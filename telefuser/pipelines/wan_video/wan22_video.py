from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.func import auto_async_call
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker
from telefuser.worker.ray_worker import create_ray_worker
from telefuser.worker.tensor_channel import WorkerTensorChannel

from ..common.rift_vfi import RiftVFIStage
from .moe_dit_denoising import MoeDitDenoisingStage
from .text_encoding import TextEncodingStage
from .vae import VAEStage


@dataclass
class Wan22VideoPipelineConfig:
    """Configuration for Wan2.2 video generation pipeline with MoE."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_high_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_low_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vfi_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"  # or mix-euler, unipc
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vae_ray: bool = False
    enable_vfi: bool = False
    enable_metrics: bool = False


class Wan22VideoPipeline(BasePipeline):
    """Wan2.2 video generation pipeline with MoE (Mixture of Experts) architecture.

    Uses two DiT models: high-quality expert for early denoising steps
    and low-quality expert for later steps, switched at a boundary timestep.
    Supports optional video frame interpolation for higher frame rates.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.base_fps = 16

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        stages = [self.vae_stage, self.denoise_stage, self.text_encoding_stage]
        if hasattr(self, "vfi_stage"):
            stages.append(self.vfi_stage)
        return stages

    def init(self, module_manager: ModuleManager, config: Wan22VideoPipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        if config.sample_solver == "euler":
            self.scheduler = FlowMatchScheduler(template="Wan")
        elif config.sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False,
            )
        elif config.sample_solver == "mix-euler":
            self.scheduler = FlowMatchScheduler(template="Wan-Mix")
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")
        self.denoise_stage = MoeDitDenoisingStage(
            "denoise",
            module_manager,
            config.dit_high_config,
            config.dit_low_config,
            self.scheduler,
        )
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)
        if config.enable_vfi:
            self.vfi_stage = RiftVFIStage("vfi", module_manager, config.vfi_config)
        self._worker_tensor_channels: list[WorkerTensorChannel] = []
        vae_to_denoise_channel = None
        denoise_to_vae_channel = None
        direct_tensor_handoff = (
            config.enable_vae_parallel and config.enable_denoising_parallel and not config.enable_vae_ray
        )
        if direct_tensor_handoff:
            timeout = max(
                config.vae_config.parallel_config.timeout,
                config.dit_high_config.parallel_config.timeout,
            )
            vae_to_denoise_channel = WorkerTensorChannel(
                config.dit_high_config.parallel_config.world_size,
                timeout=timeout,
            )
            denoise_to_vae_channel = WorkerTensorChannel(
                config.vae_config.parallel_config.world_size,
                timeout=timeout,
            )
            self._worker_tensor_channels.extend((vae_to_denoise_channel, denoise_to_vae_channel))
        if config.enable_vae_ray:
            logger.info("enable ray actor for vae")
            self.vae_stage = create_ray_worker(self.vae_stage, self.config.enable_vae_parallel)
        elif config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(
                self.vae_stage,
                tensor_output_channel=vae_to_denoise_channel,
                tensor_output_methods=("process",) if vae_to_denoise_channel is not None else (),
                tensor_input_channels=(denoise_to_vae_channel,) if denoise_to_vae_channel is not None else (),
            )
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(
                self.denoise_stage,
                tensor_output_channel=denoise_to_vae_channel,
                tensor_output_methods=("process",) if denoise_to_vae_channel is not None else (),
                tensor_input_channels=(vae_to_denoise_channel,) if vae_to_denoise_channel is not None else (),
            )
        self._uses_direct_tensor_handoff = direct_tensor_handoff

        # Auto-enable metrics if configured
        if config.enable_metrics:
            self.enable_metrics()

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | List[str],
        negative_prompt: str = "",
        input_image: Image.Image | None = None,
        end_image: Image.Image | None = None,
        seed: int | None = None,
        rand_device: str | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        cfg_scale_high: float = 5.0,
        cfg_scale_low: float = 5.0,
        boundary: float = 0.875,
        target_fps: int | None = None,
        latent_data: dict | None = None,
    ) -> List[Image.Image] | tuple[List[Image.Image], dict]:
        """Generate video from text prompt and optional input image."""
        logger.info(f"start genereate {num_frames} {width}x{height} frames")
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")

        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        # Initialize noise
        if rand_device is None:
            rand_device = self.device
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        latents = noise.to(dtype=self.torch_dtype, device=self.device)
        # Encode prompts
        prompt_emb_nega = None
        prompt_list = [prompt]
        if cfg_scale_high != 1.0:
            prompt_list.append(negative_prompt)

        prompt_emb_list_handler = auto_async_call(self.text_encoding_stage.process, prompt_list)
        # Encode image for I2V
        ref_latent = None
        if input_image is not None:
            input_image = self.preprocess_image(input_image, height, width)
            if end_image is not None:
                end_image = self.preprocess_image(end_image, height, width)

            ref_latent_handler = auto_async_call(
                self.vae_stage.process,
                "encode_image",
                input_image,
                end_image,
                num_frames,
                **tiler_kwargs,
                is_ray=self.config.enable_vae_ray,
                **(
                    {"_tensor_transport": self._uses_direct_tensor_handoff}
                    if isinstance(self.vae_stage, ParallelWorker)
                    else {}
                ),
            )
            ref_latent = ref_latent_handler()
        prompt_emb_list = prompt_emb_list_handler()
        prompt_emb_posi = prompt_emb_list[0]
        if cfg_scale_high != 1.0:
            prompt_emb_nega = prompt_emb_list[1]
        # denoise with MoE model switching
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents,
            num_inference_steps,
            ref_latent,
            prompt_emb_posi,
            prompt_emb_nega,
            cfg_scale_high,
            cfg_scale_low,
            sigma_shift,
            boundary,
            latent_data=latent_data,
            **(
                {"_tensor_transport": self._uses_direct_tensor_handoff}
                if isinstance(self.denoise_stage, ParallelWorker)
                else {}
            ),
        )
        denoise_result = latents_handler()

        # MoE stage returns a tuple when latent_data is provided.
        latent_payload: dict | None = None
        if isinstance(denoise_result, tuple):
            latents, latent_payload = denoise_result
        else:
            latents = denoise_result
        frames_handler = auto_async_call(
            self.vae_stage.process,
            "decode_video",
            latents,
            **tiler_kwargs,
            **({"_tensor_transport": False} if isinstance(self.vae_stage, ParallelWorker) else {}),
        )
        frames = frames_handler()
        frames = self.tensor2video(frames[0])

        # VFI (Video Frame Interpolation)
        if self.config.enable_vfi and target_fps is not None:
            logger.info(f"Interpolating video from {self.base_fps} fps to {target_fps} fps")
            frames_handler = auto_async_call(self.vfi_stage.process, frames, self.base_fps, target_fps)
            frames = frames_handler()
            logger.info(f"VFI complete, total frames: {len(frames)}")

        if latent_payload is not None:
            latent_payload["num_frames"] = num_frames
            return frames, latent_payload
        return frames

    def close(self) -> None:
        """Close multiprocessing workers before releasing tensor channels."""
        for stage_name in ("denoise_stage", "vae_stage"):
            stage = getattr(self, stage_name, None)
            if isinstance(stage, ParallelWorker):
                stage.close()
        for channel in getattr(self, "_worker_tensor_channels", ()):
            channel.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
        attention_config: AttentionConfig | None = None,
        enable_parallel: bool = False,
        parallel_devices: list[int] | None = None,
        sample_solver: str = "euler",
        enable_metrics: bool = False,
        **kwargs,
    ) -> "Wan22VideoPipeline":
        """Load a Wan2.2 video pipeline from HuggingFace model ID or local path.

        This method supports loading from:
        - HuggingFace model ID (e.g., "Wan-AI/Wan2.1-I2V-14B-480P")
        - Local path in HuggingFace Diffusers format

        The model folder should contain:
        - transformer/diffusion_pytorch_model.safetensors (DiT weights)
        - vae/diffusion_pytorch_model.safetensors (VAE weights)
        - text_encoder/model.safetensors (Text encoder weights)
        - tokenizer/ (Tokenizer folder, optional)
        """
        from telefuser.utils.hf_model_utils import resolve_hf_path

        # Resolve model path (download if needed)
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        logger.info(f"Loading Wan2.2 pipeline from: {model_root}")

        # Use analyzer to discover components
        from telefuser.utils.hf_model_analyzer import HFModelAnalyzer

        analyzer = HFModelAnalyzer(model_root)

        # Get component paths
        transformer_path = analyzer.get_transformer_path()
        vae_path = analyzer.get_vae_path()
        text_encoder_path = analyzer.get_text_encoder_path()
        tokenizer_path = analyzer.get_tokenizer_path()

        if not transformer_path:
            raise ValueError(f"No transformer/DiT weights found in {model_root}")
        if not vae_path:
            raise ValueError(f"No VAE weights found in {model_root}")
        if not text_encoder_path:
            raise ValueError(f"No text encoder weights found in {model_root}")

        logger.info(f"  Transformer: {transformer_path}")
        logger.info(f"  VAE: {vae_path}")
        logger.info(f"  Text Encoder: {text_encoder_path}")

        # Load models using ModuleManager
        mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")
        mm.load_model(transformer_path, device="cpu", torch_dtype=torch_dtype)
        mm.load_model(vae_path, device="cpu", torch_dtype=torch_dtype)
        mm.load_model(text_encoder_path, device="cpu", torch_dtype=torch_dtype)

        # Load tokenizer if available
        if tokenizer_path:
            mm.load_from_huggingface(
                tokenizer_path,
                module_source="transformers",
                module_name="tokenizer",
                torch_dtype=torch_dtype,
            )

        # Create pipeline
        pipeline = cls(device=device, torch_dtype=torch_dtype)

        # Create config
        config = Wan22VideoPipelineConfig()
        config.sample_solver = sample_solver
        if attention_config is not None:
            config.dit_high_config.attention_config = attention_config
            config.dit_low_config.attention_config = attention_config

        # Configure parallelism
        if enable_parallel and parallel_devices:
            config.dit_high_config.parallel_config.device_ids = parallel_devices
            config.dit_high_config.parallel_config.sp_ulysses_degree = 2
            config.dit_low_config.parallel_config.device_ids = parallel_devices
            config.dit_low_config.parallel_config.sp_ulysses_degree = 2
            config.enable_denoising_parallel = True

        # Configure metrics
        config.enable_metrics = enable_metrics

        # Apply additional config kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Initialize pipeline
        pipeline.init(mm, config)

        logger.info(f"Successfully loaded Wan2.2 pipeline from {model_id_or_path}")
        return pipeline
