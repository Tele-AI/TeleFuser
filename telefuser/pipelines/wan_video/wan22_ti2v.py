"""Wan2.2 Text-Image-to-Video Pipeline.

Unified pipeline for:
- Text-to-video (T2V): Generate video from text prompt
- Image-to-video (I2V): Generate video from text + image

Features:
- 48 latent channels (vs 16 in Wan2.1)
- Support for both UniPC and DPM++ samplers
- Optional VFI (Video Frame Interpolation)
"""

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

from ..common.rift_vfi import RiftVFIStage
from .text_encoding import TextEncodingStage
from .ti2v_denoising import TI2VDenoisingStage
from .vae import VAEStage


@dataclass
class Wan22TI2VPipelineConfig:
    """Configuration for Wan2.2 TI2V pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vfi_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "unipc"  # or "dpm++"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vae_ray: bool = False
    enable_vfi: bool = False
    enable_metrics: bool = False


class Wan22TI2VPipeline(BasePipeline):
    """Wan2.2 unified Text-Image-to-Video pipeline.

    Supports both T2V and I2V generation in a single pipeline.
    Uses Wan2.2 VAE with 48 latent channels.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 32  # VAE 16x * DiT patch 2x = 32
        self.width_division_factor = 32  # VAE 16x * DiT patch 2x = 32
        self.base_fps = 24  # Wan2.2 uses 24fps by default
        # z_dim and spatial_factor will be derived from VAE after init()
        self._z_dim: int | None = None
        self._spatial_factor: int | None = None

    @property
    def z_dim(self) -> int:
        """Get latent channel dimension from VAE model."""
        if self._z_dim is None:
            self._z_dim = getattr(self.vae_stage.vae, "z_dim", 48)
        return self._z_dim

    @property
    def spatial_factor(self) -> int:
        """Get spatial compression factor from VAE model."""
        if self._spatial_factor is None:
            self._spatial_factor = getattr(self.vae_stage.vae, "upsampling_factor", 16)
        return self._spatial_factor

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        stages = [self.vae_stage, self.denoise_stage, self.text_encoding_stage]
        if hasattr(self, "vfi_stage"):
            stages.append(self.vfi_stage)
        return stages

    def init(self, module_manager: ModuleManager, config: Wan22TI2VPipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)

        if config.sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False,
            )
        elif config.sample_solver == "dpm++":
            self.scheduler = FlowMatchScheduler(template="Wan")
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")

        self.denoise_stage = TI2VDenoisingStage(
            "denoise",
            module_manager,
            config.dit_config,
            self.scheduler,
        )
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)

        if config.enable_vfi:
            self.vfi_stage = RiftVFIStage("vfi", module_manager, config.vfi_config)

        if config.enable_vae_ray:
            logger.info("enable ray actor for vae")
            self.vae_stage = create_ray_worker(self.vae_stage, self.config.enable_vae_parallel)
        elif config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)

        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)

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
        cfg_scale: float = 5.0,
        target_fps: int | None = None,
    ) -> List[Image.Image]:
        """Generate video from text prompt and optional input image.

        Args:
            prompt: Text prompt for video generation
            negative_prompt: Negative prompt for CFG
            input_image: Input image for I2V mode (None for T2V)
            end_image: End image for I2V with start/end frames
            seed: Random seed
            rand_device: Device for random number generation
            height: Output video height
            width: Output video width
            num_frames: Number of frames (should be 4n+1)
            num_inference_steps: Number of denoising steps
            sigma_shift: Noise schedule shift parameter
            tiled: Enable tiled VAE processing
            tile_size: Tile size for VAE
            tile_stride: Tile stride for VAE
            cfg_scale: CFG scale (1.0 for no CFG)
            target_fps: Target FPS for VFI interpolation

        Returns:
            List of PIL Image frames
        """
        logger.info(f"start generate {num_frames} {width}x{height} frames")
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

        # Calculate latent shape for Wan2.2
        # vae_stride = (4, 16, 16): temporal 4x, spatial 16x compression
        latent_t = (num_frames - 1) // 4 + 1
        latent_h = height // self.spatial_factor
        latent_w = width // self.spatial_factor

        noise = self.generate_noise(
            (1, self.z_dim, latent_t, latent_h, latent_w),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        latents = noise.to(dtype=self.torch_dtype, device=self.device)

        # Encode prompts
        prompt_emb_nega = None
        prompt_list = [prompt]
        if cfg_scale != 1.0:
            prompt_list.append(negative_prompt)

        prompt_emb_list_handler = auto_async_call(self.text_encoding_stage.process, prompt_list)

        # Encode image for I2V (Wan2.2 uses blended latent approach)
        ref_latent = None
        ref_mask = None
        if input_image is not None:
            input_image = self.preprocess_image(input_image, height, width)
            if end_image is not None:
                end_image = self.preprocess_image(end_image, height, width)

            # Get image latent and mask separately for blended approach
            ref_latent_handler = auto_async_call(
                self.vae_stage.process,
                "encode_image",
                input_image,
                end_image,
                num_frames,
                **tiler_kwargs,
                concat_mask=False,  # Wan2.2 style: return latent and mask separately
                is_ray=self.config.enable_vae_ray,
            )
            ref_result = ref_latent_handler()
            # ref_result is tuple: (latent [z_dim, 1, H, W], num_frames)
            # For start + end images: ((start_latent, end_latent), num_frames)
            ref_latent, ref_mask = ref_result

        prompt_emb_list = prompt_emb_list_handler()
        prompt_emb_posi = prompt_emb_list[0]
        if cfg_scale != 1.0:
            prompt_emb_nega = prompt_emb_list[1]

        # Denoise
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents,
            num_inference_steps,
            ref_latent,
            ref_mask,
            prompt_emb_posi,
            prompt_emb_nega,
            cfg_scale,
            sigma_shift,
        )
        latents = latents_handler()

        # Decode video (keep on device for single-process decode so uint8 conversion runs before D2H)
        keep_on_device = not self.config.enable_vae_parallel and not self.config.enable_vae_ray
        frames_handler = auto_async_call(
            self.vae_stage.process,
            "decode_video",
            latents,
            return_cpu=not keep_on_device,
            **tiler_kwargs,
        )
        frames = frames_handler()
        frames = self.tensor2video(frames[0])

        # VFI (Video Frame Interpolation)
        if self.config.enable_vfi and target_fps is not None:
            logger.info(f"Interpolating video from {self.base_fps} fps to {target_fps} fps")
            frames_handler = auto_async_call(self.vfi_stage.process, frames, self.base_fps, target_fps)
            frames = frames_handler()
            logger.info(f"VFI complete, total frames: {len(frames)}")

        return frames

    def __del__(self):
        del self.vae_stage
        del self.denoise_stage
        del self.text_encoding_stage

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
        sample_solver: str = "unipc",
        enable_metrics: bool = False,
        **kwargs,
    ) -> "Wan22TI2VPipeline":
        """Load a Wan2.2 TI2V pipeline from HuggingFace model ID or local path.

        This method supports loading from:
        - HuggingFace model ID (e.g., "Wan-AI/Wan2.2-TI2V-14B")
        - Local path in HuggingFace Diffusers format

        The model folder should contain:
        - transformer/diffusion_pytorch_model.safetensors (DiT weights)
        - vae/diffusion_pytorch_model.safetensors (VAE weights with 48 channels)
        - text_encoder/model.safetensors (Text encoder weights)
        - tokenizer/ (Tokenizer folder, optional)
        """
        from telefuser.utils.hf_model_utils import resolve_hf_path

        # Resolve model path (download if needed)
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        logger.info(f"Loading Wan2.2 TI2V pipeline from: {model_root}")

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
        config = Wan22TI2VPipelineConfig()
        config.sample_solver = sample_solver

        if attention_config is not None:
            config.dit_config.attention_config = attention_config

        # Configure parallelism
        if enable_parallel and parallel_devices:
            config.dit_config.parallel_config.device_ids = parallel_devices
            config.dit_config.parallel_config.sp_ulysses_degree = 2
            config.enable_denoising_parallel = True

        # Configure metrics
        config.enable_metrics = enable_metrics

        # Apply additional config kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Initialize pipeline
        pipeline.init(mm, config)

        logger.info(f"Successfully loaded Wan2.2 TI2V pipeline from {model_id_or_path}")
        return pipeline
