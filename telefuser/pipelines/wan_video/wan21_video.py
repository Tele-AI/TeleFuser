from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.func import auto_async_call
from telefuser.utils.hf_model_utils import resolve_hf_path
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker

from ..common.rift_vfi import RiftVFIStage
from .clip_encoding import ClipEncodingStage
from .single_dit_denoising import SingleDitDenoisingStage
from .text_encoding import TextEncodingStage
from .vae import VAEStage


@dataclass
class Wan21VideoPipelineConfig:
    """Configuration for Wan2.1 video generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    clip_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    vfi_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_clip_stage: bool = False
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vfi: bool = False
    enable_metrics: bool = False


class Wan21VideoPipeline(BasePipeline):
    """Wan2.1 video generation pipeline supporting T2V and I2V.

    Supports text-to-video (T2V) and image-to-video (I2V) generation with
    optional video frame interpolation (VFI) for higher frame rates.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.base_fps = 16

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        stages = [self.vae_stage, self.denoise_stage, self.text_encoding_stage]
        if hasattr(self, "clip_stage"):
            stages.append(self.clip_stage)
        if hasattr(self, "vfi_stage"):
            stages.append(self.vfi_stage)
        return stages

    def init(self, module_manager: ModuleManager, config: Wan21VideoPipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        if config.enable_clip_stage:
            self.clip_stage = ClipEncodingStage("clip", module_manager, config.clip_config)
        if config.sample_solver == "euler":
            self.scheduler = FlowMatchScheduler(template="Wan")
        elif config.sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")
        self.denoise_stage = SingleDitDenoisingStage("denoise", module_manager, config.dit_config, self.scheduler)
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)
        if config.enable_vfi:
            self.vfi_stage = RiftVFIStage("vfi", module_manager, config.vfi_config)
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)
        if config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)

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
        cfg_scale: float = 5.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        target_fps: int | None = None,
    ) -> List[Image.Image]:
        """Generate video from text prompt and optional input image."""
        logger.info(f"start genereate {num_frames} {width}x{height} frames")
        height, width = self.check_resize_height_width(height, width)
        # Wan video requires num_frames % 4 == 1
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
        if cfg_scale != 1.0:
            prompt_list.append(negative_prompt)
        prompt_emb_list_handler = auto_async_call(self.text_encoding_stage.process, prompt_list)
        prompt_emb_list = prompt_emb_list_handler()
        prompt_emb_posi = prompt_emb_list[0]
        if cfg_scale != 1.0:
            prompt_emb_nega = prompt_emb_list[1]

        # Encode image for I2V
        clip_context = None
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
            )
            clip_context_handler = auto_async_call(self.clip_stage.process, input_image, end_image)
            ref_latent = ref_latent_handler()
            clip_context = clip_context_handler()
        # Enable sparse attention if configured
        attention_config = self.config.dit_config.attention_config
        if attention_config.is_sparse():
            sparse_config = attention_config.sparse_config
            if sparse_config is None:
                raise ValueError("Sparse attention requires sparse configuration")
            enable_sparse_attention_handler = auto_async_call(
                self.denoise_stage.enable_sparse_attention,
                height=height,
                width=width,
                num_frames=num_frames,
                sparse_config=sparse_config,
            )
            enable_sparse_attention_handler()
            logger.info(
                f"Sparse attention enabled ({sparse_config.sparse_impl}): "
                f"dense_layers={sparse_config.dense_layers}, "
                f"dense_timesteps={sparse_config.dense_timesteps}"
            )

        # denoise
        latents_handler = auto_async_call(
            self.denoise_stage.process,
            latents,
            num_inference_steps,
            ref_latent,
            prompt_emb_posi,
            prompt_emb_nega,
            clip_context,
            cfg_scale,
            sigma_shift,
        )
        latents = latents_handler()
        # Decode
        frames_handler = auto_async_call(self.vae_stage.process, "decode_video", latents, **tiler_kwargs)
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
        attn_impl: AttnImplType = AttnImplType.FLASH_ATTN_2,
        attention_config: AttentionConfig | None = None,
        enable_clip_stage: bool = False,
        enable_parallel: bool = False,
        parallel_devices: list[int] | None = None,
        sample_solver: str = "euler",
        enable_metrics: bool = False,
        **kwargs,
    ) -> "Wan21VideoPipeline":
        """Load a Wan2.1 video pipeline from HuggingFace model ID or local path.

        This method supports loading from:
        - HuggingFace model ID (e.g., "Wan-AI/Wan2.1-T2V-1.3B")
        - Local path in HuggingFace Diffusers format

        The model folder should contain:
        - transformer/diffusion_pytorch_model.safetensors (DiT weights)
        - vae/diffusion_pytorch_model.safetensors (VAE weights)
        - text_encoder/model.safetensors (Text encoder weights)
        - tokenizer/ (Tokenizer folder, optional)
        - image_encoder/ (For I2V models, optional)

        Args:
            model_id_or_path: HuggingFace model ID or local path.
            device: Device to load the model on.
            torch_dtype: Data type for model weights.
            cache_dir: Cache directory for HuggingFace models.
            attn_impl: Attention implementation type.
            attention_config: Attention configuration.
            enable_clip_stage: Enable CLIP stage for I2V.
            enable_parallel: Enable parallel processing.
            parallel_devices: Devices for parallel processing.
            sample_solver: Sampling solver (euler or unipc).
            enable_metrics: Enable metrics collection for all stages.
            **kwargs: Additional configuration options.

        Example:
            >>> pipe = Wan21VideoPipeline.from_pretrained(
            ...     "Wan-AI/Wan2.1-T2V-1.3B",
            ...     device="cuda",
            ...     torch_dtype=torch.bfloat16,
            ...     enable_metrics=True,
            ... )
            >>> video = pipe(
            ...     prompt="A cat playing piano",
            ...     num_frames=81,
            ...     height=480,
            ...     width=832,
            ... )
            >>> # Get metrics in Prometheus format
            >>> print(pipe.get_prometheus_metrics())
        """
        # Resolve model path (download if needed)
        model_root = resolve_hf_path(model_id_or_path, cache_dir)
        logger.info(f"Loading Wan2.1 pipeline from: {model_root}")

        # Use analyzer to discover components
        from telefuser.utils.hf_model_analyzer import HFModelAnalyzer

        analyzer = HFModelAnalyzer(model_root)

        # Get component paths
        transformer_path = analyzer.get_transformer_path()
        vae_path = analyzer.get_vae_path()
        text_encoder_path = analyzer.get_text_encoder_path()
        tokenizer_path = analyzer.get_tokenizer_path()
        image_encoder_path = analyzer.get_image_encoder_path()

        if not transformer_path:
            raise ValueError(f"No transformer/DiT weights found in {model_root}")
        if not vae_path:
            raise ValueError(f"No VAE weights found in {model_root}")
        if not text_encoder_path:
            raise ValueError(f"No text encoder weights found in {model_root}")

        logger.info(f"  Transformer: {transformer_path}")
        logger.info(f"  VAE: {vae_path}")
        logger.info(f"  Text Encoder: {text_encoder_path}")
        if image_encoder_path:
            logger.info(f"  Image Encoder: {image_encoder_path}")

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

        # Load image encoder if needed (for I2V)
        if enable_clip_stage and image_encoder_path:
            mm.load_model(image_encoder_path, device="cpu", torch_dtype=torch_dtype)

        # Create pipeline
        pipeline = cls(device=device, torch_dtype=torch_dtype)

        # Create config
        config = Wan21VideoPipelineConfig()
        config.sample_solver = sample_solver
        config.enable_clip_stage = enable_clip_stage
        # Create attention config from attn_impl if not provided
        if attention_config is None:
            attention_config = AttentionConfig.dense_attention(attn_impl)
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

        logger.info(f"Successfully loaded Wan2.1 pipeline from {model_id_or_path}")
        return pipeline
