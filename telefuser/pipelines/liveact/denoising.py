"""LiveAct denoising stage with audio cross-attention and KV cache.

Handles the diffusion denoising process with:
- Audio cross-attention for talking head generation
- KV cache management for streaming generation
- Memory compression for efficient long video generation
- Ulysses Sequence Parallel support
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from tqdm import tqdm

from telefuser.cache import KVCacheConfig, KVCacheManager
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_ulysses_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.metrics import with_metrics
from telefuser.platforms import current_platform
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug

# Pre-defined timesteps matching original generate.py (created once)
TIMESTEPS = [1000.0, 937.5, 833.33333333, 0.0]


class LiveActDenoisingStage(BaseStage):
    """Denoising stage for LiveAct with audio cross-attention.

    Supports:
    - Streaming video generation with KV cache
    - Audio CFG (classifier-free guidance for audio)
    - Memory-efficient KV cache with optional FP8 quantization
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        kv_cache_config: KVCacheConfig | None = None,
        mean_memory: bool = False,
    ):
        super().__init__(name, model_runtime_config)
        self.dit = module_manager.fetch_module("liveact_dit")
        # set_attention_config only exists in TeleFuser's LiveActDiT
        # Original SoulX-LiveAct WanModel uses SageAttention directly
        if hasattr(self.dit, "set_attention_config"):
            self.dit.set_attention_config(model_runtime_config.attention_config)
        if hasattr(self.dit, "set_mean_memory"):
            self.dit.set_mean_memory(mean_memory)
        self.model_names = ["dit"]

        # Enable FP8 GEMM and torch.compile only in single GPU mode
        # In distributed mode, these are applied in parallel_models() after spawn
        quant_config = model_runtime_config.quant_config
        compile_config = model_runtime_config.compile_config
        parallel_cfg = model_runtime_config.parallel_config

        if parallel_cfg.world_size == 1:
            # 1. Enable FP8 GEMM (same as original generate.py)
            if quant_config.enabled:
                try:
                    from telefuser.ops.fp8_gemm import FP8GemmOptions, enable_fp8_gemm

                    enable_fp8_gemm(self.dit, options=FP8GemmOptions())
                    logger.info("✓ FP8 GEMM enabled for DiT FFN layers")
                except ImportError:
                    logger.warning("✗ FP8 GEMM not available, skipping")

            # 2. torch.compile (exact same approach as original generate.py)
            # Original: torch.compile(model, mode="max-autotune-no-cudagraphs", backend="inductor", dynamic=False)
            if compile_config.enabled:
                self.dit = torch.compile(self.dit, **(compile_config.get_compile_kwargs()))
        # Get model architecture params from dit
        self.num_layers = len(self.dit.blocks)
        self.num_heads = self.dit.num_heads
        self.head_dim = self.dit.dim // self.dit.num_heads

        self.scheduler = FlowMatchScheduler(template="Wan")

        self.kv_cache_manager: KVCacheManager | None = None
        self.kv_cache_config = kv_cache_config or KVCacheConfig()

        self._timestep_tensors: list[torch.Tensor] | None = None
        self._kv_cache_tokens_per_frame: int | None = None
        self._kv_cache_null_audio_manager: KVCacheManager | None = None

    def parallel_models(self) -> None:
        """Configure parallel processing for the DiT model.

        This method is called by ParallelWorker to set up:
        - Device mesh for distributed communication
        - Ulysses Sequence Parallel (USP) for attention
        - FSDP for model sharding
        """
        parallel_cfg = self.model_runtime_config.parallel_config
        # device_mesh and enable_usp only exist in TeleFuser's LiveActDiT
        # Original SoulX-LiveAct uses xfuser for SP (handled externally)
        if hasattr(self.dit, "device_mesh"):
            self.dit.device_mesh = create_device_mesh_from_config(parallel_cfg)
        if hasattr(self.dit, "set_attention_config"):
            self.dit.set_attention_config(self.model_runtime_config.attention_config)

        # Enable Ulysses Sequence Parallel (TeleFuser implementation)
        if hasattr(self.dit, "enable_usp") and parallel_cfg.sp_ulysses_degree > 1:
            self.dit.enable_usp(self.dit.device_mesh)

        # Enable FSDP
        if parallel_cfg.enable_fsdp:
            logger.info(f"enable fsdp for {self.name}")
            shard_fn = partial(shard_model, wrap_module_names=self.dit.get_fsdp_module_names())
            self.dit = shard_fn(module=self.dit, device_id=self.device)
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                self.dit.cpu()
                current_platform.empty_cache()

        # Enable FP8 GEMM for FFN layers if quant_config is set (distributed mode)
        quant_config = self.model_runtime_config.quant_config
        if quant_config.enabled:
            try:
                from telefuser.ops.fp8_gemm import FP8GemmOptions, enable_fp8_gemm

                enable_fp8_gemm(self.dit, options=FP8GemmOptions())
                logger.info("✓ FP8 GEMM enabled for DiT FFN layers")
            except ImportError:
                logger.warning("✗ FP8 GEMM not available, skipping")

        # Enable torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(
                self.dit,
                mode="max-autotune-no-cudagraphs",
                backend="inductor",
                dynamic=False,
            )

    def set_kv_cache_config(self, config: KVCacheConfig) -> None:
        """Set KV cache configuration."""
        self.kv_cache_config = config

    def _get_timestep_tensors(self) -> list[torch.Tensor]:
        """Get or create timestep tensors on device.

        Matches original generate.py which creates timesteps once before all iterations.
        """
        if self._timestep_tensors is None or self._timestep_tensors[0].device != self.device:
            self._timestep_tensors = [torch.tensor([t], device=self.device, dtype=torch.float32) for t in TIMESTEPS]
        return self._timestep_tensors

    def init_kv_cache(
        self,
        tokens_per_frame: int,
        num_timesteps: int = 3,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        config: KVCacheConfig | None = None,
    ) -> KVCacheManager:
        """Initialize KV cache for streaming generation.

        Args:
            tokens_per_frame: Number of tokens per frame (H/patch * W/patch)
            num_timesteps: Number of denoising timesteps
            device: Device for KV cache
            dtype: Data type for KV cache
            config: KV cache configuration

        Returns:
            KVCacheManager instance
        """
        if config is not None:
            self.kv_cache_config = config

        # Get SP size from dit_model
        device_mesh = getattr(self.dit, "device_mesh", None)
        sp_size = get_ulysses_world_size(device_mesh) or 1

        # Create KVCacheManager from dit model
        manager = KVCacheManager.from_dit_model(
            dit_model=self.dit,
            config=self.kv_cache_config,
            tokens_per_frame=tokens_per_frame,
            num_timesteps=num_timesteps,
            sp_size=sp_size,
            device=device,
            dtype=dtype,
        )

        self.kv_cache_manager = manager
        return manager

    def clear_kv_cache(self) -> None:
        """Clear KV cache and related state."""
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.clear()
        self.kv_cache_manager = None
        self._kv_cache_tokens_per_frame = None
        self._kv_cache_null_audio_manager = None

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("denoising_process")
    @torch.inference_mode()
    def process(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        clip_fea: torch.Tensor,
        audio_embedding: torch.Tensor,
        y: torch.Tensor,
        tokens_per_frame: int,
        audio_cfg: float = 1.0,
        num_inference_steps: int = 3,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Run denoising with audio cross-attention and KV cache.

        Same structure as original generate.py for optimal torch.compile performance.
        """
        blksz_lst = [6, 8]

        if seed is not None:
            torch.manual_seed(seed)

        latent_blksz = latent.shape[1]
        f = 0 if latent_blksz == blksz_lst[0] else 1

        # Pre-compute start/end indices (f doesn't change during loop)
        start_idx = sum(blksz_lst[:f]) * tokens_per_frame
        end_idx = sum(blksz_lst[: f + 1]) * tokens_per_frame

        # Profile KV cache initialization
        with ProfilingContext4Debug("init_kv_cache_check"):
            if self._kv_cache_tokens_per_frame != tokens_per_frame:
                self.init_kv_cache(
                    tokens_per_frame,
                    num_timesteps=num_inference_steps,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
                self._kv_cache_tokens_per_frame = tokens_per_frame

        if audio_cfg > 1.0 and self._kv_cache_null_audio_manager is None:
            self._kv_cache_null_audio_manager = self.init_kv_cache(
                tokens_per_frame,
                num_timesteps=num_inference_steps,
                device=self.device,
                dtype=self.torch_dtype,
            )

        y_slice = y[:, :, start_idx // tokens_per_frame : end_idx // tokens_per_frame, ...]

        timestep_tensors = self._get_timestep_tensors()

        # Direct dit calls (same as original generate.py) - no intermediate functions
        for i in tqdm(range(len(TIMESTEPS) - 1), desc="liveact denoise"):
            timestep = timestep_tensors[i]

            # Get KVCache for this timestep (list of KVCache per layer)
            kv_cache_timestep = self.kv_cache_manager.get_timestep_caches(i)
            if audio_cfg > 1.0:
                kv_cache_null_audio_timestep = self._kv_cache_null_audio_manager.get_timestep_caches(i)

            # Profile dit forward
            with ProfilingContext4Debug(f"dit_forward_t{i}"):
                # Direct dit call matching original generate.py
                # y_slice is [1, 17, T, H, W], dit expects list of [17, T, H, W] tensors
                noise_pred = self.dit(
                    [latent],
                    t=timestep,
                    kv_cache=kv_cache_timestep,
                    skip_audio=False,
                    context=context,
                    clip_fea=clip_fea,
                    audio=audio_embedding,
                    y=[y_slice.squeeze(0)],
                    start_idx=start_idx,
                    end_idx=end_idx,
                )[0]

            # Audio CFG (same as original)
            if audio_cfg > 1.0 and i in [1, 2] and kv_cache_null_audio_timestep is not None:
                noise_pred_drop_audio = self.dit(
                    [latent],
                    t=timestep,
                    kv_cache=kv_cache_null_audio_timestep,
                    context=context,
                    clip_fea=clip_fea,
                    audio=torch.zeros_like(audio_embedding),
                    y=[y_slice.squeeze(0)],
                    start_idx=start_idx,
                    end_idx=end_idx,
                )[0]
                noise_pred = noise_pred_drop_audio + audio_cfg * (noise_pred - noise_pred_drop_audio)

            dt = (TIMESTEPS[i] - TIMESTEPS[i + 1]) / 1000
            latent = latent + (-noise_pred) * dt

        return latent
