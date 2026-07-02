"""Configuration classes for models, attention, and distributed training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from telefuser._logo import TELEFUSER_LOGO


@dataclass
class RayGPUConfig:
    """Ray GPU resource configuration."""

    num_gpus: int = 0
    memory_limit: float = 0.8  # GPU memory limit ratio (0-1)


@dataclass
class RayConfig:
    """Ray cluster configuration for distributed workers."""

    ray_address: str | None = None  # Ray cluster address (None for local)
    gpu_config: RayGPUConfig = field(default_factory=RayGPUConfig)
    num_cpus: int = 8
    memory_gb: int = 32


@dataclass
class ParallelConfig:
    """Distributed parallel processing configuration.

    Supports data parallelism (DP), classifier-free guidance parallelism (CFG),
    sequence parallelism (Ulysses/Ring), and tensor parallelism (TP).
    """

    device_ids: list | None = None
    dp_degree: int = 1  # Data parallelism degree
    cfg_degree: int = 1  # CFG parallelism degree
    sp_ulysses_degree: int = 1  # Ulysses sequence parallelism degree
    sp_ring_degree: int = 1  # Ring attention sequence parallelism degree
    pp_degree: int = 1
    tp_degree: int = 1  # Tensor parallelism degree
    enable_fsdp: bool = False
    timeout: int = 600  # Seconds
    queue_with_cpu: bool = False

    @property
    def world_size(self) -> int:
        """Total number of devices in parallel configuration."""
        if self.device_ids is None:
            return 1
        self.validate()
        return len(self.device_ids)

    def validate(self) -> None:
        """Validate that device count matches parallelism degrees."""
        device_num = 1 if self.device_ids is None else len(self.device_ids)
        degree_sum = (
            self.cfg_degree
            * self.sp_ring_degree
            * self.sp_ulysses_degree
            * self.tp_degree
            * self.dp_degree
            * self.pp_degree
        )
        if device_num != degree_sum:
            raise RuntimeError(f"device num {device_num} and world size {degree_sum} not match")


class WeightOffloadType(Enum):
    """CPU offloading strategy for model weights."""

    NO_CPU_OFFLOAD = auto()
    MODEL_CPU_OFFLOAD = auto()
    SEQUENTIAL_CPU_OFFLOAD = auto()
    ASYNC_CPU_OFFLOAD = auto()


@dataclass
class LoraConfig:
    """LoRA adapter configuration."""

    path: str = ""
    strength: float = 1.0


@dataclass
class OffloadConfig:
    """CPU memory offloading configuration."""

    offload_type: WeightOffloadType = WeightOffloadType.NO_CPU_OFFLOAD
    pin_cpu_memory: bool = True  # Pin memory for faster CPU->GPU transfer
    offload_ratio: float = 1.0  # Fraction of model to offload (0-1)
    prefetch_size: int = 1  # Number of layers to prefetch for async offload


@dataclass
class FeatureCacheConfig:
    """Configuration for feature caching in diffusion transformers.

    Feature caching accelerates inference by reusing computations from
    previous diffusion steps with Taylor series approximation.
    """

    enabled: bool = False
    model_type: str = ""  # Model type for loading cache parameters
    n_derivatives: int = 1  # Order of Taylor series expansion
    taylor_threshold: int = 2  # Threshold for switching to residual reuse


class AttnImplType(Enum):
    """Available attention implementations."""

    # Dense attention implementations
    TORCH_SDPA = auto()
    TORCH_CUDNN = auto()
    SAGE_ATTN_2_8_8 = auto()
    SAGE_ATTN_2_8_16 = auto()
    SAGE_ATTN_2_8_8_SM90 = auto()
    SPARGE_ATTN = auto()
    FLASH_ATTN_2 = auto()
    FLASH_ATTN_3 = auto()
    FLASH_ATTN_4 = auto()
    # Sparse attention implementations
    RADIAL_ATTN = auto()  # Radial attention for video generation
    LOCAL_SPARSE_ATTN = auto()  # Local window sparse attention


@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention implementations.

    Used with radial or local sparse attention to reduce memory usage
    for long sequences like videos.
    """

    sparse_impl: str | None = None  # "radial", "local", or None
    dense_timesteps: int = 40  # Initial timesteps to use dense attention
    dense_layers: int = 0  # Initial layers to use dense attention
    decay_factor: float = 1.0  # Decay for radial attention window
    local_window_size: int = 6  # Window size for local attention
    block_size: int = 128  # Block size for sparse computation
    use_sage_attention: bool = False  # Use sage attention backend

    def should_use_dense(self, numeral_timestep: int, layer_idx: int) -> bool:
        """Check if dense attention should be used for current step/layer.

        Dense attention is used during initial timesteps/layers or when
        sparse attention is disabled.
        """
        return numeral_timestep < self.dense_timesteps or layer_idx < self.dense_layers or self.sparse_impl is None


@dataclass
class AttentionConfig:
    """Unified configuration for all attention implementations."""

    attn_impl: AttnImplType = AttnImplType.TORCH_SDPA
    sparse_config: SparseAttentionConfig | None = None
    scale: float | None = None  # Optional attention scale factor
    dropout: float = 0.0
    is_causal: bool = False

    @classmethod
    def radial_attention(
        cls,
        dense_timesteps: int = 40,
        dense_layers: int = 0,
        decay_factor: float = 1.0,
        use_sage_attention: bool = False,
        **kwargs: any,
    ) -> AttentionConfig:
        """Create config for radial attention (sparse attention for video)."""
        return cls(
            attn_impl=AttnImplType.RADIAL_ATTN,
            sparse_config=SparseAttentionConfig(
                sparse_impl="radial",
                dense_timesteps=dense_timesteps,
                dense_layers=dense_layers,
                decay_factor=decay_factor,
                use_sage_attention=use_sage_attention,
            ),
            **kwargs,
        )

    @classmethod
    def local_sparse_attention(
        cls,
        dense_timesteps: int = 40,
        dense_layers: int = 0,
        local_window_size: int = 6,
        **kwargs: any,
    ) -> AttentionConfig:
        """Create config for local sparse attention."""
        return cls(
            attn_impl=AttnImplType.LOCAL_SPARSE_ATTN,
            sparse_config=SparseAttentionConfig(
                sparse_impl="local",
                dense_timesteps=dense_timesteps,
                dense_layers=dense_layers,
                local_window_size=local_window_size,
            ),
            **kwargs,
        )

    @classmethod
    def dense_attention(cls, attn_impl: AttnImplType = AttnImplType.FLASH_ATTN_2, **kwargs: any) -> AttentionConfig:
        """Create config for dense attention."""
        return cls(attn_impl=attn_impl, sparse_config=None, **kwargs)

    def is_sparse(self) -> bool:
        """Check if using sparse attention (radial or local)."""
        return self.attn_impl in (AttnImplType.RADIAL_ATTN, AttnImplType.LOCAL_SPARSE_ATTN)

    def should_use_dense(self, numeral_timestep: int, layer_idx: int) -> bool:
        """Check if dense attention should be used for current step/layer."""
        if not self.is_sparse() or self.sparse_config is None:
            return True
        return self.sparse_config.should_use_dense(numeral_timestep, layer_idx)


@dataclass
class CompileConfig:
    """Configuration for torch.compile optimization.

    This class maps directly to torch.compile() parameters and does NOT modify
    global PyTorch settings. Use set_global_compile_configs() separately for
    global-only settings like recompile_limit.

    Attributes:
        enabled: Whether to enable torch.compile
        backend: Backend to use - "inductor" (default), "eager", etc.
        fullgraph: Require full graph compilation (more restrictive but faster)
        dynamic: Support dynamic shapes (None = auto-detect, False = static)
        mode: Compilation mode preset - None, "default", "reduce-overhead",
              "max-autotune", "max-autotune-no-cudagraphs"
        coordinate_descent_tuning: Enable coordinate descent tuning for Triton kernels
        epilogue_fusion: Fuse pointwise ops into templates (epilogue/prologue)
        triton_cudagraphs: Enable CUDA graphs for Triton kernels
        max_autotune: Enable max autotune for matmul/conv selection
    """

    # torch.compile direct parameters (defaults match PyTorch defaults)
    enabled: bool = False
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool | None = None

    # Mode preset: None (default), "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
    mode: str | None = None

    # Custom options (passed to torch.compile options dict)
    # These are per-model settings, NOT global settings
    coordinate_descent_tuning: bool = False
    epilogue_fusion: bool = True  # PyTorch default is True
    triton_cudagraphs: bool = False
    max_autotune: bool = False

    def get_compile_params(self) -> tuple[str | None, dict | None]:
        """Get mode and options for torch.compile().

        If mode is set, use it directly. Otherwise build options dict from custom settings.

        Returns:
            Tuple of (mode, options). torch.compile accepts either mode OR options, NOT both.
        """
        # If mode is set (non-default), use it directly
        if self.mode is not None and self.mode != "default":
            return self.mode, None

        # No mode preset: build options dict from custom settings
        options = {}
        if self.coordinate_descent_tuning:
            options["coordinate_descent_tuning"] = True
        if not self.epilogue_fusion:
            options["epilogue_fusion"] = False
        if self.triton_cudagraphs:
            options["triton.cudagraphs"] = True
        if self.max_autotune:
            options["max_autotune"] = True

        return None, options if options else None

    def get_compile_kwargs(self) -> dict:
        """Get kwargs dict for torch.compile()."""
        mode, options = self.get_compile_params()

        kwargs = {
            "backend": self.backend,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "disable": not self.enabled,
        }

        if mode is not None:
            kwargs["mode"] = mode
        if options is not None:
            kwargs["options"] = options

        return kwargs

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        valid_modes = (None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid compile mode: {self.mode}. Must be one of {valid_modes}")

        if self.fullgraph and self.dynamic:
            raise ValueError("fullgraph=True and dynamic=True are incompatible")

        valid_backends = ("inductor", "eager", "aot_eager", "aot_nop", "nop")
        if self.backend not in valid_backends:
            raise ValueError(f"Invalid backend: {self.backend}. Must be one of {valid_backends}")


class QuantType(Enum):
    """Available quantization types for online quantization.

    Attributes:
        FP8: Standard FP8 E4M3 format.
        INT8: 8-bit integer quantization.
        MXFP8: Microscaling FP8 (OCP standard).
        MXFP6: Microscaling FP6 (OCP standard).
        MXFP4: Microscaling FP4 (OCP standard).
        NVFP4: NVIDIA FP4 format (Blackwell+).
    """

    FP8 = auto()
    INT8 = auto()
    MXFP8 = auto()
    MXFP6 = auto()
    MXFP4 = auto()
    NVFP4 = auto()


class QuantKernelBackend(Enum):
    """Kernel backends for quantized operations."""

    AUTO = auto()  # Auto-select best available
    TF_KERNEL = auto()  # TeleFuser custom kernel
    VLLM = auto()  # vLLM kernel
    CUTLASS = auto()  # NVIDIA CUTLASS


@dataclass
class QuantConfig:
    """Configuration for online quantization during model loading.

    .. warning::
        This is an interface definition only. The actual quantization
        functionality is NOT yet implemented. This config serves as a
        placeholder for future online quantization support.

    Online quantization converts bf16/fp16 weights to lower precision (FP8/INT8/MXFP4/etc.)
    at load time, reducing memory footprint without requiring pre-quantized checkpoint files.

    Attributes:
        enabled: Whether to enable online quantization.
        quant_type: Target quantization type.
        kernel_backend: Kernel backend for quantized operations.
            AUTO selects the best available backend.
        weight_block_size: Block size for block-wise quantization as (block_n, block_k).
            None disables block-wise quantization (uses per-tensor/per-channel).
            Common values: (128, 128) for FP8, (16, 16) for MX formats.

    Example:
        Future usage (not yet functional)::

            config = ModelRuntimeConfig(
                torch_dtype=torch.bfloat16,
                quant_config=QuantConfig(
                    enabled=True,
                    quant_type=QuantType.FP8,
                    weight_block_size=(128, 128),
                ),
            )
            # Will load bf16 weights and quantize to fp8 at runtime
    """

    enabled: bool = False
    quant_type: QuantType = QuantType.FP8
    kernel_backend: QuantKernelBackend = QuantKernelBackend.AUTO
    weight_block_size: tuple[int, int] | None = None


@dataclass
class ModelRuntimeConfig:
    """Complete runtime configuration for model execution."""

    offload_config: OffloadConfig = field(default_factory=OffloadConfig)
    device_type: str | None = None  # None uses platform default
    device_id: int = 0
    lora_configs: list[LoraConfig] = field(default_factory=list)
    torch_dtype: torch.dtype = torch.bfloat16
    attention_config: AttentionConfig = field(
        default_factory=lambda: AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
    )
    compile_config: CompileConfig = field(default_factory=CompileConfig)
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    ray_config: RayConfig = field(default_factory=RayConfig)
    feature_cache_config: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
