# CPU Offloading

CPU offloading is a memory optimization technique that reduces GPU VRAM usage by temporarily moving model weights from GPU to CPU memory during inference. TeleFuser provides multiple offloading strategies to balance memory usage and inference speed.

## Offloading Strategies

TeleFuser supports four offloading strategies, configured via `WeightOffloadType`:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `NO_CPU_OFFLOAD` | No offloading, all weights stay in GPU | GPU memory is sufficient |
| `MODEL_CPU_OFFLOAD` | Offload entire model to CPU between stages | Moderate memory constraints |
| `SEQUENTIAL_CPU_OFFLOAD` | Layer-by-layer offloading during forward pass | Severe memory constraints |
| `ASYNC_CPU_OFFLOAD` | Asynchronous layerwise offloading with prefetching | Best balance of speed and memory |

## Async CPU Offload

`ASYNC_CPU_OFFLOAD` is the recommended strategy for most scenarios. It uses `AsyncOffloadManager` to:

- **Offload per-layer weights** from GPU to pinned CPU memory
- **Prefetch upcoming layers** asynchronously using a dedicated CUDA stream
- **Overlap data transfer** with computation for minimal latency

### How Async Offload Works

```
Time ──────────────────────────────────────────────►

Layer 0: [Load]──[Compute]────────────────────────────
Layer 1:      [Async Load]──[Compute]─────────────────
Layer 2:           [Async Load]──[Compute]────────────
Layer 3:                [Async Load]──[Compute]───────

Data transfer (load) overlaps with computation, hiding latency
```

### Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `offload_type` | `WeightOffloadType` | `NO_CPU_OFFLOAD` | Offloading strategy |
| `pin_cpu_memory` | bool | `True` | Use pinned memory for faster H2D transfer |
| `offload_ratio` | float | `1.0` | Ratio of layers to offload (1.0 = all layers) |
| `prefetch_size` | int | `1` | Number of layers to prefetch ahead |

### AsyncOffloadManager Lazy GPU Cache

`lazy_gpu_cache` is a lower-level `AsyncOffloadManager` constructor option,
not an `OffloadConfig` field. It controls whether GPU buffers are
pre-allocated during manager initialization:

- **`lazy_gpu_cache=False` (default)**: GPU buffer pool is allocated during initialization
- **`lazy_gpu_cache=True`**: GPU buffer pool is allocated on first use (saves VRAM during initialization)

Use `lazy_gpu_cache=True` when:
- GPU memory is extremely limited during pipeline initialization
- You want to defer VRAM allocation until inference starts

Use `allocate_gpu_cache()` and `cleanup_gpu_cache()` for manual control:

```python
# Example: Manual GPU cache management
from telefuser.offload.async_offload import AsyncOffloadManager

# Initialize with lazy_gpu_cache=True
manager = AsyncOffloadManager(layers, lazy_gpu_cache=True)

# Manually allocate when ready
manager.allocate_gpu_cache()

# After inference, release cache to free VRAM
manager.cleanup_gpu_cache()
```

## Usage in Pipelines

### Basic Configuration

```python
from telefuser.core.config import OffloadConfig, WeightOffloadType
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

# Create pipeline configuration
pipe_config = Wan21VideoPipelineConfig()

# Enable async offload for DiT (most memory-intensive component)
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=1,
)

# Optionally enable offload for other stages
pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
```

### WanVideo Example

Complete example for Wan2.1 video generation with CPU offloading:

```python
import torch
from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    OffloadConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)

def get_pipeline(model_root, parallelism=1):
    """Initialize Wan2.1 pipeline with CPU offloading."""
    
    # Load models to CPU first
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth"],
        torch_dtype=torch.bfloat16,
    )
    
    # Create pipeline
    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    
    # Configure attention
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
        AttnImplType.SAGE_ATTN_2_8_8
    )
    
    # Configure offloading for different stages
    # DiT: Use async layerwise offload (best for large transformer)
    pipe_config.dit_config.offload_config = OffloadConfig(
        offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
        pin_cpu_memory=True,
        offload_ratio=1.0,
        prefetch_size=1,
    )
    
    # VAE: Use model-level offload (simpler, less frequent transfer)
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    
    # Text encoder: Use model-level offload
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    
    # Optional: Enable distributed inference
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = 2
        pipe_config.enable_denoising_parallel = True
    
    # Initialize pipeline
    pipe.init(module_manager, pipe_config)
    return pipe

# Usage
model_root = "/path/to/Wan2.1-T2V-1.3B"
pipe = get_pipeline(model_root, parallelism=1)

# Generate video
video = pipe(
    prompt="A cat playing piano",
    num_inference_steps=40,
    num_frames=81,
    height=480,
    width=832,
)
```

### Large Model Example (14B+)

For large models like Wan2.1-14B, offloading is essential:

```python
# Configuration for Wan2.1-14B (720P)
pipe_config = Wan21VideoPipelineConfig()

# Use async offload with larger prefetch for better overlap
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=2,  # Prefetch 2 layers ahead
    offload_ratio=1.0,
)

# Enable offloading for all auxiliary models
pipe_config.clip_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
```

## Performance Considerations

### Memory vs Speed Trade-off

| Strategy | VRAM Savings | Speed Impact | Recommendation |
|----------|--------------|--------------|----------------|
| `NO_CPU_OFFLOAD` | None | Fastest | 24GB+ VRAM |
| `MODEL_CPU_OFFLOAD` | High (~50%) | Moderate | 16-24GB VRAM |
| `ASYNC_CPU_OFFLOAD` | High (~60-70%) | Low | 8-16GB VRAM |
| `SEQUENTIAL_CPU_OFFLOAD` | Maximum | Slowest | <8GB VRAM |

### Tuning Prefetch Size

The `prefetch_size` parameter affects the overlap between data transfer and computation:

- **`prefetch_size=1`**: Default, balanced for most models
- **`prefetch_size=2+`**: Better overlap for larger layers, but more VRAM usage

```python
# For very large layers (e.g., 14B models)
pipe_config.dit_config.offload_config.prefetch_size = 2
```

### Qwen-Image H100 Benchmark Evidence

The following measurements validate the prefetch and compute-overlap path in
`AsyncOffloadManager`; they are not general performance guarantees for other models, GPUs,
or CPU-GPU interconnects. Tests used one NVIDIA H100 80GB and PyTorch 2.11.0+cu130. Each
configuration ran one complete warmup followed by one complete generation. DiT time was measured
with explicit CUDA synchronization at stage entry and exit, so it includes asynchronous weight
transfer, computation, and offload in that stage.

#### FP8 Lightning

Qwen-Image-2512-Lightning FP8 weights, 1328×1328, 16 steps, CFG=1, and
`offload_ratio=0.5`:

| Mode / `prefetch_size` | DiT time | Peak allocated VRAM |
|---|---:|---:|
| `NO_CPU_OFFLOAD` | 4.360 s | 40.55 GiB |
| `ASYNC_CPU_OFFLOAD`, 1 | 5.184 s | 31.61 GiB |
| `ASYNC_CPU_OFFLOAD`, 2 | 4.966 s | 32.24 GiB |
| `ASYNC_CPU_OFFLOAD`, 4 | 4.570 s | 33.50 GiB |
| `ASYNC_CPU_OFFLOAD`, 8 | 4.469 s | 36.04 GiB |
| `ASYNC_CPU_OFFLOAD`, 12 | 4.465 s | 38.57 GiB |

`prefetch_size=8` is only 2.5% slower than no offload while using 4.51 GiB less VRAM;
increasing it to 12 provides essentially no speed benefit. As an overlap control, forcing the
compute stream to wait for the copy stream after every prefetch made the same FP8
`prefetch_size=2` configuration take 6.039 s, versus 4.966 s on the normal asynchronous path.
This demonstrates effective H2D/compute overlap in the normal path.

#### Original Cache-DiT Comparison

To compare scheduling granularity, the [original Cache-DiT `layerwise_cpu_offload` implementation](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/offload/layerwise.py) was applied to the full DiT with its default leaf-module selection. Under the same FP8 model, input, warmup, and timing protocol, Cache-DiT selected 1,087 leaf targets, kept 543 persistent targets (about 50%), and used `transfer_buckets=4` with its bounded prefetch window.

| Implementation | DiT time | Peak allocated VRAM |
|---|---:|---:|
| TeleFuser block-level `ASYNC_CPU_OFFLOAD`, `prefetch_size=4` | 4.570 s | 33.50 GiB |
| Original Cache-DiT leafwise, `transfer_buckets=4` | 9.695 s | 31.20 GiB |

Cache-DiT used 2.30 GiB less peak VRAM, yet TeleFuser's DiT was still 2.12× faster (52.9% less time). For Qwen-Image, using transformer blocks as the transfer, buffer-reuse, and prefetch unit avoids the small-grained H2D, parameter-rebinding, and allocation overhead of 1,087 leaf-module targets.

#### Base BF16 Qwen-Image

Qwen-Image-2512 BF16 weights, 1664×928, 50 steps, and CFG=4:

| `offload_ratio` | `prefetch_size` | DiT time | Peak allocated VRAM |
|---:|---:|---:|---:|
| No offload | — | 26.990 s | 55.84 GiB |
| 0.5 | 2 | 46.253 s | 39.00 GiB |
| 0.5 | 4 | 42.470 s | 44.81 GiB |
| 0.5 | 8 | 39.954 s | 49.88 GiB |
| 0.6 | 8 | 47.473 s | 46.08 GiB |
| 0.75 | 8 | 56.335 s | 40.38 GiB |

For this model, increasing prefetch depth improves speed: at 50% offload, moving from two to
eight prefetched blocks reduces DiT time by 13.6% but raises peak VRAM by 10.88 GiB. Raising
`offload_ratio` reduces resident layers and VRAM, but also increases the H2D bytes required by
every denoising step. Even with eight prefetched blocks, 60% and 75% offload are slower than the
50%/two-block baseline. Tune `prefetch_size` first at a fixed `offload_ratio`, then choose the
memory-speed trade-off within the VRAM budget.


### Pinned Memory

Setting `pin_cpu_memory=True` (default) uses page-locked memory for faster H2D transfers:

- **Enabled**: Faster transfer, slightly more CPU memory usage
- **Disabled**: Slower transfer, less CPU memory usage

## Troubleshooting

### Out of Memory During Initialization

If GPU OOM occurs during pipeline initialization:

```python
# Use the most conservative pipeline offload configuration exposed by OffloadConfig
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    offload_ratio=1.0,
    prefetch_size=1,
)
```

If you need lazy GPU cache allocation specifically, wire `AsyncOffloadManager`
directly or through a pipeline integration that passes `lazy_gpu_cache=True`;
the current `OffloadConfig` dataclass does not expose that option.

### Slow Inference

If offloading causes significant slowdown:

1. **Increase prefetch size** for better overlap
2. **Reduce offload_ratio** to keep more layers resident
3. **Check CPU-GPU interconnect** (PCIe bandwidth matters)

```python
# Keep 20% of layers resident in GPU
pipe_config.dit_config.offload_config.offload_ratio = 0.8
```

### CPU Memory Issues

If CPU memory is insufficient:

```python
# Disable pinned memory
pipe_config.dit_config.offload_config.pin_cpu_memory = False
```

## API Reference

### OffloadConfig

```python
@dataclass
class OffloadConfig:
    offload_type: WeightOffloadType = WeightOffloadType.NO_CPU_OFFLOAD
    pin_cpu_memory: bool = True
    offload_ratio: float = 1.0
    prefetch_size: int = 1
```

### AsyncOffloadManager

```python
class AsyncOffloadManager:
    def __init__(
        self,
        layers: torch.nn.ModuleList,
        device: torch.device | None = None,
        *,
        enabled: bool = True,
        pin_cpu_memory: bool = True,
        offload_ratio: float = 1,
        prefetch_size: int = 1,
        lazy_gpu_cache: bool = False,
    ) -> None
    
    def allocate_gpu_cache(self) -> None:
        """Manually allocate GPU cache."""
        
    def cleanup_gpu_cache(self) -> None:
        """Release GPU cache."""
        
    def disable_offload(self) -> None:
        """Disable offloading and load all layers."""
        
    def enable_offload(self) -> None:
        """Re-enable offloading."""
```

## Sequential CPU Offload

For scenarios requiring fine-grained VRAM management, TeleFuser provides `enable_sequential_cpu_offload` - a layer-by-layer offloading mechanism that wraps individual modules with smart state management.

### Three-State System

Each wrapped module operates in one of three states:

| State | Value | Data Location | Description |
|-------|-------|---------------|-------------|
| **Offload** | `0` | `offload_device` (usually CPU) | Default state, minimal VRAM usage |
| **Onload** | `1` | `onload_device` (usually GPU) | Loaded but may use different dtype |
| **Keep** | `2` | `computation_device` (GPU) | Pinned in GPU for repeated use |

### State Transition Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      Forward Pass Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  if state == 2 (Keep):                                          │
│      → Use weights directly (fastest)                           │
│                                                                  │
│  elif onload config == computation config:                      │
│      → Use weights directly (no conversion needed)              │
│                                                                  │
│  elif vram_limit is set and GPU has free memory:                │
│      → Call keep() to promote to state 2                        │
│      → Use weights directly                                     │
│                                                                  │
│  else:                                                          │
│      → cast_to() temporary copy to GPU                          │
│      → Compute and release (state unchanged)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Usage

```python
from telefuser.offload import enable_sequential_cpu_offload, AutoWrappedLinear

# Define which modules to wrap
module_map = {
    torch.nn.Linear: AutoWrappedLinear,
}

# Configure dtype and device for each state
module_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# Enable sequential offloading
enable_sequential_cpu_offload(
    model,
    module_map=module_map,
    module_config=module_config,
    vram_limit=20.0,  # GB - promotes to Keep state when VRAM available
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | - | Model to enable offloading on |
| `module_map` | `dict` | - | Mapping from source module type to wrapper class |
| `module_config` | `dict` | - | Configuration for dtype/device in each state |
| `max_num_param` | `int/None` | `None` | Parameter threshold for using overflow config |
| `overflow_module_config` | `dict/None` | `None` | Alternative config for layers exceeding threshold |
| `vram_limit` | `float/None` | `None` | VRAM limit (GB) for automatic state promotion |

### Module Configuration

The `module_config` dictionary controls data placement:

```python
module_config = {
    # Offload state (state=0) - minimal VRAM
    "offload_dtype": torch.float32,    # FP32 for CPU storage
    "offload_device": "cpu",            # Keep on CPU
    
    # Onload state (state=1) - ready for use
    "onload_dtype": torch.bfloat16,     # Lower precision for GPU
    "onload_device": "cuda",            # Load to GPU
    
    # Computation state (state=2) - actual computation
    "computation_dtype": torch.bfloat16,  # Must match onload for promotion
    "computation_device": "cuda",         # Must be GPU
}
```

### Available Wrappers

| Wrapper | Source Module | Description |
|---------|---------------|-------------|
| `AutoWrappedModule` | `nn.Module` | Generic wrapper for any module |
| `AutoWrappedLinear` | `nn.Linear` | Optimized Linear layer with LoRA support |
| `WanAutoCastLayerNorm` | `nn.LayerNorm` | LayerNorm with automatic mixed precision |

### Layer-wise Configuration

Different configurations for different parameter sizes:

```python
# Standard config for most layers
base_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# Config for large layers (always stay on CPU)
overflow_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.float32,
    "onload_device": "cpu",  # Never load to GPU
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

enable_sequential_cpu_offload(
    model,
    module_map={nn.Linear: AutoWrappedLinear},
    module_config=base_config,
    max_num_param=1_000_000_000,  # 1B parameter threshold
    overflow_module_config=overflow_config,
    vram_limit=22.0,
)
```

### Manual State Control

After enabling, you can manually control module states:

```python
# Manual state transitions
for module in model.modules():
    if hasattr(module, 'offload'):
        module.offload()   # Force to state 0 (CPU)
        module.onload()    # Force to state 1 (onload device)
        module.keep()      # Force to state 2 (GPU)

# Check current state
if hasattr(module, 'state'):
    print(module.state)  # 0=offload, 1=onload, 2=keep
```

### vram_limit Behavior

The `vram_limit` parameter controls automatic state promotion:

| Setting | Behavior |
|---------|----------|
| `None` (default) | Conservative mode - never promotes to Keep state, always uses temporary cast |
| `20.0` | When VRAM usage < 20GB, promotes frequently used modules to Keep state |

**Recommendation**: Always set `vram_limit` for production use to improve performance.

### API Reference

```python
def enable_sequential_cpu_offload(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param: int | None = None,
    overflow_module_config: dict | None = None,
    vram_limit: float | None = None,
) -> None

class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
        vram_limit,
        name: str = "",
    )
    
    def offload(self) -> None:   # Switch to state 0
    def onload(self) -> None:    # Switch to state 1
    def keep(self) -> None:      # Switch to state 2
```

## References

- The async offloading implementation is adapted from [SGLang](https://github.com/sgl-project/sglang)'s layerwise offload utility.
- The sequential CPU offloading implementation is adapted from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
