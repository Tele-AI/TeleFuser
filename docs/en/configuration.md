# TeleFuser Configuration System

This document describes TeleFuser's three-layer configuration architecture for Python API interfaces.

## Overview

TeleFuser adopts a three-layer configuration architecture, separating concerns between model definition, inference algorithm settings, and user-adjustable parameters:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 1: Model Definition (Model-Weight Bound)                     │
│  Fixed after model loading, bound to weight files                   │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: Inference Algorithm Parameters (PipelineConfig)           │
│  Configured in PipeConfig and fixed in Pipeline.__call__ interface  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: User-Adjustable Parameters (Example run())                │
│  Exported in example file run() function                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Layer 1: Model Definition

**Location**: Model loading phase, bound to weight files

### Fixed Attributes

Attributes determined by model weights:

| Attribute | Description | Determined By |
|-----------|-------------|---------------|
| `distill` | Whether it's a distilled model | Weight file |
| `MoE` | Mixture of Experts architecture | Loading multiple DiT models |
| `fp8/bf16` | Quantization type | Weight file format |
| `meanflow` | FlowMatch type | Model architecture |

### Example

```python
# examples/wan_video/wan22_14b_image_to_video_distill_h100.py
module_manager = ModuleManager(device="cpu")

# Load distill model (fixed attribute: distill=True)
module_manager.load_model(
    f"{model_root}/dit_high_noise_distill_model_bf16_1022_ecab7.safetensors",
    torch_dtype=torch.bfloat16,
)
module_manager.load_model(
    f"{model_root}/dit_low_noise_distill_model_bf16_1022_200c2.safetensors",
    torch_dtype=torch.bfloat16,
)
```

**Key Point**: Model intrinsic properties are bound to weights and cannot be changed after loading.

## Layer 2: Inference Algorithm Parameters

**Location**: `PipelineConfig` dataclass + `Pipeline.__call__()` method

### PipelineConfig Definition

```python
@dataclass
class Wan21VideoPipelineConfig:
    """Configuration for Wan2.1 video generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    clip_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"              # Sampler type
    enable_clip_stage: bool = False
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vfi: bool = False                  # Video frame interpolation
```

### Pipeline.__call__ Fixed Parameters

```python
def __call__(
    self,
    prompt: str | List[str],
    ...
    sigma_shift: float = 5.0,         # Noise schedule parameter
    boundary: float = 0.875,          # MoE switching boundary
    tiled: bool = False,              # Tiled inference
    tile_size: tuple[int, int] = (30, 52),
    ...
)
```

### Example Configuration

```python
# examples/wan_video/wan21_1_3b_text_to_video_h100.py
PPL_CONFIG = dict(
    name="wan21_1.3B_t2v_h100",
    negative_prompt="...",
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    sigma_shift=8.0,
    enable_vfi=True,
)

def get_pipeline(parallelism=1, model_root="..."):
    pipe_config = Wan21VideoPipelineConfig()
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipe_config.enable_vfi = PPL_CONFIG["enable_vfi"]
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    ...
    pipe.init(module_manager, pipe_config)
```

**Key Point**: Layer 2 parameters are defined by developers in example files, not exposed to end users.

## Layer 3: User-Adjustable Parameters

**Location**: `run()` function in example files

### Example Interface

```python
def run(
    pipeline,
    prompt,                    # User input
    negative_prompt="",        # User input
    seed=42,                   # Adjustable
    resolution="480p",         # Adjustable
    aspect_ratio="16:9",       # Adjustable
):
    """Generate video from text prompt.

    Args:
        pipeline: Preloaded pipeline object
        prompt: Positive guidance text prompt
        negative_prompt: Negative guidance prompt
        seed: Random seed
        resolution: Resolution such as 720p, 480p
        aspect_ratio: Aspect ratio such as 16:9
    """
    width, height = get_target_video_size_from_ratio(aspect_ratio, resolution=resolution)
    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        height=height,
        width=width,
        ...
    )
    return video
```

**Key Point**: Layer 3 parameters are exposed to end users and can be modified per inference call.

## Configuration Flow Diagram

```
Model Weights (Layer 1)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Model Attributes: distill/MoE, quantization type, etc.         │
│  (Fixed, determined by weight files)                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼ module_manager.load_model()
┌─────────────────────────────────────────────────────────────────┐
│  get_pipeline()                                                 │
│  ├─ PipeConfig: sample_solver, parallel, offload (Layer 2)      │
│  └─ pipe.init(module_manager, pipe_config)                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  run() User Interface (Layer 3)                                 │
│  ├─ User inputs: prompt, seed, resolution, aspect_ratio         │
│  └─ Fixed values from PPL_CONFIG: num_inference_steps, cfg_scale│
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
     pipeline(prompt, seed, height, width, ...)
```

## Configuration Differences by Model Type

| Model Type | Layer 1 (Model) | Layer 2 (Algorithm) | Layer 3 (User) |
|------------|-----------------|---------------------|----------------|
| **Wan21 1.3B** | Single DiT | `sample_solver="euler"` | prompt, seed, resolution |
| **Wan22 A14B** | MoE (high+low DiT) | `boundary=0.875`, `cfg_scale_high/low` | Same as above |
| **Wan22 Distill** | distill weights | `cfg_scale=1.0` (no CFG needed) | Same as above |
| **HunyuanVideo + SR** | base DiT + SR DiT | `enable_sr=True`, `lq_noise_strength` | Same as above |

## Design Principles

1. **Layer 1 is Immutable**: Fixed after model loading, bound to weights
2. **Layer 2 is Semi-Fixed**: Defined in `PPL_CONFIG` within example files, controlled by developers
3. **Layer 3 is Variable**: Exposed to end users, modifiable per inference

This layered design achieves **separation of concerns**:
- Model researchers focus on Layer 1
- Algorithm engineers focus on Layer 2
- Application users focus on Layer 3

## Core Configuration Classes

Located in `telefuser/core/config.py`:

### ModelRuntimeConfig

Top-level configuration for model execution:

```python
@dataclass
class ModelRuntimeConfig:
    """Complete runtime configuration for model execution."""

    offload_config: OffloadConfig = field(default_factory=OffloadConfig)
    device_type: str | None = None
    device_id: int = 0
    lora_configs: list[LoraConfig] = field(default_factory=list)
    torch_dtype: torch.dtype = torch.bfloat16
    attention_config: AttentionConfig = field(default_factory=lambda: AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))
    compile_config: CompileConfig = field(default_factory=CompileConfig)
    quant_config: QuantConfig = field(default_factory=QuantConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    ray_config: RayConfig = field(default_factory=RayConfig)
    feature_cache_config: FeatureCacheConfig = field(default_factory=FeatureCacheConfig)
```

### ParallelConfig

Distributed parallel processing configuration:

```python
@dataclass
class ParallelConfig:
    """Distributed parallel processing configuration."""

    device_ids: list | None = None
    dp_degree: int = 1           # Data parallelism
    cfg_degree: int = 1          # CFG parallelism
    sp_ulysses_degree: int = 1   # Ulysses sequence parallelism
    sp_ring_degree: int = 1      # Ring attention sequence parallelism
    pp_degree: int = 1           # Pipeline parallelism
    tp_degree: int = 1           # Tensor parallelism
    enable_fsdp: bool = False

    def validate(self) -> None:
        """Validate that device count matches parallelism degrees."""
        ...
```

### AttentionConfig

Attention implementation configuration:

```python
@dataclass
class AttentionConfig:
    """Unified configuration for all attention implementations."""

    attn_impl: AttnImplType = AttnImplType.TORCH_SDPA
    sparse_config: SparseAttentionConfig | None = None

    @classmethod
    def radial_attention(cls, ...) -> AttentionConfig:
        """Create config for radial attention (sparse attention for video)."""
        ...

    @classmethod
    def dense_attention(cls, attn_impl: AttnImplType = AttnImplType.FLASH_ATTN_2) -> AttentionConfig:
        """Create config for dense attention."""
        ...
```

## Config Dump

TeleFuser provides a `dump_config()` method to export pipeline configuration for reproducibility and debugging.

### Use Cases

- **Reproducibility**: Capture exact configuration used for generation
- **Debugging**: Inspect effective configuration
- **Deployment**: Share configuration between environments

### Usage

```python
# After pipeline initialization
pipe = get_pipeline(parallelism=1, model_root="...")

# Dump to file
pipe.dump_config("output/pipeline_config.json")

# Or get dict directly
config = pipe.dump_config()
print(config["layer1_model_definition"]["models"])
```

### Output Format

The output JSON contains two main layers:

```json
{
  "version": "1.0",
  "timestamp": "2026-03-20T10:30:00",
  "pipeline_type": "Wan21VideoPipeline",
  "device": "cuda",
  "torch_dtype": "bfloat16",
  "layer1_model_definition": {
    "models": [
      {
        "name": "wan_video_vae",
        "path": "/dev/shm/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        "class": "TAEHV"
      },
      {
        "name": "wan_video_dit",
        "path": "/dev/shm/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "class": "WanModel"
      }
    ]
  },
  "layer2_inference_config": {
    "sample_solver": "euler",
    "enable_vfi": false,
    "vae_config": {
      "torch_dtype": "bfloat16",
      "attention_config": {
        "attn_impl": "TORCH_SDPA"
      },
      "offload_config": {
        "offload_type": "NO_CPU_OFFLOAD"
      }
    },
    "dit_config": { ... }
  }
}
```

### Implementation Details

- **Layer 1**: Model paths and class names are captured during `init()`
- **Layer 2**: PipelineConfig dataclass is serialized recursively
- **Memory efficient**: Only stores model info (name, path, class), not model weights

## Related Documentation

- [Model Loading Guide](./model_loading.md)
- [Parallel Configuration Guide](./parallel.md)
- [Attention Configuration Guide](./attention.md)
- [Offload Configuration Guide](./offload.md)
