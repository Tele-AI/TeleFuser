# Adding New Pipeline Example Guide

This document explains how to create new pipeline examples in TeleFuser, following the patterns established by existing examples like `wan_video`.

## Overview

Pipeline examples are standalone Python scripts that demonstrate how to use TeleFuser pipelines for inference. Each example should be:

1. Self-contained and runnable
2. Configurable via command-line arguments
3. Compatible with the TeleFuser server (`telefuser serve`)
4. Well-documented with clear naming conventions

## File Structure and Naming

### Directory Organization

Examples are organized by model family:

```
examples/
├── wan_video/              # WanVideo generation examples
│   ├── wan21_*.py          # Wan2.1 model examples
│   ├── wan22_*.py          # Wan2.2 model examples
├── qwen_image/             # Qwen-Image generation examples
├── hunyuan_video/          # HunyuanVideo generation examples
├── z_image/                # Z-Image generation examples
├── liveact/                # LiveAct examples
└── ...
```

### Naming Convention

Follow this pattern: `{model_version}_{feature}_{hardware/config}.py`

| Component | Examples | Description |
|-----------|----------|-------------|
| `model_version` | `wan21_14b`, `wan22_5b`, `qwen_image` | Model family and version |
| `feature` | `t2v`, `i2v`, `t2i`, `lora`, `distill` | Task type or feature |
| `hardware/config` | `h100`, `hf`, `radial`, `cache_calibrate` | Hardware target or special config |

**Examples:**
- `wan21_14b_text_to_video_h100.py` - Wan2.1 14B T2V for H100
- `wan21_1_3b_text_to_video_hf.py` - Wan2.1 1.3B T2V with HF loading
- `wan22_14b_image_to_video_lora_h100.py` - Wan2.2 14B I2V with LoRA

## Example File Structure

A standard example file follows this template:

```python
"""Brief description of what this example does.

Usage:
    python example_name.py --option value
"""

import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.{model_family}.{pipeline_module} import (
    {PipelineClass},
    {PipelineConfigClass},
)
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import save_video  # or save_image for images

# ============================================================================
# Configuration Section
# ============================================================================

PPL_CONFIG = dict(
    name="example_name",
    model_root="/path/to/model",
    negative_prompt="...",
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale=5.0,
    seed=42,
    # ... other parameters
)

# ============================================================================
# Model Loading Section
# ============================================================================

def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """Load and initialize the pipeline.
    
    Args:
        parallelism: Number of parallel GPUs (REQUIRED parameter)
        model_root: Path to model weights (REQUIRED parameter)
        
    Returns:
        Initialized pipeline instance
    """
    module_manager = ModuleManager(device="cpu")
    # Load models...
    
    pipe = PipelineClass(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = PipelineConfigClass()
    # Configure pipeline...
    
    pipe.init(module_manager, pipe_config)
    return pipe

# ============================================================================
# Inference Section
# ============================================================================

def run(pipeline, prompt, negative_prompt="", seed=PPL_CONFIG["seed"], **kwargs):
    """Run inference with the pipeline.
    
    Args:
        pipeline: Loaded pipeline instance
        prompt: Input prompt
        negative_prompt: Negative prompt
        seed: Random seed
        **kwargs: Additional parameters
        
    Returns:
        Generated output (video frames, images, etc.)
    """
    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        # ... other parameters from PPL_CONFIG
    )
    return video

def run_with_file(pipeline, prompt, negative_prompt, seed, output_path, **kwargs):
    """Run inference and save to file (optional, for server compatibility)."""
    output = run(pipeline, prompt, negative_prompt, seed, **kwargs)
    save_video(output, output_path, fps=PPL_CONFIG["target_fps"], quality=6)

# ============================================================================
# CLI Entry Point
# ============================================================================

@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use")
@click.option("--prompt", default="...", help="Input prompt")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model path")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
def main(gpu_num, prompt, model_root, seed):
    """Brief description shown in CLI help."""
    pipe = get_pipeline(gpu_num, model_root)
    
    start = time.time()
    output = run(pipe, prompt, seed=seed)
    elapsed_time = time.time() - start
    
    print(f"Generation time: {elapsed_time:.2f} seconds")
    
    # Save results
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)
    save_video(output, output_path, fps=16, quality=6)
    
    del pipe

if __name__ == "__main__":
    main()
```

## Two Loading Patterns

### Pattern 1: Hash-based Auto-recognition (Recommended for Local Weights)

Use `ModuleManager.load_model()` for local weight files. TeleFuser automatically recognizes model type by hash.

```python
def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """Load and initialize pipeline with hash-based model recognition.
    
    Args:
        parallelism: Number of parallel GPUs (REQUIRED)
        model_root: Path to model weights (REQUIRED)
    """
    module_manager = ModuleManager(device="cpu")
    
    # Load VAE (single file)
    module_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    
    # Load DiT (sharded files - use list)
    dit_path_list = [
        f"{model_root}/diffusion_pytorch_model-00001-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00002-of-00007.safetensors",
        # ...
    ]
    module_manager.load_model(
        dit_path_list,
        torch_dtype=torch.bfloat16,
    )
    
    # Load Text Encoder
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )
    
    # Create and initialize pipeline
    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe.init(module_manager, pipe_config)
    
    return pipe
```

**Key points:**
- `load_model()` accepts single path or list of paths (for sharded models)
- Models are auto-registered by hash, can be fetched by name later
- Model weights loaded on CPU, moved to GPU during `pipe.init()`

### Pattern 2: from_pretrained (Recommended for HF Format)

Use `Pipeline.from_pretrained()` for HuggingFace model IDs or local HF-format folders.

```python
def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """Create pipeline using from_pretrained.
    
    Args:
        parallelism: Number of parallel GPUs (REQUIRED)
        model_root: Path to model weights or HF model ID (REQUIRED)
    """
    model_source = model_root  # HF ID or local path
    
    pipe = Wan21VideoPipeline.from_pretrained(
        model_id_or_path=model_source,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attention_config=AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2),
        enable_clip_stage=False,  # For T2V
        enable_parallel=parallelism > 1,
        parallel_devices=list(range(parallelism)) if parallelism > 1 else None,
    )
    
    return pipe
```

**When to use from_pretrained:**
- HuggingFace model IDs (e.g., `"Wan-AI/Wan2.1-T2V-1.3B"`)
- Local folders in HF Diffusers format
- Quick prototyping and testing
- Server deployment with dynamic model selection

## Configuration Details

### PPL_CONFIG Dictionary

Centralize all default parameters. **Required fields and configuration rules:**

```python
PPL_CONFIG = dict(
    # REQUIRED fields
    name="example_identifier",       # REQUIRED: Pipeline identifier for logging and metrics
    model_root="/path/to/model",     # REQUIRED: Base directory for model files
    
    # Generation parameters
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale=5.0,
    seed=42,
    
    # Quality settings
    negative_prompt="...",
    sigma_shift=5.0,
    
    # Output settings
    target_fps=16,
    
    # Runtime settings
    tiled=False,
    sample_solver="unipc",
    attn_impl=AttnImplType.TORCH_SDPA,
)
```

**Configuration rules:**

| Rule | Description |
|------|-------------|
| `name` | **Required**. Used for logging, metrics, and pipeline identification. Should be descriptive like `"wan21_1.3B_t2v_h100"` |
| `model_root` | **Required**. Base directory containing all model files. Can be overridden via CLI `--model_root` |
| Model file paths | Use relative filenames under `model_root`, e.g., `dit_filename`, `vae_filename`. Special models can use absolute paths if needed |

**Example with special model paths:**

```python
PPL_CONFIG = dict(
    name="wan22_14B_i2v_h100",
    model_root="/nvfile/model_zoo/Wan2.2-I2V-A14B",
    # Standard models under model_root
    dit_filename="dit_model.safetensors",
    vae_filename="vae.pth",
    # Special model with absolute path (e.g., shared across pipelines)
    text_encoder_path="/shared/models/t5_umt5-xxl-enc-bf16.pth",
    # ... other parameters
)
```

### Pipeline Configuration

Configure runtime behavior through `PipelineConfig`:

```python
pipe_config = Wan21VideoPipelineConfig()

# Attention implementation
pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)

# CPU Offloading
pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD

# Sampling solver
pipe_config.sample_solver = "euler"

# Stage toggles
pipe_config.enable_clip_stage = True  # For I2V models
```

### Parallel Configuration

Configure multi-GPU inference:

```python
if parallelism > 1:
    cfg_scale = PPL_CONFIG["cfg_scale"]
    
    if cfg_scale > 1:
        # CFG parallel + Ulysses SP
        pipe_config.dit_config.parallel_config.cfg_degree = 2
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism // 2
    else:
        # Pure Ulysses SP
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
    
    pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
    pipe_config.enable_denoising_parallel = True
```

**Parallel strategies:**

| Parallelism | cfg_scale > 1 | cfg_scale = 1 |
|-------------|---------------|---------------|
| 2 GPUs | cfg_degree=2, sp=1 | cfg_degree=1, sp=2 |
| 4 GPUs | cfg_degree=2, sp=2 | cfg_degree=1, sp=4 |
| 8 GPUs | cfg_degree=2, sp=4 | cfg_degree=1, sp=8 |

### Feature Cache Configuration

Enable caching for acceleration:

```python
from telefuser.core.config import FeatureCacheConfig

if enable_feature_cache:
    pipe_config.dit_config.feature_cache_config = FeatureCacheConfig(
        enabled=True,
        model_type="Wan2_2-I2V-A14B",
    )
```

### LoRA Configuration

Add LoRA support:

```python
from telefuser.core.config import LoraConfig

pipe_config.dit_config.lora_config = LoraConfig(
    lora_path="/path/to/lora_weights.safetensors",
    lora_scale=1.0,
)
```

## Server Compatibility

Examples can be served via `telefuser serve`:

```bash
telefuser serve examples/wan_video/wan21_1_3b_text_to_video_hf.py --task t2v
```

### Required Functions for Server

The server expects these functions:

```python
def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """Must return initialized pipeline.
    
    REQUIRED parameters:
        - parallelism: Number of parallel GPUs
        - model_root: Path to model weights
    """
    pass

def run(pipeline, prompt, negative_prompt="", **kwargs):
    """Must return generation output."""
    pass

def run_with_file(pipeline, prompt, negative_prompt, seed, output_path, **kwargs):
    """Optional: Run and save to file."""
    pass
```

### Environment Variables

Use environment variables for configurable paths:

```python
model_root = os.getenv("MODEL_ROOT", "/default/path")
output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
```

## Best Practices

### 1. Clear Documentation

Add docstring at the top explaining usage:

```python
"""Wan2.1 14B Text-to-Video (T2V) example.

This example demonstrates text-to-video generation using Wan2.1 14B model.

Usage:
    python wan21_14b_text_to_video_h100.py --prompt "A cat playing piano"
    python wan21_14b_text_to_video_h100.py --gpu_num 2 --prompt "..."
"""
```

### 2. Meaningful Default Prompts

Provide interesting default prompts that showcase model capabilities:

```python
@click.option(
    "--prompt",
    default="A stylish woman walking down a Tokyo street filled with warm golden sunlight...",
    help="Positive guidance text prompt",
)
```

### 3. Consistent Parameter Naming

Follow established naming conventions:

| Parameter | Description |
|-----------|-------------|
| `gpu_num` | Number of GPUs |
| `prompt` | Positive prompt |
| `negative_prompt` | Negative prompt |
| `resolution` | 480p, 720p, etc. |
| `seed` | Random seed |
| `model_root` | Model path |
| `aspect_ratio` | 16:9, 4:3, 1:1 |

### 4. Proper Cleanup

Clean up resources at the end:

```python
def main(...):
    pipe = get_pipeline(...)
    output = run(pipe, ...)
    save_video(output, ...)
    del pipe  # Free GPU memory
```

### 5. Timing and Logging

Report generation time:

```python
start = time.time()
output = run(pipe, ...)
elapsed_time = time.time() - start
print(f"Generation time: {elapsed_time:.2f} seconds")
```

### 6. Output Naming

Use `get_example_name()` for consistent output naming:

```python
filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
```

## Complete Example Reference

For complete implementations, refer to:

| Example | Features | File |
|---------|----------|------|
| Basic T2V | Hash-based loading, parallel | `wan21_14b_text_to_video_h100.py` |
| Basic I2V | Image input, CLIP stage | `wan21_14b_image_to_video_h100.py` |
| HF Loading | from_pretrained, simple setup | `wan21_1_3b_text_to_video_hf.py` |
| LoRA | LoRA configuration | `wan21_14b_image_to_video_lora_h100.py` |
| Feature Cache | Caching acceleration | `wan22_14b_image_to_video_h100.py` |
| Distill | Dual DiT (high/low noise) | `wan22_14b_image_to_video_distill_h100.py` |

## Related Documentation

- [Adding New Model](./adding_new_model.md) - Model implementation guide
- [Adding New Stage](./adding_new_stage.md) - Stage implementation guide
- [Configuration](./configuration.md) - Configuration details
- [Parallel Inference](./parallel.md) - Multi-GPU configuration
- [CPU Offloading](./offload.md) - Memory optimization
- [Service](./service.md) - Server deployment