# WanVideo Example

Video generation using Wan2.1 and Wan2.2 models for Text-to-Video and Image-to-Video tasks.

## Model Source

### Wan2.1 Models

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| Wan2.1-T2V-1.3B | [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | [Wan-AI/Wan2.1-T2V-1.3B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B) |
| Wan2.1-T2V-14B | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | [Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1-I2V-14B-720P | [Wan-AI/Wan2.1-I2V-14B-720P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) | [Wan-AI/Wan2.1-I2V-14B-720P](https://modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P) |

### Wan2.2 Models

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| Wan2.2-T2V-14B | [Wan-AI/Wan2.2-T2V-14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-14B) | [Wan-AI/Wan2.2-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-14B) |
| Wan2.2-I2V-A14B | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | [Wan-AI/Wan2.2-I2V-A14B](https://modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) |
| Wan2.2-TI2V-5B | [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | [Wan-AI/Wan2.2-TI2V-5B](https://modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B) |

### Other

| Model | HuggingFace | ModelScope |
|-------|-------------|------------|
| RIFE v4.26 | Video Frame Interpolation | [RIFEv4.26](https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip) |

## Feature Support

| Feature | Wan2.1 | Wan2.2 |
|---------|--------|--------|
| CFG Parallel (CFGP) | ✔️ | ✔️ |
| Ulysses Sequence Parallel (USP) | ✔️ | ✔️ |
| LoRA | ✔️ | ✔️ |
| FP8 Quantization | ✔️ | ✔️ |
| FSDP | ✔️ | ✔️ |
| Encoder Parallel | ✔️ | ✔️ |
| Async Pipeline | ✔️ | ✔️ |
| Feature Cache (AdaTaylor) | ✔️ | ✔️ |
| Distilled Model | ❔ | ✔️ |
| First-Last-Frame to Video (FL2V) | ❌ | ✔️ |
| Server API | ✔️ | ✔️ |

## Parallel Configuration

The parallel config is automatically set based on `cfg_scale`:

| cfg_scale | cfg_degree | sp_ulysses_degree |
|-----------|------------|-------------------|
| > 1 | 2 | parallelism // 2 |
| == 1 | 1 | parallelism |

For Wan2.2 dual-branch models (dit_high/dit_low), each branch is configured independently based on its own `cfg_scale_high`/`cfg_scale_low` value.

**Example:**
```python
# For cfg_scale=5.0 with parallelism=4:
# cfg_degree=2, sp_ulysses_degree=2
# Total parallelism = cfg_degree * sp_ulysses_degree = 4

# For cfg_scale=1.0 (distilled) with parallelism=4:
# cfg_degree=1, sp_ulysses_degree=4
# Total parallelism = 1 * 4 = 4
```

## Files

### Text-to-Video Examples

#### wan21_1_3b_text_to_video_h100.py

Basic T2V generation with Wan2.1 1.3B model.

**Purpose:** Standard text-to-video generation with optional Video Frame Interpolation (VFI).

**Usage:**
```bash
# Basic usage
python examples/wan_video/wan21_1_3b_text_to_video_h100.py --prompt "A cat playing with a ball"

# Multi-GPU
python examples/wan_video/wan21_1_3b_text_to_video_h100.py --gpu_num 2 --prompt "A cat playing"

# Custom resolution
python examples/wan_video/wan21_1_3b_text_to_video_h100.py --resolution 480p --aspect_ratio 16:9
```

**Features:**
- Video Frame Interpolation (VFI) with RIFE model for 30fps output
- CFG parallel when cfg_scale > 1

#### wan21_1_3b_text_to_video_hf.py

T2V with HuggingFace format loading.

**Purpose:** Simplified loading using `from_pretrained()` method.

**Usage:**
```bash
# Using HF Model ID (auto-download)
python examples/wan_video/wan21_1_3b_text_to_video_hf.py --model_source "Wan-AI/Wan2.1-T2V-1.3B"

# Using local path
python examples/wan_video/wan21_1_3b_text_to_video_hf.py --model_source "/path/to/Wan2.1-T2V-1.3B"
```

#### wan21_1_3b_text_to_video_ada_taylor_cache.py

T2V with AdaTaylorCache V2 feature caching.

**Purpose:** Accelerate generation using feature caching for faster inference.

**Usage:**
```bash
python examples/wan_video/wan21_1_3b_text_to_video_ada_taylor_cache.py \
    --enable_feature_cache \
    --n_derivatives 1 \
    --taylor_threshold 2
```

**Features:**
- Adaptive skip logic based on error accumulation
- Hybrid strategy: Taylor series for small skips, residual reuse for large skips
- Better quality-speed trade-off

**Configuration:**
Feature cache is configured during pipeline initialization via `ModelRuntimeConfig.feature_cache_config`:
```python
from telefuser.core.config import FeatureCacheConfig

pipe_config.dit_config.feature_cache_config = FeatureCacheConfig(
    enabled=True,
    model_type="Wan2.1-T2V-1.3B",
    n_derivatives=1,        # Taylor series order (1 or 2)
    taylor_threshold=2,     # Hybrid strategy threshold
)
```

#### wan21_1_3b_text_to_video_radial.py

T2V with radial sparse attention.

**Purpose:** Memory-efficient video generation using sparse attention patterns.

**Usage:**
```bash
# Standard generation (dense attention)
python examples/wan_video/wan21_1_3b_text_to_video_radial.py

# With radial attention
python examples/wan_video/wan21_1_3b_text_to_video_radial.py --enable_radial

# Custom radial parameters
python examples/wan_video/wan21_1_3b_text_to_video_radial.py \
    --enable_radial \
    --dense_timesteps 20 \
    --decay_factor 0.8
```

**Features:**
- Sparse attention where nearby frames have denser attention
- Reduced memory usage for long videos
- Requires flashinfer or sageattention backend

#### wan21_1_3b_text_to_video_cache_calibrate.py

Calibration tool for AdaTaylorCache.

**Purpose:** Generate calibration parameters for optimal feature caching.

**Usage:**
```bash
python examples/wan_video/wan21_1_3b_text_to_video_cache_calibrate.py \
    --model_root /path/to/Wan2.1-T2V-1.3B/ \
    --num_inference_steps 50 \
    --sigma_shift 8.0 \
    --output_path ./cache_params.json
```

**Output:**
Generates a JSON file with:
- `K`, `retention_ratio`, `thresh`: Default values (0), need manual adjustment
- `cond_mag_ratios`, `uncond_mag_ratios`: Magnitude ratios for skip decisions

**Note:** You must adjust `K`, `retention_ratio`, and `thresh` based on your quality/speed requirements after calibration.

#### wan21_14b_text_to_video_h100.py

T2V with Wan2.1 14B model.

**Purpose:** High-quality text-to-video generation using Wan2.1 14B parameter model.

**Usage:**
```bash
# Basic usage
python examples/wan_video/wan21_14b_text_to_video_h100.py --prompt "A stylish woman walking down a Tokyo street"

# Multi-GPU
python examples/wan_video/wan21_14b_text_to_video_h100.py --gpu_num 2 --prompt "A cat playing"

# Custom resolution and aspect ratio
python examples/wan_video/wan21_14b_text_to_video_h100.py --resolution 720p --aspect_ratio 16:9
```

**Hint:**
If you encounter error like `RuntimeError: unable to open shared memory object`, `OSError: Too many open files`, solve it with:
```bash
ulimit -n 65535
```

**Features:**
- 14B parameter model for high-quality generation
- CFG parallel enabled (cfg_scale=5.0)
- UNPC scheduler with sigma_shift=5.0
- No CLIP stage required for T2V

#### wan22_t2v_5b.py

T2V with Wan2.2 TI2V 5B model.

**Purpose:** High-quality text-to-video generation using Wan2.2 5B unified model.

**Usage:**
```bash
# Basic usage
python examples/wan_video/wan22_t2v_5b.py --prompt "A stylish woman walking down a Tokyo street"

# Multi-GPU
python examples/wan_video/wan22_t2v_5b.py --gpu_num 2 --prompt "A cat playing"

# Custom resolution and aspect ratio
python examples/wan_video/wan22_t2v_5b.py --resolution 480p --aspect_ratio 16:9
```

**Features:**
- CFG parallel enabled by default (cfg_scale=5.0)
- Ulysses sequence parallelism for multi-GPU
- 50-step UNPC sampling with sigma_shift=5.0

#### wan22_14b_text_to_video_h100.py

T2V with Wan2.2 14B model (MoE architecture).

**Purpose:** High-quality text-to-video generation using Wan2.2 14B model with dual-branch (MoE) architecture.

**Usage:**
```bash
# Basic usage
python examples/wan_video/wan22_14b_text_to_video_h100.py --prompt "A stylish woman walking down a Tokyo street"

# Multi-GPU
python examples/wan_video/wan22_14b_text_to_video_h100.py --gpu_num 2 --prompt "A cat playing"

# Custom resolution and aspect ratio
python examples/wan_video/wan22_14b_text_to_video_h100.py --resolution 720p --aspect_ratio 16:9
```

**Features:**
- Dual-branch (high/low noise) MoE architecture
- CFG parallel enabled (cfg_scale_high=5.0, cfg_scale_low=5.0)
- Feature cache for acceleration
- No input image required (pure T2V)

### Image-to-Video Examples (Wan2.1 14B)

#### wan21_14b_image_to_video_h100.py

Standard I2V with Wan2.1 14B model.

**Purpose:** Generate video from image using the 14B parameter model.

**Usage:**
```bash
python examples/wan_video/wan21_14b_image_to_video_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "Make this image come alive"
```

**Features:**
- Model CPU offloading for memory efficiency
- CFG parallel (cfg_scale=5.0)

#### wan21_14b_image_to_video_lora_h100.py

I2V with LoRA acceleration.

**Purpose:** Fast I2V using distilled LoRA weights.

**Usage:**
```bash
python examples/wan_video/wan21_14b_image_to_video_lora_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- 8-step inference with LoRA distillation
- No CFG parallel (cfg_scale=1.0), uses full sp_ulysses_degree

### Image-to-Video Examples (Wan2.2 14B)

#### wan22_14b_image_to_video_h100.py

Standard I2V with Wan2.2 A14B model.

**Purpose:** High-quality I2V using Wan2.2 dual-branch architecture.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "Natural and smooth motion"
```

**Features:**
- High/low noise dual-branch architecture
- Feature cache for acceleration (configurable per branch)
- CFG parallel (cfg_scale_high=3.5, cfg_scale_low=3.5)

**Feature Cache Configuration:**
```python
# Configure feature cache for dit_high
pipe_config.dit_high_config.feature_cache_config = FeatureCacheConfig(
    enabled=True,
    model_type="Wan2_2-I2V-A14B",
)

# Configure feature cache for dit_low
pipe_config.dit_low_config.feature_cache_config = FeatureCacheConfig(
    enabled=True,
    model_type="Wan2_2-I2V-A14B",
)
```

#### wan22_14b_image_to_video_distill_h100.py

I2V with distilled model for fast inference.

**Purpose:** 8-step fast I2V using distilled Wan2.2 model.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_distill_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- 8-step inference with distilled weights
- No CFG (cfg_scale=1.0), full sequence parallel
- FSDP and VAE parallel enabled for multi-GPU

#### wan22_14b_image_to_video_distill_fp8_h100.py

I2V with FP8 quantization for memory efficiency.

**Purpose:** Memory-efficient fast I2V using FP8 quantized weights.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_distill_fp8_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- FP8 quantization (~50% memory reduction)
- 8-step inference with distilled weights
- No CFG parallel (cfg_scale=1.0)

#### wan22_14b_image_to_video_lora_h100.py

I2V with LoRA weights for fast inference.

**Purpose:** Fast I2V using LoRA-adapted Wan2.2 model.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_lora_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- LoRA weights for both dit_high and dit_low
- 8-step inference
- No CFG parallel (cfg_scale=1.0)

#### wan22_14b_image_to_video_mix_h100.py

I2V with mixed precision/optimizations.

**Purpose:** Advanced I2V with mixed optimizations including selective feature cache.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_mix_h100.py \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- Feature cache enabled for dit_high, disabled for dit_low
- LoRA weights for dit_low branch
- Mix-euler scheduler
- CFG parallel for dit_high only (cfg_scale_high=3.5, cfg_scale_low=1.0)

#### wan22_14b_image_to_video_h100_ray.py

I2V with Ray distributed inference.

**Purpose:** Multi-GPU distributed inference using Ray framework.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_h100_ray.py \
    --gpu_num 2 \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- Ray-based distributed inference
- VAE parallel processing
- No CFG parallel (cfg_scale=1.0)

#### wan22_14b_image_to_video_cache_calibrate.py

Calibration tool for Wan2.2 I2V AdaTaylorCache.

**Purpose:** Generate calibration parameters for Wan2.2 dual-branch feature caching.

**Usage:**
```bash
python examples/wan_video/wan22_14b_image_to_video_cache_calibrate.py \
    --model_root /path/to/Wan2.2-I2V-A14B/ \
    --num_inference_steps 40 \
    --sigma_shift 5.0 \
    --output_path ./cache_params.json
```

**Features:**
- Shared calibrator for both dit_high and dit_low branches
- Collects residual data across the full sampling loop
- Generates a single JSON file for the entire pipeline

**Note:** Wan2.2 uses a dual-branch architecture where dit_high and dit_low work together in the sampling loop. A single calibrator is shared between both branches to capture the complete denoising process.

#### wan22_i2v_5b.py

I2V with Wan2.2 TI2V 5B model.

**Purpose:** Image-to-video generation using Wan2.2 5B unified model.

**Usage:**
```bash
python examples/wan_video/wan22_i2v_5b.py \
    --image_path /path/to/image.jpg \
    --prompt "A stylish woman walking"
```

**Features:**
- CFG parallel enabled (cfg_scale=5.0)
- 50-step UNPC sampling

### First-Last-Frame to Video Examples (FL2V)

#### wan22_14b_first_last_frame_to_video_h100.py

Generate video from first and last frames.

**Purpose:** Create video that interpolates between start and end frames, useful for:
- Video interpolation between keyframes
- Creating smooth transitions between images
- Generating video with specific start and end content

**Usage:**
```bash
python examples/wan_video/wan22_14b_first_last_frame_to_video_h100.py \
    --first_image_path /path/to/start.png \
    --last_image_path /path/to/end.png \
    --prompt "A smooth transition between the two scenes"
```

**Features:**
- First frame (first_image) as video start
- Last frame (last_image) as video end
- CFG parallel enabled (cfg_scale_high=3.5, cfg_scale_low=3.5)
- Feature cache for acceleration

**API Usage:**
```python
video = pipeline(
    prompt=prompt,
    input_image=first_image,  # Start frame
    end_image=last_image,      # End frame
    num_inference_steps=40,
    cfg_scale_high=3.5,
    cfg_scale_low=3.5,
)
```

### Async Pipeline Examples

#### async_wan22_14b_image_to_video_distill_h100.py

Async I2V with event streaming.

**Purpose:** Asynchronous inference with progress events for API integration.

**Usage:**
```bash
python examples/wan_video/async_wan22_14b_image_to_video_distill_h100.py \
    --gpu_num 2 \
    --image_path /path/to/image.jpg \
    --prompt "A moving scene"
```

**Features:**
- Async event streaming for real-time progress
- FSDP and VAE parallel enabled
- No CFG parallel (cfg_scale=1.0)
- Suitable for API server integration

## Performance

### Text-to-Video (Wan2.1 1.3B)

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| T2V 1.3B | H100*1 | 40 | 81 | 480p | TBD | TBD |
| T2V 1.3B | H100*2 | 40 | 81 | 480p | TBD | TBD |
| T2V 1.3B + AdaTaylor | H100*1 | 40 | 81 | 480p | TBD | TBD |
| T2V 1.3B + Radial | H100*1 | 40 | 81 | 480p | TBD | TBD |

### Text-to-Video (Wan2.1 14B)

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| T2V 14B | H100*1 | 40 | 81 | 720p | TBD | TBD |

### Text-to-Video (Wan2.2 14B)

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| T2V 14B | H100*1 | 40 | 81 | 720p | TBD | TBD |
| T2V 14B | H100*2 | 40 | 81 | 720p | TBD | TBD |

### Image-to-Video (Wan2.1 14B)

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| I2V 14B | H100*1 | 40 | 81 | 720p | TBD | TBD |
| I2V 14B + LoRA | H100*1 | 8 | 81 | 720p | TBD | TBD |

### Image-to-Video (Wan2.2 A14B)

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| I2V A14B BF16 | H100*1 | 40 | 81 | 720p | TBD | TBD |
| I2V A14B Distill BF16 | H100*1 | 8 | 81 | 720p | TBD | TBD |
| I2V A14B Distill FP8 | H100*1 | 8 | 81 | 720p | TBD | TBD |
| I2V A14B Distill BF16 | H100*2 | 8 | 81 | 720p | TBD | TBD |

### First-Last-Frame to Video (Wan2.2 A14B)

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| FL2V A14B | H100*1 | 40 | 81 | 720p | TBD | TBD |
| FL2V A14B | H100*2 | 40 | 81 | 720p | TBD | TBD |

## Notes

- Wan2.1 T2V 1.3B is optimized for 480p generation
- Wan2.1 I2V 14B and Wan2.2 I2V A14B support 720p generation
- Wan2.2 TI2V 5B supports both T2V and I2V with unified model
- Wan2.2 I2V A14B supports FL2V (First-Last-frame to Video) via `end_image` parameter
- AdaTaylor cache provides 2-3x speedup with minimal quality loss
- FP8 quantization reduces memory by ~50%
- Ray distributed inference enables efficient multi-GPU scaling
- Feature cache is configured via `ModelRuntimeConfig.feature_cache_config` during pipeline initialization
- Parallel config is automatically set based on cfg_scale values
