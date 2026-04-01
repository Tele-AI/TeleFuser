# Feature Cache

Feature caching is a technique to accelerate diffusion model inference by caching intermediate features and reusing them across timesteps. TeleFuser implements AdaTaylorCache for video generation models.

## Base Interface

All feature cache implementations follow the `BaseFeatureCache` interface:

```python
class BaseFeatureCache(ABC):
    def should_compute(self, is_cond: bool) -> bool:
        """Check if real computation is needed. Auto-increments step counter."""
        pass
    
    def update(self, output: Tensor, ori_input: Tensor, is_cond: bool):
        """Update cache with computed result."""
        pass
    
    def approximate(self, input: Tensor, is_cond: bool) -> Tensor:
        """Get approximated output from cache."""
        pass
    
    def reset():
        """Reset all cached state."""
        pass
```

### Usage in Model Forward

```python
# In DiT block forward:
if self.feature_cache.should_compute(cond_flag):
    output = self.forward_blocks(x, ...)
    self.feature_cache.update(output, x, cond_flag)
else:
    output = self.feature_cache.approximate(x, cond_flag)
```

---

## Factory Function

Use `create_feature_cache` to create cache instances:

```python
from telefuser.feature_cache import create_feature_cache

# No caching (default)
cache = create_feature_cache("none")

# AdaTaylorCache for inference acceleration
cache = create_feature_cache(
    "ada_taylor",
    model_type="Wan2.1-T2V-1.3B",
    num_inference_steps=50,
)

# Calibrator for parameter collection
cache = create_feature_cache(
    "calibrator",
    num_inference_steps=50,
    sigma_shift=8.0,
    model_name="MyModel",
)
```

---

## AdaTaylorCache

AdaTaylorCache (Adaptive Taylor Cache) is a feature caching strategy that combines:
- **Adaptive skip logic**: Adaptively skips computations based on magnitude ratios between consecutive timesteps
- **Taylor series approximation**: Uses Taylor expansion for higher-order accuracy when approximating skipped steps
- **Hybrid fallback**: Falls back to residual reuse when elapsed steps exceed threshold

When `n_derivatives=0`, AdaTaylorCache reduces to simple residual caching (residual reuse only, no Taylor expansion).

### How AdaTaylorCache Works

1. **Skip Decision**: Track magnitude ratios between consecutive timesteps, accumulate error when skipping, skip when accumulated error < threshold and consecutive skips ≤ K
2. **Approximation**: When skipping and elapsed ≤ threshold, use Taylor series expansion; when elapsed > threshold, fall back to residual reuse

### Cache Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `K` | int | Maximum consecutive skip steps |
| `retention_ratio` | float | Ratio of initial steps to always compute (no skipping) |
| `thresh` | float | Error threshold for skipping decisions |
| `cond_mag_ratios` | list | Magnitude ratios for conditional path |
| `uncond_mag_ratios` | list | Magnitude ratios for unconditional path |

### Using AdaTaylorCache

To enable AdaTaylorCache in your pipeline:

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline

# Create pipeline
pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
# ... initialize pipeline ...

# Run with AdaTaylorCache enabled
video = pipe(
    prompt="A cat playing piano",
    num_inference_steps=50,
    enable_ada_taylor_cache=True,
    ada_taylor_n_derivatives=1,  # Use Taylor expansion (set to 0 for residual only)
    model_type="Wan2.1-T2V-1.3B",
    # ... other parameters ...
)
```

The `model_type` parameter specifies which pre-calibrated parameters to use. See available models in `telefuser/feature_cache/ada_taylor_cache/params/`.

---

## Cache Calibration

AdaTaylorCache requires model-specific calibration parameters. Use the calibrator to generate these parameters for new models.

### When to Calibrate

You need to run calibration when:
- Using a new model architecture
- Using different inference settings (e.g., different `num_inference_steps` or `sigma_shift`)
- Fine-tuning for specific quality/speed trade-offs

### Calibration Process

The calibration process runs the pipeline once to collect residual statistics:

1. **Initialize Calibrator**: Set up with your inference configuration
2. **Run Pipeline**: Execute one inference pass (calibration data is collected automatically)
3. **Save Parameters**: Parameters are saved to a JSON file automatically

### Running Calibration

#### Using the Example Script

```bash
python examples/wan_video/wan21_1_3b_text_to_video_cache_calibrate.py \
    --model_root /path/to/Wan2.1-T2V-1.3B/ \
    --num_inference_steps 50 \
    --sigma_shift 8.0 \
    --model_name "Wan2.1-T2V-1.3B" \
    --output_path ./my_cache_params.json
```

#### Programmatic Usage

```python
from telefuser.feature_cache import create_feature_cache

# Create calibrator
cache = create_feature_cache(
    "calibrator",
    num_inference_steps=50,
    sigma_shift=8.0,
    model_name="Wan2.1-T2V-1.3B",
)

# Set calibrator on the model
pipeline.denoise_stage.dit.set_ada_taylor_cache_calibrator(
    num_inference_steps=50,
    sigma_shift=8.0,
    model_name="Wan2.1-T2V-1.3B",
)

# Run pipeline (calibration happens automatically)
video = pipeline(
    prompt="A sample prompt",
    num_inference_steps=50,
    sigma_shift=8.0,
)
```

### Calibration Output

The generated JSON file contains:

```json
{
    "K": 0,
    "retention_ratio": 0.0,
    "thresh": 0.0,
    "sigma_shift": 8.0,
    "num_inference_steps": 50,
    "cond_mag_ratios": [1.0, 1.0124, 1.00166, ...],
    "uncond_mag_ratios": [1.0, 1.02213, 1.0041, ...]
}
```

**Important**: `K`, `retention_ratio`, and `thresh` are set to 0 by default. You need to adjust these values based on your quality/speed requirements:

- **Higher `K`**: More aggressive skipping, faster inference, potential quality loss
- **Higher `retention_ratio`**: More initial steps computed, better quality at cost of speed
- **Higher `thresh`**: More tolerant to errors, faster inference, potential quality loss

### Recommended Values

For Wan2.1 1.3B models, recommended starting values:

```json
{
    "K": 4,
    "retention_ratio": 0.2,
    "thresh": 0.12
}
```

### Parameter File Location

By default, parameters are saved to:
```
telefuser/feature_cache/ada_taylor_cache/params/{model_name}.json
```

The model name is sanitized (dots and slashes replaced with underscores) for the filename.

---

## AdaTaylorCache Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | str | Required | Model type for loading cache parameters |
| `n_derivatives` | int | 1 | Taylor expansion order (0 for residual only, 1-2 recommended) |
| `taylor_threshold` | int | 2 | Threshold for switching to residual reuse (elapsed > threshold uses residual) |

The following parameters are loaded from the pre-calibrated params:
- `K`: Maximum consecutive skip steps
- `thresh`: Error threshold for skipping decisions
- `retention_ratio`: Ratio of initial steps to always compute

### Using the Example Script

```bash
python examples/wan_video/wan21_1_3b_text_to_video_ada_taylor_cache.py \
    --gpu_num 1 \
    --n_derivatives 1 \
    --taylor_threshold 2 \
    --num_inference_steps 40
```

### When to Use Different Configurations

- **`n_derivatives=0`**: Simple residual caching, fastest, good for speed-critical scenarios
- **`n_derivatives=1`**: Taylor expansion with hybrid fallback, best quality-speed balance
- **`n_derivatives=2`**: Higher-order Taylor expansion, better accuracy at cost of memory

---

## Available Pre-calibrated Models

| Model | File | Default Steps |
|-------|------|---------------|
| Wan2.1-T2V-1.3B | `Wan2_1-T2V-1_3B.json` | 50 |
| Wan2.1-T2V-14B | `Wan2_1-T2V-14B.json` | 50 |
| Wan2.1-I2V-14B-480P | `Wan2_1-I2V-14B-480P.json` | 50 |
| Wan2.1-I2V-14B-720P | `Wan2_1-I2V-14B-720P.json` | 50 |
| Wan2.1-FL2V-14B-720P | `Wan2_1-FL2V-14B-720P.json` | 50 |
| Wan2.2-T2V-A14B | `Wan2_2-T2V-A14B.json` | 50 |
| Wan2.2-I2V-A14B | `Wan2_2-I2V-A14B.json` | 40 |
| Wan2.2-FL2V-A14B | `Wan2_2-FL2V-A14B.json` | 40 |
| HunyuanVideo-T2V | `HunyuanVideo-T2V.json` | 50 |
| HunyuanVideo-I2V | `HunyuanVideo-I2V.json` | 50 |
| Qwen-Image | `Qwen-Image.json` | 50 |
| Qwen-Image-Edit-Plus | `Qwen-Image-Edit-Plus.json` | 40 |

---

## Calibration Scripts

| Pipeline | Script | Model Type |
|----------|--------|------------|
| Wan2.1 T2V 1.3B | `examples/wan_video/wan21_1_3b_text_to_video_cache_calibrate.py` | Wan2.1-T2V-1.3B |
| Wan2.2 I2V A14B | `examples/wan_video/wan22_14b_image_to_video_cache_calibrate.py` | Wan2.2-I2V-A14B |
| HunyuanVideo T2V | `examples/hunyuan_video/hunyuan_video_t2v_cache_calibrate.py` | HunyuanVideo-T2V |
| HunyuanVideo I2V | `examples/hunyuan_video/hunyuan_video_i2v_cache_calibrate.py` | HunyuanVideo-I2V |
| Qwen-Image T2I | `examples/qwen_image/qwen_image_cache_calibrate.py` | Qwen-Image |
| Qwen-Image Edit | `examples/qwen_image/qwen_image_edit_plus_cache_calibrate.py` | Qwen-Image-Edit-Plus |

**Note for Wan2.2 I2V:** Wan2.2 uses a dual-branch architecture (dit_high + dit_low). The calibration script shares a single calibrator between both branches to capture the complete denoising process in one JSON file.

---

## CFG Parallel (CFGP) Support

AdaTaylorCache supports CFG Parallel mode where each rank processes one CFG branch (cond or uncond). Each path maintains its own step counter internally, ensuring correct synchronization across ranks.

```python
# In CFGP mode with 2 GPUs:
# Rank 0: processes cond path, cond_state step counter increments
# Rank 1: processes uncond path, uncond_state step counter increments
# Both counters stay synchronized automatically
```

---

## References

AdaTaylorCache is inspired by and builds upon the following works:

- **MagCache**: Ma, X., Fang, G., Wang, X., et al. (2025). "Semantically-aware Taylor Expansion for Diffusion Model Sampling Acceleration." arXiv preprint arXiv:2506.09045. [Link](https://arxiv.org/abs/2506.09045)

- **TaylorSeer**: Ma, X., Fang, G., Wang, X., et al. (2025). "From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeer." arXiv preprint arXiv:2503.06923. [Link](https://arxiv.org/abs/2503.06923)

We thank the authors for their pioneering work in diffusion model acceleration through feature caching and Taylor series approximation.