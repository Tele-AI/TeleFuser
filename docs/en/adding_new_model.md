# Adding New Model Development Guide

This document explains how to add support for new internal models in TeleFuser, including computing model hashes, adding configurations, and implementing necessary converters.

## Overview

TeleFuser uses a **Hash-based automatic recognition mechanism** to determine model types. To integrate a new model into the system, you need to:

1. Implement the model class (inherit from `BaseModel`)
2. Implement the `state_dict_converter` converter
3. Use `weight_viewer.py` to compute model hash
4. Add configuration and test validation

## Step-by-Step Guide

### Step 1: Implement Model Class

Create a model class inheriting from `BaseModel` (or appropriate base class based on model type):

```python
# telefuser/models/my_custom_dit.py
import torch
import torch.nn as nn
from telefuser.core.base_model import BaseModel

class MyCustomDiT(BaseModel):
    def __init__(
        self,
        in_channels=16,
        out_channels=16,
        hidden_size=2048,
        num_layers=32,
        # ... other parameters
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # ... model definition

    def forward(self, x, t, context, **kwargs):
        # Forward logic
        pass

    @classmethod
    def state_dict_converter(cls):
        """Return state dict converter class"""
        return MyCustomDiTStateDictConverter
```

#### Implementing `from_pretrained` Interface (Optional)

Models can optionally implement a `from_pretrained` class method for convenient model loading. This method provides a unified interface for loading models in pipeline examples:

```python
# telefuser/models/hunyuan_video_text_encoder.py

class TextEncoder(nn.Module):
    """Text encoder using LLM for HunyuanVideo."""

    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: str,
        text_encoder_path: str,
        # ... other parameters with internal defaults
    ):
        super().__init__()
        # ... initialization logic

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "TextEncoder":
        """Load TextEncoder from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: Path to text encoder model
            torch_dtype: Model precision (default: bfloat16)
            **kwargs: Ignored for compatibility

        Returns:
            Loaded TextEncoder instance
        """
        # Determine precision from torch_dtype
        precision = "bf16" if torch_dtype == torch.bfloat16 else "fp16"

        # All internal parameters are set with sensible defaults
        return cls(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision=precision,
            text_encoder_path=pretrained_model_name_or_path,
            tokenizer_type="llm",
            # ... other internal defaults
        )
```

**Key principles for `from_pretrained`:**
1. Only expose essential parameters like `pretrained_model_name_or_path` and `torch_dtype`
2. Set all other parameters internally with sensible defaults
3. Accept `**kwargs` for compatibility but ignore unknown parameters
4. Return a fully initialized model instance

**Note:** If `from_pretrained` is not implemented, you can still use `ModuleManager.load_model()` with hash-based auto-recognition, or manually instantiate the model and add it via `add_module()`.

#### VAE Model Example

```python
# telefuser/models/hunyuan_video_vae.py

class HunyuanVideoVAE(nn.Module):
    """HunyuanVideo VAE for video encoding/decoding."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "HunyuanVideoVAE":
        """Load HunyuanVideoVAE from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: Path to VAE checkpoint directory
            torch_dtype: Model precision (default: bfloat16)
            **kwargs: Ignored for compatibility

        Returns:
            Loaded HunyuanVideoVAE instance
        """
        # Load config from JSON
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model with config
        model = cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            # ... other config parameters
        )

        # Load state dict
        state_dict = load_state_dict(os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(state_dict, strict=False)

        return model.to(dtype=torch_dtype)
```

**Note:** Tiling/slicing settings should be handled at runtime by the VAE stage, not during model initialization.

### Step 2: Implement StateDictConverter

The converter is responsible for transforming weights from different source formats to internal format:

```python
# telefuser/models/my_custom_dit.py

class MyCustomDiTStateDictConverter:
    """
    Convert state_dict from different sources to internal format
    """
    
    @staticmethod
    def from_official(state_dict):
        """
        Convert from Civitai/Direct format
        
        Args:
            state_dict: Original state dictionary
            
        Returns:
            Converted state_dict, or (state_dict, extra_kwargs) tuple
        """
        # Create key mapping
        rename_dict = {
            "input_blocks.0.0.weight": "conv_in.weight",
            "input_blocks.0.0.bias": "conv_in.bias",
            # ... more mappings
        }
        
        converted_state_dict = {}
        for old_key, new_key in rename_dict.items():
            if old_key in state_dict:
                converted_state_dict[new_key] = state_dict[old_key]
        
        # Return extra_kwargs if model parameters need to be inferred from weights
        extra_kwargs = {
            "hidden_size": 2048,  # Infer from weights or hard-code
            "num_layers": 32,
        }
        
        return converted_state_dict, extra_kwargs
    
    @staticmethod
    def from_diffusers(state_dict):
        """Convert from Diffusers format"""
        # Similar implementation
        pass
```

### Step 3: Use Weight Viewer to Compute Model Hash

Use the built-in `weight_viewer.py` tool to analyze the model:

```bash
# Quick hash retrieval
python tools/viewer/weight_viewer.py /path/to/your/model.safetensors --quiet
```

Output example:

```
Total parameters: 14.02B
Files: 1
hash with shape: 4c3523c69fb7b24cf2db147a715b277f
```

**Record the `hash with shape` value**, which will be added to the configuration.

For more detailed analysis (view model structure to help implement StateDictConverter):

```bash
# View complete structure and export
python tools/viewer/weight_viewer.py /path/to/your/model.safetensors \
    --max-depth 10 \
    --export model_structure.json
```

**Advantages of using weight_viewer:**
- Automatically handles sharded models (using wildcards `model-*.safetensors`)
- Displays parameter statistics and data type distribution
- Automatically merges structurally identical modules (like transformer blocks)
- Exports JSON for further analysis

#### Handling Sharded Models

If the model is split into multiple files:

```bash
# Automatically merge all shards and compute hash
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors" --quiet
```

**Note**: When adding to `model_config.py`, ensure the hash is based on **merged complete weights**.

### Step 4: Add Model Configuration

Edit `telefuser/core/model_config.py` to add model configuration.

First, get information from weight_viewer output:

```bash
$ python tools/viewer/weight_viewer.py /path/to/my_model.safetensors --quiet

Total parameters: 6.91B
Files: 1
hash with shape: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

Then add configuration:

```python
from ..models.my_custom_dit import MyCustomDiT

model_loader_configs = [
    # ... existing configurations ...
    
    # MyCustomDiT - Standard version (from weight_viewer: hash=a1b2c3d4...)
    # Parameters: 6.91B
    (
        None,                                  # hash without shape (optional, for non-strict matching)
        "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",   # hash with shape (from weight_viewer)
        ["my_custom_dit"],                     # model_name (for fetch_module)
        [MyCustomDiT],                         # model_class
        "official",                             # model_resource: "official" or "diffusers"
    ),
]
```

#### Adding Multiple Variants

If the same model has multiple variants (e.g., FP8 version):

```bash
# Analyze FP8 version
$ python tools/viewer/weight_viewer.py /path/to/my_model_fp8.safetensors --quiet

Total parameters: 6.91B
Files: 1
hash with shape: b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7  # Different hash!
```

Add to configuration:

```python
    # MyCustomDiT - Standard version (hash: a1b2c3d4...)
    (
        None,
        "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        ["my_custom_dit"],
        [MyCustomDiT],
        "official",
    ),
    
    # MyCustomDiT - FP8 version (hash: b2c3d4e5...) 
    # Note: FP8 quantized weights
    (
        None,
        "b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7",
        ["my_custom_dit"],
        [MyCustomDiT],
        "official",
    ),
```

**Tip**: If variants have different tensor shapes (like pruned models), consider using non-strict matching (only using `keys_hash`).

Configuration field description:

| Field | Type | Description |
|------|------|------|
| `keys_hash` | `str \| None` | Hash based only on key names (without shape). For variants where shape may change |
| `keys_hash_with_shape` | `str` | Hash including key names and shape. Strict matching, recommended priority |
| `model_names` | `list[str]` | Model identifier name list, used for `fetch_module()` |
| `model_classes` | `list[type]` | Corresponding model class list |
| `model_resource` | `str` | Weight source format: `"official"` or `"diffusers"` |

### Step 5: Test Validation

Create a test script to verify model loading:

```python
# tests/test_my_custom_model_loading.py
import torch
import pytest
from telefuser.core.module_manager import ModuleManager

def test_my_custom_dit_loading():
    """Test MyCustomDiT model loading"""
    module_manager = ModuleManager(device="cpu")

    # Test auto-recognition
    module_manager.load_model(
        "/path/to/your/model.safetensors",
        torch_dtype=torch.bfloat16
    )

    # Verify model can be fetched
    model = module_manager.fetch_module("my_custom_dit")
    assert model is not None

    # Verify model type
    from telefuser.models.my_custom_dit import MyCustomDiT
    assert isinstance(model, MyCustomDiT)

    print("✓ MyCustomDiT loading test passed!")

if __name__ == "__main__":
    test_my_custom_dit_loading()
```

Run tests:

```bash
pytest tests/test_my_custom_model_loading.py -v
```

## Using Models in Pipeline Examples

When creating pipeline examples, use the `from_pretrained` interface and `add_module` pattern:

### Basic Pattern

```python
import os
import torch
from telefuser.utils.logging import logger
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_vae import HunyuanVideoVAE
from telefuser.models.hunyuan_video_text_encoder import HunyuanVideoTextEncoder

def get_pipeline(model_root: str = "/path/to/models"):
    """Create and initialize pipeline with all models."""
    module_manager = ModuleManager(device="cpu")

    # 1. Load VAE using from_pretrained
    vae_path = os.path.join(model_root, "vae")
    logger.info(f"Loading VAE from {vae_path}")
    vae = HunyuanVideoVAE.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(vae, name="vae")

    # 2. Load TextEncoder using from_pretrained
    text_encoder_path = os.path.join(model_root, "text_encoder", "llm")
    logger.info(f"Loading TextEncoder from {text_encoder_path}")
    text_encoder = HunyuanVideoTextEncoder.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(text_encoder, name="text_encoder")

    # 3. Load other models similarly...
    # transformer = HunyuanVideoDiT.from_pretrained(transformer_path, torch_dtype=torch.bfloat16)
    # module_manager.add_module(transformer, name="hunyuan_video_dit")

    # 4. Create and initialize pipeline
    # pipe = HunyuanVideo15Pipeline(device="cuda", torch_dtype=torch.bfloat16)
    # pipe.init(module_manager, pipe_config)

    return pipe
```

### Key Principles

1. **Use `from_pretrained` for all model loading** - This provides a consistent interface
2. **Only expose model path externally** - All other parameters should be internal defaults
3. **Use `add_module` with meaningful names** - Names like `"vae"`, `"text_encoder"`, `"hunyuan_video_dit"` are used by pipeline stages to fetch modules
4. **Let stages handle runtime settings** - Tiling, slicing, and other runtime configurations should be handled by pipeline stages, not during model initialization

### Module Naming Convention

| Module Type | Recommended Name | Used By |
|-------------|------------------|---------|
| VAE | `"vae"` | `HunyuanVideoVAEStage` |
| Text Encoder | `"text_encoder"` | `HunyuanVideoTextEncodingStage` |
| DiT/Transformer | `"hunyuan_video_dit"` | `HunyuanVideoDenoisingStage` |
| Vision Encoder (I2V) | `"vision_encoder"` | `HunyuanVideoImageEncodingStage` |
| Upsampler (SR) | `"upsampler"` | `HunyuanVideoUpsamplerStage` |
| Scheduler | `"scheduler"` | Pipeline init |

## Special Cases

### Handling Shape-Changing Variants

Some model variants (like FP8 quantized, pruned versions) may have different tensor shapes:

```python
# Main version (strict matching)
(
    None,  # No non-strict hash needed
    "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    ["my_model"],
    [MyModel],
    "official",
),

# FP8 version (different shape, use non-strict matching)
(
    "q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",  # Only key hash
    None,  # Don't use shape hash (because shape differs)
    ["my_model_fp8"],
    [MyModelFP8],  # May need different class
    "official",
),
```

### Multi-Component Models

Some model files contain multiple components (like VAE encoder + decoder):

```python
# Separate components in state_dict_converter
@staticmethod
def from_official(state_dict):
    encoder_dict = {}
    decoder_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_dict[key[8:]] = value  # Remove "encoder." prefix
        elif key.startswith("decoder."):
            decoder_dict[key[8:]] = value
    
    # Return merged dict, handle in model class
    combined_dict = {
        "encoder": encoder_dict,
        "decoder": decoder_dict,
    }
    
    return combined_dict, {"has_separate_components": True}
```

### Supporting Multiple Source Formats

If models may come from different sources (Civitai, HuggingFace Diffusers):

```python
class MyModelStateDictConverter:
    @staticmethod
    def from_official(state_dict):
        # Civitai format conversion
        return convert_official_format(state_dict)
    
    @staticmethod
    def from_diffusers(state_dict):
        # Diffusers format conversion
        return convert_diffusers_format(state_dict)
```

Then specify the correct `model_resource` in configuration.

## Debugging Tips

### 1. Use Weight Viewer to View Model Structure

```bash
# View all keys and shapes
python tools/viewer/weight_viewer.py /path/to/model.safetensors --show-all

# Export as JSON for programmatic processing
python tools/viewer/weight_viewer.py /path/to/model.safetensors --export model_info.json
```

### 2. Check Hash Matching Process

```python
from telefuser.utils.model_weight import load_state_dict, hash_state_dict_keys
from telefuser.core.model_config import model_loader_configs

sd = load_state_dict("/path/to/model.safetensors")
hash_with_shape = hash_state_dict_keys(sd, with_shape=True)
hash_without_shape = hash_state_dict_keys(sd, with_shape=False)

print(f"Model hash (with shape): {hash_with_shape}")
print(f"Model hash (without shape): {hash_without_shape}")

# Check if in configuration
found = False
for config in model_loader_configs:
    keys_hash, keys_hash_with_shape, model_names, model_classes, resource = config
    if keys_hash_with_shape == hash_with_shape:
        print(f"✓ Found match (strict): {model_names}")
        found = True
    elif keys_hash == hash_without_shape:
        print(f"✓ Found match (non-strict): {model_names}")
        found = True

if not found:
    print("✗ No matching configuration found!")
    print(f"Add this to model_config.py:")
    print(f'    (None, "{hash_with_shape}", ["your_model_name"], [YourModelClass], "official"),')
```

### 3. Verify Converter Output

```python
from telefuser.models.my_custom_dit import MyCustomDiT
from telefuser.utils.model_weight import load_state_dict

sd = load_state_dict("/path/to/model.safetensors")
converter = MyCustomDiT.state_dict_converter()
converted, extra_kwargs = converter.from_official(sd)

print(f"Extra kwargs: {extra_kwargs}")
print(f"Converted keys: {list(converted.keys())[:10]}")

# Try initialization
model = MyCustomDiT(**extra_kwargs)
model.load_state_dict(converted, strict=False)  # Test with non-strict mode first
print("✓ Model initialized successfully!")
```

### 4. Quick Configuration Verification

```bash
# After modifying configuration, quickly verify hash matches
python -c "
from telefuser.core.module_manager import ModuleManager
mm = ModuleManager(device='cpu')
mm.load_model('/path/to/your/model.safetensors')
print('✓ Configuration is correct!')
print(f'Loaded models: {mm.module_name}')
"
```

## Best Practices

1. **Keep configurations organized**
   - Group by model type
   - Keep different variants of same model together
   - Add comments explaining version differences

2. **Use strict matching when possible**
   - Provide `keys_hash_with_shape` whenever possible
   - Only use non-strict matching when shape may vary

3. **Document variants in detail**
   ```python
     # Wan2.1 T2V 14B - FP8 per-channel quantized
     # Note: This version has scaled weights for FP8 inference
     (
         None,
         "4cf556355bc7e9b6545b38f4930f60b1",
         ["wan_video_dit"],
         [WanModel],
         "official",
     ),
   ```

4. **Test all variants**
   - Original version
   - FP8 quantized version
   - Pruned version
   - Different source formats (Civitai vs Diffusers)

5. **Naming conventions**
   - Use lowercase with underscores for `model_names`
   - Prefix indicates model family: `wan_video_`, `qwen_image_`, `flashvsr_`

6. **Make full use of weight_viewer**
   ```bash
   # Analyze model before adding configuration
   python tools/viewer/weight_viewer.py /path/to/model.safetensors --export model_info.json
   
   # Compare differences between versions
   python tools/viewer/weight_viewer.py /path/to/model_v1.safetensors --export v1.json
   python tools/viewer/weight_viewer.py /path/to/model_v2.safetensors --export v2.json
   diff v1.json v2.json
   ```

## Example: Complete New Model Integration

Refer to the following files for complete implementation:

- Model implementation: `telefuser/models/wan_video_dit.py`
- Configuration definition: `telefuser/core/model_config.py` (WanModel related configurations)
- Usage example: `examples/wan_video/wan21_14b_image_to_video_h100.py`

## Optimizing Model Inference

After completing model integration, you can optimize inference performance and memory usage through the following methods.

### 1. Reuse Optimized Operators

TeleFuser's `ops` module provides high-performance neural network operator implementations. Reusing these operators in new models achieves optimal performance:

| Operator | Usage | Performance Optimization |
|----------|-------|--------------------------|
| `RMSNorm` / `LayerNorm` | Normalization layers | tf_kernel > Triton > PyTorch |
| `FeedForward` | Feed-forward networks | Supports GEGLU/SwiGLU |
| `attention` | Attention computation | Flash Attention 2/3/4, SageAttention |
| `LinearFP8` | Quantized linear layers | FP8 inference |

```python
from telefuser.ops.normalization import RMSNorm
from telefuser.ops.ffn import FeedForward
from telefuser.ops.attention import attention
from telefuser.core.config import AttentionConfig, AttnImplType

class MyTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = FeedForward(dim=dim, mult=4, activation_fn="geglu")
        self.attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
```

For detailed documentation, see [Ops Module Documentation](./ops.md).

### 2. Multi-GPU Inference

For large models or long sequence generation, various parallel strategies are available:

```python
from telefuser.core.config import ParallelConfig

# Ulysses sequence parallelism (2 GPU)
config = ParallelConfig(
    device_ids=[0, 1],
    sp_ulysses_degree=2,
)
pipe_config.dit_config.parallel_config = config
pipe_config.enable_denoising_parallel = True

# CFG + Ulysses (4 GPU)
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

| Strategy | Use Case | Description |
|----------|----------|-------------|
| Ulysses | Medium-length sequences | All-to-All communication |
| Ring | Extra-long sequences | P2P communication, supports arbitrary length |
| USP | Large-scale parallelism | Ulysses + Ring combination |
| CFG Parallel | CFG acceleration | Positive/negative prompt parallel computation |
| Pipeline Parallel | Large model inference | Layer distribution across GPUs |

For detailed configuration, see [Parallel Inference Guide](./parallel.md).

### 3. Model Quantization

Use `tools/convert/converter.py` to quantize models and significantly reduce memory usage:

**FP8 Quantization** (recommended):
```bash
python tools/convert/converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --linear_dtype fp8 \
    --non_linear_dtype torch.bfloat16 \
    --model_type wan_dit \
    --quantized \
    --single_file
```

**INT8 Quantization**:
```bash
python tools/convert/converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --linear_dtype torch.int8 \
    --model_type wan_dit \
    --quantized \
    --single_file
```

Supported quantization types: `int8`, `fp8`, `nvfp4`, `mxfp4`, `mxfp6`, `mxfp8`.

For detailed usage, see `tools/convert/README.md`.

### 4. CPU Offloading

When GPU memory is insufficient, use CPU offloading to temporarily move model weights to CPU:

```python
from telefuser.core.config import OffloadConfig, WeightOffloadType

# Async CPU offload (recommended)
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=1,
)
```

| Strategy | Memory Savings | Speed Impact | Use Case |
|----------|----------------|--------------|----------|
| `NO_CPU_OFFLOAD` | None | Fastest | Sufficient VRAM |
| `MODEL_CPU_OFFLOAD` | ~50% | Medium | Moderately constrained |
| `ASYNC_CPU_OFFLOAD` | ~60-70% | Low | 8-16GB VRAM |
| `SEQUENTIAL_CPU_OFFLOAD` | Maximum | Slowest | <8GB VRAM |

For detailed configuration, see [CPU Offloading Guide](./offload.md).

### 5. Combined Optimization Example

Here's a complete optimization configuration example:

```python
from telefuser.core.config import (
    ParallelConfig,
    AttentionConfig,
    AttnImplType,
    OffloadConfig,
    WeightOffloadType,
)

# Multi-GPU + Attention optimization + Offloading
pipe_config.dit_config.parallel_config = ParallelConfig(
    device_ids=[0, 1],
    sp_ulysses_degree=2,
)
pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
)
pipe_config.enable_denoising_parallel = True
```

## Related Documentation

- [Model Loading User Guide](./model_loading.md)
- [Hash Configuration Management Guide](./hash_config_management.md)
- [Ops Module Documentation](./ops.md) - Neural network operator implementations (activations, normalization layers, attention, etc.)
- [Parallel Inference Guide](./parallel.md) - Multi-GPU inference configuration
- [CPU Offloading Guide](./offload.md) - Memory optimization strategies
