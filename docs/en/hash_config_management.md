# Hash Configuration Management Guide

This document explains how to manage and maintain TeleFuser's model hash configurations, including the use of `weight_viewer.py` tool, configuration version control, and update workflows.

## Configuration Location

All model hash configurations are stored in:

```
telefuser/core/model_config.py
```

## Core Tool: Weight Viewer

TeleFuser provides the `weight_viewer.py` tool to assist with model analysis and management:

```bash
python tools/viewer/weight_viewer.py <model_path> [options]
```

### Basic Usage

```bash
# View single file model
python tools/viewer/weight_viewer.py /path/to/model.safetensors

# View sharded models (using wildcards)
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors"

# Show summary only (includes hash)
python tools/viewer/weight_viewer.py /path/to/model.safetensors --quiet

# Export as JSON for further analysis
python tools/viewer/weight_viewer.py /path/to/model.safetensors --export model_info.json
```

### Output Example

```
================================================================================
Model Weight Information Overview
================================================================================
Total parameters: 14.02B (14,022,154,432)
hash with shape: 4c3523c69fb7b24cf2db147a715b277f
Files loaded: 1
File list: ['/path/to/model.safetensors']

Data type distribution:
  torch.bfloat16: 14.02B (100.00%)

Detailed weight structure:
(Structurally identical modules have been merged, use --show-all to view full structure)
model
  transformer
    blocks x32
      norm1.scale                      | (2048,)              | torch.bfloat16  |     2.05K
      norm1.bias                       | (2048,)              | torch.bfloat16  |     2.05K
      ...
```

## Configuration Format

```python
model_loader_configs = [
    # Format: (keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
    (
        None,                                      # keys_hash (non-strict matching)
        "4c3523c69fb7b24cf2db147a715b277f",       # keys_hash_with_shape (strict matching)
        ["wan_video_decoder"],                     # model_names
        [TAEHV],                                   # model_classes
        "official",                                 # model_resource
    ),
    # ... more configurations
]
```

## Configuration Management Workflow

### Adding a New Model

#### 1. Obtain Model Files

```bash
# Confirm model files exist
ls /path/to/models/*.safetensors
```

#### 2. Use Weight Viewer to Analyze Model

```bash
# Get model hash and structure information
python tools/viewer/weight_viewer.py "/path/to/models/model.safetensors" --quiet
```

The `hash with shape` in the output is the `keys_hash_with_shape` needed for configuration.

#### 3. Analyze Model Structure in Detail (for implementing StateDictConverter)

```bash
# View complete structure for writing key mappings
python tools/viewer/weight_viewer.py "/path/to/models/model.safetensors" --max-depth 10 --export model_structure.json
```

Review the exported JSON file, analyze key naming patterns, and write the converter.

#### 4. Add to Configuration

Edit `telefuser/core/model_config.py` to add model configuration:

```python
from ..models.my_model import MyModel

model_loader_configs = [
    # ... existing configurations ...
    
    # MyModel - Standard version (from weight_viewer output)
    (
        None,  # Non-strict hash (optional)
        "4c3523c69fb7b24cf2db147a715b277f",  # Hash from weight_viewer
        ["my_model"],
        [MyModel],
        "official",  # or "diffusers"
    ),
]
```

#### 5. Verify Configuration

```bash
# Use weight_viewer to verify hash matches
python tools/viewer/weight_viewer.py "/path/to/models/model.safetensors" --quiet

# Then test loading
python -c "
from telefuser.core.module_manager import ModuleManager
mm = ModuleManager(device='cpu')
mm.load_model('/path/to/models/model.safetensors')
print('✓ Model loaded successfully!')
print('Available models:', mm.module_name)
"
```

### Batch Processing Multiple Model Variants

When there are multiple variants (like FP8, pruned versions), use scripts for batch processing:

```bash
#!/bin/bash
# scripts/batch_analyze_models.sh

MODEL_DIR="/path/to/models"

for model in "$MODEL_DIR"/*.safetensors; do
    echo "========================================"
    echo "Analyzing: $(basename "$model")"
    echo "========================================"
    python tools/viewer/weight_viewer.py "$model" --quiet
    echo ""
done
```

### Comparing Different Model Versions

```bash
# Analyze two versions of models
python tools/viewer/weight_viewer.py "/path/to/model_v1.safetensors" --export v1.json
python tools/viewer/weight_viewer.py "/path/to/model_v2.safetensors" --export v2.json

# Use diff tool to compare structural differences
diff <(jq '.weights_structure' v1.json) <(jq '.weights_structure' v2.json)
```

## Weight Viewer Advanced Usage

### Analyzing Sharded Models

```bash
# Automatically recognize and merge shard files
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors"

# Example: WanVideo 14B model (7 shards)
python tools/viewer/weight_viewer.py \
    "/models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-*.safetensors" \
    --quiet
```

### Viewing Specific Level Structure

```bash
# View deeper structure (default depth is 5)
python tools/viewer/weight_viewer.py /path/to/model.safetensors --max-depth 8

# View complete structure (no depth limit)
python tools/viewer/weight_viewer.py /path/to/model.safetensors --show-all
```

### Disabling Structure Merging

```bash
# Show full information for all repeated modules
python tools/viewer/weight_viewer.py /path/to/model.safetensors --no-merge
```

## Auxiliary Scripts

### Generate Configuration Template

Create script `tools/generate_config_template.py`:

> **Note**: Before running this script, ensure you have installed the project in development mode:
> ```bash
> pip install -e ".[dev]"
> ```

```python
#!/usr/bin/env python3
"""
Generate configuration template from weight_viewer output

Usage:
    python tools/generate_config_template.py <model_path> --name my_model --class MyModel
"""

import argparse
import json

from telefuser.utils.model_weight import hash_state_dict_keys


def generate_template(model_path, model_name, model_class, resource="official"):
    """Generate configuration template"""
    import glob
    
    # Handle wildcards
    files = sorted(glob.glob(model_path))
    if not files:
        print(f"Error: No files found matching {model_path}")
        sys.exit(1)
    
    # Load all weights
    from telefuser.utils.model_weight import load_state_dict
    all_weights = {}
    for f in files:
        all_weights.update(load_state_dict(f))
    
    # Compute hash
    hash_with_shape = hash_state_dict_keys(all_weights, with_shape=True)
    hash_without_shape = hash_state_dict_keys(all_weights, with_shape=False)
    
    # Generate configuration
    config = f'''    # {model_name}
    (
        "{hash_without_shape}",  # keys_hash (non-strict matching)
        "{hash_with_shape}",    # keys_hash_with_shape
        ["{model_name}"],
        [{model_class}],
        "{resource}",
    ),'''
    
    print("\n" + "="*60)
    print("Generated Configuration Template")
    print("="*60)
    print(config)
    print("\n" + "="*60)
    print(f"Model Statistics:")
    print(f"  Total tensors: {len(all_weights)}")
    print(f"  Files: {len(files)}")
    print("="*60 + "\n")
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate model config template")
    parser.add_argument("model_path", help="Model file path (supports wildcards)")
    parser.add_argument("--name", required=True, help="Model name (e.g., wan_video_dit)")
    parser.add_argument("--class", required=True, dest="model_class", help="Model class name (e.g., WanModel)")
    parser.add_argument("--resource", default="official", choices=["official", "diffusers"], help="Model source")
    
    args = parser.parse_args()
    generate_template(args.model_path, args.name, args.model_class, args.resource)


if __name__ == "__main__":
    main()
```

Usage:

```bash
python tools/generate_config_template.py \
    "/models/my_model.safetensors" \
    --name my_custom_dit \
    --class MyCustomDiT \
    --resource official
```

### Verify Configuration Integrity

> **Note**: Before running this script, ensure you have installed the project in development mode:
> ```bash
> pip install -e ".[dev]"
> ```

```python
#!/usr/bin/env python3
# tools/verify_configs.py

from telefuser.core.model_config import model_loader_configs

def verify():
    """Verify configurations"""
    print(f"Total configurations: {len(model_loader_configs)}\n")
    
    # Check for duplicates
    seen_hashes = {}
    for i, config in enumerate(model_loader_configs):
        keys_hash, keys_hash_with_shape, names, classes, resource = config
        
        if keys_hash_with_shape in seen_hashes:
            print(f"⚠️  Duplicate hash_with_shape at #{i} and #{seen_hashes[keys_hash_with_shape]}")
        else:
            seen_hashes[keys_hash_with_shape] = i
        
        print(f"#{i}: {names[0] if names else 'N/A':<30} {keys_hash_with_shape or 'N/A'}")
    
    print("\n✅ Verification complete")

if __name__ == "__main__":
    verify()
```

## Configuration Organization Recommendations

### Group by Model Family

```python
model_loader_configs = [
    # ==================== WanVideo ====================
    (None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "official"),
    (None, "1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "official"),
    
    # ==================== QwenImage ====================
    (None, "7a32c4aa3de140d48a5899ca505944b9", ["qwen_image_dit"], [QwenImageDiT], "official"),
    
    # ...
]
```

### Comment Conventions

```python
# Wan2.1 I2V 14B - 720P (from weight_viewer)
# Source: modelscope/Wan2.1-I2V-14B-720P
# Parameters: 14.02B
(
    None,
    "9269f8db9040a9d860eaca435be61814",
    ["wan_video_dit"],
    [WanModel],
    "official",
),
```

## FAQ

### Q: Weight Viewer hash doesn't match ModuleManager?

Ensure:
1. Weight Viewer loaded complete weights (including all shards)
2. Using same `with_shape=True` parameter
3. Files are intact (not corrupted)

### Q: How to handle models with dynamic shapes?

For models supporting multiple resolutions, use non-strict matching:

```python
# Use keys_hash (without shape)
(
    "q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",  # Only key hash
    None,  # Don't use shape hash
    ["flexible_model"],
    [FlexibleModel],
    "official",
),
```

### Q: How to batch add multiple model variants?

Create a script to iterate directory and generate configurations:

```bash
for f in /models/*.safetensors; do
    name=$(basename "$f" .safetensors)
    python tools/generate_config_template.py "$f" --name "${name}" --class MyModel
done
```

### Q: How to compute hash for sharded models?

`weight_viewer.py` automatically merges all shards and computes hash:

```bash
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors" --quiet
```

Ensure to use this merged hash in configuration.

## Related Documentation

- [Model Loading User Guide](./model_loading.md)
- [Adding New Model Development Guide](./adding_new_model.md)
