# Hash 配置管理指南

本文档介绍如何管理和维护 TeleFuser 的模型 hash 配置，包括 `weight_viewer.py` 工具的使用、配置版本控制和更新流程。

## 为什么使用 Hash 识别？

TeleFuser 采用 **基于 hash 的自动识别机制**，而非依赖配置文件（如 `model_index.json`）。这种设计具有以下显著优势：

### 1. 无需配置文件

与其他依赖 `model_index.json` 等元数据文件的框架不同，TeleFuser 可以直接从权重文件加载模型：

```python
# TeleFuser - 直接加载任意权重文件
from telefuser.core.module_manager import ModuleManager

mm = ModuleManager(torch_dtype=torch.bfloat16)
mm.load_model("downloads/model.safetensors")  # 无需任何配置！

# 支持的场景：
# - 单个 .safetensors 文件
# - 官方发布的原始权重（非 Diffusers 格式）
# - 自行训练的权重
# - 任意目录结构
```

**对比：**

| 场景 | TeleFuser | 需要配置文件的框架 |
|------|-----------|-------------------|
| 单个 `.safetensors` 文件 | ✅ 直接支持 | ❌ 需要配置 |
| 没有 `model_index.json` | ✅ 直接支持 | ❌ 无法加载 |
| 官方原始权重（非 Diffusers） | ✅ 直接支持 | ⚠️ 需要转换 |
| 自定义目录结构 | ✅ 直接支持 | ❌ 必须符合约定 |
| 多文件分片权重 | ✅ 自动合并 | ⚠️ 需要正确命名 |

### 2. 强模型校验

Hash 匹配提供了强大的模型校验能力：

| 校验类型 | Hash 匹配 | 字符串匹配（如 `_class_name`） |
|----------|-----------|-------------------------------|
| 加载错误的模型文件 | ❌ Hash 不匹配，拒绝加载 | ⚠️ 可能加载成功，运行时报错 |
| 模型 key 被修改 | ❌ Hash 不匹配，拒绝加载 | ⚠️ 无法检测 |
| 模型结构变化 | ❌ 拒绝（`shape=True` 时） | ⚠️ 无法检测 |
| 加载未知模型 | ❌ 未注册，拒绝加载 | ⚠️ 可能使用兜底方案 |

```python
# Hash 相当于校验和 - 确保模型完整性
# 同一模型不同量化版本会得到不同的 hash：
(
    None, "9269f8db9040a9d860eaca435be61814",  # FP16 版本
    ["wan_video_dit"], [WanModel], "official",
),
(
    None, "4cf556355bc7e9b6545b38f4930f60b1",  # FP8 版本（hash 不同！）
    ["wan_video_dit"], [WanModel], "official",
),
```

### 3. 精确区分模型变体

同一模型类的不同权重变体可以通过 hash 精确区分：

```python
# 同一个 WanModel 类，通过 hash 区分不同权重
# Wan2.1 T2V 1.3B
(None, "9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "official"),
# Wan2.2 I2V A14B
(None, "5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "official"),
# Wan2.2 TI2V 5B
(None, "1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "official"),
```

这可以防止：
- 将 T2V 权重加载到 I2V pipeline
- 混淆不同大小的模型
- 使用错误的量化变体

### 4. 安全与审计优势

- **完整性验证**：Hash 确认权重未被篡改
- **版本控制**：追踪正在使用的确切权重
- **可复现性**：相同 hash = 相同模型行为保证
- **供应链安全**：验证权重与可信来源的预期 hash 匹配

### 5. 开发者友好

```python
# 开发者无需知道确切的模型类型即可加载
mm.load_model("/downloads/unknown_model.safetensors")
model = mm.fetch_module("wan_video_dit")  # Hash 自动识别！

# 适用于任何环境：
# - 研究：下载任意 .safetensors 即可运行
# - 生产：加载前验证 hash
# - CI/CD：确保测试的是正确的模型
```

### 设计权衡

| 方面 | Hash 匹配 (TeleFuser) | 配置文件匹配 |
|------|----------------------|-------------|
| 正确性 | ✅ 强校验 | ⚠️ 弱校验 |
| 新增模型便利性 | ⚠️ 需要注册 hash | ✅ 自动检测 |
| 支持任意模型 | ⚠️ 必须在注册表中 | ✅ 有兜底方案 |
| 最适合 | 生产环境、关键任务 | 原型开发、研究实验 |

**总结**：TeleFuser 的 hash 识别机制优先考虑**正确性和可靠性**，适合加载错误模型可能导致严重问题的生产环境。

## 配置位置

所有模型 hash 配置存储在：

```
telefuser/core/model_config.py
```

## 核心工具：Weight Viewer

TeleFuser 提供了 `weight_viewer.py` 工具来辅助模型分析和管理：

```bash
python tools/viewer/weight_viewer.py <model_path> [options]
```

### 基本用法

```bash
# 查看单文件模型
python tools/viewer/weight_viewer.py /path/to/model.safetensors

# 查看分片模型（使用通配符）
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors"

# 仅显示摘要信息（包含 hash）
python tools/viewer/weight_viewer.py /path/to/model.safetensors --quiet

# 导出为 JSON 以便进一步分析
python tools/viewer/weight_viewer.py /path/to/model.safetensors --export model_info.json
```

### 输出示例

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
(结构相同的模块已合并，使用 --show-all 查看完整结构)
model
  transformer
    blocks x32
      norm1.scale                      | (2048,)              | torch.bfloat16  |     2.05K
      norm1.bias                       | (2048,)              | torch.bfloat16  |     2.05K
      ...
```

## 配置格式

```python
model_loader_configs = [
    # 格式: (keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource)
    (
        None,                                      # keys_hash (非严格匹配)
        "4c3523c69fb7b24cf2db147a715b277f",       # keys_hash_with_shape (严格匹配)
        ["wan_video_decoder"],                     # model_names
        [TAEHV],                                   # model_classes
        "official",                                 # model_resource
    ),
    # ... 更多配置
]
```

## 配置管理流程

### 添加新模型

#### 1. 获取模型文件

```bash
# 确认模型文件存在
ls /path/to/models/*.safetensors
```

#### 2. 使用 Weight Viewer 分析模型

```bash
# 获取模型 hash 和结构信息
python tools/viewer/weight_viewer.py "/path/to/models/model.safetensors" --quiet
```

输出中的 `hash with shape` 就是需要添加到配置中的 `keys_hash_with_shape`。

#### 3. 详细分析模型结构（用于实现 StateDictConverter）

```bash
# 查看完整结构，用于编写 key 映射
python tools/viewer/weight_viewer.py "/path/to/models/model.safetensors" --max-depth 10 --export model_structure.json
```

查看导出的 JSON 文件，分析 key 的命名规律，编写转换器。

#### 4. 添加到配置

编辑 `telefuser/core/model_config.py`，添加模型配置：

```python
from ..models.my_model import MyModel

model_loader_configs = [
    # ... 现有配置 ...
    
    # MyModel - Standard version (from weight_viewer output)
    (
        None,  # 非严格 hash（可选）
        "4c3523c69fb7b24cf2db147a715b277f",  # 从 weight_viewer 获取的 hash
        ["my_model"],
        [MyModel],
        "official",  # 或 "diffusers"
    ),
]
```

#### 5. 验证配置

```bash
# 使用 weight_viewer 验证 hash 是否匹配
python tools/viewer/weight_viewer.py "/path/to/models/model.safetensors" --quiet

# 然后测试加载
python -c "
from telefuser.core.module_manager import ModuleManager
mm = ModuleManager(device='cpu')
mm.load_model('/path/to/models/model.safetensors')
print('✓ Model loaded successfully!')
print('Available models:', mm.module_name)
"
```

### 批量处理多个模型变体

当有多个变体（如 FP8、pruned 版本）时，可以使用脚本批量处理：

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

### 比较不同版本模型

```bash
# 分析两个版本的模型
python tools/viewer/weight_viewer.py "/path/to/model_v1.safetensors" --export v1.json
python tools/viewer/weight_viewer.py "/path/to/model_v2.safetensors" --export v2.json

# 使用 diff 工具比较结构差异
diff <(jq '.weights_structure' v1.json) <(jq '.weights_structure' v2.json)
```

## Weight Viewer 高级用法

### 分析分片模型

```bash
# 自动识别和合并分片文件
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors"

# 示例：WanVideo 14B 模型（7个分片）
python tools/viewer/weight_viewer.py \
    "/models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-*.safetensors" \
    --quiet
```

### 查看特定层级结构

```bash
# 查看更深的结构（默认深度为5）
python tools/viewer/weight_viewer.py /path/to/model.safetensors --max-depth 8

# 查看完整结构（无深度限制）
python tools/viewer/weight_viewer.py /path/to/model.safetensors --show-all
```

### 禁用结构合并

```bash
# 显示所有重复模块的完整信息
python tools/viewer/weight_viewer.py /path/to/model.safetensors --no-merge
```

## 辅助脚本

### 生成配置模板

创建脚本 `tools/generate_config_template.py`：

> **注意**: 运行此脚本前，请确保已安装项目到开发模式：
> ```bash
> pip install -e ".[dev]"
> ```

```python
#!/usr/bin/env python3
"""
根据 weight_viewer 输出生成配置模板

Usage:
    python tools/generate_config_template.py <model_path> --name my_model --class MyModel
"""

import argparse
import json

from telefuser.utils.model_weight import hash_state_dict_keys


def generate_template(model_path, model_name, model_class, resource="official"):
    """生成配置模板"""
    import glob
    
    # 处理通配符
    files = sorted(glob.glob(model_path))
    if not files:
        print(f"Error: No files found matching {model_path}")
        sys.exit(1)
    
    # 加载所有权重
    from telefuser.utils.model_weight import load_state_dict
    all_weights = {}
    for f in files:
        all_weights.update(load_state_dict(f))
    
    # 计算 hash
    hash_with_shape = hash_state_dict_keys(all_weights, with_shape=True)
    hash_without_shape = hash_state_dict_keys(all_weights, with_shape=False)
    
    # 生成配置
    config = f'''    # {model_name}
    (
        "{hash_without_shape}",  # keys_hash (非严格匹配)
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

使用：

```bash
python tools/generate_config_template.py \
    "/models/my_model.safetensors" \
    --name my_custom_dit \
    --class MyCustomDiT \
    --resource official
```

### 验证配置完整性

> **注意**: 运行此脚本前，请确保已安装项目到开发模式：
> ```bash
> pip install -e ".[dev]"
> ```

```python
#!/usr/bin/env python3
# tools/verify_configs.py

from telefuser.core.model_config import model_loader_configs

def verify():
    """验证配置"""
    print(f"Total configurations: {len(model_loader_configs)}\n")
    
    # 检查重复
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

## 配置组织建议

### 按模型家族分组

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

### 注释规范

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

## 常见问题

### Q: Weight Viewer 显示的 hash 与 ModuleManager 不匹配？

确保：
1. Weight Viewer 加载了完整的权重（包括所有分片）
2. 使用相同的 `with_shape=True` 参数
3. 检查文件是否完整（没有损坏）

### Q: 如何处理动态 shape 的模型？

对于支持多种分辨率的模型，使用非严格匹配：

```python
# 使用 keys_hash（不包含 shape）
(
    "q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",  # 仅 key hash
    None,  # 不使用 shape hash
    ["flexible_model"],
    [FlexibleModel],
    "official",
),
```

### Q: 如何批量添加多个模型变体？

创建一个脚本遍历目录并生成配置：

```bash
for f in /models/*.safetensors; do
    name=$(basename "$f" .safetensors)
    python tools/generate_config_template.py "$f" --name "${name}" --class MyModel
done
```

### Q: 分片模型的 hash 如何计算？

`weight_viewer.py` 会自动合并所有分片并计算 hash：

```bash
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors" --quiet
```

确保在配置中使用这个合并后的 hash。

## 相关文档

- [模型加载用户指南](./model_loading.md)
- [添加新模型开发指南](./adding_new_model.md)
