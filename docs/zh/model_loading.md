# TeleFuser 模型加载指南

本文档介绍 TeleFuser 中如何通过 `ModuleManager` 加载内部实现的模型。

## 概述

TeleFuser 采用 **Hash-based 自动模型识别**机制。系统通过计算模型权重文件的 key 的 MD5 hash 值，自动识别模型类型并初始化对应的 model class。这种设计确保了对内部实现模型的完全控制，避免意外错误。

## 核心概念

### ModuleManager

`ModuleManager` 是 TeleFuser 的模型加载管理器，负责：
- 自动识别模型类型（通过 weight hash）
- 加载并初始化模型权重
- 管理多个模型的生命周期

### Hash 匹配机制

```
模型文件 → 提取 state_dict keys → 计算 MD5 hash → 匹配预配置 → 初始化对应 model class
```

预配置的模型信息存储在 `telefuser/core/model_config.py` 中。

## 快速开始

### 基本用法

```python
from telefuser.core.module_manager import ModuleManager
import torch

# 创建 ModuleManager 实例
module_manager = ModuleManager(
    torch_dtype=torch.bfloat16,
    device="cpu"  # 加载时放在 CPU，后续可 offload
)

# 加载模型（自动识别类型）
module_manager.load_model("/path/to/model.safetensors")

# 或使用 load_models 批量加载
module_manager.load_models([
    "/path/to/vae.safetensors",
    "/path/to/text_encoder.safetensors",
])
```

### 在 Pipeline 中使用

```python
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline, Wan21VideoPipelineConfig

# 1. 加载模型
module_manager = ModuleManager(device="cpu")
module_manager.load_models([
    "/path/to/clip_encoder.pth",
    "/path/to/vae.safetensors",
    "/path/to/dit.safetensors",
    "/path/to/text_encoder.safetensors",
], torch_dtype=torch.bfloat16)

# 2. 初始化 Pipeline
pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
pipe_config = Wan21VideoPipelineConfig()
pipe.init(module_manager, pipe_config)

# 3. 获取特定模型（可选）
vae_model = module_manager.fetch_module("wan_video_vae")
text_encoder = module_manager.fetch_module("wan_video_text_encoder")
```

## 高级用法

### 指定数据类型

可以为不同模型指定不同的数据类型：

```python
# Image Encoder 使用 float16
module_manager.load_models(
    ["/path/to/image_encoder.pth"],
    torch_dtype=torch.float16
)

# DiT 和 VAE 使用 bfloat16
module_manager.load_models(
    ["/path/to/dit.safetensors", "/path/to/vae.safetensors"],
    torch_dtype=torch.bfloat16
)
```

### 低内存加载

启用 `low_cpu_mem_usage` 可减少 CPU 内存占用：

```python
module_manager.load_model(
    "/path/to/large_model.safetensors",
    low_cpu_mem_usage=True  # 不复制到 CPU，直接加载到目标设备
)
```

### 多文件模型加载

对于分片的模型（如 sharded safetensors）：

```python
module_manager.load_model([
    "/path/to/model-00001-of-00007.safetensors",
    "/path/to/model-00002-of-00007.safetensors",
    # ... 其他分片
], torch_dtype=torch.bfloat16)
```

### 获取已加载的模型

```python
# 获取单个模型
vae = module_manager.fetch_module("wan_video_vae")

# 获取模型及其来源路径
vae, path = module_manager.fetch_module("wan_video_vae", require_model_path=True)

# 当有多个同名模型时，指定索引
dit = module_manager.fetch_module("wan_video_dit", index=0)
```

### HuggingFace 模型加载

对于不在预配置 hash 列表中的模型，可以使用 HuggingFace 加载方式：

```python
# 从 HuggingFace 加载
module_manager.load_from_huggingface(
    module_path="stabilityai/stable-diffusion-xl-base-1.0",
    module_source="diffusers",  # 或 "transformers"
    module_name="sdxl_unet",
    torch_dtype=torch.bfloat16,
)
```

## 支持的模型格式

ModuleManager 支持以下模型文件格式：

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| Safetensors | `.safetensors` | 推荐格式，安全且高效 |
| PyTorch | `.bin`, `.pt`, `.pth`, `.ckpt` | 标准 PyTorch 格式 |

## 故障排除

### 模型无法识别

如果模型无法被自动识别，可能原因：

1. **模型未在预配置列表中**
   - 检查 `telefuser/core/model_config.py` 是否包含该模型的 hash
   - 如果是新模型，需要按照[开发文档](./adding_new_model.md)添加配置

2. **模型文件损坏或不完整**
   - 验证文件完整性
   - 重新下载模型文件

3. **使用了不支持的格式**
   - 转换为 `.safetensors` 格式

### 内存不足

```python
# 使用低内存模式
module_manager.load_model(
    "/path/to/model.safetensors",
    low_cpu_mem_usage=True
)

# 或者先加载到 CPU，后续再 offload
module_manager = ModuleManager(device="cpu")
module_manager.load_model(...)
# 然后在 Pipeline 配置中设置 offload 策略
```

### Hash 不匹配

如果看到日志中出现 hash 但不匹配：

```
load model /path/to/model.safetensors with state hash xxxxxxxxxx
```

这表示该模型不在预配置列表中。你需要：
1. 使用 `weight_viewer.py` 工具计算 hash：
   ```bash
   python tools/viewer/weight_viewer.py /path/to/model.safetensors --quiet
   ```
2. 按照开发文档添加模型配置

## 最佳实践

1. **始终在 CPU 上加载模型**
   ```python
   module_manager = ModuleManager(device="cpu")
   ```
   让 Pipeline 负责将模型移动到 GPU 并管理 offload。

2. **合理选择数据类型**
   - Image Encoder: `float16` 通常足够
   - DiT/VAE/Text Encoder: `bfloat16` 提供更好的数值稳定性
   - 如需 FP8 量化，加载时使用 `float8_e4m3fn`

3. **批量加载相关模型**
   ```python
   # 好：一次加载相关模型
   module_manager.load_models([vae_path, dit_path, text_encoder_path])
   
   # 避免：多次单独调用（除非需要不同 dtype）
   ```

4. **使用 Safetensors 格式**
   - 加载更快
   - 更安全（防止代码执行）
   - 更好的跨平台兼容性

5. **使用 weight_viewer.py 工具**
   ```bash
   # 在添加新模型前，先用工具分析
   python tools/viewer/weight_viewer.py /path/to/new_model.safetensors
   ```

## 相关文档

- [添加新模型开发指南](./adding_new_model.md)
- [Hash 配置管理指南](./hash_config_management.md)
- [Service 指南](./service.md)
