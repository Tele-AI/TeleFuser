# 添加新模型开发指南

本文档介绍如何为 TeleFuser 添加新的内部模型支持，包括计算模型 hash、添加配置以及实现必要的转换器。

## 概述

TeleFuser 使用 **Hash-based 自动识别机制**来确定模型类型。要将新模型接入系统，需要：

1. 实现模型类（继承 `BaseModel`）
2. 实现 `state_dict_converter` 转换器
3. 使用 `weight_viewer.py` 计算模型 hash
4. 添加配置并测试验证

## 步骤详解

### 步骤 1：实现模型类

创建模型类并继承 `BaseModel`（或根据模型类型选择合适的基类）：

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
        # ... 其他参数
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # ... 模型定义

    def forward(self, x, t, context, **kwargs):
        # 前向逻辑
        pass

    @classmethod
    def state_dict_converter(cls):
        """返回状态字典转换器类"""
        return MyCustomDiTStateDictConverter
```

#### 实现 `from_pretrained` 接口（可选）

模型可以可选地实现 `from_pretrained` 类方法，以便在 pipeline 示例中方便地加载模型。该方法提供统一的模型加载接口：

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
        # ... 其他参数使用内部默认值
    ):
        super().__init__()
        # ... 初始化逻辑

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "TextEncoder":
        """Load TextEncoder from pretrained checkpoint.

        Args:
            pretrained_model_name_or_path: 模型路径
            torch_dtype: 模型精度（默认：bfloat16）
            **kwargs: 忽略未知参数以保持兼容性

        Returns:
            加载完成的 TextEncoder 实例
        """
        # 从 torch_dtype 确定精度
        precision = "bf16" if torch_dtype == torch.bfloat16 else "fp16"

        # 所有内部参数使用合理的默认值
        return cls(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision=precision,
            text_encoder_path=pretrained_model_name_or_path,
            tokenizer_type="llm",
            # ... 其他内部默认值
        )
```

**`from_pretrained` 实现原则：**
1. 只对外暴露必要参数，如 `pretrained_model_name_or_path` 和 `torch_dtype`
2. 所有其他参数在内部设置合理的默认值
3. 接受 `**kwargs` 以保持兼容性，但忽略未知参数
4. 返回完全初始化的模型实例

**注意：** 如果未实现 `from_pretrained`，仍可使用 `ModuleManager.load_model()` 配合 hash 自动识别加载模型，或手动实例化模型后通过 `add_module()` 添加。

#### VAE 模型示例

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
            pretrained_model_name_or_path: VAE 检查点目录路径
            torch_dtype: 模型精度（默认：bfloat16）
            **kwargs: 忽略未知参数以保持兼容性

        Returns:
            加载完成的 HunyuanVideoVAE 实例
        """
        # 从 JSON 加载配置
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # 使用配置创建模型
        model = cls(
            in_channels=config.get("in_channels", 3),
            out_channels=config.get("out_channels", 3),
            # ... 其他配置参数
        )

        # 加载状态字典
        state_dict = load_state_dict(os.path.join(pretrained_model_name_or_path, "diffusion_pytorch_model.safetensors"))
        model.load_state_dict(state_dict, strict=False)

        return model.to(dtype=torch_dtype)
```

**注意：** 分块/切片设置应在运行时由 VAE stage 处理，而非在模型初始化时设置。

### 步骤 2：实现 StateDictConverter

转换器负责将不同来源的权重格式转换为内部格式：

```python
# telefuser/models/my_custom_dit.py

class MyCustomDiTStateDictConverter:
    """
    转换不同来源的 state_dict 到内部格式
    """
    
    @staticmethod
    def from_official(state_dict):
        """
        从 Civitai/Direct 格式转换
        
        Args:
            state_dict: 原始状态字典
            
        Returns:
            转换后的 state_dict，或 (state_dict, extra_kwargs) 元组
        """
        # 创建 key 映射
        rename_dict = {
            "input_blocks.0.0.weight": "conv_in.weight",
            "input_blocks.0.0.bias": "conv_in.bias",
            # ... 更多映射
        }
        
        converted_state_dict = {}
        for old_key, new_key in rename_dict.items():
            if old_key in state_dict:
                converted_state_dict[new_key] = state_dict[old_key]
        
        # 如果需要根据权重推断模型参数，返回 extra_kwargs
        extra_kwargs = {
            "hidden_size": 2048,  # 从权重推断或硬编码
            "num_layers": 32,
        }
        
        return converted_state_dict, extra_kwargs
    
    @staticmethod
    def from_diffusers(state_dict):
        """从 Diffusers 格式转换"""
        # 类似实现
        pass
```

### 步骤 3：使用 Weight Viewer 计算模型 Hash

使用内置的 `weight_viewer.py` 工具分析模型：

```bash
# 快速获取 hash
python tools/viewer/weight_viewer.py /path/to/your/model.safetensors --quiet
```

输出示例：

```
Total parameters: 14.02B
Files: 1
hash with shape: 4c3523c69fb7b24cf2db147a715b277f
```

**记录 `hash with shape` 值**，这将被添加到配置中。

对于更详细的分析（查看模型结构以帮助实现 StateDictConverter）：

```bash
# 查看完整结构并导出
python tools/viewer/weight_viewer.py /path/to/your/model.safetensors \
    --max-depth 10 \
    --export model_structure.json
```

**使用 weight_viewer 的优势：**
- 自动处理分片模型（使用通配符 `model-*.safetensors`）
- 显示参数统计和数据类型分布
- 自动合并结构相同的模块（如 transformer blocks）
- 导出 JSON 便于后续分析

#### 分片模型处理

如果模型分为多个文件：

```bash
# 自动合并所有分片并计算 hash
python tools/viewer/weight_viewer.py "/path/to/model-*.safetensors" --quiet
```

**注意**：在添加到 `model_config.py` 时，确保 hash 是基于**合并后的完整权重**计算的。

### 步骤 4：添加模型配置

编辑 `telefuser/core/model_config.py`，添加模型配置。

首先，从 weight_viewer 输出中获取信息：

```bash
$ python tools/viewer/weight_viewer.py /path/to/my_model.safetensors --quiet

Total parameters: 6.91B
Files: 1
hash with shape: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
```

然后添加配置：

```python
from ..models.my_custom_dit import MyCustomDiT

model_loader_configs = [
    # ... 现有配置 ...
    
    # MyCustomDiT - Standard version (from weight_viewer: hash=a1b2c3d4...)
    # Parameters: 6.91B
    (
        None,                                  # hash without shape (可选，用于非严格匹配)
        "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",   # hash with shape（来自 weight_viewer）
        ["my_custom_dit"],                     # model_name（用于 fetch_module）
        [MyCustomDiT],                         # model_class
        "official",                             # model_resource: "official" 或 "diffusers"
    ),
]
```

#### 添加多个变体

如果同一模型有多个变体（如 FP8 版本）：

```bash
# 分析 FP8 版本
$ python tools/viewer/weight_viewer.py /path/to/my_model_fp8.safetensors --quiet

Total parameters: 6.91B
Files: 1
hash with shape: b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7  # 不同的 hash！
```

添加到配置：

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

**提示**：如果变体的 tensor shape 不同（如 pruned 模型），考虑使用非严格匹配（仅使用 `keys_hash`）。

配置字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `keys_hash` | `str \| None` | 仅基于 key 名称的 hash（不含 shape）。用于 shape 可能变化的变体 |
| `keys_hash_with_shape` | `str` | 包含 key 名称和 shape 的 hash。严格匹配，推荐优先使用 |
| `model_names` | `list[str]` | 模型标识名称列表，用于 `fetch_module()` |
| `model_classes` | `list[type]` | 对应的模型类列表 |
| `model_resource` | `str` | 权重来源格式：`"official"` 或 `"diffusers"` |

### 步骤 5：测试验证

创建测试脚本验证模型加载：

```python
# tests/test_my_custom_model_loading.py
import torch
import pytest
from telefuser.core.module_manager import ModuleManager

def test_my_custom_dit_loading():
    """测试 MyCustomDiT 模型加载"""
    module_manager = ModuleManager(device="cpu")

    # 测试自动识别
    module_manager.load_model(
        "/path/to/your/model.safetensors",
        torch_dtype=torch.bfloat16
    )

    # 验证可以获取模型
    model = module_manager.fetch_module("my_custom_dit")
    assert model is not None

    # 验证模型类型
    from telefuser.models.my_custom_dit import MyCustomDiT
    assert isinstance(model, MyCustomDiT)

    print("✓ MyCustomDiT loading test passed!")

if __name__ == "__main__":
    test_my_custom_dit_loading()
```

运行测试：

```bash
pytest tests/test_my_custom_model_loading.py -v
```

## 在 Pipeline 示例中使用模型

创建 pipeline 示例时，使用 `from_pretrained` 接口和 `add_module` 模式：

### 基本模式

```python
import os
import torch
from telefuser.utils.logging import logger
from telefuser.core.module_manager import ModuleManager
from telefuser.models.hunyuan_video_vae import HunyuanVideoVAE
from telefuser.models.hunyuan_video_text_encoder import HunyuanVideoTextEncoder

def get_pipeline(model_root: str = "/path/to/models"):
    """创建并初始化包含所有模型的 pipeline。"""
    module_manager = ModuleManager(device="cpu")

    # 1. 使用 from_pretrained 加载 VAE
    vae_path = os.path.join(model_root, "vae")
    logger.info(f"Loading VAE from {vae_path}")
    vae = HunyuanVideoVAE.from_pretrained(vae_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(vae, name="vae")

    # 2. 使用 from_pretrained 加载 TextEncoder
    text_encoder_path = os.path.join(model_root, "text_encoder", "llm")
    logger.info(f"Loading TextEncoder from {text_encoder_path}")
    text_encoder = HunyuanVideoTextEncoder.from_pretrained(text_encoder_path, torch_dtype=torch.bfloat16)
    module_manager.add_module(text_encoder, name="text_encoder")

    # 3. 其他模型类似加载...
    # transformer = HunyuanVideoDiT.from_pretrained(transformer_path, torch_dtype=torch.bfloat16)
    # module_manager.add_module(transformer, name="hunyuan_video_dit")

    # 4. 创建并初始化 pipeline
    # pipe = HunyuanVideo15Pipeline(device="cuda", torch_dtype=torch.bfloat16)
    # pipe.init(module_manager, pipe_config)

    return pipe
```

### 关键原则

1. **所有模型使用 `from_pretrained` 加载** - 提供一致的接口
2. **只对外暴露模型路径** - 所有其他参数应为内部默认值
3. **使用有意义的名称调用 `add_module`** - 如 `"vae"`、`"text_encoder"`、`"hunyuan_video_dit"` 等，pipeline stages 使用这些名称获取模块
4. **由 stage 处理运行时设置** - 分块、切片等运行时配置应由 pipeline stage 处理，而非模型初始化时

### 模块命名规范

| 模块类型 | 推荐名称 | 使用方 |
|---------|---------|--------|
| VAE | `"vae"` | `HunyuanVideoVAEStage` |
| Text Encoder | `"text_encoder"` | `HunyuanVideoTextEncodingStage` |
| DiT/Transformer | `"hunyuan_video_dit"` | `HunyuanVideoDenoisingStage` |
| Vision Encoder (I2V) | `"vision_encoder"` | `HunyuanVideoImageEncodingStage` |
| Upsampler (SR) | `"upsampler"` | `HunyuanVideoUpsamplerStage` |
| Scheduler | `"scheduler"` | Pipeline init |

## 特殊情况处理

### 处理 Shape 变化的变体

某些模型变体（如 FP8 量化版、pruned 版）可能有不同的 tensor shape：

```python
# 主版本（严格匹配）
(
    None,  # 不需要非严格 hash
    "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    ["my_model"],
    [MyModel],
    "official",
),

# FP8 版本（shape 不同，使用非严格匹配）
(
    "q7r8s9t0u1v2w3x4y5z6a7b8c9d0e1f2",  # 仅 key hash
    None,  # 不使用 shape hash（因为 shape 不同）
    ["my_model_fp8"],
    [MyModelFP8],  # 可能需要不同的类
    "official",
),
```

### 多组件模型

某些模型文件包含多个组件（如 VAE encoder + decoder）：

```python
# 在 state_dict_converter 中分离组件
@staticmethod
def from_official(state_dict):
    encoder_dict = {}
    decoder_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            encoder_dict[key[8:]] = value  # 去掉 "encoder." 前缀
        elif key.startswith("decoder."):
            decoder_dict[key[8:]] = value
    
    # 返回合并的 dict，在模型类中处理
    combined_dict = {
        "encoder": encoder_dict,
        "decoder": decoder_dict,
    }
    
    return combined_dict, {"has_separate_components": True}
```

### 支持多种来源格式

如果模型可能来自不同来源（Civitai、HuggingFace Diffusers）：

```python
class MyModelStateDictConverter:
    @staticmethod
    def from_official(state_dict):
        # Civitai 格式转换
        return convert_official_format(state_dict)
    
    @staticmethod
    def from_diffusers(state_dict):
        # Diffusers 格式转换
        return convert_diffusers_format(state_dict)
```

然后在配置中指定正确的 `model_resource`。

## 调试技巧

### 1. 使用 Weight Viewer 查看模型结构

```bash
# 查看所有 keys 和 shape
python tools/viewer/weight_viewer.py /path/to/model.safetensors --show-all

# 导出为 JSON 便于程序处理
python tools/viewer/weight_viewer.py /path/to/model.safetensors --export model_info.json
```
### 2. 检查 hash 匹配过程

```python
from telefuser.utils.model_weight import load_state_dict, hash_state_dict_keys
from telefuser.core.model_config import model_loader_configs

sd = load_state_dict("/path/to/model.safetensors")
hash_with_shape = hash_state_dict_keys(sd, with_shape=True)
hash_without_shape = hash_state_dict_keys(sd, with_shape=False)

print(f"Model hash (with shape): {hash_with_shape}")
print(f"Model hash (without shape): {hash_without_shape}")

# 检查是否在配置中
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

### 3. 验证转换器输出

```python
from telefuser.models.my_custom_dit import MyCustomDiT
from telefuser.utils.model_weight import load_state_dict

sd = load_state_dict("/path/to/model.safetensors")
converter = MyCustomDiT.state_dict_converter()
converted, extra_kwargs = converter.from_official(sd)

print(f"Extra kwargs: {extra_kwargs}")
print(f"Converted keys: {list(converted.keys())[:10]}")

# 尝试初始化
model = MyCustomDiT(**extra_kwargs)
model.load_state_dict(converted, strict=False)  # 先用非严格模式测试
print("✓ Model initialized successfully!")
```

### 4. 快速验证配置

```bash
# 修改配置后，快速验证 hash 是否匹配
python -c "
from telefuser.core.module_manager import ModuleManager
mm = ModuleManager(device='cpu')
mm.load_model('/path/to/your/model.safetensors')
print('✓ Configuration is correct!')
print(f'Loaded models: {mm.module_name}')
"
```

## 最佳实践

1. **保持配置有序**
   - 按模型类型分组
   - 同一模型的不同变体放在一起
   - 添加注释说明版本差异

2. **使用严格匹配优先**
   - 尽可能提供 `keys_hash_with_shape`
   - 仅在 shape 可能变化时使用非严格匹配

3. **详细记录变体**
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

4. **测试所有变体**
   - 原始版本
   - FP8 量化版本
   - Pruned 版本
   - 不同来源格式（Civitai vs Diffusers）

5. **命名规范**
   - `model_names` 使用小写下划线格式
   - 前缀表示模型家族：`wan_video_`, `qwen_image_`, `flashvsr_`

6. **充分利用 weight_viewer**
   ```bash
   # 在添加配置前分析模型
   python tools/viewer/weight_viewer.py /path/to/model.safetensors --export model_info.json
   
   # 比较不同版本的差异
   python tools/viewer/weight_viewer.py /path/to/model_v1.safetensors --export v1.json
   python tools/viewer/weight_viewer.py /path/to/model_v2.safetensors --export v2.json
   diff v1.json v2.json
   ```

## 示例：完整的新模型集成

参考以下文件了解完整实现：

- 模型实现：`telefuser/models/wan_video_dit.py`
- 配置定义：`telefuser/core/model_config.py`（WanModel 相关配置）
- 使用示例：`examples/wan_video/wan21_14b_image_to_video_h100.py`

## 优化模型推理

完成模型集成后，可以通过以下方式优化推理性能和显存使用。

### 1. 复用优化的算子

TeleFuser 的 `ops` 模块提供了高性能的神经网络算子实现。在新模型中复用这些算子可以获得最佳性能：

| 算子 | 用途 | 性能优化 |
|------|------|----------|
| `RMSNorm` / `LayerNorm` | 归一化层 | tf_kernel > Triton > PyTorch |
| `FeedForward` | 前馈网络 | 支持 GEGLU/SwiGLU |
| `attention` | 注意力计算 | Flash Attention 2/3/4, SageAttention |
| `LinearFP8` | 量化线性层 | FP8 推理 |

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

详细文档请参考 [Ops 模块文档](./ops.md)。

### 2. 多卡推理

对于大模型或长序列生成，可以使用多种并行策略：

```python
from telefuser.core.config import ParallelConfig

# Ulysses 序列并行（2 GPU）
config = ParallelConfig(
    device_ids=[0, 1],
    sp_ulysses_degree=2,
)
pipe_config.dit_config.parallel_config = config
pipe_config.enable_denoising_parallel = True

# CFG + Ulysses（4 GPU）
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

| 策略 | 适用场景 | 说明 |
|------|----------|------|
| Ulysses | 中等长度序列 | All-to-All 通信 |
| Ring | 超长序列 | P2P 通信，支持任意长度 |
| USP | 大规模并行 | Ulysses + Ring 组合 |
| CFG 并行 | CFG 加速 | 正/负 prompt 并行计算 |
| 流水线并行 | 大模型推理 | 层分割到多 GPU |

详细配置请参考 [并行推理指南](./parallel.md)。

### 3. 模型量化

使用 `tools/convert/converter.py` 对模型进行量化，显著减少显存占用：

**FP8 量化**（推荐）：
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

**INT8 量化**：
```bash
python tools/convert/converter.py \
    --source /path/to/model/ \
    --output /path/to/output \
    --linear_dtype torch.int8 \
    --model_type wan_dit \
    --quantized \
    --single_file
```

支持的量化类型：`int8`、`fp8`、`nvfp4`、`mxfp4`、`mxfp6`、`mxfp8`。

详细使用方法请参考 `tools/convert/README.md`。

### 4. CPU 卸载 (Offloading)

当显存不足时，可以使用 CPU 卸载将模型权重临时移到 CPU：

```python
from telefuser.core.config import OffloadConfig, WeightOffloadType

# 异步 CPU 卸载（推荐）
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=1,
)
```

| 策略 | 显存节省 | 速度影响 | 适用场景 |
|------|----------|----------|----------|
| `NO_CPU_OFFLOAD` | 无 | 最快 | 显存充足 |
| `MODEL_CPU_OFFLOAD` | ~50% | 中等 | 中度受限 |
| `ASYNC_CPU_OFFLOAD` | ~60-70% | 低 | 8-16GB 显存 |
| `SEQUENTIAL_CPU_OFFLOAD` | 最大 | 最慢 | <8GB 显存 |

详细配置请参考 [CPU 卸载指南](./offload.md)。

### 5. 组合优化示例

以下是一个完整的优化配置示例：

```python
from telefuser.core.config import (
    ParallelConfig,
    AttentionConfig,
    AttnImplType,
    OffloadConfig,
    WeightOffloadType,
)

# 多卡 + 注意力优化 + 卸载
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

## 相关文档

- [模型加载用户指南](./model_loading.md)
- [Hash 配置管理指南](./hash_config_management.md)
- [Ops 模块文档](./ops.md) - 神经网络算子实现（激活函数、归一化层、注意力等）
- [并行推理指南](./parallel.md) - 多 GPU 推理配置
- [CPU 卸载指南](./offload.md) - 显存优化策略
