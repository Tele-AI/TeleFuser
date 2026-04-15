# 添加新 Pipeline Example 指南

本文档介绍如何在 TeleFuser 中创建新的 pipeline 示例，遵循现有示例（如 `wan_video`）的模式。

## 概述

Pipeline 示例是独立的 Python 脚本，演示如何使用 TeleFuser pipeline 进行推理。每个示例应该：

1. 自包含且可直接运行
2. 支持命令行参数配置
3. 与 TeleFuser 服务兼容（`telefuser serve`）
4. 文档清晰，命名规范明确

## 文件结构与命名

### 目录组织

示例按模型家族组织：

```
examples/
├── wan_video/              # WanVideo 生成示例
│   ├── wan21_*.py          # Wan2.1 模型示例
│   ├── wan22_*.py          # Wan2.2 模型示例
├── qwen_image/             # Qwen-Image 生成示例
├── hunyuan_video/          # HunyuanVideo 生成示例
├── z_image/                # Z-Image 生成示例
├── liveact/                # LiveAct 示例
└── ...
```

### 命名规范

遵循此模式：`{model_version}_{feature}_{hardware/config}.py`

| 组成部分 | 示例 | 描述 |
|---------|------|------|
| `model_version` | `wan21_14b`, `wan22_5b`, `qwen_image` | 模型家族和版本 |
| `feature` | `t2v`, `i2v`, `t2i`, `lora`, `distill` | 任务类型或特性 |
| `hardware/config` | `h100`, `hf`, `radial`, `cache_calibrate` | 硬件目标或特殊配置 |

**命名示例：**
- `wan21_14b_text_to_video_h100.py` - Wan2.1 14B T2V，适用于 H100
- `wan21_1_3b_text_to_video_hf.py` - Wan2.1 1.3B T2V，使用 HF 加载
- `wan22_14b_image_to_video_lora_h100.py` - Wan2.2 14B I2V，带 LoRA

## 示例文件结构

标准示例文件遵循此模板：

```python
"""简要描述本示例的功能。

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
from telefuser.utils.video import save_video  # 图像用 save_image

# ============================================================================
# 配置部分
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
    # ... 其他参数
)

# ============================================================================
# 模型加载部分
# ============================================================================

def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """加载并初始化 pipeline。
    
    Args:
        parallelism: 并行 GPU 数量（必填参数）
        model_root: 模型权重路径（必填参数）
        
    Returns:
        已初始化的 pipeline 实例
    """
    module_manager = ModuleManager(device="cpu")
    # 加载模型...
    
    pipe = PipelineClass(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = PipelineConfigClass()
    # 配置 pipeline...
    
    pipe.init(module_manager, pipe_config)
    return pipe

# ============================================================================
# 推理部分
# ============================================================================

def run(pipeline, prompt, negative_prompt="", seed=PPL_CONFIG["seed"], **kwargs):
    """使用 pipeline 执行推理。
    
    Args:
        pipeline: 已加载的 pipeline 实例
        prompt: 输入提示词
        negative_prompt: 负面提示词
        seed: 随机种子
        **kwargs: 其他参数
        
    Returns:
        生成的输出（视频帧、图像等）
    """
    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        # ... 其他参数从 PPL_CONFIG 获取
    )
    return video

def run_with_file(pipeline, prompt, negative_prompt, seed, output_path, **kwargs):
    """执行推理并保存到文件（可选，用于服务兼容性）。"""
    output = run(pipeline, prompt, negative_prompt, seed, **kwargs)
    save_video(output, output_path, fps=PPL_CONFIG["target_fps"], quality=6)

# ============================================================================
# CLI 入口
# ============================================================================

@click.command()
@click.option("--gpu_num", default=1, help="使用的 GPU 数量")
@click.option("--prompt", default="...", help="输入提示词")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="模型路径")
@click.option("--seed", default=PPL_CONFIG["seed"], help="随机种子")
def main(gpu_num, prompt, model_root, seed):
    """CLI help 中显示的简要描述。"""
    pipe = get_pipeline(gpu_num, model_root)
    
    start = time.time()
    output = run(pipe, prompt, seed=seed)
    elapsed_time = time.time() - start
    
    print(f"生成时间: {elapsed_time:.2f} 秒")
    
    # 保存结果
    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)
    save_video(output, output_path, fps=16, quality=6)
    
    del pipe

if __name__ == "__main__":
    main()
```

## 两种加载模式

### 模式 1：Hash-based 自动识别（推荐用于本地权重）

使用 `ModuleManager.load_model()` 加载本地权重文件。TeleFuser 通过 hash 自动识别模型类型。

```python
def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """使用 hash 自动识别加载并初始化 pipeline。
    
    Args:
        parallelism: 并行 GPU 数量（必填）
        model_root: 模型权重路径（必填）
    """
    module_manager = ModuleManager(device="cpu")
    
    # 加载 VAE（单文件）
    module_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    
    # 加载 DiT（分片文件 - 使用列表）
    dit_path_list = [
        f"{model_root}/diffusion_pytorch_model-00001-of-00007.safetensors",
        f"{model_root}/diffusion_pytorch_model-00002-of-00007.safetensors",
        # ...
    ]
    module_manager.load_model(
        dit_path_list,
        torch_dtype=torch.bfloat16,
    )
    
    # 加载文本编码器
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )
    
    # 创建并初始化 pipeline
    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    pipe.init(module_manager, pipe_config)
    
    return pipe
```

**关键点：**
- `load_model()` 接受单个路径或路径列表（用于分片模型）
- 模型通过 hash 自动注册，后续可通过名称获取
- 模型权重在 CPU 上加载，在 `pipe.init()` 时移至 GPU

### 模式 2：from_pretrained（推荐用于 HF 格式）

使用 `Pipeline.from_pretrained()` 加载 HuggingFace 模型 ID 或本地 HF 格式文件夹。

```python
def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """使用 from_pretrained 创建 pipeline。
    
    Args:
        parallelism: 并行 GPU 数量（必填）
        model_root: 模型权重路径或 HF 模型 ID（必填）
    """
    model_source = model_root  # HF ID 或本地路径
    
    pipe = Wan21VideoPipeline.from_pretrained(
        model_id_or_path=model_source,
        device="cuda",
        torch_dtype=torch.bfloat16,
        attention_config=AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2),
        enable_clip_stage=False,  # T2V 不需要 CLIP
        enable_parallel=parallelism > 1,
        parallel_devices=list(range(parallelism)) if parallelism > 1 else None,
    )
    
    return pipe
```

**何时使用 from_pretrained：**
- HuggingFace 模型 ID（如 `"Wan-AI/Wan2.1-T2V-1.3B"`）
- 本地 HF Diffusers 格式文件夹
- 快速原型开发和测试
- 服务部署时动态选择模型

## 配置详解

### PPL_CONFIG 字典

集中管理所有默认参数。**必填字段和配置规则：**

```python
PPL_CONFIG = dict(
    # 必填字段
    name="example_identifier",      # 必填：Pipeline 标识符，用于日志和指标
    model_root="/path/to/model",    # 必填：模型文件的基础目录
    
    # 生成参数
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    cfg_scale=5.0,
    seed=42,
    
    # 质量设置
    negative_prompt="...",
    sigma_shift=5.0,
    
    # 输出设置
    target_fps=16,
    
    # 运行时设置
    tiled=False,
    sample_solver="unipc",
    attn_impl=AttnImplType.TORCH_SDPA,
)
```

**配置规则：**

| 规则 | 说明 |
|------|------|
| `name` | **必填**。用于日志、指标和 pipeline 标识。应具有描述性，如 `"wan21_1.3B_t2v_h100"` |
| `model_root` | **必填**。包含所有模型文件的基础目录。可通过 CLI `--model_root` 覆盖 |
| 模型文件路径 | 使用相对于 `model_root` 的文件名，如 `dit_filename`、`vae_filename`。特殊模型可使用绝对路径 |

**特殊模型路径示例：**

```python
PPL_CONFIG = dict(
    name="wan22_14B_i2v_h100",
    model_root="/nvfile/model_zoo/Wan2.2-I2V-A14B",
    # 标准模型位于 model_root 下
    dit_filename="dit_model.safetensors",
    vae_filename="vae.pth",
    # 特殊模型使用绝对路径（如多个 pipeline 共享的模型）
    text_encoder_path="/shared/models/t5_umt5-xxl-enc-bf16.pth",
    # ... 其他参数
)
```

### Pipeline 配置

通过 `PipelineConfig` 配置运行时行为：

```python
pipe_config = Wan21VideoPipelineConfig()

# Attention 实现
pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)

# CPU Offloading
pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD

# 采样 solver
pipe_config.sample_solver = "euler"

# Stage 开关
pipe_config.enable_clip_stage = True  # I2V 模型需要
```

### 并行配置

配置多 GPU 推理：

```python
if parallelism > 1:
    cfg_scale = PPL_CONFIG["cfg_scale"]
    
    if cfg_scale > 1:
        # CFG 并行 + Ulysses SP
        pipe_config.dit_config.parallel_config.cfg_degree = 2
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism // 2
    else:
        # 纯 Ulysses SP
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = parallelism
    
    pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
    pipe_config.enable_denoising_parallel = True
```

**并行策略配置表：**

| GPU 数量 | cfg_scale > 1 | cfg_scale = 1 |
|---------|---------------|---------------|
| 2 GPU | cfg_degree=2, sp=1 | cfg_degree=1, sp=2 |
| 4 GPU | cfg_degree=2, sp=2 | cfg_degree=1, sp=4 |
| 8 GPU | cfg_degree=2, sp=4 | cfg_degree=1, sp=8 |

### Feature Cache 配置

启用缓存以加速推理：

```python
from telefuser.core.config import FeatureCacheConfig

if enable_feature_cache:
    pipe_config.dit_config.feature_cache_config = FeatureCacheConfig(
        enabled=True,
        model_type="Wan2_2-I2V-A14B",
    )
```

### LoRA 配置

添加 LoRA 支持：

```python
from telefuser.core.config import LoraConfig

pipe_config.dit_config.lora_config = LoraConfig(
    lora_path="/path/to/lora_weights.safetensors",
    lora_scale=1.0,
)
```

## 服务兼容性

示例可通过 `telefuser serve` 部署为服务：

```bash
telefuser serve examples/wan_video/wan21_1_3b_text_to_video_hf.py --task t2v
```

### 服务所需函数

服务期望以下函数：

```python
def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """必须返回已初始化的 pipeline。
    
    必填参数：
        - parallelism: 并行 GPU 数量
        - model_root: 模型权重路径
    """
    pass

def run(pipeline, prompt, negative_prompt="", **kwargs):
    """必须返回生成输出。"""
    pass

def run_with_file(pipeline, prompt, negative_prompt, seed, output_path, **kwargs):
    """可选：执行推理并保存到文件。"""
    pass
```

### 环境变量

使用环境变量配置可变路径：

```python
model_root = os.getenv("MODEL_ROOT", "/default/path")
output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
```

## 最佳实践

### 1. 清晰的文档

在文件顶部添加说明用途的 docstring：

```python
"""Wan2.1 14B 文本生成视频 (T2V) 示例。

本示例演示如何使用 Wan2.1 14B 模型进行文本生成视频。

Usage:
    python wan21_14b_text_to_video_h100.py --prompt "一只猫在弹钢琴"
    python wan21_14b_text_to_video_h100.py --gpu_num 2 --prompt "..."
"""
```

### 2. 有意义的默认提示词

提供能展示模型能力的有趣默认提示词：

```python
@click.option(
    "--prompt",
    default="一位时尚女性在东京街头漫步，温暖的阳光...",
    help="正向引导文本提示词",
)
```

### 3. 一致的参数命名

遵循已建立的命名规范：

| 参数 | 描述 |
|------|------|
| `gpu_num` | GPU 数量 |
| `prompt` | 正向提示词 |
| `negative_prompt` | 负面提示词 |
| `resolution` | 480p, 720p 等 |
| `seed` | 随机种子 |
| `model_root` | 模型路径 |
| `aspect_ratio` | 16:9, 4:3, 1:1 |

### 4. 正确的资源清理

在结束时清理资源：

```python
def main(...):
    pipe = get_pipeline(...)
    output = run(pipe, ...)
    save_video(output, ...)
    del pipe  # 释放 GPU 内存
```

### 5. 计时与日志

报告生成时间：

```python
start = time.time()
output = run(pipe, ...)
elapsed_time = time.time() - start
print(f"生成时间: {elapsed_time:.2f} 秒")
```

### 6. 输出命名

使用 `get_example_name()` 保持一致的输出命名：

```python
filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
```

## 完整示例参考

完整实现请参考：

| 示例 | 特性 | 文件 |
|------|------|------|
| 基础 T2V | Hash-based 加载，并行 | `wan21_14b_text_to_video_h100.py` |
| 基础 I2V | 图像输入，CLIP stage | `wan21_14b_image_to_video_h100.py` |
| HF 加载 | from_pretrained，简单配置 | `wan21_1_3b_text_to_video_hf.py` |
| LoRA | LoRA 配置 | `wan21_14b_image_to_video_lora_h100.py` |
| Feature Cache | 缓存加速 | `wan22_14b_image_to_video_h100.py` |
| Distill | 双 DiT（高/低噪声） | `wan22_14b_image_to_video_distill_h100.py` |

## 相关文档

- [添加新模型](./adding_new_model.md) - 模型实现指南
- [添加新 Stage](./adding_new_stage.md) - Stage 实现指南
- [配置](./configuration.md) - 配置详解
- [并行推理](./parallel.md) - 多 GPU 配置
- [CPU 卸载](./offload.md) - 显存优化
- [服务](./service.md) - 服务部署