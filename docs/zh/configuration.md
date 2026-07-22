# TeleFuser 配置体系

本文档介绍 TeleFuser 的三层配置架构设计。

## 概述

TeleFuser 采用三层配置架构，实现了模型定义、推理算法参数和用户可调参数的关注点分离：

```
┌─────────────────────────────────────────────────────────────────────┐
│  第一层: 模型定义层 (Model-Weight Bound)                             │
│  模型加载后固定，与权重文件绑定                                        │
├─────────────────────────────────────────────────────────────────────┤
│  第二层: 推理算法参数层 (PipelineConfig)                             │
│  PipeConfig 配置 + Pipeline.__call__ 接口固定设置                    │
├─────────────────────────────────────────────────────────────────────┤
│  第三层: 用户可调参数层 (Example run())                              │
│  Example 文件 run() 函数对外暴露                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 第一层：模型定义层

**位置**：模型加载阶段，与权重文件绑定

### 固有属性

由模型权重决定的属性：

| 属性 | 说明 | 决定方式 |
|------|------|----------|
| `distill` | 是否为蒸馏模型 | 权重文件 |
| `MoE` | MoE 架构 | 加载多个 DiT 模型 |
| `fp8/bf16` | 量化类型 | 权重文件格式 |
| `meanflow` | FlowMatch 类型 | 模型架构 |

### 示例

```python
# examples/wan_video/wan22_14b_image_to_video_distill_h100.py
module_manager = ModuleManager(device="cpu")

# 加载蒸馏模型（固定属性：distill=True）
module_manager.load_model(
    f"{model_root}/dit_high_noise_distill_model_bf16_1022_ecab7.safetensors",
    torch_dtype=torch.bfloat16,
)
module_manager.load_model(
    f"{model_root}/dit_low_noise_distill_model_bf16_1022_200c2.safetensors",
    torch_dtype=torch.bfloat16,
)
```

**关键点**：模型固有属性与权重绑定，加载后不可更改。

## 第二层：推理算法参数层

**位置**：`PipelineConfig` dataclass + `Pipeline.__call__()` 方法

### PipelineConfig 定义

```python
@dataclass
class Wan21VideoPipelineConfig:
    """Configuration for Wan2.1 video generation pipeline."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    clip_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"              # 采样器类型
    enable_clip_stage: bool = False
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vfi: bool = False                  # 视频帧插值
```

### Pipeline.__call__ 固定参数

```python
def __call__(
    self,
    prompt: str | List[str],
    ...
    sigma_shift: float = 5.0,         # 噪声调度参数
    boundary: float = 0.875,          # MoE 切换边界
    tiled: bool = False,              # 分块推理
    tile_size: tuple[int, int] = (30, 52),
    ...
)
```

### Example 中的配置

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

**关键点**：第二层参数由开发者在 example 文件中定义，不暴露给最终用户。

## 第三层：用户可调参数层

**位置**：example 文件中的 `run()` 函数

### 示例接口

```python
def run(
    pipeline,
    prompt,                    # 用户输入
    negative_prompt="",        # 用户输入
    seed=42,                   # 可调参数
    resolution="480p",         # 可调参数
    aspect_ratio="16:9",       # 可调参数
):
    """从文本提示生成视频。

    Args:
        pipeline: 预加载的 pipeline 对象
        prompt: 正向引导文本提示
        negative_prompt: 负向引导提示
        seed: 随机种子
        resolution: 分辨率，如 720p、480p
        aspect_ratio: 宽高比，如 16:9
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

**关键点**：第三层参数暴露给最终用户，每次推理可修改。

## 配置流动图

```
模型权重 (第一层)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  模型固有属性: distill/MoE, 量化类型等                            │
│  (固定，由权重文件决定)                                           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼ module_manager.load_model()
┌─────────────────────────────────────────────────────────────────┐
│  get_pipeline()                                                  │
│  ├─ PipeConfig: sample_solver, parallel, offload (第二层)        │
│  └─ pipe.init(module_manager, pipe_config)                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  run() 用户接口 (第三层)                                          │
│  ├─ 用户输入: prompt, seed, resolution, aspect_ratio             │
│  └─ PPL_CONFIG 固定值: num_inference_steps, cfg_scale            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
     pipeline(prompt, seed, height, width, ...)
```

## 不同模型类型的配置差异

| 模型类型 | 第一层 (模型) | 第二层 (算法) | 第三层 (用户) |
|----------|---------------|---------------|---------------|
| **Wan21 1.3B** | 单 DiT | `sample_solver="euler"` | prompt, seed, resolution |
| **Wan22 A14B** | MoE (high+low DiT) | `boundary=0.875`, `cfg_scale_high/low` | 同上 |
| **Wan22 Distill** | distill 权重 | `cfg_scale=1.0` (无需 CFG) | 同上 |
| **HunyuanVideo + SR** | base DiT + SR DiT | `enable_sr=True`, `lq_noise_strength` | 同上 |

## 设计原则

1. **第一层不可变**：模型加载后固定，与权重绑定
2. **第二层半固定**：在 example 中通过 `PPL_CONFIG` 定义，开发者控制
3. **第三层可变**：暴露给最终用户，每次推理可修改

这种分层设计实现了**关注点分离**：
- 模型研究者关注第一层
- 算法工程师关注第二层
- 应用用户关注第三层

## 核心配置类

位于 `telefuser/core/config.py`：

### ModelRuntimeConfig

模型执行的最高层配置：

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

分布式并行处理配置：

```python
@dataclass
class ParallelConfig:
    """Distributed parallel processing configuration."""

    device_ids: list | None = None
    dp_degree: int = 1           # 数据并行
    cfg_degree: int = 1          # CFG 并行
    sp_ulysses_degree: int = 1   # Ulysses 序列并行
    sp_ring_degree: int = 1      # Ring Attention 序列并行
    pp_degree: int = 1           # 流水线并行
    tp_degree: int = 1           # 张量并行
    enable_fsdp: bool = False
    worker_intra_op_threads: int = 1  # 每个 worker 的 CPU intra-op 线程数

    def validate(self) -> None:
        """验证设备数量与并行度匹配。"""
        ...
```

### AttentionConfig

注意力实现配置：

```python
@dataclass
class AttentionConfig:
    """Unified configuration for all attention implementations."""

    attn_impl: AttnImplType = AttnImplType.TORCH_SDPA
    sparse_config: SparseAttentionConfig | None = None

    @classmethod
    def radial_attention(cls, ...) -> AttentionConfig:
        """创建 radial attention 配置（视频生成稀疏注意力）。"""
        ...

    @classmethod
    def dense_attention(cls, attn_impl: AttnImplType = AttnImplType.FLASH_ATTN_2) -> AttentionConfig:
        """创建稠密注意力配置。"""
        ...
```

## 配置导出

TeleFuser 提供 `dump_config()` 方法用于导出 pipeline 配置，支持复现和调试。

### 使用场景

- **复现性**：捕获生成任务使用的精确配置
- **调试**：检查实际生效的配置
- **部署**：在不同环境间共享配置

### 使用方法

```python
# pipeline 初始化后
pipe = get_pipeline(parallelism=1, model_root="...")

# 导出到文件
pipe.dump_config("output/pipeline_config.json")

# 或直接获取字典
config = pipe.dump_config()
print(config["layer1_model_definition"]["models"])
```

### 输出格式

输出的 JSON 包含两个主要层级：

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

### 实现细节

- **第一层**：在 `init()` 时捕获模型路径和类名
- **第二层**：递归序列化 PipelineConfig dataclass
- **内存高效**：只存储模型信息（名称、路径、类名），不存储模型权重

## 相关文档

- [模型加载指南](./model_loading.md)
- [并行配置指南](./parallel.md)
- [注意力配置指南](./attention.md)
- [Offload 配置指南](./offload.md)
