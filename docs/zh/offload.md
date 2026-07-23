# CPU 卸载 (Offloading)

CPU 卸载是一种内存优化技术，通过在推理过程中临时将模型权重从 GPU 移动到 CPU 内存来减少 GPU 显存使用。TeleFuser 提供多种卸载策略来平衡内存使用和推理速度。

## 卸载策略

TeleFuser 支持四种卸载策略，通过 `WeightOffloadType` 配置：

| 策略 | 描述 | 适用场景 |
|----------|-------------|----------|
| `NO_CPU_OFFLOAD` | 不卸载，所有权重保留在 GPU | GPU 显存充足 |
| `MODEL_CPU_OFFLOAD` | 在阶段之间将整个模型卸载到 CPU | 显存中度受限 |
| `SEQUENTIAL_CPU_OFFLOAD` | 前向传播时逐层卸载 | 显存严重受限 |
| `ASYNC_CPU_OFFLOAD` | 带预取的异步逐层卸载 | 速度和内存的最佳平衡 |

## 异步 CPU 卸载

`ASYNC_CPU_OFFLOAD` 是大多数场景推荐的策略。它使用 `AsyncOffloadManager` 来：

- **逐层卸载权重** 从 GPU 到固定（pinned）CPU 内存
- **异步预取即将使用的层** 使用专用 CUDA 流
- **重叠数据传输** 与计算以最小化延迟

### 异步卸载工作原理

```
时间 ──────────────────────────────────────────────►

第0层: [加载]──[计算]────────────────────────────
第1层:      [异步加载]──[计算]─────────────────
第2层:           [异步加载]──[计算]────────────
第3层:                [异步加载]──[计算]───────

数据传输（加载）与计算重叠，隐藏延迟
```

### 关键参数

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `offload_type` | `WeightOffloadType` | `NO_CPU_OFFLOAD` | 卸载策略 |
| `pin_cpu_memory` | bool | `True` | 使用固定内存加速 H2D 传输 |
| `offload_ratio` | float | `1.0` | 要卸载的层比例（1.0 = 所有层） |
| `prefetch_size` | int | `1` | 提前预取的层数 |

### AsyncOffloadManager 延迟 GPU 缓存

`lazy_gpu_cache` 是底层 `AsyncOffloadManager` 的构造参数，不是
`OffloadConfig` 字段。它控制是否在 manager 初始化时预分配 GPU 缓冲区：

- **`lazy_gpu_cache=False`（默认）**：在初始化期间分配 GPU 缓冲区池
- **`lazy_gpu_cache=True`**：在首次使用时分配 GPU 缓冲区池（在初始化期间节省显存）

在以下情况使用 `lazy_gpu_cache=True`：
- 管道初始化期间 GPU 显存极其有限
- 希望将显存分配推迟到推理开始

使用 `allocate_gpu_cache()` 和 `cleanup_gpu_cache()` 进行手动控制：

```python
# 示例：手动 GPU 缓存管理
from telefuser.offload.async_offload import AsyncOffloadManager

# 使用 lazy_gpu_cache=True 初始化
manager = AsyncOffloadManager(layers, lazy_gpu_cache=True)

# 在准备好时手动分配
manager.allocate_gpu_cache()

# 推理完成后，释放缓存以释放显存
manager.cleanup_gpu_cache()
```

## 在管道中使用

### 基础配置

```python
from telefuser.core.config import OffloadConfig, WeightOffloadType
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

# 创建管道配置
pipe_config = Wan21VideoPipelineConfig()

# 为 DiT 启用异步卸载（最消耗内存的组件）
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=1,
)

# 可选：为其他阶段启用卸载
pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
```

### WanVideo 示例

带 CPU 卸载的 Wan2.1 视频生成完整示例：

```python
import torch
from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    OffloadConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)

def get_pipeline(model_root, parallelism=1):
    """使用 CPU 卸载初始化 Wan2.1 管道。"""
    
    # 首先将模型加载到 CPU
    module_manager = ModuleManager(device="cpu")
    module_manager.load_models(
        [f"{model_root}/Wan2.1_VAE.pth"],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [[f"{model_root}/diffusion_pytorch_model.safetensors"]],
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_models(
        [f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth"],
        torch_dtype=torch.bfloat16,
    )
    
    # 创建管道
    pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan21VideoPipelineConfig()
    
    # 配置注意力
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
        AttnImplType.SAGE_ATTN_2_8_8
    )
    
    # 为不同阶段配置卸载
    # DiT：使用异步逐层卸载（最适合大型 Transformer）
    pipe_config.dit_config.offload_config = OffloadConfig(
        offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
        pin_cpu_memory=True,
        offload_ratio=1.0,
        prefetch_size=1,
    )
    
    # VAE：使用模型级卸载（更简单，传输频率更低）
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    
    # 文本编码器：使用模型级卸载
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    
    # 可选：启用分布式推理
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.sp_ulysses_degree = 2
        pipe_config.enable_denoising_parallel = True
    
    # 初始化管道
    pipe.init(module_manager, pipe_config)
    return pipe

# 使用
model_root = "/path/to/Wan2.1-T2V-1.3B"
pipe = get_pipeline(model_root, parallelism=1)

# 生成视频
video = pipe(
    prompt="A cat playing piano",
    num_inference_steps=40,
    num_frames=81,
    height=480,
    width=832,
)
```

### 大模型示例（14B+）

对于 Wan2.1-14B 等大型模型，卸载是必不可少的：

```python
# Wan2.1-14B (720P) 配置
pipe_config = Wan21VideoPipelineConfig()

# 使用异步卸载，更大的预取数以获得更好的重叠效果
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    prefetch_size=2,  # 提前预取 2 层
    offload_ratio=1.0,
)

# 为所有辅助模型启用卸载
pipe_config.clip_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
```

## 性能考虑

### 内存与速度权衡

| 策略 | 显存节省 | 速度影响 | 推荐场景 |
|----------|--------------|--------------|----------------|
| `NO_CPU_OFFLOAD` | 无 | 最快 | 24GB+ 显存 |
| `MODEL_CPU_OFFLOAD` | 高（~50%）| 中等 | 16-24GB 显存 |
| `ASYNC_CPU_OFFLOAD` | 高（~60-70%）| 低 | 8-16GB 显存 |
| `SEQUENTIAL_CPU_OFFLOAD` | 最大 | 最慢 | <8GB 显存 |

### 调整预取大小

`prefetch_size` 参数影响数据传输与计算之间的重叠：

- **`prefetch_size=1`**：默认，适合大多数模型
- **`prefetch_size=2+`**：更大层的更好重叠，但更多显存使用

```python
# 对于非常大的层（例如 14B 模型）
pipe_config.dit_config.offload_config.prefetch_size = 2
```

### Qwen-Image H100 实测证据

以下数据用于验证 `AsyncOffloadManager` 的预取和计算重叠路径，并不是对所有模型、GPU 或
CPU-GPU 互联的通用性能承诺。测试使用单张 NVIDIA H100 80GB，PyTorch 2.11.0+cu130；
每个配置先运行一次完整 warmup，再运行一次完整生成。DiT 时间在 stage 入口和出口显式同步
CUDA 后测量，因此包含该阶段的异步权重传输、计算和卸载。

#### FP8 Lightning

Qwen-Image-2512-Lightning FP8 权重，1328×1328，16 steps，CFG=1，`offload_ratio=0.5`：

| 模式 / `prefetch_size` | DiT 时间 | 峰值已分配显存 |
|---|---:|---:|
| `NO_CPU_OFFLOAD` | 4.360 s | 40.55 GiB |
| `ASYNC_CPU_OFFLOAD`, 1 | 5.184 s | 31.61 GiB |
| `ASYNC_CPU_OFFLOAD`, 2 | 4.966 s | 32.24 GiB |
| `ASYNC_CPU_OFFLOAD`, 4 | 4.570 s | 33.50 GiB |
| `ASYNC_CPU_OFFLOAD`, 8 | 4.469 s | 36.04 GiB |
| `ASYNC_CPU_OFFLOAD`, 12 | 4.465 s | 38.57 GiB |

`prefetch_size=8` 比无卸载只慢 2.5%，同时少用 4.51 GiB 显存；增至 12 已基本没有速度
收益。作为重叠对照，在同一 FP8、`prefetch_size=2` 配置中，强制计算流在每次预取后等待
copy stream 的 DiT 时间为 6.039 s，而正常异步路径为 4.966 s。这证明 H2D 传输与计算在正常
路径中存在有效重叠。

#### BF16 基础 Qwen-Image

Qwen-Image-2512 BF16 权重，1664×928，50 steps，CFG=4：

| `offload_ratio` | `prefetch_size` | DiT 时间 | 峰值已分配显存 |
|---:|---:|---:|---:|
| 无卸载 | — | 26.990 s | 55.84 GiB |
| 0.5 | 2 | 46.253 s | 39.00 GiB |
| 0.5 | 4 | 42.470 s | 44.81 GiB |
| 0.5 | 8 | 39.954 s | 49.88 GiB |
| 0.6 | 8 | 47.473 s | 46.08 GiB |
| 0.75 | 8 | 56.335 s | 40.38 GiB |

该模型中，增加预取深度能够加速：50% offload 下从 2 提升到 8 个预取 block，DiT 时间缩短
13.6%，但增加 10.88 GiB 峰值显存。提高 `offload_ratio` 会减少常驻层和显存，但也会增加
每个 denoising step 的 H2D 数据量；即使保留 8 个预取 block，60% 和 75% offload 都比
50%/2-block 基线更慢。因此应先在固定 `offload_ratio` 下调大 `prefetch_size`，再根据显存
预算选择折中点。


### 固定内存

设置 `pin_cpu_memory=True`（默认）使用页锁定内存以实现更快的 H2D 传输：

- **启用**：传输更快，CPU 内存使用略高
- **禁用**：传输更慢，CPU 内存使用更少

## 故障排除

### 初始化期间内存不足

如果管道初始化期间发生 GPU OOM：

```python
# 使用 OffloadConfig 当前暴露的最保守管线卸载配置
pipe_config.dit_config.offload_config = OffloadConfig(
    offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
    pin_cpu_memory=True,
    offload_ratio=1.0,
    prefetch_size=1,
)
```

如果明确需要延迟 GPU cache 分配，需要直接使用 `AsyncOffloadManager`，
或通过会传递 `lazy_gpu_cache=True` 的 pipeline 集成接入；当前
`OffloadConfig` dataclass 不暴露这个选项。

### 推理速度慢

如果卸载导致显著减速：

1. **增加预取大小** 以获得更好的重叠
2. **降低 offload_ratio** 以保留更多层常驻
3. **检查 CPU-GPU 互联**（PCIe 带宽很重要）

```python
# 保留 20% 的层常驻在 GPU 中
pipe_config.dit_config.offload_config.offload_ratio = 0.8
```

### CPU 内存问题

如果 CPU 内存不足：

```python
# 禁用固定内存
pipe_config.dit_config.offload_config.pin_cpu_memory = False
```

## API 参考

### OffloadConfig

```python
@dataclass
class OffloadConfig:
    offload_type: WeightOffloadType = WeightOffloadType.NO_CPU_OFFLOAD
    pin_cpu_memory: bool = True
    offload_ratio: float = 1.0
    prefetch_size: int = 1
```

### AsyncOffloadManager

```python
class AsyncOffloadManager:
    def __init__(
        self,
        layers: torch.nn.ModuleList,
        device: torch.device | None = None,
        *,
        enabled: bool = True,
        pin_cpu_memory: bool = True,
        offload_ratio: float = 1,
        prefetch_size: int = 1,
        lazy_gpu_cache: bool = False,
    ) -> None
    
    def allocate_gpu_cache(self) -> None:
        """手动分配 GPU 缓存。"""
        
    def cleanup_gpu_cache(self) -> None:
        """释放 GPU 缓存。"""
        
    def disable_offload(self) -> None:
        """禁用卸载并加载所有层。"""
        
    def enable_offload(self) -> None:
        """重新启用卸载。"""
```

## 顺序 CPU 卸载 (Sequential CPU Offload)

对于需要细粒度显存管理的场景，TeleFuser 提供了 `enable_sequential_cpu_offload` —— 一种逐层卸载机制，为单个模块包装智能状态管理。

### 三态系统

每个被包装的模块在三种状态之一中运行：

| 状态 | 值 | 数据位置 | 描述 |
|------|-----|----------|------|
| **卸载 (Offload)** | `0` | `offload_device` (通常是 CPU) | 默认状态，最小显存占用 |
| **加载 (Onload)** | `1` | `onload_device` (通常是 GPU) | 已加载但可能使用不同数据类型 |
| **保持 (Keep)** | `2` | `computation_device` (GPU) | 固定在 GPU 中供重复使用 |

### 状态转换流程

```
┌─────────────────────────────────────────────────────────────────┐
│                       前向传播流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  如果 state == 2 (Keep):                                        │
│      → 直接使用权重（最快）                                      │
│                                                                  │
│  否则如果 onload 配置 == 计算配置:                              │
│      → 直接使用权重（无需转换）                                  │
│                                                                  │
│  否则如果设置了 vram_limit 且 GPU 有空闲显存:                   │
│      → 调用 keep() 提升到 state 2                               │
│      → 直接使用权重                                              │
│                                                                  │
│  否则:                                                          │
│      → cast_to() 临时复制到 GPU                                  │
│      → 计算完成后释放（状态不变）                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 使用方法

```python
from telefuser.offload import enable_sequential_cpu_offload, AutoWrappedLinear

# 定义要包装的模块
module_map = {
    torch.nn.Linear: AutoWrappedLinear,
}

# 为每个状态配置数据类型和设备
module_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# 启用顺序卸载
enable_sequential_cpu_offload(
    model,
    module_map=module_map,
    module_config=module_config,
    vram_limit=20.0,  # GB - 当显存充足时提升到 Keep 状态
)
```

### 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | - | 要启用卸载的模型 |
| `module_map` | `dict` | - | 源模块类型到包装类的映射 |
| `module_config` | `dict` | - | 每个状态的数据类型/设备配置 |
| `max_num_param` | `int/None` | `None` | 使用溢出配置的参数阈值 |
| `overflow_module_config` | `dict/None` | `None` | 超过阈值层的替代配置 |
| `vram_limit` | `float/None` | `None` | 自动状态提升的显存限制 (GB) |

### 模块配置

`module_config` 字典控制数据放置：

```python
module_config = {
    # 卸载状态 (state=0) - 最小显存占用
    "offload_dtype": torch.float32,    # CPU 存储使用 FP32
    "offload_device": "cpu",            # 保留在 CPU 上
    
    # 加载状态 (state=1) - 准备使用
    "onload_dtype": torch.bfloat16,     # GPU 使用较低精度
    "onload_device": "cuda",            # 加载到 GPU
    
    # 计算状态 (state=2) - 实际计算
    "computation_dtype": torch.bfloat16,  # 必须与 onload 匹配才能提升
    "computation_device": "cuda",         # 必须是 GPU
}
```

### 可用包装器

| 包装器 | 源模块 | 描述 |
|---------|---------------|-------------|
| `AutoWrappedModule` | `nn.Module` | 任意模块的通用包装器 |
| `AutoWrappedLinear` | `nn.Linear` | 优化的 Linear 层，支持 LoRA |
| `WanAutoCastLayerNorm` | `nn.LayerNorm` | 支持自动混合精度的 LayerNorm |

### 分层配置

为不同参数大小的层使用不同配置：

```python
# 大多数层的标准配置
base_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.bfloat16,
    "onload_device": "cuda",
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

# 大层的配置（始终保留在 CPU）
overflow_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.float32,
    "onload_device": "cpu",  # 从不加载到 GPU
    "computation_dtype": torch.bfloat16,
    "computation_device": "cuda",
}

enable_sequential_cpu_offload(
    model,
    module_map={nn.Linear: AutoWrappedLinear},
    module_config=base_config,
    max_num_param=1_000_000_000,  # 10 亿参数阈值
    overflow_module_config=overflow_config,
    vram_limit=22.0,
)
```

### 手动状态控制

启用后，您可以手动控制模块状态：

```python
# 手动状态转换
for module in model.modules():
    if hasattr(module, 'offload'):
        module.offload()   # 强制切换到 state 0 (CPU)
        module.onload()    # 强制切换到 state 1 (加载设备)
        module.keep()      # 强制切换到 state 2 (GPU)

# 检查当前状态
if hasattr(module, 'state'):
    print(module.state)  # 0=卸载, 1=加载, 2=保持
```

### vram_limit 行为

`vram_limit` 参数控制自动状态提升：

| 设置 | 行为 |
|---------|----------|
| `None` (默认) | 保守模式 - 从不提升到 Keep 状态，始终使用临时转换 |
| `20.0` | 当显存使用 < 20GB 时，将常用模块提升到 Keep 状态 |

**建议**：生产环境始终设置 `vram_limit` 以提高性能。

### API 参考

```python
def enable_sequential_cpu_offload(
    model: torch.nn.Module,
    module_map: dict,
    module_config: dict,
    max_num_param: int | None = None,
    overflow_module_config: dict | None = None,
    vram_limit: float | None = None,
) -> None

class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype,
        offload_device,
        onload_dtype,
        onload_device,
        computation_dtype,
        computation_device,
        vram_limit,
        name: str = "",
    )
    
    def offload(self) -> None:   # 切换到 state 0
    def onload(self) -> None:    # 切换到 state 1
    def keep(self) -> None:      # 切换到 state 2
```

## 参考

- 异步卸载实现改编自 [SGLang](https://github.com/sgl-project/sglang) 的逐层卸载工具。
- 顺序 CPU 卸载实现改编自 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)。
