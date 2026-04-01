# Profiler 性能分析系统

TeleFuser 提供了基于 PyTorch Profiler 的性能分析系统，支持分布式环境下的性能调试和内存追踪。

## 功能特性

- **上下文管理器和装饰器** - 灵活的使用方式
- **同步和异步支持** - 同时支持同步和异步函数
- **分布式感知** - 自动处理多 rank 性能分析
- **内存追踪** - 监控峰值内存分配
- **PyTorch Profiler 集成** - 可选的详细 kernel 级别分析
- **Chrome trace 导出** - 在 Chrome DevTools 中可视化分析结果
- **环境变量控制** - 无需修改代码即可启用/禁用

## 快速开始

### 基本用法

```python
from telefuser.utils.profiler import ProfilingContext

# 作为上下文管理器
with ProfilingContext("my_operation"):
    # 你的代码
    result = model(input_data)

# 作为装饰器
@ProfilingContext("my_function")
def process_data(data):
    return model(data)

# 作为异步装饰器
@ProfilingContext("async_operation")
async def process_async(data):
    return await model(data)
```

### 启用 PyTorch Profiler

设置环境变量启用详细性能分析：

```bash
# 启用特定名称的 profiler
export ENABLE_PROFILER_NAMES="vae_decode,text_encoding,dit_denoising"

# 设置 trace 文件输出目录
export PROFILER_OUTPUT_DIR="./profiler_output"

# 运行你的应用
python your_script.py
```

## 环境变量

| 变量 | 描述 | 默认值 |
|------|------|--------|
| `ENABLE_PROFILER_NAMES` | 逗号分隔的 profiler 名称列表 | ""（空） |
| `PROFILER_OUTPUT_DIR` | Chrome trace 文件输出目录 | "./profiler_output" |
| `TELEFUSER_PROFILE_DEBUG` | 启用所有调试 profiling 上下文 | "false" |

### 程序化控制 Profiler

```python
from telefuser.utils.profiler import (
    enable_profiler_for_names,
    set_profiler_output_dir,
    get_enabled_profiler_names,
)

# 编程方式启用 profilers
enable_profiler_for_names("vae_decode,text_encoding")

# 设置输出目录
set_profiler_output_dir("/path/to/traces")

# 检查已启用的名称
names = get_enabled_profiler_names()  # 返回: {"vae_decode", "text_encoding"}
```

## ProfilingContext 与 ProfilingContext4Debug

### ProfilingContext

始终激活的性能分析上下文：

```python
from telefuser.utils.profiler import ProfilingContext

@ProfilingContext("operation_name")
def process():
    # 总是记录执行时间和峰值内存
    pass
```

### ProfilingContext4Debug

根据 `TELEFUSER_PROFILE_DEBUG` 条件激活：

```python
from telefuser.utils.profiler import ProfilingContext4Debug

@ProfilingContext4Debug("debug_operation")
def process():
    # 仅当 TELEFUSER_PROFILE_DEBUG=true 时进行性能分析
    # 否则无任何开销
    pass
```

**推荐在 Stage 中使用：**

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.utils.profiler import ProfilingContext4Debug

class MyStage(BaseStage):
    @with_model_offload(["model"])
    @ProfilingContext4Debug("my_stage_process")
    @torch.inference_mode()
    def process(self, input_data):
        # 仅在调试模式下进行性能分析
        return self.model(input_data)
```

## 输出结果

### 控制台日志

使用 `ProfilingContext` 时，会记录以下信息：

```
[Profile] my_operation cost 0.123456 seconds
Rank 0 - Function 'my_operation' Peak Memory: 4.50 GB
```

当 PyTorch Profiler 启用时：

```
Rank 0 - Starting PyTorch profiler for 'my_operation'
Rank 0 - PyTorch profiler trace saved to: ./profiler_output/my_operation_rank0_run1.json
Rank 0 - Profiler summary for 'my_operation': Total operations: 150
Rank 0 -   1. aten::addmm: CPU=12.34 ms, CUDA=8.56 ms
Rank 0 -   2. aten::copy_: CPU=5.23 ms, CUDA=3.21 ms
...
```

### Chrome Trace 文件

当 PyTorch Profiler 启用时，会导出 Chrome trace 文件：

```
profiler_output/
├── vae_decode_rank0_run1.json
├── vae_decode_rank0_run2.json
├── text_encoding_rank0_run1.json
└── dit_denoising_rank0_run1.json
```

可视化方法：

1. 打开 Chrome DevTools (`chrome://tracing`)
2. 点击 "Load" 并选择 JSON trace 文件
3. 分析 kernel 时序、内存操作和 CPU/GPU 时间线

## 参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `name` | str | 必填 | Profiler 名称，用于标识 |
| `reset_peak_memory` | bool | True | 性能分析前重置峰值内存统计 |

```python
# 自定义内存追踪行为
with ProfilingContext("operation", reset_peak_memory=False):
    # 不重置峰值内存 - 捕获累积峰值
    pass
```

## 分布式支持

Profiler 自动处理分布式环境：

```python
# 在分布式环境中（如 2 个 GPU）
with ProfilingContext("distributed_op"):
    # Rank 0 日志: "Rank 0 - Function 'distributed_op' Peak Memory: 4.50 GB"
    # Rank 1 日志: "Rank 1 - Function 'distributed_op' Peak Memory: 4.50 GB"
    pass
```

Trace 文件包含 rank 信息：

```
profiler_output/
├── operation_rank0_run1.json
├── operation_rank1_run1.json
```

## 硬件平台支持

Profiler 支持多种硬件平台：

| 平台 | Profiler Activity |
|------|-------------------|
| CUDA (NVIDIA) | `torch.profiler.ProfilerActivity.CUDA` |
| XPU (Intel) | `torch.profiler.ProfilerActivity.XPU` |
| NPU (华为) | `torch.profiler.ProfilerActivity.PrivateUse1` |
| CPU | `torch.profiler.ProfilerActivity.CPU` (始终启用) |

## Stage 中的集成使用

### 典型使用模式

```python
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.utils.profiler import ProfilingContext4Debug
import torch

class VAEDecodeStage(BaseStage):
    def __init__(self, name, module_manager, model_runtime_config):
        super().__init__(name, model_runtime_config)
        self.vae = module_manager.fetch_module("vae")
        self.model_names = ["vae"]

    @with_model_offload(["vae"])
    @ProfilingContext4Debug("vae_decode")
    @torch.inference_mode()
    def process(self, latents):
        with torch.autocast(device_type=self.device_type, dtype=self.torch_dtype):
            return self.vae.decode(latents)
```

### 分析多个操作

```python
class TextEncodingStage(BaseStage):
    @with_model_offload(["text_encoder"])
    @ProfilingContext4Debug("text_encoding")
    @torch.inference_mode()
    def encode_text(self, prompts):
        # 整体编码被分析
        with ProfilingContext4Debug("tokenization"):
            tokens = self.tokenizer(prompts)
        with ProfilingContext4Debug("embedding"):
            embeddings = self.text_encoder(tokens)
        return embeddings
```

## 最佳实践

### 1. 使用有意义的名称

```python
# 推荐 - 描述性强且唯一
@ProfilingContext4Debug("vae_decode_video")
@ProfilingContext4Debug("dit_denoising_step_0")

# 避免 - 通用或重复
@ProfilingContext4Debug("process")
@ProfilingContext4Debug("model")
```

### 2. 在 Stage 中使用 ProfilingContext4Debug

```python
# 推荐 - 生产环境无开销
@ProfilingContext4Debug("stage_name")
def process(self, data):
    pass

# 避免在生产代码中使用 - 总是激活
@ProfilingContext("stage_name")
def process(self, data):
    pass
```

### 3. 与其他装饰器组合使用

装饰器顺序很重要 - profiler 应包裹实际计算：

```python
@with_model_offload(["model"])      # 外层: 处理模型加载
@ProfilingContext4Debug("process")  # 中层: 分析计算
@torch.inference_mode()             # 内层: 禁用梯度
def process(self, data):
    return self.model(data)
```

### 4. 精确启用

```bash
# 仅启用需要的
export ENABLE_PROFILER_NAMES="vae_decode"

# 避免全部启用（trace 文件过大）
export ENABLE_PROFILER_NAMES="*"  # 不推荐
```

### 5. 合理使用 reset_peak_memory

```python
# 每个独立操作时重置
with ProfilingContext("independent_op", reset_peak_memory=True):
    pass

# 追踪累积内存时不重置
with ProfilingContext("sequence_op", reset_peak_memory=False):
    pass
```

## 故障排除

### Trace 文件过大

如果 trace 文件太大：

1. 仅启用特定的 profiler 名称
2. 减少性能分析持续时间
3. 分析更少的操作

```bash
export ENABLE_PROFILER_NAMES="dit_denoising"  # 仅一个操作
```

### GPU Activity 缺失

如果 GPU 活动未被记录：

1. 验证平台支持（CUDA、XPU、NPU）
2. 检查 CUDA 同步是否正常工作

```python
from telefuser.platforms import current_platform
print(current_platform.device_type)  # 应为 "cuda"、"xpu" 或 "npu"
```

### 内存统计不准确

确保性能分析前进行同步：

```python
# Profiler 自动同步，但自定义计时需手动同步
from telefuser.platforms import current_platform
current_platform.synchronize()
with ProfilingContext("operation"):
    pass
```

## 相关文档

- [添加新 Stage](./adding_new_stage.md) - Stage 开发中的 profiler 集成
- [Metrics 指标系统](./metrics.md) - 生产环境监控和可观测性
- [Logging 日志系统](./logging.md) - 日志配置和使用
- [Configuration 配置](./configuration.md) - 运行时配置选项