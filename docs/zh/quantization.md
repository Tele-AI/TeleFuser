# 量化

量化用较低精度存储或计算部分张量。TeleFuser 分别对模型权重、Linear 输入、注意力张量和 KV cache 提供了量化路径；启用其中一项不等于整个 pipeline 都使用低精度。

## 基本原理

`W8A16` 表示 8 bit 权重和 16 bit 激活，`W8A8` 表示权重与激活都是 8 bit。累加器和输出仍可使用 BF16 或 FP32，因此该记号并不描述算子里的所有张量。

TeleFuser 离线转换器的对称 INT8 量化为：

$$
s = \frac{\max |x|}{127}, \qquad
q = \mathrm{clamp}\left(\mathrm{round}\left(\frac{x}{s}\right), -128, 127\right),
\qquad \hat{x} = s q.
$$

一个 scale 可以对应整个 tensor、一个输出通道、一个 token 或一个小 block。分组越细通常误差越小，但 scale 数量和 kernel 复杂度也越高。

| 粒度 | TeleFuser 中的例子 |
| --- | --- |
| Per-tensor | ComfyUI INT8/FP8 转换 |
| Per-output-channel | 离线 INT8/FP8 权重、`LinearFP8` 权重 |
| Per-token | `LinearFP8`、LiveAct、LingBot FP8 激活 |
| Per-block | MXFP 和 NVFP4 转换 kernel |

## 已有路径

| 路径 | 精度 | 入口 |
| --- | --- | --- |
| TorchAO 在线量化 | W8A8 或仅权重 W8A16 | `QuantType.TORCHAO_FP8` |
| bitsandbytes 在线量化 | NF4 仅权重，W4A16 | `QuantType.BNB_NF4` |
| scaled FP8 checkpoint | FP8 权重和动态 FP8 激活，W8A8 | `torch_dtype=torch.float8_e4m3fn` |
| 离线 checkpoint 转换 | INT8、FP8、MXFP4/6/8 或 NVFP4 权重 | `tools/convert/converter.py` |

`QuantType` 中还有尚未接入通用在线加载的格式。枚举值存在，不代表具体模型已经实现该路径。

## 在线 Linear 量化

### TorchAO FP8：取决于版本

TeleFuser 会探测 TorchAO 的动态激活与仅权重 FP8 API，首先找到的 API 决定实际模式：动态激活和权重
FP8 是 W8A8；BF16 输入配合 `Float8WeightOnlyConfig` 是 W8A16。应检查转换日志并运行真实前向，
不能只根据 `QuantType.TORCHAO_FP8` 推断模式。

当前接入 Wan、Qwen-Image 和 LTX 的 transformer blocks。默认过滤器会跳过 `head`、`time_embedding`、`time_projection` 和 `patch_embedding` 等名称。

```python
import torch

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager

quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.TORCHAO_FP8,
    kernel_backend=QuantKernelBackend.TORCHAO,
)
manager = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
manager.load_model(
    dit_paths,
    device="cuda",
    torch_dtype=torch.bfloat16,
    quant_config=quant_config,
)
```

完整的 Qwen-Image 示例：

```bash
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --prompt "A cat playing piano" \
  --output qwen_image_fp8.png
```

FP8 会减少权重显存和读取流量；W8A8 还会量化 Linear 输入。是否降低延迟取决于实际模式、矩阵形状、
GPU、TorchAO 版本以及 `torch.compile`。

TorchAO 必须与 PyTorch 版本兼容。不要直接安装最新版，应先查看
[TorchAO release 兼容表](https://github.com/pytorch/ao/releases)；出现 import warning 或失败，说明该组合尚未通过验证。

### bitsandbytes NF4：W4A16

NF4 使用针对近似正态分布权重设计的非均匀 4 bit 码本。TeleFuser 把选中的 Linear 替换为 `bitsandbytes.nn.Linear4bit`，使用 BF16 计算，并压缩量化统计量。

```python
quant_config = QuantConfig(
    enabled=True,
    quant_type=QuantType.BNB_NF4,
    kernel_backend=QuantKernelBackend.BITSANDBYTES,
)
```

完整示例是 `examples/qwen_image/qwen_image_t2i_telefuser_nf4_h100.py`。NF4 通常比 FP8 更省权重显存，但 4 bit 解码不保证延迟更低。

## Scaled FP8 checkpoint：W8A8

这条路径与 TorchAO 不同。兼容的 checkpoint 已经包含 E4M3FN 权重和逐输出通道 scale。`LinearFP8` 在每次 forward 时把输入逐行量化为 FP8，再通过 `tf_kernel` 或 vLLM/CUTLASS 执行 scaled GEMM，输出恢复为 BF16 或 FP16。

每一行的 scaled FP8 同样使用 absmax：

$$
s = \frac{\max |x|}{\mathrm{max}(\mathrm{E4M3FN})}, \qquad
q = \mathrm{cast}_{\mathrm{E4M3FN}}\left(
\mathrm{clamp}\left(\frac{x}{s}, f_{\min}, f_{\max}\right)\right).
$$

只能加载符合 TeleFuser 权重和 scale 布局的 checkpoint：

```python
manager.load_model(
    fp8_checkpoint,
    device="cuda",
    torch_dtype=torch.float8_e4m3fn,
)
```

只修改 `torch_dtype` 不能把任意 BF16 checkpoint 变成 scaled FP8 checkpoint。应从仓库提供的 Qwen-Image 或 Wan FP8 示例开始。

## 离线转换

转换器量化选中的二维权重，并把 scale 一同写入 checkpoint。它只生成 artifact，不会自动提供推理 kernel。

```bash
python tools/convert/converter.py \
  --source /path/to/source \
  --output /path/to/output \
  --model_type wan_dit \
  --quantized \
  --linear_dtype fp8 \
  --non_linear_dtype torch.bfloat16 \
  --single_file
```

`--linear_dtype` 支持 `int8`、`fp8`、`mxfp4`、`mxfp6`、`mxfp8` 和 `nvfp4`。

- `int8` 和 `fp8` 默认每个输出行使用一个 absmax scale；ComfyUI 模式每个 tensor 使用一个 scale。
- MXFP4/6/8 通过 `lightx2v_kernel` 使用 32 元素 block 和 E8M0 scale。
- NVFP4 每个字节打包两个值，每 16 个值使用一个 E4M3 scale，另有一个全 tensor 的 global scale。
- 未量化 tensor 会转换为 `--non_linear_dtype`。

默认 scale key 是 `<weight>_scale`；NVFP4 还会写入 `<weight>_global_scale`。MXFP 和 NVFP4 转换依赖 CUDA
与 `lightx2v_kernel`。TeleFuser 通用 loader 不执行这些 artifact；消费端必须实现相同的布局和 GEMM。

## 其他量化数据路径

- **LiveAct FP8：** 使用 vLLM 风格的动态 W8A8 GEMM；权重缓存为 FP8，激活逐 token 量化。
- **LingBot-Video MoE FP8：** expert 权重逐输出通道量化，路由后的激活逐行量化，再调用 `torch._scaled_mm`。
- **SageAttention：** TeleFuser 的三个变体都把 Q/K 量化为 INT8；`2_8_16` 使用 FP16 P/V，`2_8_8` 及其 SM90 变体使用 FP8 P/V。这些 kernel 不修改模型权重。
- **LiveAct FP8 KV cache：** K/V 保存为 E4M3FN，每个末维向量使用一个 FP32 scale；加载时反量化为注意力请求的 dtype。

## 验证方法

固定与 BF16 baseline 相同的 prompt、输入、seed、scheduler 和推理步数，并检查：

1. 日志中的 Linear 转换数量不为零。
2. 可选后端通过真实的 `torch.inference_mode()` 前向；仅 import 成功不够。
3. 分开记录加载峰值显存和稳态显存。
4. warmup 后使用 CUDA 同步或 CUDA event 测量延迟。
5. 对比生成图像或视频；进程成功退出不代表精度合格。

算子对比可记录：

$$
E_{\max} = \max |y_q-y|, \qquad
E_{\mathrm{rel}} = \frac{\lVert y_q-y \rVert_2}{\lVert y \rVert_2}.
$$

相关测试包括 `tests/unit/quantize/test_quantized_linear.py` 和 `tests/unit/models/test_lingbot_video_moe.py`。

TorchAO 对 weight-only 与动态 FP8 的定义见其[量化推理文档](https://docs.pytorch.org/ao/stable/workflows/inference.html)。
