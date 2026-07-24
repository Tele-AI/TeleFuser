# tf-kernel 安装与使用

`tf-kernel` 是 TeleFuser 的可选 CUDA 扩展包，提供融合逐元素算子、量化 GEMM、SageAttention 和块稀疏
注意力内核。它位于本仓库的 `tf-kernel/` 目录，但拥有独立的包元数据、版本、wheel、CI 和发布 tag。

TeleFuser 不安装 `tf-kernel` 也可以运行：对于已经实现回退的算子，`telefuser.ops` 层会保留 PyTorch 原生
或 Triton 路径。当 Pipeline 需要 `tf-kernel` 提供的优化 CUDA 路径时再安装它。

!!! important "重要"

    TeleFuser 的模型代码仍然必须从 `telefuser.ops` 导入算子，不应直接依赖 `tf_kernel`。下文的直接导入
    仅用于独立使用、安装诊断和内核开发。

## 兼容性

| 组件 | 要求或目标 |
|------|------------|
| Python | 3.10 或更高版本 |
| PyTorch | `2.11.0`（`2.11.0+cu128` 这样的 CUDA local version 也满足要求） |
| CUDA Toolkit | 从源码编译要求 12.8 或更高版本 |
| CMake | 从源码编译要求 3.26 或更高版本 |
| GPU 目标 | SM80、SM90 和 SM100 |

具体可用内核取决于编译目标。FP4 内核要求 Blackwell（SM100 或更高）；在 Ampere 或 Hopper 上看到
`no fp4 operator available` 属于预期行为。目前核心算子已在 Python 3.11、PyTorch 2.11.0+cu128、
CUDA 12.8 和 H100（SM90a）组合上验证。其他目标和算子族用于生产前仍应在目标 GPU 上验证。

!!! warning "当前 H100 SageAttention 限制"

    当前验证的 H100 build 中，架构选择接口 `tf_kernel.sageattn()` 会进入 SM90 专用 FP8 内核，并可能报
    `CUDA error: misaligned address`。RMSNorm、融合激活、FP8 量化和通用 FP8 SageAttention 路径已通过
    smoke test。在部署 wheel 的专项 GPU 测试通过前，不应在生产环境启用 SM90 专用 SageAttention 后端。

## 选择安装方式

### 随 TeleFuser 安装已发布版本

普通包安装：

```bash
python -m pip install "telefuser[kernel]"
```

在 TeleFuser checkout 中安装：

```bash
python -m pip install -e ".[kernel]"
```

`kernel` extra 有意不固定 `tf-kernel` 版本，它会从当前配置的包索引解析最新兼容版本；PyTorch 依赖和
CUDA ABI 要求由 `tf-kernel` 自身管理。该命令**不会**编译仓库中的同级 `tf-kernel/` 源码目录。

### 联合开发 TeleFuser 和 tf-kernel

在仓库根目录使用同一个 Python 解释器安装两个 editable 项目：

```bash
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel
```

等价命令：

```bash
/path/to/venv/bin/python -m pip install -e ./tf-kernel -e ".[dev]"
```

editable 安装会调用 CUDA 编译后端。如果更关注编译时间、二进制体积或可复现性，建议使用下文的
指定架构 wheel 编译方式。

### 独立安装 tf-kernel

不安装 TeleFuser 也可以单独安装和升级 CUDA 扩展：

```bash
python -m pip install --upgrade tf-kernel
```

该包的版本与 TeleFuser 版本相互独立；`tf-kernel-v<version>` 源码 tag 会生成独立发布产物。

## 从源码编译

克隆 TeleFuser 单仓库，选择已经安装 PyTorch 2.11.0 的解释器，然后进入内核项目：

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser/tf-kernel
```

本地工作站可以自动检测当前 GPU：

```bash
make build-auto PYTHON=/path/to/venv/bin/python
```

需要可复现的指定架构编译时：

```bash
make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere 和 Ada
make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper，包括 H100
make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell
```

H100 上限制主机资源占用的完整示例：

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
make build-sm90 \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

`make build` 会编译所有支持的目标。每个编译目标都会把 wheel 写入 `dist/`，添加包含 Torch/CUDA ABI
信息的合法 wheel build tag，并将其安装到 `PYTHON` 指定的解释器。首次编译需要联网获取固定版本的
CUTLASS、FlashInfer 和其他 CMake 依赖。

## 验证安装

使用实际启动 TeleFuser 的同一个解释器运行：

```bash
python - <<'PY'
from pathlib import Path

import torch
import tf_kernel

print("tf-kernel:", tf_kernel.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("extension:", Path(tf_kernel.common_ops.__file__).resolve())

x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
y = tf_kernel.rmsnorm(x, weight)
assert y.shape == x.shape and torch.isfinite(y).all()
print("RMSNorm smoke test: OK")
PY
```

H100 专用 wheel 的 common extension 应从 `sm90` 包目录加载。还应运行 `python -m pip check` 检查环境中的
依赖冲突。

## 使用示例

TeleFuser 用户应调用公共 ops 层；它会在支持的 eager CUDA 路径选择 `tf-kernel`，同时保留框架回退：

```python
import torch

from telefuser.ops.activations import silu_and_mul

x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
y = silu_and_mul(x)  # 最后一维拆分成两个宽度为 1024 的张量。
assert y.shape == (4, 1024)
```

独立用户可以直接调用内核包：

```python
import torch
import tf_kernel

# RMSNorm
x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
y = tf_kernel.rmsnorm(x, weight, eps=1e-6)

# 已在 H100 验证的通用 FP8 SageAttention 路径。
# HND 布局：[batch, heads, sequence, head_dim]
q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
attn_output = tf_kernel.sageattn_qk_int8_pv_fp8_cuda(
    q,
    k,
    v,
    tensor_layout="HND",
    is_causal=False,
    pv_accum_dtype="fp32",
)
```

per-token FP8 量化会写入调用者预先分配的输出张量：

```python
x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
x_scale = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
tf_kernel.tf_per_token_quant_fp8(x, x_q, x_scale)
```

更多底层接口契约请参考 `tf-kernel/docs/` 下的 API 文档和 `tf-kernel/tests/` 中的测试。

## 常见问题

### 使用了错误的 Python 环境

始终使用 `python -m pip`，并向 Make 传入 `PYTHON=/path/to/venv/bin/python`。可通过
`python -m pip show tf-kernel` 和 `python -c "import sys; print(sys.executable)"` 确认路径。

### PyTorch 被替换或依赖解析失败

`tf-kernel` 要求 PyTorch 2.11.0，因为已编译扩展与 PyTorch/CUDA ABI 绑定。如果其他包固定了不兼容的
PyTorch 版本，请在干净环境中安装 TeleFuser 和 `tf-kernel`。除非重新编译并针对替代版本完成验证，
否则不要绕过该约束。

### CMake 找不到 CUDA 或使用了错误 Toolkit

检查 `nvcc --version`，将 `CUDA_HOME` 指向 CUDA 12.8+ Toolkit，并在 `PATH` 中把 `$CUDA_HOME/bin` 放在
旧版 Toolkit 之前。PyTorch CUDA runtime 与所选 Toolkit 应保持 ABI 兼容。

### 扩展针对错误的 GPU 架构编译

在目标机器使用 `make build-auto` 重新编译，或显式选择 `build-sm80`、`build-sm90`、`build-sm100`。
指定架构 wheel 无法提供编译时未包含的内核。

### H100 上 SageAttention 报 `misaligned address`

当前架构选择接口会把 H100 路由到 SM90 专用 FP8 实现。如果该路径失败，应将其视为当前 wheel 不支持的
后端并选择其他 TeleFuser 注意力实现；异步 CUDA 错误发生后不要继续复用该进程。上文的通用 FP8 函数
可用于独立验证，但生产启用仍需要完成精度对齐和目标工作负载 benchmark。

### 编译耗尽 CPU 或内存

降低 `MAX_JOBS` 和 `CMAKE_BUILD_PARALLEL_LEVEL`，并设置
`CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"`。只编译一个 SM 架构也能显著降低编译时间和产物体积。
