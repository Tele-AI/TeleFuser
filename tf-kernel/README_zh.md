# tf-kernel

[English](README.md) | 中文

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)](https://pytorch.org/)

**tf-kernel** 是为 TeleFuser 设计的高性能 CUDA 内核库，为 Transformer 和扩散模型提供优化的 GPU 运算。它使用 CUTLASS 和 FlashInfer 实现自定义 CUDA 内核，并提供 PyTorch Python 绑定。

## 功能特性

- **逐元素运算**：激活函数（SiLU、GELU）、RMS 归一化、旋转位置编码（RoPE）、类型转换
- **GEMM 运算**：FP8、INT8 和 FP4 量化矩阵乘法，支持多种量化方案
- **注意力变体**：
  - SageAttention v2：INT8 QK 量化，FP8/FP16 值
  - SageAttention v3：适用于 Blackwell（SM100+）的 FP4 量化
  - 块稀疏注意力：高效的块稀疏模式注意力
- **多架构支持**：SM80（Ampere）、SM90（Hopper）、SM100+（Blackwell）

## 安装

`tf-kernel` 要求 PyTorch 2.11.0 和 NVIDIA CUDA 环境。独立安装命令：

```bash
python -m pip install --upgrade tf-kernel
```

在 TeleFuser checkout 中，`python -m pip install -e ".[kernel]"` 会从当前配置的包索引安装已发布版本，
不会编译本源码目录。如需联合开发两个项目：

```bash
cd /path/to/TeleFuser
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel
```

兼容性、验证方法、可运行 API 示例和常见问题见
[TeleFuser 完整安装与使用指南](../docs/zh/tf_kernel.md)。

## 源码编译

### 环境要求

- CUDA Toolkit ≥12.8
- CMake ≥3.26
- Python ≥3.10
- PyTorch == 2.11.0
- scikit-build-core
- ninja（可选）

### 开发环境安装

`tf-kernel` 是存放在 TeleFuser 单仓库中的独立版本 Python distribution。如需联合开发，请克隆 TeleFuser
并使用同一个解释器安装两个 editable 项目：

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel

# 等价命令：
/path/to/venv/bin/python -m pip install -e ./tf-kernel -e ".[dev]"
```

可选依赖组：`dev`（全部）、`test`（测试）、`docs`（文档）、`lint`（代码检查）。

如果只开发 `tf-kernel`，不安装 TeleFuser 的开发依赖：

```bash
cd tf-kernel
python -m pip install -e ".[dev]"
```

### 独立发布

包版本在 `pyproject.toml` 中声明，与 TeleFuser 版本相互独立。tf-kernel 不提供 GitHub Actions 自动
CUDA 编译或发布 workflow。每次发布都必须在明确配置好 CUDA/NVCC 的构建机上手动构建、验证和上传：

```bash
make update <version>
make build-sm90 PYTHON=/path/to/venv/bin/python  # 按需选择目标架构。
python -m pip install twine
python -m twine check dist/*.whl
python -m twine upload dist/*.whl
```

产物验证完成后，可以创建匹配的 `tf-kernel-v<version>` tag 作为源码追溯标记；该 tag 不会触发构建或
发布。只有 TeleFuser 需要显式兼容性边界时才约束根目录的 `kernel` extra；默认保持不固定版本。

### 使用 Makefile 编译 tf-kernel

```bash
# 编译所有支持的 SM 架构（默认：ALL）
make build

# 自动检测本地 GPU 架构并编译（单机上使用推荐）
make build-auto

# 仅编译特定 SM 架构
make build-sm80   # Ampere（A100、RTX 3090 等）
make build-sm90   # Hopper（H100）
make build-sm100  # Blackwell（RTX 5090、B100/B200）
```

每个目标都会将 wheel 写入 `dist/`，并安装到 `PYTHON` 指定的解释器。例如，在 H100 上限制资源占用：

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
make build-sm90 \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

### 目标 SM 架构选择

编译系统支持通过 `TF_KERNEL_TARGET_SM` CMake 变量选择目标 SM 架构：

| 选项 | 描述 |
|------|------|
| `ALL` | 编译所有支持的 SM 架构（默认） |
| `AUTO` | 自动检测本地 GPU 架构并编译 |
| `SM80` | 仅编译 SM 80-89（Ampere、Ada Lovelace） |
| `SM90` | 仅编译 SM 90（Hopper H100） |
| `SM100` | 仅编译 SM 100+（Blackwell） |

直接使用 CMake：
```bash
cmake -DTF_KERNEL_TARGET_SM=AUTO ..
cmake -DTF_KERNEL_TARGET_SM=SM80 ..
```

**注意：** 针对特定 SM 架构编译可以显著减少编译时间和二进制文件大小，相比编译所有架构。

### 限制编译资源占用（CPU / 并行度）

默认情况下，`make build` 会使用所有可用的 CPU 核心。你可以覆盖编译并行度和 NVCC 编译线程数：

```bash
# 限制并行作业数（控制 make 和 cmake 的并行度）
make build MAX_JOBS=2

# 额外限制 NVCC 内部线程数（减少 CPU 和峰值内存占用）
make build MAX_JOBS=2 CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

### 验证已安装扩展

```bash
python - <<'PY'
from pathlib import Path
import torch
import tf_kernel

print("tf-kernel:", tf_kernel.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name())
print("extension:", Path(tf_kernel.common_ops.__file__).resolve())

x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
assert torch.isfinite(tf_kernel.rmsnorm(x, weight)).all()
print("RMSNorm smoke test: OK")
PY
```

在 Ampere 或 Hopper 上，导入时提示 FP4 算子不可用属于预期行为；FP4 要求 SM100+。安装后运行
`python -m pip check` 可检查是否有其他包要求不兼容的 PyTorch 版本。
当前验证的 H100 wheel 在架构选择的 SM90 SageAttention 路径存在已知 `misaligned address` 错误；专项
SM90 测试通过前应改用其他 TeleFuser 注意力后端。

## 贡献代码

### 添加新内核的步骤：

1. 在 [csrc](csrc) 目录中实现内核
2. 在 [include/tf_kernel_ops.h](include/tf_kernel_ops.h) 中暴露接口
3. 在 [csrc/common_extension.cc](csrc/common_extension.cc) 中创建 Torch 扩展
4. 更新 [CMakeLists.txt](CMakeLists.txt) 添加新的 CUDA 源文件
5. 在 [python](python/tf_kernel) 中暴露 Python 接口
6. 添加测试和基准测试

### 开发技巧

1. 创建 Torch 扩展时，使用 `m.def` 添加函数定义，使用 `m.impl` 绑定设备：

- 如何编写 schema：[Schema 参考](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // 为了支持 torch.compile，我们需要在这里使用 schema 进行 def
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### 适配 C++ 原生类型以兼容 Torch

第三方 C++ 库通常使用 int 和 float，但 PyTorch 绑定需要 int64_t 和 double，这是由于 Python 的类型映射。

使用 tf_kernel_torch_shim.h 中的 make_pytorch_shim 自动处理类型转换：

```cpp

// 添加 int -> int64_t 的类型转换
template <>
struct pytorch_library_compatible_type<int> {
  using type = int64_t;
  static int convert_from_type(int64_t arg) {
    TORCH_CHECK(arg <= std::numeric_limits<int>::max(), "value too large");
    TORCH_CHECK(arg >= std::numeric_limits<int>::min(), "value too small");
    return arg;
  }
};
```
```cpp
// 包装你的函数
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### 测试与基准测试

1. 在 [tests/](/tests) 中添加 pytest 测试，如果需要跳过某些测试，请使用 `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 需要计算能力 10 或以上。"
)
```

2. 使用 [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) 在 [benchmark/](benchmark) 中添加基准测试

   **我们推荐使用 `triton.testing.do_bench_cudagraph` 进行内核基准测试**：

   相比 `triton.testing.do_bench`，`do_bench_cudagraph` 提供：
   - 减少 CPU 开销影响，获得更准确的内核性能测量
   - 将 PDL（程序化依赖启动）效果纳入单个内核结果
   - 在支持 PDL 的架构（SM >= 90）上提供更真实的性能数据

3. 运行测试套件

## 内核大小分析

分析编译后的 wheel 文件中的 CUDA 内核大小，以识别过大的内核和模板实例化膨胀：

此工具需要 `cubloaty`（使用 `pip install cubloaty` 安装）。

```bash
# 安装 cubloaty
pip install cubloaty

# 分析 wheel 文件
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl

# 自定义输出文件
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl --output my_analysis.txt
```

工具生成：
- 文本报告，包含：
  - 内核分组（按名称前缀）
  - 单个内核大小（按大小排序）

使用此工具可以识别大内核和潜在的模板实例化膨胀。

## 致谢

本项目基于以下优秀开源项目构建：

- **[SGL-Kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** - SGLang 项目的一部分，为 LLM 推理提供高性能 CUDA 内核
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - 量化注意力实现，相比标准注意力机制实现显著加速
- **[Block-Sparse-Attention](https://github.com/Dao-AILab/flash-attention)** - FlashAttention 项目中的块稀疏注意力实现

我们衷心感谢这些项目的作者和贡献者对开源社区的杰出贡献。

## 参与贡献

我们欢迎社区贡献！在提交 issue 或 pull request 之前，请阅读我们的[贡献指南](CONTRIBUTING.md)和[行为准则](CODE_OF_CONDUCT.md)。
