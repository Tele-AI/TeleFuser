# tf-kernel

English | [中文](README_zh.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-orange)](https://pytorch.org/)

**tf-kernel** is a high-performance CUDA kernel library for TeleFuser, providing optimized GPU operations for transformer and diffusion models. It implements custom CUDA kernels using CUTLASS and FlashInfer, with PyTorch bindings for Python accessibility.

## Features

- **Elementwise Operations**: Activation functions (SiLU, GELU), RMS normalization, rotary positional embedding (RoPE), casting
- **GEMM Operations**: FP8, INT8, and FP4 quantized matrix multiplication with various quantization schemes
- **Attention Variants**:
  - SageAttention v2: INT8 QK quantization with FP8/FP16 value
  - SageAttention v3: FP4 quantization for Blackwell (SM100+)
  - Block Sparse Attention: Efficient block-sparse pattern attention
- **Multi-Architecture Support**: SM80 (Ampere), SM90 (Hopper), SM100+ (Blackwell)

## Installation

`tf-kernel` requires PyTorch 2.11.0 and an NVIDIA CUDA environment. Install the independent package with:

```bash
python -m pip install --upgrade tf-kernel
```

From a TeleFuser checkout, `python -m pip install -e ".[kernel]"` installs the published package from the configured
index. It does not compile this source directory. To develop both projects together, use:

```bash
cd /path/to/TeleFuser
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel
```

See the [full TeleFuser installation and usage guide](../docs/en/tf_kernel.md) for compatibility, verification,
runnable API examples, and troubleshooting.

## Building from Source

### Requirements

- CUDA Toolkit ≥12.8
- CMake ≥3.26
- Python ≥3.10
- PyTorch == 2.11.0
- scikit-build-core
- ninja (optional)

### Development Installation

`tf-kernel` is an independently versioned Python distribution stored in the TeleFuser monorepo. For joint
development, clone TeleFuser and install both editable projects with the same interpreter:

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel

# Equivalent command:
/path/to/venv/bin/python -m pip install -e ./tf-kernel -e ".[dev]"
```

Dependency groups available: `dev` (all), `test`, `docs`, `lint`.

To work on `tf-kernel` without installing TeleFuser's development dependencies:

```bash
cd tf-kernel
python -m pip install -e ".[dev]"
```

### Independent Releases

The package version is declared in `pyproject.toml` and is independent of the TeleFuser version. Update it with
`make update <version>`, then create a matching `tf-kernel-v<version>` tag. The root-level release workflow builds
the CUDA 12.8 wheel and publishes it separately from the `telefuser` distribution. Configure PyPI trusted publishing
for the `tf-kernel-pypi` GitHub environment before the first release; after publishing, update the root `kernel` extra
only if TeleFuser needs an explicit compatibility bound. The root extra is intentionally unpinned by default.

### Use Makefile to build tf-kernel

```bash
# Build for all supported SM architectures (default: ALL)
make build

# Build for auto-detected GPU architecture (recommended for single-machine use)
make build-auto

# Build for specific SM architecture only
make build-sm80   # Ampere (A100, RTX 3090, etc.)
make build-sm90   # Hopper (H100)
make build-sm100  # Blackwell (RTX 5090, B100/B200)
```

Each target writes a wheel to `dist/` and installs it into the interpreter selected by `PYTHON`. For example, a
resource-bounded H100 build is:

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
make build-sm90 \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

### Target SM Architecture Selection

The build system supports selecting target SM architectures via the `TF_KERNEL_TARGET_SM` CMake variable:

| Option | Description |
|--------|-------------|
| `ALL` | Build for all supported SM architectures (default) |
| `AUTO` | Auto-detect local GPU and build for its architecture |
| `SM80` | Build for SM 80-89 (Ampere, Ada Lovelace) |
| `SM90` | Build for SM 90 (Hopper H100) |
| `SM100` | Build for SM 100+ (Blackwell) |

Using CMake directly:
```bash
cmake -DTF_KERNEL_TARGET_SM=AUTO ..
cmake -DTF_KERNEL_TARGET_SM=SM80 ..
```

**Note:** Building for a specific SM architecture reduces build time and binary size significantly compared to building for all architectures.

### Limit build resource usage (CPU / parallelism)

By default, `make build` uses all available CPU cores. You can override build parallelism and NVCC compile threads:

```bash
# Limit parallel jobs (controls both make and cmake parallelism)
make build MAX_JOBS=2

# Additionally limit NVCC internal threads (reduces CPU and peak memory)
make build MAX_JOBS=2 CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

### Verify the installed extension

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

On Ampere or Hopper, the import message that FP4 operators are unavailable is expected; FP4 requires SM100+.
Run `python -m pip check` after installation to detect packages that require an incompatible PyTorch version.
The currently validated H100 wheel has a known `misaligned address` failure in the architecture-selected SM90
SageAttention path; use another TeleFuser attention backend until the focused SM90 test passes.

## Contribution

### Steps to add a new kernel:

1. Implement the kernel in [csrc](csrc)
2. Expose the interface in [include/tf_kernel_ops.h](include/tf_kernel_ops.h)
3. Create torch extension in [csrc/common_extension.cc](csrc/common_extension.cc)
4. Update [CMakeLists.txt](CMakeLists.txt) to include new CUDA source
5. Expose Python interface in [python](python/tf_kernel)
6. Add test and benchmark

### Development Tips

1. When creating torch extensions, add the function definition with `m.def`, and device binding with `m.impl`:

- How to write schema: [Schema reference](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)

   ```cpp
   // We need def with schema here for torch.compile
   m.def(
    "bmm_fp8(Tensor A, Tensor B, Tensor! D, Tensor A_scale, Tensor B_scale, Tensor workspace_buffer, "
    "int cublas_handle) -> ()");
   m.impl("bmm_fp8", torch::kCUDA, &bmm_fp8);
   ```

### Adapting C++ Native Types for Torch Compatibility

Third-party C++ libraries often use int and float, but PyTorch bindings require int64_t and double due to Python's type mapping.

Use make_pytorch_shim from tf_kernel_torch_shim.h to handle conversions automatically:

```cpp

// Add type conversion for int -> int64_t
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
// Wrap your function
m.impl("fwd", torch::kCUDA, make_pytorch_shim(&mha_fwd));
```

### Testing & Benchmarking

1. Add pytest tests in [tests/](/tests), if you need to skip some test, please use `@pytest.mark.skipif`

```python
@pytest.mark.skipif(
    skip_condition, reason="Nvfp4 Requires compute capability of 10 or above."
)
```

2. Add benchmarks using [triton benchmark](https://triton-lang.org/main/python-api/generated/triton.testing.Benchmark.html) in [benchmark/](benchmark)

   **We recommend using `triton.testing.do_bench_cudagraph` for kernel benchmarking**:

   Compared to `triton.testing.do_bench`, `do_bench_cudagraph` provides:
   - Reduced CPU overhead impact for more accurate kernel performance measurements
   - Incorporation of PDL (Programmatic Dependent Launch) effects into individual kernel results
   - More realistic performance data on PDL-supported architectures (SM >= 90)

3. Run test suite

## Kernel Size Analysis

Analyze CUDA kernel sizes in compiled wheel files to identify oversized kernels and template-instantiation bloat:

This tool requires `cubloaty` (install with `pip install cubloaty`) to work.

```bash
# Install cubloaty
pip install cubloaty

# Analyze a wheel file
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl

# Custom output file
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl --output my_analysis.txt
```

The tool generates:
- A text report with:
  - Kernel groups (by name prefix)
  - Individual kernel sizes (sorted by size)

Use this to identify large kernels and potential template instantiation bloat.

## Acknowledgments

This project is built upon the excellent work of the following open-source projects:

- **[SGL-Kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel)** - Part of the SGLang project, providing high-performance CUDA kernels for LLM serving
- **[SageAttention](https://github.com/thu-ml/SageAttention)** - Quantized attention implementation achieving significant speedups over standard attention mechanisms
- **[Block-Sparse-Attention](https://github.com/Dao-AILab/flash-attention)** - Block sparse attention implementation from the FlashAttention project

We sincerely thank the authors and contributors of these projects for their outstanding contributions to the open-source community.

## Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting issues or pull requests.
