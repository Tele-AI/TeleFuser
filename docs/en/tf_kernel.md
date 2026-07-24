# tf-kernel installation and usage

`tf-kernel` is TeleFuser's optional CUDA extension package. It provides fused elementwise operations, quantized
GEMM, SageAttention, and block-sparse attention kernels. The package lives in the `tf-kernel/` directory of this
repository, but it has its own package metadata, version, wheel, CI, and release tag.

TeleFuser can run without `tf-kernel`: the `telefuser.ops` layer keeps native PyTorch or Triton fallbacks where they
are implemented. Install `tf-kernel` when a pipeline uses one of its optimized CUDA paths.

!!! important

    TeleFuser model code must continue importing operations from `telefuser.ops`, not directly from `tf_kernel`.
    Direct imports below are intended for standalone kernel use, diagnostics, and kernel development.

## Compatibility

| Component | Requirement or target |
|-----------|-----------------------|
| Python | 3.10 or newer |
| PyTorch | `2.11.0` (a CUDA local version such as `2.11.0+cu128` satisfies this requirement) |
| CUDA Toolkit | 12.8 or newer for source builds |
| CMake | 3.26 or newer for source builds |
| GPU targets | SM80, SM90, and SM100 |

Kernel availability depends on the selected build target. FP4 kernels require Blackwell (SM100 or newer); seeing
`no fp4 operator available` on Ampere or Hopper is expected. Core operations are currently validated with Python
3.11, PyTorch 2.11.0+cu128, CUDA 12.8, and H100 (SM90a). Other targets and operation families should be validated on
their target GPU before production use.

!!! warning "Current H100 SageAttention limitation"

    In the currently validated H100 build, the architecture-selected `tf_kernel.sageattn()` path chooses the
    SM90-specific FP8 kernel and can fail with `CUDA error: misaligned address`. RMSNorm, fused activations, FP8
    quantization, and the generic FP8 SageAttention path pass smoke tests. Do not enable the SM90-specific SageAttention
    backend in production until its focused GPU test passes on the wheel being deployed.

## Choose an installation path

### Install the released package with TeleFuser

For a normal package installation:

```bash
python -m pip install "telefuser[kernel]"
```

From a TeleFuser checkout:

```bash
python -m pip install -e ".[kernel]"
```

The `kernel` extra intentionally has no `tf-kernel` version pin. It resolves the latest compatible `tf-kernel` from
the configured package index; `tf-kernel` itself owns its PyTorch dependency and CUDA ABI requirements. This command
does **not** compile the sibling `tf-kernel/` source directory.

### Develop TeleFuser and tf-kernel together

From the repository root, install both editable projects with the same interpreter:

```bash
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel
```

Equivalent command:

```bash
/path/to/venv/bin/python -m pip install -e ./tf-kernel -e ".[dev]"
```

Editable installation invokes the CUDA build backend. Use an architecture-specific wheel build instead when build
time, binary size, or reproducibility matters.

### Install tf-kernel independently

The CUDA extension can be installed and upgraded without installing TeleFuser:

```bash
python -m pip install --upgrade tf-kernel
```

The package version is independent of the TeleFuser version. A source tag named `tf-kernel-v<version>` produces the
standalone release artifact.

## Build from source

Clone the TeleFuser monorepo, select the interpreter that already contains PyTorch 2.11.0, and enter the kernel
project:

```bash
git clone https://github.com/Tele-AI/TeleFuser.git
cd TeleFuser/tf-kernel
```

For a local workstation, auto-detect the installed GPU:

```bash
make build-auto PYTHON=/path/to/venv/bin/python
```

For a reproducible target-specific build:

```bash
make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere and Ada
make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper, including H100
make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell
```

An H100 build with bounded host resource use can be run as follows:

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH \
CUDA_HOME=/usr/local/cuda-12.8 \
make build-sm90 \
  PYTHON=/path/to/venv/bin/python \
  MAX_JOBS=2 \
  CMAKE_BUILD_PARALLEL_LEVEL=2 \
  CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"
```

`make build` builds all supported targets. Every build target writes a wheel to `dist/`, adds a legal wheel build
tag containing the Torch/CUDA ABI, and installs that wheel into the interpreter selected by `PYTHON`. The initial
build needs network access to obtain pinned CUTLASS, FlashInfer, and other CMake dependencies.

## Verify the installation

Run the check with the same interpreter that will start TeleFuser:

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

An H100-specific wheel should load its common extension from an `sm90` package directory. Also run
`python -m pip check` to expose dependency conflicts in the environment.

## Usage

TeleFuser users should call the public ops layer; it selects `tf-kernel` for supported eager CUDA paths and keeps the
framework fallback behavior:

```python
import torch

from telefuser.ops.activations import silu_and_mul

x = torch.randn(4, 2048, device="cuda", dtype=torch.float16)
y = silu_and_mul(x)  # The last dimension is split into two 1024-wide tensors.
assert y.shape == (4, 1024)
```

Standalone users can call the kernel package directly:

```python
import torch
import tf_kernel

# RMSNorm
x = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
weight = torch.ones(1024, device="cuda", dtype=torch.float16)
y = tf_kernel.rmsnorm(x, weight, eps=1e-6)

# H100-tested generic FP8 SageAttention path.
# HND layout: [batch, heads, sequence, head_dim]
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

Per-token FP8 quantization writes into caller-provided output tensors:

```python
x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
x_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
x_scale = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
tf_kernel.tf_per_token_quant_fp8(x, x_q, x_scale)
```

See the API reference under `tf-kernel/docs/` and the tests in `tf-kernel/tests/` for lower-level contracts.

## Troubleshooting

### The wrong Python environment is used

Always use `python -m pip` and pass `PYTHON=/path/to/venv/bin/python` to Make. Confirm both paths with
`python -m pip show tf-kernel` and `python -c "import sys; print(sys.executable)"`.

### PyTorch is replaced or dependency resolution fails

`tf-kernel` requires PyTorch 2.11.0 because compiled extensions are tied to the PyTorch/CUDA ABI. Install TeleFuser
and `tf-kernel` into a clean environment if another package pins an incompatible PyTorch version. Do not bypass the
constraint unless you rebuild and validate the extension against the replacement version.

### CMake cannot find CUDA or uses the wrong toolkit

Check `nvcc --version`, set `CUDA_HOME` to the CUDA 12.8+ toolkit, and put `$CUDA_HOME/bin` before older toolkits in
`PATH`. The PyTorch CUDA runtime and the selected toolkit should be ABI-compatible.

### The extension was built for the wrong GPU

Rebuild with `make build-auto` on the target machine or use the explicit `build-sm80`, `build-sm90`, or
`build-sm100` target. Architecture-specific wheels cannot provide kernels that were omitted at build time.

### SageAttention fails with `misaligned address` on H100

The architecture selector currently routes H100 to the SM90-specific FP8 implementation. Treat a failure in that
path as an unsupported backend for the current wheel; select another TeleFuser attention implementation instead of
continuing in the CUDA process after the asynchronous error. The generic FP8 function shown above is useful for
isolated validation, but production enablement still requires parity and workload benchmarks.

### The build exhausts CPU or memory

Lower `MAX_JOBS` and `CMAKE_BUILD_PARALLEL_LEVEL`, and set
`CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"`. Targeting one SM architecture also substantially reduces build time and
artifact size.
