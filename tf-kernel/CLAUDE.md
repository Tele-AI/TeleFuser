# TF-Kernel - AI Coding Agent Guide

Essential information for AI coding agents working on `tf-kernel` - a high-performance CUDA kernel library for TeleFuser with PyTorch bindings.

## Tech Stack

- **Python**: >= 3.10 | **PyTorch**: == 2.11.0 | **CUDA**: 12.8+ to 13.0+ | **CMake**: >= 3.26
- **Dependencies**: CUTLASS, FlashInfer, cuBLAS, cuBLASLt, libtorch

## Project Structure

```
tf-kernel/
├── csrc/                      # CUDA kernels
│   ├── elementwise/           # Activation, norm, rope, cast, copy, topk
│   ├── gemm/                  # INT8/FP8/FP4 GEMM, quantization
│   ├── sageattn2/             # SageAttention v2 (INT8 QK)
│   ├── sageattn3/             # SageAttention v3 (FP4 Blackwell)
│   ├── block_sparse_attn/     # Block-sparse attention
│   ├── expert_specialization/ # Expert specialization kernels
│   ├── cutlass_extensions/    # CUTLASS extensions
│   ├── local_flashinfer/      # Local FlashInfer integration
│   └── memory/                # Memory operations
├── tf_kernel/                 # Python package
│   ├── elementwise.py         # Elementwise ops wrappers
│   ├── gemm.py                # GEMM ops wrappers
│   ├── sageattn2.py           # SageAttention v2 interface
│   ├── sageattn3.py           # SageAttention v3 interface
│   ├── block_sparse_attn.py   # Block sparse attn interface
│   ├── memory.py              # Memory ops wrappers
│   ├── sampling.py            # Sampling ops wrappers
│   ├── scalar_type.py         # Scalar type definitions
│   ├── load_utils.py          # Arch-specific library loading
│   ├── triton/                # Triton kernels
│   ├── testing/               # Testing utilities
│   └── sm80/sm90/sm100/       # Architecture-specific libs
├── include/                   # C++ headers
│   ├── tf_kernel_ops.h        # Operator declarations
│   ├── tf_kernel_torch_shim.h # PyTorch type shims
│   ├── scalar_type.hpp        # Scalar type definitions
│   └── utils.h                # Utility headers
├── tests/                     # pytest test suite
├── benchmark/                 # Benchmark scripts
├── cmake/                     # CMake utilities
└── 3rd/                       # Third-party code (CUTLASS, FlashInfer)
```

## Build Commands

```bash
make build          # Build all architectures (default)
make build-auto     # Auto-detect GPU architecture
make build-sm80     # Ampere/Ada (SM 80-89)
make build-sm90     # Hopper (SM 90)
make build-sm100    # Blackwell (SM 100+)
make test           # Run tests
make format         # Format code
```

CMake options: `TF_KERNEL_TARGET_SM` (ALL/AUTO/SM80/SM90/SM100), `TF_KERNEL_ENABLE_BF16`, `TF_KERNEL_ENABLE_FP8`, `TF_KERNEL_ENABLE_FP4`

## Packaging and Releases

- `tf-kernel` is a separate Python distribution inside the TeleFuser monorepo. Its version is declared explicitly in
  `pyproject.toml` and must not be derived from TeleFuser release tags.
- Do not add GitHub Actions workflows that compile or publish `tf-kernel`. Use `make update <version>`, build and
  validate the wheel on an explicitly provisioned CUDA/NVCC host, and upload it manually.
- `pip install -e ".[kernel]"` at the repository root selects the released wheel. Joint source development installs
  both editable projects with `scripts/install_dev.sh --kernel`.
- Never trigger a nested pip install from the TeleFuser or tf-kernel build backend. Keep local source paths out of
  published dependency metadata.
- Keep the exact PyTorch requirement, Docker build version, wheel build tag, and tested ABI synchronized.

## Key Kernels

| Category | Location | Key Operations |
|----------|----------|----------------|
| Elementwise | `csrc/elementwise/` | silu_and_mul, gelu_and_mul, rmsnorm, fused_add_rmsnorm, apply_rope |
| GEMM | `csrc/gemm/` | int8_scaled_mm, fp8_scaled_mm, fp8_blockwise_scaled_mm, bmm_fp8, per_token_quant_fp8 |
| FP4 (SM100+) | `csrc/gemm/` | cutlass_scaled_fp4_mm, scaled_fp4_quant |
| Attention | `csrc/sageattn2/`, `csrc/sageattn3/` | SageAttention v2 (INT8 QK), v3 (FP4 Blackwell) |
| Block Sparse | `csrc/block_sparse_attn/` | Block-sparse attention patterns |

## Adding a New Kernel

1. Implement in `csrc/<category>/your_kernel.cu`
2. Declare in `include/tf_kernel_ops.h`
3. Register in `csrc/common_extension.cc`: `m.def("kernel(...)"); m.impl("kernel", torch::kCUDA, &kernel);`
4. Add to CMakeLists.txt `SOURCES`
5. Create Python wrapper in `tf_kernel/<category>.py`
6. Export in `tf_kernel/__init__.py`
7. Add tests in `tests/test_your_kernel.py`

For type conversion between C++ native types and PyTorch types, use `make_pytorch_shim` from `tf_kernel_torch_shim.h`.

## Code Style

- **C++/CUDA**: clang-format (Google style, 2-space indent, 120 column limit)
- **Python**: black, isort, ruff (config in `.pre-commit-config.yaml`)

```bash
make format  # Format all code
```

## Architecture Loading

Python auto-loads the correct library based on GPU:
- SM 90 → `sm90/common_ops.so`
- SM 100+ → `sm100/common_ops.so`  
- SM 80-89 → `sm80/common_ops.so`

## Common Issues

| Issue | Solution |
|-------|----------|
| Build OOM | `make build MAX_JOBS=2 CMAKE_ARGS="-DTF_KERNEL_COMPILE_THREADS=1"` |
| CUDA 12.6 segfault | Update ptxas to 12.8 |
| CUDA runtime not found | Set `CUDA_HOME` or `CUDA_PATH` |
| FP4 symbols in SM80/SM90 | Ensure FP4 sources only in `SM_100_SOURCES` |
| Symbol signature mismatch | Match `int64_t`/`double` between header and implementation |

## Wheel Verification

```bash
make test-wheel  # Check wheel symbols
python analyze_whl_kernel_sizes.py path/to/tf_kernel-*.whl  # Analyze kernel sizes
```

## AI Agent Rules

1. **Documentation Consistency**: Update `CLAUDE.md` when adding modules, changing structure, or modifying APIs
2. **Response Format**: Start all responses with "Developer" prefix

## References

- [CUTLASS](https://github.com/NVIDIA/cutlass) | [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [PyTorch C++ Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [PyTorch Operator Schema](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func)
