# Hardware Platforms

TeleFuser abstracts execution hardware behind the platform layer in `telefuser/platforms/`. Each process resolves a
single `current_platform` object at import time, and the [Ops](ops.md) dispatch layer uses it to select operator
implementations. Pipelines and examples configure a `torch.device`-style device and do not branch on the vendor.

## Platform selection

`_resolve_current_platform()` in `telefuser/platforms/__init__.py` probes the environment in a fixed order and
instantiates the first match:

1. **ROCm** — a PyTorch `+rocm` (HIP) build with at least one visible AMD GPU
2. **CUDA** — a CUDA build with at least one visible NVIDIA GPU
3. **NPU** — a `torch_npu` installation with a visible Ascend device
4. **CPU** — the fallback when no accelerator is detected

HIP builds expose the `torch.cuda` API, so the ROCm platform reuses it (`device_type` stays `cuda`); check
`torch.version.hip` to distinguish a ROCm host from an NVIDIA one. A HIP or CUDA build without a visible GPU falls
back to CPU. `CUDA_VISIBLE_DEVICES` controls device visibility on both GPU platforms, and
`ASCEND_RT_VISIBLE_DEVICES` on NPU.

## Platform matrix

| Platform | `device_type` | Distributed backend | Operator dispatch | `tf-kernel` | torch.compile |
|----------|---------------|---------------------|-------------------|-------------|---------------|
| CUDA (NVIDIA) | `cuda` | NCCL | Optimized `forward_cuda` paths | Supported | Experimental ([details](torch_compile_compatibility.md)) |
| ROCm (AMD) | `cuda` | RCCL (`nccl`) | Triton `forward_cuda` paths and native fallbacks | Not available | Not validated |
| NPU (Ascend) | `npu` | HCCL | Native fallbacks | Not available | Not validated |
| CPU | `cpu` | Gloo | Native fallbacks | Not available | Native path |

## Attention backends by platform

Attention backend availability is resolved at import time (see [Attention](attention.md)). A backend whose
dependencies are missing falls back to `TORCH_SDPA` with a one-time warning.

- **CUDA**: all dense backends — `TORCH_SDPA`, FlashAttention 2/3/4, SageAttention through `tf-kernel` or
  `sageattention`, and cuDNN — subject to GPU architecture; sparse backends likewise.
- **ROCm**: `TORCH_SDPA` only. `flash_attn` has no official ROCm wheels for consumer RDNA GPUs, and `tf-kernel`,
  SageAttention, and SpargeAttn are CUDA-only. The AOTriton-backed SDPA path is the fast attention kernel on
  RDNA4.
- **NPU / CPU**: `TORCH_SDPA` through the native fallback paths; no vendor attention kernels are integrated.

## CUDA

CUDA is the primary validated path: Python 3.10–3.13, PyTorch 2.6 or newer, CUDA toolkit 12.8 or newer, with H100
as the validated target for optimized kernels. Optional `tf-kernel` provides fused elementwise operations,
quantized GEMM, SageAttention, and block-sparse attention — see [tf-kernel](tf_kernel.md) for build and artifact
compatibility. Multi-GPU inference uses NCCL; see [Parallel Inference](parallel.md).

## ROCm

ROCm support targets AMD GPUs with ROCm 7.x and a PyTorch `+rocm` build; see [Installation](installation.md) for the
setup path.

- Attention uses `TORCH_SDPA`; no `tf-kernel`, `flash_attn`, or `sageattention` installation is required.
- The ops layer selects `forward_rocm` where a kernel defines one and otherwise reuses the CUDA Triton path (Triton
  supports ROCm), falling back to native PyTorch.
- `tf-kernel` imports are gated to `CudaPlatform`, so no CUDA-only extension is loaded on ROCm hosts.
- Multi-GPU inference uses RCCL, AMD's NCCL-compatible collectives library. PyTorch's ROCm build exposes it through
  the `nccl` backend string, so the platform layer requires no special configuration.
- `torch.compile` is not validated on ROCm; ROCm examples run eager.
- Validated entry points are the `*_rocm.py` examples, for example
  [Wan2.1 1.3B text-to-video](https://github.com/Tele-AI/TeleFuser/tree/main/examples/wan_video) on a Radeon RX 9070
  (gfx1201, ROCm 7.2). Multi-GPU branches reuse the `_h100.py` parallel configuration but are not yet validated.

## NPU and CPU

The NPU platform targets Huawei Ascend devices through `torch_npu` with the HCCL distributed backend.

- Attention uses `TORCH_SDPA` through the native fallback paths; no `tf-kernel`, `flash_attn`, `sageattention`, or
  `triton` installation is required.
- The ops layer selects `forward_npu` where an op defines one and otherwise falls back to native PyTorch. No
  NPU-optimized kernels are integrated yet, so pipelines run entirely on the native paths.
- Multi-card inference uses HCCL through the `hccl` backend string. Parallel-worker queues marshal tensors through
  CPU because `torch_npu` has no reliable cross-process device IPC, and each spawned worker group receives a
  distinct `HCCL_IF_BASE_PORT` so concurrent groups do not collide on HCCL's data-plane socket range.
- `torch.compile` is not validated on NPU; NPU examples run eager.
- The validated entry point is
  [Wan2.2 TI2V-5B text-to-video](https://github.com/Tele-AI/TeleFuser/tree/main/examples/wan_video)
  (`wan22_t2v_5b.py`), which auto-detects the platform and runs unmodified on an Atlas 910B (CANN 8.2, torch 2.9
  with a matching `torch_npu`): single-card, and four-card CFG × Ulysses parallelism over HCCL. Wan2.2 A14B shares
  these code paths but has not been exercised on NPU hardware; validate other examples on your target NPU before
  production use.

The CPU platform is the fallback when no accelerator is detected; it is intended for tests and for pipelines that
explicitly request CPU execution.
