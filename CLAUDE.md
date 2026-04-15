# TeleFuser - Agent Guidelines

## Project Overview

TeleFuser is a high-performance framework for efficient multimodal generation model inference (image/video generation, video super-resolution). 

**Tech Stack:** Python 3.10-3.13, PyTorch 2.6+, CUDA 12.8+, FastAPI, Ray

**Supported Models:** WanVideo, Qwen-Image, Z-Image, FlashVSR, RealESRGAN, RIFT-HDV3

## Commands

```bash
pip install -e ".[dev]"           # Development installation
pre-commit run --all-files        # Linting checks
pytest tests/                     # Run tests
bash scripts/run_ci_tests.sh      # Full CI suite
telefuser serve /path/to/pipeline.py --port 8000  # Start API server
```

## Architecture

```
telefuser/
├── core/             # Base abstractions: BasePipeline, BaseStage, configs
├── pipelines/        # Model-specific pipelines
├── models/           # Model architectures: DiT, VAE, text encoders
├── ops/              # Custom operations: attention, FFN, normalization
├── kernel/           # Triton kernels: RMSNorm, rotary, quant, fused ops
├── platforms/        # Hardware abstraction: CUDA, NPU, CPU
├── distributed/      # FSDP, TP, PP, SP, Ring/Ulysses attention
│   ├── ulysses_comm.py   # Ulysses All-to-All: ulysses_scatter_heads, ulysses_gather_heads
│   ├── ring.py            # Ring P2P communication for long sequences
│   └── pp_comm.py         # Pipeline parallelism
├── schedulers/       # Diffusion schedulers
├── feature_cache/    # Feature caching: AdaTaylorCache
├── service/          # FastAPI service
└── client/           # Python SDK
```

### Layer Architecture Principles

TeleFuser follows a strict layered architecture for operations:

```
┌─────────────────────────────────────────────────────────────┐
│                      models/                                 │
│  (DiT, VAE, text encoders - ONLY import from ops/)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       ops/                                   │
│  (Compile-aware dispatch: native for compile, kernel for    │
│   eager mode. Base classes: CustomOp, CustomOpFunction)     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   kernel/triton/                             │
│  (Pure Triton kernels, custom ops. NOT directly used by     │
│   models. May have torch.library.custom_op registration.)   │
└─────────────────────────────────────────────────────────────┘
```

**Key Rules:**

1. **models/** layer MUST only import from `telefuser.ops/`
   - ✅ `from telefuser.ops.normalization import RMSNorm, LayerNorm, modulate`
   - ✅ `from telefuser.ops.rotary import apply_rotary_emb`
   - ❌ `from telefuser.kernel.triton import apply_rotary_embedding`

2. **ops/** layer handles compile-aware dispatch:
   - `torch.compiler.is_compiling()` → PyTorch native implementation
   - Eager mode + CUDA → Optimized Triton kernel
   - Other platforms → PyTorch native fallback

3. **kernel/triton/** contains pure Triton code:
   - No `torch.compiler.is_compiling()` checks
   - May use `torch.library.custom_op` for torch.compile compatibility
   - Only used by ops/ layer, never directly by models/

## Code Style

- PEP8 with ruff (line length: 120)
- Comments and docstrings **must be in English**
- All public function parameters **must have type annotations** (return types optional)
- Use Python 3.10+ syntax: `str | None`, `list[int]`

## Documentation Links

| Topic | English | Chinese |
|-------|---------|---------|
| Adding New Example | [docs/en/adding_new_example.md](docs/en/adding_new_example.md) | [docs/zh/adding_new_example.md](docs/zh/adding_new_example.md) |
| Adding New Model | [docs/en/adding_new_model.md](docs/en/adding_new_model.md) | [docs/zh/adding_new_model.md](docs/zh/adding_new_model.md) |
| Adding New Stage | [docs/en/adding_new_stage.md](docs/en/adding_new_stage.md) | [docs/zh/adding_new_stage.md](docs/zh/adding_new_stage.md) |
| Attention | [docs/en/attention.md](docs/en/attention.md) | [docs/zh/attention.md](docs/zh/attention.md) |
| Configuration | [docs/en/configuration.md](docs/en/configuration.md) | [docs/zh/configuration.md](docs/zh/configuration.md) |
| Feature Cache | [docs/en/feature_cache.md](docs/en/feature_cache.md) | [docs/zh/feature_cache.md](docs/zh/feature_cache.md) |
| Logging | [docs/en/logging.md](docs/en/logging.md) | [docs/zh/logging.md](docs/zh/logging.md) |
| Metrics | [docs/en/metrics.md](docs/en/metrics.md) | [docs/zh/metrics.md](docs/zh/metrics.md) |
| Model Loading | [docs/en/model_loading.md](docs/en/model_loading.md) | [docs/zh/model_loading.md](docs/zh/model_loading.md) |
| Offload | [docs/en/offload.md](docs/en/offload.md) | [docs/zh/offload.md](docs/zh/offload.md) |
| Ops | [docs/en/ops.md](docs/en/ops.md) | [docs/zh/ops.md](docs/zh/ops.md) |
| Parallel | [docs/en/parallel.md](docs/en/parallel.md) | [docs/zh/parallel.md](docs/zh/parallel.md) |
| Profiler | [docs/en/profiler.md](docs/en/profiler.md) | [docs/zh/profiler.md](docs/zh/profiler.md) |
| Service | [docs/en/service.md](docs/en/service.md) | [docs/zh/service.md](docs/zh/service.md) |
| Testing | [docs/en/testing.md](docs/en/testing.md) | [docs/zh/testing.md](docs/zh/testing.md) |

## Key Configuration Classes

Located in `telefuser/core/config.py`:
- `AttnImplType`: Attention implementations (TORCH_SDPA, FLASH_ATTN_*, SAGE_ATTN_*, RADIAL_ATTN, etc.)
- `ParallelConfig`: Distributed processing (device_ids, sp_ulysses_degree, sp_ring_degree)
- `ModelRuntimeConfig`: Runtime settings (dtype, attention impl, offloading)

## Key Distributed APIs

Located in `telefuser/distributed/`:
- `ulysses_scatter_heads`: Scatter heads, gather sequence (for QKV in Ulysses SP)
- `ulysses_gather_heads`: Gather heads, scatter sequence (for output in Ulysses SP)
- `RingP2PComm`: P2P communication for Ring Attention on long sequences
- `PipelineP2PComm`: Pipeline parallelism communication between stages

## Key Triton Kernels

Located in `telefuser/kernel/triton/`:
- `fused_add_rms_norm`: Fused residual add + RMSNorm
- `fused_scale_shift`: Fused scale and shift operations
- `fused_layernorm_scale_shift_gate_select01`: LayerNorm + scale/shift + gate selection
- `fused_residual_layernorm_scale_shift_gate_select01`: Residual add + LayerNorm + scale/shift + gate selection
- `apply_rotary_embedding`: Rotary Position Embedding (RoPE)

## Test Markers

```python
@pytest.mark.gpu           # Requires GPU
@pytest.mark.multi_gpu     # Requires multiple GPUs
@pytest.mark.distributed   # Requires distributed setup
@pytest.mark.slow          # Long-running tests
```

**GPU tests in CPU CI:** Wrap GPU-dependent imports in `try-except` with `pytest.skip(..., allow_module_level=True)` to skip in CPU-only environments.

## Development Guidelines

- Start responses with "**Developer,**" prefix
- DO NOT use `sys.path.insert()` in test files
- Keep this file synchronized with codebase changes
- See `CONTRIBUTING.md` for contribution guidelines
- Update CLAUDE.md if needed (new patterns, new modules, architecture changes)

## Task Completion Checklist

**After completing a coding task, ask the user:**

> Code changes completed. Would you like to run `/simplify` for code review?

If the user agrees:
1. Run `/simplify` to review and optimize the changed code for reuse, quality, and efficiency
