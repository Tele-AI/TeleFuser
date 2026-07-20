# TeleFuser - Agent Guidelines

## Project Overview

TeleFuser is a high-performance framework for efficient multimodal generation model inference (image/video generation, video super-resolution). 

**Tech Stack:** Python 3.10-3.13, PyTorch 2.6+, CUDA 12.8+, FastAPI, Ray

**Supported Models:** WanVideo (Wan2.1/2.2), Qwen-Image, Z-Image, FlashVSR, HunyuanVideo, Flux2 Klein, LTX Video, LiveAct, LongCat-Video, LingBot-World

## Commands

```bash
pip install -e ".[dev]"           # Development installation
pre-commit run --all-files        # Linting checks
pytest tests/                     # Run tests
bash scripts/run_ci_tests.sh      # Full CI suite
telefuser serve /path/to/pipeline.py --port 8000  # Start API server
```

## Troubleshooting

When multi-GPU inference hangs, zombie processes may remain. Clean them up with:

```bash
ps aux | grep -E 'spawn_main' | grep -v grep | awk '{print $2}' | xargs kill -9
```

## Architecture

```
telefuser/
├── core/             # Base abstractions: BasePipeline, BaseStage, configs
├── pipelines/        # Model-specific pipelines
│   ├── wan_video/    # Wan2.1/2.2: T2V, I2V, FL2V
│   ├── qwen_image/   # Qwen-Image: T2I, Edit
│   ├── z_image/      # Z-Image: T2I
│   ├── flashvsr/     # FlashVSR: VSR
│   ├── hunyuan_video_1_5/  # HunyuanVideo: T2V, I2V
│   ├── flux2_klein/  # Flux2 Klein: T2I
│   ├── ltx_video/    # LTX Video: I2V + Audio
│   ├── liveact/      # LiveAct: S2V (speech-to-video)
│   ├── longcat_video/ # LongCat-Video: T2V, I2V
│   ├── lingbot_world_fast/  # LingBot shared causal-fast engine
│   ├── lingbot_world_v2/    # LingBot-World v2 causal-fast facade
│   └── common/       # Shared pipeline utilities
├── models/           # Model architectures: DiT, VAE, text encoders
├── ops/              # Custom operations: attention, FFN, normalization
├── kernel/           # Triton kernels: RMSNorm, rotary, quant, fused ops
│   └── triton/       # Pure Triton implementations
├── platforms/        # Hardware abstraction: CUDA, NPU, CPU
├── distributed/      # FSDP, TP, PP, SP, Ring/Ulysses attention
│   ├── ulysses_comm.py   # Ulysses All-to-All: ulysses_scatter_heads, ulysses_gather_heads
│   ├── ring.py            # Ring P2P communication for long sequences
│   ├── pp_comm.py         # Pipeline parallelism
│   ├── fsdp.py            # Fully Sharded Data Parallel
│   ├── tp_parallelize.py  # Tensor Parallelism
│   └── parallel_shard.py  # Parallel sharding utilities
├── schedulers/       # Diffusion schedulers
├── feature_cache/    # Feature caching: AdaTaylorCache
├── cache/            # General cache management
├── offload/          # CPU offload strategies
├── metrics/          # Metrics collection and monitoring
├── orchestrator/     # Pipeline orchestration
├── worker/           # Distributed worker management
├── entrypoints/      # CLI entry points
├── service/          # FastAPI service
└── client/           # Python SDK
```

### LingBot Streaming State

- Long-running LingBot sessions generate noise and VAE condition latents per chunk; do not retain duration-sized chunk lists.
- Incremental VAE encoder and decoder feature caches must be session-owned so concurrent sessions remain isolated.
- A streaming stage worker is owned by exactly one actor. Drain session work and release
  stage caches through that actor in reverse topological order; session facades must not
  call actor-owned workers directly.
- LingBot VAE encode, decode, and DiT stages may overlap on the same GPU. Do not infer
  resource groups from device placement; resolve memory pressure by moving stages.
  Use scheduler session metrics and bounded-attention long replays to validate latency
  and memory without retaining duration-sized tensor lists.

### Layer Architecture Principles For Models

TeleFuser's model follows a strict layered architecture for operations:

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
| Hash Config Management | [docs/en/hash_config_management.md](docs/en/hash_config_management.md) | [docs/zh/hash_config_management.md](docs/zh/hash_config_management.md) |
| Logging | [docs/en/logging.md](docs/en/logging.md) | [docs/zh/logging.md](docs/zh/logging.md) |
| Metrics | [docs/en/metrics.md](docs/en/metrics.md) | [docs/zh/metrics.md](docs/zh/metrics.md) |
| Model Loading | [docs/en/model_loading.md](docs/en/model_loading.md) | [docs/zh/model_loading.md](docs/zh/model_loading.md) |
| Offload | [docs/en/offload.md](docs/en/offload.md) | [docs/zh/offload.md](docs/zh/offload.md) |
| Ops | [docs/en/ops.md](docs/en/ops.md) | [docs/zh/ops.md](docs/zh/ops.md) |
| Parallel | [docs/en/parallel.md](docs/en/parallel.md) | [docs/zh/parallel.md](docs/zh/parallel.md) |
| Profiler | [docs/en/profiler.md](docs/en/profiler.md) | [docs/zh/profiler.md](docs/zh/profiler.md) |
| Service | [docs/en/service.md](docs/en/service.md) | [docs/zh/service.md](docs/zh/service.md) |
| Service Metadata | [docs/en/service_metadata.md](docs/en/service_metadata.md) | [docs/zh/service_metadata.md](docs/zh/service_metadata.md) |
| Stream Server | [docs/en/stream_server.md](docs/en/stream_server.md) | [docs/zh/stream_server.md](docs/zh/stream_server.md) |
| Testing | [docs/en/testing.md](docs/en/testing.md) | [docs/zh/testing.md](docs/zh/testing.md) |
| torch.compile Compatibility | [docs/en/torch_compile_compatibility.md](docs/en/torch_compile_compatibility.md) | [docs/zh/torch_compile_compatibility.md](docs/zh/torch_compile_compatibility.md) |

## Key Configuration Classes

Located in `telefuser/core/config.py`:
- `AttnImplType`: Attention implementations (TORCH_SDPA, FLASH_ATTN_*, SAGE_ATTN_*, RADIAL_ATTN, etc.)
- `AttentionConfig`: Attention configuration with sparse attention support
- `ParallelConfig`: Distributed processing (device_ids, sp_ulysses_degree, sp_ring_degree)
- `ModelRuntimeConfig`: Runtime settings (dtype, attention impl, offloading)
- `FeatureCacheConfig`: Feature caching configuration for AdaTaylorCache
- `CompileConfig`: torch.compile configuration
- `QuantConfig`: Quantization settings (FP8, INT8)
- `OffloadConfig`: CPU offload configuration
- `LoraConfig`: LoRA weight loading configuration
- `SparseAttentionConfig`: Sparse attention pattern configuration
- `RayConfig`: Ray distributed inference configuration
- `RayGPUConfig`: Ray GPU allocation configuration

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
- `fused_scale_shift_gate_select`: Scale/shift + gate selection for dual-branch models
- `apply_rotary_embedding`: Rotary Position Embedding (RoPE)
- `fused_merge_attn_states`: Merge attention states with optional gating
- `per_token_quant_fp8`: FP8 per-token quantization
- `per_token_dequant_fp8`: FP8 per-token dequantization
- `per_block_int8`: INT8 per-block quantization
- `norm_infer`: Optimized normalization for inference
- `triton_one_pass_rms_norm`: Single-pass RMSNorm

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
- When creating commits, use a conventional-commit summary and a detailed body. The body should list the main changes and the verification commands or checks that were run. Do not use a one-line commit message for non-trivial changes.
- Keep this file synchronized with codebase changes
- See `CONTRIBUTING.md` for contribution guidelines
- Update CLAUDE.md if needed (new patterns, new modules, architecture changes)

## Interaction Workflow

### Plan-first mode

When the user explicitly asks for planning before execution, the agent MUST stop
before making edits or running mutating commands. Treat any of the following as
plan-first triggers:

- "先 plan"
- "先计划"
- "先不要改"
- "只分析"
- "不要执行"
- "等我确认"
- "每阶段确认"
- "先给 TODO"

In plan-first mode:

1. Inspect only the context needed to produce a concrete plan.
2. Present a concise plan and TODO list.
3. Wait for the user's confirmation before editing files, staging changes,
   committing, installing dependencies, or running other mutating operations.
4. If the user asks for stage-by-stage confirmation, stop after each completed
   stage, report the result, and wait for confirmation before continuing.

Use the task/TODO mechanism available in the current agent environment:

- Codex: use `update_plan` when available.
- Claude Code: use TaskCreate/TaskUpdate when available.
- Other environments: maintain a short markdown checklist in the response.

### Default execution mode

If the user does not request plan-first mode, follow the active agent's higher
priority instructions for autonomy and execution. For routine code or
documentation changes, it is acceptable to proceed directly after briefly
stating what will be changed.

When clarification is genuinely required, ask a direct question using the
current environment's available user-input mechanism. Do not require a specific
tool name such as AskUserQuestion unless that tool exists in the current
environment.

### Completion

When work is complete, report:

- what changed
- what verification was run
- any verification that could not be run and why
- whether unrelated dirty worktree files were left untouched

## Task Completion Checklist

**After completing a coding task, ask the user:**

> Code changes completed. Would you like to:
> 1. Run `/simplify` for code review?
> 2. Use GPT 5.4 for code review? (via MCP tool `review_code`)

If the user chooses `/simplify`:
1. Run `/simplify` to review and optimize the changed code for reuse, quality, and efficiency

If the user chooses GPT 5.4 review:
1. Call the MCP tool `mcp__gpt-review__review_code` to send the diff to GPT 5.4 for review
2. Present the review results to the user
