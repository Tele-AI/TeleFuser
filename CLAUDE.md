# TeleFuser - Agent Guidelines

## Scope And Sources Of Truth

TeleFuser is a high-performance multimodal inference framework built with Python, PyTorch, CUDA, FastAPI, and Ray.

- Treat the current repository, tests, and documentation as the API source of truth.
- Use [README.md](README.md) for project structure, supported models, and documentation discovery.
- Use [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, contribution workflow, and coding standards.
- Read the relevant guides under `docs/en/` or `docs/zh/` before changing a subsystem.
- Prefer the closest maintained implementation and its tests over adding a new pattern.

## Repository Map

```text
telefuser/
  core/             Base abstractions, configuration, and module management
  pipelines/        Model-specific pipeline and stage implementations
  models/           DiT, VAE, text encoder, and other model architectures
  ops/              Public compile-aware operation dispatch
  kernel/triton/    Internal Triton kernel implementations
  schedulers/       Diffusion schedulers
  distributed/      FSDP, tensor/pipeline/sequence parallelism, and communication
  feature_cache/    Feature caching implementations
  cache/            General cache management
  offload/          CPU and device offload strategies
  metrics/          Runtime measurement and raw metrics
  orchestrator/     Request and actor-based streaming orchestration
  worker/           Distributed worker management
  service/          FastAPI and LiveKit-backed services
  client/           Python client SDK
  entrypoints/      CLI entry points
  platforms/        CUDA, NPU, and CPU abstractions
examples/           Runnable examples and model-specific usage guides
tests/              Unit, integration, and regression tests
docs/               English and Chinese documentation
tf-kernel/          Independently packaged optional CUDA kernels
```

## Common Commands

```bash
pip install -e ".[dev]"
pre-commit run --all-files
pytest tests/
bash scripts/run_ci_tests.sh
telefuser serve /path/to/pipeline.py --port 8000
telefuser stream-serve /path/to/pipeline.py --port 8088
```

## Development Rules

- Follow PEP8 and ruff with a line length of 120.
- Write comments and docstrings in English.
- Add type annotations to all public function parameters.
- Use Python 3.10+ syntax such as `str | None` and `list[int]`.
- Do not use `sys.path.insert()` in tests.
- Preserve unrelated worktree changes.
- Use a conventional-commit summary and a detailed body for non-trivial commits. Include the main changes and the
  verification performed.
- Update this file only when a change introduces a durable, cross-cutting rule that agents must know before editing.

## Architecture Boundaries

### Models, Ops, And Kernels

- Code under `telefuser/models/` must import operations through `telefuser.ops`, never directly from
  `telefuser.kernel.triton` or optional kernel packages.
- Code under `telefuser/ops/` owns compile-aware dispatch: use native PyTorch while compiling, optimized kernels for
  supported eager execution, and native fallbacks elsewhere.
- Code under `telefuser/kernel/triton/` contains kernel implementations. Do not add
  `torch.compiler.is_compiling()` branches there.
- Keep model code on the public ops layer even when adding or changing an optimized backend.

### TF-Kernel Packaging

- `telefuser` and `tf-kernel` are independent Python distributions with separate metadata, versions, wheels, tests,
  and releases.
- Do not publish prebuilt `tf-kernel` wheels or a source distribution to a public package index, and do not add a
  TeleFuser `kernel` extra.
- Local `tf-kernel` source builds use its Makefile. Direct and editable pip builds are intentionally rejected.
- Locally built wheels may be shared only through direct artifacts or scoped artifact repositories for compatible
  environments. Keep different target SM families isolated because the wheel filename does not encode the SM target.
- Do not make the TeleFuser build invoke pip or compile `tf-kernel`, and do not publish a local-path dependency or a
  TeleFuser `kernel` extra.
- Do not add GitHub Actions workflows that compile or publish `tf-kernel`; kernel wheels require an explicitly
  provisioned CUDA/NVCC host and manual validation.
- Load only the wheel extension matching the visible GPU architecture family. Keep build compatibility facts in
  `tf_kernel._build_info` and validated SageAttention dispatch in `tf_kernel.capabilities`.
- `telefuser.ops.attention` may prefer `tf_kernel` and fall back to `sageattention`; callers must use the ops layer.

### Metrics

- `telefuser/metrics/runtime.py` measures synchronized target-side phase duration and allocator peaks.
- Target services emit only raw, bounded phase, chunk, and runtime facts. AIPerf owns warmup exclusion, aggregation,
  semantic mapping, artifacts, and visualization.
- Keep client delivery and target compute metrics distinct, including `stream_fps` and `chunk_compute_fps`.

## Pipeline Integration Contract

When adding or porting a pipeline:

- Select the closest maintained pipeline, public example, and tests as structural baselines. Read the relevant
  adding-new-example, adding-new-model, adding-new-stage, model-loading, configuration, and service guides.
- Base each new model-family example README on `examples/README_TEMPLATE.md`; keep its required section order and
  remove inapplicable optional sections and all placeholders.
- Inventory model-specific classes and configuration fields, then map them to upstream behavior and the selected
  baseline.
- Reuse `BasePipeline`, `BaseStage`, `ModuleManager`, existing configuration dataclasses, example contracts, and
  service schemas.
- The expected list of new framework-level interfaces, shared configuration fields, environment variables, loaders,
  registries, CLI options, and service schema fields is empty.
- Reusing a framework API does not authorize changing it. Use existing file-list, wildcard, loading, registry,
  orchestration, configuration, and service behavior as-is.
- If an existing extension point cannot express a requirement, report the exact gap, alternatives, affected callers,
  and compatibility impact, then obtain explicit approval before changing shared framework code.
- Do not attach ad-hoc configuration attributes or create parallel interfaces for convenience.
- Do not add an environment variable unless explicitly requested or an existing documented variable has exactly the
  required semantics. If a new process-level variable is unavoidable, obtain approval first and add documentation,
  defaults, validation, precedence rules, and tests.
- Prefer function parameters for request inputs, dataclass fields for runtime configuration, CLI options for command
  inputs, and service schemas for API inputs.
- Establish a faithful upstream path before stage splitting or optimization. Preserve computation order, tensor
  shapes, conditioning paths, scheduler semantics, defaults, and output format; read checkpoint metadata instead of
  guessing architecture values.
- Do not combine initial integration with sparse attention, caching, quantization, refactoring, or other optimizations
  unless explicitly requested.
- Before completion, inspect the diff for new environment lookups and public configuration or service surfaces. Report
  every intentional addition and every difference from upstream and the selected baseline.

## Testing

- Follow [docs/en/testing.md](docs/en/testing.md) and the pytest configuration in `pyproject.toml` for markers and test
  selection.
- In CPU CI, wrap GPU-dependent imports in `try-except` and call `pytest.skip(..., allow_module_level=True)` when the
  dependency is unavailable.
- Scale verification with risk: run focused tests for narrow changes and broader tests for shared contracts or
  cross-module behavior.

## Documentation Map

- Pipeline integration: [adding_new_example.md](docs/en/adding_new_example.md),
  [adding_new_model.md](docs/en/adding_new_model.md), [adding_new_stage.md](docs/en/adding_new_stage.md),
  [model_loading.md](docs/en/model_loading.md), and [configuration.md](docs/en/configuration.md).
- Runtime architecture: [ops.md](docs/en/ops.md), [attention.md](docs/en/attention.md),
  [parallel.md](docs/en/parallel.md), and [torch_compile_compatibility.md](docs/en/torch_compile_compatibility.md).
- Services and streaming: [service.md](docs/en/service.md), [stream_server.md](docs/en/stream_server.md), and
  [stream_scheduler.md](docs/en/stream_scheduler.md).
- LingBot work: [LingBot-World examples](examples/lingbot/README.md) and
  [LingBot-Video examples](examples/lingbot_video/README.md).
- Benchmarking: [metrics.md](docs/en/metrics.md) and [benchmark_aiperf.md](docs/en/benchmark_aiperf.md).

## Interaction Workflow

- Start responses with the `**Developer,**` prefix.

### Plan-First Requests

Treat phrases such as "先 plan", "先计划", "先不要改", "只分析", "不要执行", "等我确认", "每阶段确认", and
"先给 TODO" as explicit requests to plan before execution. Do not infer plan-first mode from an ordinary change
request.

1. Inspect only enough context to produce a concrete plan.
2. Present a concise plan and TODO list using the environment's native planning mechanism when available.
3. Wait for confirmation before editing, installing dependencies, staging, committing, or running other mutating
   commands.
4. When stage-by-stage confirmation is requested, stop after each completed stage and wait before continuing.

### Default Execution

- For change requests without a plan-first trigger, work autonomously through implementation and verification after a
  brief progress update.
- Do not request confirmation for routine inspection, scoped edits, tests, formatting, or other reversible actions
  needed to complete the task.
- Make reasonable assumptions when they keep the work within the requested scope. Pause only for an unresolved choice
  that would materially change the result, a destructive action, or a required expansion of public interfaces or task
  scope.
- On completion, report what changed, verification performed, verification that could not run and why, and whether
  unrelated worktree changes were left untouched.
