# SGLang-Diffusion LingBot Stream Baseline

This directory contains the SGLang-Diffusion stream-world baseline for the
TeleFuser benchmark suite.

SGLang-Diffusion is served through the diffusion runtime in the main
`sgl-project/sglang` package (`sglang.multimodal_gen`). There is no separate
`sgl-project/SGLang-Diffusion` repository required for this adapter.

The goal is to compare framework/runtime behavior for realtime world-model
serving:

- TeleFuser `stream-serve` with `LingBot-World-Fast`
- SGLang-Diffusion `sglang serve` with `robbyant/lingbot-world-fast-diffusers`
- the same prompt, first frame, output size, FPS target, and timed control trace
- while adapting only the transport protocol and runtime framework

SGLang-Diffusion uses WebSocket + MessagePack for this path, not WebRTC. The
adapter is implemented in AIPerf and maps the shared control trace into SGLang
`camera_actions` events before writing the common stream artifacts.

## Layout

- `benchmark_contract.yaml`: stream-world baseline contract
- `configs/stream_lingbot_world_fast_quick.json`: default quick benchmark config
- `configs/stream_lingbot_world_fast_h100_gpu_resident.json`: 1xH100 tuned
  GPU-resident config
- `scripts/run_service.sh`: SGLang-Diffusion service launch helper
- `scripts/run_mock_stream_service.sh`: SGLang-style WebSocket mock stream helper
- `scripts/run_stream_bench.py`: compatibility launcher for AIPerf
- `scripts/run_stream_bench.sh`: thin AIPerf launch helper

## Start The Service

Install SGLang-Diffusion in the service environment first, then run:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

For an offline model zoo path, override the model path:

```bash
SGLANG_LINGBOT_MODEL_PATH=<lingbot-world-fast-model-path> \
  bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

If the `sglang` console script points at a stale Python path, run through the
environment's Python executable instead:

```bash
SGLANG_PYTHON=/path/to/venv/bin/python \
SGLANG_LINGBOT_MODEL_PATH=<lingbot-world-fast-model-path> \
  bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

The launcher prepends `python_shims/` to `PYTHONPATH` so the benchmark can run
against SGLang-Diffusion environments that omitted the small `addict.Dict`
configuration dependency.

For a fair performance comparison against the TeleFuser default LingBot service,
run SGLang-Diffusion GPU-resident on one GPU first. Do not enable
`--dit-cpu-offload`, `--dit-layerwise-offload`, `--text-encoder-cpu-offload`,
`--vae-cpu-offload`, or `SGLANG_LINGBOT_NATIVE_NORM_FALLBACK=1` for the
performance baseline. The launcher defaults to `--performance-mode speed`;
`auto` may silently enable VAE layerwise offload even when the individual CPU
offload flags are false. If the GPU-resident configuration OOMs, record that
failure instead of relabeling an auto-offloaded run. CPU/offload settings are
useful for smoke testing the adapter and protocol path, but their metrics are
not directly comparable with TeleFuser's default GPU-resident result.

If the SGLang service environment is missing an import-time dependency that is
available elsewhere, place only that dependency in a narrow directory and pass
it through `SGLANG_EXTRA_PYTHONPATH`. Avoid pointing this at a full
site-packages directory because it can override SGLang's own `torch`, `sglang`,
or CUDA packages.

```bash
SGLANG_EXTRA_PYTHONPATH=/path/to/websockets-only \
SGLANG_PYTHON=/path/to/venv/bin/python \
SGLANG_LINGBOT_MODEL_PATH=<lingbot-world-fast-model-path> \
  bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

If the environment lacks `cuda.bindings`, set
`SGLANG_LINGBOT_NATIVE_NORM_FALLBACK=1` to force the LingBot norm shim onto
PyTorch-native scale/shift code. Keep this disabled on correctly provisioned
SGLang environments because it bypasses CuTe fused norm kernels.

Default address:

- `http://<sglang-stream-host>:30000`

Health check:

```bash
curl http://<sglang-stream-host>:30000/health
```

## Run The Benchmark

Set up the AIPerf dependency repository first:

```bash
bash scripts/setup_aiperf_repo.sh
```

Then run:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh
```

For the validated 1xH100 tuned workload, run:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh \
  benchmarks/baseline/sglang_lingbot_stream/configs/stream_lingbot_world_fast_h100_gpu_resident.json
```

The launcher delegates to:

```bash
uv run --project benchmarks/aiperf aiperf profile \
  --stream-config benchmarks/baseline/sglang_lingbot_stream/configs/stream_lingbot_world_fast_quick.json
```

This uses the same control trace as the TeleFuser stream benchmark:

- `benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`

The adapter maps:

- `ArrowUp` to `w`
- `ArrowDown` to `s`
- `ArrowLeft` to `a`
- `ArrowRight` to `d`

SGLang emits no separate control-ack message. The benchmark records control
acknowledgement latency when `chunk_stats.event_id` shows that a control event
has been sampled, and records control-to-next-frame latency when
`frame_batch.event_id` appears on a video batch.

## Target Chunk Metrics

AIPerf directly normalizes the native SGLang-Diffusion `chunk_stats` fields. It
maps scheduler forward time to target compute, payload-build time to encoding,
and separately retains request preparation, output pacing, header/payload/write
time, total chunk time, frames, batches, raw bytes, WebSocket payload bytes, and
content type. The same warmup exclusion, percentiles, weighted compute FPS,
JSONL artifacts, and HTML report used by TeleFuser are reused without adding an
AIPerf dependency to SGLang.

The instrumented SGLang realtime endpoint also sends its existing reset-scoped
`OutputBatch.peak_memory_mb` fact. AIPerf maps it only to
`chunk_peak_reserved_bytes`, because SGLang obtains that value from
`max_memory_reserved()`; allocated peak remains unavailable. Pipeline/runtime
creation durations are still unavailable. The contract snapshots `/v1/models`
as descriptive model and pipeline identity.

## Validated 1xH100 Result

The 2026-07-13 validation used one 80GB H100, SGLang `performance_mode=speed`,
no CPU or layerwise offload, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
and one warmup plus one measured session. The host's FlashInfer/CUDA compiler
headers were incompatible, so the documented RoPE fallback was enabled; this
environment qualification must remain attached to the result.

The 9-sink / 18-frame cache geometry in the compare config failed in both
warmup and profiling with CUDA OOM at approximately 79.16 GiB process memory.
The tuned 6-sink / 9-frame config completed successfully:

| Metric | Measured value |
|---|---:|
| Profile sessions | 1 / 1 |
| Frames / chunks | 93 / 8 |
| Steady chunks after warmup exclusion | 7 |
| Target compute FPS | 5.1906 |
| Client stream FPS | 5.6788 |
| First-frame latency | 3076.18 ms |
| Session runtime | 19.277 s |
| Chunk compute mean / p90 | 2.3119 / 2.3362 s |
| Chunk encode mean | 0.0551 s |
| Chunk total mean | 2.3699 s |
| Peak reserved allocator bytes | 76,919,341,056 (73,356 MiB) |

The OOM and tuned success are different cache configurations and must not be
combined into a same-configuration comparison. Their AIPerf artifacts are under
`work_dirs/benchmarks/sglang_lingbot_stream/` in
`h100_gpu_resident_speed_official_cache_oom_20260713` and
`h100_gpu_resident_speed_rope_fallback_peak_20260713`, respectively.

## Mock Stream Transport

The mock service does not load SGLang or any diffusion model. It exposes the
same `/health`, `/v1/models`, and `/v1/realtime_video/generate` shape used by the
stream benchmark adapter, sends synthetic `frame_batch` payloads at a fixed FPS,
acknowledges controls, and emits synthetic native `chunk_stats` to validate the
mapping path.

Start it with:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_mock_stream_service.sh \
    --host <sglang-stream-bind-host> \
    --port <sglang-stream-port>
```

Then run:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh \
    benchmarks/baseline/sglang_lingbot_stream/configs/stream_transport_mock_compare.json \
    --stream-server-url http://<sglang-stream-host>:<sglang-stream-port>
```

Use this only for streaming/transport overhead comparisons.
