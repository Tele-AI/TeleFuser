# TeleFuser and AIPerf Benchmark

This page describes how to benchmark TeleFuser batch video serving, TeleFuser WebRTC stream serving, and the SGLang-Diffusion LingBot stream baseline through AIPerf.

Current scope:

- `telefuser serve`
- OpenAI-compatible `/v1/videos` API
- AIPerf end-to-end request latency and throughput
- optional HTTP trace and TeleFuser metrics scraping
- `telefuser stream-serve`
- WebRTC / DataChannel world-model sessions
- SGLang-Diffusion LingBot World WebSocket baseline
- frame-level and control-latency metrics

Stream sessions are now executed by AIPerf's native `profile --stream-config` path. The TeleFuser and SGLang scripts only select the target config and URL.

AIPerf defaults to route-based ICE selection (`ice_host_ips: ["auto"]`) so a multi-homed host advertises the source address used to reach the stream target. Set the comma-separated `TELEFUSER_STREAM_BENCH_ICE_HOST_IPS` variable or repeat `--stream-ice-host-ip` only for container, TURN, or special-routing overrides; an empty list restores all aioice-discovered addresses.

## Assets

The benchmark assets live in:

```text
benchmarks/telefuser_aiperf/
├── benchmark_contract.yaml
├── stream_benchmark_contract.yaml
├── configs/
├── data/
└── scripts/
```

## Prerequisites

1. Start a TeleFuser batch video server.
2. Set up the AIPerf dependency repository.

Example:

```bash
telefuser serve \
    examples/wan_video/wan21_14b_image_to_video_h100.py \
    --port 8000 \
    --task i2v

bash scripts/setup_aiperf_repo.sh
```

The setup script defaults to the `teleai` branch of `https://github.com/ActivePeter/aiperf`.

## Quick Start

Run the default benchmark:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh
```

The script checks:

```bash
curl http://127.0.0.1:8000/v1/service/health
```

and then runs AIPerf with:

```bash
aiperf profile --config benchmarks/telefuser_aiperf/configs/video_generation_quick.yaml
```

## Configs

- `video_generation_quick.yaml`: minimal connectivity and latency check
- `video_generation_e2e.yaml`: fuller E2E run with warmup, download, trace, and optional server metrics
- `video_generation_rate.yaml`: Poisson arrival benchmark for rate-based load

## Why Built-In Readiness Probe Is Disabled

These configs intentionally do not use AIPerf's built-in `waitForModel*` readiness probe.

Reason:

- TeleFuser is being benchmarked through `video_generation`
- AIPerf readiness probing is better aligned with chat/completions/embeddings-style endpoints
- TeleFuser does not currently expose a stable `/v1/models` path for this use case

So the helper script uses TeleFuser's `/v1/service/health` instead.

## Stream Benchmark

The stream benchmark path uses AIPerf's contract, runner, WebRTC adapter, observability, and artifact layers:

- `benchmarks/aiperf/src/aiperf/streaming/`
- `benchmarks/aiperf/src/aiperf/streaming/adapters/telefuser_webrtc.py`
- `benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_compare.json`
- `benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json`
- `benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`

It records offer RTT, connection latency, first-frame latency, stream FPS, control acknowledgement latency, and control-to-next-frame latency. With `benchmark_metrics: true`, it also captures target pipeline/runtime initialization, per-chunk compute and encoding time, allocator peaks, steady-state compute FPS, runtime/cache dimensions, and target environment identity. `warmup_chunks` controls the leading chunks excluded from steady-state aggregates.
Each completed stream run also writes `target_metadata.json` and a self-contained `stream_report.html` from AIPerf. The same report UI renders TeleFuser, SGLang, and registered third-party adapters without target-specific frontend code.

The thin launcher is equivalent to:

```bash
uv run --project benchmarks/aiperf --extra streaming-webrtc \
  aiperf profile \
  --stream-config benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json
```

## SGLang-Diffusion Stream Baseline

This adapter uses the diffusion runtime in the main `sgl-project/sglang`
package (`sglang.multimodal_gen`); it does not require a separate
`SGLang-Diffusion` repository.

Start SGLang-Diffusion:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

Then run:

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh
```

This baseline uses `robbyant/lingbot-world-fast-diffusers`, WebSocket + MessagePack, and the same `stream_lingbot_controls.json` trace as the TeleFuser stream benchmark.

The compare configs use a 90 second session window so the baseline has enough time to emit first frames and control-loop metrics; the `quick` configs remain local smoke checks.

Fairness note: TeleFuser's LingBot world-model service runs in its default 1-GPU GPU-resident configuration. The SGLang-Diffusion performance baseline must therefore use `--performance-mode speed` and disable CPU, DiT layerwise, VAE, and text-encoder offload. SGLang's `auto` mode may enable VAE layerwise offload even when the individual CPU-offload flags are false. If the GPU-resident configuration OOMs, record that failure instead of relabeling an auto-offloaded result.

AIPerf normalizes SGLang's native `chunk_stats` into the shared target chunk
schema: scheduler compute, request preparation, output encoding/pacing/writes,
total chunk duration, frame/batch counts, and raw/wire bytes appear in
`summary.json`, `sessions.jsonl`, and `stream_report.html`. An instrumented SGLang
endpoint also sends its reset-scoped `OutputBatch.peak_memory_mb`; AIPerf maps
this only to `chunk_peak_reserved_bytes`, matching SGLang's
`max_memory_reserved()` source. Initialization phase durations and allocated
peaks remain unavailable instead of being inferred.

On 2026-07-13, a true 1xH100 GPU-resident run established two separate outcomes.
The 9-sink / 18-frame cache workload failed with CUDA OOM at approximately
79.16 GiB process memory. The tuned 6-sink / 9-frame workload completed one
warmup and one measured session: 93 frames in 8 chunks, 5.1906 target compute
FPS, 5.6788 client stream FPS, 3076.18 ms first-frame latency, and a maximum
reset-scoped reserved peak of 76,919,341,056 bytes (73,356 MiB). The host
required the documented RoPE fallback because its FlashInfer CUDA compiler and
headers were incompatible. These two cache geometries must not be reported as a
same-configuration comparison.

## Historical Metrics Service (GreptimeDB)

Cross-run storage, APIs, and visualization are owned by AIPerf. TeleFuser and SGLang
produce canonical artifacts; neither target repository contains a database client or a
target-specific history frontend.

With GreptimeDB running, import existing profile and stream artifacts:

```bash
uv run --project benchmarks/aiperf aiperf history ingest \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --artifact-root work_dirs/benchmarks
```

Then start the API and bundled Vue application:

```bash
uv run --project benchmarks/aiperf aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --artifact-root work_dirs/benchmarks \
  --host 127.0.0.1 \
  --port 8095
```

Open `http://127.0.0.1:8095/` to browse runs, compare metrics across runs, and inspect
session, control, phase, chunk, timeslice, GPU, Prometheus, and normalized points. Warmup
and profiling points retain separate phase values. Client delivery `stream_fps` and
target `chunk_compute_fps` remain separate metrics.

GreptimeDB is mandatory. A connection or table-creation failure stops startup, query
failures return HTTP 503, and the API/UI never switch to SQLite, an in-memory index, or
direct artifact queries. JSON and JSONL artifacts remain replayable inputs only.

See the AIPerf
[deployment guide](https://github.com/ActivePeter/aiperf/blob/teleai/docs/tutorials/history-dashboard.md), the
[history service design](https://github.com/ActivePeter/aiperf/blob/teleai/docs/dev/history-service-design.md),
and the TeleFuser
[benchmark design](/TeleFuser/zh/benchmark_aiperf_design/) for the complete ownership and
metric-boundary rules.
