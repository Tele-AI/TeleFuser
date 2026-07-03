# TeleFuser and AIPerf Benchmark

This page describes how to benchmark TeleFuser batch video serving with AIPerf.

Current scope:

- `telefuser serve`
- OpenAI-compatible `/v1/videos` API
- end-to-end request latency and throughput
- optional HTTP trace and TeleFuser metrics scraping

Not covered yet:

- `telefuser stream-serve`
- WebRTC / DataChannel world-model sessions
- frame-level and control-latency metrics

For remote stream setup, use:

```bash
python3 scripts/remote_bench_sync.py stream-bootstrap
```

That helper syncs the stream assets, installs WebRTC dependencies, auto-detects a usable `model_zoo` root, and writes a non-destructive `model_zoo` symlink inside the remote repo for later `telefuser stream-serve` startup.

Both `run_stream_bench.sh` and `run_video_bench.sh` raise the open-file limit to `8192` by default to avoid AIPerf/WebRTC startup failures; override with `TELEFUSER_BENCH_NOFILE_LIMIT` if your environment needs a different value.

For stream runs, `run_stream_bench.py` also auto-detects non-loopback local IPs for ICE candidate gathering when no explicit allowlist is provided. Use `--ice-host-ip` or `TELEFUSER_WEBRTC_ICE_HOST_IPS` only when you need to pin a specific address.

## Assets

The benchmark assets live in:

```text
benchmarks/telefuser_aiperf/
├── README.md
├── configs/
├── data/
└── scripts/
```

## Prerequisites

1. Start a TeleFuser batch video server.
2. Install AIPerf from the vendored clone.

Example:

```bash
telefuser serve \
    examples/wan_video/wan21_14b_image_to_video_h100.py \
    --port 8000 \
    --task i2v

pip install -e ./benchmarks/aiperf
```

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

## Limitation

This is a **batch video serving benchmark**, not yet a **real-time world-model streaming benchmark**.

For `stream-serve`, the next layer should add:

1. a WebRTC transport
2. a world-model session endpoint
3. an interactive trace workload
4. frame-level latency metrics
