# TeleFuser AIPerf Benchmarks

This directory contains TeleFuser-specific benchmark assets for running end-to-end video serving tests with AIPerf.

Current scope:
- `telefuser serve` batch video generation over the OpenAI-compatible `/v1/videos` API
- `telefuser stream-serve` WebRTC end-to-end session benchmarking
- End-to-end latency, throughput, HTTP trace, frame-level latency, and optional server-metrics scraping

## Layout

- `configs/`: AIPerf YAML configs for batch video serving and JSON configs for stream benchmarking
- `data/`: Prompt files and control traces
- `scripts/`: Helper scripts for launching batch and stream benchmarks

## Prerequisites

1. Install TeleFuser and start a batch video server.
2. Install AIPerf from the vendored clone under `benchmarks/aiperf`.

Example TeleFuser startup:

```bash
telefuser serve \
    examples/wan_video/wan21_14b_image_to_video_h100.py \
    --port 8000 \
    --task i2v
```

Example AIPerf install:

```bash
pip install -e ./benchmarks/aiperf
```

## Quick Start

### Batch Video

Run a small benchmark against TeleFuser `/v1/videos`:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh
```

This uses:
- `configs/video_generation_quick.yaml`
- file-backed prompts from `data/video_prompts.jsonl`
- TeleFuser server at `http://127.0.0.1:8000`

### Stream Serve

Start a TeleFuser stream server first:

```bash
telefuser stream-serve \
    examples/lingbot/stream_lingbot_world_fast.py \
    -p 8088 \
    --skip-validation
```

Then run the WebRTC benchmark:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

This uses:
- `configs/stream_lingbot_world_fast_quick.json`
- control trace from `data/stream_lingbot_controls.json`
- TeleFuser stream server at `http://127.0.0.1:8088`

If no explicit ICE allowlist is provided, `run_stream_bench.py` will auto-detect a non-loopback local IP for WebRTC host candidate gathering. Only pin `--ice-host-ip` or `TELEFUSER_WEBRTC_ICE_HOST_IPS` when you need a specific address.

The script records:
- SDP offer / answer RTT
- WebRTC connect latency
- first-frame latency
- steady-state stream FPS
- control acknowledge latency
- control-to-next-frame latency
- the helper raises the open-file limit to `8192` by default; override with `TELEFUSER_BENCH_NOFILE_LIMIT` if needed

## Config Notes

TeleFuser and AIPerf line up well for batch video serving:
- TeleFuser exposes `/v1/videos`, `/v1/videos/{id}`, and `/v1/videos/{id}/content`
- AIPerf `video_generation` already speaks the same async submit/poll/download flow

The provided configs intentionally:
- use `endpoint.type: video_generation`
- enable `downloadVideoContent` only in the full E2E variant
- point `serverMetrics.urls` at `/v1/service/metrics` only when metrics scraping is desired
- do not enable AIPerf's built-in readiness probe, because TeleFuser batch video serving does not expose a stable `/v1/models` or chat-style readiness path for `video_generation`

Instead, the helper script checks:

```bash
curl http://127.0.0.1:8000/v1/service/health
```

## Stream Scope Notes

The current stream benchmark is a TeleFuser-specific harness layered next to AIPerf, not inside AIPerf core yet.

Reasons:
- `telefuser stream-serve` is WebRTC session based, not plain HTTP request / response
- world-model streaming needs frame-level and control-loop metrics
- current `LingBotWorldFastService` only allows one active session at a time, so the harness supports concurrency parameters but this pipeline is effectively single-session today
- `python3 scripts/remote_bench_sync.py stream-bootstrap` will auto-detect the usable model zoo root and create a repo-local `model_zoo` symlink on the remote host, so the default relative path works without extra manual env setup
