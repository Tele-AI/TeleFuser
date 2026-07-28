# TeleFuser and AIPerf

TeleFuser exposes raw target-side facts; AIPerf owns workload execution, aggregation, resource collection, artifacts,
GreptimeDB history, and visualization. The checked-in integration covers batch video generation through the
OpenAI-compatible `/v1/videos` API and LingBot streaming through LiveKit.

AIPerf's stream runner and result schema are transport-neutral. The LiveKit adapter is maintained by
TeleFuser, loads from source at process startup, and produces AIPerf's standard session results. The contract records WebRTC as
the media transport and LiveKit as its provider, preserving the SFU topology without adding LiveKit code to AIPerf.

For installation, workload configs, launch commands, history setup, and focused tests, use the canonical
[`benchmarks/telefuser_aiperf/README.md`](https://github.com/Tele-AI/TeleFuser/tree/main/benchmarks/telefuser_aiperf#readme). AIPerf is installed from a pinned
Git commit with `pip`; no retained AIPerf checkout or adapter `pyproject.toml` is required.

## Quick start

From the TeleFuser repository root, install the streaming-capable AIPerf Git commit into its isolated environment:

```bash
bash scripts/setup_aiperf.sh
```

Start a local LiveKit development server in terminal 1:

```bash
livekit-server --dev --bind 127.0.0.1
```

Start the four-GPU LingBot-World v2 target in terminal 2, replacing the model path:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
    --livekit-url ws://127.0.0.1:7880 \
    --livekit-api-key devkey \
    --livekit-api-secret secret \
    --num-workers 1 \
    --worker-gpu-map 0,1,2,3 \
    --port 8088 \
    --skip-validation
```

Wait for `"ready":true`, `"workers_idle":1`, and `"workers_failed":0`:

```bash
curl --noproxy '*' --fail --silent --show-error \
  http://127.0.0.1:8088/v1/service/health
```

An idle service reports `"livekit_connected":false`; that is expected. Run the benchmark in terminal 3:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

The request contains 59.75 seconds of media, but the configured AIPerf active window is 240 seconds. Allow about four
minutes for the command to finish. Success is `Stream profile sessions: 1/1 succeeded`; reports are created under
`artifacts/telefuser_aiperf/stream_lingbot_v2_1min/`. See the canonical README for model file layout, manual Python
environment selection, history setup, and troubleshooting.

## Ownership and metric semantics

| Component | Owner | Responsibility |
|---|---|---|
| TeleFuser runtime | TeleFuser | Emit synchronized phase, chunk, runtime, cache, and environment facts |
| Batch target adapter | AIPerf | Convert `/v1/videos` HTTP events into the standard request timeline |
| LiveKit source adapter | TeleFuser | Convert room, track, status, metrics, and control events into session results |
| Aggregation and history | AIPerf | Apply warmup, percentiles, throughput, artifacts, GreptimeDB, and visualization |
| Contracts and workloads | TeleFuser | Fix target capabilities, inputs, settings, and reproducible launch commands |

Target facts follow these rules:

- Durations use a monotonic clock; cross-process samples also retain source UTC timestamps.
- CUDA phase boundaries synchronize the measured target device.
- Values are finite and non-negative; unavailable values are omitted or `null`, never fabricated as zero.
- Memory uses bytes in the raw protocol and is converted only for display.
- The target does not exclude warmup, calculate percentiles, or produce cross-run conclusions.

| Scope | Examples | Aggregation rule |
|---|---|---|
| Event | Frame or response arrival | Preserve the event timeline |
| Request/session | First output, session latency | Calculate independently for each request or session |
| Run | Success rate, throughput, percentiles | Aggregate after AIPerf excludes warmup |

Client delivery, target pipeline residence, target phase time, and resource utilization remain separate dimensions.
Fields without equivalent semantics remain private or unavailable instead of being forced into a common metric.

## Validated one-minute LingBot-World v2 replay

The `stream_lingbot_world_v2_1min.json` workload was validated on 2026-07-28 with four H100 80 GB GPUs, BF16 DiT,
FP32 VAE, SageAttention SM90, `torch.compile` enabled, FSDP disabled, `chunk_size=4`, and 16 FPS. A 60-second request
is truncated to 60 complete latent chunks: 957 output frames representing 59.75 seconds of media. LingBot-World v2
used its fixed `local_attn_size=18` and `sink_size=6` window, so the 240 latent-frame request retained a fixed
27,144-token KV capacity rather than a duration-sized global KV cache.

| Measurement | Result |
|---|---:|
| Successful sessions | 1 / 1 |
| Target chunks / generated frames | 60 / 957 |
| Client frames received | 947 |
| Target steady chunks / frames | 59 / 944 |
| Configured session runtime | 242.255 s |
| First-frame latency | 6,252.773 ms |
| Client delivery rate (`stream_fps`) | 9.397 FPS |
| Weighted steady chunk compute rate | 4.708 FPS |
| Chunk pipeline residence, mean / p99 | 3.399 / 3.901 s |

The target emitted all 957 frames and the LiveKit client received 947. AIPerf excluded the first target chunk from
steady-state compute aggregation, leaving 944 generated frames across 59 chunks. The 242-second session runtime is
the configured 240-second active-window limit plus connection overhead; it is not the generation time of the
59.75-second media payload.

Chunk pipeline residence spans actor admission through output and includes time shared by overlapping encode, DiT,
and decode work. It is therefore expected to exceed adjacent delivery intervals; the report's 4.708 weighted
steady chunk compute FPS must not be read as client stream throughput. The Git-installed AIPerf run exited with code
0. The replay artifact is
`artifacts/telefuser_aiperf/stream_lingbot_v2_1min/20260728_083948_fc4344ba/stream_report.html`.

## Reproducibility

Every result should retain the TeleFuser commit and AIPerf package version, model revision, accelerator model/count,
driver, CUDA, PyTorch, dtype, complete workload config, warmup rule, success/failure counts, and
offload/cache/attention settings.
Use the dated validation above as one-run evidence, not a universal performance guarantee. Ongoing comparisons belong
in GreptimeDB and replayable artifacts.
