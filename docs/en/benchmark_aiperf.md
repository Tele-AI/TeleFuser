# TeleFuser and AIPerf

TeleFuser exposes raw target-side facts; AIPerf owns workload execution, aggregation, resource collection, artifacts,
GreptimeDB history, and visualization. The checked-in integration covers batch video generation through the
OpenAI-compatible `/v1/videos` API, TeleFuser LingBot streaming through LiveKit, and SGLang LingBot streaming through
its native realtime WebSocket endpoint.

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

The request contains 59.75 seconds of media. The 240-second active window is a timeout ceiling; a successful run
exits after the target completion status and normally takes about 66 seconds after admission. Success is
`Stream profile sessions: 1/1 succeeded`; reports are created under
`artifacts/telefuser_aiperf/stream_lingbot_v2_1min/`.

To profile SGLang on the same four-GPU workload instead, start its server and select the SGLang config:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_sglang_lingbot_world_v2_4gpu.sh

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_sglang_lingbot_world_v2_4gpu_1min.json
```

This launch explicitly uses `--flow-shift 10` for parity with the official model and TeleFuser. SGLang's source
default is `5`, so runs made with `SGLANG_FLOW_SHIFT=5` describe SGLang's default but are not model-configuration
parity comparisons. See the benchmark README for model, GPU, port, and executable overrides.

## Ownership and metric semantics

| Component | Owner | Responsibility |
|---|---|---|
| TeleFuser runtime | TeleFuser | Emit synchronized phase, chunk, runtime, cache, and environment facts |
| Batch target adapter | AIPerf | Convert `/v1/videos` HTTP events into the standard request timeline |
| LiveKit source adapter | TeleFuser | Convert room, track, status, metrics, and control events into session results |
| SGLang source adapter | TeleFuser | Convert MessagePack frames, chunk timings, and camera events into session results |
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

Commit `663c385b179012c5c3de613212d10e8e6eac5f5d` was validated on 2026-08-02 with the
`stream_lingbot_world_v2_1min.json` workload, AIPerf 0.11.0 at commit
`e977ffbb1648510acec431b2a3fbd1a0f7bb8a35`, and four H100 80 GB GPUs. The current H100 example used BF16 DiT,
FP32 VAE, FlashAttention-4, disabled `torch.compile`, disabled FSDP, `chunk_size=4`, and 16 FPS. The 60-second
request was truncated to 60 complete latent chunks: 957 generated frames representing 59.75 seconds of media.
LingBot-World v2 used `local_attn_size=18` and `sink_size=6`; the session reported a fixed 28,080-token KV capacity
for its 240 latent frames.

| Runtime / target | Compute FPS | Mean / p99 chunk | Stream FPS | Client frames | Artifact |
|---|---:|---:|---:|---:|---|
| TeleFuser `.venv`, torch cu128 | 16.191 | 0.988 / 1.099 s | 12.697 | 756 | `20260802_084922_d7ae0931` |
| TeleFuser `.venv-sglang`, torch cu130 | 15.897 | 1.006 / 1.208 s | 14.089 | 871 | `20260802_090301_af6c433c` |
| SGLang `.venv-sglang`, torch cu130 | 16.617 | 0.963 / 0.974 s | 16.772 | 957 | `20260801_104829_2320fd7f` |

Every row completed 60 target chunks and generated 957 frames. AIPerf excluded only target chunk 0, leaving 944
frames across 59 chunks. The aligned TeleFuser run used 59.381647 seconds of synchronized compute time and was 4.33%
below the SGLang compute rate. The cu130 TeleFuser result was 1.81% below its cu128 run, so the environment change is
reported separately and is not counted as an optimization gain. The aligned TeleFuser report is
`artifacts/telefuser_aiperf/stream_lingbot_v2_1min/20260802_090301_af6c433c/stream_report.html`.

`stream_fps` is not used for the compute comparison. TeleFuser published LiveKit video with real-time 16 FPS pacing;
its aligned run averaged 18.99 ms from decoded-ready to publish start, 941.66 ms in paced publication, and 2.10 ms
from publish completion to client metadata. SGLang used unpaced burst WebSocket output. Those delivery semantics are
not equivalent even though both include network and client decoding. TeleFuser's 9.740-second first-frame latency
comprised 0.630 seconds to create the session, another 1.979 seconds to connect, 3.206 seconds from connection to
admission, and 3.925 seconds from admission to the first client frame; runtime creation occupied 1.564 seconds of the
last interval.

## Reproducibility

Every result should retain the TeleFuser commit and AIPerf package version, model revision, accelerator model/count,
driver, CUDA, PyTorch, dtype, complete workload config, warmup rule, success/failure counts, and
offload/cache/attention settings.
Use the dated validation above as one-run evidence, not a universal performance guarantee. Ongoing comparisons belong
in GreptimeDB and replayable artifacts.
