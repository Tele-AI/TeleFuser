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

## Four-H100 LingBot-World v2 validation

Both runs below used four H100 80 GB GPUs, BF16 DiT, FP32 VAE, FlashAttention-4, disabled FSDP, disabled
`torch.compile`, `chunk_size=4`, and 16 FPS output. They validate different workloads and code revisions, so their
results should not be interpreted as a before-and-after performance comparison.

### Current four-H100 real-time gate

Commit `540b579` was validated on 2026-08-03 through the direct LingBot pipeline-service path with four H100 80 GB
GPUs and PyTorch 2.11.0+cu128. The request used 832x480 output, 77 frames, and five four-latent-frame chunks.

| Metric | Result |
|---|---:|
| Generated frames / target chunks | 77 / 5 |
| Steady chunks after excluding chunk 0 | 4 |
| Steady compute FPS | **17.1399** |
| Chunk compute mean / p50 / p90 / max | 0.9335 / 0.9409 / 0.9410 / 1.0058 s |
| First generated frame from measured session start | 3.2182 s |

This run clears the average target-side 16 FPS compute gate. It does not claim that every chunk clears the one-second
budget: the maximum was 1.0058 seconds. The synchronized compute interval includes condition handling, DiT,
clean-KV update, spatial VAE decode, GPU-to-CPU transfer, and frame conversion. It excludes model loading, runtime
creation, LiveKit pacing/encoding, network delivery, and client rendering.

The exact reproduction command is in the
[LingBot example guide](https://github.com/Tele-AI/TeleFuser/tree/main/examples/lingbot#validated-four-h100-real-time-gate)
and uses `tools/validation/benchmark_lingbot_world_v2_direct.py`.

### Current one-minute streaming replay

The one-minute workload was rerun on 2026-08-03 at TeleFuser commit
`284996dd616cfd44a55523687b7f2a63a281abb9`. It validates sustained target generation, bounded KV-cache capacity,
and the paced LiveKit delivery path on the current communication-optimized revision.

The run used the `stream_lingbot_world_v2_1min.json` workload and AIPerf 0.11.0 at commit
`e977ffbb1648510acec431b2a3fbd1a0f7bb8a35`. The 60-second request was truncated to 60 complete latent chunks:
957 generated frames representing 59.75 seconds of media. With `local_attn_size=18` and `sink_size=6`, the
240-latent-frame session reported a fixed 28,080-token KV capacity.

| Metric | Result |
|---|---:|
| Generated target frames / chunks | 957 / 60 |
| Steady frames / chunks after excluding chunk 0 | 944 / 59 |
| Steady target compute time / FPS | 58.2791 s / **16.1979** |
| Chunk compute mean / p50 / p90 / p99 / max | 0.9878 / 0.9593 / 1.0624 / 1.0932 / 1.1149 s |
| LiveKit stream FPS / client frames | 13.1967 / 803 |
| First client frame / session runtime | 6.0682 / 66.8948 s |
| Runtime creation | 1.4176 s |
| Artifact | `20260803_095518_62ec043c` |

The target completed all 60 chunks and cleared the average 16 FPS compute gate. It did not keep every chunk below one
second: p99 was 1.0932 seconds and the maximum was 1.1149 seconds. The lower client frame count belongs to the paced
delivery measurement and must not be conflated with target generation completeness.

## Reproducibility

Every result should retain the TeleFuser commit and AIPerf package version, model revision, accelerator model/count,
driver, CUDA, PyTorch, dtype, complete workload config, warmup rule, success/failure counts, and
offload/cache/attention settings.
Use the dated validation above as one-run evidence, not a universal performance guarantee. Ongoing comparisons belong
in GreptimeDB and replayable artifacts.
