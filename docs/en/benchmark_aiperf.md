# TeleFuser and AIPerf

TeleFuser exposes raw target-side facts; AIPerf owns workload execution, aggregation, resource collection, artifacts,
GreptimeDB history, and visualization. The checked-in integration covers batch video generation through the
OpenAI-compatible `/v1/videos` API, LingBot-VLA structured actions through native HTTP task polling, TeleFuser LingBot
streaming through LiveKit, and SGLang LingBot streaming through its native realtime WebSocket endpoint.

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
| VLA structured adapter | TeleFuser | Validate action results and convert native submit/poll events into bounded AIPerf records |
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

The one-minute workload was rerun on 2026-08-06 with the current source tree, four H100 80 GB GPUs, Python
3.11.13, PyTorch 2.11.0+cu128, CUDA 12.8, BF16 DiT, FP32 VAE, FlashAttention-4, disabled FSDP, and disabled
`torch.compile`. The source tree includes the tagged Q/K/V Copy Engine Ulysses path.

The run used the `stream_lingbot_world_v2_1min.json` workload and AIPerf 0.11.0 at commit
`e977ffbb1648510acec431b2a3fbd1a0f7bb8a35`. Its `delivery_mode=lossless` request uses FIFO backpressure and
keeps the LiveKit video track open until the AIPerf client confirms the sender declared frame count. The 60-second
request generated 957 frames across 60 complete latent chunks, representing 59.75 seconds of media. The steady
summary excludes the first 13-frame chunk; the 240-latent-frame session reported a fixed 28,080-token KV capacity
with `local_attn_size=18` and `sink_size=6`.

| Metric | Result |
|---|---:|
| Successful sessions | 1 / 1 |
| Generated target frames / chunks | 957 / 60 |
| Steady frames / chunks after excluding chunk 0 | 944 / 59 |
| Steady target compute time / FPS | 53.6901 s / **17.5824** |
| Chunk compute mean / p50 / p90 / p99 / max | 0.9100 / 0.9088 / 0.9177 / 0.9751 / 1.0549 s |
| LiveKit declared / decoded client frames | 958 / 961 |
| Client callback FPS after first frame | 15.2313 |
| First client frame / session runtime | 5.9822 / 79.2523 s |
| Runtime creation | 1.4279 s |
| Artifact | `20260806_132517_46419f8f` |

The target completed all 60 chunks and clears the average 16 FPS compute gate. Lossless delivery completed only
after the client confirmed all 958 declared frames. The three additional client decoder callbacks come from the
LiveKit startup track and must not be interpreted as generated model frames.


## Reproducibility

Every result should retain the TeleFuser commit and AIPerf package version, model revision, accelerator model/count,
driver, CUDA, PyTorch, dtype, complete workload config, warmup rule, success/failure counts, and
offload/cache/attention settings.
Use the dated validation above as one-run evidence, not a universal performance guarantee. Ongoing comparisons belong
in GreptimeDB and replayable artifacts.

## TeleFuser performance baselines

The AIPerf result remains the source for workload execution and warmup exclusion. TeleFuser adds a small, versioned
baseline file so CI and local runs can compare the same fields across T2I, T2V, VLA, and real-time streaming:

- first-frame latency (`first_frame_latency_ms.p95`)
- chunk latency (`p50`, `p95`, and `p99`, in milliseconds)
- steady FPS (`steady_fps.p50`)
- GPU peak memory (`gpu_peak_memory_mb.maximum`)
- multi-request throughput (`throughput_per_second.p50`)
- request success rate
- output consistency (exact-match rate, maximum absolute error, and minimum cosine similarity)

The implementation is dependency-free and lives in `telefuser.metrics.baseline`. It consumes normalized AIPerf records,
accepts the existing stream aliases (`stream_fps`, `chunk_measurements`, and `first_frame_latency_s`), and never runs
GPU code while creating or comparing a baseline. A normalized record file can be summarized with:

Adapters that hold a reference output can produce the consistency fields with
`compare_numeric_outputs()` or `compare_output_hashes()` before constructing a `BenchmarkRecord`; full output arrays
are not written to the baseline artifact.

```bash
python -m telefuser.metrics.baseline summarize \
  --input artifacts/telefuser_aiperf/run/records.json \
  --output artifacts/telefuser_aiperf/run/baseline.json \
  --project-commit "$(git rev-parse HEAD)"
```

Compare a candidate run with the checked-in or downloaded baseline. The command exits with status 1 when a regression
is detected, which makes it suitable for a CI gate:

```bash
python -m telefuser.metrics.baseline compare \
  --baseline baselines/lingbot_world_v2.json \
  --candidate artifacts/telefuser_aiperf/run/baseline.json \
  --output artifacts/telefuser_aiperf/run/regression.json
```

The default relative tolerance is 5%. Latency and memory regress when they increase; FPS and throughput regress when
they decrease. Output consistency is checked independently and requires exact matches when the adapter provides an
exact-match result, cosine similarity at least `0.9999`, and maximum absolute error at most `1e-4`. Missing metrics or
workloads fail the comparison instead of being treated as zero. GPU collection and workload execution stay in the
existing AIPerf adapters, so this layer can be unit-tested on CPU.
