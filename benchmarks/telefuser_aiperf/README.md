# TeleFuser AIPerf Integration

This directory contains the TeleFuser-owned benchmark assets for AIPerf. AIPerf itself is not
vendored here, and the LiveKit adapter is loaded directly from this source tree instead of being built or published as
a separate Python distribution.

Run every command in this guide from the TeleFuser repository root. The user-facing metric definitions and the
latest validated result are documented in
[`docs/en/benchmark_aiperf.md`](../../docs/en/benchmark_aiperf.md). This README is the canonical installation and
operation guide.

## Prerequisites

- A working TeleFuser installation with the LingBot-World v2 checkpoints.
- Four CUDA GPUs visible to TeleFuser. The validated configuration used four H100 80 GB GPUs.
- Python 3.10 or newer, Git, curl, and the `livekit-server` executable.

Install the local LiveKit development server if it is not already available:

```bash
curl -sSL https://get.livekit.io | bash
```

The development server and its default `devkey` / `secret` credentials are only for trusted local testing.

## Install AIPerf

The official AIPerf 0.11.0 wheel does not contain the streaming runner. Install the pinned `teleai` source commit
from GitHub. The recommended helper creates an isolated `.venv-aiperf` so AIPerf dependencies do not change the
TeleFuser runtime environment:

```bash
bash scripts/setup_aiperf.sh
```

Successful setup ends by printing the AIPerf version, the pinned VCS commit, and the TeleFuser adapter path. The
default commit is `e977ffbb1648510acec431b2a3fbd1a0f7bb8a35`.

To install into an existing benchmark environment instead, run:

```bash
python -m pip uninstall -y aiperf
python -m pip install \
  'aiperf @ git+https://github.com/ActivePeter/aiperf.git@e977ffbb1648510acec431b2a3fbd1a0f7bb8a35' \
  'livekit>=1.1.13,<2'
```

The uninstall is intentional: the official wheel and the streaming source currently both report version `0.11.0`,
so pip may otherwise keep the non-streaming wheel.

Pip uses a temporary Git checkout while building the package; no AIPerf checkout is retained in this repository. No
adapter `pyproject.toml`, wheel, or editable install is required. The stream launcher adds this directory to
`PYTHONPATH` before registering the `telefuser_livekit` adapter.

Use `AIPERF_ENV_DIR` to select another helper-created environment. If AIPerf was installed manually, set
`TELEFUSER_AIPERF_PYTHON` to its Python executable when running the stream launcher, for example:

```bash
TELEFUSER_AIPERF_PYTHON=/path/to/benchmark-env/bin/python \
  bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

## Batch Video

Start a compatible TeleFuser service, for example:

```bash
telefuser serve \
  examples/wan_video/wan21_14b_image_to_video_480p_service.py \
  --port 8000 \
  --task i2v
```

Run the smoke workload or the fixed Wan2.1 comparison:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_wan21_i2v_480p_compare.yaml
```

The launcher checks `/v1/service/health` before profiling. Its common overrides are `TELEFUSER_AIPERF_URL`,
`TELEFUSER_AIPERF_HEALTH_URL`, `TELEFUSER_BENCH_NOFILE_LIMIT`, and `AIPERF_BIN`.

Available batch configs:

| Config | Purpose |
|---|---|
| `configs/video_generation_quick.yaml` | Connectivity and latency smoke test |
| `configs/video_generation_e2e.yaml` | Warmup, trace, records, and target metrics |
| `configs/video_generation_rate.yaml` | Poisson-arrival load |
| `configs/video_generation_wan21_i2v_480p_compare.yaml` | Fixed Wan2.1 I2V comparison |

## LingBot-VLA v2 Structured Actions

Start the native VLA service from its isolated model environment, then run the AIPerf workload from the repository
root:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

The repository-owned `telefuser_vla_structured` endpoint and `telefuser_structured_http` transport submit
`POST /v1/tasks/structured`, poll `GET /v1/tasks/{task_id}/status`, and pass request latency, throughput, success,
trace, and server metric facts into AIPerf's normal warmup and aggregation pipeline. Defaults are two excluded warmup
requests followed by 20 measured requests at concurrency one. Override them without changing the checked-in config:

```bash
TELEFUSER_AIPERF_REQUESTS=100 \
TELEFUSER_AIPERF_CONCURRENCY=2 \
  bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

Each terminal result is required to contain a finite `50x55` action chunk and the frozen structured result fields.
The VLA adapter registers `vla_inference_time` and `vla_peak_memory` as AIPerf record metrics, so the normal AIPerf
artifact contains their per-request values and aggregated p50/p95/p99 summaries alongside request latency, throughput,
and success rate. It retains an action hash, bounds, dimensions, and verification status; it does not copy full action
arrays or Base64 cameras into AIPerf response records. This validates service execution and normalized action structure,
not physical robot control semantics.

For a target-process RSS and per-GPU process-memory artifact, pass the native service PID. This is an external bounded
sampler and does not change the TeleFuser service or any shared metric endpoint:

```bash
TELEFUSER_AIPERF_SERVICE_PID=<service-pid> \
  bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

The resource summary is written to
`artifacts/telefuser_aiperf/vla_structured/resource_summary.json` and includes sample count, RSS mean/p50/p95/p99,
and per-GPU process-memory mean/p50/p95/p99. It is intentionally separate from AIPerf server metrics because RSS is
an observer-side process-tree fact, while GPU telemetry remains target-side Prometheus data.

The checked-in AIPerf configuration defaults to two excluded warmup requests and 20 measured requests. For a longer
distribution, override the request count and concurrency without editing the configuration:

```bash
TELEFUSER_AIPERF_REQUESTS=100 \
TELEFUSER_AIPERF_CONCURRENCY=2 \
  bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

## VLA GPU Parallelism Assessment

The current VLA policy validates `world_size == 1`, so this assessment is deliberately read-only and does not enable
FSDP, tensor parallelism, or pipeline parallelism. It reports visible GPU capacity, current free memory, checkpoint
size, and how many complete replicas fit using a measured resident-memory estimate:

```bash
.venv-vla/bin/python tools/validation/inspect_lingbot_vla_v2_gpu_plan.py \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --replica-memory-mib 13302 \
  --output work_dirs/vla_gpu_plan.json
```

On the validated four-H100 host, the report showed four visible 80 GB GPUs and four estimated complete replicas at
the 13,302 MiB measured process-memory baseline. This supports request-level multi-replica service capacity; it is
not evidence that one model replica can be split across GPUs. Any future FSDP or tensor-parallel implementation must
be introduced behind the VLA pipeline boundary and re-run the frozen preprocessing, velocity, action, and HTTP
contract tests before it is considered equivalent.

## LingBot-World v2 Streaming

The v2 pipeline expects the following files below `TF_MODEL_ZOO_PATH`:

```text
Wan2.2-I2V-A14B/Wan2.1_VAE.pth
Wan2.2-I2V-A14B/models_t5_umt5-xxl-enc-bf16.pth
lingbot/lingbot-world-v2-14b-causal-fast/transformers/model-00001-of-00008.safetensors
...
lingbot/lingbot-world-v2-14b-causal-fast/transformers/model-00008-of-00008.safetensors
```

Use three terminals for the local benchmark. In terminal 1, start LiveKit:

```bash
livekit-server --dev --bind 127.0.0.1
```

Leave this process running. In terminal 2, start the four-GPU LingBot-World v2 service. Replace
`/path/to/model_zoo`; local LiveKit connections must not use host proxy variables:

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

Initial model loading and pipeline warmup can take several minutes. Do not start AIPerf until the service reports
`"ready":true`, `"workers_idle":1`, and `"workers_failed":0`:

```bash
curl --noproxy '*' --fail --silent --show-error \
  http://127.0.0.1:8088/v1/service/health
```

`"livekit_connected":false` is normal while no session is active; it does not mean the service is unhealthy.

In terminal 3, run the one-minute LingBot-World v2 workload:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

The v2 workload is the default TeleFuser stream workload. It requests 59.75 seconds of media using the model's fixed
attention window. The 240-second AIPerf active window is a timeout ceiling; a successful run exits after the target
emits its completion status and normally takes about 66 seconds after admission, excluding model loading.

A successful run prints `Stream profile sessions: 1/1 succeeded`, an artifact directory, and an HTML report path.
Results are written below:

```text
artifacts/telefuser_aiperf/stream_lingbot_v2_1min/<run-id>/
```

The service command clears common proxy variables for the local LiveKit connection. The adapter also bypasses proxy
variables when the target returns a loopback LiveKit URL; remote LiveKit deployments are unchanged.

The contract records `transport: webrtc` and `transport_provider: livekit`. The adapter creates the TeleFuser
session, joins its room, receives native video tracks, sends reliable controls on `tf.control`, and consumes
`tf.status` and bounded `tf.metrics` messages. It produces AIPerf's standard session results without requiring any
LiveKit-specific changes in AIPerf.

## SGLang LingBot-World v2, Four GPUs

This target uses SGLang's native MessagePack WebSocket endpoint and does not start LiveKit. It requires an SGLang
installation containing the `/v1/realtime_video/generate` endpoint and
`LingBotWorldCausalDMDPipeline`. In terminal 1, launch the four-GPU server:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_sglang_lingbot_world_v2_4gpu.sh
```

The launcher defaults to GPUs `0,1,2,3`, port `30000`, and model
`robbyant/lingbot-world-v2-14b-causal-fast-diffusers`. Explicit `SGLANG_SOURCE_DIR` and `SGLANG_PYTHON` values take
precedence over an installed `sglang` command. Override the defaults when needed:

```bash
SGLANG_BIN=/path/to/sglang \
SGLANG_PYTHON=/path/to/sglang-env/bin/python \
SGLANG_SOURCE_DIR=/path/to/sglang-source \
SGLANG_MODEL_PATH=/path/to/lingbot-world-v2-14b-causal-fast-diffusers \
SGLANG_CUDA_VISIBLE_DEVICES=4,5,6,7 \
  bash benchmarks/telefuser_aiperf/scripts/run_sglang_lingbot_world_v2_4gpu.sh
```

Wait until `http://127.0.0.1:30000/health` succeeds, then run AIPerf in terminal 2:

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_sglang_lingbot_world_v2_4gpu_1min.json
```

Artifacts are written below
`artifacts/telefuser_aiperf/stream_sglang_lingbot_v2_4gpu_1min/`. The adapter counts combined and split SGLang frame
batches, converts the shared keyboard trace to `camera_actions` state events, and maps scheduler, WebP encoding,
pacing, and WebSocket write timings into AIPerf's standard stream result.

The launcher intentionally passes `--flow-shift 10`, matching the official LingBot-World v2 implementation and the
TeleFuser workload. The SGLang source default is `5`; use `SGLANG_FLOW_SHIFT=5` only to benchmark SGLang's default
behavior, and do not compare that run as a numerically equivalent model configuration. The workload also fixes four
DMD steps, 16 FPS, 60 chunks, a KV window of 18 latent frames plus a six-frame sink, WebP quality 95, and no output
pacing. SGLang permits one active realtime generation session, so the contract fixes concurrency to one.

## Troubleshooting

- `The pinned streaming-capable AIPerf or LiveKit is not installed`: rerun `bash scripts/setup_aiperf.sh`.
- `SGLang is not installed or SGLANG_BIN is invalid`: activate the SGLang environment or set `SGLANG_BIN` to its
  executable.
- SGLang connection refused on port 30000: wait for model loading to finish and check `/health`.
- Connection refused on port 8088: the TeleFuser process is still warming up or has exited; inspect terminal 2.
- `0/1 succeeded` with zero received frames: confirm LiveKit is still running, restart the TeleFuser service, wait for
  one idle worker, and rerun the benchmark.
- LiveKit connection errors on localhost: start TeleFuser with the proxy variables removed exactly as shown above.
- `aiperf_commit` is `unknown` inside a source-install artifact: use the commit printed by `setup_aiperf.sh`; AIPerf
  only embeds `_build_info.py` in its CI-built wheels.

## History And Resources

AIPerf history and active resource collection require GreptimeDB. Start persistent storage, then the AIPerf history
service:

```bash
docker volume create aiperf-greptime-data
docker run -d --name aiperf-greptime --restart unless-stopped \
  -p 127.0.0.1:4000:4000 \
  -v aiperf-greptime-data:/greptimedb_data \
  greptime/greptimedb:latest \
  standalone start \
  --http-addr 0.0.0.0:4000 \
  --data-home /greptimedb_data

.venv-aiperf/bin/aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --host 127.0.0.1 \
  --port 8095
```

Set `AIPERF_HISTORY_URL` and `AIPERF_RESOURCE_TARGET_PID` before the batch launcher to collect the target process
tree. History failures do not silently fall back to an in-memory or file-only database.

## Layout And Verification

```text
configs/             Reproducible batch and streaming workloads
data/                Prompt and control inputs
scripts/             Batch and streaming launchers
telefuser_aiperf/    Source-loaded LiveKit and SGLang realtime adapters
tests/               Adapter tests
*_contract.yaml      Target and transport capability contracts
```

Runtime use does not require pytest. To run the optional adapter checks, install the test-only dependencies into the
AIPerf environment first, then run the checks from the repository root:

```bash
.venv-aiperf/bin/python -m pip install \
  'pytest>=7' \
  'pytest-asyncio>=0.21'

PYTHONPATH=benchmarks/telefuser_aiperf \
  .venv-aiperf/bin/python -m pytest \
  benchmarks/telefuser_aiperf/tests/test_livekit_adapter.py \
  benchmarks/telefuser_aiperf/tests/test_sglang_adapter.py \
  benchmarks/telefuser_aiperf/tests/test_vla_structured.py

bash -n \
  scripts/setup_aiperf.sh \
  benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/scripts/run_sglang_lingbot_world_v2_4gpu.sh \
  benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```
