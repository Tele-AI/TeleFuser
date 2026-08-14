# LingBot-VLA v2

This example loads the official LingBot-VLA v2 6B base checkpoint through TeleFuser and returns its normalized
55-dimensional canonical action chunk. The RobotWin profile is used only to prepare the example observation; the
result is not converted to physical RobotWin actions.

The native HTTP and full-project CI evidence for TeleFuser commit `baf3d18` is recorded in
[VALIDATION_BAF3D18.md](VALIDATION_BAF3D18.md).

## Model Directory

The examples use the existing model-zoo layout:

```text
${TF_MODEL_ZOO_PATH}/
  lingbot/lingbot-vla-v2-6b/
  Qwen3-VL-4B-Instruct/
```

Set the model root before running an example or service:

```bash
export TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo
```

## Feature Support

| Feature | Support |
| --- | --- |
| Official 6B base checkpoint | Supported |
| Public loader and RobotWin preprocessing | Supported |
| Canonical action output (`50 x 55`) | Supported |
| Strict upstream parity | Frozen 38-tensor baseline |
| Native structured HTTP service | Supported |
| AIPerf structured workload | Supported |
| Request-level replicas | Supported |
| Single-policy FSDP/TP/PP | Not enabled |
| Physical robot action mapping and safety control | Not included |

Request-level replicas use one GPU per policy copy; this is not tensor or pipeline parallelism inside one policy.

## Files

| File | Purpose |
| --- | --- |
| `lingbot_vla_v2_inference.py` | Direct in-process inference |
| `lingbot_vla_v2_native_service.py` | Native TeleFuser structured service contract |
| `../../telefuser/pipelines/lingbot_vla_v2/` | Pipeline, preprocessing, policy, and service adapter |
| `../../tools/validation/` | Parity, runtime, service, and fault validators |
| `VALIDATION_BAF3D18.md` | Native HTTP and full-project CI evidence |

Generated captures and benchmark artifacts belong under `work_dirs/` and are not committed.

## Validated H100 Development Environment

The strict upstream parity, runtime comparison, and native structured-service validation used the environment below.
TeleFuser supports broader versions through its normal dependency ranges, but reproduce VLA parity and performance
results with these versions before attributing a difference to code changes.

| Component | Validated value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (SM90) |
| NVIDIA driver | `590.48.01` |
| Python | `3.10.12` |
| PyTorch | `2.11.0+cu130` |
| TorchVision | `0.26.0+cu130` |
| TorchAudio | `2.11.0+cu130` |
| PyTorch CUDA runtime | `13.0` |
| Transformers | `4.57.3` |
| Triton | `3.6.0` |

Create the VLA environment inside the repository. Install the CUDA 13.0 PyTorch wheels before TeleFuser so dependency
resolution retains the validated PyTorch/CUDA ABI. Obtain the three exact wheels from the CUDA 13.0 PyTorch wheel
index or artifact repository used by the deployment:

```bash
python3.10 -m venv .venv-vla
source .venv-vla/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Replace /path/to/cu130-wheels with the deployment's CUDA 13.0 wheel directory.
python -m pip install --no-index --find-links /path/to/cu130-wheels \
  "torch==2.11.0+cu130" \
  "torchvision==0.26.0+cu130" \
  "torchaudio==2.11.0+cu130"

python -m pip install -e ".[dev]"
```

The `.venv-vla` directory is ignored by Git and does not modify the system or Conda base environment. The commands
below use its interpreter explicitly, so activating it is optional after installation. Verify the runtime before
loading the checkpoint or comparing benchmark results:

```bash
.venv-vla/bin/python - <<'PY'
import importlib.metadata as metadata

import torch
import transformers

print("PyTorch:", torch.__version__)
print("TorchVision:", metadata.version("torchvision"))
print("TorchAudio:", metadata.version("torchaudio"))
print("PyTorch CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Triton:", metadata.version("triton"))
print("GPU:", torch.cuda.get_device_name(0))

assert torch.__version__ == "2.11.0+cu130"
assert metadata.version("torchvision") == "0.26.0+cu130"
assert metadata.version("torchaudio") == "2.11.0+cu130"
assert torch.version.cuda == "13.0"
assert transformers.__version__ == "4.57.3"
assert metadata.version("triton") == "3.6.0"
assert torch.cuda.is_available()
PY
```

## Inputs

- Three RGB cameras in the upstream RobotWin order: high, left wrist, right wrist.
- A raw 14-dimensional RobotWin state.
- A non-empty task string.

The SDK applies the bundled upstream RobotWin `bounds_99_woclip` statistics and maps the observation into
LingBot's 55-dimensional canonical state.

## Output

The pipeline returns `LingBotVlaV2CanonicalActionChunk` with:

- `canonical_normalized_actions`: `[H, 55]` base-model output.
- `horizon`: action chunk length, normally 50 for the official base config.
- `action_dim`: canonical action dimension, normally 55.
- `checkpoint_variant`: `base`.
- `policy_verified=False` and `verification_status="unverified_official_6b_base"`.

## Checkpoints

The VLA directory must contain `model.safetensors.index.json` and every referenced shard. The Qwen3-VL directory
supplies the visual-language backbone configuration and processor.

## Direct Inference

```bash
.venv-vla/bin/python examples/lingbot_vla_v2/lingbot_vla_v2_inference.py \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' \
  --output canonical_action_chunk.npz
```

The example saves canonical actions and checkpoint metadata in an `.npz` file. The base output must not be sent to
a robot without an embodiment-specific post-training checkpoint, action mapping, and policy validation.

## Native TeleFuser Service

The native service uses the shared `PIPELINE_CONTRACT`, asynchronous task scheduler, pipeline pool, status API, runtime
metrics, and `TFClient`.

The example resolves checkpoints under the existing `TF_MODEL_ZOO_PATH` layout:

- `lingbot/lingbot-vla-v2-6b`
- `Qwen3-VL-4B-Instruct`

Start one replica on one visible GPU:

```bash
TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action \
  --parallelism 1 \
  --host 127.0.0.1 \
  --port 18080
```

Submit `POST /v1/tasks/structured` with `task="vla_action"`, an `instruction`, the 14-dimensional `state`, and
the three Base64 camera fields. The creation response contains a task ID. Poll
`GET /v1/tasks/{task_id}/status`; a completed response contains the action payload under `result` and includes
`inference_time_s` and the optional `peak_memory_mb`.

The unified client handles image encoding, submission, polling, and result extraction:

```python
from telefuser.client import TFClient

client = TFClient("http://127.0.0.1:18080")
actions = client.predict_vla_actions(
    instruction="pick up the red block",
    state=[0.0] * 14,
    camera_high_path="/data/cam_high.png",
    camera_left_wrist_path="/data/cam_left_wrist.png",
    camera_right_wrist_path="/data/cam_right_wrist.png",
    seed=7,
)
print(actions["horizon"], actions["action_dim"])
```

For independent replicas, expose one GPU per replica through the existing pipeline pool:

```bash
CUDA_VISIBLE_DEVICES=0,1 TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action \
  --parallelism 2 \
  --num-replicas 2 \
  --port 18080
```

This is request-level replication, not tensor parallelism inside one policy replica. The response remains a normalized
base-model canonical action chunk and must not be treated as a physical robot command.

## Single-GPU Service Benchmark

Use the VLA-specific benchmark to measure checkpoint construction, first-request latency, steady-state latency,
sequential throughput, process RSS, CUDA allocator peaks, and source-image-size overhead. The pipeline always converts
the three source images to the official `256x256` model input, so source size affects boundary and preprocessing cost,
not the model token shape.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vla/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_service.py \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --image examples/data/lingbot_world_fast/image.jpg \
  --image-sizes 256x256,640x480,1280x720 \
  --warmup 1 \
  --runs 20 \
  --output work_dirs/vla_service_benchmark/report.json
```

The native service moves the policy to its target GPU and runs one synthetic fixed-shape warmup before readiness. It
also keeps the allocator cache between requests. The report records construction and startup warmup separately, while
the first accepted request represents a ready replica. The default `service-thread` execution mode matches the native
service runner's fixed worker thread; use `--execution-mode direct` only to measure the in-process pipeline ceiling.
Shutdown still offloads the policy explicitly.

## Native Structured API Validation

Use the VLA-specific HTTP validator after the native service reports ready. This is the structured-output counterpart
to the model-specific direct and AIPerf workloads used by the video and LingBot-World integrations: it exercises the
real TeleFuser HTTP boundary, asynchronous scheduler, task status polling, pipeline pool, and result serialization.
It emits raw request facts and aggregate latency distributions to a JSON artifact; it does not add a VLA-specific
service interface or change shared metric semantics.

Run a single-replica smoke and latency check:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg \
  --warmup 1 \
  --requests 20 \
  --concurrency 1 \
  --output work_dirs/vla_service_validation/smoke_20.json
```

When the target was started with two independent replicas, validate request-level concurrency with:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg \
  --warmup 2 \
  --requests 100 \
  --concurrency 2 \
  --output work_dirs/vla_service_validation/two_replica_100.json
```

Use duration mode for a bounded soak. Workers use closed-loop scheduling: each worker submits its next request only
after its previous task reaches a terminal state.

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --duration-seconds 7200 \
  --concurrency 1 \
  --service-pid <telefuser-parent-pid> \
  --gpu-indexes 0 \
  --resource-interval-seconds 1 \
  --output work_dirs/vla_service_validation/soak_2h.json
```

Resource sampling is opt-in and local-only. `--service-pid` must identify the parent `telefuser serve` process; its
replica descendants are discovered on every sample. RSS is summed across that process tree, while `nvidia-smi`
process memory is grouped by physical GPU index. For a two-replica service on physical GPUs 0 and 1, pass
`--gpu-indexes 0,1`. Omitting `--service-pid` keeps remote-service validation lightweight and does not invoke
`nvidia-smi`. Reports retain bounded raw samples plus distributions and first/last 10% trends for latency, RSS, and
per-GPU process memory.

The validator freezes the current structured contract. Requests contain exactly `task`, `instruction`, `state`, the
three camera fields, and optional `seed`. Action results contain exactly `canonical_normalized_actions`, `horizon`,
`action_dim`, `checkpoint_variant`, `policy_verified`, and `verification_status`. Safe additive task-status metadata
remains allowed, but status responses must not echo the three Base64 camera fields.

The command exits nonzero if readiness or contract checks fail, any measured request fails, task IDs are duplicated,
or the queue is not drained at the end. Each successful record validates the expected `50x55` finite action tensor
and retains only statistics and a float64 action fingerprint. Full actions and Base64 camera contents are deliberately
excluded from the artifact. `--max-records` bounds retained per-request samples during long runs while aggregate
latency and success counters still cover the complete run. `--max-resource-samples` independently bounds retained
resource samples.

For fault handling, run the independent validator against a ready service:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_service_faults.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg
```

It checks missing cameras, invalid state size, invalid Base64, and cancellation. Replica termination is opt-in and
requires a disposable two-replica service: add `--service-pid <telefuser-parent-pid>` and
`--kill-replica-gpu-index <physical-index>`. The tool only selects a GPU compute process inside that parent process
tree, sends `SIGTERM`, and verifies one-replica capacity degradation plus a subsequent valid `50x55` response. It does
not promise automatic replica restart.

The same structured API is available through the repository-owned AIPerf workload. Install the pinned isolated
AIPerf environment once, then run the workload while the native service is ready:

```bash
bash scripts/setup_aiperf.sh
bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

AIPerf excludes the configured warmup, aggregates request latency, throughput, success, traces, and server metrics,
and writes normal AIPerf artifacts. The adapter strictly validates the action contract but retains only bounded action
facts, not full arrays or Base64 inputs. Passing either validator proves serving and normalized action structure, not
embodiment-specific control semantics.

## TeleFuser Regression Baseline

The validation capture runs through the public loader and pipeline, then records preprocessing tensors, fixed initial
noise, every flow-matching `x_t` and velocity step, and the final canonical action. Run it twice before changing VLA
model code to establish and verify a strict local baseline:

```bash
.venv-vla/bin/python tools/validation/capture_lingbot_vla_v2_telefuser.py \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]' \
  --seed 7 \
  --deterministic-moe \
  --output work_dirs/vla_regression/baseline_seed7.npz

# Repeat the same command with:
#   --output work_dirs/vla_regression/replay_seed7.npz

.venv-vla/bin/python tools/validation/run_lingbot_vla_v2_parity.py \
  --reference work_dirs/vla_regression/baseline_seed7.npz \
  --candidate work_dirs/vla_regression/replay_seed7.npz \
  --profile strict \
  --output work_dirs/vla_regression/strict_report.json
```

Each `.npz` has a same-name `.json` sidecar containing the checkpoint, processor, input, runtime, and tensor contract
metadata. The default checkpoint identity is a fast filename-and-size manifest. Add `--full-checkpoint-hash` when a
content hash of every checkpoint shard is required. Keep generated artifacts under `work_dirs`; do not commit them.

This is a TeleFuser regression check, not upstream parity. It detects changes to the current implementation but does
not establish equivalence with the official repository.

## Official Upstream Parity

The strict upstream baseline pins `Robbyant/lingbot-vla-v2` at commit
`be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`. Keep the checkout, uv environment, cache, and artifacts under
`work_dirs`; Git ignores them. Create the isolated runtime with:

```bash
mkdir -p work_dirs/.uv-cache-upstream work_dirs/.uv-tmp-upstream
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" uv venv work_dirs/.venv-lingbot-upstream --python .venv-vla/bin/python
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" uv pip install --python work_dirs/.venv-lingbot-upstream/bin/python -r tools/validation/requirements-lingbot-vla-v2-upstream.txt
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" uv pip install --python work_dirs/.venv-lingbot-upstream/bin/python --no-deps "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
git clone https://github.com/Robbyant/lingbot-vla-v2 work_dirs/lingbot-vla-v2-upstream
git -C work_dirs/lingbot-vla-v2-upstream checkout be27333c9b5f2663b0ec33f069dd7dfd67fa32b5
```

Generate the reference with `capture_lingbot_vla_v2_upstream.py` in the upstream uv environment and the candidate
with `capture_lingbot_vla_v2_telefuser.py` in `.venv-vla`. Pass identical model, processor, camera, task, state, seed,
and device arguments to both commands, add `--deterministic-moe`, and pass `--upstream-root` to the upstream command.
Then compare them with the strict comparator shown above. Generated artifacts belong in `work_dirs/vla_upstream_parity`.

This is a minimal inference-parity runtime, not a LeRobot training environment. The upstream setup itself combines
LeRobot 0.4.2 metadata constraints with versions outside those constraints, so LeRobot is installed with `--no-deps`;
the capture import and end-to-end run are the runtime checks.

The official code hard-codes FlashAttention during construction. The upstream capture replaces that selection only
inside its validation process so both sides use eager attention on the Python 3.10.12 / PyTorch 2.11 stack. Production
inference keeps the upstream Triton MoE path through `telefuser.ops`; strict capture uses `--deterministic-moe` because
the upstream kernel uses atomic accumulation and is not bitwise repeatable across separate processes. Artifact metadata
records both `attention_backend` and `moe_backend`, and the comparator rejects mixed-backend artifacts.

## Upstream vs TeleFuser Runtime

Use the no-capture runtime benchmark after strict parity has passed. Unlike the parity capture, this benchmark does
not install layer hooks or copy every intermediate tensor to CPU. Both implementations consume the same frozen
preprocessed tensors and use the same initial noise, checkpoint paths, device type, software versions, attention
backend, and MoE backend. The comparator rejects reports when any of these conditions differ.

Run the official checkout and TeleFuser sequentially on the same otherwise-idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 work_dirs/.venv-lingbot-upstream/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_runtime.py \
  --implementation upstream \
  --upstream-root work_dirs/lingbot-vla-v2-upstream \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --input-artifact work_dirs/vla_upstream_parity/upstream_seed7.npz \
  --seed 7 --device cuda:0 --warmup 3 --runs 20 \
  --output work_dirs/vla_runtime_comparison/upstream_h100_runs20.json

CUDA_VISIBLE_DEVICES=0 .venv-vla/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_runtime.py \
  --implementation telefuser \
  --model-root /hhb-data/aigc/model_zoo/lingbot/lingbot-vla-v2-6b \
  --qwen3vl-root /hhb-data/aigc/model_zoo/Qwen3-VL-4B-Instruct \
  --input-artifact work_dirs/vla_upstream_parity/upstream_seed7.npz \
  --seed 7 --device cuda:0 --warmup 3 --runs 20 \
  --output work_dirs/vla_runtime_comparison/telefuser_h100_runs20.json

.venv-vla/bin/python tools/validation/compare_lingbot_vla_v2_runtime_benchmarks.py \
  --upstream work_dirs/vla_runtime_comparison/upstream_h100_runs20.json \
  --telefuser work_dirs/vla_runtime_comparison/telefuser_h100_runs20.json \
  --output-json work_dirs/vla_runtime_comparison/upstream_vs_telefuser_h100_runs20.json \
  --output-markdown work_dirs/vla_runtime_comparison/upstream_vs_telefuser_h100_runs20.md
```

The following controlled run used upstream commit
`be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`, TeleFuser model source commit
`86278d4a22d35f7cd8606dddd80ae3a4637e396c`, one NVIDIA H100 80GB HBM3, Python 3.10.12,
PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 4.57.3, eager attention, and the upstream Robby Triton MoE kernel.
It used three warmup requests and 20 measured requests per scope.

| Scope | Metric | Upstream | TeleFuser | TeleFuser change |
|---|---:|---:|---:|---:|
| Core model | mean | 669.382 ms | 660.100 ms | -1.39% |
| Core model | p50 | 668.373 ms | 657.364 ms | -1.65% |
| Core model | p95 | 677.683 ms | 669.343 ms | -1.23% |
| Core model | p99 | 678.620 ms | 685.735 ms | +1.05% |
| Runtime request | mean | 662.462 ms | 658.935 ms | -0.53% |
| Runtime request | p50 | 661.707 ms | 656.974 ms | -0.72% |
| Runtime request | p95 | 666.779 ms | 678.023 ms | +1.69% |
| Runtime request | p99 | 669.705 ms | 682.456 ms | +1.90% |

Negative change means TeleFuser is faster. `Core model` measures device-resident `sample_actions` with fixed inputs
and noise. `Runtime request` additionally includes CPU-to-GPU tensor transfer, seeded noise construction, output
validation, and CPU action delivery; image decoding and preprocessing are excluded equally on both sides. Peak CUDA
allocated memory was 12,454.8 MiB for both implementations.

The mean results differ by less than 1.5%, so this run shows no material TeleFuser inference overhead under the
matched model boundary. The slightly higher TeleFuser p99 is within a small 20-sample run and should be tracked with
more repetitions before drawing a tail-latency conclusion. Model loading time is recorded but not compared because
the official and TeleFuser loaders construct processors and framework objects at different boundaries. This runtime
benchmark does not replace the strict 38-tensor upstream parity result, include HTTP scheduling, or prove physical
robot control semantics. Keep the generated reports under `work_dirs`; do not commit them.
