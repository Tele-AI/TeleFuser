# LingBot-VLA v2 Examples

This example runs the official LingBot-VLA v2 6B base checkpoint through TeleFuser. It accepts a RobotWin
observation and returns a normalized `50 x 55` canonical action chunk through direct Python inference or the native
structured service.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LingBot-VLA v2 6B base | [robbyant/lingbot-vla-v2-6b](https://huggingface.co/robbyant/lingbot-vla-v2-6b) | [Robbyant/lingbot-vla-v2-6b](https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b) | Vision-language-action policy checkpoint supplied as local shards |
| Qwen3-VL-4B-Instruct | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | [Qwen/Qwen3-VL-4B-Instruct](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) | Backbone configuration and processor |

The parity reference uses [Robbyant/lingbot-vla-v2](https://github.com/Robbyant/lingbot-vla-v2) at commit
`be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`.

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Official 6B base checkpoint | Supported | Local sharded safetensors checkpoint |
| RobotWin preprocessing | Supported | Three RGB cameras, task text, and a raw 14-dimensional state |
| Canonical action output | Supported | Normally `50 x 55` normalized actions |
| BF16 inference | Supported | Default path; strict 38-tensor upstream parity passed |
| CUDA Graph | Supported | Dynamic eager prefix with an opt-in fixed-shape action-denoising graph |
| Quantization | Partial | Profile-specific release status; see Configuration and Performance |
| Native server API | Supported | Asynchronous structured task API and `TFClient` |
| Request replicas | Supported | One complete policy copy per GPU |
| Single-policy FSDP, TP, or PP | Unsupported | The integration does not split one policy across GPUs |
| RoboTwin action mapping | Supported | Unnormalizes canonical output to absolute-position `50 x 14` chunks |
| Semantic VLA contract | Supported | Model and robot action spaces are explicit; see [VLA Action Integration](../../docs/en/vla.md) |
| Generic VLA session server | Preferred | Versioned JSON WebSocket with replica-affine sessions |

## Requirements

- GPU: one H100 80 GB was used for parity, performance, quantization, and service validation.
- Software: Python 3.10.12, PyTorch 2.11.0+cu130, CUDA 13.0, Transformers 5.14.1, and Triton 3.6.0.
- Quantization: TorchAO 0.17.0 and bitsandbytes 0.48.0 are pinned; tf-kernel FP8 needs a compatible SM90 wheel.
- Input assets: three RGB camera images, a non-empty instruction, and a finite 14-dimensional RobotWin state.

Install TeleFuser after installing the PyTorch build that matches the target CUDA runtime:

```bash
python3.10 -m venv .venv-vla
source .venv-vla/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
|-- lingbot/
|   `-- lingbot-vla-v2-6b/
|       |-- model.safetensors.index.json
|       `-- model-*.safetensors
`-- Qwen3-VL-4B-Instruct/
```

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

The VLA directory must contain every shard referenced by `model.safetensors.index.json`.

## Quick Start

```bash
mkdir -p work_dirs/lingbot_vla_v2
.venv-vla/bin/python examples/lingbot_vla_v2/lingbot_vla_v2_inference.py \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0,0,0,0,0,0,0,0,0,0,0,0,0,0]' \
  --seed 7 \
  --output work_dirs/lingbot_vla_v2/action_chunk.npz
```

The output NPZ contains the normalized canonical action chunk, shape metadata, checkpoint variant, and policy
verification status.

## Examples

### RobotWin Action Inference

#### `lingbot_vla_v2_inference.py`

This is the smallest in-process entry point. The input processor loads images in high, left-wrist, right-wrist order,
applies the bundled `bounds_99_woclip` statistics, and maps the raw 14-dimensional state into the 55-dimensional
canonical space.

```bash
python examples/lingbot_vla_v2/lingbot_vla_v2_inference.py \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --camera-high /path/to/cam_high.png \
  --camera-left-wrist /path/to/cam_left_wrist.png \
  --camera-right-wrist /path/to/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0,0,0,0,0,0,0,0,0,0,0,0,0,0]' \
  --output work_dirs/lingbot_vla_v2/action_chunk.npz
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--device` | `cuda` | Inference device |
| `--seed` | None | Optional deterministic noise seed |
| `--cuda-graph` | Disabled | Keep the dynamic prefix eager and graph the fixed-shape denoising loop |
| `--quantization` | None | Select an optional quantization profile |
| `--output` | `canonical_action_chunk.npz` | Output NPZ path |

The returned action chunk normally has horizon 50 and action dimension 55. The official base checkpoint intentionally
returns `policy_verified=False` and `verification_status="unverified_official_6b_base"`.

## Configuration

### CUDA Graph

`--cuda-graph` keeps vision-language prefix encoding and 36-layer KV-cache construction eager for every request, then
lazily captures all 10 fixed-shape action-denoising steps. Different instructions and language padding masks therefore
reuse the same denoising graph without being tied to the warmup instruction.

The denoising graph remains specialized to its tensor shapes, dtypes, and device. Standard preprocessing keeps these
layouts fixed at batch 1, language length 72, and action shape `1 x 50 x 55`. Prefix and graph execution are serialized
per policy instance. Close the pipeline to release graph buffers.

CUDA Graph cannot be combined with `torch.compile` or online quantization other than `fused-fp8-graph`. Invalid
combinations fail before model loading.

### Online Quantization

BF16 remains the default and the only profile covered by strict upstream parity. Quantization is applied in memory and
does not modify checkpoint files.

| CLI value | Backend | Scope | Validation status |
| --- | --- | --- | --- |
| `fused-fp8-graph` | Native scaled GEMM and Triton | Repeated denoising Linear and routed-MoE weights | H100 functional/AIPerf validated; release gate failed exact HTTP replay |
| `torchao-fp8` | TorchAO | 492 selected Qwen/action-expert Linear layers | H100 functional/AIPerf validated; release gate failed exact HTTP replay |
| `bnb-nf4` | bitsandbytes | Same 492 Linear-layer manifest, NF4 weights and BF16 compute | H100 functional/AIPerf validated; release gate failed exact HTTP replay |
| `tf-kernel-fp8` | TeleFuser tf-kernel | Per-token activation and per-output-channel weight FP8 | Code/unit tested; hardware unverified |

Use the direct example with one of the following variants:

```bash
# Fused FP8 requires CUDA Graph.
--quantization fused-fp8-graph --cuda-graph

# Online capacity profiles without CUDA Graph.
--quantization torchao-fp8
--quantization bnb-nf4
--quantization tf-kernel-fp8
```

The tf-kernel path requires an SM90 wheel built for the exact PyTorch/CUDA ABI. It remains "code support, hardware
unverified" until that real-model run succeeds on a compatible installation.

The complete H100 release suite passed BF16 eager and BF16 Graph. The three runnable quantized profiles passed
direct/HTTP numerical thresholds, AIPerf, dynamic-instruction, fault, and shutdown checks, but did not produce
bit-exact HTTP replays. They therefore remain code-supported capacity profiles rather than release-validated profiles.

The public loader accepts the same options:

```python
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline

pipeline = get_lingbot_vla_v2_pipeline(
    "/path/to/lingbot-vla-v2-6b",
    "/path/to/Qwen3-VL-4B-Instruct",
    device="cuda:0",
    quantization="torchao-fp8",
)
```

Compare a deterministic quantized capture with the corresponding TeleFuser BF16 capture:

```bash
.venv-vla/bin/python tools/validation/compare_lingbot_vla_v2_quantization.py \
  --reference work_dirs/vla_quantization/bf16_seed7.npz \
  --candidate work_dirs/vla_quantization/torchao_seed7.npz \
  --candidate-replay work_dirs/vla_quantization/torchao_replay_seed7.npz \
  --min-cosine 0.995 --max-relative-l2 0.10 --max-abs 0.5 \
  --require-exact-replay \
  --output work_dirs/vla_quantization/bf16_vs_torchao.json
```

## Performance

The measurements below are point results from one NVIDIA H100 80 GB system with CUDA 13.0, PyTorch 2.11.0+cu130,
Transformers 5.14.1, batch size 1, and fixed seed 7. Core-model timings reuse device-resident parity inputs and
exclude image decoding and preprocessing. They are environment-specific measurements, not universal performance
guarantees.

### CUDA Graph A/B

The graph captures only the fixed ten-step action-denoising loop; vision-language prefix encoding remains eager. The
comparison used five warmup requests and twenty serial requests with the same SDPA configuration in both paths.

| Scope | BF16 eager | BF16 denoising Graph | Change | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Direct core model | 649.790 ms | 164.226 ms | -74.73% | 3.957x |
| Direct runtime request | 649.843 ms | 164.910 ms | -74.62% | 3.941x |
| Service target inference | 1317.571 ms | 883.986 ms | -32.91% | 1.490x |
| Service HTTP end-to-end | 1362.134 ms | 929.761 ms | -31.74% | 1.465x |

### Quantization profiles

The table reports core-model latency and peak allocated VRAM from the H100 release profiles. The relative value is
`BF16 eager latency / profile latency`; values below `1.0x` are slower than BF16 eager. `fused-fp8-graph` combines
quantization with CUDA Graph and must be compared with BF16 Graph when isolating the quantization effect.

| Profile | Mean core latency | Relative to BF16 eager | Peak allocated VRAM | Interpretation |
| --- | ---: | ---: | ---: | --- |
| BF16 eager | 637.7 ms | 1.00x | 12,457 MiB | Reference |
| BF16 CUDA Graph | 131.6 ms | 4.84x | 12,487 MiB | Graph-only reference |
| `fused-fp8-graph` | 162.5 ms | 3.92x | 10,862 MiB | Combined FP8 + Graph path; 0.81x vs BF16 Graph |
| `torchao-fp8` | 1335.0 ms | 0.48x | 8,438 MiB | Lower memory, slower than BF16 eager |
| `bnb-nf4` | 924.7 ms | 0.69x | 6,479 MiB | Lower memory, slower than BF16 eager |

The CUDA Graph A/B table and the release-profile table use separate benchmark harnesses and run sets; compare
absolute timings only within the same table.

The quantized profiles are capacity and memory trade-off profiles rather than release-validated speedup claims. The
runnable profiles passed functional, numerical-threshold, AIPerf, fault, and shutdown checks, but did not produce
bit-exact HTTP replays. The `tf-kernel-fp8` profile is excluded from this table because compatible hardware validation
is still pending. Re-run the release suite to regenerate measurements for a different GPU or software environment.

## Serving

The three inference entrypoints share one `LingBotVlaV2Pipeline`; they are access modes, not separate model
implementations:

| Entry point | Role | Recommendation |
| --- | --- | --- |
| `lingbot_vla_v2_inference.py` | Direct Python reference and offline baseline | Keep for regression/debugging |
| `lingbot_vla_v2_native_service.py` via `telefuser serve` | Native HTTP structured requests | Keep for TeleFuser compatibility |
| `lingbot_vla_v2_vla_server.py` | Stateful generic VLA WebSocket | **Primary simulator path** |

For a production or simulation deployment, start only the generic WebSocket server. The other entries remain for
baseline comparison and backward compatibility and do not change the model or session implementation.

Start the native structured service:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action --parallelism 1 --host 127.0.0.1 --port 18080
```

Submit `POST /v1/tasks/structured` with `task="vla_action"`, the instruction, 14-dimensional state, three Base64
camera fields, and an optional seed. Poll `GET /v1/tasks/{task_id}/status`. Each encoded camera is limited to 10 MiB
and 16,777,216 decoded pixels.

```python
from telefuser.client import TFClient

client = TFClient("http://127.0.0.1:18080")
result = client.predict_vla_actions(
    instruction="pick up the red block",
    state=[0.0] * 14,
    camera_high_path="/data/cam_high.png",
    camera_left_wrist_path="/data/cam_left_wrist.png",
    camera_right_wrist_path="/data/cam_right_wrist.png",
    seed=7,
)
```

Use request-level replicas when multiple GPUs are available:

```bash
CUDA_VISIBLE_DEVICES=0,1 TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action --parallelism 2 --num-replicas 2 --port 18080
```

This creates one complete policy per GPU; it does not enable tensor or pipeline parallelism within a policy.

### Generic VLA Session Server

Start the preferred additive WebSocket service without mounting routes into `telefuser serve`:

```bash
CUDA_VISIBLE_DEVICES=0 TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  .venv-vla/bin/python -m examples.lingbot_vla_v2.lingbot_vla_v2_vla_server \
  --parallelism 1 --num-replicas 1 --host 0.0.0.0 --port 8000
```

The standalone server uses TeleFuser's default `STRICT` pipeline validation. Use `--skip-validation` only for trusted
local development after reviewing the pipeline file.

The service exposes `GET /healthz` and `/v1/vla/session`. Each `OPEN` reserves one pipeline replica for that session;
`PREDICT`, `RESET`, and `CLOSE` are sent to the same worker-local `VLASession`. Closing or disconnecting releases the
replica. The H100 cuDNN SDPA guard is applied inside each LingBot worker before model loading.

This protocol returns semantic `RobotActionChunk` values. The simulator process should deserialize the chunk, pass it
to `SimulatorChunkRuntime`, and then execute it through its own `SimulatorAdapter`. The inference server tracks
pending/ready/expired inference only and does not infer simulator execution. The simulator side can call
`execute_ready_with_report()` (or `execute_ready_async()` from an async client) to obtain a small transport-neutral
report containing `executed`, `failed`, or `no_action`, the chunk sequence, and the executed step count. The report can
be forwarded on an existing control channel; it is intentionally not a new WebSocket operation.

For a minimal P0 check, use one fake adapter test for report status and one async test for non-blocking execution. A
long-running soak, cross-machine clock comparison, and full simulator episode are not required to validate this
runtime API.

### RoboTwin via Generic VLA Session

RoboTwin clients use the same `/v1/vla/session` endpoint as MuJoCo. The XPolicyLab-side `TeleFuser_VLA` policy maps
RoboTwin observations to the negotiated `robotwin` embodiment contract and returns validated absolute-position action
chunks. No model-specific server or MessagePack dependency is required in TeleFuser.

Run the generic server on the inference host, then point the XPolicyLab adapter at
`ws://INFERENCE_HOST:8000/v1/vla/session` with `model_id=lingbot-vla-v2` and `embodiment_id=robotwin`. Keep the generic
WebSocket and the XPolicyLab policy server on a trusted private network or an SSH/VPN tunnel because neither endpoint
provides authentication.

## Local MuJoCo smoke simulation

When the RTX RoboTwin workstation is unavailable, the generic VLA action path can be exercised locally with MuJoCo.
The setup script installs MuJoCo into the existing TeleFuser `.venv` and stages EGL/OSMesa packages under the ignored
`.venv-mujoco-libs` directory; it never runs `apt install` or changes system Python. The scene reads the existing
RoboTwin ALOHA-Agilex URDF and meshes directly from `/data/RoboTwin` and adds a tabletop, cube, and three cameras.

```bash
bash examples/lingbot_vla_v2/setup_mujoco_local.sh

# Physics + three-camera local smoke (no VLA model required)
.venv/bin/python examples/lingbot_vla_v2/lingbot_vla_v2_mujoco.py \
  --mode local --image-size 256 --execute-horizon 8 \
  --output-dir work_dirs/lingbot_vla_v2/mujoco_local

# Complete local simulator -> generic VLA WebSocket -> simulator loop
.venv/bin/python examples/lingbot_vla_v2/lingbot_vla_v2_mujoco.py \
  --mode websocket --server-url ws://127.0.0.1:8000/v1/vla/session \
  --chunks 2 --execute-horizon 8 \
  --output-dir work_dirs/lingbot_vla_v2/mujoco_websocket
```

The adapter uses a semantic 14-dimensional `absolute_qpos` contract, PD torque control, and the existing
`SimulatorChunkRuntime`. `--mode local` validates model loading, rendering, action mapping, and execution without
loading LingBot. `--mode websocket` additionally validates the live generic VLA session and requires a running VLA
server. This is a chain/physics smoke test, not a RoboTwin task-success or checkpoint-quality evaluation.

## Validation

The repository includes strict upstream parity, runtime, quantization, structured-service, fault, and AIPerf
validators under `tools/validation/` and `benchmarks/telefuser_aiperf/`.

This integration is inference-only: it consumes the official LingBot-VLA v2 checkpoint without post-training or
fine-tuning. The MuJoCo example validates the simulator boundary and action execution path; it is not a substitute
for physical robot task-success evaluation. Keep direct/HTTP/WebSocket speed comparisons and their generated reports
under the ignored `work_dirs/` directory.

Compare previously captured upstream and TeleFuser artifacts:

```bash
.venv-vla/bin/python tools/validation/run_lingbot_vla_v2_parity.py \
  --reference work_dirs/vla_parity/upstream_seed7.npz \
  --candidate work_dirs/vla_parity/telefuser_seed7.npz \
  --profile strict --output work_dirs/vla_parity/strict_report.json
```

Validate a running structured service or run a bounded soak:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg \
  --quantization-profile bf16 --warmup 1 --requests 20 --concurrency 1 \
  --output work_dirs/vla_service_validation/smoke_20.json

# Replace --requests 20 with --duration-seconds 3600 for a one-hour run.
```

Run the repository-owned AIPerf workload with:

```bash
bash scripts/setup_aiperf.sh
bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

Run the complete real-GPU release suite after installing AIPerf. It executes every runtime profile in an isolated
process, compares direct and HTTP actions, changes instruction layout under CUDA Graph, runs bounded load and fault
checks, verifies shutdown/restart, and records full checkpoint, processor, software, CUDA, and GPU identity:

```bash
.venv-vla/bin/python tools/validation/run_lingbot_vla_v2_release_suite.py suite \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --image examples/data/lingbot_world_fast/image.jpg \
  --gpu-index 0 \
  --output-dir work_dirs/lingbot_vla_v2_release
```

Use `--profiles bf16-eager,bf16-graph` for an intermediate run. Such a partial run is useful for development but is
not a complete quantization support-matrix release result.

These checks establish framework parity and serving contracts, not physical robot task success.

## Troubleshooting

### CUDA Graph Or Quantization Is Rejected

Use CUDA Graph only with BF16 or `fused-fp8-graph`; the fused profile must include `--cuda-graph`. Other online
quantization profiles run without CUDA Graph and require CUDA.

### tf-kernel FP8 Cannot Load

Install a tf-kernel wheel built for the visible GPU architecture and exact PyTorch/CUDA ABI, or use BF16, TorchAO
FP8, or BNB NF4. Do not promote tf-kernel FP8 to supported status based only on unit tests.

## Notes

- Canonical normalized actions are not physical robot commands. Deployment requires de-normalization, embodiment
  mapping, control frequency, limits, safety policy, feedback, and emergency-stop behavior.
- `unverified_official_6b_base` remains intentional until an embodiment checkpoint and task-success evaluation are
  available.
- Quantized profiles require separate numerical, performance, and robot task-success acceptance.
- Generated captures and benchmark reports belong under the Git-ignored `work_dirs/` directory.
