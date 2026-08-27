# LingBot-VLA v2 Examples

This example runs the official LingBot-VLA v2 6B base checkpoint through TeleFuser. It accepts a RobotWin
observation and returns a normalized `50 x 55` canonical action chunk through direct Python inference or the native
structured service.

## Model Source

| Model | Hugging Face | ModelScope | Purpose |
| --- | --- | --- | --- |
| LingBot-VLA v2 6B base | N/A | N/A | Vision-language-action policy checkpoint supplied as local shards |
| Qwen3-VL-4B-Instruct | [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | N/A | Backbone configuration and processor |

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
| Physical robot action mapping | Unsupported | Output remains in normalized canonical space |

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
| `fused-fp8-graph` | Native scaled GEMM and Triton | Repeated denoising Linear and routed-MoE weights | Code support; hardware unverified until the current release gate passes |
| `torchao-fp8` | TorchAO | 492 selected Qwen/action-expert Linear layers | H100 forward, action, replay, and lifecycle validated |
| `bnb-nf4` | bitsandbytes | Same 492 Linear-layer manifest, NF4 weights and BF16 compute | H100 forward, action, replay, and lifecycle validated |
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

## Serving

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

## Validation

The repository includes strict upstream parity, runtime, quantization, structured-service, fault, and AIPerf
validators under `tools/validation/` and `benchmarks/telefuser_aiperf/`.

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

## Performance

The following measurements used one H100 80 GB, Python 3.10.12, PyTorch 2.11.0+cu130, CUDA 13.0, Transformers
5.14.1, TorchAO 0.17.0, and bitsandbytes 0.48.0. Runtime means cover the same fixed-shape action request after warmup.

### Runtime And Quantization

| Profile | Runtime mean | Steady GPU allocated | Action comparison | Status |
| --- | ---: | ---: | --- | --- |
| BF16 eager | 636.107 ms | 12,299.5 MiB | Strict upstream parity 38/38; max abs `0.0` | Supported |
| BF16 dual CUDA Graph (historical) | 132.081 ms | 12,454.3 MiB | Cosine `0.999913`; relative L2 `0.013195` | Superseded; rerun denoising-only graph |
| Fused FP8 dual graph (historical) | 163.194 ms | 10,828.0 MiB | Cosine `0.999040`; relative L2 `0.044357` | Superseded; current release gate pending |
| TorchAO FP8 | 1,321.630 ms | 8,266.4 MiB | Cosine `0.999714`; relative L2 `0.024001`; exact replay | Experimental capacity profile |
| BNB NF4 | 915.299 ms | 6,297.4 MiB | Cosine `0.998031`; relative L2 `0.063843`; exact replay | Experimental capacity profile |
| tf-kernel FP8 | Not measured | Not measured | No compatible CUDA 13/SM90 wheel installed | Code support; hardware unverified |

Reproduce a measured profile from a frozen input artifact by changing `--quantization` and the output name. Add
`--cuda-graph` when measuring BF16 graph or `fused-fp8-graph`:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vla/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_runtime.py \
  --implementation telefuser \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --input-artifact work_dirs/vla_quantization/bf16_seed7.npz \
  --seed 7 --device cuda:0 --quantization torchao-fp8 \
  --warmup 5 --runs 20 \
  --output work_dirs/vla_quantization/torchao_runtime.json
```

The two dual-graph rows preserve the previous implementation's timing comparison; they are not measurements of the
current dynamic-prefix implementation. Rerun the release suite and fixed-input runtime benchmark before publishing new
graph numbers. TorchAO FP8 and BNB NF4 reduced memory but were slower than BF16 eager at batch 1, so they remain
capacity options rather than latency recommendations.

The quantization gate requires finite `50 x 55` actions, cosine similarity at least `0.995`, relative L2 at most
`0.10`, max absolute error at most `0.5`, and exact deterministic replay. It does not establish robot task success.

### Official Upstream Comparison

The matched eager BF16 run used three warmups and 20 measured requests on the same H100 and frozen inputs:

| Scope | Upstream mean | TeleFuser mean | TeleFuser change |
| --- | ---: | ---: | ---: |
| Core model | 669.382 ms | 660.100 ms | -1.39% |
| Runtime request | 662.462 ms | 658.935 ms | -0.53% |

Negative change means TeleFuser was faster. Both implementations allocated 12,454.8 MiB at peak in this run.

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
