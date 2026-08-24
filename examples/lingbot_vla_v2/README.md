# LingBot-VLA v2

This example runs the official LingBot-VLA v2 6B base checkpoint through TeleFuser and returns a normalized
`50 x 55` canonical action chunk. The RobotWin profile prepares the observation only; the base output is not a
physical robot command.

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
  lingbot/lingbot-vla-v2-6b/
  Qwen3-VL-4B-Instruct/
```

```bash
export TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo
```

The VLA directory must contain `model.safetensors.index.json` and every referenced shard. The Qwen3-VL directory
provides the visual-language backbone configuration and processor.

## Validated H100 Environment

Strict parity, runtime comparison, quantization screening, and structured-service validation used this environment:

| Component | Validated value |
| --- | --- |
| GPU | NVIDIA H100 80GB HBM3 (SM90) |
| NVIDIA driver | `590.48.01` |
| Python | `3.10.12` |
| PyTorch | `2.11.0+cu130` |
| TorchVision | `0.26.0+cu130` |
| TorchAudio | `2.11.0+cu130` |
| PyTorch CUDA runtime | `13.0` |
| Transformers | `5.14.1` |
| Triton | `3.6.0` |

Create the model-specific environment inside this repository. Install the matching CUDA 13.0 PyTorch wheels first
so dependency resolution retains the validated ABI:

```bash
python3.10 -m venv .venv-vla
source .venv-vla/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-index --find-links /path/to/cu130-wheels \
  "torch==2.11.0+cu130" "torchvision==0.26.0+cu130" "torchaudio==2.11.0+cu130"
python -m pip install -e ".[dev]"
```

`.venv-vla` is ignored by Git and does not modify Conda base or the system Python. Use its interpreter explicitly in
the commands below.

## Feature Support

| Feature | Support |
| --- | --- |
| Official 6B base checkpoint | Supported |
| Public loader and RobotWin preprocessing | Supported |
| Canonical action output | Supported, normally `50 x 55` |
| Strict official-upstream parity | Passed, frozen 38-tensor baseline |
| Matched upstream speed comparison | Recorded below for H100 BF16 |
| Native structured HTTP service | Supported |
| AIPerf structured workload | Supported |
| Request-level replicas | Supported, one policy copy per GPU |
| BF16 vision SDPA | Enabled when flash-attn is unavailable; H100 A/B recorded below |
| CUDA Graph prefix and action denoising | Opt-in BF16 path; two graphs share static KV buffers |
| Online quantization | BF16 default; capacity-oriented profiles are opt-in |
| Fused FP8 graph | Experimental H100 path; reduces memory but does not improve batch-1 latency |
| tf-kernel FP8 | Code/unit tested; compatible SM90 wheel not validated on this host |
| Single-policy FSDP, TP, or PP | Not enabled |
| Physical action mapping and safety control | Not included |

## Files

| File | Purpose |
| --- | --- |
| `lingbot_vla_v2_inference.py` | Direct in-process inference |
| `lingbot_vla_v2_native_service.py` | Native TeleFuser structured-service contract |
| `../../telefuser/pipelines/lingbot_vla_v2/` | Pipeline, preprocessing, policy, and service adapter |
| `../../tools/validation/` | Parity, runtime, service, fault, and quantization validators |

Generated captures and benchmark reports belong under the Git-ignored `work_dirs/` directory.

## Usage

### Inputs and Output

Inputs are three RGB cameras in upstream RobotWin order (high, left wrist, right wrist), a raw 14-dimensional state,
and a non-empty task instruction. The SDK applies the bundled upstream `bounds_99_woclip` statistics and maps the
observation into LingBot's 55-dimensional canonical state.

The returned `LingBotVlaV2CanonicalActionChunk` contains:

- `canonical_normalized_actions`: `[H, 55]` base-model output.
- `horizon`: normally 50 for the official base configuration.
- `action_dim`: normally 55.
- `checkpoint_variant`: `base`.
- `policy_verified=False` and `verification_status="unverified_official_6b_base"`.

### Direct Inference

```bash
.venv-vla/bin/python examples/lingbot_vla_v2/lingbot_vla_v2_inference.py \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --camera-high /data/cam_high.png \
  --camera-left-wrist /data/cam_left_wrist.png \
  --camera-right-wrist /data/cam_right_wrist.png \
  --task "pick up the red block" \
  --state-json '[0,0,0,0,0,0,0,0,0,0,0,0,0,0]' \
  --seed 7 \
  --output canonical_action_chunk.npz
```

The `.npz` contains canonical actions and checkpoint metadata.

### Native TeleFuser Service

The service uses the shared `PIPELINE_CONTRACT`, asynchronous scheduler, pipeline pool, task-status API, runtime
metrics, and `TFClient`.

```bash
TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action --parallelism 1 --host 127.0.0.1 --port 18080
```

Submit `POST /v1/tasks/structured` with `task="vla_action"`, `instruction`, the 14-dimensional `state`, the three
Base64 camera fields, and an optional `seed`. Poll `GET /v1/tasks/{task_id}/status`; a completed result includes the
action payload, `inference_time_s`, and optional `peak_memory_mb`.

Each encoded camera is limited to 10 MiB and 16,777,216 decoded pixels before RGB conversion. Both limits are
model-specific settings in `PPL_CONFIG` and apply independently to all three cameras.

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
print(result["horizon"], result["action_dim"])
```

Use independent request-level replicas when multiple GPUs are available:

```bash
CUDA_VISIBLE_DEVICES=0,1 TF_MODEL_ZOO_PATH=/hhb-data/aigc/model_zoo \
  .venv-vla/bin/telefuser serve \
  examples/lingbot_vla_v2/lingbot_vla_v2_native_service.py \
  --task vla_action --parallelism 2 --num-replicas 2 --port 18080
```

This creates one complete policy per GPU. It does not split one policy with tensor or pipeline parallelism.

## Optional Dual CUDA Graph

The BF16 runtime uses two lazily captured CUDA Graphs. The prefix graph encodes the fixed-shape visual/language prefix
and constructs all 36 KV-cache layers. The denoising graph captures all 10 action-denoising steps. Prefix KV outputs
are the denoising graph's static KV inputs, so steady-state replay does not perform a second per-layer KV copy. This
removes prefix Python dispatch, the denoising Python loop, and repeated kernel-launch dispatch from the request path.

```python
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline

pipeline = get_lingbot_vla_v2_pipeline(
    "/path/to/lingbot-vla-v2-6b",
    "/path/to/Qwen3-VL-4B-Instruct",
    device="cuda:0",
    cuda_graph=True,
)
```

The graphs are specialized to the first request's tensor shapes, dtypes, device, image-grid values, camera-valid mask,
and language-padding mask. LingBot-VLA v2's public
preprocessing contract supplies fixed shapes (`batch=1`, language length 72, and action shape `1 x 50 x 55`), so
subsequent standard requests meet that contract while image content, language tokens, state, and noise may change.
Changing a static layout value requires a new policy instance. Replay is serialized per policy instance because its
static buffers are shared. Call `pipeline.close()` to release both graphs and their buffers.

CUDA Graph mode is mutually exclusive with `torch.compile` and with online quantization other than the experimental
`fused-fp8-graph` profile; invalid combinations fail explicitly. BF16 eager execution remains the default.

## Optional Online Quantization

BF16 remains the default and the only profile covered by strict upstream parity. Quantization is opt-in and does not
modify checkpoint files. The legacy online profiles keep the fused action MoE weights and action heads in BF16. The
`fused-fp8-graph` profile instead targets only repeated denoising computation: 144 action-attention Linear layers, 108
shared-expert Linear layers, two action-time MLP Linear layers, and all 36 routed-MoE expert tensors. Router gates,
state/action input-output projections, AdaNorm projections, the visual/language prefix, and action head remain BF16.

| CLI value | Backend | Path | Validation status |
| --- | --- | --- | --- |
| `fused-fp8-graph` | Native scaled GEMM + Triton | Pre-materialized W8A8 denoise weights; BF16 copies discarded | H100 graph, action, latency, and memory validated; experimental capacity option |
| `torchao-fp8` | TorchAO | Dynamic FP8 activation/weight or FP8 weight-only fallback | H100 real forward, action comparison, lifecycle |
| `tf-kernel-fp8` | TeleFuser tf-kernel | Per-token activation and per-output-channel weight FP8 | Code/unit tested; compatible SM90 wheel unavailable |
| `bnb-nf4` | bitsandbytes | NF4 weight-only, BF16 compute | H100 real forward, action comparison, lifecycle |

TeleFuser's base dependency set currently declares TorchAO and bitsandbytes, while selecting a quantized VLA profile
remains optional. The validated VLA environment uses the exact versions below. If either module is unavailable or a
different version was resolved, repair only `.venv-vla` without changing the system environment:

```bash
uv pip install --python .venv-vla/bin/python --reinstall --no-deps \
  "torchao==0.17.0" "bitsandbytes==0.48.0"
```

Add `--quantization fused-fp8-graph --cuda-graph` to direct inference for the experimental fused graph profile. Other
accepted values are `torchao-fp8`, `bnb-nf4`, and `tf-kernel-fp8`; the tf-kernel path requires an SM90 wheel built for
the exact PyTorch/CUDA ABI from `tf-kernel/`. Do not use a wheel built for another SM family or CUDA ABI. For the native
service, set `PPL_CONFIG["quantization"] = "fused-fp8-graph"` and `PPL_CONFIG["cuda_graph"] = True` in
`lingbot_vla_v2_native_service.py`; they default to `None` and `False`.

The public loader uses the same option:

```python
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline

pipeline = get_lingbot_vla_v2_pipeline(
    "/path/to/lingbot-vla-v2-6b",
    "/path/to/Qwen3-VL-4B-Instruct",
    device="cuda:0",
    quantization="torchao-fp8",
)
```

The official manifest SHA-256 is
`f9efe28620796060ccc46bd18ac153a580b28d01c7719fa55a8e80631f2ce833`. A changed layer count or name manifest fails
before conversion. Reports record this hash, selected groups, wrapper and weight types, package versions, and backend.

To compare a quantized backend against unchanged TeleFuser BF16, capture both with identical input, seed, and
`--deterministic-moe`, then run:

```bash
.venv-vla/bin/python tools/validation/compare_lingbot_vla_v2_quantization.py \
  --reference work_dirs/vla_quantization/bf16_seed7.npz \
  --candidate work_dirs/vla_quantization/torchao_seed7.npz \
  --output work_dirs/vla_quantization/bf16_vs_torchao.json
```

Optional gates are `--min-cosine`, `--max-relative-l2`, `--max-abs`, and
`--candidate-replay ... --require-exact-replay`. This is a quantization regression against TeleFuser BF16, not strict
official-upstream parity or RoboTwin task-success evidence.

## Validation

### Strict Official-Upstream Parity

The official reference pins `Robbyant/lingbot-vla-v2` at commit
`be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`. Its isolated checkout, uv environment, cache, and artifacts remain under
`work_dirs/`:

```bash
mkdir -p work_dirs/.uv-cache-upstream work_dirs/.uv-tmp-upstream
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" \
  uv venv work_dirs/.venv-lingbot-upstream --python .venv-vla/bin/python
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" \
  uv pip install --python work_dirs/.venv-lingbot-upstream/bin/python \
  -r tools/validation/requirements-lingbot-vla-v2-upstream.txt
UV_CACHE_DIR="$PWD/work_dirs/.uv-cache-upstream" TMPDIR="$PWD/work_dirs/.uv-tmp-upstream" \
  uv pip install --python work_dirs/.venv-lingbot-upstream/bin/python --no-deps \
  "lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz"
git clone https://github.com/Robbyant/lingbot-vla-v2 work_dirs/lingbot-vla-v2-upstream
git -C work_dirs/lingbot-vla-v2-upstream checkout be27333c9b5f2663b0ec33f069dd7dfd67fa32b5
```

Generate the official artifact with `capture_lingbot_vla_v2_upstream.py`, the TeleFuser artifact with
`capture_lingbot_vla_v2_telefuser.py`, and compare them with `run_lingbot_vla_v2_parity.py --profile strict`. Both
captures must use identical checkpoints, cameras, task, state, seed, device, `--deterministic-moe`, eager attention,
and deterministic reference MoE metadata. TeleFuser capture metadata records both the commit and whether tracked
files were dirty; release evidence must use a clean worktree and `--full-checkpoint-hash`.

| Layer | Compared | Passed | Failed | Global max abs |
| --- | ---: | ---: | ---: | ---: |
| Preprocessing tensors | 6 | 6 | 0 | `0.0` |
| Initial action noise | 1 | 1 | 0 | `0.0` |
| Timesteps | 10 | 10 | 0 | `0.0` |
| Per-step `x_t` | 10 | 10 | 0 | `0.0` |
| Per-step velocity | 10 | 10 | 0 | `0.0` |
| Final normalized action (`50 x 55`) | 1 | 1 | 0 | `0.0` |
| **Total** | **38** | **38** | **0** | **`0.0`** |

The official constructor hard-codes FlashAttention, so the upstream capture selects eager attention only inside the
validation process. Production inference still reaches the upstream Triton MoE through `telefuser.ops`; strict
capture uses deterministic reference MoE because atomic accumulation is not bitwise repeatable across processes.
The comparator rejects mixed attention or MoE backends.

### TeleFuser Regression Baseline

For changes that do not require a fresh official capture, run the TeleFuser capture twice with identical arguments
and compare the artifacts with the same strict comparator:

```bash
.venv-vla/bin/python tools/validation/capture_lingbot_vla_v2_telefuser.py \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --camera-high /data/cam_high.png --camera-left-wrist /data/cam_left.png \
  --camera-right-wrist /data/cam_right.png --task "pick up the red block" \
  --state-json '[0,0,0,0,0,0,0,0,0,0,0,0,0,0]' --seed 7 --deterministic-moe \
  --output work_dirs/vla_regression/baseline_seed7.npz

# Repeat the capture with:
# --output work_dirs/vla_regression/replay_seed7.npz

.venv-vla/bin/python tools/validation/run_lingbot_vla_v2_parity.py \
  --reference work_dirs/vla_regression/baseline_seed7.npz \
  --candidate work_dirs/vla_regression/replay_seed7.npz --profile strict \
  --output work_dirs/vla_regression/strict_report.json
```

Each capture has a JSON sidecar with checkpoint, processor, input, runtime, and tensor-contract metadata. This detects
TeleFuser regressions but does not independently establish equivalence with the official repository.

### Native Structured API

Run the validator after the service reports ready:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg \
  --quantization-profile bf16 --warmup 1 --requests 20 --concurrency 1 \
  --output work_dirs/vla_service_validation/smoke_20.json
```

For a two-replica run, use `--warmup 2 --requests 100 --concurrency 2`. For a bounded soak, use
`--duration-seconds 7200 --service-pid <telefuser-parent-pid> --gpu-indexes 0`. Local resource sampling sums RSS over
the service process tree and groups `nvidia-smi` process memory by physical GPU. Reports keep bounded samples,
distributions, and first/last 10% trends without storing full actions or Base64 images.

The validator freezes the VLA request and result fields, checks readiness, unique task IDs, queue drain, and finite
`50 x 55` actions, and exits nonzero on any failure. `--quantization-profile` labels the running configuration; it does
not modify the service or inspect its quantized wrappers.

Fault checks cover missing cameras, invalid state size, invalid Base64, and cancellation:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_service_faults.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg
```

Replica termination is opt-in for a disposable two-replica service through `--service-pid` and
`--kill-replica-gpu-index`. The validator checks one-replica capacity degradation and a subsequent valid response; it
does not promise automatic replica restart.

The same contract is available as a repository-owned AIPerf workload:

```bash
bash scripts/setup_aiperf.sh
bash benchmarks/telefuser_aiperf/scripts/run_vla_structured_bench.sh
```

AIPerf owns warmup exclusion, aggregation, traces, server metrics, and artifacts. Passing these validators proves the
serving path and normalized action structure, not physical control semantics.

### Historical Native HTTP and CI Evidence (`baf3d18`)

On 2026-08-12, TeleFuser commit `baf3d18840a71363984edb46222ef86200efb689` was validated with one H100 80GB,
Python 3.10.12, PyTorch 2.11.0+cu130/CUDA 13.0, one replica at `127.0.0.1:18080`, one warmup, and 20 sequential
requests. All three camera fields used the same source image; the request used seed 7 and a zero-valued 14-dimensional
RobotWin state.

The service used the single-replica command above. The measured workload added process and GPU sampling:

```bash
.venv-vla/bin/python tools/validation/validate_lingbot_vla_v2_structured_service.py \
  --base-url http://127.0.0.1:18080 \
  --image examples/data/lingbot_world_fast/image.jpg \
  --warmup 1 --requests 20 --concurrency 1 \
  --service-pid <service-pid> --gpu-indexes 0 \
  --output work_dirs/vla_service_validation/smoke_20_baf3d18.json
```

| Check | Result |
| --- | --- |
| Overall validation | Passed |
| Measured requests | 20 |
| Successful / failed | 20 (100%) / 0 |
| Unique task IDs / completed | 20 / 20 |
| Action contract | 20 finite `50 x 55` canonical normalized chunks |
| Policy status | 20 `unverified_official_6b_base` |
| Ready before and after / warmup / queue drain | Passed |
| Resource sampling | Passed, 26 CPU and GPU samples |

| Latency | Mean | p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| End to end | 1.370 s | 1.372 s | 1.391 s | 1.391 s | 1.391 s |
| Accepted to terminal | 1.337 s | 1.339 s | 1.355 s | 1.356 s | 1.356 s |
| Target inference | 1.267 s | 1.252 s | 1.330 s | 1.330 s | 1.330 s |
| Submission | 0.033 s | 0.032 s | 0.035 s | 0.036 s | 0.036 s |

Throughput was 0.729 requests/s. GPU process memory remained at 13,302 MiB. Process-tree RSS peaked at 3,617.9 MiB
and changed by -0.6 MiB between the first and last sample windows. The service stopped normally, leaving no process or
GPU allocation. The ignored raw report contains bounded statistics and action fingerprints, not complete action or
Base64 payloads.

The strict 38-item calculation was not repeated at `baf3d18`: the model, loader, preprocessing, velocity sampling,
and action path had not changed since `2d40ee2`; changes through `baf3d18` affected the service and validation boundary.
This HTTP evidence supplements rather than replaces the frozen strict parity result.

Full-project CI ran in a separate `.venv`; cross-model dependencies such as PyAV, OpenCV, Diffusers, and ImageIO were
not installed into `.venv-vla`:

```bash
PATH=/data/telefuser_vla_test/.venv/bin:$PATH \
  bash scripts/run_ci_tests.sh --skip-install
```

| Stage | Result |
| --- | --- |
| Ruff check | Passed |
| Ruff format check | Passed, 510 files checked |
| Ruff import check | Passed |
| CPU-only runtime assertion | Passed |
| Unit tests | 1,236 passed, 8 skipped, 114 deselected, 5 subtests passed |
| Server and OpenAI API tests | 62 passed |
| Overall CI | Passed |

Expected skips covered CUDA-only operations, optional local tf-kernel, and two LingBot-Video refiner parity checks
whose separate upstream checkout was unavailable. No VLA, structured-service, shared-service, or cross-model test
failed. This run proves that the real 6B checkpoint crossed HTTP, scheduler, pipeline service, serialization, and
status polling while preserving `50 x 55`; it does not change `unverified_official_6b_base` or prove robot semantics.

## Performance

### Official Upstream vs TeleFuser

Establish strict numerical parity before comparing speed. The runtime benchmark consumes the same frozen preprocessed
tensors and initial noise, and requires matching checkpoint, device, software, attention, and MoE identities. It times
the device-resident core and a runtime request boundary without parity hooks or intermediate CPU copies.

Run both implementations sequentially on the same idle GPU, then compare their reports:

```bash
CUDA_VISIBLE_DEVICES=0 work_dirs/.venv-lingbot-upstream/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_runtime.py \
  --implementation upstream --upstream-root work_dirs/lingbot-vla-v2-upstream \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --input-artifact work_dirs/vla_upstream_parity/upstream_seed7.npz \
  --seed 7 --device cuda:0 --warmup 3 --runs 20 \
  --output work_dirs/vla_runtime_comparison/upstream_h100_runs20.json

CUDA_VISIBLE_DEVICES=0 .venv-vla/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_runtime.py \
  --implementation telefuser \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --input-artifact work_dirs/vla_upstream_parity/upstream_seed7.npz \
  --seed 7 --device cuda:0 --warmup 3 --runs 20 \
  --output work_dirs/vla_runtime_comparison/telefuser_h100_runs20.json

.venv-vla/bin/python tools/validation/compare_lingbot_vla_v2_runtime_benchmarks.py \
  --upstream work_dirs/vla_runtime_comparison/upstream_h100_runs20.json \
  --telefuser work_dirs/vla_runtime_comparison/telefuser_h100_runs20.json \
  --output-json work_dirs/vla_runtime_comparison/upstream_vs_telefuser_h100_runs20.json \
  --output-markdown work_dirs/vla_runtime_comparison/upstream_vs_telefuser_h100_runs20.md
```

The recorded run used upstream commit `be27333c9b5f2663b0ec33f069dd7dfd67fa32b5`, TeleFuser model-source commit
`86278d4a22d35f7cd8606dddd80ae3a4637e396c`, one H100 80GB, the validated software versions above, eager attention,
the upstream Robby Triton MoE, three warmups, and 20 measured requests per scope.

| Scope | Metric | Upstream | TeleFuser | TeleFuser change |
| --- | ---: | ---: | ---: | ---: |
| Core model | mean | 669.382 ms | 660.100 ms | -1.39% |
| Core model | p50 | 668.373 ms | 657.364 ms | -1.65% |
| Core model | p95 | 677.683 ms | 669.343 ms | -1.23% |
| Core model | p99 | 678.620 ms | 685.735 ms | +1.05% |
| Runtime request | mean | 662.462 ms | 658.935 ms | -0.53% |
| Runtime request | p50 | 661.707 ms | 656.974 ms | -0.72% |
| Runtime request | p95 | 666.779 ms | 678.023 ms | +1.69% |
| Runtime request | p99 | 669.705 ms | 682.456 ms | +1.90% |

Negative change means TeleFuser was faster. `Core model` measures `sample_actions` with device-resident fixed inputs
and noise. `Runtime request` also includes tensor transfer, seeded-noise construction, output validation, and CPU
action delivery; both exclude image decoding and preprocessing. Peak allocated CUDA memory was 12,454.8 MiB for both.
Mean differences below 1.5% show no material TeleFuser overhead in this matched run. The small 20-sample p99 result is
not a tail-latency conclusion, and loader time is not compared because construction boundaries differ.

### BF16 Vision SDPA

When flash-attn is unavailable, the Qwen3-VL vision encoder now uses BF16 SDPA while the language/action attention
backend remains eager. A fixed-input H100 A/B used the same weights and inputs for both attention implementations:

| Scope | Eager | BF16 SDPA | Change |
| --- | ---: | ---: | ---: |
| Vision encoding | 18.595 ms | 15.005 ms | -19.3% |
| Full model | 175.557 ms | 172.048 ms | -2.0% |

The action difference did not exceed the eager replay variation from the fused Robby MoE atomic accumulation.

### Dual CUDA Graph

Run the TeleFuser benchmark twice on the same idle GPU with the same frozen input. Omit `--cuda-graph` for the eager
baseline and include it for the graph run:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vla/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_runtime.py \
  --implementation telefuser \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --input-artifact work_dirs/vla_quantization/current_bf16_seed7.npz \
  --seed 7 --device cuda:0 --cuda-graph --warmup 5 --runs 20 \
  --output work_dirs/vla_prefix_graph/bf16_sdpa_dual_graph_h100.json
```

The recorded matched run used one H100 80GB, BF16 vision SDPA, eager action attention, the Robby Triton MoE, five
warmups, and 20 measured requests. Each core request clones the same initial noise before inference. The graph column
captures both prefix/KV construction and all 10 denoising steps.

| Scope | Metric | Eager | CUDA Graph | Change | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Core model | mean | 631.254 ms | 131.905 ms | -79.10% | 4.79x |
| Core model | p50 | 629.088 ms | 131.897 ms | -79.03% | 4.77x |
| Core model | p95 | 642.094 ms | 131.966 ms | -79.45% | 4.87x |
| Core model | throughput | 1.584 req/s | 7.581 req/s | +378.60% | 4.79x |
| Runtime request | mean | 632.654 ms | 132.669 ms | -79.03% | 4.77x |
| Runtime request | p50 | 629.594 ms | 132.645 ms | -78.93% | 4.75x |
| Runtime request | p95 | 643.463 ms | 132.867 ms | -79.35% | 4.84x |
| Runtime request | throughput | 1.581 req/s | 7.538 req/s | +376.79% | 4.77x |
| Peak allocated CUDA memory | peak | 12,456.838 MiB | 12,487.122 MiB | +0.24% | n/a |

The graph report also compares full `1 x 50 x 55` action chunks inside the same loaded process. Robby Triton MoE
atomic accumulation is not bitwise repeatable, so the relevant acceptance criterion is whether graph-vs-eager error
exceeds the eager replay baseline:

| Comparison | Max abs | Relative L2 | Cosine similarity |
| --- | ---: | ---: | ---: |
| Eager vs eager | 0.081543 | 0.014549 | 0.999895 |
| CUDA Graph vs eager | 0.148438 | 0.015322 | 0.999884 |
| CUDA Graph vs CUDA Graph | 0.132812 | 0.014528 | 0.999896 |

The graph-vs-eager error stayed at the same scale as the eager replay baseline; no additional numerical degradation
beyond the non-deterministic atomic MoE variation was observed. The deterministic unit model additionally requires
exact equality across all 10 captured denoising steps and subsequent replays.

### Single-GPU Service Benchmark

```bash
CUDA_VISIBLE_DEVICES=0 .venv-vla/bin/python \
  tools/validation/benchmark_lingbot_vla_v2_service.py \
  --model-root "$TF_MODEL_ZOO_PATH/lingbot/lingbot-vla-v2-6b" \
  --qwen3vl-root "$TF_MODEL_ZOO_PATH/Qwen3-VL-4B-Instruct" \
  --image examples/data/lingbot_world_fast/image.jpg \
  --image-sizes 256x256,640x480,1280x720 --warmup 1 --runs 20 \
  --output work_dirs/vla_service_benchmark/report.json
```

The report includes construction, startup warmup, first/steady request latency, p50/p90/p95/p99, throughput, phase
timing, RSS, CUDA peaks, shutdown, and allocator state. The default `service-thread` mode matches the native runner;
`--execution-mode direct` measures only the in-process ceiling. Source images are always converted to the official
`256 x 256` input, so source size changes boundary/preprocessing cost rather than model token shape.

### Quantization Screening

Five fixed-input H100 requests produced this screening result. It verifies the path but is not a production baseline:

| Profile | Mean request | Throughput | Steady GPU allocated | Action cosine vs BF16 | Relative action L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.668 s | 1.496 req/s | 12,328 MiB | 1.00000 | 0.00% |
| TorchAO FP8 | 1.385 s | 0.722 req/s | 8,293 MiB | 0.99969 | 2.48% |
| BNB NF4 | 0.965 s | 1.037 req/s | 6,327 MiB | 0.99822 | 6.06% |

Both online formats reduced allocated memory but were slower than BF16 on this H100. Treat them as capacity options
until longer performance and RoboTwin task-success evaluations establish deployment benefit. tf-kernel FP8 was not
run because the available CUDA toolkit was 12.8 while `.venv-vla` used CUDA 13.0.

The graph-compatible fused FP8 path was screened separately with five warmups and 10 measured dual-graph requests.
It pre-materialized exactly 254 action-path Linear weights plus all 36 routed-MoE expert tensors as E4M3, used static
activation/scale/workspace addresses during replay, and retained no BF16 routed-expert weight copies.

| Profile | Core mean | Runtime mean | Throughput | Steady GPU allocated | Cosine vs frozen BF16 | Relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 dual graph | 131.646 ms | 131.511 ms | 7.604 req/s | 12,454 MiB | 0.999885 | 1.52% |
| Fused FP8 dual graph | 162.770 ms | 163.335 ms | 6.122 req/s | 10,828 MiB | 0.998643 | 5.28% |

Fused FP8 reduced steady allocation by 1,626 MiB (13.1%) but increased runtime latency by 24.2%. Actual-shape H100
microbenchmarks showed the same cause: routed MoE changed from 0.108 to 0.140 ms per layer and action QKV from 0.073
to 0.123 ms per layer because batch-1 activation quantization overhead exceeded the FP8 tensor-core gain. This profile
is therefore an explicit capacity experiment, not a recommended latency optimization. BF16 dual graph remains the
default deployment path; task-success evaluation is still required before using any quantized action output.

## Notes and Limitations

- BF16 strict parity proves numerical equivalence at the captured model boundary; it does not certify robot behavior.
- `unverified_official_6b_base` is intentional until an embodiment-specific checkpoint and task-success evaluation
  establish control semantics.
- Do not execute canonical actions directly. Real control requires de-normalization, joint/action mapping, control
  frequency, limits, safety policy, feedback, and emergency-stop behavior.
- Quantized profiles require separate numerical, performance, and task-success acceptance criteria.
- Request replicas scale independent policy copies; single-policy FSDP, TP, and PP are outside this integration.
- Runtime benchmarks exclude HTTP scheduling unless the structured-service validator or AIPerf workload is used.
