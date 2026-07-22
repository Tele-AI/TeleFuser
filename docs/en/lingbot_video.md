# LingBot-Video

LingBot-Video supports Dense and MoE base DiTs for T2I, T2V, and TI2V. The
MoE checkpoint also includes a separate low-noise refiner. The integration is
precision-first: use the upstream reference capture artifacts before enabling
backend or distributed optimizations.

## Checkpoints

The Dense loader consumes a Diffusers `transformer/` directory directly:

```python
from telefuser.pipelines.lingbot_video import load_lingbot_video_dense_transformer

transformer = load_lingbot_video_dense_transformer(
    "/path/to/lingbot-video-dense-1.3b/transformer"
)
```

The MoE and refiner directories are sharded. Use
`load_lingbot_video_moe_transformer` for either `transformer/` or `refiner/`.
The default sorted eager expert path preserves upstream route ordering and keeps
a `where`-based diagnostic fallback. It is a validated single-GPU BF16 path,
but not a grouped-GEMM, FP8, or distributed production-throughput backend. For
`variant="moe"`, the runtime defaults to stage CPU offload so the base DiT,
text encoder, VAE, and separately loaded refiner do not need to reside on one
GPU. Set `cpu_offload=False` only when GPU capacity is known to be sufficient.

## Prompt preparation

The generation pipeline intentionally consumes the structured JSON caption, while
prompt rewriting remains an optional, separately deployable workflow. Preserve the
official two-stage contract: EXPAND uses the base VLM with its LoRA disabled, and
MAP uses the same VLM with the LingBot rewriter LoRA enabled. For TI2V, provide
the identical first frame to both the rewriter and TeleFuser generation.

```bash
REWRITER_BASE_MODEL=/path/to/Qwen3.6-27B \
REWRITER_ADAPTER=/path/to/lingbot-video-rewriter-lora \
python work_dirs/lingbot-video-master/rewriter/inference.py \
  --mode t2v --prompt "<plain prompt>" --duration 5 --output prompt.json
```

Pass the resulting `prompt.json` to `--caption-json`, or serialize its
`caption` object as the service `prompt`. The rewriter is deliberately not loaded
inside a DiT service process because the two models have separate deployment and
capacity requirements.

Unless explicitly overridden, the pipeline, CLI, and service use the checkpoint's
structured negative CFG caption. T2I uses the source still-image variant, while
T2V and TI2V use the source video variant, including its temporal-stability
constraints. Do not replace an omitted negative prompt with an empty string when
reproducing an upstream sample: it changes the Qwen3-VL negative condition and
can materially alter color and image quality.

## Runtime composition

`LingBotVideoPipeline` composes independently loaded stages:

- `LingBotVideoTextEncodingStage` encodes the structured JSON caption with
  Qwen3-VL.
- `LingBotVideoDenoisingStage` runs source-order, two-forward CFG.
- `LingBotVideoVAEEncodeStage` and `LingBotVideoVAEDecodeStage` apply the
  checkpoint VAE's latent mean/std normalization.
- `FlowUniPCMultistepScheduler` owns the sigma/timestep sequence.

Attach stages with `pipeline.set_runtime(...)` after initializing the pipeline
config. Provide a structured JSON caption, not casual unstructured text.

For the standard checkpoint layout, `build_lingbot_video_pipeline` loads those
components directly without importing the upstream runtime:

```python
from telefuser.pipelines.lingbot_video import LingBotVideoRequest, build_lingbot_video_pipeline

pipeline = build_lingbot_video_pipeline("/path/to/lingbot-video-dense-1.3b", num_inference_steps=40)
frames = pipeline(LingBotVideoRequest(caption=structured_caption, height=480, width=832, num_frames=121))
```

Direct API and CLI heights and widths must be divisible by 16: the Wan VAE
downsamples by eight and the DiT spatially patchifies the resulting latents by two.

The default `AttentionConfig` uses TeleFuser's SDPA dispatcher and remains the
source-equivalent numerical path. Alternative attention backends are opt-in
through `attention_config=` and require a separate L2 parity report; they are
not enabled by the service or CLI defaults.

The VAE decode stage returns RGB video in `[0,1]`. Video callers must pass these
float frames directly to Diffusers `export_to_video`, which performs the uint8
conversion itself. Converting to uint8 before that call applies a second 255
scale and wraps channel values, producing a negative-like MP4.

## TI2V

TI2V has two independent first-frame condition paths:

1. The image is supplied to Qwen3-VL as visual input with the caption.
2. The image is VAE-encoded to a clean temporal-prefix latent. It is written
   before every denoising step and once after the final scheduler step.

The current pipeline accepts a raw RGB condition image in the range [0,255] with shape
`[B, 3, H, W]` (or `[B, 3, F, H, W]`, using frame zero). It resizes and center-crops the image before separately passing it to Qwen3-VL and the VAE.

## Service

The service entrypoint exposes `t2i`, `t2v`, and `i2v` through TeleFuser service APIs and requires a structured JSON string as `prompt`:

```bash
LINGBOT_VIDEO_MODEL_ROOT=/path/to/lingbot-video-dense-1.3b \
  telefuser serve examples/lingbot_video/lingbot_video_service.py --port 8000
```

Set `LINGBOT_VIDEO_VARIANT=moe` and `LINGBOT_VIDEO_ENABLE_REFINER=1` to use the MoE checkpoint with its separate refiner. The service releases base-stage weights before loading the refiner; `refine` is also an explicit per-request boolean parameter.
Requested service resolutions are rounded up to the LingBot VAE-and-DiT
sixteen-pixel spatial grid; for example, `480p` at `16:9` resolves to 864x480.
Pass `negative_prompt` only to override the source-compatible default; an
explicit empty string remains a supported override.

## Refiner

`LingBotVideoRefinerStage` takes the base RGB output in memory, VAE-encodes it,
mixes it with noise at `t_thresh`, and samples the low-noise sigma tail. It can
also preserve a clean TI2V frame zero. Base and refiner are separate runtime
stages. Call `base_pipeline.release_gpu_resources()` before loading the refiner
when they share a GPU.
The included CLI implements this lifecycle for MoE checkpoints:

```bash
python examples/lingbot_video/lingbot_video_generate.py \
  --model-dir /path/to/lingbot-video-moe-30b-a3b --variant moe --refine \
  --caption-json /path/to/caption.json --output result.mp4
```

With `--image first_frame.png`, the CLI also applies the upstream TI2V frame-zero geometry to the refiner condition.

For an in-memory base-to-refiner handoff, call `prepare_refiner_video(...)` before
`LingBotVideoRefinerStage.refine(...)`. It matches upstream training-aligned frame
selection and bicubic resize without an MP4 write/read round trip; pass the base
output FPS explicitly. Validate this path against the corresponding source MP4 baseline.
The MP4 compatibility test uses the upstream Diffusers writer and compares the upstream loader (through a PyAV-backed decord adapter when decord is unavailable) with this loader tensor-for-tensor.


## Validation

Capture upstream artifacts before comparing a TeleFuser run:

```bash
python tools/validation/capture_lingbot_video_reference.py --dry-run
python tools/validation/capture_lingbot_video_reference.py --all-cases --mode t2i --mode t2v --mode ti2v --trace sampled
python tools/validation/inspect_lingbot_video_checkpoint.py --model-dir /path/to/lingbot-video-dense-1.3b --variant dense --output dense-load-report.json
python tools/validation/inspect_lingbot_video_checkpoint.py --model-dir /path/to/lingbot-video-moe-30b-a3b --variant moe --output moe-load-report.json
python tools/validation/inspect_lingbot_video_checkpoint.py --model-dir /path/to/lingbot-video-moe-30b-a3b --variant refiner --output refiner-load-report.json
python tools/validation/compare_lingbot_video_parity.py REFERENCE CANDIDATE
python tools/validation/replay_lingbot_video_dense_reference.py --reference-dir work_dirs/lingbot_video_reference/t2v/example_1/run-00
python tools/validation/replay_lingbot_video_dense_reference.py --validate-text --reference-dir work_dirs/lingbot_video_reference/ti2v/example_1/run-00
python tools/validation/replay_lingbot_video_dense_reference.py --validate-text --validate-ti2v-vae --reference-dir work_dirs/lingbot_video_reference/ti2v/example_1/run-00
python tools/validation/replay_lingbot_video_dense_reference.py --reference-root work_dirs/lingbot_video_reference_all_cases --assert-exact --output dense-all-cases-replay.json
PYTHONPATH=work_dirs/lingbot-video-master python tools/validation/run_lingbot_video_moe_parity.py --transformer-dir /path/to/lingbot-video-moe-30b-a3b/transformer --assert-exact
PYTHONPATH=work_dirs/lingbot-video-master python tools/validation/run_lingbot_video_refiner_core_parity.py --model-root /path/to/lingbot-video-moe-30b-a3b --assert-exact
python tools/validation/validate_lingbot_video_refiner_handoff.py --input base.mp4 --height 1088 --width 1920 --assert-exact
python tools/validation/validate_lingbot_video_refiner_output_handoff.py --model-dir /path/to/lingbot-video-moe-30b-a3b --caption-json prompt.json --height 64 --width 64 --num-frames 5 --steps 1 --refiner-height 64 --refiner-width 64 --refiner-steps 1 --output handoff-output-report.json --comparison-output handoff-comparison.mp4
python tools/validation/benchmark_lingbot_video.py --model-dir /path/to/lingbot-video-dense-1.3b --caption-json prompt.json --output result.mp4 --report benchmark.json --warmup 1 --runs 3
python tools/validation/benchmark_lingbot_video.py --model-dir /path/to/lingbot-video-moe-30b-a3b --variant moe --refine --caption-json prompt.json --output result.mp4 --report benchmark.json --warmup 1 --runs 3
```

`validate_lingbot_video_refiner_handoff.py` checks that the TeleFuser MP4
compatibility loader is tensor-identical to the source loader. Add
`--in-memory-video` and `--in-memory-fps` to quantify the input difference
introduced by MP4 encoding; this comparison does not replace a final refiner
output-quality evaluation.
Use `--assert-exact` to make the source MP4 compatibility comparison fail on
metadata, tensor shape, dtype, or value drift. It intentionally does not judge
the native in-memory handoff, whose difference from lossy MP4 is expected.
`validate_lingbot_video_refiner_output_handoff.py` generates one MoE base
sample, drives the refiner with both the native RGB tensor and a temporary MP4
round trip using identical prompt conditions and RNG state, then reports the
final-output L2 difference. This is intentionally a quality comparison, not an
equivalence test: the upstream refiner uses the lossy MP4 round trip, while the
native path removes that intermediate encoding. The report includes decoded-frame
PSNR and local SSIM; `--comparison-output` writes memory output on the left and
MP4-round-trip output on the right for human review.

The capture tool records scheduler tensors, prompt tensors, selected denoising steps, latent inputs/outputs, the generation seed, RNG state hashes, and decoded frame hashes. Add `--validate-text` to the replay command to compare Qwen3-VL processor inputs and final embeddings; TI2V validation also compares the preprocessed first frame. Add `--validate-ti2v-vae` to compare the sampled clean condition latent; use `--seed` only with pre-existing captures that lack seed metadata.
Use `--reference-root` for an all-case Dense DiT/VAE replay. It retains one
loaded Dense transformer and VAE across captured runs, while instantiating a
fresh scheduler for each run so each capture preserves its own sampling setup.
Add `--assert-exact` to make the command fail when any recorded tensor differs
in shape or value, so it can serve as a CI parity gate rather than a report-only
diagnostic.
The checkpoint-inspection tool performs the normal strict load and records the
consumed config fields, checkpoint-key coverage, component/block parameter
counts, dtype/device distribution, retained FP32 parameter count, and model
memory allocation evidence.
The benchmark tool reports one-time setup separately from warmup and measured metrics for checkpoint load, text encoding, each denoising step, VAE, refiner, output encoding, and peak GPU memory. Use it to establish a baseline before enabling an optimization; record full-resolution measurements separately from smoke runs.
When `--negative-caption` is omitted, the benchmark uses the same source-compatible
T2I or video negative caption as the pipeline, CLI, and service. Pass an explicit
empty string only when intentionally benchmarking that semantic override.
For a base-plus-refiner run, it also records the serial base release and refiner
load phases. The default sorted eager MoE path is source-equivalent and has an
explicit diagnostic fallback, but it is not a grouped-GEMM or FP8
production-throughput backend.
For T2I/T2V, the refiner reuses the exact CFG text conditions from the base
generation and reports this as `refiner_prompt_conditions_reused`; TI2V keeps
the source-compatible text-only refiner encoding path.
The refiner core CLI injects identical latent, noise, prompt, and frame-zero condition tensors into the upstream and TeleFuser low-noise paths. It offloads the upstream DiT before loading the TeleFuser DiT, so both 30B models do not overlap in GPU memory.
Add `--assert-exact` to the MoE or refiner core validator to enforce the
zero-drift numerical-oracle gate instead of only writing metrics.

## Requirements and limitations

The numerical-oracle path requires CUDA, PyTorch, Diffusers, Transformers, and the
checkpoint components `transformer/`, `text_encoder/`, `processor/`, `vae/`, and
`scheduler/`. Dense runs source-equivalently on one GPU. The current MoE expert
implementation is an eager correctness path; do not use it as a 30B production
backend. FSDP, Ulysses/CFG parallelism, FlashAttention, and FP8 experts are intentionally not
enabled. The service and CLI both support a serial single-process base-plus-refiner lifecycle; distributed alignment is outside the current support scope.
