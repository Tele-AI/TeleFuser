# MiniMax H3

These examples run the local MiniMax H3 Base release from its original FL2VA and Ref2VA partitions. They generate
24 FPS video with synchronized 32 kHz stereo audio. The local path supports 768p-class output; hosted Context-IR and
Regenerate-2K services are not implemented or implied.

## Requirements

- Linux with ffmpeg and ffprobe.
- One NVIDIA H100 80GB for sequential stage offload, or two/four H100 80GB GPUs for resident multi-GPU execution.
- Enough host memory for the approximately 63 GB encoder and 62 GB DiT partitions.
- The repository development environment and the unmodified model directory at
  `/hhb-data/aigc/model_zoo/MiniMaxAI_MiniMax-H3`.

Run commands from the repository root. The examples load original checkpoint shards through `ModuleManager`. A
one-GPU run uses stage-level model CPU offload. Multi-GPU runs keep the stages resident: two GPUs use Ulysses2 for
the DiT, TP2 for the text encoder, and TP2 video-VAE tiling; four GPUs use DiT Ulysses2 x TP2, text TP4, and TP4
video-VAE tiling. The audio VAE remains on GPU 0. The encoder and DiT use BF16; both VAEs remain FP32, with the
reference FP16 autocast boundary applied only to CUDA video decode.

The source-controlled default inputs live in `examples/data/minimax-h3/`. They are the exact inputs frozen for the
official SGLang parity runs; `provenance.json` records their original URLs, byte sizes, and SHA-256 hashes.

## T2VA And FL2VA

Use the explicit mode names when demonstrating a particular task. T2VA has no reference input:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode t2va \
  --prompt "A cinematic coastal landscape with synchronized ambient sound." \
  --duration 5 \
  --output outputs/minimax_h3_t2va.mp4
```

First-frame FL2VA uses the bundled reference image when `--image` is omitted:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode first-frame \
  --prompt "Steam rises from the ramen while the family talks in the background." \
  --duration 8 \
  --output outputs/minimax_h3_first_frame.mp4
```

Last-frame-only FL2VA accepts either `--last-image` or the bundled image:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode last-frame \
  --prompt "The camera settles on a warm family dinner at the final frame." \
  --output outputs/minimax_h3_last_frame.mp4
```

First-and-last-frame FL2VA accepts two images. When both are omitted, the bundled image is used at both endpoints;
that default is useful for a contract smoke run, while meaningful motion requires distinct endpoint images.

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode first-last \
  --image /path/to/first.png \
  --last-image /path/to/last.png \
  --prompt "Move smoothly from the first composition to the last composition." \
  --output outputs/minimax_h3_first_last.mp4
```

For compatibility, omitting `--mode` infers T2VA, first-frame, last-frame, or first-last from `--image` and
`--last-image`. Explicit modes are preferable in reproducible commands.

## Feature Cache

MiniMax H3 uses AdaTaylorCache around the complete joint audio-video DiT block stack. Calibrate once on one H100
with the same step count and scheduler shifts used for inference:

```bash
python examples/minimax_h3/minimax_h3_cache_calibrate.py \
  --steps 50 \
  --duration 4
```

The default output is
`telefuser/feature_cache/ada_taylor_cache/params/MiniMax-H3-Base.json`, the location selected by the existing
`FeatureCacheConfig` loader. Calibration runs full compute and derives its skip decisions from audio-token residuals;
video tokens otherwise dominate the joint sequence and can hide audio error. A 50-point H3 sigma schedule performs
49 DiT calls, so the generated file records `num_inference_steps: 49`. Calibration JSON files are local ignored
artifacts and are not bundled with the repository.

After calibration, enable the cache on the FL2VA example:

```bash
python examples/minimax_h3/minimax_h3_fl2va_h100.py \
  --mode t2va \
  --gpu-num 4 \
  --steps 50 \
  --duration 4 \
  --enable-feature-cache \
  --output outputs/minimax_h3_t2va_cached.mp4
```

The calibrated defaults use first-order Taylor approximation, at most two consecutive skips, 20% initial-step
retention, and a `0.03` schedule threshold. See the unified warm benchmark in
[Measured Four-GPU Profile](#measured-four-gpu-profile). Recalibrate when step count, scheduler shifts, checkpoint,
or target workload changes.

## Ref2VA

With no material arguments, the simple Ref2VA script uses the bundled reference video followed by the bundled voice
reference:

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --prompt "Preserve the source identity and motion, and use the reference voice for the dialogue." \
  --duration 5 \
  --output outputs/minimax_h3_ref2va.mp4
```

Custom material paths may be repeated:

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --image /path/to/subject.png \
  --video /path/to/motion.mp4 \
  --audio /path/to/voice.wav \
  --prompt "Keep the subject identity and follow the reference motion." \
  --duration 5 \
  --output outputs/minimax_h3_ref2va_custom.mp4
```

For an explicit mixed-media order, repeat `--material TYPE=URI`. Its order is passed through unchanged; it cannot be
combined with the grouped flags:

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --material video=https://example.com/motion.mp4 \
  --material image=/path/to/subject.png \
  --material audio=/path/to/voice.wav \
  --prompt "Use <Video 1>, preserve <Image 2>, and speak with <Audio 3>." \
  --duration 5 \
  --output outputs/minimax_h3_ref2va_ordered.mp4
```

The legacy convenience flags still group repeated arguments as images, videos, then audio. Use `--material TYPE=URI` or
the Ref2VA JSON request mode whenever heterogeneous ordering is semantic. Relative material URIs are resolved from the
request file's directory.

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --request examples/data/minimax-h3/ref2va.json \
  --output outputs/minimax_h3_ordered_request.mp4
```

The JSON `conditions` array is passed in its original order. Each entry accepts `type` (`image`, `video`,
`video_audio`, or `audio`), `role: "reference"`, `uri`, and optional `start_time_seconds` for video inputs.
`video_audio` requires both tracks; `video` uses its original soundtrack when one is present. For example:

```json
{
  "task": "ref2va",
  "prompt": "Use <Image 1>, then <Audio 1>, then the motion and soundtrack from <Video 1>.",
  "conditions": [
    {"type": "image", "role": "reference", "uri": "subject.png"},
    {"type": "audio", "role": "reference", "uri": "voice.wav"},
    {
      "type": "video_audio",
      "role": "reference",
      "uri": "motion.mp4",
      "start_time_seconds": 1.5
    }
  ],
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
  "seed": 0,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "num_inference_steps": 50
}
```

Ref2VA may omit `target.duration_seconds` when exactly one audio-bearing condition supplies the duration. With
multiple audio-bearing conditions, duration must be explicit. Published limits are enforced before model execution:
at most 9 images, 3 videos, 3 audio-bearing inputs, and 12 files total; each audio/video clip must be 2-15 seconds,
total video and total audio duration must each be at most 15 seconds, and audio requires an image or video reference.

## Inference-Only AdaLN Cache

The FL2VA and Ref2VA H100 examples enable online AdaLN caching by default. The first complete request computes AdaLN
normally; after successful denoising, the model releases the AdaLN and timestep-projection weights and later requests
reuse the in-memory cache. A later request with a different schedule requires a fresh pipeline or the full-weight path.

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --material video=/path/to/motion.mp4 \
  --material audio=/path/to/voice.wav \
  --output outputs/minimax_h3_ref2va_online_adaln.mp4
```

Ref2VA JSON request mode retains explicit `--online-adaln-cache` and `--no-online-adaln-cache` switches; it defaults
to cache-off to match the former standalone request runner.

FSDP remains unsupported for cache mode; single-GPU, Ulysses, and DiT TP are supported. Online TP collection gathers
each step modulation output across TP ranks before releasing the projection weights.

### Memory Accounting

Cache mode reduces persistent DiT weight allocation after the first successful online request. The following values are
calculated from the released H3 configuration, not sampled end-to-end peak-memory measurements. They use the standard
50-step video and audio schedules: the two schedules contain 99 unique timesteps, and each cached timestep stores all
50 block projections plus the final projection in BF16.

| Topology | Released AdaLN and timestep weights per rank | Device-resident cache per rank | Net persistent-weight reduction per rank |
|---|---:|---:|---:|
| Single GPU or Ulysses | 24.35 GiB | 0.89 GiB | 23.45 GiB |
| DiT TP2 | 12.20 GiB | 0.89 GiB | 11.31 GiB |
| DiT TP4 | 6.13 GiB | 0.89 GiB | 5.24 GiB |

The calculation excludes activations, the text encoder, both VAEs, communication buffers, and PyTorch allocator
reservation. Therefore, it describes the steady-state capacity returned by AdaLN removal rather than a universal
end-to-end peak reduction. During online collection, the first request keeps the full weights; the reduction applies
only after that request finalizes successfully. NVIDIA SMI can retain cached allocator pages until PyTorch releases
them, so use allocated memory and a steady cached request when validating this accounting on a deployment.

## Standard Python And Serve Entrypoints

All three executable modules expose `PPL_CONFIG`, `get_pipeline`, `run`, and `run_with_file`. The two fixed-partition
generation modules also expose `PIPELINE_MANIFEST` for serving. `get_pipeline(parallelism, model_root)` interprets
`parallelism` as the total GPU count and selects the corresponding profile below. `run` returns the in-memory
`MiniMaxH3Generation`; `run_with_file` writes the synchronized MP4 and returns its `output_path`.

```python
from examples.minimax_h3.minimax_h3_fl2va_h100 import get_pipeline, run_with_file

pipeline = get_pipeline(4, "/path/to/MiniMaxAI_MiniMax-H3")
try:
    artifact = run_with_file(
        pipeline,
        task="i2v",
        prompt="Steam rises from the ramen while the family talks.",
        first_image_path="examples/data/minimax-h3/fl2va-reference.png",
        output_path="outputs/minimax_h3_i2v.mp4",
    )
finally:
    pipeline.stop()
```

Serve T2VA, first-frame I2VA, and first-and-last-frame FL2VA from the shared FL2VA checkpoint partition:

```bash
telefuser serve examples/minimax_h3/minimax_h3_fl2va_h100.py --gpu-num 4 --port 8000
```

The service contract advertises `t2v`, `i2v`, and `fl2v`. Submit `first_image_path` for `i2v`, and both
`first_image_path` and `last_image_path` for `fl2v`. Output duration must be between 4 and 15 seconds:

```bash
curl -X POST http://127.0.0.1:8000/v1/tasks/create \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"i2v\",\"prompt\":\"Animate this dinner scene with synchronized dialogue.\",\"first_image_path\":\"https://example.com/first.png\",\"resolution\":\"768p\",\"aspect_ratio\":\"16:9\",\"target_video_length\":5}"
```

Ref2VA uses its own checkpoint partition and advertises the standard `s2v` service task, which maps to the
model-specific Ref2VA task. Its required `conditions` parameter is the same ordered array accepted by the local
pipeline, so heterogeneous image, video, `video_audio`, and audio references are not reordered by the example.
Output duration must also be between 4 and 15 seconds:

```bash
telefuser serve examples/minimax_h3/minimax_h3_ref2va_h100.py --task s2v --gpu-num 4 --port 8001

curl -X POST http://127.0.0.1:8001/v1/tasks/create \
  -H "Content-Type: application/json" \
  -d "{\"task\":\"s2v\",\"prompt\":\"Use <Video 1>, then <Audio 2>.\",\"conditions\":[{\"type\":\"video\",\"role\":\"reference\",\"uri\":\"https://example.com/motion.mp4\"},{\"type\":\"audio\",\"role\":\"reference\",\"uri\":\"https://example.com/voice.mp3\"}],\"resolution\":\"768p\",\"aspect_ratio\":\"16:9\",\"target_video_length\":5}"
```
Use `/v1/service/metadata` to inspect the active task contract. Ref2VA's `--request` mode is the convenient local
entrypoint for request files and resolves relative material paths beside the JSON file.

## Generation And Parallel Options

The simple CLIs expose `--steps`, `--seed`, `--duration`, `--aspect-ratio`, `--flow-shift`, and
`--audio-flow-shift`. The FL2VA CLI additionally exposes the existing feature-cache initialization controls.
Supported explicit aspect ratios are `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, and `9:16`; `auto` follows the task
policy or first FL2VA keyframe.

`--gpu-num` selects the total worker count. `--ulysses-degree` remains a compatibility alias for the same CLI option;
it no longer means that every selected GPU is necessarily an Ulysses rank.

| `--gpu-num` | DiT | Text encoder | Video VAE | Residency |
|---:|---|---|---|---|
| 1 | single GPU | single GPU | single GPU | sequential model CPU offload |
| 2 | Ulysses2 | TP2 | TP2 tiling | resident |
| 4 | Ulysses2 x TP2 | TP4 | TP4 tiling | resident |

The Ulysses degree must divide 56 attention heads. Scripts must run from their guarded entry points so worker
processes can spawn safely. H100 examples request packed FlashAttention 4 and fall back to packed PyTorch SDPA when
FlashAttention 4 is unavailable.

## Online DiT Quantization

MiniMax H3 supports two single-GPU online quantization backends for the DiT transformer Linear layers:

| CLI value | Backend | Weight/activation path |
|---|---|---|
| torchao-fp8 | TorchAO | FP8 dynamic activation and FP8 weight when supported, otherwise TorchAO's FP8 weight-only path |
| bnb-nf4 | bitsandbytes | NF4 weight-only with BF16 compute |

Both paths convert the 258 Linear layers in the main and token-refiner transformer blocks. The FP32 video/audio
patch projections, timestep embedding, output projections, text encoder, and VAEs retain their reference dtypes.
The BF16 DiT is loaded from the original shards, moved to CUDA after text encoding, quantized on first denoising use,
and then kept resident for the pipeline lifetime. This ordering avoids a simultaneous BF16 text encoder and DiT on
one GPU and avoids unsupported CPU transfers of quantized tensor subclasses.
TorchAO conversion has a transient memory peak near the BF16 footprint; use the full 80 GB device without colocated workloads.

Use the dedicated TorchAO FP8 example:

~~~bash
python examples/minimax_h3/minimax_h3_fl2va_torchao_fp8_h100.py \
  --mode t2va \
  --duration 5 \
  --output outputs/minimax_h3_torchao_fp8.mp4
~~~

Or the dedicated bitsandbytes NF4 example:

~~~bash
python examples/minimax_h3/minimax_h3_fl2va_bnb_nf4_h100.py \
  --mode t2va \
  --duration 5 \
  --output outputs/minimax_h3_bnb_nf4.mp4
~~~

The standard FL2VA, Ref2VA, and JSON request CLIs also accept
--quantization with either torchao-fp8 or bnb-nf4. The Python loader accepts the same names:

~~~python
from examples.minimax_h3.common import load_minimax_h3_pipeline

pipeline = load_minimax_h3_pipeline(
    "/path/to/MiniMaxAI_MiniMax-H3",
    partition="FL2VA",
    quantization="torchao-fp8",
)
~~~

Online quantization currently requires ulysses_degree=1, tp_degree=1, and FSDP disabled. Quantizing before TP/FSDP
would invalidate those wrappers' BF16 parameter-sharding contract, so unsupported combinations fail before checkpoint
loading.

For matched BF16/FP8/NF4 profiling, use the validation benchmark. It writes the synchronized MP4 plus a JSON report
containing load time, end-to-end generation time, stage timings, and denoising allocator peaks:

~~~bash
python tools/validation/benchmark_minimax_h3_quantization.py \
  --backend torchao-fp8 \
  --duration 5 \
  --steps 50 \
  --output outputs/minimax_h3_torchao_fp8_50step.mp4
~~~

For multi-GPU resident profiles, `WorkerTensorChannel` transports text conditioning, visual condition rows, and the
final video latent directly between worker groups. CUDA intermediates therefore do not stage through the parent
process or CPU. The pipeline reports media, text, condition VAE, denoising, video/audio decode, allocator peak, and
computed/skipped feature-cache steps in `MiniMaxH3Generation.runtime_metrics`.

H3 also uses eager BF16 Triton paths for Q/K RMSNorm plus partial NeoX RoPE, indexed modulation, SwiGLU, and Ulysses
relayout when their input contracts match. Compatible `tf-kernel` builds may accelerate public RMSNorm, SwiGLU, and
RoPE operations. All public ops retain native PyTorch fallbacks for unsupported devices, dtypes, and compile mode.

FSDP2 remains available for an SP-only DiT profile and cannot be combined with DiT TP. The standard CLI can select
it explicitly for a two-GPU Ulysses run:

```bash
python examples/minimax_h3/minimax_h3_ref2va_h100.py \
  --ulysses-degree 2 \
  --enable-fsdp \
  --output outputs/minimax_h3_ref2va_fsdp.mp4
```

The standard four-GPU profile already uses Ulysses2 x TP2 and therefore leaves FSDP disabled. Use
`load_minimax_h3_pipeline` directly to construct another supported combination; the product of Ulysses and TP degrees
must be 1, 2, or 4.

Ring attention, CFG parallelism, pipeline parallelism, sparse attention, and `torch.compile` are not enabled for H3.
Video-VAE parallelism is spatial tiling over the existing TP process group, not parameter tensor parallelism. The
dedicated service manifests expose the pipeline without adding framework-level configuration fields or changing the
shared request schema.

## Four-GPU Regression

The example regression registry includes `minimax_h3_t2va_4gpu`, a 768p, five-second, 50-step T2VA request with seed
0. It reserves four GPUs and uses the resident Ulysses2 x TP2 profile. Regression runs force packed PyTorch SDPA,
matching the repository-wide deterministic regression policy; ordinary example and service execution still defaults
to packed FlashAttention 4. The standard file entrypoint preserves synchronized video and audio in the baseline MP4;
regression additionally validates the audio stream contract, duration, and decoded waveform similarity.

Initialize or intentionally replace the local ignored baseline:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
python examples/run_examples.py \
  --pipeline minimax_h3_t2va_4gpu \
  --gpus 0,1,2,3 \
  --update-baseline
```

Run the regression against that baseline:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
python examples/run_examples.py \
  --pipeline minimax_h3_t2va_4gpu \
  --gpus 0,1,2,3
```

## Measured Four-GPU Profile

The comparison below was retested on 2026-08-06 using the frozen 768p, five-second, 50-point T2VA request with seed
0, online AdaLN cache enabled, and the resident Ulysses2 x TP2 four-H100 profile. Each configuration starts
a fresh pipeline, runs one unmeasured warmup request, then measures the second request. Pipeline time includes text
encoding, DiT, video/audio decode, host materialization, and orchestration; it excludes model/worker initialization
and MP4 encoding. Wall time surrounds the same `run()` call. Full-device memory is sampled from `nvidia-smi`
every 100 ms during the measured request.

The measurements use PyTorch 2.11.0, CUDA 12.8, NCCL 2.28.9, and the source-built SM90 `tf-kernel` wheel from this
checkout. Reproduce the three rows from the repository root:

```bash
python -m tools.validation.benchmark_minimax_h3_four_gpu \
  --attention FLASH_ATTN_4 \
  --output /tmp/minimax_h3_flash.json
python -m tools.validation.benchmark_minimax_h3_four_gpu \
  --attention SAGE_ATTN_2_8_8_SM90 \
  --output /tmp/minimax_h3_sage.json
python -m tools.validation.benchmark_minimax_h3_four_gpu \
  --attention FLASH_ATTN_4 \
  --feature-cache \
  --output /tmp/minimax_h3_flash_cache.json
```

| Attention | Feature cache | Computed / skipped DiT calls | Pipeline time | Wall time | DiT time | Pipeline speedup | Peak memory GPU 0 / 1 / 2 / 3 |
|---|---|---:|---:|---:|---:|---:|---:|
| FlashAttention 4 | Disabled | 49 / 0 | 77.32 s | 77.63 s | 74.64 s | 1.00x | 51.48 / 50.28 / 50.30 / 50.28 GiB |
| SageAttention 2_8_8 SM90 | Disabled | 49 / 0 | 72.41 s | 72.72 s | 69.85 s | 1.07x | 51.42 / 50.20 / 50.24 / 50.24 GiB |
| FlashAttention 4 | AdaTaylorCache | 26 / 23 | 42.39 s | 42.68 s | 39.82 s | 1.82x | 52.73 / 52.09 / 51.56 / 51.54 GiB |

The immediately preceding revision (`ebbcf9f`) used PyTorch/NCCL scatter under the same documented request, warmup,
hardware, and parallel profile. Holding each attention/cache configuration fixed gives the communication-path
comparison below:

| Fixed configuration | NCCL pipeline / DiT | CUDA IPC pipeline / DiT | Pipeline reduction | DiT reduction |
|---|---:|---:|---:|---:|
| FlashAttention 4, cache disabled | 79.10 / 76.48 s | **77.32 / 74.64 s** | **2.25%** | **2.41%** |
| SageAttention 2_8_8 SM90, cache disabled | 75.96 / 73.37 s | **72.41 / 69.85 s** | **4.67%** | **4.80%** |
| FlashAttention 4, AdaTaylorCache | 43.53 / 40.87 s | **42.39 / 39.82 s** | **2.62%** | **2.57%** |

These are separate fresh revision runs rather than an in-process toggle, so ordinary run-to-run variance remains.
See the [CUDA IPC Ulysses technical article](../../docs/en/blog/cuda_ipc_ulysses.md) for the execution trace, design,
claim boundary, and related work.

Sage SM90 reduces pipeline latency by 6.34% and DiT latency by 6.43%. It is approximate and remains an H100 opt-in:

```bash
python -m examples.minimax_h3.minimax_h3_fl2va_h100 \
  --gpu-num 4 \
  --attn-impl SAGE_ATTN_2_8_8_SM90 \
  --target-video-length 5 \
  --output outputs/minimax_h3_sage_sm90.mp4
```

Against the FlashAttention 4 output from the same seed, the Sage run measured video PSNR 20.45 dB, mean SSIM
0.7683, and audio cosine similarity 0.98505. Review generated quality for the target workload before selecting it in
production; FlashAttention 4 remains the default.

AdaTaylorCache reduces steady-state pipeline latency by 45.2% and increases maximum single-GPU occupancy by
1.25 GiB (2.4%) in these measurements. Against the previously matched uncached MP4, PSNR is 26.91, SSIM is 0.8619,
audio cosine similarity is 0.9562, and audio duration is unchanged. The earlier matched local SGLang SP2+TP2 parity
run measured 79.37 seconds and 67.8 GiB on GPU 0 under the same request shape.

With the source-built tf-kernel available, MiniMax H3 uses the direct CUDA IPC Copy Engine Ulysses scatter. The
fused-QKV projection is passed as a strided V view first; Q/K normalization and RoPE then overlap that transfer,
followed by tagged Q/K transfers and one shared GPU-memory handshake. The three destination buffers are cached per
Ulysses group, so this avoids a QKV packing copy and repeated target allocation. If the optional backend is
unavailable, the same calls fall back to NCCL.

These numbers describe this request and environment, not a general performance or quality guarantee.
