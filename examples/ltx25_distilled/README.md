# LTX-2.5 Distilled Examples

One-to-four-H100 text-to-video (T2V) and image-to-video (I2V) generation with the distilled LTX-2.5 pipeline. The example
produces an MP4 containing generated video and synchronized 48 kHz stereo audio.

The LTX-2.5 implementation is isolated under `telefuser/models/ltx25` and
`telefuser/pipelines/ltx25_distilled`; it does not reuse the legacy LTX-2.3 model or pipeline modules.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LTX-2.5 22B distilled model pack | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) | N/A | Transformer, text encoder, video/audio VAEs, spatial upsampler, and duration head |

This example does not auto-download weights. Download the official repository while preserving its directory layout:

```bash
hf download Lightricks/LTX-2.5 --local-dir /path/to/LTX-2.5
```

The example requires the exact split LTX-2.5 checkpoint layout shown below; a consolidated checkpoint or Diffusers
directory is not accepted.

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-video | Supported | Generates video and audio from a text prompt |
| Image-to-video | Supported | Accepts one still image at a non-negative output-frame index |
| Multi-GPU inference | Supported | Ulysses sequence parallelism and block-level FSDP2 on 2 or 4 H100s |
| Attention backend | Supported | Select a dense `AttnImplType` with `--attn-impl`; FlashAttention 4 is the default |
| Video VAE | Supported | DiffVAE is the default; ConvVAE is selectable with `--video-vae conv` |
| CPU offload | Supported | `cpu` streams transformer blocks and releases modules between phases; `none` retains modules on the GPU |
| LoRA | Unsupported | The example does not expose a LoRA loader |
| Quantization | Unsupported | The example loads BF16 checkpoints |
| Feature cache | Unsupported | The example does not configure feature caching |
| Server API | Partial | Legacy `get_pipeline()` and `run_with_file()` entry points exist, but no explicit pipeline contract is declared |

## Requirements

- GPU: one, two, or four NVIDIA H100s; other GPU targets are not validated by this example
- Software: the standard TeleFuser environment, `transformers==5.14.1`, and an `ffmpeg` executable on `PATH`
- DiffVAE: a matching NATTEN/libnatten build is required for the formal 1536x1024, 121-frame workload
- I2V input: a PIL-readable still image such as PNG or JPEG

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup). For the formal
DiffVAE path, select the command matching the installed PyTorch and CUDA versions from the
[NATTEN installation guide](https://natten.org/install/), then verify that its CUDA kernel library is available:

```bash
python -c "import natten; print(natten.HAS_LIBNATTEN)"
```

The command must print `True`. Without NATTEN, the DiffVAE decoder uses the Triton/eager compatibility fallback,
which is not the formal performance or accuracy baseline.

## Model Directory

`--model-root` must point to the root of this exact split-checkpoint layout:

```text
/path/to/LTX-2.5/
|-- diffusion_models/
|   \-- ltx-2.5-22b-distilled-transformer-bf16.safetensors
|-- text_encoders/
|   \-- gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
|-- vae/
|   |-- ltx-2.5-video-vae-bf16.safetensors
|   |-- ltx-2.5-video-vae-conv-bf16.safetensors
|   \-- ltx-2.5-audio-vae-bf16.safetensors
|-- latent_upscale_models/
|   \-- ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
\-- model_patches/
    \-- ltx-2.5-duration-head-bf16.safetensors
```

The current checkpoint resolver validates all seven files at pipeline construction time, including both video VAE
checkpoints regardless of the `--video-vae` selection.

Validate the split model pack without loading checkpoint tensors:

```bash
python tools/validation/inspect_ltx25_checkpoints.py \
  --model-root /path/to/LTX-2.5 \
  --output work_dirs/ltx25-checkpoints.json
```

## Quick Start

Run the default T2V workload from the repository root:

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --prompt "A cinematic camera orbit around the subject." \
  --output-path work_dirs/ltx25-t2v.mp4
```

The command writes a 1536x1024, 121-frame, 24 FPS video with synchronized audio to
`work_dirs/ltx25-t2v.mp4`.

Run the denoising stage with four-way Ulysses sequence parallelism and FSDP2 shards:

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_h100.py \
  --gpu-num 4 \
  --attn-impl FLASH_ATTN_4 \
  --model-root /path/to/LTX-2.5 \
  --prompt "A cinematic camera orbit around the subject." \
  --output-path work_dirs/ltx25-t2v-sp4.mp4
```

## Examples

### `ltx25_distilled_t2v_h100.py`

This is the standalone text-to-video entry point.

```bash
python examples/ltx25_distilled/ltx25_distilled_t2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --prompt "Ocean waves roll beneath a cloudy sky as distant thunder echoes." \
  --output-path work_dirs/ltx25-t2v.mp4
```

### `ltx25_distilled_i2v_h100.py`

This is the standalone image-to-video entry point. It uses the repository's frozen reference image by default;
`--image-path` can override it when reproducing another 896x512, 121-frame workload:

```bash
python examples/ltx25_distilled/ltx25_distilled_i2v_h100.py \
  --model-root /path/to/LTX-2.5 \
  --image-path examples/data/ltx25/official_guitar_man.png \
  --image-frame-index 0 \
  --image-strength 1.0 \
  --prompt "A man with short gray hair plays a red electric guitar." \
  --output-path work_dirs/ltx25-i2v.mp4
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--prompt` | Script-specific | Text prompt used for video and audio generation |
| `--model-root` | Deployment-specific | Root of the required split model pack; pass it explicitly |
| `--output-path` | Required | Destination MP4; parent directories are created automatically |
| `--image-path` | `examples/data/ltx25/official_guitar_man.png` | Still image supplied to `ltx25_distilled_i2v_h100.py` |
| `--image-frame-index` | `0` (I2V) | Non-negative output-frame index for the image condition |
| `--image-strength` | `1.0` (I2V) | Image-conditioning strength in the inclusive range `[0, 1]` |
| `--height` | T2V: `1024`; I2V: `512` | Output height; must be a positive multiple of 64 |
| `--width` | T2V: `1536`; I2V: `896` | Output width; must be a positive multiple of 64 |
| `--num-frames` | `121` | Output frame count; must satisfy `num_frames = 8k + 1` |
| `--frame-rate` | `24.0` | Positive output frame rate in FPS |
| `--seed` | `42` | Random seed |
| `--video-vae` | `diff` | Video decoder: `diff` or `conv` |
| `--offload` | `cpu` | Model residency policy: `cpu` or `none` |
| `--gpu-num` | `1` | GPU count: `1`, `2`, or `4`; multi-GPU runs use Ulysses and FSDP2 |
| `--attn-impl` | `FLASH_ATTN_4` | Dense attention implementation from `AttnImplType` |

Key behavior:

- Both examples load checkpoint components into a CPU `ModuleManager`, then initialize six independently managed
  stages for text encoding, video conditioning, denoising, latent upsampling, video decoding, and audio decoding.
- The distilled pipeline runs fixed two-stage sampling and jointly generates video and audio.
- Output audio is written as stereo PCM at 48 kHz and muxed into the MP4 as AAC.
- `LTX25DistilledOutput` also exposes the final video and audio VAE latents.
- Both examples support one, two, or four GPUs; only the I2V entry point accepts a still-image condition.

## Regression

`examples/run_examples.py` registers complete two-H100 T2V and I2V workloads. Both generate 121 frames, preserve the
audio stream, and use the runner's deterministic SDPA regression backend:

```bash
python examples/run_examples.py --pipeline ltx25_distilled_t2v_2gpu --gpus 0,1
python examples/run_examples.py --pipeline ltx25_distilled_i2v_2gpu --gpus 0,1
```

Initialize or intentionally replace local baselines with `--update-baseline`. Normal regression runs require those
baselines and compare video PSNR/SSIM plus the audio stream contract and waveform similarity.

## Configuration

### Pipeline Composition

`load_ltx25_distilled_modules()` constructs the split-checkpoint components on CPU and registers them with
`ModuleManager`. `LTX25DistilledPipeline.init()` composes the six flat stage modules from manager-owned components.
The compatibility constructor `LTX25DistilledPipeline.from_model_root()` follows the same loading and stage path.

### Video VAE

`--video-vae diff` selects DiffVAE and is the formal output path. It uses NATTEN when the compatible CUDA extension
is installed and otherwise falls back to the Triton/eager implementation. `--video-vae conv` selects ConvVAE as a
compatibility alternative. The selected VAE is used consistently for image conditioning, latent-statistics
normalization around spatial upsampling, and video decoding. Both variants return RGB chunks in `[F, H, W, C]` with
values in `[0, 1]`. Image conditions are applied to the clean latent state before initial noising.

### Model Residency

`--offload cpu` is the default. It streams transformer blocks between CPU and GPU, releases other modules at phase
boundaries, and lowers peak GPU residency at the cost of transfers. Transformer weights remain in pinned CPU memory
and stream through reusable GPU buffers; the other stage-owned modules move through GPU memory sequentially. Use
`--offload none` only when the GPU has enough memory to retain modules between phases.

With `--gpu-num 2` or `--gpu-num 4`, the denoiser remains GPU-resident because FSDP2 and transformer CPU offload are
mutually exclusive. When `--offload cpu` is selected, the other five stages still follow their normal CPU-offload
lifecycle.

### Sequence Parallelism

`--gpu-num 2` and `--gpu-num 4` shard video and audio tokens across the denoising workers with Ulysses. Self-attention
and audio/video cross-attention exchange sequence and head partitions with all-to-all collectives; text context stays
replicated. The implementation pads non-divisible token counts and masks the padding before attention, then gathers
and crops model outputs before each sampler step. The 48 transformer blocks are also sharded with FSDP2. BF16 SP runs
are not bitwise-identical to single-GPU runs because the per-rank token dimensions can select different GEMM kernels.

### Attention Backend

`--attn-impl` selects a dense backend through `ModelRuntimeConfig.attention_config` and the public `telefuser.ops`
attention dispatcher. `FLASH_ATTN_4` preserves the validated single-H100 baseline. Other choices require their normal
runtime dependencies and use the dispatcher's documented fallback behavior where applicable. When SP must pad a
non-divisible sequence, only the affected masked attention calls use SDPA because the FlashAttention kernels do not
consume arbitrary additive masks.

### Request Constraints

- `height` and `width` must be positive multiples of 64.
- `num_frames` must satisfy `num_frames = 8k + 1`; examples include 1, 9, 17, and 121.
- `frame_rate` must be positive.
- `image_frame_index` must be non-negative and `image_strength` must be in `[0, 1]`.

## Performance

The formal quality and performance gates use BF16 on one H100 80 GB with PyTorch 2.11.0, CUDA 12.8, NATTEN 0.21.6,
the upstream eager DiffVAE tiling, and matching request/runtime provenance. The 1536x1024, 121-frame T2V comparison
recorded 61.90 dB PSNR and 0.999685 SSIM; the frozen 896x512, 121-frame I2V comparison recorded 62.95 dB PSNR and
0.999671 SSIM.

The timings below are synchronized end-to-end p50 seconds from five cold and five warm samples:

| Workload | Mode | Upstream cold / warm | TeleFuser cold / warm |
| --- | --- | ---: | ---: |
| T2V 1536x1024 / 121 | `offload=cpu` | 76.78 / 77.29 | 64.44 / 60.09 |
| I2V 896x512 / 121 | `offload=cpu` | 65.02 / 64.52 | 51.98 / 44.08 |
| T2V 1536x1024 / 121 | `offload=none` | 58.28 / 55.29 | 46.79 / 46.93 |
| I2V 896x512 / 121 | `offload=none` | 48.45 / 42.34 | 30.24 / 30.29 |

The no-offload TeleFuser run reserved 79.14 GB at peak. Lower resolutions and shorter valid frame counts are useful
for diagnostics but do not replace these formal gates.

## Troubleshooting

### Missing Checkpoint

The pipeline reports the first missing component and its resolved path. Compare `--model-root` with the complete
layout above; both video VAE files are currently required even when only one decoder is selected.

### Output Has No Audio

TeleFuser keeps the generated video if audio muxing fails. Confirm that `ffmpeg` is installed and available on
`PATH`:

```bash
ffmpeg -version
```

### DiffVAE Uses the Compatibility Fallback

Confirm that NATTEN is importable and includes libnatten for the active PyTorch/CUDA environment:

```bash
python -c "import torch, natten; print(torch.__version__, torch.version.cuda, natten.HAS_LIBNATTEN)"
```

Use the [NATTEN installation matrix](https://natten.org/install/) to select a matching build when the final value is
`False`.

## Notes

- The formal workload is 1536x1024, 121 frames, 24 FPS, DiffVAE, and NATTEN on one H100.
- Lower resolutions and shorter valid frame counts are useful for smoke tests but are not the formal quality baseline.
- The frozen I2V input is `examples/data/ltx25/official_guitar_man.png`, used at frame index 0 and strength 1.0.
- CUDA audio-vocoder convolution can vary slightly across replays. The validation capture tools provide
  `--deterministic-audio` for exact waveform comparisons without changing production inference behavior.
