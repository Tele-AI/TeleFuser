# LTX 2.3 Examples

This example performs two-stage image-to-video generation with the LTX 2.3 22B checkpoint and writes one MP4
containing generated video and audio.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LTX 2.3 22B | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | [Lightricks/LTX-2.3](https://modelscope.cn/models/Lightricks/LTX-2.3) | Main transformer, spatial upsampler, and distilled stage-two LoRA |
| Gemma 3 12B | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | N/A | Text encoder |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Image-to-video with audio | Supported | Produces an MP4 with an AAC audio track |
| Two-stage denoising | Supported | Base generation, latent upsampling, then refinement |
| Multi-GPU inference | Supported | Ulysses sequence parallelism and FSDP through `--gpu_num` |
| LoRA | Supported | Uses the stage-two distilled LoRA when present |
| VAE parallelism | Supported | Enabled with multi-GPU execution |
| Ray VAE actor | Unsupported | The checked-in script does not expose this mode |
| Server API | Unsupported | The script is an offline CLI entry point |

## Requirements

- GPU: two CUDA GPUs by default; use another supported count through `--gpu_num`
- Software: the standard TeleFuser installation and an `ffmpeg` executable on `PATH`
- Input assets: a PIL-readable source image supplied with `--image_path`

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
|-- LTX-2.3/
|   |-- ltx-2.3-22b-dev.safetensors
|   |-- ltx-2.3-spatial-upscaler-x2-1.0.safetensors
|   \-- ltx-2.3-22b-distilled-lora-384.safetensors
\-- gemma-3-12b-it-qat-q4_0-unquantized/
    |-- model-00001-of-00005.safetensors
    |-- ...
    \-- model-00005-of-00005.safetensors
```

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Quick Start

```bash
TELEAI_EXAMPLE_OUTPUT_DIR=work_dirs \
python examples/ltx_video/ltx23_22b_image_to_video_two_stage_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/LTX-2.3" \
  --image_path /path/to/input.png \
  --prompt "A slow camera move through a sunlit garden."
```

The command writes `work_dirs/ltx23_22b_image_to_video_two_stage_h100_2gpu.mp4`.

## Examples

### Two-Stage Image-To-Video

#### `ltx23_22b_image_to_video_two_stage_h100.py`

```bash
TELEAI_EXAMPLE_OUTPUT_DIR=work_dirs \
python examples/ltx_video/ltx23_22b_image_to_video_two_stage_h100.py \
  --gpu_num 4 \
  --model_root "$TF_MODEL_ZOO_PATH/LTX-2.3" \
  --image_path /path/to/input.png \
  --resolution 1080p \
  --prompt "A cinematic outdoor scene with natural motion."
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--gpu_num` | `2` | Number of inference workers |
| `--image_path` | Legacy local sample path | Pass an existing image explicitly |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/LTX-2.3` | LTX checkpoint directory |
| `--resolution` | `1080p` | One of `720p`, `1080p`, `2k`, or `4k` |
| `--num_inference_steps` | `30` | Steps in each denoising stage |
| `--num_frames` | `121` | Generated frame count |
| `--seed` | `42` | Random seed |

## Notes

- The default input asset is not packaged, so pass `--image_path` explicitly.
- The stage-two LoRA is optional at load time but recommended for the intended output quality.
