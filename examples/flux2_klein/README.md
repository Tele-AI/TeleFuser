# FLUX.2 Klein Examples

These examples generate images with the FLUX.2 Klein 9B model through either TeleFuser or the official Diffusers
pipeline.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| FLUX.2 Klein 9B | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | [black-forest-labs/FLUX.2-klein-9B](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-9B) | Four-step distilled text-to-image model |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-image | Supported | TeleFuser and Diffusers reference paths |
| Multi-GPU inference | Partial | The TeleFuser script accepts `--gpu_num`; use only validated degrees |
| LoRA | Unsupported | No LoRA option is exposed |
| Quantization | Unsupported | The examples use BF16 weights |
| CPU offload | Unsupported | No CPU-offload option is exposed |
| Feature cache | Unsupported | No cache configuration is exposed |
| Server API | Supported | The TeleFuser example exposes the standard pipeline functions |

## Requirements

- GPU: one H100 or A100 80 GB is recommended for the 9B BF16 model
- Software: the standard TeleFuser installation; Diffusers is required for the official comparison script
- Input assets: none

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
/path/to/FLUX.2-klein-9B/
|-- transformer/
|-- vae/
|-- text_encoder/
\-- tokenizer/
```

The TeleFuser script expects this local Diffusers-style directory. The official script also accepts a Hugging Face
model ID through `--model_id`.

## Quick Start

```bash
TELEAI_EXAMPLE_OUTPUT_DIR=work_dirs \
python examples/flux2_klein/flux2_klein_text_to_image_h100.py \
  --model_root /path/to/FLUX.2-klein-9B \
  --prompt "A sunlit mountain lake"
```

The command writes `work_dirs/flux2_klein_text_to_image_h100.png`.

## Examples

### TeleFuser Inference

#### `flux2_klein_text_to_image_h100.py`

```bash
python examples/flux2_klein/flux2_klein_text_to_image_h100.py \
  --model_root /path/to/FLUX.2-klein-9B \
  --prompt "A detailed architectural photograph"
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/FLUX.2-klein-9B` | Local model directory |
| `--gpu_num` | `1` | GPU count |
| `--height`, `--width` | `1024` | Output dimensions, divisible by 16 |
| `--seed` | `42` | Random seed |

### Diffusers Reference

#### `flux2_klein_text_to_image_official.py`

```bash
python examples/flux2_klein/flux2_klein_text_to_image_official.py \
  --model_id /path/to/FLUX.2-klein-9B \
  --prompt "A detailed architectural photograph"
```
