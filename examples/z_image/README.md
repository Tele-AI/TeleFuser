# Z-Image Examples

These examples generate images with the distilled Z-Image-Turbo model through TeleFuser or a fixed Diffusers
reference path.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| Z-Image-Turbo | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | [Tongyi-MAI/Z-Image-Turbo](https://modelscope.cn/models/Tongyi-MAI/Z-Image-Turbo) | Distilled text-to-image checkpoint |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-image | Supported | TeleFuser and Diffusers reference paths |
| Multi-GPU inference | Supported | CFG and Ulysses parallelism through `--gpu_num` |
| LoRA | Unsupported | No LoRA loader is exposed |
| Quantization | Supported | The TeleFuser model path supports FP8 configuration |
| CPU offload | Unsupported | No CPU-offload option is exposed |
| Feature cache | N/A | Not used by this distilled model example |
| Server API | Supported | The TeleFuser script exposes the standard pipeline functions |

## Requirements

- GPU: one CUDA GPU; additional GPUs may be selected with `--gpu_num`
- Software: the standard TeleFuser installation; Diffusers is required for the reference script
- Input assets: none

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
/path/to/Z-Image-Turbo/
|-- transformer/
|   \-- diffusion_pytorch_model-*.safetensors
|-- vae/
|-- text_encoder/
|   \-- model-*.safetensors
|-- tokenizer/
\-- scheduler/
```

The default root is `${TF_MODEL_ZOO_PATH}/Z-Image-Turbo`.

## Quick Start

```bash
python examples/z_image/z_image_turbo_t2i_h100.py \
  --model_root /path/to/Z-Image-Turbo \
  --prompt "A watercolor landscape at sunrise" \
  --output work_dirs/z-image-turbo.png
```

The command writes the generated image to `work_dirs/z-image-turbo.png`.

## Examples

### TeleFuser Inference

#### `z_image_turbo_t2i_h100.py`

```bash
python examples/z_image/z_image_turbo_t2i_h100.py \
  --model_root /path/to/Z-Image-Turbo \
  --aspect_ratio 16:9 \
  --gpu_num 1 \
  --output work_dirs/z-image-turbo.png
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/Z-Image-Turbo` | Local model directory |
| `--aspect_ratio` | `16:9` | Output aspect ratio |
| `--gpu_num` | `1` | GPU count |
| `--output` | Generated name | Output PNG path |

### Diffusers Reference

#### `z_image_turbo_official.py`

This fixed-configuration script uses the same model directory and writes `z_image_example.png`.

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo python examples/z_image/z_image_turbo_official.py
```
