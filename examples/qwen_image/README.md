# Qwen-Image Examples

These examples provide text-to-image generation, image editing, quantized inference, and feature-cache calibration
with Qwen-Image checkpoints, including the unified Qwen-Image 2.1 pipeline.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| Qwen-Image | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | [Qwen/Qwen-Image](https://modelscope.cn/models/Qwen/Qwen-Image) | Text-to-image base weights |
| Qwen-Image 2.1 | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | [Qwen/Qwen-Image-2.1](https://modelscope.cn/models/Qwen/Qwen-Image-2.1) | Native generation, editing, and reference examples |
| Qwen-Image-Lightning | [Qwen/Qwen-Image-Lightning](https://huggingface.co/Qwen/Qwen-Image-Lightning) | [Qwen/Qwen-Image-Lightning](https://modelscope.cn/models/Qwen/Qwen-Image-Lightning) | Distilled LoRA and FP8 variants |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | [Qwen/Qwen-Image-Edit](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) | Image editing |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-image | Supported | BF16, Lightning LoRA, TeleFuser FP8, and NF4 examples |
| Image editing | Supported | TeleFuser and Diffusers reference paths |
| Qwen-Image 2.1 reference generation | Supported | One to ten condition images in the CLI example |
| Multi-GPU inference | Supported | CFG and Ulysses parallelism on scripts exposing `--gpu_num` |
| LoRA | Supported | Lightning LoRA example |
| Quantization | Supported | Pre-quantized FP8, online TeleFuser FP8, and NF4 |
| CPU offload | Supported | Used by native Qwen pipelines |
| Feature cache | Supported | Separate T2I and edit calibration tools |
| Server API | Supported | Native examples expose standard pipeline functions |

Qwen-Image 2.1 uses a 64-channel latent VAE, Qwen3-VL prompt encoding, and a
single-stream block-causal transformer. The native TeleFuser path uses 40 steps
by default with classifier-free guidance disabled. The same stage pipeline handles
text-to-image, image editing, and reference-guided generation.

## Requirements

- GPU: one H100-class CUDA GPU for the documented configurations; use supported multi-GPU degrees as needed
- Software: the standard TeleFuser installation plus Diffusers standalone Qwen-Image 2.1 VAE classes and Transformers Qwen3-VL classes
- Input assets: T2I requires no input asset; edit uses `examples/data/edit2511input.png`, and reference generation
  uses five official demo images under `examples/data/qwen_image_21/`

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

The TeleFuser entry point implements the 2.1 DiT locally. It uses Diffusers only for the standalone VAE class and Transformers for the Qwen3-VL text encoder and processor.

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
|-- Qwen-Image-2512/
|   |-- transformer/
|   |-- vae/
|   |-- text_encoder/
|   \-- tokenizer/
|-- Qwen-Image-2.1/
|   |-- transformer/
|   |-- vae/
|   |-- text_encoder/
|   |-- processor/
|   \-- scheduler/
|-- Qwen-Image-2512-Lightning/
|   \-- Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors
|-- Qwen-Image-Edit-2509/
\-- Qwen-Image-Edit-2511/
    |-- transformer/
    |-- vae/
    |-- text_encoder/
    \-- tokenizer/
```

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Quick Start

```bash
python examples/qwen_image/qwen_image_t2i_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --prompt "A sunset over snow-covered mountains" \
  --output work_dirs/qwen-image.png
```

The command writes the generated image to `work_dirs/qwen-image.png`.

The 2.1 standard example uses the same regression prompt, negative prompt, seed, and default `16:9` aspect ratio
as the existing Qwen-Image example. These defaults are declared in the Python scripts. Both examples map `16:9`
to **1664×928** pixels.

Select physical GPU 3 by masking it into the process (it is then visible as
`cuda:0`):

```bash
CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_t2i_h100.py \
  --model_root /hhb-data/aigc/model_zoo/Qwen-Image-2.1 \
  --output_path work_dirs/qwen-image-2.1.png
```

## Examples

### Text-To-Image

#### `qwen_image_21_t2i_h100.py`

This standard example loads Qwen-Image 2.1 through TeleFuser for text-to-image
generation.

```bash
CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_t2i_h100.py \
  --model_root /hhb-data/aigc/model_zoo/Qwen-Image-2.1 \
  --prompt "A neon shop sign that reads QWEN IMAGE 2.1 in the rain" \
  --output_path work_dirs/qwen-image-2.1.png
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--model_root` | `/hhb-data/aigc/model_zoo/Qwen-Image-2.1` | Diffusers model directory |
| `--gpu_num` | `1` | Qwen-Image 2.1 currently uses one GPU |
| `--aspect_ratio` | `16:9` | Uses the existing Qwen-Image size mapping; default is 1664×928 |
| `--height`, `--width` | unset | Optional overrides for the mapped output dimensions |
| `--output_path` | generated example name | PNG output path |

#### `qwen_image_t2i_h100.py`

```bash
python examples/qwen_image/qwen_image_t2i_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --gpu_num 2 --aspect_ratio 16:9
```

#### `qwen_image_t2i_lora_h100.py`

Uses the Lightning LoRA configured in the script:

```bash
python examples/qwen_image/qwen_image_t2i_lora_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --prompt "A portrait photograph"
```

#### `qwen_image_t2i_lightning_fp8_h100.py`

Loads the pre-quantized Lightning FP8 transformer configured in `PPL_CONFIG`.

```bash
python examples/qwen_image/qwen_image_t2i_lightning_fp8_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --prompt "A mountain landscape"
```

#### `qwen_image_t2i_telefuser_fp8_h100.py`

```bash
python examples/qwen_image/qwen_image_t2i_telefuser_fp8_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --prompt "A ceramic vase in a studio"
```

#### `qwen_image_t2i_telefuser_nf4_h100.py`

```bash
python examples/qwen_image/qwen_image_t2i_telefuser_nf4_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --prompt "A ceramic vase in a studio"
```

### Image Editing

#### `qwen_image_edit_plus_h100.py`

```bash
python examples/qwen_image/qwen_image_edit_plus_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-Edit-2511" \
  --image_path /path/to/input.jpg \
  --prompt "Change the background to a beach" \
  --output work_dirs/qwen-image-edit.png
```

#### `qwen_image_21_edit_h100.py`

The native 2.1 edit example uses the existing Qwen-Image edit test image. Without
`--height` or `--width`, output dimensions follow the input image aspect ratio
at approximately one megapixel.

```bash
CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_edit_h100.py \
  --image_path examples/data/edit2511input.png \
  --output_path work_dirs/qwen-image-2.1-edit.png
```

### Reference-Guided Generation

#### `qwen_image_21_reference_h100.py`

The default test uses the official Qwen-Image 2.1 ["Outfit styling (5 images)" demo
case](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1/blob/main/examples/cases.json). Its prompt and five input
images are included in the script and `examples/data/qwen_image_21/` in the official order: model, down jacket,
Mary Jane shoes, handbag, and fur hat. Run it without extra arguments, or pass `--image_path` once per custom
reference image. The output aspect ratio follows the last reference unless dimensions are supplied.

```bash
CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_reference_h100.py \
  --output_path work_dirs/qwen-image-2.1-reference.png
```

### Cache Calibration

#### `qwen_image_cache_calibrate.py`

```bash
python examples/qwen_image/qwen_image_cache_calibrate.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-2512" \
  --output_path work_dirs/qwen-image-cache.json
```

#### `qwen_image_edit_plus_cache_calibrate.py`

```bash
python examples/qwen_image/qwen_image_edit_plus_cache_calibrate.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-Edit-2511" \
  --image_path /path/to/input.jpg \
  --output_path work_dirs/qwen-image-edit-cache.json
```

### Diffusers References

#### `qwen_image_t2i_official.py`

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo python examples/qwen_image/qwen_image_t2i_official.py
```

#### `qwen_image_edit_plus_official.py`

```bash
python examples/qwen_image/qwen_image_edit_plus_official.py \
  --model_root "$TF_MODEL_ZOO_PATH/Qwen-Image-Edit-2509" \
  --image_path /path/to/input.jpg \
  --prompt "Replace the screen text" \
  --output work_dirs/qwen-image-edit-official.png
```

This reference keeps the official fixed sampling parameters while making model, input, prompt, and output paths
explicit.

## Serving

The standard entry points expose `t2i` and `i2i` service contracts. The edit
example uses the existing `i2i` service task because `telefuser serve --task`
does not offer a separate `edit` value.
The existing service schema supplies one image through `first_image_path` for
`edit` and `i2i`; use the reference CLI for multiple images.

```bash
CUDA_VISIBLE_DEVICES=3 telefuser serve examples/qwen_image/qwen_image_21_t2i_h100.py \
  --task t2i --port 8091
```

To serve either image-conditioned example, select its script with the `i2i`
task. For example:

```bash
CUDA_VISIBLE_DEVICES=3 telefuser serve examples/qwen_image/qwen_image_21_edit_h100.py \
  --task i2i --port 8092
```

Upload the source image through `/v1/tasks/form` with `first_image_file`.

The service loads one pipeline replica on the selected GPU and writes each
request result to the service output directory.

## Configuration

Supported aspect ratios include `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, and `2:3`. The exact
resolution mapping is defined in each entry point.

## Troubleshooting

If loading fails with `safetensors ... invalid JSON in header`, one or more
checkpoint shards are incomplete. Verify the four files under
`text_encoder/` and copy them again before starting the service.
