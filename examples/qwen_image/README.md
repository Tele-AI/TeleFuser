# Qwen-Image Examples

These examples provide text-to-image generation, image editing, quantized inference, and feature-cache calibration
with Qwen-Image checkpoints.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| Qwen-Image | [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) | [Qwen/Qwen-Image](https://modelscope.cn/models/Qwen/Qwen-Image) | Text-to-image base weights |
| Qwen-Image-Lightning | [Qwen/Qwen-Image-Lightning](https://huggingface.co/Qwen/Qwen-Image-Lightning) | [Qwen/Qwen-Image-Lightning](https://modelscope.cn/models/Qwen/Qwen-Image-Lightning) | Distilled LoRA and FP8 variants |
| Qwen-Image-Edit | [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | [Qwen/Qwen-Image-Edit](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) | Image editing |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-image | Supported | BF16, Lightning LoRA, TeleFuser FP8, and NF4 examples |
| Image editing | Supported | TeleFuser and Diffusers reference paths |
| Multi-GPU inference | Supported | CFG and Ulysses parallelism on scripts exposing `--gpu_num` |
| LoRA | Supported | Lightning LoRA example |
| Quantization | Supported | Pre-quantized FP8, online TeleFuser FP8, and NF4 |
| CPU offload | Supported | Used by native Qwen pipelines |
| Feature cache | Supported | Separate T2I and edit calibration tools |
| Server API | Supported | Native examples expose standard pipeline functions |

## Requirements

- GPU: one H100-class CUDA GPU for the documented configurations; use supported multi-GPU degrees as needed
- Software: the standard TeleFuser installation; Diffusers is required for the official reference scripts
- Input assets: a readable image for editing; T2I requires no input asset

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
|-- Qwen-Image-2512/
|   |-- transformer/
|   |-- vae/
|   |-- text_encoder/
|   \-- tokenizer/
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

## Examples

### Text-To-Image

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

## Configuration

Supported aspect ratios include `1:1`, `16:9`, `9:16`, `4:3`, `3:4`, `3:2`, and `2:3`. The exact
resolution mapping is defined in each entry point.
