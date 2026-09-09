# LongCat-Video Examples

These examples provide text-to-video, image-to-video, video continuation, unified generation, and refinement with
LongCat-Video.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LongCat-Video | [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video) | [meituan-longcat/LongCat-Video](https://modelscope.cn/models/meituan-longcat/LongCat-Video) | Base video generation |
| Wan2.1 VAE | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | [Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B) | Video encoder and decoder |
| RIFE v4.26 | [RIFEv4.26](https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip) | N/A | Optional frame interpolation |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text-to-video | Supported | Standard and refinement entry points |
| Image-to-video | Supported | Requires a source image |
| Video continuation | Supported | Requires a source video |
| Multi-GPU inference | Supported | CFG and Ulysses parallelism through `--gpu_num` |
| LoRA | Supported | Distillation and refinement LoRAs |
| Quantization | Unsupported | No quantized example is checked in |
| CPU offload | Supported | Base scripts use model offload for memory control |
| Feature cache | N/A | The implementation uses its model-specific KV cache |
| Server API | Supported | Generation examples expose standard pipeline functions |

## Requirements

- GPU: one CUDA GPU for base examples; additional GPUs may be selected through `--gpu_num`
- Software: the standard TeleFuser installation; RIFE is required only for frame interpolation
- Input assets: a readable image for I2V or a readable video for continuation

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
|-- LongCat-Video/
|   |-- diffusion_pytorch_model-*.safetensors
|   |-- text_encoder/
|   |-- tokenizer/
|   \-- lora/
|       \-- refinement_lora.safetensors
|-- Wan2.1-T2V-1.3B/
|   \-- Wan2.1_VAE.pth
\-- RIFEv4.26_0921/
    \-- flownet.pkl
```

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Quick Start

```bash
TELEAI_EXAMPLE_OUTPUT_DIR=work_dirs \
python examples/longcat_video/longcat_text_to_video.py \
  --model_root "$TF_MODEL_ZOO_PATH/LongCat-Video" \
  --prompt "A boat sailing across a calm ocean"
```

The command writes `work_dirs/longcat_text_to_video_1gpu.mp4`.

## Examples

### Text-To-Video

#### `longcat_text_to_video.py`

```bash
python examples/longcat_video/longcat_text_to_video.py \
  --model_root "$TF_MODEL_ZOO_PATH/LongCat-Video" \
  --resolution 720p --aspect_ratio 16:9 \
  --prompt "A wide landscape with slow camera motion"
```

### Image-To-Video

#### `longcat_image_to_video.py`

```bash
python examples/longcat_video/longcat_image_to_video.py \
  --model_root "$TF_MODEL_ZOO_PATH/LongCat-Video" \
  --image_path /path/to/input.jpg \
  --prompt "Natural motion in the scene"
```

### Video Continuation

#### `longcat_video_continue.py`

```bash
python examples/longcat_video/longcat_video_continue.py \
  --model_root "$TF_MODEL_ZOO_PATH/LongCat-Video" \
  --video_path /path/to/input.mp4 \
  --prompt "Continue the scene naturally"
```

### Unified Tasks

#### `longcat_video_unify.py`

Use the unified pipeline when one loaded model must serve multiple generation modes:

```bash
python examples/longcat_video/longcat_video_unify.py \
  --model_root "$TF_MODEL_ZOO_PATH/LongCat-Video" \
  --task t2v --prompt "A sunset over the sea"
```

### Refinement

#### `longcat_text_to_video_refine.py`

```bash
python examples/longcat_video/longcat_text_to_video_refine.py \
  --model_root "$TF_MODEL_ZOO_PATH/LongCat-Video" \
  --height 480 --width 832 \
  --refine_height 720 --refine_width 1280
```

Key behavior:

- Base generation supports one or more GPUs through `--gpu_num`.
- Refinement can enable block-sparse attention and temporal extension.
