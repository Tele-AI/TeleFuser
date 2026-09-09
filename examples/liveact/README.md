# LiveAct Examples

LiveAct generates a speech-driven portrait video from one source image, one audio file, and a text prompt.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LiveAct | [Soul-AILab/LiveAct](https://huggingface.co/Soul-AILab/LiveAct) | N/A | Speech-driven video transformer, VAE, CLIP, and T5 weights |
| Chinese Wav2Vec2 Base | [TencentGameMate/chinese-wav2vec2-base](https://huggingface.co/TencentGameMate/chinese-wav2vec2-base) | N/A | Audio feature encoder |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Speech-to-video | Supported | Conditions generation on an image and audio file |
| Multi-GPU inference | Supported | Ulysses sequence parallelism for DiT and VAE through `--gpu_num` |
| Quantization | Supported | The checked-in configuration enables FP8 Linear layers |
| Compilation | Supported | DiT and VAE compilation are enabled in `PPL_CONFIG` |
| CPU offload | Unsupported | No CPU-offload mode is exposed |
| Feature cache | Unsupported | No feature cache is configured |
| Server API | Partial | Standard loader and run functions exist, but no explicit pipeline contract is declared |

## Requirements

- GPU: one CUDA GPU, or multiple compatible GPUs for Ulysses sequence parallelism
- Software: the standard TeleFuser installation, a compatible `tf-kernel` build for FP8, and `ffmpeg`
- Input assets: a PIL-readable portrait image and an audio file supported by the audio pipeline

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup). Build and install
`tf-kernel` with the Makefile matching the visible GPU architecture.

## Model Directory

```text
${TF_MODEL_ZOO_PATH}/
|-- LiveAct/
|   |-- diffusion_pytorch_model-*.safetensors
|   |-- Wan2.1_VAE.pth
|   |-- models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
|   \-- models_t5_umt5-xxl-enc-bf16.pth
\-- chinese-wav2vec2-base/
```

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Quick Start

```bash
python examples/liveact/liveact_s2v_h100.py \
  --model_root "$TF_MODEL_ZOO_PATH/LiveAct" \
  --image_path /path/to/portrait.png \
  --audio_path /path/to/speech.wav \
  --prompt "A person talking naturally" \
  --output work_dirs/liveact.mp4
```

The command writes a 20 FPS speech-driven video with the source audio to `work_dirs/liveact.mp4`.

## Examples

### Speech-Driven Video

#### `liveact_s2v_h100.py`

```bash
python examples/liveact/liveact_s2v_h100.py \
  --gpu_num 2 \
  --model_root "$TF_MODEL_ZOO_PATH/LiveAct" \
  --image_path /path/to/portrait.png \
  --audio_path /path/to/speech.wav \
  --height 720 --width 416 --fps 20 \
  --output work_dirs/liveact-sp2.mp4
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/LiveAct` | LiveAct checkpoint directory |
| `--image_path` | Legacy local sample path | Pass an existing source image explicitly |
| `--audio_path` | Legacy local sample path | Pass an existing source audio file explicitly |
| `--gpu_num` | `1` | Multi-GPU runs enable DiT and VAE sequence parallelism |
| `--height`, `--width` | `720`, `416` | Output dimensions |
| `--fps` | `20` | Output frame rate |
| `--output` | Generated name | Output MP4 path |

## Notes

- The default image and audio assets are not packaged; pass both input paths explicitly.
- The default attention, quantization, and compile settings target compatible NVIDIA GPUs.
