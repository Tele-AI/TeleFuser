# FlashVSR Examples

FlashVSR performs streaming video super-resolution from a low-resolution input video and writes the restored frames
to an MP4 file.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| FlashVSR v1.1 BF16 | [lzx1413/FlashVSR-v1.1-BF16](https://huggingface.co/lzx1413/FlashVSR-v1.1-BF16) | [lzx1413/FlashVSR-v1.1-BF16](https://modelscope.cn/models/lzx1413/FlashVSR-v1.1-BF16) | Streaming video restoration checkpoint |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Streaming video super-resolution | Supported | Processes the input in stateful chunks |
| Multi-GPU inference | Supported | Ulysses sequence parallelism and FSDP through `--gpu_num` |
| Quantization | Unsupported | The example loads the released BF16 checkpoint |
| CPU offload | Unsupported | No CPU-offload option is exposed |
| Feature cache | N/A | Not used by the streaming restoration pipeline |
| Server API | Unsupported | The script is an offline CLI entry point |

## Requirements

- GPU: one or more CUDA GPUs; the example has been exercised on NVIDIA RTX 5090 GPUs
- Software: the standard TeleFuser installation, `ffmpeg`, `tf-kernel`, and Block-Sparse-Attention
- Input assets: a readable input video

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup). Build `tf-kernel`
with its repository Makefile and install Block-Sparse-Attention according to its upstream instructions.

## Model Directory

```text
/path/to/FlashVSR-v1.1-BF16/
|-- flashvsr11_dit_streaming_dmd_5dc619.safetensors
\-- TCDecoder.ckpt
```

The default root is `${TF_MODEL_ZOO_PATH}/FlashVSR-v1.1-BF16`; pass `--model_root` to use another location.

## Quick Start

```bash
python examples/flashvsr/flashvsr_stream.py \
  --input_video /path/to/input.mp4 \
  --scale 4 \
  --model_root /path/to/FlashVSR-v1.1-BF16 \
  --output work_dirs/flashvsr-restored.mp4
```

The command restores the input at 4x scale and writes `work_dirs/flashvsr-restored.mp4`.

## Examples

### Video Restoration

#### `flashvsr_stream.py`

Use this entry point for stateful, chunked restoration of a local video.

```bash
python examples/flashvsr/flashvsr_stream.py \
  -i /path/to/input.mp4 -s 4 --gpu_num 2 \
  --model_root /path/to/FlashVSR-v1.1-BF16 \
  -o work_dirs/flashvsr-restored-sp2.mp4
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `-i`, `--input_video` | Required | Input low-resolution video |
| `-s`, `--scale` | `4` | Upscaling factor |
| `--height`, `--width` | Auto-detected | Optional input dimensions |
| `--gpu_num` | `1` | Number of sequence-parallel workers |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/FlashVSR-v1.1-BF16` | Checkpoint directory |
| `-o`, `--output` | Generated name | Output MP4 path |
| `--seed` | `0` | Random seed |

## Notes

- `local_range=9` emphasizes sharp detail; `local_range=11` favors temporal stability.
