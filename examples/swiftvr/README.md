# SwiftVR Examples

SwiftVR restores videos with a stateful 24-frame causal protocol and writes the restored frames to an MP4 file.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| SwiftVR | [H-oliday/SwiftVR](https://huggingface.co/H-oliday/SwiftVR) | N/A | Video restoration checkpoint |

The integration follows upstream SwiftVR commit `5ca168cef6ca7200f135fdfea85e5e13d12c5b53`.

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Stateful video restoration | Supported | Uses the released 24-frame causal chunk protocol |
| Multi-GPU inference | Supported | Ulysses degrees must divide 40 attention heads |
| Stage parallelism | Supported | Three GPUs split ReAE encode, DiT, and ReAE decode |
| Quantization | Partial | TorchAO FP8 and tf-kernel FP8 are opt-in quality/performance tradeoffs |
| Compilation | Supported | Single-GPU DiT compilation through `--compile_dit` |
| CPU offload | Unsupported | No CPU-offload option is exposed |
| Server API | Unsupported | The current service protocol has no video-input transport |

## Requirements

- GPU: one CUDA GPU; multi-GPU Ulysses supports degrees dividing 40, and stage parallelism requires three GPUs
- Software: the standard TeleFuser installation and `ffmpeg`; optional FP8 paths require their matching backend
- Input assets: a readable low-resolution input video supplied with `--input_video`

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup).

## Model Directory

```text
/path/to/SwiftVR/
|-- reae.safetensors
|-- prompt_embedding.safetensors
\-- transformer/
    |-- config.json
    \-- diffusion_pytorch_model.safetensors
```

## Quick Start

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /path/to/SwiftVR \
  --input_video /path/to/input.mp4 \
  --height 360 --width 640 --scale 3 \
  --output work_dirs/swiftvr-restored.mp4
```

The command restores a 640x360 input to 1920x1080 and writes `work_dirs/swiftvr-restored.mp4`.

## Examples

### Offline Restoration

#### `swiftvr_restore_h100.py`

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /path/to/SwiftVR \
  --input_video /path/to/input.mp4 \
  --attn_impl TORCH_SDPA --compile_dit \
  --output work_dirs/swiftvr-compiled.mp4
```

For two-way Ulysses:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /path/to/SwiftVR \
  --input_video /path/to/input.mp4 --gpu_num 2
```

For three-stage execution:

```bash
python examples/swiftvr/swiftvr_restore_h100.py \
  --model_root /path/to/SwiftVR \
  --input_video /path/to/input.mp4 \
  --gpu_num 3 --enable_stage_parallel --stage_devices 0,1,2
```

Ulysses and stage parallelism are mutually exclusive. Compilation is a single-GPU optimization. In stage-parallel
mode, only one active stream session is supported because worker stages retain causal state.

## Configuration

`SwiftVRPipeline.stream()` accepts uint8 `[T,H,W,3]` tensors and returns PIL RGB frames. A partial chunk may return
an empty list until enough causal context is available. The CLI performs shape-specific warmup before timing.

## Notes

- The default input asset is not packaged; pass `--input_video` explicitly.
- See the [SwiftVR integration guide](../../docs/en/swiftvr.md) for architecture and measured performance.
