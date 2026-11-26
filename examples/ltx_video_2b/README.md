# LTX-Video 2B Examples

Generate text-to-video with the Diffusers-format LTX-Video 2B checkpoint through a TeleFuser pipeline. The RTX 5090 path uses native dynamic-activation W4A4 NVFP4 for the transformer FFN GEMMs.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| LTX-Video 2B | [Lightricks/LTX-Video](https://huggingface.co/Lightricks/LTX-Video) | N/A | Text-to-video transformer, T5 encoder and VAE |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text to video | Supported | Official Diffusers conditioning and sampler |
| NVFP4 | Supported | Real W4A4 compute on Blackwell; quality profile quantizes FFNs |
| BF16 | Supported | Official sequential model CPU offload baseline |
| Multi-GPU / LoRA / feature cache / server API | Unsupported | Not exposed by this integration |

## Requirements

- GPU: NVIDIA Blackwell with NVFP4 support; tested on one 24 GiB RTX 5090.
- Software: TeleFuser CUDA extra, including `torchao==0.17.0`.
- Input assets: none.

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup), including the CUDA extra.

## Model Directory

```text
ltx-video-2b/
|-- transformer/
|-- text_encoder/
|-- tokenizer/
|-- scheduler/
\-- vae/
```

## Quick Start

```bash
python examples/ltx_video_2b/ltx_video_2b_t2v_nvfp4_5090.py \
  --model_root /path/to/ltx-video-2b \
  --output_path work_dirs/ltx-video-2b-nvfp4.mp4
```

The command writes the official 161-frame, 512x768, 50-step workload at 24 fps.

## Examples

### Text-to-video

#### `ltx_video_2b_t2v_nvfp4_5090.py`

```bash
python examples/ltx_video_2b/ltx_video_2b_t2v_nvfp4_5090.py \
  --model_root /path/to/ltx-video-2b --nvfp4_profile quality

python examples/ltx_video_2b/ltx_video_2b_t2v_nvfp4_5090.py \
  --model_root /path/to/ltx-video-2b --quantization bf16
```

| Option | Default | Description |
| --- | --- | --- |
| `--quantization` | `nvfp4` | `nvfp4` or the BF16 CPU-offload baseline |
| `--nvfp4_profile` | `quality` | `quality` keeps attention/conditioning in BF16; `full` quantizes all Linear layers |
| `--num_frames` | `161` | Must be `8*k+1` |
| `--gpu_num` | `1` | Only single-GPU execution is supported |

The quality profile keeps T5 in BF16, encodes the prompts once, then releases T5 from GPU. The NVFP4 transformer and VAE remain resident; there is no per-block CPU offload during denoising.

## Performance

Measured on one NVIDIA GeForce RTX 5090 Laptop GPU (24 GiB), PyTorch 2.14.0+cu132, CUDA 13.2 and TorchAO 0.17.0. Both rows use the official prompt, seed 123, 512x768, 161 frames, 50 steps, CFG 3.0 and the same VAE decode settings. Time includes prompt encoding, denoising and VAE decode; it excludes model loading and MP4 encoding.

| Configuration | GPUs | Resolution | Frames | Steps | Time (s) | Generated FPS | Peak VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 official model CPU offload | 1 | 768x512 | 161 | 50 | 80.88 | 1.99 | 8.92 |
| TeleFuser NVFP4 quality, resident DiT+VAE | 1 | 768x512 | 161 | 50 | 65.63 | 2.45 | 12.06 |

NVFP4 is 1.23x faster for this end-to-end workload. Neither result is real-time at 24 fps. BF16 reports lower peak allocated VRAM because the official baseline moves complete model stages between CPU and GPU; NVFP4 intentionally keeps the quantized transformer and VAE resident to avoid denoising-step PCIe transfers.

Reproduce the NVFP4 row:

```bash
python tools/validation/benchmark_ltx_nvfp4.py \
  --model-dir /path/to/ltx-video-2b \
  --quantization nvfp4 --nvfp4-profile quality \
  --output-dir outputs/ltx-telefuser-nvfp4
```

## Configuration

```python
from telefuser.pipelines.ltx_video_2b import LTXVideo2BPipeline

pipe = LTXVideo2BPipeline.from_pretrained("/path/to/ltx-video-2b", quantization="nvfp4", nvfp4_profile="quality")
frames = pipe("A cinematic close-up of two women talking", seed=123)
```

## Notes

- Quality NVFP4 intentionally leaves temporal attention, cross-attention, conditioning and output projections in BF16. Full W4A4 caused visible frame-to-frame illumination instability in validation.
- `mean luminance deviation` is not an error relative to BF16. It is each frame's mean luminance minus that video's temporal mean, so a BF16 video with real lighting or camera changes is expected to have non-zero values. The less ambiguous name is **temporal mean-luminance variation**.
