# PAN2 Examples

Generate a video from text or a first frame with PAN2, using TeleFuser model loading, stages and distributed workers.
The transformer supports the existing Ulysses SP, TP, Sol-Attn and tf-kernel FP8 paths.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| PAN2 | [IFM/PAN2](https://huggingface.co/IFM/PAN2) | N/A | Transformer, provided Qwen3.5 text encoder, Wan2.2 VAE and tokenizer |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text to video / image to video | Supported | One prompt and optional first RGB frame |
| Sequence parallel | Supported | Ulysses; odd video/text lengths are padded and compacted before attention |
| Tensor parallel | Supported | Joint attention, refiner, FFNs and modulation projections; combines with SP |
| Sol-Attn | Supported | Joint layers; text-only refiner stays dense; optional FP8 attention |
| FP8 linear quantization | Supported | Existing tf-kernel FP8 backend, FP32 modulation retained |
| CPU offload | Supported | Model CPU offload including SP/TP shards and repeated requests |
| LoRA / feature cache / compilation / FSDP / Ring SP | Unsupported | Not exposed by this integration |
| Server API | Unsupported | Local pipeline and command line example only |

Numerical checks cover SP2, TP2 and SP2×TP2, including FP8+Sol-Attn and repeated CPU offload.
See [validation notes](../../docs/en/h3_image_pan2_validation.md) for full checkpoint verification status.

## Requirements

- An 80GB CUDA GPU for BF16 transformer inference with sequential model CPU offload; TP can reduce parameter memory.
- Sufficient host memory for all components and spawned worker storage.
- Transformers with `Qwen3_5TextModel` (tested 5.14.1), Diffusers with `AutoencoderKLWan`, Pillow and an MP4 writer.
- A compatible, locally provisioned tf-kernel wheel for Sol-Attn and FP8.

Install TeleFuser through the [development setup](../../CONTRIBUTING.md#development-setup).
Validation used PyTorch 2.11.0+cu130 and the Diffusers revision recorded in the validation notes.
The Qwen3.5 text encoder works with Transformers' PyTorch fallback when optional FLA kernels are absent.

## Model Directory

```text
PAN2/
|-- transformer/config.json
|-- transformer/diffusion_pytorch_model-*-of-*.safetensors
|-- text_encoder/config.json
|-- text_encoder/model-*-of-*.safetensors
|-- tokenizer/                         # complete released tokenizer assets
|-- scheduler/scheduler_config.json
|-- vae/config.json
\-- vae/diffusion_pytorch_model.safetensors
```

Use the **provided text-only encoder** in `IFM/PAN2`; a separate complete Qwen3.5 multimodal checkpoint is unnecessary.

## Quick Start

```bash
python examples/pan2/generate.py \
  --model_root /path/to/PAN2 \
  --prompt "A red fox walks through a snowy forest, tracking camera, natural light." \
  --output_path work_dirs/pan2.mp4
```

The command writes a 121-frame RGB MP4 at 24 fps, 704×1248, using 50 steps, CFG 2.5 and the released flow shift 9.
PAN2 was trained with structured captions; the official repository provides richer structured prompt examples.
The example encodes the supplied string directly and does not call an external captioning service.

## Examples

### Video generation

#### `generate.py`

Pass `--image` for first-frame conditioning. Frame counts must be `4*k+1`; both spatial dimensions must divide 32.

```bash
python examples/pan2/generate.py \
  --model_root /path/to/PAN2 --image inputs/fox.png \
  --prompt "The fox walks forward through the snow." --output_path work_dirs/pan2-i2v.mp4

python examples/pan2/generate.py \
  --model_root /path/to/PAN2 --prompt "A red fox walks through snow." \
  --device_ids 0 1 2 3 --sp 2 --tp 2 --fp8 --sol --sol_fp8 \
  --num_frames 241 --output_path work_dirs/pan2-10s.mp4
```

| Option | Default | Description |
| --- | --- | --- |
| `--device_ids` | `0` | Visible device IDs; count must equal SP×TP |
| `--sp`, `--tp` | `1`, `1` | Ulysses and tensor parallel degrees |
| `--fp8` | off | tf-kernel FP8 linear quantization |
| `--sol`, `--sol_fp8` | off | Sparse joint attention and optional FP8 Q/K/V |
| `--height`, `--width` | `704`, `1248` | Positive multiples of 32 |
| `--num_frames` | `121` | `4*k+1`; 241 frames span 10 seconds at 24 fps |
| `--num_inference_steps` | `50` | Number of flow-matching Euler steps |
| `--guidance_scale` | `2.5` | CFG; values at most one use conditional prediction only |
| `--fps` | `24` | MP4 frame rate and default negative-caption time span |
| `--negative_prompt` | released default | Optional replacement negative caption |
| `--seed` | `0` | CPU noise generator seed |

## Configuration

The text conditioner applies the released chat template, reads `hidden_states[-3]` and removes the system/user
header tokens, with the released 1536-token budget. The default negative caption has time ranges filled from the requested clip length.
Sampling retains FP32 latents across Euler updates. The optional first-frame condition uses VAE posterior mode,
per-channel normalization and a channel mask; its predicted first latent frame is not CFG amplified.

SP compacts padding before attention so padded keys cannot change the softmax. TP row reductions remain outside
linear modules so replacing those modules with FP8 preserves collective semantics. Modulation projections remain
FP32. The example uses five initial dense steps and one dense layer before Sol-Attn.

## Troubleshooting

- Missing `Qwen3_5TextModel`: use a compatible Transformers release.
- Out of memory: use TP, reduce video length or resolution, and retain model CPU offload. SP does not shard weights.
- Invalid geometry: use dimensions divisible by 32 and frame counts `4*k+1`.
- Missing optional FLA kernels: Transformers uses its PyTorch fallback; this affects conditioning speed.

## Notes

The source model is [ifm-ai/PAN2](https://github.com/ifm-ai/PAN2).
The transformer and preprocessing preserve Apache-2.0 notices from the pinned Diffusers implementation.
This example integrates local generation components; it does not port the hosted prompt-rewriting or guardrail service.
