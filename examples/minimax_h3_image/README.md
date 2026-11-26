# MiniMax-H3-Image Examples

Generate images from text or edit up to sixteen reference images with the native six-step PDD checkpoint.
The pipeline uses TeleFuser model loading, stages, workers, attention dispatch and quantization.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| MiniMax-H3-Image | [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | N/A | Native transformer, fused PDD heads, image VAE and Qwen3-VL conditioner |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Text to image / reference editing | Supported | One prompt, at most 16 references, fixed six-step schedule |
| Sequence parallel | Supported | Existing Ulysses SP; divides attention heads after TP |
| Tensor parallel | Supported | DiT SP/TP and independent Qwen3-VL text-encoder TP |
| Resident inference | Supported | Keep DiT, sharded text encoder and VAE on GPUs |
| VAE tile parallel / CUDA graph | Supported | Preserve tile geometry and blending; graph buffers retain one tile shape |
| Sol-Attn | Supported | Dense text prefix; BF16 attention or native FP8 Q/K/V with `--sol-fp8` |
| FP8 linear quantization | Supported | Existing tf-kernel FP8 backend; FP32 boundary projections retained |
| CPU offload | Supported | Model CPU offload, including sharded DiT; sequential conditioning phases |
| LoRA / feature cache / torch.compile / FSDP | Unsupported | Not exposed by this image integration |
| Server API | Unsupported | Local Python and CLI entrypoints; service schema not validated |

Numerical feature checks use small models, including SP2, TP2 and SP2×TP2 with FP8 and Sol-Attn together.
See [validation notes](VALIDATION.md) for the distinction between these checks and full weights.

## Requirements

- An 80GB CUDA GPU for the BF16 transformer and sequential CPU offload; more devices can reduce TP memory.
- Sufficient host memory for the transformer, Qwen3-VL encoder, VAE and spawned worker storage.
- TeleFuser, a Transformers version supporting Qwen3-VL, Diffusers primitives, and Pillow.
- Sol-Attn and FP8 require a compatible, locally provisioned tf-kernel wheel; use the existing installation instructions.
- Reference inputs are upright RGB images after EXIF orientation is applied.

Install TeleFuser through the [development setup](../../CONTRIBUTING.md#development-setup).
Validation uses PyTorch 2.11.0+cu130, Transformers 5.22.2 and the Diffusers revision recorded in the validation notes.
No new TeleFuser package extra is required.

## Model Directory

Download the released native transformer and PDD heads, plus the official **image** VAE in Diffusers format.
The native `vae/` video-style weights are not consumed by this example.

```text
MiniMax-H3-Image/
|-- transformer/config.json
|-- transformer/model-*-of-*.safetensors
|-- pdd_config.json
|-- pdd_fused_heads.safetensors
|-- scheduler/scheduler_config.json
|-- processor/                         # tokenizer and image processor assets
|-- text_encoder/config.json
|-- text_encoder/model-*-of-*.safetensors
|-- vae_diffusers/config.json
\-- vae_diffusers/diffusion_pytorch_model.safetensors
```

`--text_encoder_path` can point to an existing identical H3 Qwen3-VL text encoder to avoid duplicating weights.
It must contain the released encoder configuration and all fourteen shards.

Set the shared model root (the default is `/hhb-data/aigc/model_zoo`):

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

The default checkpoint directory is `$TF_MODEL_ZOO_PATH/MiniMaxAI_MiniMax-H3-Image`. Use `--model_root` to override it.

## Quick Start

```bash
python examples/minimax_h3_image/minimax_h3_image_t2i_h100.py \
  --model_root /path/to/MiniMax-H3-Image \
  --text_encoder_path /path/to/MiniMax-H3-Image/text_encoder \
  --prompt "A red fox in a snowy forest, detailed natural photograph." \
  --output_path work_dirs/h3-image.png
```

The command writes a 2048×2048 PNG using all six published heads and the published sigma grid.
The default uses model CPU offload for single-GPU memory capacity. For lower latency on four H100s, shard the DiT
with TP and keep all stages resident:

```bash
python examples/minimax_h3_image/minimax_h3_image_t2i_h100.py \
  --model_root /path/to/MiniMax-H3-Image \
  --text_encoder_path /path/to/MiniMax-H3-Image/text_encoder \
  --gpu_num 4 --sp 1 --tp 4 --resident \
  --prompt "A red bicycle leaning against a white plaster wall." --seed 42
```

`--resident` uses encoder TP across the selected devices, a resident mixed-precision VAE, parallel tiles and CUDA graph
decode on up to six independent streams per device. The VAE is loaded in FP32 before casting decoder GEMMs
to BF16; encoder weights, decoder normalization/scales and rotary bases stay FP32. The existing CPU-offload
default is unchanged. For an ablation, disable individual VAE optimizations with
`--no-vae-parallel` or `--no-vae-cuda-graph`. CUDA graph decode retains one local tile shape at a time and requires
resident VAE weights; it does not cache prompts, denoising results or completed images.
The resident encoder selects native cuDNN SDPA through its ordinary `AttentionConfig`. As with vLLM-Omni's cuDNN
encoder path, a fixed noise seed is not a guarantee of bitwise-identical images across executions or layouts.

On H100, `--sol-fp8` replays single-split local attention below 16384 tokens in a CUDA graph shared across the
image DiT's layers. Each replay copies fresh Q/K/V; it retains one shape and returns an independently owned output.
The graph covers smoothing/quantization, native Sol and exact-prefix correction, with no model weights or
distributed collectives captured. Larger contexts keep native eager FP8 Sol to avoid copying large graph inputs.
Short exact prefixes use native Flash SDPA. Long-sequence statistics use parallel
FP32 reductions, and FP8 padding copies/clears Q/K/V in a single kernel launch. These optimizations preserve the configured Sol threshold, layer range and six-step sampling schedule.

## Examples

### Image generation and editing

#### `minimax_h3_image_t2i_h100.py`

Use the same command for text-only generation or image editing; the prompt can refer to `<Picture 1>`, etc.
An editing request deliberately changes its reference according to a new instruction. For example, requesting
an autumn forest should replace a snowy background with colored leaves. Compare precision/parallel settings
using the same task, prompt, seed and references; show edited results separately.

```bash
python examples/minimax_h3_image/minimax_h3_image_t2i_h100.py \
  --model_root /path/to/MiniMax-H3-Image \
  --text_encoder_path /path/to/MiniMax-H3-Image/text_encoder \
  --prompt "Change the sky in <Picture 1> to sunset." \
  --image inputs/landscape.png --output_path work_dirs/edit.png

python examples/minimax_h3_image/minimax_h3_image_t2i_h100.py \
  --model_root /path/to/MiniMax-H3-Image \
  --text_encoder_path /path/to/MiniMax-H3-Image/text_encoder \
  --prompt "A red fox in a snowy forest." \
  --gpu_num 4 --sp 2 --tp 2 --fp8 --sol
```

Key options:

| Option | Default | Description |
| --- | --- | --- |
| `--model_root` | `PPL_CONFIG["model_root"]` | Released checkpoint directory |
| `--text_encoder_path` | `MODEL_ROOT/text_encoder` | Optional shared Qwen3-VL encoder directory |
| `--gpu_num` | `1` | Number of GPUs; defaults to Ulysses SP unless `--tp` is set |
| `--device_ids` | first `gpu_num` devices | Optional explicit visible CUDA IDs; count must equal SP×TP |
| `--sp`, `--tp` | `gpu_num / tp`, `1` | Ulysses and tensor parallel degrees |
| `--fp8` | off | Online linear weight/activation FP8 through tf-kernel |
| `--sol` | off | Sparse BF16 attention |
| `--sol-fp8` | off | FP8 Q/K/V Sol; also enables `--sol` |
| `--dit-resident` | off | Keep DiT on GPU and offload text encoder; requires sufficient VRAM |
| `--resident` | off | Keep all stages resident; default encoder TP to GPU count and enable VAE tile parallel/graphs |
| `--text-encoder-tp` | GPU count when resident, otherwise `1` | Independent encoder TP degree; uses the first selected devices |
| `--vae-parallel` / `--no-vae-parallel` | on in multi-GPU resident mode | Distribute independent VAE encode/decode tiles |
| `--vae-cuda-graph` / `--no-vae-cuda-graph` | on in resident mode | Replay local decoder tiles; preserves eager arithmetic and blend order |
| `--height`, `--width` | `2048`, `2048` | Returned image dimensions |
| `--min_generation_area` | `4194304` | Generation canvas floor; smaller requested images resize after decoding |
| `--seed` | `0` | CPU request generator seed |
| `--image` | empty | Ordered reference paths, at most sixteen |

Hybrid four-GPU execution uses `--gpu_num 4 --sp 2 --tp 2`; this shards DiT projections with TP while distributing sequence attention with Ulysses SP.
Add `--resident --fp8 --sol-fp8` to combine resident text-encoder TP4 and VAE acceleration with hybrid DiT execution.
Encoder and VAE workers use their own one-dimensional groups; the DiT SP×TP layout does not restrict their TP degree.

Key behavior:

- `PPL_CONFIG` declares the example defaults. `get_pipeline(parallelism, model_root, ...)` loads one reusable pipeline.
- `run()` returns a PIL image; `run_with_file()` saves the output and returns `{"output_path": ...}`.
- The CLI stops workers in `finally`. Python callers must call `pipeline.stop()` when finished.
- These local entrypoints do not claim a validated `telefuser serve` contract.

## Configuration

### Python entrypoints

```python
from examples.minimax_h3_image.minimax_h3_image_t2i_h100 import get_pipeline, run_with_file

pipeline = get_pipeline(parallelism=2, model_root="/path/to/MiniMax-H3-Image")
try:
    result = run_with_file(pipeline, prompt="A red fox in a snowy forest.")
    print(result["output_path"])
finally:
    pipeline.stop()
```

### Sampling and parallel configuration

Text is encoded verbatim without a chat template. Image references retain their own aspect ratio and are resized
separately for the VAE (short edge 1024–2048) and vision conditioner (short edge 1024). The image VAE uses ImageNet
pixel normalization, posterior sampling rounded through FP16, per-channel latent statistics and tiled decoding.

Sampling is guidance-distilled: there is no CFG or arbitrary step-count option. The scheduler must specify
unshifted `velocity` updates. Changing the head count or silently substituting another flow schedule is rejected.
The example enables Sol-Attn after one dense step and one dense layer; adjust the ordinary `AttentionConfig` in
`common.py` for different quality/performance tradeoffs.

## Troubleshooting

- Missing PDD heads: load `pdd_fused_heads.safetensors` together with every native transformer shard.
- Out of memory: use TP or model CPU offload; SP alone does not shard parameter storage. Full residency requires
  room for all stages and CUDA contexts. On four H100s, start with `--sp 1 --tp 4 --resident` rather than BF16 SP4.
- Worker startup timeout: check the existing `ParallelConfig.timeout`, CUDA visibility and NCCL connectivity.
- Incorrect VAE keys: use `vae_diffusers/`, not the unrelated native video-style `vae/` serialization.

## Notes

The source model is [MiniMax-AI/MiniMax-H3-Image](https://github.com/MiniMax-AI/MiniMax-H3-Image).
The model-specific image VAE and packing helpers retain Apache-2.0 notices from their pinned Diffusers source.
This local example does not perform hosted prompt rewriting or expose a service schema.

`--fp8` quantizes linear GEMMs. Add `--sol-fp8` for native FP8 Q/K/V Sol attention (it also enables `--sol`). The native SM90 kernel is validated with SP2×TP2; NVVM bindings are selected by their actual API signature, so the CUDA 12.9 label does not incorrectly force an older binding API.
