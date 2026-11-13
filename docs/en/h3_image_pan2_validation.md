# MiniMax-H3-Image and PAN2 integration validation

This document distinguishes architecture/parallel numerical checks from full-checkpoint generation.
A successful small-model check does not establish output quality, throughput or memory requirements of the released model.

## Sources and runtime

| Item | Revision / value |
| --- | --- |
| TeleFuser fork baseline | `f21457a92e92f4a7570f0cb6b445ee334f1072b0` |
| MiniMax-AI/MiniMax-H3-Image source | `40850901ab938303b5274122b5f305ebc5b75c80` |
| ifm-ai/PAN2 source | `0fb907b6e16a82ead6d691d1de2eef6299c71e9a` |
| Independent Diffusers reference | `843f1743e259e510f22065c0cb917d417b98d9f5` |
| MiniMaxAI/MiniMax-H3-Image weights | `5f90473023975e6271b0ed1dd84c4e88e4bbc6d3` |
| IFM/PAN2 weights | `82f54c6cea3e20a9c451b61d7360016e5970503d` |
| GPU | NVIDIA H100 80GB HBM3, four NVLink-connected devices, 81,559 MiB each |
| Driver | 590.48.01 |
| CPU / container memory limit | Two Intel Xeon Platinum 8468 sockets, 192 logical CPUs / 800 GiB |
| Python / PyTorch | 3.11.13 / 2.11.0+cu130 |
| Transformers / xFormers | 5.14.1 / 0.0.35 (cu130) |
| Accelerate (independent reference loading only) | 1.15.0 |

Runtime dependencies were installed in an isolated task environment. Existing unrelated GPU workloads were preserved;
full-component tests use the two free devices. Four-device numerical fixtures allocate only small models.

## Released transformer architecture

| Model | Parameters | Blocks / refiner | Attention | Latents / patch | Conditioning |
| --- | --- | --- | --- | --- | --- |
| MiniMax-H3-Image | 33,125,224,384 | 50 / 2 | Hidden 5376, 56 heads × 128 | 24 channels, 1×2×2 patch | Qwen3-VL layer 50, 5120 features, six fused PDD heads |
| PAN2 | 29,201,996,992 | 44 / 4 | Hidden 4096, 32 heads × 128 | 48 channels; noisy + condition + mask = 97 input channels, 1×2×2 patch | Provided Qwen3.5 text encoder, hidden_states[-3], 4096 features |

Counts are obtained from meta-device models built from the released configuration, including H3's six PDD heads.
H3 boundary projections and PAN2 timestep/modulation projections retain their checkpoint FP32 precision.

## Numerical checks

- H3-Image DiT: exact agreement with the independent Diffusers tiny model after checkpoint conversion, including
  gate/value ordering and native FP32 boundary modules.
- PAN2 DiT: exact FP32 agreement with the independent Diffusers tiny model.
- H3-Image VAE: exact encoder and decoder agreement, including overlapping tile stitching. The full released
  mixed-precision VAE also matches the official from_pretrained loader with zero encoder/decoder error.
- PAN2 full text encoder: exact agreement with the official prompt-embedding helper for a short prompt and the
  1477-token released negative caption. H3-Image pure-text and 1024-pixel reference-image conditioning also match
  the full 64-layer official Qwen3-VL reference exactly at hidden_states[50]. Non-persistent rotary frequencies remain FP32 when model weights cast to BF16.
- Sampling: H3 positive velocity direction, six step-dependent heads, frozen reference rows, per-row reference timestep,
  image packing and optimized-attention padding; PAN2 CFG, conditional first frame, FP32 accumulation and negative-caption
  duration are covered by unit tests.
- New architecture, conditioning precision, VAE and sampling tests: 20 passed. Existing H3 model and FP8 smoothing regressions: 41 passed.
- Complete six-head H3-Image edit sampling agrees exactly with the independent reference scheduler and model.
- Actual ParallelWorker sampling (with reference/first-frame inputs, odd text lengths, all sampling steps and two
  consecutive requests) passed SP2, TP2 and SP2×TP2 for both models with FP8 linear and FP8 Sol-Attn enabled.

The deterministic model fixtures use two transformer blocks, one text refiner and seed 2026 (CPU input generator 71).
H3 uses hidden 256, four 128-dimensional heads, FFN 512, latent channels 2 and text features 32; the Sol fixture has
384 packed rows with four text rows. PAN2 uses four 128-dimensional heads, latent output channels 2, five input channels,
text features 32 and FFN ratios 2. Its Sol fixture input is `[1,5,3,16,32]` with five text tokens; dense SP also tests
27 video tokens and five text tokens. These are intentionally reduced architectures.

The distributed fixture compares an unchanged single-device BF16 model to the same model after actual sharding and
quantization. PAN2 uses odd video/text sequence lengths to exercise padding removal. FP8 checks require materialized
E4M3 weight caches. Repeated-offload checks run two complete CPU unload/reload cycles and require identical outputs.

| Model | SP | TP | FP8 linear + FP8 Sol-Attn relative L2 | Maximum absolute error | Materialized FP8 modules |
| --- | --- | --- | --- | --- | --- |
| H3-Image tiny | 2 | 1 | 0.004660 | 0.007718 | 14 |
| H3-Image tiny | 1 | 2 | 0.004576 | 0.006151 | 14 |
| H3-Image tiny | 2 | 2 | 0.004576 | 0.006151 | 14 |
| PAN2 tiny | 2 | 1 | 0.003891 | 0.007813 | 30 |
| PAN2 tiny | 1 | 2 | 0.003835 | 0.007813 | 30 |
| PAN2 tiny | 2 | 2 | 0.003835 | 0.007813 | 30 |

All six configurations also passed the repeated-offload identity check. Sol-Attn used `tau=1`, exact threshold estimation,
no dense warmup, and FP8 attention. These tolerances/results describe numerical fixtures; use calibrated dense warmup
and quality evaluation for real generation.

Reproduce a fixture from the repository root (use an environment with compatible optional GPU kernels):

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/validation/validate_h3_image_pan2_parallel.py \
  --model pan2 --sp 2 --tp 1 --fp8 --sol --sol-fp8 --offload \
  --output work_dirs/pan2-parallel.json
```

The actual worker sampling path has a separate reproduction command:

```bash
NCCL_IB_DISABLE=1 python tools/validation/validate_h3_image_pan2_sampling.py \
  --model pan2 --scheduler-config /path/to/PAN2/scheduler/scheduler_config.json \
  --device-ids 0 1 --sp 2 --tp 1 --fp8 --sol --sol-fp8 \
  --output work_dirs/pan2-worker-sampling.json
```

For H3-Image use `--model h3-image` and omit `--scheduler-config`. Both checks use random tiny models.

Replace `pan2` with `h3-image`. For SP2×TP2 use four visible devices, four processes and `--sp 2 --tp 2`.
`NCCL_IB_DISABLE=1` was needed by the validation host; it is an existing NCCL option, not a TeleFuser setting.

## Full components

| Component | Check | Result |
| --- | --- | --- |
| PAN2 released Qwen3.5 text encoder | Load all four shards, chat-template prompt and negative prompt, model CPU offload | Passed; output `[1,14,4096]`, negative `[1,9,4096]`; peak allocated 15,938,346,496 bytes |
| H3-Image released Qwen3-VL conditioner | All fourteen shard hashes match H3-Image metadata; encode verbatim prompt through layer 50 and offload | Passed; output `[8,5120]`; peak allocated 51,557,238,784 bytes |
| H3-Image released visual text conditioning | Encode a 1024×1024 reference with image-token presentation and offload | Passed; output `[1043,5120]`; peak allocated 51,845,369,856 bytes |
| H3-Image released image VAE | Mixed precision, encode/decode 256×256 RGB fixture, offload between phases | Passed; output `[1,3,256,256]`; peak allocated 2,305,942,528 bytes |
| PAN2 released Wan2.2 VAE | Full file SHA256 matches the shared Wan2.2 VAE; encode/reconstruct RGB image and decode 9 video frames, with model offload | Passed; latent `[1,48,1,16,16]`, video `[1,3,9,256,256]`; peak allocated 2,211,739,136 bytes |
| Released H3-Image / PAN2 complete generation | Transformer downloads and end-to-end verification | Pending |

The component peaks are synchronized-process allocator observations for those specific inputs, not advertised minimum
VRAM or full-model throughput. Full-generation quality/performance conclusions require completed transformer weights.

## Integration scope

The implementation adds model-family classes, pipeline/stage classes and example options. It adds no shared framework
configuration fields, environment variables, loaders, registries, worker protocols or service schemas.
The existing H3 attention call site makes tensors contiguous before existing FP8 prefix/correction ops; no operator
signature changed. This fixes the SP + FP8 Sol-Attn combination inherited by H3-Image.

H3-Image uses the native transformer plus fused PDD heads, and the officially supplied `vae_diffusers` image VAE.
PAN2 uses the provided text-only Qwen3.5 checkpoint and the maintained Diffusers Wan VAE inside a TeleFuser BaseModel.
PAN2 follows the pinned official Diffusers inference contract: FP32 initial noise/latent accumulation and separate
conditional/unconditional forwards. The native training repository initializes BF16 noise and batches CFG; therefore
a fixed seed need not reproduce its command line output byte for byte. The dense and optimized cases in this validation
use the same declared noise, conditioning and schedule.

Hosted prompt rewriting and guardrail services are outside this local generation integration. The command line
examples do not offer LoRA, feature caching, FSDP, Ring SP or serving routes.

See the [H3-Image example](../../examples/minimax_h3_image/README.md) and
[PAN2 example](../../examples/pan2/README.md) for checkpoint layout and generation commands.
