# PAN2 validation

## Scope and provenance

This integration adds PAN2 only. MiniMax-H3 video/audio generation is an existing, separate integration;
its LightX2V video benchmark does not measure MiniMax-H3-Image.

The released-checkpoint results below were recorded on November 13, 2026, at TeleFuser commit
`e09df7a01875298248d343167ed3a9fee615dc1b`. They are historical validation evidence, not reruns of the split PR.
Every listed output was checked against its original SHA256 record during this split.

## Hardware and runtime

- Host: 4 NVIDIA H100 80GB HBM3 GPUs with NVLink; released runs use **one or two** devices (GPU 2/3).
- CPU: dual Intel Xeon Platinum 8468, 192 logical CPUs; container memory limit 800 GiB.
- Driver 590.48.01; Python 3.11.13; PyTorch 2.11.0+cu130; Transformers 5.14.1; xFormers 0.0.35.
- Model CPU offload, `pin_cpu_memory=False`; separate model loading excluded from `pipeline_seconds`.
- Single-request wall time includes conditioning, denoising, decoding and runtime CPU offload/onload;
  no repeated-run throughput or pure-kernel speedup is inferred.
- FP8+Sol below is **FP8 linear GEMM with BF16 Sol attention**, not FP8 Q/K/V attention.
- Independent Diffusers reference revision: `843f1743e259e510f22065c0cb917d417b98d9f5`.

## Released checkpoint runs

Checkpoint: `IFM/PAN2`, revision `82f54c6cea3e20a9c451b61d7360016e5970503d`.
Output: 704×1248, 121 frames (241 for the longer case), 24 fps, 50 steps, CFG 2.5, flow shift 9.
The example consumes the supplied caption directly; hosted prompt rewriting is outside this integration.

| Case | GPUs | SP×TP | Pipeline (s) | Output shape |
| --- | ---: | --- | ---: | --- |
| pan2-10s-tp2-fp8-sol | 2 | 1×2 | 338.505 | [241, 704, 1248, 3] |
| pan2-bf16 | 1 | 1×1 | 360.615 | [121, 704, 1248, 3] |
| pan2-fp8-sol | 1 | 1×1 | 304.212 | [121, 704, 1248, 3] |
| pan2-i2v-tp2-fp8-sol | 2 | 1×2 | 191.858 | [121, 704, 1248, 3] |
| pan2-sp2 | 2 | 2×1 | 226.276 | [121, 704, 1248, 3] |
| pan2-sp2-fp8-sol | 2 | 2×1 | 218.658 | [121, 704, 1248, 3] |
| pan2-tp2 | 2 | 1×2 | 215.876 | [121, 704, 1248, 3] |
| pan2-tp2-fp8-sol | 2 | 1×2 | 206.637 | [121, 704, 1248, 3] |

These are generation-completion checks, not a scored quality evaluation or a matched official speed benchmark.
Four-GPU hybrid SP2×TP2 is covered by small numerical fixtures; the full-checkpoint results above use at most two GPUs.

## Numerical and sampling checks

The unit tests cover checkpoint conversion, conditioning precision, scheduling, output conversion and VAE behavior.
The distributed tools compare tiny models against the same dense model and exercise repeated offload/worker requests;
they do not measure full-model quality. Independent Diffusers comparisons require that pinned reference implementation.

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_IB_DISABLE=1 python -m torch.distributed.run \
  --standalone --nproc_per_node=2 tools/validation/validate_pan2_parallel.py \
  --sp 2 --tp 1 --fp8 --sol --offload --output work_dirs/pan2-parallel.json

NCCL_IB_DISABLE=1 python tools/validation/validate_pan2_sampling.py \
  --device-ids 0 1 --sp 2 --tp 1 --fp8 --sol \
  --scheduler-config /path/to/PAN2/scheduler/scheduler_config.json \
  --output work_dirs/pan2-sampling.json
```

See the [example](README.md) for released-checkpoint generation commands.
