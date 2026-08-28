# MiniMax-H3 FP8 Sol Attention Quality

Date: 2026-08-28

## Scope

- Model: `MiniMaxAI/MiniMax-H3`, FL2VA partition
- Hardware: one NVIDIA H100 80 GB; no sequence or tensor parallelism
- Request: T2VA, 1344x768, 4 seconds, 50 denoising steps, seed 0
- Prompt: `Steam rises from the ramen while the family talks in the background.`
- Sol policy: 10 dense steps, 2 dense layers, `tau=1.0`, exact threshold
- FP8 profiles: tf-kernel W8A8 Linear plus post-RoPE E4M3 Q/K/V
- Timing: one cold measured request per clean process; no warm-up

The three FP8 runs differ only at the QKV quantization boundary. `K` subtracts
the per-head sequence mean from K. `K+V+bias` also centers V, adds the V mean
back to the attention output, and corrects the residual V mean after E4M3
rounding. These are attention-equivalent transforms, not Linear SmoothQuant.

## Results

| Profile | Denoise (s) | Throughput (step/s) | Peak allocated (GiB) | Video cosine | Audio cosine | DINOv3 mean / min |
|---|---:|---:|---:|---:|---:|---:|
| BF16 Linear + FA4 | 212.474 | 0.2353 | 64.67 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| FP8 Sol, unsmoothed | 150.940 | 0.3313 | 37.11 | 0.8882 | 0.4220 | 0.8456 / 0.7367 |
| FP8 Sol, K smoothing | 151.802 | 0.3294 | 37.11 | 0.8753 | 0.4045 | **0.8768** / 0.7935 |
| FP8 Sol, K+V smoothing + V bias correction | 154.019 | 0.3246 | 37.11 | 0.8770 | **0.5094** | 0.8748 / **0.8011** |

Pixel metrics compare all decoded uint8 frames and float32 audio samples against
the matched BF16 output. DINOv3 metrics are corresponding-frame feature cosine
over every fourth frame using
`facebook/dinov3-vith16plus-pretrain-lvd1689m`. Pixel similarity is not
monotonic with the attention-level error reduction because small numerical
changes can move a diffusion trajectory while preserving its composition. The
DINOv3 result and synchronized frame review capture that perceptual behavior:
both smoothing modes improve corresponding-frame structure, while K+V also
substantially improves the synchronized audio match and the worst video frame.

The selected K+V profile has a 2.0% denoising cost versus unsmoothed FP8 Sol. It
still improves throughput by 38.0% and lowers peak allocated memory by 42.6%
versus the BF16 baseline.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python -m tools.validation.benchmark_minimax_h3_fp8_sol_sp \
  --gpu-num 1 --profile baseline --duration 4 --steps 50 --no-warmup \
  --output outputs/h3_bf16.mp4

CUDA_VISIBLE_DEVICES=0 python -m tools.validation.benchmark_minimax_h3_fp8_sol_sp \
  --gpu-num 1 --profile optimized --duration 4 --steps 50 --no-warmup \
  --sol-fp8-smoothing none --no-sol-fp8-v-bias-correction \
  --output outputs/h3_fp8_sol_unsmoothed.mp4

CUDA_VISIBLE_DEVICES=0 python -m tools.validation.benchmark_minimax_h3_fp8_sol_sp \
  --gpu-num 1 --profile optimized --duration 4 --steps 50 --no-warmup \
  --sol-fp8-smoothing k --no-sol-fp8-v-bias-correction \
  --output outputs/h3_fp8_sol_k.mp4

CUDA_VISIBLE_DEVICES=0 python -m tools.validation.benchmark_minimax_h3_fp8_sol_sp \
  --gpu-num 1 --profile optimized --duration 4 --steps 50 --no-warmup \
  --sol-fp8-smoothing kv --sol-fp8-v-bias-correction \
  --output outputs/h3_fp8_sol_kv_bias.mp4
```

The benchmark saves synchronized MP4, `.frames.npy`, `.audio.npy`, and metrics
JSON artifacts. MP4 and NumPy artifacts remain local and are excluded from Git.
