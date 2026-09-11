# MiniMax-H3 FP8 Sol Attention Quality

Date: 2026-09-08

## Scope

- Model: `MiniMaxAI/MiniMax-H3`, FL2VA partition
- Hardware: one NVIDIA H100 80 GB; no sequence or tensor parallelism
- Request: T2VA, 1344x768, 4 seconds, 107 decoded frames, 50 denoising steps, seed 0
- Prompt: `Steam rises from the ramen while the family talks in the background.`
- Sol policy: 10 dense steps, 2 dense layers, `tau=1.0`, exact threshold
- FP8 profile: tf-kernel W8A8 Linear plus post-RoPE E4M3 Q/K/V
- Timing: cold measured requests in clean processes with no warm-up. FP8 values are means of two runs; the BF16
  reference is one run.

The two FP8 profiles differ only at the QKV quantization boundary. The final profile subtracts per-head sequence
means from K and V, adds the V mean back to the attention output, and corrects the residual V mean after E4M3
rounding. These are attention-equivalent transforms, not Linear SmoothQuant.

## End-to-end results

| Profile | Denoise mean / range (s) | Throughput (step/s) | Peak allocated (GiB) |
|---|---:|---:|---:|
| BF16 Linear + FA4 | 211.442 | 0.23647 | 64.67 |
| FP8 Sol, unsmoothed | 148.446 / 148.372-148.520 | **0.33682** | 37.11 |
| FP8 Sol, fused KV smoothing + V bias correction | 151.651 / 151.328-151.974 | 0.32971 | **37.11** |

The final profile is 39.4% faster and uses 42.6% less peak allocated memory than the matched BF16 reference. Against
unsmoothed FP8 Sol, smoothing adds 2.16% mean denoising time and reduces mean throughput by 2.11%; peak allocated
memory is unchanged. The duplicate FP8 runs produced identical frame and audio SHA256 hashes within each profile.

## Attention-boundary error

Real post-QK-norm, post-RoPE Q/K/V were captured from the first active Sol layer. The live tensor shape was
`(1, 32626, 56, 128)` within a 32640-token padded request. The measurement used heads 0-3 over the complete live K/V
context. Dense-attention output error used 64 evenly spaced query positions and FP32 math against the original BF16
Q/K/V. This isolates the quantization boundary; it is not a full-generation metric.

| Tensor boundary | Unsmoothed MSE | Smoothed MSE | MSE reduction |
|---|---:|---:|---:|
| K at the quantizer input distribution | 9.380e-4 | 7.349e-4 | **21.65%** |
| V at the quantizer input distribution | 1.647e-2 | 1.611e-2 | 2.16% |
| Reconstructed V | 1.647e-2 | 1.611e-2 | 2.17% |
| Per-head/channel reconstructed V mean bias | 1.044e-6 | 2.561e-14 | **>99.99999%** |
| Dense attention output | 7.034e-4 | 6.459e-4 | **8.18%** |

For the dense attention output, smoothing also improved cosine similarity from 0.999403 to 0.999452, relative L2
error from 0.03455 to 0.03311, and SQNR from 29.23 to 29.60 dB. Q is intentionally unchanged by smoothing and had
identical quantization output in both profiles.

## Generated media comparison

The following metrics compare decoded outputs with the same-seed BF16 trajectory. They measure numerical trajectory
similarity, not absolute perceptual quality: a small attention perturbation can select a different valid diffusion
trajectory.

| Profile | Frame cosine | PSNR (dB) | SSIM mean / min |
|---|---:|---:|---:|
| FP8 Sol, unsmoothed | 0.87488 | 14.695 | 0.5464 / 0.5151 |
| FP8 Sol, KV smoothing + V correction | **0.87729** | **14.800** | **0.5659 / 0.5382** |

SSIM was evaluated on every fourth frame (27 synchronized frames). All-frame cosine and PSNR use all 107 decoded
uint8 frames.

| Profile | Waveform cosine | Waveform MSE | SI-SDR (dB) | Spectral convergence | Log-spectral distance (dB) |
|---|---:|---:|---:|---:|---:|
| FP8 Sol, unsmoothed | **0.55217** | **9.842e-5** | **-3.579** | **0.6301** | **12.484** |
| FP8 Sol, KV smoothing + V correction | 0.53450 | 1.026e-4 | -3.980 | 0.6388 | 12.522 |

Both FP8 videos are coherent and preserve the prompt's ramen-dining composition without corrupted frames. Smoothing
improves every reported video trajectory metric in this seed, while the unsmoothed output is closer to BF16 on the
reported audio trajectory metrics. This single-seed media comparison therefore does not establish a universal audio
quality ranking; the attention-boundary metrics directly measure the error targeted by the implementation.

## Fusion results

The optimized path combines K and V sequence statistics into one Triton reduction and merges the BF16 dense prefix
with the corrected sparse suffix in one output pass. At the current H3 live shape `(1, 32626, 56, 128)`, H100
100-repetition means are:

| Boundary operation | Before fusion (ms) | After fusion (ms) | Change |
|---|---:|---:|---:|
| Exact K/V sequence statistics | 1.0097 | 0.5319 | -47.3% |
| Smoothed QKV quantization and correction statistics | 2.5795 | 2.1054 | -18.4% |
| Output correction and dense-prefix merge | 0.8776 | 0.5540 | -36.9% |
| Combined quantization/correction and output boundary | 3.4571 | 2.6594 | **-23.1%** |

The GPU tests verify the fused K/V statistics, quantized tensors, correction, and fused prefix merge are bitwise
identical to their unfused exact implementations. Fusion adds no large partial-statistics buffer.

Several faster-looking approximations were rejected. A two-warp reduction changed long-sequence FP32 summation,
sampling K/V centers damaged Sol routing quality, and a block-partial V-bias fusion was slower than directly scanning
the compact FP8 V tensor. The remaining end-to-end cost is the exact global-statistics work; it was not removed by
sacrificing the target error reduction.

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
  --output outputs/h3_fp8_sol_kv_bias.mp4
```

The benchmark saves synchronized MP4, `.frames.npy`, `.audio.npy`, and metrics JSON artifacts. MP4 and NumPy
artifacts remain local and are excluded from Git.
