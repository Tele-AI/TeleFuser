# ABot-World LightVAE LF=3 single-GPU microbatch capacity

## Configuration

- GPU: one NVIDIA H100 80 GiB (CUDA device 5)
- Checkpoint: `ABot-World-0-5B-LF` with official `taew2_2` lightweight decoder
- `control_latent_frames=3`; each continuation chunk delivers 12 display frames per independent session
- 2 warmup chunks, then 5 synchronized steady-state samples per batch size
- Input: `examples/data/1.png`; fixed prompt and deterministic session seeds

## Results

| Concurrent sessions / DiT batch | Mean chunk time (s) | Aggregate FPS | FPS/session | Peak allocated GiB | 8 FPS deadline (1.5 s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.387 | 30.98 | 30.98 | 28.91 | pass |
| 2 | 0.703 | 34.13 | 17.07 | 34.56 | pass |
| 3 | 1.021 | 35.26 | 11.75 | 40.28 | pass |
| 4 | 1.297 | 37.01 | 9.25 | 45.95 | pass |
| 5 | 1.599 | 37.53 | 7.51 | 51.73 | fail |
| 6 | 1.961 | 36.71 | 6.12 | 57.36 | fail |

The 8 FPS service limit is four simultaneous sessions: B=4 p95 is 1.371 s, while B=5 mean latency already exceeds the 1.5 s deadline. Aggregate throughput saturates at about 37 FPS; this is an SLO limit, not an HBM OOM limit. At B=6, mean DiT time is 1.402 s and LightVAE decode is 0.056 s.

Raw local artifacts (`results.json`, `results.csv`, `run.log`) are intentionally ignored by repository rules.
