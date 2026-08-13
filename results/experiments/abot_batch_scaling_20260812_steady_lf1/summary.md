# ABot steady-state batch scaling: one latent control frame

Hardware: one NVIDIA H100 80 GB (GPU 0). The ABot-World-0-5B-LF service was
preloaded, then each point used continuously active retained sessions, one warmup
chunk per session, and two measured chunks per session. Values below are from the
LiveKit service scheduler, not a synthetic model loop.

| Sessions | Batch cap | Observed batch | Aggregate FPS | Per-session FPS | p95 chunk latency (s) | p95 queue wait (s) | Peak allocated GiB | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1.0 | 11.00 | 11.00 | 0.376 | 0.000 | 39.05 | OK |
| 2 | 1 | 1.0 | 11.73 | 5.86 | 0.692 | 0.347 | 44.33 | OK |
| 2 | 2 | 2.0 | 14.31 | 7.16 | 0.572 | 0.000 | 54.89 | OK |
| 4 | 1 | 1.0 | 11.49 | 2.87 | 1.422 | 1.070 | 54.91 | OK |
| 4 | 2 | 2.0 | 14.31 | 3.58 | 1.137 | 0.570 | 65.47 | OK |
| 4 | 4 | -- | -- | -- | -- | -- | -- | OOM in VAE temporal decode (requested 3.81 GiB) |

The valid batch-2 points increase aggregate throughput by 22.0% (two sessions)
and 24.6% (four sessions) over batch cap 1. However, the four-session latency
remains above one second and batch 4 is infeasible despite 80 GB device memory.

Raw data: [results.csv](results.csv) and [results.json](results.json).
