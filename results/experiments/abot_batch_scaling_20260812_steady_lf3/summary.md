# ABot steady-state batch scaling: three latent control frames

Hardware and warmup are the same as the LF=1 experiment. Each scheduled chunk
generates three latent frames and was measured through the LiveKit scheduler.

| Sessions | Batch cap | Observed batch | Aggregate FPS | Per-session FPS | p95 chunk latency (s) | p95 queue wait (s) | Peak allocated GiB | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 1 | 1.0 | 15.66 | 15.66 | 0.776 | 0.000 | 39.14 | OK |
| 2 | 1 | 1.0 | 15.54 | 7.77 | 1.546 | 0.774 | 44.42 | OK |
| 2 | 2 | 2.0 | 16.14 | 8.07 | 1.508 | 0.001 | 55.07 | OK |

Batching two long-control sessions improves aggregate FPS only 3.8% while the
batch compute time rises from about 0.77 s to 1.51 s. VAE decode consumes about
20% of each batch, and DiT about 28-32%; the remaining time is cache collation,
output conversion, and scheduler-side work. This differs sharply from the LF=1
case and motivates a workload-aware policy rather than a fixed batch cap.

Raw data: [results.csv](results.csv) and [results.json](results.json).
