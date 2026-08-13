# Single-H100 ABot-World retained-session microbatch benchmark (LF=3)

Date: 2026-08-13. One NVIDIA H100 80 GB (GPU 4), ABot-World-0-5B-LF,
832x480, `control_latent_frames=3`.

For each batch size B, the benchmark creates B independent retained sessions.
It discards the special 9-frame seed chunk, warms three 12-frame continuation
chunks, then measures eight synchronized calls to
`generate_next_blocks(B sessions)`. Every timed call generates exactly 12
frames for every active session. `T(B)` below is the mean timed batch-call
duration. It excludes the service scheduler, client delivery, browser/WebRTC,
and initial-session creation.

| Batch B | Chunk Time T(B) | Aggregate FPS = 12B/T(B) | FPS/session = 12/T(B) |
|---:|---:|---:|---:|
| 1 | 0.7979 s | 15.04 | 15.04 |
| 2 | 1.5656 s | 15.33 | 7.66 |
| 3 | 2.2802 s | 15.79 | 5.26 |
| 4 | OOM | OOM | OOM |

Measurement variation (standard deviation over eight samples): B=1 6.9 ms,
B=2 13.3 ms, B=3 23.9 ms. Peak PyTorch allocated memory was 39.1 GiB,
55.0 GiB, and 71.1 GiB for B=1,2,3 respectively. B=4 failed while attempting
to allocate a further 4.63 GiB, with only 0.86 GiB free.

The aggregate gain from B=1 to B=3 is only 5.0%, so this ABot implementation's
current native model batch path is close to linear-time in B. This is a model
execution/state-layout result, not a TurboServe scheduling artifact.

Raw machine-readable results: `results.json` and `results.csv`. The benchmark
implementation is `tools/validation/benchmark_abot_microbatch.py`.
