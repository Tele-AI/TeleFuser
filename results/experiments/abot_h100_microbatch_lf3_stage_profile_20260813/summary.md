# ABot-World LF=3 microbatch stage profile (one H100)

The setup matches the synchronous retained-session microbenchmark: B independent
sessions, 9-frame seed chunk excluded, three continuation warmups, then six
timed 12-frame continuation batches. Stage times use CUDA events.

| B | End-to-end chunk | DiT denoise | VAE decode | Other state/Python/tensor work | Aggregate FPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 807.0 ms | 289.2 ms | 415.9 ms | 101.8 ms | 14.87 |
| 2 | 1567.3 ms | 505.9 ms | 865.3 ms | 196.0 ms | 15.31 |
| 3 | 2279.1 ms | 736.9 ms | 1260.1 ms | 282.0 ms | 15.80 |

DiT scales sublinearly (B=3 is 2.55x B=1), demonstrating some GPU batch
parallelism. VAE decode is nearly linear (B=2: 2.08x, B=3: 3.03x); it is the
largest stage and is the primary reason aggregate FPS remains nearly flat.
The remainder also grows near-linearly because the current retained-session
implementation collates KV/VAE state before a batch and scatters it afterward.

This profile is not a scheduler measurement: it invokes the native batched
model path directly, after state creation and before any service delivery.
