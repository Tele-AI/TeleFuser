# ABot-World explicit cross-session batching experiment (LF=3)

Date: 2026-08-13. This experiment uses four independent single-GPU replicas
(GPUs 4--7) and **explicitly selects** `scheduler_mode=batched`. It is not
the TurboServe baseline: TurboServe's per-worker open-source loop is
single-session round-robin. The purpose here is to evaluate the experimental
TeleFuser cross-session model-batching path against that baseline.

## Fixed workload

- ABot-World-0-5B-LF, default 832x480 image, `control_latent_frames=3`.
- Four replicas; simultaneous active clients, control heartbeat every 0.3 s,
  30 s run, no idle intervals; immediate consumer and lossless delivery.
- `max_batch_size=4`, batching window 2 ms. FPS is local consumer-visible
  end-to-end FPS, excluding browser/WebRTC encode and network transport.

| Users / GPU | Total users | Per-user FPS | Aggregate / GPU | Approx. cluster FPS | Mean observed batch | Mean compute / batch (s) | Mean queue wait (ms) | Mean first frame (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 14.340 | 14.340 | 57.360 | 1.000 | 0.809 | 0.000 | 0.665 |
| 2 | 8 | 7.188 | 14.376 | 57.504 | 1.649 | 1.304 | 0.065 | 1.067 |
| 3 | 12 | 4.694 | 14.082 | 56.328 | 2.243 | 1.730 | 1.943 | 1.438 |

Observed batch histograms per GPU were respectively `{1:37}`, `{1:13,2:24}`
and `{1:9,2:10,3:18}`. Thus batching does form after warmup, but it does not
produce throughput scaling: batch=2 has a roughly 1.58 s steady batch time,
close to twice a single-session 0.81 s step. Batch=3 reaches about 1.73 s and
requires about 81 GB per H100. The bottleneck is therefore the current
batched execution/state layout, not waiting for the scheduler to collect
requests.

Raw per-GPU JSON and logs are in `users_per_gpu_{1,2,3}/`.
