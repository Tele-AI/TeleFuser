# ABot-World four-GPU concurrent-user baseline (LF=3)

Date: 2026-08-13.  GPUs 4--7 are four independent single-GPU service replicas;
this is a per-replica capacity baseline, not a global multi-GPU TurboServe result.

## Fixed workload

- Model: `ABot-World-0-5B-LF`; input: `84b90ad568b693d2.png` at the default 832x480.
- `control_latent_frames=3` (the original ABot-World streaming setting).
- Four replicas, continuous active controls every 0.3 s, 30 s per run, no idle periods.
- Consumer pulls immediately (`consumer_playback_fps=0`), lossless delivery, batch window 2 ms,
  and `max_batch_size=4`.  Reported FPS is consumer-visible end-to-end FPS in the local
  service harness; it excludes browser/WebRTC encode and network transport.

| Users / GPU | Total users | Per-user FPS | Aggregate / GPU | Approx. cluster FPS | Mean batch | Mean compute / batch (s) | Mean queue wait (ms) | Mean first frame (s) | Outcome |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 14.361 | 14.361 | 57.444 | 1.000 | 0.811 | 0.000 | 0.664 | admitted |
| 2 | 8 | 7.189 | 14.378 | 57.510 | 1.649 | 1.304 | 0.065 | 1.067 | admitted |
| 3 | 12 | 4.715 | 14.144 | 56.576 | 2.243 | 1.722 | 1.957 | 1.425 | admitted; each H100 reached about 81 GB during the run |
| 4 | 16 | -- | -- | -- | -- | -- | -- | -- | rejected: `capacity=3` |

`users_per_gpu_4` does not produce JSON because the fourth retained session is rejected by
the service's admission controller (`ABot retained-session capacity is exhausted (capacity=3)`).
The consumer-close timeout subsequently printed by the harness is a cleanup artifact, not a
model-inference latency measurement.

## Interpretation

For LF=3, the single-user result is 14.24--14.51 FPS across the four cards (mean 14.36),
consistent with the previously matched direct single-GPU result (about 15 FPS).  Increasing
the number of active sessions does form batches, but raises batch compute time almost
proportionally, leaving per-GPU throughput flat at about 14.1--14.4 FPS.  Thus the current
baseline's limiting factor in this workload is model/state memory and batched compute scaling,
not scheduler queueing.  This is a useful pre-experiment gap for a workload-aware world-model
scheduler: it should avoid admitting a fourth retained LF=3 state locally and should use global
placement/migration or state offload rather than merely increasing the local batch.

Raw files: `users_per_gpu_{1,2,3}/gpu{4,5,6,7}.json`; logs, including the four admission
rejections, are co-located in `users_per_gpu_4/`.
