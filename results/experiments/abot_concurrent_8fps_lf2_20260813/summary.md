# ABot-World 8-FPS / two-latent concurrent baseline

## Target and workload

- Model: `ABot-World-0-5B-LF`, real public checkpoint, 832x480.
- Target: 8 FPS per user; one continuation chunk contains 2 latent frames and decodes to 8 RGB frames.
- Controls: every active user holds a valid control snapshot; it is refreshed once per second.
- Consumer metric: lossless consumer displays frames at 8 FPS. `consumer_end_to_end_fps` includes startup and final queued-frame drain, so it is deliberately a user-visible, conservative metric.
- GPU: one NVIDIA H100 80 GB (physical GPU 4), one model replica.
- Each run uses 18 seconds of sustained input, no intentional idle intervals, and synchronized session arrivals.

## Results

| Active users | Scheduler | Batch cap | Mean observed batch | Mean displayed FPS/user | p95 compute per scheduled chunk (s) | p95 queue wait (s) | p95 inter-chunk interval (s) | 8-FPS target met? |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | strict round-robin | 1 | 1.00 | 7.736 | 0.608 | 0.089 | 1.003 | Steady-state yes; end-to-end aggregate is conservative |
| 2 | strict round-robin | 1 | 1.00 | 6.392 | 0.603 | 0.381 | 1.212 | No |
| 3 | strict round-robin | 1 | 1.00 | 4.262 | 0.602 | 0.387 | 1.802 | No |
| 2 | coalesced batch | 2 | 1.406 | 6.347 | 1.142 | 0.407 | 1.741 | No |

## Interpretation

A one-user continuation chunk stabilizes at about 0.60 seconds. Strict round-robin therefore needs about 1.20 seconds for two continuously active users and about 1.80 seconds for three, while the playback/control period is one second. This is the primary pre-improvement bottleneck: a per-GPU 8-FPS deadline miss caused by serial session scheduling, not video delivery or dropped frames.

Coalesced batch size two is not a sufficient fix in the current ABot path. Its actual batch-2 compute is about 1.12 seconds, so it too misses the one-second deadline. The result is only a small end-to-end change (6.392 to 6.347 FPS/user) and uses substantially more memory (about 66.5 GiB while loaded in this run). This motivates improving both model-stage batching efficiency and workload-aware placement/admission rather than merely turning on batching.

## Reproduce

```bash
cd /public/fanyk1/lwb/TeleFuser-abot-world

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. \
/public/fanyk1/lwb/envs/telefuser_sage291/bin/python \
tools/validation/benchmark_abot_turboserve_concurrent.py \
  --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \
  --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \
  --sessions 2 --duration-seconds 18 --arrival-window-seconds 0 \
  --fps 8 --consumer-playback-fps 8 --control-latent-frames 2 \
  --scheduler-mode round_robin --max-batch-size 1 --batching-window-ms 0 \
  --delivery-mode lossless --control-update-min-seconds 1 \
  --control-update-max-seconds 1 --idle-probability 0 \
  --output results/experiments/abot_concurrent_8fps_lf2_20260813/sessions_2_round_robin.json
```

For the batching comparison, change `--scheduler-mode batched --max-batch-size 2 --batching-window-ms 2`.
