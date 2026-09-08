---
title: "ABot Multi-Session Serving: State Isolation, Compatible Batching, and Multi-GPU Scheduling"
description: Session-isolated ABot-World serving with compatible batching, capacity-aware admission, and multi-GPU workers.
date: 2026-08-19
status: validated
validated_revision: 52491cb
hardware: 4 x NVIDIA H100 80 GB HBM3
tags:
  - abot-world
  - multi-session
  - batched-serving
  - process-nccl
  - livekit
---

# ABot Multi-Session Serving: State Isolation, Compatible Batching, and Multi-GPU Scheduling

An ABot-World interaction is not a request that can be released immediately. Once a user connects, the service keeps
the prompt embedding, DiT KV cache, scheduler, and video temporal state until the interaction ends. A single-session
path makes this state easy to manage, but concurrent users normally execute denoising independently: they share model
weights, not the computation.

The sessions also cannot always be concatenated safely. They may be at different chunk stages, use different control
latent counts, or refer to different latent-frame positions. Multi-session serving therefore does more than concatenate
requests: it checks state compatibility, shares one model execution, and returns state and output to the original
session.

## Design Goals

The implementation follows five boundaries:

1. Keep generation and decode state independent per session.
2. Batch only sessions with compatible control frames, cache shapes, and RoPE positions.
3. Make admission depend on actual GPU capacity, not only connection count.
4. Separate LiveKit control from model workers and support multi-GPU migration.
5. Preserve B1/eager fallback when batching, memory, or CUDA Graph constraints are not met.

## Architecture Overview

```text
LiveKit / HTTP control plane
  | session creation, admission, ownership, control messages
  v
ABotWorldLiveKitService
  | deadline-aware scheduling, compatibility, backpressure
  v
Model workers (one per GPU)
  | collate session state -> one DiT execution -> scatter state
  v
Per-session TAEW/VAE decode and output queues
```

The main ownership boundaries are:

| Area | Owner | Responsibility |
|---|---|---|
| Session state | `ABotWorldInteractiveSession` | Prompt, KV cache, scheduler, RNG, and temporal decode state |
| Single-GPU service | `ABotWorldLiveKitService` | Admission, scheduling, batching, pacing, and lifecycle |
| Process workers | `process_worker_pool.py` | Model execution outside the control plane |
| Multi-GPU workers | `nccl_process_worker_pool.py` | One model worker per GPU and fixed NCCL communicator |
| Cluster routing | `turboserve.py` | Load-aware placement, ownership, and migration decisions |

Model code owns model inputs, cache updates, and decode. Connection management, GPU selection, and cross-worker routing
remain outside the model.

## Session Isolation

Each `ABotWorldInteractiveSession` owns its prompt embedding, first-frame latent, self/cross KV caches, scheduler and
RNG state, VAE/TAEW temporal decode state, and chunk/frame counters. An idle session can move retained tensors to CPU
and later restore them without changing its logical owner.

The batch therefore shares only short-lived model computation. Mutable generation state is never shared: after one batch
finishes, caches, decode state, and outputs return to their respective sessions.

## Compatible Batching

The scheduler orders ready sessions by their earliest playout deadline. Before forming a batch it validates the control
latent count, first-versus-continuation stage, absolute-RoPE `next_latent_frame`, and latent/cache shape and dtype.

Compatible sessions form B2 or B3 within a short formation window:

```text
ready sessions
      | compatibility check
      v
collate prompt / noise / action / KV cache
      | one denoise_interactive_blocks call
      v
scatter cache and latent outputs
      | per-session decode and queue
      v
independent LiveKit delivery
```

When first chunks and continuations cannot be mixed safely, the scheduler reduces the batch or uses B1. This trades
some throughput for deterministic cache pointers, position encoding, and frame ordering.

## Multi-GPU Workers and Migration

The parent process owns LiveKit, HTTP admission, and ownership; worker processes own model instances and GPU tensors.
In `process-nccl` mode, one worker is started per GPU with a fixed NCCL communicator. Clients see one service endpoint
and never select a GPU.

Migration is a chunk-boundary transaction: quiesce new work, drain output and publisher credit, snapshot state, restore
it on the target worker, and commit a new route with a monotonically increasing ownership epoch. If the commit fails,
the source owner remains authoritative instead of leaving two workers with partial state.

## Backpressure and Fallback

Real-time output supports `latest`, which drops the oldest complete block when a queue is full, and `lossless`, which
blocks only the affected session. CUDA Graph replays are used only for profiled-compatible shapes and state; other inputs
use eager execution. Failed compatibility checks similarly fall back to a smaller batch or B1.

## Performance Results

### Target compute on one H100

In the LF3 microbenchmark, aggregate throughput increased from 30.98 FPS at B1 to 37.01 FPS at B4, a 19.5% gain. B4
met an 8 FPS, 1.5-second chunk deadline; larger batches did not. This measures model compute only and excludes
network, LiveKit, and browser delivery.

### Four-GPU trace

The 30-minute four-GPU trace on 2026-08-19 produced the following results:

| Metric | Result |
|---|---:|
| Admission | 295/295 immediate, 0 failed |
| Real model calls | 5,475, all successful |
| B1 / B2 / B3 calls | 4,533 / 907 / 35 |
| B2 chunk-item share | 28.1% |
| B3 chunk-item share | 1.6% |
| Fewer calls than all-B1 execution | 977 (15.1%) |
| Graph fallback | 0 |
| Demand SLO | 81.8% |
| Mean active-user FPS | 11.13 |
| A2F p95 | 4.13 s |

The dispatch histogram represents:

```text
4533 + 2 × 907 + 3 × 35 = 6452
```

All-B1 execution would require 6,452 model calls. The trace used 5,475, saving 977 calls. This difference reflects
model-execution reuse from compatible batching rather than dropping sessions.

## Correctness and Applicability

Tests cover session isolation, 30-block generation, batch ordering, cache scatter, worker lifecycle, and migration
transactions. The long trace also exercises admission, output queues, and multi-worker dispatch. Every model dispatch in
the trace returned `ok`, and none of the four workers used Graph fallback.

Mean active-user FPS was 11.13 and demand-SLO attainment was 81.8%. Batching gains depend on control-message arrival
timing and tensor compatibility; these results primarily show the capacity and compute-reuse benefits of multi-user
serving.

## Reproduction Record

The experiment record is under:

```text
results/experiments/abot_4gpu1237_lf3_12fps_publicdemo_b3_f36_graph_30min_20260819T060933Z/
```

The workload is normalized from a public TurboServe trace and covers admission, batch dispatch, state retention, output
delivery, and multi-GPU worker behavior.

## Limitations

- Results cover one host with four H100 GPUs;
- batch formation depends on session arrival timing and input compatibility;
- migration cost depends on worker topology, memory capacity, and transfer configuration;
- demand-SLO attainment in this trace is below 100%.

## Related Work

The implementation combines continuous batching, deadline-aware scheduling, and multi-GPU worker pools with
ABot-World KV caches, TAEW/VAE temporal state, and LiveKit real-time delivery.
