# Streaming Pipeline Scheduler

## Purpose

`StreamingPipelineOrchestrator` coordinates long-running, stateful generation pipelines whose work arrives and
completes as ordered chunks. It is intended for interactive workloads such as LingBot, where new control input can
arrive while earlier chunks are encoding, denoising, or decoding.

The scheduler is distinct from `FlexiblePipelineOrchestrator`. The flexible orchestrator coordinates request-level
stage groups; the streaming scheduler owns bounded, per-session dataflow and persistent stage actors.

## Architecture

The scheduler executes a directed acyclic graph of typed artifacts:

```text
external input
     |
     v
  encode -- condition --> denoise -- latent --> decode -- frames --> output
                            ^
                            |
                         control
```

Each logical stage is represented by one long-lived actor. Independent actors may run concurrently even when their
workers use the same physical GPU. CUDA device placement alone does not imply serialization or resource ownership.

| Component | Responsibility |
| --- | --- |
| `StreamingStageSpec` | Declares stage inputs, outputs, ordering, admission limits, and an optional resource group. |
| `StreamingEdgeSpec` | Declares a bounded artifact path and its per-session capacity. |
| `StreamingPipelineSpec` | Defines the complete graph, outputs, and resource groups. |
| `LocalStageActor` | Serializes execution for one local state-owning stage. |
| `ParallelWorkerStageActor` | Gives one `ParallelWorker` a single actor owner. |
| `StreamingPipelineOrchestrator` | Validates the graph and schedules ready sequence items across sessions. |

## Dataflow and Ordering

Every input, intermediate artifact, and output is associated with a session ID and sequence ID. The default
`StageOrdering.PER_SESSION_STRICT` preserves the causal order of state mutations within a session while allowing fair
interleaving between sessions.

Edges and outputs have explicit capacities. When a downstream stage cannot accept more work, the scheduler applies
backpressure rather than retaining unbounded tensors. Pipeline implementations must therefore treat submission as
admission-controlled, not as an unbounded queue.

## Actor Ownership and Session Lifecycle

A state-owning worker has exactly one actor owner for its entire lifetime. In particular, one `ParallelWorker` must
not be invoked directly by a session facade or shared by multiple stage actors. This preserves result ordering and
ensures that cache mutation and release occur in one well-defined execution context.

Session shutdown is ordered as follows:

1. Stop admitting new work.
2. Drain or cancel accepted work according to the session policy.
3. Release stage-owned state in reverse topological order through the owning actors.
4. Release scheduler artifact references and verify that no capacity slots remain allocated.
5. Record cleanup failures and do not reuse partially released state.

LingBot uses this lifecycle for both offline chunked generation and bidirectional WebRTC sessions.

## Resource Groups and Placement

`StreamingResourceGroupSpec` represents an explicit shared concurrency constraint. A stage participates only when
its `StreamingStageSpec.resource_group` names a group declared by `StreamingPipelineSpec.resource_groups`.

Do not infer a resource group from `device_id` or `ParallelConfig.device_ids`. For LingBot, VAE encode, DiT, and VAE
decode are independent actors and may overlap on the same GPU. If a placement exceeds memory capacity, move stages to
different devices or define a deliberate deployment constraint; do not add an implicit global mutex.

LingBot uses independent `vae_encode_config` and `vae_decode_config`. Each VAE
stage receives its own complete `ModelRuntimeConfig`; there is no shared VAE
placement fallback.

## Observability and Real-Time Operation

`StreamingSessionMetrics` records scheduler-observed timing and lifecycle data, including:

| Signal | Operational use |
| --- | --- |
| First-output latency | Time from first accepted ingress to first emitted output. |
| Control-to-output latency | Time from an accepted control/input to its corresponding output. |
| Chunk period | Cadence between consecutive output chunks. |
| Stage timing | Input-ready, admitted, and completed timestamps for each invocation. |
| Idle intervals | Admission gaps and their blocking reason. |
| Diagnostics | Stale, orphaned, duplicate, cleanup-failure, and slot-leak counters. |

For real-time operation, compare p95 chunk period with the media duration represented by a chunk:

```text
real-time factor = p95 chunk period / chunk media duration
```

A value below one indicates that generation normally stays ahead of playback. Production capacity planning must still
reserve margin for encoding, transport, and scheduling jitter.

## Integration Requirements

When integrating a streaming pipeline:

- Keep model-specific preprocessing and cache behavior outside the generic scheduler.
- Give every state-owning worker exactly one actor owner.
- Define a bounded edge for every tensor-bearing artifact path.
- Preserve session and sequence IDs from ingress through output.
- Isolate session state and release it through the owning actor.
- Declare resource groups only for real, explicit deployment constraints.
- Validate interleaved sessions, backpressure, cancellation, actor failures, and cleanup failures.
