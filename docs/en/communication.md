# Communication Architecture

This guide describes how TeleFuser moves tensors between GPUs and worker processes, how communication ownership is
split across modules, and which invariants keep the implementation correct and efficient. For configuration of DP,
CFG, SP, PP, TP, and FSDP, see the [Parallel Inference Guide](parallel.md).

## Design Goals

TeleFuser communication follows five rules:

1. Keep the control plane separate from large tensor data.
2. Keep reusable collective mechanics out of model implementations.
3. Avoid host staging and device-wide synchronization on same-host GPU paths.
4. Bound retained GPU memory and provide explicit cancellation and shutdown semantics.
5. Preserve a native PyTorch fallback when a specialized transport is not applicable.

The implementation is layered rather than hidden behind one universal transport. NCCL collectives, CUDA IPC, Ray,
and service networking have different topology and lifetime requirements, so they share ownership rules but not one
runtime protocol.

## Architecture Overview

```text
Pipeline / model code
  | declares topology, tensor layout, and stage connections
  v
Parallel strategy and worker adapters
  | DeviceMesh groups, Ulysses, Ring, VAE spatial, ParallelWorker
  v
Shared communication mechanisms
  | collectives.py                 | worker/tensor_channel.py
  | process-group tensor movement | same-host cross-worker tensor movement
  v                                v
PyTorch distributed / NCCL        CUDA IPC + multiprocessing metadata
```

The main ownership boundaries are:

| Area | Owner | Responsibility |
|------|-------|----------------|
| Process topology | `telefuser/distributed/device_mesh.py` | Build and expose named process groups |
| Shared collectives | `telefuser/distributed/collectives.py` | Contiguous gather buffers and grouped reductions |
| Sequence attention | `ulysses_comm.py`, `ring.py` | Strategy-specific All-to-All and P2P protocols |
| Sequence/CFG shards | `parallel_shard.py` | Tensor padding, slicing, gathering, and restoration |
| Spatial VAE | `vae_spatial.py` | Height shards, neighbor halos, and output gathering |
| Pipeline P2P | `pp_comm.py` | Rank-to-rank communication inside a PP process group |
| Worker execution | `worker/parallel_worker.py` | Process-group and worker lifecycle, command dispatch |
| Cross-worker tensors | `worker/tensor_channel.py` | CPU shared memory and persistent CUDA IPC pools |
| Cluster actors | `worker/ray_worker.py` | Ray resource assignment and optional local workers |

Model code owns model-specific layout and reconstruction. It should call a shared collective or a strategy module
instead of allocating rank buffers or invoking tensor collectives directly.

## Process Groups and Device Mesh

`ParallelWorker` starts one spawned process per local rank. For groups larger than one rank it selects the assigned
device, initializes the platform distributed backend, and asks the stage to parallelize its models. CUDA normally
uses NCCL through the platform abstraction.

`create_device_mesh_from_config()` creates named dimensions in this order:

```text
DP -> CFG -> SP (ring, ulysses) -> PP -> TP
```

When Ring and Ulysses are both enabled, SP is a two-dimensional `(ring, ulysses)` submesh. Accessors such as
`get_cfg_group()`, `get_ring_group()`, `get_ulysses_group()`, and `get_pp_group()` prevent models from reconstructing
rank lists independently. The configured world size must equal the product of all enabled degrees. SP and TP are
currently mutually exclusive.

## Shared Collective Primitives

`telefuser/distributed/collectives.py` is an internal implementation boundary. It is intentionally not exported from
the top-level `telefuser.distributed` API.

### Equal-shape gather

`all_gather_stacked()` gathers equal-shaped local tensors into one rank-major allocation:

```text
local [D0, ...]
  -> all_gather_into_tensor
buffer [world_size * D0, ...]
  -> view
result [world_size, D0, ...]
```

This replaces one allocation per rank with one contiguous output buffer. Consumers can retain the rank dimension for
model-specific reconstruction or use `all_gather_cat()` to concatenate along any tensor dimension in rank order.

`parallel_shard.py`, LingBot Video sequence restoration, Wan/Wan2.2 VAE reconstruction, and VAE spatial gathering use
these primitives. Unequal VAE height shards are padded to the maximum local height before gather and trimmed after it.

### Grouped reductions

`all_reduce_sum_()` submits all independent sum reductions asynchronously before waiting for their work handles. Tile
blending uses it for value and weight tensors, avoiding duplicated synchronization code in model implementations.

These helpers assume that all participating ranks call collectives in the same order with compatible shapes and
dtypes. Violating collective order is a distributed deadlock, not a recoverable per-rank error.

## Sequence Parallel Communication

### Ulysses

Ulysses converts sequence shards with global heads into full-sequence tensors with local heads:

```text
[B, S_local, H_global, D]
  -> All-to-All scatter heads / gather sequence
[B, S_global, H_local, D]
  -> local attention
  -> All-to-All gather heads / scatter sequence
[B, S_local, H_global, D]
```

`ulysses_scatter_heads()` and `ulysses_gather_heads()` use functional `all_to_all_single` collectives. They return wait
closures so callers can separate submission from consumption. Attention implementations submit Q, K, and V before
waiting for any of them, allowing NCCL to schedule the three transfers without Python-side serialization. The output
All-to-All restores the original sequence/head layout.

The number of attention heads must be divisible by the Ulysses world size, and the gathered sequence length must be
divisible on the inverse path. The helpers validate these constraints before communication.

### Ring Attention

Ring Attention keeps Q local and rotates K/V through neighboring ranks. `RingP2PComm` resolves group-local neighbors
to global ranks, batches `isend` and `irecv` operations with `batch_isend_irecv`, and reuses caller-provided receive
buffers when available.

K and V may be concatenated into one transfer and split into views after receive. Communication for the next KV block
is submitted before attention on the current block; the implementation waits only before consuming the next block.
Partial attention results are combined with an online log-sum-exp merge.

The AllGather Ring variant is simpler but materializes global K/V on every rank. It is a memory-heavy alternative, not
the preferred long-context path.

## Spatial VAE Communication

Height-sharded VAE decode has two communication patterns:

1. Neighbor halo exchange before a spatial convolution.
2. Rank-ordered gather when a full-height tensor is required.

Halo exchange uses reusable send and receive buffers and one `batch_isend_irecv` call for the available top and bottom
neighbors. Boundary ranks fill missing halos with zero. Buffer reuse avoids allocating contiguous halo tensors at
every layer and every frame.

Full-height reconstruction uses the shared contiguous gather primitive. Rank-local heights are gathered first so
uneven shards can be padded and trimmed correctly. The final tensor restores its original channels-last or contiguous
memory format.

## Cross-Worker Tensor Channel

`WorkerTensorChannel` connects one producer worker group to one consumer worker group on the same host. It separates
small control metadata from tensor storage:

```text
Producer worker                 Parent / control path              Consumer worker
      |                                  |                               |
      | stage tensor                     |                               |
      |-- stage into IPC slot ---------->| WorkerTensorRef metadata ---->|
      |                                  |                               |-- map pool once
      |<------- generation ACK / completion event -----------------------|-- peer copy
```

The parent receives `WorkerTensorRef` objects and never materializes CUDA tensor contents. Nested dictionaries,
tuples, and lists preserve their structure; duplicate tensor leaves are transported once per artifact.

### Persistent CUDA IPC pools

Stable CUDA tensor profiles are keyed by tensor index, shape, dtype, and source device. Each profile owns a persistent
allocation with two slots by default. Slots are selected round-robin so sequential traffic uses real double buffering.

The pool allocation and IPC handle are created once. Consumers cache imported storage and event handles, so steady
state does not reopen CUDA IPC allocations. At most eight profiles are pooled per channel. Additional dynamic profiles
fall back to PyTorch multiprocessing tensor transport rather than retaining unbounded HBM.

When `shard_dim` is set, every consumer rank receives only its rank-local view. The producer stages the tensor once,
and aggregate peer-copy traffic remains one logical tensor rather than one full tensor per consumer. LingBot's spatial
VAE uses height sharding with `shard_dim=-2`.

### Stream ordering protocol

Each slot has a reusable producer-ready event:

1. The producer copies the source tensor into the slot on its current stream.
2. The producer records the ready event and publishes its handle with metadata.
3. The consumer stream waits for the ready event only when it is not already complete.
4. The consumer copies its mapped slot view into an output tensor.
5. The consumer records a reusable completion event before publishing its generation acknowledgement.
6. Before overwriting a reused slot, the producer stream waits for completion events from ranks that copied it.

This protocol contains no device-wide synchronization in the transport path. Event `query()` provides a fast path
when producer staging or consumer copy has already completed.

Acknowledgements use a lock-free shared generation array. A positive generation means that a rank copied the payload;
a negative generation means that it discarded the payload. The producer waits for completion metadata only from
ranks that performed a copy, so cancellation cannot wait for an event that was never recorded.

### CPU and fallback transport

CPU tensors use multiprocessing shared memory. CUDA profiles that cannot be pooled use PyTorch's multiprocessing CUDA
tensor transport. In both cases, one FIFO exists per consumer rank, and the receiving process performs final device
placement.

## Control Plane and Lifecycle

`ParallelWorker` command and result queues carry method names, arguments, small results, and tensor references. They
use `SimpleQueue` to avoid background feeder scheduling tails. Large tensors connected through a
`WorkerTensorChannel` stay on the direct data path.

The channel contract is ordered and bounded:

- Exactly one producer and one consumer group bind to a channel.
- Consumer rank count must match the channel configuration.
- Consumers resolve artifacts in producer order.
- A terminal cancelled artifact must be released with `discard_tensor_refs(..., sync=True)`.
- Shutdown stops consumers before producers and closes the channel last.
- Worker cleanup synchronizes pending device work before releasing local IPC mappings.

Timeouts mark a worker failed and terminate its processes. Reusing a failed worker is rejected rather than risking a
partially ordered channel.

## Pipeline Parallel and Ray Boundaries

`PipelineP2PComm` is a different transport from `WorkerTensorChannel`. It communicates between ranks inside one PP
process group with NCCL send/recv and batched P2P operations. Existing Wan PP shape/grid broadcasts and latent
convenience methods remain owned by that PP path.

CUDA IPC is a same-host mechanism. `RayWorker` respects the logical devices assigned by Ray and may host a local
`ParallelWorker`, but TeleFuser does not replace Ray's cross-node object transport with CUDA IPC. A deployment that
needs cross-node GPU-direct transfer requires a separately designed transport and topology contract.

## Efficiency Invariants

The communication implementation preserves these performance properties:

- No parent-process CUDA tensor materialization on direct worker edges.
- No host staging on the pooled same-host CUDA path.
- Two logical device copies for a full handoff: producer staging and consumer output copy.
- Persistent pool, storage, and event handles in steady state.
- Bounded slots and profile count, preventing unbounded retained HBM.
- Stream events instead of device-wide synchronization.
- Rank-local copying when consumers operate on disjoint shards.
- One contiguous output allocation for equal-shape gather.
- Q/K/V collective submission before waits in Ulysses.
- Batched neighbor P2P and reusable halo buffers in spatial VAE and Ring paths.

## Verification and Benchmarking

Focused tests cover pure collective layout, real two-GPU NCCL ordering, CUDA IPC readiness and slot reuse, cancellation,
multi-consumer acknowledgement, and spatial VAE parity:

```bash
pytest tests/unit/distributed/
pytest tests/integration/test_collectives.py
pytest tests/integration/test_worker_tensor_channel.py
pytest tests/integration/test_wan_video_vae_spatial.py
```

The local SGLang comparison includes producer staging, metadata transport, target copy, target synchronization, and
slot acknowledgement for both implementations:

```bash
python tools/validation/benchmark_tensor_channel_vs_sglang.py
```

Its default gate uses 200 measured transfers. TeleFuser p50 must be no more than 5% above SGLang, while p95 allows the
larger of 10% or 0.05 ms to account for sub-millisecond multiprocessing scheduling jitter. Compare copy counts and
mean latency as well as percentiles; a single process-scheduling tail is not evidence of a transport regression.

Pipeline-level validation should rerun every pipeline whose communication call site changed. The example runner
provides baseline output comparisons:

```bash
python examples/run_examples.py --pipeline <name> --gpus 0,1,2,3
```

## Extension Rules

When adding a communication path:

1. Put generic equal-shape gather or reduction mechanics in `distributed/collectives.py`.
2. Put algorithm-specific protocols in a focused module under `telefuser/distributed/`.
3. Keep model code responsible only for tensor layout and model semantics.
4. Use `WorkerTensorChannel` only for a same-host, single-producer/single-consumer-group edge.
5. Do not add a new fallback, environment variable, or public configuration field without a concrete topology gap.
6. Specify ordering, ownership, cancellation, timeout, and shutdown before optimizing the happy path.
7. Add a real multi-process test for any new collective or IPC synchronization rule.

Do not route large tensors through the parent merely because the control path already exists. Do not use a device-wide
synchronize to repair an ordering bug; express the dependency with process-group work handles or stream events.

## Known Limits

- CUDA IPC pools are same-host only.
- Stable pooled profiles require fixed tensor index, shape, dtype, and source device.
- Ring AllGather trades implementation simplicity for replicated K/V memory.
- Spatial VAE halo exchange reuses buffers but currently waits before the dependent convolution.
- WAN pipeline-parallel communication remains a separate, model-specific compatibility area.
- Ray cross-node tensor performance depends on Ray transport and cluster configuration.

## Related Documentation

- [Parallel Inference Guide](parallel.md)
- [Attention Implementation Guide](attention.md)
- [Streaming Scheduler](stream_scheduler.md)
- [Testing Guide](testing.md)
