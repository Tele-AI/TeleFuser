# ABot-World-0-5B-LF

TeleFuser provides a TurboServe-style concurrent integration for the public
ABot-World-0-5B-LF long-forcing checkpoint. Each model replica remains single-GPU,
while a LiveKit deployment can run one replica per configured GPU and continuously
batch compatible retained sessions. There are two supported transport entry points:

* Native HTTP controller for local debugging:

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --host 127.0.0.1 \
  --port 7860
```

* The shared `stream-serve` LiveKit path, which reuses TeleFuser room admission,
  reliable `tf.control` messages, WebRTC media publication, pacing, and the
  existing LingBot browser controller:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=0 \
telefuser stream-serve examples/abot_world/abot_world_livekit_service.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --worker-gpu-map 0 --max-sessions-per-worker 1 \
  --port 8088 --skip-validation
```

Start coturn with one fixed relay port for this single-session SSH setup, then start LiveKit:

```bash
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 \
  --listening-port=3478 --min-port=49160 --max-port=49160 \
  --user=livekit-demo:livekit-demo-password --realm=livekit.local \
  --fingerprint --lt-cred-mech --no-tls --no-dtls --no-cli \
  --allow-loopback-peers
```

Then run `livekit-server --dev` as described in the [stream
server guide](stream_server.md), and serve the reused ABot browser defaults:

```bash
python examples/abot_world/abot_world_livekit.py \
  --server-url http://127.0.0.1:8088 --port 8092 --no-open
```

For an SSH session, forward remote TCP ports `8092`, `7880`, `3478`, and `49160` to the
same local ports. The browser wrapper proxies the TeleFuser API, so port 8088
does not need forwarding.

## Checkpoint And Image

The loader expects the unmodified checkpoint layout:

```text
ABot-World-0-5B-LF/
  diffusion_pytorch_model.safetensors
  Wan2.2_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
```

The browser's default sample image comes from the official ABot-World web
client asset at `../ABot-World/web_client/datasets/images/84b90ad568b693d2.png`.
Pass a different server-side image path from the page when needed.

## Pipeline Structure

`ABotWorldPipeline` is a TeleFuser `BasePipeline`. It uses the existing Wan
VAE and text stages plus the model-specific `ABotWorldDenoisingStage`.
`ABotWorldDiT` uses the public TeleFuser attention operations and the official
four-step x0-prediction causal sampler.

`ABotWorldInteractivePipeline` retains the prompt embedding, initial-image
latent, self/cross KV caches, scheduler, RNG, and VAE temporal cache per session.
The scheduler owns one GPU execution thread per replica and batches compatible
sessions through DiT and cached VAE decode. Per-session RNG draws, KV/VAE cache
scatter, relative RoPE positions, and chunk counters remain isolated.

The service performs a real one-session warmup before admission, separates retained
state from temporary workspace memory, and applies the resulting capacity ceiling. The
planner treats the measured one-item workspace as batch-scaled: for each candidate
capacity it budgets every retained session plus `min(candidate, max_batch_size)`
workspace items under a 10% free-memory reserve. This avoids advertising a capacity
that is safe only for batch size one.
Idle state can be suspended to CPU. A two-phase chunk-boundary migration snapshot
contains prompt state, RNG, KV, VAE caches, counters, and ownership epoch; the
in-process LiveKit router changes model ownership only after target import succeeds.

## Controls And Idle Behavior

WASD and arrow keys control movement. IJKL controls camera rotation. Connect
creates the image-conditioned session and displays the input preview; it does
not advance the DiT with an empty action state. A non-empty control snapshot
starts the next three-latent causal block. Releasing all keys stops new model
execution without discarding frames already queued for playback.

The browser consumes decoded frames in order at 12 FPS. Every session has a
bounded output queue. The default `latest` delivery mode evicts the oldest complete
video block when a slow client fills that queue and increments drop metrics, keeping
control latency bounded. `delivery_mode=lossless` instead applies per-session
scheduling backpressure; it does not block other ready sessions.

## KV And RoPE

The default causal window is 18 latent frames: six sink frames plus a
twelve-frame rolling tail. KV cache entries remain unrotated; RoPE is applied
when keys are read using bounded logical positions. Sink positions are `0..5`
and the rolling tail occupies the remaining local window, so the global session
frame number does not grow the RoPE index past the precomputed table.

This fixed logical position policy is an intentional difference from the
original non-sink ABot baseline and must be evaluated as part of any future
long-horizon quality claim.

## Four-GPU Black-Box User-Wave Baseline

The checked-in ABot factory fixes the real-time baseline at **12 FPS**,
**three control latents per chunk** (12 decoded frames), `scheduler_mode=batched`,
and `max_batch_size=2`. The following launch uses four physical GPUs (4--7),
one spawned model worker per GPU, two retained sessions per worker, and a bounded
HTTP admission queue. It is a black-box deployment: clients call only the public
HTTP/LiveKit interfaces and never select a GPU.

Start a local LiveKit server in a separate terminal. A loopback experiment does
not need the public TURN setup; retain the TURN setup above when clients are remote.

```bash
livekit-server --dev --bind 127.0.0.1
```

Use the source-tree CLI below. The currently installed `telefuser` console script
may be older than this checkout and omit `process-nccl`; `python -m` with
`PYTHONPATH=$PWD` is therefore intentional.

```bash
cd /public/fanyk1/lwb/TeleFuser-abot-world

export PYTHONPATH=$PWD
export TF_MODEL_ZOO_PATH=/public/fanyk1/lwb/model_zoo
export CUDA_VISIBLE_DEVICES=4,5,6,7
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

/public/fanyk1/lwb/envs/telefuser_sage291/bin/python -m telefuser.entrypoints.cli.main \
  stream-serve examples/abot_world/abot_world_livekit_service.py \
  --host 127.0.0.1 --port 8088 \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --num-workers 4 --worker-gpu-map '0;1;2;3' \
  --worker-mode process-nccl \
  --max-sessions-per-worker 2 --queue-size 16 \
  --skip-validation
```

`CUDA_VISIBLE_DEVICES=4,5,6,7` remaps physical GPUs 4--7 to logical IDs 0--3,
which is why the worker map is exactly `'0;1;2;3'` rather than `'4;5;6;7'`.
The parent admission scheduler assigns each incoming session to the least-loaded
worker; no browser or workload-client GPU argument exists. `process-nccl` keeps
LiveKit transport in the parent, holds model state in the children, batches each
worker's compatible ready sessions, and can migrate retained state at a chunk
boundary. Rebalancing is enabled by default.

Wait for readiness and record the public configuration before the load starts:

```bash
curl --noproxy '*' --fail --silent http://127.0.0.1:8088/v1/service/ready
curl --noproxy '*' --fail --silent http://127.0.0.1:8088/v1/service/metadata | python -m json.tool
```

The metadata must report `worker_mode: process-nccl`, `num_workers: 4`, and
`configured_max_sessions_per_worker: 2`. It also reports the routing snapshot and
any rebalance decision. Do not claim a migration result merely because
`migration_supported` is true: first pass the focused ABot TAeW migration validation
after its state-snapshot patch has landed. A balanced user wave normally exercises
placement, local batching, admission queueing, and recovery; it need not create an
imbalanced placement worth migrating.

### Experimental EDF deadline-aware micro-batching

`TELEFUSER_ABOT_MAX_DEADLINE_BATCH_WAIT_MS` is a separate, opt-in upper bound for
waiting to form a compatible continuation batch. It does **not** replace
`TELEFUSER_ABOT_BATCHING_WINDOW_MS`, which remains the scheduler's legacy/pacing
coalescing window. Its default is `0`, preserving the baseline behavior.

For an initial B=2 experiment, use a conservative 100 ms cap:

```bash
export TELEFUSER_ABOT_SCHEDULER_MODE=batched
export TELEFUSER_ABOT_MAX_BATCH_SIZE=2
export TELEFUSER_ABOT_BATCHING_WINDOW_MS=2
export TELEFUSER_ABOT_MAX_DEADLINE_BATCH_WAIT_MS=100
```

### Offline H100 batch-time priors

Before the first real B=2 dispatch, the generic scheduler would otherwise estimate
it as two B=1 calls. For the measured ABot-World-0.5B-LF, LF=3, H100 full-pipeline
profile, select a named, explicit prior table instead:

```bash
# Eager P95 raw seconds: B2=0.7405, B3=1.0691, B4=1.4073.
export TELEFUSER_ABOT_BATCH_COMPUTE_PROFILE=h100_lf3_eager_full_pipeline_v1

# Or use the separately validated CUDA-Graph profile: B2=0.6923, B3=1.0319.
# export TELEFUSER_ABOT_BATCH_COMPUTE_PROFILE=h100_lf3_cuda_graph_v1

# Optional: use raw P95 × 1.05 for this run (default: ×1.10).
export TELEFUSER_ABOT_BATCH_COMPUTE_SAFETY_FACTOR=1.05
```

The scheduler applies this factor once to raw profile/observed timing, then retains the
maximum of the selected prior and every observed runtime. It must be finite and at least
`1.0`; the default is `1.10`. The default profile is `none`; do not select an H100
profile for a mismatched GPU, model shape, LF, or execution backend.

When the GPU becomes free, the scheduler holds the EDF-earliest compatible
continuation only until the earlier of this cap and its deadline-safe B=2 start.
With `MAX_BATCH_SIZE=3`, once a compatible B arrives, it waits for C only when C's
predicted release is before both the retained B=2 fallback start and the B=3
latest-safe-start; otherwise it dispatches B=2 immediately. If no peer arrives, the
held session falls back to B=1. First chunks never wait. If another EDF job is
ready during the hold, it runs first only when both its own deadline and the held
session's B=1 fallback deadline remain safe. Inspect `deadline_batch_waits_started`,
`deadline_batch_wait_timeouts`, and `deadline_batch_filler_dispatches` in
`/v1/service/metrics/json` when evaluating the policy.

### Experimental publisher-frame-credit EDF

To make EDF use playout slack rather than only ABot's local output queue, enable
the following **opt-in** policy:

```bash
export TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_ENABLED=true
# Default ABot FPS is 12, so 3 seconds yields F=36 frames.
export TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_SECONDS=3.0
# Optional exact override; it takes precedence over TARGET_SECONDS.
export TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_TARGET_FRAMES=36
export TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_RESERVE_FRAMES=4
export TELEFUSER_ABOT_PUBLISHER_FRAME_CREDIT_GUARD_MS=50
```

The service tracks `F = queued chunk frames + frames dequeued by the publisher but
not yet accepted by LiveKit's `capture_frame`. `capture_frame` is a server-side
transport handoff, not a browser render acknowledgement. For latest-mode
continuations, EDF computes its safe start from `F`, keeps the configured reserve and
guard, and waits for a compatible peer only while the B=2 fallback remains safe.
First chunks and lossless sessions retain their existing behavior.
`TARGET_FRAMES` changes only the low-watermark at which a latest-mode session becomes
eligible again; it does not replace the reserve used to calculate the deadline. With
`F=36`, 12 FPS, a 4-frame reserve, and a 50 ms guard, the maximum modeled completion
slack is `(36 - 4) / 12 - 0.05 = 2.617s`.

Inspect `queued_video_frames`, `publisher_unsubmitted_frames`, `frame_credit_frames`, and
`frame_credit_deadline_in_seconds` in `/v1/service/metrics/json` or dispatch JSONL.

### Run the arrival/burst/recovery workload

The tracked scenario at
`tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_wave.json` drives real
LiveKit controller clients. Each client creates a session through
`POST /v1/stream/sessions`, joins its room, sends reliable `tf.control` states,
counts received WebRTC video frames, and deletes its public session on departure.
The workload has four phases: 4-user warmup, 8-user SLO-capacity ramp, 12-user
burst (four requests should queue under the 4 x 2 admission cap), then recovery to
4 users. Scale-down removes the newest sessions first, so long-lived session state
is preserved; a client stays counted until its scheduled shutdown actually runs.

```bash
cd /public/fanyk1/lwb/TeleFuser-abot-world

PYTHONPATH=$PWD /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \
  tools/validation/benchmark_abot_livekit_burst.py \
  --scenario tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_wave.json \
  --output /public/fanyk1/lwb/results/experiments/abot_livekit_4gpu_lf3_12fps_wave/result.json
```

Use `--dry-run` first to validate the JSON without contacting the service. The
result artifact keeps raw one-second delivery samples, session admissions, phase
events, server metadata snapshots, and a concise phase table. `Aggregate FPS` is
all generated frames actually received by the WebRTC clients. `FPS/all-user` and SLO
attainment use every requested session after its 15-second grace interval; an
unassigned, disconnected, or stalled request contributes zero FPS. The legacy
`FPS/controlled` field remains available for comparison with active media clients.

This is an end-to-end client-delivery experiment, not a model-only benchmark.
Correlate its artifact with `/metrics` or the Prometheus/Grafana stack for GPU,
queue, batching, and stage telemetry; do not infer GPU utilization from delivery
FPS alone.

### All-active peak-16 capacity trace

Use the following trace to measure fixed four-GPU overload behavior without hiding
users behind an admission queue. It requires four warm workers, four retained
sessions per worker, batch cap four, and `--queue-size 0`: the 16 requested users
must all receive an immediate `assigned` response. A queued, rejected, disconnected,
or zero-output user remains in the `FPS/all-user` and SLO denominator after grace;
there is no per-client GPU selection.

| Phase | Duration | Arrival/departure window | Target users |
| --- | ---: | ---: | ---: |
| warmup | 45 s | arrivals over 10 s | 4 |
| ramps | 55 s + 55 s | arrivals over 15 s | 8, 12 |
| peak | 80 s | arrivals over 15 s | 16 |
| recovery | 50 s + 45 s | departures over 15 s | 8, 4 |

Start the service on physical GPUs 0--3 with the all-active profile:

```bash
cd /public/fanyk1/lwb/TeleFuser-abot-world
export PYTHONPATH=$PWD
export TF_MODEL_ZOO_PATH=/public/fanyk1/lwb/model_zoo
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TELEFUSER_ABOT_SCHEDULER_MODE=batched
export TELEFUSER_ABOT_MAX_BATCH_SIZE=4
export TELEFUSER_ABOT_BATCHING_WINDOW_MS=2

/public/fanyk1/lwb/envs/telefuser_sage291/bin/python -m telefuser.entrypoints.cli.main \
  stream-serve examples/abot_world/abot_world_livekit_service.py \
  --host 127.0.0.1 --port 8088 \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --num-workers 4 --worker-gpu-map '0;1;2;3' \
  --worker-mode process-nccl \
  --max-sessions-per-worker 4 --queue-size 0 \
  --skip-validation
```

Then run the black-box client trace:

```bash
PYTHONPATH=$PWD /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \
  tools/validation/benchmark_abot_livekit_burst.py \
  --scenario tools/validation/workloads/abot_livekit_4gpu_lf3_12fps_all_active_peak16_wave.json \
  --output /public/fanyk1/lwb/results/experiments/abot_livekit_4gpu_lf3_12fps_all_active_peak16_wave/result.json
```

The printed `admitted` column must be `16/16` in the peak phase. The JSON phase
summary also records `max_requested_users`, all-user FPS, and immediate-assignment
contract status, so a queue or failed admission invalidates rather than improves the
reported serving result.

## Multi-GPU and autoscaling

`process-nccl` deliberately uses a fixed one-GPU-per-worker NCCL group, so it does
not combine with process autoscaling. Plain `--worker-mode process` remains
independent-replica batching and reports `migration_supported: false`; it is not a
TurboServe migration baseline. In-process migration remains useful for debugging,
but stages state via CPU.

For plain `process` mode, optional cold-replica autoscaling starts only the requested
minimum and scales within the GPUs declared above. For example:

```bash
  --enable-autoscaling --autoscaling-min-workers 1 \
  --autoscaling-target-utilization 0.75 \
  --autoscaling-hysteresis 0.10 \
  --autoscaling-cooldown-seconds 30 --autoscaling-interval-seconds 5
```

Because scale-out loads a checkpoint, autoscaling with multiple workers requires
a non-zero session queue.

## Tests and benchmark

CPU contract tests cover model conversion, sink KV rolling, RoPE boundaries,
session cleanup, idle behavior, FIFO backpressure, and action layout:

```bash
python -m pytest tests/unit/models/test_wan22_video_vae.py \
  tests/unit/pipelines/abot_world -q
```

The opt-in GPU smoke uses the release checkpoint, the public `480x832` shape,
a fixed seed, and 30 control blocks:

```bash
CUDA_VISIBLE_DEVICES=0 \
ABOT_WORLD_MODEL_ROOT=/path/to/ABot-World-0-5B-LF \
ABOT_WORLD_TEST_IMAGE=/path/to/initial.png \
python -m pytest -m "gpu and slow" \
  tests/integration/test_abot_world_smoke.py -v -s
```

The deterministic continuous-batching benchmark runs multiple sessions for
30 blocks and writes stage/batch latency plus throughput JSON:

```bash
python tools/validation/benchmark_abot_turboserve.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --image /path/to/initial.png \
  --sessions 2 --chunks 30 --batch-size 2 \
  --output /tmp/abot-turboserve.json
```

The smoke and benchmark are generation, cache-isolation, batching, and ordering
contract tests. They do not establish visual quality, prompt fidelity, or parity
over an unbounded session. Every ABot replica remains single-GPU; multi-GPU
deployments scale with independent replicas rather than tensor parallelism.

For a service-level workload with bursty arrivals, independent keyboard activity,
and playback-paced consumers, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/validation/benchmark_abot_turboserve_concurrent.py \
  --model-root /path/to/ABot-World-0-5B-LF --image /path/to/initial.png \
  --sessions 4 --duration-seconds 12 --arrival-window-seconds 1.5 \
  --max-batch-size 4 --output /tmp/abot-concurrent.json
```

This reports delivered FPS, first-chunk latency, scheduler queue wait, model compute
time, observed batch-size distribution, and per-session latest-queue drops.
