# VLA Action Integration

TeleFuser keeps model lifecycle, GPU placement, request cancellation, and monitoring in the existing pipeline and
service layers. VLA integration adds semantic action boundaries around those facilities; it does not create a second
model server.

## Data Flow

```text
robot observation
    -> EmbodimentAdapter.encode_observation
    -> VLAPolicy.predict
    -> ModelActionChunk
    -> EmbodimentAdapter.decode_actions
    -> RobotActionChunk
    -> ChunkExecutor (horizon, age, safety)
    -> SimulatorAdapter.execute
```

Tensor width is never used to infer action meaning. Every chunk carries an `ActionSpaceSpec` with representation,
ordered dimension names, units, frame, control rate, and normalization identity. Session opening rejects a policy and
embodiment pair whose action spaces are semantically incompatible.

Each embodiment also declares an `ObservationSpaceSpec` with ordered state dimensions, named image dtype, layout,
and channel requirements, plus the timestamp unit and clock domain. Image height and width remain dynamic so
simulators can choose their native camera resolution.

The public modules are:

- `telefuser.vla.contracts`: observations, requests, capabilities, action spaces, and model/robot chunks.
- `telefuser.vla.policy`: the model-facing `VLAPolicy` protocol.
- `telefuser.vla.embodiment`: observation encoding and model-to-robot action mapping.
- `telefuser.vla.registry`: explicit registration of already-loaded policies and embodiments.
- `telefuser.vla.session`: transport-neutral OPEN, PREDICT, RESET, and CLOSE lifecycle.
- `telefuser.vla.serialization`: versioned JSON/Base64 wire formats for action and observation spaces, observations,
  and chunks.
- `telefuser.vla.runtime`: scheduling, deterministic chunk state, action trimming, age checks, and safety policies.
- `telefuser.integrations.sim`: simulator protocol and the dependency-free RoboTwin callback adapter.
- `telefuser.service.vla_session`: the additive generic VLA WebSocket application factory.
- `telefuser.service.vla_replica`: worker-local OPEN, PREDICT, RESET, and CLOSE dispatch for pipeline replicas.
- `telefuser.client.AsyncVLAClient`: remote session client with concurrent request correlation.

## LingBot-VLA v2 Compatibility

`LingBotVlaV2VLAPolicy` wraps `LingBotVlaV2Pipeline`; the pipeline's existing tensor input and return types are not
changed. It labels the normalized canonical `[T,55]` result as a `ModelActionChunk`.

`RobotWinProfile` is the first `EmbodimentAdapter`. It preserves the existing normalization code and maps a canonical
chunk to an absolute-position `[H,14]` `RobotActionChunk` in the declared dual-arm joint order. Its model and robot
action spaces are exported as `LINGBOT_VLA_V2_ACTION_SPACE` and `ROBOTWIN_ACTION_SPACE`.

The generic VLA WebSocket is the primary online integration path. The existing LingBot RoboTwin WebSocket endpoint
is a compatibility adapter for unmodified upstream `WebsocketClientPolicy` clients and calls the same policy,
embodiment, session, and runtime path internally. Its URL, MessagePack request fields, metadata frame, response fields,
reset behavior, and latest-wins
scheduler behavior remain compatible. The old model-specific scheduler import is retained as an alias to the common
runtime scheduler.

The standalone endpoint owns one resident pipeline, so each connection session is inherently pinned to that policy
instance. The shared HTTP structured-task route remains unchanged and continues to return the existing canonical
JSON result.

## Session Lifecycle

Create and register loaded components, then open a typed session:

```python
from telefuser.pipelines.lingbot_vla_v2 import LingBotVlaV2VLAPolicy, RobotWinProfile
from telefuser.vla import VLARegistry, VLASessionManager

registry = VLARegistry()
registry.register_policy("lingbot-vla-v2", LingBotVlaV2VLAPolicy(pipeline))
registry.register_embodiment("robotwin", RobotWinProfile.default())

sessions = VLASessionManager(registry)
session = sessions.open(
    "connection-1",
    model_id="lingbot-vla-v2",
    embodiment_id="robotwin",
    episode_id="episode-1",
    execute_horizon=8,
)
robot_chunk = session.predict(robot_observation, "pick up the block", sequence_id=0, seed=7)
sessions.reset("connection-1")
sessions.close("connection-1")
```

Sequence IDs must strictly increase between resets. A policy result must echo the request episode, sequence, and
observation timestamp. `ChunkExecutor` can reject stale observations when `max_observation_age_ns` and a same-clock
`now_ns` are supplied. Network round-trip deadlines must still be enforced by the simulator client because monotonic
clocks on separate machines are not comparable.

## Generic WebSocket Protocol

`create_vla_session_app` exposes `/v1/vla/session` for an in-process registry, while
`create_pipeline_pool_vla_session_app` exposes the same protocol through worker-local sessions pinned by
`PipelinePool`. Neither factory changes an existing TeleFuser service route. The server
first sends `HELLO` with protocol version `1.0`, wire encoding, supported operations, registered component IDs, and
payload limits. A client then uses `OPEN`, ordered `PREDICT`, `RESET`, and `CLOSE`. `OPEN` can declare the expected
robot action and observation spaces; semantic incompatibility is rejected before inference.

The protocol returns machine-readable error codes for malformed messages, unsupported versions, unknown components
or sessions, contract mismatches, superseded work, out-of-order or expired observations, request timeout, unavailable
sessions, and replica failure. Tensor payloads are dense, typed, shape-checked Base64 data inside bounded JSON
messages. This is the stable interoperability format; the LingBot-RoboTwin compatibility endpoint continues to use
its existing MessagePack format. New model and simulator integrations must target the generic protocol rather than
add behavior to the model-specific compatibility endpoint.

`request_ttl_ms` is measured with server monotonic time and bounds inference delivery. Observation age is checked
separately using `observation_timestamp_ns` and `observation_clock_now_ns`, which must come from the same clock domain.
The server permits inference and observation delivery to overlap. Per session it runs one replica request and retains
only the newest waiting observation; replaced requests return `superseded`. `RESET` and `CLOSE` are barriers: they
invalidate earlier tickets and wait for any non-cancellable replica call before acknowledging. A discarded stateful
inference triggers policy reset before the retained observation runs. After a timed-out inference finishes, the
session is reset before it accepts more prediction work.

## Replica Session Leases

`PipelinePool.open_session` removes one healthy replica from the ordinary request pool and binds it to a
`session_id`. Calls through `acquire_session` always reach that replica and are serialized per session.
`close_session` returns a live replica to ordinary capacity. Replica exit removes the binding and raises
`ReplicaDeadError`; pool shutdown marks sessions as closing, waits for active session calls, clears every binding, and
then stops replicas.

This API is additive. Existing HTTP work continues to use `PipelinePool.acquire()`, whose allocation behavior and
callers are unchanged. When `PipelinePool.start_all(..., vla_provider_factory="get_vla_provider")` is requested, each
worker constructs its own provider around the already-loaded pipeline and handles VLA lifecycle RPC. The provider is
optional; pipeline files and callers that do not enable it keep the original task and shutdown protocols.

## Chunk State Machine

`ActionChunkStateMachine` makes action lifecycle explicit: pending inference, ready buffer, execution, and terminal
executed, superseded, expired, or rejected states. It supports latest-sequence admission, separate observation-age and
network-TTL checks, configurable discard/retain handling after `execute_horizon`, reset, and deterministic hold/stop
states after disconnect. Stateful policies must provide a recovery callback; it is invoked when an inference that may
have mutated history is discarded.

The generic server tracks inference admission through PENDING, READY, SUPERSEDED, EXPIRED, and REJECTED. It does not
claim that a returned action was executed. `SimulatorChunkRuntime` runs on the simulator client and owns READY,
EXECUTING, EXECUTED, HOLD, and STOP transitions while calling a `SimulatorAdapter` one action at a time.

The bundled RoboTwin statistics do not declare an authoritative simulator control frequency, so both exported
LingBot/RoboTwin specs use `control_hz=None`. The remote simulator must resolve its actual control period rather than
guessing one in the inference process.

## Simulator Boundary

`RoboTwinSimulatorAdapter` has no RoboTwin, SAPIEN, Vulkan, or ROS dependency. The RTX-side process provides observe,
execute, and reset callbacks; the adapter validates semantics and converts each action to contiguous CPU `float32`
before invoking RoboTwin. This keeps simulator dependencies outside the H100 inference environment.

Adding another simulator requires a new `SimulatorAdapter`; it must not add model-specific decoding. Adding another
robot requires an `EmbodimentAdapter`. Adding another VLA requires a `VLAPolicy` wrapper around its maintained
pipeline.

## Service Boundary

The generic WebSocket application remains an explicit factory rather than an automatically mounted `telefuser serve`
route. The LingBot example supplies a standalone server that starts `PipelinePool` with the optional VLA provider.
This keeps existing HTTP routing and every non-VLA pipeline unchanged. A real simulator still owns control timing,
actuator feedback, and verification that each returned action was actually applied.

For LingBot-VLA v2, the standalone generic WebSocket is the primary continuous-control entrypoint. The direct Python
entrypoint remains the reference/offline baseline, and the native HTTP structured service remains for existing
TeleFuser callers. The RoboTwin MessagePack server is a legacy compatibility entrypoint only; it can be removed after
all upstream clients migrate to `/v1/vla/session`.

The direct inference CLI and structured HTTP task remain separate because they provide reference and batch workflows,
not simulator session transports. The legacy RoboTwin MessagePack endpoint should be removed only after the RTX client
passes generic-protocol action delivery, reset, timeout, reconnect, and latest-wins parity checks.

No new dependencies, environment variables, shared model configuration fields, CLI options, HTTP schemas, or
existing service routes are introduced by this semantic and transport layer.
