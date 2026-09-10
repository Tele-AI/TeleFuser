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

The public modules are:

- `telefuser.vla.contracts`: observations, requests, capabilities, action spaces, and model/robot chunks.
- `telefuser.vla.policy`: the model-facing `VLAPolicy` protocol.
- `telefuser.vla.embodiment`: observation encoding and model-to-robot action mapping.
- `telefuser.vla.registry`: explicit registration of already-loaded policies and embodiments.
- `telefuser.vla.session`: transport-neutral OPEN, PREDICT, RESET, and CLOSE lifecycle.
- `telefuser.vla.serialization`: versioned JSON/Base64 wire formats for action spaces, observations, and chunks.
- `telefuser.vla.runtime`: scheduling, deterministic chunk state, action trimming, age checks, and safety policies.
- `telefuser.integrations.sim`: simulator protocol and the dependency-free RoboTwin callback adapter.
- `telefuser.service.vla_session`: the additive generic VLA WebSocket application factory.

## LingBot-VLA v2 Compatibility

`LingBotVlaV2VLAPolicy` wraps `LingBotVlaV2Pipeline`; the pipeline's existing tensor input and return types are not
changed. It labels the normalized canonical `[T,55]` result as a `ModelActionChunk`.

`RobotWinProfile` is the first `EmbodimentAdapter`. It preserves the existing normalization code and maps a canonical
chunk to an absolute-position `[H,14]` `RobotActionChunk` in the declared dual-arm joint order. Its model and robot
action spaces are exported as `LINGBOT_VLA_V2_ACTION_SPACE` and `ROBOTWIN_ACTION_SPACE`.

The existing LingBot RoboTwin WebSocket endpoint now calls this policy, embodiment, session, and runtime path
internally. Its URL, MessagePack request fields, metadata frame, response fields, reset behavior, and latest-wins
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

`create_vla_session_app` exposes `/v1/vla/session` without changing an existing TeleFuser service route. The server
first sends `HELLO` with protocol version `1.0`, wire encoding, supported operations, registered component IDs, and
payload limits. A client then uses `OPEN`, ordered `PREDICT`, `RESET`, and `CLOSE`. `OPEN` can declare the expected
robot action space; semantic incompatibility is rejected before inference.

The protocol returns machine-readable error codes for malformed messages, unsupported versions, unknown components
or sessions, action-space mismatches, out-of-order or expired observations, request timeout, unavailable sessions, and
replica failure. Tensor payloads are dense, typed, shape-checked Base64 data inside bounded JSON messages. This is the
stable interoperability format; the LingBot-RoboTwin compatibility endpoint continues to use its existing MessagePack
format.

`request_ttl_ms` is measured with server monotonic time and bounds inference delivery. Observation age is checked
separately using `observation_timestamp_ns` and `observation_clock_now_ns`, which must come from the same clock domain.
After a timed-out stateful inference finishes, the session is reset before it accepts more prediction work.

## Replica Session Leases

`PipelinePool.open_session` removes one healthy replica from the ordinary request pool and binds it to a
`session_id`. Calls through `acquire_session` always reach that replica and are serialized per session.
`close_session` returns a live replica to ordinary capacity. Replica exit removes the binding and raises
`ReplicaDeadError`; pool shutdown marks sessions as closing, waits for active session calls, clears every binding, and
then stops replicas.

This API is additive. Existing HTTP work continues to use `PipelinePool.acquire()`, whose allocation behavior and
callers are unchanged. The generic in-process WebSocket factory accepts a `VLASessionManager`; a deployment backed by
subprocess replicas must use these lease methods when adding its replica RPC adapter.

## Chunk State Machine

`ActionChunkStateMachine` makes action lifecycle explicit: pending inference, ready buffer, execution, and terminal
executed, superseded, expired, or rejected states. It supports latest-sequence admission, separate observation-age and
network-TTL checks, configurable discard/retain handling after `execute_horizon`, reset, and deterministic hold/stop
states after disconnect. Stateful policies must provide a recovery callback; it is invoked when an inference that may
have mutated history is discarded.

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

## Current Boundary

The generic WebSocket application is intentionally an explicit factory rather than an automatically mounted
`telefuser serve` route. Subprocess replica transport still needs VLA-specific OPEN/PREDICT/RESET/CLOSE RPC commands
before the WebSocket factory can directly own `PipelinePool` leases. The lease primitive is implemented and tested,
but this RPC bridge is outside the current P0 scope.

The chunk state machine is also transport-neutral. The existing LingBot compatibility server retains its proven
latest-wins scheduler until a later compatibility-preserving integration uses the generic state machine end to end.

No new dependencies, environment variables, shared model configuration fields, CLI options, HTTP schemas, or
existing service routes are introduced by this semantic and transport layer.
