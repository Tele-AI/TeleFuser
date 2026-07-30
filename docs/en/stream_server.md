# Stream Server

TeleFuser uses LiveKit as its only streaming transport. The `telefuser stream-serve` command accepts pipeline files
whose `get_service()` returns either `ServerPushService` or `BidirectionalService`; there is no separate backend
selector or direct SDP endpoint.

LiveKit terminates browser WebRTC connections and provides rooms, reconnect handling, media delivery, and reliable
data messages. TeleFuser owns model workers, admission, session state, pipeline execution, and token issuance. A
LiveKit Cloud project or a self-hosted LiveKit Server is therefore required.

This guide uses **service instance** for the single object returned by the pipeline file's `get_service()` function.
It is loaded once by a model worker and may own multiple user sessions; a user session is not another service
instance.

## Why LiveKit

TeleFuser targets high-performance multimodal generation inference. Its streaming service must support continuous
media output, bidirectional control, and long-running stateful model sessions. LiveKit's realtime transport
capabilities align with these goals:

- A room provides a stable transport boundary for a model session, keeping browser connections independent from
  session-owned model state.
- Media tracks carry progressive video and audio, while data topics carry controls, status, and bounded telemetry.
- Scoped tokens distinguish controller, viewer, and worker roles, matching TeleFuser's session ownership and permission
  model.
- `ServerPushService` and `BidirectionalService` share one streaming entrypoint and client connection model.
- LiveKit owns connections, reconnects, and media delivery so TeleFuser can focus on model workers, admission, pipeline
  execution, and resource release.

## Install and start locally

The LiveKit Python SDKs are included in the base TeleFuser installation:

```bash
pip install -e .
```

Install the development LiveKit Server and your platform's `coturn` package separately:

```bash
# Debian/Ubuntu; use the equivalent coturn package on other platforms.
sudo apt-get update
sudo apt-get install -y coturn

curl -sSL https://get.livekit.io | bash
livekit-server --dev
```

The development server listens on `ws://127.0.0.1:7880` and uses `devkey` / `secret`. Do not use development
credentials in production.

Start TeleFuser:

```bash
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey \
  --livekit-api-secret secret \
  --port 8088 \
  --skip-validation
```

The same command serves server-push pipelines:

```bash
telefuser stream-serve examples/stream_server/stream_video_replay.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey \
  --livekit-api-secret secret \
  --port 8088 \
  --skip-validation
```

Configuration may instead use `TELEFUSER_LIVEKIT_*` environment variables. Explicit CLI values take precedence.

## Browser demo

The checked-in page sets `iceTransportPolicy: relay`, so its matching TCP TURN service is required even though a
production LiveKit deployment may provide TURN differently. Start this development-only coturn process first:

```bash
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 \
  --relay-ip=127.0.0.1 \
  --listening-port=3478 \
  --min-port=49160 --max-port=49200 \
  --user=livekit-demo:livekit-demo-password \
  --realm=livekit.local \
  --fingerprint --lt-cred-mech \
  --no-tls --no-dtls --no-cli \
  --allow-loopback-peers
```

Start the LingBot control page in a fourth terminal:

```bash
python examples/stream_server/livekit_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8092 \
  --no-open
```

Open `http://127.0.0.1:8092`, select an initial image, and click **Start**. The demo proxies `/v1/stream/*` requests
to TeleFuser, obtains a controller token, joins the assigned LiveKit room, renders the published video track, and
sends the on-page or keyboard camera controls on `tf.control`.

For VS Code Remote SSH, forward the demo HTTP port, LiveKit signaling port, and the TURN listener configured for
LiveKit. The checked-in demo uses TCP relay at `turn:127.0.0.1:3478?transport=tcp` with development credentials
`livekit-demo` / `livekit-demo-password`. Change this browser configuration and the LiveKit deployment together for
production.

Forward remote TCP ports `8092`, `7880`, and `3478` to the same local ports, then open
`http://127.0.0.1:8092`. The loopback listener, static password, disabled TLS, and `--allow-loopback-peers` are only
for a trusted development host reached through the tunnel. Do not copy them to a public deployment.

The complete browser stack is now coturn (`3478`), LiveKit (`7880`), TeleFuser (`8088`), and the page (`8092`).
`curl http://127.0.0.1:8088/v1/service/health` should report a ready idle worker before a session starts. During a
successful run, the page shows the video track and status messages including `control_state`, generation stages,
and `chunk_sent`. Stop or close the browser session before stopping the four processes in reverse order; this avoids
the browser reconnecting while LiveKit and the model worker drain.

## Architecture and lifecycle

```text
Browser A ── HTTP API / LiveKit room A ── session runner A ─┐
                                                            ├─ one model worker
Browser B ── HTTP API / LiveKit room B ── session runner B ─┘        │
                                                                     └─ one shared service instance
                                                                            ├─ pipeline session A state
                                                                            └─ pipeline session B state
```

| Term | Lifetime and ownership |
|---|---|
| Service process | One `telefuser stream-serve` process containing the HTTP API, admission scheduler, and current in-process worker. |
| Model worker | Loads one service instance and owns retained-session capacity. The current runtime creates exactly one. |
| Service instance | The result of one `get_service()` call. Model weights and the pipeline actor graph are loaded once and shared by its sessions. |
| Session runner | One LiveKit room connection and transport task for an admitted HTTP session. Runners share the worker's service instance. |
| Pipeline session | Per-user state created through `BidirectionalService.create_session()`, including model/cache/control state owned by that session. |

`--worker-gpu-map 0,1,2,3` assigns four devices to the one model worker and its one service instance. It does not
create four workers or four model replicas.

1. The controller creates a session through `POST /v1/stream/sessions`.
2. The scheduler admits, queues, or rejects it and assigns one retained-session slot on a worker.
3. TeleFuser creates the LiveKit room and returns a scoped controller token.
4. The worker joins the room and starts the pipeline.
5. Video and PCM16 audio are published as LiveKit tracks. Status and metrics use reliable data topics.
6. For `BidirectionalService`, only the controller can send normalized control messages to the pipeline.
7. Deletion, timeout, controller departure, or pipeline completion drains actor-owned state and closes the room.

Each streaming stage worker remains owned by one pipeline actor. Here, a stage worker is an internal pipeline
execution object, not the LiveKit model worker in the table above. Reconnects do not move actor-owned cache state. A
server-push pipeline starts from its request config and produces chunks without incoming controls; a bidirectional
pipeline additionally exposes create, pull, control, and close operations.

## LingBot multi-session time slicing

`--max-sessions-per-worker` controls how many LiveKit sessions one model worker may retain; it does not create more
service instances. Capacity above one requires a `BidirectionalService`, because server-push sessions do not have a
control-idle signal. The framework provides the room runners and retained-session admission, but the underlying
service must provide session isolation and any required compute serialization.

The checked-in LingBot-World-Fast and LingBot-World v2 services implement that serialization with one execution lease
inside their shared service instance. Enable two retained sessions with, for example:

```bash
telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
  --max-sessions-per-worker 2 --control-idle-timeout 10 --skip-validation
```

Each admitted LingBot room creates an independent pipeline session, control queue, noise state, VAE state, and model
cache within the shared service instance. A valid `control_state`, `control`, `prompt`, or `reset` message renews that
session's lease activity. When another session is waiting and the current holder has received no valid control for
`control_idle_timeout` seconds, the holder stops admitting new chunks and yields after its current chunk completes.
The waiting session is then granted the lease; the parked session can request the lease again when its controller
sends a new control message.

This is chunk-boundary time slicing, not continuous batching or replication: the LingBot service instance has at most
one session chunk in flight. `control_idle_timeout` does not close the holder, release its retained-session slot, or
free its cache; it only permits a waiting session to take the lease at a chunk boundary. A parked session therefore
still consumes its per-session GPU memory. Set `max_sessions_per_worker` from measured memory headroom. Controllers
that represent a held input must resend `control_state` periodically; the checked-in browser demo sends a one-second
heartbeat while a control remains pressed.

There are two separate FIFO waits:

| Wait | Applies to | Capacity and observable result |
|---|---|---|
| HTTP admission queue | Requests made after all retained-session slots are occupied | Bounded by `queue_size`; returns HTTP 202 with `queue_position`, or HTTP 429 when disabled/full. |
| LingBot execution-lease queue | Admitted sessions requesting model execution | Bounded by the number of retained LingBot sessions; emits `lease_queued`, `lease_granted`, and `lease_parked` status events. |

Releasing the execution lease does not move a session back to the HTTP admission queue. Only terminal session cleanup
releases its retained-session slot and admits the next HTTP-queued request.

## HTTP API

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/stream/sessions` | POST | Create and admit a controller session |
| `/v1/stream/sessions/{session_id}` | GET | Read session status |
| `/v1/stream/sessions/{session_id}` | DELETE | Drain and close a session |
| `/v1/stream/sessions/{session_id}/tokens` | POST | Create a viewer token |
| `/v1/stream/health` | GET | LiveKit scheduler and worker health |
| `/v1/service/health` | GET | Generic service health |
| `/v1/service/ready` | GET | Readiness probe |
| `/v1/service/metadata` | GET | Pipeline and transport metadata |
| `/v1/service/metrics` | GET | Prometheus metrics |

Create a controller session:

```bash
curl -X POST http://127.0.0.1:8088/v1/stream/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "identity": "controller-1",
    "prompt": "A first-person view moving through a forest",
    "image_path": "examples/lingbot/assets/test_1.jpeg",
    "config": {"fps": 16}
  }'
```

For a one-minute LingBot-World v2 replay, start
`examples/lingbot/lingbot_world_v2_image_to_video_h100.py` and use:

```json
{
  "fps": 16,
  "chunk_size": 4,
  "frame_num": 957,
  "max_duration_seconds": 60.0
}
```

The complete-chunk policy maps this request to 60 chunks and 59.75 seconds of output media. The v2 example uses
`local_attn_size=18` and `sink_size=6`; its KV capacity therefore remains fixed while the session-owned noise and
VAE state advance incrementally. The reproducible LiveKit workload and dated four-H100 result are documented in
[TeleFuser and AIPerf](benchmark_aiperf.md).

A successful response includes `session_id`, `room`, `livekit_url`, `token`, `worker_id`, and `status`. Sessions fill
retained slots on the worker before entering the HTTP admission queue. A queued session returns HTTP 202 with
`queue_position`; a full zero-length queue returns HTTP 429.

In `/v1/stream/health`, `workers_busy` counts model workers with at least one retained session; it does not count
sessions or currently executing chunks. `workers_idle` counts workers with no retained sessions, and `queued_sessions`
is only the HTTP admission queue depth. `livekit_connected=false` is normal before any session runner has connected;
it does not mean that the model or service instance failed to load.

Create a viewer token without granting control permission:

```bash
curl -X POST http://127.0.0.1:8088/v1/stream/sessions/<session_id>/tokens \
  -H 'Content-Type: application/json' \
  -d '{"identity":"viewer-1"}'
```

Close the session explicitly:

```bash
curl -X DELETE http://127.0.0.1:8088/v1/stream/sessions/<session_id>
```

## LiveKit data protocol

| Topic | Direction | Content |
|---|---|---|
| `tf.control` | controller to worker | Reliable JSON control messages |
| `tf.status` | worker to room | Lifecycle and chunk status |
| `tf.metrics` | worker to room | Bounded runtime metrics |
| `tf.asset` | reserved | Future bounded asset messages |

Accepted control types are `control_state`, `control`, `prompt`, `reset`, and `stop`. For example:

```json
{"type":"control_state","controls":["w","j"]}
```

An optional versioned envelope is also accepted:

```json
{"version":1,"session_id":"<id>","type":"control_state","payload":{"controls":["w"]}}
```

Messages are bounded by `TELEFUSER_LIVEKIT_MAX_DATA_MESSAGE_BYTES` (12 KiB by default). Unknown controls, duplicate
entries, invalid JSON, wrong topics, session mismatches, and control messages from viewers are rejected.

## CLI and environment configuration

```text
telefuser stream-serve PIPE_PATH [OPTIONS]
```

Important options are `--host`, `--port`, `--livekit-url`, `--livekit-api-key`, `--livekit-api-secret`,
`--num-workers`, `--worker-gpu-map`, `--max-sessions-per-worker`, `--control-idle-timeout`, `--queue-size`,
`--session-timeout`, `--token-ttl`,
`--controller-timeout`, `--room-empty-timeout`, and `--worker-mode`.

| Environment variable | Default | Meaning |
|---|---:|---|
| `TELEFUSER_LIVEKIT_URL` | required | LiveKit WebSocket URL |
| `TELEFUSER_LIVEKIT_API_KEY` | required | API key used to mint room tokens |
| `TELEFUSER_LIVEKIT_API_SECRET` | required | API secret used to mint room tokens |
| `TELEFUSER_LIVEKIT_HOST` | `0.0.0.0` | HTTP API bind host |
| `TELEFUSER_LIVEKIT_PORT` | `8088` | HTTP API port |
| `TELEFUSER_LIVEKIT_NUM_WORKERS` | `1` | Model workers; the current in-process runtime requires exactly `1` |
| `TELEFUSER_LIVEKIT_WORKER_GPU_MAP` | unset | GPU group assigned to the one worker, such as `0,1,2,3` |
| `TELEFUSER_LIVEKIT_MAX_SESSIONS_PER_WORKER` | `1` | Retained bidirectional sessions per model worker |
| `TELEFUSER_LIVEKIT_CONTROL_IDLE_TIMEOUT` | `10` | Inactive seconds before a waiting session may take the execution lease |
| `TELEFUSER_LIVEKIT_QUEUE_SIZE` | `0` | Queued sessions; zero rejects when all retained-session slots are full |
| `TELEFUSER_LIVEKIT_SESSION_TIMEOUT` | `1800` | Maximum session lifetime in seconds |
| `TELEFUSER_LIVEKIT_TOKEN_TTL` | `3600` | Join-token lifetime in seconds |
| `TELEFUSER_LIVEKIT_CONTROLLER_TIMEOUT` | `60` | Grace period after controller departure |
| `TELEFUSER_LIVEKIT_ROOM_EMPTY_TIMEOUT` | `30` | Grace period after the room becomes empty |

The current runtime requires `worker_mode=in-process` and starts exactly one model worker. Additional model replicas
require separate `stream-serve` processes and external request routing until process-worker isolation is implemented;
their admission queues and session state are independent. `--skip-validation` is intended for trusted local files,
not production.

## Production deployment

- Use LiveKit Cloud or the official self-hosted deployment guidance; do not expose `livekit-server --dev`.
- Use unique API credentials and keep the API secret only on the TeleFuser server.
- Configure TLS, advertised node addresses, UDP/TCP media ports, and TURN in LiveKit itself.
- Restrict TeleFuser's HTTP API with the deployment's authentication and network policy.
- Monitor `/v1/service/ready`, worker failures, queue depth, and session expiration.
- Interpret chunk period as adjacent output cadence: p95 cadence must remain below one chunk's media duration with
  margin for transport and encoding. Pipeline residence and client delivery FPS are separate measurements.

## Troubleshooting

- **HTTP health is ready but no media arrives:** verify that both browser and worker can reach the LiveKit URL and
  inspect LiveKit participant/track logs.
- **Browser reconnects repeatedly:** verify signaling, TURN, firewall, and advertised LiveKit node addresses.
- **Controls are ignored:** ensure the sender used the controller token, topic `tf.control`, and a supported control.
- **HTTP 429:** all retained-session slots are full and `queue_size` is zero, or the queue is full.
- **Session remains active:** call the session DELETE endpoint and check controller/room timeout settings.
- **Local LiveKit connection returns proxy HTTP 503:** some native SDK paths honor `HTTP_PROXY` but not
  `NO_PROXY`. Start TeleFuser with `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and their lowercase variants unset when
  connecting to `ws://127.0.0.1:7880`.
- **Stale GPU workers after a forced exit:** terminate remaining `spawn_main` processes before restarting.
