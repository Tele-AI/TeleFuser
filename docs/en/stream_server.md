# TeleFuser Stream Server Guide

This guide covers TeleFuser's real-time stream server, which delivers continuous video (and optional audio) over **WebRTC** connections — as opposed to the batch request-response mode of `telefuser serve`.

---

## Quick Start

```bash
# 1. Install
pip install -e .

# WebRTC support is included in the default install.

# 2. Start the stream server
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation

# 3. Open the WebRTC client demo
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088
```

Open `http://localhost:8090` in a browser, type a prompt, and click **Connect** to see real-time video.

---

## Stream Modes

TeleFuser stream server supports two interaction modes, both available over WebRTC:

| Mode | Transport | Direction | Use Case |
|------|-----------|-----------|----------|
| **Server Push** | WebRTC (RTP) | Server → Client | Real-time preview, text-to-video streaming |
| **Bidirectional** | WebRTC (RTP + DataChannel) | Client ↔ Server | Interactive generation, keyboard/camera control, speech-to-video |

`telefuser stream-serve` starts a stream-only app. It exposes `/v1/stream/*`, `/v1/stream/webrtc/*`, and
`/v1/service/*`; it does not expose request-response task routes, file-download routes, or OpenAI-compatible
`/v1/images` and `/v1/videos` routes. Use `telefuser serve` for batch task submission.

### Server Push (WebRTC)

```
Client                          Server
  │                               │
  │  POST /v1/stream/webrtc/offer │
  │  (SDP offer + prompt)         │
  │──────────────────────────────►│
  │                               │  stream_task() → serve()
  │  SDP answer                   │  on background thread
  │◄──────────────────────────────│
  │                               │
  │  ◄──── RTP video frames ────  │
  │  ◄──── RTP audio frames ────  │  (optional)
  │                               │
  │  DELETE /v1/stream/webrtc/{id}│
  │──────────────────────────────►│  cleanup
```

The pipeline's `serve()` method yields chunks containing JPEG-encoded frames. The WebRTC layer decodes them into `av.VideoFrame` objects and streams them to the browser via RTP at the target frame rate.

### Bidirectional (WebRTC)

```
Client                          Server
  │                               │
  │  pc.createDataChannel("telefuser")
  │  pc.addTransceiver("video", recvonly)
  │  (optionally add camera/mic tracks)
  │                               │
  │  POST /v1/stream/webrtc/offer │
  │  (SDP offer + config)         │
  │──────────────────────────────►│
  │                               │  create_session(config)
  │                               │  pull_chunks(session_id)
  │  SDP answer                   │  start ChunkRouter
  │◄──────────────────────────────│
  │                               │
  │  ──── DataChannel JSON ─────► │  push_chunk(session_id, data)
  │  ◄──── RTP video frames ────  │  ChunkRouter → FrameGeneratorTrack
  │  ◄──── RTP audio frames ────  │  ChunkRouter → AudioGeneratorTrack
  │  ◄──── DataChannel JSON ────  │  ChunkRouter → metadata
  │                               │
  │  ──── Media tracks (opt) ───► │  IncomingVideoRelay / AudioRelay
  │                               │  → push_chunk(session_id, frames)
  │                               │
  │  {"type": "stop"}             │
  │  or DELETE /v1/stream/webrtc/{}│
  │──────────────────────────────►│  close_session + cleanup
```

The client **must** create a DataChannel named `"telefuser"` before generating the SDP offer. The server reuses that single channel for receiving control input and sending metadata output. Video and audio are transported over RTP media tracks in both directions.

**ChunkRouter** is the server-side fan-out adapter that consumes the pipeline's `pull_chunks()` generator exactly once and dispatches:

- `frames_b64` → decoded JPEG → `av.VideoFrame` → pushed to `FrameGeneratorTrack` (RTP video)
- `audio_b64` → decoded PCM16 → fed to `AudioGeneratorTrack` (RTP audio)
- Remaining metadata → serialized as `StreamChunkMessage` JSON → sent over DataChannel

---

## CLI Usage

```bash
telefuser stream-serve <pipeline_file> [OPTIONS]
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port`, `-p` | `8088` | Server port |
| `--host` | `0.0.0.0` | Bind address |
| `--gpu-num`, `-g` | `1` | GPU count passed to `get_service(gpu_num=...)` when supported |
| `--security-level` | `strict` | Pipeline validation level (`none`, `basic`, `strict`, `sandbox`). `sandbox` is a best-effort restricted-load check, not runtime isolation. |
| `--skip-validation` | `false` | Skip pipeline file security checks |

### Examples

```bash
# Start with default settings
telefuser stream-serve examples/stream_server/stream_video_replay.py

# Custom port and host
telefuser stream-serve my_pipeline.py -p 9000 --host 127.0.0.1

# Skip validation (development only)
telefuser stream-serve my_pipeline.py -p 8088 --skip-validation
```

---

## Creating a Stream Pipeline

A stream pipeline is a Python file that defines a `get_service()` function returning a service object. It may accept
an optional `gpu_num` parameter supplied by `stream-serve --gpu-num`. The service must implement one of two protocols.

### ServerPushService Protocol

For pipelines that accept a single request and stream output continuously.

```python
from __future__ import annotations
from collections.abc import AsyncGenerator

class MyService:
    def start(self) -> None:
        """Called once at server startup. Load models, open files, etc."""
        ...

    def stop(self) -> None:
        """Called at server shutdown. Release resources."""
        ...

    async def serve(self, request: dict) -> AsyncGenerator[dict, None]:
        """Yield output chunks for a single request.

        The request dict contains all fields from the WebRTC offer
        (prompt, task, fps, duration_s, etc.).
        """
        for i in range(10):
            yield {
                "type": "chunk",
                "index": i,
                "frames_b64": [encode_frame(frame)],
                "fps": 24,
            }

def get_service() -> MyService:
    return MyService()
```

### BidirectionalService Protocol

For pipelines that accept continuous input and produce continuous output.

```python
from __future__ import annotations
from collections.abc import AsyncGenerator

class MyBidirectionalService:
    def start(self) -> None: ...
    def stop(self) -> None: ...

    def create_session(self, config: dict) -> str:
        """Create a new session. Return session_id."""
        ...

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        """Push an input chunk into the session."""
        ...

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        """Yield output chunks for the session."""
        ...

    def close_session(self, session_id: str) -> None:
        """Close the session and free resources."""
        ...

def get_service() -> MyBidirectionalService:
    return MyBidirectionalService()
```

### Chunk Format

Server-push chunks should contain the following fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | `str` | Yes | Always `"chunk"` |
| `index` | `int` | Yes | Chunk sequence number |
| `frames_b64` | `list[str]` | Yes | Base64-encoded JPEG frames |
| `fps` | `int` | Yes | Target frame rate |
| `num_frames` | `int` | No | Number of frames in this chunk |
| `resolution` | `str` | No | Frame resolution (e.g., `"1920x1080"`) |
| `prompt` | `str` | No | Echo of input prompt |
| `timestamp` | `float` | No | Chunk generation timestamp |
| `audio_b64` | `str` | No | Base64-encoded PCM16 audio data |
| `audio_sample_rate` | `int` | No | Audio sample rate (default: 48000) |
| `audio_channels` | `int` | No | Number of audio channels (default: 1) |

> Audio fields are optional. When present, the WebRTC layer creates an audio track alongside the video track.

---

## API Reference

### Endpoints Overview

| Endpoint | Method | Mode | Description |
|----------|--------|------|-------------|
| `/v1/stream/webrtc/offer` | POST | Both | WebRTC SDP offer/answer exchange |
| `/v1/stream/webrtc/{session_id}` | DELETE | Both | Close WebRTC session |
| `/v1/stream/sessions/{session_id}` | DELETE | Both | Close session (pipeline + WebRTC) |
| `/v1/stream/sessions/{session_id}/status` | GET | Both | Get session status |

The stream app also exposes service endpoints such as `/v1/service/health`, `/v1/service/ready`,
`/v1/service/metadata`, and `/v1/service/metrics/json`.

### WebRTC: SDP Offer

**POST** `/v1/stream/webrtc/offer`

This endpoint serves both **server-push** and **bidirectional** modes. The server auto-detects the mode based on the pipeline's service type.

Request:

```json
{
  "sdp": "<SDP offer string>",
  "type": "offer",
  "session_id": "optional-uuid",
  "task": "t2v",
  "prompt": "A sunset over the ocean",
  "fps": 24,
  "duration_s": 10,
  "config": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sdp` | `str` | Yes | SDP offer from browser |
| `type` | `str` | Yes | SDP type (usually `"offer"`) |
| `session_id` | `str` | No | Custom session ID (auto-generated if omitted) |
| `task` | `str` | Yes | Task type (e.g. `t2v`, `i2v`, `bidirectional`) |
| `prompt` | `str` | No | Input prompt |
| `fps` | `int` | No | Target video FPS (default: 24) |
| `config` | `dict` | No | Extra config passed to `BidirectionalService.create_session()` |

> Additional fields are allowed (`"extra": "allow"`) and forwarded to the pipeline.

Response (`200 OK`):

```json
{
  "session_id": "abc123",
  "sdp": "<SDP answer string>",
  "type": "answer"
}
```

Error responses:

| Status | Condition |
|--------|-----------|
| `400` | Invalid SDP or negotiation failure |
| `503` | Stream service not running or max sessions reached |

### WebRTC: Close Session

**DELETE** `/v1/stream/webrtc/{session_id}`

Response (`200 OK`):

```json
{
  "session_id": "abc123",
  "status": "closed"
}
```

**DELETE** `/v1/stream/sessions/{session_id}` closes the pipeline session and WebRTC session together when both
owners are present. If one side has already disappeared but the other closes successfully, the endpoint still returns
`200 OK`; pipeline-side close failures are logged as warnings instead of being silently ignored.

### WebRTC: DataChannel Protocol (Bidirectional)

In bidirectional mode, the client-created `"telefuser"` DataChannel carries JSON messages in both directions.

**Client → Server messages:**

```json
// Control input (e.g., keyboard, prompt)
{"type": "control", "key": "ArrowUp", "action": "press"}
{"type": "control", "prompt": "new prompt text"}

// Stop session
{"type": "stop"}
```

**Server → Client messages:**

```json
// Output chunk metadata (media sent separately via RTP)
{
  "type": "chunk",
  "session_id": "abc123",
  "index": 0,
  "data": {"type": "chunk", "index": 0, "fps": 24, "timestamp": 1714000000.0},
  "timestamp": 1714000000.0
}

// Generation complete
{
  "type": "done",
  "session_id": "abc123",
  "total_chunks": 240,
  "timestamp": 1714000010.0
}
```

> The control message format is pipeline-defined. The examples above show the convention used by `ArrowOverlayService`. Your pipeline's `push_chunk()` receives whatever JSON the client sends.

### Stream-Specific Health Fields

**GET** `/v1/service/health` returns additional fields when stream service is active:

```json
{
  "status": "healthy",
  "stream_ready": true,
  "stream_mode": "server_push",
  "webrtc_active_sessions": 1,
  "webrtc_server_push_sessions": 1,
  "webrtc_bidirectional_sessions": 0,
  "webrtc_max_sessions": 10
}
```

### Stream-Specific Metadata Fields

**GET** `/v1/service/metadata`:

```json
{
  "service_type": "stream",
  "stream_mode": "server_push",
  "pipeline_file": "examples/stream_server/stream_video_replay.py",
  "security_level": "STRICT",
  "runner": "StreamPipelineService",
  "webrtc_active_sessions": 0,
  "webrtc_server_push_sessions": 0,
  "webrtc_bidirectional_sessions": 0,
  "webrtc_max_sessions": 10
}
```

---

## Client Integration

### WebRTC: Server Push (JavaScript)

Minimal browser client for server-push mode:

```javascript
// For LAN: no iceServers needed (or use default STUN)
// For public network: configure STUN + TURN servers
const pc = new RTCPeerConnection({
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    // Add TURN server for production / NAT traversal:
    // { urls: "turn:your-domain.com:3478", username: "user", credential: "pass" },
  ],
});
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });

const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const resp = await fetch("http://localhost:8088/v1/stream/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    task: "t2v",
    prompt: "A cat playing piano",
  }),
});

const answer = await resp.json();
await pc.setRemoteDescription(answer);

pc.ontrack = (event) => {
  document.getElementById("video").srcObject = event.streams[0];
};
```

### WebRTC: Bidirectional (JavaScript)

Full-duplex client with DataChannel for control and optional camera/mic input:

```javascript
const pc = new RTCPeerConnection();

// 1. Client MUST create DataChannel before generating the offer
const dc = pc.createDataChannel("telefuser");
dc.onopen = () => {
  // Send control messages after channel opens
  dc.send(JSON.stringify({ type: "control", key: "ArrowUp", action: "press" }));
};
dc.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.type === "done") console.log("Generation complete:", msg.total_chunks, "chunks");
};

// 2. Optionally send camera/mic to the server
// const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
// stream.getTracks().forEach(t => pc.addTrack(t, stream));

// 3. Receive server output video/audio
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });

pc.ontrack = (evt) => {
  if (evt.track.kind === "video") {
    document.getElementById("video").srcObject = evt.streams[0];
  }
};

// 4. SDP exchange (same endpoint as server-push)
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const resp = await fetch("http://localhost:8088/v1/stream/webrtc/offer", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    sdp: pc.localDescription.sdp,
    type: pc.localDescription.type,
    task: "bidirectional",
    prompt: "A dog running",
    config: { fps: 24 },
  }),
});

const answer = await resp.json();
await pc.setRemoteDescription(new RTCSessionDescription({
  sdp: answer.sdp,
  type: answer.type,
}));

// 5. Stop session
// dc.send(JSON.stringify({ type: "stop" }));
// await fetch(`http://localhost:8088/v1/stream/webrtc/${answer.session_id}`, { method: "DELETE" });
```

### Gradio UI

TeleFuser includes a Gradio-based web UI for WebRTC streaming:

```bash
# Start the stream server
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088

# Launch the Gradio UI (separate terminal)
python webui/stream_app.py --server-url http://localhost:8088 --port 7860
```

The Gradio UI provides a prompt input, duration slider, and embedded WebRTC video player.

---

## Examples

### Video Replay Service

`examples/stream_server/stream_video_replay.py` — A server-push pipeline that loads a local video file and streams it as frame chunks:

- Loads video frames and audio with PyAV at `start()` time
- Pre-encodes all frames as JPEG (cached, not re-encoded per request)
- Streams chunks at configurable pacing with optional audio
- Supports `duration_s`, `realtime`, and `prompt` request parameters

```bash
telefuser stream-serve examples/stream_server/stream_video_replay.py -p 8088 --skip-validation
```

### WebRTC Client Demo

`examples/stream_server/webrtc_client_demo.py` — A standalone HTML page served via Python's HTTP server:

```bash
python examples/stream_server/webrtc_client_demo.py --server-url http://localhost:8088 --port 8090
```

Opens a browser page with video player, prompt input, and connect/stop/unmute buttons.

### Arrow Overlay Service (Bidirectional)

`examples/stream_server/stream_arrow_overlay.py` — A bidirectional pipeline that loads a video and overlays a D-pad HUD based on keyboard input:

- Implements `BidirectionalService` protocol with session state management
- `push_chunk()` receives `{"type": "control", "key": "ArrowUp", "action": "press"}` and updates `pressed_keys`
- `pull_chunks()` yields video frames with D-pad overlay drawn based on `pressed_keys`
- Loops through video frames indefinitely until session closes

```bash
# Start the server
telefuser stream-serve examples/stream_server/stream_arrow_overlay.py -p 8088 --skip-validation

# Start the client (opens browser at localhost:8092)
python examples/stream_server/webrtc_arrow_overlay_demo.py --server-url http://localhost:8088
```

Press arrow keys in the browser to see the D-pad overlay respond in real time.

### Bidirectional Client Demo

`examples/stream_server/webrtc_bidirectional_demo.py` — A general-purpose bidirectional WebRTC client with:

- DataChannel for sending prompts and control messages
- Optional camera and microphone input via media tracks
- Server output video player and DataChannel message log

```bash
python examples/stream_server/webrtc_bidirectional_demo.py --server-url http://localhost:8088
```

### LingBot-World-Fast Streaming

`examples/lingbot/lingbot_world_fast_image_to_video_h100.py` provides a bidirectional streaming service for LingBot-World-Fast. The service generates video over WebRTC RTP and receives prompts and direction control messages over DataChannel. The current demo page does not capture the browser camera or microphone; LingBot currently outputs video only, no audio.

#### Model Files

LingBot-World-Fast requires two sets of weights:

| Setting | Example | Description |
|---------|---------|-------------|
| `TF_MODEL_ZOO_PATH` | `/storage/model_zoo` | Required environment variable that locates both model trees |
| Base model subdirectory | `/storage/model_zoo/Wan2.2-I2V-A14B` | Base Wan2.2 I2V weights containing VAE, T5 text encoder, and tokenizer |
| Fast model subdirectory | `/storage/model_zoo/lingbot/lingbot-world-fast` | LingBot-World-Fast DiT weights |

#### VS Code Remote SSH: TURN over TCP

When the page is opened by a laptop browser through VS Code Remote SSH, the browser and TeleFuser are not on the
same network interface even though both URLs use `localhost`. SDP signaling can use the demo's HTTP proxy, but RTP
media still needs an ICE route. VS Code forwards TCP rather than arbitrary UDP relay ports, so use TURN over TCP.

For development on a trusted host, install coturn and run a loopback-only server in a separate terminal:

```bash
sudo apt-get install -y coturn

turnserver -n -m 1 \
  --listening-ip=127.0.0.1 \
  --relay-ip=127.0.0.1 \
  --listening-port=3478 \
  --min-port=49160 --max-port=49200 \
  --user=telefuser:telefuser-turn \
  --realm=telefuser.local \
  --fingerprint --lt-cred-mech \
  --no-tls --no-dtls --no-cli \
  --allow-loopback-peers \
  --simple-log --log-file=/tmp/telefuser-turn.log
```

`--allow-loopback-peers` is needed only because coturn and TeleFuser are peers on the same host. This command is for
local development and must not be exposed directly to the internet. Verify the TCP allocation and credentials:

```bash
turnutils_uclient -t -y -c \
  -u telefuser -w telefuser-turn -p 3478 127.0.0.1
```

#### Start the Server

Set `TF_MODEL_ZOO_PATH` before starting the service and pass the worker count with `--gpu-num`. The service creates
Ulysses workers internally, so the server command does not need `torchrun` or distributed environment variables.
Use `CUDA_VISIBLE_DEVICES` to select the physical GPUs.

```bash
TF_MODEL_ZOO_PATH=/storage/model_zoo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TELEFUSER_TURN_SERVER='turn:127.0.0.1:3478?transport=tcp' \
TELEFUSER_TURN_USERNAME=telefuser \
TELEFUSER_TURN_CREDENTIAL=telefuser-turn \
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  --gpu-num 4 -p 8088 --host 0.0.0.0 --skip-validation
```

The streaming example enables FSDP and Ulysses sequence parallelism. All four worker logs should show a device mesh
with `dims=[4]` and `Enabling FSDP for lingbot_world_fast_denoise`.

Wait for the following log line before connecting the browser demo:

```text
Starting stream server on 0.0.0.0:8088
```

Verify the service is ready:

```bash
curl --noproxy '*' http://127.0.0.1:8088/v1/service/health
```

#### Start the Browser Demo

When using VS Code Remote SSH, run the demo on the remote TeleFuser host. The demo proxies signaling and session
requests to 8088 by default.

```bash
python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8091 \
  --image-path examples/data/lingbot_world_fast/image.jpg \
  --intrinsics-path examples/data/lingbot_world_fast/intrinsics.npy \
  --frame-num 321 \
  --chunk-size 3 \
  --sample-shift 10.0 \
  --fps 16 \
  --turn-url 'turn:localhost:3478?transport=tcp' \
  --turn-username telefuser \
  --turn-credential telefuser-turn \
  --force-turn-relay \
  --ice-gather-timeout-ms 30000 \
  --no-open
```

Open in browser:

```text
http://localhost:8091
```

In the VS Code **Ports** panel, forward the following remote TCP ports:

| Remote port | Local port | Purpose |
|-------------|------------|---------|
| `8091` | Any available port | Demo HTML and proxied `/v1/stream/webrtc/*` requests |
| `3478` | `3478` | TURN-over-TCP connection referenced by `turn:localhost:3478` |

Port 8088 does not need forwarding while the demo proxy is enabled. Port 3478 must retain local port 3478 unless
the demo's `--turn-url` is changed to match another local port. Open the exact 8091 URL displayed by VS Code using
`http://`; `localhost` is a browser secure context, so HTTPS is not needed for this development workflow.

Do not forward the coturn relay range (`49160-49200`) through VS Code. Browser relay traffic stays inside the
forwarded TURN TCP connection; the relay range is used on the remote host between coturn and the WebRTC peer.
Internet-facing production TURN is different and must expose its configured relay range through the firewall.

If you use a normal SSH client instead of the VS Code Ports panel, start the tunnel in a separate laptop terminal.
Replace `USER` and `SERVER_HOST` with the server login:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -L 8091:127.0.0.1:8091 \
  -L 3478:127.0.0.1:3478 \
  USER@SERVER_HOST
```

Keep the tunnel running and open `http://localhost:8091`. Add `-p SSH_PORT` for a non-default SSH port or
`-i /path/to/private_key` for a specific identity file. Omit `-N` if the same connection should also open an
interactive shell.

If local port 8091 is occupied, use `-L 18091:127.0.0.1:8091` and open `http://localhost:18091`. If local port 3478
is occupied, use `-L 13478:127.0.0.1:3478` and pass
`--turn-url 'turn:localhost:13478?transport=tcp'` to the browser demo. Do not change the server-side
`TELEFUSER_TURN_SERVER`, which still connects to coturn at remote port 3478.

`--image-path` and `--intrinsics-path` are server-side paths, not laptop-local paths. For real-time keyboard control,
the service loads `--intrinsics-path` and keeps its first row fixed for the session. The demo
enables proxying by default; the browser only needs to access the demo port. Requests to `/v1/stream/webrtc/*` are
forwarded by the demo process to `--server-url`.

#### Run the Browser and Service on the Same Machine

If the browser and GPU service run on the same physical machine, you can skip coturn and SSH forwarding. Examples
include Chrome opened directly on the workstation or through remote desktop, VNC, or noVNC. An SSH login alone does
not make the laptop browser local: if the browser still runs on the laptop, use the TURN and port-forwarding setup
above.

Do not set `TELEFUSER_TURN_*`, and do not pass `--turn-url`, TURN credentials, or `--force-turn-relay` to the demo.

Server:

```bash
env -u TELEFUSER_TURN_SERVER \
  -u TELEFUSER_TURN_USERNAME \
  -u TELEFUSER_TURN_CREDENTIAL \
  TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  --gpu-num 4 -p 8088 --host 127.0.0.1 --skip-validation
```

Demo:

```bash
env -u TELEFUSER_TURN_SERVER \
  -u TELEFUSER_TURN_USERNAME \
  -u TELEFUSER_TURN_CREDENTIAL \
  python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8091 \
  --image-path examples/data/lingbot_world_fast/image.jpg \
  --intrinsics-path examples/data/lingbot_world_fast/intrinsics.npy \
  --frame-num 321 \
  --chunk-size 3 \
  --sample-shift 10.0 \
  --fps 16 \
  --no-open
```

Open in the browser on the same machine:

```text
http://localhost:8091
```

#### Direction Control

The demo supports the page D-pad, arrow keys, and the source-compatible `WASD`/`IJKL` keyboard controls:

| Input | cam mode meaning |
|-------|------------------|
| `↑` / `W` | Move forward |
| `↓` / `S` | Move backward |
| `←` / `A` | Strafe left |
| `→` / `D` | Strafe right |
| `J` / `L` | Yaw left / right |
| `I` / `K` | Pitch up / down |

Controls only affect chunks that have not yet started generating; chunks already denoising or decoding are not
immediately changed. Holding a key continuously generates controlled chunks. After all keys are released, the
service stops requesting new chunks and WebRTC repeats the most recently emitted frame.

The camera pose starts at identity. Each latent interval integrates four video-frame control steps, matching the
LingBot source trajectory generator. Camera intrinsics are fixed when the session is created. Absolute pose and
pitch state are accumulated across chunks, including the relative-pose delta across each chunk boundary.

The following states in the DataChannel log indicate that direction control has been consumed by the server and applied to a generation chunk:

```text
"stage":"control_state"
"stage":"applying_direction_control"
```

The demo enables `Control HUD` by default. The HUD overlay appears in the top-left corner of output chunks that received direction control, confirming the control pipeline is active. Once confirmed, you can uncheck the HUD on the page to observe pure model output.

Common control strength parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--control-move-step` | `0.05` | Forward/backward displacement per video frame |
| `--control-yaw-step-degrees` | `2.0` | Yaw angle per video frame |
| `--control-lateral-step` | `0.05` | Lateral displacement per video frame |
| `--control-pitch-step-degrees` | `2.0` | Pitch angle per video frame |
| `--control-pitch-limit-degrees` | `85.0` | Absolute pitch limit |
| `--show-control-hud / --no-show-control-hud` | `true` | Whether to overlay direction HUD on controlled chunks |

#### `cam` vs `act` Control Modes

| Mode | Input | Description |
|------|-------|-------------|
| `cam` | `poses + intrinsics` | Camera trajectory control. The server converts arrow keys into camera poses, then builds a 6-channel camera control tensor. |
| `act` | `poses + intrinsics + action` | Action control. Requires 7-channel action-control weights. |

The current demo defaults to `cam`. If the model weights are camera-control weights, keep:

```bash
--control-mode cam
```

Only switch to `act` when using action-control weights:

```bash
--control-mode act
```

The service example sets `PPL_CONFIG["control_mode"]="cam"` inside
`examples/lingbot/lingbot_world_fast_image_to_video_h100.py`. A session's control mode must
match the mode used to initialize the pipeline. To use action-control weights,
set that value to `"act"` before starting the service and pass `--control-mode
act` to the demo.

#### Frame and Control Contract

LingBot accepts complete latent chunks only. `frame_num` must be `4n + 1`, and
the resulting latent-frame count must divide evenly by `chunk_size`. With the
default chunk size of 3, valid output lengths are 9, 21, 33, ..., 321 frames.
The browser accepts a duration and rounds down to the largest complete chunk:
10 seconds becomes 153 frames and 20 seconds becomes 321 frames at 16 FPS.
Direct API requests with invalid values are rejected. Offline control files
must provide enough poses and action samples for the requested video window.
Intrinsics may be either one static `(4,)` value or one `(frames, 4)` value per
video frame. Explicit control tensors must match the pipeline's selected mode,
device, dtype, and full latent-chunk shape.

#### VRAM and Resolution

LingBot's global KV cache grows with `frame_num` and output resolution. FSDP shards model parameters, while Ulysses
shards KV heads across workers. More GPUs therefore allow longer sessions, but the scaling is not perfectly linear.
The following 832x480 configurations were tested with FSDP, `chunk_size=3`, and `sample_shift=10.0`:

| GPUs | Duration | Frames | Result |
|------|----------|--------|--------|
| 2 H100 80 GB | 10 seconds | 153 | Passed |
| 2 H100 80 GB | 20 seconds | 321 | KV-cache CUDA OOM |
| 4 H100 80 GB | 20 seconds | 321 | Passed, 27/27 chunks |

The four-GPU run peaked at approximately 58.6 GiB on GPU 0 and 41.6 GiB on GPUs 1-3. Start with five seconds to
verify the pipeline before selecting the maximum duration:

```bash
# In the browser demo or session configuration:
--frame-num 81
```

Common tuning parameters:

| Parameter | Effect |
|-----------|--------|
| `PPL_CONFIG["resolution"]` | Select either the 480p or 720p output area |
| `--frame-num` | Reduce total generated frames and latent chunk count |
| `--chunk-size` | Affect the latent chunk size per generation step |

An OOM marks the parallel worker as failed. Closing the browser session is not enough; restart the stream service
before trying another duration.

The service currently allows only one LingBot active session. Before reconnecting, click **Stop** on the demo, or call:

```bash
curl -X DELETE http://127.0.0.1:8088/v1/stream/webrtc/<session_id>
```

If still experiencing OOM, check whether other processes are using the GPU:

```bash
nvidia-smi
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TELEFUSER_WEBRTC_MAX_SESSIONS` | `10` | Maximum concurrent WebRTC sessions (1-100) |
| `TELEFUSER_STUN_SERVERS` | `["stun:stun.l.google.com:19302"]` | STUN server URLs (JSON array) |
| `TELEFUSER_TURN_SERVER` | `None` | TURN server URL (e.g. `turn:your-domain.com:3478`) |
| `TELEFUSER_TURN_USERNAME` | `None` | TURN server username |
| `TELEFUSER_TURN_CREDENTIAL` | `None` | TURN server credential |

### CORS

CORS middleware is automatically added when stream service is initialized (required for browser-based WebRTC clients). All origins are allowed by default.

### Security Levels

Pipeline files are validated before loading. Use `--skip-validation` only for development:

| Level | Description |
|-------|-------------|
| `none` | Disable validation |
| `basic` | Static AST analysis |
| `strict` | Static AST analysis plus import restrictions |
| `sandbox` | Strict checks plus a best-effort restricted-load validation step. This is not runtime isolation. |

---

## Public Network Deployment

When deploying the stream server on a public network (server and client on different networks), WebRTC requires additional configuration for NAT traversal and browser security.

### 1. HTTPS (Required)

Browsers block WebRTC on non-localhost HTTP pages. You must serve over HTTPS:

```bash
# Option A: Let's Encrypt (recommended for production)
certbot certonly --standalone -d your-domain.com

# Option B: Self-signed certificate (development/testing)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start with TLS
uvicorn telefuser.service.api.app:app \
  --host 0.0.0.0 --port 443 \
  --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### 2. STUN/TURN Servers

**STUN** discovers the client's public IP (lightweight, free). **TURN** relays media when direct connections fail (requires bandwidth, self-host recommended).

```bash
# Install coturn (popular open-source TURN server)
apt install coturn
```

Minimal `/etc/turnserver.conf`:

```ini
listening-port=3478
tls-listening-port=5349
realm=your-domain.com
server-name=your-domain.com
fingerprint
lt-cred-mech
user=telefuser:your-secret-password
external-ip=203.0.113.10
min-port=49152
max-port=65535
```

```bash
systemctl enable coturn && systemctl start coturn
```

Configure TeleFuser to use the TURN server:

```bash
export TELEFUSER_TURN_SERVER="turn:your-domain.com:3478"
export TELEFUSER_TURN_USERNAME="telefuser"
export TELEFUSER_TURN_CREDENTIAL="your-secret-password"
telefuser stream-serve pipeline.py -p 8000
```

### 3. Firewall Ports

| Port | Protocol | Purpose |
|------|----------|---------|
| 443 | TCP | HTTPS (API + SDP signaling) |
| 3478 | TCP+UDP | STUN/TURN |
| 5349 | TCP | TURN over TLS |
| 49152-65535 | UDP | TURN relay media ports |

### Deployment Architecture

```
┌─────────┐     HTTPS/443      ┌──────────────┐
│  Client  │◄──────────────────►│  Nginx/CDN   │
│ (Browser)│                    │  (TLS term.) │
│          │     UDP            ├──────────────┤
│          │◄──────────────────►│  TeleFuser   │
│          │   (WebRTC media)   │  :8000       │
│          │                    ├──────────────┤
│          │◄──────────────────►│  coturn      │
│          │   UDP 3478 +       │  (TURN relay)│
│          │   49152-65535      │              │
└─────────┘                    └──────────────┘
```

> **LAN deployment** requires no STUN/TURN configuration. The default settings work out of the box when server and client are on the same network.

---

## Troubleshooting

### WebRTC Connection Fails (ICE Error)

For **LAN**: ensure the server binds to `0.0.0.0` (not `127.0.0.1`) and no firewall blocks UDP.

For **public network**:

1. Verify STUN/TURN servers are configured (see [Public Network Deployment](#public-network-deployment))
2. Ensure TURN server ports (3478, 49152-65535) are open
3. Test TURN connectivity: `turnutils_uclient -u user -w pass your-domain.com`
4. Check browser DevTools → `chrome://webrtc-internals` for ICE candidate details

### No Audio in Browser

Click the **Unmute** button. Browsers require a user gesture before playing audio. The video element starts muted by default.

### Port Already in Use

```bash
# Find and kill the process using the port
lsof -ti:8088 | xargs kill -9
```

### "Stream service is not running" (503)

The pipeline's `start()` method failed. Check server logs for errors (e.g., missing video file, import errors).

### High Memory Usage

Each WebRTC session holds video frames in a queue. Limit `webrtc_max_sessions` and ensure pipelines don't buffer excessive frames. The server automatically cleans up sessions when clients disconnect.
