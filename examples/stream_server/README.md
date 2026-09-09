# LiveKit Stream Examples

TeleFuser has one streaming backend: LiveKit. The examples cover both service contracts:

- `stream_video_replay.py`: server-push video.
- `stream_arrow_overlay.py`: bidirectional control rendered into output frames.
- `livekit_bidirectional_demo.py`: browser UI for LingBot camera control.
- `_control_demo_ui.py`: private shared HTML/CSS/control asset used by the LiveKit demo.

## Requirements

Install the [TeleFuser development environment](../../CONTRIBUTING.md#development-setup), then prepare the weights
and GPU environment in the [LingBot-World guide](../lingbot/README.md). The command below uses four H100 GPUs.
Model-free replay and overlay scripts are separate service examples; they do not validate model inference speed.

## Model-Free Preflight

Start with `stream_arrow_overlay.py`. It generates its own moving frames when no video file is present and accepts
the same browser control protocol as LingBot-World, so it validates signaling, TURN relay, controls, and return video
without loading a checkpoint. Follow the four-terminal procedure in the
[WebRTC Core Experience](../../docs/en/streaming_quickstart.md), using the overlay service as Terminal 3.

## LingBot-World v2

Run all commands from the repository root. The checked-in browser demo forces TCP TURN relay, so the complete
interactive stack has four services. Install the
LiveKit Server once with `curl -sSL https://get.livekit.io | bash`, install your platform's `coturn` package, then
run these commands in four terminals:

```bash
# Terminal 1: TURN relay matching the browser configuration
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 \
  --listening-port=3478 --min-port=49160 --max-port=49200 \
  --user=livekit-demo:livekit-demo-password --realm=livekit.local \
  --fingerprint --lt-cred-mech --no-tls --no-dtls --no-cli \
  --allow-loopback-peers

# Terminal 2: LiveKit signaling and SFU
livekit-server --dev

# Terminal 3: TeleFuser model and session API
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey \
  --livekit-api-secret secret \
  --worker-gpu-map 0,1,2,3 \
  --max-sessions-per-worker 1 \
  --control-idle-timeout 10 \
  --port 8088 \
  --skip-validation

# Terminal 4: browser page and API proxy
python examples/stream_server/livekit_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8092 \
  --no-open
```

Open `http://127.0.0.1:8092`, choose an image, and click **Start**. For VS Code Remote SSH, forward TCP `8092`,
`7880`, and `3478` to the same local ports; the API proxy means `8088` does not need forwarding. Stop the browser
session first, then stop terminals 4 through 1 in reverse order.

## Expected Output

The browser shows an initial preview followed by camera-controlled video while movement keys are held. Check that
frames advance in order and that releasing the keys returns generation to idle. A connected room alone does not
establish that the model is producing frames.

## Configuration and Troubleshooting

This command starts one process, one in-process model worker, and one shared LingBot service instance. It exposes four
physical GPUs as process-local devices 0-3, declares one four-device logical worker group, and retains one session.
The LingBot execution lease serializes model chunks; it is not a generic replication option.

The LiveKit Python SDK is part of TeleFuser's base dependencies; the LiveKit Server is installed and operated
separately. See the [Stream Server guide](../../docs/en/stream_server.md) for room roles and viewer fan-out, the exact
GPU-map boundary, session API, queues, lifecycle, observability, remote development, and production deployment.

Documentation builds do not start LiveKit or load model weights. For measured first-frame latency, control response,
target compute FPS, and client delivery FPS, use the [AIPerf benchmark guide](../../docs/en/benchmark_aiperf.md).
