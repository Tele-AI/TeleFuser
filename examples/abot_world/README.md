# ABot-World-0-5B-LF

Generate image-conditioned, camera-controlled video using a local single-GPU HTTP controller or the LiveKit
streaming service. The public workload is 480x832 at a 12 FPS playback target.

## Requirements

Install the standard [TeleFuser development environment](../../CONTRIBUTING.md#development-setup). This example
requires one CUDA GPU, the release checkpoint below, and an initial image readable by the server. The maintained
smoke test checks the 480x832 generation contract; this README does not record a minimum VRAM or a dated throughput
measurement. The LiveKit path additionally needs LiveKit Server and coturn as described in the
[stream server guide](../../docs/en/stream_server.md).

## Model Directory

Prepare the unmodified release checkpoint layout:

```text
ABot-World-0-5B-LF/
  diffusion_pytorch_model.safetensors
  Wan2.2_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
```

The default sample is an external upstream checkout asset at
`../ABot-World/web_client/datasets/images/84b90ad568b693d2.png`. Supply another server-side image path in the HTTP
controller when that checkout is unavailable. The LiveKit browser also accepts an uploaded image.

## Quick Start

Run all commands from the repository root. Start the local HTTP controller:

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860`, provide an image path, and connect. The browser controls WASD/arrow movement and IJKL
camera rotation. Confirm that the preview appears, then hold a movement key and check that new frames arrive;
release the keys and confirm that generation becomes idle. Disconnect before stopping the server with Ctrl+C.

Connecting
creates the image-conditioned causal session but does not advance the DiT
until a non-empty control state is received. Generated blocks remain ordered
in a bounded FIFO and the producer waits when the browser is behind.
The six sink latents and rolling tail use fixed logical RoPE positions, so the
global session frame number does not index beyond the trained local window.

## Configuration

The HTTP entry point exposes `--height` (480), `--width` (832), `--fps` (12), and `--control-latent-frames` (3).
Three causal latents per control update match the official streaming checkpoint; the one-latent mode is experimental.
The playback FPS is not a measured compute throughput. See the [pipeline architecture](../../docs/en/abot_world.md)
for model and cache behavior.

## LiveKit

The LiveKit path uses TeleFuser's existing `stream-serve` service and the
shared LingBot browser controls. Start coturn with one fixed relay port and
LiveKit Server first, then run the model worker:

```bash
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 \
  --listening-port=3478 --min-port=49160 --max-port=49160 \
  --user=livekit-demo:livekit-demo-password --realm=livekit.local \
  --fingerprint --lt-cred-mech --no-tls --no-dtls --no-cli \
  --allow-loopback-peers
```

```bash
livekit-server --dev
```

Then run the model worker:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=0 \
telefuser stream-serve examples/abot_world/abot_world_livekit_service.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --worker-gpu-map 0 --max-sessions-per-worker 1 \
  --port 8088 --skip-validation
```

Serve the reused browser page in another terminal:

```bash
python examples/abot_world/abot_world_livekit.py \
  --server-url http://127.0.0.1:8088 --port 8092 --no-open
```

Open `http://127.0.0.1:8092`, upload an image, and click **Start**. Hold a movement key to receive generated video.
Stop the browser session before stopping the browser proxy, model worker, LiveKit, and TURN relay in that order.

The SSH connection must also forward relay port `49160` in addition to
`8092`, `7880`, and `3478`. The default image requires the upstream checkout described above, but an uploaded image
is sent as a data URL in the session request. It sends the existing `tf.control`
`control_state` and press/release messages; the ABot service emits a preview
first and then ordered 12 FPS chunks only while controls are held.

## Test tiers

CPU contract tests cover action-channel layout, checkpoint conversion, sink
KV rolling, RoPE boundary validation, session cleanup, and the direct runtime
idle/FIFO behavior:

```bash
pytest tests/unit/pipelines/abot_world
```

The 30-block GPU smoke is opt-in because it loads the release checkpoint:

```bash
ABOT_WORLD_MODEL_ROOT=/path/to/ABot-World-0-5B-LF \
ABOT_WORLD_TEST_IMAGE=/path/to/initial.png \
pytest -m "gpu and slow" tests/integration/test_abot_world_smoke.py -v
```

The smoke uses the public 480x832 shape, a fixed seed, and a fixed control
state. It checks that every block decodes frames and that the session's
emitted-frame counter matches the observed count. It is a generation contract
test, not a visual-quality or long-horizon parity claim.

## Troubleshooting

- No initial image: provide an existing server-side path in the HTTP controller, or upload an image in the LiveKit UI.
- Preview without new frames: hold a movement or camera key; an empty control state intentionally keeps generation idle.
- No LiveKit media connection: check the TURN relay and forwarded ports using the
  [stream server troubleshooting guide](../../docs/en/stream_server.md#production-deployment).

Documentation builds validate publication and links only. Run the test tiers above separately to validate runtime behavior.
