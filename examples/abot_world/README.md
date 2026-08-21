# ABot-World-0-5B-LF

This example exposes a local single-GPU HTTP entry point and a concurrent
TurboServe-style LiveKit entry point. The HTTP controller is useful for model debugging:

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --host 127.0.0.1 \
  --port 7860
```

The browser controls WASD/arrow movement and IJKL camera rotation. Connecting
creates the image-conditioned causal session but does not advance the DiT
until a non-empty control state is received. Generated blocks remain ordered in a bounded per-session queue. The default
`latest` mode drops the oldest complete block, with metrics, only when a slow
browser fills the queue; `lossless` mode applies scheduling backpressure instead.
The six sink latents and rolling tail use fixed logical RoPE positions, so the
global session frame number does not index beyond the trained local window.

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

For multiple GPUs, use one worker per GPU, for example
`--num-workers 4 --worker-gpu-map '0;1;2;3' --worker-mode process-nccl`. This mode loads each
GPU replica in a spawned child so Python, asyncio, CUDA contexts, and model execution are isolated across GPUs.
It keeps room transport in the parent and enables NCCL session migration; its NCCL group is fixed, so do not
enable process autoscaling. Use plain `--worker-mode process` plus a non-zero queue and
`--enable-autoscaling --autoscaling-min-workers 1` for on-demand independent replicas.
Each worker continuously batches compatible retained sessions through both DiT
and cached VAE decode; GPU IDs are passed explicitly to the ABot model factory.

For a reproducible four-GPU deployment, use one worker per GPU with
`process-nccl` and an explicit worker map. The parent scheduler assigns each
public session; clients do not choose a GPU.

Serve the reused browser page in another terminal:

```bash
python examples/abot_world/abot_world_livekit.py \
  --server-url http://127.0.0.1:8088 --port 8092 --no-open
```

The SSH connection must also forward relay port `49160` in addition to
`8092`, `7880`, and `3478`. The page defaults to the checked-in ABot sample image, but an uploaded image
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

The multi-session benchmark exercises 30 continuously batched blocks:

```bash
python tools/validation/benchmark_abot_turboserve.py \
  --model-root /path/to/ABot-World-0-5B-LF --image /path/to/initial.png \
  --sessions 2 --chunks 30 --batch-size 2 --output /tmp/abot-turboserve.json
```

The smoke and benchmark check generation, session-state isolation, block ordering,
and batching; they are not visual-quality or long-horizon parity claims.
