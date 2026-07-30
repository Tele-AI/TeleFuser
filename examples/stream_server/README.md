# LiveKit Stream Examples

TeleFuser has one streaming backend: LiveKit. The examples cover both service contracts:

- `stream_video_replay.py`: server-push video.
- `stream_arrow_overlay.py`: bidirectional control rendered into output frames.
- `livekit_bidirectional_demo.py`: browser UI for LingBot camera control.
- `_control_demo_ui.py`: private shared HTML/CSS/control asset used by the LiveKit demo.

The checked-in browser demo forces TCP TURN relay, so the complete interactive stack has four services. Install the
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
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey \
  --livekit-api-secret secret \
  --max-sessions-per-worker 2 \
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

This command loads one LingBot service instance, not one instance per user. The one model worker retains up to two
independent LiveKit/pipeline sessions and the LingBot execution lease time-slices their model chunks. The setting is
not a generic replication option: server-push capacity remains one, and other bidirectional services must implement
their own safe cross-session execution policy.

The LiveKit Python SDK is part of TeleFuser's base dependencies; the LiveKit Server is installed and operated
separately. See the [Stream Server guide](../../docs/en/stream_server.md) for the session API, data topics, worker
lifecycle, remote development, and production deployment.
