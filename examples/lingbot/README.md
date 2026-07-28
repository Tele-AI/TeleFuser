# LingBot-World Examples

Offline image-to-video and interactive WebRTC streaming generation for LingBot-World-Fast (v1) and LingBot-World
v2. Both variants use the shared causal-fast streaming engine; their checkpoint layout and PPL defaults differ.

## Model Directory

Both offline examples require the Wan2.2 I2V base model. v1 and v2 use separate LingBot checkpoint directories:

```text
${TF_MODEL_ZOO_PATH}/
├── Wan2.2-I2V-A14B/
└── lingbot/
    ├── lingbot-world-fast/
    └── lingbot-world-v2-14b-causal-fast/
        └── transformers/
```

Set the model root before running the example:

```bash
export TF_MODEL_ZOO_PATH=/path/to/model_zoo
```

## Feature Support

| Feature | Support |
| --- | --- |
| Offline image-to-video | ✔️ |
| Camera control | ✔️ |
| Continuous chunked generation | ✔️ |
| Single-GPU inference | ✔️ |
| Ulysses Sequence Parallel | ✔️ |
| FSDP | Configurable through PPL_CONFIG |
| H100 Sage Attention | ✔️ |

## Files

### lingbot_world_fast_image_to_video_h100.py

Offline generation and stream-server entry point for LingBot-World-Fast with camera control.

Default configuration:

- Resolution: `480p`
- Output frames: `81`
- Frame rate: `16 FPS`
- Chunk size: `3` latent frames
- Seed: `42`
- Control mode: `cam`
- Attention backend: `SAGE_ATTN_2_8_8_SM90`
- Default input: `examples/data/lingbot_world_fast/image.jpg`
- Default control directory: `examples/data/lingbot_world_fast/`
- Default output: `work_dirs/lingbot_world_fast_i2v_<gpu_num>gpu.mp4`

### lingbot_world_v2_image_to_video_h100.py

Offline generation and stream-server entry point for camera-controlled v2. The default is 77 frames at 16 FPS: 20 latent frames, exactly five
complete chunks of four. With complete chunk streaming, 81 output frames cannot be represented by `chunk_size=4`.
The v2 checkpoint only supports camera control and uses its PPL-configured SageAttention SM90 backend, local attention, sink size,
and timesteps.

```bash
python examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
    --gpu_num 4 \
    --model_root "${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B" \
    --v2_model_root "${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-v2-14b-causal-fast/transformers"
```

## Usage

### Four H100 GPUs

The recommended configuration uses four H100 GPUs with Ulysses sequence parallelism for DiT inference:

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
    --gpu_num 4 \
    --model_root "${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B" \
    --fast_model_root "${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast"
```

### Single H100 GPU

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
    --gpu_num 1 \
    --model_root "${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B" \
    --fast_model_root "${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast"
```

### Custom Input and Output

`--action_path` accepts a directory containing `poses.npy` and, in action-control mode, `action.npy`.
Pass the camera calibration file separately with `--intrinsics-path`. If it was calibrated at a resolution other
than `832x480`, also pass `--intrinsics-width` and `--intrinsics-height`; the pipeline transforms the calibration to
the generated frame size.

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
    --gpu_num 4 \
    --model_root "${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B" \
    --fast_model_root "${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast" \
    --image_path /path/to/input.jpg \
    --action_path /path/to/camera_control \
    --intrinsics-path /path/to/intrinsics.npy \
    --intrinsics-width 1920 \
    --intrinsics-height 1080 \
    --prompt "A cinematic scene with smooth camera motion" \
    --resolution 720p \
    --fps 16 \
    --seed 42 \
    --output work_dirs/lingbot_world_fast_custom.mp4
```

### Repository-Provided Input

The repository includes an image, poses, and intrinsics that can be used directly. Omit the input, control, and
output options to use these defaults:

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
    --gpu_num 4 \
    --model_root "${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B" \
    --fast_model_root "${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast"
```

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--gpu_num` | `1` | Total visible GPUs; selects the LingBot VAE and DiT placement strategy |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B` | Wan2.2 I2V base model directory |
| `--fast_model_root` | `${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast` | LingBot-World-Fast model directory |
| `--image_path` | Bundled `image.jpg` | Input image path |
| `--action_path` | Bundled control directory | Directory containing `poses.npy` and optional `action.npy` |
| `--intrinsics-path` | Bundled `intrinsics.npy` | Camera intrinsics in `[fx, fy, cx, cy]` order |
| `--intrinsics-width` | `832` | Pixel width of the calibration coordinate system |
| `--intrinsics-height` | `480` | Pixel height of the calibration coordinate system |
| `--prompt` | Bundled English prompt | Positive guidance prompt |
| `--resolution` | `480p` | Output resolution; available values are `480p` and `720p` |
| `--fps` | `16` | Output video frame rate |
| `--seed` | `42` | Random seed |
| `--output` | `work_dirs/lingbot_world_fast_i2v_<gpu_num>gpu.mp4` | Output MP4 path |

Display the command-line help:

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py --help
```

## Notes

- Offline frame counts are fixed in each example's `PPL_CONFIG`: v1 uses 81 frames (seven complete chunks of
  three latent frames) and v2 uses 77 frames (five complete chunks of four latent frames).
- `--gpu_num` must not exceed the number of GPUs visible to the process. For example, set
  `CUDA_VISIBLE_DEVICES=0,1,2,3` to select four devices.
- The example explicitly closes its parallel workers on exit. If the process is forcibly terminated, check for
  residual `spawn_main` child processes.

## Real-Time Streaming

The same examples expose both offline generation and a stream-server `get_service()` entry point. Configure the
service topology with `stream-serve --worker-gpu-map`; the size of the assigned GPU group is passed to
`get_service(gpu_num=...)`. Use `CUDA_VISIBLE_DEVICES` to select the physical devices. Do not use `torchrun` because
TeleFuser creates workers internally.

### Scheduler and Stage Placement

LingBot offline and stream-server execution share the actor-based streaming scheduler. The bundled examples use
fixed placement for the following total GPU counts:

| Total GPUs | DiT GPUs | VAE encode GPU | VAE decode GPU |
| --- | --- | --- | --- |
| 2 | `0-1` | `0` | `1` |
| 4 | `0-3` | `0` | `1` |
| 5 | `0-3` | `4` | `4` |
| 6 | `0-4` | `5` | `5` |

For other counts, the examples retain the PPL-configured VAE devices and assign all visible GPUs to DiT. Direct
`LingBotWorldFastPipelineConfig` users may set `vae_encode_config`, `vae_decode_config`, and `dit_config` independently.

### H100 Compile Benchmark

The v2 example was measured at 480p (832x464 internal size), 77 output frames, five latent chunks of four frames,
BF16 DiT, FP32 VAE, SageAttention SM90, `torch.compile` enabled, and FSDP disabled. Each value is the mean from a
second session after a complete warmup session. Pure DiT measures synchronous `denoise_and_update_cache`; chunk
period is the mean interval between decoded chunk outputs while encode, DiT, and decode overlap.

| Total H100 GPUs | Pure DiT seconds/chunk | Overlapped chunk period seconds/chunk |
| --- | --- | --- |
| 2 | 1.587 | 2.096 |
| 4 | 0.911 | 1.615 |

The scheduler does not infer a resource group from overlapping device IDs, so VAE encode, DiT, and VAE decode may
overlap on a shared GPU.

See the [streaming scheduler guide](../../docs/en/stream_scheduler.md) for
architecture, metric definitions, and lifecycle guarantees.

### Tested GPU and Duration Limits

LingBot-World-Fast uses a global KV cache that grows with the requested frame count even when FSDP is enabled. The
following 832x480 limits were verified on H100 80 GB GPUs with `chunk_size=3`, `16 FPS`, and
`sample_shift=10.0`:

| GPUs | Duration selected in the page | Frame count | Result |
| --- | --- | --- | --- |
| 2 H100 | 10 seconds | 153 | Passed |
| 2 H100 | 20 seconds | 321 | CUDA OOM while allocating KV cache |
| 4 H100 | 20 seconds | 321 | Passed, 27/27 chunks |

The four-GPU 20-second test used FSDP and Ulysses degree 4. Peak memory was approximately 58.6 GiB on GPU 0 and
41.6 GiB on GPUs 1-3. These are tested values, not universal limits; other resolutions and concurrent GPU users
change the available capacity.

LingBot-World v2 instead uses the fixed `local_attn_size=18`, `sink_size=6` sliding window configured by its
example. A four-H100, one-minute AIPerf replay on 2026-07-28 resolved to 60 complete chunks, 957 output frames, and
59.75 seconds of media. Its runtime metadata reported 240 latent frames but a fixed 27,144-token KV capacity.

| One-minute v2 measurement | Result |
| --- | ---: |
| Successful sessions | 1 / 1 |
| Target chunks / generated frames | 60 / 957 |
| Client frames / steady frames | 946 / 944 |
| Output cadence mean / p50 / p95 | 1.666 / 1.661 / 1.865 s |
| First / middle / last 20-chunk mean | 1.694 / 1.646 / 1.658 s |
| Session runtime | 104.031 s |

During initial WebRTC track startup, the client received 2 of the first chunk's 13 frames and then received all
`59 * 16 = 944` steady frames. Similar cadence in the first, middle, and last thirds confirms that the fixed attention
window avoided duration-driven degradation in this run. It did not reach real time: each chunk represents 1.0 second
of media, while p95 cadence was 1.865 seconds. See the
[AIPerf benchmark guide](../../docs/en/benchmark_aiperf.md) for the workload,
metric boundary, artifact path, and the observed client cleanup issue.

### LiveKit Transport

LiveKit provides room lifecycle, signaling, reconnects, adaptive media transport, and reliable control messages.
The Python clients are base TeleFuser dependencies; install the LiveKit Server and your platform's `coturn` package
separately for local development:

```bash
curl -sSL https://get.livekit.io | bash
```

The checked-in browser page forces TCP TURN relay. Start its matching development coturn service in terminal 1:

```bash
turnserver -n -m 1 \
    --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 \
    --listening-port=3478 --min-port=49160 --max-port=49200 \
    --user=livekit-demo:livekit-demo-password --realm=livekit.local \
    --fingerprint --lt-cred-mech --no-tls --no-dtls --no-cli \
    --allow-loopback-peers
```

Start LiveKit in terminal 2:

```bash
livekit-server --dev
```

Start the four-GPU LingBot worker and its session API in terminal 3:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
    --livekit-url ws://127.0.0.1:7880 \
    --livekit-api-key devkey \
    --livekit-api-secret secret \
    --num-workers 1 \
    --worker-gpu-map 0,1,2,3 \
    --port 8088 \
    --skip-validation
```

For the fixed-window v2 model, replace the pipeline path in that command with
`examples/lingbot/lingbot_world_v2_image_to_video_h100.py`; the LiveKit and worker options stay the same.

Then start the bundled browser demo in terminal 4:

```bash
python examples/stream_server/livekit_bidirectional_demo.py \
    --server-url http://127.0.0.1:8088 \
    --port 8092 \
    --no-open
```

The demo proxies the TeleFuser session API, so VS Code Remote SSH needs TCP forwarding for demo port `8092`, LiveKit
signaling port `7880`, and TURN listener `3478`; `8088` does not need forwarding. Open `http://127.0.0.1:8092`,
select an initial image, and click **Start**. See the [Stream Server guide](../../docs/en/stream_server.md) for the API
contract, control topics, shutdown order, troubleshooting, and production deployment notes.

The current runtime supports exactly one `in-process` worker. One admitted LiveKit room owns one model pipeline
instance until the session is closed or expires. Use separate service processes for additional workers until
process-worker mode is implemented.

Select the initial image in the browser before connecting; it is included in the session request. Real-time camera
poses come from LiveKit control messages. The LingBot-World v2 service uses the bundled `intrinsics.npy` and its `832x480`
calibration size by default, matching the offline example. A request can override that calibration. Other LingBot
services with neither configured nor request-provided intrinsics center the principal point on the selected image and
use its width as both focal lengths. Requests with calibrated intrinsics should also send `intrinsics_width` and
`intrinsics_height` so the service can transform them from calibration pixels to output pixels.

### Camera Controls

The page has separate translation and rotation pads:

| Input | Camera operation |
| --- | --- |
| `W` or `↑` | Move forward |
| `S` or `↓` | Move backward |
| `A` | Strafe left |
| `D` | Strafe right |
| `J` | Yaw left |
| `L` | Yaw right |
| `I` | Pitch up |
| `K` | Pitch down |

Multiple keys can be held together, for example `W+J`. Move/strafe steps default to `0.05` per video frame;
yaw/pitch steps default to `2°` per video frame, with pitch limited to `±85°`. Camera pose and pitch accumulate
across chunks. The browser sends the complete held-key snapshot; the service retains only the newest pending
short-press snapshot, so stale taps cannot override newer input. Releasing all keys stops requesting new chunks, and
WebRTC repeats the most recent frame while idle. **Release Controls** clears held and pending keys without changing
the accumulated pose; **Reset Camera Pose** explicitly clears keys and returns the pose to identity.

Every output chunk includes its immutable `applied_controls` snapshot. Its `MOVE`/`ROTATE` HUD is rendered from that
same snapshot, so the indicators describe the translation and rotation that actually generated the displayed frames.

#### Camera Motion Integration

The service maintains a camera-to-world matrix and a scalar accumulated pitch for each session. Every video-frame
integration step first updates yaw and pitch, then derives horizontal movement directions from the new camera
rotation:

```text
R_new   = Ry(yaw_delta) @ R_old @ Rx(pitch_delta)
forward = normalize([R_new[0, 2], 0, R_new[2, 2]])
right   = normalize([R_new[0, 0], 0, R_new[2, 0]])
t_new   = t_old + forward_or_backward + left_or_right
```

Yaw is applied around the world Y axis, pitch around the local camera X axis, and translation is projected onto the
world XZ plane. Pitch therefore changes the viewing direction without making forward motion fly upward. Simultaneous
translation keys are added directly, so diagonal motion is faster than a single-axis move.

The Wan VAE has a temporal compression factor of four. The service consequently performs four video-frame camera
steps between adjacent latent poses. With the default `chunk_size=3`, holding `W` produces the following positions
for the first two chunks:

```text
first chunk:  z = [0.0, 0.2, 0.4]
next chunk:   previous z = 0.4, current z = [0.6, 0.8, 1.0]
```

The previous chunk's final pose is carried across the boundary before framewise relative poses are computed. This
keeps the first motion in every subsequent chunk continuous instead of resetting it to the identity pose. Control is
sampled when a chunk is submitted; releasing or changing a key does not alter control chunks that were already
submitted or prefetched.

#### From Camera Poses to DiT Control

The accumulated absolute poses are converted into frame-to-frame relative poses. Relative translation is normalized
by the largest translation norm in that control chunk, matching the current LingBot preprocessing. As a result,
`control_move_step` controls the accumulated camera path but does not necessarily scale the model-visible translation
strength proportionally for constant-speed motion. Real-time control then multiplies the normalized translation by
`control_translation_scale` (default `3.0`), so a non-zero single-direction step has model-visible magnitude `3`
rather than `1`. Rotation deltas are not normalized in this way.

The relative poses and transformed camera intrinsics define a ray origin and ray direction for every output pixel.
Camera-control mode concatenates them into six-channel Plücker ray features, rearranges each spatial VAE-stride block
onto the latent grid, and sends the resulting tensor through the DiT camera-control embedding. Camera control does
not pass through the VAE; only the reference image and generated video use the VAE encode/decode paths.

### Troubleshooting

- **Blank 8092 page:** confirm the demo process is listening, open the exact forwarded URL from the VS Code Ports
  panel using `http://`, and hard-refresh. The demo uses a threaded HTTP server so VS Code probe connections do not
  block page requests.
- **No relay candidate:** confirm the configured TURN listener is reachable and its credentials match the browser
  RTC configuration. Browser logs should report at least one `typ=relay` candidate.
- **Static preview:** inspect LiveKit messages for `control_state` and `applying_direction_control`. If the server
  logged CUDA OOM, restart the service because a failed parallel worker cannot process another session.
- **Session already active:** click **Stop** or delete it with
  `curl -X DELETE http://127.0.0.1:8088/v1/stream/sessions/<session_id>`.
- **Residual workers after a forced exit:** terminate stale `spawn_main` processes before restarting.

For production TLS, firewall, media-port, and TURN requirements, see the
[stream server guide](../../docs/en/stream_server.md#production-deployment).
