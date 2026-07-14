# LingBot-World-Fast Examples

Offline image-to-video and online streaming generation with LingBot-World-Fast. This page describes how to run
the `lingbot_world_fast_image_to_video_h100.py` offline image-to-video example on H100 GPUs.

## Model Directory

The offline example requires the Wan2.2 I2V base model and the LingBot-World-Fast model. The recommended directory
layout is:

```text
${TF_MODEL_ZOO_PATH}/
├── Wan2.2-I2V-A14B/
└── lingbot/
    └── lingbot-world-fast/
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
| FSDP | Disabled by default in this example |
| H100 Sage Attention | ✔️ |

## Files

### lingbot_world_fast_image_to_video_h100.py

Offline video generation on H100 GPUs using LingBot-World-Fast with camera control.

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

`--action_path` accepts a directory. In camera-control mode, the directory must contain `poses.npy` and
`intrinsics.npy`.

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
    --gpu_num 4 \
    --model_root "${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B" \
    --fast_model_root "${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast" \
    --image_path /path/to/input.jpg \
    --action_path /path/to/camera_control \
    --prompt "A cinematic scene with smooth camera motion" \
    --resolution 720p \
    --frame_num 81 \
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
| `--gpu_num` | `1` | Number of GPUs used for Ulysses sequence parallelism |
| `--model_root` | `${TF_MODEL_ZOO_PATH}/Wan2.2-I2V-A14B` | Wan2.2 I2V base model directory |
| `--fast_model_root` | `${TF_MODEL_ZOO_PATH}/lingbot/lingbot-world-fast` | LingBot-World-Fast model directory |
| `--image_path` | Bundled `image.jpg` | Input image path |
| `--action_path` | Bundled control directory | Directory containing `poses.npy` and `intrinsics.npy` |
| `--prompt` | Bundled English prompt | Positive guidance prompt |
| `--resolution` | `480p` | Output resolution; available values are `480p` and `720p` |
| `--frame_num` | `81` | Number of output video frames |
| `--fps` | `16` | Output video frame rate |
| `--seed` | `42` | Random seed |
| `--output` | `work_dirs/lingbot_world_fast_i2v_<gpu_num>gpu.mp4` | Output MP4 path |

Display the command-line help:

```bash
python examples/lingbot/lingbot_world_fast_image_to_video_h100.py --help
```

## Notes

- `frame_num` must satisfy the pipeline's complete latent-chunk constraint. The default 81 frames correspond to
  21 latent frames, which are split into seven complete chunks.
- `--gpu_num` must not exceed the number of GPUs visible to the process. For example, set
  `CUDA_VISIBLE_DEVICES=0,1,2,3` to select four devices.
- The example explicitly closes its parallel workers on exit. If the process is forcibly terminated, check for
  residual `spawn_main` child processes.

## Real-Time Streaming

Start the bidirectional WebRTC service with two physical H100 GPUs (2 and 3):

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=2,3 \
telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py \
    --gpu-num 2 -p 8088 --host 0.0.0.0 --skip-validation
```

Run the browser controller on the server. The action directory supplies the same fixed camera intrinsics used by
the offline example; camera poses are generated from keyboard input in real time.

```bash
python examples/stream_server/webrtc_bidirectional_demo.py \
    --server-url http://127.0.0.1:8088 \
    --port 8091 \
    --image-path examples/data/lingbot_world_fast/image.jpg \
    --action-path examples/data/lingbot_world_fast \
    --frame-num 81 \
    --chunk-size 3 \
    --no-open
```

Use `W/S` to move forward/backward, `A/D` to strafe, `J/L` (or the left/right arrows) to yaw, and `I/K` to
pitch. Releasing all controls stops new chunk generation, so the WebRTC stream holds the last output frame.
