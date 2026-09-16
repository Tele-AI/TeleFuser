# ABot-World Examples

ABot-World-0-5B-LF provides a single-GPU interactive world-model controller with persistent causal state through a
local HTTP UI or the shared LiveKit streaming service.

## Model Source

| Model | HuggingFace | ModelScope | Purpose |
| --- | --- | --- | --- |
| ABot-World-0-5B-LF | [acvlab/ABot-World-0-5B-LF](https://huggingface.co/acvlab/ABot-World-0-5B-LF) | [amap_cvlab/ABot-World-0-5B-LF](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF) | Interactive causal DiT checkpoint |
| Wan2.2 VAE and T5 | [Bundled in ABot-World-0-5B-LF](https://huggingface.co/acvlab/ABot-World-0-5B-LF) | [Bundled in ABot-World-0-5B-LF](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF) | Video codec and prompt encoder |
| TAEHW2 decoder | [Bundled in ABot-World-0-5B-LF](https://huggingface.co/acvlab/ABot-World-0-5B-LF) | [Bundled in ABot-World-0-5B-LF](https://modelscope.cn/models/amap_cvlab/ABot-World-0-5B-LF) | Low-latency browser preview |

## Feature Support

| Feature | Support | Notes |
| --- | --- | --- |
| Interactive world generation | Supported | WASD/arrow movement and IJKL camera controls |
| Persistent causal sessions | Supported | Bounded KV cache and fixed local RoPE positions |
| Multi-GPU inference | Partial | Independent one-GPU replicas; one model session is not tensor-sharded |
| CPU offload | Supported | VAE, T5, and DiT use model CPU offload |
| Quantization | Unsupported | The loader uses BF16 DiT/T5 and FP32 VAE weights |
| Server API | Supported | Local HTTP and LiveKit stream-service entry points |

## Requirements

- GPU: one CUDA GPU per model worker; the optimized SageAttention path requires an SM90 H100-class GPU
- Software: the standard TeleFuser installation; LiveKit Server and coturn are required only for the LiveKit path
- Input assets: a source image, either uploaded in the UI or provided by path

Install TeleFuser by following the [development setup](../../CONTRIBUTING.md#development-setup). See the
[ABot architecture guide](../../docs/en/abot_world.md) for cache and service details.

## Model Directory

```text
/path/to/ABot-World-0-5B-LF/
|-- diffusion_pytorch_model.safetensors
|-- Wan2.2_VAE.pth
|-- taew2_2.pth
\-- models_t5_umt5-xxl-enc-bf16.pth
```

## Quick Start

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`, upload an image, connect, and hold a movement key to generate frames.

## Examples

### Local Interactive Controller

#### `abot_world_interactive_web.py`

```bash
python examples/abot_world/abot_world_interactive_web.py \
  --model-root /path/to/ABot-World-0-5B-LF \
  --height 480 --width 832 \
  --fps 8 --control-latent-frames 2
```

The default two-latent control block targets 8 FPS. Three causal latents match the official streaming checkpoint;
one latent remains experimental. Height and width must be divisible by 32, and `latent-frames` must equal `1 mod 3`.

### LiveKit Service

#### `abot_world_livekit_service.py`

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo CUDA_VISIBLE_DEVICES=0 \
telefuser stream-serve examples/abot_world/abot_world_livekit_service.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --worker-gpu-map 0 --max-sessions-per-worker 1 \
  --port 8088 --skip-validation
```

Start coturn and LiveKit Server before this command. For multiple independent replicas, use an explicit worker map
such as `--num-workers 4 --worker-gpu-map '0;1;2;3'`.

### LiveKit Browser

#### `abot_world_livekit.py`

```bash
python examples/abot_world/abot_world_livekit.py \
  --server-url http://127.0.0.1:8088 --port 8092 --no-open
```

Open `http://127.0.0.1:8092`, upload an image, and start the session.

#### `abot_world_livekit_service.py` and `_loader.py`

The service file provides the stream-service factory. `_loader.py` is a shared checkpoint loader rather than a
standalone entry point.

## Configuration

The output queue is bounded per session. `latest` mode drops the oldest complete block when a slow browser fills the
queue; `lossless` mode applies scheduling backpressure. Stop the browser session before shutting down the proxy,
model worker, LiveKit, and TURN relay.

## Validation

```bash
pytest tests/unit/pipelines/abot_world
```
