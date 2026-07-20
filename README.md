<div align="center">
  <img src="assets/telefuser_logo.png" width="80%">
</div>

<p align="center">
  <a href="README_zh.md">中文</a> | English
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-green" alt="CUDA">
</p>

TeleFuser is a high-performance runtime for world model inference and multimodal generation. It is designed for continuous, low-latency, stateful visual generation workloads such as real-time world models, speech-driven animation, and streaming visual systems.

## News 📰

- ✨ **2026-07-15**: Added [**LingBot-World v2**](https://github.com/Robbyant/lingbot-world-v2) support for offline generation, interactive WebRTC streaming, and multi-GPU inference.

- ✨ **2026-07-06**: Added external **CacheSeek** latent cache integration for service-mode cross-request reuse. Cache hits can skip the first N denoising steps; the Wan2.2 cache-enabled service example snapshots `[5, 10, 15, 20, 25]` by default. See [docs/en/latent_cache.md](docs/en/latent_cache.md).

## Why TeleFuser

Most open-source inference stacks are optimized for one of three cases:

- one-shot image generation
- offline video generation
- general LLM serving

Real-time world models need a different runtime profile: continuous execution, streaming output, bidirectional interaction, stateful sessions, long-context efficiency, and stable performance under concurrency. TeleFuser focuses on those runtime problems directly.

The project treats a world model as more than a function that returns a single clip. It provides the infrastructure needed to run a model as a continuously updated system that can receive input, keep state, and emit frames progressively.

## What TeleFuser Provides

- **World-model-oriented runtime**: Support for continuous video generation, interactive sessions, and bidirectional control loops.
- **ADF (AI Dev First)**: Repository layers, pipeline contracts, examples, and docs are structured for coding agents to discover capabilities, follow project conventions, and extend pipelines efficiently.
- **Streaming pipeline scheduler**: Actor-owned stateful stages, bounded artifact edges, per-session ordering, backpressure, lifecycle cleanup, and explicit resource groups.
- **Streaming transport**: WebRTC-based streaming with media tracks plus DataChannel control for real-time inference.
- **Scalable GPU runtime**: Multi-GPU execution with tensor parallelism, sequence parallelism, optional Ray workers, and distributed service replicas.
- **Inference optimization stack**: Triton kernels, optimized attention backends, quantization, offload, feature caching, and CacheSeek latent cache integration.
- **Unified serving**: Local Python API, `telefuser serve` for task APIs, and `telefuser stream-serve` for continuous streaming services.

## Quick Start

### Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

WebRTC streaming support is included in the default installation through `aiortc`.

### 1. Batch Video Inference

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline
import torch

pipe = Wan21VideoPipeline.from_pretrained(
    model_id_or_path="Wan-AI/Wan2.1-T2V-1.3B",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

video = pipe(
    prompt="A cat playing piano",
    num_frames=81,
    height=480,
    width=832,
)
```

### 2. Real-Time World Model Demo

TeleFuser includes a bidirectional WebRTC demo for `LingBot-World v2`.
LingBot-World v2 uses camera control and its v2 PPL defaults; its streaming example caps a session at two minutes.

LingBot streaming uses the actor-based scheduler for both offline and service execution. Encode, DiT, and decode may
overlap even on the same GPU; move stages only when memory placement requires it. See the
[streaming scheduler guide](docs/en/stream_scheduler.md).

For a laptop browser connected through VS Code Remote SSH, coturn is the only additional system package required;
no extra Python package is needed. On Debian or Ubuntu, install it with:

```bash
sudo apt-get update
sudo apt-get install -y coturn
```

The package provides both `turnserver` and the `turnutils_uclient` verification tool. Skip this step when both
commands already exist, or when the browser and GPU service run on the same physical machine.

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TELEFUSER_TURN_SERVER='turn:127.0.0.1:3478?transport=tcp' \
TELEFUSER_TURN_USERNAME=telefuser \
TELEFUSER_TURN_CREDENTIAL=telefuser-turn \
telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
  --gpu-num 4 -p 8088 --host 0.0.0.0 --skip-validation

python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 \
  --port 8091 \
  --image-path examples/data/lingbot_world_fast/image.jpg \
  --turn-url 'turn:localhost:3478?transport=tcp' \
  --turn-username telefuser --turn-credential telefuser-turn \
  --force-turn-relay --ice-gather-timeout-ms 30000 --no-open
```

This starts a continuous session where the client sends control messages over a WebRTC DataChannel and receives
generated video frames over media tracks. When the browser runs on a laptop through VS Code Remote SSH, configure
TURN over TCP and forward ports `8091` and `3478`; port `8088` does not need forwarding because the demo proxies
signaling requests. Keep local port `3478` equal to remote port `3478`; the forwarded 8091 port may use any available
local port. Without VS Code, run the equivalent tunnel from a terminal on the laptop:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 8091:127.0.0.1:8091 \
  -L 3478:127.0.0.1:3478 \
  USER@SERVER_HOST
```

Then open `http://localhost:8091`. The TURN command and credentials above are development examples. See the
[stream server guide](docs/en/stream_server.md) and the
[LingBot example README](examples/lingbot/README.md) for coturn startup and the tested four-H100 setup.

If the browser runs on the same physical machine as TeleFuser, no SSH tunnel or TURN server is needed. Unset all
`TELEFUSER_TURN_*` variables, start the service on `127.0.0.1:8088`, run the demo without any `--turn-*` or
`--force-turn-relay` arguments, and open `http://localhost:8091`. This does not apply when only the shell is on the
server through SSH but the browser still runs on a laptop.

### 3. Batch Service Mode

```bash
telefuser serve examples/wan_video/wan22_14b_text_to_video_h100.py --task t2v --port 8000
```

TeleFuser exposes:

- native task APIs under `/v1/tasks/*`
- OpenAI-compatible image and video APIs under `/v1/images` and `/v1/videos`
- service metadata that reflects the pipeline contract

See [docs/en/service.md](docs/en/service.md) for full API details.

## Architecture

TeleFuser uses a layered runtime architecture that maps cleanly to the repository structure:

1. **Access layer**: FastAPI task APIs and WebRTC streaming entrypoints.
2. **Service layer**: request routing, task management, stream sessions, replica pools, and integration with pipeline execution.
3. **Pipeline abstraction layer**: model-specific `BasePipeline` / `BaseStage` components, with an actor-based streaming orchestrator for bounded dataflow, session ordering, metrics, and cleanup.
4. **Model and optimization layer**: model loading, attention selection, quantization, offload, LoRA, and cache integration.
5. **Execution backend layer**: optimized ops, Triton kernels, and device-specific implementations.

Relevant directories:

```text
telefuser/
├── service/         # REST APIs, streaming APIs, WebRTC integration
├── orchestrator/    # Request orchestration and actor-based streaming scheduler
├── pipelines/       # Model-specific pipelines
├── distributed/     # TP / SP / FSDP / Ray utilities
├── feature_cache/   # AdaTaylorCache
├── ops/             # Compile-aware operator dispatch
├── kernel/triton/   # Triton kernels
└── models/          # DiT, VAE, encoders, decoders
```

## Supported Pipelines

### World Model and Real-Time Oriented

| Pipeline | Task | Notes |
|----------|------|-------|
| `LingBot-World v2` | Bidirectional world-model streaming | Interactive WebRTC control loop via [examples/lingbot/lingbot_world_v2_image_to_video_h100.py](examples/lingbot/lingbot_world_v2_image_to_video_h100.py) |
| `LiveAct` | S2V | Speech-driven talking head generation via [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py) |
| `FlashVSR` | VSR | Streaming video super-resolution via [examples/flashvsr/README.md](examples/flashvsr/README.md) |

### Video Generation

| Pipeline | Task | Notes |
|----------|------|-------|
| `WanVideo` (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | Main video generation family, including async and service examples in [examples/wan_video/README.md](examples/wan_video/README.md) |
| `HunyuanVideo` | T2V, I2V | Supported via [examples/hunyuan_video/README.md](examples/hunyuan_video/README.md) |
| `LTX Video` | I2V + Audio | Unified audio-video generation via [examples/ltx_video/README.md](examples/ltx_video/README.md) |
| `LongCat-Video` | T2V, I2V, VC | Long-form generation and continuation via [examples/longcat_video/README.md](examples/longcat_video/README.md) |

### Image Generation and Other Multimodal Pipelines

| Pipeline | Task | Notes |
|----------|------|-------|
| `Qwen-Image` | T2I, Edit | [examples/qwen_image/README.md](examples/qwen_image/README.md) |
| `Z-Image` | T2I | [examples/z_image/README.md](examples/z_image/README.md) |
| `Flux2 Klein` | T2I | [examples/flux2_klein/README.md](examples/flux2_klein/README.md) |

See [examples/README.md](examples/README.md) for the example runner and baseline comparison workflow.

## Documentation

- [docs/en/service.md](docs/en/service.md): REST serving, task APIs, OpenAI-compatible APIs
- [docs/en/stream_server.md](docs/en/stream_server.md): continuous streaming and WebRTC protocols
- [docs/en/stream_scheduler.md](docs/en/stream_scheduler.md): actor-based stage scheduling, backpressure, lifecycle, metrics, and LingBot placement
- [docs/en/parallel.md](docs/en/parallel.md): distributed inference architecture
- [docs/en/latent_cache.md](docs/en/latent_cache.md): CacheSeek latent cache integration
- [docs/en/feature_cache.md](docs/en/feature_cache.md): `AdaTaylorCache`
- [docs/en/model_loading.md](docs/en/model_loading.md): model loading patterns
- [docs/en/attention.md](docs/en/attention.md): attention backends and configuration
- [docs/en/torch_compile_compatibility.md](docs/en/torch_compile_compatibility.md): compile-related constraints
- [docs/en/adding_new_model.md](docs/en/adding_new_model.md): integrating new models
- [docs/en/adding_new_example.md](docs/en/adding_new_example.md): authoring examples and pipeline contracts

## Known Limitations

- `AdaTaylorCache` is only calibrated for selected model families.
- `torch.compile` support is still experimental in parts of the stack.
- Some optimized paths require specific GPU architectures and CUDA versions.
- World-model examples such as `LingBot-World v2` require external checkpoints and environment setup.
- Multi-machine deployment exists in the architecture but may require project-specific integration and validation.

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution workflow and [AGENTS.md](AGENTS.md) for project-specific agent guidance.

## License

Apache 2.0 License. See [LICENSE](LICENSE).

## Acknowledgements

TeleFuser builds on and is inspired by a broad set of open-source efforts in multimodal generation and inference systems, including:

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)
- [LightX2V](https://github.com/ModelTC/LightX2V)
- [cache-dit](https://github.com/vipshop/cache-dit)
- [diffusers](https://github.com/huggingface/diffusers)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
