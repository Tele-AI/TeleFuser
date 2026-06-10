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

TeleFuser is a high-performance runtime for world model inference and multimodal generation. It is designed for continuous, low-latency, stateful visual generation workloads such as real-time world models, interactive agents, speech-driven animation, and streaming visual systems.

## Why TeleFuser

Most open-source inference stacks are optimized for one of three cases:

- one-shot image generation
- offline video generation
- general LLM serving

Real-time world models need a different runtime profile: continuous execution, streaming output, bidirectional interaction, stateful sessions, long-context efficiency, and stable performance under concurrency. TeleFuser focuses on those runtime problems directly.

The project treats a world model as more than a function that returns a single clip. It provides the infrastructure needed to run a model as a continuously updated system that can receive input, keep state, and emit frames progressively.

## What TeleFuser Provides

- **World-model-oriented runtime**: Support for continuous video generation, interactive sessions, and bidirectional control loops.
- **AI Dev First interfaces**: Pipelines can publish `PIPELINE_CONTRACT` / `PIPELINE_MANIFEST` metadata so agents and services can discover tasks, inputs, and parameters programmatically.
- **Asynchronous pipeline scheduling**: Stage-based execution with request isolation, resource locking, and parallel stage groups.
- **Streaming transport**: WebRTC-based streaming with media tracks plus DataChannel control for real-time inference.
- **Scalable GPU runtime**: Multi-GPU execution with tensor parallelism, sequence parallelism, Ray-based deployment, and distributed worker orchestration.
- **Inference optimization stack**: Triton kernels, optimized attention backends, quantization, offload, and feature caching.
- **Unified serving**: Local Python API, `telefuser serve` for task APIs, and `telefuser stream-serve` for continuous streaming services.

## World Model Inference Focus

TeleFuser is built around the runtime requirements that world models expose in production:

- **Continuous execution instead of one-shot calls**: stream frames as they are produced instead of waiting for full completion.
- **Interactive control**: accept prompts, controls, images, audio, or action signals while a session is active.
- **Stateful sessions**: keep runtime state across chunks rather than rebuilding the full pipeline every step.
- **Low first-frame latency**: expose partial outputs quickly through async scheduling and streaming transport.
- **Long-horizon efficiency**: reduce memory pressure for long videos and repeated denoising through sequence parallelism, offload, and caching.

Today that maps to concrete capabilities in this repo, including:

- bidirectional WebRTC sessions for `LingBot-World-Fast`
- speech-to-video generation for `LiveAct`
- streaming video processing for `FlashVSR`
- long-form and continuation workflows for `LongCat-Video`
- batch and async video generation for `WanVideo`, `HunyuanVideo`, and `LTX Video`

## Quick Start

### Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

For WebRTC streaming:

```bash
pip install -e ".[webrtc]"
```

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

TeleFuser includes a bidirectional WebRTC demo for `LingBot-World-Fast`.

```bash
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/LingBot-World

telefuser stream-serve examples/stream_server/stream_lingbot_world_fast.py \
  -p 8088 \
  --skip-validation

python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://localhost:8088 \
  --image-path /path/to/input.png
```

This starts a continuous session where the client sends control messages over a WebRTC DataChannel and receives generated video frames over media tracks.

### 3. Batch Service Mode

```bash
telefuser serve examples/wan_video/wan22_14b_text_to_video_service.py --port 8000
```

TeleFuser exposes:

- native task APIs under `/v1/tasks/*`
- OpenAI-compatible image and video APIs under `/v1/images` and `/v1/videos`
- service metadata that reflects the pipeline contract

See [docs/en/service.md](docs/en/service.md) for full API details.

## Serving Modes

### `telefuser serve`

Use this mode for request-response inference with task management, standard REST APIs, and service metadata.

- good fit for batch text-to-video, image-to-video, image generation, and super-resolution
- supports pipeline contracts for structured parameter exposure
- supports OpenAI-compatible routes for easier client integration

### `telefuser stream-serve`

Use this mode for continuous streaming workloads.

- server-push WebRTC for progressive video output
- bidirectional WebRTC for interactive control loops
- useful for real-time world models, speech-driven generation, and streaming media pipelines

See [docs/en/stream_server.md](docs/en/stream_server.md) for stream protocol details.

## AI Dev First Runtime

TeleFuser is designed so pipelines are understandable not only to human developers, but also to automated systems and agents.

- `PIPELINE_CONTRACT` and `PIPELINE_MANIFEST` define supported tasks, required file inputs, defaults, and user-facing parameters.
- the service layer uses those contracts to expose machine-readable metadata
- the same pipeline can be used locally, through REST APIs, or through streaming services

This is the core of the project's "AI Dev First" direction: standardize runtime behavior so orchestration systems can discover and use pipelines without reverse-engineering internal code paths.

## Architecture

TeleFuser uses a layered runtime architecture that maps cleanly to the repository structure:

1. **Access layer**: FastAPI task APIs and WebRTC streaming entrypoints.
2. **Service and scheduling layer**: request routing, task management, stream sessions, and orchestration.
3. **Pipeline abstraction layer**: stage-based pipelines with async execution, request isolation, and resource locks.
4. **Model and optimization layer**: model loading, attention selection, quantization, offload, LoRA, and cache integration.
5. **Execution backend layer**: optimized ops, Triton kernels, and device-specific implementations.

Relevant directories:

```text
telefuser/
├── service/         # REST APIs, streaming APIs, WebRTC integration
├── orchestrator/    # Pipeline orchestration
├── pipelines/       # Model-specific pipelines
├── distributed/     # TP / SP / FSDP / Ray utilities
├── feature_cache/   # AdaTaylorCache
├── ops/             # Compile-aware operator dispatch
├── kernel/triton/   # Triton kernels
└── models/          # DiT, VAE, encoders, decoders
```

## Runtime Capabilities

- **Async pipeline scheduling**: run independent stages concurrently and gate shared resources with lock groups.
- **Distributed inference**: tensor parallelism, sequence parallelism, Ray-based multi-GPU deployment, and pipeline-scale orchestration.
- **Attention backends**: Torch SDPA, FlashAttention, SageAttention, sparse attention variants, and other configurable implementations.
- **Feature caching**: `AdaTaylorCache` accelerates supported diffusion models with calibrated skip/reuse logic.
- **Memory optimization**: CPU offload, weight reuse, and runtime-aware loading strategies for large video models.
- **Quantization**: FP8 and INT8-related runtime support where the model/backend path allows it.
- **Streaming output**: progressive frame delivery over WebRTC with optional audio tracks.

## Supported Pipelines

### World Model and Real-Time Oriented

| Pipeline | Task | Notes |
|----------|------|-------|
| `LingBot-World-Fast` | Bidirectional world-model streaming | Interactive WebRTC control loop via [examples/stream_server/stream_lingbot_world_fast.py](examples/stream_server/stream_lingbot_world_fast.py) |
| `LiveAct` | S2V | Speech-driven talking head generation via [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py) |
| `FlashVSR` | VSR | Streaming video super-resolution via [examples/flashvsr/README.md](examples/flashvsr/README.md) |
| `LongCat-Video` | T2V, I2V, VC | Long-form generation and continuation via [examples/longcat_video/README.md](examples/longcat_video/README.md) |

### Video Generation

| Pipeline | Task | Notes |
|----------|------|-------|
| `WanVideo` (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | Main video generation family, including async and service examples in [examples/wan_video/README.md](examples/wan_video/README.md) |
| `HunyuanVideo` | T2V, I2V | Supported via [examples/hunyuan_video/README.md](examples/hunyuan_video/README.md) |
| `LTX Video` | I2V + Audio | Unified audio-video generation via [examples/ltx_video/README.md](examples/ltx_video/README.md) |

### Image Generation and Other Multimodal Pipelines

| Pipeline | Task | Notes |
|----------|------|-------|
| `Qwen-Image` | T2I, Edit | [examples/qwen_image/README.md](examples/qwen_image/README.md) |
| `Z-Image` | T2I | [examples/z_image/README.md](examples/z_image/README.md) |
| `Flux2 Klein` | T2I | [examples/flux2_klein/README.md](examples/flux2_klein/README.md) |

## Examples

Key entry points:

- [examples/wan_video/README.md](examples/wan_video/README.md)
- [examples/longcat_video/README.md](examples/longcat_video/README.md)
- [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py)
- [examples/flashvsr/README.md](examples/flashvsr/README.md)
- [examples/ltx_video/README.md](examples/ltx_video/README.md)
- [examples/stream_server/stream_lingbot_world_fast.py](examples/stream_server/stream_lingbot_world_fast.py)
- [examples/stream_server/webrtc_bidirectional_demo.py](examples/stream_server/webrtc_bidirectional_demo.py)

To inspect CLI options for an example:

```bash
python examples/wan_video/wan21_1_3b_text_to_video_hf.py --help
```

The [examples/README.md](examples/README.md) file documents the example runner and baseline comparison workflow.

## Documentation

- [docs/en/service.md](docs/en/service.md): REST serving, task APIs, OpenAI-compatible APIs
- [docs/en/stream_server.md](docs/en/stream_server.md): continuous streaming and WebRTC protocols
- [docs/en/parallel.md](docs/en/parallel.md): distributed inference architecture
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
- World-model examples such as `LingBot-World-Fast` require external checkpoints and environment setup.
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
