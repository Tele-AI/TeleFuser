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

- ✨ **2026-08-03**: Validated LingBot-World v2 target-side real-time generation on **4 x H100 80 GB** at
  832x480 and 16 FPS. The current 77-frame gate reached **17.14 steady compute FPS**; see the
  [reproducible benchmark](docs/en/benchmark_aiperf.md#current-four-h100-real-time-gate).
- ✨ **2026-07-27**: Unified streaming on LiveKit with room sessions, retained multi-session admission, LingBot
  chunk-boundary time slicing, reconnect-friendly browser transport, and server-push/bidirectional contracts.
- ✨ **2026-07-22**: Added [**LingBot-Video**](examples/lingbot_video/README.md) support for Dense and MoE T2I/T2V/TI2V generation, native four-GPU CFG/SP execution, and in-memory MoE refinement.
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
- **Streaming transport**: LiveKit-backed WebRTC for server-push media and resilient bidirectional sessions, with
  room lifecycle, reconnect handling, participant roles, and reliable controls.
- **Scalable GPU runtime**: Multi-GPU execution with tensor parallelism, sequence parallelism, optional Ray workers, and distributed service replicas.
- **Inference optimization stack**: Triton kernels, optimized attention backends, quantization, offload, feature caching, and CacheSeek latent cache integration.
- **Unified serving**: Local Python API, `telefuser serve` for task APIs, and `telefuser stream-serve` for LiveKit
  rooms and media.

## Quick Start

### Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

TeleFuser does not require `tf-kernel` to run. The project does not publish prebuilt tf-kernel wheels or a source
distribution to a public package index. Build the optional extension with the Makefile under `tf-kernel/`; a locally
built wheel may be distributed only to compatible environments. See the [tf-kernel README](tf-kernel/README.md) and
[installation and usage guide](docs/en/tf_kernel.md) for build, verification, and artifact compatibility details.

The base installation includes the LiveKit Python SDKs used by `telefuser stream-serve`. A LiveKit Cloud project or
self-hosted LiveKit Server is operated separately.

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

### 2. Real-Time World Model WebRTC Demo

TeleFuser streams `LingBot-World v2` through LiveKit. LingBot-World v2 uses camera control and its v2 PPL defaults;
its streaming example caps a session at two minutes.

The validated four-H100 configuration sustains 17.14 target-side compute FPS for the default 77-frame, 832x480
request, above its 16 FPS playback target. This is a synchronized pipeline-compute metric; model loading, LiveKit
encoding, network delivery, and client rendering are measured separately. See the
[LingBot example guide](examples/lingbot/README.md#validated-four-h100-real-time-gate) for the exact command and
chunk timings.

LingBot streaming uses the actor-based scheduler for both offline and service execution. Encode, DiT, and decode may
overlap even on the same GPU; move stages only when memory placement requires it. See the
[streaming scheduler guide](docs/en/stream_scheduler.md).

The checked-in browser page forces a TCP TURN relay so the same setup works through VS Code Remote SSH. The complete
local development stack therefore has four processes: coturn, LiveKit Server, TeleFuser, and the browser page.
Install the LiveKit Server and your platform's `coturn` package once:

```bash
# Debian/Ubuntu; use the equivalent coturn package on other platforms.
sudo apt-get update
sudo apt-get install -y coturn

# Install LiveKit Server once.
curl -sSL https://get.livekit.io | bash
```

Then run each command below in a separate terminal from the repository root.

Terminal 1 — start the development-only TCP TURN relay:

```bash
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 \
  --listening-port=3478 --min-port=49160 --max-port=49200 \
  --user=livekit-demo:livekit-demo-password \
  --realm=livekit.local --fingerprint --lt-cred-mech \
  --no-tls --no-dtls --no-cli --allow-loopback-peers
```

Terminal 2 — start LiveKit with its development credentials (`devkey` / `secret`):

```bash
livekit-server --dev
```

Terminal 3 — load the four-GPU LingBot-World v2 service:

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --num-workers 1 --worker-gpu-map 0,1,2,3 \
  --max-sessions-per-worker 2 --control-idle-timeout 10 \
  --port 8088 --skip-validation
```

This is one four-GPU model worker and one loaded LingBot service instance, not four replicas. It can retain two
independent user sessions; the shared LingBot execution lease runs at most one session chunk at a time and yields at
a chunk boundary after the active controller becomes idle while another session waits.

Terminal 4 — serve the browser controller and proxy its session API:

```bash
python examples/stream_server/livekit_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 --port 8092 --no-open
```

For VS Code Remote SSH, forward remote TCP ports `8092`, `7880`, and `3478` to the same local ports; `8088` does not
need forwarding because the page proxies the TeleFuser API. Open `http://127.0.0.1:8092`, select an initial image,
click **Start**, and use the on-page controls or `W/A/S/D` and arrow keys. A successful connection shows a video
track plus `control_state`, generation-stage, and chunk status messages.

Check the server independently with `curl http://127.0.0.1:8088/v1/service/health`. To stop the stack, stop the
browser session or close the page first, then press Ctrl+C in terminals 4, 3, 2, and 1. These loopback addresses,
static credentials, disabled TURN TLS, and `--allow-loopback-peers` are for trusted development only. See the
[stream server guide](docs/en/stream_server.md) for LiveKit Cloud, production networking, session APIs, and
troubleshooting.

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

1. **Access layer**: FastAPI task APIs and LiveKit-backed stream room/session entrypoints.
2. **Service layer**: request routing, task management, stream sessions, replica pools, and integration with pipeline execution.
3. **Pipeline abstraction layer**: model-specific `BasePipeline` / `BaseStage` components, with an actor-based streaming orchestrator for bounded dataflow, session ordering, metrics, and cleanup.
4. **Model and optimization layer**: model loading, attention selection, quantization, offload, LoRA, and cache integration.
5. **Execution backend layer**: optimized ops, Triton kernels, and device-specific implementations.

Relevant directories:

```text
telefuser/
├── service/         # REST APIs and LiveKit-backed streaming
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
| `LingBot-World v2` | Bidirectional world-model streaming | LiveKit control loop via [examples/lingbot/lingbot_world_v2_image_to_video_h100.py](examples/lingbot/lingbot_world_v2_image_to_video_h100.py) |
| `LiveAct` | S2V | Speech-driven talking head generation via [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py) |
| `FlashVSR` | VSR | Streaming video super-resolution via [examples/flashvsr/README.md](examples/flashvsr/README.md) |

### Video Generation

| Pipeline | Task | Notes |
|----------|------|-------|
| `WanVideo` (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | Main video generation family, including async and service examples in [examples/wan_video/README.md](examples/wan_video/README.md) |
| `HunyuanVideo` | T2V, I2V | Supported via [examples/hunyuan_video/README.md](examples/hunyuan_video/README.md) |
| `LTX Video` | I2V + Audio | Unified audio-video generation via [examples/ltx_video/README.md](examples/ltx_video/README.md) |
| `LongCat-Video` | T2V, I2V, VC | Long-form generation and continuation via [examples/longcat_video/README.md](examples/longcat_video/README.md) |
| **NEW** `LingBot-Video` | T2I, T2V, TI2V, MoE refiner | Dense/MoE generation with native CFG/SP and an in-memory base-to-refiner path; see [examples/lingbot_video/README.md](examples/lingbot_video/README.md) |

### Image Generation and Other Multimodal Pipelines

| Pipeline | Task | Notes |
|----------|------|-------|
| `Qwen-Image` | T2I, Edit | [examples/qwen_image/README.md](examples/qwen_image/README.md) |
| `Z-Image` | T2I | [examples/z_image/README.md](examples/z_image/README.md) |
| `Flux2 Klein` | T2I | [examples/flux2_klein/README.md](examples/flux2_klein/README.md) |

See [examples/README.md](examples/README.md) for the example runner and baseline comparison workflow.

## Documentation

- [docs/en/service.md](docs/en/service.md): REST serving, task APIs, OpenAI-compatible APIs
- [docs/en/stream_server.md](docs/en/stream_server.md): LiveKit streaming, session APIs, data topics, and deployment
- [docs/en/stream_scheduler.md](docs/en/stream_scheduler.md): actor-based stage scheduling, backpressure, lifecycle, metrics, and LingBot placement
- [docs/en/parallel.md](docs/en/parallel.md): distributed inference architecture
- [docs/en/communication.md](docs/en/communication.md): collectives, CUDA IPC, synchronization, and transport lifecycle
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
