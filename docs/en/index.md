# TeleFuser

A **high-performance runtime** for world model inference and multimodal generation.

## Features

- 🌍 **World Model Runtime** — Continuous execution, stateful sessions, bidirectional control loops
- 🚀 **High Performance** — Optimized Triton kernels, feature caching, and multi-GPU inference
- 🎨 **Multimodal Generation** — Image/video generation, super-resolution, speech-to-video
- 📡 **Streaming Transport** — WebRTC with media tracks plus DataChannel for real-time inference
- 🔧 **Flexible Configuration** — Attention implementations, parallel strategies, quantization, offloading
- 📦 **Extensible** — Easy to add new models, stages, and pipelines

## Supported Models

### World Model and Real-Time

| Model | Tasks | Description |
|-------|-------|-------------|
| LingBot-World-Fast | Bidirectional streaming | Interactive world model via WebRTC DataChannel |

### Video Generation

| Model | Tasks | Description |
|-------|-------|-------------|
| WanVideo (Wan2.1/2.2) | T2V, I2V, FL2V | Video generation and editing |
| HunyuanVideo | T2V, I2V | Video generation |
| LTX Video | I2V + Audio | Video generation with audio |
| FlashVSR | VSR | Video super-resolution |
| LiveAct | S2V | Speech-to-video |
| LongCat-Video | T2V, I2V | Long video generation |

### Image Generation

| Model | Tasks | Description |
|-------|-------|-------------|
| Qwen-Image | T2I, Edit | Image generation and editing |
| Z-Image | T2I | Image generation |
| Flux2 Klein | T2I | Image generation |

## Quick Start

```bash
# Install
pip install telefuser

# Batch serving
telefuser serve /path/to/pipeline.py --port 8000

# Stream serving (requires WebRTC)
pip install -e ".[webrtc]"
telefuser stream-serve examples/stream_server/stream_lingbot_world_fast.py -p 8088
```

## Documentation Sections

- **[Service Guide](service.md)** — Batch serving, task APIs, and SDK
- **[Stream Server](stream_server.md)** — WebRTC streaming and bidirectional control
- **[Configuration](configuration.md)** — Runtime and model configuration
- **[Parallel Inference](parallel.md)** — Distributed processing strategies
- **[Adding New Model](adding_new_model.md)** — Integrate new models
- **[Profiler](profiler.md)** — Performance analysis tools

---

[Switch to Chinese 🇨🇳](/TeleFuser/zh/)
