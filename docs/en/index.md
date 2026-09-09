---
title: "TeleFuser: World Model Streaming and Multimodal Inference"
description: >-
  TeleFuser is an open-source streaming inference and serving framework for real-time world models,
  multimodal generation, and distributed video generation.
---

<section class="tf-hero" markdown>

# TeleFuser

An open-source **streaming inference and serving framework** for real-time world models and multimodal generation,
built for continuous pipelines, distributed GPU execution, and production service interfaces.

<div class="tf-badge-row" markdown>
<span class="tf-badge">PyTorch 2.6+</span>
<span class="tf-badge">CUDA 12.8+</span>
<span class="tf-badge">Triton kernels</span>
<span class="tf-badge">FastAPI service</span>
<span class="tf-badge">Ray distributed</span>
</div>

</section>

## Runtime Capabilities

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**[World Model Runtime](world_model_streaming_inference.md)**

Continuous execution, stateful sessions, and bidirectional control loops.
</div>
<div class="feature-card" markdown>
**Parallel Inference**

Ulysses, Ring Attention, tensor parallelism, pipeline parallelism, and FSDP.
</div>
<div class="feature-card" markdown>
**Optimized Operators**

Compile-aware ops with eager CUDA Triton kernels and PyTorch native fallbacks.
</div>
<div class="feature-card" markdown>
**Streaming Service**

FastAPI batch serving and LiveKit-backed rooms for server-push and resilient interactive WebRTC.
</div>
<div class="feature-card" markdown>
**Feature Cache**

AdaTaylorCache and runtime cache controls for repeated generation workloads.
</div>
<div class="feature-card" markdown>
**Extensible Pipelines**

Reusable stages, model configs, schedulers, and pipeline orchestration.
</div>
</div>

## Supported Models

### World Model and Real-Time

| Model | Tasks | Description |
|-------|-------|-------------|
| [LingBot-World v2](/TeleFuser/cookbook/lingbot-world/) | Bidirectional streaming | Camera-controlled interactive world model via LiveKit |
| [LingBot-World-Fast](/TeleFuser/cookbook/lingbot-world/) | Bidirectional streaming | Legacy/causal-fast model via LiveKit reliable data messages |
| [ABot-World-0-5B-LF](/TeleFuser/cookbook/abot-world/) | Single-GPU interactive generation | Direct HTTP or LiveKit browser control with persistent causal KV state |

### Video Generation

| Model | Tasks | Description |
|-------|-------|-------------|
| [WanVideo (Wan2.1 / Wan2.2)](/TeleFuser/cookbook/wan-video/) | T2V, I2V, FL2V | Video generation and editing |
| [LTX Video](/TeleFuser/cookbook/ltx23/) | I2V + Audio | Video generation with audio |
| [LTX-2.5 Distilled](/TeleFuser/cookbook/ltx25-distilled/) | T2V, I2V + Audio | ModuleManager-backed six-stage pipeline with 1/2/4-H100 Ulysses SP |
| [MiniMax H3](/TeleFuser/cookbook/minimax-h3/) | T2VA, FL2VA, Ref2VA + Audio | Local 768p joint audio-video generation |
| [FlashVSR](/TeleFuser/cookbook/flashvsr/) | VSR | Video super-resolution |
| [SwiftVR](/TeleFuser/cookbook/swiftvr/) | Causal video restoration | Stateful restoration with BF16, compile, FP8Linear, Ulysses SP, and stage-parallel options |
| [LiveAct](/TeleFuser/cookbook/liveact/) | S2V | Speech-to-video |
| [LongCat-Video](/TeleFuser/cookbook/longcat-video/) | T2V, I2V | Long video generation |
| [LingBot-Video](/TeleFuser/cookbook/lingbot-video/) | T2I, T2V, TI2V, MoE refiner | Precision-first Dense and MoE video generation |

### Image Generation

| Model | Tasks | Description |
|-------|-------|-------------|
| [Qwen-Image](/TeleFuser/cookbook/qwen-image/) | T2I, Edit | Image generation and editing |
| [Z-Image](/TeleFuser/cookbook/z-image/) | T2I | Image generation |
| [Flux2 Klein](/TeleFuser/cookbook/flux2-klein/) | T2I | Image generation |

### Vision-Language-Action

| Model | Tasks | Description |
|-------|-------|-------------|
| [LingBot-VLA v2](/TeleFuser/cookbook/lingbot-vla-v2/) | Robot manipulation | Vision-language-action inference for supported robot profiles |

## Quick Start

```bash
# Install
pip install telefuser

# Batch serving
telefuser serve /path/to/pipeline.py --port 8000

# LiveKit-backed streaming (Python SDK included in the base install)
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  -p 8088
```

## Documentation Sections

<div class="tf-link-grid">
<a href="service/"><strong>Service Guide</strong><span>Batch serving, task APIs, and SDK.</span></a>
<a href="stream_server/"><strong>Stream Server</strong><span>LiveKit sessions, retained capacity, LingBot time slicing, and bidirectional control.</span></a>
<a href="stream_scheduler/"><strong>Stream Scheduler</strong><span>Actor ownership, bounded dataflow, lifecycle, metrics, and GPU placement.</span></a>
<a href="benchmark_aiperf/"><strong>AIPerf Benchmark</strong><span>Batch video and LingBot LiveKit workflows.</span></a>
<a href="blog/"><strong>Technical Blog</strong><span>Optimization design, profiling evidence, results, and related work.</span></a>
<a href="configuration/"><strong>Configuration</strong><span>Runtime, attention, quantization, and offload settings.</span></a>
<a href="tf_kernel/"><strong>TF-Kernel</strong><span>Install, build, verify, and use the optional CUDA extension.</span></a>
<a href="parallel/"><strong>Parallel Inference</strong><span>Distributed processing strategies.</span></a>
<a href="communication/"><strong>Communication Architecture</strong><span>NCCL collectives, CUDA IPC, ordering, and efficiency.</span></a>
<a href="adding_new_model/"><strong>Adding New Model</strong><span>Integrate new model architectures and stages.</span></a>
<a href="profiler/"><strong>Profiler</strong><span>Performance analysis tools.</span></a>
</div>

---

[Switch to Chinese 🇨🇳](/TeleFuser/zh/)
