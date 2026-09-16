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
**[Parallel Inference](parallel.md)**

Ulysses, Ring Attention, tensor parallelism, pipeline parallelism, and FSDP.
</div>
<div class="feature-card" markdown>
**[Optimized Operators](ops.md)**

Compile-aware ops with eager CUDA Triton kernels and PyTorch native fallbacks.
</div>
<div class="feature-card" markdown>
**[Streaming Service](serving.md)**

FastAPI batch serving and LiveKit-backed rooms for server-push and resilient interactive WebRTC.
</div>
<div class="feature-card" markdown>
**[Feature Cache](feature_cache.md)**

AdaTaylorCache and runtime cache controls for repeated generation workloads.
</div>
<div class="feature-card" markdown>
**[Extensible Pipelines](adding_new_model.md)**

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

## Start Here

<div class="tf-link-grid">
<a href="streaming_quickstart/"><strong>Core WebRTC Experience</strong><span>Control LingBot-World v2 and receive generated video in the browser.</span></a>
<a href="installation/"><strong>Installation</strong><span>Install the package and verify CUDA availability.</span></a>
<a href="quickstart/"><strong>Basic Inference</strong><span>Run Wan2.1 1.3B locally and submit an HTTP task.</span></a>
<a href="supported_models/"><strong>Supported Models</strong><span>Select a model, checkpoint source, and validated profile.</span></a>
</div>

## Documentation Sections

<div class="tf-link-grid">
<a href="configuration/"><strong>Runtime and Optimization</strong><span>Configuration, parallelism, attention, caching, quantization, and offload.</span></a>
<a href="operations_reference/"><strong>Operations and Reference</strong><span>Metrics, logging, profiling, benchmarks, and troubleshooting.</span></a>
<a href="adding_new_model/"><strong>Developer Guide</strong><span>Integrate models, stages, examples, and public operations.</span></a>
<a href="blog/"><strong>Technical Blog</strong><span>Optimization design, profiling evidence, results, and related work.</span></a>
</div>

---

[Switch to Chinese 🇨🇳](/TeleFuser/zh/)
