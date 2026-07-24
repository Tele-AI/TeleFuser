<section class="tf-hero" markdown>

# TeleFuser

A **high-performance runtime** for world model inference and multimodal generation, built for long-running
pipelines, distributed execution, and production service interfaces.

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
**World Model Runtime**

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

FastAPI batch serving plus WebRTC media tracks and DataChannel control.
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
| LingBot-World v2 | Bidirectional streaming | Camera-controlled interactive world model via WebRTC |
| LingBot-World-Fast | Bidirectional streaming | Legacy/causal-fast interactive world model via WebRTC DataChannel |

### Video Generation

| Model | Tasks | Description |
|-------|-------|-------------|
| WanVideo (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | Video generation and editing |
| HunyuanVideo | T2V, I2V | Video generation |
| LTX Video | I2V + Audio | Video generation with audio |
| FlashVSR | VSR | Video super-resolution |
| LiveAct | S2V | Speech-to-video |
| LongCat-Video | T2V, I2V | Long video generation |
| LingBot-Video | T2I, T2V, TI2V, MoE refiner | Precision-first Dense and MoE video generation |

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

# Stream serving (WebRTC support is included in the default install)
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py -p 8088
```

## Documentation Sections

<div class="tf-link-grid">
<a href="service/"><strong>Service Guide</strong><span>Batch serving, task APIs, and SDK.</span></a>
<a href="stream_server/"><strong>Stream Server</strong><span>WebRTC streaming and bidirectional control.</span></a>
<a href="stream_scheduler/"><strong>Stream Scheduler</strong><span>Actor ownership, bounded dataflow, lifecycle, metrics, and GPU placement.</span></a>
<a href="benchmark_aiperf/"><strong>AIPerf Benchmark</strong><span>Batch, stream, baseline, and history workflows.</span></a>
<a href="configuration/"><strong>Configuration</strong><span>Runtime, attention, quantization, and offload settings.</span></a>
<a href="tf_kernel/"><strong>TF-Kernel</strong><span>Install, build, verify, and use the optional CUDA extension.</span></a>
<a href="parallel/"><strong>Parallel Inference</strong><span>Distributed processing strategies.</span></a>
<a href="adding_new_model/"><strong>Adding New Model</strong><span>Integrate new model architectures and stages.</span></a>
<a href="profiler/"><strong>Profiler</strong><span>Performance analysis tools.</span></a>
</div>

---

[Switch to Chinese 🇨🇳](/TeleFuser/zh/)
