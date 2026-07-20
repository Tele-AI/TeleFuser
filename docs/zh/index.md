<section class="tf-hero" markdown>

# TeleFuser

一个面向世界模型推理和多模态生成的**高性能运行时**，覆盖长时运行流水线、分布式执行和生产服务接口。

<div class="tf-badge-row" markdown>
<span class="tf-badge">PyTorch 2.6+</span>
<span class="tf-badge">CUDA 12.8+</span>
<span class="tf-badge">Triton kernels</span>
<span class="tf-badge">FastAPI service</span>
<span class="tf-badge">Ray distributed</span>
</div>

</section>

## 运行时能力

<div class="feature-grid" markdown>
<div class="feature-card" markdown>
**世界模型运行时**

连续执行、有状态会话和双向控制循环。
</div>
<div class="feature-card" markdown>
**并行推理**

Ulysses、Ring Attention、张量并行、流水线并行和 FSDP。
</div>
<div class="feature-card" markdown>
**优化算子**

编译感知 ops，支持 eager CUDA Triton 内核和 PyTorch 原生回退。
</div>
<div class="feature-card" markdown>
**流式服务**

FastAPI 批量服务，以及 WebRTC 媒体轨道和 DataChannel 控制。
</div>
<div class="feature-card" markdown>
**特征缓存**

AdaTaylorCache 和运行时缓存控制，面向重复生成工作负载。
</div>
<div class="feature-card" markdown>
**可扩展流水线**

可复用阶段、模型配置、调度器和流水线编排。
</div>
</div>

## 支持的模型

### 世界模型和实时推理

| 模型 | 任务 | 描述 |
|------|------|------|
| LingBot-World-Fast | 双向流式推理 | 通过 WebRTC DataChannel 的交互式世界模型 |

### 视频生成

| 模型 | 任务 | 描述 |
|------|------|------|
| WanVideo (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | 视频生成和编辑 |
| HunyuanVideo | T2V, I2V | 视频生成 |
| LTX Video | I2V + Audio | 视频生成 + 音频 |
| FlashVSR | VSR | 视频超分辨率 |
| LiveAct | S2V | 语音转视频 |
| LongCat-Video | T2V, I2V | 长视频生成 |

### 图像生成

| 模型 | 任务 | 描述 |
|------|------|------|
| Qwen-Image | T2I, Edit | 图像生成和编辑 |
| Z-Image | T2I | 图像生成 |
| Flux2 Klein | T2I | 图像生成 |

## 快速开始

```bash
# 安装
pip install telefuser

# 批量服务
telefuser serve /path/to/pipeline.py --port 8000

# 流式服务（默认安装已包含 WebRTC 支持）
telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py -p 8088
```

## 文档分区

<div class="tf-link-grid">
<a href="service/"><strong>服务指南</strong><span>批量服务、任务 API 和 SDK。</span></a>
<a href="stream_server/"><strong>流式服务</strong><span>WebRTC 流式传输和双向控制。</span></a>
<a href="stream_scheduler/"><strong>流式调度器</strong><span>Actor 所有权、有界数据流、生命周期、指标和 GPU 卡位。</span></a>
<a href="benchmark_aiperf/"><strong>AIPerf 基准测试</strong><span>Batch、stream、baseline 与历史指标工作流。</span></a>
<a href="configuration/"><strong>配置</strong><span>运行时、注意力、量化和卸载配置。</span></a>
<a href="parallel/"><strong>并行推理</strong><span>分布式处理策略。</span></a>
<a href="adding_new_model/"><strong>新增模型</strong><span>集成新的模型架构和阶段。</span></a>
<a href="profiler/"><strong>性能分析</strong><span>性能分析工具。</span></a>
</div>

---

[切换到英文 🇬🇧](/TeleFuser/)
