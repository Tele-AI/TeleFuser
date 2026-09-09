---
title: TeleFuser 世界模型与多模态生成流式推理框架
description: >-
  TeleFuser 是开源的世界模型与多模态生成流式推理和服务框架，支持实时视频生成、有状态会话、
  多 GPU 分布式推理及 WebRTC 交互。
---

<section class="tf-hero" markdown>

# TeleFuser

一个面向实时世界模型和多模态生成的开源**流式推理与服务框架**，覆盖连续流水线、多 GPU 分布式执行和生产服务接口。

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
**[世界模型运行时](world_model_streaming_inference.md)**

连续执行、有状态会话和双向控制循环。
</div>
<div class="feature-card" markdown>
**[并行推理](parallel.md)**

Ulysses、Ring Attention、张量并行、流水线并行和 FSDP。
</div>
<div class="feature-card" markdown>
**[优化算子](ops.md)**

编译感知 ops，支持 eager CUDA Triton 内核和 PyTorch 原生回退。
</div>
<div class="feature-card" markdown>
**[流式服务](serving.md)**

FastAPI 批量服务，以及同时支持 server-push 和稳定交互式 WebRTC 的 LiveKit room。
</div>
<div class="feature-card" markdown>
**[特征缓存](feature_cache.md)**

AdaTaylorCache 和运行时缓存控制，面向重复生成工作负载。
</div>
<div class="feature-card" markdown>
**[可扩展流水线](adding_new_model.md)**

可复用阶段、模型配置、调度器和流水线编排。
</div>
</div>

## 支持的模型

### 世界模型和实时推理

| 模型 | 任务 | 描述 |
|------|------|------|
| [LingBot-World v2](/TeleFuser/zh/cookbook/lingbot-world/) | 双向流式推理 | 通过 LiveKit 进行相机控制的交互式世界模型 |
| [LingBot-World-Fast](/TeleFuser/zh/cookbook/lingbot-world/) | 双向流式推理 | 通过 LiveKit 可靠数据消息控制的 legacy/causal-fast 模型 |
| [ABot-World-0-5B-LF](/TeleFuser/zh/cookbook/abot-world/) | 单卡交互式生成 | 通过直接 HTTP 或 LiveKit 浏览器控制，并保持因果 KV 状态 |

### 视频生成

| 模型 | 任务 | 描述 |
|------|------|------|
| [WanVideo (Wan2.1 / Wan2.2)](/TeleFuser/zh/cookbook/wan-video/) | T2V, I2V, FL2V | 视频生成和编辑 |
| [LTX Video](/TeleFuser/zh/cookbook/ltx23/) | I2V + Audio | 视频生成 + 音频 |
| [LTX-2.5 Distilled](/TeleFuser/zh/cookbook/ltx25-distilled/) | T2V、I2V + Audio | 基于 ModuleManager 的六阶段流水线，支持 1/2/4 张 H100 的 Ulysses SP |
| [MiniMax H3](/TeleFuser/zh/cookbook/minimax-h3/) | T2VA, FL2VA, Ref2VA + Audio | 本地 768p 音视频联合生成 |
| [FlashVSR](/TeleFuser/zh/cookbook/flashvsr/) | VSR | 视频超分辨率 |
| [SwiftVR](/TeleFuser/zh/cookbook/swiftvr/) | 因果视频修复 | 支持 BF16、torch.compile、FP8Linear、Ulysses SP 和 stage-parallel |
| [LiveAct](/TeleFuser/zh/cookbook/liveact/) | S2V | 语音转视频 |
| [LongCat-Video](/TeleFuser/zh/cookbook/longcat-video/) | T2V, I2V | 长视频生成 |
| [LingBot-Video](/TeleFuser/zh/cookbook/lingbot-video/) | T2I, T2V, TI2V, MoE refiner | 精度优先的 Dense/MoE 视频生成 |

### 图像生成

| 模型 | 任务 | 描述 |
|------|------|------|
| [Qwen-Image](/TeleFuser/zh/cookbook/qwen-image/) | T2I, Edit | 图像生成和编辑 |
| [Z-Image](/TeleFuser/zh/cookbook/z-image/) | T2I | 图像生成 |
| [Flux2 Klein](/TeleFuser/zh/cookbook/flux2-klein/) | T2I | 图像生成 |

### 视觉语言动作

| 模型 | 任务 | 描述 |
|------|------|------|
| [LingBot-VLA v2](/TeleFuser/zh/cookbook/lingbot-vla-v2/) | 机器人操作 | 面向受支持机器人配置的视觉语言动作推理 |

## 从这里开始

<div class="tf-link-grid">
<a href="streaming_quickstart/"><strong>WebRTC 核心体验</strong><span>在浏览器控制 LingBot-World v2 并接收生成视频。</span></a>
<a href="installation/"><strong>安装</strong><span>安装软件包并验证 CUDA 可用性。</span></a>
<a href="quickstart/"><strong>基础推理</strong><span>在本地运行 Wan2.1 1.3B 并提交 HTTP 任务。</span></a>
<a href="supported_models/"><strong>支持的模型</strong><span>选择模型、权重来源和经过验证的运行配置。</span></a>
</div>

## 文档分区

<div class="tf-link-grid">
<a href="configuration/"><strong>运行时与优化</strong><span>配置、并行、注意力、缓存、量化和卸载。</span></a>
<a href="operations_reference/"><strong>运维与参考</strong><span>指标、日志、性能分析、基准和故障排查。</span></a>
<a href="adding_new_model/"><strong>开发者指南</strong><span>集成模型、阶段、示例和公共算子。</span></a>
<a href="blog/"><strong>技术博客</strong><span>优化设计、profiling 证据、效果与 Related Work。</span></a>
</div>

---

[切换到英文 🇬🇧](/TeleFuser/)
