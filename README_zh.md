<div align="center">
  <img src="assets/telefuser_logo.png" width="80%">
</div>

<p align="center">
  中文 | <a href="README.md">English</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.8%2B-green" alt="CUDA">
</p>

TeleFuser 是一个面向世界模型推理与多模态生成的高性能运行时框架。它重点服务于实时世界模型、视觉交互式Agent、语音驱动动画、流式视觉生成等连续、低时延、有状态的视觉生成任务。

## 为什么是 TeleFuser

大多数开源推理框架主要优化以下三类场景：

- 单次图像生成
- 离线视频生成
- 通用大语言模型服务

而实时世界模型需要的是另一种运行时能力：连续执行、流式输出、双向交互、会话状态保持、长上下文效率，以及并发场景下的稳定吞吐。TeleFuser 重点解决的正是这些问题。

在 TeleFuser 中，世界模型不只是“输入一次、返回一个视频”的函数，而是一个可以持续接收输入、保持状态、逐步产出结果的动态系统。

## TeleFuser 提供什么

- **面向世界模型的运行时**：支持连续视频生成、交互式会话和双向控制闭环。
- **AI Dev First 接口**：Pipeline 可以通过 `PIPELINE_CONTRACT` / `PIPELINE_MANIFEST` 暴露机器可读的任务、输入和参数定义。
- **异步 Pipeline 调度**：基于 Stage 的执行模型，支持请求隔离、资源锁和并行 Stage 组。
- **流式传输能力**：基于 WebRTC 的媒体流传输，并结合 DataChannel 实现实时控制。
- **可扩展 GPU 运行时**：支持多 GPU、张量并行、序列并行、Ray 部署和分布式工作节点编排。
- **推理优化栈**：包含 Triton Kernel、优化注意力后端、量化、卸载和特征缓存。
- **统一服务方式**：既支持本地 Python 调用，也支持 `telefuser serve` 和 `telefuser stream-serve` 两种服务模式。

## 世界模型推理导向

TeleFuser 围绕世界模型在生产环境中的运行时需求来设计：

- **连续执行，而不是一次性推理**：边生成边返回，不必等待整个结果完成。
- **可交互控制**：会话运行过程中可以接收 prompt、控制信号、图像、音频或动作输入。
- **有状态会话**：跨 chunk 保留运行时状态，而不是每一步都重建整条 Pipeline。
- **低首帧时延**：通过异步调度和流式传输尽快返回部分结果。
- **长时程效率**：通过序列并行、offload、cache 等机制降低长视频和重复去噪的内存压力。

这些设计已经映射到当前仓库中的具体能力，包括：

- `LingBot-World-Fast` 的双向 WebRTC 会话
- `LiveAct` 的语音驱动视频生成
- `FlashVSR` 的流式视频超分
- `LongCat-Video` 的长视频与续写工作流
- `WanVideo`、`HunyuanVideo`、`LTX Video` 的批处理与异步视频生成能力

## 快速开始

### 安装

```bash
pip install -e .
```

开发环境安装：

```bash
pip install -e ".[dev]"
```

默认安装已通过 `aiortc` 包含 WebRTC 流式服务能力。

### 1. 批量视频推理

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline
import torch

pipe = Wan21VideoPipeline.from_pretrained(
    model_id_or_path="Wan-AI/Wan2.1-T2V-1.3B",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

video = pipe(
    prompt="一只猫在弹钢琴",
    num_frames=81,
    height=480,
    width=832,
)
```

### 2. 实时世界模型 Demo

TeleFuser 当前提供了 `LingBot-World-Fast` 的双向 WebRTC Demo。

```bash
export LINGBOT_WORLD_CHECKPOINT_DIR=/path/to/LingBot-World

telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py \
  -p 8088 \
  --skip-validation

python examples/stream_server/webrtc_bidirectional_demo.py \
  --server-url http://localhost:8088 \
  --image-path /path/to/input.png
```

该流程会启动一个持续运行的会话：客户端通过 WebRTC DataChannel 发送控制消息，服务端通过媒体轨道持续回传生成视频。

### 3. 批处理服务模式

```bash
telefuser serve examples/wan_video/wan22_14b_text_to_video_service.py --port 8000
```

TeleFuser 对外提供：

- 原生任务接口 `/v1/tasks/*`
- OpenAI 兼容图像与视频接口 `/v1/images` 和 `/v1/videos`
- 基于 Pipeline Contract 自动生成的服务元数据

完整 API 说明见 [docs/zh/service.md](docs/zh/service.md)。

## 服务模式

### `telefuser serve`

适用于请求-响应式推理服务，提供任务管理、标准 REST API 和服务元数据。

- 适合文生视频、图生视频、文生图、视频超分等批处理场景
- 支持通过 Pipeline Contract 对外暴露结构化参数
- 支持 OpenAI 兼容接口，便于接入现有客户端

### `telefuser stream-serve`

适用于连续流式生成场景。

- 支持服务端推送式 WebRTC 视频流
- 支持双向 WebRTC 交互控制
- 适合实时世界模型、语音驱动生成和流式媒体处理

流式协议说明见 [docs/zh/stream_server.md](docs/zh/stream_server.md)。

## AI Dev First 运行时

TeleFuser 的设计目标并不只是让人类开发者更容易写 Pipeline，也希望让 Agent 或自动化编排系统能够直接理解并调用这些能力。

- `PIPELINE_CONTRACT` 和 `PIPELINE_MANIFEST` 用来定义支持的 task、所需文件输入、默认值和用户可见参数
- 服务层会读取这些 contract，并对外暴露机器可读的能力信息
- 同一条 Pipeline 可以同时服务本地调用、REST API 和流式服务

这也是 TeleFuser “AI Dev First” 的核心方向：把运行时能力标准化，让上层调度系统不需要反向阅读内部实现，就能发现并使用 Pipeline。

## 架构

TeleFuser 采用分层运行时架构，并与仓库目录结构保持一致：

1. **接入层**：FastAPI 任务接口与 WebRTC 流式入口。
2. **服务与调度层**：请求路由、任务管理、流式会话和整体编排。
3. **Pipeline 抽象层**：基于 Stage 的 Pipeline，支持异步执行、请求隔离和资源锁。
4. **模型与优化层**：模型加载、注意力选择、量化、offload、LoRA、cache 集成。
5. **执行后端层**：优化算子、Triton Kernel 和设备相关实现。

关键目录：

```text
telefuser/
├── service/         # REST API、流式 API、WebRTC 集成
├── orchestrator/    # Pipeline 编排
├── pipelines/       # 模型级 Pipeline 实现
├── distributed/     # TP / SP / FSDP / Ray 等并行能力
├── feature_cache/   # AdaTaylorCache
├── ops/             # 面向 compile 的算子分发层
├── kernel/triton/   # Triton Kernel
└── models/          # DiT、VAE、编码器、解码器
```

## 运行时能力

- **异步 Pipeline 调度**：独立 Stage 可并行运行，共享资源通过锁组协调。
- **分布式推理**：支持张量并行、序列并行、Ray 多 GPU 部署和大规模推理编排。
- **注意力后端**：支持 Torch SDPA、FlashAttention、SageAttention、稀疏注意力等实现。
- **特征缓存**：`AdaTaylorCache` 可为已校准的扩散模型提供跳步与复用加速。
- **内存优化**：支持 CPU offload、权重复用和面向大视频模型的运行时加载策略。
- **量化能力**：在模型和后端支持的路径上提供 FP8 / INT8 相关能力。
- **流式输出**：通过 WebRTC 逐步输出视频帧，并支持可选音频轨道。

## 已支持 Pipeline

### 世界模型与实时生成导向

| Pipeline | 任务 | 说明 |
|----------|------|------|
| `LingBot-World-Fast` | 双向世界模型流式推理 | 交互式 WebRTC 控制闭环，见 [examples/lingbot/stream_lingbot_world_fast.py](examples/lingbot/stream_lingbot_world_fast.py) |
| `LiveAct` | S2V | 语音驱动数字人视频生成，见 [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py) |
| `FlashVSR` | VSR | 流式视频超分，见 [examples/flashvsr/README.md](examples/flashvsr/README.md) |
| `LongCat-Video` | T2V, I2V, VC | 长视频生成与续写，见 [examples/longcat_video/README.md](examples/longcat_video/README.md) |

### 视频生成

| Pipeline | 任务 | 说明 |
|----------|------|------|
| `WanVideo` (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | 主力视频生成家族，含异步和服务示例，见 [examples/wan_video/README.md](examples/wan_video/README.md) |
| `HunyuanVideo` | T2V, I2V | 见 [examples/hunyuan_video/README.md](examples/hunyuan_video/README.md) |
| `LTX Video` | I2V + Audio | 统一音视频生成，见 [examples/ltx_video/README.md](examples/ltx_video/README.md) |

### 图像与其他多模态生成

| Pipeline | 任务 | 说明 |
|----------|------|------|
| `Qwen-Image` | T2I, Edit | [examples/qwen_image/README.md](examples/qwen_image/README.md) |
| `Z-Image` | T2I | [examples/z_image/README.md](examples/z_image/README.md) |
| `Flux2 Klein` | T2I | [examples/flux2_klein/README.md](examples/flux2_klein/README.md) |

## 示例入口

重点示例：

- [examples/wan_video/README.md](examples/wan_video/README.md)
- [examples/longcat_video/README.md](examples/longcat_video/README.md)
- [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py)
- [examples/flashvsr/README.md](examples/flashvsr/README.md)
- [examples/ltx_video/README.md](examples/ltx_video/README.md)
- [examples/lingbot/stream_lingbot_world_fast.py](examples/lingbot/stream_lingbot_world_fast.py)
- [examples/stream_server/webrtc_bidirectional_demo.py](examples/stream_server/webrtc_bidirectional_demo.py)

查看某个示例的 CLI 参数：

```bash
python examples/wan_video/wan21_1_3b_text_to_video_hf.py --help
```

[examples/README.md](examples/README.md) 中还提供了统一的 example runner 与 baseline 对比流程说明。

## 文档

- [docs/zh/service.md](docs/zh/service.md)：REST 服务、任务 API、OpenAI 兼容接口
- [docs/zh/stream_server.md](docs/zh/stream_server.md)：连续流式推理与 WebRTC 协议
- [docs/zh/parallel.md](docs/zh/parallel.md)：分布式推理架构
- [docs/zh/feature_cache.md](docs/zh/feature_cache.md)：`AdaTaylorCache`
- [docs/zh/model_loading.md](docs/zh/model_loading.md)：模型加载方式
- [docs/zh/attention.md](docs/zh/attention.md)：注意力后端与配置
- [docs/zh/torch_compile_compatibility.md](docs/zh/torch_compile_compatibility.md)：`torch.compile` 相关约束
- [docs/zh/adding_new_model.md](docs/zh/adding_new_model.md)：新模型接入
- [docs/zh/adding_new_example.md](docs/zh/adding_new_example.md)：Example 与 Pipeline Contract 编写方式

## 已知限制

- `AdaTaylorCache` 目前只对部分模型家族提供了校准参数。
- `torch.compile` 在部分路径上仍处于实验阶段。
- 一些优化能力依赖特定 GPU 架构和 CUDA 环境。
- `LingBot-World-Fast` 这类世界模型示例依赖外部权重和额外环境配置。
- 多机部署在架构上已有支持，但实际落地通常还需要项目级集成与验证。

## 开发

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

贡献流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，项目内 Agent 约束见 [AGENTS.md](AGENTS.md)。

## 许可证

Apache 2.0，详见 [LICENSE](LICENSE)。

## 致谢

TeleFuser 建立在多模态生成与推理系统相关的开源工作之上，也受到了这些项目的启发，包括：

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)
- [LightX2V](https://github.com/ModelTC/LightX2V)
- [cache-dit](https://github.com/vipshop/cache-dit)
- [diffusers](https://github.com/huggingface/diffusers)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image)
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image)
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR)
