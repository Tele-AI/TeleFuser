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

TeleFuser 是一个面向世界模型推理与多模态生成的高性能运行时框架。它重点服务于实时世界模型、语音驱动动画、流式视觉生成等连续、低时延、有状态的视觉生成任务。

## News 📰

- ✨ **2026-07-27**：统一使用 LiveKit 流式后端，支持 room 会话、worker 准入、浏览器自动重连，以及
  server-push 和 bidirectional 两种 pipeline contract。
- ✨ **2026-07-22**：**NEW** 新增 [**LingBot-Video**](examples/lingbot_video/README.md) 支持，覆盖 Dense/MoE T2I、T2V、TI2V、原生四卡 CFG/SP 推理与内存直传 MoE refiner。
- ✨ **2026-07-15**：新增 [**LingBot-World v2**](https://github.com/Robbyant/lingbot-world-v2) 支持，支持离线生成、交互式 WebRTC 流和多卡推理。

- ✨ **2026-07-06**：新增外部 **CacheSeek** latent cache 集成，支持服务模式下跨请求复用；命中后可跳过前 N 步去噪。Wan2.2 服务示例默认快照 `[5, 10, 15, 20, 25]`。配置和安装方式见 [docs/zh/latent_cache.md](docs/zh/latent_cache.md)。

## 为什么是 TeleFuser

大多数开源推理框架主要优化以下三类场景：

- 单次图像生成
- 离线视频生成
- 通用大语言模型服务

而实时世界模型需要的是另一种运行时能力：连续执行、流式输出、双向交互、会话状态保持、长上下文效率，以及并发场景下的稳定吞吐。TeleFuser 重点解决的正是这些问题。

在 TeleFuser 中，世界模型不只是“输入一次、返回一个视频”的函数，而是一个可以持续接收输入、保持状态、逐步产出结果的动态系统。

## TeleFuser 提供什么

- **面向世界模型的运行时**：支持连续视频生成、交互式会话和双向控制闭环。
- **ADF (AI Dev First)**：通过清晰的仓库分层、Pipeline Contract、示例和文档约束，让 AI Agent 能理解能力边界、遵循项目开发流程，并高效扩展 Pipeline。
- **流式 Pipeline 调度器**：基于 actor 管理有状态 Stage，提供有界 artifact edge、session 顺序、backpressure、生命周期清理和显式 resource group。
- **流式传输能力**：LiveKit-backed WebRTC 同时支持 server-push 媒体和稳定的双向会话，提供 room 生命周期、
  重连、参与者角色和可靠控制消息。
- **可扩展 GPU 运行时**：支持多 GPU、张量并行、序列并行、Ray 部署和分布式工作节点编排。
- **推理优化栈**：包含 Triton Kernel、优化注意力后端、量化、卸载、特征缓存和 CacheSeek latent cache 集成。
- **统一服务方式**：支持本地 Python 调用、任务 API `telefuser serve`，以及基于 LiveKit room/media 的
  `telefuser stream-serve`。

## 快速开始

### 安装

```bash
pip install -e .
```

开发环境安装：

```bash
pip install -e ".[dev]"
```

TeleFuser 不依赖 `tf-kernel` 也能运行。目前没有发布 tf-kernel 预编译包；该可选扩展仅支持使用
`tf-kernel/` 下的 Makefile 从源码构建。编译方法见 [tf-kernel README](tf-kernel/README_zh.md)，安装验证、
支持配置和常见问题见 [tf-kernel 安装与使用指南](docs/zh/tf_kernel.md)。

基础安装已包含 `telefuser stream-serve` 使用的 LiveKit Python SDK；LiveKit Cloud 项目或自托管 LiveKit
Server 需要单独运行。

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

### 2. 实时世界模型 WebRTC Demo

TeleFuser 通过 LiveKit 传输 `LingBot-World v2`。LingBot-World v2 使用相机控制和 v2 PPL 默认值；其流式
示例将单个会话上限设为两分钟。

LingBot 的离线与服务执行共用 actor scheduler。即使位于同一张 GPU，encode、DiT 和 decode 也可以重叠；
仅在显存放置需要时移动 Stage。详见[流式调度器指南](docs/zh/stream_scheduler.md)。

仓库内浏览器页面强制使用 TCP TURN relay，以便同一套配置可通过 VS Code Remote SSH 工作。因此完整的本地
开发环境包含四个进程：coturn、LiveKit Server、TeleFuser 和浏览器页面。先安装一次 LiveKit Server，并
通过操作系统的包管理器安装 `coturn`：

```bash
# Debian/Ubuntu；其他平台请安装对应的 coturn 软件包。
sudo apt-get update
sudo apt-get install -y coturn

# LiveKit Server 只需安装一次。
curl -sSL https://get.livekit.io | bash
```

然后从仓库根目录在四个独立终端中依次运行以下命令。

终端 1——启动仅供开发使用的 TCP TURN relay：

```bash
turnserver -n -m 1 \
  --listening-ip=127.0.0.1 --relay-ip=127.0.0.1 \
  --listening-port=3478 --min-port=49160 --max-port=49200 \
  --user=livekit-demo:livekit-demo-password \
  --realm=livekit.local --fingerprint --lt-cred-mech \
  --no-tls --no-dtls --no-cli --allow-loopback-peers
```

终端 2——使用开发凭据（`devkey` / `secret`）启动 LiveKit：

```bash
livekit-server --dev
```

终端 3——加载四卡 LingBot-World v2 服务：

```bash
TF_MODEL_ZOO_PATH=/path/to/model_zoo \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
  --livekit-url ws://127.0.0.1:7880 \
  --livekit-api-key devkey --livekit-api-secret secret \
  --num-workers 1 --worker-gpu-map 0,1,2,3 \
  --port 8088 --skip-validation
```

终端 4——启动浏览器控制页面及其 session API 代理：

```bash
python examples/stream_server/livekit_bidirectional_demo.py \
  --server-url http://127.0.0.1:8088 --port 8092 --no-open
```

使用 VS Code Remote SSH 时，把远端 TCP `8092`、`7880` 和 `3478` 映射到相同本地端口；页面会代理
TeleFuser API，因此无需映射 `8088`。打开 `http://127.0.0.1:8092`，选择初始图片，点击 **Start**，
再使用页面按钮或 `W/A/S/D` 和方向键控制相机。成功连接后会显示视频轨道以及 `control_state`、生成 Stage
和 chunk 状态消息。

可用 `curl http://127.0.0.1:8088/v1/service/health` 独立检查服务。停止时先结束浏览器 session 或关闭页面，
再按终端 4、3、2、1 的顺序按 Ctrl+C。Loopback 地址、静态凭据、禁用 TURN TLS 和
`--allow-loopback-peers` 仅适用于可信开发环境。LiveKit Cloud、生产网络、session API 和故障排查见
[流服务文档](docs/zh/stream_server.md)。

### 3. 批处理服务模式

```bash
telefuser serve examples/wan_video/wan22_14b_text_to_video_h100.py --task t2v --port 8000
```

TeleFuser 对外提供：

- 原生任务接口 `/v1/tasks/*`
- OpenAI 兼容图像与视频接口 `/v1/images` 和 `/v1/videos`
- 基于 Pipeline Contract 自动生成的服务元数据

完整 API 说明见 [docs/zh/service.md](docs/zh/service.md)。

## 架构

TeleFuser 采用分层运行时架构，并与仓库目录结构保持一致：

1. **接入层**：FastAPI 任务接口和 LiveKit-backed stream room/session 入口。
2. **服务层**：请求路由、任务管理、流式会话、副本池，以及与 Pipeline 执行过程的集成。
3. **Pipeline 抽象层**：模型相关的 `BasePipeline` / `BaseStage` 组件；actor-based streaming orchestrator 提供有界数据流、session 顺序、指标和清理。
4. **模型与优化层**：模型加载、注意力选择、量化、offload、LoRA、cache 集成。
5. **执行后端层**：优化算子、Triton Kernel 和设备相关实现。

关键目录：

```text
telefuser/
├── service/         # REST API 和 LiveKit-backed 流服务
├── orchestrator/    # 请求编排与基于 actor 的流式调度
├── pipelines/       # 模型级 Pipeline 实现
├── distributed/     # TP / SP / FSDP / Ray 等并行能力
├── feature_cache/   # AdaTaylorCache
├── ops/             # 面向 compile 的算子分发层
├── kernel/triton/   # Triton Kernel
└── models/          # DiT、VAE、编码器、解码器
```

## 已支持 Pipeline

### 世界模型与实时生成导向

| Pipeline | 任务 | 说明 |
|----------|------|------|
| `LingBot-World v2` | 双向世界模型流式推理 | LiveKit 控制闭环，见 [examples/lingbot/lingbot_world_v2_image_to_video_h100.py](examples/lingbot/lingbot_world_v2_image_to_video_h100.py) |
| `LiveAct` | S2V | 语音驱动数字人视频生成，见 [examples/liveact/liveact_s2v_h100.py](examples/liveact/liveact_s2v_h100.py) |
| `FlashVSR` | VSR | 流式视频超分，见 [examples/flashvsr/README.md](examples/flashvsr/README.md) |

### 视频生成

| Pipeline | 任务 | 说明 |
|----------|------|------|
| `WanVideo` (Wan2.1 / Wan2.2) | T2V, I2V, FL2V | 主力视频生成家族，含异步和服务示例，见 [examples/wan_video/README.md](examples/wan_video/README.md) |
| `HunyuanVideo` | T2V, I2V | 见 [examples/hunyuan_video/README.md](examples/hunyuan_video/README.md) |
| `LTX Video` | I2V + Audio | 统一音视频生成，见 [examples/ltx_video/README.md](examples/ltx_video/README.md) |
| `LongCat-Video` | T2V, I2V, VC | 长视频生成与续写，见 [examples/longcat_video/README.md](examples/longcat_video/README.md) |
| **NEW** `LingBot-Video` | T2I, T2V, TI2V, MoE refiner | 支持原生 CFG/SP 的 Dense/MoE 生成与内存直传 base-to-refiner，见 [examples/lingbot_video/README.md](examples/lingbot_video/README.md) |

### 图像与其他多模态生成

| Pipeline | 任务 | 说明 |
|----------|------|------|
| `Qwen-Image` | T2I, Edit | [examples/qwen_image/README.md](examples/qwen_image/README.md) |
| `Z-Image` | T2I | [examples/z_image/README.md](examples/z_image/README.md) |
| `Flux2 Klein` | T2I | [examples/flux2_klein/README.md](examples/flux2_klein/README.md) |

[examples/README.md](examples/README.md) 中提供了统一的 example runner 与 baseline 对比流程说明。

## 文档

- [docs/zh/service.md](docs/zh/service.md)：REST 服务、任务 API、OpenAI 兼容接口
- [docs/zh/stream_server.md](docs/zh/stream_server.md)：LiveKit 流服务、session API、data topic 和部署
- [docs/zh/stream_scheduler.md](docs/zh/stream_scheduler.md)：基于 actor 的 Stage 调度、backpressure、生命周期、指标和 LingBot 卡位
- [docs/zh/parallel.md](docs/zh/parallel.md)：分布式推理架构
- [docs/zh/latent_cache.md](docs/zh/latent_cache.md)：CacheSeek latent cache 集成
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
- `LingBot-World v2` 这类世界模型示例依赖外部权重和额外环境配置。
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
