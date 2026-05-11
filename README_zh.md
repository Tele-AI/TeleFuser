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

TeleFuser 是一个用于高效多模态生成模型推理的高性能框架，支持图像生成、视频生成和视频超分辨率管道。

## 概述

TeleFuser 旨在为大规模多模态生成模型提供优化的推理能力。该框架采用基于 Stage 的模块化管道架构设计，提供统一的 API 接口支持本地调用和服务器部署，并提供全面的优化技术。

## 特性

- **基于 Stage 的管道架构**：模块化的管道设计，由可复用的 Stage（文本编码、去噪、VAE 等）组成
- **统一 API 设计**：本地 Python 调用和服务器部署的一致 API 接口
- **异步管道调度**：Stage 级异步执行，支持并行组调度和共享资源锁机制
- **分布式并行**：基于 Ray 框架的多 GPU 处理，支持数据并行、张量并行和序列并行（Ulysses/Ring/USP）
- **量化支持**：FP8 量化，降低内存占用
- **LoRA 挂载**：运行时 LoRA 权重挂载，支持微调模型推理
- **自定义注意力实现**：针对不同硬件优化的注意力内核，包括 SageAttention、FlashAttention 和稀疏注意力变体
- **内存优化**：灵活的 CPU 卸载策略和智能权重缓存
- **特征缓存**：AdaTaylorCache 支持 Wan2.1/2.2 模型，采用预计算跳过步数和混合 Taylor-残差近似策略
- **REST API 服务器**：FastAPI 服务器，支持 OpenAI 兼容端点（`/v1/images`, `/v1/videos`）和原生 API（`/v1/tasks/*`）
- **HuggingFace 格式支持**：从 HuggingFace Hub 或本地 Diffusers 格式文件夹加载模型

## 安装

```bash
pip install -e .
```

开发环境安装：

```bash
pip install -e ".[dev]"
```

## 快速开始

### 基本用法

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline
import torch

# 从 HuggingFace Hub 加载
pipe = Wan21VideoPipeline.from_pretrained(
    model_id_or_path="Wan-AI/Wan2.1-T2V-1.3B",
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# 生成视频
video = pipe(
    prompt="一只猫在弹钢琴",
    num_frames=81,
    height=480,
    width=832,
)
```

### 示例

📁 **查看 `examples/` 目录获取完整示例：**

| 示例 | 说明 |
|---------|-------------|
| `examples/wan_video/wan21_1_3b_text_to_video_hf.py` | 使用 HuggingFace 格式的文生视频 |
| `examples/wan_video/wan21_14b_image_to_video_h100.py` | 图生视频 (I2V) |
| `examples/wan_video/wan21_1_3b_text_to_video_h100.py` | 手动加载模型的文生视频 |
| `examples/wan_video/wan22_14b_first_last_frame_to_video_h100.py` | 首尾帧到视频 (FL2V) |
| `examples/qwen_image/qwen_image_t2i_h100.py` | 文生图 |
| `examples/wan_video/async_wan22_14b_image_to_video_distill_h100.py` | 异步管道演示 |
| `examples/ltx_video/ltx23_22b_image_to_video_two_stage_h100.py` | LTX 2.3 两阶段音频视频生成 |
| `examples/flux2_klein/flux2_klein_text_to_image_h100.py` | Flux2 Klein 文生图 |
| `examples/liveact/liveact_s2v_h100.py` | LiveAct 语音驱动视频生成 |
| `examples/stream_server/stream_lingbot_world_fast.py` | Lingbot-World 动作控制实时生成

运行任意示例时使用 `--help` 查看可用选项：

```bash
python examples/wan_video/wan21_1_3b_text_to_video_hf.py --help
```

支持 `telefuser serve` 的 example 还可以通过 `PIPELINE_MANIFEST`、`PIPELINE_CONTRACT` 或对应的工厂函数声明
一份 pipeline contract。服务端会以它作为能力来源，判断支持哪些 task、需要哪些文件输入，以及对外暴露哪些用户参数，
而不是把内部 pipeline 调参项全部暴露出去。

📖 具体编写方式见 [添加新 Example](docs/zh/adding_new_example.md)，服务端消费方式见 [Service 文档](docs/zh/service.md)。

## 架构亮点

### 异步管道调度

TeleFuser 提供基于 Stage 的异步调度引擎，支持并发请求处理：

- **并行 Stage 组**：独立的 Stage（如文本编码和 VAE 编码）在并行组内并发执行
- **共享资源锁**：访问共享资源的 Stage 使用可配置的锁组防止资源争用
- **请求隔离**：每个请求维护独立的状态和队列
- **事件驱动执行**：整个管道采用非阻塞的 async/await 模式

异步使用示例：
```python
from telefuser.pipelines.wan_video import AsyncWan22VideoPipeline

pipe = AsyncWan22VideoPipeline(device="cuda")
pipe.init(module_manager, config)

async for event in pipe.agenerate(
    request_id="demo-1",
    prompt="一幅美丽的风景",
    input_image=image,
):
    if event["type"] == "stage_end":
        print(f"Stage {event['stage']['name']} 完成")
    elif event["type"] == "final":
        print(f"视频已保存至 {event['payload']['artifacts'][0]['uri']}")
```

## CLI 使用

TeleFuser 提供命令行接口以便快速访问功能：

```bash
# 启动 API 服务器
telefuser serve ./examples/wan_video/wan21_14b_image_to_video_h100.py --port 8000

# 支持多 GPU
telefuser serve ./examples/wan_video/wan21_14b_image_to_video_h100.py --task i2v --port 8000 --gpu-num 2

# 验证管道文件
telefuser validate /path/to/pipeline.py

# 扫描目录中的管道文件
telefuser scan /path/to/pipelines/
```

所有示例脚本都支持命令行参数。使用 `--help` 查看可用选项：

```bash
python examples/wan_video/wan21_1_3b_text_to_video_h100.py --help
```

📖 **详细 CLI 使用说明和 API 参考请参阅 [Service 文档](docs/zh/service.md)。**

## API 服务器

TeleFuser 提供基于 FastAPI 的 REST API 服务器，支持双 API 模式：

- **TeleFuser 原生 API** (`/v1/tasks/*`) - 异步任务管理
- **OpenAI 兼容 API** (`/v1/images`, `/v1/videos`) - 行业标准格式

```bash
# 启动服务器
telefuser serve /path/to/pipeline.py --port 8000
```

当 pipeline 文件提供 manifest/contract 时，`telefuser serve` 会把它作为以下内容的唯一事实来源：

- 支持的 task，例如 `t2v`、`i2v`、`fl2v`、`vc`
- 必需的上传输入，例如图片或视频
- 用户可见请求参数的默认值与必填约束
- `/v1/service/metadata` 返回的服务元数据

应当把用户可见参数放进 contract，把内部调参项继续保留在 `PPL_CONFIG` 或实现代码里。

📖 **详细 API 文档、客户端 SDK 和示例请参阅 [Service 文档](docs/zh/service.md)。**

## 配置

### 模型路径

在管道配置中配置模型路径：

```python
PPL_CONFIG = dict(
    dit_path=["/path/to/dit/model-*.safetensors"],
    vae_path=["/path/to/vae/model.safetensors"],
    text_encoder_path=["/path/to/text_encoder/model-*.safetensors"],
)
```

### HuggingFace 格式加载

直接从 HuggingFace Hub 或本地目录加载模型：

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline

pipe = Wan21VideoPipeline.from_pretrained(
    model_id_or_path="Wan-AI/Wan2.1-T2V-1.3B",  # HF Model ID 或本地路径
    device="cuda",
    torch_dtype=torch.bfloat16,
)
```

### 性能优化

- 在 H100 GPU 上使用 `attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90` 进行优化的注意力
- 启用 FP8 量化以减少内存使用
- 为多 GPU 设置配置并行处理

📖 **更多配置详情请参阅：**
- [模型加载指南](docs/zh/model_loading.md) - 使用 ModuleManager 加载模型
- [添加新模型](docs/zh/adding_new_model.md) - 新模型集成开发指南
- [Ops 模块文档](docs/zh/ops.md) - 神经网络算子实现
- [并行推理指南](docs/zh/parallel.md) - 分布式并行推理架构和使用方法
- [注意力实现指南](docs/zh/attention.md) - 注意力配置和后端
- [日志指南](docs/zh/logging.md) - 日志系统配置和使用
- [指标监控指南](docs/zh/metrics.md) - 指标收集和监控

## 已知限制

- **特征缓存**：目前支持 Wan2.1、Wan2.2、Qwen-Image 和 HunyuanVideo 模型
- **编译**：Torch compile 支持处于实验阶段，详见 [torch.compile 兼容性指南](docs/zh/torch_compile_compatibility.md)
- **多机**：跨多机分布式推理尚未测试
- **GPU 要求**：部分功能（FP8、SageAttention）需要特定 GPU 架构（H100+）
- **模型覆盖**：仅测试了选定模型；其他 Diffusers 模型可能需要适配

## 开发

我们采用 PEP8 作为代码风格。

快速设置：

```bash
pip install -e ".[dev]"
pre-commit install
```

📖 **详细贡献指南、测试文档和开发工作流请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。**

## 许可证

Apache 2.0 许可证 - 详见 LICENSE 文件。

## 贡献

欢迎贡献！请查看项目仓库了解贡献指南。

## 致谢

本项目建立并受益于许多优秀的开源项目：

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) - ModelScope 社区开发的扩散模型引擎
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine) - 高性能扩散模型推理引擎
- [LightX2V](https://github.com/ModelTC/LightX2V) - 轻量级图像/视频生成推理框架
- [cache-dit](https://github.com/vipshop/cache-dit) - 原生 PyTorch DiT 模型缓存库
- [Wan2.1](https://github.com/Wan-Video/Wan2.1) / [Wan2.2](https://github.com/Wan-Video/Wan2.2) - 视频生成模型
- [diffusers](https://github.com/huggingface/diffusers) - 最先进的扩散模型库
- [Qwen-Image](https://github.com/QwenLM/Qwen-Image) - 图像生成基础模型
- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) - 支持双语文本渲染的逼真图像生成模型
- [FlashVSR](https://github.com/OpenImagingLab/FlashVSR) - 实时扩散视频超分辨率框架

## 支持

如有问题和疑问，请查看项目文档或在仓库中创建问题。
