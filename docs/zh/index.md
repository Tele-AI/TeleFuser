# TeleFuser

一个**高性能运行时**，用于世界模型推理和多模态生成。

## 特性

- 🌍 **世界模型运行时** — 连续执行、有状态会话、双向控制循环
- 🚀 **高性能** — 优化的 Triton 内核、特征缓存和多 GPU 推理
- 🎨 **多模态生成** — 图像/视频生成、超分辨率、语音转视频
- 📡 **流式传输** — WebRTC 媒体轨道 + DataChannel 实时推理
- 🔧 **灵活配置** — 注意力实现、并行策略、量化、卸载
- 📦 **可扩展** — 轻松添加新模型、阶段和流水线

## 支持的模型

### 世界模型和实时推理

| 模型 | 任务 | 描述 |
|------|------|------|
| LingBot-World-Fast | 双向流式推理 | 通过 WebRTC DataChannel 的交互式世界模型 |

### 视频生成

| 模型 | 任务 | 描述 |
|------|------|------|
| WanVideo (Wan2.1/2.2) | T2V, I2V, FL2V | 视频生成和编辑 |
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
telefuser stream-serve examples/lingbot/stream_lingbot_world_fast.py -p 8088
```

## 文档分区

- **[服务指南](service.md)** — 批量服务、任务 API 和 SDK
- **[流式服务](stream_server.md)** — WebRTC 流式传输和双向控制
- **[配置](configuration.md)** — 运行时和模型配置
- **[并行推理](parallel.md)** — 分布式处理策略
- **[新增模型](adding_new_model.md)** — 集成新模型
- **[性能分析](profiler.md)** — 性能分析工具

---

[切换到英文 🇬🇧](/TeleFuser/)
