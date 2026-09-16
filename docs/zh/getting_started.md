# 开始使用

根据目标选择最短路径。模型执行需要支持 CUDA 的系统；构建文档和运行 CPU 单元测试不需要 GPU。

## 核心路径

1. [安装 TeleFuser](installation.md) 并验证命令行入口。
2. 使用自包含的无模型预检[验证 WebRTC 链路](streaming_quickstart.md)。
3. 按同一指南运行 LingBot-World v2，验证浏览器中的交互式世界模型生成。

如需更轻量的单卡安装与批量 API 检查，请使用[基础推理](quickstart.md)。

## 选择工作流

| 目标 | 起点 |
| --- | --- |
| 在浏览器体验交互式 LingBot-World v2 | [WebRTC 核心体验](streaming_quickstart.md) |
| 直接从 Python 运行模型 | [基础推理](quickstart.md) |
| 通过 HTTP 提供批量图像或视频生成 | [服务与 API](serving.md) |
| 设计或运维流式部署 | [流式服务](stream_server.md) |
| 选择模型权重和示例 | [支持的模型](supported_models.md) |
| 排查安装或运行错误 | [故障排查](troubleshooting.md) |
| 集成新流水线 | [开发者指南](adding_new_model.md) |

## 范围

TeleFuser 覆盖离线生成、批量 HTTP 服务和有状态流式推理。WebRTC 预检用于分离传输问题与模型、GPU
问题，基础推理路径用于隔离软件包与批量服务配置。请根据需要验证的能力选择入口。
