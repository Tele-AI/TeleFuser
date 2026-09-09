# 服务与 API

TeleFuser 提供两种生命周期和传输契约不同的服务模式。应根据工作负载选择模式，而不是把它们视为可互换的
服务配置。

## 选择服务模式

| 工作负载 | 命令 | 传输 | 生命周期 |
| --- | --- | --- | --- |
| 图像、视频、音视频、修复或 VLA 任务 | `telefuser serve` | HTTP 请求和轮询 | 有限任务 |
| 连续生成或交互式世界模型 | `telefuser stream-serve` | HTTP 控制及 LiveKit 媒体/数据 | 有状态会话 |

有限生成任务请使用[批量服务与 API 参考](service.md)，其中包含任务 API、OpenAI 兼容的图像和视频接口、
Python 客户端、流水线副本、指标和错误响应。

server-push 和双向会话请使用[流式服务指南](stream_server.md)，其中包含 LiveKit room 角色、worker 准入、
GPU 卡位、重连行为和生产网络配置。

## 批量服务契约

```bash
telefuser serve /path/to/pipeline.py --task t2v --port 8000
```

启动后应读取服务实际暴露的契约，不要假设所有流水线接受相同参数：

| 接口 | 用途 |
| --- | --- |
| `/docs` | Swagger UI |
| `/openapi.json` | 机器可读 HTTP schema |
| `/v1/service/health` | 存活检查 |
| `/v1/service/metadata` | 已加载流水线和参数契约 |
| `/v1/tasks/create` | 提交有限任务 |
| `/v1/tasks/{task_id}/status` | 轮询任务状态 |

## 流式服务契约

`telefuser stream-serve` 管理常驻模型 worker 和会话准入。LiveKit 承载媒体及可靠控制消息；HTTP API
负责创建、查询和删除会话。批量任务 ID 与流式会话 ID 的所有权和清理语义不同。

## 相关指南

- [LingBot-World v2 WebRTC 核心体验](streaming_quickstart.md)验证交互式浏览器链路。
- [基础推理快速上手](quickstart.md)完成单机和批量服务请求。
- [服务元数据](service_metadata.md)定义自省字段。
- [流式调度器](stream_scheduler.md)解释 actor 所有权和有界数据流。
- [运维与参考](operations_reference.md)覆盖可观测性和诊断。
