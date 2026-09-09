# 运维与参考

流水线在单机模式正确运行后，再使用本分区。这里集中服务检查、指标、日志、性能分析、测试和按症状组织的
诊断信息，避免将它们混入首次运行教程。

## 查找参考信息

| 需求 | 参考文档 |
| --- | --- |
| 查询 CLI 命令和批量 HTTP 接口 | [批量服务与 API 参考](service.md) |
| 查询已加载流水线接受的参数 | [服务元数据](service_metadata.md) |
| 配置流式 worker 和 LiveKit 会话 | [流式服务](stream_server.md) |
| 解释运行时原始指标 | [监控指标](metrics.md) |
| 配置进程和请求日志 | [日志](logging.md) |
| 捕获阶段耗时和 GPU trace | [性能分析](profiler.md) |
| 复现批量与流式基准 | [TeleFuser 与 AIPerf](benchmark_aiperf.md) |
| 运行 CPU、GPU、分布式或回归测试 | [测试](testing.md) |
| 排查常见故障 | [故障排查](troubleshooting.md) |

## 运行时检查

对于批量服务，首先检查：

```bash
curl --fail http://127.0.0.1:8000/v1/service/health
curl --fail http://127.0.0.1:8000/v1/service/status
curl --fail http://127.0.0.1:8000/v1/service/metadata
curl --fail http://127.0.0.1:8000/v1/service/metrics
```

报告结果或故障时，应记录 TeleFuser commit、示例文件、权重 ID、GPU 型号、CUDA 和 PyTorch 版本、精度、
并行配置及完整命令。

## 参考信息原则

实际运行的 CLI `--help`、OpenAPI schema、Pydantic 服务 schema、配置 dataclass 和测试是 API 的事实来源。
叙述性文档负责解释这些契约，不应引入并行的配置项或环境变量。
