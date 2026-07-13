# TeleFuser 与 AIPerf Benchmark

本文档是 TeleFuser 的 benchmark 执行入口，主要维护：

- 批处理视频服务的压测运行方式
- AIPerf 配置、参数和结果文件
- 本地与 baseline benchmark 的运行说明

更细的 benchmark 设计、接口边界和已实现资产清单放在：

- [当前 Benchmark 对比快照](benchmark_current_comparison.md)
- [TeleFuser 与 AIPerf Benchmark 设计](benchmark_aiperf_design.md)

## 目录结构

仓库中新增的 benchmark 资产位于：

```text
benchmarks/telefuser_aiperf/
├── benchmark_contract.yaml
├── stream_benchmark_contract.yaml
├── configs/
│   ├── video_generation_quick.yaml
│   ├── video_generation_e2e.yaml
│   ├── video_generation_rate.yaml
│   ├── stream_lingbot_world_fast_compare.json
│   ├── stream_lingbot_world_fast_quick.json
│   ├── stream_transport_mock_compare.json
│   └── stream_transport_mock_quick.json
├── data/
│   ├── video_prompts.jsonl
│   └── stream_lingbot_controls.json
└── scripts/
    ├── run_mock_webrtc_service.sh
    ├── run_video_bench.sh
    └── run_stream_bench.sh

benchmarks/baseline/sglang_lingbot_stream/
├── benchmark_contract.yaml
├── configs/
│   ├── stream_lingbot_world_fast_compare.json
│   ├── stream_lingbot_world_fast_quick.json
│   ├── stream_transport_mock_compare.json
│   └── stream_transport_mock_quick.json
└── scripts/
    ├── run_service.sh
    ├── run_mock_stream_service.sh
    └── run_stream_bench.sh
```

## 前置条件

### 1. 启动 TeleFuser 批处理服务

例如启动一个视频生成服务：

```bash
telefuser serve \
    examples/wan_video/wan21_14b_image_to_video_h100.py \
    --port 8000 \
    --task i2v
```

确认服务健康检查可访问：

```bash
curl http://<telefuser-batch-host>:8000/v1/service/health
```

### 2. 安装 AIPerf

TeleFuser 不直接 vendored AIPerf 源码。通过 setup 脚本拉取依赖仓库：

```bash
bash scripts/setup_aiperf_repo.sh
```

脚本默认使用 `https://github.com/ActivePeter/aiperf` 的 `teleai` 分支，并把 checkout 放在 `benchmarks/aiperf` 后做 editable install。

安装后确认 `aiperf` 命令可用：

```bash
aiperf --help
```

## 快速开始

### 批处理视频

运行一个最小视频 benchmark：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh
```

默认使用：

- 配置文件：`benchmarks/telefuser_aiperf/configs/video_generation_quick.yaml`
- 服务地址：`http://<telefuser-batch-host>:8000`
- prompt 文件：`benchmarks/telefuser_aiperf/data/video_prompts.jsonl`

脚本会先检查：

```bash
curl http://<telefuser-batch-host>:8000/v1/service/health
```

然后执行：

```bash
aiperf profile --config benchmarks/telefuser_aiperf/configs/video_generation_quick.yaml
```

### 流式世界模型

先启动 TeleFuser 流式服务：

```bash
telefuser stream-serve \
    examples/lingbot/stream_lingbot_world_fast.py \
    -p 8088 \
    --skip-validation
```

确认流式服务健康检查可访问：

```bash
curl http://<telefuser-stream-host>:8088/v1/service/health
```

然后运行最小流式 benchmark：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

默认使用：

- 配置文件：`benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json`
- 控制 trace：`benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`
- 服务地址：`http://<telefuser-stream-host>:8088`

这条路径由 AIPerf 原生 `profile --stream-config` 模式执行；TeleFuser 脚本只是薄启动器。等价的直接命令是：

```bash
uv run --project benchmarks/aiperf --extra streaming-webrtc \
  aiperf profile \
  --stream-config benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json
```

AIPerf 默认使用 `ice_host_ips: ["auto"]`，按到 stream target 的系统路由自动选择本机源地址，适用于常见多网卡主机。
容器、TURN 或特殊路由需要固定 candidate 时，设置逗号分隔的 `TELEFUSER_STREAM_BENCH_ICE_HOST_IPS`，
或直接向 AIPerf 重复传入 `--stream-ice-host-ip`；配置空列表才恢复全部地址枚举。

Quick/compare 配置启用 `benchmark_metrics: true`，除客户端首帧、FPS 和控制延迟外，还采集 pipeline/runtime 初始化、
逐 chunk compute/encode、分阶段 allocator peak、KV-cache/runtime 参数及 target 环境信息。`warmup_chunks` 按每个 measured
session 排除开头 chunk 后再计算 mean、p50、p90、std 和加权 compute FPS；raw chunk 仍保留在 `sessions.jsonl`。

### SGLang-Diffusion stream baseline

SGLang-Diffusion 对接的是 `sgl-project/sglang` 主仓库里的 `sglang.multimodal_gen` diffusion runtime，不依赖单独的 `SGLang-Diffusion` 仓库。

先启动 SGLang-Diffusion LingBot World 服务：

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
```

确认服务健康检查可访问：

```bash
curl http://<sglang-stream-host>:30000/health
```

然后运行 SGLang stream baseline：

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh
```

默认使用：

- 配置文件：`benchmarks/baseline/sglang_lingbot_stream/configs/stream_lingbot_world_fast_quick.json`
- 控制 trace：`benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`
- 服务地址：`http://<sglang-stream-host>:30000`

这条 baseline 使用 `robbyant/lingbot-world-fast-diffusers`，协议是 WebSocket + MessagePack。AIPerf adapter 将 `ArrowUp/ArrowDown/ArrowLeft/ArrowRight` 映射为 SGLang LingBot 的 `w/s/a/d` camera actions，结果写入和 TeleFuser stream benchmark 相同的 AIPerf streaming summary 结构。

AIPerf 同时直接归一化 SGLang 原生 `chunk_stats`：scheduler compute、request prepare、输出编码/pacing/write、chunk 总时长、帧数、batch 数及 raw/wire bytes 都进入统一的 `summary.json`、`sessions.jsonl` 和 `stream_report.html`。插桩后的 SGLang endpoint 还会上报已有的 reset-scoped `OutputBatch.peak_memory_mb`；由于它来自 `max_memory_reserved()`，AIPerf 只映射到 `chunk_peak_reserved_bytes`，不会伪造 allocated peak。初始化 phase 时长仍保持不可用。

正式 GPU-resident baseline 必须使用 `--performance-mode speed`。SGLang 的 `auto` 模式即使收到三个 `*-cpu-offload=false`，仍可能自动启用 VAE layerwise offload；这类结果只能单独标为 auto-offload，不能并入 TeleFuser 默认 GPU-resident 公平对比。

### 纯流式 mock 压测

这组 benchmark 只测 streaming 层，不加载底层生成模型。它适合回答：

- WebRTC offer/answer 和连接建立本身需要多久
- WebRTC media track 按固定 FPS 推帧时能否稳定
- DataChannel 控制消息的 ack 和下一帧反馈延迟
- WebSocket + MessagePack frame batch 协议在同样 session/count 下的开销

TeleFuser mock target 走 TeleFuser-compatible WebRTC mock service，服务端只生成合成视频帧：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_mock_webrtc_service.sh \
  --host <telefuser-stream-bind-host> \
  --port <telefuser-stream-port>

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_transport_mock_compare.json \
  --stream-server-url http://<telefuser-stream-host>:<telefuser-stream-port>
```

SGLang mock target 只启动兼容 `/health` 和 `/v1/realtime_video/generate` 的 WebSocket mock server，不调用 SGLang runtime 或任何模型：

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_mock_stream_service.sh \
  --host <sglang-stream-bind-host> \
  --port <sglang-stream-port>

bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh \
  benchmarks/baseline/sglang_lingbot_stream/configs/stream_transport_mock_compare.json \
  --stream-server-url http://<sglang-stream-host>:<sglang-stream-port>
```

默认 compare 配置使用：

- `session_count=4`
- `warmup_sessions=1`
- `session_duration_s=20`
- `fps=16`
- 同一份 `stream_lingbot_controls.json` 控制 trace
- TeleFuser mock：`320x180` 合成帧，经 WebRTC media track 发送
- SGLang mock：每个 `frame_batch` 携带固定大小二进制 payload，经 WebSocket + MessagePack 发送

这组结果只能说明流式协议、harness 和控制消息路径开销，不能说明真实模型推理速度。

## 可用配置

### `video_generation_quick.yaml`

适合快速联通性验证。

特点：

- 5 个请求
- 并发 1
- 不下载视频内容
- 只看最基本的端到端结果

使用：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_quick.yaml
```

### `video_generation_e2e.yaml`

适合更完整的 E2E benchmark。

特点：

- 1 个 warmup 请求
- 默认 profiling 并发 2
- 默认 profiling 请求数 6
- 开启 `downloadVideoContent: true`
- 开启 HTTP trace 导出
- 可抓取 TeleFuser `/v1/service/metrics`

使用：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_e2e.yaml
```

### `video_generation_rate.yaml`

适合看低频到中频请求率下的表现。

特点：

- `poisson` 到达模式
- 默认速率 `0.2 req/s`
- 默认最大并发 4
- 适合观察排队与时延波动

使用：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_rate.yaml
```

### `stream_lingbot_world_fast_quick.json`

适合流式世界模型最小联通和端到端时延验证。

特点：

- 1 个 session
- `WebRTC + DataChannel`
- 默认运行 `12s`
- 按固定 control trace 注入方向控制
- 输出 `offer RTT`、`首帧时延`、`稳态 FPS`、`control ack latency`、`control-to-next-frame latency`

使用：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json
```

## 常用环境变量

这些配置支持通过环境变量覆盖：

```bash
export TELEFUSER_AIPERF_URL=http://<telefuser-batch-host>:8000
export TELEFUSER_AIPERF_METRICS_URL=http://<telefuser-batch-host>:8000/v1/service/metrics
export TELEFUSER_AIPERF_CONCURRENCY=4
export TELEFUSER_AIPERF_REQUESTS=12
export TELEFUSER_AIPERF_SIZE=1280x720
export TELEFUSER_AIPERF_SECONDS=4
export TELEFUSER_AIPERF_SERVER_METRICS=true
```

例如跑一个更高并发的完整 E2E：

```bash
export TELEFUSER_AIPERF_CONCURRENCY=4
export TELEFUSER_AIPERF_REQUESTS=12

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_e2e.yaml
```

流式 benchmark 也支持通过 CLI 覆盖，例如：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json \
  --session-count 1 \
  --session-duration-s 20 \
  --print-events
```

## 为什么没有启用 AIPerf 自带 readiness probe

当前这套配置没有使用 AIPerf 的 `waitForModelTimeout` / `waitForModelMode`。

原因是：

- TeleFuser 这里对接的是 `video_generation` endpoint
- AIPerf 的 readiness probe 更偏向 `chat` / `completions` / `embeddings`
- TeleFuser 当前没有稳定的 `/v1/models` 路径可用于这类 probe

因此这里改为更直接的做法：在 benchmark 脚本里先检查 TeleFuser 的 `/v1/service/health`。

## 指标解释

这套 benchmark 里最值得关注的通常是：

- `Request Latency`
  表示一次完整视频请求从提交到完成的端到端耗时。

- `Request Throughput`
  表示单位时间内完成的请求数。

- HTTP Trace 相关指标
  用于区分时间主要耗在：
  - 连接建立
  - 请求发送
  - 服务端等待
  - 响应接收

- TeleFuser 服务指标
  如果启用了 `serverMetrics`，可以同时对照：
  - 队列长度
  - 已创建 / 已完成 / 已失败任务数
  - GPU 指标

## 结果文件

结果默认写到：

```text
artifacts/telefuser_aiperf/
```

不同配置写入不同子目录，例如：

- `artifacts/telefuser_aiperf/video_quick`
- `artifacts/telefuser_aiperf/video_e2e`
- `artifacts/telefuser_aiperf/video_rate`
- `artifacts/telefuser_aiperf/stream_lingbot_quick/<timestamp>`
- `artifacts/telefuser_aiperf/stream_lingbot_compare/<timestamp>`
- `artifacts/telefuser_aiperf/stream_transport_mock_compare/<timestamp>`
- `artifacts/sglang_lingbot_stream/stream_lingbot_compare/<timestamp>`
- `artifacts/sglang_lingbot_stream/stream_transport_mock_compare/<timestamp>`

通常可以关注：

- summary JSON
- raw records JSONL
- server metrics JSON / CSV
- stream benchmark 的逐 session JSONL
- stream benchmark 的逐事件 JSONL
- stream benchmark 的 `target_metadata.json`，包含 target 启动 phase 和软件/硬件环境 snapshot
- stream benchmark 的 `stream_report.html`；这是 AIPerf 生成的跨系统统一查看界面，可直接用浏览器打开

## 历史指标服务（GreptimeDB）

跨任务历史存储、API 和前端统一由 AIPerf 提供。TeleFuser 与 SGLang 只生成标准产物，不在 target 仓库实现数据库 client、历史 API
或专用前端分支。

GreptimeDB 可用后，先导入已有的 profile 和 stream 产物：

```bash
uv run --project benchmarks/aiperf aiperf history ingest \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --artifact-root work_dirs/benchmarks
```

再启动只读 API 和内置 Vue 前端：

```bash
uv run --project benchmarks/aiperf aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --artifact-root work_dirs/benchmarks \
  --host 127.0.0.1 \
  --port 8095
```

打开 `http://127.0.0.1:8095/` 查看历史任务、跨 Run 指标曲线，以及单 Run 的 session、control、phase、chunk、timeslice、GPU、
Prometheus 和 normalized 指标。warmup 与 profiling 通过 `phase` 区分；客户端 `stream_fps` 与 target
`chunk_compute_fps` 保持为两个指标。

GreptimeDB 是强依赖。连接或建表失败会终止启动，运行期查询失败返回 HTTP 503；API 和 UI 不会切换到 SQLite、内存索引、文件直查
或缓存结果。JSON/JSONL 仍是可重放输入，但不属于在线查询路径。

完整部署说明见 [AIPerf Benchmark History Dashboard](https://github.com/ActivePeter/aiperf/blob/teleai/docs/tutorials/history-dashboard.md)，实现边界见
[AIPerf Benchmark History Service Design](https://github.com/ActivePeter/aiperf/blob/teleai/docs/dev/history-service-design.md) 和
[TeleFuser 与 AIPerf Benchmark 设计](benchmark_aiperf_design.md)。

## 实测记录

### 2026-07-09，Wan2.1 I2V 480P batch benchmark

运行环境：

- 机器：远端 H100 测试环境，具体机器信息已脱敏
- GPU：4 卡可见配置
- 模型：Wan2.1-I2V-14B-480P
- 服务入口：TeleFuser Wan2.1 I2V 480P 固定 workload 服务
- benchmark 配置：TeleFuser Wan2.1 I2V 480P compare 配置
- artifact：已归档，具体路径不在公开文档中暴露

结果：

| 指标 | 数值 |
| --- | ---: |
| warmup 请求数 | 1 |
| profiling 请求数 | 2 |
| profiling errors | 0 |
| benchmark duration | 325.49 s |
| request latency avg | 163084.80 ms |
| request latency p50 | 163084.80 ms |
| request latency p90 | 163373.28 ms |
| request latency p99 | 163438.19 ms |
| request latency min / max | 162724.20 / 163445.40 ms |
| request throughput | 0.00614 requests/s |

备注：该配置的 `server_metrics.enabled=false`，所以 AIPerf 没有导出 GPU telemetry；运行过程中 GPU 4/5/6/7 处于高负载。

### 2026-07-09，LingBot-World-Fast stream benchmark

运行环境：

- 机器：远端 H100 测试环境，具体机器信息已脱敏
- GPU：4 卡可见配置；LingBot-World-Fast 当前默认单卡 GPU-resident
- 模型：Wan2.2-I2V-A14B + LingBot-World-Fast
- 服务入口：TeleFuser LingBot stream 服务
- benchmark 配置：TeleFuser LingBot stream compare 配置
- artifact：已归档，具体路径不在公开文档中暴露

结果：

| 指标 | 数值 |
| --- | ---: |
| profile sessions | 1 / 1 |
| failed sessions | 0 |
| success rate | 100% |
| configured session duration | 90.0 s |
| actual session runtime | 40.15 s |
| frames received | 406 |
| offer RTT | 6783.97 ms |
| WebRTC connected latency | 14432.81 ms |
| first metadata latency | 14540.01 ms |
| first frame latency | 14598.90 ms |
| stream FPS | 16.05 |
| control ack latency avg / p50 / p90 / max | 8.88 / 2.11 / 22.56 / 42.69 ms |
| control-to-next-frame latency avg / p50 / p90 / max | 37.60 / 36.63 / 53.76 / 55.50 ms |

备注：

- 本次是单 session benchmark，没有并发；当前 `LingBotWorldFastService` 也只允许一个 active session。
- `stream_lingbot_world_fast_compare.json` 的 `session_duration_s=90` 是等待上限；本次服务在完成生成后发送 `done`，所以实际 session runtime 为 40.15 s。
- 运行结束后 stream 服务仍为 `stream_ready=true`，服务所在 GPU 常驻显存约 47 GB。

### 2026-07-09，SGLang-Diffusion LingBot-World-Fast 早期可用性记录

运行环境：

- 机器：远端 H100 测试环境，具体机器信息已脱敏
- GPU：单卡可见配置
- 模型：LingBot-World-Fast Diffusers layout
- 临时 Diffusers layout：已脱敏
- benchmark 配置：SGLang LingBot stream compare 配置
- artifact：已归档，具体路径不在公开文档中暴露

结果：

| 指标 | 数值 |
| --- | ---: |
| profile sessions | 1 / 1 |
| failed sessions | 0 |
| success rate | 100% |
| configured session duration | 90.0 s |
| actual session runtime | 27.68 s |
| frames received | 93 |
| offer RTT | 81.00 ms |
| WebSocket connected latency | 81.01 ms |
| first metadata latency | 9896.79 ms |
| first frame latency | 9896.79 ms |
| stream FPS | 5.18 |
| control ack latency avg / p50 / p90 / max | 6622.36 / 6622.36 / 6622.36 / 6622.36 ms |
| control-to-next-frame latency avg / p50 / p90 / max | 6621.65 / 6621.65 / 6621.65 / 6621.65 ms |

备注：

- 本次是单 session benchmark，没有并发。
- 该记录早于 `performance_mode=speed` 公平性约束，不作为当前 GPU-resident 正式对比表输入。
- `request_extra` 显式设置 `realtime_causal_sink_size=9` 和 `realtime_causal_kv_cache_num_frames=18`；未设置时 SGLang 默认 45 帧 causal KV cache，在单张 H100 上 runtime OOM。
- `SGLANG_LINGBOT_DISABLE_FLASHINFER_ROPE=1` 让 SGLang 使用 RoPE fallback。当前远端只有 pip CUDA 13 toolkit，`flashinfer` 首次 JIT RoPE kernel 会因为 CUDA compiler/header 不兼容失败。
- 服务启动后常驻显存约 48 GB；运行过程中日志记录 peak memory usage 约 73522 MB。

### 2026-07-13，SGLang 1xH100 GPU-resident 正式 baseline

运行条件：

- 单张 80GB H100，`performance_mode=speed`
- DiT、text encoder、VAE 均驻留 GPU，无 CPU 或 layerwise offload
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- warmup 1 session，profile 1 session，每个 session 8 chunks
- 远端 FlashInfer 的 CUDA compiler/header 不兼容，因此启用了已记录的 PyTorch RoPE fallback

同一服务得到两个必须分开解释的结果：

| Cache 配置 | 结果 | 解释 |
| --- | --- | --- |
| sink 9 / window 18 | warmup 0/1、profile 0/1，CUDA OOM | 进程显存约 79.16 GiB，不能作为成功性能样本 |
| sink 6 / window 9 | warmup 1/1、profile 1/1 | 单卡 GPU-resident tuned baseline |

6/9 tuned profile 指标：

| 指标 | 数值 |
| --- | ---: |
| frames / chunks | 93 / 8 |
| warmup 排除后的 chunks | 7 |
| target compute FPS | 5.1906 |
| client stream FPS | 5.6788 |
| first frame latency | 3076.18 ms |
| session runtime | 19.277 s |
| chunk compute mean / p90 | 2.3119 / 2.3362 s |
| chunk encode mean | 0.0551 s |
| chunk total mean | 2.3699 s |
| control ack mean | 3333.92 ms |
| peak reserved allocator | 76,919,341,056 bytes（73,356 MiB） |

成功产物位于 `work_dirs/benchmarks/sglang_lingbot_stream/h100_gpu_resident_speed_rope_fallback_peak_20260713/20260713_115343_a2eae617`；9/18 OOM 产物位于 `work_dirs/benchmarks/sglang_lingbot_stream/h100_gpu_resident_speed_official_cache_oom_20260713/20260713_115523_418b0f5e`。两者 cache 几何不同，不能合并成同配置对比结论。

### 2026-07-10，纯流式 mock benchmark

运行环境：

- 机器：本地/远端测试环境信息已脱敏
- 模型：不加载真实模型
- TeleFuser target：TeleFuser-compatible WebRTC mock service
- SGLang target：SGLang-style WebSocket + MessagePack mock service
- benchmark 配置：纯流式 mock compare 配置
- artifact：已归档，具体路径不在公开文档中暴露

运行口径：

- `session_count=4`
- `warmup_sessions=1`
- `session_duration_s=20`
- `fps=16`
- 同一份控制 trace

结果：

| 指标 | TeleFuser WebRTC mock | SGLang WebSocket mock |
| --- | ---: | ---: |
| profile sessions | 4 / 4 | 4 / 4 |
| failed sessions | 0 | 0 |
| offer RTT avg | 5009.97 ms | 1.99 ms |
| connected latency avg | 10518.66 ms | 2.00 ms |
| first frame latency avg | 10583.69 ms | 64.73 ms |
| stream FPS avg | 16.00 | 16.05 |
| frames received avg | 319 | 320 |
| control ack latency avg | 0.70 ms | 0.35 ms |
| control-to-next-frame latency avg | 25.63 ms | 21.93 ms |

备注：

- 这组 benchmark 不加载任何生成模型，只测 streaming / transport / control path。
- 两边 steady-state FPS 和 control-to-next-frame 都接近目标帧率下的帧间隔，说明纯流式层不是真实世界模型低 FPS 的主要瓶颈。
- TeleFuser mock 的首帧和 connected latency 主要受当前 WebRTC SDP/ICE 建连路径影响；它影响首帧，不代表模型推理慢。

### 2026-07-10，纯流式 mock 极限 FPS sweep

运行环境：

- 机器：本地/远端测试环境信息已脱敏
- 模型：不加载真实模型
- TeleFuser target：TeleFuser-compatible WebRTC mock service
- SGLang target：SGLang-style WebSocket + MessagePack mock service
- benchmark 配置：纯流式 mock 配置，逐次覆盖目标 FPS
- artifact：已归档，具体路径不在公开文档中暴露

运行口径：

- 单 session sweep，每个 target 单独运行
- `session_duration_s=6`
- TeleFuser mock 使用 `320x180` 合成视频帧
- SGLang mock 每个 `frame_batch` 携带固定大小二进制 payload
- 两边都不加载 LingBot、Wan 或 SGLang diffusion runtime

结果：

| Target FPS | TeleFuser WebRTC mock 实测 FPS | SGLang WebSocket mock 实测 FPS |
| ---: | ---: | ---: |
| `30` | `30.02` | `28.82` |
| `60` | `60.18` | `54.04` |
| `120` | `120.05` | `104.58` |
| `240` | `241.25` | `179.87` |
| `480` | `355.92` | `260.40` |
| `960` | `394.97` | `394.82` |
| `1920` | 未运行 | `350.64` |

补充指标：

| Target | TeleFuser first-frame | TeleFuser control ack | SGLang first-frame | SGLang control ack |
| ---: | ---: | ---: | ---: | ---: |
| `30` | `10635.24 ms` | `1.79 ms` | `275.56 ms` | `4.60 ms` |
| `60` | `10673.16 ms` | `8.73 ms` | `73.53 ms` | `3.56 ms` |
| `120` | `10990.64 ms` | `5.34 ms` | `52.43 ms` | `2.23 ms` |
| `240` | `10728.59 ms` | `0.97 ms` | `36.18 ms` | `4.29 ms` |
| `480` | `10708.52 ms` | `4.53 ms` | `91.68 ms` | `0.66 ms` |
| `960` | `10768.45 ms` | `2.95 ms` | `48.53 ms` | `3.01 ms` |
| `1920` | 未运行 | 未运行 | `60.51 ms` | `3.09 ms` |

备注：

- TeleFuser WebRTC mock 在 `240 FPS` 前基本贴住目标，`480/960 FPS` 后开始饱和，本次最高观测约 `395 FPS`。
- SGLang WebSocket mock 本次最高观测也约 `395 FPS`，出现在 `960 FPS` target；继续提高到 `1920 FPS` 后下降到约 `351 FPS`。
- TeleFuser first-frame 仍约 `10.6s-11.0s`，主要反映当前 WebRTC SDP/ICE 建连和 harness 等待路径，不是模型耗时。
- SGLang 这里统计的是 WebSocket + MessagePack `frame_batch`，不是 RTP video frame；这组 sweep 只能解释纯 streaming 层上限。
