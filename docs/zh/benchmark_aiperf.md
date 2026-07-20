# TeleFuser 与 AIPerf

TeleFuser 只暴露目标侧原始事实；AIPerf 统一负责 workload 执行、指标聚合、资源采集、产物、GreptimeDB
历史服务和前端展示。这样同一套 benchmark 与界面可以复用于 TeleFuser、SGLang-Diffusion 和后续实现。

当前资产覆盖：

- 通过 OpenAI 兼容 `/v1/videos` API 测试 Wan2.1 图生视频；
- 通过 WebRTC 与 DataChannel 测试 LingBot-World-Fast；
- 通过 WebSocket 与 MessagePack 测试 SGLang-Diffusion LingBot baseline。

## 仓库边界

```text
benchmarks/
├── telefuser_aiperf/                 # TeleFuser contract、配置、数据和启动器
├── baseline/sglang_lingbot_stream/   # Stream baseline
└── aiperf/                            # 被 Git 忽略的外部 AIPerf checkout
```

TeleFuser 不 vendoring AIPerf 实现。安装脚本与所有 launcher 都固定使用
`<TeleFuser>/benchmarks/aiperf`，不提供 checkout 路径覆盖。先安装
[uv](https://docs.astral.sh/uv/getting-started/installation/)，然后在 TeleFuser 仓库根目录执行一次：

```bash
bash scripts/setup_aiperf_repo.sh
```

脚本会 clone AIPerf、创建包含 WebRTC 支持的隔离运行环境，并创建用于 benchmark 输出与 History
导入的 `<TeleFuser>/artifacts`。前端产物已经内置，普通用户不需要安装 Node.js，也不需要单独启动
前端进程。正式实验应固定 AIPerf commit：

```bash
AIPERF_REF=<commit> bash scripts/setup_aiperf_repo.sh
```

`AIPERF_REPO_URL`、`AIPERF_BRANCH` 和 `AIPERF_REF` 只控制来源与 revision，不改变 checkout 位置。

## Batch 视频测试

启动固定 Wan2.1 I2V target：

```bash
telefuser serve \
  examples/wan_video/wan21_14b_image_to_video_480p_service.py \
  --port 8000 \
  --task i2v
```

执行快速测试或固定对比 workload：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_wan21_i2v_480p_compare.yaml
```

启动器会先检查 `/v1/service/health`。常用覆盖变量包括
`TELEFUSER_AIPERF_URL`、`TELEFUSER_AIPERF_CONCURRENCY`、
`TELEFUSER_AIPERF_REQUESTS`、`TELEFUSER_AIPERF_SIZE` 和
`TELEFUSER_AIPERF_SECONDS`。

## LingBot Stream 测试

启动 TeleFuser：

```bash
telefuser stream-serve \
  examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
  -p 8088 \
  --skip-validation
```

执行测试：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_compare.json
```

Stream 配置通过 `benchmark_metrics: true` 开启目标侧原始事实。TeleFuser 同步记录 runtime 创建、actor graph
chunk 计算、cache 几何和运行环境。生成工作位于子 actor 中，服务进程的 allocator 无法代表完整 actor graph，
因此不对 LingBot 上报不完整的 allocator 峰值；完整进程树显存曲线由 AIPerf 主动资源采集提供。原生 WebRTC
的编码位于 chunk fact 之后，因此当前不伪造独立编码耗时。AIPerf 负责跳过 warmup 并生成聚合结果。

TeleFuser 与 SGLang 共用
`benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json` 中的定时控制 trace。

## SGLang-Diffusion baseline

使用兼容且固定版本的 `sgl-project/sglang` 环境。TeleFuser 仓库不会在 import 时 monkeypatch SGLang
内部模块。

```bash
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_service.sh
bash benchmarks/baseline/sglang_lingbot_stream/scripts/run_stream_bench.sh
```

Baseline 固定 prompt、首帧、FPS、session 时长和控制 trace，只由 adapter 转换 transport 语义。正式性能
对比必须记录 SGLang commit 与模型 revision，使用 GPU-resident speed mode，并保持 offload、fallback、cache
和 attention 设置一致。某个配置 OOM 就应记录为该配置失败，不能用 mock 或 offload 结果替代。

本文两条启动命令默认都使用 1 张 GPU；比较其他卡数时必须同时显式覆盖两个 target。

## 配置清单

| 配置 | 用途 |
|---|---|
| `video_generation_quick.yaml` | Batch 连通性与延迟 smoke test |
| `video_generation_e2e.yaml` | Batch warmup、trace、records 和服务指标 |
| `video_generation_rate.yaml` | Poisson 到达负载 |
| `video_generation_wan21_i2v_480p_compare.yaml` | 固定 Wan I2V 对比 workload |
| `stream_lingbot_world_fast_quick.json` | 有界 Stream smoke test |
| `stream_lingbot_world_fast_compare.json` | 固定 LingBot Stream 对比 workload |

SGLang 对应配置位于 `benchmarks/baseline/sglang_lingbot_stream/configs`。

## 指标解释

必须区分指标 scope：

| 指标 | 含义 |
|---|---|
| `stream_fps` | 客户端收到帧数除以客户端 session 时间 |
| `chunk_compute_fps` | 单个目标 chunk 的帧数除以计算时间 |
| `chunk_compute_fps_weighted` | 排除 warmup 后的 `sum(frames) / sum(compute_seconds)` |

AIPerf 按交付、时延、吞吐、目标执行和资源五个稳定维度展示指标。不同实现的细分上报先保留为原始证据，
再映射到 canonical leaf，不扩张成新的顶层指标。

## 主动资源上报与历史曲线

使用 Docker 可以直接启动带持久化卷的 GreptimeDB：

```bash
docker volume create aiperf-greptime-data
docker run -d --name aiperf-greptime --restart unless-stopped \
  -p 127.0.0.1:4000:4000 \
  -v aiperf-greptime-data:/greptimedb_data \
  greptime/greptimedb:latest \
  standalone start \
  --http-addr 0.0.0.0:4000 \
  --data-home /greptimedb_data
```

生产环境应固定镜像 tag 或 digest。命名卷会在容器重启后保留历史数据。随后在 TeleFuser 仓库根目录
启动内置的 AIPerf 后端和中文前端；`artifacts` 与 benchmark launcher 的输出目录一致：

```bash
uv run --frozen --no-dev --project benchmarks/aiperf aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --host 127.0.0.1 \
  --port 8095
```

检查前后端和数据库：

```bash
curl --fail http://127.0.0.1:8095/api/v1/history/health
curl --fail -X POST 'http://127.0.0.1:4000/v1/sql?db=public' \
  --data-urlencode 'sql=SELECT 1 AS ready'
```

在 target 所在机器开启主动采集：

```bash
export AIPERF_HISTORY_URL=http://<history-host>:8095
export AIPERF_RESOURCE_TARGET_PID=<service-pid>

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

Agent 递归观测目标进程树，默认每 1 秒采样、每 15 秒上报，并在任务结束时 flush。它采集 CPU、内存、
GPU、显存、Ethernet 和 RDMA 的进程、可探测容器与整机事实。用量曲线与整机容量始终分开；CPU 以一个
逻辑核为 100%，多核和多卡允许超过 100%。

GreptimeDB 是 History 与主动上报的强依赖。启动、查询或最终 flush 失败会直接暴露，不会切换到 SQLite、
内存索引或文件直查。

打开 `http://127.0.0.1:8095/` 查看中文桌面界面。页面支持左右两组 Run、按 canonical 指标树选择图表、
同时展示 avg/P95/P99、资源时间折线和跨实验对比。

远端实验机建议保持默认 loopback 监听，并通过 SSH 安全转发：

```bash
ssh -L 8095:127.0.0.1:8095 user@benchmark-host
```

## 产物与复现要求

Batch 和 Stream 启动器默认将带时间戳的结果写入 `artifacts/`。Stream 产物包含 summary、session/event
JSONL、目标 metadata、normalized metrics 和独立 HTML 报告。

正式性能结果至少应保留：

- TeleFuser 或 SGLang commit 与模型 revision；
- GPU 型号/数量、driver、CUDA、PyTorch 和 dtype；
- workload 配置与 control trace；
- warmup 规则及成功/失败 session 数；
- offload、cache、attention 和 fallback 设置。

协议和职责边界见 [TeleFuser 与 AIPerf Benchmark 设计](benchmark_aiperf_design.md)。动态实验数值保存在
GreptimeDB 和可重放产物中，不写入稳定用户文档。
