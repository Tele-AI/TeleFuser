# TeleFuser 与 AIPerf

TeleFuser 只暴露目标侧原始事实；AIPerf 负责 workload 执行、聚合、资源采集、产物、GreptimeDB 历史服务和
展示。仓库内当前集成只覆盖通过 OpenAI 兼容 `/v1/videos` API 执行的 batch 视频生成。

随旧传输后端一起删除的内容包括 LingBot 直接 WebRTC adapter 和 SGLang 对比资产。AIPerf 目前
尚未集成 LiveKit benchmark adapter，因此本仓库不会把不受支持的 stream benchmark 标记为可运行。流服务
输出的 target compute 指标仍可供未来 LiveKit-aware benchmark client 使用。

## 仓库边界

```text
benchmarks/
├── telefuser_aiperf/   # Batch contract、配置、数据和 launcher
└── aiperf/             # 被 Git 忽略的外部 AIPerf checkout
```

TeleFuser 不 vendoring AIPerf。安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，在仓库
根目录执行：

```bash
bash scripts/setup_aiperf_repo.sh
```

脚本把 AIPerf clone 到 `benchmarks/aiperf`，安装其非开发运行环境，并创建 `artifacts/`。正式实验应固定
commit：

```bash
AIPERF_REF=<commit> bash scripts/setup_aiperf_repo.sh
```

`AIPERF_REPO_URL`、`AIPERF_BRANCH` 和 `AIPERF_REF` 可以选择来源与 revision，但不改变 checkout 位置。

## Batch 视频测试

启动固定 Wan2.1 I2V target：

```bash
telefuser serve \
  examples/wan_video/wan21_14b_image_to_video_480p_service.py \
  --port 8000 \
  --task i2v
```

执行 smoke profile 或固定对比 workload：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_wan21_i2v_480p_compare.yaml
```

Launcher 会先检查 `/v1/service/health`。常用覆盖变量包括 `TELEFUSER_AIPERF_URL`、
`TELEFUSER_AIPERF_CONCURRENCY`、`TELEFUSER_AIPERF_REQUESTS`、`TELEFUSER_AIPERF_SIZE` 和
`TELEFUSER_AIPERF_SECONDS`。

| 配置 | 用途 |
|---|---|
| `video_generation_quick.yaml` | 连通性和时延 smoke test |
| `video_generation_e2e.yaml` | Warmup、trace、records 和服务指标 |
| `video_generation_rate.yaml` | Poisson 到达负载 |
| `video_generation_wan21_i2v_480p_compare.yaml` | 固定 Wan I2V 对比 |

## 主动资源上报与历史曲线

启动持久化 GreptimeDB：

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

再启动 AIPerf history API 与内置 dashboard：

```bash
uv run --frozen --no-dev --project benchmarks/aiperf aiperf history serve \
  --greptime-url http://127.0.0.1:4000 \
  --greptime-database public \
  --artifact-root artifacts \
  --host 127.0.0.1 \
  --port 8095
```

如需采集目标进程树，在执行 batch launcher 前设置 `AIPERF_HISTORY_URL` 和
`AIPERF_RESOURCE_TARGET_PID`。History 与主动上报强依赖 GreptimeDB；失败时不会静默回退到内存或文件数据库。

## 复现要求

每个结果都应保留 TeleFuser/AIPerf commit、模型 revision、加速器型号/数量、driver、CUDA、PyTorch、dtype、
完整 workload、warmup 规则、成功/失败数量，以及 offload/cache/attention 设置。动态结果保存在 GreptimeDB
和可重放产物中，不写入稳定文档。

稳定职责与指标边界见 [AIPerf benchmark 设计](benchmark_aiperf_design.md)。
