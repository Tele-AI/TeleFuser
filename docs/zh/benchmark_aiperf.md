# TeleFuser 与 AIPerf Benchmark

本文档是 TeleFuser 的 benchmark 执行入口，主要维护：

- 批处理视频服务的压测运行方式
- AIPerf 配置、参数和结果文件
- 远程同步、部署和复现实验脚本

更细的 benchmark 设计、分层、对标 shortlist 和 `stream-serve` 规划放在：

- [TeleFuser 与 AIPerf Benchmark 设计](benchmark_aiperf_design.md)

## 目录结构

仓库中新增的 benchmark 资产位于：

```text
benchmarks/telefuser_aiperf/
├── README.md
├── configs/
│   ├── video_generation_quick.yaml
│   ├── video_generation_e2e.yaml
│   ├── video_generation_rate.yaml
│   └── stream_lingbot_world_fast_quick.json
├── data/
│   ├── video_prompts.jsonl
│   └── stream_lingbot_controls.json
└── scripts/
    ├── run_video_bench.sh
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
curl http://127.0.0.1:8000/v1/service/health
```

### 2. 安装 AIPerf

当前仓库已经 vendored 一份 AIPerf clone 到 `benchmarks/aiperf`，可直接本地安装：

```bash
pip install -e ./benchmarks/aiperf
```

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
- 服务地址：`http://127.0.0.1:8000`
- prompt 文件：`benchmarks/telefuser_aiperf/data/video_prompts.jsonl`

脚本会先检查：

```bash
curl http://127.0.0.1:8000/v1/service/health
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
curl http://127.0.0.1:8088/v1/service/health
```

然后运行最小流式 benchmark：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

默认使用：

- 配置文件：`benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json`
- 控制 trace：`benchmarks/telefuser_aiperf/data/stream_lingbot_controls.json`
- 服务地址：`http://127.0.0.1:8088`

如果远端机器有很多网卡或容器 veth，`run_stream_bench.py` 现在会默认自动探测本机的非 loopback 地址并用于 ICE host candidate 收敛。

只有在你明确知道要锁定某个地址时，才手动传 `--ice-host-ip` 或设置 `TELEFUSER_WEBRTC_ICE_HOST_IPS`。不要默认写 `127.0.0.1`，否则在远程主机 / 容器网络里很容易把 WebRTC 连接锁死在本机回环地址。

这条路径当前不是 AIPerf 原生 endpoint，而是 TeleFuser 侧独立的 WebRTC benchmark harness。

如果在远程环境执行，建议直接用：

```bash
python3 scripts/remote_bench_sync.py stream-bootstrap
```

它会同步代码、安装 WebRTC 依赖、自动探测可用的 `model_zoo` 根目录，并在远程仓库里写回一个非破坏性的 `model_zoo` 软链，方便后续直接起 `telefuser stream-serve`。

另外，`run_stream_bench.sh` 和 `run_video_bench.sh` 会默认把 open-file limit 提升到 `8192`，避免 AIPerf/WebRTC 在启动时撞到文件描述符上限；如果你的环境需要不同值，可以用 `TELEFUSER_BENCH_NOFILE_LIMIT` 覆盖。

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
export TELEFUSER_AIPERF_URL=http://127.0.0.1:8000
export TELEFUSER_AIPERF_METRICS_URL=http://127.0.0.1:8000/v1/service/metrics
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

## 远程一键同步与校验

为了方便把这套 benchmark 资产同步到远程测试机，仓库中新增了：

```bash
python3 scripts/remote_bench_sync.py --help
```

这个脚本默认面向当前测试机：

- `root@116.238.240.2`
- SSH 端口 `30724`
- 远程仓库路径 `/workspace/TeleFuser`
- TeleFuser 服务 Python `/root/venv/cu128/bin/python`
- 独立 benchmark Python `/root/venv/telefuser-bench/bin/python`

推荐模式是**分离部署**：

- 已经部署好的 TeleFuser 服务环境继续保留
- 额外创建一个独立的 benchmark venv，只安装 AIPerf 和压测依赖

这样可以避免 benchmark 依赖覆盖 TeleFuser 现有运行环境里的：

- `transformers`
- `protobuf`
- `pillow`
- `huggingface_hub`

如果你要跑 `telefuser stream-serve` 的世界模型流式 benchmark，推荐直接走一键自动化预备：

```bash
python3 scripts/remote_bench_sync.py stream-bootstrap
```

这条命令会自动完成：

- 同步流式 benchmark 资产
- 安装 AIPerf 和 `aiortc` / `opencv-python-headless`
- 安装 TeleFuser 到独立服务环境
- 校验 `examples/lingbot/stream_lingbot_world_fast.py`
- 校验 `model_zoo` 路径
- 打印远程 GPU 状态

如果你只想更新安装而不重新同步：

```bash
python3 scripts/remote_bench_sync.py install --install-webrtc --install-telefuser
```

如果目标机器变化，可以通过参数覆盖：

```bash
python3 scripts/remote_bench_sync.py \
  --host <host> \
  --port <port> \
  --user <user> \
  --remote-repo <remote_repo> \
  --remote-python <remote_service_python> \
  --remote-bench-python <remote_bench_python> \
  gpu-status
```

### 1. 查看远程 GPU 状态

```bash
python3 scripts/remote_bench_sync.py gpu-status
```

这个命令会通过 SSH 执行 `nvidia-smi`，输出：

- 每张卡的利用率
- 显存占用
- 当前活跃计算进程

### 2. 首次全量同步

```bash
python3 scripts/remote_bench_sync.py bootstrap
```

这个命令会把当前管理的 benchmark 资产全量同步到远程，包括：

- `benchmarks/aiperf`
- `benchmarks/telefuser_aiperf`
- `docs/en/benchmark_aiperf.md`
- `docs/zh/benchmark_aiperf.md`
- `docs/en/index.md`
- `docs/zh/index.md`
- `mkdocs.yml`

如果你还希望同步完成后直接安装并校验：

```bash
python3 scripts/remote_bench_sync.py bootstrap --install --verify
```

默认行为是：

- 创建独立 benchmark venv
- 在独立环境中安装 vendored `benchmarks/aiperf`
- 复用同一个 benchmark venv 运行流式 benchmark Python 脚本
- 不修改当前 TeleFuser 服务环境

如果你明确希望同时刷新 TeleFuser 服务环境：

```bash
python3 scripts/remote_bench_sync.py bootstrap --install --verify --install-telefuser
```

如果你想强制重建独立 benchmark venv：

```bash
python3 scripts/remote_bench_sync.py bootstrap --install --verify --recreate-bench-venv
```

如果当前只想演练命令而不真正执行：

```bash
python3 scripts/remote_bench_sync.py --dry-run bootstrap --install --verify
```

### 3. 后续增量同步

首次同步后，脚本会在本地写一个状态文件，用于记录上次已同步的文件哈希：

```text
~/.cache/telefuser/remote_bench_sync/<target>.json
```

后续直接运行：

```bash
python3 scripts/remote_bench_sync.py sync
```

脚本只会上传发生变化的受管文件。

如果你想看本次计划同步哪些文件：

```bash
python3 scripts/remote_bench_sync.py sync --print-files
```

如果你想忽略状态文件并强制重新全量上传：

```bash
python3 scripts/remote_bench_sync.py sync --full
```

### 4. 远程安装

```bash
python3 scripts/remote_bench_sync.py install
```

默认会在远程执行：

- 创建独立 venv：`/root/venv/telefuser-bench`
- `pip install -e /workspace/TeleFuser/benchmarks/aiperf`

也就是说，默认**不会**修改现有 TeleFuser 服务环境。

如果你确实要刷新 TeleFuser 服务环境：

```bash
python3 scripts/remote_bench_sync.py install --install-telefuser
```

如果你想强制重建独立 benchmark venv：

```bash
python3 scripts/remote_bench_sync.py install --recreate-bench-venv
```

如果你只想更新源码，不想拉依赖，可以使用：

```bash
python3 scripts/remote_bench_sync.py install --no-deps
```

如果你当前不想安装 AIPerf，只想做其它步骤：

```bash
python3 scripts/remote_bench_sync.py install --skip-aiperf
```

### 5. 远程校验

```bash
python3 scripts/remote_bench_sync.py verify
```

校验内容包括：

- benchmark 配置文件是否存在
- TeleFuser 服务环境中的 `import telefuser`
- TeleFuser 服务环境中的 `telefuser --help`
- 独立 benchmark 环境中的 `import aiperf`
- 独立 benchmark 环境中的 `aiperf --help`
- 独立 benchmark 环境中的 `python benchmarks/telefuser_aiperf/scripts/run_stream_bench.py --help`

注意：

- 如果你之前使用的是 `install --no-deps`，那么对应环境的 CLI 校验可能失败
- 如果你只想检查 benchmark 环境，可以使用：

```bash
python3 scripts/remote_bench_sync.py verify --skip-telefuser
```

### 6. 运行 benchmark

#### 6.1 一键跑 `Wan2.1-I2V-14B-480P` 对比

如果远程机器上的 benchmark 资产和隔离环境已经准备好，可以直接用一条命令顺序跑：

- `TeleFuser`
- `Diffusers` baseline

```bash
python3 scripts/remote_bench_sync.py batch-compare \
  --framework both \
  --gpu 0
```

这条命令会自动完成：

- 校验远程 `telefuser` / `aiperf` 环境
- 解析 TeleFuser batch 模型目录
- 对指定 GPU 执行 `gpu_burner.sh stop <gpu>`
- 检查该 GPU 是否真的空闲
- 拉起 TeleFuser 480P 服务
- 跑对齐后的 TeleFuser AIPerf config
- 停掉 TeleFuser 服务
- 拉起 Diffusers baseline 服务
- 跑对齐后的 Diffusers AIPerf config
- 输出两边最新 `summary.json` 路径和关键指标

默认行为是**非破坏性的**：

- 只会停止指定 GPU 上的 burner 占位
- 不会主动杀掉远程机器上已有的其它计算任务
- 如果 GPU 在停止 burner 之后仍然被真实任务占用，会直接失败

如果你希望它等待 GPU 空闲，而不是立即失败，可以显式给等待窗口：

```bash
python3 scripts/remote_bench_sync.py batch-compare \
  --framework both \
  --gpu 0 \
  --gpu-idle-timeout 600
```

如果你只想单独跑一边：

```bash
python3 scripts/remote_bench_sync.py batch-compare --framework telefuser --gpu 0
python3 scripts/remote_bench_sync.py batch-compare --framework diffusers --gpu 0
```

这条自动化入口当前固定使用：

- TeleFuser config:
  `benchmarks/telefuser_aiperf/configs/video_generation_wan21_i2v_480p_compare.yaml`
- Diffusers config:
  `benchmarks/baseline/diffusers_wan_i2v/configs/video_generation_compare.yaml`

如果远程环境还没准备好，先执行：

```bash
python3 scripts/remote_bench_sync.py sync
python3 scripts/remote_bench_sync.py install --install-telefuser
python3 scripts/remote_bench_sync.py verify
```

#### 6.2 手工跑 TeleFuser batch benchmark

当 TeleFuser 服务已经在远程机器上启动后，可以直接指定独立 benchmark 环境中的 `aiperf` 可执行文件：

```bash
export AIPERF_BIN=/root/venv/telefuser-bench/bin/aiperf

bash benchmarks/telefuser_aiperf/scripts/run_video_bench.sh \
  benchmarks/telefuser_aiperf/configs/video_generation_e2e.yaml
```

这样 benchmark 走独立环境，TeleFuser 服务继续走原来的运行环境，两边互不覆盖。

#### 6.3 手工跑流式 benchmark

流式 benchmark 则直接走独立环境里的 Python：

```bash
export TELEFUSER_STREAM_BENCH_PYTHON=/root/venv/telefuser-bench/bin/python

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_lingbot_world_fast_quick.json
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

通常可以关注：

- summary JSON
- raw records JSONL
- server metrics JSON / CSV
- stream benchmark 的逐 session JSONL
- stream benchmark 的逐事件 JSONL
