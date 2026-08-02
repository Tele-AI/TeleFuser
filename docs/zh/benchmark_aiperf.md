# TeleFuser 与 AIPerf

TeleFuser 只暴露目标侧原始事实；AIPerf 负责 workload 执行、聚合、资源采集、产物、GreptimeDB 历史服务和
展示。仓库内集成覆盖 OpenAI 兼容 `/v1/videos` API 的 batch 视频生成，以及通过 LiveKit 执行的 LingBot
streaming benchmark。

AIPerf stream runner 与结果 schema 不感知具体传输。LiveKit adapter 由 TeleFuser 维护，在进程启动时从源码
注册，并生成 AIPerf 标准 session result。Contract 将 WebRTC 记录为媒体 transport，将 LiveKit 记录为
provider；这既保留 SFU 拓扑信息，也无需在 AIPerf 中加入 LiveKit 代码。

安装方法、workload 配置、运行命令、历史服务和专项测试统一维护在
[`benchmarks/telefuser_aiperf/README.md`](https://github.com/Tele-AI/TeleFuser/tree/main/benchmarks/telefuser_aiperf#readme)。AIPerf 通过 `pip` 安装固定的 Git
commit，不保留 AIPerf checkout 或 adapter `pyproject.toml`。

## 快速开始

以下命令都在 TeleFuser 仓库根目录执行。先把支持 streaming 的 AIPerf Git commit 安装到独立环境：

```bash
bash scripts/setup_aiperf.sh
```

在终端 1 启动本地 LiveKit 开发服务器：

```bash
livekit-server --dev --bind 127.0.0.1
```

在终端 2 启动四卡 LingBot-World v2 target，并替换实际模型目录：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  TF_MODEL_ZOO_PATH=/path/to/model_zoo \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  telefuser stream-serve examples/lingbot/lingbot_world_v2_image_to_video_h100.py \
    --livekit-url ws://127.0.0.1:7880 \
    --livekit-api-key devkey \
    --livekit-api-secret secret \
    --num-workers 1 \
    --worker-gpu-map 0,1,2,3 \
    --port 8088 \
    --skip-validation
```

模型加载和 pipeline warmup 可能需要几分钟。等待健康接口同时出现 `"ready":true`、
`"workers_idle":1` 和 `"workers_failed":0`：

```bash
curl --noproxy '*' --fail --silent --show-error \
  http://127.0.0.1:8088/v1/service/health
```

空闲时 `"livekit_connected":false` 是正常状态。在终端 3 启动测试：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh
```

请求媒体时长为 59.75 秒。240 秒 active window 是超时上限；成功运行会在 target 发出完成状态后退出，
从准入起通常约 66 秒。成功输出为 `Stream profile sessions: 1/1 succeeded`，报告写入
`artifacts/telefuser_aiperf/stream_lingbot_v2_1min/`。

使用相同四卡 workload 测试 SGLang 时，先启动服务，再选择 SGLang 配置：

```bash
bash benchmarks/telefuser_aiperf/scripts/run_sglang_lingbot_world_v2_4gpu.sh

bash benchmarks/telefuser_aiperf/scripts/run_stream_bench.sh \
  benchmarks/telefuser_aiperf/configs/stream_sglang_lingbot_world_v2_4gpu_1min.json
```

## 职责与指标语义

| 组件 | 归属 | 职责 |
|---|---|---|
| TeleFuser runtime | TeleFuser | 输出同步的 phase、chunk、runtime、cache 和环境原始事实 |
| Batch target adapter | AIPerf | 将 `/v1/videos` HTTP 事件转为标准 request 时间线 |
| LiveKit 源码 adapter | TeleFuser | 将 room、track、status、metrics 和 control 事件转为 session result |
| SGLang 源码 adapter | TeleFuser | 将 MessagePack frame、chunk timing 和 camera event 转为 session result |
| 聚合与历史 | AIPerf | 负责 warmup、percentile、throughput、artifact、GreptimeDB 和展示 |
| Contract 与 workload | TeleFuser | 固定 target 能力、输入、设置和可复现启动命令 |

Target 原始事实遵守以下规则：

- Duration 使用单调时钟；跨进程样本同时保留源端 UTC 时间戳。
- CUDA phase 在开始和结束边界同步目标设备。
- 数值必须有限且非负；不可用值省略或使用 `null`，不能伪造为零。
- Memory 在原始协议中使用 bytes，只在展示层转换单位。
- Target 不排除 warmup、不计算 percentile，也不生成跨 run 结论。

| Scope | 示例 | 聚合规则 |
|---|---|---|
| Event | frame 或 response arrival | 保留事件时间线 |
| Request/session | first output、session latency | 对每个 request 或 session 独立计算 |
| Run | success rate、throughput、percentile | AIPerf 排除 warmup 后聚合 |

客户端交付、target pipeline residence、target phase time 和资源利用率保持为不同维度。无法等价的字段保留为
private 或 unavailable，不强行映射为同一指标。

## LingBot-World v2 一分钟回放实测

2026-08-02 使用 4 张 H100 80 GB 验证了 TeleFuser commit
`663c385b179012c5c3de613212d10e8e6eac5f5d` 和 `stream_lingbot_world_v2_1min.json` workload；AIPerf 为
0.11.0、commit `e977ffbb1648510acec431b2a3fbd1a0f7bb8a35`。当前 H100 example 使用 BF16 DiT、FP32 VAE、
FlashAttention-4，关闭 `torch.compile` 和 FSDP，`chunk_size=4`，输出 16 FPS。60 秒请求按完整 latent
chunk 截断为 60 个 chunk、957 帧，对应 59.75 秒媒体时长。LingBot-World v2 使用
`local_attn_size=18`、`sink_size=6`，本次 240 latent frame session 报告的固定 KV 容量为 28,080 token。

| 运行环境 / target | Compute FPS | Chunk mean / p99 | Stream FPS | 客户端帧数 | Artifact |
|---|---:|---:|---:|---:|---|
| TeleFuser `.venv`，torch cu128 | 16.191 | 0.988 / 1.099 s | 12.697 | 756 | `20260802_084922_d7ae0931` |
| TeleFuser `.venv-sglang`，torch cu130 | 15.897 | 1.006 / 1.208 s | 14.089 | 871 | `20260802_090301_af6c433c` |
| SGLang `.venv-sglang`，torch cu130 | 16.617 | 0.963 / 0.974 s | 16.772 | 957 | `20260801_104829_2320fd7f` |

三次运行均完成 60 个 target chunk、生成 957 帧。AIPerf 只排除 target chunk 0，稳态统计包含 59 个 chunk、
944 帧。对齐环境后的 TeleFuser 同步计算时间为 59.381647 秒，compute FPS 比 SGLang 低 4.33%。TeleFuser
cu130 结果比 cu128 低 1.81%，因此环境变化单独报告，不计作代码优化收益。对齐环境 TeleFuser 报告位于
`artifacts/telefuser_aiperf/stream_lingbot_v2_1min/20260802_090301_af6c433c/stream_report.html`。

Compute 对比不使用 `stream_fps`。TeleFuser 的 LiveKit 视频按实时 16 FPS pacing 发布；对齐环境运行中，
decoded-ready 到 publish start 平均 18.99 ms，paced publish 平均 941.66 ms，publish 完成到客户端 metadata
平均 2.10 ms。SGLang 使用无 pacing 的 WebSocket burst 输出，两者交付语义不等价，尽管两边都包含网络
传输和客户端解码。TeleFuser 首帧为 9.740 秒：session 创建 0.630 秒，其后连接 1.979 秒，连接到准入
3.206 秒，准入到客户端首帧 3.925 秒；最后一段中的 runtime creation 为 1.564 秒。

## 复现要求

每个结果都应保留 TeleFuser commit、AIPerf 包版本、模型 revision、加速器型号/数量、driver、CUDA、
PyTorch、dtype、完整 workload、warmup 规则、成功/失败数量，以及 offload/cache/attention 设置。上面的日期化
实测只代表单次验证，不是通用性能保证；持续对比结果应保存在 GreptimeDB 和可重放产物中。
OOM 是被测配置的结果，不能用 mock 或改变 offload 策略后的结果替代。
