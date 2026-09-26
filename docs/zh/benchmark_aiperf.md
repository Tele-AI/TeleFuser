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

## 四卡 H100 LingBot-World v2 验证

以下两次运行均使用 4 张 H100 80 GB、BF16 DiT、FP32 VAE、FlashAttention-4，关闭 FSDP 和
`torch.compile`，设置 `chunk_size=4` 并输出 16 FPS。两次运行的 workload 和代码版本不同，不能将结果
解释为优化前后的性能对比。

### 当前 77 帧实时计算门禁

2026-08-03 使用 PyTorch 2.11.0+cu128，通过 direct LingBot pipeline-service 路径验证了 commit
`540b579`。请求使用 832x480、77 帧，共生成 5 个、每个包含 4 个 latent frame 的 chunk。

| 指标 | 结果 |
|---|---:|
| 生成帧数 / target chunk | 77 / 5 |
| 排除 chunk 0 后的 steady chunk | 4 |
| Steady compute FPS | **17.1399** |
| Chunk compute mean / p50 / p90 / max | 0.9335 / 0.9409 / 0.9410 / 1.0058 秒 |
| 从计时 session 开始到首个生成帧 | 3.2182 秒 |

该运行通过了平均 target-side 16 FPS 计算门禁，但不表示每个 chunk 都低于一秒：最大值为 1.0058 秒。
设备同步后的 compute 区间包含 condition handling、DiT、clean-KV update、空间 VAE decode、GPU-to-CPU
传输和 frame conversion；不包含模型加载、runtime creation、LiveKit pacing/encoding、网络交付和客户端渲染。

复现命令见 [LingBot 示例文档](https://github.com/Tele-AI/TeleFuser/tree/main/examples/lingbot#validated-four-h100-real-time-gate)，
使用 `tools/validation/benchmark_lingbot_world_v2_direct.py`。

### 当前一分钟流式回放

2026-08-03 在 TeleFuser commit `284996dd616cfd44a55523687b7f2a63a281abb9` 上重新运行了一分钟 workload，
用于验证当前通信优化版本的持续 target 生成、固定 KV cache 容量和带 pacing 的 LiveKit 交付路径。

运行使用 `stream_lingbot_world_v2_1min.json` workload，以及 commit
`e977ffbb1648510acec431b2a3fbd1a0f7bb8a35` 对应的 AIPerf 0.11.0。60 秒请求按完整 latent chunk 截断为
60 个 chunk、957 帧，对应 59.75 秒媒体时长。使用 `local_attn_size=18` 和 `sink_size=6` 时，240 个
latent frame 的 session 报告固定 KV 容量为 28,080 token。

| 指标 | 结果 |
|---|---:|
| Target 生成帧数 / chunk | 957 / 60 |
| 排除 chunk 0 后的 steady 帧数 / chunk | 944 / 59 |
| Steady target compute 时间 / FPS | 58.2791 秒 / **16.1979** |
| Chunk compute mean / p50 / p90 / p99 / max | 0.9878 / 0.9593 / 1.0624 / 1.0932 / 1.1149 秒 |
| LiveKit stream FPS / 客户端帧数 | 13.1967 / 803 |
| 客户端首帧 / session runtime | 6.0682 / 66.8948 秒 |
| Runtime creation | 1.4176 秒 |
| Artifact | `20260803_095518_62ec043c` |

Target 完成全部 60 个 chunk，平均 compute 通过 16 FPS 门禁，但并非每个 chunk 都低于一秒：p99 为
1.0932 秒，最大值为 1.1149 秒。客户端帧数较低属于带 pacing 的交付测量，不能与 target 生成完整性混为一谈。

## 复现要求

每个结果都应保留 TeleFuser commit、AIPerf 包版本、模型 revision、加速器型号/数量、driver、CUDA、
PyTorch、dtype、完整 workload、warmup 规则、成功/失败数量，以及 offload/cache/attention 设置。上面的日期化
实测只代表单次验证，不是通用性能保证；持续对比结果应保存在 GreptimeDB 和可重放产物中。
OOM 是被测配置的结果，不能用 mock 或改变 offload 策略后的结果替代。
## TeleFuser 性能回归基线

AIPerf 继续负责 workload 执行、warmup 排除和原始结果聚合。TeleFuser 在此之上增加一个带版本的基线文件，
统一比较 T2I、T2V、VLA 和实时流四类 workload：

- 首帧延迟 `first_frame_latency_ms.p95`
- chunk 延迟 `p50/p95/p99`（毫秒）
- 稳态 FPS `steady_fps.p50`
- GPU 峰值显存 `gpu_peak_memory_mb.maximum`
- 多请求吞吐 `throughput_per_second.p50`
- 请求成功率
- 输出一致性（精确匹配率、最大绝对误差、最小余弦相似度）

实现位于 `telefuser.metrics.baseline`，不依赖 GPU 或 AIPerf，可以在 CPU CI 中生成和比较基线。它接受标准化
AIPerf record，也兼容实时流已有的 `stream_fps`、`chunk_measurements` 和 `first_frame_latency_s` 字段：

适配器如果保留参考输出，可以先调用 `compare_numeric_outputs()` 或 `compare_output_hashes()` 生成一致性字段，
再构造 `BenchmarkRecord`；基线产物不会写入完整输出数组。

```bash
python -m telefuser.metrics.baseline summarize \
  --input artifacts/telefuser_aiperf/run/records.json \
  --output artifacts/telefuser_aiperf/run/baseline.json \
  --project-commit "$(git rev-parse HEAD)"
```

使用候选结果进行回归检查：

```bash
python -m telefuser.metrics.baseline compare \
  --baseline baselines/lingbot_world_v2.json \
  --candidate artifacts/telefuser_aiperf/run/baseline.json \
  --output artifacts/telefuser_aiperf/run/regression.json
```

默认相对容差为 5%。延迟和显存升高、FPS 和吞吐下降会触发失败；缺少指标或 workload 也会失败。输出一致性单独
检查，适配器提供精确匹配结果时要求全部匹配，同时要求余弦相似度至少 `0.9999`、最大绝对误差不超过 `1e-4`。
GPU 数据采集和 workload 执行仍由现有 AIPerf adapter 负责，因此本层测试不运行 GPU。
