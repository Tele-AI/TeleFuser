# 当前 Benchmark 对比快照

本文档汇总截至 `2026-07-13` 已经跑通并记录到文档中的 benchmark 结果。更完整的运行命令、环境和单次记录见 [TeleFuser 与 AIPerf Benchmark](benchmark_aiperf.md)，benchmark 设计和公平对比边界见 [TeleFuser 与 AIPerf Benchmark 设计](benchmark_aiperf_design.md)。

## 总览

| 模式 | Workload | TeleFuser 当前结果 | Baseline 当前结果 | 当前结论 |
| --- | --- | --- | --- | --- |
| Batch 视频生成 | Wan2.1-I2V-14B-480P，`832x480`，`81` 帧，`40` steps | 已完成，平均请求时延 `163.08s`，吞吐 `0.00614 requests/s` | Diffusers target 已落地，当前快照未记录实测结果 | 只能展示 TeleFuser 单边结果，暂不下性能对比结论 |
| 世界模型流式会话 | LingBot-World-Fast，`832x480`，目标 `16 FPS`，同一份控制 trace | 已完成，首帧 `14.60s`，稳态 `16.05 FPS`，控制到下一帧平均 `37.60ms` | SGLang-Diffusion 6/9 tuned 已完成，首帧 `3.08s`，stream `5.68 FPS`，控制到下一帧平均 `3329.12ms` | SGLang 正式 9/18 cache 在 1xH100 OOM；6/9 可运行结果因 cache 几何不同只作受限参考 |
| 纯流式 mock 会话 | 不加载模型，固定 FPS 合成帧，同一份控制 trace | 已完成，首帧 `10.58s`，稳态 `16.00 FPS`，控制到下一帧平均 `25.63ms` | 已完成，首帧 `64.73ms`，稳态 `16.05 FPS`，控制到下一帧平均 `21.93ms` | 稳态和控制闭环接近；差异主要在 WebRTC SDP/ICE 建连和首帧 |
| 纯流式 mock 极限 FPS sweep | 不加载模型，逐步提高目标 FPS，压测 transport/harness 上限 | 最高观测 `394.97 FPS`；`30/60/120/240 FPS` 基本贴住目标，`480/960 FPS` 开始饱和 | 最高观测 `394.82 FPS`；`1920 FPS` 目标下降到 `350.64 FPS` | 两边纯传输上限都在约 `395 FPS`；这不是模型推理 FPS |

注意：SGLang-Diffusion 6/9 tuned baseline 使用真正的 1-GPU GPU-resident `performance_mode=speed`，没有 CPU 或 layerwise offload；测试机仍需要 RoPE fallback，而且 causal KV cache 从 9/18 缩到 6/9。它证明了单卡可运行性并给出真实指标，但由于 cache 几何不同，不能直接计算 TeleFuser 与 SGLang 的严格公平性能差值。

## Batch 视频生成

当前已记录的是 TeleFuser 侧的 Wan2.1 I2V 480P benchmark。

| 项目 | 数值 |
| --- | --- |
| 运行环境 | 远端 H100 测试环境，具体机器信息已脱敏 |
| GPU | 4 卡可见配置 |
| 模型 | Wan2.1-I2V-14B-480P |
| 服务入口 | TeleFuser Wan2.1 I2V 480P 固定 workload 服务 |
| Benchmark 配置 | TeleFuser Wan2.1 I2V 480P compare 配置 |
| Artifact | 已归档，具体路径不在公开文档中暴露 |
| Warmup 请求数 | `1` |
| Profiling 请求数 | `2` |
| Profiling errors | `0` |
| Benchmark duration | `325.49s` |
| Request latency avg / p50 | `163084.80ms` / `163084.80ms` |
| Request latency p90 / p99 | `163373.28ms` / `163438.19ms` |
| Request latency min / max | `162724.20ms` / `163445.40ms` |
| Request throughput | `0.00614 requests/s` |

Diffusers batch baseline 的服务、contract 和 compare config 已经在仓库中，但当前结果快照没有记录 Diffusers 侧实测 summary。因此 batch 视频生成现在只能说明 TeleFuser 固定 workload 已跑通，不能给出 TeleFuser vs Diffusers 的速度结论。

## 世界模型流式会话

这组结果是单 session stream benchmark。两边使用同一类 LingBot world-model 实时控制语义、同 prompt/first frame/输出尺寸/FPS 目标/控制 trace；TeleFuser 使用 WebRTC + DataChannel，SGLang-Diffusion baseline 使用 WebSocket + MessagePack。

### 指标对比

| 指标 | TeleFuser stream | SGLang-Diffusion stream | 对比 |
| --- | ---: | ---: | --- |
| Profile sessions | `1 / 1` | `1 / 1` | 相同 |
| Failed sessions | `0` | `0` | 相同 |
| Configured session duration | `90.0s` | `90.0s` | 相同 |
| Actual session runtime | `40.15s` | `19.277s` | 两边都提前结束 |
| Frames received | `406` | `93` | TeleFuser 多 `4.37x` |
| Connection / offer latency | WebRTC connected `14432.81ms` | WebSocket connected `12.40ms` | SGLang 建连更轻 |
| First frame latency | `14598.90ms` | `3076.18ms` | SGLang tuned 首帧更快 |
| Stream FPS | `16.05` | `5.68` | 配置不同，不计算公平倍率 |
| Target compute FPS | 当前旧产物未上报 | `5.1906` | 只陈述 SGLang target 计算吞吐 |
| Control ack latency avg | `8.88ms` | `3333.92ms` | 语义不同，SGLang 为 chunk 采样近似 |
| Control-to-next-frame avg | `37.60ms` | `3329.12ms` | SGLang tuned 仍为秒级 |
| Chunk reserved peak max | 当前旧产物未上报 | `76,919,341,056 bytes` | SGLang reset-scoped `max_memory_reserved()` |

### 结果归档

| Target | 归档状态 |
| --- | --- |
| TeleFuser | 已归档，具体路径不在公开文档中暴露 |
| SGLang-Diffusion | 已归档，具体路径不在公开文档中暴露 |

### 当前解读

TeleFuser 在这组数据里不是首帧更快，而是稳态流式出帧和控制闭环更快。TeleFuser 首帧包含 WebRTC offer/ICE 和 stream runtime 初始化，首帧为 `14.60s`；SGLang WebSocket 建连更轻，6/9 tuned 首帧为 `3.08s`。

稳态阶段的瓶颈差异更明显：TeleFuser 能按目标播放节奏输出约 `16 FPS`，SGLang-Diffusion 6/9 tuned 每个 chunk 的模型 forward 成为主耗时，target compute 为 `5.1906 FPS`、客户端接收为 `5.68 FPS`。SGLang 日志中的 WebSocket 写入约毫秒级，主要瓶颈不在传输链路，而在 scheduler/model forward。由于两端 cache 几何未完全对齐，这里不把差值写成公平加速比。

控制链路也不同。TeleFuser 的控制通过 DataChannel 异步进入 session，ack 和下一帧反馈都在几十毫秒级；SGLang-Diffusion baseline 没有单独 control ack 消息，benchmark 用 chunk/frame batch 事件推导控制被采样和反映到下一帧的时间，因此当前观测到秒级控制闭环。

## 纯流式 mock 会话

这组 target 不加载 LingBot、Wan 或 SGLang diffusion runtime，只用合成帧和固定 payload 压测 streaming 层。下面是 `2026-07-10` 本地时区实测结果。

运行口径：

- `session_count=4`
- `warmup_sessions=1`
- `session_duration_s=20`
- `fps=16`
- 同一份控制 trace
- TeleFuser mock 走 WebRTC media track + DataChannel
- SGLang mock 走 WebSocket + MessagePack `frame_batch` / `chunk_stats`

| 指标 | TeleFuser WebRTC mock | SGLang WebSocket mock | 说明 |
| --- | ---: | ---: | --- |
| Profile sessions | `4 / 4` | `4 / 4` | 两边都成功 |
| Offer / connect latency avg | `5009.97ms` / `10518.66ms` | `1.99ms` / `2.00ms` | WebRTC 当前 harness 有 SDP/ICE 等待 |
| First frame latency avg | `10583.69ms` | `64.73ms` | 差异主要来自建连链路，不是模型 |
| Stream FPS avg | `16.00` | `16.05` | 稳态都达到目标 FPS |
| Frames received avg | `319` | `320` | 20s 窗口内基本一致 |
| Control ack latency avg | `0.70ms` | `0.35ms` | 都是毫秒级 |
| Control-to-next-frame avg | `25.63ms` | `21.93ms` | 都是帧间隔量级 |

这组结果说明：纯 streaming 层在稳态出帧和控制闭环上不是当前世界模型真实 benchmark 的主要瓶颈。真实 SGLang 6/9 tuned 世界模型约 `5.68 FPS`，而 mock 可以稳定到 `16 FPS`，因此真实慢点主要还在模型 forward / scheduler，而不是 WebSocket 写入本身。TeleFuser mock 的首帧和 connected latency 高，主要暴露的是当前 WebRTC offer/ICE 建连路径成本；这会影响首帧，但不影响 steady-state FPS。

### 极限 FPS sweep

这组 sweep 继续使用纯流式 mock，不加载任何真实生成模型。测试口径是让服务端 mock 尽可能快地按目标 FPS 推送合成帧或固定 payload，再由 benchmark harness 统计实际接收 FPS。

运行口径：

- 单 session sweep，每个 target 单独运行
- `session_duration_s=6`
- TeleFuser mock：`320x180` 合成视频帧，经 WebRTC media track 发送
- SGLang mock：每个 `frame_batch` 携带固定大小二进制 payload，经 WebSocket + MessagePack 发送
- 两边都不加载 LingBot、Wan 或 SGLang diffusion runtime

| Target FPS | TeleFuser WebRTC mock 实测 FPS | SGLang WebSocket mock 实测 FPS |
| ---: | ---: | ---: |
| `30` | `30.02` | `28.82` |
| `60` | `60.18` | `54.04` |
| `120` | `120.05` | `104.58` |
| `240` | `241.25` | `179.87` |
| `480` | `355.92` | `260.40` |
| `960` | `394.97` | `394.82` |
| `1920` | 未运行 | `350.64` |

结论：

- TeleFuser WebRTC mock 在 `240 FPS` 之前基本贴住目标；`480/960 FPS` 后本次测试环境下开始饱和，最高观测约 `395 FPS`。
- SGLang WebSocket mock 在 `960 FPS` 目标下达到本次最高观测约 `395 FPS`，`1920 FPS` 目标反而下降到约 `351 FPS`。
- TeleFuser mock 的 first-frame latency 在 sweep 中仍然约 `10.6s-11.0s`，主要来自 WebRTC SDP/ICE 建连和当前 harness 等待路径；这不代表 steady-state FPS 低。
- SGLang mock 的 first-frame latency 为几十到数百毫秒，但它统计的是 WebSocket `frame_batch` 消息，不是 RTP video frame。两边 transport 语义不同，因此这组结果用于判断流式层上限，不用于替代真实模型 benchmark。

## 延迟与控制实时性分析口径

流式 benchmark 的延迟不能只看一个总耗时，需要拆成“启动链路”和“稳态控制链路”两部分。首帧快不一定代表控制实时，控制 ack 快也不一定代表画面已经响应。

### 启动链路

| 指标 | 含义 | 主要判断 |
| --- | --- | --- |
| `offer_rtt_ms` | 客户端发起 offer/init 到服务端返回 answer/ready 的耗时 | HTTP/SDP 或 WebSocket init 开销 |
| `connected_latency_ms` | transport 建连完成耗时 | WebRTC ICE、DataChannel 或 WebSocket 建连成本 |
| `first_metadata_latency_ms` | 首个服务端 metadata、chunk stats 或状态事件到达 | 服务端是否开始产出 |
| `first_frame_latency_ms` | 客户端收到第一帧视频或第一批 frame payload | 用户可见首帧时间 |

当前数据里，SGLang WebSocket 建连更轻，所以 6/9 tuned 真实世界模型首帧为 `3.08s`，低于 TeleFuser 的 `14.60s`。但纯流式 mock 显示 TeleFuser 的首帧主要受 WebRTC SDP/ICE 建连和当前 harness 等待路径影响：mock first-frame 约 `10.58s`，而不是模型 forward 耗时。

### 稳态控制链路

| 指标 | 含义 | 主要判断 |
| --- | --- | --- |
| `stream_fps` | session 稳态实际接收帧率 | 模型和服务是否能跟上目标 FPS |
| `control_ack_latency_ms` | 控制消息发出到服务端确认收到或采样的耗时 | 网络、transport、server event loop 是否及时 |
| `control_to_next_frame_latency_ms` | 控制消息发出到下一帧到达客户端的耗时 | 控制能多快进入视频时间线 |
| p90 / p99 / max | 尾延迟 | 是否存在偶发长时间失控或卡顿 |

在 `16 FPS` 目标下，一帧间隔约 `62.5ms`。如果 `control_to_next_frame_latency_ms` 能稳定落在一个帧间隔量级内，用户通常会感到控制是实时的；如果达到秒级，即使平均 FPS 尚可，交互也会明显滞后。

当前真实世界模型结果：

- TeleFuser：`control_ack_latency_ms` 平均 `8.88ms`，`control_to_next_frame_latency_ms` 平均 `37.60ms`，低于一个 `16 FPS` 帧间隔，说明控制可以在下一帧量级进入流式时间线。
- SGLang-Diffusion：没有独立 control ack 消息，benchmark 用 `chunk_stats.event_id` 推导控制被采样时间；当前 `control_to_next_frame_latency_ms` 平均 `3329.12ms`，说明控制事件被 scheduler/model chunk 采样得很晚，用户会感到秒级滞后。
- 纯流式 mock：TeleFuser `25.63ms`、SGLang `21.93ms` 的 control-to-next-frame 都是帧间隔量级，说明当前真实 SGLang 世界模型的秒级控制滞后主要不在 WebSocket 写入本身，而在模型 forward / scheduler / control sampling 路径。

更严格的“控制是否真的影响画面”需要服务端把 `event_id` 写入 frame/chunk metadata，客户端统计该 `event_id` 首次出现在第几帧。当前 benchmark 已经能衡量控制消息到达下一帧的时间，但如果要证明按键后画面内容已经转向，还需要把视觉状态变化或事件应用帧号记录进 artifact。

## SGLang 当前配置限制

当前可运行的 SGLang-Diffusion 6/9 tuned baseline 使用以下边界：

- `SGLANG_LINGBOT_DISABLE_FLASHINFER_ROPE=1`，绕过 flashinfer RoPE JIT 问题，使用 fallback。
- `performance_mode=speed`，DiT、text encoder、VAE 全部 GPU-resident。
- `realtime_causal_sink_size=6`。
- `realtime_causal_kv_cache_num_frames=9`。
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- `VSA_sparsity=0.0`。

正式 compare 的 sink 9 / window 18 在同一 `speed` 服务、单张 H100 上 warmup 与 profile 都 OOM，进程显存约 `79.16 GiB`。6/9 tuned 跑通版本由 AIPerf 原生记录 steady reserved peak 最大 `76,919,341,056 bytes`（`73,356 MiB`）。因此 9/18 的结论是容量失败，6/9 的结论是独立 tuned 可运行结果；两者不能混为同配置性能样本。

## 下一步对比口径

后续补齐结果时建议按下面顺序更新本文档：

1. 补 Diffusers batch baseline 的同 workload 实测 summary，给出 TeleFuser vs Diffusers 的请求时延和吞吐对比。
2. 修复 SGLang FlashInfer RoPE / CUDA JIT 兼容问题后，在不使用 fallback 的环境中复核 1-GPU GPU-resident stream baseline。
3. 对 stream 世界模型分别记录首帧、稳态 FPS、控制 ack、控制到下一帧，不把首帧和稳态吞吐混成一个结论。
4. 若测试多 session 并发，需要单独标注 session_count、stagger_s 和服务是否允许多 active session。
