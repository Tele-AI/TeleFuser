---
title: "ABot 多会话服务：状态隔离、兼容批处理与多卡调度"
description: 为 ABot-World 引入 session 状态隔离、兼容批处理、容量感知 admission 和多 GPU worker 调度。
date: 2026-08-19
status: validated
validated_revision: 52491cb
hardware: 4 x NVIDIA H100 80 GB HBM3
tags:
  - abot-world
  - multi-session
  - batched-serving
  - process-nccl
  - livekit
---

# ABot 多会话服务：状态隔离、兼容批处理与多卡调度

ABot-World 的一次交互并不是一个可以立即释放的请求。用户建立连接后，服务需要持续保存 prompt
embedding、DiT 的 KV cache、scheduler 以及视频解码的 temporal state，直到这次交互结束。单会话路径很容易
实现这些状态，但当多个用户同时在线时，每个用户各自执行一次 denoising，GPU 只共享了模型权重，计算却没有
得到复用。

问题的另一面是，这些 session 并不总是可以直接拼成一个 batch：它们可能处在不同的 chunk 阶段，拥有不同的
控制帧数量，或者对应不同的 latent frame 位置。multi-session serving 的核心不是“把请求拼起来”，而是先
确认状态兼容，再共享一次模型执行，最后把状态和输出分别还给原来的 session。

## 设计目标

实现遵循以下边界：

1. session 的生成状态和解码状态始终独立；
2. 只有控制帧、cache shape 和 RoPE 位置兼容的 session 才进入同一个 batch；
3. admission 受实际 GPU 容量约束，而不是只由连接数决定；
4. LiveKit 控制面与模型 worker 解耦，支持多 GPU 和 chunk 边界迁移；
5. batch 不兼容、显存不足或 CUDA Graph 不适用时，保留 B1/eager fallback。

## 架构总览

```text
LiveKit / HTTP 控制面
  | session 创建、admission、ownership、控制消息
  v
ABotWorldLiveKitService
  | deadline-aware scheduler、batch compatibility、backpressure
  v
模型 worker（每张 GPU 一个）
  | collate session state -> 一次 DiT 执行 -> scatter session state
  v
每个 session 的 TAEW/VAE decode 与输出队列
```

主要职责由以下组件承担：

| 范围 | 负责模块 | 职责 |
|---|---|---|
| Session 状态 | `ABotWorldInteractiveSession` | prompt、KV cache、scheduler、随机数和 temporal decode state |
| 单卡服务 | `ABotWorldLiveKitService` | admission、调度、批处理、输出节流和 session 生命周期 |
| 进程 worker | `process_worker_pool.py` | 将模型执行移出控制面，管理 worker 生命周期 |
| 多卡 worker | `nccl_process_worker_pool.py` | 每 GPU 一个模型 worker，固定 NCCL communicator |
| 集群路由 | `turboserve.py` | 负载感知 placement、ownership 和迁移决策 |

模型代码只负责模型输入、cache 更新和输出解码；连接管理、GPU 选择和跨 worker 路由不下沉到模型内部。

## Session 状态隔离

每个 `ABotWorldInteractiveSession` 持有自己的 prompt embedding、首帧 latent、self/cross KV cache、scheduler
和 RNG state，同时保存 VAE/TAEW temporal decode state 以及 chunk/frame 计数。session 被挂起时，retained
tensor 可以移到 CPU；恢复时再回到原有的逻辑 owner。

因此，batch 只共享短暂的模型计算，不共享可变的生成状态：一次 batch 执行结束后，cache、解码状态和输出
仍然回到各自的 session。

## 兼容批处理

调度器按照最早 playout deadline 选择 ready session。进入 batch 前会验证 control latent frame 数、首 chunk
与 continuation 阶段、absolute RoPE 的 `next_latent_frame`，以及 latent/cache 的 shape 和 dtype。

兼容的 session 在短 formation window 内组成 B2 或 B3：

```text
ready sessions
      | compatibility check
      v
collate prompt / noise / action / KV cache
      | one denoise_interactive_blocks call
      v
scatter cache and latent outputs
      | per-session decode and queue
      v
independent LiveKit delivery
```

首 chunk 与 continuation 不能安全混合时，调度器会缩小 batch 或退回 B1。这样做牺牲一部分吞吐，换取
cache 指针、位置编码和帧顺序的确定性。

## 多 GPU 与 session 迁移

父进程拥有 LiveKit、HTTP admission 和 ownership；模型 worker 拥有模型实例和 GPU tensor。`process-nccl`
模式为每张 GPU 启动一个 worker，并在 worker 之间建立固定 NCCL communicator。客户端只看到一个服务端点，
不需要选择 GPU。

迁移是一个 chunk-boundary transaction：先 quiesce session，排空输出和 publisher credit，快照状态，在目标
worker 恢复，再用递增的 ownership epoch 提交新路由。提交失败时保留 source owner，不会让一半状态同时被两个
worker 接管。

## 输出背压与 fallback

实时视频输出有两种语义：`latest` 在队列满时丢弃最旧的完整 block，`lossless` 则只阻塞当前 session。
CUDA Graph 只在输入 shape 和状态满足 profile 时 replay；其他情况使用 eager 路径。batch 兼容性检查失败时，
同样回退到较小 batch 或 B1。

## 性能结果

### 单 H100 目标端计算

LF3 microbenchmark 中，聚合吞吐从 B1 的 30.98 FPS 增加到 B4 的 37.01 FPS，提升约 19.5%。B4 满足 8 FPS、
1.5 秒 chunk deadline；继续增加 batch 后无法满足该 deadline。这里测量的是模型计算，不包含网络、LiveKit
和浏览器交付。

### 四卡 trace

2026-08-19 的四卡、30 分钟 trace 得到以下结果：

| 指标 | 结果 |
|---|---:|
| Admission | 295/295 立即分配，0 failed |
| 真实模型调用 | 5,475，全部成功 |
| B1 / B2 / B3 调用次数 | 4,533 / 907 / 35 |
| B2 chunk item 占比 | 28.1% |
| B3 chunk item 占比 | 1.6% |
| 相比全部 B1 少的模型调用 | 977 次（15.1%） |
| Graph fallback | 0 |
| Demand SLO | 81.8% |
| 活跃用户平均 FPS | 11.13 |
| A2F p95 | 4.13 s |

调用直方图对应的 chunk item 数为：

```text
4533 + 2 × 907 + 3 × 35 = 6452
```

如果每个 item 都使用 B1，需要 6,452 次模型调用；实际执行 5,475 次，减少 977 次。这个差值直接反映了
兼容批处理带来的模型执行复用，而不是通过丢弃 session 得到的数字。

## 正确性与适用边界

测试覆盖 session 状态隔离、30-block 生成、batch ordering、cache scatter、worker 生命周期和迁移事务。
长时间 trace 还验证了 admission、输出队列和多 worker dispatch。实验中四个 worker 均未发生 Graph fallback，
所有模型 dispatch 都返回 `ok`。

平均活跃用户 FPS 为 11.13，demand SLO 达成率为 81.8%。batch 收益取决于 session 控制消息的到达时间和
tensor 兼容性，这组结果主要体现了多用户场景下的容量和计算复用收益。

## 复现记录

实验记录位于：

```text
results/experiments/abot_4gpu1237_lf3_12fps_publicdemo_b3_f36_graph_30min_20260819T060933Z/
```

workload 使用公开 TurboServe trace 归一化得到，覆盖 admission、batch dispatch、状态保持、输出交付和多
GPU worker 行为。

## 限制

- 当前结果覆盖单机四张 H100；
- batch 形成依赖 session 到达时间和输入兼容性；
- 迁移开销依赖 worker 拓扑、显存容量和传输配置；
- 当前 trace 的 demand SLO 尚未达到 100%。

## Related Work

实现结合了 continuous batching、deadline-aware scheduling 和多 GPU worker pool，并将这些机制接入
ABot-World 的 KV cache、TAEW/VAE temporal state 和 LiveKit 实时交付路径。
