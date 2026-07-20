# 流式 Pipeline 调度器

## 目的

`StreamingPipelineOrchestrator` 用于协调按顺序 chunk 执行的长时、有状态生成管线。它适用于 LingBot 一类的
交互式任务：较早 chunk 仍在 encode、denoise 或 decode 时，后续 control 输入可以持续进入。

该调度器与 `FlexiblePipelineOrchestrator` 职责不同。后者协调请求级 stage group；流式调度器管理有界的
per-session 数据流和长期存活的 stage actor。

## 架构

调度器执行由带类型 artifact 构成的有向无环图：

```text
外部输入
   |
   v
encode -- condition --> denoise -- latent --> decode -- frames --> 输出
                           ^
                           |
                        control
```

每个逻辑 stage 对应一个长期存活的 actor。相互独立的 actor 可以并发执行，即使其 worker 使用同一张物理 GPU。
CUDA device placement 本身不代表串行执行或资源所有权。

| 组件 | 职责 |
| --- | --- |
| `StreamingStageSpec` | 声明 stage 的输入、输出、顺序、准入上限和可选 resource group。 |
| `StreamingEdgeSpec` | 声明有界 artifact 路径及其 per-session 容量。 |
| `StreamingPipelineSpec` | 定义完整图、输出和 resource groups。 |
| `LocalStageActor` | 串行执行一个本地有状态 stage。 |
| `ParallelWorkerStageActor` | 为一个 `ParallelWorker` 提供唯一 actor owner。 |
| `StreamingPipelineOrchestrator` | 校验图并在多个 session 之间调度已就绪的 sequence item。 |

## 数据流与顺序

每个输入、中间 artifact 和输出都关联 session ID 与 sequence ID。默认的
`StageOrdering.PER_SESSION_STRICT` 保证单个 session 内有状态更新的因果顺序，同时允许不同 session 公平交错。

edge 和输出均有显式容量。下游 stage 无法继续接收任务时，调度器施加 backpressure，而不是无限保留 tensor。
因此管线实现必须把提交视为受准入控制的操作，而不是无界队列。

## Actor 所有权与 Session 生命周期

一个有状态 worker 在整个生命周期内只能有一个 actor owner。特别是，一个 `ParallelWorker` 不得由 session
facade 直接调用，也不得被多个 stage actor 共享。该约束保证 result ordering，并让 cache 更新与释放发生在
唯一、明确的执行上下文中。

session 关闭按以下顺序执行：

1. 停止接收新任务。
2. 根据 session 策略排空或取消已接收任务。
3. 通过 owning actor 按逆拓扑顺序释放 stage-owned state。
4. 释放 scheduler artifact 引用，并确认没有遗留容量 slot。
5. 记录清理失败；不得复用只完成部分释放的状态。

LingBot 的离线 chunked generation 与双向 WebRTC session 均使用此生命周期。

## Resource Group 与放置

`StreamingResourceGroupSpec` 表示显式的共享并发约束。只有当 `StreamingStageSpec.resource_group` 引用
`StreamingPipelineSpec.resource_groups` 中声明的 group 时，stage 才会参与该约束。

不要根据 `device_id` 或 `ParallelConfig.device_ids` 推断 resource group。对于 LingBot，VAE encode、DiT 和
VAE decode 是独立 actor，即使位于同一张 GPU 也可以重叠执行。若放置超过显存容量，应移动 stage 到其他设备，
或声明明确的部署约束；不要增加隐式的全局互斥锁。

LingBot 的 `vae_encode_config` 和 `vae_decode_config` 是两个独立且完整的
`ModelRuntimeConfig`，不再提供共享的 VAE placement fallback。

## 可观测性与实时运行

`StreamingSessionMetrics` 记录 scheduler 观测到的时序和生命周期数据，包括：

| 信号 | 运行用途 |
| --- | --- |
| 首帧延迟 | 从首个 ingress 被接收到首个输出发出的时间。 |
| Control-to-output 延迟 | 从 control/input 被接收到对应输出发出的时间。 |
| Chunk period | 相邻输出 chunk 的节奏。 |
| Stage timing | 每次调用的 input-ready、admitted 和 completed 时间。 |
| Idle interval | 准入间隔及其阻塞原因。 |
| Diagnostics | stale、orphaned、duplicate、cleanup failure 和 slot leak 计数。 |

实时运行时，应比较 p95 chunk period 与一个 chunk 代表的媒体时长：

```text
实时系数 = p95 chunk period / chunk 媒体时长
```

小于一表示生成通常快于播放消耗。生产容量规划仍应为编码、传输和调度抖动保留余量。

## 接入要求

接入流式管线时：

- 模型专属预处理和 cache 行为必须保留在通用 scheduler 之外。
- 每个有状态 worker 必须只有一个 actor owner。
- 每条携带 tensor 的 artifact 路径都必须定义有界 edge。
- 从 ingress 到输出持续保留 session ID 和 sequence ID。
- session state 必须隔离，并通过 owning actor 释放。
- 只为真实且明确的部署约束声明 resource group。
- 应验证 session 交错、backpressure、取消、actor failure 和 cleanup failure。
