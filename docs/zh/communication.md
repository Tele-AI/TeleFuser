# 通信架构

本文说明 TeleFuser 如何在 GPU、进程和 worker 之间传递 tensor，通信职责如何划分，以及实现通过哪些约束
保证正确性和效率。DP、CFG、SP、PP、TP 与 FSDP 的配置方法见[并行推理指南](parallel.md)。

## 设计目标

TeleFuser 的通信实现遵循五条原则：

1. 控制面与大 tensor 数据面分离。
2. 可复用的 collective 机制不放在模型实现中。
3. 同机 GPU 路径避免 host staging 和 device-wide synchronization。
4. 限制常驻显存，并明确取消、超时和关闭语义。
5. 优化传输不适用时保留 PyTorch 原生回退路径。

实现采用分层设计，而不是用一个通用 transport 隐藏所有差异。NCCL collective、CUDA IPC、Ray 和服务网络
具有不同的拓扑与生命周期要求；它们共享职责规则，但不共享同一种运行时协议。

## 架构总览

```text
Pipeline / 模型代码
  | 声明拓扑、tensor layout 和 stage 连接关系
  v
并行策略与 worker adapter
  | DeviceMesh group、Ulysses、Ring、VAE spatial、ParallelWorker
  v
共享通信机制
  | collectives.py                 | worker/tensor_channel.py
  | 进程组内 tensor 搬运            | 同机跨 worker tensor 搬运
  v                                v
PyTorch distributed / NCCL        CUDA IPC + multiprocessing 元数据
```

主要职责边界如下：

| 范围 | 负责模块 | 职责 |
|------|----------|------|
| 进程拓扑 | `telefuser/distributed/device_mesh.py` | 创建并暴露具名 process group |
| 共享 collective | `telefuser/distributed/collectives.py` | 连续 gather buffer 与成组 reduction |
| 序列注意力 | `ulysses_comm.py`、`ring.py` | 策略专用的 All-to-All 与 P2P 协议 |
| 序列/CFG 分片 | `parallel_shard.py` | tensor padding、切分、gather 和恢复 |
| 空间 VAE | `vae_spatial.py` | 高度分片、邻居 halo 和输出 gather |
| Pipeline P2P | `pp_comm.py` | PP process group 内 rank-to-rank 通信 |
| Worker 执行 | `worker/parallel_worker.py` | 进程组、worker 生命周期和命令派发 |
| 跨 worker tensor | `worker/tensor_channel.py` | CPU shared memory 和持久 CUDA IPC pool |
| 集群 actor | `worker/ray_worker.py` | Ray 资源分配和可选的本地 worker group |

模型代码负责模型特有的 layout 和重建语义；不应自行分配 rank buffer 或直接调用 tensor collective，而应
调用共享 collective 或策略模块。

## 进程组与 DeviceMesh

`ParallelWorker` 为每个本地 rank 启动一个 spawn 进程。当 group 大于一个 rank 时，它选择分配的 device，
初始化平台对应的 distributed backend，再让 stage 并行化模型。CUDA 平台通常通过平台抽象使用 NCCL。

`create_device_mesh_from_config()` 按以下顺序创建具名维度：

```text
DP -> CFG -> SP (ring, ulysses) -> PP -> TP
```

同时启用 Ring 和 Ulysses 时，SP 是二维 `(ring, ulysses)` 子 mesh。`get_cfg_group()`、
`get_ring_group()`、`get_ulysses_group()` 和 `get_pp_group()` 等 accessor 避免模型重复推导 rank 列表。
配置的 world size 必须等于所有并行 degree 的乘积。目前 SP 和 TP 互斥。

## 共享 Collective 原语

`telefuser/distributed/collectives.py` 是内部实现边界，有意不从顶层 `telefuser.distributed` API 导出。

### 等形状 Gather

`all_gather_stacked()` 把各 rank 的等形状 tensor gather 到一个 rank-major allocation：

```text
local [D0, ...]
  -> all_gather_into_tensor
buffer [world_size * D0, ...]
  -> view
result [world_size, D0, ...]
```

这样每次 gather 只分配一个连续输出 buffer，不再为每个 rank 单独分配 tensor。调用方可以保留 rank 维进行
模型特有的重建，也可以使用 `all_gather_cat()` 按 rank 顺序沿任意维拼接。

`parallel_shard.py`、LingBot Video 序列恢复、Wan/Wan2.2 VAE 重建和 VAE spatial gather 都复用这些原语。
VAE 高度分片不等长时，先 pad 到最大本地高度，gather 后再裁剪。

### 成组 Reduction

`all_reduce_sum_()` 先异步提交所有相互独立的 sum reduction，再统一等待 work handle。Tile blending 用它同时
归约 value 和 weight，模型中不再重复实现同步逻辑。

这些 helper 要求所有参与 rank 以相同顺序调用 collective，并提供兼容的 shape 和 dtype。Collective 顺序
不一致会造成分布式死锁，不能作为单 rank 异常恢复。

## 序列并行通信

### Ulysses

Ulysses 把“序列分片、完整 heads”转换成“完整序列、本地 heads”：

```text
[B, S_local, H_global, D]
  -> All-to-All：scatter heads / gather sequence
[B, S_global, H_local, D]
  -> 本地 attention
  -> All-to-All：gather heads / scatter sequence
[B, S_local, H_global, D]
```

`ulysses_scatter_heads()` 和 `ulysses_gather_heads()` 使用 functional `all_to_all_single`，并返回 wait closure，
使提交与消费分离。Attention 会先提交 Q、K、V 三个 collective，再等待其中任何一个，让 NCCL 不受 Python
串行提交限制。输出 All-to-All 恢复原始 sequence/head layout。

Attention head 数必须能被 Ulysses world size 整除，反向 layout 恢复时 gathered sequence length 也必须可整除。
Helper 会在通信前验证这些约束。

### Ring Attention

Ring Attention 保持 Q 本地不动，让 K/V 在相邻 rank 之间轮转。`RingP2PComm` 把 group-local 邻居解析为
global rank，使用 `batch_isend_irecv` 批量提交 `isend` 和 `irecv`，并优先复用调用方提供的 receive buffer。

K/V 可以拼成一次传输，接收后再切成 view。下一块 KV 的通信先提交，再计算当前块 attention，只在消费下一块
之前等待。各块的 attention 结果通过 online log-sum-exp 合并。

Ring AllGather 变体实现更简单，但每个 rank 都会物化全局 K/V；它是显存开销更大的备选实现，不是长上下文
首选路径。

## 空间 VAE 通信

按高度分片的 VAE decode 包含两类通信：

1. 空间卷积前的邻居 halo 交换。
2. 需要完整高度 tensor 时的 rank-order gather。

Halo 交换复用 send/receive buffer，并用一次 `batch_isend_irecv` 提交当前 rank 存在的上下邻居操作。边界 rank
把缺失 halo 填零。Buffer 复用避免每层、每帧重复创建 contiguous halo tensor。

完整高度重建使用共享连续 gather 原语。先 gather 各 rank 的本地高度，才能正确处理不等分片的 padding 和裁剪。
最终 tensor 会恢复原来的 channels-last 或 contiguous memory format。

## 跨 Worker Tensor Channel

`WorkerTensorChannel` 在同一主机上连接一个 producer worker group 和一个 consumer worker group。小型控制
元数据与 tensor storage 分开传输：

```text
Producer worker                 父进程 / 控制路径                 Consumer worker
      |                                  |                               |
      | stage tensor                     |                               |
      |-- staging 到 IPC slot ---------->| WorkerTensorRef 元数据 ------>|
      |                                  |                               |-- pool 只映射一次
      |<------- generation ACK / completion event -----------------------|-- peer copy
```

父进程只接收 `WorkerTensorRef`，不会物化 CUDA tensor 内容。嵌套 dict、tuple 和 list 保持结构；同一 artifact
中的重复 tensor leaf 只传输一次。

### 持久 CUDA IPC Pool

稳定 CUDA tensor profile 由 tensor index、shape、dtype 和 source device 共同标识。每个 profile 持有一块
持久 allocation，默认包含两个 slot。Slot 使用 round-robin 选择，使顺序流量真正使用双缓冲。

Pool allocation 和 IPC handle 只创建一次。Consumer 缓存导入后的 storage 和 event handle，steady state 不会
反复打开 CUDA IPC allocation。每个 channel 最多缓存八个 profile；更多动态 profile 回退到 PyTorch
multiprocessing tensor transport，避免常驻 HBM 无界增长。

配置 `shard_dim` 后，每个 consumer rank 只接收自己的 rank-local view。Producer 只 staging 一次，聚合
peer-copy 流量保持为一个 logical tensor，而不是为每个 consumer 拷贝一份完整 tensor。LingBot 空间 VAE 使用
`shard_dim=-2` 按高度分片。

### Stream 顺序协议

每个 slot 持有一个可复用的 producer-ready event：

1. Producer 在当前 stream 把 source tensor copy 到 slot。
2. Producer 记录 ready event，并随元数据发布 event handle。
3. Ready event 尚未完成时，consumer stream 才等待它。
4. Consumer 把映射后的 slot view copy 到 output tensor。
5. Consumer 先记录可复用 completion event，再发布 generation ACK。
6. Producer 覆盖复用 slot 前，其 staging stream 等待所有实际执行 copy 的 rank completion event。

整个 transport 路径不包含 device-wide synchronization。若 producer staging 或 consumer copy 已完成，event
`query()` 提供快速路径。

ACK 使用 lock-free shared generation array。正 generation 表示该 rank 已 copy payload；负 generation 表示
该 rank 已 discard。Producer 只为真正 copy 的 rank 等待 completion metadata，因此取消路径不会等待一个
从未记录的 event。

### CPU 与回退传输

CPU tensor 使用 multiprocessing shared memory。无法池化的 CUDA profile 使用 PyTorch multiprocessing 的
CUDA tensor transport。两种情况都为每个 consumer rank 保留一个 FIFO，最终 device placement 由接收进程完成。

## 控制面与生命周期

`ParallelWorker` 的 command/result queue 传输方法名、参数、小型结果和 tensor reference。它们使用
`SimpleQueue`，避免 background feeder 引入调度尾延迟。通过 `WorkerTensorChannel` 连接的大 tensor 留在直接
数据路径。

Channel contract 是有序且有界的：

- 一个 channel 只能绑定一个 producer 和一个 consumer group。
- Consumer rank 数必须与 channel 配置匹配。
- Consumer 必须按 producer 顺序解析 artifact。
- 被取消的末端 artifact 必须调用 `discard_tensor_refs(..., sync=True)` 释放。
- 关闭时先停止 consumer，再停止 producer，最后关闭 channel。
- Worker cleanup 在释放本地 IPC mapping 前同步尚未完成的 device work。

Timeout 会把 worker 标记为 failed 并终止其进程。失败 worker 不允许继续复用，避免破坏 channel 的部分顺序。

## Pipeline Parallel 与 Ray 边界

`PipelineP2PComm` 与 `WorkerTensorChannel` 是不同传输。它在同一 PP process group 内使用 NCCL send/recv 和
batched P2P。现有 Wan PP 的 shape/grid broadcast 与 latent convenience method 仍由 PP 路径负责。

CUDA IPC 只能用于同一主机。`RayWorker` 遵守 Ray 分配的逻辑 device，并可在 actor 内运行本地
`ParallelWorker`，但 TeleFuser 不会用 CUDA IPC 替代 Ray 的跨节点 object transport。需要跨节点 GPU-direct
传输时，必须单独设计 transport 和拓扑 contract。

## 效率约束

通信实现保持以下性能属性：

- 直接 worker edge 不在父进程物化 CUDA tensor。
- 同机池化 CUDA 路径不经过 host staging。
- 完整 handoff 只包含两个 logical device copy：producer staging 和 consumer output copy。
- Steady state 复用 pool、storage 和 event handle。
- Slot 和 profile 数量有界，避免常驻 HBM 无界增长。
- 使用 stream event，不使用 device-wide synchronization。
- Consumer 处理互斥 shard 时只 copy rank-local 数据。
- 等形状 gather 只分配一个连续输出 buffer。
- Ulysses 的 Q/K/V collective 先全部提交再等待。
- Spatial VAE 和 Ring 使用批量邻居 P2P，并复用 halo/receive buffer。

## 验证与基准测试

专项测试覆盖纯 collective layout、真实双卡 NCCL 顺序、CUDA IPC readiness 与 slot reuse、取消、多 consumer
ACK，以及空间 VAE parity：

```bash
pytest tests/unit/distributed/
pytest tests/integration/test_collectives.py
pytest tests/integration/test_worker_tensor_channel.py
pytest tests/integration/test_wan_video_vae_spatial.py
```

本地 SGLang 对比同时包含两种实现的 producer staging、metadata transport、target copy、target synchronization
和 slot ACK：

```bash
python tools/validation/benchmark_tensor_channel_vs_sglang.py
```

默认门禁测量 200 次传输。TeleFuser p50 不得比 SGLang 高 5% 以上；p95 上限取 10% 和 0.05 ms 中较宽者，
用于覆盖亚毫秒 multiprocessing 调度抖动。判断回归时还应比较 copy 次数和 mean latency；单个进程调度尾点
不能单独证明 transport 退化。

修改通信调用点后，应复跑所有受影响 pipeline。Example runner 提供 baseline 输出对比：

```bash
python examples/run_examples.py --pipeline <name> --gpus 0,1,2,3
```

## 扩展规则

新增通信路径时：

1. 通用的等形状 gather 或 reduction 机制放在 `distributed/collectives.py`。
2. 算法专用协议放在 `telefuser/distributed/` 下的聚焦模块中。
3. 模型代码只负责 tensor layout 和模型语义。
4. `WorkerTensorChannel` 只用于同机、单 producer、单 consumer group 的 edge。
5. 没有明确拓扑缺口时，不新增 fallback、环境变量或公共配置字段。
6. 优化 happy path 前，先定义顺序、所有权、取消、timeout 和 shutdown。
7. 新 collective 或 IPC 同步规则必须添加真实多进程测试。

不要因为控制路径已经存在就让大 tensor 绕行父进程；也不要用 device-wide synchronize 修补顺序问题，应通过
process-group work handle 或 stream event 表达依赖。

## 已知边界

- CUDA IPC pool 只支持同一主机。
- 稳定池化 profile 要求 tensor index、shape、dtype 和 source device 固定。
- Ring AllGather 用实现简单性换取每个 rank 的 K/V 副本显存。
- 空间 VAE halo exchange 已复用 buffer，但当前仍会在依赖它的卷积前等待。
- WAN pipeline-parallel 通信仍是独立的模型专用兼容区域。
- Ray 跨节点 tensor 性能取决于 Ray transport 与集群配置。

## 相关文档

- [并行推理指南](parallel.md)
- [Attention 实现指南](attention.md)
- [流式调度器](stream_scheduler.md)
- [测试指南](testing.md)
