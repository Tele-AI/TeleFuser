# 并行推理指南

本文档详细介绍 TeleFuser 的分布式并行推理架构，包括原理介绍、配置方法和使用示例。

## 概述

TeleFuser 提供多维度并行推理能力，支持以下并行策略：

| 并行类型 | 描述 | 适用场景 |
|---------|------|---------|
| **数据并行 (DP)** | 复制模型到多个 GPU，并行处理不同数据 | 吞吐量优化 |
| **CFG 并行** | 并行计算 positive/negative prompt | CFG 加速 |
| **序列并行 (SP)** | 将长序列分割到多个 GPU | 长视频生成 |
| **流水线并行 (PP)** | 将模型层分割到多个 GPU | 大模型推理 |
| **张量并行 (TP)** | 将张量维度分割到多个 GPU | 大模型推理 |

## 架构设计

### Device Mesh 布局

TeleFuser 使用 PyTorch DeviceMesh 管理分布式并行，维度顺序为：

```
DP -> CFG -> SP (ring, ulysses) -> PP -> TP
```

```python
from telefuser.distributed import create_device_mesh_from_config
from telefuser.core.config import ParallelConfig

config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    dp_degree=1,
    cfg_degree=2,
    sp_ulysses_degree=2,
    sp_ring_degree=1,
    pp_degree=1,
    tp_degree=1,
)

device_mesh = create_device_mesh_from_config(config)
```

### 核心模块

```
telefuser/distributed/
├── device_mesh.py      # DeviceMesh 创建和进程组管理
├── pp_comm.py          # 流水线并行 P2P 通信
├── ulysses_comm.py     # Ulysses All-to-All 通信原语
├── ring.py             # Ring Attention P2P 通信
├── parallel_shard.py   # 序列并行张量分片/反分片
├── fsdp.py             # FSDP 数据并行
└── tp_parallelize.py   # 张量并行工具
```

## 序列并行

序列并行用于处理超长序列（如长视频），将序列维度分割到多个 GPU。

### Ulysses Attention

基于 All-to-All 通信的序列并行：

**原理**：
1. 每个 GPU 持有序列的一部分
2. 通过 All-to-All 将 heads 重新分配
3. 每个 GPU 拥有完整序列的部分 heads
4. 本地计算注意力后，再通过 All-to-All 恢复

**数据流**：
```
输入: (B, S_LOCAL, H_GLOBAL, D)
  -> All-to-All QKV -> (B, S_GLOBAL, H_LOCAL, D)
  -> 本地注意力计算
  -> All-to-All O -> (B, S_LOCAL, H_GLOBAL, D)
```

**特点**：
- 通信开销：2次 All-to-All（QKV + Output）
- 适合中等长度序列
- 需要头数能被 GPU 数整除

### Ring Attention

基于 P2P 通信的序列并行：

**原理**：
1. 每个 GPU 持有 Q 的一部分和 K/V 的一部分
2. K/V 在 GPU 环中轮转
3. 每个 GPU 依次看到所有 K/V 块
4. 使用在线 softmax 合并注意力输出

**算法流程**：
```python
for step in range(world_size):
    # 1. 计算当前 KV 块的注意力
    out, lse = attention(q, k, v)
    
    # 2. 发送当前 KV 到下一个 GPU
    # 3. 从上一个 GPU 接收新的 KV
    next_k, next_v = send_recv_kv(k, v)
    
    # 4. 使用在线 softmax 合并结果
    out, lse = merge_attn_states(prev_out, prev_lse, out, lse)
    
    # 5. 更新 KV
    k, v = next_k, next_v
```

**特点**：
- 支持任意长度序列
- 通信与计算可重叠
- 需要支持 log-sum-exp 的注意力实现

### USP (Unified Sequence Parallelism)

Ulysses + Ring 组合策略，支持更大规模并行：

**原理**：
1. Ring 维度：序列分割
2. Ulysses 维度：heads 分割
3. 两种策略互补，支持更多 GPU

**配置示例**：
```python
# 4 GPU: ring=2, ulysses=2
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    sp_ring_degree=2,
    sp_ulysses_degree=2,
)
```

### 异步 Ulysses (async_usp_forward)

异步 All-to-All 实现，重叠计算和通信：

```python
# 发起异步 All-to-All
q_wait = ulysses_scatter_heads(q, group)
k_wait = ulysses_scatter_heads(k, group)
v_wait = ulysses_scatter_heads(v, group)

# 等待完成
q = q_wait()
k = k_wait()
v = v_wait()

# 计算注意力
x = attention(q, k, v)

# 异步 All-to-All 输出
out_wait = ulysses_gather_heads(x, group, num_heads=num_heads)
out = out_wait()
```

## 流水线并行 (PP)

将模型层分割到多个 GPU，实现大模型推理。

### 原理

```
Stage 0 (GPU 0):  Embedding + Layers [0:N/4]
       ↓ send hidden states
Stage 1 (GPU 1):  Layers [N/4:N/2]
       ↓ send hidden states
Stage 2 (GPU 2):  Layers [N/2:3N/4]
       ↓ send hidden states
Stage 3 (GPU 3):  Layers [3N/4:N] + Head
       ↓ output
```

### P2P 通信

```python
from telefuser.distributed import PipelineP2PComm, get_pp_group

pp_group = get_pp_group(device_mesh)
comm = PipelineP2PComm(pp_group)

# 发送隐藏状态到下一阶段
if not comm.is_last_stage:
    comm.send_latent(hidden_states)

# 从上一阶段接收隐藏状态
if not comm.is_first_stage:
    hidden_states = comm.recv_latent(shape=latent_shape)
```

### 层分配

```python
# 自动分配层到各阶段
start_idx, end_idx = comm.get_stage_indices(num_layers)

# 示例：40 层，4 阶段
# Stage 0: [0:10]
# Stage 1: [10:20]
# Stage 2: [20:30]
# Stage 3: [30:40]
```

### PP Forward 实现

```python
def pp_forward(self, x, timestep, context, ...):
    # 第一阶段：Embedding + 首批层
    if self.is_pp_first_stage:
        x = self.patch_embedding(x)
        x, grid_size = self.patchify(x)
        x = self.forward_blocks_pp(x, ...)  # 处理本阶段的层
        self.pp_comm.send_latent(x)
        return None
    
    # 中间阶段：接收 + 处理 + 发送
    elif not self.is_pp_last_stage:
        x = self.pp_comm.recv_latent(...)
        x = self.forward_blocks_pp(x, ...)
        self.pp_comm.send_latent(x)
        return None
    
    # 最后阶段：接收 + 处理 + 输出
    else:
        x = self.pp_comm.recv_latent(...)
        x = self.forward_blocks_pp(x, ...)
        x = self.head(x)
        return x
```

## CFG 并行

将 Classifier-Free Guidance 的 positive/negative prompt 并行计算：

### 原理

标准 CFG 计算：
```python
noise_pred = noise_neg + cfg_scale * (noise_pos - noise_neg)
```

CFG 并行将 positive 和 negative 分配到不同 GPU：

```
GPU 0: 计算 noise_pos (positive prompt)
GPU 1: 计算 noise_neg (negative prompt)
然后：All-Gather 合并结果
```

### 使用方法

```python
# 2 GPU CFG 并行
config = ParallelConfig(
    device_ids=[0, 1],
    cfg_degree=2,
)

# 在模型中启用
dit.enable_cfgp()
```

### 实现

```python
# 分片
cfg_parallel_shard(device_mesh, [x, timestep, context, ...])

# 根据 CFG rank 选择 positive/negative
cond_flag = False if get_cfg_rank(device_mesh) else True

# 计算
output = model(x, context, cond_flag=cond_flag)

# 合并
output = cfg_parallel_unshard(device_mesh, [output])[0]
```

## 数据并行 (DP)

使用 FSDP 进行数据并行训练/推理：

### FSDP1

```python
from telefuser.distributed.fsdp import shard_model

model = shard_model(
    model,
    device_id=device_id,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    wrap_module_names=["blocks"],
    param_dtype=torch.bfloat16,
)
```

### FSDP2

```python
from telefuser.distributed.fsdp import shard_model_fsdp2

model = shard_model_fsdp2(
    model,
    wrap_module_names=["blocks"],
    param_dtype=torch.bfloat16,
)
```

## 张量并行 (TP)

将张量维度分割到多个 GPU：

### 使用方法

```python
from telefuser.distributed import parallelize_module
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

tp_plan = {
    "self_attn.q": ColwiseParallel(),
    "self_attn.k": ColwiseParallel(),
    "self_attn.v": ColwiseParallel(),
    "self_attn.o": RowwiseParallel(),
    "ffn.0": ColwiseParallel(),
    "ffn.2": RowwiseParallel(),
}

model = parallelize_module(model, device_mesh, tp_plan)
```

### 注意事项

- SP 和 TP 不能同时启用
- 需要确保头数能被 TP 度数整除

## Worker 实现

### ParallelWorker

多进程并行 Worker，使用 `multiprocessing.spawn`：

```python
from telefuser.worker import ParallelWorker

worker = ParallelWorker(stage)

# 调用方法
result = worker.process(latents, ...)

# 关闭
del worker
```

**特点**：
- 每个 GPU 一个进程
- 自动初始化进程组
- 支持同步/异步调用

### RayWorker

Ray 集群分布式 Worker：

```python
from telefuser.worker import create_ray_worker

worker = create_ray_worker(stage, enable_parallel=True)
result = worker.process.remote(latents, ...)
```

## 配置说明

### ParallelConfig

```python
@dataclass
class ParallelConfig:
    device_ids: list | None = None        # GPU ID 列表
    dp_degree: int = 1                    # 数据并行度
    cfg_degree: int = 1                   # CFG 并行度
    sp_ulysses_degree: int = 1            # Ulysses 序列并行度
    sp_ring_degree: int = 1               # Ring 序列并行度
    pp_degree: int = 1                    # 流水线并行度
    tp_degree: int = 1                    # 张量并行度
    enable_fsdp: bool = False             # 启用 FSDP
    timeout: int = 600                    # 超时时间（秒）
    queue_with_cpu: bool = False          # 使用 CPU 队列
    worker_intra_op_threads: int = 1      # 每个 worker 的 CPU intra-op 线程数
```

默认值 1 与 `torchrun` 一致，可避免多个 GPU worker 的 CPU 线程过度竞争。只有分布式
stage 包含大量 CPU 计算时才需要提高该值；它不会修改父进程的线程池。

### 验证规则

```python
# 设备数必须等于各并行度乘积
world_size = dp * cfg * sp_ring * sp_ulysses * pp * tp

# SP 和 TP 不能同时启用
if sp_degree > 1 and tp_degree > 1:
    raise ValueError("SP and TP are mutually exclusive")
```

## 使用示例

### 单 GPU 推理

```python
from telefuser.core.config import ParallelConfig

config = ParallelConfig(device_ids=[0])
```

### 2 GPU Ulysses 序列并行

```python
config = ParallelConfig(
    device_ids=[0, 1],
    sp_ulysses_degree=2,
)
pipe_config.dit_config.parallel_config = config
pipe_config.enable_denoising_parallel = True
```

### 4 GPU CFG + Ulysses

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

### 4 GPU USP

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    sp_ring_degree=2,
    sp_ulysses_degree=2,
)
```

### 4 GPU 流水线并行

```python
config = ParallelConfig(
    device_ids=[0, 1, 2, 3],
    pp_degree=4,
)
```

### 8 GPU 混合并行

```python
# DP=2, CFG=2, Ulysses=2
config = ParallelConfig(
    device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
    dp_degree=2,
    cfg_degree=2,
    sp_ulysses_degree=2,
)
```

## Wan Video 示例

```python
from telefuser.pipelines.wan_video.wan21_video import (
    Wan21VideoPipeline,
    Wan21VideoPipelineConfig,
)
from telefuser.core.config import AttentionConfig, AttnImplType

# 创建 Pipeline
pipe = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
pipe_config = Wan21VideoPipelineConfig()

# 配置并行
if gpu_num > 1:
    pipe_config.dit_config.parallel_config.device_ids = list(range(gpu_num))
    pipe_config.dit_config.parallel_config.sp_ulysses_degree = 2
    pipe_config.enable_denoising_parallel = True

# 配置注意力
pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)

# 初始化
pipe.init(module_manager, pipe_config)

# 推理
video = pipe(
    prompt="A stylish girl playing with her dog",
    num_inference_steps=40,
    num_frames=81,
    cfg_scale=6.0,
)
```

## Device Mesh 工具函数

```python
from telefuser.distributed import (
    # 进程组
    get_dp_group, get_dp_rank, get_dp_world_size,
    get_cfg_group, get_cfg_rank, get_cfg_world_size,
    get_ulysses_group, get_ulysses_rank, get_ulysses_world_size,
    get_ring_group, get_ring_rank, get_ring_world_size,
    get_pp_group, get_pp_rank, get_pp_world_size,
    get_tp_group, get_tp_rank, get_tp_world_size,
    
    # PP 辅助
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    
    # 策略检测
    get_attention_strategy,  # 返回 "local", "ulysses", "ring", "usp"

    # 通信
    ulysses_scatter_heads,
    ulysses_gather_heads,
    RingP2PComm,
    PipelineP2PComm,
    merge_attn_states,
    ring_attention_forward,
)

# 获取当前注意力策略
strategy = get_attention_strategy(device_mesh)
# "local": 无序列并行
# "ulysses": 仅 Ulysses
# "ring": 仅 Ring
# "usp": Ulysses + Ring 组合
```

## 性能优化建议

### 选择并行策略

| 场景 | 推荐策略 | 说明 |
|------|---------|------|
| 短视频 (81帧) | 单 GPU 或 CFG=2 | 通信开销小 |
| 中等视频 (161帧) | Ulysses=2 | All-to-All 效率高 |
| 长视频 (241+帧) | Ring 或 USP | 支持任意长度 |
| 大模型 (14B) | PP 或 FSDP | 模型分割 |

### FSDP vs TP 选择

在多 GPU 推理场景下，选择 FSDP 还是 TP 取决于显存和通信条件：

| 条件 | 推荐策略 | 原因 |
|------|---------|------|
| 单卡显存可容纳单层 Layer | **FSDP** | TP 对通信带宽要求更高，FSDP 通信开销更低 |
| 单卡显存无法容纳单层 Layer | **TP** | 必须使用 TP 将张量切分到多卡 |
| 多机/低带宽网络 | **FSDP** | TP 需要高带宽低延迟的 GPU 互联 |
| 单机 NVLink/InfiniBand | **TP** | 高带宽互联下 TP 效率更高 |

**选择建议**：
- 优先评估单卡显存是否能容纳单层 Layer（包括激活值）
- 如果可以，优先选择 FSDP，因为它对通信要求更低
- 仅当单卡显存不足时，才考虑使用 TP
- FSDP 可与 PP、SP 等策略组合使用

### 通信优化

1. **使用异步通信**：`async_usp_forward` 重叠计算和通信
2. **批量通信**：`batch_isend_irecv` 减少通信次数
3. **FP8 量化**：减少通信数据量

### 内存优化

1. **序列并行**：减少每个 GPU 的序列长度
2. **流水线并行**：减少每个 GPU 的层数
3. **CPU Offload**：将权重卸载到 CPU

## 故障排除

### 设备数不匹配

```
RuntimeError: device num 4 and world size 2 not match
```

**解决方案**：确保 `len(device_ids) == dp * cfg * sp_ring * sp_ulysses * pp * tp`

### SP 和 TP 冲突

```
ValueError: Not allowed to enable sequence parallel and tensor parallel together
```

**解决方案**：SP 和 TP 不能同时启用，选择其中一种。

### Ring Attention 需要 LSE

```
RuntimeError: Ring attention requires log-sum-exp from attention implementation
```

**解决方案**：使用支持 LSE 的注意力实现（Flash Attention 2/3/4）。

### 通信超时

```
RuntimeError: ParallelWorker timeout
```

**解决方案**：增加 `timeout` 参数值，或检查网络连接。

## 参考资料

- [Ulysses: Sequence Parallelism](https://arxiv.org/abs/2309.14509)
- [Ring Attention](https://arxiv.org/abs/2310.01889)
- [GPipe: Pipeline Parallelism](https://arxiv.org/abs/1811.06965)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)
