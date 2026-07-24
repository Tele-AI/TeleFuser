# Attention 实现指南

本文档描述 TeleFuser 中的注意力实现架构，包括配置、调用流程、Pipeline 支持情况，以及不同注意力后端的安装教程。

## 概述

TeleFuser 提供统一的注意力配置接口，支持密集注意力和稀疏注意力实现。架构包括：

- **配置层**: `AttentionConfig`, `SparseAttentionConfig`, `AttnImplType`
- **运行时状态**: 用于稀疏注意力的 `SparseAttentionState`
- **实现层**: 支持多种后端的 `attention()` 函数

## 配置类

### AttnImplType

定义所有可用注意力实现的枚举：

```python
class AttnImplType(Enum):
    # 密集注意力
    TORCH_SDPA = auto()
    TORCH_CUDNN = auto()
    FLASH_ATTN_2 = auto()
    FLASH_ATTN_3 = auto()
    FLASH_ATTN_4 = auto()  # 适用于 Hopper (SM90) 和 Blackwell (SM100+) GPU
    SAGE_ATTN_2_8_8 = auto()
    SAGE_ATTN_2_8_16 = auto()
    SAGE_ATTN_2_8_8_SM90 = auto()
    SPARGE_ATTN = auto()
    # 稀疏注意力
    RADIAL_ATTN = auto()
    LOCAL_SPARSE_ATTN = auto()
```

### AttentionConfig

所有注意力类型的统一配置：

```python
@dataclass
class AttentionConfig:
    attn_impl: AttnImplType = AttnImplType.TORCH_SDPA
    sparse_config: SparseAttentionConfig | None = None
    scale: float | None = None
    dropout: float = 0.0
    is_causal: bool = False
```

工厂方法：
- `AttentionConfig.dense_attention(attn_impl)` - 创建密集注意力配置
- `AttentionConfig.radial_attention(**kwargs)` - 创建径向稀疏注意力配置
- `AttentionConfig.local_sparse_attention(**kwargs)` - 创建局部稀疏注意力配置

### SparseAttentionConfig

稀疏注意力配置：

```python
@dataclass
class SparseAttentionConfig:
    sparse_impl: str | None = None           # "radial", "local" 等
    dense_timesteps: int = 40               # 初始时间步使用密集注意力
    dense_layers: int = 0                   # 初始层使用密集注意力
    decay_factor: float = 1.0               # 注意力窗口衰减因子
    local_window_size: int = 6              # 局部稀疏注意力窗口大小
    block_size: int = 128                   # 稀疏计算块大小
    use_sage_attention: bool = False        # 使用 sage attention 后端
```

## 调用流程

### 1. 配置

```python
from telefuser.core.config import AttentionConfig, AttnImplType

# 密集注意力
config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)

# 径向稀疏注意力
config = AttentionConfig.radial_attention(
    dense_timesteps=40,
    dense_layers=0,
    decay_factor=1.0,
)
```

### 2. Pipeline 配置

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

pipe_config = Wan21VideoPipelineConfig()
pipe_config.dit_config.attention_config = config
```

### 3. 模型初始化

`ModelRuntimeConfig` 包含 `attention_config`：

```python
from telefuser.core.config import ModelRuntimeConfig

runtime_config = ModelRuntimeConfig()
runtime_config.attention_config = config  # 默认: TORCH_SDPA 密集
```

### 4. 模型设置

Pipeline Stage 将配置传递给模型：

```python
# 在 SingleDitDenoisingStage.__init__ 中
self.dit.set_attention_config(model_runtime_config.attention_config)
```

### 5. Attention 执行

模型调用 `attention()` 函数：

```python
from telefuser.ops.attention import attention

output = attention(
    q, k, v,
    attention_config=self.attention_config,
    sparse_state=sparse_state,  # 稀疏注意力必需
    input_layout="BSND",
    output_layout="BSND",
)
```

### 6. 稀疏 Attention 状态（仅稀疏）

对于稀疏注意力，运行时状态跟踪当前步/层：

```python
from telefuser.ops.attention import MaskMap, SparseAttentionState

# 创建状态
sparse_config = config.sparse_config
mask_map = MaskMap(video_token_num=3840, num_frame=16)
sparse_state = SparseAttentionState(sparse_config, mask_map, model_type="wan")

# 每步更新
sparse_state.update(numeral_timestep=20, layer_idx=5)

# 检查是否应使用密集
if sparse_state.should_use_dense():
    # 使用密集注意力
else:
    # 使用稀疏注意力
```

## Pipeline 支持情况

| Pipeline | 密集注意力 | 稀疏 (径向) | 说明 |
|----------|-----------|------------|------|
| `Wan21VideoPipeline` | ✅ | ✅ | 视频生成完整支持 |
| `Wan22VideoPipeline` | ✅ | ✅ | 视频生成完整支持 |
| `QwenImagePipeline` | ✅ | ❌ | 图像生成不需要时序稀疏注意力 |
| `ZImagePipeline` | ✅ | ❌ | 图像生成不需要时序稀疏注意力 |

### Wan21VideoPipeline / Wan22VideoPipeline

支持密集和径向注意力：

```python
# 径向注意力用于内存高效的长视频生成
from telefuser.core.config import AttentionConfig

config = AttentionConfig.radial_attention(
    dense_timesteps=40,      # 前40个时间步使用密集
    dense_layers=0,          # 前N层使用密集
    decay_factor=1.0,        # 窗口衰减因子
    use_sage_attention=False,
)
pipe_config.dit_config.attention_config = config
```

使用径向注意力时：
1. Pipeline 在 `__call__` 中调用 `dit.enable_radial_attention()`
2. 使用 `MaskMap` 创建 `SparseAttentionState`
3. 在去噪循环中每时间步/层更新状态
4. 早期时间步/层自动回退到密集注意力

### QwenImagePipeline / ZImagePipeline

仅支持密集注意力（图像生成没有时序维度）：

```python
from telefuser.core.config import AttentionConfig, AttnImplType

config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
pipe_config.dit_config.attention_config = config
```

## 实现细节

### 密集注意力后端

| 后端 | 描述 | 依赖 |
|------|------|------|
| `TORCH_SDPA` | PyTorch 缩放点积注意力 | PyTorch 2.0+ |
| `TORCH_CUDNN` | cuDNN 注意力后端 | cuDNN |
| `FLASH_ATTN_2` | Flash Attention 2 | `flash_attn` 包 |
| `FLASH_ATTN_3` | Flash Attention 3 | `flash_attn_interface` 包 |
| `FLASH_ATTN_4` | Flash Attention 4 | `flash_attn` 包（含 `cute` 子模块） |
| `SAGE_ATTN_*` | SageAttention 变体 | `sageattention` 包 |
| `SPARGE_ATTN` | Sparge Attention | `spas_sage_attn` 包 |

**Flash Attention 4 说明**: Flash Attention 4 针对 **Hopper (SM90, H100)** 和 **Blackwell (SM100+, B100/B200)** GPU 架构进行了优化，在这些架构上提供显著的性能提升。对于旧版 GPU（Ampere、Ada Lovelace），请使用 Flash Attention 2 或 3。

### 稀疏注意力后端

| 后端 | 描述 | 依赖 |
|------|------|------|
| `RADIAL_ATTN` | 视频径向注意力 | `flashinfer` 或 `sageattention` (优先使用 tf-kernel) |
| `LOCAL_SPARSE_ATTN` | 局部窗口稀疏注意力 | `block_sparse_attn` |

**SageAttention 优先级说明**: 当设置 `use_sage_attention=True` 时，如果 tf-kernel 和独立的 `sageattention` 包都可用，系统将优先使用 tf-kernel 的 sageattention 实现。这提供了更好的性能和与 TeleFuser 内核库的集成。

### 安装 Sparge Attention

要使用 `SPARGE_ATTN` 后端或径向注意力中的稀疏 sage attention，需要从源码安装 `spas_sage_attn`：

```bash
git clone https://github.com/spa-lab/spas-sage-attn.git
cd spas-sage-attn
pip install -e .
```

**`spas_sage_attn` 的要求：**
- CUDA 12.0+
- PyTorch 2.0+
- 兼容 SM80 (A100), SM86 (RTX 3090), SM89 (RTX 4090), SM90 (H100)

**替代方案**：如果 `spas_sage_attn` 不可用，系统将自动回退到 `sparse_sageattn`（如果已安装）：

```bash
pip install sparse-sageattn
```

## 安装教程

本节提供 TeleFuser 中不同注意力后端的安装说明。

### Flash Attention

Flash Attention 提供内存高效的注意力实现，具有硬件优化。

#### Flash Attention 2

Flash Attention 2 推荐用于大多数 GPU 架构（Ampere SM80、Ada Lovelace SM89 和部分 Hopper SM90）。

```bash
# 从 PyPI 安装（推荐）
pip install flash-attn --no-build-isolation

# 或从源码编译以获得特定优化
pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation
```

**要求：**
- CUDA 11.6+
- PyTorch 2.0+
- 计算能力 8.0+ 的 GPU（A100、RTX 3090/4090、H100 等）

#### Flash Attention 3

Flash Attention 3 专门针对 Hopper (H100) GPU 优化。

```bash
pip install flash-attn-interface
```

**要求：**
- H100 GPU (SM90)
- CUDA 12.0+
- PyTorch 2.2+

#### Flash Attention 4

Flash Attention 4（Cute 接口）针对 Hopper (SM90) 和 Blackwell (SM100+) GPU 优化。

```bash
# 从源码安装（包含 cute 子模块）
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation
```

**要求：**
- H100 (SM90) 或 B100/B200/RTX 5090 (SM100+)
- CUDA 12.8+
- PyTorch 2.4+

**注意**：Flash Attention 4 可能没有预编译的 wheel，建议从源码编译。

### SageAttention

SageAttention 提供量化注意力，使用 INT8 Q/K 量化以提高性能。

#### 方式一：通过 tf-kernel 安装（TeleFuser 推荐）

参见 [tf-kernel](#tf-kernel) 章节的安装说明。

tf-kernel 为以下架构提供 SageAttention 变体：
- **SM80 (A100)**: `sageattn_qk_int8_pv_fp16_cuda` - FP16 PV 累加
- **SM86 (RTX 3090)**: `sageattn_qk_int8_pv_fp16_triton` - Triton 实现
- **SM89 (RTX 4090)**: `sageattn_qk_int8_pv_fp8_cuda` - FP8 PV 累加
- **SM90 (H100)**: `sageattn_qk_int8_pv_fp8_cuda_sm90` - H100 优化版
- **SM100+ (Blackwell)**: `sageattn_qk_int8_pv_fp8_cuda` 使用 per-warp 量化

对于 Blackwell (SM100+) 的 FP4 注意力，使用 FP4 支持编译 tf-kernel：

```bash
TF_KERNEL_ENABLE_FP4=ON make build-sm100
```

#### 方式二：从官方源码安装

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
pip install -e .
```

**要求：**
- CUDA 11.8+
- PyTorch 2.0+
- 计算能力 8.0+ 的 GPU

### Radial Attention

Radial attention 是一种用于视频生成的稀疏注意力模式，可减少内存使用。

**依赖：**
- `flashinfer` 或 `tf-kernel`（带 sageattention）

#### 方式一：通过 tf-kernel 安装（TeleFuser 推荐）

参见 [tf-kernel](#tf-kernel) 章节的安装说明。

#### 方式二：从 FlashInfer 官方源码安装

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
pip install -e .
```

**要求：**
- CUDA 11.8+
- PyTorch 2.0+
- 计算能力 8.0+ 的 GPU

### Block Sparse Attention

用于局部稀疏注意力（`LOCAL_SPARSE_ATTN`）：

#### 方式一：通过 tf-kernel 安装（TeleFuser 推荐）

参见 [tf-kernel](#tf-kernel) 章节的安装说明。

#### 方式二：从官方源码安装

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
pip install -e .
```

**要求：**
- CUDA 11.6+
- PyTorch 2.0+
- 计算能力 8.0+ 的 GPU

### tf-kernel

`tf-kernel` 是 TeleFuser 的可选 CUDA 扩展包，提供优化的 SageAttention 和块稀疏注意力实现。从当前配置
的包索引安装已发布版本：

```bash
python -m pip install -e ".[kernel]"
```

如需随 TeleFuser 编译仓库内源码：

```bash
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel
```

`kernel` extra 不会编译 `tf-kernel/`。需要指定架构源码编译时，在 `tf-kernel/` 目录执行：

```bash
make build-auto PYTHON=/path/to/venv/bin/python
make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere 和 Ada
make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper
make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell
```

源码编译要求 Python 3.10+、PyTorch 2.11.0、CUDA Toolkit 12.8+ 和 CMake 3.26+；FP4 内核仅在 SM100+
可用。完整兼容矩阵、H100 编译命令、安装验证、使用示例和常见问题见
[tf-kernel 安装与使用指南](./tf_kernel.md)。

当前验证的 H100 wheel 在架构选择的 SM90 SageAttention 路径存在已知 `misaligned address` 错误；部署的
wheel 通过专项 SM90 GPU 测试前，应选择其他注意力后端。

### 检查可用后端

安装后，验证哪些后端可用：

```python
from telefuser.ops.attention.attention_impl import (
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_4_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    FLASHINFER_AVAILABLE,
)

print(f"Flash Attention 2: {FLASH_ATTN_2_AVAILABLE}")
print(f"Flash Attention 3: {FLASH_ATTN_3_AVAILABLE}")
print(f"Flash Attention 4: {FLASH_ATTN_4_AVAILABLE}")
print(f"Sage Attention: {SAGE_ATTN_AVAILABLE}")
print(f"FlashInfer: {FLASHINFER_AVAILABLE}")
```

### 快速安装汇总

| 后端 | 安装命令 | GPU 支持 |
|------|----------|----------|
| Flash Attention 2 | `pip install flash-attn --no-build-isolation` | SM80+ (A100, RTX 3090/4090, H100) |
| Flash Attention 3 | `pip install flash-attn-interface` | SM90 (H100) |
| Flash Attention 4 | 从源码编译（cute 接口） | SM90+ (H100, B100/B200) |
| SageAttention | tf-kernel 或 [官方源码](https://github.com/thu-ml/SageAttention) | SM80+ |
| Radial Attention | tf-kernel 或 [FlashInfer 源码](https://github.com/flashinfer-ai/flashinfer) | SM80+ |
| Block Sparse | tf-kernel 或 [官方源码](https://github.com/mit-han-lab/Block-Sparse-Attention) | SM80+ |
| Sparge Attention | 从源码安装（见上文） | SM80, SM86, SM89, SM90 |

## 示例

### 示例 1: 使用 Flash Attention 2 的密集注意力

```python
from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

config = Wan21VideoPipelineConfig()
config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)
```

### 示例 2: 长视频的径向注意力

```python
from telefuser.core.config import AttentionConfig
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

config = Wan21VideoPipelineConfig()
config.dit_config.attention_config = AttentionConfig.radial_attention(
    dense_timesteps=40,
    dense_layers=2,
    decay_factor=1.0,
)
```

### 示例 3: H100 上的 Sage Attention

```python
from telefuser.core.config import AttentionConfig, AttnImplType

config = AttentionConfig.dense_attention(
    AttnImplType.SAGE_ATTN_2_8_8_SM90  # 针对 SM90 (H100) 优化
)
```

## 长上下文注意力

TeleFuser 支持跨多 GPU 的分布式注意力处理长序列。提供三种策略：

### 策略

| 策略 | 描述 | GPU 要求 | 通信方式 |
|------|------|----------|----------|
| **Ulysses** | 基于 All-to-All 的序列并行 | 2+ GPU | 对 heads 做 All-to-All |
| **Ring** | 基于 P2P 的序列并行 | 2+ GPU | P2P 传递 KV |
| **USP** | Ulysses + Ring 组合 | 4+ GPU (ring×ulysses) | 两者都用 |

### Ulysses 注意力

将序列分割到各 GPU，使用 All-to-All 重新分配 heads：

```
输入: (B, S_LOCAL, H_GLOBAL, D)
  -> All-to-All QKV -> (B, S_GLOBAL, H_LOCAL, D)
  -> 本地注意力计算
  -> All-to-All O -> (B, S_LOCAL, H_GLOBAL, D)
```

### Ring 注意力

在 GPU 环中轮转 KV 块，使用在线 softmax 合并结果：

```
对于环中每一步：
  1. 使用当前 KV 计算本地注意力
  2. 发送 KV 到下一个 GPU，从上一个接收
  3. 使用在线 softmax 合并注意力输出
```

**注意**: Ring 注意力需要支持 log-sum-exp (lse) 的注意力实现以进行在线 softmax 合并。支持 Flash Attention（2、3 或 4）和 SageAttention。

### USP (Ulysses + Ring)

组合两种策略以支持更大规模：

```
1. Ulysses All-to-All: (B, S_LOCAL, H_GLOBAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
2. 对聚合序列做 Ring 注意力
3. Ulysses All-to-All: (B, S_GLOBAL, H_LOCAL, D) -> (B, S_LOCAL, H_GLOBAL, D)
```

### 配置

```python
from telefuser.core.config import ParallelConfig
from telefuser.distributed import create_device_mesh_from_config
from telefuser.ops.attention.attention_impl import long_context_attention

# Ulysses: 2 GPU
config = ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2)

# Ring: 2 GPU（需要 Flash Attention）
config = ParallelConfig(device_ids=[0, 1], sp_ring_degree=2)

# USP: 4 GPU (ring=2, ulysses=2)
config = ParallelConfig(device_ids=[0, 1, 2, 3], sp_ring_degree=2, sp_ulysses_degree=2)

device_mesh = create_device_mesh_from_config(config)

output = long_context_attention(q, k, v, device_mesh=device_mesh)
```

### Device Mesh 工具函数

```python
from telefuser.distributed import (
    get_attention_strategy,      # 返回 "local", "ulysses", "ring" 或 "usp"
    get_ulysses_group,           # 获取 Ulysses 进程组
    get_ring_group,              # 获取 Ring 进程组
    get_ulysses_world_size,      # 获取 Ulysses 度数
    get_ring_world_size,         # 获取 Ring 度数
)

strategy = get_attention_strategy(device_mesh)
```

## 异步 Ulysses 注意力 (async_usp_forward)

`async_usp_forward` 是一种高效的 Ulysses 注意力实现，使用异步 All-to-All 通信来重叠计算和通信，从而提高性能。

### 原理

标准的 Ulysses 注意力需要等待所有 All-to-All 操作完成后才能进行计算。而 `async_usp_forward` 使用异步通信：

```
1. 发起 Q 的 All-to-All 异步操作
2. 发起 K 的 All-to-All 异步操作
3. 发起 V 的 All-to-All 异步操作
4. 等待所有操作完成
5. 计算注意力
6. 发起 O 的 All-to-All 异步操作
7. 等待完成
```

### 使用方式

在模型中启用 USP 后，`async_usp_forward` 会自动被调用：

```python
# 启用 Ulysses 序列并行
dit.enable_usp()

# 前向传播时会自动使用 async_usp_forward
output = dit(x, timestep, context, ...)
```

### 实现示例

以下是 `async_usp_forward` 的典型实现模式（来自 `wan_video_dit.py`）：

```python
def async_usp_forward(self, x, freqs, sparse_state=None, device_mesh=None):
    # 注意：此方法仅支持 Ulysses-style SP
    from telefuser.distributed.ulysses_comm import (
        ulysses_scatter_heads,
        ulysses_gather_heads,
    )
    from telefuser.distributed.device_mesh import get_ulysses_group

    group = get_ulysses_group(device_mesh)

    # 计算 Q, K, V
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    # 应用 RoPE
    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)

    # 重塑为 (B, S, H, D)
    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

    # 异步 All-to-All QKV
    q_wait = ulysses_scatter_heads(q, group)
    k_wait = ulysses_scatter_heads(k, group)
    v_wait = ulysses_scatter_heads(v, group)

    # 等待完成
    q = q_wait()
    k = k_wait()
    v = v_wait()

    # 计算注意力
    x = attention(q, k, v, input_layout="BSND", output_layout="BSND")

    # 异步 All-to-All 输出
    out_wait = ulysses_gather_heads(x, group, num_heads=self.num_heads)
    out = out_wait()

    # 重塑并应用输出投影
    out = rearrange(out, "b s n d -> b s (n d)", n=self.num_heads)
    return self.o(out)
```

### 支持的模型

| 模型 | async_usp_forward | 说明 |
|------|-------------------|------|
| `WanVideoDiT` | ✅ | 视频生成模型 |
| `QwenImageDiT` | ✅ | 图像生成模型（双流注意力） |
| `FlashVSRDiT` | ✅ | 视频超分辨率模型 |
| `ZImageDiT` | ❌ | 暂不支持 |

### 双流注意力 (QwenImageDiT)

`QwenImageDiT` 使用双流注意力，同时处理图像和文本流：

```python
def async_usp_forward(self, image, text, image_rotary_emb, attention_mask, device_mesh):
    group = get_ulysses_group(device_mesh)
    seq_txt = text.shape[1]

    # 计算图像和文本的 Q, K, V
    img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
    txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)

    # 拼接为联合注意力
    joint_q = torch.cat([txt_q, img_q], dim=1)
    joint_k = torch.cat([txt_k, img_k], dim=1)
    joint_v = torch.cat([txt_v, img_v], dim=1)

    # 异步 All-to-All
    joint_q_wait = ulysses_scatter_heads(joint_q, group)
    joint_k_wait = ulysses_scatter_heads(joint_k, group)
    joint_v_wait = ulysses_scatter_heads(joint_v, group)

    # ... 计算联合注意力并分割输出
```

### 通信原语

`async_usp_forward` 使用以下通信原语（定义在 `telefuser/distributed/ulysses_comm.py`）：

| 函数 | 描述 |
|------|------|
| `ulysses_scatter_heads(x, group)` | 将 head 维度分散到各 rank，收集 sequence 维度 |
| `ulysses_gather_heads(x, group, num_heads)` | 将 head 维度从各 rank 收集，分散 sequence 维度 |

这些原语返回一个 waitable 对象，调用 `()` 会阻塞直到操作完成。

### 与 long_context_attention 的区别

| 特性 | async_usp_forward | long_context_attention |
|------|-------------------|------------------------|
| 支持策略 | 仅 Ulysses | Ulysses, Ring, USP |
| 通信方式 | 异步 All-to-All | 同步通信 |
| 计算通信重叠 | ✅ 支持 | ❌ 不支持 |
| 使用场景 | 模型内部优化 | 通用长上下文 API |

## 故障排除

### 警告："Sparse attention requires sparse_state, falling back to FLASH_ATTN_2"

发生在：
1. 使用径向注意力但 `sparse_state` 为 `None`
2. 在密集时间步（早期时间步使用密集注意力）
3. 在密集层（早期层使用密集注意力）

**解决方案**: 这是预期行为。代码自动回退到密集注意力。如需要，确保在需要时正确初始化 `sparse_state`。

### 检查可用后端

```python
from telefuser.ops.attention.attention_impl import (
    FLASH_ATTN_2_AVAILABLE,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_4_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
    RADIAL_ATTN_AVAILABLE,
)

print(f"Flash Attention 2: {FLASH_ATTN_2_AVAILABLE}")
print(f"Flash Attention 3: {FLASH_ATTN_3_AVAILABLE}")
print(f"Flash Attention 4: {FLASH_ATTN_4_AVAILABLE}")
print(f"Sage Attention: {SAGE_ATTN_AVAILABLE}")
print(f"Radial Attention: {RADIAL_ATTN_AVAILABLE}")
```
