# Ops 模块文档

本文档介绍 TeleFuser 的 `ops` 模块，提供高效的视频生成神经网络算子实现。

## 架构原则

TeleFuser 遵循严格的分层架构：

```
┌─────────────────────────────────────────────────────────────┐
│                      models/                                 │
│  (DiT, VAE, text encoders - 只能从 ops/ 导入)               │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       ops/                                   │
│  (编译感知分发：compile 时用 native，eager 时用 kernel。     │
│   基类：CustomOp, CustomOpFunction)                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   kernel/triton/                             │
│  (纯 Triton 内核，custom ops。不直接被 models 使用。         │
│   可能有 torch.library.custom_op 注册)                       │
└─────────────────────────────────────────────────────────────┘
```

### 关键规则

1. **models/** 层只能从 `telefuser.ops/` 导入：
   ```python
   # ✅ 正确
   from telefuser.ops.normalization import RMSNorm, LayerNorm, modulate
   from telefuser.ops.rotary import apply_rotary_emb
   from telefuser.ops.attention import attention

   # ❌ 错误 - 模型中绝不要从 kernel 层导入
   from telefuser.kernel.triton import apply_rotary_embedding
   from telefuser.kernel.triton import fused_scale_shift
   ```

2. **ops/** 层负责编译感知分发：
   ```python
   # 在 ops/normalization.py 中
   class RMSNorm(CustomOp):
       def forward(self, x):
           if torch.compiler.is_compiling():
               return self.forward_native(x)  # PyTorch 原生实现
           return self.forward_cuda(x)  # Triton 内核
   ```

3. **kernel/triton/** 包含纯 Triton 代码：
   - 无编译状态检查（由 ops 层处理）
   - 可使用 `torch.library.custom_op` 以支持 torch.compile
   - 只被 ops 层调用，绝不直接被 models 调用

### 为什么采用此架构？

- **torch.compile 兼容性**：ops 层在编译时分发到 PyTorch 原生实现，让 Inductor 可以跨层融合操作
- **性能优化**：ops 层在 eager 模式下使用优化的 Triton 内核
- **关注点分离**：kernel 层专注纯内核实现，ops 层处理分发逻辑

AMD ROCm 上，未定义 `forward_rocm` 内核时分发会复用 `forward_cuda` 的 Triton 路径。各平台的完整分发行为与
`tf-kernel` 门控见[硬件平台](platforms.md)。

### 不同算子类型的 torch.compile 策略

TeleFuser 采用**混合策略**处理 torch.compile 兼容性，根据算子特性优化：

| 算子类型 | 策略 | 原因 |
|:---------|:-----|:-----|
| **Attention**（高计算密度） | `@torch.compiler.disable` | FlashAttention/SageAttention 性能远优于原生 PyTorch；融合收益有限 |
| **RoPE**（中等计算密度） | `@torch.compiler.disable` | Triton 内核优于原生实现；后续 Attention 已阻断融合 |
| **RMSNorm/LayerNorm**（低计算密度） | 编译时使用原生实现 | Overhead-bound；Inductor 可与相邻算子融合获得更好收益 |
| **modulate**（点操作） | 编译时使用原生实现 | 计算量极小；Inductor 自动融合最优 |

**执行流程示例**：
```
Linear → RMSNorm(q_norm) → RoPE → Attention
                      ↑        ↑         ↑
               原生+融合    Triton    Triton (disabled)
```

由于 Attention 使用了 `@torch.compiler.disable`，RoPE 之后融合已被阻断。因此 RoPE 使用 Triton 内核最大化单算子性能。

## 概述

`telefuser/ops` 模块包含以下核心组件：

| 模块 | 描述 | 文件 |
|------|------|------|
| 激活函数 | GELU、SiLU、GEGLU、SwiGLU 等 | `activations.py` |
| 前馈网络 | 可配置的 FFN 实现 | `ffn.py` |
| 归一化层 | RMSNorm、LayerNorm、AdaLayerNorm | `normalization.py` |
| 量化线性层 | FP8 量化线性层 | `quantized_linear.py` |
| 注意力 | 密集和稀疏注意力实现 | `attention/` |

## 激活函数 (`activations.py`)

### 标准激活函数

```python
from telefuser.ops.activations import get_activation

# 获取标准激活函数
silu = get_activation("silu")
gelu = get_activation("gelu")
mish = get_activation("mish")
```

### FP32SiLU

SiLU 激活函数的 FP32 版本，用于数值稳定性：

```python
from telefuser.ops.activations import FP32SiLU

activation = FP32SiLU()
output = activation(inputs)  # 内部转换为 FP32 计算
```

### 门控线性单元

#### GELU

标准 GELU 激活函数，支持 tanh 近似：

```python
from telefuser.ops.activations import GELU

# 精确 GELU
gelu = GELU(dim_in=1024, dim_out=4096, approximate="none")

# tanh 近似 GELU（更快）
gelu_approx = GELU(dim_in=1024, dim_out=4096, approximate="tanh")
```

#### GEGLU

门控 GELU 单元，将输入分割后应用门控：

```python
from telefuser.ops.activations import GEGLU

geglu = GEGLU(dim_in=1024, dim_out=4096)
# 输出: hidden_states * gelu(gate)
```

#### SwiGLU

门控 SiLU 单元，类似 GEGLU 但使用 SiLU 激活：

```python
from telefuser.ops.activations import SwiGLU

swiglu = SwiGLU(dim_in=1024, dim_out=4096)
# 输出: hidden_states * silu(gate)
```

#### ApproximateGELU

快速 GELU 近似，使用 sigmoid 函数：

```python
from telefuser.ops.activations import ApproximateGELU

approx_gelu = ApproximateGELU(dim_in=1024, dim_out=4096)
# 公式: x * sigmoid(1.702 * x)
```

### 激活函数对照表

| 类名 | 公式 | 参考文献 |
|------|------|----------|
| `GELU` | `GELU(x)` | [Gaussian Error Linear Units](https://huggingface.co/papers/1606.08415) |
| `GEGLU` | `x * GELU(gate)` | [GLU Variants](https://huggingface.co/papers/2002.05202) |
| `SwiGLU` | `x * SiLU(gate)` | [GLU Variants](https://huggingface.co/papers/2002.05202) |
| `ApproximateGELU` | `x * sigmoid(1.702x)` | [GELU Approximation](https://huggingface.co/papers/1606.08415) |

## 前馈网络 (`ffn.py`)

### FeedForward

可配置的前馈网络，支持多种激活函数：

```python
from telefuser.ops.ffn import FeedForward

# 标准 FFN（4倍扩展）
ffn = FeedForward(dim=1024, mult=4, activation_fn="geglu")

# 自定义隐藏维度
ffn = FeedForward(dim=1024, inner_dim=4096, activation_fn="swiglu")

# 带 dropout
ffn = FeedForward(dim=1024, dropout=0.1, final_dropout=True)
```

### 支持的激活函数

| 激活函数名 | 描述 |
|-----------|------|
| `"gelu"` | 标准 GELU |
| `"gelu-approximate"` | tanh 近似 GELU |
| `"geglu"` | 门控 GELU |
| `"geglu-approximate"` | 近似门控 GELU |
| `"swiglu"` | 门控 SiLU |
| `"linear-silu"` | 线性投影 + SiLU |

### 使用示例

```python
import torch
from telefuser.ops.ffn import FeedForward

# 创建 FFN
ffn = FeedForward(
    dim=1024,           # 输入/输出维度
    mult=4,             # 隐藏层扩展倍数
    dropout=0.0,        # dropout 概率
    activation_fn="geglu",  # 激活函数
    bias=True,          # 是否使用偏置
)

# 前向传播
x = torch.randn(2, 256, 1024)  # (batch, seq, dim)
output = ffn(x)
print(output.shape)  # (2, 256, 1024)
```

## 归一化层 (`normalization.py`)

### RMSNorm

Root Mean Square Layer Normalization，比 LayerNorm 更高效：

```python
from telefuser.ops.normalization import RMSNorm

# 创建 RMSNorm
norm = RMSNorm(dim=1024, eps=1e-5, elementwise_affine=True)

# 前向传播
output = norm(hidden_states)
```

**性能优化**：
- CUDA 上优先使用 `tf_kernel.rmsnorm`（最佳性能）
- 回退到 Triton 内核
- 非CUDA张量使用 PyTorch 实现

### LayerNorm

带 Triton 内核优化的 Layer Normalization：

```python
from telefuser.ops.normalization import LayerNorm

# 创建 LayerNorm
norm = LayerNorm(dim=1024, eps=1e-6, elementwise_affine=True, bias=True)

# 前向传播
output = norm(hidden_states)
```

**性能优化**：
- CUDA 上使用 Triton 内核
- 非CUDA张量回退到 `nn.functional.layer_norm`

### AdaLayerNormContinuous

自适应层归一化，支持连续条件输入：

```python
from telefuser.ops.normalization import AdaLayerNormContinuous

# 创建自适应归一化
ada_norm = AdaLayerNormContinuous(
    embedding_dim=1024,           # 归一化维度
    conditioning_embedding_dim=256,  # 条件嵌入维度
    elementwise_affine=True,
    norm_type="layer_norm",  # 或 "rms_norm"
)

# 前向传播
x = torch.randn(2, 256, 1024)
cond = torch.randn(2, 256)
output = ada_norm(x, cond)
```

### modulate 函数

调制函数，用于自适应归一化：

```python
from telefuser.ops.normalization import modulate

# 应用调制: x * (1 + scale) + shift
output = modulate(x, shift, scale)
```

**性能优化**：CUDA 上使用 Triton 内核的 `fused_scale_shift`。

### 归一化层对照表

| 类名 | 描述 | 核心优化 |
|------|------|----------|
| `RMSNorm` | RMS 归一化 | tf_kernel > Triton > PyTorch |
| `LayerNorm` | 层归一化 | Triton > PyTorch |
| `AdaLayerNormContinuous` | 自适应层归一化 | 内部使用 LayerNorm 或 RMSNorm |

## 量化线性层 (`quantized_linear.py`)

### LinearFP8

FP8 量化的线性层，用于内存高效推理：

```python
import torch.nn as nn
from telefuser.ops.quantized_linear import LinearFP8

# 从现有 Linear 层创建
original_linear = nn.Linear(1024, 4096)
fp8_linear = LinearFP8(original_linear, data_type=torch.float8_e4m3fn)

# 前向传播
x = torch.randn(2, 256, 1024, device="cuda")
output = fp8_linear(x)
```

**后端支持**：
- 使用 `tf_kernel` 执行逐 token 量化和 scaled FP8 GEMM

### 模型量化工具

```python
from telefuser.ops.quantized_linear import replace_linear_layers, convert_params_to_buffers

# 替换所有 Linear 层为 FP8 版本
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)

# 将非 FP8 参数转换为 buffer（减少内存开销）
model = convert_params_to_buffers(model)
```

### 完整量化示例

```python
import torch
import torch.nn as nn
from telefuser.ops.quantized_linear import replace_linear_layers, convert_params_to_buffers

# 加载模型
model = load_my_model()
model = model.to("cuda")

# 替换 Linear 层为 FP8 版本
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)

# 转换参数为 buffer
model = convert_params_to_buffers(model)

# 推理
with torch.no_grad():
    output = model(input_tensor)
```

## 注意力模块 (`attention/`)

注意力模块提供统一的密集和稀疏注意力实现。详细文档请参考 [Attention 实现指南](./attention.md)。

### 快速参考

```python
from telefuser.ops.attention import attention, long_context_attention
from telefuser.core.config import AttentionConfig, AttnImplType

# 密集注意力
config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
output = attention(q, k, v, attention_config=config)

# 稀疏注意力（径向）
config = AttentionConfig.radial_attention(dense_timesteps=40)
output = attention(q, k, v, attention_config=config, sparse_state=sparse_state)

# 长上下文注意力（分布式）
output = long_context_attention(q, k, v, device_mesh=device_mesh)
```

### 模块结构

| 文件 | 描述 |
|------|------|
| `attention_impl.py` | 统一注意力实现，支持多种后端 |
| `radial_attention_core.py` | 径向稀疏注意力核心 |
| `local_sparse_attn.py` | 局部窗口稀疏注意力 |
| `sparse_attention.py` | 稀疏注意力接口 |

### 支持的注意力后端

| 后端 | 类型 | 依赖 |
|------|------|------|
| `TORCH_SDPA` | 密集 | PyTorch 2.0+ |
| `TORCH_CUDNN` | 密集 | cuDNN |
| `FLASH_ATTN_2/3/4` | 密集 | flash-attn |
| `SAGE_ATTN_*` | 密集 | sageattention |
| `RADIAL_ATTN` | 稀疏 | flashinfer / sageattention |
| `LOCAL_SPARSE_ATTN` | 稀疏 | block_sparse_attn |

## 在新模型中使用 Ops

### 示例：自定义 Transformer Block

```python
import torch
import torch.nn as nn
from telefuser.ops.normalization import RMSNorm, modulate
from telefuser.ops.ffn import FeedForward
from telefuser.ops.attention import attention
from telefuser.core.config import AttentionConfig, AttnImplType

class MyTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attention_config: AttentionConfig = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # 注意力
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.num_heads = num_heads
        self.attention_config = attention_config or AttentionConfig.dense_attention(
            AttnImplType.FLASH_ATTN_2
        )
        
        # FFN
        self.ffn = FeedForward(
            dim=dim,
            mult=mlp_ratio,
            activation_fn="geglu",
        )
        
        # 自适应调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
    
    def forward(self, x, cond):
        # 自适应调制
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=1)
        
        # 注意力残差
        x = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # FFN 残差
        x = x + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        return x
    
    def attention(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(2)
        
        output = attention(
            q, k, v,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        
        return self.proj(output.flatten(2))
```

### 示例：使用量化层

```python
import torch
import torch.nn as nn
from telefuser.ops.quantized_linear import LinearFP8, replace_linear_layers

class MyQuantizedModel(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        return x

# 创建并量化模型
model = MyQuantizedModel(dim=1024).cuda()
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)
```

## 性能优化建议

### 1. 选择合适的注意力后端

| GPU | 推荐后端 |
|-----|----------|
| H100/B100+ | `FLASH_ATTN_4` 或 `SAGE_ATTN_2_8_8_SM90` |
| A100/RTX 4090 | `FLASH_ATTN_2` 或 `SAGE_ATTN_2_8_16` |
| 其他 CUDA GPU | `TORCH_SDPA` 或 `FLASH_ATTN_2` |

### 2. 使用 FP8 量化减少显存

```python
# 对于大模型推理
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)
model = convert_params_to_buffers(model)
```

### 3. 长视频使用稀疏注意力

```python
# 径向注意力可减少 50%+ 显存
config = AttentionConfig.radial_attention(
    dense_timesteps=40,  # 早期时间步使用密集注意力
    decay_factor=1.0,
)
```

### 4. 分布式训练使用长上下文注意力

```python
# Ulysses 序列并行
from telefuser.distributed import create_device_mesh_from_config
from telefuser.core.config import ParallelConfig

config = ParallelConfig(device_ids=[0, 1, 2, 3], sp_ulysses_degree=4)
device_mesh = create_device_mesh_from_config(config)

# 在模型中启用
dit.enable_usp()
```

## 相关文档

- [Attention 实现指南](./attention.md) - 注意力模块详细文档
- [添加新模型](./adding_new_model.md) - 如何添加新模型
- [并行处理](./parallel.md) - 分布式训练指南
