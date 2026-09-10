# Ops Module Documentation

This document introduces the `ops` module of TeleFuser, providing efficient neural network operator implementations for video generation.

## Architecture Principles

TeleFuser follows a strict layered architecture for operations:

```
┌─────────────────────────────────────────────────────────────┐
│                      models/                                 │
│  (DiT, VAE, text encoders - ONLY import from ops/)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       ops/                                   │
│  (Compile-aware dispatch: native for compile, kernel for    │
│   eager mode. Base classes: CustomOp, CustomOpFunction)     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   kernel/triton/                             │
│  (Pure Triton kernels, custom ops. NOT directly used by     │
│   models. May have torch.library.custom_op registration.)   │
└─────────────────────────────────────────────────────────────┘
```

### Key Rules

1. **models/** layer MUST only import from `telefuser.ops/`:
   ```python
   # ✅ Correct
   from telefuser.ops.normalization import RMSNorm, LayerNorm, modulate
   from telefuser.ops.rotary import apply_rotary_emb
   from telefuser.ops.attention import attention

   # ❌ Wrong - Never import from kernel layer in models
   from telefuser.kernel.triton import apply_rotary_embedding
   from telefuser.kernel.triton import fused_scale_shift
   ```

2. **ops/** layer handles compile-aware dispatch:
   ```python
   # In ops/normalization.py
   class RMSNorm(CustomOp):
       def forward(self, x):
           if torch.compiler.is_compiling():
               return self.forward_native(x)  # PyTorch native
           return self.forward_cuda(x)  # Triton kernel
   ```

3. **kernel/triton/** contains pure Triton code:
   - No compile-state checks (handled by ops layer)
   - May use `torch.library.custom_op` for torch.compile compatibility
   - Only called by ops/ layer, never directly by models/

### Why This Architecture?

- **torch.compile compatibility**: ops layer dispatches to native PyTorch when compiling, allowing Inductor to fuse operations across layers
- **Performance**: ops layer uses optimized Triton kernels in eager mode
- **Separation of concerns**: kernel layer focuses on pure kernel implementation, ops layer handles dispatch logic

On AMD ROCm, dispatch reuses the `forward_cuda` Triton path when no `forward_rocm` kernel is defined. See
[Hardware Platforms](platforms.md) for the full per-platform dispatch behavior and `tf-kernel` gating.

### torch.compile Strategy by Operator Type

TeleFuser uses a **mixed strategy** for torch.compile compatibility, optimizing based on operator characteristics:

| Operator Type | Strategy | Reason |
|:--------------|:---------|:-------|
| **Attention** (High compute density) | `@torch.compiler.disable` | FlashAttention/SageAttention outperform native PyTorch significantly; fusion gains limited |
| **RoPE** (Medium compute density) | `@torch.compiler.disable` | Triton kernel outperforms native; subsequent Attention blocks fusion anyway |
| **RMSNorm/LayerNorm** (Low compute density) | Native in compile mode | Overhead-bound; Inductor can fuse with adjacent ops for better gains |
| **modulate** (Point operations) | Native in compile mode | Minimal compute; Inductor auto-fusion optimal |

**Execution flow example**:
```
Linear → RMSNorm(q_norm) → RoPE → Attention
                      ↑        ↑         ↑
               Native+Fuse  Triton    Triton (disabled)
```

Since Attention uses `@torch.compiler.disable`, fusion beyond RoPE is blocked anyway. Therefore, RoPE uses Triton kernel to maximize single-operator performance.

## Overview

The `telefuser/ops` module contains the following core components:

| Module | Description | File |
|--------|-------------|------|
| Activations | GELU, SiLU, GEGLU, SwiGLU, etc. | `activations.py` |
| Feed-Forward Networks | Configurable FFN implementations | `ffn.py` |
| Normalization Layers | RMSNorm, LayerNorm, AdaLayerNorm | `normalization.py` |
| Quantized Linear Layers | FP8 quantized linear layers | `quantized_linear.py` |
| Attention | Dense and sparse attention implementations | `attention/` |

## Activations (`activations.py`)

### Standard Activation Functions

```python
from telefuser.ops.activations import get_activation

# Get standard activation functions
silu = get_activation("silu")
gelu = get_activation("gelu")
mish = get_activation("mish")
```

### FP32SiLU

FP32 version of SiLU activation for numerical stability:

```python
from telefuser.ops.activations import FP32SiLU

activation = FP32SiLU()
output = activation(inputs)  # Internally converts to FP32 for computation
```

### Gated Linear Units

#### GELU

Standard GELU activation function with tanh approximation support:

```python
from telefuser.ops.activations import GELU

# Exact GELU
gelu = GELU(dim_in=1024, dim_out=4096, approximate="none")

# Tanh approximate GELU (faster)
gelu_approx = GELU(dim_in=1024, dim_out=4096, approximate="tanh")
```

#### GEGLU

Gated GELU unit, splits input and applies gating:

```python
from telefuser.ops.activations import GEGLU

geglu = GEGLU(dim_in=1024, dim_out=4096)
# Output: hidden_states * gelu(gate)
```

#### SwiGLU

Gated SiLU unit, similar to GEGLU but uses SiLU activation:

```python
from telefuser.ops.activations import SwiGLU

swiglu = SwiGLU(dim_in=1024, dim_out=4096)
# Output: hidden_states * silu(gate)
```

#### ApproximateGELU

Fast GELU approximation using sigmoid function:

```python
from telefuser.ops.activations import ApproximateGELU

approx_gelu = ApproximateGELU(dim_in=1024, dim_out=4096)
# Formula: x * sigmoid(1.702 * x)
```

### Activation Functions Reference

| Class | Formula | Reference |
|-------|---------|-----------|
| `GELU` | `GELU(x)` | [Gaussian Error Linear Units](https://huggingface.co/papers/1606.08415) |
| `GEGLU` | `x * GELU(gate)` | [GLU Variants](https://huggingface.co/papers/2002.05202) |
| `SwiGLU` | `x * SiLU(gate)` | [GLU Variants](https://huggingface.co/papers/2002.05202) |
| `ApproximateGELU` | `x * sigmoid(1.702x)` | [GELU Approximation](https://huggingface.co/papers/1606.08415) |

## Feed-Forward Networks (`ffn.py`)

### FeedForward

Configurable feed-forward network supporting multiple activation functions:

```python
from telefuser.ops.ffn import FeedForward

# Standard FFN (4x expansion)
ffn = FeedForward(dim=1024, mult=4, activation_fn="geglu")

# Custom hidden dimension
ffn = FeedForward(dim=1024, inner_dim=4096, activation_fn="swiglu")

# With dropout
ffn = FeedForward(dim=1024, dropout=0.1, final_dropout=True)
```

### Supported Activation Functions

| Activation Name | Description |
|-----------------|-------------|
| `"gelu"` | Standard GELU |
| `"gelu-approximate"` | Tanh approximate GELU |
| `"geglu"` | Gated GELU |
| `"geglu-approximate"` | Approximate gated GELU |
| `"swiglu"` | Gated SiLU |
| `"linear-silu"` | Linear projection + SiLU |

### Usage Example

```python
import torch
from telefuser.ops.ffn import FeedForward

# Create FFN
ffn = FeedForward(
    dim=1024,           # Input/output dimension
    mult=4,             # Hidden layer expansion multiplier
    dropout=0.0,        # Dropout probability
    activation_fn="geglu",  # Activation function
    bias=True,          # Whether to use bias
)

# Forward pass
x = torch.randn(2, 256, 1024)  # (batch, seq, dim)
output = ffn(x)
print(output.shape)  # (2, 256, 1024)
```

## Normalization Layers (`normalization.py`)

### RMSNorm

Root Mean Square Layer Normalization, more efficient than LayerNorm:

```python
from telefuser.ops.normalization import RMSNorm

# Create RMSNorm
norm = RMSNorm(dim=1024, eps=1e-5, elementwise_affine=True)

# Forward pass
output = norm(hidden_states)
```

**Performance Optimization**:
- Uses `tf_kernel.rmsnorm` on CUDA for best performance
- Falls back to Triton kernel
- Uses PyTorch implementation for non-CUDA tensors

### LayerNorm

Layer Normalization with Triton kernel optimization:

```python
from telefuser.ops.normalization import LayerNorm

# Create LayerNorm
norm = LayerNorm(dim=1024, eps=1e-6, elementwise_affine=True, bias=True)

# Forward pass
output = norm(hidden_states)
```

**Performance Optimization**:
- Uses Triton kernel on CUDA
- Falls back to `nn.functional.layer_norm` for non-CUDA tensors

### AdaLayerNormContinuous

Adaptive layer normalization with continuous conditioning:

```python
from telefuser.ops.normalization import AdaLayerNormContinuous

# Create adaptive normalization
ada_norm = AdaLayerNormContinuous(
    embedding_dim=1024,           # Normalization dimension
    conditioning_embedding_dim=256,  # Conditioning embedding dimension
    elementwise_affine=True,
    norm_type="layer_norm",  # or "rms_norm"
)

# Forward pass
x = torch.randn(2, 256, 1024)
cond = torch.randn(2, 256)
output = ada_norm(x, cond)
```

### modulate Function

Modulation function for adaptive normalization:

```python
from telefuser.ops.normalization import modulate

# Apply modulation: x * (1 + scale) + shift
output = modulate(x, shift, scale)
```

**Performance Optimization**: Uses Triton kernel's `fused_scale_shift` on CUDA.

### Normalization Layers Reference

| Class | Description | Kernel Optimization |
|-------|-------------|---------------------|
| `RMSNorm` | RMS normalization | tf_kernel > Triton > PyTorch |
| `LayerNorm` | Layer normalization | Triton > PyTorch |
| `AdaLayerNormContinuous` | Adaptive layer normalization | Uses LayerNorm or RMSNorm internally |

## Quantized Linear Layers (`quantized_linear.py`)

### LinearFP8

FP8 quantized linear layer for memory-efficient inference:

```python
import torch.nn as nn
from telefuser.ops.quantized_linear import LinearFP8

# Create from existing Linear layer
original_linear = nn.Linear(1024, 4096)
fp8_linear = LinearFP8(original_linear, data_type=torch.float8_e4m3fn)

# Forward pass
x = torch.randn(2, 256, 1024, device="cuda")
output = fp8_linear(x)
```

**Backend Support**:
- Requires `tf_kernel` for per-token quantization and scaled FP8 GEMM

### Model Quantization Tools

```python
from telefuser.ops.quantized_linear import replace_linear_layers, convert_params_to_buffers

# Replace all Linear layers with FP8 versions
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)

# Convert non-FP8 parameters to buffers (reduce memory overhead)
model = convert_params_to_buffers(model)
```

### Complete Quantization Example

```python
import torch
import torch.nn as nn
from telefuser.ops.quantized_linear import replace_linear_layers, convert_params_to_buffers

# Load model
model = load_my_model()
model = model.to("cuda")

# Replace Linear layers with FP8 versions
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)

# Convert parameters to buffers
model = convert_params_to_buffers(model)

# Inference
with torch.no_grad():
    output = model(input_tensor)
```

## Attention Module (`attention/`)

The attention module provides unified dense and sparse attention implementations. For detailed documentation, please refer to the [Attention Implementation Guide](./attention.md).

### Quick Reference

```python
from telefuser.ops.attention import attention, long_context_attention
from telefuser.core.config import AttentionConfig, AttnImplType

# Dense attention
config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
output = attention(q, k, v, attention_config=config)

# Sparse attention (radial)
config = AttentionConfig.radial_attention(dense_timesteps=40)
output = attention(q, k, v, attention_config=config, sparse_state=sparse_state)

# Long context attention (distributed)
output = long_context_attention(q, k, v, device_mesh=device_mesh)
```

### Module Structure

| File | Description |
|------|-------------|
| `attention_impl.py` | Unified attention implementation supporting multiple backends |
| `radial_attention_core.py` | Radial sparse attention core |
| `local_sparse_attn.py` | Local window sparse attention |
| `sparse_attention.py` | Sparse attention interface |

### Supported Attention Backends

| Backend | Type | Dependencies |
|---------|------|--------------|
| `TORCH_SDPA` | Dense | PyTorch 2.0+ |
| `TORCH_CUDNN` | Dense | cuDNN |
| `FLASH_ATTN_2/3/4` | Dense | flash-attn |
| `SAGE_ATTN_*` | Dense | sageattention |
| `RADIAL_ATTN` | Sparse | flashinfer / sageattention |
| `LOCAL_SPARSE_ATTN` | Sparse | block_sparse_attn |

## Using Ops in New Models

### Example: Custom Transformer Block

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
        
        # Attention
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
        
        # Adaptive modulation parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
    
    def forward(self, x, cond):
        # Adaptive modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(cond).chunk(6, dim=1)
        
        # Attention residual
        x = x + gate_msa.unsqueeze(1) * self.attention(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # FFN residual
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

### Example: Using Quantized Layers

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

# Create and quantize model
model = MyQuantizedModel(dim=1024).cuda()
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)
```

## Performance Optimization Tips

### 1. Choose the Right Attention Backend

| GPU | Recommended Backend |
|-----|---------------------|
| H100/B100+ | `FLASH_ATTN_4` or `SAGE_ATTN_2_8_8_SM90` |
| A100/RTX 4090 | `FLASH_ATTN_2` or `SAGE_ATTN_2_8_16` |
| Other CUDA GPUs | `TORCH_SDPA` or `FLASH_ATTN_2` |

### 2. Use FP8 Quantization to Reduce Memory

```python
# For large model inference
replace_linear_layers(model, quant_type=torch.float8_e4m3fn)
model = convert_params_to_buffers(model)
```

### 3. Use Sparse Attention for Long Videos

```python
# Radial attention can reduce 50%+ memory
config = AttentionConfig.radial_attention(
    dense_timesteps=40,  # Use dense attention for early timesteps
    decay_factor=1.0,
)
```

### 4. Use Long Context Attention for Distributed Training

```python
# Ulysses sequence parallelism
from telefuser.distributed import create_device_mesh_from_config
from telefuser.core.config import ParallelConfig

config = ParallelConfig(device_ids=[0, 1, 2, 3], sp_ulysses_degree=4)
device_mesh = create_device_mesh_from_config(config)

# Enable in model
dit.enable_usp()
```

## Related Documentation

- [Attention Implementation Guide](./attention.md) - Detailed attention module documentation
- [Adding New Models](./adding_new_model.md) - How to add new models
- [Parallel Processing](./parallel.md) - Distributed training guide
