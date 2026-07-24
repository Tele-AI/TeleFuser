# Attention Implementation Guide

This document describes the attention implementation architecture in TeleFuser, including configuration, calling flow, pipeline support status, and installation guides for different attention backends.

## Overview

TeleFuser provides a unified attention configuration interface that supports both dense and sparse attention implementations. The architecture consists of:

- **Configuration Layer**: `AttentionConfig`, `SparseAttentionConfig`, `AttnImplType`
- **Runtime State**: `SparseAttentionState` for sparse attention
- **Implementation Layer**: `attention()` function with multiple backends

## Configuration Classes

### AttnImplType

Enum defining all available attention implementations:

```python
class AttnImplType(Enum):
    # Dense attention
    TORCH_SDPA = auto()
    TORCH_CUDNN = auto()
    FLASH_ATTN_2 = auto()
    FLASH_ATTN_3 = auto()
    FLASH_ATTN_4 = auto()  # For Hopper (SM90) and Blackwell (SM100+) GPUs
    SAGE_ATTN_2_8_8 = auto()
    SAGE_ATTN_2_8_16 = auto()
    SAGE_ATTN_2_8_8_SM90 = auto()
    SPARGE_ATTN = auto()
    # Sparse attention
    RADIAL_ATTN = auto()
    LOCAL_SPARSE_ATTN = auto()
```

### AttentionConfig

Unified configuration for all attention types:

```python
@dataclass
class AttentionConfig:
    attn_impl: AttnImplType = AttnImplType.TORCH_SDPA
    sparse_config: SparseAttentionConfig | None = None
    scale: float | None = None
    dropout: float = 0.0
    is_causal: bool = False
```

Factory methods:
- `AttentionConfig.dense_attention(attn_impl)` - Create dense attention config
- `AttentionConfig.radial_attention(**kwargs)` - Create radial sparse attention config
- `AttentionConfig.local_sparse_attention(**kwargs)` - Create local sparse attention config

### SparseAttentionConfig

Configuration for sparse attention:

```python
@dataclass
class SparseAttentionConfig:
    sparse_impl: str | None = None           # "radial", "local", etc.
    dense_timesteps: int = 40               # Use dense attention for initial timesteps
    dense_layers: int = 0                   # Use dense attention for initial layers
    decay_factor: float = 1.0               # Decay factor for attention window
    local_window_size: int = 6              # Window size for local sparse attention
    block_size: int = 128                   # Block size for sparse computation
    use_sage_attention: bool = False        # Use sage attention backend
```

## Calling Flow

### 1. Configuration

```python
from telefuser.core.config import AttentionConfig, AttnImplType

# Dense attention
config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)

# Radial sparse attention
config = AttentionConfig.radial_attention(
    dense_timesteps=40,
    dense_layers=0,
    decay_factor=1.0,
)
```

### 2. Pipeline Configuration

```python
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

pipe_config = Wan21VideoPipelineConfig()
pipe_config.dit_config.attention_config = config
```

### 3. Model Initialization

The `ModelRuntimeConfig` contains `attention_config`:

```python
from telefuser.core.config import ModelRuntimeConfig

runtime_config = ModelRuntimeConfig()
runtime_config.attention_config = config  # Default: TORCH_SDPA dense
```

### 4. Model Setup

Pipeline stages pass config to models:

```python
# In SingleDitDenoisingStage.__init__
self.dit.set_attention_config(model_runtime_config.attention_config)
```

### 5. Attention Execution

Models call `attention()` with the config:

```python
from telefuser.ops.attention import attention

output = attention(
    q, k, v,
    attention_config=self.attention_config,
    sparse_state=sparse_state,  # Required for sparse attention
    input_layout="BSND",
    output_layout="BSND",
)
```

### 6. Sparse Attention State (for sparse only)

For sparse attention, runtime state tracks current step/layer:

```python
from telefuser.ops.attention import MaskMap, SparseAttentionState

# Create state
sparse_config = config.sparse_config
mask_map = MaskMap(video_token_num=3840, num_frame=16)
sparse_state = SparseAttentionState(sparse_config, mask_map, model_type="wan")

# Update per step
sparse_state.update(numeral_timestep=20, layer_idx=5)

# Check if should use dense
if sparse_state.should_use_dense():
    # Use dense attention
else:
    # Use sparse attention
```

## Pipeline Support Status

| Pipeline | Dense Attention | Sparse (Radial) | Notes |
|----------|-----------------|-----------------|-------|
| `Wan21VideoPipeline` | ✅ | ✅ | Full support for video generation |
| `Wan22VideoPipeline` | ✅ | ✅ | Full support for video generation |
| `QwenImagePipeline` | ✅ | ❌ | Image generation doesn't need temporal sparse attention |
| `ZImagePipeline` | ✅ | ❌ | Image generation doesn't need temporal sparse attention |

### Wan21VideoPipeline / Wan22VideoPipeline

Supports both dense and radial attention:

```python
# Radial attention for memory-efficient long video generation
from telefuser.core.config import AttentionConfig

config = AttentionConfig.radial_attention(
    dense_timesteps=40,      # Dense for first 40 timesteps
    dense_layers=0,          # Dense for first N layers
    decay_factor=1.0,        # Window decay factor
    use_sage_attention=False,
)
pipe_config.dit_config.attention_config = config
```

When using radial attention:
1. Pipeline calls `dit.enable_radial_attention()` in `__call__`
2. Creates `SparseAttentionState` with `MaskMap`
3. Updates state per timestep/layer in denoising loop
4. Automatically falls back to dense for early timesteps/layers

### QwenImagePipeline / ZImagePipeline

Supports only dense attention (image generation doesn't have temporal dimension):

```python
from telefuser.core.config import AttentionConfig, AttnImplType

config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_2)
pipe_config.dit_config.attention_config = config
```

## Implementation Details

### Dense Attention Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `TORCH_SDPA` | PyTorch scaled dot-product attention | PyTorch 2.0+ |
| `TORCH_CUDNN` | cuDNN attention backend | cuDNN |
| `FLASH_ATTN_2` | Flash Attention 2 | `flash_attn` package |
| `FLASH_ATTN_3` | Flash Attention 3 | `flash_attn_interface` package |
| `FLASH_ATTN_4` | Flash Attention 4 | `flash_attn` package (with `cute` submodule) |
| `SAGE_ATTN_*` | SageAttention variants | `sageattention` package |
| `SPARGE_ATTN` | Sparge Attention | `spas_sage_attn` package |

**Note on Flash Attention 4**: Flash Attention 4 is optimized for **Hopper (SM90, H100)** and **Blackwell (SM100+, B100/B200)** GPUs. It provides significant performance improvements on these architectures. For older GPUs (Ampere, Ada Lovelace), use Flash Attention 2 or 3 instead.

### Sparse Attention Backends

| Backend | Description | Requirements |
|---------|-------------|--------------|
| `RADIAL_ATTN` | Radial attention for video | `flashinfer` or `sageattention` (tf-kernel prioritized) |
| `LOCAL_SPARSE_ATTN` | Local window sparse attention | `block_sparse_attn` |

**Note on SageAttention Priority**: When `use_sage_attention=True` is set, the system will prioritize tf-kernel's sageattention implementation over the standalone `sageattention` package if both are available. This provides better performance and integration with the TeleFuser kernel library.

### Installing Sparge Attention

To use `SPARGE_ATTN` backend or sparse sage attention in radial attention, you need to install `spas_sage_attn` from source:

```bash
git clone https://github.com/spa-lab/spas-sage-attn.git
cd spas-sage-attn
pip install -e .
```

**Requirements for `spas_sage_attn`:**
- CUDA 12.0+
- PyTorch 2.0+
- Compatible with SM80 (A100), SM86 (RTX 3090), SM89 (RTX 4090), SM90 (H100)

**Alternative**: If `spas_sage_attn` is not available, the system will automatically fall back to `sparse_sageattn` (if installed):

```bash
pip install sparse-sageattn
```

## Installation Guide

This section provides installation instructions for different attention backends used in TeleFuser.

### Flash Attention

Flash Attention provides memory-efficient attention implementations with hardware optimization.

#### Flash Attention 2

Flash Attention 2 is recommended for most GPU architectures (Ampere SM80, Ada Lovelace SM89, and some Hopper SM90).

```bash
# Install from PyPI (recommended)
pip install flash-attn --no-build-isolation

# Or build from source for specific optimizations
pip install git+https://github.com/Dao-AILab/flash-attention.git --no-build-isolation
```

**Requirements:**
- CUDA 11.6+
- PyTorch 2.0+
- GPU with compute capability 8.0+ (A100, RTX 3090/4090, H100, etc.)

#### Flash Attention 3

Flash Attention 3 is optimized specifically for Hopper (H100) GPUs.

```bash
pip install flash-attn-interface
```

**Requirements:**
- H100 GPU (SM90)
- CUDA 12.0+
- PyTorch 2.2+

#### Flash Attention 4

Flash Attention 4 (Cute interface) is optimized for Hopper (SM90) and Blackwell (SM100+) GPUs.

```bash
# Install from source with cute submodule
git clone --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install . --no-build-isolation
```

**Requirements:**
- H100 (SM90) or B100/B200/RTX 5090 (SM100+)
- CUDA 12.8+
- PyTorch 2.4+

**Note**: Flash Attention 4 may not be available as a pre-built wheel. Building from source is recommended.

### SageAttention

SageAttention provides quantized attention with INT8 Q/K quantization for improved performance.

#### Option 1: Install via tf-kernel (Recommended for TeleFuser)

See [tf-kernel](#tf-kernel) section for installation instructions.

tf-kernel provides SageAttention variants for:
- **SM80 (A100)**: `sageattn_qk_int8_pv_fp16_cuda` - FP16 PV accumulation
- **SM86 (RTX 3090)**: `sageattn_qk_int8_pv_fp16_triton` - Triton implementation
- **SM89 (RTX 4090)**: `sageattn_qk_int8_pv_fp8_cuda` - FP8 PV accumulation
- **SM90 (H100)**: `sageattn_qk_int8_pv_fp8_cuda_sm90` - Optimized for H100
- **SM100+ (Blackwell)**: `sageattn_qk_int8_pv_fp8_cuda` with per-warp quantization

For FP4 attention on Blackwell (SM100+), build tf-kernel with FP4 support:

```bash
TF_KERNEL_ENABLE_FP4=ON make build-sm100
```

#### Option 2: Install from Official Source

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
pip install -e .
```

**Requirements:**
- CUDA 11.8+
- PyTorch 2.0+
- GPU with compute capability 8.0+

### Radial Attention

Radial attention is a sparse attention pattern for video generation that reduces memory usage.

**Dependencies:**
- `flashinfer` OR `tf-kernel` (with sageattention)

#### Option 1: Install via tf-kernel (Recommended for TeleFuser)

See [tf-kernel](#tf-kernel) section for installation instructions.

#### Option 2: Install FlashInfer from Official Source

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer
pip install -e .
```

**Requirements:**
- CUDA 11.8+
- PyTorch 2.0+
- GPU with compute capability 8.0+

### Block Sparse Attention

For local sparse attention (`LOCAL_SPARSE_ATTN`):

#### Option 1: Install via tf-kernel (Recommended for TeleFuser)

See [tf-kernel](#tf-kernel) section for installation instructions.

#### Option 2: Install from Official Source

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
pip install -e .
```

**Requirements:**
- CUDA 11.6+
- PyTorch 2.0+
- GPU with compute capability 8.0+

### tf-kernel

`tf-kernel` is TeleFuser's optional CUDA extension package and provides optimized SageAttention and block-sparse
attention implementations. Install the released package from the configured package index with:

```bash
python -m pip install -e ".[kernel]"
```

To compile the in-tree source together with TeleFuser instead:

```bash
PYTHON=/path/to/venv/bin/python scripts/install_dev.sh --kernel
```

The `kernel` extra does not compile `tf-kernel/`. For an architecture-specific source build, run one of these commands
from the `tf-kernel/` directory:

```bash
make build-auto PYTHON=/path/to/venv/bin/python
make build-sm80 PYTHON=/path/to/venv/bin/python   # Ampere and Ada
make build-sm90 PYTHON=/path/to/venv/bin/python   # Hopper
make build-sm100 PYTHON=/path/to/venv/bin/python  # Blackwell
```

Source builds require Python 3.10+, PyTorch 2.11.0, CUDA Toolkit 12.8+, and CMake 3.26+. FP4 kernels are available only
on SM100+. See the [tf-kernel installation and usage guide](./tf_kernel.md) for the complete compatibility matrix,
H100 build command, installation verification, usage examples, and troubleshooting.

The currently validated H100 wheel has a known `misaligned address` failure in the architecture-selected SM90
SageAttention path. Use another attention backend until the focused SM90 GPU test passes on the deployed wheel.

### Checking Available Backends

After installation, verify which backends are available:

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

### Quick Installation Summary

| Backend | Install Command | GPU Support |
|---------|----------------|-------------|
| Flash Attention 2 | `pip install flash-attn --no-build-isolation` | SM80+ (A100, RTX 3090/4090, H100) |
| Flash Attention 3 | `pip install flash-attn-interface` | SM90 (H100) |
| Flash Attention 4 | Build from source (cute interface) | SM90+ (H100, B100/B200) |
| SageAttention | tf-kernel or [official source](https://github.com/thu-ml/SageAttention) | SM80+ |
| Radial Attention | tf-kernel or [FlashInfer source](https://github.com/flashinfer-ai/flashinfer) | SM80+ |
| Block Sparse | tf-kernel or [official source](https://github.com/mit-han-lab/Block-Sparse-Attention) | SM80+ |
| Sparge Attention | Install from source (see above) | SM80, SM86, SM89, SM90 |

## Examples

### Example 1: Dense Attention with Flash Attention 2

```python
from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipelineConfig

config = Wan21VideoPipelineConfig()
config.dit_config.attention_config = AttentionConfig.dense_attention(
    AttnImplType.FLASH_ATTN_2
)
```

### Example 2: Radial Attention for Long Videos

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

### Example 3: Sage Attention on H100

```python
from telefuser.core.config import AttentionConfig, AttnImplType

config = AttentionConfig.dense_attention(
    AttnImplType.SAGE_ATTN_2_8_8_SM90  # Optimized for SM90 (H100)
)
```

## Long Context Attention

TeleFuser supports distributed attention for processing long sequences across multiple GPUs. Three strategies are available:

### Strategies

| Strategy | Description | GPU Requirement | Communication |
|----------|-------------|-----------------|---------------|
| **Ulysses** | All-to-All based sequence parallelism | 2+ GPUs | All-to-All on heads |
| **Ring** | P2P based sequence parallelism | 2+ GPUs | P2P for KV rotation |
| **USP** | Combined Ulysses + Ring | 4+ GPUs (ring×ulysses) | Both |

### Ulysses Attention

Splits sequence across GPUs and uses All-to-All to redistribute heads:

```
Input: (B, S_LOCAL, H_GLOBAL, D)
  -> All-to-All QKV -> (B, S_GLOBAL, H_LOCAL, D)
  -> Local attention
  -> All-to-All O -> (B, S_LOCAL, H_GLOBAL, D)
```

### Ring Attention

Rotates KV chunks through a ring of GPUs, using online softmax to merge results:

```
For each step in ring:
  1. Compute local attention with current KV
  2. Send KV to next GPU, receive from previous
  3. Merge attention outputs using online softmax
```

**Note**: Ring attention requires an attention implementation that supports log-sum-exp (lse) for online softmax merging. Flash Attention (2, 3, or 4) and SageAttention are supported.

### USP (Ulysses + Ring)

Combines both strategies for larger scale:

```
1. Ulysses All-to-All: (B, S_LOCAL, H_GLOBAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
2. Ring attention on gathered sequence
3. Ulysses All-to-All: (B, S_GLOBAL, H_LOCAL, D) -> (B, S_LOCAL, H_GLOBAL, D)
```

### Configuration

```python
from telefuser.core.config import ParallelConfig
from telefuser.distributed import create_device_mesh_from_config
from telefuser.ops.attention.attention_impl import long_context_attention

# Ulysses: 2 GPUs
config = ParallelConfig(device_ids=[0, 1], sp_ulysses_degree=2)

# Ring: 2 GPUs (requires Flash Attention)
config = ParallelConfig(device_ids=[0, 1], sp_ring_degree=2)

# USP: 4 GPUs (ring=2, ulysses=2)
config = ParallelConfig(device_ids=[0, 1, 2, 3], sp_ring_degree=2, sp_ulysses_degree=2)

device_mesh = create_device_mesh_from_config(config)

output = long_context_attention(q, k, v, device_mesh=device_mesh)
```

### Device Mesh Utilities

```python
from telefuser.distributed import (
    get_attention_strategy,      # Returns "local", "ulysses", "ring", or "usp"
    get_ulysses_group,           # Get Ulysses process group
    get_ring_group,              # Get Ring process group
    get_ulysses_world_size,      # Get Ulysses degree
    get_ring_world_size,         # Get Ring degree
)

strategy = get_attention_strategy(device_mesh)
```

## Asynchronous Ulysses Attention (async_usp_forward)

`async_usp_forward` is an efficient Ulysses attention implementation that uses asynchronous All-to-All communication to overlap computation and communication, thereby improving performance.

### Principle

Standard Ulysses attention requires waiting for all All-to-All operations to complete before computation. `async_usp_forward` uses asynchronous communication:

```
1. Initiate async All-to-All for Q
2. Initiate async All-to-All for K
3. Initiate async All-to-All for V
4. Wait for all operations to complete
5. Compute attention
6. Initiate async All-to-All for O
7. Wait for completion
```

### Usage

After enabling USP in the model, `async_usp_forward` is automatically called:

```python
# Enable Ulysses sequence parallelism
dit.enable_usp()

# async_usp_forward will be used automatically during forward pass
output = dit(x, timestep, context, ...)
```

### Implementation Example

Here's a typical implementation pattern for `async_usp_forward` (from `wan_video_dit.py`):

```python
def async_usp_forward(self, x, freqs, sparse_state=None, device_mesh=None):
    # Note: This method only supports Ulysses-style SP
    from telefuser.distributed.ulysses_comm import (
        ulysses_scatter_heads,
        ulysses_gather_heads,
    )
    from telefuser.distributed.device_mesh import get_ulysses_group

    group = get_ulysses_group(device_mesh)

    # Compute Q, K, V
    q = self.norm_q(self.q(x))
    k = self.norm_k(self.k(x))
    v = self.v(x)

    # Apply RoPE
    q = rope_apply(q, freqs, self.num_heads)
    k = rope_apply(k, freqs, self.num_heads)

    # Reshape to (B, S, H, D)
    q = rearrange(q, "b s (n d) -> b s n d", n=self.num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=self.num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=self.num_heads)

    # Async All-to-All for QKV
    q_wait = ulysses_scatter_heads(q, group)
    k_wait = ulysses_scatter_heads(k, group)
    v_wait = ulysses_scatter_heads(v, group)

    # Wait for completion
    q = q_wait()
    k = k_wait()
    v = v_wait()

    # Compute attention
    x = attention(q, k, v, input_layout="BSND", output_layout="BSND")

    # Async All-to-All for output
    out_wait = ulysses_gather_heads(x, group, num_heads=self.num_heads)
    out = out_wait()

    # Reshape and apply output projection
    out = rearrange(out, "b s n d -> b s (n d)", n=self.num_heads)
    return self.o(out)
```

### Supported Models

| Model | async_usp_forward | Notes |
|-------|-------------------|-------|
| `WanVideoDiT` | ✅ | Video generation model |
| `QwenImageDiT` | ✅ | Image generation model (dual-stream attention) |
| `FlashVSRDiT` | ✅ | Video super-resolution model |
| `ZImageDiT` | ❌ | Not supported yet |

### Dual-Stream Attention (QwenImageDiT)

`QwenImageDiT` uses dual-stream attention, processing both image and text streams:

```python
def async_usp_forward(self, image, text, image_rotary_emb, attention_mask, device_mesh):
    group = get_ulysses_group(device_mesh)
    seq_txt = text.shape[1]

    # Compute Q, K, V for image and text
    img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
    txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)

    # Concatenate for joint attention
    joint_q = torch.cat([txt_q, img_q], dim=1)
    joint_k = torch.cat([txt_k, img_k], dim=1)
    joint_v = torch.cat([txt_v, img_v], dim=1)

    # Async All-to-All
    joint_q_wait = ulysses_scatter_heads(joint_q, group)
    joint_k_wait = ulysses_scatter_heads(joint_k, group)
    joint_v_wait = ulysses_scatter_heads(joint_v, group)

    # ... compute joint attention and split output
```

### Communication Primitives

`async_usp_forward` uses the following communication primitives (defined in `telefuser/distributed/ulysses_comm.py`):

| Function | Description |
|----------|-------------|
| `ulysses_scatter_heads(x, group)` | Scatter heads across ranks, gather sequence dimension |
| `ulysses_gather_heads(x, group, num_heads)` | Gather heads from ranks, scatter sequence dimension |

These primitives return a waitable object; calling `()` will block until the operation completes.

### Comparison with long_context_attention

| Feature | async_usp_forward | long_context_attention |
|---------|-------------------|------------------------|
| Supported strategies | Ulysses only | Ulysses, Ring, USP |
| Communication | Async All-to-All | Synchronous |
| Compute-communication overlap | ✅ Supported | ❌ Not supported |
| Use case | Model-internal optimization | General long context API |

## Troubleshooting

### Warning: "Sparse attention requires sparse_state, falling back to FLASH_ATTN_2"

This occurs when:
1. Using radial attention but `sparse_state` is `None`
2. In dense timestep (early timesteps use dense attention)
3. In dense layer (early layers use dense attention)

**Solution**: This is expected behavior. The code automatically falls back to dense attention. To suppress the warning, ensure `sparse_state` is properly initialized when needed.

### Check Available Backends

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
