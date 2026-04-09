# LiveAct Model Definition Analysis

## Model Architecture Overview

LiveAct extends the Wan I2V model with audio-conditioning capabilities for talking head video generation.

## Core Components

### 1. WanModel (DiT Backbone)

**Location**: `work_dirs/SoulX-LiveAct/model_liveact/model_memory.py`

**Architecture**:
```python
WanModel(
    model_type='i2v',
    patch_size=(1, 2, 2),
    text_len=512,
    in_dim=16,           # VAE latent channels
    dim=2048,            # Hidden dimension
    ffn_dim=8192,        # FFN intermediate dimension
    freq_dim=256,        # Time embedding dimension
    text_dim=4096,       # T5 embedding dimension
    out_dim=16,          # Output channels
    num_heads=16,        # Attention heads
    num_layers=40,       # Transformer layers
    audio_window=5,      # Audio window size
    output_dim=768,      # Audio cross-attention output dimension
    context_tokens=32,   # Audio context tokens per frame
)
```

**Submodules**:
- `patch_embedding`: Conv3d (16 → 2048, kernel=(1,2,2), stride=(1,2,2))
- `text_embedding`: Linear(4096 → 2048) + GELU + Linear(2048 → 2048)
- `time_embedding`: Linear(256 → 2048) + SiLU + Linear(2048 → 2048)
- `time_projection`: SiLU + Linear(2048 → 12288)  # 6 * 2048
- `blocks`: ModuleList of 40 `WanAttentionBlock`
- `head`: Output head with modulation
- `img_emb`: MLPProj(1280 → 2048)  # CLIP embedding projection
- `audio_proj`: AudioProjModel  # Audio embedding projection

### 2. WanAttentionBlock

**Components**:
- `norm1`: WanLayerNorm(2048)
- `self_attn`: WanSelfAttention (with KV cache)
- `norm3`: WanLayerNorm(2048, elementwise_affine=True)
- `cross_attn`: WanI2VCrossAttention (text + image)
- `norm2`: WanLayerNorm(2048)
- `ffn`: Linear(2048 → 8192) + GELU + Linear(8192 → 2048)
- `modulation`: Parameter(1, 6, 2048)  # Adaptive modulation
- `audio_cross_attn`: SingleStreamAttention  # Audio conditioning
- `norm_x`: WanLayerNorm(2048)  # For audio cross-attention input

### 3. WanSelfAttention

**Features**:
- QK normalization (RMSNorm)
- KV cache management with FP8 support
- Memory compression via Conv1d (kernel=5, stride=5)
- 3D RoPE (time, height, width)

**Parameters**:
```python
WanSelfAttention(
    dim=2048,
    num_heads=16,
    head_dim=128,
    q = Linear(2048 → 2048),
    k = Linear(2048 → 2048),
    v = Linear(2048 → 2048),
    o = Linear(2048 → 2048),
    norm_q = WanRMSNorm(2048),
    norm_k = WanRMSNorm(2048),
    memory_proj_k = Conv1d(2048, 2048, kernel=5, stride=5, groups=2048),
    memory_proj_v = Conv1d(2048, 2048, kernel=5, stride=5, groups=2048),
)
```

### 4. WanI2VCrossAttention

**Features**:
- Cross-attention for text conditioning
- Separate K/V projections for CLIP image features (257 tokens)

**Parameters**:
```python
WanI2VCrossAttention(
    dim=2048,
    num_heads=16,
    q = Linear(2048 → 2048),
    k = Linear(2048 → 2048),        # For text
    v = Linear(2048 → 2048),
    k_img = Linear(2048 → 2048),    # For CLIP image
    v_img = Linear(2048 → 2048),
    norm_q = WanRMSNorm(2048),
    norm_k = WanRMSNorm(2048),
    norm_k_img = WanRMSNorm(2048),
)
```

### 5. SingleStreamAttention (Audio Cross-Attention)

**Features**:
- Cross-attention for audio conditioning
- Per-frame audio embedding projection

**Parameters**:
```python
SingleStreamAttention(
    dim=2048,
    encoder_hidden_states_dim=768,  # Audio embedding dimension
    num_heads=16,
    q_linear = Linear(2048 → 2048),
    kv_linear = Linear(768 → 4096),  # 2 * 2048
    proj = Linear(2048 → 2048),
)
```

### 6. AudioProjModel

**Purpose**: Project wav2vec2 embeddings to audio cross-attention input

**Architecture**:
```python
AudioProjModel(
    seq_len=5,          # Audio window for first frame
    seq_len_vf=12,      # Audio window for subsequent frames
    blocks=12,          # Wav2vec2 hidden layers
    channels=768,       # Wav2vec2 hidden dimension
    intermediate_dim=512,
    output_dim=768,
    context_tokens=32,  # Tokens per frame
    proj1 = Linear(5*12*768 → 512),      # For first frame
    proj1_vf = Linear(12*12*768 → 512),  # For subsequent frames
    proj2 = Linear(512 → 512),
    proj3 = Linear(512 → 32*768),        # context_tokens * output_dim
)
```

### 7. Head

**Features**:
- Output projection with modulation
- Unpatchify to latent space

**Parameters**:
```python
Head(
    dim=2048,
    out_dim=16,
    patch_size=(1, 2, 2),
    norm = WanLayerNorm(2048),
    head = Linear(2048 → 64),  # 16 * 1 * 2 * 2
    modulation = Parameter(1, 2, 2048),
)
```

## 3D RoPE Implementation

```python
def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).div(dim))
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

# Combined 3D frequencies
freqs = torch.cat([
    rope_params(1024, d - 4*(d//6)),  # Time dimension
    rope_params(1024, 2*(d//6)),       # Height dimension
    rope_params(1024, 2*(d//6)),       # Width dimension
], dim=1)
```

## Normalization Layers

### WanRMSNorm
```python
class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
```

### WanLayerNorm
```python
class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, inputs):
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps
        ).to(inputs.dtype)
```

## KV Cache Management

### Memory Compression Strategy
1. **Conv1d Compression**: Use grouped Conv1d to compress KV cache
   - kernel_size=5, stride=5
   - Groups = dim (2048) for independent channel compression

2. **Mean Compression**: Alternative using mean pooling
   - Average every 5 frames into 1 compressed frame

### FP8 Quantization
```python
def _quantize_kv_tensor(self, kv):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = kv.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = torch.clamp(scale / fp8_max, min=1e-12)
    q_kv = (kv / scale.to(dtype=kv.dtype)).to(torch.float8_e4m3fn)
    return q_kv, scale
```

## Distributed Inference (model_memory_sp.py)

### Sequence Parallelism
- Uses `xfuser` for Ulysses-style sequence parallelism
- `xFuserLongContextAttention` for distributed attention
- `get_sequence_parallel_world_size()` and `get_sequence_parallel_rank()` for SP coordination

### Key Differences from Single-GPU Version
- `causal_rope_apply` takes `sp_size` and `sp_rank` parameters
- KV cache indices adjusted for SP rank
- Sequence parallel shard/unshard around attention