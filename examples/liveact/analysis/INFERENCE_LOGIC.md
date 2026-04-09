# LiveAct Inference Logic Analysis

## Forward Pass Flow

### 1. Input Preparation

```python
# Inputs to WanModel.forward()
x = [latent]               # List of latent tensors [B, C, T, H, W]
t = timestep               # Tensor [1] - current timestep
context = [text_embedding] # List of T5 embeddings [B, 512, 4096]
clip_fea = clip_context    # CLIP visual features [B, 257, 1280]
y = vae_latent            # VAE encoded latent with mask [B, 17, T', H', W']
audio = audio_embs        # Audio embeddings [B, T_audio, 5, 12, 768]
kv_cache = {...}          # KV cache dictionary
start_idx = ...           # Start index for KV cache
end_idx = ...             # End index for KV cache
```

### 2. Patch Embedding

```python
# Concatenate latent with VAE condition
x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

# Apply patch embedding (Conv3d)
x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
# Shape: [B, 2048, f, h, w]

# Flatten spatial dimensions
x = [u.flatten(2).transpose(1, 2) for u in x]
# Shape: [B, f*h*w, 2048]
```

### 3. Time Embedding

```python
# Sinusoidal time embedding
e = self.time_embedding(sinusoidal_embedding_1d(freq_dim, t))
# Shape: [B, 256]

# Time projection for modulation
e0 = self.time_projection(e).unflatten(1, (6, dim))
# Shape: [B, 6, 2048]
```

### 4. Text & Image Context

```python
# Text embedding
context = self.text_embedding(context)
# Shape: [B, 512, 2048]

# CLIP image embedding
context_clip = self.img_emb(clip_fea)
# Shape: [B, 257, 2048]

# Concatenate CLIP + text
context = torch.concat([context_clip, context], dim=1)
# Shape: [B, 769, 2048]
```

### 5. Audio Embedding Processing

```python
# Split audio into first frame and subsequent frames
first_frame_audio_emb_s = audio_cond[:, :1, ...]
latter_frame_audio_emb = audio_cond[:, 1:, ...]

# Rearrange for VAE-scale alignment
latter_frame_audio_emb = rearrange(
    latter_frame_audio_emb,
    "b (n_t n) w s c -> b n_t n w s c",
    n=vae_scale  # 4
)

# Process audio with AudioProjModel
audio_embedding = self.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)
# Shape: [B, T, 32, 768]

# Concatenate along context_tokens dimension
audio_embedding = torch.concat(audio_embedding.split(1), dim=2)
# Shape: [B, T, 32*?, 768]
```

### 6. Transformer Block Forward

For each block (40 layers):
```python
# Modulation
modulation = self.modulation + e  # [1, 6, 2048]
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)

# Self-attention with KV cache
y = self.self_attn(
    norm1(x) * (1 + scale_msa) + shift_msa,
    seq_lens, grid_sizes, freqs,
    kv_cache=kv_cache[layer_id],
    start_idx=start_idx,
    end_idx=end_idx,
    update_cache=update_cache
)
x = x + y * gate_msa

# Cross-attention for text
x = x + self.cross_attn(norm3(x), context)

# Audio cross-attention
x_a = self.audio_cross_attn(
    norm_x(x),
    encoder_hidden_states=audio_embedding,
    shape=grid_sizes[0],
    start_f=start_f
)
if start_f == 0:
    x_a[:, :frame_seqlen] = 0  # Zero out first frame
x = x + x_a

# FFN
y = self.ffn(norm2(x) * (1 + scale_mlp) + shift_mlp)
x = x + y * gate_mlp
```

### 7. Self-Attention with KV Cache

```python
def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache, start_idx, end_idx, update_cache):
    # Compute Q, K, V
    q = self.norm_q(self.q(x)).view(b, s, n, d)
    k = self.norm_k(self.k(x)).view(b, s, n, d)
    v = self.v(x).view(b, s, n, d)

    # Load KV cache
    k_cache, v_cache = self._load_kv_cache(kv_cache, device, dtype)

    # Update cache with memory compression
    if update_cache:
        # Compress old KV and shift
        k_cache[:, 2*frame_seqlen:3*frame_seqlen].copy_(
            k_compress(k_cache[:, 2*frame_seqlen:7*frame_seqlen])
        )
        # ... similar for v_cache

    # Update current KV
    if start_idx != 0:
        k_cache[:, 6*frame_seqlen:] = k
        v_cache[:, 6*frame_seqlen:] = v
    else:
        k_cache[:, :6*frame_seqlen] = k
        v_cache[:, :6*frame_seqlen] = v

    # Apply RoPE
    roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame)
    roped_key = causal_rope_apply(k_cache, grid_sizes, freqs, start_frame=0)

    # Compute attention
    x = attention(roped_query, roped_key[:, :end_idx], v_cache[:, :end_idx])

    # Store KV cache
    self._store_kv_cache(kv_cache, k_cache, v_cache)

    return self.o(x)
```

### 8. Audio Cross-Attention

```python
def forward(self, x, encoder_hidden_states, shape, start_f):
    # x: [B, N_t * N_h * N_w, 2048]
    # encoder_hidden_states: [B, T_total, 32, 768]

    # Reshape for per-frame processing
    x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=N_t)

    # Get Q from visual features
    q = self.q_linear(x).view(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
    # [B*num_heads, N, head_dim]

    # Get K, V from audio embeddings
    encoder_kv = self.kv_linear(encoder_hidden_states[start_f:start_f+B])
    encoder_k, encoder_v = encoder_kv.chunk(2, dim=-1)

    # Compute attention
    x = scaled_dot_product_attention(q, encoder_k, encoder_v)

    # Project output
    x = self.proj(x)
    return rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=N_t)
```

### 9. Output Head

```python
def forward(self, x, e):
    # Modulation for head
    e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)

    # Project to output
    x = self.head(self.norm(x) * (1 + e[1]) + e[0])
    # Shape: [B, f*h*w, 64]

    return x
```

### 10. Unpatchify

```python
def unpatchify(self, x, grid_sizes):
    # x: [B, f*h*w, 64]
    # grid_sizes: [B, 3] = [f, h, w]

    for u, v in zip(x, grid_sizes.tolist()):
        # Reshape to [f, h, w, 1, 2, 2, 16]
        u = u[:prod(v)].view(*v, *patch_size, c)

        # Rearrange to [16, f*1, h*2, w*2]
        u = torch.einsum('fhwpqrc->cfphqwr', u)
        u = u.reshape(c, *[i*j for i,j in zip(v, patch_size)])

    return out  # [B, 16, T, H, W]
```

## Denoising Loop

```python
# Fixed timesteps (3 steps)
timesteps = [1000.0, 937.5, 833.33333333, 0.0]

for i in range(len(timesteps) - 1):
    timestep = timesteps[i]

    # Forward pass
    noise_pred = wan_i2v_model(
        [latent],
        t=timestep,
        kv_cache=kv_cache[i],
        context=context,
        clip_fea=clip_context,
        audio=audio_embs,
        y=y_cut,
        start_idx=...,
        end_idx=...,
        update_cache=...,
    )[0]

    # Audio CFG (optional)
    if audio_cfg > 1.0 and i in [1, 2]:
        noise_pred_drop_audio = wan_i2v_model(
            ..., audio=torch.zeros_like(audio_embs), ...
        )[0]
        noise_pred = noise_pred_drop_audio + audio_cfg * (noise_pred - noise_pred_drop_audio)

    # Euler step
    dt = (timesteps[i] - timesteps[i+1]) / 1000
    latent = latent + (-noise_pred) * dt[0]
```

## Streaming Generation Pattern

```
Iteration 0 (first chunk):
- Generate 6 frames latent
- Decode 6 frames video
- Store latent for next iteration

Iteration 1+ (subsequent chunks):
- Generate 8 frames latent
- Concatenate last 3 frames from previous with current 5 frames
- Decode 8 frames video
- Skip first 9 frames (overlap with previous)
```

## Key Data Transformations

| Stage | Input Shape | Output Shape | Notes |
|-------|-------------|--------------|-------|
| VAE Encode | [B, 3, T, H, W] | [B, 16, T/4, H/8, W/8] | 4x temporal, 8x spatial |
| Patch Embed | [B, 16, T, H, W] | [B, T*H*W/patch, 2048] | patch_size=(1,2,2) |
| Self-Attn Q | [B, S, 2048] | [B, S, 16, 128] | 16 heads, 128 head_dim |
| Audio Proj | [B, T, 5, 12, 768] | [B, T, 32, 768] | 32 context tokens |
| Head Output | [B, S, 2048] | [B, S, 64] | 16*1*2*2=64 |
| Unpatchify | [B, S, 64] | [B, 16, T, H, W] | Reverse of patch embed |