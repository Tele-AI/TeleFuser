# LiveAct Integration Progress

## Overview
- **Model**: LiveAct (Audio-conditioned Image-to-Video Generation)
- **Type**: I2V with Audio Control (Talking Head Video Generation)
- **Source**: work_dirs/SoulX-LiveAct/generate.py
- **Started**: 2026-04-07

## Phase Status
| Phase | Status | Notes |
|-------|--------|-------|
| 1. Analyze | ✅ Complete | Created PIPELINE_LOGIC, MODEL_DEFINITION, INFERENCE_LOGIC |
| 2. Refactor | ✅ Complete | Created custom stages for original model interfaces |
| 3. Integrate | 🔄 In Progress | Testing pipeline with original model loading |
| 4. Cleanup | ⏳ Pending | |
| 5. Review | ⏳ Pending | |

## Files Created

### Pipeline Files (`telefuser/pipelines/liveact/`)
| File | Description |
|------|-------------|
| `__init__.py` | Package exports |
| `pipeline.py` | `LiveActPipeline` class |
| `text_encoding.py` | `LiveActTextEncodingStage` (wraps original T5) |
| `clip_encoding.py` | `LiveActClipEncodingStage` (wraps original CLIP) |
| `vae.py` | `LiveActVAEStage` (wraps original LightVAE) |
| `audio_encoding.py` | `AudioEncodingStage` + `AudioProjModel` |
| `denoising.py` | `LiveActDenoisingStage` with KV cache |

### Example Files (`examples/liveact/`)
| File | Description |
|------|-------------|
| `liveact_i2v_1gpu.py` | Single GPU example using original model initialization |

## Custom Stage Design

The custom stages wrap the original SoulX-LiveAct models:

1. **LiveActTextEncodingStage**: Wraps `wan.modules.t5.T5EncoderModel`
   - Interface: `text_encoder([prompt], device=device)` → returns list of tensors

2. **LiveActClipEncodingStage**: Wraps `wan.modules.clip.CLIPModel`
   - Interface: `clip.visual(video_list)` → returns [B, 257, 1280]

3. **LiveActVAEStage**: Wraps `lightx2v.models.video_encoders.hf.wan.vae.WanVAE`
   - Interface: `vae.encode(video)`, `vae.decode(latent)`

4. **AudioEncodingStage**: Handles Wav2Vec2 + AudioProjModel
   - Processes audio and projects to context tokens

## Key Findings

### Model Architecture
- **DiT Backbone**: WanModel (40 layers, 2048 dim, 16 heads) with audio cross-attention
- **VAE**: WanVAE (16 latent channels, 4x temporal / 8x spatial compression)
- **Text Encoder**: T5EncoderModel (UMT5-XXL, 512 token length)
- **Visual Encoder**: CLIP (XLM-Roberta-Large-ViT-Huge-14, 257 tokens)
- **Audio Encoder**: Wav2Vec2Model (wav2vec2-base, 768-dim embeddings)

### Unique Features (Not in Existing WanVideo Pipeline)
1. **Audio Cross-Attention**: `SingleStreamAttention` module in each block
2. **Audio Projection**: `AudioProjModel` projects wav2vec2 embeddings to 768-dim with 32 context tokens
3. **KV Cache with Memory Compression**: Conv1d (kernel=5, stride=5) for KV compression
4. **FP8 KV Cache Support**: Optional FP8 quantization for KV cache
5. **Streaming Video Generation**: Iterative generation with KV cache updates
6. **Audio CFG**: Classifier-free guidance for audio control

### Implementation Challenges
1. ✅ Implemented `LiveActDiT` with audio cross-attention - Using original WanModel directly
2. ✅ Handled streaming KV cache management - Implemented in LiveActDenoisingStage
3. ✅ Integrated audio encoding pipeline (wav2vec2) - AudioEncodingStage with AudioProjModel
4. State dict converter for LiveAct weights - Not needed, using original weights format

## Wrapper Class Handling
The original SoulX-LiveAct uses wrapper classes (not nn.Module):
- `T5EncoderModel` - wrapper with `self.model` (actual nn.Module)
- `CLIPModel` - wrapper with `self.model`
- `LightVAE` - wrapper with `self.model`

These required custom handling in stages:
- Remove `@with_model_offload` decorator
- Add setter methods (`set_text_encoder()`, `set_clip()`, `set_vae()`)
- Override `onload_models()`/`offload_models()` to access `.model` attribute

`WanModel` is a proper `nn.Module` (inherits from ModelMixin, ConfigMixin), so `@with_model_offload` works correctly.

## Component Integration Plan
| Component | Integration Method | Status |
|-----------|-------------------|--------|
| DiT (LiveAct) | Original WanModel (wrapper-compatible) | ✅ Works with @with_model_offload |
| VAE | Custom LiveActVAEStage (wrapper) | ✅ Implemented |
| Text Encoder | Custom LiveActTextEncodingStage (wrapper) | ✅ Implemented |
| CLIP | Custom LiveActClipEncodingStage (wrapper) | ✅ Implemented |
| Audio Encoder | AudioEncodingStage (wav2vec2 nn.Module) | ✅ Implemented |
| Audio Projection | AudioProjModel (nn.Module) | ✅ Implemented |

## Optimization Settings (Match Original SoulX-LiveAct)

| Optimization | Default | Description |
|--------------|---------|-------------|
| `enable_fp8_gemm` | True | FP8 for FFN linear layers |
| `enable_torch_compile` | True | torch.compile for DiT, VAE encode/decode |
| `fp8_kv_cache` | False | FP8 quantization for KV cache |
| `offload_cache` | True | Offload KV cache to CPU (single GPU) |
| `mean_memory` | False | Mean compression for KV cache |

### Memory Requirements (480x832)
| Configuration | GPU Memory | CPU Memory |
|---------------|------------|------------|
| Default (offload_cache=True) | ~10 GB | ~200 GB |
| FP8 KV + offload | ~10 GB | ~100 GB |
| No offload | ~210 GB | ~0 |

### Usage
```bash
python examples/liveact/liveact_i2v_1gpu.py \
    --ckpt_dir path/to/checkpoints \
    --wav2vec_dir path/to/wav2vec2 \
    --image image.jpg \
    --audio audio.wav \
    --output output.mp4

# Disable optimizations for debugging
python examples/liveact/liveact_i2v_1gpu.py \
    --disable_fp8_gemm \
    --disable_compile \
    ...
```