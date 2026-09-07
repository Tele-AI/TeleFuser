# LongCat Video Example

Long-form video generation using LongCat-Video model with extended context support.

## Model Source

| Platform | Link |
|----------|------|
| HuggingFace | [meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video) |
| ModelScope | [meituan-longcat/LongCat-Video](https://modelscope.cn/models/meituan-longcat/LongCat-Video) |

## Feature Support

| Feature | Support |
|---------|---------|
| CFG Parallel (CFGP) | ✔️ |
| Ulysses Sequence Parallel (USP) | ✔️ |
| LoRA | ✔️ |
| FP8 Quantization | ❌ |
| FSDP | ✔️ |
| Encoder Parallel | N/A |
| Async Pipeline | ✔️ |
| Feature Cache (AdaTaylor) | N/A |
| Distillation (cfg_step_lora) | ✔️ |
| Server API | ✔️ |
| Refinement (LoRA + BSA + Temporal) | ✔️ |
| BSA (Block Sparse Attention) | ✔️ |

## Additional Dependencies

| Model | Purpose | Link |
|-------|---------|------|
| Wan2.1 VAE | Video decoder | [Wan-AI/Wan2.1-T2V-14B](https://modelscope.cn/models/Wan-AI/Wan2___1-T2V-14B) |
| RIFE v4.26 | Video Frame Interpolation | [RIFEv4.26](https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip) |

## Files

### longcat_text_to_video.py

Text-to-Video generation example.

**Purpose:** Generate video from text prompt using LongCat-Video model.

**Usage:**
```bash
# Basic usage
python examples/longcat_video/longcat_text_to_video.py --prompt "A boat sailing on the ocean"

# Custom resolution
python examples/longcat_video/longcat_text_to_video.py --height 720 --width 1280

# Multi-GPU inference
python examples/longcat_video/longcat_text_to_video.py --gpu_num 2 --prompt "A beautiful scene"
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs (1, 2, 4, 8) |
| `--height` | 480 | Video height |
| `--width` | 832 | Video width |
| `--prompt` | (default prompt) | Text prompt |
| `--negative_prompt` | "" | Negative prompt |
| `--seed` | 42 | Random seed |

**Features:**
- Extended context for long video generation
- KV cache for efficient inference
- CPU offloading for memory efficiency

### longcat_image_to_video.py

Image-to-Video generation example.

**Purpose:** Generate video from a reference image and text prompt.

**Usage:**
```bash
# Basic usage
python examples/longcat_video/longcat_image_to_video.py \
    --image_path /path/to/image.jpg \
    --prompt "Make this image come alive"

# With custom resolution
python examples/longcat_video/longcat_image_to_video.py \
    --image_path /path/to/image.jpg \
    --prompt "Natural motion" \
    --resolution 720p
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs |
| `--image_path` | (default path) | Input image path |
| `--prompt` | (default prompt) | Text prompt |
| `--negative_prompt` | "" | Negative prompt |
| `--seed` | 42 | Random seed |
| `--resolution` | 720p | Target resolution |

**Features:**
- I2V with extended temporal coherence
- KV cache optimization

### longcat_video_continue.py

Video continuation example.

**Purpose:** Continue generating video from an existing video, enabling long-form video creation.

**Usage:**
```bash
python examples/longcat_video/longcat_video_continue.py \
    --input_video /path/to/video.mp4 \
    --prompt "Continue the scene naturally" \
    --num_frames 93
```

**Features:**
- Seamless video continuation
- Temporal consistency with previous frames
- VFI support for smooth output

### longcat_video_unify.py

Unified pipeline for multiple video tasks.

**Purpose:** Single pipeline supporting T2V, I2V, and video continuation.

**Usage:**
```bash
# Text-to-Video
python examples/longcat_video/longcat_video_unify.py --mode t2v --prompt "A sunset scene"

# Image-to-Video
python examples/longcat_video/longcat_video_unify.py --mode i2v --image_path /path/to/image.jpg --prompt "Make it move"

# Video continuation
python examples/longcat_video/longcat_video_unify.py --mode continue --input_video /path/to/video.mp4 --prompt "Continue"
```

**Features:**
- Unified interface for all tasks
- VFI integration for 24fps output
- Efficient KV cache utilization

### longcat_text_to_video_refine.py

Text-to-Video with official LongCat refinement (LoRA-based).

**Purpose:** Generate video at base resolution (e.g. 480p), then refine to higher resolution (e.g. 720p) using the official refinement LoRA.

**Usage:**
```bash
# 480p base -> 720p refined output
python examples/longcat_video/longcat_text_to_video_refine.py \
    --height 480 --width 832 \
    --refine_height 720 --refine_width 1280

# Multi-GPU with custom refine settings
python examples/longcat_video/longcat_text_to_video_refine.py \
    --gpu_num 2 --refine_num_steps 50 --refine_t_thresh 0.5
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gpu_num` | 1 | Number of GPUs (1, 2, 4, 8) |
| `--height` | 480 | Base generation height |
| `--width` | 832 | Base generation width |
| `--refine_height` | 720 | Refine target height |
| `--refine_width` | 1280 | Refine target width |
| `--refine_num_steps` | 50 | Refine denoising steps |
| `--refine_t_thresh` | 0.5 | Noise threshold [0,1] (denoising starts from t_thresh * 1000) |
| `--prompt` | (default prompt) | Text prompt |
| `--seed` | 42 | Random seed |

**Features:**
- Official LongCat refinement LoRA for high-quality upscaling
- No CFG during refinement (faster inference)
- Pixel-space bilinear upsampling + VAE re-encode + LoRA-guided denoising

## Performance

### Text-to-Video

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| T2V | H100*1 | 50 | 93 | 1280x720 | 1376.8 | 52.6 |
| T2V | H100*2 | 50 | 93 | 1280x720 | TBD | TBD |
| T2V | H100*4 | 50 | 93 | 1280x720 | TBD | TBD |

### Image-to-Video

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| I2V | H100*1 | 50 | 93 | 1280x720 | 1336.7 | 52.7 |
| I2V | H100*2 | 50 | 93 | 1280x720 | TBD | TBD |

### Video Continuation

| Config | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|-------|--------|------------|----------|---------------|
| Continue | H100*1 | 50 | 93 | 1280x720 | 1183.1 | 59.7 |

### Refinement Pipeline

Refinement pipeline supports multiple modes with different attention mechanisms and temporal extensions:

| Mode | Device | Steps | Frames | Resolution | Time (s) | Max VRAM (GB) | Size (MB) |
|------|--------|-------|--------|------------|----------|---------------|-----------|
| Base Only | H100*1 | 50 | 93 | 832x480 | 624.2 | 43.3 | 5.0 |
| Spatial Refine | H100*1 | 50 | 93 | 1280x720 | 741.9 | 52.7 | 12.3 |
| BSA Refine | H100*1 | 50 | 90 | 1280x704 | 679.9 | 52.3 | 11.1 |
| Temporal Refine | H100*1 | 50 | 185 | 1280x720 | 1023.0 | 63.8 | 23.3 |
| LoRA Isolation Test | H100*1 | 50 | 93 | 832x480 | 603.8 | 43.3 | 5.0 |
| Determinism Test | H100*1 | 50 | 93 | 1280x720 | 1479.4 | 52.7 | 12.3 |

**Refinement Modes:**
- **Base Only**: 480p generation without refinement
- **Spatial Refine**: Official LoRA-based spatial upscaling (480p → 720p)
- **BSA Refine**: Block-Sparse Attention for efficient high-res generation
- **Temporal Refine**: Extended temporal generation (93 → 185 frames)
- **LoRA Isolation**: Verifies no LoRA interference in base generation
- **Determinism**: Validates reproducibility with fixed seed

### Refinement (T2V 480p → 720p)

| Config | Device | Base Steps | Refine Steps | Frames | Base Resolution | Refine Resolution | Time (s) | Max VRAM (GB) |
|--------|--------|------------|--------------|--------|-----------------|-------------------|----------|---------------|
| T2V+Refine | H100*1 | 50 | 50 | 93 | 832x480 | 1280x720 | TBD | TBD |

## Notes

- LongCat-Video is optimized for long-form video generation
- Supports extended temporal context compared to standard video models
- VFI (Video Frame Interpolation) can be enabled for smoother output (15fps -> 24fps)
- KV cache significantly improves inference speed
- Multi-GPU supports CFG parallel and Ulysses sequence parallel
