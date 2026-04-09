Now I have a complete understanding of the system. Let me analyze the execution flow with tensor shapes.

## 执行流程分析 (以 1.wav 为输入)

### 1. 初始化阶段

**参数解析:**
- `size = 416*720` → `width=720, height=416`
- `fps = 20`
- `vae_stride = (4, 8, 8)` (时间压缩4x, 空间压缩8x)
- `patch_size = (1, 2, 2)`
- `blksz_lst = [6, 8]` (两轮迭代生成的latent帧数)

**关键计算:**
```python
frame_len = (height // 16) * (width // 16) = (416/16) * (720/16) = 26 * 45 = 1170  # 每帧的token数
kv_cache_tokens = frame_len * sum(blksz_lst) // world_size = 1170 * 14 // 2 = 8190  # KV cache容量
frame_num = (sum(blksz_lst) - 1) * 4 + 1 = 13 * 4 + 1 = 53  # 初始视频帧数
```

**模型加载:**
- WanModel (14B): 主生成模型，40个Transformer blocks
- LightVAE: 视频编解码器
- CLIP: 图像特征提取 (输出 257×1280)
- T5: 文本编码 (输出 512×4096)
- Wav2Vec2: 音频编码 (12层隐藏状态，每层768维)

---

### 2. 输入预处理阶段

**图像处理:**
```python
image → center_rescale_crop → resize(416, 720)
cond_image shape: [1, 3, 1, 416, 720]  # B C T H W
```

**CLIP编码:**
```python
clip_context shape: [1, 257, 1280]  # 图像特征
```

**T5编码:**
```python
context shape: [1, 512, 2048]  # 文本特征 (经过text_embedding后)
```

**音频处理 (1.wav约27秒 @ 44.1kHz → 重采样到16kHz):**

由于 `--steam_audio` 开启，音频按分段处理:

```python
# 重采样策略 (fps=20)
rate = 25 / 20 = 1.25  # 慢放系数
audio_resampled shape: [1, samples] @ 16kHz

# Wav2Vec2编码 (generate.py:285)
audio_embedding shape: [T_frames, 12, 768]  
# T_frames = audio_duration * fps = 27 * 20 ≈ 540 帧
# 12 = wav2vec隐藏层数
# 768 = 每层维度
```

---

### 3. VAE编码阶段

**mask构建:**
```python
msk shape: [1, 4, 53, 26, 45]  # B 4 T_lat H_lat W_lat (bfloat16)
# 第一帧mask=1，其余=0，表示已知/生成区域
```

**VAE编码:**
```python
# padding_frames (包含cond_image)
padding_frames shape: [1, 3, 53, 416, 720]  # B C T H W

# VAE encode
y shape: [1, 16, 14, 26, 45]  # B C_lat T_lat H_lat W_lat
# 16=latent channels, 14=53/4+1 (时间维度压缩后)

# 拼接mask
y = concat([msk, y]) → [1, 20, 14, 26, 45]  # 4+16=20 channels
```

---

### 4. 迭代生成阶段 (核心流程)

**迭代次数计算:**
```python
iter_total_num = int(audio_len / (vae_stride[0] * blksz_lst[-1] / fps)) + 1
             = int(27 / (4 * 8 / 20)) + 1 ≈ 17 次
```

**每次迭代的流程:**

#### 第0轮迭代 (_=0, f=0, 生成6帧latent):

**音频分段提取:**
```python
# steam_audio模式下
audio_start_idx = 0
audio_end_idx = 53  # frame_num

# 从原始音频切片并重新编码
audio_slice = audio_ori[:, sr_ori*audio_start_idx/fps : sr_ori*audio_end_idx/fps]
              # 对于1.wav: slice ≈ 0 ~ 2.65秒

# 重新Wav2Vec2编码
audio_embedding shape: [53, 12, 768]  # 对当前视频段的音频编码

# 提取带窗口的音频embedding
audio_embs = get_audio_emb(audio_embedding, 0, 53, device)
indices = [-2, -1, 0, 1, 2] * 1  # 5帧窗口
audio_embs shape: [1, 53, 5, 12, 768]
# B=1, frames=53, window=5, layers=12, dim=768
```

**Latent初始化:**
```python
latent shape: [16, 6, 26, 45]  # C T_lat H_lat W_lat (bfloat16)
# 随机噪声，6帧latent对应24帧视频 (6*4=24)
```

**Diffusion去噪 (3步timesteps):**
```python
timesteps = [1000.0, 937.5, 833.33, 0.0]  # 4个值，3次迭代

# 每步:
y_cut = y[:, :, :14, ...]  # [1, 20, 14, 26, 45]

arg_c = {
    'context': context,          # [1, 512, 2048]
    'clip_fea': clip_context,    # [1, 257, 1280]
    'audio': audio_embs,         # [1, 53, 5, 12, 768]
    'y': y_cut[:, :, :6],        # [1, 20, 6, 26, 45] 对应6帧latent
    'start_idx': 0,
    'end_idx': 6 * frame_len,    # 6 * 1170 = 7020
}
```

**WanModel forward (model_memory_sp.py:992-1128):**

```python
# Patch embedding
x = patch_embedding(latent) → [1, 2048, 6, 13, 22]  
# 2048=dim, 6=T_patch, 13=26/2, 22=45/2 (spatial patch)

x flatten → [1, 1716, 2048]  # 1716 = 6*13*22

# Sequence Parallel切分
x = chunk(x, 2)[rank] → [1, 858, 2048]  # 每GPU一半tokens

# Audio projection (AudioProjModel)
audio_cond shape: [1, 53, 5, 12, 768]

# 分离首帧和后续帧音频
first_frame_audio shape: [1, 1, 5, 12, 768]
latter_frame_audio shape: [1, 52, 5, 12, 768]
                      → reshape为 [1, 13, 4, 5, 12, 768]  # 52帧分成13组，每组4帧(VAE scale)

# 提取各部分音频特征后投影
audio_embedding = audio_proj(first_frame_audio, latter_frame_audio)
                → [1, 53, 32, 768]  # B frames context_tokens dim

# 最终audio_embedding
audio_embedding shape: [1, 53*32, 768] = [1, 1696, 768]  # 合并后
```

**Transformer Block处理 (40层):**

```python
# WanAttentionBlock forward
x输入: [1, 858, 2048]  # 每GPU的tokens

# Self-Attention with KV Cache
# 第一次迭代 (update_cache=False, start_idx=0):
q, k, v shape: [1, 858, 40, 128]  # B S H head_dim
kv_cache初始化为零

# 存入KV cache (前3帧latent的KV)
k_cache[:, :3*frame_seqlen] = k  # 3*285 = 855 tokens
v_cache[:, :3*frame_seqlen] = v

# Cross-Attention (文本+图像)
context shape: [1, 257+512, 2048] = [1, 769, 2048]

# Audio Cross-Attention (SingleStreamAttention)
x_a = audio_cross_attn(norm_x(x), audio_embedding)
audio_embedding slice: [53, 32, 768] (对应当前帧)
q shape: [f, 40, N_h*N_w, 128] = [6, 40, 286, 128]  
# f=6帧，每帧286个spatial tokens

# FFN
x输出: [1, 858, 2048]

# Head
x = head(x, e) → [1, 858, 512]  # 512 = 16*1*2*2 (patch output)
```

**去噪更新:**
```python
latent = latent + (-noise_pred) * dt
```

**VAE解码:**
```python
_videos = vae.decode(_latent.squeeze(0))
_videos shape: [1, 3, 24, 416, 720]  # B C T H W (24帧视频)
# f=0时直接解码6帧latent → 24帧
```

---

#### 第1轮迭代 (_=1, f=1, 生成8帧latent):

**音频分段:**
```python
audio_start_idx = 0 + 1*8*4 = 32  # 上一轮结束位置
audio_end_idx = 53 + 32 = 85
# 切片音频: 32~85帧 (约1.6~4.25秒)
audio_embs shape: [1, 53, 5, 12, 768]  # 重新编码当前段
```

**Latent初始化:**
```python
latent shape: [16, 8, 26, 45]  # 8帧latent → 32帧视频
```

**KV Cache更新:**
```python
# f=1时，update_cache=True
# 压缩之前的KV cache (ConvKV压缩)
k_compress = k[:, :5*frame_seqlen] → Conv1d(kernel=5, stride=5)
           → [1, frame_seqlen, 40, 128]  # 5帧压缩为1帧

# KV cache滑动
k_cache[:, 0:frame_seqlen] = compressed_cache  # 历史压缩
k_cache[:, 1*frame_seqlen:3*frame_seqlen] = k_cache[:, 5*frame_seqlen:7*frame_seqlen]  # 滑动
k_cache[:, 3*frame_seqlen:] = k[:, -5*frame_seqlen:]  # 当前帧KV
```

**y_cut:**
```python
y_cut = y[:, :, 6:14, ...]  # [1, 20, 8, 26, 45] 对应8帧latent
```

**解码与合并:**
```python
_latent = concat([pre_latent[:, -3:], latent])  # 拼接前一轮最后3帧
         → [16, 11, 26, 45]
_videos = vae.decode(_latent)[:, :, 9:]  # 跳过重叠帧
_videos shape: [1, 3, 32, 416, 720]  # 32帧新视频
```

---

### 5. 视频合并阶段

```python
# 所有迭代结果拼接
videos = torch.concat(gen_video_list, dim=2)  
       → [1, 3, 17*32+24, 416, 720]  # 约560帧

# 后处理
videos = (videos + 1.0) / 2  # [-1,1] → [0,1]
videos = videos.permute(0, 2, 3, 4, 1)  # B T H W C

# 导出视频
export_to_video(videos, fps=20)  # tmp.mp4

# 添加原始音频
add_audio_to_video('tmp.mp4', '1.wav', 'output.mp4')
```

---

### 流程总结图

```
┌─────────────────────────────────────────────────────────────┐
│                    初始化                                    │
│  加载模型: WanModel, VAE, CLIP, T5, Wav2Vec2                │
│  KV Cache: [1, 8190, 40, 128] × 3步 × 40层                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   输入预处理                                  │
│  Image: [1,3,1,416,720] → CLIP [1,257,1280]                │
│  Text: "一个人在说话" → T5 [1,512,2048]                      │
│  Audio: 1.wav → resample → Wav2Vec2 [T,12,768]             │
│  VAE Encode: [1,3,53,416,720] → [1,16,14,26,45]            │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│           迭代生成 (iter_total_num ≈ 17轮)                   │
│                                                             │
│  Iter 0 (f=0, 6帧latent):                                   │
│    audio_slice: 0~2.65秒 → Wav2Vec2 → [53,12,768]          │
│    audio_embs: [1,53,5,12,768] → audio_proj [1,1696,768]   │
│    latent: [16,6,26,45] → diffusion → decode → 24帧        │
│    KV Cache: 初始化前3帧KV                                  │
│                                                             │
│  Iter 1~16 (f=1, 8帧latent):                               │
│    audio_slice: 滑动窗口 (每轮+32帧索引)                    │
│    audio_embs: steam重新编码 → audio_proj                   │
│    latent: [16,8,26,45] → diffusion → decode → 32帧        │
│    KV Cache: 压缩历史KV + 滑动 + 更新当前                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   视频合并                                    │
│  concat(gen_video_list) → [1,3,~560,416,720]               │
│  ffmpeg: 拼接音频 → 最终mp4                                 │
└─────────────────────────────────────────────────────────────┘
```

### 音频-视频对齐机制

**关键公式:**
```python
# 时间映射
frame_idx → audio_time = frame_idx / fps  # 视频帧对应音频时间

# VAE时间压缩
video_frame → latent_frame = video_frame / 4  # 4帧视频对应1帧latent

# 音频窗口
每帧视频对应5帧音频特征 (前后±2帧窗口)，中心对齐

# steam_audio分段
每轮生成8帧latent (32帧视频) → 对应音频段长度32/fps ≈ 1.6秒
```

### KV Cache压缩

```python
# 每5帧latent的KV压缩为1帧 (ConvKV)
原始: [1, 5*frame_seqlen, 40, 128]  # 5帧完整KV
压缩: Conv1d(k, kernel=5, stride=5) → [1, frame_seqlen, 40, 128]

# 内存恒定: 无论生成多长视频，KV cache大小固定
# 可支持小时级视频生成
```