"""LiveAct Example: Audio-conditioned Image-to-Video Generation.

This example demonstrates how to use the LiveAct pipeline for generating
talking head videos from an input image and audio.

Prerequisites:
    SoulX-LiveAct must be installed or available in PYTHONPATH:
    export PYTHONPATH=/path/to/SoulX-LiveAct:$PYTHONPATH

Usage:
    python examples/liveact/liveact_i2v_1gpu.py \
        --ckpt_dir path/to/checkpoints \
        --wav2vec_dir path/to/wav2vec2 \
        --image path/to/image.jpg \
        --audio path/to/audio.wav \
        --prompt "A person talking naturally" \
        --output output.mp4

Optimizations:
    --enable_fp8_gemm: Use FP8 for FFN linear layers (default: True)
    --enable_compile: Enable torch.compile for DiT (default: True)
    --fp8_kv_cache: Use FP8 for KV cache (default: False)
    --offload_cache: Offload KV cache to CPU (default: True)

Config:
    Modify PPL_CONFIG to change attention implementation.
    Set PPL_CONFIG["attention_config"] = AttentionConfig.dense_attention(AttnImplType.XXX).
"""

import os

import click
import torch
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.wav2vec2 import Wav2Vec2Model
from telefuser.pipelines.liveact import LiveActPipeline, LiveActPipelineConfig
from telefuser.utils.logging import logger
from telefuser.utils.video import save_video

# Check SageAttention availability
try:
    from sageattention import sageattn  # noqa: F401

    USE_SAGEATTN = True
except ImportError:
    USE_SAGEATTN = False

PPL_CONFIG = dict(
    name="liveact_i2v_1gpu",
    negative_prompt="",
    num_inference_steps=3,
    audio_cfg=1.0,
    fps=24,
    attention_config=AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
)


def get_pipeline(
    ckpt_dir: str,
    wav2vec_dir: str,
    device: str = "cuda",
    enable_fp8_gemm: bool = True,
    enable_compile: bool = True,
    fp8_kv_cache: bool = False,
    offload_cache: bool = True,
    mean_memory: bool = False,
):
    """Load LiveAct pipeline using original model initialization.

    Args:
        ckpt_dir: Path to model checkpoints (LiveAct weights)
        wav2vec_dir: Path to wav2vec2 weights
        device: Device to load model on
        enable_fp8_gemm: Use FP8 for FFN linear layers
        enable_compile: Enable torch.compile for DiT
        fp8_kv_cache: Use FP8 quantization for KV cache
        offload_cache: Offload KV cache to CPU
        mean_memory: Use mean compression for KV cache

    Returns:
        LiveActPipeline instance
    """
    torch_dtype = torch.bfloat16

    # Initialize module manager
    mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")

    # ============== Check optimization availability ==============
    fp8_gemm_available = False

    if USE_SAGEATTN:
        logger.info("✓ SageAttention is available")
    else:
        logger.warning("✗ SageAttention is NOT available, will use SDPA fallback")

    if enable_fp8_gemm:
        try:
            from telefuser.ops.fp8_gemm import FP8GemmOptions, enable_fp8_gemm  # noqa: F401

            fp8_gemm_available = True
            logger.info("✓ FP8 GEMM is available")
        except ImportError:
            logger.warning("✗ FP8 GEMM is NOT available")

    # ============== Load LiveAct DiT ==============
    dit_path = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors")
    mm.load_model(dit_path, name="liveact_dit", torch_dtype=torch_dtype)

    # ============== Load VAE ==============
    vae_path = os.path.join(ckpt_dir, "Wan2.1_VAE.pth")
    mm.load_model(vae_path, name="wan_video_vae", torch_dtype=torch_dtype)

    # ============== Load CLIP ==============
    clip_path = os.path.join(ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    mm.load_model(clip_path, name="wan_video_image_encoder", torch_dtype=torch_dtype)

    # ============== Load T5 Text Encoder ==============
    text_encoder_path = os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth")
    mm.load_model(text_encoder_path, name="wan_video_text_encoder", torch_dtype=torch_dtype)

    # ============== Load Wav2Vec2 Audio Encoder ==============
    audio_encoder = (
        Wav2Vec2Model.from_pretrained(
            wav2vec_dir,
            local_files_only=True,
            torch_dtype=torch_dtype,
        )
        .to(device, dtype=torch_dtype)
        .eval()
    )
    audio_encoder.feature_extractor._freeze_parameters()

    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir,
        local_files_only=True,
    )

    mm.add_module(audio_encoder, "wav2vec2")
    mm.add_module(wav2vec_feature_extractor, "wav2vec2_feature_extractor")

    # ============== Create Pipeline ==============
    pipeline = LiveActPipeline(device=device, torch_dtype=torch_dtype)

    # Create config with optimization settings
    config = LiveActPipelineConfig()
    config.audio_cfg = PPL_CONFIG["audio_cfg"]
    config.fps = PPL_CONFIG["fps"]
    config.num_inference_steps = PPL_CONFIG["num_inference_steps"]

    # KV cache settings
    config.fp8_kv_cache = fp8_kv_cache
    config.offload_cache = offload_cache
    config.mean_memory = mean_memory

    # Optimization flags
    config.enable_fp8_gemm = enable_fp8_gemm
    config.enable_torch_compile = enable_compile

    # Attention implementation (from PPL_CONFIG)
    config.dit_config.attention_config = PPL_CONFIG["attention_config"]

    # Initialize pipeline
    pipeline.init(mm, config)

    # Print optimization summary
    logger.info("=" * 60)
    logger.info("Optimization Summary:")
    logger.info(f"  Attention:      {PPL_CONFIG['attention_config'].attn_impl}")
    logger.info(f"  SageAttention:  {'✓ available' if USE_SAGEATTN else '✗ not available'}")
    logger.info(f"  FP8 GEMM:       {'✓ enabled' if (enable_fp8_gemm and fp8_gemm_available) else '✗ disabled'}")
    logger.info(f"  torch.compile:  {'✓ enabled' if enable_compile else '✗ disabled'}")
    logger.info(f"  FP8 KV Cache:   {'✓ enabled' if fp8_kv_cache else '✗ disabled'}")
    logger.info(f"  Cache Offload:  {'✓ CPU' if offload_cache else '✗ GPU'}")
    logger.info("=" * 60)

    return pipeline


def run(
    pipeline: LiveActPipeline,
    prompt: str,
    input_image: str,
    audio_path: str,
    height: int = 480,
    width: int = 832,
    fps: int = 24,
    audio_cfg: float = 1.0,
    seed: int = 42,
    output_path: str = "output.mp4",
):
    """Run LiveAct inference.

    Args:
        pipeline: LiveActPipeline instance
        prompt: Text prompt
        input_image: Path to input image
        audio_path: Path to audio file
        height: Video height
        width: Video width
        fps: Video fps
        audio_cfg: Audio CFG scale
        seed: Random seed
        output_path: Output video path
    """
    # Load input image
    image = Image.open(input_image).convert("RGB")

    # Generate video frames
    frames = pipeline(
        prompt=prompt,
        input_image=image,
        audio_path=audio_path,
        height=height,
        width=width,
        fps=fps,
        audio_cfg=audio_cfg,
        seed=seed,
    )

    # Save video with audio
    save_video(frames, output_path, fps=fps, audio_path=audio_path)
    print(f"Video saved to: {output_path}")


@click.command()
@click.option("--ckpt_dir", required=True, help="Path to LiveAct checkpoints")
@click.option("--wav2vec_dir", required=True, help="Path to wav2vec2 weights")
@click.option("--image", required=True, help="Path to input image")
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--prompt", default="A person talking naturally", help="Text prompt")
@click.option("--height", default=480, help="Video height")
@click.option("--width", default=832, help="Video width")
@click.option("--fps", default=24, help="Video fps")
@click.option("--audio_cfg", default=1.0, type=float, help="Audio CFG scale")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--output", default="output.mp4", help="Output video path")
@click.option("--enable_fp8_gemm/--no_fp8_gemm", default=True, help="Enable FP8 GEMM for FFN (default: True)")
@click.option("--enable_compile/--no_compile", default=True, help="Enable torch.compile (default: True)")
@click.option("--fp8_kv_cache", is_flag=True, default=False, help="Use FP8 for KV cache")
@click.option("--offload_cache/--no_offload_cache", default=True, help="Offload KV cache to CPU (default: True)")
@click.option("--mean_memory", is_flag=True, default=False, help="Use mean compression for KV cache")
def main(
    ckpt_dir: str,
    wav2vec_dir: str,
    image: str,
    audio: str,
    prompt: str,
    height: int,
    width: int,
    fps: int,
    audio_cfg: float,
    seed: int,
    output: str,
    enable_fp8_gemm: bool,
    enable_compile: bool,
    fp8_kv_cache: bool,
    offload_cache: bool,
    mean_memory: bool,
):
    """LiveAct: Generate talking head video from image and audio."""
    # Load pipeline
    print(f"Loading model from: {ckpt_dir}")
    pipeline = get_pipeline(
        ckpt_dir=ckpt_dir,
        wav2vec_dir=wav2vec_dir,
        enable_fp8_gemm=enable_fp8_gemm,
        enable_compile=enable_compile,
        fp8_kv_cache=fp8_kv_cache,
        offload_cache=offload_cache,
        mean_memory=mean_memory,
    )

    # Run inference
    print(f"Generating video: {width}x{height} @ {fps}fps")
    run(
        pipeline=pipeline,
        prompt=prompt,
        input_image=image,
        audio_path=audio,
        height=height,
        width=width,
        fps=fps,
        audio_cfg=audio_cfg,
        seed=seed,
        output_path=output,
    )


if __name__ == "__main__":
    main()
