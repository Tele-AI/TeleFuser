"""LiveAct Example: Audio-conditioned Image-to-Video Generation.

This example demonstrates how to use the LiveAct pipeline for generating
talking head videos from an input image and audio.

Usage:
    python examples/liveact/liveact_i2v_1gpu.py \
        --ckpt_dir path/to/checkpoints \
        --wav2vec_dir path/to/wav2vec2 \
        --image path/to/image.jpg \
        --audio path/to/audio.wav \
        --prompt "A person talking naturally" \
        --output output.mp4
"""

import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, CompileConfig, QuantConfig, QuantType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.wav2vec2 import Wav2Vec2Model
from telefuser.pipelines.liveact import LiveActPipeline, LiveActPipelineConfig
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import save_video

PPL_CONFIG = dict(
    num_inference_steps=3,
    audio_cfg=1.0,
    fps=24,
    seed=42,
    height=480,
    width=832,
    attention_config=AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90),
    quant_config=QuantConfig(enabled=True, quant_type=QuantType.FP8),
    compile_config=CompileConfig(enabled=True, dynamic=False),
)


def get_pipeline(ckpt_dir: str, wav2vec_dir: str):
    """Load LiveAct pipeline.

    Args:
        ckpt_dir: Path to model checkpoints (LiveAct weights)
        wav2vec_dir: Path to wav2vec2 weights

    Returns:
        LiveActPipeline instance
    """
    torch_dtype = torch.bfloat16

    mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")

    mm.load_models(
        [
            os.path.join(ckpt_dir, "diffusion_pytorch_model-*.safetensors"),
        ],
        torch_dtype=torch_dtype,
    )
    mm.load_models(
        [
            os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
            os.path.join(ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        ],
        torch_dtype=torch_dtype,
    )

    # Load Wav2Vec2 Audio Encoder (model includes integrated audio_processor)
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True, torch_dtype=torch_dtype).eval()
    audio_encoder.feature_extractor._freeze_parameters()

    mm.add_module(audio_encoder, "wav2vec2")

    pipeline = LiveActPipeline(device="cuda", torch_dtype=torch_dtype)
    config = LiveActPipelineConfig()
    config.dit_config.attention_config = PPL_CONFIG["attention_config"]
    config.dit_config.quant_config = PPL_CONFIG["quant_config"]
    config.dit_config.compile_config = PPL_CONFIG["compile_config"]

    pipeline.init(mm, config)
    return pipeline


def run(
    pipeline: LiveActPipeline,
    image: str,
    audio_path: str,
    prompt: str,
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    fps: int = PPL_CONFIG["fps"],
    seed: int = PPL_CONFIG["seed"],
):
    """Run LiveAct inference.

    Args:
        pipeline: LiveActPipeline instance
        image: Path to input image
        audio_path: Path to audio file
        prompt: Text prompt
        height: Video height
        width: Video width
        fps: Video fps
        seed: Random seed

    Returns:
        Generated video frames
    """
    frames = pipeline(
        prompt=prompt,
        input_image=Image.open(image).convert("RGB"),
        audio_path=audio_path,
        height=height,
        width=width,
        fps=fps,
        audio_cfg=PPL_CONFIG["audio_cfg"],
        seed=seed,
    )
    return frames


@click.command()
@click.option("--ckpt_dir", required=True, help="Path to LiveAct checkpoints")
@click.option("--wav2vec_dir", required=True, help="Path to wav2vec2 weights")
@click.option("--image", required=True, help="Path to input image")
@click.option("--audio", required=True, help="Path to audio file")
@click.option("--prompt", default="A person talking naturally", help="Text prompt")
@click.option("--height", default=PPL_CONFIG["height"], help="Video height")
@click.option("--width", default=PPL_CONFIG["width"], help="Video width")
@click.option("--fps", default=PPL_CONFIG["fps"], help="Video fps")
@click.option("--output", default=None, help="Output video path (default: liveact_i2v_1gpu.mp4)")
def main(
    ckpt_dir: str,
    wav2vec_dir: str,
    image: str,
    audio: str,
    prompt: str,
    height: int,
    width: int,
    fps: int,
    output: str | None,
):
    """LiveAct: Generate talking head video from image and audio."""
    pipeline = get_pipeline(ckpt_dir, wav2vec_dir)

    start = time.time()
    frames = run(pipeline, image, audio, prompt, height, width, fps)
    elapsed = time.time() - start
    print(f"Video generation time: {elapsed:.2f} seconds")

    if output is None:
        filename = get_example_name(__file__)
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        output = os.path.join(output_dir, filename)
    save_video(frames, output, fps=fps, audio_path=audio, quality=6)
    print(f"Video saved to: {output}")


if __name__ == "__main__":
    main()
