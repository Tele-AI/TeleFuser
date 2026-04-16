import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, CompileConfig, QuantConfig, QuantType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.wav2vec2 import Wav2Vec2Model
from telefuser.pipelines.liveact import LiveActPipeline, LiveActPipelineConfig
from telefuser.platforms.cuda import CudaPlatform
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
CudaPlatform.init_cudnn_optimizations()

PPL_CONFIG = dict(
    name="liveact_s2v",
    model_root=TF_MODEL_ZOO_PATH + "/LiveAct",
    wav2vec_dir=TF_MODEL_ZOO_PATH + "/chinese-wav2vec2-base",
    num_inference_steps=3,
    audio_cfg=1.0,
    fps=20,
    seed=42,
    height=720,
    width=416,
    # Attention configuration - modify this to change attention implementation
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90,
    # Quantization and compile configuration
    quant_config=QuantConfig(enabled=True, quant_type=QuantType.FP8),
    compile_config=CompileConfig(
        enabled=True,
        mode="max-autotune-no-cudagraphs",
        backend="inductor",
        dynamic=False,
    ),
    vae_compile=True,
)


def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """Load LiveAct pipeline with optional SP parallelism.

    Args:
        parallelism: Number of GPUs for parallel inference (REQUIRED)
        model_root: Path to model checkpoints (REQUIRED)

    Returns:
        LiveActPipeline instance
    """
    wav2vec_dir = PPL_CONFIG["wav2vec_dir"]
    torch_dtype = torch.bfloat16

    mm = ModuleManager(torch_dtype=torch_dtype, device="cpu")

    mm.load_models(
        [
            os.path.join(model_root, "diffusion_pytorch_model-*.safetensors"),
        ],
        torch_dtype=torch_dtype,
    )
    mm.load_models(
        [
            os.path.join(model_root, "Wan2.1_VAE.pth"),
            os.path.join(model_root, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),
        ],
        torch_dtype=torch_dtype,
    )

    # Load Wav2Vec2 Audio Encoder (model includes integrated audio_processor)
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True, torch_dtype=torch_dtype).eval()
    audio_encoder.feature_extractor._freeze_parameters()

    mm.add_module(audio_encoder, "wav2vec2")

    pipeline = LiveActPipeline(device="cuda", torch_dtype=torch_dtype)
    config = LiveActPipelineConfig()
    config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    config.dit_config.quant_config = PPL_CONFIG["quant_config"]
    config.dit_config.compile_config = PPL_CONFIG["compile_config"]
    config.vae_config.compile_config = CompileConfig(enabled=PPL_CONFIG["vae_compile"])

    # Configure SP parallelism for multi-GPU
    if parallelism > 1:
        config.dit_config.parallel_config.sp_ulysses_degree = parallelism
        config.dit_config.parallel_config.device_ids = list(range(parallelism))
        config.vae_config.parallel_config.sp_ulysses_degree = parallelism
        config.vae_config.parallel_config.device_ids = list(range(parallelism))
        config.enable_denoising_parallel = True
        config.enable_vae_parallel = True
        print(f"Enabled Ulysses Sequence Parallel with {parallelism} GPUs")

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
        audio_path=audio_path,
        input_image=image,
        height=height,
        width=width,
        fps=fps,
        audio_cfg=PPL_CONFIG["audio_cfg"],
        seed=seed,
    )
    return frames


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use (1, 2, or 4), default is 1")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Path to LiveAct checkpoints")
@click.option("--image_path", default=f"{os.path.dirname(__file__)}/../data/1.png", help="Path to input image")
@click.option("--audio_path", default=f"{os.path.dirname(__file__)}/../data/1.wav", help="Path to audio file")
@click.option("--prompt", default="A person talking naturally", help="Text prompt")
@click.option("--height", default=PPL_CONFIG["height"], help="Video height")
@click.option("--width", default=PPL_CONFIG["width"], help="Video width")
@click.option("--fps", default=PPL_CONFIG["fps"], help="Video fps")
@click.option("--output", default=None, help="Output video path (default: liveact_i2v_sp_{gpu_num}gpu.mp4)")
def main(
    gpu_num: int,
    model_root: str,
    image_path: str,
    audio_path: str,
    prompt: str,
    height: int,
    width: int,
    fps: int,
    output: str | None,
):
    """LiveAct: Generate talking head video from image and audio with SP parallelism."""
    pipeline = get_pipeline(gpu_num, model_root)
    input_image = Image.open(image_path).convert("RGB")
    start = time.time()
    frames = run(pipeline, input_image, audio_path, prompt, height, width, fps)
    elapsed = time.time() - start
    print(f"Video generation time: {elapsed:.2f} seconds")

    if output is None:
        filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
        output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
        output = os.path.join(output_dir, filename)
    save_video(frames, output, fps=fps, audio_path=audio_path, quality=6)
    print(f"Video saved to: {output}")


if __name__ == "__main__":
    main()
