import os
import tempfile
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, LoraConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.ltx_video.ltx23_video import LTX23VideoPipeline, LTXVideoPipelineConfig
from telefuser.utils.audio import save_wav
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_image_size, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="ltx23_22B_i2v_h100",
    model_root=TF_MODEL_ZOO_PATH + "/LTX-2.3",
    negative_prompt="worst quality, low quality, blurry, distorted, deformed, artifacts, overexposed, underexposed",
    num_inference_steps=30,
    num_frames=121,
    resolution="1080p",
    frame_rate=24.0,
    seed=42,
    tiled=False,
    input_image_frame_idx=0,
    input_image_strength=1.0,
    video_cfg_scale=3.0,
    video_stg_scale=1.0,
    video_rescale_scale=0.7,
    video_modality_scale=3.0,
    audio_cfg_scale=7.0,
    audio_stg_scale=1.0,
    audio_rescale_scale=0.7,
    audio_modality_scale=3.0,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    gemma_path_list=[
        TF_MODEL_ZOO_PATH + "/gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
        TF_MODEL_ZOO_PATH + "/gemma-3-12b-it-qat-q4_0-unquantized/model-00002-of-00005.safetensors",
        TF_MODEL_ZOO_PATH + "/gemma-3-12b-it-qat-q4_0-unquantized/model-00003-of-00005.safetensors",
        TF_MODEL_ZOO_PATH + "/gemma-3-12b-it-qat-q4_0-unquantized/model-00004-of-00005.safetensors",
        TF_MODEL_ZOO_PATH + "/gemma-3-12b-it-qat-q4_0-unquantized/model-00005-of-00005.safetensors",
    ],
    lora_configs=[
        LoraConfig(
            TF_MODEL_ZOO_PATH + "/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors",
            1.0,
        )
    ],
)


def get_pipeline(parallelism: int = 2, model_root: str = PPL_CONFIG["model_root"]) -> LTX23VideoPipeline:
    """
    Args:
        parallelism: Number of parallel GPUs for inference (REQUIRED)
        model_root: Root directory containing model files (REQUIRED)
    """
    # Load models
    module_manager = ModuleManager(device="cpu")
    # vae + dit
    module_manager.load_model(f"{model_root}/ltx-2.3-22b-dev.safetensors", torch_dtype=torch.bfloat16)
    # upsampler
    module_manager.load_model(f"{model_root}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors", torch_dtype=torch.bfloat16)
    # gemma encoder
    module_manager.load_models([PPL_CONFIG["gemma_path_list"]], torch_dtype=torch.bfloat16)

    pipe = LTX23VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = LTXVideoPipelineConfig()
    attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_stage1_config.attention_config = attention_config
    pipe_config.dit_stage2_config.attention_config = attention_config

    pipe_config.dit_stage1_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_stage2_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_stage2_config.lora_configs = PPL_CONFIG["lora_configs"]
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.upsampler_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    if parallelism > 1:
        device_ids = list(range(parallelism))
        for dit_cfg in (pipe_config.dit_stage1_config, pipe_config.dit_stage2_config):
            dit_cfg.parallel_config.device_ids = device_ids
            dit_cfg.parallel_config.sp_ulysses_degree = parallelism
            dit_cfg.parallel_config.enable_fsdp = True

        pipe_config.enable_denoising_parallel = True
        pipe_config.enable_vae_parallel = True
        pipe_config.vae_config.parallel_config.device_ids = device_ids
        pipe_config.vae_config.parallel_config.dp_degree = parallelism

    pipe.init(module_manager, pipe_config)
    return pipe


def run(
    pipeline: LTX23VideoPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    num_frames: int = PPL_CONFIG["num_frames"],
):
    """
       Convert static images to video sequences (along with audio) using unified audio-video generation model.
    Args:
        pipeline (VideoGenerationPipeline): Preloaded video generation pipeline object
        image (PIL.Image/ndarray): Input image, resolution should match height/width parameters
        prompt (str): Positive guidance text prompt
        negative_prompt (str, optional): Negative guidance prompt, will be merged with base negative prompt. Default is empty
        seed (int, optional): Random seed. Default is 42
        resolution (str): Resolution such as 720p, 480p
        num_inference_steps (int): Number of inference steps. Default is 8
        num_frames (int): Number of frames to generate. Default is 81

    Returns:
        List[PIL.Image]: Generated video sequence
    """
    width, height = get_target_image_size(
        image.size[0],
        image.size[1],
        resolution=resolution,
        height_division_factor=64,
        width_division_factor=64,
    )
    if height is None or width is None:
        raise ValueError(f"Unsupported resolution preset: {resolution}")

    video, waveform, sample_rate = pipeline(
        prompt=prompt,
        input_image=image,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}".strip(),
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=int(height),
        width=int(width),
        frame_rate=PPL_CONFIG["frame_rate"],
        input_image_frame_idx=PPL_CONFIG["input_image_frame_idx"],
        input_image_strength=PPL_CONFIG["input_image_strength"],
        video_cfg_scale=PPL_CONFIG["video_cfg_scale"],
        video_stg_scale=PPL_CONFIG["video_stg_scale"],
        video_rescale_scale=PPL_CONFIG["video_rescale_scale"],
        video_modality_scale=PPL_CONFIG["video_modality_scale"],
        audio_cfg_scale=PPL_CONFIG["audio_cfg_scale"],
        audio_stg_scale=PPL_CONFIG["audio_stg_scale"],
        audio_rescale_scale=PPL_CONFIG["audio_rescale_scale"],
        audio_modality_scale=PPL_CONFIG["audio_modality_scale"],
    )
    return video, waveform, sample_rate


@click.command()
@click.option("--gpu_num", default=2, help="Number of GPUs to use, default is 2")
@click.option(
    "--image_path",
    default=f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png",
    help="Input image path",
)
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard. Perfect for pet and family content, or videos aiming to showcase love, style, and the bond between kids and their pets.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Root directory containing model files")
@click.option(
    "--resolution",
    default=PPL_CONFIG["resolution"],
    type=click.Choice(["720p", "1080p", "2k", "4k"], case_sensitive=False),
    help="Resolution preset",
)
@click.option("--num_inference_steps", default=PPL_CONFIG["num_inference_steps"], help="Number of inference steps")
@click.option("--num_frames", default=PPL_CONFIG["num_frames"], help="Number of frames to generate")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
def main(  # noqa: PLR0913
    gpu_num: int,
    image_path: str,
    prompt: str,
    negative_prompt: str,
    model_root: str,
    resolution: str,
    num_inference_steps: int,
    num_frames: int,
    seed: int,
) -> None:
    """Image to video conversion using LTX2.3 22B model with two-stage denoising"""
    pipe = get_pipeline(parallelism=gpu_num, model_root=model_root)
    image = Image.open(image_path).convert("RGB")

    start = time.time()
    video, waveform, sample_rate = run(
        pipe,
        image,
        prompt,
        negative_prompt,
        seed=seed,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
    )
    elapsed_time = time.time() - start

    print(f"Video generation time: {elapsed_time:.2f} seconds")

    output_dir = os.getenv("TELEAI_EXAMPLE_OUTPUT_DIR", "./")
    filename = get_example_name(__file__).replace(".py", f"_{gpu_num}gpu.mp4")
    output_path = os.path.join(output_dir, filename)

    temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio_file.name
    temp_audio_file.close()
    try:
        save_wav(waveform, sample_rate, temp_audio_path)
        save_video(video, output_path, fps=float(PPL_CONFIG["frame_rate"]), quality=6, audio_path=temp_audio_path)
    finally:
        try:
            os.remove(temp_audio_path)
        except FileNotFoundError:
            pass
    print(f"Video saved to: {output_path}")

    del pipe


if __name__ == "__main__":
    main()
