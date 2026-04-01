import asyncio
import os
import time

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.async_wan22_video import (
    AsyncWan22VideoPipeline,
    AsyncWan22VideoPipelineConfig,
)
from telefuser.utils.logging import logger

PPL_CONFIG = dict(
    name="wan22_A14B_i2v_h100_distill",
    model_root="/nvfile-heatstorage/model_zoo/modelscope/Wan2.2-I2V-A14B",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_inference_steps=8,
    num_frames=81,
    resolution="720p",
    cfg_scale_high=1.0,
    cfg_scale_low=1.0,
    seed=42,
    tiled=False,
    attn_impl=AttnImplType.TORCH_SDPA,
    model_type="Wan2.2-I2V-A14B",
    sigma_shift=5.0,
    boundary=0.9,
    sample_solver="euler",
    target_fps=16,
)


def get_pipeline(parallelism: int = 1, model_root=None):
    if model_root is None:
        model_root = PPL_CONFIG["model_root"]

    module_manager = ModuleManager(device="cpu")

    # vae
    module_manager.load_model(
        f"{model_root}/Wan2.1_VAE.pth",
        torch_dtype=torch.bfloat16,
    )
    # dit high/low
    dit_high_path = f"{model_root}/dit_high_noise_distill_model_bf16_1022_ecab7.safetensors"
    dit_low_path = f"{model_root}/dit_low_noise_distill_model_bf16_1022_200c2.safetensors"
    module_manager.load_model([dit_high_path], torch_dtype=torch.bfloat16)
    module_manager.load_model([dit_low_path], torch_dtype=torch.bfloat16)
    # t5
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )

    pipe = AsyncWan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = AsyncWan22VideoPipelineConfig()
    pipe_config.dit_high_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_high_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_low_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_low_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.text_encoding_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    if parallelism > 1:
        # Configure parallel based on cfg_scale
        # For cfg_scale > 1: cfg_degree=2, sp_ulysses_degree=parallelism//2
        # For cfg_scale == 1: cfg_degree=1, sp_ulysses_degree=parallelism
        cfg_scale_high = PPL_CONFIG["cfg_scale_high"]
        cfg_scale_low = PPL_CONFIG["cfg_scale_low"]

        if cfg_scale_high > 1:
            pipe_config.dit_high_config.parallel_config.cfg_degree = 2
            pipe_config.dit_high_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_high_config.parallel_config.sp_ulysses_degree = parallelism

        if cfg_scale_low > 1:
            pipe_config.dit_low_config.parallel_config.cfg_degree = 2
            pipe_config.dit_low_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_low_config.parallel_config.sp_ulysses_degree = parallelism

        pipe_config.dit_high_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_high_config.parallel_config.enable_fsdp = True
        pipe_config.dit_low_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_low_config.parallel_config.enable_fsdp = True
        pipe_config.enable_denoising_parallel = True
        pipe_config.vae_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.vae_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.enable_vae_parallel = True
    pipe_config.vae_config.offload_type = WeightOffloadType.NO_CPU_OFFLOAD
    pipe.init(module_manager, pipe_config)
    return pipe


async def run(pipe, req_id, seed, prompt, image, ppl_config=None):
    if ppl_config is None:
        ppl_config = PPL_CONFIG

    saved_uri = ""
    total_ms = 0
    request_start = time.time()
    stage_times = []

    async for evt in pipe.agenerate(
        request_id=req_id,
        prompt=prompt,
        input_image=image,
        negative_prompt=ppl_config["negative_prompt"],
        seed=seed,
        resolution=ppl_config["resolution"],
        num_frames=ppl_config["num_frames"],
        num_inference_steps=ppl_config["num_inference_steps"],
        cfg_scale_high=ppl_config["cfg_scale_high"],
        cfg_scale_low=ppl_config["cfg_scale_low"],
        sigma_shift=ppl_config["sigma_shift"],
        boundary=ppl_config["boundary"],
        tiled=ppl_config["tiled"],
        artifact_fps=16,
        artifact_quality=6,
    ):
        if evt["type"] == "stage_end":
            stage_time = evt["timing"].get("duration_ms", 0)
            stage_times.append(
                {"stage": evt["stage"].get("name"), "role": evt["stage"].get("role"), "time_ms": stage_time}
            )
            logger.info(
                f"[{req_id}] stage_end role={evt['stage'].get('role')} "
                f"name={evt['stage'].get('name')} "
                f"time_ms={stage_time:.2f}"
            )
        elif evt["type"] == "final":
            artifacts = evt["payload"].get("artifacts", [])
            total_ms = evt["timing"].get("total_ms", 0)
            if artifacts:
                saved_uri = artifacts[0]["uri"]
                logger.info(f"[{req_id}] saved: {saved_uri}")
            else:
                logger.warning(f"[{req_id}] final without artifacts")
        elif evt["type"] == "error":
            raise RuntimeError(f"[{req_id}] {evt['error']['message']}")

    request_elapsed = time.time() - request_start

    if not saved_uri:
        raise RuntimeError(f"[{req_id}] finished without artifact uri")

    return {"uri": saved_uri, "wall_time_s": request_elapsed, "pipeline_time_ms": total_ms, "stage_times": stage_times}


async def run_with_file(pipe, req_id, seed, prompt, image, ppl_config=None):
    """Run async pipeline and save to file.

    Note: The async pipeline handles file saving internally through agenerate().
    The output URI is returned in the result dict.

    Args:
        pipe: Async pipeline instance
        req_id: Request ID for tracking
        seed: Random seed
        prompt: Positive guidance text prompt
        image: Input image
        ppl_config: Pipeline configuration

    Returns:
        Result dict with uri, wall_time_s, pipeline_time_ms, stage_times
    """
    if ppl_config is None:
        ppl_config = PPL_CONFIG

    result = await run(pipe, req_id, seed, prompt, image, ppl_config)
    logger.info(f"Video saved to: {result['uri']}")
    return result


@click.command()
@click.option("--gpu_num", default=1, help="Number of GPUs to use, default is 1")
@click.option(
    "--image_path", default=f"{os.path.dirname(__file__)}/../data/101235-video-720_0.png", help="Input image path"
)
@click.option(
    "--prompt",
    default="A stylish little girl gently caressing her dog while they relax in a sunny, beautiful backyard.",
    help="Positive guidance text prompt",
)
@click.option("--negative_prompt", default="", help="Negative guidance prompt")
@click.option("--resolution", default=PPL_CONFIG["resolution"], help="480p or 720p")
@click.option("--seed", default=PPL_CONFIG["seed"], help="Random seed")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Root directory of the model files")
def main(
    gpu_num,
    image_path,
    prompt,
    negative_prompt,
    resolution,
    seed,
    model_root,
):
    """Async Image to video conversion using Wan2.2 14B distillation model"""
    # Update PPL_CONFIG with command line parameters
    ppl_config = PPL_CONFIG.copy()
    ppl_config["model_root"] = model_root
    ppl_config["seed"] = seed
    ppl_config["resolution"] = resolution

    asyncio.run(async_main(gpu_num, image_path, prompt, negative_prompt, ppl_config))


async def async_main(gpu_num, image_path, prompt, negative_prompt, ppl_config):
    pipe = get_pipeline(parallelism=gpu_num, model_root=ppl_config["model_root"])

    image = Image.open(image_path).convert("RGB")

    await pipe.astart()

    ts = int(time.time())
    rid = f"demo-{ts}-1"
    logger.info(f"starting request: {rid}")

    result = await run(pipe, rid, ppl_config["seed"], prompt, image, ppl_config)

    logger.info(f"  [{rid}] {result['uri']}")

    await pipe.astop()


if __name__ == "__main__":
    main()
