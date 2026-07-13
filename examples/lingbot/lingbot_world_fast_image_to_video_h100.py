"""LingBot-World-Fast offline image-to-video example.

Single GPU:
    python examples/lingbot/lingbot_world_fast_image_to_video_h100.py

Four GPUs with Ulysses sequence parallelism:
    python examples/lingbot/lingbot_world_fast_image_to_video_h100.py --gpu_num 4
"""

from __future__ import annotations

import time
from pathlib import Path

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_world_fast.control import (
    LingBotWorldFastControlBuilder,
    LingBotWorldFastOfflineControlSource,
    load_action_control_inputs,
    load_camera_control_inputs,
    truncate_control_sequence,
)
from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastChunkRequest,
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
)
from telefuser.utils.video import save_video

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "examples" / "data" / "lingbot_world_fast"
TF_MODEL_ZOO_PATH = "/hhb-data/aigc/model_zoo"
DEFAULT_IMAGE_PATH = str(_DATA_ROOT / "image.jpg")
DEFAULT_ACTION_PATH = str(_DATA_ROOT)
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "work_dirs"
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
    "mountains under a bright blue sky with drifting white clouds. Gentle ripples reflect the tree and sky."
)
RESOLUTION_AREAS = {"480p": 480 * 832, "720p": 720 * 1280}

PPL_CONFIG = dict(
    name="lingbot_world_fast_i2v_h100",
    model_root=TF_MODEL_ZOO_PATH + "/Wan2.2-I2V-A14B",
    fast_model_root=TF_MODEL_ZOO_PATH + "/lingbot/lingbot-world-fast",
    control_mode="cam",
    resolution="480p",
    num_inference_steps=4,
    frame_num=81,
    chunk_size=3,
    sample_shift=10.0,
    seed=42,
    target_fps=16,
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90,
    enable_fsdp=False,
    local_attn_size=-1,
    sink_size=0,
    max_attention_size=None,
    torch_dtype=torch.bfloat16,
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    fast_model_root: str = PPL_CONFIG["fast_model_root"],
) -> LingBotWorldFastPipeline:
    """Load LingBot-World-Fast for offline chunked generation."""
    if parallelism < 1:
        raise ValueError(f"parallelism must be positive, got {parallelism}")

    dtype = PPL_CONFIG["torch_dtype"]
    pipeline = LingBotWorldFastPipeline(device="cuda", torch_dtype=dtype)
    pipeline.init(
        ModuleManager(device="cpu"),
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=model_root,
            fast_checkpoint_subdir=fast_model_root,
            vae_config=ModelRuntimeConfig(device_type="cuda", device_id=0, torch_dtype=dtype),
            text_encoding_config=ModelRuntimeConfig(device_type="cuda", device_id=0, torch_dtype=dtype),
            dit_torch_dtype=dtype,
            control_type=PPL_CONFIG["control_mode"],
            max_area=RESOLUTION_AREAS[PPL_CONFIG["resolution"]],
            local_attn_size=PPL_CONFIG["local_attn_size"],
            sink_size=PPL_CONFIG["sink_size"],
            attention_config=AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"]),
            parallel_config=ParallelConfig(
                device_ids=list(range(parallelism)) if parallelism > 1 else None,
                sp_ulysses_degree=parallelism,
                enable_fsdp=PPL_CONFIG["enable_fsdp"],
            ),
        ),
    )
    return pipeline


def run(
    pipeline: LingBotWorldFastPipeline,
    image: Image.Image,
    prompt: str,
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
    action_path: str = DEFAULT_ACTION_PATH,
    frame_num: int | None = None,
    fps: int | None = None,
) -> list[Image.Image]:
    """Generate a complete offline video through the pipeline core API."""
    if resolution not in RESOLUTION_AREAS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    pipeline.config.max_area = RESOLUTION_AREAS[resolution]
    frame_num = PPL_CONFIG["frame_num"] if frame_num is None else frame_num
    fps = PPL_CONFIG["target_fps"] if fps is None else fps

    session_config = LingBotWorldFastSessionConfig(
        prompt=prompt,
        image=image,
        control_mode=PPL_CONFIG["control_mode"],
        fps=fps,
        chunk_size=PPL_CONFIG["chunk_size"],
        frame_num=frame_num,
        sample_shift=PPL_CONFIG["sample_shift"],
        seed=seed,
        max_attention_size=PPL_CONFIG["max_attention_size"],
    )
    control_context = pipeline.control_context(session_config)
    control_builder = LingBotWorldFastControlBuilder(control_context)
    if session_config.control_mode == "act":
        poses, intrinsics, action = load_action_control_inputs(action_path)
    else:
        poses, intrinsics = load_camera_control_inputs(action_path)
        action = None
    poses, intrinsics, action = truncate_control_sequence(poses, intrinsics, action, session_config.frame_num)
    control_source = LingBotWorldFastOfflineControlSource(control_builder, poses, intrinsics, action)
    session = LingBotWorldFastGenerationSession(config=session_config)
    frames: list[Image.Image] = []
    try:
        for chunk_index in range(control_context.latent_frames // control_context.chunk_size):
            result = pipeline(
                session,
                LingBotWorldFastChunkRequest(
                    chunk_index=chunk_index,
                    control=control_source.control_at(chunk_index),
                ),
            )
            frames.extend(result.frames)
    finally:
        pipeline.release_session(session)
    return frames


@click.command()
@click.option(
    "--gpu_num",
    default=1,
    type=int,
    help="Number of GPUs used for Ulysses sequence parallelism",
)
@click.option("--image_path", default=DEFAULT_IMAGE_PATH, type=click.Path(exists=True))
@click.option("--action_path", default=DEFAULT_ACTION_PATH, type=click.Path(exists=True, file_okay=False))
@click.option("--prompt", default=DEFAULT_PROMPT, help="Positive guidance prompt")
@click.option("--seed", default=PPL_CONFIG["seed"], type=int)
@click.option("--resolution", default=PPL_CONFIG["resolution"], type=click.Choice(["480p", "720p"]))
@click.option("--frame_num", default=PPL_CONFIG["frame_num"], type=int, help="Number of output frames")
@click.option("--fps", default=PPL_CONFIG["target_fps"], type=int, help="Output video frame rate")
@click.option("--model_root", default=PPL_CONFIG["model_root"], type=click.Path(exists=True, file_okay=False))
@click.option(
    "--fast_model_root",
    default=PPL_CONFIG["fast_model_root"],
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--output", default=None, type=click.Path(dir_okay=False), help="Output video path")
def main(
    gpu_num: int,
    image_path: str,
    action_path: str,
    prompt: str,
    seed: int,
    resolution: str,
    frame_num: int,
    fps: int,
    model_root: str,
    fast_model_root: str,
    output: str | None,
) -> None:
    """Generate an offline video with LingBot-World-Fast."""
    pipeline = get_pipeline(gpu_num, model_root, fast_model_root)
    try:
        image = Image.open(image_path).convert("RGB")

        start = time.perf_counter()
        frames = run(
            pipeline,
            image,
            prompt,
            seed=seed,
            resolution=resolution,
            action_path=action_path,
            frame_num=frame_num,
            fps=fps,
        )
        elapsed = time.perf_counter() - start

        output_path = Path(output) if output else DEFAULT_OUTPUT_DIR / f"lingbot_world_fast_i2v_{gpu_num}gpu.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video(frames, str(output_path), fps=fps, quality=6)
        print(f"Video generation time: {elapsed:.2f} seconds")
        print(f"Video saved to: {output_path}")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
