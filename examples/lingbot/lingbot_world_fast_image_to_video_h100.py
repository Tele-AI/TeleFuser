"""LingBot-World-Fast offline and streaming example.

Single GPU:
    python examples/lingbot/lingbot_world_fast_image_to_video_h100.py

Four GPUs with Ulysses sequence parallelism:
    python examples/lingbot/lingbot_world_fast_image_to_video_h100.py --gpu_num 4
WebRTC streaming service:
    telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py \
        --gpu-num 4 -p 8088 --skip-validation

"""

from __future__ import annotations

import os
import time
from pathlib import Path

import click
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline, LingBotWorldFastPipelineConfig
from telefuser.pipelines.lingbot_world_fast.service import LingBotWorldFastService
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig
from telefuser.utils.video import save_video

TF_MODEL_ZOO_PATH = Path(os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")).expanduser()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = _PROJECT_ROOT / "examples" / "data" / "lingbot_world_fast"
DEFAULT_IMAGE_PATH = str(_DATA_ROOT / "image.jpg")
DEFAULT_ACTION_PATH = str(_DATA_ROOT)
DEFAULT_INTRINSICS_PATH = str(_DATA_ROOT / "intrinsics.npy")
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "work_dirs"
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
    "mountains under a bright blue sky with drifting white clouds. Gentle ripples reflect the tree and sky."
)
RESOLUTION_AREAS = {"480p": 480 * 832, "720p": 720 * 1280}

PPL_CONFIG = dict(
    parallelism=1,
    control_mode="cam",
    resolution="480p",
    frame_num=81,
    chunk_size=3,
    frame_policy="truncate",
    sample_shift=10.0,
    seed=42,
    target_fps=16,
    max_duration_seconds=5.0,
    attn_impl=AttnImplType.SAGE_ATTN_2_8_8_SM90,
    enable_fsdp=False,
    local_attn_size=-1,
    sink_size=0,
    timestep_indices=(0, 179, 358, 679),
    max_attention_size=None,
    vae_torch_dtype=torch.float32,
    torch_dtype=torch.bfloat16,
)


def get_pipeline(
    parallelism: int = PPL_CONFIG["parallelism"],
    model_root: str | None = None,
    fast_model_root: str | None = None,
) -> LingBotWorldFastPipeline:
    """Load LingBot-World-Fast for offline chunked generation."""
    if model_root is None or fast_model_root is None:
        default_model_root = str(TF_MODEL_ZOO_PATH / "Wan2.2-I2V-A14B")
        default_fast_model_root = str(TF_MODEL_ZOO_PATH / "lingbot" / "lingbot-world-fast")
    else:
        default_model_root, default_fast_model_root = model_root, fast_model_root
    if parallelism < 1:
        raise ValueError(f"parallelism must be positive, got {parallelism}")
    dtype = PPL_CONFIG["torch_dtype"]
    pipeline = LingBotWorldFastPipeline(device="cuda", torch_dtype=dtype)
    pipeline.init(
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=model_root or default_model_root,
            fast_checkpoint_path=fast_model_root or default_fast_model_root,
            vae_config=ModelRuntimeConfig(device_type="cuda", device_id=0, torch_dtype=PPL_CONFIG["vae_torch_dtype"]),
            text_encoding_config=ModelRuntimeConfig(device_type="cuda", device_id=0, torch_dtype=dtype),
            dit_torch_dtype=dtype,
            control_type=PPL_CONFIG["control_mode"],
            max_area=RESOLUTION_AREAS[PPL_CONFIG["resolution"]],
            local_attn_size=PPL_CONFIG["local_attn_size"],
            sink_size=PPL_CONFIG["sink_size"],
            timestep_indices=PPL_CONFIG["timestep_indices"],
            attention_config=AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"]),
            parallel_config=ParallelConfig(
                device_ids=list(range(parallelism)) if parallelism > 1 else None,
                sp_ulysses_degree=parallelism,
                enable_fsdp=PPL_CONFIG["enable_fsdp"],
            ),
            vae_parallel_config=ParallelConfig(device_ids=[0]),
        ),
    )
    return pipeline


def get_service(gpu_num: int = PPL_CONFIG["parallelism"]) -> LingBotWorldFastService:
    """Build the service loaded by the TeleFuser stream server."""
    pipeline = get_pipeline(parallelism=gpu_num)
    return LingBotWorldFastService(
        pipeline,
        default_fps=PPL_CONFIG["target_fps"],
        default_session_config={
            "control_mode": PPL_CONFIG["control_mode"],
            "max_duration_seconds": PPL_CONFIG["max_duration_seconds"],
            "chunk_size": PPL_CONFIG["chunk_size"],
            "frame_policy": PPL_CONFIG["frame_policy"],
            "sample_shift": PPL_CONFIG["sample_shift"],
            "max_attention_size": PPL_CONFIG["max_attention_size"],
        },
    )


def run(
    pipeline: LingBotWorldFastPipeline,
    image: Image.Image,
    prompt: str,
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
    action_path: str = DEFAULT_ACTION_PATH,
    intrinsics_path: str = DEFAULT_INTRINSICS_PATH,
    intrinsics_width: int = 832,
    intrinsics_height: int = 480,
    fps: int | None = None,
) -> list[Image.Image]:
    """Generate a complete offline video through the pipeline core API."""
    if resolution not in RESOLUTION_AREAS:
        raise ValueError(f"Unsupported resolution: {resolution}")
    pipeline.config.max_area = RESOLUTION_AREAS[resolution]
    fps = PPL_CONFIG["target_fps"] if fps is None else fps

    session_config = LingBotWorldFastSessionConfig(
        prompt=prompt,
        image=image,
        control_mode=PPL_CONFIG["control_mode"],
        fps=fps,
        chunk_size=PPL_CONFIG["chunk_size"],
        frame_policy=PPL_CONFIG["frame_policy"],
        frame_num=PPL_CONFIG["frame_num"],
        sample_shift=PPL_CONFIG["sample_shift"],
        seed=seed,
        max_attention_size=PPL_CONFIG["max_attention_size"],
    )
    controls = pipeline.prepare_offline_controls(
        session_config,
        action_path,
        intrinsics_path,
        intrinsics_width=intrinsics_width,
        intrinsics_height=intrinsics_height,
    )
    return pipeline.generate_video(session_config, controls)


@click.command()
@click.option(
    "--gpu_num",
    default=PPL_CONFIG["parallelism"],
    type=int,
    help="Number of GPUs used for Ulysses sequence parallelism",
)
@click.option("--image_path", default=DEFAULT_IMAGE_PATH, type=click.Path(exists=True))
@click.option("--action_path", default=DEFAULT_ACTION_PATH, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--intrinsics-path",
    "intrinsics_path",
    default=DEFAULT_INTRINSICS_PATH,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--intrinsics-width", default=832, type=click.IntRange(min=1), show_default=True)
@click.option("--intrinsics-height", default=480, type=click.IntRange(min=1), show_default=True)
@click.option("--prompt", default=DEFAULT_PROMPT, help="Positive guidance prompt")
@click.option("--seed", default=PPL_CONFIG["seed"], type=int)
@click.option("--resolution", default=PPL_CONFIG["resolution"], type=click.Choice(list(RESOLUTION_AREAS)))
@click.option("--fps", default=PPL_CONFIG["target_fps"], type=int, help="Output video frame rate")
@click.option("--model_root", default=None, type=click.Path(exists=True, file_okay=False))
@click.option(
    "--fast_model_root",
    default=None,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("--output", default=None, type=click.Path(dir_okay=False), help="Output video path")
def main(
    gpu_num: int,
    image_path: str,
    action_path: str,
    intrinsics_path: str,
    intrinsics_width: int,
    intrinsics_height: int,
    prompt: str,
    seed: int,
    resolution: str,
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
            intrinsics_path=intrinsics_path,
            intrinsics_width=intrinsics_width,
            intrinsics_height=intrinsics_height,
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
