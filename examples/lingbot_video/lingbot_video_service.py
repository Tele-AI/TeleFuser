"""Service entrypoint for LingBot-Video Dense generation."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from PIL import Image

from telefuser.pipelines.lingbot_video import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    LingBotVideoRequest,
    build_lingbot_video_pipeline,
    build_lingbot_video_refiner_stage,
    default_negative_caption,
    load_refiner_first_frame,
    parse_lingbot_video_prompt,
    prepare_refiner_video,
)
from telefuser.utils.video import get_target_video_size_from_ratio

PPL_CONFIG = {
    "model_root": os.environ.get(
        "LINGBOT_VIDEO_MODEL_ROOT", "/hhb-data/aigc/model_zoo/lingbot/lingbot-video-dense-1.3b"
    ),
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "fps": 24,
    "variant": os.environ.get("LINGBOT_VIDEO_VARIANT", "dense"),
    "enable_refiner": os.environ.get("LINGBOT_VIDEO_ENABLE_REFINER", "0").lower() in {"1", "true", "yes"},
    "refiner_height": 1088,
    "refiner_width": 1920,
    "refiner_steps": 8,
    "refiner_guidance_scale": 3.0,
    "refiner_shift": 3.0,
    "refiner_t_thresh": 0.85,
    "refiner_tail_steps": 2,
}


def get_pipeline_contract() -> dict:
    """Declare the task contract consumed by TeleFuser service APIs."""
    parameters = {
        "seed": {"type": "integer", "default": 42},
        "target_video_length": {"type": "integer", "default": 4},
        "resolution": {"type": "string", "default": "480p"},
        "aspect_ratio": {"type": "string", "default": "16:9"},
        "refine": {"type": "boolean", "default": PPL_CONFIG["enable_refiner"]},
    }
    image_parameters = {
        **parameters,
        "negative_prompt": {"type": "string", "default": DEFAULT_NEGATIVE_PROMPT_IMAGE},
    }
    video_parameters = {
        **parameters,
        "negative_prompt": {"type": "string", "default": DEFAULT_NEGATIVE_PROMPT},
    }
    return {
        "pipeline_name": "lingbot_video",
        "supported_tasks": ["t2i", "t2v", "i2v"],
        "supported_media_types": ["image", "video"],
        "execution_mode": "serial_single_pipeline",
        "effective_max_concurrent_tasks": 1,
        "task_contracts": {
            "t2i": {"media_type": "image", "parameters": image_parameters},
            "t2v": {"media_type": "video", "parameters": video_parameters},
            "i2v": {
                "media_type": "video",
                "required_inputs": ["first_image_path"],
                "parameters": video_parameters,
            },
        },
    }


def get_pipeline(parallelism: int = 1, model_root: str = PPL_CONFIG["model_root"]):
    """Load a single-GPU source-equivalent LingBot base pipeline."""
    if parallelism != 1:
        raise ValueError("LingBot-Video service currently supports parallelism=1")
    return build_lingbot_video_pipeline(
        model_root,
        variant=PPL_CONFIG["variant"],
        guidance_scale=PPL_CONFIG["guidance_scale"],
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
    )


def _load_condition_image(path: str) -> torch.Tensor:
    """Load a user-uploaded RGB image as a raw [0,255] TI2V tensor."""
    array = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float()


def _num_frames(seconds: int, fps: int) -> int:
    """Convert service duration to the nearest valid ``4n+1`` LingBot count."""
    return max(1, 4 * round(seconds * fps / 4) + 1)


def _lingbot_video_size(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """Resolve a service resolution for the combined VAE and DiT spatial contract."""
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution,
        height_division_factor=16,
        width_division_factor=16,
    )
    if width is None or height is None:
        raise ValueError(f"unsupported LingBot-Video resolution: {resolution}")
    return width, height


def run_with_file(
    pipeline,
    prompt: str,
    *,
    first_image_path: str = "",
    negative_prompt: str | None = None,
    seed: int = 42,
    output_path: str = "output.mp4",
    target_video_length: int = 4,
    resolution: str = "480p",
    aspect_ratio: str = "16:9",
    task: str = "t2v",
    refine: bool | None = None,
    **_: object,
) -> dict[str, str]:
    """Generate and encode a T2I, T2V, or TI2V service result."""
    width, height = _lingbot_video_size(aspect_ratio, resolution)
    caption, _ = parse_lingbot_video_prompt(json.loads(prompt))
    if task not in {"t2i", "t2v", "i2v"}:
        raise ValueError(f"unsupported LingBot-Video service task: {task}")
    if task == "i2v" and not first_image_path:
        raise ValueError("LingBot-Video i2v requires first_image_path")
    refine_enabled = PPL_CONFIG["enable_refiner"] if refine is None else refine
    if refine_enabled and pipeline.variant != "moe":
        raise ValueError("LingBot-Video refiner requires a pipeline loaded with variant=moe")
    num_frames = 1 if task == "t2i" else _num_frames(target_video_length, PPL_CONFIG["fps"])
    resolved_negative_prompt = negative_prompt if negative_prompt is not None else default_negative_caption(num_frames)
    generation = pipeline.generate(
        LingBotVideoRequest(
            caption=caption,
            height=height,
            width=width,
            num_frames=num_frames,
            image=_load_condition_image(first_image_path) if first_image_path else None,
        ),
        negative_caption=resolved_negative_prompt,
        generator=torch.Generator("cuda").manual_seed(seed),
    )
    frames = generation.output
    if refine_enabled:
        if pipeline.text_stage is None or pipeline.model_dir is None:
            raise RuntimeError("LingBot-Video refiner requires text-stage and checkpoint metadata")
        if generation.prompt_conditions.has_visual_condition:
            positive, positive_mask = pipeline.text_stage.encode(caption)
            negative, negative_mask = pipeline.text_stage.encode(resolved_negative_prompt)
        else:
            positive = generation.prompt_conditions.positive_prompt_embeds
            negative = generation.prompt_conditions.negative_prompt_embeds
            positive_mask = generation.prompt_conditions.positive_attention_mask
            negative_mask = generation.prompt_conditions.negative_attention_mask
        pipeline.release_gpu_resources()
        refiner = build_lingbot_video_refiner_stage(pipeline.model_dir)
        lowres_video, _ = prepare_refiner_video(
            frames,
            source_fps=PPL_CONFIG["fps"],
            height=PPL_CONFIG["refiner_height"],
            width=PPL_CONFIG["refiner_width"],
        )
        clean_first_frame = (
            load_refiner_first_frame(
                first_image_path,
                target_height=PPL_CONFIG["refiner_height"],
                target_width=PPL_CONFIG["refiner_width"],
                geometry_height=height,
                geometry_width=width,
            )
            if first_image_path
            else None
        )
        frames = refiner.refine(
            lowres_video,
            positive,
            negative,
            positive_mask,
            negative_mask,
            num_inference_steps=PPL_CONFIG["refiner_steps"],
            guidance_scale=PPL_CONFIG["refiner_guidance_scale"],
            shift=PPL_CONFIG["refiner_shift"],
            t_thresh=PPL_CONFIG["refiner_t_thresh"],
            tail_steps=PPL_CONFIG["refiner_tail_steps"],
            clean_first_frame=clean_first_frame,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
    video = frames[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    if task == "t2i":
        Image.fromarray((video[0] * 255).round().astype("uint8")).save(output_path)
        return {"output_path": output_path}
    from diffusers.utils import export_to_video

    export_to_video(list(video), output_path, fps=PPL_CONFIG["fps"])
    return {"output_path": output_path}
