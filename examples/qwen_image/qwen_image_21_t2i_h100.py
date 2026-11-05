"""Qwen-Image 2.1 text-to-image generation.

Usage:
    CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_t2i_h100.py \
        --model_root /hhb-data/aigc/model_zoo/Qwen-Image-2.1 \
        --prompt "A paper boat on a moonlit lake" \
        --output_path work_dirs/qwen-image-2.1.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import click
import torch
from PIL import Image

from telefuser.pipelines.qwen_image import QwenImage21Pipeline
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template
from telefuser.utils.utils import get_example_name

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG = {
    "name": "qwen_image_2.1_t2i",
    "model_root": os.path.join(TF_MODEL_ZOO_PATH, "Qwen-Image-2.1"),
    "prompt": (
        "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and "
        "lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied "
        "in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, "
        "cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at "
        "an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no "
        "staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming "
        "with vivid, fresh, youthful charm."
    ),
    "negative_prompt": (
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
        "画面具有AI感。构图混乱。文字模糊，扭曲。"
    ),
    "seed": 42,
    "aspect_ratio": "16:9",
    "num_inference_steps": 40,
    "cfg_scale": 1.0,
    "output_resolution": 1024,
}

PIPELINE_CONTRACT = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=["t2i"],
    task_contracts={
        "t2i": build_task_contract_template(
            "t2i",
            parameter_overrides={
                "prompt": {"default": PPL_CONFIG["prompt"]},
                "negative_prompt": {"default": PPL_CONFIG["negative_prompt"]},
                "seed": {"default": PPL_CONFIG["seed"]},
                "aspect_ratio": {"default": PPL_CONFIG["aspect_ratio"]},
            },
            excluded_parameters=["resolution"],
        )
    },
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    device: str | None = None,
) -> QwenImage21Pipeline:
    """Load Qwen-Image 2.1 for a single GPU."""

    if parallelism != 1:
        raise ValueError("Qwen-Image 2.1 example currently supports one GPU; use CUDA_VISIBLE_DEVICES to select it")
    runtime_device = device or "cuda"
    return QwenImage21Pipeline.from_pretrained(
        model_root,
        device=runtime_device,
        torch_dtype=torch.bfloat16,
    )


def run(
    pipeline: QwenImage21Pipeline,
    prompt: str,
    negative_prompt: str = PPL_CONFIG["negative_prompt"],
    seed: int = PPL_CONFIG["seed"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    cfg_scale: float = PPL_CONFIG["cfg_scale"],
) -> list[Image.Image]:
    """Generate images using the existing Qwen-Image aspect-ratio sizes."""

    default_width, default_height = ASPECT_RATIO_TO_SIZE[aspect_ratio]

    return pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt if cfg_scale > 1 else None,
        seed=seed,
        height=height if height is not None else default_height,
        width=width if width is not None else default_width,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        output_resolution=PPL_CONFIG["output_resolution"],
    )


def run_with_file(
    pipeline: QwenImage21Pipeline,
    prompt: str,
    output_path: str,
    negative_prompt: str = PPL_CONFIG["negative_prompt"],
    seed: int = PPL_CONFIG["seed"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    height: int | None = None,
    width: int | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Generate an image and save it for the TeleFuser service contract."""

    images = run(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        aspect_ratio=aspect_ratio,
        height=height,
        width=width,
        cfg_scale=float(kwargs.get("cfg_scale", PPL_CONFIG["cfg_scale"])),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination)
    return {"output_path": str(destination)}


@click.command()
@click.option("--gpu_num", default=1, type=int, show_default=True, help="Number of GPUs; Qwen-Image 2.1 uses one GPU")
@click.option("--model_root", default=PPL_CONFIG["model_root"], show_default=True, help="Model directory")
@click.option("--prompt", default=PPL_CONFIG["prompt"], show_default=True, help="Text prompt")
@click.option("--negative_prompt", default=PPL_CONFIG["negative_prompt"], show_default=True)
@click.option("--aspect_ratio", "-ar", default=PPL_CONFIG["aspect_ratio"], show_default=True)
@click.option("--height", type=int, default=None)
@click.option("--width", type=int, default=None)
@click.option("--seed", default=PPL_CONFIG["seed"], type=int, show_default=True)
@click.option("--output_path", type=click.Path(path_type=Path), default=None)
def main(
    gpu_num: int,
    model_root: str,
    prompt: str,
    negative_prompt: str,
    aspect_ratio: str,
    height: int | None,
    width: int | None,
    seed: int,
    output_path: Path | None,
) -> None:
    """Run Qwen-Image 2.1 inference."""

    pipeline = get_pipeline(gpu_num, model_root)
    destination = output_path or Path(get_example_name(__file__, "png"))
    run_with_file(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        aspect_ratio=aspect_ratio,
        seed=seed,
        height=height,
        width=width,
        output_path=str(destination),
    )
    print(f"Image saved to: {destination}")


if __name__ == "__main__":
    main()
