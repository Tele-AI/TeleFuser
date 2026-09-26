"""Generate a new image from one or more references with native Qwen-Image 2.1.

Usage:
    CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_reference_h100.py \
        --output_path work_dirs/qwen-image-2.1-reference.png
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import torch
from PIL import Image

from telefuser.pipelines.qwen_image import QwenImage21Pipeline
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template

PPL_CONFIG = {
    "name": "qwen_image_2.1_reference",
    "model_root": os.path.join(os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo"), "Qwen-Image-2.1"),
    "prompt": (
        "让【图1】中的模特换上【图3】中的玛丽珍鞋，拿着【图4】中的手提包，并戴上【图5】中的绒毛帽。"
        "将【图2】中的羽绒服敞开穿在外面，露出原有的内搭上衣。保持模特姿势和背景不变。"
    ),
    "image_paths": (
        "examples/data/qwen_image_21/447da49d-dcd0-46ec-9ca8-20127c0f07d1.webp",
        "examples/data/qwen_image_21/0f2d052b-6ccf-46e2-821d-d073ead679ac.webp",
        "examples/data/qwen_image_21/3f4d7519-6c33-491f-bad8-661f1bf03601.webp",
        "examples/data/qwen_image_21/0f5d38f9-d04a-498e-aaef-9dae01da8c6d.webp",
        "examples/data/qwen_image_21/3acf3276-0572-490e-b8ce-334833a6ca20.webp",
    ),
    "seed": 42,
    "num_inference_steps": 40,
}

PIPELINE_CONTRACT = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=["i2i"],
    task_contracts={
        "i2i": build_task_contract_template(
            "i2i",
            parameter_overrides={
                "prompt": {"default": PPL_CONFIG["prompt"]},
                "seed": {"default": PPL_CONFIG["seed"]},
            },
            excluded_parameters=["negative_prompt", "resolution", "aspect_ratio"],
        )
    },
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    device: str | None = None,
) -> QwenImage21Pipeline:
    """Load the shared native Qwen-Image 2.1 pipeline."""

    if parallelism != 1:
        raise ValueError("Qwen-Image 2.1 example currently supports one GPU")
    return QwenImage21Pipeline.from_pretrained(model_root, device=device or "cuda", torch_dtype=torch.bfloat16)


def run(
    pipeline: QwenImage21Pipeline,
    prompt: str,
    images: list[Image.Image],
    seed: int = PPL_CONFIG["seed"],
    height: int | None = None,
    width: int | None = None,
) -> list[Image.Image]:
    """Generate a new image from one to ten shared reference images."""

    if not 1 <= len(images) <= 10:
        raise ValueError("Pass one to ten reference images")
    return pipeline(
        prompt=prompt,
        image=images,
        seed=seed,
        height=height,
        width=width,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
    )


def run_with_file(
    pipeline: QwenImage21Pipeline,
    prompt: str,
    first_image_path: str,
    output_path: str,
    reference_image_paths: tuple[str, ...] = (),
    seed: int = PPL_CONFIG["seed"],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, str]:
    """Save reference-guided output for CLI or the single-image service entrypoint."""

    paths = (first_image_path, *reference_image_paths)
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.copy())
    outputs = run(pipeline, prompt, images, seed=seed, height=height, width=width)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    outputs[0].save(destination)
    return {"output_path": str(destination)}


@click.command()
@click.option("--gpu_num", default=1, type=int, show_default=True)
@click.option("--model_root", default=PPL_CONFIG["model_root"], show_default=True)
@click.option(
    "--image_path",
    "image_paths",
    multiple=True,
    type=click.Path(exists=True),
    default=PPL_CONFIG["image_paths"],
    show_default=True,
)
@click.option("--prompt", default=PPL_CONFIG["prompt"], show_default=True)
@click.option("--seed", default=PPL_CONFIG["seed"], type=int, show_default=True)
@click.option("--height", type=int, default=None)
@click.option("--width", type=int, default=None)
@click.option("--output_path", type=click.Path(path_type=Path), default=Path("work_dirs/qwen-image-2.1-reference.png"))
def main(
    gpu_num: int,
    model_root: str,
    image_paths: tuple[str, ...],
    prompt: str,
    seed: int,
    height: int | None,
    width: int | None,
    output_path: Path,
) -> None:
    """Run Qwen-Image 2.1 reference-guided generation."""

    if not 1 <= len(image_paths) <= 10:
        raise click.BadParameter("Pass one to ten --image_path values")
    pipeline = get_pipeline(gpu_num, model_root)
    result = run_with_file(
        pipeline,
        prompt,
        image_paths[0],
        str(output_path),
        reference_image_paths=image_paths[1:],
        seed=seed,
        height=height,
        width=width,
    )
    print(f"Image saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
