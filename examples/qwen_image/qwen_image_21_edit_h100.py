"""Edit an existing image with the native Qwen-Image 2.1 pipeline.

Usage:
    CUDA_VISIBLE_DEVICES=3 python examples/qwen_image/qwen_image_21_edit_h100.py \
        --image_path examples/data/edit2511input.png \
        --output_path work_dirs/qwen-image-2.1-edit.png
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
    "name": "qwen_image_2.1_edit",
    "model_root": os.path.join(os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo"), "Qwen-Image-2.1"),
    "prompt": '这个女生看着面前的电视屏幕，屏幕上面写着"阿里巴巴"',
    "image_path": "examples/data/edit2511input.png",
    "seed": 42,
    "num_inference_steps": 40,
}

PIPELINE_CONTRACT = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=["i2i"],
    task_contracts={
        "i2i": build_task_contract_template(
            "i2i",
            required_inputs=["first_image_path"],
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
    image: Image.Image,
    seed: int = PPL_CONFIG["seed"],
    height: int | None = None,
    width: int | None = None,
) -> list[Image.Image]:
    """Apply an editing instruction to one source image."""

    return pipeline(
        prompt=prompt,
        image=image,
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
    seed: int = PPL_CONFIG["seed"],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, str]:
    """Save an edited image for the standard service entrypoint."""

    with Image.open(first_image_path) as source:
        images = run(pipeline, prompt, source.copy(), seed=seed, height=height, width=width)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(destination)
    return {"output_path": str(destination)}


@click.command()
@click.option("--gpu_num", default=1, type=int, show_default=True)
@click.option("--model_root", default=PPL_CONFIG["model_root"], show_default=True)
@click.option("--image_path", default=PPL_CONFIG["image_path"], type=click.Path(exists=True), show_default=True)
@click.option("--prompt", default=PPL_CONFIG["prompt"], show_default=True)
@click.option("--seed", default=PPL_CONFIG["seed"], type=int, show_default=True)
@click.option("--height", type=int, default=None)
@click.option("--width", type=int, default=None)
@click.option("--output_path", type=click.Path(path_type=Path), default=Path("work_dirs/qwen-image-2.1-edit.png"))
def main(
    gpu_num: int,
    model_root: str,
    image_path: str,
    prompt: str,
    seed: int,
    height: int | None,
    width: int | None,
    output_path: Path,
) -> None:
    """Run Qwen-Image 2.1 image editing."""

    pipeline = get_pipeline(gpu_num, model_root)
    result = run_with_file(pipeline, prompt, image_path, str(output_path), seed=seed, height=height, width=width)
    print(f"Image saved to: {result['output_path']}")


if __name__ == "__main__":
    main()
