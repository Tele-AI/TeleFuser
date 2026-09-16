import os

import click
import torch
from PIL import Image

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")


@click.command()
@click.option(
    "--model_root",
    default=os.path.join(TF_MODEL_ZOO_PATH, "Qwen-Image-Edit-2509"),
    help="Local Diffusers checkpoint directory",
)
@click.option("--image_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Input image path")
@click.option(
    "--prompt",
    default='这个女生看着面前的电视屏幕，屏幕上面写着"阿里巴巴"',
    help="Image editing instruction",
)
@click.option("--output", default="output_image_edit_plus.png", help="Output image path")
def main(model_root: str, image_path: str, prompt: str, output: str) -> None:
    """Run the fixed-parameter Diffusers comparison for Qwen-Image-Edit."""
    from diffusers import QwenImageEditPlusPipeline

    pipeline = QwenImageEditPlusPipeline.from_pretrained(model_root, torch_dtype=torch.bfloat16)
    pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=None)
    image = Image.open(image_path).convert("RGB")
    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(0),
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    with torch.inference_mode():
        output_image = pipeline(**inputs).images[0]
    output_image.save(output)
    print("image saved at", os.path.abspath(output))


if __name__ == "__main__":
    main()
