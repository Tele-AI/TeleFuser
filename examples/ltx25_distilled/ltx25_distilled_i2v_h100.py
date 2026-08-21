"""Run the faithful LTX-2.5 distilled image-to-video pipeline on H100 GPUs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import click
import torch
from PIL import Image

from telefuser.core.config import AttnImplType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.ltx25_distilled import (
    LTX25DistilledOutput,
    LTX25DistilledPipeline,
    LTX25ImageCondition,
    build_ltx25_distilled_config,
    load_ltx25_distilled_modules,
)
from telefuser.utils.audio import save_wav
from telefuser.utils.video import save_video

DEFAULT_IMAGE_PATH = Path(__file__).resolve().parents[1] / "data" / "ltx25" / "official_guitar_man.png"

PPL_CONFIG: dict[str, Any] = {
    "name": "ltx25_distilled_i2v_h100",
    "model_root": "/hhb-data/aigc/model_zoo/Lightricks/LTX-2.5/LTX-2.5",
    "height": 512,
    "width": 896,
    "num_frames": 121,
    "frame_rate": 24.0,
    "seed": 42,
    "prompt": "A man with short gray hair plays a red electric guitar.",
    "input_image_path": str(DEFAULT_IMAGE_PATH),
    "video_vae": "diff",
    "attn_impl": "FLASH_ATTN_4",
}

DENSE_ATTN_IMPLS = tuple(
    implementation.name
    for implementation in AttnImplType
    if implementation not in {AttnImplType.RADIAL_ATTN, AttnImplType.LOCAL_SPARSE_ATTN, AttnImplType.SOL_ATTN}
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    video_vae: str = PPL_CONFIG["video_vae"],
    offload: str = "cpu",
    attn_impl: str | AttnImplType = PPL_CONFIG["attn_impl"],
) -> LTX25DistilledPipeline:
    """Load the isolated LTX-2.5 distilled I2V pipeline on one or more H100s."""
    if parallelism not in (1, 2, 4):
        raise ValueError(f"parallelism must be 1, 2, or 4, got {parallelism}")
    if video_vae not in ("diff", "conv"):
        raise ValueError(f"video_vae must be 'diff' or 'conv', got {video_vae!r}")
    if offload not in ("none", "cpu"):
        raise ValueError(f"offload must be 'none' or 'cpu', got {offload!r}")
    selected_video_vae = cast(Literal["diff", "conv"], video_vae)
    selected_offload = cast(Literal["none", "cpu"], offload)
    if isinstance(attn_impl, str):
        try:
            selected_attn_impl = AttnImplType[attn_impl]
        except KeyError as exc:
            raise ValueError(f"Unknown attention implementation {attn_impl!r}; choose from {DENSE_ATTN_IMPLS}") from exc
    else:
        selected_attn_impl = attn_impl
    if selected_attn_impl.name not in DENSE_ATTN_IMPLS:
        raise ValueError(f"LTX-2.5 supports dense attention implementations only, got {selected_attn_impl.name}")
    module_manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    load_ltx25_distilled_modules(
        module_manager,
        model_root,
        video_vae=selected_video_vae,
        torch_dtype=torch.bfloat16,
    )
    pipeline = LTX25DistilledPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipeline.init(
        module_manager,
        build_ltx25_distilled_config(
            "cuda",
            torch.bfloat16,
            selected_video_vae,
            selected_offload,
            parallelism=parallelism,
            attn_impl=selected_attn_impl,
        ),
    )
    return pipeline


def run(
    pipeline: LTX25DistilledPipeline,
    image: Image.Image,
    prompt: str,
    *,
    seed: int = PPL_CONFIG["seed"],
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    num_frames: int = PPL_CONFIG["num_frames"],
    frame_rate: float = PPL_CONFIG["frame_rate"],
    image_frame_index: int = 0,
    image_strength: float = 1.0,
) -> LTX25DistilledOutput:
    """Generate video and synchronized audio from an image and text prompt."""
    images = (
        LTX25ImageCondition(
            image.convert("RGB"),
            frame_idx=image_frame_index,
            strength=image_strength,
        ),
    )
    return pipeline(
        prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        images=images,
    )


def run_with_file(
    pipeline: LTX25DistilledPipeline,
    prompt: str,
    output_path: str,
    input_image_path: str = PPL_CONFIG["input_image_path"],
    seed: int = PPL_CONFIG["seed"],
    height: int = PPL_CONFIG["height"],
    width: int = PPL_CONFIG["width"],
    num_frames: int = PPL_CONFIG["num_frames"],
    frame_rate: float = PPL_CONFIG["frame_rate"],
    image_frame_index: int = 0,
    image_strength: float = 1.0,
    first_image_path: str | None = None,
    **_: object,
) -> dict[str, str]:
    """Generate and save an MP4 with synchronized LTX-2.5 audio."""
    image = Image.open(first_image_path or input_image_path).convert("RGB")
    result = run(
        pipeline,
        image,
        prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        image_frame_index=image_frame_index,
        image_strength=image_strength,
    )
    frames = torch.cat(result.video_chunks).mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as stream:
        audio_path = Path(stream.name)
    try:
        save_wav(result.audio, 48000, str(audio_path))
        save_video(list(frames), str(destination), fps=result.frame_rate, quality=6, audio_path=str(audio_path))
    finally:
        audio_path.unlink(missing_ok=True)
    return {"output_path": str(destination)}


@click.command()
@click.option("--prompt", default=PPL_CONFIG["prompt"], show_default=True)
@click.option("--model-root", default=PPL_CONFIG["model_root"], show_default=True)
@click.option("--output-path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--image-path",
    "input_image_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=PPL_CONFIG["input_image_path"],
    show_default=True,
)
@click.option("--image-frame-index", default=0, show_default=True)
@click.option("--image-strength", default=1.0, show_default=True)
@click.option("--height", default=PPL_CONFIG["height"], show_default=True)
@click.option("--width", default=PPL_CONFIG["width"], show_default=True)
@click.option("--num-frames", default=PPL_CONFIG["num_frames"], show_default=True)
@click.option("--frame-rate", default=PPL_CONFIG["frame_rate"], show_default=True)
@click.option("--seed", default=PPL_CONFIG["seed"], show_default=True)
@click.option("--video-vae", type=click.Choice(["diff", "conv"]), default=PPL_CONFIG["video_vae"], show_default=True)
@click.option("--offload", type=click.Choice(["none", "cpu"]), default="cpu", show_default=True)
@click.option("--gpu-num", type=click.Choice(["1", "2", "4"]), default="1", show_default=True)
@click.option(
    "--attn-impl",
    type=click.Choice(DENSE_ATTN_IMPLS),
    default=PPL_CONFIG["attn_impl"],
    show_default=True,
)
def main(
    prompt: str,
    model_root: str,
    output_path: Path,
    input_image_path: Path,
    image_frame_index: int,
    image_strength: float,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    seed: int,
    video_vae: str,
    offload: str,
    gpu_num: str,
    attn_impl: str,
) -> None:
    """Generate an LTX-2.5 video and synchronized audio from an image."""
    pipeline = get_pipeline(
        parallelism=int(gpu_num),
        model_root=model_root,
        video_vae=video_vae,
        offload=offload,
        attn_impl=attn_impl,
    )
    try:
        run_with_file(
            pipeline,
            input_image_path=str(input_image_path),
            prompt=prompt,
            output_path=str(output_path),
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            image_frame_index=image_frame_index,
            image_strength=image_strength,
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
