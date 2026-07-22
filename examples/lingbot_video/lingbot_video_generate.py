"""Generate a LingBot-Video T2I, T2V, or TI2V sample with native TeleFuser stages."""

from __future__ import annotations

import argparse

import torch
from PIL import Image

from telefuser.pipelines.lingbot_video import (
    LingBotVideoRequest,
    build_lingbot_video_pipeline,
    build_lingbot_video_refiner_stage,
    default_negative_caption,
    load_lingbot_video_prompt,
    load_refiner_first_frame,
    num_frames_from_duration,
    prepare_refiner_video,
)


def _image_to_tensor(path: str) -> torch.Tensor:
    """Load an RGB image as the raw [0,255] tensor required by TI2V."""
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--caption-json", required=True, help="Path to the structured JSON caption.")
    parser.add_argument("--output", required=True, help="Output MP4 path, or PNG when --num-frames=1.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--num-frames", type=int, help="Override prompt-file duration with an explicit 4n+1 frame count."
    )
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image", help="Optional first-frame image for TI2V.")
    parser.add_argument("--variant", choices=("dense", "moe"), default="dense")
    parser.add_argument("--negative-caption", default=None, help="Override the official structured negative caption.")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refiner-height", type=int, default=1088)
    parser.add_argument("--refiner-width", type=int, default=1920)
    parser.add_argument("--refiner-steps", type=int, default=8)
    parser.add_argument("--refiner-guidance-scale", type=float, default=3.0)
    parser.add_argument("--refiner-shift", type=float, default=3.0)
    parser.add_argument("--refiner-t-thresh", type=float, default=0.85)
    parser.add_argument("--refiner-tail-steps", type=int, default=2)
    args = parser.parse_args()
    if args.refine and args.variant != "moe":
        raise ValueError("--refine requires --variant moe because the refiner is shipped with the MoE checkpoint")
    caption_text, duration = load_lingbot_video_prompt(args.caption_json)
    num_frames = (
        args.num_frames
        if args.num_frames is not None
        else num_frames_from_duration(duration)
        if duration is not None
        else 121
    )
    source_image = _image_to_tensor(args.image) if args.image else None
    negative_caption = (
        args.negative_caption if args.negative_caption is not None else default_negative_caption(num_frames)
    )
    pipeline = build_lingbot_video_pipeline(
        args.model_dir,
        variant=args.variant,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
    )
    generator = torch.Generator("cuda").manual_seed(args.seed)
    generation = pipeline.generate(
        LingBotVideoRequest(
            caption=caption_text,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            image=source_image,
        ),
        negative_caption=negative_caption,
        generator=generator,
    )
    frames = generation.output
    if args.refine:
        if pipeline.text_stage is None:
            raise RuntimeError("LingBot-Video text stage is required for refiner prompt encoding")
        # The upstream TI2V refiner consumes text-only prompt conditioning. Its
        # frame-zero visual condition is injected through the VAE latent instead.
        if generation.prompt_conditions.has_visual_condition:
            positive, positive_mask = pipeline.text_stage.encode(caption_text)
            negative, negative_mask = pipeline.text_stage.encode(negative_caption)
        else:
            positive = generation.prompt_conditions.positive_prompt_embeds
            negative = generation.prompt_conditions.negative_prompt_embeds
            positive_mask = generation.prompt_conditions.positive_attention_mask
            negative_mask = generation.prompt_conditions.negative_attention_mask
        pipeline.release_gpu_resources()
        refiner = build_lingbot_video_refiner_stage(args.model_dir)
        lowres_video, _ = prepare_refiner_video(
            frames,
            source_fps=24.0,
            height=args.refiner_height,
            width=args.refiner_width,
        )
        clean_first_frame = (
            load_refiner_first_frame(
                args.image,
                target_height=args.refiner_height,
                target_width=args.refiner_width,
                geometry_height=args.height,
                geometry_width=args.width,
            )
            if args.image
            else None
        )
        frames = refiner.refine(
            lowres_video,
            positive,
            negative,
            positive_mask,
            negative_mask,
            num_inference_steps=args.refiner_steps,
            guidance_scale=args.refiner_guidance_scale,
            shift=args.refiner_shift,
            t_thresh=args.refiner_t_thresh,
            tail_steps=args.refiner_tail_steps,
            clean_first_frame=clean_first_frame,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )
    video = frames[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    if video.shape[0] == 1:
        Image.fromarray((video[0] * 255).round().astype("uint8")).save(args.output)
        return
    from diffusers.utils import export_to_video

    export_to_video(list(video), args.output, fps=24)


if __name__ == "__main__":
    main()
