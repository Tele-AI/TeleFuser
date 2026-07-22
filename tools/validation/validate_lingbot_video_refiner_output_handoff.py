"""Compare final Refiner outputs from in-memory and MP4 base-video handoffs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lingbot_video.lingbot_video_moe_30b import build_pipeline, build_refiner
from telefuser.pipelines.lingbot_video import (
    LingBotVideoRequest,
    default_negative_caption,
    load_lingbot_video_prompt,
    load_refiner_first_frame,
    load_refiner_video_file,
    num_frames_from_duration,
    prepare_refiner_video,
)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    """Return L0/L1 metrics for one in-memory versus MP4 handoff tensor pair."""
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    ref = reference.float().cpu()
    got = candidate.float().cpu()
    delta = (got - ref).abs()
    exact_mismatch_count = int(torch.count_nonzero(reference.cpu() != candidate.cpu()).item())
    cosine = (
        1.0
        if exact_mismatch_count == 0
        else torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    )
    return {
        "shape_match": True,
        "dtype_match": reference.dtype == candidate.dtype,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm().div(ref.norm().clamp_min(1e-12)).item()),
        "cosine": float(max(-1.0, min(1.0, cosine))),
        "exact_mismatch_count": exact_mismatch_count,
    }


def decoded_frame_quality_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | None | bool]:
    """Return local-SSIM and PSNR for decoded ``[B,3,F,H,W]`` RGB tensors."""
    if reference.shape != candidate.shape:
        return {"shape_match": False, "psnr_db": None, "ssim": None}
    if reference.ndim != 5 or reference.shape[1] != 3:
        raise ValueError("decoded frames must have shape [B,3,F,H,W]")
    ref = reference.float().cpu().clamp(0.0, 1.0)
    got = candidate.float().cpu().clamp(0.0, 1.0)
    mse = torch.mean((got - ref).square())
    psnr = None if mse == 0 else float(-10.0 * torch.log10(mse).item())
    batch, channels, frames, height, width = ref.shape
    kernel = max(1, min(11, height, width) | 1)
    padding = kernel // 2
    ref_2d = ref.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    got_2d = got.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    mean_ref = torch.nn.functional.avg_pool2d(ref_2d, kernel, stride=1, padding=padding)
    mean_got = torch.nn.functional.avg_pool2d(got_2d, kernel, stride=1, padding=padding)
    variance_ref = (
        torch.nn.functional.avg_pool2d(ref_2d.square(), kernel, stride=1, padding=padding) - mean_ref.square()
    )
    variance_got = (
        torch.nn.functional.avg_pool2d(got_2d.square(), kernel, stride=1, padding=padding) - mean_got.square()
    )
    covariance = (
        torch.nn.functional.avg_pool2d(ref_2d * got_2d, kernel, stride=1, padding=padding) - mean_ref * mean_got
    )
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_ref * mean_got + c1) * (2 * covariance + c2)) / (
        (mean_ref.square() + mean_got.square() + c1) * (variance_ref + variance_got + c2)
    )
    return {"shape_match": True, "psnr_db": psnr, "ssim": float(ssim.mean().clamp(-1.0, 1.0).item())}


def side_by_side_frames(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Place matching decoded videos side by side for visual handoff review."""
    if reference.shape != candidate.shape:
        raise ValueError("comparison videos must have matching shapes")
    if reference.ndim != 5 or reference.shape[1] != 3:
        raise ValueError("comparison videos must have shape [B,3,F,H,W]")
    return torch.cat((reference, candidate), dim=-1)


def _write_video_tensor(video: torch.Tensor, path: Path, fps: int = 24) -> None:
    """Write a normalized ``[B,3,F,H,W]`` RGB tensor without rescaling uint8 frames."""
    frames = video[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.shape[0] == 1:
        Image.fromarray((frames[0] * 255).round().astype("uint8")).save(path)
        return
    from diffusers.utils import export_to_video

    export_to_video(list(frames), str(path), fps=fps)


def write_side_by_side_video(reference: torch.Tensor, candidate: torch.Tensor, path: Path, fps: int = 24) -> None:
    """Write in-memory and MP4-refined decoded frames as a visual comparison artifact."""
    comparison = side_by_side_frames(reference, candidate)
    _write_video_tensor(comparison, path, fps)


def _measure(operation: Any) -> tuple[Any, float]:
    _synchronize()
    started = time.perf_counter()
    result = operation()
    _synchronize()
    return result, time.perf_counter() - started


def _image_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(__import__("numpy").asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()


def build_refiner_handoff_report(
    *,
    memory_input: torch.Tensor,
    mp4_input: torch.Tensor,
    memory_metadata: dict[str, Any],
    mp4_metadata: dict[str, Any],
    memory_output: torch.Tensor,
    mp4_output: torch.Tensor,
    memory_seconds: float,
    mp4_seconds: float,
) -> dict[str, Any]:
    """Report effects of replacing the source's lossy MP4 handoff."""
    return {
        "comparison_baseline": "source_mp4_round_trip",
        "mp4_round_trip_is_lossy": True,
        "metadata_match": memory_metadata == mp4_metadata,
        "in_memory_to_mp4_input": tensor_metrics(memory_input.cpu(), mp4_input.cpu()),
        "final_output": tensor_metrics(memory_output.cpu(), mp4_output.cpu()),
        "final_output_quality": decoded_frame_quality_metrics(memory_output, mp4_output),
        "memory_refiner_seconds": memory_seconds,
        "mp4_refiner_seconds": mp4_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--caption-json", type=Path, required=True)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--negative-caption", default=None)
    parser.add_argument("--image")
    parser.add_argument("--refiner-height", type=int, default=1088)
    parser.add_argument("--refiner-width", type=int, default=1920)
    parser.add_argument("--refiner-steps", type=int, default=8)
    parser.add_argument("--refiner-guidance-scale", type=float, default=3.0)
    parser.add_argument("--refiner-shift", type=float, default=3.0)
    parser.add_argument("--refiner-t-thresh", type=float, default=0.85)
    parser.add_argument("--refiner-tail-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-cpu-offload", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--comparison-output",
        type=Path,
        help="Optional side-by-side memory|MP4 Refiner output video for human review.",
    )
    args = parser.parse_args()

    caption, duration = load_lingbot_video_prompt(args.caption_json)
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
    pipeline, base_load_seconds = _measure(
        lambda: build_pipeline(
            args.model_dir,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            cpu_offload=False if args.disable_cpu_offload else None,
        )
    )
    generation, base_generation_seconds = _measure(
        lambda: pipeline.generate(
            LingBotVideoRequest(
                caption=caption,
                height=args.height,
                width=args.width,
                num_frames=num_frames,
                image=source_image,
            ),
            negative_caption=negative_caption,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        )
    )
    frames = generation.output
    if pipeline.text_stage is None:
        raise RuntimeError("LingBot refiner requires the base text stage")
    if generation.prompt_conditions.has_visual_condition:
        positive, positive_mask = pipeline.text_stage.encode(caption)
    else:
        positive = generation.prompt_conditions.positive_prompt_embeds
        positive_mask = generation.prompt_conditions.positive_attention_mask
    negative = torch.zeros_like(positive)
    negative_mask = positive_mask.clone()
    _, base_release_seconds = _measure(pipeline.release_gpu_resources)
    refiner, refiner_load_seconds = _measure(
        lambda: build_refiner(args.model_dir, cpu_offload=not args.disable_cpu_offload)
    )
    memory_input, memory_metadata = prepare_refiner_video(
        frames, source_fps=24.0, height=args.refiner_height, width=args.refiner_width
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

    def refine(video: torch.Tensor) -> tuple[torch.Tensor, float]:
        return _measure(
            lambda: refiner.refine(
                video,
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
        )

    memory_output, memory_seconds = refine(memory_input)
    with tempfile.TemporaryDirectory() as temporary_directory:
        mp4_path = Path(temporary_directory) / "base.mp4"
        _write_video_tensor(frames, mp4_path)
        mp4_input, mp4_metadata = load_refiner_video_file(
            mp4_path, height=args.refiner_height, width=args.refiner_width
        )
        mp4_output, mp4_seconds = refine(mp4_input)
    report = build_refiner_handoff_report(
        memory_input=memory_input,
        mp4_input=mp4_input,
        memory_metadata=memory_metadata,
        mp4_metadata=mp4_metadata,
        memory_output=memory_output,
        mp4_output=mp4_output,
        memory_seconds=memory_seconds,
        mp4_seconds=mp4_seconds,
    )
    report.update(
        model_dir=str(args.model_dir),
        height=args.height,
        width=args.width,
        num_frames=num_frames,
        steps=args.steps,
        refiner_height=args.refiner_height,
        refiner_width=args.refiner_width,
        refiner_steps=args.refiner_steps,
        cpu_offload=not args.disable_cpu_offload,
        base_load_seconds=base_load_seconds,
        base_generation_seconds=base_generation_seconds,
        base_release_seconds=base_release_seconds,
        refiner_load_seconds=refiner_load_seconds,
    )
    if args.comparison_output is not None:
        write_side_by_side_video(memory_output, mp4_output, args.comparison_output)
        report["comparison_output"] = str(args.comparison_output)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
