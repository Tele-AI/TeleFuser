"""Wan2.1 1.3B T2V with tf-kernel FP8 Linear layers and dense attention."""

from __future__ import annotations

import time

import click
import torch

if __package__:
    from examples.wan_video.wan21_1_3b_text_to_video_sol_fp8_h100 import (
        PPL_CONFIG,
        configure_attention_backends,
        run,
    )
    from examples.wan_video.wan21_1_3b_text_to_video_sol_fp8_h100 import (
        get_pipeline as get_quantized_pipeline,
    )
else:
    from wan21_1_3b_text_to_video_sol_fp8_h100 import (
        PPL_CONFIG,
        configure_attention_backends,
        run,
    )
    from wan21_1_3b_text_to_video_sol_fp8_h100 import (
        get_pipeline as get_quantized_pipeline,
    )

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import save_video


def get_pipeline(
    *,
    model_root: str = PPL_CONFIG["model_root"],
    quantization: str = "tf-kernel-fp8",
    sample_solver: str = "euler",
):
    """Load Wan2.1 with online FP8 quantization and dense SDPA attention."""
    return get_quantized_pipeline(
        model_root=model_root,
        quantization=quantization,
        attention_config=AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
        sample_solver=sample_solver,
    )


@click.command()
@click.option("--prompt", default="A small paper boat floating down a sunlit stream.")
@click.option("--seed", default=42, type=int)
@click.option("--resolution", default="480p", type=click.Choice(["480p", "720p"]))
@click.option("--width", type=int)
@click.option("--height", type=int)
@click.option("--num-inference-steps", default=PPL_CONFIG["num_inference_steps"], type=int)
@click.option("--num-frames", default=PPL_CONFIG["num_frames"], type=int)
@click.option("--cfg-scale", default=PPL_CONFIG["cfg_scale"], type=float)
@click.option("--sigma-shift", default=PPL_CONFIG["sigma_shift"], type=float)
@click.option("--sample-solver", default="euler", type=click.Choice(["euler", "unipc"]))
@click.option("--model-root", default=PPL_CONFIG["model_root"])
@click.option(
    "--quantization",
    default="tf-kernel-fp8",
    type=click.Choice(["tf-kernel-fp8", "torchao-fp8", "none"]),
)
@click.option("--output", default=get_example_name(__file__, "mp4"))
def main(
    prompt: str,
    seed: int,
    resolution: str,
    width: int | None,
    height: int | None,
    num_inference_steps: int,
    num_frames: int,
    cfg_scale: float,
    sigma_shift: float,
    sample_solver: str,
    model_root: str,
    quantization: str,
    output: str,
) -> None:
    """Run dense Wan2.1 with tf-kernel/TorchAO FP8 or a BF16 baseline."""
    configure_attention_backends()
    pipeline = get_pipeline(model_root=model_root, quantization=quantization, sample_solver=sample_solver)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    video = run(
        pipeline,
        prompt,
        seed=seed,
        resolution=resolution,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        sigma_shift=sigma_shift,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    save_video(video, output, fps=16, quality=6)
    peak_allocated = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    click.echo(
        f"quantization={quantization} sol_attention=false elapsed_s={elapsed:.2f} "
        f"throughput_fps={num_frames / elapsed:.4f} "
        f"peak_allocated_gib={peak_allocated:.3f} peak_reserved_gib={peak_reserved:.3f} output={output}"
    )


if __name__ == "__main__":
    main()
