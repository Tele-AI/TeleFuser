"""Wan2.1 1.3B T2V with optional attention and quantization optimizations.

Attention can use dense SDPA or Sol-Attn. DiT Linear layers can remain BF16 or
use tf-kernel FP8, TorchAO FP8, or bitsandbytes NF4. The example keeps the DiT
on CUDA so quantized Linear modules are not repeatedly reconstructed.
"""

from __future__ import annotations

import os
import time

import click
import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan21_video import Wan21VideoPipeline, Wan21VideoPipelineConfig
from telefuser.utils.utils import get_example_name
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = {
    "model_root": TF_MODEL_ZOO_PATH + "/Wan2.1-T2V-1.3B",
    "negative_prompt": (
        "Camera shake, overly saturated colors, overexposed, static, blurry details, subtitles, "
        "worst quality, low quality, JPEG compression artifacts, ugly, incomplete, deformed limbs"
    ),
    "num_inference_steps": 40,
    "num_frames": 81,
    "resolution": "480p",
    "cfg_scale": 5.0,
    "sigma_shift": 8.0,
}


def configure_attention_backends() -> None:
    """Configure dense attention backends used directly or by SOL fallbacks."""
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def make_quant_config(quantization: str) -> QuantConfig:
    """Build the online quantization config used for the Wan DiT."""
    if quantization == "none":
        return QuantConfig()
    if quantization == "tf-kernel-fp8":
        return QuantConfig(
            enabled=True,
            quant_type=QuantType.FP8,
            kernel_backend=QuantKernelBackend.TF_KERNEL,
        )
    if quantization == "torchao-fp8":
        return QuantConfig(
            enabled=True,
            quant_type=QuantType.TORCHAO_FP8,
            kernel_backend=QuantKernelBackend.TORCHAO,
        )
    if quantization == "bnb-nf4":
        return QuantConfig(
            enabled=True,
            quant_type=QuantType.BNB_NF4,
            kernel_backend=QuantKernelBackend.BITSANDBYTES,
        )
    raise ValueError("quantization must be 'none', 'tf-kernel-fp8', 'torchao-fp8', or 'bnb-nf4'")


def make_attention_config(
    attention: str,
    *,
    dense_timesteps: int = 10,
    dense_layers: int = 1,
    tau: float = 1.0,
    threshold_type: str = "diag",
    kv_splits: int | str = "auto",
) -> AttentionConfig:
    """Build the selected dense or Sol-Attn configuration."""
    if attention == "dense":
        return AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
    if attention not in ("sol", "sol-fp8"):
        raise ValueError("attention must be 'dense', 'sol', or 'sol-fp8'")
    return AttentionConfig.sol_attention(
        dense_timesteps=dense_timesteps,
        dense_layers=dense_layers,
        tau=tau,
        threshold_type=threshold_type,
        kv_splits=kv_splits,
        sol_fp8=attention == "sol-fp8",
    )


def get_pipeline(
    *,
    model_root: str = PPL_CONFIG["model_root"],
    attention: str = "dense",
    quantization: str = "none",
    dense_timesteps: int = 10,
    dense_layers: int = 1,
    tau: float = 1.0,
    threshold_type: str = "diag",
    kv_splits: int | str = "auto",
    sample_solver: str = "euler",
) -> Wan21VideoPipeline:
    """Load Wan2.1 with independently selectable attention and quantization."""
    quant_config = make_quant_config(quantization)
    module_manager = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    module_manager.load_model(f"{model_root}/Wan2.1_VAE.pth", device="cpu", torch_dtype=torch.bfloat16)
    module_manager.load_model(
        f"{model_root}/diffusion_pytorch_model.safetensors",
        device="cuda",
        torch_dtype=torch.bfloat16,
        quant_config=quant_config,
    )
    module_manager.load_model(f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth", device="cpu", torch_dtype=torch.bfloat16)

    pipeline = Wan21VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    config = Wan21VideoPipelineConfig()
    config.dit_config.attention_config = make_attention_config(
        attention,
        dense_timesteps=dense_timesteps,
        dense_layers=dense_layers,
        tau=tau,
        threshold_type=threshold_type,
        kv_splits=kv_splits,
    )
    config.dit_config.quant_config = quant_config
    config.dit_config.offload_config.offload_type = WeightOffloadType.NO_CPU_OFFLOAD
    config.sample_solver = sample_solver
    config.enable_metrics = True
    pipeline.init(module_manager, config)
    return pipeline


def run(
    pipeline: Wan21VideoPipeline,
    prompt: str,
    *,
    seed: int = 42,
    resolution: str = "480p",
    width: int | None = None,
    height: int | None = None,
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    num_frames: int = PPL_CONFIG["num_frames"],
    cfg_scale: float = PPL_CONFIG["cfg_scale"],
    sigma_shift: float = PPL_CONFIG["sigma_shift"],
):
    """Generate one deterministic validation video."""
    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if width is None or height is None:
        width, height = get_target_video_size_from_ratio(
            "16:9", resolution=resolution, height_division_factor=2, width_division_factor=2
        )
    return pipeline(
        prompt=prompt,
        negative_prompt=PPL_CONFIG["negative_prompt"],
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        cfg_scale=cfg_scale,
        seed=seed,
        height=height,
        width=width,
        sigma_shift=sigma_shift,
        tiled=True,
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
@click.option("--attention", default="dense", type=click.Choice(["dense", "sol", "sol-fp8"]))
@click.option(
    "--quantization",
    default="none",
    type=click.Choice(["none", "tf-kernel-fp8", "torchao-fp8", "bnb-nf4"]),
)
@click.option("--dense-timesteps", default=10, type=int)
@click.option("--dense-layers", default=1, type=int)
@click.option("--tau", default=1.0, type=float)
@click.option("--threshold-type", default="diag", type=click.Choice(["diag", "exact"]))
@click.option("--kv-splits", default="auto", type=click.Choice(["auto", "1", "2", "4"]))
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
    attention: str,
    quantization: str,
    dense_timesteps: int,
    dense_layers: int,
    tau: float,
    threshold_type: str,
    kv_splits: str,
    output: str,
) -> None:
    """Run Wan2.1 with optional attention and quantization optimizations."""
    configure_attention_backends()
    pipeline = get_pipeline(
        model_root=model_root,
        attention=attention,
        quantization=quantization,
        dense_timesteps=dense_timesteps,
        dense_layers=dense_layers,
        tau=tau,
        threshold_type=threshold_type,
        kv_splits=kv_splits if kv_splits == "auto" else int(kv_splits),
        sample_solver=sample_solver,
    )
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
        f"attention={attention} quantization={quantization} elapsed_s={elapsed:.2f} "
        f"throughput_fps={num_frames / elapsed:.4f} "
        f"peak_allocated_gib={peak_allocated:.3f} peak_reserved_gib={peak_reserved:.3f} output={output}"
    )


if __name__ == "__main__":
    main()
