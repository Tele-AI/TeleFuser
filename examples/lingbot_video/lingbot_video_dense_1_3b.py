"""LingBot-Video Dense 1.3B CLI and service example."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from PIL import Image
from diffusers.utils import export_to_video

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_video import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NEGATIVE_PROMPT_IMAGE,
    LingBotVideoPipeline,
    LingBotVideoPipelineConfig,
    LingBotVideoRequest,
    default_negative_caption,
    load_lingbot_video_dense_transformer,
    load_lingbot_video_model_config,
    num_frames_from_duration,
    parse_lingbot_video_prompt,
)
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template
from telefuser.utils.video import get_target_video_size_from_ratio

_DEFAULT_PROMPT = (Path(__file__).parent / "assets" / "t2v_5s.json.example").read_text(encoding="utf-8")
TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")

PPL_CONFIG: dict[str, Any] = {
    "name": "lingbot_video_dense_1_3b",
    "model_root": TF_MODEL_ZOO_PATH + "/lingbot/lingbot-video-dense-1.3b",
    "variant": "dense",
    "supports_refiner": False,
    "prompt": _DEFAULT_PROMPT,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "fps": 24,
    "target_video_length": 5,
    "resolution": "480p",
    "aspect_ratio": "16:9",
    "seed": 42,
    "cfg_parallel_degree": 1,
}


def _build_contract() -> dict[str, Any]:
    """Build the standard service contract with LingBot-specific defaults."""
    task_contracts = {}
    for task in ("t2i", "t2v", "i2v"):
        overrides = {
            "negative_prompt": {"default": DEFAULT_NEGATIVE_PROMPT_IMAGE if task == "t2i" else DEFAULT_NEGATIVE_PROMPT},
            "seed": {"default": PPL_CONFIG["seed"]},
            "resolution": {"default": PPL_CONFIG["resolution"]},
            "aspect_ratio": {"default": PPL_CONFIG["aspect_ratio"]},
        }
        if task != "t2i":
            overrides["target_video_length"] = {"default": PPL_CONFIG["target_video_length"]}
        task_contracts[task] = build_task_contract_template(task, parameter_overrides=overrides)
    return build_pipeline_manifest(
        pipeline_name=PPL_CONFIG["name"],
        supported_tasks=task_contracts,
        task_contracts=task_contracts,
    )


CONTRACT = _build_contract()
PIPELINE_CONTRACT = CONTRACT


def _resolve_runtime_device(device: str, parallel_config: ParallelConfig) -> tuple[str, int]:
    """Resolve the torchrun-local device or the parent device for native workers."""
    if parallel_config.world_size == 1:
        parsed = torch.device(device)
        return device, parsed.index or 0
    if torch.device(device).type != "cuda":
        raise ValueError("LingBot distributed inference requires CUDA devices")
    local_rank = torch.cuda.current_device()
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}", local_rank


def build_pipeline(
    model_root: str | Path,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
    guidance_scale: float = PPL_CONFIG["guidance_scale"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    shift: float = 3.0,
    attention_config: AttentionConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    batch_cfg: bool | None = None,
) -> LingBotVideoPipeline:
    """Load Dense modules and initialize the complete pipeline."""
    try:
        from diffusers import AutoencoderKLWan
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("LingBot-Video requires diffusers and transformers") from exc

    parallel_config = parallel_config or ParallelConfig()
    if batch_cfg is None:
        batch_cfg = parallel_config.world_size > 1 and parallel_config.cfg_degree == 1
    if batch_cfg and parallel_config.cfg_degree > 1:
        raise ValueError("LingBot CFG parallel and batch CFG are mutually exclusive")
    if parallel_config.enable_fsdp and cpu_offload:
        raise ValueError("LingBot FSDP inference requires cpu_offload=False")
    device, device_id = _resolve_runtime_device(device, parallel_config)
    root = Path(model_root)
    transformer_dir = root / "transformer"
    offload_config = OffloadConfig(
        offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD if cpu_offload else WeightOffloadType.NO_CPU_OFFLOAD
    )
    runtime_config = ModelRuntimeConfig(
        device_type=torch.device(device).type,
        device_id=device_id,
        torch_dtype=torch_dtype,
        attention_config=attention_config or AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
        offload_config=offload_config,
        parallel_config=parallel_config,
    )

    module_manager = ModuleManager(torch_dtype=torch_dtype, device="cpu")
    module_manager.add_module(
        load_lingbot_video_dense_transformer(transformer_dir, device="cpu", torch_dtype=torch_dtype),
        name="transformer",
        path=str(transformer_dir),
    )
    module_manager.load_from_huggingface(
        str(root / "processor"),
        module_source="transformers",
        module_name="processor",
        module_class=AutoProcessor,
    )
    module_manager.load_from_huggingface(
        str(root / "text_encoder"),
        module_source="transformers",
        module_name="text_encoder",
        module_class=Qwen3VLForConditionalGeneration,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    module_manager.load_from_huggingface(
        str(root / "vae"),
        module_source="diffusers",
        module_name="vae",
        module_class=AutoencoderKLWan,
        torch_dtype=torch.float32,
    )
    module_manager.load_from_huggingface(
        str(root / "scheduler"),
        module_source="diffusers",
        module_name="scheduler",
        module_class=FlowUniPCMultistepScheduler,
    )

    pipeline = LingBotVideoPipeline(device=device, torch_dtype=torch_dtype)
    pipeline.init(
        module_manager,
        LingBotVideoPipelineConfig(
            model=load_lingbot_video_model_config(transformer_dir, variant="dense"),
            text_encoding_config=runtime_config,
            dit_config=runtime_config,
            vae_config=runtime_config,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            shift=shift,
            batch_cfg=batch_cfg,
            enable_denoising_parallel=parallel_config.world_size > 1,
        ),
    )
    return pipeline


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    cfg_parallel_degree: int | None = None,
) -> object:
    """Load the fixed Dense 1.3B checkpoint on one or four GPUs."""
    if parallelism not in {1, 4}:
        raise ValueError("LingBot-Video Dense supports parallelism=1 or 4")
    cfg_parallel_degree = int(PPL_CONFIG["cfg_parallel_degree"]) if cfg_parallel_degree is None else cfg_parallel_degree
    if cfg_parallel_degree not in {1, 2} or parallelism % cfg_parallel_degree:
        raise ValueError("LingBot CFG parallel degree must be 1 or 2 and divide parallelism")
    return build_pipeline(
        model_root,
        parallel_config=ParallelConfig(
            device_ids=list(range(parallelism)),
            cfg_degree=cfg_parallel_degree,
            sp_ulysses_degree=parallelism // cfg_parallel_degree,
            enable_fsdp=parallelism > 1,
        ),
    )


def _load_condition_image(path: str) -> torch.Tensor:
    """Load an RGB image path as a raw [0,255] TI2V tensor."""
    pixels = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()


def _resolve_video_size(aspect_ratio: str, resolution: str) -> tuple[int, int]:
    """Resolve output geometry using the validated LingBot 480p landscape preset."""
    if resolution == "480p" and aspect_ratio == "16:9":
        return 832, 480
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution,
        height_division_factor=16,
        width_division_factor=16,
    )
    if width is None or height is None:
        raise ValueError(f"unsupported LingBot-Video resolution: {resolution}")
    return width, height


def _generator(pipeline: object, seed: int) -> torch.Generator:
    """Create a deterministic generator on the pipeline device type."""
    device_type = torch.device(getattr(pipeline, "device", "cpu")).type
    return torch.Generator(device_type).manual_seed(seed)


def run(
    pipeline: object,
    prompt: str = PPL_CONFIG["prompt"],
    negative_prompt: str | None = None,
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    target_video_length: float | None = None,
    task: str = "t2v",
    first_image_path: str = "",
) -> torch.Tensor:
    """Run Dense T2I, T2V, or I2V and return normalized RGB frames."""
    if task not in {"t2i", "t2v", "i2v"}:
        raise ValueError(f"unsupported LingBot-Video task: {task}")
    if task == "i2v" and not first_image_path:
        raise ValueError("LingBot-Video i2v requires first_image_path")

    caption, prompt_duration = parse_lingbot_video_prompt(json.loads(prompt))
    width, height = _resolve_video_size(aspect_ratio, resolution)
    duration = target_video_length if target_video_length is not None else prompt_duration
    if duration is None:
        duration = float(PPL_CONFIG["target_video_length"])
    num_frames = 1 if task == "t2i" else num_frames_from_duration(duration, fps=int(PPL_CONFIG["fps"]))
    resolved_negative_prompt = negative_prompt if negative_prompt is not None else default_negative_caption(num_frames)
    return pipeline.generate(
        LingBotVideoRequest(
            caption=caption,
            height=height,
            width=width,
            num_frames=num_frames,
            image=_load_condition_image(first_image_path) if first_image_path else None,
        ),
        negative_caption=resolved_negative_prompt,
        generator=_generator(pipeline, seed),
    ).output


def _save_output(frames: torch.Tensor, output_path: str) -> dict[str, str]:
    """Encode normalized RGB without an intermediate uint8 video conversion."""
    path = Path(output_path)
    video = frames[0].permute(1, 2, 3, 0).float().clamp(0.0, 1.0).cpu().numpy()
    if video.shape[0] == 1:
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((video[0] * 255).round().astype(np.uint8)).save(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(list(video), str(path), fps=int(PPL_CONFIG["fps"]))
    return {"output_path": str(path)}


def run_with_file(
    pipeline: object,
    prompt: str = PPL_CONFIG["prompt"],
    negative_prompt: str | None = None,
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "output.mp4",
    target_video_length: float | None = None,
    resolution: str = PPL_CONFIG["resolution"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    task: str = "t2v",
    first_image_path: str = "",
    **_: object,
) -> dict[str, str]:
    """Run the Dense example and save its image or video output."""
    frames = run(
        pipeline,
        prompt,
        negative_prompt,
        seed,
        resolution,
        aspect_ratio,
        target_video_length,
        task,
        first_image_path,
    )
    return _save_output(frames, output_path)


@click.command()
@click.option("--gpu_num", default=1, type=int, help="Number of GPUs: 1 or 4")
@click.option("--cfg_parallel_degree", default=PPL_CONFIG["cfg_parallel_degree"], type=click.Choice([1, 2]))
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Dense 1.3B checkpoint root")
@click.option("--prompt", default=PPL_CONFIG["prompt"], help="Structured JSON caption")
@click.option("--negative_prompt", default=None, help="Optional structured negative caption")
@click.option("--seed", default=PPL_CONFIG["seed"], type=int)
@click.option("--resolution", default=PPL_CONFIG["resolution"])
@click.option("--aspect_ratio", default=PPL_CONFIG["aspect_ratio"])
@click.option("--target_video_length", default=None, type=float)
@click.option("--task", default="t2v", type=click.Choice(["t2i", "t2v", "i2v"]))
@click.option("--first_image_path", default="", help="Required for i2v")
@click.option("--output_path", default="output.mp4")
def main(
    gpu_num: int,
    cfg_parallel_degree: int,
    model_root: str,
    prompt: str,
    negative_prompt: str | None,
    seed: int,
    resolution: str,
    aspect_ratio: str,
    target_video_length: float | None,
    task: str,
    first_image_path: str,
    output_path: str,
) -> None:
    """Generate with the LingBot-Video Dense 1.3B checkpoint."""
    pipeline = get_pipeline(gpu_num, model_root, cfg_parallel_degree)
    try:
        result = run_with_file(
            pipeline,
            prompt,
            negative_prompt,
            seed,
            output_path,
            target_video_length,
            resolution,
            aspect_ratio,
            task,
            first_image_path,
        )
        click.echo(f"Output saved to {result['output_path']}")
    finally:
        stop = getattr(pipeline, "stop", None)
        if callable(stop):
            stop()


if __name__ == "__main__":
    main()
