"""LingBot-Video MoE 30B plus refiner CLI and service example."""

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
    LingBotVideoDenoisingStage,
    LingBotVideoPipeline,
    LingBotVideoPipelineConfig,
    LingBotVideoRefinerStage,
    LingBotVideoRequest,
    LingBotVideoVAEDecodeStage,
    LingBotVideoVAEEncodeStage,
    default_negative_caption,
    load_lingbot_video_model_config,
    load_refiner_first_frame,
    num_frames_from_duration,
    parse_lingbot_video_prompt,
    prepare_refiner_video,
)
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template
from telefuser.utils.video import get_target_video_size_from_ratio
from telefuser.worker.parallel_worker import ParallelWorker

_DEFAULT_PROMPT = (Path(__file__).parent / "assets" / "t2v_5s.json.example").read_text(encoding="utf-8")
TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")

PPL_CONFIG: dict[str, Any] = {
    "name": "lingbot_video_moe_30b",
    "model_root": TF_MODEL_ZOO_PATH + "/lingbot/lingbot-video-moe-30b-a3b",
    "variant": "moe",
    "supports_refiner": True,
    "prompt": _DEFAULT_PROMPT,
    "num_inference_steps": 40,
    "guidance_scale": 3.0,
    "fps": 24,
    "target_video_length": 5,
    "resolution": "480p",
    "aspect_ratio": "16:9",
    "seed": 42,
    "expert_backend": "auto",
    "enable_refiner": True,
    "refiner_height": 1088,
    "refiner_width": 1920,
    "refiner_steps": 8,
    "refiner_guidance_scale": 3.0,
    "refiner_shift": 3.0,
    "refiner_t_thresh": 0.85,
    "refiner_tail_steps": 2,
    "refiner_parallelism": 0,
    "refiner_batch_cfg": False,
    "refiner_null_cond_clone_zero": True,
    "cfg_parallel_degree": 1,
    "refiner_cfg_parallel_degree": 1,
    "refiner_co_resident": True,
}


def _build_contract() -> dict[str, Any]:
    """Build the standard service contract with LingBot-specific refiner defaults."""
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
            overrides["refine"] = {"type": "boolean", "default": PPL_CONFIG["enable_refiner"]}
        task_contracts[task] = build_task_contract_template(task, parameter_overrides=overrides)
    return build_pipeline_manifest(
        pipeline_name=PPL_CONFIG["name"],
        supported_tasks=task_contracts,
        task_contracts=task_contracts,
    )


PIPELINE_CONTRACT = _build_contract()


def build_pipeline(
    model_root: str | Path,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = True,
    guidance_scale: float = PPL_CONFIG["guidance_scale"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    shift: float = 3.0,
    attention_config: AttentionConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    batch_cfg: bool | None = None,
    expert_backend: str = PPL_CONFIG["expert_backend"],
) -> LingBotVideoPipeline:
    """Load MoE base modules and initialize the complete pipeline."""
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
    runtime_device = torch.device(device)
    root = Path(model_root)
    transformer_dir = root / "transformer"
    offload_config = OffloadConfig(
        offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD if cpu_offload else WeightOffloadType.NO_CPU_OFFLOAD
    )
    runtime_config = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch_dtype,
        attention_config=attention_config or AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
        offload_config=offload_config,
        parallel_config=parallel_config,
    )

    if expert_backend == "auto":
        expert_backend = "grouped_mm" if parallel_config.world_size > 1 else "sorted"
    if expert_backend not in {"fp8", "grouped_mm", "sorted"}:
        raise ValueError("LingBot MoE expert_backend must be 'auto', 'fp8', 'grouped_mm', or 'sorted'")
    if expert_backend == "grouped_mm" and not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("LingBot grouped_mm requires a recent CUDA-enabled PyTorch build")
    module_manager = ModuleManager(torch_dtype=torch_dtype, device="cpu")
    module_manager.load_model(str(transformer_dir), name="transformer", torch_dtype=torch_dtype)
    transformer = module_manager.fetch_module("transformer")
    if transformer is None:
        raise RuntimeError(f"Unable to load LingBot-Video transformer from {transformer_dir}")
    transformer.promote_stability_layers_to_fp32()
    if expert_backend == "fp8":
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("LingBot FP8 experts require torch._scaled_mm")
        transformer.quantize_experts_fp8_()
    transformer.set_expert_execution_backend(expert_backend)

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
            model=load_lingbot_video_model_config(transformer_dir, variant="moe"),
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


def build_refiner(
    model_root: str | Path,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = True,
    attention_config: AttentionConfig | None = None,
    parallel_config: ParallelConfig | None = None,
    batch_cfg: bool | None = None,
    expert_backend: str = PPL_CONFIG["expert_backend"],
) -> LingBotVideoRefinerStage:
    """Load refiner modules through ModuleManager and assemble its stages."""
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise RuntimeError("LingBot-Video refiner requires diffusers") from exc

    parallel_config = parallel_config or ParallelConfig()
    if batch_cfg is None:
        batch_cfg = False
    if batch_cfg and parallel_config.cfg_degree > 1:
        raise ValueError("LingBot CFG parallel and batch CFG are mutually exclusive")
    if parallel_config.enable_fsdp and cpu_offload:
        raise ValueError("LingBot refiner FSDP inference requires cpu_offload=False")
    runtime_device = torch.device(device)
    root = Path(model_root)
    offload_config = OffloadConfig(
        offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD if cpu_offload else WeightOffloadType.NO_CPU_OFFLOAD
    )
    runtime_config = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch_dtype,
        attention_config=attention_config or AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
        offload_config=offload_config,
        parallel_config=parallel_config,
    )

    if expert_backend == "auto":
        expert_backend = "grouped_mm" if parallel_config.world_size > 1 else "sorted"
    if expert_backend not in {"fp8", "grouped_mm", "sorted"}:
        raise ValueError("LingBot refiner expert_backend must be 'auto', 'fp8', 'grouped_mm', or 'sorted'")
    module_manager = ModuleManager(torch_dtype=torch_dtype, device="cpu")
    module_manager.load_model(str(root / "refiner"), name="transformer", torch_dtype=torch_dtype)
    transformer = module_manager.fetch_module("transformer")
    if transformer is None:
        raise RuntimeError(f"Unable to load LingBot-Video refiner from {root / 'refiner'}")
    transformer.promote_stability_layers_to_fp32()
    if expert_backend == "fp8":
        if not hasattr(torch, "_scaled_mm"):
            raise RuntimeError("LingBot FP8 experts require torch._scaled_mm")
        transformer.quantize_experts_fp8_()
    transformer.set_expert_execution_backend(expert_backend)

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
    scheduler = module_manager.fetch_module("scheduler")
    denoising_stage = LingBotVideoDenoisingStage("refiner", module_manager, runtime_config, batch_cfg=batch_cfg)
    refiner = LingBotVideoRefinerStage(
        denoising_stage=denoising_stage,
        vae_encode_stage=LingBotVideoVAEEncodeStage("refiner_vae_encode", module_manager, runtime_config),
        vae_decode_stage=LingBotVideoVAEDecodeStage("refiner_vae_decode", module_manager, runtime_config),
        scheduler=scheduler,
    )
    if parallel_config.world_size > 1 and not torch.distributed.is_initialized():
        refiner.denoising_stage = ParallelWorker(denoising_stage)
    else:
        denoising_stage.parallel_models()
    return refiner


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    refiner_parallelism: int | None = None,
    refiner_batch_cfg: bool | None = None,
    cfg_parallel_degree: int | None = None,
    refiner_cfg_parallel_degree: int | None = None,
    refiner_co_resident: bool | None = None,
    expert_backend: str = PPL_CONFIG["expert_backend"],
) -> object:
    """Load the fixed MoE 30B checkpoint and configure its separate refiner."""
    if parallelism not in {1, 4}:
        raise ValueError("LingBot-Video MoE supports parallelism=1 or 4")
    cfg_parallel_degree = int(PPL_CONFIG["cfg_parallel_degree"]) if cfg_parallel_degree is None else cfg_parallel_degree
    if cfg_parallel_degree not in {1, 2} or parallelism % cfg_parallel_degree:
        raise ValueError("LingBot CFG parallel degree must be 1 or 2 and divide parallelism")
    pipeline = build_pipeline(
        model_root,
        cpu_offload=parallelism == 1,
        parallel_config=ParallelConfig(
            device_ids=list(range(parallelism)),
            cfg_degree=cfg_parallel_degree,
            sp_ulysses_degree=parallelism // cfg_parallel_degree,
            enable_fsdp=parallelism > 1,
        ),
        expert_backend=expert_backend,
    )
    refiner_parallelism = (
        int(PPL_CONFIG["refiner_parallelism"]) if refiner_parallelism is None else refiner_parallelism
    ) or parallelism
    if refiner_parallelism not in {1, 4}:
        raise ValueError("LingBot-Video refiner supports parallelism=1 or 4")
    refiner_cfg_parallel_degree = (
        int(PPL_CONFIG["refiner_cfg_parallel_degree"])
        if refiner_cfg_parallel_degree is None
        else refiner_cfg_parallel_degree
    )
    if refiner_cfg_parallel_degree not in {1, 2} or refiner_parallelism % refiner_cfg_parallel_degree:
        raise ValueError("LingBot refiner CFG parallel degree must be 1 or 2 and divide refiner parallelism")
    pipeline.refiner_parallel_config = ParallelConfig(
        device_ids=list(range(refiner_parallelism)),
        cfg_degree=refiner_cfg_parallel_degree,
        sp_ulysses_degree=refiner_parallelism // refiner_cfg_parallel_degree,
        enable_fsdp=refiner_parallelism > 1,
    )
    pipeline.refiner_cpu_offload = refiner_parallelism == 1
    pipeline.refiner_batch_cfg = (
        bool(PPL_CONFIG["refiner_batch_cfg"]) if refiner_batch_cfg is None else refiner_batch_cfg
    )
    if pipeline.refiner_batch_cfg and refiner_cfg_parallel_degree > 1:
        raise ValueError("LingBot refiner CFG parallel and batch CFG are mutually exclusive")
    pipeline.refiner_co_resident = (
        bool(PPL_CONFIG["refiner_co_resident"]) if refiner_co_resident is None else refiner_co_resident
    ) and parallelism > 1
    pipeline.refiner_expert_backend = expert_backend
    return pipeline


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
    refine: bool | None = None,
    model_root: str = PPL_CONFIG["model_root"],
) -> torch.Tensor:
    """Run MoE T2I, T2V, or I2V with an optional refiner stage."""
    if task not in {"t2i", "t2v", "i2v"}:
        raise ValueError(f"unsupported LingBot-Video task: {task}")
    if task == "i2v" and not first_image_path:
        raise ValueError("LingBot-Video i2v requires first_image_path")
    if task == "t2i" and refine is True:
        raise ValueError("LingBot-Video refiner does not support t2i")
    if refine is None:
        refine = task != "t2i" and bool(PPL_CONFIG["enable_refiner"])

    caption, prompt_duration = parse_lingbot_video_prompt(json.loads(prompt))
    width, height = _resolve_video_size(aspect_ratio, resolution)
    duration = target_video_length if target_video_length is not None else prompt_duration
    if duration is None:
        duration = float(PPL_CONFIG["target_video_length"])
    num_frames = 1 if task == "t2i" else num_frames_from_duration(duration, fps=int(PPL_CONFIG["fps"]))
    resolved_negative_prompt = negative_prompt if negative_prompt is not None else default_negative_caption(num_frames)
    generation = pipeline.generate(
        LingBotVideoRequest(
            caption=caption,
            height=height,
            width=width,
            num_frames=num_frames,
            image=_load_condition_image(first_image_path) if first_image_path else None,
        ),
        negative_caption=resolved_negative_prompt,
        generator=_generator(pipeline, seed),
    )
    frames = generation.output
    if not refine:
        return frames
    if pipeline.text_stage is None:
        raise RuntimeError("LingBot-Video refiner requires the base text stage")

    if generation.prompt_conditions.has_visual_condition:
        positive, positive_mask = pipeline.text_stage.encode(caption)
    else:
        positive = generation.prompt_conditions.positive_prompt_embeds
        positive_mask = generation.prompt_conditions.positive_attention_mask
    if PPL_CONFIG["refiner_null_cond_clone_zero"]:
        negative = torch.zeros_like(positive)
        negative_mask = positive_mask.clone()
    elif (
        generation.prompt_conditions.has_visual_condition or generation.prompt_conditions.negative_prompt_embeds is None
    ):
        negative, negative_mask = pipeline.text_stage.encode(resolved_negative_prompt)
    else:
        negative = generation.prompt_conditions.negative_prompt_embeds
        negative_mask = generation.prompt_conditions.negative_attention_mask

    if not getattr(pipeline, "refiner_co_resident", False):
        pipeline.release_gpu_resources()
    refiner = build_refiner(
        model_root,
        cpu_offload=getattr(pipeline, "refiner_cpu_offload", True),
        parallel_config=getattr(pipeline, "refiner_parallel_config", None),
        batch_cfg=getattr(pipeline, "refiner_batch_cfg", False),
        expert_backend=getattr(pipeline, "refiner_expert_backend", PPL_CONFIG["expert_backend"]),
    )
    try:
        lowres_video, _ = prepare_refiner_video(
            frames,
            source_fps=float(PPL_CONFIG["fps"]),
            height=int(PPL_CONFIG["refiner_height"]),
            width=int(PPL_CONFIG["refiner_width"]),
        )
        clean_first_frame = (
            load_refiner_first_frame(
                first_image_path,
                target_height=int(PPL_CONFIG["refiner_height"]),
                target_width=int(PPL_CONFIG["refiner_width"]),
                geometry_height=height,
                geometry_width=width,
            )
            if first_image_path
            else None
        )
        return refiner.refine(
            lowres_video,
            positive,
            negative,
            positive_mask,
            negative_mask,
            num_inference_steps=int(PPL_CONFIG["refiner_steps"]),
            guidance_scale=float(PPL_CONFIG["refiner_guidance_scale"]),
            shift=float(PPL_CONFIG["refiner_shift"]),
            t_thresh=float(PPL_CONFIG["refiner_t_thresh"]),
            tail_steps=int(PPL_CONFIG["refiner_tail_steps"]),
            clean_first_frame=clean_first_frame,
            generator=_generator(pipeline, seed),
            before_decode=(pipeline.release_gpu_resources if getattr(pipeline, "refiner_co_resident", False) else None),
        )
    finally:
        refiner.close()


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
    refine: bool | None = None,
    model_root: str = PPL_CONFIG["model_root"],
    **_: object,
) -> dict[str, str]:
    """Run the MoE example and save its base or refined output."""
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
        refine,
        model_root=model_root,
    )
    return _save_output(frames, output_path)


@click.command()
@click.option("--gpu_num", default=1, type=int, help="Number of GPUs: 1 or 4")
@click.option("--cfg_parallel_degree", default=PPL_CONFIG["cfg_parallel_degree"], type=click.Choice([1, 2]))
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="MoE 30B checkpoint root")
@click.option("--prompt", default=PPL_CONFIG["prompt"], help="Structured JSON caption")
@click.option("--negative_prompt", default=None, help="Optional structured negative caption")
@click.option("--seed", default=PPL_CONFIG["seed"], type=int)
@click.option(
    "--expert_backend",
    default=PPL_CONFIG["expert_backend"],
    type=click.Choice(["auto", "fp8", "grouped_mm", "sorted"]),
    help="MoE expert backend; fp8 uses native dynamic W8A8 scaled GEMMs",
)
@click.option("--resolution", default=PPL_CONFIG["resolution"])
@click.option("--aspect_ratio", default=PPL_CONFIG["aspect_ratio"])
@click.option("--target_video_length", default=None, type=float)
@click.option("--task", default="t2v", type=click.Choice(["t2i", "t2v", "i2v"]))
@click.option("--first_image_path", default="", help="Required for i2v")
@click.option(
    "--refiner_gpu_num", default=PPL_CONFIG["refiner_parallelism"], type=int, help="Refiner GPUs; 0 inherits gpu_num"
)
@click.option("--refiner_batch_cfg/--no-refiner_batch_cfg", default=PPL_CONFIG["refiner_batch_cfg"])
@click.option(
    "--refiner_cfg_parallel_degree",
    default=PPL_CONFIG["refiner_cfg_parallel_degree"],
    type=click.Choice([1, 2]),
)
@click.option("--refiner_co_resident/--no-refiner_co_resident", default=PPL_CONFIG["refiner_co_resident"])
@click.option("--refine/--no-refine", default=None)
@click.option("--output_path", default="output.mp4")
def main(
    gpu_num: int,
    cfg_parallel_degree: int,
    model_root: str,
    prompt: str,
    negative_prompt: str | None,
    seed: int,
    expert_backend: str,
    resolution: str,
    aspect_ratio: str,
    target_video_length: float | None,
    task: str,
    first_image_path: str,
    refiner_gpu_num: int,
    refiner_batch_cfg: bool,
    refiner_cfg_parallel_degree: int,
    refiner_co_resident: bool,
    refine: bool | None,
    output_path: str,
) -> None:
    """Generate with the LingBot-Video MoE 30B checkpoint and refiner."""
    pipeline = get_pipeline(
        gpu_num,
        model_root,
        refiner_parallelism=refiner_gpu_num,
        refiner_batch_cfg=refiner_batch_cfg,
        cfg_parallel_degree=cfg_parallel_degree,
        refiner_cfg_parallel_degree=refiner_cfg_parallel_degree,
        refiner_co_resident=refiner_co_resident,
        expert_backend=expert_backend,
    )
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
            refine,
            model_root=model_root,
        )
        click.echo(f"Output saved to {result['output_path']}")
    finally:
        stop = getattr(pipeline, "stop", None)
        if callable(stop):
            stop()


if __name__ == "__main__":
    main()
