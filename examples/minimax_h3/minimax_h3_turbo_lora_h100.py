# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 FL2VA with the released Turbo LoRA on one, two, or four GPUs."""

from __future__ import annotations

import argparse
import os
from typing import Any

try:
    from examples.minimax_h3.common import MINIMAX_H3_DEFAULT_FL2VA_IMAGE, load_minimax_h3_pipeline, save_generation
except ModuleNotFoundError:
    from common import MINIMAX_H3_DEFAULT_FL2VA_IMAGE, load_minimax_h3_pipeline, save_generation
from telefuser.core.config import AttnImplType
from telefuser.pipelines.minimax_h3.pipeline import MiniMaxH3Generation, MiniMaxH3Pipeline
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG: dict[str, Any] = {
    "name": "minimax_h3_turbo_lora_h100",
    "model_root": TF_MODEL_ZOO_PATH + "/MiniMaxAI_MiniMax-H3",
    "partition": "FL2VA",
    "lora_path": "lightx2v/Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
    "lora_strength": 1.0,
    "num_inference_steps": 9,
    "prompt": "Steam rises from the ramen while the family talks in the background.",
    "input_image_path": str(MINIMAX_H3_DEFAULT_FL2VA_IMAGE),
    "target_video_length": 8,
    "seed": 0,
    "flow_shift": 6.0,
    "audio_flow_shift": 3.0,
    "resolution": "768p",
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "device": "cuda:0",
    "enable_fsdp": False,
    "attn_impl": AttnImplType.FLASH_ATTN_4,
}

PIPELINE_MANIFEST = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=("i2v",),
    task_contracts={
        "i2v": build_task_contract_template(
            "i2v",
            parameter_overrides={
                "prompt": {"default": PPL_CONFIG["prompt"]},
                "resolution": {"default": PPL_CONFIG["resolution"], "enum": [PPL_CONFIG["resolution"]]},
                "target_video_length": {"default": PPL_CONFIG["target_video_length"]},
            },
            excluded_parameters=("negative_prompt",),
        )
    },
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    device: str = PPL_CONFIG["device"],
    lora_path: str = PPL_CONFIG["lora_path"],
    lora_strength: float = PPL_CONFIG["lora_strength"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    enable_fsdp: bool | None = PPL_CONFIG["enable_fsdp"],
    attn_impl: AttnImplType | str = PPL_CONFIG["attn_impl"],
) -> MiniMaxH3Pipeline:
    """Load FL2VA and merge Turbo LoRA with the requested GPU parallelism."""
    if parallelism not in {1, 2, 4}:
        raise ValueError("parallelism must be 1, 2, or 4")
    tp_degree = 2 if parallelism == 4 else 1
    return load_minimax_h3_pipeline(
        model_root,
        partition=PPL_CONFIG["partition"],
        device=device,
        num_inference_steps=num_inference_steps,
        ulysses_degree=parallelism // tp_degree,
        tp_degree=tp_degree,
        text_encoder_tp_degree=parallelism,
        enable_fsdp=enable_fsdp,
        attn_impl=attn_impl,
        lora_path=lora_path,
        lora_strength=lora_strength,
    )


def run(
    pipeline: MiniMaxH3Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "minimax_h3_turbo_lora.mp4",
    target_video_length: float = PPL_CONFIG["target_video_length"],
    input_image_path: str = PPL_CONFIG["input_image_path"],
) -> MiniMaxH3Generation:
    """Generate an image-conditioned Turbo H3 clip."""
    result = pipeline(
        task="fl2va",
        prompt=prompt,
        conditions=[{"type": "image", "role": "keyframe", "uri": input_image_path, "frame_index": 0}],
        target={
            "short_edge": PPL_CONFIG["short_edge"],
            "aspect_ratio": PPL_CONFIG["aspect_ratio"],
            "duration_seconds": target_video_length,
        },
        seed=seed,
        flow_shift=PPL_CONFIG["flow_shift"],
        audio_flow_shift=PPL_CONFIG["audio_flow_shift"],
    )
    save_generation(result, output_path)
    return result


def run_with_file(
    pipeline: MiniMaxH3Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "minimax_h3_turbo_lora.mp4",
    target_video_length: float = PPL_CONFIG["target_video_length"],
    input_image_path: str = PPL_CONFIG["input_image_path"],
    first_image_path: str | None = None,
    **_: object,
) -> dict[str, str]:
    """Service-compatible wrapper that returns the generated output path."""
    run(
        pipeline,
        prompt=prompt,
        seed=seed,
        output_path=output_path,
        target_video_length=target_video_length,
        input_image_path=first_image_path or input_image_path,
    )
    return {"output_path": output_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MiniMax H3 Turbo LoRA FL2VA on H100 GPUs")
    parser.add_argument("--gpu-num", "--ulysses-degree", dest="gpu_num", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--lora-path", default=PPL_CONFIG["lora_path"])
    parser.add_argument("--image", dest="input_image_path", default=PPL_CONFIG["input_image_path"])
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument("--duration", type=float, default=PPL_CONFIG["target_video_length"])
    parser.add_argument("--steps", type=int, default=PPL_CONFIG["num_inference_steps"])
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--device", default=PPL_CONFIG["device"])
    parser.add_argument(
        "--attn-impl",
        choices=("FLASH_ATTN_4", "SAGE_ATTN_2_8_8_SM90"),
        default=PPL_CONFIG["attn_impl"].name,
    )
    fsdp_group = parser.add_mutually_exclusive_group()
    fsdp_group.add_argument("--enable-fsdp", dest="enable_fsdp", action="store_true")
    fsdp_group.add_argument("--disable-fsdp", dest="enable_fsdp", action="store_false")
    parser.set_defaults(enable_fsdp=PPL_CONFIG["enable_fsdp"])
    parser.add_argument("--output", default="minimax_h3_turbo_lora.mp4")
    args = parser.parse_args()
    pipeline = get_pipeline(
        args.gpu_num,
        args.model_root,
        device=args.device,
        lora_path=args.lora_path,
        num_inference_steps=args.steps,
        enable_fsdp=args.enable_fsdp,
        attn_impl=args.attn_impl,
    )
    try:
        run(
            pipeline,
            prompt=args.prompt,
            seed=args.seed,
            output_path=args.output,
            target_video_length=args.duration,
            input_image_path=args.input_image_path,
        )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
