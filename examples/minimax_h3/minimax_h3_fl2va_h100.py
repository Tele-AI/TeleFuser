# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from examples.minimax_h3.common import (
    MINIMAX_H3_DEFAULT_FL2VA_IMAGE,
    load_minimax_h3_pipeline,
    save_generation,
)
from telefuser.core.config import AttnImplType, FeatureCacheConfig
from telefuser.pipelines.minimax_h3.pipeline import MiniMaxH3Generation, MiniMaxH3Pipeline
from telefuser.pipelines.minimax_h3.task_profiles import MINIMAX_H3_FINITE_ASPECT_RATIOS
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG: dict[str, Any] = {
    "name": "minimax_h3_fl2va_h100",
    "model_root": TF_MODEL_ZOO_PATH + "/MiniMaxAI_MiniMax-H3",
    "partition": "FL2VA",
    "prompt": "Steam rises from the ramen while the family talks in the background.",
    "num_inference_steps": 50,
    "short_edge": 768,
    "resolution": "768p",
    "aspect_ratio": "16:9",
    "cli_aspect_ratio": "auto",
    "target_video_length": 8,
    "seed": 0,
    "flow_shift": None,
    "audio_flow_shift": None,
    "device": "cuda:0",
    "enable_fsdp": None,
    "online_adaln_cache": True,
    "attn_impl": AttnImplType.FLASH_ATTN_4,
    "feature_cache_model_type": "MiniMax-H3-Base",
    "feature_cache_n_derivatives": 1,
    "feature_cache_taylor_threshold": 2,
    "quantization": None,
}


def _task_contract(task: str) -> dict[str, Any]:
    return build_task_contract_template(
        task,
        parameter_overrides={
            "prompt": {
                "default": PPL_CONFIG["prompt"],
                "description": "Positive prompt for MiniMax H3 audio-video generation.",
            },
            "seed": {"default": PPL_CONFIG["seed"]},
            "resolution": {
                "default": PPL_CONFIG["resolution"],
                "enum": [PPL_CONFIG["resolution"]],
                "description": "MiniMax H3 local checkpoints support 768p-class output.",
            },
            "aspect_ratio": {
                "default": PPL_CONFIG["aspect_ratio"],
                "enum": list(MINIMAX_H3_FINITE_ASPECT_RATIOS),
            },
            "target_video_length": {
                "default": PPL_CONFIG["target_video_length"],
                "description": "Output duration in seconds; MiniMax H3 supports values from 4 through 15.",
            },
        },
        excluded_parameters=("negative_prompt",),
    )


PIPELINE_MANIFEST = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=("t2v", "i2v", "fl2v"),
    task_contracts={task: _task_contract(task) for task in ("t2v", "i2v", "fl2v")},
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    device: str = PPL_CONFIG["device"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    enable_fsdp: bool | None = PPL_CONFIG["enable_fsdp"],
    online_adaln_cache: bool = PPL_CONFIG["online_adaln_cache"],
    attn_impl: AttnImplType | str = PPL_CONFIG["attn_impl"],
    enable_feature_cache: bool = False,
    feature_cache_model_type: str = PPL_CONFIG["feature_cache_model_type"],
    feature_cache_n_derivatives: int = PPL_CONFIG["feature_cache_n_derivatives"],
    feature_cache_taylor_threshold: int = PPL_CONFIG["feature_cache_taylor_threshold"],
    quantization: str | None = PPL_CONFIG["quantization"],
) -> MiniMaxH3Pipeline:
    """Load the FL2VA checkpoint partition for one, two, or four GPUs."""
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
        online_adaln_cache=online_adaln_cache,
        attn_impl=attn_impl,
        feature_cache_config=FeatureCacheConfig(
            enabled=enable_feature_cache,
            model_type=feature_cache_model_type,
            n_derivatives=feature_cache_n_derivatives,
            taylor_threshold=feature_cache_taylor_threshold,
        ),
        quantization=quantization,
    )


def build_fl2va_conditions(
    *,
    mode: str | None,
    image: str | None,
    last_image: str | None,
) -> list[dict[str, object]]:
    """Build the three supported FL2VA keyframe signatures, preserving legacy inference."""
    if mode is None:
        if image and last_image:
            mode = "first-last"
        elif image:
            mode = "first-frame"
        elif last_image:
            mode = "last-frame"
        else:
            mode = "t2va"
    if mode == "t2va":
        if image or last_image:
            raise ValueError("--mode t2va does not accept --image or --last-image")
        return []

    default_image = str(MINIMAX_H3_DEFAULT_FL2VA_IMAGE)
    if mode == "first-frame":
        return [{"type": "image", "role": "keyframe", "uri": image or default_image, "frame_index": 0}]
    if mode == "last-frame":
        return [
            {
                "type": "image",
                "role": "keyframe",
                "uri": last_image or image or default_image,
                "frame_index": -1,
            }
        ]
    if mode == "first-last":
        first = image or default_image
        last = last_image or image or default_image
        return [
            {"type": "image", "role": "keyframe", "uri": first, "frame_index": 0},
            {"type": "image", "role": "keyframe", "uri": last, "frame_index": -1},
        ]
    raise ValueError(f"unsupported FL2VA mode {mode!r}")


def _mode_for_service_task(task: str) -> str:
    modes = {"t2v": "t2va", "i2v": "first-frame", "fl2v": "first-last"}
    try:
        return modes[task]
    except KeyError as exc:
        raise ValueError(f"unsupported MiniMax H3 FL2VA service task: {task}") from exc


def run(
    pipeline: MiniMaxH3Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    target_video_length: float = PPL_CONFIG["target_video_length"],
    task: str = "t2v",
    first_image_path: str = "",
    last_image_path: str = "",
    *,
    mode: str | None = None,
    flow_shift: float | None = PPL_CONFIG["flow_shift"],
    audio_flow_shift: float | None = PPL_CONFIG["audio_flow_shift"],
) -> MiniMaxH3Generation:
    """Run T2VA or a supported FL2VA keyframe signature."""
    if resolution != PPL_CONFIG["resolution"]:
        expected = PPL_CONFIG["resolution"]
        raise ValueError(f"MiniMax H3 only supports resolution={expected!r}")
    resolved_mode = mode or _mode_for_service_task(task)
    if mode is None and task == "i2v" and not first_image_path:
        raise ValueError("MiniMax H3 i2v requires first_image_path")
    if mode is None and task == "fl2v" and (not first_image_path or not last_image_path):
        raise ValueError("MiniMax H3 fl2v requires first_image_path and last_image_path")
    conditions = build_fl2va_conditions(
        mode=resolved_mode,
        image=first_image_path or None,
        last_image=last_image_path or None,
    )
    return pipeline(
        task="fl2va" if conditions else "t2va",
        prompt=prompt,
        conditions=conditions,
        target={
            "short_edge": PPL_CONFIG["short_edge"],
            "aspect_ratio": aspect_ratio,
            "duration_seconds": target_video_length,
        },
        seed=seed,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )


def run_with_file(
    pipeline: MiniMaxH3Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "minimax_h3_fl2va.mp4",
    resolution: str = PPL_CONFIG["resolution"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    target_video_length: float = PPL_CONFIG["target_video_length"],
    task: str = "t2v",
    first_image_path: str = "",
    last_image_path: str = "",
    mode: str | None = None,
    flow_shift: float | None = PPL_CONFIG["flow_shift"],
    audio_flow_shift: float | None = PPL_CONFIG["audio_flow_shift"],
    **_: object,
) -> dict[str, str]:
    """Run MiniMax H3 FL2VA and save its synchronized audio-video output."""
    result = run(
        pipeline,
        prompt,
        seed,
        resolution,
        aspect_ratio,
        target_video_length,
        task,
        first_image_path,
        last_image_path,
        mode=mode,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )
    save_generation(result, output_path)
    return {"output_path": str(Path(output_path))}


def _main(default_quantization: str | None = PPL_CONFIG["quantization"]) -> None:
    parser = argparse.ArgumentParser(description="Generate MiniMax H3 T2VA/FL2VA audio-video on H100 GPUs")
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--mode", choices=("t2va", "first-frame", "last-frame", "first-last"))
    parser.add_argument("--image", dest="first_image_path", default="")
    parser.add_argument("--last-image", dest="last_image_path", default="")
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument(
        "--target-video-length",
        "--duration",
        dest="target_video_length",
        type=float,
        default=PPL_CONFIG["target_video_length"],
    )
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--steps", type=int, default=PPL_CONFIG["num_inference_steps"])
    parser.add_argument(
        "--aspect-ratio",
        choices=("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        default=PPL_CONFIG["cli_aspect_ratio"],
    )
    parser.add_argument("--flow-shift", type=float, default=PPL_CONFIG["flow_shift"])
    parser.add_argument("--audio-flow-shift", type=float, default=PPL_CONFIG["audio_flow_shift"])
    parser.add_argument("--device", default=PPL_CONFIG["device"])
    parser.add_argument(
        "--quantization",
        choices=("torchao-fp8", "tf-kernel-fp8", "bnb-nf4"),
        default=default_quantization,
        help="Online DiT Linear quantization backend (single GPU only).",
    )
    parser.add_argument("--gpu-num", "--ulysses-degree", dest="gpu_num", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument(
        "--attn-impl",
        choices=("FLASH_ATTN_4", "SAGE_ATTN_2_8_8_SM90"),
        default=PPL_CONFIG["attn_impl"].name,
    )
    parser.add_argument("--enable-feature-cache", action="store_true")
    parser.add_argument("--feature-cache-model-type", default=PPL_CONFIG["feature_cache_model_type"])
    parser.add_argument(
        "--feature-cache-n-derivatives",
        type=int,
        choices=(0, 1, 2),
        default=PPL_CONFIG["feature_cache_n_derivatives"],
    )
    parser.add_argument(
        "--feature-cache-taylor-threshold",
        type=int,
        default=PPL_CONFIG["feature_cache_taylor_threshold"],
    )
    fsdp_group = parser.add_mutually_exclusive_group()
    fsdp_group.add_argument("--enable-fsdp", dest="enable_fsdp", action="store_true")
    fsdp_group.add_argument("--disable-fsdp", dest="enable_fsdp", action="store_false")
    parser.set_defaults(enable_fsdp=PPL_CONFIG["enable_fsdp"])
    parser.add_argument("--output-path", "--output", dest="output_path", default="minimax_h3_fl2va.mp4")
    args = parser.parse_args()

    resolved_mode = args.mode
    if resolved_mode is None:
        if args.first_image_path and args.last_image_path:
            resolved_mode = "first-last"
        elif args.first_image_path:
            resolved_mode = "first-frame"
        elif args.last_image_path:
            resolved_mode = "last-frame"
        else:
            resolved_mode = "t2va"
    try:
        build_fl2va_conditions(
            mode=resolved_mode, image=args.first_image_path or None, last_image=args.last_image_path or None
        )
    except ValueError as exc:
        parser.error(str(exc))

    pipeline = get_pipeline(
        args.gpu_num,
        args.model_root,
        device=args.device,
        num_inference_steps=args.steps,
        enable_fsdp=args.enable_fsdp,
        attn_impl=args.attn_impl,
        enable_feature_cache=args.enable_feature_cache,
        feature_cache_model_type=args.feature_cache_model_type,
        feature_cache_n_derivatives=args.feature_cache_n_derivatives,
        feature_cache_taylor_threshold=args.feature_cache_taylor_threshold,
        quantization=args.quantization,
    )
    try:
        result = run_with_file(
            pipeline,
            prompt=args.prompt,
            seed=args.seed,
            output_path=args.output_path,
            aspect_ratio=args.aspect_ratio,
            target_video_length=args.target_video_length,
            first_image_path=args.first_image_path,
            last_image_path=args.last_image_path,
            mode=resolved_mode,
            flow_shift=args.flow_shift,
            audio_flow_shift=args.audio_flow_shift,
        )
        print("Output saved to {}".format(result["output_path"]))
    finally:
        pipeline.stop()


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
