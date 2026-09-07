# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from examples.minimax_h3.common import (
    MINIMAX_H3_DEFAULT_REF2VA_AUDIO,
    MINIMAX_H3_DEFAULT_REF2VA_VIDEO,
    load_minimax_h3_pipeline,
    load_minimax_h3_request,
    run_minimax_h3_request,
    save_generation,
)
from telefuser.pipelines.minimax_h3.pipeline import MiniMaxH3Generation, MiniMaxH3Pipeline
from telefuser.pipelines.minimax_h3.task_profiles import MINIMAX_H3_FINITE_ASPECT_RATIOS
from telefuser.service.core.contract_templates import build_pipeline_manifest, build_task_contract_template

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "/hhb-data/aigc/model_zoo")
PPL_CONFIG: dict[str, Any] = {
    "name": "minimax_h3_ref2va_h100",
    "model_root": TF_MODEL_ZOO_PATH + "/MiniMaxAI_MiniMax-H3",
    "partition": "Ref2VA",
    "prompt": "Preserve the source identity and motion, and use the reference voice for the dialogue.",
    "num_inference_steps": 50,
    "short_edge": 768,
    "resolution": "768p",
    "aspect_ratio": "16:9",
    "cli_aspect_ratio": "auto",
    "target_video_length": 5,
    "seed": 0,
    "flow_shift": None,
    "audio_flow_shift": None,
    "device": "cuda:0",
    "enable_fsdp": None,
    "online_adaln_cache": True,
    "attention_chunks": 2,
    "ulysses_sequence_mode": "valid_only",
    "quantization": None,
}

PIPELINE_MANIFEST = build_pipeline_manifest(
    pipeline_name=PPL_CONFIG["name"],
    supported_tasks=("s2v",),
    task_contracts={
        "s2v": build_task_contract_template(
            "s2v",
            media_type="video",
            excluded_parameters=("negative_prompt",),
            parameter_overrides={
                "prompt": {
                    "type": "string",
                    "required": True,
                    "default": PPL_CONFIG["prompt"],
                    "description": "Positive prompt referencing ordered MiniMax H3 materials.",
                },
                "conditions": {
                    "type": "array",
                    "required": True,
                    "description": "Ordered image, video, video_audio, and audio reference materials.",
                },
                "seed": {"type": "integer", "default": PPL_CONFIG["seed"]},
                "resolution": {
                    "type": "string",
                    "default": PPL_CONFIG["resolution"],
                    "enum": [PPL_CONFIG["resolution"]],
                },
                "aspect_ratio": {
                    "type": "string",
                    "default": PPL_CONFIG["aspect_ratio"],
                    "enum": list(MINIMAX_H3_FINITE_ASPECT_RATIOS),
                },
                "target_video_length": {
                    "type": "integer",
                    "default": PPL_CONFIG["target_video_length"],
                    "description": "Output duration in seconds; MiniMax H3 supports values from 4 through 15.",
                },
                "flow_shift": {"type": "number", "default": PPL_CONFIG["flow_shift"]},
                "audio_flow_shift": {"type": "number", "default": PPL_CONFIG["audio_flow_shift"]},
                "output_path": {"type": "string", "default": ""},
            },
        )
    },
)


def get_pipeline(
    parallelism: int = 1,
    model_root: str = PPL_CONFIG["model_root"],
    *,
    device: str = PPL_CONFIG["device"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
    enable_fsdp: bool | None = PPL_CONFIG["enable_fsdp"],
    online_adaln_cache: bool = PPL_CONFIG["online_adaln_cache"],
    attention_chunks: int = PPL_CONFIG["attention_chunks"],
    ulysses_sequence_mode: str = PPL_CONFIG["ulysses_sequence_mode"],
    quantization: str | None = PPL_CONFIG["quantization"],
) -> MiniMaxH3Pipeline:
    """Load the Ref2VA checkpoint partition for one, two, or four GPUs."""
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
        attention_chunks=attention_chunks,
        ulysses_sequence_mode=ulysses_sequence_mode,
        quantization=quantization,
    )


def default_ref2va_conditions() -> list[dict[str, str]]:
    """Return the frozen video-then-audio reference order used by the example."""
    return [
        {"type": "video", "role": "reference", "uri": str(MINIMAX_H3_DEFAULT_REF2VA_VIDEO)},
        {"type": "audio", "role": "reference", "uri": str(MINIMAX_H3_DEFAULT_REF2VA_AUDIO)},
    ]


_REF2VA_REFERENCE_TYPES = frozenset({"image", "video", "video_audio", "audio"})


def parse_ref2va_ordered_materials(materials: list[str]) -> list[dict[str, str]]:
    """Parse repeated TYPE=URI CLI values without changing their order."""
    conditions: list[dict[str, str]] = []
    for index, material in enumerate(materials):
        condition_type, separator, uri = material.partition("=")
        if not separator or not condition_type or not uri:
            raise ValueError(f"material[{index}] must use TYPE=URI syntax.")
        if condition_type not in _REF2VA_REFERENCE_TYPES:
            allowed = ", ".join(sorted(_REF2VA_REFERENCE_TYPES))
            raise ValueError(f"material[{index}] has unsupported type {condition_type!r}; expected one of {allowed}.")
        conditions.append({"type": condition_type, "role": "reference", "uri": uri})
    return conditions


def build_ref2va_conditions(
    *,
    images: list[str],
    videos: list[str],
    audios: list[str],
    materials: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build conditions, preserving explicit ordered material inputs exactly."""
    if materials is not None:
        if images or videos or audios:
            raise ValueError("--material cannot be combined with --image, --video, or --audio.")
        return parse_ref2va_ordered_materials(materials)
    return [
        *({"type": "image", "role": "reference", "uri": path} for path in images),
        *({"type": "video", "role": "reference", "uri": path} for path in videos),
        *({"type": "audio", "role": "reference", "uri": path} for path in audios),
    ]


def run(
    pipeline: MiniMaxH3Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    conditions: list[dict[str, Any]] | None = None,
    seed: int = PPL_CONFIG["seed"],
    resolution: str = PPL_CONFIG["resolution"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    target_video_length: float | None = PPL_CONFIG["target_video_length"],
    task: str = "s2v",
    *,
    flow_shift: float | None = PPL_CONFIG["flow_shift"],
    audio_flow_shift: float | None = PPL_CONFIG["audio_flow_shift"],
) -> MiniMaxH3Generation:
    """Run Ref2VA with an ordered heterogeneous condition list."""
    if task != "s2v":
        raise ValueError(f"unsupported MiniMax H3 Ref2VA service task: {task}")
    if resolution != PPL_CONFIG["resolution"]:
        expected = PPL_CONFIG["resolution"]
        raise ValueError(f"MiniMax H3 only supports resolution={expected!r}")
    resolved_conditions = default_ref2va_conditions() if conditions is None else conditions
    return pipeline(
        task="ref2va",
        prompt=prompt,
        conditions=resolved_conditions,
        target={
            "short_edge": PPL_CONFIG["short_edge"],
            "aspect_ratio": aspect_ratio,
            "duration_seconds": target_video_length,
        },
        seed=seed,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )


def run_request(
    pipeline: MiniMaxH3Pipeline,
    request_path: str | Path,
    *,
    num_inference_steps: int | None = None,
) -> MiniMaxH3Generation:
    """Run an ordered Ref2VA JSON request without changing its condition order."""
    return run_minimax_h3_request(
        pipeline,
        request_path,
        num_inference_steps=num_inference_steps,
        expected_partition=PPL_CONFIG["partition"],
    )


def run_with_file(
    pipeline: MiniMaxH3Pipeline,
    prompt: str = PPL_CONFIG["prompt"],
    conditions: list[dict[str, Any]] | None = None,
    seed: int = PPL_CONFIG["seed"],
    output_path: str = "minimax_h3_ref2va.mp4",
    resolution: str = PPL_CONFIG["resolution"],
    aspect_ratio: str = PPL_CONFIG["aspect_ratio"],
    target_video_length: float | None = PPL_CONFIG["target_video_length"],
    task: str = "s2v",
    flow_shift: float | None = PPL_CONFIG["flow_shift"],
    audio_flow_shift: float | None = PPL_CONFIG["audio_flow_shift"],
    **_: object,
) -> dict[str, str]:
    """Run MiniMax H3 Ref2VA and save its synchronized audio-video output."""
    result = run(
        pipeline,
        prompt,
        conditions,
        seed,
        resolution,
        aspect_ratio,
        target_video_length,
        task,
        flow_shift=flow_shift,
        audio_flow_shift=audio_flow_shift,
    )
    save_generation(result, output_path)
    return {"output_path": str(Path(output_path))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MiniMax H3 Ref2VA audio-video on H100 GPUs")
    parser.add_argument("--model-root", default=PPL_CONFIG["model_root"])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--material", action="append", default=None, metavar="TYPE=URI")
    parser.add_argument(
        "--request", type=Path, help="JSON request; its prompt, conditions, and target are authoritative"
    )
    parser.add_argument("--prompt", default=PPL_CONFIG["prompt"])
    parser.add_argument(
        "--target-video-length",
        "--duration",
        dest="target_video_length",
        type=float,
        default=PPL_CONFIG["target_video_length"],
    )
    parser.add_argument("--seed", type=int, default=PPL_CONFIG["seed"])
    parser.add_argument("--steps", type=int, help="Override the configured request step count")
    parser.add_argument(
        "--aspect-ratio",
        choices=("auto", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        default=PPL_CONFIG["cli_aspect_ratio"],
    )
    parser.add_argument("--flow-shift", type=float, default=PPL_CONFIG["flow_shift"])
    parser.add_argument("--audio-flow-shift", type=float, default=PPL_CONFIG["audio_flow_shift"])
    parser.add_argument("--device", default=PPL_CONFIG["device"])
    parser.add_argument("--quantization", choices=("fp8", "torchao-fp8", "tf-kernel-fp8", "bnb-nf4"))
    parser.add_argument("--gpu-num", "--ulysses-degree", dest="gpu_num", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--attention-chunks", type=int, choices=(1, 2), default=PPL_CONFIG["attention_chunks"])
    parser.add_argument(
        "--ulysses-sequence-mode",
        choices=("padded", "valid_only"),
        default=PPL_CONFIG["ulysses_sequence_mode"],
    )
    fsdp_group = parser.add_mutually_exclusive_group()
    fsdp_group.add_argument("--enable-fsdp", dest="enable_fsdp", action="store_true")
    fsdp_group.add_argument("--disable-fsdp", dest="enable_fsdp", action="store_false")
    parser.set_defaults(enable_fsdp=PPL_CONFIG["enable_fsdp"])
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--online-adaln-cache", dest="online_adaln_cache", action="store_true")
    cache_group.add_argument("--no-online-adaln-cache", dest="online_adaln_cache", action="store_false")
    parser.set_defaults(online_adaln_cache=None)
    parser.add_argument("--output-path", "--output", dest="output_path", default="minimax_h3_ref2va.mp4")
    args = parser.parse_args()

    if args.request is not None and (args.image or args.video or args.audio or args.material):
        parser.error("--request cannot be combined with --image, --video, --audio, or --material")

    if args.request is not None:
        request_steps = args.steps
        if request_steps is None:
            request_steps = int(
                load_minimax_h3_request(args.request).get("num_inference_steps", PPL_CONFIG["num_inference_steps"])
            )
        conditions = None
    else:
        request_steps = PPL_CONFIG["num_inference_steps"] if args.steps is None else args.steps
        conditions = build_ref2va_conditions(
            images=args.image, videos=args.video, audios=args.audio, materials=args.material
        )

    online_adaln_cache = args.online_adaln_cache
    if online_adaln_cache is None:
        online_adaln_cache = False if args.request is not None else PPL_CONFIG["online_adaln_cache"]

    pipeline = get_pipeline(
        args.gpu_num,
        args.model_root,
        device=args.device,
        num_inference_steps=request_steps,
        enable_fsdp=args.enable_fsdp,
        online_adaln_cache=online_adaln_cache,
        attention_chunks=args.attention_chunks,
        ulysses_sequence_mode=args.ulysses_sequence_mode,
        quantization=args.quantization,
    )
    try:
        if args.request is not None:
            generation = run_request(pipeline, args.request, num_inference_steps=args.steps)
            save_generation(generation, args.output_path)
            result = {"output_path": str(Path(args.output_path))}
        else:
            result = run_with_file(
                pipeline,
                prompt=args.prompt,
                conditions=conditions or None,
                seed=args.seed,
                output_path=args.output_path,
                aspect_ratio=args.aspect_ratio,
                target_video_length=args.target_video_length,
                flow_shift=args.flow_shift,
                audio_flow_shift=args.audio_flow_shift,
            )
        print("Output saved to {}".format(result["output_path"]))
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
