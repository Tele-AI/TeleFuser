"""Run the LingBot-VLA v2 base checkpoint with a RobotWin observation adapter."""

from __future__ import annotations

import json

import click
import numpy as np

from telefuser.pipelines.lingbot_vla_v2 import (
    ROBOTWIN_CAMERA_KEYS,
    LingBotVlaV2Observation,
    LingBotVlaV2Pipeline,
)
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline


def get_pipeline(
    model_root: str,
    qwen3vl_root: str,
    device: str = "cuda",
    quantization: str | None = None,
    cuda_graph: bool = False,
) -> LingBotVlaV2Pipeline:
    """Load the official 6B checkpoint and Qwen3-VL processor."""
    return get_lingbot_vla_v2_pipeline(
        model_root,
        qwen3vl_root,
        device=device,
        quantization=quantization,
        cuda_graph=cuda_graph,
    )


@click.command()
@click.option("--model-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--qwen3vl-root", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--camera-high", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--camera-left-wrist", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--camera-right-wrist", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--task", required=True)
@click.option("--state-json", required=True, help="Raw 14-D RobotWin state as a JSON list")
@click.option("--output", default="canonical_action_chunk.npz", type=click.Path(dir_okay=False))
@click.option("--seed", default=None, type=int)
@click.option("--device", default="cuda")
@click.option("--cuda-graph", is_flag=True, help="Enable fixed-shape CUDA Graph inference")
@click.option(
    "--quantization",
    type=click.Choice(("fused-fp8-graph", "torchao-fp8", "tf-kernel-fp8", "bnb-nf4")),
    default=None,
)
def main(
    model_root: str,
    qwen3vl_root: str,
    camera_high: str,
    camera_left_wrist: str,
    camera_right_wrist: str,
    task: str,
    state_json: str,
    output: str,
    seed: int | None,
    device: str,
    cuda_graph: bool,
    quantization: str | None,
) -> None:
    """Predict and save a normalized canonical action chunk."""
    try:
        state = json.loads(state_json)
    except json.JSONDecodeError as error:
        raise click.BadParameter("state-json must be valid JSON") from error
    if not isinstance(state, list) or len(state) != 14:
        raise click.BadParameter("state-json must decode to a 14-element JSON list")
    observation = LingBotVlaV2Observation(
        task=task,
        state=state,
        images=dict(
            zip(
                ROBOTWIN_CAMERA_KEYS,
                (camera_high, camera_left_wrist, camera_right_wrist),
                strict=True,
            )
        ),
    )
    pipeline = get_pipeline(
        model_root,
        qwen3vl_root,
        device=device,
        quantization=quantization,
        cuda_graph=cuda_graph,
    )
    try:
        chunk = pipeline(observation, seed=seed)
        arrays = {
            "canonical_normalized_actions": chunk.canonical_normalized_actions.numpy(),
            "horizon": np.asarray(chunk.horizon),
            "action_dim": np.asarray(chunk.action_dim),
            "checkpoint_variant": np.asarray(chunk.checkpoint_variant),
            "policy_verified": np.asarray(chunk.policy_verified),
            "verification_status": np.asarray(chunk.verification_status),
        }
        np.savez(output, **arrays)
        click.echo(
            f"Saved {chunk.horizon}-step normalized canonical action chunk to {output}; "
            f"policy status: {chunk.verification_status}"
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
