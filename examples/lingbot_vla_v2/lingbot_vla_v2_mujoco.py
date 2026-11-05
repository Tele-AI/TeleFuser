"""Run the generic LingBot VLA loop against a local headless MuJoCo scene."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from PIL import Image

from telefuser.client import AsyncVLAClient
from telefuser.integrations.sim import MuJoCoJointBinding, MuJoCoSimulatorAdapter
from telefuser.pipelines.lingbot_vla_v2.robot_profile import (
    ROBOTWIN_ACTION_ORDER,
    ROBOTWIN_ACTION_SPACE,
    ROBOTWIN_CAMERA_KEYS,
    ROBOTWIN_OBSERVATION_SPACE,
)
from telefuser.vla import RobotActionChunk
from telefuser.vla.runtime import ChunkStatus, SimulatorChunkRuntime

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = Path("/data/RoboTwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf")
LOCAL_MESA_LIBDIR = REPOSITORY_ROOT / ".venv-mujoco-libs/root/usr/lib/x86_64-linux-gnu"
CAMERA_NAMES = {
    ROBOTWIN_CAMERA_KEYS[0]: "telefuser_cam_high",
    ROBOTWIN_CAMERA_KEYS[1]: "telefuser_cam_left",
    ROBOTWIN_CAMERA_KEYS[2]: "telefuser_cam_right",
}


def _ensure_headless_runtime() -> None:
    """Re-exec once so the dynamic loader sees repository-local OSMesa."""
    library_path = str(LOCAL_MESA_LIBDIR)
    current_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if os.environ.get("MUJOCO_GL") == "osmesa" and library_path in current_paths:
        return
    if not (LOCAL_MESA_LIBDIR / "libOSMesa.so.8").is_file():
        raise RuntimeError(
            "repository-local OSMesa is missing; run examples/lingbot_vla_v2/setup_mujoco_local.sh first"
        )
    environment = os.environ.copy()
    environment["MUJOCO_GL"] = "osmesa"
    environment["LD_LIBRARY_PATH"] = ":".join(path for path in (library_path, *current_paths) if path)
    os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def _build_smoke_model(urdf_path: Path) -> Any:
    """Add a small tabletop scene and three fixed cameras to the RoboTwin URDF."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.option.timestep = 0.002
    spec.worldbody.add_light(name="telefuser_key_light", pos=[1.0, 0.0, 2.5], dir=[-0.3, 0.0, -1.0])
    spec.worldbody.add_geom(
        name="telefuser_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3.0, 3.0, 0.05],
        rgba=[0.24, 0.26, 0.28, 1.0],
    )
    spec.worldbody.add_geom(
        name="telefuser_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.7, 0.0, 0.66],
        size=[0.55, 0.7, 0.05],
        rgba=[0.45, 0.32, 0.22, 1.0],
        friction=[1.0, 0.01, 0.001],
    )
    cube = spec.worldbody.add_body(name="telefuser_cube", pos=[0.65, 0.0, 0.77])
    cube.add_freejoint(name="telefuser_cube_free")
    cube.add_geom(
        name="telefuser_cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.05, 0.05, 0.05],
        mass=0.1,
        rgba=[0.85, 0.12, 0.08, 1.0],
        friction=[1.0, 0.01, 0.001],
    )
    spec.worldbody.add_camera(
        name=CAMERA_NAMES[ROBOTWIN_CAMERA_KEYS[0]],
        pos=[2.2, 0.0, 1.55],
        zaxis=[1.55, 0.0, 0.75],
        fovy=55.0,
    )
    spec.worldbody.add_camera(
        name=CAMERA_NAMES[ROBOTWIN_CAMERA_KEYS[1]],
        pos=[1.15, 1.25, 1.15],
        zaxis=[0.5, 0.95, 0.35],
        fovy=60.0,
    )
    spec.worldbody.add_camera(
        name=CAMERA_NAMES[ROBOTWIN_CAMERA_KEYS[2]],
        pos=[1.15, -1.25, 1.15],
        zaxis=[0.5, -0.95, 0.35],
        fovy=60.0,
    )
    return spec.compile()


def _joint_bindings() -> tuple[MuJoCoJointBinding, ...]:
    return (
        *(MuJoCoJointBinding(ROBOTWIN_ACTION_ORDER[index], (f"fl_joint{index + 1}",)) for index in range(6)),
        MuJoCoJointBinding(ROBOTWIN_ACTION_ORDER[6], ("fl_joint7", "fl_joint8"), True),
        *(MuJoCoJointBinding(ROBOTWIN_ACTION_ORDER[index + 7], (f"fr_joint{index + 1}",)) for index in range(6)),
        MuJoCoJointBinding(ROBOTWIN_ACTION_ORDER[13], ("fr_joint7", "fr_joint8"), True),
    )


def _create_adapter(urdf_path: Path, image_size: int, steps_per_action: int) -> MuJoCoSimulatorAdapter:
    model = _build_smoke_model(urdf_path)
    return MuJoCoSimulatorAdapter(
        model,
        ROBOTWIN_ACTION_SPACE,
        ROBOTWIN_OBSERVATION_SPACE,
        _joint_bindings(),
        camera_names=CAMERA_NAMES,
        image_height=image_size,
        image_width=image_size,
        steps_per_action=steps_per_action,
    )


def _save_images(images: dict[str, Any], output_dir: Path | None) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        Image.fromarray(np.asarray(image)).save(output_dir / f"{name.rsplit('.', maxsplit=1)[-1]}.png")


def _run_local_smoke(
    adapter: MuJoCoSimulatorAdapter,
    *,
    execute_horizon: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    initial = adapter.reset()
    targets = initial.state.values.repeat(execute_horizon, 1)
    targets[:, 0] += torch.linspace(0.01, 0.06, execute_horizon)
    targets[:, 7] -= torch.linspace(0.01, 0.04, execute_horizon)
    targets[:, 6] = 0.5
    targets[:, 13] = 0.5
    chunk = RobotActionChunk(
        actions=targets,
        action_space=ROBOTWIN_ACTION_SPACE,
        valid_length=execute_horizon,
        observation_timestamp_ns=initial.state.timestamp_ns,
        sequence_id=0,
        episode_id="mujoco-local-smoke",
    )
    runtime = SimulatorChunkRuntime(
        adapter,
        ROBOTWIN_ACTION_SPACE,
        chunk.episode_id,
        execute_horizon=execute_horizon,
    )
    status = runtime.accept(chunk, observation_clock_now_ns=initial.state.timestamp_ns)
    if status is not ChunkStatus.READY:
        raise RuntimeError(f"local action chunk was not ready: {status.value}")
    executed = runtime.execute_ready()
    final = adapter.observe()
    _save_images(dict(final.images), output_dir)
    image_std = {name: float(np.asarray(image).std()) for name, image in final.images.items()}
    if any(value <= 1.0 for value in image_std.values()):
        raise RuntimeError(f"MuJoCo returned an empty or nearly uniform camera frame: {image_std}")
    return {
        "mode": "local",
        "executed_actions": executed,
        "simulation_time_s": final.metadata["simulation_time_s"],
        "state_delta_l2": float(torch.linalg.vector_norm(final.state.values - initial.state.values)),
        "image_shape": {name: list(np.asarray(image).shape) for name, image in final.images.items()},
        "image_std": image_std,
        "runtime_state": runtime.state.value,
    }


async def _run_websocket_loop(
    adapter: MuJoCoSimulatorAdapter,
    *,
    server_url: str,
    instruction: str,
    seed: int,
    chunks: int,
    execute_horizon: int,
    output_dir: Path | None,
) -> dict[str, Any]:
    episode_id = "mujoco-websocket-smoke"
    runtime = SimulatorChunkRuntime(
        adapter,
        ROBOTWIN_ACTION_SPACE,
        episode_id,
        execute_horizon=execute_horizon,
    )
    observation = adapter.reset()
    executed_total = 0
    async with AsyncVLAClient(server_url) as client:
        await client.open_session(
            "mujoco-local",
            model_id="lingbot-vla-v2",
            embodiment_id="robotwin",
            episode_id=episode_id,
            execute_horizon=execute_horizon,
            expected_robot_action_space=ROBOTWIN_ACTION_SPACE,
            expected_robot_observation_space=ROBOTWIN_OBSERVATION_SPACE,
        )
        for sequence_id in range(chunks):
            chunk = await client.predict(
                "mujoco-local",
                observation,
                instruction,
                sequence_id,
                seed=seed,
            )
            status = runtime.accept(chunk, observation_clock_now_ns=observation.state.timestamp_ns)
            if status is not ChunkStatus.READY:
                raise RuntimeError(f"server action chunk was not ready: {status.value}")
            executed_total += runtime.execute_ready()
            observation = adapter.observe()
    _save_images(dict(observation.images), output_dir)
    return {
        "mode": "websocket",
        "chunks": chunks,
        "executed_actions": executed_total,
        "simulation_time_s": observation.metadata["simulation_time_s"],
        "final_state": observation.state.values.tolist(),
        "runtime_state": runtime.state.value,
    }


@click.command()
@click.option("--mode", type=click.Choice(("local", "websocket")), default="local", show_default=True)
@click.option("--urdf", "urdf_path", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=DEFAULT_URDF)
@click.option("--server-url", default="ws://127.0.0.1:8000/v1/vla/session", show_default=True)
@click.option("--instruction", default="pick up the red block", show_default=True)
@click.option("--seed", type=int, default=7, show_default=True)
@click.option("--chunks", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--execute-horizon", type=click.IntRange(min=1), default=8, show_default=True)
@click.option("--steps-per-action", type=click.IntRange(min=1), default=5, show_default=True)
@click.option("--image-size", type=click.IntRange(min=64), default=256, show_default=True)
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), default=None)
def main(
    mode: str,
    urdf_path: Path,
    server_url: str,
    instruction: str,
    seed: int,
    chunks: int,
    execute_horizon: int,
    steps_per_action: int,
    image_size: int,
    output_dir: Path | None,
) -> None:
    """Validate local physics alone or the complete generic VLA WebSocket loop."""
    _ensure_headless_runtime()
    adapter = _create_adapter(urdf_path.resolve(), image_size, steps_per_action)
    try:
        if mode == "local":
            result = _run_local_smoke(adapter, execute_horizon=execute_horizon, output_dir=output_dir)
        else:
            result = asyncio.run(
                _run_websocket_loop(
                    adapter,
                    server_url=server_url,
                    instruction=instruction,
                    seed=seed,
                    chunks=chunks,
                    execute_horizon=execute_horizon,
                    output_dir=output_dir,
                )
            )
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    finally:
        adapter.close()


if __name__ == "__main__":
    main()
