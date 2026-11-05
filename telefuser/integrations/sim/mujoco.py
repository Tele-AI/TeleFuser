"""Optional MuJoCo implementation of the generic simulator boundary."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from telefuser.vla.contracts import ActionSpaceSpec, ObservationSpaceSpec, RobotObservation, RobotState
from telefuser.vla.runtime.executor import RobotAction


@dataclass(frozen=True)
class MuJoCoJointBinding:
    """Map one semantic action dimension to one or more MuJoCo joints."""

    dimension_name: str
    joint_names: tuple[str, ...]
    normalized_to_joint_range: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dimension_name, str) or not self.dimension_name:
            raise ValueError("dimension_name must be a non-empty string")
        if not isinstance(self.joint_names, tuple) or not self.joint_names:
            raise ValueError("joint_names must be a non-empty tuple")
        if any(not isinstance(name, str) or not name for name in self.joint_names):
            raise ValueError("joint_names must contain non-empty strings")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be unique within one binding")
        if not isinstance(self.normalized_to_joint_range, bool):
            raise TypeError("normalized_to_joint_range must be a boolean")


@dataclass(frozen=True)
class _ResolvedJoint:
    qpos_address: int
    dof_address: int
    limited: bool
    lower: float
    upper: float


class MuJoCoSimulatorAdapter:
    """Apply semantic robot actions to a MuJoCo model with PD control.

    MuJoCo is imported only when this adapter is instantiated, so the package
    remains optional and cannot affect existing TeleFuser pipelines.
    """

    def __init__(
        self,
        model: Any,
        action_space: ActionSpaceSpec,
        observation_space: ObservationSpaceSpec,
        joint_bindings: Sequence[MuJoCoJointBinding],
        *,
        camera_names: Mapping[str, str] | None = None,
        image_height: int = 256,
        image_width: int = 256,
        steps_per_action: int = 5,
        position_gain: float = 80.0,
        damping_gain: float = 8.0,
        max_force: float = 200.0,
    ) -> None:
        self._mujoco = self._import_mujoco()
        if not isinstance(action_space, ActionSpaceSpec):
            raise TypeError("action_space must be an ActionSpaceSpec")
        if not isinstance(observation_space, ObservationSpaceSpec):
            raise TypeError("observation_space must be an ObservationSpaceSpec")
        if isinstance(steps_per_action, bool) or not isinstance(steps_per_action, int) or steps_per_action < 1:
            raise ValueError("steps_per_action must be a positive integer")
        if isinstance(image_height, bool) or not isinstance(image_height, int) or image_height < 1:
            raise ValueError("image_height must be a positive integer")
        if isinstance(image_width, bool) or not isinstance(image_width, int) or image_width < 1:
            raise ValueError("image_width must be a positive integer")
        for name, value in (
            ("position_gain", position_gain),
            ("damping_gain", damping_gain),
            ("max_force", max_force),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

        bindings = tuple(joint_bindings)
        if tuple(binding.dimension_name for binding in bindings) != action_space.dimension_names:
            raise ValueError("joint binding order must match action_space.dimension_names")
        physical_names = [name for binding in bindings for name in binding.joint_names]
        if len(set(physical_names)) != len(physical_names):
            raise ValueError("a MuJoCo joint cannot be controlled by multiple action dimensions")

        self.model = model
        self.data = self._mujoco.MjData(model)
        self.action_space = action_space
        self.observation_space = observation_space
        self.joint_bindings = bindings
        self.steps_per_action = steps_per_action
        self.position_gain = float(position_gain)
        self.damping_gain = float(damping_gain)
        self.max_force = float(max_force)
        self._resolved_bindings = tuple(self._resolve_binding(binding) for binding in bindings)
        self._camera_names = dict(camera_names or {})
        expected_images = {image.name for image in observation_space.images}
        if set(self._camera_names) != expected_images:
            raise ValueError("camera_names keys must exactly match observation_space image names")
        for camera_name in self._camera_names.values():
            if self._mujoco.mj_name2id(model, self._mujoco.mjtObj.mjOBJ_CAMERA, camera_name) < 0:
                raise ValueError(f"MuJoCo model does not contain camera {camera_name!r}")
        self._renderer = (
            self._mujoco.Renderer(model, height=image_height, width=image_width) if self._camera_names else None
        )
        self._lock = threading.RLock()
        self._closed = False
        self._mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _import_mujoco() -> ModuleType:
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError(
                "MuJoCo is optional; install it in the simulator virtual environment before using this adapter"
            ) from error
        return mujoco

    def observe(self) -> RobotObservation:
        """Return current joint state and named RGB camera frames."""
        with self._lock:
            self._ensure_open()
            self._mujoco.mj_forward(self.model, self.data)
            timestamp_ns = time.monotonic_ns()
            state = torch.tensor(self._read_action_state(), dtype=torch.float32)
            images: dict[str, np.ndarray] = {}
            if self._renderer is not None:
                for observation_name, camera_name in self._camera_names.items():
                    self._renderer.update_scene(self.data, camera=camera_name)
                    images[observation_name] = np.ascontiguousarray(self._renderer.render().copy())
            observation = RobotObservation(
                state=RobotState(state, self.action_space.dimension_names, timestamp_ns),
                images=images,
                metadata={"simulator": "mujoco", "simulation_time_s": float(self.data.time)},
            )
            self.observation_space.validate(observation)
            return observation

    def execute(self, action: RobotAction) -> None:
        """Advance physics while driving joints toward one absolute action."""
        if not isinstance(action, RobotAction):
            raise TypeError("MuJoCo execute expects RobotAction")
        self.action_space.require_compatible(action.action_space, context="MuJoCo action space")
        values = action.values.detach().to(device="cpu", dtype=torch.float64).numpy()
        if values.shape != (self.action_space.dimension,):
            raise ValueError(f"MuJoCo action must have shape ({self.action_space.dimension},), got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("MuJoCo action must contain only finite values")

        with self._lock:
            self._ensure_open()
            targets = self._resolve_targets(values)
            for _ in range(self.steps_per_action):
                self._mujoco.mj_forward(self.model, self.data)
                self.data.qfrc_applied.fill(0.0)
                for target, joints in zip(targets, self._resolved_bindings, strict=True):
                    for joint in joints:
                        position_error = target - float(self.data.qpos[joint.qpos_address])
                        velocity = float(self.data.qvel[joint.dof_address])
                        force = (
                            float(self.data.qfrc_bias[joint.dof_address])
                            + self.position_gain * position_error
                            - self.damping_gain * velocity
                        )
                        self.data.qfrc_applied[joint.dof_address] = np.clip(
                            force,
                            -self.max_force,
                            self.max_force,
                        )
                self._mujoco.mj_step(self.model, self.data)
            self.data.qfrc_applied.fill(0.0)

    def reset(self) -> RobotObservation:
        """Reset model state and return the first rendered observation."""
        with self._lock:
            self._ensure_open()
            self._mujoco.mj_resetData(self.model, self.data)
            self._mujoco.mj_forward(self.model, self.data)
            return self.observe()

    def close(self) -> None:
        """Release the optional offscreen renderer."""
        with self._lock:
            if self._closed:
                return
            if self._renderer is not None:
                self._renderer.close()
            self._closed = True

    def _resolve_binding(self, binding: MuJoCoJointBinding) -> tuple[_ResolvedJoint, ...]:
        resolved: list[_ResolvedJoint] = []
        for name in binding.joint_names:
            joint_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model does not contain joint {name!r}")
            joint_type = int(self.model.jnt_type[joint_id])
            supported_types = {
                int(self._mujoco.mjtJoint.mjJNT_HINGE),
                int(self._mujoco.mjtJoint.mjJNT_SLIDE),
            }
            if joint_type not in supported_types:
                raise ValueError(f"MuJoCo joint {name!r} must be a hinge or slide joint")
            limited = bool(self.model.jnt_limited[joint_id])
            lower, upper = (float(value) for value in self.model.jnt_range[joint_id])
            if binding.normalized_to_joint_range and (not limited or upper <= lower):
                raise ValueError(f"normalized MuJoCo joint {name!r} requires a finite increasing range")
            resolved.append(
                _ResolvedJoint(
                    qpos_address=int(self.model.jnt_qposadr[joint_id]),
                    dof_address=int(self.model.jnt_dofadr[joint_id]),
                    limited=limited,
                    lower=lower,
                    upper=upper,
                )
            )
        return tuple(resolved)

    def _resolve_targets(self, values: np.ndarray) -> tuple[float, ...]:
        targets: list[float] = []
        for value, binding, joints in zip(values, self.joint_bindings, self._resolved_bindings, strict=True):
            target = float(value)
            if binding.normalized_to_joint_range:
                target = float(np.clip(target, 0.0, 1.0))
                target = joints[0].lower + target * (joints[0].upper - joints[0].lower)
            lower = max((joint.lower for joint in joints if joint.limited), default=-np.inf)
            upper = min((joint.upper for joint in joints if joint.limited), default=np.inf)
            targets.append(float(np.clip(target, lower, upper)))
        return tuple(targets)

    def _read_action_state(self) -> np.ndarray:
        values = np.empty(len(self.joint_bindings), dtype=np.float32)
        for index, (binding, joints) in enumerate(zip(self.joint_bindings, self._resolved_bindings, strict=True)):
            joint_values = np.asarray([self.data.qpos[joint.qpos_address] for joint in joints], dtype=np.float64)
            value = float(joint_values.mean())
            if binding.normalized_to_joint_range:
                value = (value - joints[0].lower) / (joints[0].upper - joints[0].lower)
                value = float(np.clip(value, 0.0, 1.0))
            values[index] = value
        return values

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuJoCo simulator adapter is closed")
