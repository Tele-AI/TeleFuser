from __future__ import annotations

import asyncio
import base64
import gc
import math
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug

from .control import LingBotWorldFastControlBuilder, LingBotWorldFastControlContext
from .pipeline import LingBotWorldFastPipeline
from .session import (
    LingBotWorldFastChunkRequest,
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionState,
    resolve_lingbot_frame_count,
)

_DIRECTION_ALIASES = {
    "ArrowUp": "w",
    "ArrowDown": "s",
    "ArrowLeft": "j",
    "ArrowRight": "l",
    "KeyW": "w",
    "KeyA": "a",
    "KeyS": "s",
    "KeyD": "d",
    "KeyI": "i",
    "KeyJ": "j",
    "KeyK": "k",
    "KeyL": "l",
    "w": "w",
    "a": "a",
    "s": "s",
    "d": "d",
    "i": "i",
    "j": "j",
    "k": "k",
    "l": "l",
    "up": "w",
    "down": "s",
    "left": "j",
    "right": "l",
    "forward": "w",
    "backward": "s",
}
_ACTION_DIRECTIONS = ("w", "a", "s", "d")
MAX_GENERATION_SECONDS = 20.0


class LingBotWorldFastService:
    """Bidirectional WebRTC service for LingBot-World-Fast."""

    def __init__(
        self,
        pipeline: LingBotWorldFastPipeline,
        default_fps: int = 16,
        default_session_config: Mapping[str, object] | None = None,
        max_generation_seconds: float = MAX_GENERATION_SECONDS,
    ) -> None:
        self.pipeline = pipeline
        self.default_fps = default_fps
        self.default_session_config = dict(default_session_config or {})
        if max_generation_seconds <= 0:
            raise ValueError(f"max_generation_seconds must be positive, got {max_generation_seconds}")
        self.max_generation_seconds = float(max_generation_seconds)
        self._sessions: dict[str, LingBotWorldFastSessionState] = {}

    def start(self) -> None:
        logger.info("LingBotWorldFastService started")

    def stop(self) -> None:
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)
        self.pipeline.close()

    def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    @staticmethod
    def _load_image(config: dict) -> Image.Image:
        image = config.get("image")
        image_path = config.get("image_path")
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
        raise ValueError("LingBotWorldFastService requires 'image' or 'image_path'")

    @staticmethod
    def _frame_num_for_duration(max_duration_seconds: float, fps: int, chunk_size: int) -> int:
        """Return the longest 4n+1 output length with complete latent chunks."""
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if max_duration_seconds <= 0:
            raise ValueError(f"max_duration_seconds must be positive, got {max_duration_seconds}")
        max_latent_frames = int(math.floor(max_duration_seconds * fps / 4)) + 1
        latent_frames = (max_latent_frames // chunk_size) * chunk_size
        if latent_frames < chunk_size:
            raise ValueError(
                f"max_duration_seconds={max_duration_seconds} is too short for chunk_size={chunk_size} at fps={fps}"
            )
        return 4 * (latent_frames - 1) + 1

    def create_session(self, config: dict) -> str:
        for stale_session_id, stale_state in list(self._sessions.items()):
            if not stale_state.active:
                self.close_session(stale_session_id)
        if self._sessions:
            raise RuntimeError(
                "LingBotWorldFastService supports one active session at a time; stop it before reconnecting"
            )
        defaults = self.default_session_config

        session_id = config.get("session_id") or str(uuid.uuid4())
        image = self._load_image(config)
        intrinsics = config.get("intrinsics")
        if intrinsics is None and config.get("intrinsics_path"):
            intrinsics = np.load(Path(config["intrinsics_path"]))

        fps_value = config.get("fps", defaults.get("fps", self.default_fps))
        if fps_value is None:
            fps_value = self.default_fps
        chunk_size_value = config.get("chunk_size", defaults.get("chunk_size", 3))
        if chunk_size_value is None:
            chunk_size_value = 3
        max_duration_value = config.get(
            "max_duration_seconds",
            defaults.get("max_duration_seconds", self.max_generation_seconds),
        )
        if max_duration_value is None:
            max_duration_value = self.max_generation_seconds

        fps = int(fps_value)
        chunk_size = int(chunk_size_value)
        max_duration_seconds = float(max_duration_value)
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if max_duration_seconds <= 0:
            raise ValueError(f"max_duration_seconds must be positive, got {max_duration_seconds}")
        if max_duration_seconds > self.max_generation_seconds:
            raise ValueError(f"max_duration_seconds must not exceed {self.max_generation_seconds:g}")

        frame_policy = str(config.get("frame_policy", defaults.get("frame_policy", "truncate")))
        requested_frame_num = config.get("frame_num")
        frame_num = (
            int(requested_frame_num)
            if requested_frame_num is not None
            else self._frame_num_for_duration(max_duration_seconds, fps, chunk_size)
        )
        frame_num, _ = resolve_lingbot_frame_count(frame_num, chunk_size, frame_policy)
        duration_seconds = (frame_num - 1) / fps
        if duration_seconds > max_duration_seconds:
            raise ValueError(
                f"LingBot streaming duration must not exceed {max_duration_seconds:g} seconds, "
                f"got {duration_seconds:g} seconds"
            )

        session_config = LingBotWorldFastSessionConfig(
            prompt=config.get("prompt", defaults.get("prompt", "")),
            image=image,
            control_mode=config.get("control_mode", defaults.get("control_mode", "cam")),
            fps=fps,
            chunk_size=chunk_size,
            frame_num=frame_num,
            frame_policy=frame_policy,
            sample_shift=float(config.get("sample_shift", defaults.get("sample_shift", 10.0))),
            seed=int(config.get("seed", defaults.get("seed", 42))),
            max_attention_size=config.get("max_attention_size", defaults.get("max_attention_size")),
            max_sequence_length=int(config.get("max_sequence_length", defaults.get("max_sequence_length", 512))),
            intrinsics=intrinsics,
            control_move_step=float(config.get("control_move_step", defaults.get("control_move_step", 0.05))),
            control_yaw_step_degrees=float(
                config.get(
                    "control_yaw_step_degrees",
                    defaults.get("control_yaw_step_degrees", 2.0),
                )
            ),
            control_lateral_step=float(config.get("control_lateral_step", defaults.get("control_lateral_step", 0.05))),
            control_pitch_step_degrees=float(
                config.get(
                    "control_pitch_step_degrees",
                    defaults.get("control_pitch_step_degrees", 2.0),
                )
            ),
            control_pitch_limit_degrees=float(
                config.get(
                    "control_pitch_limit_degrees",
                    defaults.get("control_pitch_limit_degrees", 85.0),
                )
            ),
            show_control_hud=bool(config.get("show_control_hud", defaults.get("show_control_hud", True))),
        )
        control_context = self.pipeline.control_context(session_config)
        state = LingBotWorldFastSessionState(
            config=session_config,
            control_context=control_context,
            output_queue=asyncio.Queue(),
        )
        self._sessions[session_id] = state
        logger.info(f"LingBotWorld session created: {session_id}")
        return session_id

    @staticmethod
    def _put_output(state: LingBotWorldFastSessionState, payload: dict) -> None:
        if state.output_queue is None or state.loop is None:
            return
        try:
            state.loop.call_soon_threadsafe(state.output_queue.put_nowait, payload)
        except Exception as exc:
            logger.warning(f"Failed to enqueue LingBotWorld output: {exc}")

    @staticmethod
    def _encode_frames_to_b64(frames: list[Image.Image], quality: int = 85) -> list[str]:
        """Serialize generated frames for the streaming transport."""
        encoded: list[str] = []
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        return encoded

    def _release_generation_session(self, state: LingBotWorldFastSessionState) -> None:
        if state.generation_session is not None:
            self.pipeline.release_session(state.generation_session)
            state.generation_session = None
        with state.control_lock:
            state.pressed_controls.clear()
            state.queued_controls.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _emit_preview_frame(self, state: LingBotWorldFastSessionState) -> None:
        image = state.config.image.convert("RGB")
        control_context = state.control_context or self.pipeline.control_context(state.config)
        width, height = control_context.width, control_context.height
        preview = image.resize((width, height), Image.BICUBIC)
        frames = [preview]
        if state.config.show_control_hud:
            frames = self._overlay_control_hud(frames, controls=None)
        self._put_output(
            state,
            {
                "type": "preview",
                "index": -1,
                "fps": state.config.fps,
                "timestamp": time.time(),
                "frames_b64": self._encode_frames_to_b64(frames),
            },
        )

    @staticmethod
    def _direction_from_chunk(chunk: dict) -> str | None:
        raw = chunk.get("control") or chunk.get("direction") or chunk.get("key")
        if raw is None:
            return None
        return _DIRECTION_ALIASES.get(str(raw))

    @staticmethod
    def _is_explicit_control_chunk(chunk: dict) -> bool:
        return "control_tensor" in chunk or chunk.get("poses") is not None

    @staticmethod
    def _rotation_matrix(axis: str, angle: float) -> np.ndarray:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        if axis == "x":
            return np.asarray([[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]])
        if axis == "y":
            return np.asarray([[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]])
        raise ValueError(f"Unsupported rotation axis: {axis}")

    @classmethod
    def _integrate_camera_step(
        cls,
        c2w: np.ndarray,
        pitch: float,
        controls: set[str],
        config: LingBotWorldFastSessionConfig,
    ) -> tuple[np.ndarray, float]:
        pitch_delta = 0.0
        pitch_step = math.radians(float(config.control_pitch_step_degrees))
        if "i" in controls:
            pitch_delta += pitch_step
        if "k" in controls:
            pitch_delta -= pitch_step
        pitch_limit = math.radians(float(config.control_pitch_limit_degrees))
        new_pitch = pitch + pitch_delta
        if -pitch_limit <= new_pitch <= pitch_limit:
            pitch = new_pitch
        else:
            pitch_delta = 0.0

        yaw_delta = 0.0
        yaw_step = math.radians(float(config.control_yaw_step_degrees))
        if "j" in controls:
            yaw_delta -= yaw_step
        if "l" in controls:
            yaw_delta += yaw_step

        rotation = c2w[:3, :3]
        rotation_new = cls._rotation_matrix("y", yaw_delta) @ rotation @ cls._rotation_matrix("x", pitch_delta)
        forward = np.asarray([rotation_new[0, 2], 0.0, rotation_new[2, 2]])
        right = np.asarray([rotation_new[0, 0], 0.0, rotation_new[2, 0]])
        forward_norm = np.linalg.norm(forward)
        right_norm = np.linalg.norm(right)
        if forward_norm > 0:
            forward /= forward_norm + 1e-6
        if right_norm > 0:
            right /= right_norm + 1e-6

        movement = np.zeros(3)
        if "w" in controls:
            movement += forward * float(config.control_move_step)
        if "s" in controls:
            movement -= forward * float(config.control_move_step)
        if "d" in controls:
            movement += right * float(config.control_lateral_step)
        if "a" in controls:
            movement -= right * float(config.control_lateral_step)

        result = np.eye(4)
        result[:3, :3] = rotation_new
        result[:3, 3] = c2w[:3, 3] + movement
        return result, pitch

    @staticmethod
    def _draw_triangle(
        draw: ImageDraw.ImageDraw,
        direction: str,
        center_x: int,
        center_y: int,
        size: int,
        color: tuple[int, int, int, int],
    ) -> None:
        half = size // 2
        tip = max(6, size // 4)
        if direction == "up":
            points = [(center_x, center_y - half), (center_x - tip, center_y - tip), (center_x + tip, center_y - tip)]
        elif direction == "down":
            points = [(center_x, center_y + half), (center_x - tip, center_y + tip), (center_x + tip, center_y + tip)]
        elif direction == "left":
            points = [(center_x - half, center_y), (center_x - tip, center_y - tip), (center_x - tip, center_y + tip)]
        else:
            points = [(center_x + half, center_y), (center_x + tip, center_y - tip), (center_x + tip, center_y + tip)]
        draw.polygon(points, fill=color)

    @classmethod
    def _draw_control_panel(
        cls,
        draw: ImageDraw.ImageDraw,
        left: int,
        top: int,
        cell: int,
        active: set[str],
        active_fill: tuple[int, int, int, int],
        outline: tuple[int, int, int, int],
        label: str,
    ) -> None:
        panel_size = cell * 3
        font = ImageFont.load_default(size=max(12, cell // 3))
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_x = left + (panel_size - label_width) // 2
        label_y = top - label_height - max(4, cell // 10)
        draw.text((label_x, label_y), label, fill=outline, font=font)
        draw.rectangle(
            (left, top, left + panel_size, top + panel_size),
            fill=(10, 18, 32, 150),
            outline=outline,
            width=max(1, cell // 14),
        )
        centers = {
            "up": (left + cell + cell // 2, top + cell // 2),
            "left": (left + cell // 2, top + cell + cell // 2),
            "right": (left + cell * 2 + cell // 2, top + cell + cell // 2),
            "down": (left + cell + cell // 2, top + cell * 2 + cell // 2),
        }
        for direction, (center_x, center_y) in centers.items():
            fill = active_fill if direction in active else (100, 116, 139, 170)
            cls._draw_triangle(draw, direction, center_x, center_y, cell, fill)

    @classmethod
    def _overlay_control_hud(cls, frames: list[Image.Image], controls: list[str] | None) -> list[Image.Image]:
        controls_active = set(controls or ())
        movement_active = {
            direction
            for direction, control in {"up": "w", "down": "s", "left": "a", "right": "d"}.items()
            if control in controls_active
        }
        rotation_active = {
            direction
            for direction, control in {"up": "i", "down": "k", "left": "j", "right": "l"}.items()
            if control in controls_active
        }
        out: list[Image.Image] = []
        for frame in frames:
            image = frame.convert("RGB")
            width, height = image.size
            pad = max(10, min(width, height) // 32)
            cell = max(28, min(width, height) // 12)
            panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)
            panel_size = cell * 3
            top = height - pad - panel_size
            cls._draw_control_panel(
                draw,
                left=pad,
                top=top,
                cell=cell,
                active=movement_active,
                active_fill=(37, 99, 235, 230),
                outline=(96, 165, 250, 210),
                label="MOVE",
            )
            cls._draw_control_panel(
                draw,
                left=width - pad - panel_size,
                top=top,
                cell=cell,
                active=rotation_active,
                active_fill=(245, 158, 11, 230),
                outline=(251, 191, 36, 210),
                label="ROTATE",
            )

            out.append(Image.alpha_composite(image.convert("RGBA"), panel).convert("RGB"))
        return out

    def _build_directional_control_chunk(
        self,
        state: LingBotWorldFastSessionState,
        control_context: LingBotWorldFastControlContext,
    ) -> dict | None:
        with state.control_lock:
            controls = set(state.pressed_controls) | set(state.queued_controls)
            state.queued_controls.clear()
            c2w = np.asarray(state.control_c2w, dtype=np.float64)
            pitch = float(state.control_pitch)
            initialized = state.control_initialized

        if not controls:
            return None

        latent_frames = control_context.chunk_size
        poses: list[list[list[float]]] = []
        action_rows: list[list[float]] = []
        previous_pose = c2w.copy() if initialized else None

        if not initialized:
            poses.append(c2w.tolist())
            action_rows.append([1.0 if name in controls else 0.0 for name in _ACTION_DIRECTIONS])
        intervals = latent_frames if initialized else latent_frames - 1
        for _ in range(intervals):
            for _ in range(4):
                c2w, pitch = self._integrate_camera_step(c2w, pitch, controls, state.config)
            poses.append(c2w.tolist())
            action_rows.append([1.0 if name in controls else 0.0 for name in _ACTION_DIRECTIONS])

        with state.control_lock:
            state.control_c2w = c2w.tolist()
            state.control_pitch = pitch
            state.control_initialized = True

        chunk: dict = {
            "type": "control",
            "poses": poses,
            "controls": sorted(controls),
        }
        if previous_pose is not None:
            chunk["previous_pose"] = previous_pose.tolist()
        if control_context.control_type == "act":
            chunk["action"] = action_rows
        return chunk

    def _update_direction_controls(self, state: LingBotWorldFastSessionState, chunk: dict) -> bool:
        direction = self._direction_from_chunk(chunk)
        if direction is None:
            return False

        event = str(chunk.get("event") or chunk.get("action") or "press").lower()
        with state.control_lock:
            if event in {"release", "keyup", "end"}:
                state.pressed_controls.discard(direction)
            elif event == "reset":
                state.pressed_controls.clear()
                state.queued_controls.clear()
                state.control_c2w = np.eye(4).tolist()
                state.control_pitch = 0.0
                state.control_initialized = False
            else:
                state.pressed_controls.add(direction)
                state.queued_controls.add(direction)
            controls = sorted(state.pressed_controls)

        self._put_output(
            state,
            {
                "type": "status",
                "stage": "control_state",
                "controls": controls,
                "timestamp": time.time(),
            },
        )
        return True

    def _worker_loop(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None or state.output_queue is None or state.loop is None:
            return

        def emit_status(stage: str, **data: object) -> None:
            payload = {
                "type": "status",
                "stage": stage,
                "timestamp": time.time(),
            }
            payload.update(data)
            self._put_output(state, payload)

        with ProfilingContext4Debug("workloop"):
            self._run_worker_loop(session_id, state, emit_status)

    def _run_worker_loop(
        self,
        session_id: str,
        state: LingBotWorldFastSessionState,
        emit_status: Callable[..., None],
    ) -> None:
        try:
            self._emit_preview_frame(state)
            control_context = state.control_context or self.pipeline.control_context(state.config)
            control_builder = LingBotWorldFastControlBuilder(control_context)
            state.generation_session = LingBotWorldFastGenerationSession(config=state.config)
            runtime = state.generation_session
            emit_status(
                "runtime_ready",
                width=control_context.width,
                height=control_context.height,
                latent_frames=control_context.latent_frames,
                total_chunks=control_context.latent_frames // control_context.chunk_size,
            )
            chunk_index = 0
            while state.active:
                with state.control_lock:
                    controls_held = bool(state.pressed_controls)
                if controls_held:
                    try:
                        incoming = state.pending_inputs.get_nowait()
                    except queue.Empty:
                        incoming = {"type": "direction_control"}
                else:
                    incoming = state.pending_inputs.get()
                if incoming.get("type") == "stop":
                    break

                explicit_control = incoming if self._is_explicit_control_chunk(incoming) else None
                applied_controls = None
                while True:
                    try:
                        next_item = state.pending_inputs.get_nowait()
                    except queue.Empty:
                        break
                    if next_item.get("type") == "stop":
                        incoming = next_item
                        break
                    if self._is_explicit_control_chunk(next_item):
                        explicit_control = next_item

                if incoming and incoming.get("type") == "stop":
                    break
                if explicit_control is not None:
                    control = control_builder.defer(explicit_control)
                else:
                    directional_chunk = self._build_directional_control_chunk(state, control_context)
                    if directional_chunk is None:
                        continue
                    applied_controls = directional_chunk["controls"]
                    emit_status(
                        "applying_direction_control",
                        index=chunk_index,
                        controls=applied_controls,
                        move_step=state.config.control_move_step,
                        yaw_step_degrees=state.config.control_yaw_step_degrees,
                        lateral_step=state.config.control_lateral_step,
                        pitch_step_degrees=state.config.control_pitch_step_degrees,
                    )
                    control = control_builder.defer(directional_chunk)

                emit_status("generating_chunk", index=chunk_index)
                result = self.pipeline(
                    runtime,
                    LingBotWorldFastChunkRequest(
                        chunk_index=runtime.current_chunk_index,
                        session_id=session_id,
                        control=control,
                    ),
                    progress_callback=emit_status,
                )
                frames = result.frames
                if not frames:
                    break
                if state.config.show_control_hud:
                    frames = self._overlay_control_hud(frames, applied_controls)
                payload = {
                    "type": "chunk",
                    "index": chunk_index,
                    "fps": state.config.fps,
                    "timestamp": time.time(),
                    "frames_b64": self._encode_frames_to_b64(frames),
                }
                self._put_output(state, payload)
                emit_status("chunk_sent", index=chunk_index, frames=len(frames))
                chunk_index += 1
                if result.done:
                    break

        except Exception as exc:
            logger.exception(f"LingBotWorld worker failed: session={session_id}, error={exc}")
            self._put_output(
                state,
                {
                    "type": "error",
                    "stage": "worker_failed",
                    "error": str(exc),
                    "timestamp": time.time(),
                },
            )
        finally:
            state.active = False
            self._put_output(state, {"type": "done"})
            self._release_generation_session(state)

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        is_direction_action = chunk.get("type") == "control" and self._update_direction_controls(state, chunk)
        if is_direction_action:
            event = str(chunk.get("event") or chunk.get("action") or "press").lower()
            if event not in {"release", "keyup", "end", "reset"}:
                state.pending_inputs.put({"type": "direction_control"})
            return
        state.pending_inputs.put(chunk)

    async def pull_chunks(self, session_id: str) -> AsyncGenerator[dict, None]:
        state = self._sessions.get(session_id)
        if state is None or state.output_queue is None:
            return

        if state.loop is None:
            state.loop = asyncio.get_running_loop()
        if state.worker_thread is None or not state.worker_thread.is_alive():
            state.worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(session_id,),
                daemon=True,
                name=f"lingbot-world-{session_id[:8]}",
            )
            state.worker_thread.start()

        while True:
            chunk = await state.output_queue.get()
            if chunk.get("type") == "done":
                break
            yield chunk

    def close_session(self, session_id: str) -> None:
        state = self._sessions.pop(session_id, None)
        if state is None:
            return
        state.active = False
        state.pending_inputs.put({"type": "stop"})
        self._release_generation_session(state)
        logger.info(f"LingBotWorld session closed: {session_id}")
