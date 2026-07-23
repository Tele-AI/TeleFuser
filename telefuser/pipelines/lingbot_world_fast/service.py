from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import io
import math
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from telefuser.metrics.runtime import RuntimeMeasurement, finish_runtime_measurement, start_runtime_measurement
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug

from .control import LingBotWorldFastControlBuilder, LingBotWorldFastControlContext
from .pipeline import LingBotWorldFastPipeline
from .session import (
    LingBotWorldFastDirectionCommand,
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionState,
    resolve_lingbot_frame_count,
)

_DIRECTION_ALIASES = {
    "ArrowUp": "w",
    "ArrowDown": "s",
    "ArrowLeft": "a",
    "ArrowRight": "d",
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
    "left": "a",
    "right": "d",
    "forward": "w",
    "backward": "s",
}
_ACTION_DIRECTIONS = ("w", "a", "s", "d")
MAX_GENERATION_SECONDS = 20.0
DEFAULT_OUTPUT_QUEUE_SIZE = 4
_VIDEO_OUTPUT_TYPES = frozenset({"chunk", "preview"})
_TERMINAL_OUTPUT_TYPES = frozenset({"done", "error"})
_MAX_INPUT_IMAGE_BYTES = 10 * 1024 * 1024


class LingBotWorldFastService:
    """Bidirectional WebRTC service for LingBot-World-Fast."""

    def __init__(
        self,
        pipeline: LingBotWorldFastPipeline,
        default_fps: int = 16,
        default_session_config: Mapping[str, object] | None = None,
        max_generation_seconds: float = MAX_GENERATION_SECONDS,
        output_queue_size: int = DEFAULT_OUTPUT_QUEUE_SIZE,
        close_timeout: float = 300.0,
    ) -> None:
        self.pipeline = pipeline
        self.default_fps = default_fps
        self.default_session_config = dict(default_session_config or {})
        if max_generation_seconds <= 0:
            raise ValueError(f"max_generation_seconds must be positive, got {max_generation_seconds}")
        if output_queue_size <= 0:
            raise ValueError(f"output_queue_size must be positive, got {output_queue_size}")
        if close_timeout <= 0:
            raise ValueError(f"close_timeout must be positive, got {close_timeout}")
        self.max_generation_seconds = float(max_generation_seconds)
        self.output_queue_size = int(output_queue_size)
        self.close_timeout = float(close_timeout)
        self._sessions: dict[str, LingBotWorldFastSessionState] = {}

    def start(self) -> None:
        self.pipeline.warmup(self._warmup_session_config())
        logger.info("LingBotWorldFastService started")

    def _warmup_session_config(self) -> LingBotWorldFastSessionConfig:
        """Build a single-chunk request matching the service's default shape."""
        defaults = self.default_session_config
        chunk_size = int(defaults.get("chunk_size", 3))
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        config = self.pipeline.config
        return LingBotWorldFastSessionConfig(
            prompt=str(defaults.get("prompt", "")),
            image=Image.new("RGB", (config.orig_width, config.orig_height)),
            control_mode=str(defaults.get("control_mode", config.control_type)),
            fps=int(defaults.get("fps", self.default_fps)),
            chunk_size=chunk_size,
            frame_num=4 * (2 * chunk_size - 1) + 1,
            frame_policy="strict",
            sample_shift=float(defaults.get("sample_shift", 10.0)),
            seed=int(defaults.get("seed", 42)),
            max_attention_size=defaults.get("max_attention_size"),
            max_sequence_length=int(defaults.get("max_sequence_length", 512)),
        )

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
            if image.startswith("data:image/"):
                return LingBotWorldFastService._load_image_data_url(image)
            return Image.open(image).convert("RGB")
        if image_path:
            return Image.open(image_path).convert("RGB")
        raise ValueError("LingBotWorldFastService requires an image")

    @staticmethod
    def _load_image_data_url(image_data_url: str) -> Image.Image:
        header, separator, payload = image_data_url.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Browser image must be a base64 data URL")
        max_base64_chars = 4 * ((_MAX_INPUT_IMAGE_BYTES + 2) // 3)
        if len(payload) > max_base64_chars:
            raise ValueError("Browser image exceeds the 10 MiB size limit")
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Browser image contains invalid base64 data") from exc
        if len(image_bytes) > _MAX_INPUT_IMAGE_BYTES:
            raise ValueError("Browser image exceeds the 10 MiB size limit")
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        except OSError as exc:
            raise ValueError("Browser image has an unsupported or invalid format") from exc

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
            intrinsics_width=config.get("intrinsics_width", defaults.get("intrinsics_width")),
            intrinsics_height=config.get("intrinsics_height", defaults.get("intrinsics_height")),
            control_move_step=float(config.get("control_move_step", defaults.get("control_move_step", 0.05))),
            control_yaw_step_degrees=float(
                config.get(
                    "control_yaw_step_degrees",
                    defaults.get("control_yaw_step_degrees", 2.0),
                )
            ),
            control_lateral_step=float(config.get("control_lateral_step", defaults.get("control_lateral_step", 0.05))),
            control_translation_scale=self._positive_float(
                config.get("control_translation_scale", defaults.get("control_translation_scale", 3.0)),
                "control_translation_scale",
            ),
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
            benchmark_metrics=bool(config.get("benchmark_metrics", defaults.get("benchmark_metrics", False))),
        )
        control_context = self.pipeline.control_context(session_config)
        state = LingBotWorldFastSessionState(
            config=session_config,
            control_context=control_context,
            output_queue=asyncio.Queue(maxsize=self.output_queue_size),
        )
        self._sessions[session_id] = state
        logger.info(f"LingBotWorld session created: {session_id}")
        return session_id

    @staticmethod
    def _put_output(state: LingBotWorldFastSessionState, payload: dict) -> None:
        if state.output_queue is None or state.loop is None:
            return
        try:
            state.loop.call_soon_threadsafe(LingBotWorldFastService._enqueue_output, state, payload)
        except Exception as exc:
            logger.warning(f"Failed to enqueue LingBotWorld output: {exc}")

    @staticmethod
    def _enqueue_output(state: LingBotWorldFastSessionState, payload: dict) -> None:
        """Enqueue output without allowing a slow client to retain stale video."""
        output_queue = state.output_queue
        if output_queue is None:
            return

        payload_type = str(payload.get("type", ""))
        queued = output_queue._queue

        def discard_first(predicate: Callable[[dict], bool]) -> bool:
            for item in queued:
                if predicate(item):
                    queued.remove(item)
                    return True
            return False

        if output_queue.full():
            discarded = False
            if payload_type in _VIDEO_OUTPUT_TYPES:
                discarded = discard_first(lambda item: item.get("type") in _VIDEO_OUTPUT_TYPES)
                if not discarded:
                    with state.metrics_lock:
                        state.dropped_video_payloads += 1
                    return
            elif payload_type == "status":
                stage = payload.get("stage")
                discarded = discard_first(lambda item: item.get("type") == "status" and item.get("stage") == stage)
                if not discarded:
                    with state.metrics_lock:
                        state.dropped_status_payloads += 1
                    return
            elif payload_type in _TERMINAL_OUTPUT_TYPES:
                discarded = discard_first(lambda item: item.get("type") in _VIDEO_OUTPUT_TYPES)
                if not discarded:
                    discarded = discard_first(lambda item: item.get("type") == "status")
                if not discarded:
                    discarded = discard_first(lambda _item: True)
                if not discarded:
                    logger.warning("LingBot output queue is full; unable to enqueue terminal output")
                    return

            if discarded:
                with state.metrics_lock:
                    if payload_type in _VIDEO_OUTPUT_TYPES:
                        state.dropped_video_payloads += 1
                    else:
                        state.dropped_status_payloads += 1

        output_queue.put_nowait(payload)
        with state.metrics_lock:
            state.output_queue_high_watermark = max(state.output_queue_high_watermark, output_queue.qsize())

    @staticmethod
    def _runtime_metrics(state: LingBotWorldFastSessionState) -> dict[str, float | int]:
        """Return a stable runtime snapshot suitable for status messages and benchmarks."""
        now = time.monotonic()
        with state.metrics_lock:
            metrics: dict[str, float | int] = {
                "session_age_seconds": round(now - state.created_at_monotonic, 6),
                "output_queue_high_watermark": state.output_queue_high_watermark,
                "dropped_video_payloads": state.dropped_video_payloads,
                "dropped_status_payloads": state.dropped_status_payloads,
                "overwritten_explicit_controls": state.overwritten_explicit_controls,
                "dropped_control_signals": state.dropped_control_signals,
                "overwritten_direction_commands": state.overwritten_direction_commands,
                "last_applied_control_revision": state.last_applied_control_revision or 0,
                "pending_direction_commands": int(state.pending_direction_command is not None),
            }
            if state.worker_started_at_monotonic is not None:
                metrics["worker_start_seconds"] = round(
                    state.worker_started_at_monotonic - state.created_at_monotonic, 6
                )
            if state.first_chunk_sent_at_monotonic is not None:
                metrics["first_chunk_seconds"] = round(
                    state.first_chunk_sent_at_monotonic - state.created_at_monotonic, 6
                )
            return metrics

    def _stream_progress(
        self,
        state: LingBotWorldFastSessionState,
        runtime: LingBotWorldFastGenerationSession,
    ) -> dict[str, float | int]:
        """Build client-facing generation progress for WebRTC telemetry."""
        target_frames = state.config.frame_num
        fps = state.config.fps
        generated_frames = min(runtime.emitted_frames, target_frames)
        target_duration_seconds = (target_frames - 1) / fps
        generated_duration_seconds = min(generated_frames / fps, target_duration_seconds)
        return {
            "service_max_duration_seconds": self.max_generation_seconds,
            "target_duration_seconds": round(target_duration_seconds, 3),
            "generated_duration_seconds": round(generated_duration_seconds, 3),
            "target_frames": target_frames,
            "generated_frames": generated_frames,
            "fps": fps,
            "total_chunks": (((target_frames - 1) // 4 + 1) // state.config.chunk_size),
            "completed_chunks": runtime.current_chunk_index,
        }

    def _release_generation_session(self, state: LingBotWorldFastSessionState) -> None:
        if state.streaming_session is not None:
            streaming_session = state.streaming_session
            streaming_runtime = self.pipeline._get_streaming_runtime()
            scheduler_metrics = streaming_runtime.session_metrics(streaming_session)
            state.scheduler_metrics = asdict(scheduler_metrics) if is_dataclass(scheduler_metrics) else None
            streaming_runtime.close_session(streaming_session)
            state.streaming_session = None
            state.generation_session = None
        elif state.generation_session is not None:
            self.pipeline.release_session(state.generation_session)
            state.generation_session = None
        with state.control_lock:
            state.pressed_controls.clear()
            state.pending_direction_command = None
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
                "frames": frames,
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
        controls: set[str],
    ) -> dict | None:
        with state.control_lock:
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
            "translation_scale": state.config.control_translation_scale,
        }
        if previous_pose is not None:
            chunk["previous_pose"] = previous_pose.tolist()
        if control_context.control_type == "act":
            chunk["action"] = action_rows
        return chunk

    @staticmethod
    def _positive_float(value: object, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive number") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive number")
        return result

    @staticmethod
    def _queue_direction_snapshot(state: LingBotWorldFastSessionState) -> None:
        """Store only the newest short-press snapshot for the next chunk."""
        if not state.pressed_controls:
            return
        state.next_control_revision += 1
        if state.pending_direction_command is not None:
            with state.metrics_lock:
                state.overwritten_direction_commands += 1
        state.pending_direction_command = LingBotWorldFastDirectionCommand(
            revision=state.next_control_revision,
            controls=frozenset(state.pressed_controls),
        )

    @staticmethod
    def _controls_from_snapshot(chunk: dict) -> set[str] | None:
        """Validate a browser control-state message and return canonical controls."""
        raw_controls = chunk.get("controls")
        if not isinstance(raw_controls, list) or not all(isinstance(control, str) for control in raw_controls):
            return None
        controls = {_DIRECTION_ALIASES.get(control) for control in raw_controls}
        if None in controls:
            return None
        return controls

    def _update_direction_controls(self, state: LingBotWorldFastSessionState, chunk: dict) -> bool:
        if chunk.get("type") == "control_state":
            controls = self._controls_from_snapshot(chunk)
            if controls is None:
                logger.warning("Ignoring LingBot control state with unsupported controls")
                return False
            with state.control_lock:
                previous_controls = set(state.pressed_controls)
                state.pressed_controls = controls
                if controls - previous_controls:
                    self._queue_direction_snapshot(state)
                active_controls = sorted(state.pressed_controls)
                pending_count = int(state.pending_direction_command is not None)
                revision = state.next_control_revision
            self._put_output(
                state,
                {
                    "type": "status",
                    "stage": "control_state",
                    "controls": active_controls,
                    "revision": revision,
                    "pending_direction_commands": pending_count,
                    "timestamp": time.time(),
                },
            )
            return True

        direction = self._direction_from_chunk(chunk)
        if direction is None:
            return False

        event = str(chunk.get("event") or chunk.get("action") or "press").lower()
        if event not in {"press", "release", "keyup", "end", "reset", "reset_pose"}:
            logger.warning(f"Ignoring LingBot direction control with unsupported event={event!r}")
            return False
        with state.control_lock:
            if event in {"release", "keyup", "end"}:
                state.pressed_controls.discard(direction)
            elif event == "reset":
                state.pressed_controls.clear()
                state.pending_direction_command = None
            elif event == "reset_pose":
                state.pressed_controls.clear()
                state.pending_direction_command = None
                state.control_c2w = np.eye(4).tolist()
                state.control_pitch = 0.0
                state.control_initialized = False
            else:
                if direction not in state.pressed_controls:
                    state.pressed_controls.add(direction)
                    self._queue_direction_snapshot(state)
            controls = sorted(state.pressed_controls)
            pending_count = int(state.pending_direction_command is not None)
            revision = state.next_control_revision

        self._put_output(
            state,
            {
                "type": "status",
                "stage": "control_state",
                "controls": controls,
                "revision": revision,
                "pending_direction_commands": pending_count,
                "timestamp": time.time(),
            },
        )
        return True

    @staticmethod
    def _wake_control_worker(state: LingBotWorldFastSessionState, signal: dict | None = None) -> None:
        """Wake a waiting worker without retaining obsolete input events."""
        try:
            state.pending_inputs.put_nowait(signal or {"type": "control_update"})
        except queue.Full:
            with state.metrics_lock:
                state.dropped_control_signals += 1

    @staticmethod
    def _stop_control_worker(state: LingBotWorldFastSessionState) -> None:
        """Replace a stale wake-up with the terminal signal needed by the worker."""
        while True:
            try:
                state.pending_inputs.get_nowait()
            except queue.Empty:
                break
        try:
            state.pending_inputs.put_nowait({"type": "stop"})
        except queue.Full:  # pragma: no cover - defensive for alternate Queue implementations
            pass

    def _next_realtime_control(
        self,
        state: LingBotWorldFastSessionState,
        control_context: LingBotWorldFastControlContext,
        control_builder: LingBotWorldFastControlBuilder,
        chunk_index: int,
        emit_status: Callable[..., None],
        block: bool,
    ) -> tuple[object, list[str] | None] | None:
        """Select one queued tap or continue a direction that remains held."""
        while state.active:
            with state.control_lock:
                explicit_control = state.latest_explicit_control
                state.latest_explicit_control = None
                held_controls = frozenset(state.pressed_controls)
                direction_command = None
                if explicit_control is None and state.pending_direction_command is not None:
                    direction_command = state.pending_direction_command
                    state.pending_direction_command = None

            if explicit_control is not None:
                return control_builder.defer(explicit_control), None

            if direction_command is not None:
                controls = set(direction_command.controls)
                revision: int | None = direction_command.revision
            elif held_controls and block:
                controls = set(held_controls)
                revision = None
            else:
                if not block:
                    return None
                try:
                    incoming = state.pending_inputs.get(block=True)
                except queue.Empty:  # pragma: no cover - Queue.get blocks here
                    return None
                if incoming.get("type") == "stop":
                    state.active = False
                    return None
                continue

            directional_chunk = self._build_directional_control_chunk(state, control_context, controls)
            if directional_chunk is None:
                continue
            applied_controls = directional_chunk["controls"]
            with state.control_lock:
                state.last_applied_control_revision = revision
            emit_status(
                "applying_direction_control",
                index=chunk_index,
                controls=applied_controls,
                revision=revision,
                source="tap" if revision is not None else "held",
                move_step=state.config.control_move_step,
                yaw_step_degrees=state.config.control_yaw_step_degrees,
                lateral_step=state.config.control_lateral_step,
                pitch_step_degrees=state.config.control_pitch_step_degrees,
            )
            return control_builder.defer(directional_chunk), applied_controls
        return None

    def _benchmark_devices(self) -> tuple[str | torch.device, ...]:
        """Return every CUDA-capable pipeline device for benchmark synchronization."""
        devices: list[str | torch.device] = []
        for name in ("device", "text_device", "vae_encode_device", "vae_decode_device", "vae_device"):
            device = getattr(self.pipeline, name, None)
            if device is not None and device not in devices:
                devices.append(device)
        parallel_config = getattr(getattr(self.pipeline, "config", None), "parallel_config", None)
        for device_id in getattr(parallel_config, "device_ids", None) or ():
            device = f"cuda:{int(device_id)}"
            if device not in devices:
                devices.append(device)
        return tuple(devices)

    def _start_benchmark_measurement(
        self,
        state: LingBotWorldFastSessionState,
    ) -> RuntimeMeasurement | None:
        if not state.config.benchmark_metrics:
            return None
        try:
            # Generation runs in child actors, while this timer lives in the service
            # process. Process-local allocator peaks would therefore be incomplete.
            return start_runtime_measurement(self._benchmark_devices(), capture_peak_memory=False)
        except Exception as exc:
            logger.warning(f"Could not start LingBot benchmark measurement: {exc}")
            return None

    @staticmethod
    def _finish_benchmark_measurement(
        measurement: RuntimeMeasurement | None,
    ) -> dict[str, object] | None:
        if measurement is None:
            return None
        try:
            return finish_runtime_measurement(measurement)
        except Exception as exc:
            logger.warning(f"Could not finish LingBot benchmark measurement: {exc}")
            return None

    def _run_actor_worker_loop(
        self,
        state: LingBotWorldFastSessionState,
        control_context: LingBotWorldFastControlContext,
        control_builder: LingBotWorldFastControlBuilder,
        emit_status: Callable[..., None],
    ) -> None:
        """Drive dynamic control ingress and ordered output through the shared actor graph."""
        first_item = self._next_realtime_control(state, control_context, control_builder, 0, emit_status, block=True)
        if first_item is None:
            return
        runtime_measurement = self._start_benchmark_measurement(state)
        try:
            runtime = self.pipeline._create_initialized_session(state.config, progress_callback=emit_status)
        finally:
            runtime_facts = self._finish_benchmark_measurement(runtime_measurement)
        state.generation_session = runtime
        if not state.active:
            return
        streaming_runtime = self.pipeline._get_streaming_runtime()
        streaming_session = streaming_runtime.create_session(runtime, progress_callback=emit_status)
        state.streaming_session = streaming_session
        if not state.active:
            return
        emit_status(
            "runtime_ready",
            width=runtime.width,
            height=runtime.height,
            latent_frames=runtime.latent_f,
            total_chunks=runtime.chunk_count,
            stream_progress=self._stream_progress(state, runtime),
            runtime=self._runtime_metadata(runtime),
            **({"measurement": {"name": "runtime_creation", **runtime_facts}} if runtime_facts is not None else {}),
        )

        submitted = 0
        controls_by_chunk: dict[int, list[str] | None] = {}
        measurements_by_chunk: dict[int, RuntimeMeasurement] = {}

        def raise_scheduler_error() -> None:
            error = streaming_runtime.error(streaming_session)
            if error is not None:
                raise RuntimeError("LingBot streaming scheduler failed") from error

        def submit_chunk(item: tuple[object, list[str] | None]) -> None:
            nonlocal submitted
            deferred_control, applied_controls = item
            control = self.pipeline._resolve_control(deferred_control)
            self.pipeline._validate_control(runtime, control)
            chunk_measurement = self._start_benchmark_measurement(state)
            if not streaming_runtime.try_submit_chunk(streaming_session, submitted, control):
                self._finish_benchmark_measurement(chunk_measurement)
                raise RuntimeError("LingBot streaming ingress became unavailable after capacity check")
            controls_by_chunk[submitted] = applied_controls
            if chunk_measurement is not None:
                measurements_by_chunk[submitted] = chunk_measurement
            emit_status(
                "generating_chunk",
                index=submitted,
                controls=applied_controls or [],
                prefetched=submitted > runtime.current_chunk_index,
            )
            with state.metrics_lock:
                state.chunk_started_at_monotonic[submitted] = time.monotonic()
            submitted += 1

        submit_chunk(first_item)
        while state.active and runtime.current_chunk_index < runtime.chunk_count:
            raise_scheduler_error()
            outputs = streaming_runtime.poll_frames(streaming_session)
            for result_index, frames in outputs:
                if result_index != runtime.current_chunk_index:
                    raise RuntimeError(
                        f"LingBot actor output index {result_index} does not match {runtime.current_chunk_index}"
                    )
                if not frames:
                    raise RuntimeError(f"LingBot actor emitted no frames for chunk {result_index}")
                applied_controls = controls_by_chunk.pop(result_index, None)
                chunk_facts = self._finish_benchmark_measurement(measurements_by_chunk.pop(result_index, None))
                if state.config.show_control_hud:
                    frames = self._overlay_control_hud(frames, applied_controls)
                self._put_output(
                    state,
                    {
                        "type": "chunk",
                        "index": result_index,
                        "fps": state.config.fps,
                        "timestamp": time.time(),
                        "frames": frames,
                        "applied_controls": applied_controls or [],
                    },
                )
                now = time.monotonic()
                with state.metrics_lock:
                    started_at = state.chunk_started_at_monotonic.pop(result_index, now)
                    if state.first_chunk_sent_at_monotonic is None:
                        state.first_chunk_sent_at_monotonic = now
                    control_to_chunk_seconds = (
                        now - state.last_control_at_monotonic if state.last_control_at_monotonic is not None else None
                    )
                runtime.current_chunk_index += 1
                runtime.emitted_frames += len(frames)
                emit_status(
                    "chunk_sent",
                    index=result_index,
                    controls=applied_controls or [],
                    frames=len(frames),
                    chunk_elapsed_seconds=round(now - started_at, 6),
                    control_to_chunk_seconds=(
                        round(control_to_chunk_seconds, 6) if control_to_chunk_seconds is not None else None
                    ),
                    runtime_metrics=self._runtime_metrics(state),
                    stream_progress=self._stream_progress(state, runtime),
                    **(
                        {
                            "measurement": {
                                "index": result_index,
                                "frames": len(frames),
                                "compute_seconds": chunk_facts["seconds"],
                                "memory": chunk_facts["memory"],
                            }
                        }
                        if chunk_facts is not None
                        else {}
                    ),
                )

            admitted_prefetch = False
            while (
                state.active
                and submitted < runtime.chunk_count
                and streaming_runtime.can_submit_chunk(streaming_session)
            ):
                item = self._next_realtime_control(
                    state,
                    control_context,
                    control_builder,
                    submitted,
                    emit_status,
                    block=False,
                )
                if item is None:
                    break
                submit_chunk(item)
                admitted_prefetch = True

            if runtime.current_chunk_index >= runtime.chunk_count or not state.active:
                break
            if submitted == runtime.current_chunk_index:
                item = self._next_realtime_control(
                    state,
                    control_context,
                    control_builder,
                    submitted,
                    emit_status,
                    block=True,
                )
                if item is None:
                    break
                if not streaming_runtime.can_submit_chunk(streaming_session):
                    raise RuntimeError("LingBot actor ingress remained blocked without in-flight work")
                submit_chunk(item)
            elif not outputs and not admitted_prefetch:
                streaming_runtime.wait_until_idle(streaming_session, timeout=0.05)

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
        """Run dynamic service ingress through the pipeline-owned actor scheduler."""
        try:
            with state.metrics_lock:
                state.worker_started_at_monotonic = time.monotonic()
            self._emit_preview_frame(state)
            control_context = state.control_context or self.pipeline.control_context(state.config)
            control_builder = LingBotWorldFastControlBuilder(control_context)
            self._run_actor_worker_loop(state, control_context, control_builder, emit_status)
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
            try:
                self._release_generation_session(state)
            except Exception as exc:
                logger.exception(f"LingBotWorld cleanup failed: session={session_id}, error={exc}")
                self._put_output(
                    state,
                    {
                        "type": "error",
                        "stage": "cleanup_failed",
                        "error": str(exc),
                        "timestamp": time.time(),
                    },
                )
            finally:
                self._put_output(
                    state,
                    {
                        "type": "status",
                        "stage": "runtime_summary",
                        "runtime_metrics": self._runtime_metrics(state),
                        "scheduler_metrics": state.scheduler_metrics,
                    },
                )
                self._put_output(state, {"type": "done"})
                if self._sessions.get(session_id) is state:
                    self._sessions.pop(session_id, None)

    def _runtime_metadata(self, runtime: LingBotWorldFastGenerationSession) -> dict[str, int]:
        return {
            "height": int(runtime.height),
            "width": int(runtime.width),
            "latent_frames": int(runtime.latent_f),
            "frame_tokens": int(runtime.frame_tokens),
            "chunk_size": int(runtime.chunk_size),
            "max_attention_size": int(runtime.max_attention_size),
            "kv_local_attn_size": int(self.pipeline.dit.local_attn_size),
            "kv_sink_size": int(self.pipeline.dit.sink_size),
            "kv_cache_capacity_tokens": int(runtime.kv_cache_capacity_tokens),
        }

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        state = self._sessions.get(session_id)
        if state is None or not state.active:
            return
        with state.metrics_lock:
            state.last_control_at_monotonic = time.monotonic()
        is_direction_action = chunk.get("type") in {"control", "control_state"} and self._update_direction_controls(
            state, chunk
        )
        if is_direction_action:
            self._wake_control_worker(state, {"type": "direction_control"})
            return
        with state.control_lock:
            if state.latest_explicit_control is not None:
                with state.metrics_lock:
                    state.overwritten_explicit_controls += 1
            state.latest_explicit_control = chunk
        self._wake_control_worker(state)

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

        completed = False
        try:
            while True:
                chunk = await state.output_queue.get()
                if chunk.get("type") == "done":
                    completed = True
                    break
                yield chunk
        finally:
            if not completed:
                await asyncio.to_thread(self.close_session, session_id)

    def close_session(self, session_id: str, timeout: float | None = None) -> None:
        """Request cancellation and wait only for the configured cleanup deadline."""
        effective_timeout = self.close_timeout if timeout is None else timeout
        if effective_timeout <= 0:
            raise ValueError(f"timeout must be positive, got {effective_timeout}")
        state = self._sessions.get(session_id)
        if state is None:
            return
        state.active = False
        self._stop_control_worker(state)
        worker = state.worker_thread
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=effective_timeout)
        if worker is not None and worker.is_alive():
            logger.warning(
                f"LingBotWorld session cleanup is still running after {effective_timeout:g}s: session={session_id}"
            )
            return
        if worker is None or not worker.is_alive():
            self._release_generation_session(state)
            self._sessions.pop(session_id, None)
        logger.info(f"LingBotWorld session closed: {session_id}")
