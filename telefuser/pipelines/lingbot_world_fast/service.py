from __future__ import annotations

import asyncio
import gc
import math
import queue
import threading
import time
import uuid
from collections.abc import AsyncGenerator

import torch
from PIL import Image, ImageDraw

from telefuser.metrics.runtime import (
    RuntimeMeasurement,
    finish_runtime_measurement,
    start_runtime_measurement,
)
from telefuser.utils.logging import logger

from .pipeline import LingBotWorldFastPipeline
from .session import (
    LingBotWorldFastRuntimeState,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionState,
)

_DIRECTION_ALIASES = {
    "ArrowUp": "up",
    "ArrowDown": "down",
    "ArrowLeft": "left",
    "ArrowRight": "right",
    "KeyW": "up",
    "KeyS": "down",
    "KeyA": "left",
    "KeyD": "right",
    "w": "up",
    "s": "down",
    "a": "left",
    "d": "right",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "forward": "up",
    "backward": "down",
}
_ACTION_DIRECTIONS = ("up", "down", "left", "right")


class LingBotWorldFastService:
    """Bidirectional WebRTC service for LingBot-World-Fast."""

    def __init__(self, pipeline: LingBotWorldFastPipeline, default_fps: int = 16) -> None:
        self.pipeline = pipeline
        self.default_fps = default_fps
        self._sessions: dict[str, LingBotWorldFastSessionState] = {}

    def start(self) -> None:
        logger.info("LingBotWorldFastService started")

    def stop(self) -> None:
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)

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

    def create_session(self, config: dict) -> str:
        for stale_session_id, stale_state in list(self._sessions.items()):
            if not stale_state.active:
                self.close_session(stale_session_id)
        if self._sessions:
            raise RuntimeError(
                "LingBotWorldFastService supports one active session at a time; stop it before reconnecting"
            )

        session_id = config.get("session_id") or str(uuid.uuid4())
        image = self._load_image(config)

        session_config = LingBotWorldFastSessionConfig(
            prompt=config.get("prompt", ""),
            image=image,
            control_mode=config.get("control_mode", "cam"),
            fps=int(config.get("fps") or self.default_fps),
            chunk_size=int(config.get("chunk_size", 3)),
            frame_num=int(config.get("frame_num", 81)),
            sample_shift=float(config.get("sample_shift", 5.0)),
            seed=int(config.get("seed", 42)),
            max_attention_size=config.get("max_attention_size"),
            offload_model=bool(config.get("offload_model", False)),
            max_sequence_length=int(config.get("max_sequence_length", 512)),
            action_path=config.get("action_path"),
            poses=config.get("poses"),
            intrinsics=config.get("intrinsics"),
            action=config.get("action"),
            control_move_step=float(config.get("control_move_step", 0.18)),
            control_yaw_step_degrees=float(config.get("control_yaw_step_degrees", 10.0)),
            control_lateral_step=float(config.get("control_lateral_step", 0.12)),
            show_control_hud=bool(config.get("show_control_hud", True)),
            benchmark_metrics=bool(config.get("benchmark_metrics", False)),
        )
        state = LingBotWorldFastSessionState(
            config=session_config,
            output_queue=asyncio.Queue(),
            loop=None,
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
    def _release_runtime(state: LingBotWorldFastSessionState) -> None:
        if state.runtime is not None:
            state.runtime.active = False
            state.runtime = None
        with state.control_lock:
            state.pressed_controls.clear()
            state.queued_controls.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _emit_preview_frame(self, state: LingBotWorldFastSessionState) -> None:
        image = state.config.image.convert("RGB")
        width, height = self.pipeline._best_output_size(image.width, image.height, self.pipeline.config.max_area)
        height, width = self.pipeline.check_resize_height_width(height, width)
        preview = image.resize((width, height), Image.BICUBIC)
        self._put_output(
            state,
            {
                "type": "preview",
                "index": -1,
                "fps": state.config.fps,
                "timestamp": time.time(),
                "frames_b64": self.pipeline.encode_frames_to_b64([preview]),
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
        return "control_tensor" in chunk or (chunk.get("poses") is not None and chunk.get("intrinsics") is not None)

    @staticmethod
    def _pose_matrix(yaw: float, position: list[float]) -> list[list[float]]:
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        return [
            [cos_y, 0.0, sin_y, position[0]],
            [0.0, 1.0, 0.0, position[1]],
            [-sin_y, 0.0, cos_y, position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _default_intrinsics(self) -> list[list[float]]:
        width = float(self.pipeline.config.orig_width)
        height = float(self.pipeline.config.orig_height)
        focal = max(width, height)
        return [[focal, focal, width * 0.5, height * 0.5]]

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
    def _overlay_control_hud(cls, frames: list[Image.Image], controls: list[str] | None) -> list[Image.Image]:
        if not controls:
            return frames

        active = set(controls)
        out: list[Image.Image] = []
        for frame in frames:
            image = frame.convert("RGB")
            width, height = image.size
            pad = max(10, min(width, height) // 32)
            cell = max(28, min(width, height) // 12)
            panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(panel)

            left = pad
            top = pad
            panel_size = cell * 3
            draw.rectangle(
                (left, top, left + panel_size, top + panel_size),
                fill=(10, 18, 32, 150),
                outline=(96, 165, 250, 210),
                width=max(1, cell // 14),
            )

            centers = {
                "up": (left + cell + cell // 2, top + cell // 2),
                "left": (left + cell // 2, top + cell + cell // 2),
                "right": (left + cell * 2 + cell // 2, top + cell + cell // 2),
                "down": (left + cell + cell // 2, top + cell * 2 + cell // 2),
            }
            for direction, (cx, cy) in centers.items():
                is_active = direction in active
                fill = (37, 99, 235, 230) if is_active else (100, 116, 139, 170)
                cls._draw_triangle(draw, direction, cx, cy, cell, fill)

            out.append(Image.alpha_composite(image.convert("RGBA"), panel).convert("RGB"))
        return out

    def _build_directional_control_chunk(
        self,
        state: LingBotWorldFastSessionState,
        runtime: LingBotWorldFastRuntimeState,
    ) -> dict | None:
        with state.control_lock:
            controls = set(state.pressed_controls) | set(state.queued_controls)
            state.queued_controls.clear()
            yaw = float(state.control_yaw)
            position = [float(v) for v in state.control_position]

        if not controls or runtime.current_chunk_index >= len(runtime.noise_chunks):
            return None

        latent_frames = int(runtime.noise_chunks[runtime.current_chunk_index].shape[2])
        move_step = float(state.config.control_move_step)
        yaw_step = math.radians(float(state.config.control_yaw_step_degrees))
        lateral_step = float(state.config.control_lateral_step)
        poses: list[list[list[float]]] = []
        action_rows: list[list[float]] = []

        for _ in range(latent_frames):
            strafe = 0
            if "left" in controls and "right" not in controls:
                yaw += yaw_step
                strafe = -1
            elif "right" in controls and "left" not in controls:
                yaw -= yaw_step
                strafe = 1

            forward_x = math.sin(yaw)
            forward_z = math.cos(yaw)
            right_x = math.cos(yaw)
            right_z = -math.sin(yaw)
            if "up" in controls and "down" not in controls:
                position[0] += forward_x * move_step
                position[2] += forward_z * move_step
            elif "down" in controls and "up" not in controls:
                position[0] -= forward_x * move_step
                position[2] -= forward_z * move_step
            if strafe:
                position[0] += right_x * lateral_step * strafe
                position[2] += right_z * lateral_step * strafe

            poses.append(self._pose_matrix(yaw, position))
            action_rows.append([1.0 if name in controls else 0.0 for name in _ACTION_DIRECTIONS])

        with state.control_lock:
            state.control_yaw = yaw
            state.control_position = position

        chunk: dict = {
            "type": "control",
            "poses": poses,
            "intrinsics": self._default_intrinsics(),
            "controls": sorted(controls),
        }
        model_control_type = getattr(
            getattr(self.pipeline, "dit", None),
            "control_type",
            self.pipeline.config.control_type,
        )
        if model_control_type == "act":
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
                state.control_position = [0.0, 0.0, 0.0]
                state.control_yaw = 0.0
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

        devices = (
            self.pipeline.device,
            self.pipeline.vae_device,
            self.pipeline.text_device,
        )

        def start_measurement() -> RuntimeMeasurement | None:
            if not state.config.benchmark_metrics:
                return None
            try:
                return start_runtime_measurement(
                    devices,
                    capture_peak_memory=True,
                )
            except Exception as exc:
                logger.warning(f"Could not start LingBot benchmark measurement: {exc}")
                return None

        def finish_measurement(
            measurement: RuntimeMeasurement | None,
        ) -> dict[str, object] | None:
            if measurement is None:
                return None
            try:
                return finish_runtime_measurement(measurement)
            except Exception as exc:
                logger.warning(f"Could not finish LingBot benchmark measurement: {exc}")
                return None

        try:
            self._emit_preview_frame(state)
            emit_status("initializing_runtime")
            runtime_measurement = start_measurement()
            state.runtime = self.pipeline.create_runtime(state.config, progress_callback=emit_status)
            runtime = state.runtime
            runtime_facts = finish_measurement(runtime_measurement)
            runtime_metadata = self._runtime_metadata(runtime)
            runtime_ready: dict[str, object] = {
                "width": runtime.width,
                "height": runtime.height,
                "latent_frames": runtime.latent_f,
                "total_chunks": len(runtime.noise_chunks),
                "runtime": runtime_metadata,
            }
            if runtime_facts is not None:
                runtime_ready["measurement"] = {
                    "name": "runtime_creation",
                    **runtime_facts,
                }
            emit_status(
                "runtime_ready",
                **runtime_ready,
            )
            chunk_index = 0
            while state.active and runtime.active:
                control_override = None
                incoming = None
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
                        incoming = next_item

                if incoming and incoming.get("type") == "stop":
                    break
                if incoming:
                    control_override = self.pipeline.build_control_override(runtime, incoming)
                if control_override is None:
                    directional_chunk = self._build_directional_control_chunk(state, runtime)
                    if directional_chunk is not None:
                        applied_controls = directional_chunk["controls"]
                        emit_status(
                            "applying_direction_control",
                            index=chunk_index,
                            controls=applied_controls,
                            move_step=state.config.control_move_step,
                            yaw_step_degrees=state.config.control_yaw_step_degrees,
                            lateral_step=state.config.control_lateral_step,
                        )
                        control_override = self.pipeline.build_control_override(runtime, directional_chunk)

                emit_status("generating_chunk", index=chunk_index)
                chunk_measurement = start_measurement()
                frames = self.pipeline.generate_next_chunk(
                    runtime,
                    control_override=control_override,
                    progress_callback=emit_status,
                )
                chunk_facts = finish_measurement(chunk_measurement)
                if not frames:
                    break
                if state.config.show_control_hud and applied_controls:
                    frames = self._overlay_control_hud(frames, applied_controls)
                encode_started_at = time.perf_counter()
                frames_b64 = self.pipeline.encode_frames_to_b64(frames)
                encode_seconds = time.perf_counter() - encode_started_at
                payload = {
                    "type": "chunk",
                    "index": chunk_index,
                    "fps": state.config.fps,
                    "timestamp": time.time(),
                    "frames_b64": frames_b64,
                }
                self._put_output(state, payload)
                chunk_sent: dict[str, object] = {
                    "index": chunk_index,
                    "frames": len(frames),
                }
                if chunk_facts is not None:
                    chunk_sent["measurement"] = {
                        "index": chunk_index,
                        "frames": len(frames),
                        "compute_seconds": chunk_facts["seconds"],
                        "encode_seconds": encode_seconds,
                        "memory": chunk_facts["memory"],
                    }
                emit_status("chunk_sent", **chunk_sent)
                chunk_index += 1
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
            self._release_runtime(state)

    def _runtime_metadata(
        self,
        runtime: LingBotWorldFastRuntimeState,
    ) -> dict[str, int]:
        cache_capacity = int(runtime.self_kv_cache[0]["k"].shape[1])
        return {
            "height": int(runtime.height),
            "width": int(runtime.width),
            "latent_frames": int(runtime.latent_f),
            "frame_tokens": int(runtime.frame_tokens),
            "chunk_size": int(runtime.chunk_size),
            "max_attention_size": int(runtime.max_attention_size),
            "kv_local_attn_size": int(self.pipeline.dit.local_attn_size),
            "kv_sink_size": int(self.pipeline.dit.sink_size),
            "kv_cache_capacity_tokens": cache_capacity,
        }

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            return
        if chunk.get("type") == "control" and self._update_direction_controls(state, chunk):
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

        while state.active:
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
        self._release_runtime(state)
        logger.info(f"LingBotWorld session closed: {session_id}")
