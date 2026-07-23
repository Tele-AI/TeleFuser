from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from PIL import Image

from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from .pipeline import LingBotWorldFastPipeline


@dataclass
class AsyncVAEChunkHandle:
    session_id: str | None
    generation_id: int
    chunk_id: int
    is_last_clip: bool
    enqueue_ns: int
    denoise_profile: dict[str, object] = field(default_factory=dict)
    queue_wait_ms: float = 0.0
    queue_depth_after_enqueue: int = 0
    frames: list[Image.Image] | None = field(default=None, repr=False)
    exception: BaseException | None = field(default=None, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    vae_start_ns: int | None = None
    vae_end_ns: int | None = None
    output_ready_ns: int | None = None
    vae_gpu_ms: float | None = None
    overlap_ms: float | None = None
    canceled: bool = False


@dataclass
class AsyncVAEDecodeTask:
    session_id: str | None
    generation_id: int
    chunk_id: int
    latent: torch.Tensor = field(repr=False)
    latent_ready_event: torch.cuda.Event | None = field(repr=False)
    is_first_clip: bool
    is_last_clip: bool
    handle: AsyncVAEChunkHandle = field(repr=False)


class AsyncVAEManager:
    """Single-process, single-VAE async decoder for LingBot streaming."""

    _SENTINEL = object()

    def __init__(self, pipeline: LingBotWorldFastPipeline, queue_size: int) -> None:
        self.pipeline = pipeline
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[AsyncVAEDecodeTask | object] = queue.Queue(maxsize=self.queue_size)
        self._lock = threading.Condition()
        self._cancelled_generations: set[int] = set()
        self._next_chunk_by_generation: dict[int, int] = {}
        self._active_generation_id: int | None = None
        self._exception: BaseException | None = None
        self._closed = False
        self._stream: torch.cuda.Stream | None = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="lingbot-async-vae",
        )
        self._worker.start()

    def _is_cancelled(self, generation_id: int) -> bool:
        with self._lock:
            return generation_id in self._cancelled_generations

    def _mark_failed(self, exc: BaseException) -> None:
        with self._lock:
            if self._exception is None:
                self._exception = exc
            self._lock.notify_all()

    def raise_if_failed(self) -> None:
        with self._lock:
            if self._exception is not None:
                raise RuntimeError(f"LingBot async VAE worker failed: {self._exception}") from self._exception
            if self._closed:
                raise RuntimeError("LingBot async VAE worker is closed")

    def enqueue(self, task: AsyncVAEDecodeTask) -> AsyncVAEChunkHandle:
        start_ns = time.perf_counter_ns()
        while True:
            self.raise_if_failed()
            try:
                self._queue.put(task, timeout=0.1)
                break
            except queue.Full:
                continue
        task.handle.queue_wait_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        task.handle.queue_depth_after_enqueue = self._safe_qsize()
        logger.info(
            "lingbot_async_vae vae_enqueue "
            f"session_id={task.session_id} generation_id={task.generation_id} chunk_id={task.chunk_id} "
            f"queue_wait_ms={task.handle.queue_wait_ms:.3f} "
            f"queue_depth={task.handle.queue_depth_after_enqueue}/{self.queue_size}"
        )
        return task.handle

    def cancel_generation(self, generation_id: int, timeout: float = 30.0) -> None:
        with self._lock:
            self._cancelled_generations.add(generation_id)
            self._next_chunk_by_generation.pop(generation_id, None)
            self._lock.notify_all()

        kept: list[AsyncVAEDecodeTask | object] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, AsyncVAEDecodeTask) and item.generation_id == generation_id:
                self._cancel_handle(item.handle)
            else:
                kept.append(item)
        for item in kept:
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                logger.warning("LingBot async VAE queue refilled while cancelling old generation")
                break

        deadline = time.perf_counter() + timeout
        with self._lock:
            while self._active_generation_id == generation_id and self._exception is None:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    logger.warning(f"LingBot async VAE cancel timed out for generation_id={generation_id}")
                    break
                self._lock.wait(timeout=remaining)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._lock.notify_all()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            try:
                item = self._queue.get_nowait()
                if isinstance(item, AsyncVAEDecodeTask):
                    self._cancel_handle(item.handle)
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(self._SENTINEL)
            except queue.Full:
                logger.warning("LingBot async VAE worker close could not enqueue sentinel")
        self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            logger.warning("LingBot async VAE worker did not exit within timeout")

    def _cancel_handle(self, handle: AsyncVAEChunkHandle) -> None:
        handle.canceled = True
        handle.frames = []
        handle.output_ready_ns = time.perf_counter_ns()
        handle.done.set()

    def _safe_qsize(self) -> int:
        try:
            return self._queue.qsize()
        except NotImplementedError:
            return -1

    def _get_stream(self) -> torch.cuda.Stream | None:
        device = torch.device(self.pipeline.async_vae_device or self.pipeline.vae_device)
        if device.type != "cuda" or not torch.cuda.is_available():
            return None
        if self._stream is not None:
            return self._stream
        with torch.cuda.device(device):
            least_priority = 0
            greatest_priority = 0
            try:
                least_priority, greatest_priority = torch.cuda.Stream.priority_range()
            except Exception:
                try:
                    least_priority, greatest_priority = torch.cuda.priority_range()
                except Exception:
                    pass
            self._stream = torch.cuda.Stream(device=device, priority=least_priority)
            logger.info(
                "lingbot_async_vae stream_created "
                f"device={device} least_priority={least_priority} greatest_priority={greatest_priority} "
                f"selected_priority={least_priority} selected=least_priority"
            )
        return self._stream

    def _decode_task(self, task: AsyncVAEDecodeTask) -> None:
        handle = task.handle
        if self._is_cancelled(task.generation_id):
            self._cancel_handle(handle)
            return
        with self._lock:
            expected_chunk = self._next_chunk_by_generation.get(task.generation_id, task.chunk_id)
            if task.chunk_id != expected_chunk:
                raise RuntimeError(
                    f"Async VAE chunk order violation for generation_id={task.generation_id}: "
                    f"expected {expected_chunk}, got {task.chunk_id}"
                )
            self._next_chunk_by_generation[task.generation_id] = task.chunk_id + 1

        device = torch.device(self.pipeline.async_vae_device or self.pipeline.vae_device)
        handle.vae_start_ns = time.perf_counter_ns()
        if device.type == "cuda" and torch.cuda.is_available():
            with torch.cuda.device(device):
                stream = self._get_stream()
                assert stream is not None
                start_event = torch.cuda.Event(enable_timing=True)
                done_event = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    if task.latent_ready_event is not None:
                        stream.wait_event(task.latent_ready_event)
                    start_event.record(stream)
                    frames_tensor = self.pipeline.decode_video_cached_async(
                        self.pipeline._async_vae_runtime(task.generation_id),
                        task.latent,
                        is_first_clip=task.is_first_clip,
                        is_last_clip=task.is_last_clip,
                    )
                    done_event.record(stream)
                done_event.synchronize()
                handle.vae_gpu_ms = start_event.elapsed_time(done_event)
        else:
            frames_tensor = self.pipeline.decode_video_cached_async(
                self.pipeline._async_vae_runtime(task.generation_id),
                task.latent,
                is_first_clip=task.is_first_clip,
                is_last_clip=task.is_last_clip,
            )

        if self._is_cancelled(task.generation_id):
            self._cancel_handle(handle)
            return

        images = self.pipeline.tensor2video(frames_tensor)
        handle.vae_end_ns = time.perf_counter_ns()
        handle.output_ready_ns = handle.vae_end_ns
        handle.frames = images
        dit_ms = float(handle.denoise_profile.get("dit_total_ms", 0.0) or 0.0)
        vae_ms = (handle.vae_end_ns - handle.vae_start_ns) / 1_000_000.0 if handle.vae_start_ns else 0.0
        if dit_ms > 0 and vae_ms > 0:
            overlap_start = max(handle.enqueue_ns, handle.vae_start_ns or handle.enqueue_ns)
            overlap_end = min(handle.output_ready_ns, handle.enqueue_ns + int(dit_ms * 1_000_000))
            handle.overlap_ms = max(0.0, (overlap_end - overlap_start) / 1_000_000.0)
        handle.done.set()

        memory = self._memory_snapshot(device)
        extra_vae_model_copy = bool(getattr(self.pipeline, "has_separate_async_vae", False))
        logger.info(
            "lingbot_async_vae output_ready "
            f"session_id={task.session_id} generation_id={task.generation_id} chunk_id={task.chunk_id} "
            f"frames={len(images)} vae_ms={vae_ms:.3f} vae_gpu_ms={handle.vae_gpu_ms} "
            f"vae_device={device} "
            f"queue_depth={self._safe_qsize()}/{self.queue_size} overlap_ms={handle.overlap_ms} "
            f"vae_peak_allocated={memory['peak_allocated']} vae_peak_reserved={memory['peak_reserved']} "
            f"vae_allocated={memory['allocated']} vae_reserved={memory['reserved']} "
            f"vae_activation_peak_bytes={memory['vae_activation_peak_bytes']} "
            f"extra_vae_model_copy={extra_vae_model_copy}"
        )

    @staticmethod
    def _memory_snapshot(device: torch.device) -> dict[str, int]:
        if device.type != "cuda" or not torch.cuda.is_available():
            return {
                "allocated": 0,
                "reserved": 0,
                "peak_allocated": 0,
                "peak_reserved": 0,
                "vae_activation_peak_bytes": 0,
            }
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        return {
            "allocated": int(allocated),
            "reserved": int(reserved),
            "peak_allocated": int(peak_allocated),
            "peak_reserved": int(peak_reserved),
            "vae_activation_peak_bytes": int(max(0, peak_allocated - allocated)),
        }

    def _worker_loop(self) -> None:
        logger.info("LingBot async VAE worker started")
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            if not isinstance(item, AsyncVAEDecodeTask):
                continue
            with self._lock:
                self._active_generation_id = item.generation_id
                self._lock.notify_all()
            try:
                self._decode_task(item)
            except BaseException as exc:
                item.handle.exception = exc
                item.handle.done.set()
                self._mark_failed(exc)
                logger.exception(
                    "LingBot async VAE worker failed: "
                    f"session_id={item.session_id} generation_id={item.generation_id} chunk_id={item.chunk_id}"
                )
            finally:
                item.latent = torch.empty(0)
                with self._lock:
                    self._active_generation_id = None
                    self._lock.notify_all()
        logger.info("LingBot async VAE worker stopped")
