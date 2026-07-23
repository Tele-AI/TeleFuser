from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from PIL import Image

from telefuser.models.wan_video_vae import (
    WanVideoVAEStreamingDecodeState,
    WanVideoVAEStreamingEncodeState,
)

from .control import LingBotWorldFastControlContext


def resolve_lingbot_frame_count(frame_num: int, chunk_size: int, frame_policy: str) -> tuple[int, int]:
    """Return the effective video and latent frame counts for a session."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
    if not isinstance(frame_num, int) or isinstance(frame_num, bool):
        raise ValueError(f"frame_num must be an integer, got {frame_num!r}")
    if frame_num < 1 or (frame_num - 1) % 4:
        raise ValueError(f"frame_num must be 4n+1, got {frame_num}")
    if frame_policy not in {"strict", "truncate"}:
        raise ValueError(f"frame_policy must be 'strict' or 'truncate', got {frame_policy!r}")

    latent_frames = (frame_num - 1) // 4 + 1
    if frame_policy == "truncate":
        latent_frames -= latent_frames % chunk_size
    if latent_frames < chunk_size or latent_frames % chunk_size:
        raise ValueError(f"frame_num {frame_num} does not contain a whole number of latent chunks of size {chunk_size}")
    effective_frame_num = 4 * (latent_frames - 1) + 1
    return effective_frame_num, latent_frames


@dataclass
class LingBotWorldFastSessionConfig:
    prompt: str
    image: Image.Image
    control_mode: str = "cam"
    fps: int = 16
    chunk_size: int = 3
    frame_num: int = 81
    frame_policy: str = "truncate"
    sample_shift: float = 10.0
    seed: int = 42
    max_attention_size: int | None = None
    max_sequence_length: int = 512
    # Optional CacheSeek world_kv reuse. None preserves baseline behavior.
    world_kv_binding: object | None = None
    intrinsics: object | None = None
    control_move_step: float = 0.05
    control_yaw_step_degrees: float = 2.0
    control_lateral_step: float = 0.05
    control_pitch_step_degrees: float = 2.0
    control_pitch_limit_degrees: float = 85.0
    show_control_hud: bool = True


@dataclass
class LingBotWorldFastChunkRequest:
    """Inputs for one explicitly indexed LingBot video chunk."""

    chunk_index: int
    control: torch.Tensor | Callable[[], torch.Tensor] = field(repr=False)
    session_id: str | None = None

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if not isinstance(self.control, torch.Tensor) and not callable(self.control):
            raise TypeError("Each chunk request requires a model control tensor or deferred control factory")


@dataclass
class LingBotWorldFastChunkResult:
    """Output and progress metadata for one generated LingBot chunk."""

    chunk_index: int
    frames: list[Image.Image] = field(repr=False)
    emitted_frames: int = 0
    done: bool = False
    session_id: str | None = None


class LingBotWorldFastSessionStatus(str, Enum):
    """Transaction and lifecycle state for one generation session."""

    READY = "ready"
    RUNNING = "running"
    NEW = "new"
    COMMITTED = "committed"
    POISONED = "poisoned"
    RELEASED = "released"


@dataclass
class LingBotWorldFastConditionPrefetch:
    """One in-flight condition chunk encoded ahead of its DiT use."""

    chunk_index: int
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    condition_chunk: torch.Tensor | None = field(default=None, repr=False)
    condition_encode_ms: float = 0.0
    start_ns: int | None = None
    end_ns: int | None = None
    exception: BaseException | None = field(default=None, repr=False)


@dataclass
class LingBotWorldFastGenerationSession:
    """Externally owned state for one chunked LingBot generation."""

    config: LingBotWorldFastSessionConfig
    prompt_emb: torch.Tensor | None = field(default=None, repr=False)
    condition_image: torch.Tensor | None = field(default=None, repr=False)
    noise_generator: torch.Generator | None = field(default=None, repr=False)
    encoder_state: WanVideoVAEStreamingEncodeState = field(default_factory=WanVideoVAEStreamingEncodeState)
    latent_h: int = 0
    latent_w: int = 0
    latent_f: int = 0
    height: int = 0
    width: int = 0
    frame_tokens: int = 0
    chunk_size: int = 0
    max_attention_size: int = 0
    cache_handle: int | None = None
    decoder_state: WanVideoVAEStreamingDecodeState = field(default_factory=WanVideoVAEStreamingDecodeState)
    current_chunk_index: int = 0
    emitted_frames: int = 0
    status: LingBotWorldFastSessionStatus = LingBotWorldFastSessionStatus.NEW
    poisoned_reason: str | None = None
    transaction_lock: object = field(default_factory=threading.RLock, repr=False)
    # CacheSeek world_kv binding and decode-only latents for fast-forward hits.
    world_kv_binding: object | None = None
    world_kv_cached_latents: dict[int, torch.Tensor] = field(default_factory=dict)
    async_vae_generation_id: int | None = None
    condition_prefetch: LingBotWorldFastConditionPrefetch | None = field(default=None, repr=False)

    @property
    def chunk_count(self) -> int:
        if self.chunk_size < 1:
            raise RuntimeError("LingBot generation session has not been initialized")
        return self.latent_f // self.chunk_size


@dataclass
class LingBotWorldFastSessionState:
    config: LingBotWorldFastSessionConfig
    control_context: LingBotWorldFastControlContext | None = None
    generation_session: LingBotWorldFastGenerationSession | None = None
    pending_inputs: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    output_queue: asyncio.Queue | None = None
    worker_thread: threading.Thread | None = None
    loop: asyncio.AbstractEventLoop | None = None
    active: bool = True
    pressed_controls: set[str] = field(default_factory=set)
    queued_controls: set[str] = field(default_factory=set)
    control_c2w: list[list[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    control_pitch: float = 0.0
    control_initialized: bool = False
    control_lock: object = field(default_factory=threading.Lock)
