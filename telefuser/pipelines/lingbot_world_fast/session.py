from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
from PIL import Image

from telefuser.models.wan_video_vae import WanVideoVAEStreamingDecodeState

from .control import LingBotWorldFastControlContext


@dataclass
class LingBotWorldFastSessionConfig:
    prompt: str
    image: Image.Image
    control_mode: str = "cam"
    fps: int = 16
    chunk_size: int = 3
    frame_num: int = 81
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
class LingBotWorldFastGenerationSession:
    """Externally owned state for one chunked LingBot generation."""

    config: LingBotWorldFastSessionConfig
    prompt_emb: torch.Tensor | None = field(default=None, repr=False)
    noise_chunks: list[torch.Tensor] = field(default_factory=list, repr=False)
    condition_chunks: list[torch.Tensor] = field(default_factory=list, repr=False)
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
    active: bool = True
    status: LingBotWorldFastSessionStatus = LingBotWorldFastSessionStatus.NEW
    poisoned_reason: str | None = None
    transaction_lock: object = field(default_factory=threading.RLock, repr=False)
    # CacheSeek world_kv binding and decode-only latents for fast-forward hits.
    world_kv_binding: object | None = None
    world_kv_cached_latents: dict[int, torch.Tensor] = field(default_factory=dict)


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
