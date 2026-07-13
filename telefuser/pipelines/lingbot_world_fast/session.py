from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field
from enum import Enum

import torch
from PIL import Image

from telefuser.models.wan_video_vae import WanVideoVAEStreamingDecodeState


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
    offload_model: bool = False
    max_sequence_length: int = 512
    action_path: str | None = None
    poses: object | None = None
    intrinsics: object | None = None
    action: object | None = None
    # Optional CacheSeek world_kv reuse. None preserves baseline behavior.
    world_kv_binding: object | None = None
    control_move_step: float = 0.18
    control_yaw_step_degrees: float = 10.0
    control_lateral_step: float = 0.12
    show_control_hud: bool = True


@dataclass
class LingBotWorldFastChunkRequest:
    """Inputs for one explicitly indexed LingBot video chunk."""

    chunk_index: int
    session_id: str | None = None
    action: dict[str, object] | None = field(default=None, repr=False)
    control_override: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if self.action is not None and self.control_override is not None:
            raise ValueError("Provide either action or control_override, not both")
        if self.action is None and self.control_override is None:
            raise ValueError("Each chunk request requires action or control_override")


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
    COMMITTED = "committed"
    POISONED = "poisoned"
    RELEASED = "released"


@dataclass
class LingBotWorldFastGenerationSession:
    """Externally owned state for one chunked LingBot generation."""

    prompt_emb: torch.Tensor
    encoded_image_latent: torch.Tensor
    noise_chunks: list[torch.Tensor]
    condition_chunks: list[torch.Tensor]
    control_chunks: list[torch.Tensor] | None
    latent_h: int
    latent_w: int
    latent_f: int
    height: int
    width: int
    max_seq_len: int
    frame_tokens: int
    chunk_size: int
    max_attention_size: int
    cache_handle: int | None
    decoder_state: WanVideoVAEStreamingDecodeState = field(default_factory=WanVideoVAEStreamingDecodeState)
    current_chunk_index: int = 0
    emitted_frames: int = 0
    active: bool = True
    status: LingBotWorldFastSessionStatus = LingBotWorldFastSessionStatus.READY
    poisoned_reason: str | None = None
    transaction_lock: object = field(default_factory=threading.RLock, repr=False)
    # KV geometry in latent frames; -1 means full-length KV.
    kv_local_attn_size: int = -1
    kv_sink_size: int = 0
    # CacheSeek world_kv binding and decode-only latents for fast-forward hits.
    world_kv_binding: object | None = None
    world_kv_cached_latents: dict[int, torch.Tensor] = field(default_factory=dict)


# Compatibility alias for callers migrating from the Stage 1 API.
LingBotWorldFastRuntimeState = LingBotWorldFastGenerationSession


@dataclass
class LingBotWorldFastSessionState:
    config: LingBotWorldFastSessionConfig
    generation_session: LingBotWorldFastGenerationSession | None = None
    pending_inputs: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    output_queue: asyncio.Queue | None = None
    worker_thread: threading.Thread | None = None
    loop: asyncio.AbstractEventLoop | None = None
    active: bool = True
    pressed_controls: set[str] = field(default_factory=set)
    queued_controls: set[str] = field(default_factory=set)
    control_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    control_yaw: float = 0.0
    control_lock: object = field(default_factory=threading.Lock)
