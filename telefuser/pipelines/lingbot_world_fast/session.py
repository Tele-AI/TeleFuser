from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass, field

import torch
from PIL import Image

from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler


@dataclass
class LingBotWorldFastSessionConfig:
    prompt: str
    image: Image.Image
    control_mode: str = "cam"
    fps: int = 16
    chunk_size: int = 3
    frame_num: int = 81
    sample_shift: float = 5.0
    seed: int = 42
    max_attention_size: int | None = None
    offload_model: bool = False
    max_sequence_length: int = 512
    action_path: str | None = None
    poses: object | None = None
    intrinsics: object | None = None
    action: object | None = None
    control_move_step: float = 0.18
    control_yaw_step_degrees: float = 10.0
    control_lateral_step: float = 0.12
    show_control_hud: bool = True


@dataclass
class LingBotWorldFastRuntimeState:
    prompt_emb: torch.Tensor
    encoded_image_latent: torch.Tensor
    noise_chunks: list[torch.Tensor]
    condition_chunks: list[torch.Tensor]
    control_chunks: list[torch.Tensor] | None
    timesteps: torch.Tensor
    self_kv_cache: list[dict[str, torch.Tensor | int]]
    crossattn_cache: list[dict[str, torch.Tensor | bool]]
    latent_h: int
    latent_w: int
    latent_f: int
    height: int
    width: int
    max_seq_len: int
    frame_tokens: int
    chunk_size: int
    max_attention_size: int
    scheduler: FlowUniPCMultistepScheduler
    current_chunk_index: int = 0
    emitted_frames: int = 0
    active: bool = True
    generator: torch.Generator | None = None


@dataclass
class LingBotWorldFastSessionState:
    config: LingBotWorldFastSessionConfig
    runtime: LingBotWorldFastRuntimeState | None = None
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
