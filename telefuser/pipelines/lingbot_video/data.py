"""Data contracts shared by LingBot-Video pipeline stages.

TI2V geometry and structured-caption behavior are adapted from the
Apache-2.0 licensed upstream LingBot-Video implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


def validate_frame_count(num_frames: int) -> int:
    """Validate LingBot's temporal frame contract (1 or ``4n + 1``)."""
    if num_frames < 1 or (num_frames - 1) % 4:
        raise ValueError("LingBot-Video requires num_frames to be 1 or 4n+1")
    return num_frames


_RUNTIME_PROMPT_FIELDS = frozenset({"duration", "fps", "height", "width", "num_frames", "resolution", "ratio"})

# Source-compatible default CFG conditions, adapted from the Apache-2.0
# licensed upstream LingBot-Video pipeline. Keep the strings serialized exactly
# so Qwen3-VL receives the same structured negative caption by default.
DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", '
    '"jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", '
    '"invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", '
    '"drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": '
    '["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image '
    'in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", '
    '"temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static '
    'object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", '
    '"unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
)
DEFAULT_NEGATIVE_PROMPT_IMAGE = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", '
    '"jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden '
    'in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", '
    '"cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", '
    '"logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": '
    '["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'
)


def default_negative_caption(num_frames: int) -> str:
    """Return the source-compatible T2I or video default CFG caption."""
    return DEFAULT_NEGATIVE_PROMPT_IMAGE if num_frames == 1 else DEFAULT_NEGATIVE_PROMPT


def parse_lingbot_video_prompt(payload: object) -> tuple[str, float | None]:
    """Extract the source-compatible caption and optional duration from a prompt sample."""
    if isinstance(payload, list):
        if not payload:
            raise ValueError("LingBot-Video prompt sample list must not be empty")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("LingBot-Video prompt sample must be a dictionary or a non-empty list of dictionaries")
    caption = (
        payload["caption"]
        if "caption" in payload
        else {key: value for key, value in payload.items() if key not in _RUNTIME_PROMPT_FIELDS}
    )
    if isinstance(caption, (dict, list)):
        caption_text = json.dumps(caption, ensure_ascii=False, separators=(",", ":"))
    else:
        caption_text = str(caption)
    if not caption_text:
        raise ValueError("LingBot-Video prompt caption must not be empty")
    duration = payload.get("duration")
    return caption_text, None if duration is None else float(duration)


def load_lingbot_video_prompt(path: str | Path) -> tuple[str, float | None]:
    """Load an upstream rewriter prompt file and return its caption and duration."""
    return parse_lingbot_video_prompt(json.loads(Path(path).read_text(encoding="utf-8")))


def num_frames_from_duration(duration: float, fps: int = 24) -> int:
    """Match upstream duration conversion for a ``4n+1`` LingBot video length."""
    if duration < 0 or fps <= 0:
        raise ValueError("duration must be non-negative and fps must be positive")
    frame_count = int(float(duration) * fps)
    return ((frame_count - 1) // 4 + 1) * 4 + 1


@dataclass(frozen=True)
class LingBotVideoModelConfig:
    """Architecture values encoded by the official Dense/MoE checkpoints."""

    variant: str = "dense"
    num_layers: int = 24
    hidden_size: int = 2048
    num_heads: int = 16
    in_channels: int = 16
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 2560
    intermediate_size: int = 6144
    num_experts: int = 0
    top_k: int = 0

    def __post_init__(self) -> None:
        if self.variant not in {"dense", "moe", "refiner"}:
            raise ValueError(f"Unsupported LingBot-Video variant: {self.variant}")
        if self.num_layers <= 0 or self.hidden_size <= 0 or self.num_heads <= 0:
            raise ValueError("LingBot-Video architecture dimensions must be positive")
        if self.variant in {"moe", "refiner"} and (self.num_experts <= 0 or self.top_k <= 0):
            raise ValueError("MoE and refiner variants require num_experts and top_k")


@dataclass(frozen=True)
class LingBotVideoRequest:
    """Structured request consumed by the numerical pipeline."""

    caption: str
    height: int
    width: int
    num_frames: int
    image: object | None = None

    def __post_init__(self) -> None:
        validate_frame_count(self.num_frames)
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.height % 16 or self.width % 16:
            raise ValueError("LingBot-Video height and width must be divisible by 16")
        if self.image is not None and self.num_frames == 1:
            raise ValueError("TI2V image conditioning requires a video frame count")


def load_lingbot_video_model_config(path: str | Path, *, variant: str = "dense") -> LingBotVideoModelConfig:
    """Load the architecture subset from a Diffusers transformer config.json."""
    config_path = Path(path)
    if config_path.is_dir():
        config_path = config_path / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    required = (
        "depth",
        "hidden_size",
        "num_attention_heads",
        "in_channels",
        "patch_size",
        "text_dim",
        "intermediate_size",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"LingBot transformer config is missing: {missing}")
    return LingBotVideoModelConfig(
        variant=variant,
        num_layers=int(payload["depth"]),
        hidden_size=int(payload["hidden_size"]),
        num_heads=int(payload["num_attention_heads"]),
        in_channels=int(payload["in_channels"]),
        patch_size=tuple(int(v) for v in payload["patch_size"]),
        text_dim=int(payload["text_dim"]),
        intermediate_size=int(payload["intermediate_size"]),
        num_experts=int(payload.get("num_experts", 0)),
        top_k=int(payload.get("num_experts_per_tok", 0)) if int(payload.get("num_experts", 0)) else 0,
    )


def smart_resize(
    height: int, width: int, *, factor: int, min_pixels: int = 4 * 28**2, max_pixels: int = 16384 * 28**2
) -> tuple[int, int]:
    """Match upstream Qwen3-VL image smart-resize geometry."""
    import math

    if min_pixels > max_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels")
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("image dimensions have an unsupported aspect ratio")
    resize_height = max(factor, round(height / factor) * factor)
    resize_width = max(factor, round(width / factor) * factor)
    if resize_height * resize_width > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        resize_height = math.floor(height / scale / factor) * factor
        resize_width = math.floor(width / scale / factor) * factor
    elif resize_height * resize_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resize_height = math.ceil(height * scale / factor) * factor
        resize_width = math.ceil(width * scale / factor) * factor
    return resize_height, resize_width


def preprocess_ti2v_image(image: "torch.Tensor", *, height: int, width: int) -> "torch.Tensor":
    """Resize-short-side then center-crop an RGB image to the condition frame."""
    import math

    import torch.nn.functional as F

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")
    old_height, old_width = image.shape[-2:]
    scale = max(height / old_height, width / old_width)
    resized_height = max(math.ceil(old_height * scale), height)
    resized_width = max(math.ceil(old_width * scale), width)
    resized = F.interpolate(
        image.to(torch.uint8), size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )
    top = int(round((resized_height - height) / 2.0))
    left = int(round((resized_width - width) / 2.0))
    return resized[:, :, top : top + height, left : left + width].float().div(255.0).unsqueeze(2)
