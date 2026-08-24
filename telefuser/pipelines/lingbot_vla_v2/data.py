"""RobotWin input preparation for LingBot-VLA v2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import Resize

from .robot_profile import ROBOTWIN_CAMERA_KEYS, RobotWinProfile

ImageInput = Image.Image | np.ndarray | torch.Tensor | str | Path


@dataclass(frozen=True)
class LingBotVlaV2Observation:
    """One RobotWin observation accepted by the public SDK."""

    task: str
    state: torch.Tensor | Sequence[float]
    images: Mapping[str, ImageInput]


@dataclass(frozen=True)
class LingBotVlaV2Inputs:
    """Tensor contract consumed by ``LingBotVlaV2PolicyStage``."""

    images: torch.Tensor
    img_masks: torch.Tensor
    lang_tokens: torch.Tensor
    lang_masks: torch.Tensor
    state: torch.Tensor
    image_grid_thw: torch.Tensor


def _image_to_chw_uint8(image: ImageInput) -> torch.Tensor:
    """Convert one RGB image to the format used by the upstream processor."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = np.asarray(opened.convert("RGB"))
    elif isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(np.asarray(image).copy())
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"unsupported image type: {type(image)!r}")
    image = image.detach().to(device="cpu")
    if image.ndim != 3:
        raise ValueError(f"each image must have three dimensions, got {tuple(image.shape)}")
    if image.shape[0] == 3:
        chw = image
    elif image.shape[-1] == 3:
        chw = image.permute(2, 0, 1)
    else:
        raise ValueError(f"each image must have three RGB channels, got {tuple(image.shape)}")

    if chw.dtype == torch.uint8:
        return chw.contiguous()
    chw = chw.to(dtype=torch.float32)
    if not torch.isfinite(chw).all():
        raise ValueError("images must contain only finite values")
    if chw.numel() and float(chw.max()) <= 2.0 and float(chw.min()) >= 0.0:
        chw = chw * 255.0
    return chw.round().clamp_(0, 255).to(dtype=torch.uint8).contiguous()


class LingBotVlaV2InputProcessor:
    """Prepare RobotWin images, task text, and canonical state tensors."""

    def __init__(
        self,
        processor: Any,
        model_config: Any,
        robot_profile: RobotWinProfile,
        *,
        image_size: int = 256,
    ) -> None:
        if processor is None or not hasattr(processor, "image_processor") or not hasattr(processor, "tokenizer"):
            raise TypeError("LingBot-VLA v2 requires a Qwen3-VL AutoProcessor")
        self.processor = processor
        self.robot_profile = robot_profile
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.image_size = int(image_size)
        self.image_resize = Resize((self.image_size, self.image_size), antialias=True)
        self.max_state_dim = int(getattr(model_config, "max_state_dim", 55))
        self.tokenizer_max_length = int(getattr(model_config, "tokenizer_max_length", 72))
        if self.max_state_dim != robot_profile.canonical_dim:
            raise ValueError(
                f"model max_state_dim is {self.max_state_dim}, RobotWin requires {robot_profile.canonical_dim}"
            )

    def _process_images(self, images: Mapping[str, ImageInput]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        missing = [key for key in ROBOTWIN_CAMERA_KEYS if key not in images]
        if missing:
            raise ValueError(f"RobotWin observation is missing camera keys: {missing}")

        resized_images = [
            self.image_resize(_image_to_chw_uint8(images[key]).to(dtype=torch.float32))
            for key in self.robot_profile.camera_keys
        ]
        output = self.processor.image_processor(resized_images)
        pixels = output["pixel_values"] if isinstance(output, dict) else output.pixel_values
        grid = output.get("image_grid_thw") if isinstance(output, dict) else getattr(output, "image_grid_thw", None)
        pixels = torch.as_tensor(pixels)
        grid = None if grid is None else torch.as_tensor(grid)
        num_images = len(resized_images)
        if grid is None or grid.numel() != num_images * 3:
            raise ValueError(f"Qwen3-VL image processor must return one image_grid_thw row per camera, got {grid}")
        grids = grid.reshape(num_images, 3).to(dtype=torch.long)

        if pixels.ndim == 3:
            if pixels.shape[0] != num_images:
                raise ValueError(
                    f"Qwen3-VL image processor returned {pixels.shape[0]} image batches for {num_images} cameras"
                )
            processed_images = list(pixels.unbind(0))
        elif pixels.ndim == 2:
            patch_counts = grids.prod(dim=-1).tolist()
            if sum(patch_counts) != pixels.shape[0]:
                raise ValueError(
                    "Qwen3-VL image processor pixel count does not match image_grid_thw: "
                    f"pixels={pixels.shape[0]}, expected={sum(patch_counts)}"
                )
            processed_images = list(pixels.split(patch_counts, dim=0))
        else:
            raise ValueError(
                "Qwen3-VL image processor must return packed [patches, features] or "
                f"batched [images, patches, features], got {tuple(pixels.shape)}"
            )

        first_shape = processed_images[0].shape
        if any(image.shape != first_shape for image in processed_images[1:]):
            shapes = [tuple(image.shape) for image in processed_images]
            raise ValueError(f"all RobotWin cameras must produce equal patch shapes, got {shapes}")
        return (
            torch.stack(processed_images, dim=0).unsqueeze(0),
            torch.ones(1, len(processed_images), dtype=torch.bool),
            grids.unsqueeze(0),
        )

    def _process_language(self, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        tokenizer = self.processor.tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": task}],
            tokenize=False,
            add_generation_prompt=False,
        )
        tokens = tokenizer(
            [rendered],
            padding="max_length",
            padding_side="right",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        )
        return tokens["input_ids"], tokens["attention_mask"].to(dtype=torch.bool)

    def prepare(self, observation: LingBotVlaV2Observation) -> LingBotVlaV2Inputs:
        """Prepare one RobotWin observation for model inference."""
        if not isinstance(observation, LingBotVlaV2Observation):
            raise TypeError("observation must be a LingBotVlaV2Observation")
        image_tensors, image_masks, image_grid_thw = self._process_images(observation.images)
        state = self.robot_profile.normalize_state(observation.state).unsqueeze(0)
        lang_tokens, lang_masks = self._process_language(observation.task)
        return LingBotVlaV2Inputs(
            images=image_tensors,
            img_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            image_grid_thw=image_grid_thw,
        )
