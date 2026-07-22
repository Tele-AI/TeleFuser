"""Qwen3-VL prompt encoding for LingBot-Video.

Prompt templates and Qwen3-VL image preparation behavior are adapted from the
Apache-2.0 licensed upstream LingBot-Video implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

from .data import smart_resize

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    'or a video reference alone, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMAGE_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
_VISION_SPATIAL_MERGE_SIZE = 2
_VISION_MIN_TOKEN_COUNT = 4
_VISION_MAX_TOKEN_COUNT = 16384


class LingBotVideoTextEncodingStage(BaseStage):
    """Encode structured captions and optional TI2V images with Qwen3-VL."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        *,
        token_length: int = 37698,
        hidden_state_skip_layer: int | None = 0,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.text_encoder = module_manager.fetch_module("text_encoder")
        self.processor = module_manager.fetch_module("processor")
        self.token_length = token_length
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self._crop_start: int | None = None
        self.model_names = ["text_encoder"]

    def _crop_prefix_length(self) -> int:
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            prefix = PROMPT_TEMPLATE.format(marker).split(marker, maxsplit=1)[0]
            inputs = self.processor(text=prefix, images=None, videos=None, return_tensors="pt")
            self._crop_start = int(inputs["input_ids"].shape[1])
        return self._crop_start

    def _vision_patch_size(self) -> int:
        """Return the Qwen vision patch size using the source lookup order."""
        for obj in (
            getattr(getattr(self.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(self.processor, "image_processor", None), "config", None),
            getattr(self.processor, "image_processor", None),
        ):
            patch_size = getattr(obj, "patch_size", None)
            if patch_size is not None:
                return int(patch_size)
        return 16

    def prepare_ti2v_vlm_image(self, pixel_values: torch.Tensor) -> Image.Image:
        """Convert a condition frame to the source-equivalent Qwen3-VL image input."""
        if (
            pixel_values.ndim != 5
            or pixel_values.shape[0] != 1
            or pixel_values.shape[1] != 3
            or pixel_values.shape[2] < 1
        ):
            raise ValueError("TI2V pixels must have shape [1,3,F,H,W]")
        frame = pixel_values[0, :, 0].detach().cpu().clamp(0.0, 1.0)
        image = Image.fromarray(frame.permute(1, 2, 0).mul(255).byte().numpy(), mode="RGB")
        patch_factor = self._vision_patch_size() * _VISION_SPATIAL_MERGE_SIZE
        width, height = image.size
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=patch_factor,
            min_pixels=_VISION_MIN_TOKEN_COUNT * patch_factor**2,
            max_pixels=_VISION_MAX_TOKEN_COUNT * patch_factor**2,
        )
        return image.resize((resized_width, resized_height))

    @with_model_offload(["text_encoder"])
    @torch.no_grad()
    def encode(
        self,
        prompt: str | list[str],
        *,
        images: Any | None = None,
        video_metadata: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cropped prompt embeddings and their attention mask."""
        prompts = [prompt] if isinstance(prompt, str) else prompt
        visual_prefix = IMAGE_PROMPT_TEMPLATE if images is not None else ""
        texts = [PROMPT_TEMPLATE.format(visual_prefix + item) for item in prompts]
        inputs = self.processor(
            text=texts,
            images=images,
            videos=None,
            video_metadata=video_metadata,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
        ).to(self.device)
        output = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        embeddings = (
            output.hidden_states[-(self.hidden_state_skip_layer + 1)]
            if self.hidden_state_skip_layer is not None
            else output.last_hidden_state
        )
        mask = inputs["attention_mask"]
        crop_start = self._crop_prefix_length()
        embeddings, mask = embeddings[:, crop_start:], mask[:, crop_start:]
        if embeddings.shape[0] == 1:
            length = int(mask[0].sum().item())
            embeddings, mask = embeddings[:, :length], mask[:, :length]
        return embeddings, mask
