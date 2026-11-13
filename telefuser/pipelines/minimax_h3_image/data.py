# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from Diffusers 843f1743e259e510f22065c0cb917d417b98d9f5.
import math

import numpy as np
import torch

# The spatial rotary axes are normalized by the square root of the latent area and scaled by 32.
_ROPE_SPATIAL_SCALE = 32


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise.patchify_video_latents
def patchify_video_latents(latents: torch.Tensor, patch_size: tuple[int, int, int]) -> torch.Tensor:
    r"""
    Pack video latents into transformer rows.

    Args:
        latents (`torch.Tensor` of shape `(batch_size, channels, num_frames, height, width)`):
            The latents to pack.
        patch_size (`tuple[int, int, int]`): The `(t, h, w)` patch.

    Returns:
        `torch.Tensor` of shape `(batch_size * num_patches, channels * prod(patch_size))`: The packed rows, ordered
        frame-major then row-major.
    """
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"Latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}.")

    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w).contiguous()


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._spatial_position_grid
def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    r"""
    One aspect-normalized spatial rotary axis: `dim // patch` coordinates centred on the unit interval, scaled up by
    32. The right endpoint is excluded, so a square canvas spans `[0, 32)`.
    """
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    # Built with numpy: `np.linspace(..., endpoint=False)` is `start + arange(num) * (stop - start) / num`, which is
    # not what `torch.linspace` computes, and the float64 grid has to be reproduced exactly.
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


# Copied from diffusers.modular_pipelines.minimax_h3.before_denoise._frame_position_grid
def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""The `(h, w)` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def get_image_canvas_size(height: int, width: int, min_generation_area: int, multiple: int) -> tuple[int, int]:
    r"""
    The canvas MiniMax-H3-Image generates a `height x width` image on.

    An image below `min_generation_area` pixels is generated larger at the same aspect ratio: both sides are scaled by
    `sqrt(min_generation_area / (height * width))`. Either way both sides round to the nearest `multiple`.

    Returns:
        `tuple[int, int]`: the `(height, width)` of the canvas.
    """
    scale = max(1.0, math.sqrt(min_generation_area / (width * height)))
    return (
        max(multiple, round(height * scale / multiple) * multiple),
        max(multiple, round(width * scale / multiple) * multiple),
    )


def build_image_packed_sequence(
    text_token_tags: torch.Tensor,
    reference_grids: list[tuple[int, int]],
    latent_height: int,
    latent_width: int,
    patch_size: tuple[int, int, int],
    image_tag: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    r"""
    Build the `[text | reference images | generated image]` layout of MiniMax-H3-Image.

    Args:
        text_token_tags (`torch.Tensor` of shape `(num_text_tokens,)`):
            The modality tag of every text row. Text is tagged `1`, except for the rows of a reference's vision block,
            which are tagged `0` (image).
        reference_grids (`list[tuple[int, int]]`):
            The `(latent_height, latent_width)` of every reference image, in packed order. Empty for `text2image`.
        latent_height (`int`): Latent height of the generated image.
        latent_width (`int`): Latent width of the generated image.
        patch_size (`tuple[int, int, int]`): The transformer's `(t, h, w)` patch.
        image_tag (`int`): The modality tag an image row carries, i.e. `components.image_tag`.

    Returns:
        `tuple`: `position_ids`, `token_tags`, `image_indices`, `text_indices`, and the number of leading image rows
        that are reference rows rather than generated ones.
    """
    _, patch_h, patch_w = patch_size
    num_text_tokens = text_token_tags.shape[0]
    num_reference_rows = sum((height // patch_h) * (width // patch_w) for height, width in reference_grids)
    num_target_rows = (latent_height // patch_h) * (latent_width // patch_w)
    sequence_length = num_text_tokens + num_reference_rows + num_target_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    # Every image is one frame on the rotary clock, which starts where the text rows end: a reference takes one
    # integer slot each, and the generated image takes the slot after the last reference.
    cursor = num_text_tokens
    rotary_time = float(num_text_tokens)
    for height, width in reference_grids:
        rows = slice(cursor, cursor + (height // patch_h) * (width // patch_w))
        cursor = rows.stop
        frame_grid, _ = _frame_position_grid(height, width, patch_h, patch_w)
        position_ids[rows, 0] = rotary_time
        position_ids[rows, 1:] = frame_grid
        rotary_time += 1.0
    frame_grid, _ = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    position_ids[cursor:, 0] = rotary_time
    position_ids[cursor:, 1:] = frame_grid

    text_indices = torch.arange(num_text_tokens)
    image_indices = torch.arange(num_text_tokens, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[image_indices] = image_tag

    return position_ids, token_tags, image_indices, text_indices, num_reference_rows


def bound_short_edge(
    width: int, height: int, min_short_edge: int, max_short_edge: int, multiple: int
) -> tuple[int, int]:
    r"""
    The size a MiniMax-H3-Image reference is resized to for one of its consumers.

    A short edge within `[min_short_edge, max_short_edge]` keeps the original size. Otherwise the image is scaled,
    keeping its ratio, so that the short edge lands on the nearer bound, and both sides round to `multiple`.

    Returns:
        `tuple[int, int]`: the `(width, height)` to resize to.
    """
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )
