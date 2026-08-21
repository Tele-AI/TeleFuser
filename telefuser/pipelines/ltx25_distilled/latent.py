from __future__ import annotations

import math

"""Isolated LTX-2.5 latent patchification and conditioning helpers."""

from dataclasses import dataclass, field, replace
from typing import Any, Callable, NamedTuple, Protocol

import einops
import torch
from torch._prims_common import DeviceLikeType

from telefuser.models.ltx25.diff_vae.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoLatentShape

STAGE_2_DISTILLED_SIGMA_VALUES = [0.909375, 0.725, 0.421875, 0.0]
VIDEO_LATENT_CHANNELS = 128


class AudioLatentShape(NamedTuple):
    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_torch_shape(self) -> torch.Size:
        return torch.Size([self.batch, self.channels, self.frames, self.mel_bins])

    def token_count(self) -> int:
        return self.frames

    def mask_shape(self) -> AudioLatentShape:
        return AudioLatentShape(self.batch, 1, self.frames, 1)

    @staticmethod
    def from_duration(
        batch: int,
        duration: float,
        channels: int = 8,
        mel_bins: int = 16,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
    ) -> AudioLatentShape:
        latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
        return AudioLatentShape(batch, channels, round(duration * latents_per_second), mel_bins)


@dataclass(frozen=True)
class LatentState:
    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None

    def clone(self) -> LatentState:
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            keyframes_mask=self.keyframes_mask.clone() if self.keyframes_mask is not None else None,
        )


class Patchifier(Protocol):
    def patchify(self, latents: torch.Tensor) -> torch.Tensor: ...

    def unpatchify(self, latents: torch.Tensor, output_shape: AudioLatentShape | VideoLatentShape) -> torch.Tensor: ...

    def get_token_count(self, target_shape: AudioLatentShape | VideoLatentShape) -> int: ...

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor: ...


class VideoLatentPatchifier:
    def __init__(self, patch_size: int):
        self.patch_size = (1, patch_size, patch_size)

    def get_token_count(self, target_shape: VideoLatentShape) -> int:
        return target_shape.frames * target_shape.height * target_shape.width // math.prod(self.patch_size)

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )

    def unpatchify(self, latents: torch.Tensor, output_shape: VideoLatentShape) -> torch.Tensor:
        return einops.rearrange(
            latents,
            "b (f h w) (c p q) -> b c f (h p) (w q)",
            f=output_shape.frames // self.patch_size[0],
            h=output_shape.height // self.patch_size[1],
            w=output_shape.width // self.patch_size[2],
            p=self.patch_size[1],
            q=self.patch_size[2],
        )

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if not isinstance(output_shape, VideoLatentShape):
            raise ValueError("VideoLatentPatchifier expects VideoLatentShape when computing coordinates")
        grid_coords = torch.meshgrid(
            torch.arange(start=0, end=output_shape.frames, step=self.patch_size[0], device=device),
            torch.arange(start=0, end=output_shape.height, step=self.patch_size[1], device=device),
            torch.arange(start=0, end=output_shape.width, step=self.patch_size[2], device=device),
            indexing="ij",
        )
        patch_starts = torch.stack(grid_coords, dim=0)
        patch_size_delta = torch.tensor(
            self.patch_size,
            device=patch_starts.device,
            dtype=patch_starts.dtype,
        ).view(3, 1, 1, 1)
        patch_ends = patch_starts + patch_size_delta
        latent_coords = torch.stack((patch_starts, patch_ends), dim=-1)
        return einops.repeat(latent_coords, "c f h w bounds -> b c (f h w) bounds", b=output_shape.batch, bounds=2)


class AudioPatchifier:
    def __init__(self, patch_size: int):
        # Keep the same latent layout as upstream LTX:
        # audio latents are shaped (B, C, T, F) and patchify flattens along time -> (B, T, C*F).
        # Positions encode real time in seconds so RoPE max_pos can be expressed in seconds.
        self.patch_size = (patch_size, 1, 1)
        self.sample_rate = 16000
        self.hop_length = 160
        self.audio_latent_downsample_factor = 4
        self.is_causal = True
        self.shift = 0

    def get_token_count(self, target_shape: AudioLatentShape) -> int:
        return target_shape.frames // self.patch_size[0]

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(latents, "b c (f p) m -> b f (c p m)", p=self.patch_size[0])

    def unpatchify(self, latents: torch.Tensor, output_shape: AudioLatentShape) -> torch.Tensor:
        return einops.rearrange(
            latents,
            "b f (c p m) -> b c (f p) m",
            c=output_shape.channels,
            p=self.patch_size[0],
            m=output_shape.mel_bins,
        )

    def get_patch_grid_bounds(
        self,
        output_shape: AudioLatentShape | VideoLatentShape,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if isinstance(output_shape, VideoLatentShape):
            raise ValueError("AudioPatchifier expects AudioLatentShape when computing coordinates")
        if device is None:
            device = torch.device("cpu")

        start_latent = self.shift
        end_latent = output_shape.frames + self.shift
        audio_latent_frame_start = torch.arange(start_latent, end_latent, dtype=torch.float32, device=device)
        audio_latent_frame_end = torch.arange(start_latent + 1, end_latent + 1, dtype=torch.float32, device=device)

        downsample = float(self.audio_latent_downsample_factor)
        audio_mel_frame_start = audio_latent_frame_start * downsample
        audio_mel_frame_end = audio_latent_frame_end * downsample

        if self.is_causal:
            causal_offset = 1.0
            audio_mel_frame_start = (audio_mel_frame_start + causal_offset - downsample).clamp_min(0.0)
            audio_mel_frame_end = (audio_mel_frame_end + causal_offset - downsample).clamp_min(0.0)

        start_timings = audio_mel_frame_start * float(self.hop_length) / float(self.sample_rate)
        end_timings = audio_mel_frame_end * float(self.hop_length) / float(self.sample_rate)

        start_timings = start_timings.unsqueeze(0).expand(output_shape.batch, -1).unsqueeze(1)
        end_timings = end_timings.unsqueeze(0).expand(output_shape.batch, -1).unsqueeze(1)
        return torch.stack([start_timings, end_timings], dim=-1)


def get_pixel_coords(
    latent_coords: torch.Tensor,
    scale_factors: SpatioTemporalScaleFactors,
    causal_fix: bool = False,
) -> torch.Tensor:
    pixel_coords = latent_coords.clone()
    pixel_coords[:, 0] *= scale_factors.time
    pixel_coords[:, 1] *= scale_factors.height
    pixel_coords[:, 2] *= scale_factors.width
    if causal_fix:
        pixel_coords[:, 0, :, :] -= scale_factors.time - 1
        pixel_coords[:, 0, :, :] = pixel_coords[:, 0, :, :].clamp_min(0)
    return pixel_coords


@dataclass(frozen=True)
class LatentTools:
    patchifier: Patchifier
    target_shape: AudioLatentShape | VideoLatentShape

    def patchify(self, latent_state: LatentState) -> LatentState:
        latent_state = latent_state.clone()
        return replace(
            latent_state,
            latent=self.patchifier.patchify(latent_state.latent),
            denoise_mask=self.patchifier.patchify(latent_state.denoise_mask),
            clean_latent=self.patchifier.patchify(latent_state.clean_latent),
            keyframes_mask=(
                self.patchifier.patchify(latent_state.keyframes_mask)
                if latent_state.keyframes_mask is not None
                else None
            ),
        )

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        latent_state = latent_state.clone()
        return replace(
            latent_state,
            latent=self.patchifier.unpatchify(latent_state.latent, output_shape=self.target_shape),
            denoise_mask=self.patchifier.unpatchify(
                latent_state.denoise_mask,
                output_shape=self.target_shape.mask_shape(),
            ),
            clean_latent=self.patchifier.unpatchify(latent_state.clean_latent, output_shape=self.target_shape),
            keyframes_mask=(
                self.patchifier.unpatchify(latent_state.keyframes_mask, output_shape=self.target_shape.mask_shape())
                if latent_state.keyframes_mask is not None
                else None
            ),
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        num_tokens = self.patchifier.get_token_count(self.target_shape)
        return LatentState(
            latent=latent_state.latent[:, :num_tokens],
            denoise_mask=torch.ones_like(latent_state.denoise_mask)[:, :num_tokens],
            positions=latent_state.positions[:, :, :num_tokens],
            clean_latent=latent_state.clean_latent[:, :num_tokens],
            attention_mask=None,
            keyframes_mask=(
                latent_state.keyframes_mask[:, :num_tokens] if latent_state.keyframes_mask is not None else None
            ),
        )


@dataclass(frozen=True)
class VideoLatentTools(LatentTools):
    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS
    causal_fix: bool = True

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is None:
            initial_latent = torch.zeros(*self.target_shape.to_torch_shape(), device=device, dtype=dtype)
        else:
            initial_latent = initial_latent.to(device=device, dtype=dtype)
        denoise_mask = torch.ones(*self.target_shape.mask_shape().to_torch_shape(), device=device, dtype=torch.float32)
        latent_coords = self.patchifier.get_patch_grid_bounds(output_shape=self.target_shape, device=device)
        positions = get_pixel_coords(
            latent_coords,
            self.scale_factors,
            causal_fix=self.causal_fix,
        ).to(dtype=torch.float32)
        positions[:, 0, ...] /= self.fps
        state = self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions,
                clean_latent=initial_latent.clone(),
                keyframes_mask=torch.zeros_like(denoise_mask),
            )
        )
        assert state.keyframes_mask is not None
        first_frame_tokens = self.patchifier.get_token_count(self.target_shape._replace(frames=1))
        keyframes_mask = state.keyframes_mask.clone()
        keyframes_mask[:, :first_frame_tokens] = 1.0
        return replace(state, keyframes_mask=keyframes_mask)


@dataclass(frozen=True)
class AudioLatentTools(LatentTools):
    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is None:
            initial_latent = torch.zeros(*self.target_shape.to_torch_shape(), device=device, dtype=dtype)
        else:
            initial_latent = initial_latent.to(device=device, dtype=dtype)
        denoise_mask = torch.ones(*self.target_shape.mask_shape().to_torch_shape(), device=device, dtype=torch.float32)
        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=self.patchifier.get_patch_grid_bounds(
                    output_shape=self.target_shape,
                    device=device,
                ).to(dtype=torch.float32),
                clean_latent=initial_latent.clone(),
            )
        )


class ConditioningError(RuntimeError):
    pass


class ConditioningItem(Protocol):
    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState: ...


class VideoConditionByLatentIndex:
    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        cond_batch, cond_channels, _, cond_height, cond_width = self.latent.shape
        target_shape = latent_tools.target_shape
        tgt_batch, tgt_channels, tgt_frames, tgt_height, tgt_width = target_shape.to_torch_shape()
        if (cond_batch, cond_channels, cond_height, cond_width) != (tgt_batch, tgt_channels, tgt_height, tgt_width):
            raise ConditioningError(
                f"Can't apply image conditioning item to latent with shape {target_shape}, expected shape is "
                f"({tgt_batch}, {tgt_channels}, {tgt_frames}, {tgt_height}, {tgt_width})."
            )
        tokens = latent_tools.patchifier.patchify(self.latent)
        start_token = latent_tools.patchifier.get_token_count(target_shape._replace(frames=self.latent_idx))
        stop_token = start_token + tokens.shape[1]
        latent_state = latent_state.clone()
        latent_state.latent[:, start_token:stop_token] = tokens
        latent_state.clean_latent[:, start_token:stop_token] = tokens
        latent_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength
        return latent_state


class VideoConditionByKeyframeIndex:
    def __init__(self, keyframes: torch.Tensor, frame_idx: int, strength: float):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength

    def apply_to(self, latent_state: LatentState, latent_tools: VideoLatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        positions = get_pixel_coords(
            latent_coords=latent_tools.patchifier.get_patch_grid_bounds(
                output_shape=VideoLatentShape.from_torch_shape(self.keyframes.shape),
                device=self.keyframes.device,
            ),
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix if self.frame_idx == 0 else False,
        ).to(dtype=torch.float32)
        positions[:, 0, ...] += self.frame_idx
        positions[:, 0, ...] /= latent_tools.fps
        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )
        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=update_attention_mask(
                latent_state=latent_state,
                attention_mask=None,
                num_noisy_tokens=latent_tools.target_shape.token_count(),
                num_new_tokens=tokens.shape[1],
                batch_size=tokens.shape[0],
                device=self.keyframes.device,
                dtype=self.keyframes.dtype,
            ),
        )


class ConditioningItemAttentionStrengthWrapper:
    def __init__(self, conditioning: ConditioningItem, attention_mask: float | torch.Tensor):
        self.conditioning = conditioning
        self.attention_mask = attention_mask

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        original_state = latent_state
        new_state = self.conditioning.apply_to(latent_state, latent_tools)
        num_new_tokens = new_state.latent.shape[1] - original_state.latent.shape[1]
        if num_new_tokens == 0:
            return new_state
        return replace(
            new_state,
            attention_mask=update_attention_mask(
                latent_state=original_state,
                attention_mask=self.attention_mask,
                num_noisy_tokens=latent_tools.target_shape.token_count(),
                num_new_tokens=num_new_tokens,
                batch_size=new_state.latent.shape[0],
                device=new_state.latent.device,
                dtype=new_state.latent.dtype,
            ),
        )


def resolve_cross_mask(
    attention_mask: float | torch.Tensor,
    num_new_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(attention_mask, float):
        return torch.full((batch_size, num_new_tokens), attention_mask, device=device, dtype=dtype)
    if attention_mask.ndim == 1:
        return attention_mask[None].expand(batch_size, -1).to(device=device, dtype=dtype)
    if attention_mask.ndim == 2:
        return attention_mask.to(device=device, dtype=dtype)
    raise ValueError(f"Unsupported attention mask shape: {attention_mask.shape}")


def build_attention_mask(
    existing_mask: torch.Tensor | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    num_existing_tokens: int,
    cross_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = cross_mask.shape[0]
    total_tokens = num_existing_tokens + num_new_tokens
    attention_mask = torch.zeros((batch_size, total_tokens, total_tokens), device=device, dtype=dtype)
    if existing_mask is not None:
        attention_mask[:, :num_existing_tokens, :num_existing_tokens] = existing_mask
    else:
        attention_mask[:, :num_existing_tokens, :num_existing_tokens] = 1.0
    attention_mask[:, num_existing_tokens:, num_existing_tokens:] = 1.0
    attention_mask[:, :num_noisy_tokens, num_existing_tokens:] = cross_mask.unsqueeze(1)
    attention_mask[:, num_existing_tokens:, :num_noisy_tokens] = cross_mask.unsqueeze(2)
    return attention_mask


def update_attention_mask(
    latent_state: LatentState,
    attention_mask: float | torch.Tensor | None,
    num_noisy_tokens: int,
    num_new_tokens: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if attention_mask is None:
        if latent_state.attention_mask is None:
            return None
        cross_mask = torch.ones(batch_size, num_new_tokens, device=device, dtype=dtype)
        return build_attention_mask(
            existing_mask=latent_state.attention_mask,
            num_noisy_tokens=num_noisy_tokens,
            num_new_tokens=num_new_tokens,
            num_existing_tokens=latent_state.latent.shape[1],
            cross_mask=cross_mask,
            device=device,
            dtype=dtype,
        )
    return build_attention_mask(
        existing_mask=latent_state.attention_mask,
        num_noisy_tokens=num_noisy_tokens,
        num_new_tokens=num_new_tokens,
        num_existing_tokens=latent_state.latent.shape[1],
        cross_mask=resolve_cross_mask(attention_mask, num_new_tokens, batch_size, device, dtype),
        device=device,
        dtype=dtype,
    )


@dataclass(frozen=True)
class MultiModalGuiderParams:
    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    stg_blocks: list[int] | None = field(default_factory=list)
    rescale_scale: float = 0.0
    modality_scale: float = 1.0
    skip_step: int = 0


@dataclass(frozen=True)
class MultiModalGuider:
    params: MultiModalGuiderParams
    negative_context: torch.Tensor | None = None

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
    ) -> torch.Tensor:
        pred = (
            cond
            + (self.params.cfg_scale - 1) * (cond - uncond_text)
            + self.params.stg_scale * (cond - uncond_perturbed)
            + (self.params.modality_scale - 1) * (cond - uncond_modality)
        )
        if self.params.rescale_scale != 0:
            factor = cond.std() / pred.std()
            factor = self.params.rescale_scale * factor + (1 - self.params.rescale_scale)
            pred = pred * factor
        return pred

    def do_unconditional_generation(self) -> bool:
        return not math.isclose(self.params.cfg_scale, 1.0)

    def do_perturbed_generation(self) -> bool:
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step_index: int) -> bool:
        if self.params.skip_step == 0:
            return False
        return step_index % (self.params.skip_step + 1) != 0
