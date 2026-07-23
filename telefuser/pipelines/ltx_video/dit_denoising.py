from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, NamedTuple, Protocol

import einops
import torch
from loguru import logger
from torch._prims_common import DeviceLikeType

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx_dit import LTXVideoTransformer
from telefuser.models.ltx_video_vae import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoLatentShape
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.lora_loader import LoRALoader
from telefuser.utils.profiler import ProfilingContext4Debug

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

    def clone(self) -> LatentState:
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
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
        )

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        num_tokens = self.patchifier.get_token_count(self.target_shape)
        return LatentState(
            latent=latent_state.latent[:, :num_tokens],
            denoise_mask=torch.ones_like(latent_state.denoise_mask)[:, :num_tokens],
            positions=latent_state.positions[:, :, :num_tokens],
            clean_latent=latent_state.clean_latent[:, :num_tokens],
            attention_mask=None,
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
        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions.to(dtype),
                clean_latent=initial_latent.clone(),
            )
        )


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
                ).to(dtype),
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


class GaussianNoiser:
    def __init__(self, noise: torch.Tensor):
        self.noise = noise

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        noise = self.noise.to(device=latent_state.latent.device, dtype=latent_state.latent.dtype)
        if noise.shape != latent_state.latent.shape:
            raise ValueError(
                f"Noise shape {tuple(noise.shape)} does not match latent shape {tuple(latent_state.latent.shape)}."
            )
        # Avoid type promotion (e.g., bf16 latent * fp32 mask -> fp32) which would later break bf16 Linear kernels.
        mask = latent_state.denoise_mask.to(dtype=latent_state.latent.dtype)
        scaled_mask = mask * noise_scale
        return replace(latent_state, latent=(noise * scaled_mask + latent_state.latent * (1 - scaled_mask)))


class DitDenoisingStage(BaseStage):
    """Denoising stage for LTX video generation."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
        model_name: str,
        scheduler: object,
    ):
        super().__init__(name, model_runtime_config)
        self.dit: LTXVideoTransformer = module_manager.fetch_module(model_name)
        if self.dit is not None and hasattr(self.dit, "set_attention_config"):
            self.dit.set_attention_config(model_runtime_config.attention_config)
        self.apply_model_quantization(self.dit)
        self.model_names = ["dit"]
        self.scheduler = scheduler
        self.flow_match_scheduler = FlowMatchScheduler(template="LTX.2")
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)
        self._lora_applied = False

        # Handle torch.compile
        if model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for ltx dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    def _maybe_apply_loras(self) -> None:
        """Apply configured LoRA weights to the underlying velocity model (once).

        Note: LoRA is merged into weights in-place. For two-stage LTX pipelines, stage-2 typically supplies
        distilled LoRA weights while stage-1 keeps base weights. In multiprocessing mode, applying inside the
        worker process avoids mutating shared module instances in the parent process.
        """
        if self._lora_applied:
            return
        lora_configs = self.model_runtime_config.lora_configs
        if not lora_configs:
            self._lora_applied = True
            return

        # Apply LoRA directly to the inner velocity model so state_dict keys match `transformer_blocks.*`.
        # The LoRA loader already strips common prefixes like `diffusion_model.` and `model.`.
        target = getattr(self.dit, "velocity_model", self.dit)
        lora_loader = LoRALoader()
        for lora_config in lora_configs:
            lora_path = lora_config.path
            strength = lora_config.strength
            applied = lora_loader.apply_lora(target, lora_path, strength=strength)
            logger.info(f"Loaded LoRA: {lora_path} with strength: {strength} for ltx dit (applied={applied})")

        self._lora_applied = True

    def _prepare_schedule(
        self,
        num_inference_steps: int,
        sigmas: torch.Tensor | None,
        initial_video_latent: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sigmas is None:
            self.flow_match_scheduler.set_timesteps(
                num_inference_steps=num_inference_steps,
                latent=initial_video_latent,
            )
            sigmas = self.flow_match_scheduler.sigmas.to(dtype=torch.float32, device=self.device)
        else:
            sigmas = sigmas.to(dtype=torch.float32, device=self.device)

        timesteps = (sigmas[:-1] * self.scheduler.num_train_timesteps).to(dtype=torch.float32, device="cpu")
        self.scheduler.sigmas = sigmas
        self.scheduler.timesteps = timesteps
        return sigmas, timesteps

    def _build_denoise_fn(
        self,
        positive_context: object,
        negative_context: object | None,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        simple_denoise: bool,
    ) -> Callable[[LatentState, LatentState, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]]:
        def resolve_context(context: object) -> tuple[torch.Tensor, torch.Tensor]:
            """Resolve (video_encoding, audio_encoding) from supported context formats.

            The LTX text encoder returns `EmbeddingsProcessorOutput` (a NamedTuple) in the normal code path.
            When passing through `ParallelWorker`, tuples are converted to lists, so we also accept list/tuple.
            """
            if isinstance(context, dict):
                video_encoding = context.get("video_encoding")
                audio_encoding = context.get("audio_encoding")
            elif hasattr(context, "video_encoding") and hasattr(context, "audio_encoding"):
                video_encoding = getattr(context, "video_encoding")
                audio_encoding = getattr(context, "audio_encoding")
            elif isinstance(context, (list, tuple)):
                if len(context) < 2:
                    raise TypeError(
                        "Expected context as (video_encoding, audio_encoding, ...), got "
                        f"{type(context).__name__} with length {len(context)}."
                    )
                video_encoding = context[0]
                audio_encoding = context[1]
            else:
                raise TypeError(
                    "Unsupported context type. Expected dict, EmbeddingsProcessorOutput-like, or list/tuple, got "
                    f"{type(context)!r}."
                )

            if not isinstance(video_encoding, torch.Tensor):
                raise TypeError(f"video_encoding must be a torch.Tensor, got {type(video_encoding)!r}.")
            if not isinstance(audio_encoding, torch.Tensor):
                raise TypeError(f"audio_encoding must be a torch.Tensor, got {type(audio_encoding)!r}.")
            return video_encoding, audio_encoding

        positive_video_encoding, positive_audio_encoding = resolve_context(positive_context)
        negative_video_encoding = None
        negative_audio_encoding = None
        if negative_context is not None:
            negative_video_encoding, negative_audio_encoding = resolve_context(negative_context)

        def build_modality(state: LatentState, context: torch.Tensor, sigma: torch.Tensor, enabled: bool = True):
            from telefuser.models.ltx_dit import Modality

            return Modality(
                enabled=enabled,
                latent=state.latent,
                sigma=sigma,
                timesteps=state.denoise_mask * sigma,
                positions=state.positions,
                context=context,
                context_mask=None,
                attention_mask=state.attention_mask,
            )

        if simple_denoise:

            def denoise_step(
                video_state: LatentState,
                audio_state: LatentState,
                sigmas: torch.Tensor,
                step_index: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                sigma = sigmas[step_index]
                video = build_modality(video_state, positive_video_encoding, sigma)
                audio = build_modality(audio_state, positive_audio_encoding, sigma)
                return self.dit(video=video, audio=audio, perturbations=None)

            return denoise_step

        video_guider = MultiModalGuider(
            video_guider_params,
            negative_video_encoding,
        )
        audio_guider = MultiModalGuider(
            audio_guider_params,
            negative_audio_encoding,
        )
        last_denoised_video: torch.Tensor | None = None
        last_denoised_audio: torch.Tensor | None = None

        def denoise_step(
            video_state: LatentState,
            audio_state: LatentState,
            sigmas: torch.Tensor,
            step_index: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal last_denoised_video, last_denoised_audio
            if video_guider.should_skip_step(step_index) and audio_guider.should_skip_step(step_index):
                return last_denoised_video, last_denoised_audio

            from telefuser.models.ltx_dit import (
                BatchedPerturbationConfig,
                Perturbation,
                PerturbationConfig,
                PerturbationType,
            )

            sigma = sigmas[step_index]
            pos_video = build_modality(
                video_state,
                positive_video_encoding,
                sigma,
                enabled=not video_guider.should_skip_step(step_index),
            )
            pos_audio = build_modality(
                audio_state,
                positive_audio_encoding,
                sigma,
                enabled=not audio_guider.should_skip_step(step_index),
            )
            denoised_video, denoised_audio = self.dit(video=pos_video, audio=pos_audio, perturbations=None)

            neg_denoised_video, neg_denoised_audio = 0.0, 0.0
            if video_guider.do_unconditional_generation() or audio_guider.do_unconditional_generation():
                neg_video = build_modality(
                    video_state,
                    video_guider.negative_context if video_guider.negative_context is not None else pos_video.context,
                    sigma,
                )
                neg_audio = build_modality(
                    audio_state,
                    audio_guider.negative_context if audio_guider.negative_context is not None else pos_audio.context,
                    sigma,
                )
                neg_denoised_video, neg_denoised_audio = self.dit(video=neg_video, audio=neg_audio, perturbations=None)

            ptb_denoised_video, ptb_denoised_audio = 0.0, 0.0
            if video_guider.do_perturbed_generation() or audio_guider.do_perturbed_generation():
                perturbations = []
                if video_guider.do_perturbed_generation():
                    perturbations.append(
                        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks)
                    )
                if audio_guider.do_perturbed_generation():
                    perturbations.append(
                        Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks)
                    )
                ptb_config = BatchedPerturbationConfig(perturbations=[PerturbationConfig(perturbations=perturbations)])
                ptb_denoised_video, ptb_denoised_audio = self.dit(
                    video=pos_video,
                    audio=pos_audio,
                    perturbations=ptb_config,
                )

            mod_denoised_video, mod_denoised_audio = 0.0, 0.0
            if video_guider.do_isolated_modality_generation() or audio_guider.do_isolated_modality_generation():
                mod_config = BatchedPerturbationConfig(
                    perturbations=[
                        PerturbationConfig(
                            perturbations=[
                                Perturbation(type=PerturbationType.SKIP_A2V_CROSS_ATTN, blocks=None),
                                Perturbation(type=PerturbationType.SKIP_V2A_CROSS_ATTN, blocks=None),
                            ]
                        )
                    ]
                )
                mod_denoised_video, mod_denoised_audio = self.dit(
                    video=pos_video,
                    audio=pos_audio,
                    perturbations=mod_config,
                )

            denoised_video = (
                last_denoised_video
                if video_guider.should_skip_step(step_index)
                else video_guider.calculate(denoised_video, neg_denoised_video, ptb_denoised_video, mod_denoised_video)
            )
            denoised_audio = (
                last_denoised_audio
                if audio_guider.should_skip_step(step_index)
                else audio_guider.calculate(denoised_audio, neg_denoised_audio, ptb_denoised_audio, mod_denoised_audio)
            )
            last_denoised_video = denoised_video
            last_denoised_audio = denoised_audio
            return denoised_video, denoised_audio

        return denoise_step

    def _create_initial_states(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        conditionings: torch.Tensor | list[ConditioningItem] | None,
        video_noiser: GaussianNoiser,
        audio_noiser: GaussianNoiser,
        noise_scale: float,
        initial_video_latent: torch.Tensor | None,
        initial_audio_latent: torch.Tensor | None,
    ) -> tuple[LatentState, LatentState, VideoLatentTools, AudioLatentTools]:
        video_latent_shape = VideoLatentShape(
            batch=1,
            channels=VIDEO_LATENT_CHANNELS,
            frames=(num_frames - 1) // VIDEO_SCALE_FACTORS.time + 1,
            height=height // VIDEO_SCALE_FACTORS.height,
            width=width // VIDEO_SCALE_FACTORS.width,
        )
        video_tools = VideoLatentTools(self.video_patchifier, video_latent_shape, frame_rate)
        video_state = video_tools.create_initial_state(self.device, self.torch_dtype, initial_video_latent)
        if conditionings is None:
            conditionings_list: list[ConditioningItem] = []
            conditioning_latents = None
        elif isinstance(conditionings, torch.Tensor):
            conditionings_list = []
            conditioning_latents = conditionings
        else:
            conditionings_list = conditionings
            conditioning_latents = None

        if conditioning_latents is not None:
            video_state = self._apply_conditioning_latents(video_state, video_tools, conditioning_latents)
        for conditioning in conditionings_list:
            video_state = conditioning.apply_to(latent_state=video_state, latent_tools=video_tools)
        video_state = video_noiser(video_state, noise_scale)

        audio_latent_shape = AudioLatentShape.from_duration(batch=1, duration=float(num_frames) / float(frame_rate))
        audio_tools = AudioLatentTools(self.audio_patchifier, audio_latent_shape)
        audio_state = audio_tools.create_initial_state(self.device, self.torch_dtype, initial_audio_latent)
        audio_state = audio_noiser(audio_state, noise_scale)
        return video_state, audio_state, video_tools, audio_tools

    def _apply_conditioning_latents(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
        conditionings: torch.Tensor,
    ) -> LatentState:
        """Apply tensor-based image conditioning exported by `ltx_video.vae.VAEStage.encode_image`.

        The conditioning tensor packs a per-token denoise mask and the corresponding clean latents.
        Shape: (B, 1 + C, F_cond, H, W) where C == VIDEO_LATENT_CHANNELS.
        Only tokens with mask < 1.0 are applied.
        """
        if conditionings.ndim != 5:
            raise ValueError(f"Expected conditioning latents with 5 dimensions, got {tuple(conditionings.shape)}.")
        if conditionings.shape[1] != 1 + VIDEO_LATENT_CHANNELS:
            raise ValueError(
                f"Expected conditioning latents channel dimension {1 + VIDEO_LATENT_CHANNELS}, got "
                f"{conditionings.shape[1]}."
            )

        conditionings = conditionings.to(device=self.device, dtype=self.torch_dtype)
        cond_mask = conditionings[:, :1]
        cond_latents = conditionings[:, 1:]
        cond_frames = cond_latents.shape[2]
        if cond_latents.shape[0] != latent_state.latent.shape[0]:
            raise ValueError(
                f"Conditioning batch size {cond_latents.shape[0]} does not match latent batch size "
                f"{latent_state.latent.shape[0]}."
            )
        if cond_frames <= 0:
            return latent_state

        # Validate that the conditioning window fits within the target video latent sequence.
        target_shape = latent_tools.target_shape
        tokens_per_frame = target_shape.height * target_shape.width
        if cond_latents.shape[3] != target_shape.height or cond_latents.shape[4] != target_shape.width:
            raise ValueError(
                "Conditioning spatial shape does not match target latent shape: "
                f"cond_hw=({cond_latents.shape[3]}, {cond_latents.shape[4]}), "
                f"target_hw=({target_shape.height}, {target_shape.width})."
            )
        max_tokens = target_shape.frames * tokens_per_frame
        cond_token_limit = cond_frames * tokens_per_frame
        if cond_token_limit > max_tokens:
            raise ValueError(
                "Conditioning temporal window exceeds target latent frames: "
                f"cond_frames={cond_frames}, target_frames={target_shape.frames}."
            )

        mask_tokens = latent_tools.patchifier.patchify(cond_mask).to(dtype=torch.float32)
        latent_tokens = latent_tools.patchifier.patchify(cond_latents)
        if latent_tokens.shape[1] != cond_token_limit or mask_tokens.shape[1] != cond_token_limit:
            raise ValueError(
                "Conditioning token count does not match expected layout: "
                f"expected={cond_token_limit}, mask_tokens={mask_tokens.shape[1]}, "
                f"latent_tokens={latent_tokens.shape[1]}."
            )
        if latent_tokens.shape[1] != mask_tokens.shape[1]:
            raise ValueError("Conditioning mask token count does not match conditioning latent token count.")
        if latent_tokens.shape[2] != latent_state.latent.shape[2]:
            raise ValueError(
                f"Conditioning latent dim {latent_tokens.shape[2]} does not match target latent dim "
                f"{latent_state.latent.shape[2]}."
            )

        update = (mask_tokens < 1.0).squeeze(-1)
        if not torch.any(update):
            return latent_state

        latent_state = latent_state.clone()
        latent_state.latent[:, :cond_token_limit][update] = latent_tokens[:, :cond_token_limit][update]
        latent_state.clean_latent[:, :cond_token_limit][update] = latent_tokens[:, :cond_token_limit][update]
        latent_state.denoise_mask[:, :cond_token_limit][update] = mask_tokens[:, :cond_token_limit][update]
        return latent_state

    def _denoise(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        sigmas: torch.Tensor,
        timesteps: torch.Tensor,
        denoise_fn: Callable[[LatentState, LatentState, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[LatentState, LatentState]:
        for step_index, timestep in enumerate(timesteps):
            denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_index)
            denoised_video = (
                denoised_video * video_state.denoise_mask
                + video_state.clean_latent.float() * (1 - video_state.denoise_mask)
            ).to(denoised_video.dtype)
            denoised_audio = (
                denoised_audio * audio_state.denoise_mask
                + audio_state.clean_latent.float() * (1 - audio_state.denoise_mask)
            ).to(denoised_audio.dtype)
            to_final = step_index + 1 >= len(timesteps)
            if hasattr(self.scheduler, "template"):
                next_video = self.scheduler.step(denoised_video, timestep, video_state.latent, to_final=to_final)
                next_audio = self.scheduler.step(denoised_audio, timestep, audio_state.latent, to_final=to_final)
            else:
                next_video = self.scheduler.step(denoised_video, timestep, video_state.latent, return_dict=False)
                next_audio = self.scheduler.step(denoised_audio, timestep, audio_state.latent, return_dict=False)
                next_video = next_video[0] if isinstance(next_video, tuple) else next_video
                next_audio = next_audio[0] if isinstance(next_audio, tuple) else next_audio
            video_state = replace(video_state, latent=next_video)
            audio_state = replace(audio_state, latent=next_audio)
        return video_state, audio_state

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("ltx denoising")
    @torch.inference_mode()
    def process(
        self,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        conditionings: torch.Tensor | list[ConditioningItem] | None,
        positive_context: object,
        negative_context: object | None,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        initial_video_latent: torch.Tensor | None = None,
        initial_audio_latent: torch.Tensor | None = None,
        sigmas: torch.Tensor | None = None,
        noise_scale: float = 1.0,
        simple_denoise: bool = False,
    ):
        """Run a denoising pass and return video/audio latent states."""
        self._maybe_apply_loras()
        video_noiser = GaussianNoiser(video_noise)
        audio_noiser = GaussianNoiser(audio_noise)
        sigmas, timesteps = self._prepare_schedule(num_inference_steps, sigmas, initial_video_latent)
        denoise_fn = self._build_denoise_fn(
            positive_context=positive_context,
            negative_context=negative_context,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            simple_denoise=simple_denoise,
        )
        video_state, audio_state, video_tools, audio_tools = self._create_initial_states(
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            conditionings=conditionings,
            video_noiser=video_noiser,
            audio_noiser=audio_noiser,
            noise_scale=noise_scale,
            initial_video_latent=initial_video_latent,
            initial_audio_latent=initial_audio_latent,
        )
        video_state, audio_state = self._denoise(video_state, audio_state, sigmas, timesteps, denoise_fn)
        video_state = video_tools.unpatchify(video_tools.clear_conditioning(video_state))
        audio_state = audio_tools.unpatchify(audio_tools.clear_conditioning(audio_state))
        return video_state, audio_state

    @staticmethod
    def stage2_sigmas(device: torch.device) -> torch.Tensor:
        """Return the fixed distilled sigma schedule used by stage 2."""
        return torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)

