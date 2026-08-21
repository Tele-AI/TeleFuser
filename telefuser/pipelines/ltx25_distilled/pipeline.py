"""Multi-stage LTX-2.5 distilled pipeline runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Sequence

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.models.ltx25.diff_vae.types import VideoLatentShape
from telefuser.models.ltx25.sampler import ANCESTRAL_NOISE_SEED_OFFSET, LTX25_STAGE2_DISTILLED_SIGMAS
from telefuser.utils.func import auto_async_call
from telefuser.worker.parallel_worker import ParallelWorker

from .audio_decoding import LTX25AudioDecodingStage
from .denoising import LTX25DenoisingStage
from .latent import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    LatentState,
    VideoLatentPatchifier,
    VideoLatentTools,
)
from .latent_upsampling import LTX25LatentUpsamplingStage
from .loader import load_ltx25_distilled_modules
from .text_encoding import LTX25TextEncodingStage
from .video_conditioning import LTX25VideoConditioningStage
from .video_decoding import LTX25VideoDecodingStage


@dataclass(frozen=True, slots=True)
class LTX25ImageCondition:
    image: Image.Image
    frame_idx: int = 0
    strength: float = 1.0
    crf: int | None = None


@dataclass(frozen=True, slots=True)
class LTX25DistilledOutput:
    video_chunks: tuple[torch.Tensor, ...]
    audio: torch.Tensor
    video_latent: torch.Tensor
    audio_latent: torch.Tensor
    num_frames: int
    frame_rate: float


@dataclass
class LTX25DistilledConfig:
    """Runtime settings for independently managed LTX-2.5 stages."""

    video_vae: Literal["diff", "conv"] = "diff"
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    video_conditioning_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    denoising_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    upsampling_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    video_decoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    audio_decoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)


class LTX25DistilledPipeline(BasePipeline):
    """Two-stage distilled pipeline composed from ModuleManager-backed stages."""

    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: LTX25DistilledConfig | None = None
        self.text_stage: LTX25TextEncodingStage | None = None
        self.conditioning_stage: LTX25VideoConditioningStage | None = None
        self.denoising_stage: LTX25DenoisingStage | ParallelWorker | None = None
        self.upsampling_stage: LTX25LatentUpsamplingStage | None = None
        self.video_decoding_stage: LTX25VideoDecodingStage | None = None
        self.audio_decoding_stage: LTX25AudioDecodingStage | None = None

    def init(self, module_manager: ModuleManager, config: LTX25DistilledConfig) -> None:
        self.config = config
        self.text_stage = LTX25TextEncodingStage(module_manager, config.text_encoding_config)
        self.conditioning_stage = LTX25VideoConditioningStage(module_manager, config.video_conditioning_config)
        denoising_stage = LTX25DenoisingStage(module_manager, config.denoising_config)
        self.denoising_stage = (
            ParallelWorker(denoising_stage)
            if config.denoising_config.parallel_config.world_size > 1
            else denoising_stage
        )
        self.upsampling_stage = LTX25LatentUpsamplingStage(module_manager, config.upsampling_config)
        self.video_decoding_stage = LTX25VideoDecodingStage(
            module_manager, config.video_decoding_config, video_vae=config.video_vae
        )
        self.audio_decoding_stage = LTX25AudioDecodingStage(module_manager, config.audio_decoding_config)
        self._model_info = module_manager.get_model_info()

    @classmethod
    def from_model_root(
        cls,
        model_root: str | Path,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        video_vae: Literal["diff", "conv"] = "diff",
        offload: Literal["none", "cpu"] = "cpu",
        parallelism: int = 1,
        attn_impl: AttnImplType = AttnImplType.FLASH_ATTN_4,
    ) -> "LTX25DistilledPipeline":
        module_manager = ModuleManager(device="cpu", torch_dtype=torch_dtype)
        load_ltx25_distilled_modules(module_manager, model_root, video_vae=video_vae, torch_dtype=torch_dtype)
        pipeline = cls(device=device, torch_dtype=torch_dtype)
        pipeline.init(
            module_manager,
            build_ltx25_distilled_config(
                device,
                torch_dtype,
                video_vae,
                offload,
                parallelism=parallelism,
                attn_impl=attn_impl,
            ),
        )
        return pipeline

    def close(self) -> None:
        """Release distributed denoising workers, when configured."""
        if isinstance(self.denoising_stage, ParallelWorker):
            self.denoising_stage.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _denoising_input(self, state: LatentState) -> LatentState | dict[str, torch.Tensor | None]:
        if not isinstance(self.denoising_stage, ParallelWorker):
            return state
        return {
            "latent": state.latent,
            "denoise_mask": state.denoise_mask,
            "positions": state.positions,
            "clean_latent": state.clean_latent,
            "attention_mask": state.attention_mask,
            "keyframes_mask": state.keyframes_mask,
        }

    def _get_stages(self) -> list[object]:
        return [
            stage
            for stage in (
                self.text_stage,
                self.conditioning_stage,
                self.denoising_stage,
                self.upsampling_stage,
                self.video_decoding_stage,
                self.audio_decoding_stage,
            )
            if stage is not None
        ]

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        seed: int,
        height: int,
        width: int,
        num_frames: int | None = None,
        frame_rate: float = 24.0,
        images: Sequence[LTX25ImageCondition] = (),
    ) -> LTX25DistilledOutput:
        if (
            self.config is None
            or self.text_stage is None
            or self.conditioning_stage is None
            or self.denoising_stage is None
            or self.upsampling_stage is None
            or self.video_decoding_stage is None
            or self.audio_decoding_stage is None
        ):
            raise RuntimeError("LTX25DistilledPipeline.init must be called before generation")

        _validate_resolution(height, width, frame_rate)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video_context, audio_context = self.text_stage.encode(prompt)
        if num_frames is None:
            num_frames = self.text_stage.predict_num_frames(video_context, audio_context, frame_rate)
        _validate_request(height, width, num_frames, frame_rate)

        stage_1_tools = _video_tools(
            batch=1, frames=num_frames, height=height // 2, width=width // 2, frame_rate=frame_rate
        )
        audio_tools = _audio_tools(num_frames=num_frames, frame_rate=frame_rate)
        stage_1_video = stage_1_tools.create_initial_state(self.device, self.torch_dtype)
        if images:
            stage_1_video = self.conditioning_stage.apply(stage_1_video, stage_1_tools, images, height // 2, width // 2)
        stage_1_video = _noised_state(stage_1_video, 1.0, generator)
        stage_1_audio = _noised_state(audio_tools.create_initial_state(self.device, self.torch_dtype), 1.0, generator)

        try:
            stage_1_video, stage_1_audio = auto_async_call(
                self.denoising_stage.denoise_stage1,
                self._denoising_input(stage_1_video),
                self._denoising_input(stage_1_audio),
                video_context,
                audio_context,
                noise_seed=seed + ANCESTRAL_NOISE_SEED_OFFSET,
            )()
            low_resolution_latent = stage_1_tools.unpatchify(stage_1_tools.clear_conditioning(stage_1_video)).latent
            upscaled_video = self.upsampling_stage.process(low_resolution_latent)

            stage_2_tools = _video_tools(batch=1, frames=num_frames, height=height, width=width, frame_rate=frame_rate)
            stage_2_video = stage_2_tools.create_initial_state(self.device, self.torch_dtype, upscaled_video)
            if images:
                stage_2_video = self.conditioning_stage.apply(stage_2_video, stage_2_tools, images, height, width)
            stage_2_video = _noised_state(stage_2_video, LTX25_STAGE2_DISTILLED_SIGMAS[0], generator)
            stage_2_audio = _noised_state(
                audio_tools.create_initial_state(
                    self.device, self.torch_dtype, audio_tools.unpatchify(stage_1_audio).latent
                ),
                LTX25_STAGE2_DISTILLED_SIGMAS[0],
                generator,
            )
            stage_2_video, stage_2_audio = auto_async_call(
                self.denoising_stage.denoise_stage2,
                self._denoising_input(stage_2_video),
                self._denoising_input(stage_2_audio),
                video_context,
                audio_context,
            )()
        finally:
            if not isinstance(self.denoising_stage, ParallelWorker) or not self.denoising_stage.failed:
                auto_async_call(self.denoising_stage.finish_request)()

        video_latent = stage_2_tools.unpatchify(stage_2_tools.clear_conditioning(stage_2_video)).latent
        audio_latent = audio_tools.unpatchify(stage_2_audio).latent
        return LTX25DistilledOutput(
            self.video_decoding_stage.decode(video_latent, generator),
            self.audio_decoding_stage.decode(audio_latent),
            video_latent,
            audio_latent,
            num_frames,
            frame_rate,
        )


def build_ltx25_distilled_config(
    device: str,
    torch_dtype: torch.dtype,
    video_vae: Literal["diff", "conv"],
    offload: Literal["none", "cpu"],
    *,
    parallelism: int = 1,
    attn_impl: AttnImplType = AttnImplType.FLASH_ATTN_4,
) -> LTX25DistilledConfig:
    if video_vae not in ("diff", "conv"):
        raise ValueError(f"video_vae must be 'diff' or 'conv', got {video_vae!r}")
    if offload not in ("none", "cpu"):
        raise ValueError(f"offload must be 'none' or 'cpu', got {offload!r}")
    if parallelism < 1 or 32 % parallelism:
        raise ValueError(f"parallelism must be a positive divisor of 32, got {parallelism}")
    if not isinstance(attn_impl, AttnImplType):
        raise TypeError(f"attn_impl must be an AttnImplType, got {type(attn_impl).__name__}")
    attention_config = AttentionConfig.dense_attention(attn_impl)
    if attention_config.is_sparse():
        raise ValueError(f"LTX-2.5 supports dense attention implementations only, got {attn_impl.name}")
    target_device = torch.device(device)
    device_type = target_device.type
    device_id = 0 if target_device.index is None else target_device.index
    regular = WeightOffloadType.MODEL_CPU_OFFLOAD if offload == "cpu" else WeightOffloadType.NO_CPU_OFFLOAD
    denoising = (
        WeightOffloadType.ASYNC_CPU_OFFLOAD
        if offload == "cpu" and parallelism == 1
        else WeightOffloadType.NO_CPU_OFFLOAD
    )

    def runtime(offload_type: WeightOffloadType, *, distributed: bool = False) -> ModelRuntimeConfig:
        return ModelRuntimeConfig(
            device_type=device_type,
            device_id=device_id,
            torch_dtype=torch_dtype,
            offload_config=OffloadConfig(offload_type=offload_type),
            attention_config=attention_config,
            parallel_config=(
                ParallelConfig(
                    device_ids=list(range(parallelism)),
                    sp_ulysses_degree=parallelism,
                    enable_fsdp=True,
                    timeout=1800,
                )
                if distributed
                else ParallelConfig()
            ),
        )

    return LTX25DistilledConfig(
        video_vae=video_vae,
        text_encoding_config=runtime(regular),
        video_conditioning_config=runtime(regular),
        denoising_config=runtime(denoising, distributed=parallelism > 1),
        upsampling_config=runtime(regular),
        video_decoding_config=runtime(regular),
        audio_decoding_config=runtime(regular),
    )


def _video_tools(*, batch: int, frames: int, height: int, width: int, frame_rate: float) -> VideoLatentTools:
    shape = VideoLatentShape(
        batch=batch,
        channels=128,
        frames=(frames - 1) // 8 + 1,
        height=height // 32,
        width=width // 32,
    )
    return VideoLatentTools(VideoLatentPatchifier(1), shape, frame_rate)


def _audio_tools(*, num_frames: int, frame_rate: float) -> AudioLatentTools:
    return AudioLatentTools(
        AudioPatchifier(1), AudioLatentShape.from_duration(batch=1, duration=num_frames / frame_rate)
    )


def _noised_state(state: LatentState, noise_scale: float, generator: torch.Generator) -> LatentState:
    noise = torch.randn(state.latent.shape, generator=generator, dtype=state.latent.dtype, device=state.latent.device)
    latent = torch.lerp(state.latent.float(), noise.float(), noise_scale)
    latent = torch.lerp(state.clean_latent.float(), latent, state.denoise_mask)
    return replace(state, latent=latent.to(state.latent.dtype))


def _validate_resolution(height: int, width: int, frame_rate: float) -> None:
    if height <= 0 or width <= 0 or height % 64 or width % 64:
        raise ValueError("LTX-2.5 distilled height and width must be positive multiples of 64")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")


def _validate_request(height: int, width: int, num_frames: int, frame_rate: float) -> None:
    _validate_resolution(height, width, frame_rate)
    if num_frames < 1 or (num_frames - 1) % 8:
        raise ValueError("LTX-2.5 num_frames must satisfy num_frames = 8k + 1")


__all__ = [
    "LTX25DistilledConfig",
    "LTX25DistilledOutput",
    "LTX25DistilledPipeline",
    "LTX25ImageCondition",
    "build_ltx25_distilled_config",
]
