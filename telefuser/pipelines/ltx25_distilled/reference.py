"""Faithful monolithic LTX-2.5 distilled text-to-video diffusion reference path."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

import torch
from PIL import Image

from telefuser.models.ltx25.checkpoint import LTX25ModelPaths, inspect_checkpoint
from telefuser.models.ltx25.diff_vae.types import VideoLatentShape, VideoPixelShape
from telefuser.models.ltx25.embeddings import LTX25EmbeddingsProcessorOutput
from telefuser.models.ltx25.sampler import LTX25EulerAncestralStep, ancestral_noise_generator, distilled_sigmas
from telefuser.models.ltx25.transformer import BatchedPerturbationConfig, Modality

from .image import default_image_crf, preprocess_ltx25_image
from .latent import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    LatentState,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoLatentPatchifier,
    VideoLatentTools,
)


def _release_modules(*modules: object) -> None:
    """Return lazily loaded checkpoint modules to CPU between reference phases."""
    for module in modules:
        if isinstance(module, (_LazyTextEncoder, _LazyCallable)):
            module.release()
        elif isinstance(module, torch.nn.Module):
            module.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class TextEncoder(Protocol):
    """Minimal Gemma interface consumed by the distilled reference path."""

    def encode(self, prompts: list[str]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]: ...


class EmbeddingsProcessor(Protocol):
    """Minimal dual-context connector interface consumed by the reference path."""

    def __call__(
        self, hidden_states: tuple[torch.Tensor, ...], attention_mask: torch.Tensor
    ) -> LTX25EmbeddingsProcessorOutput: ...


class X0Transformer(Protocol):
    """Joint video/audio x0 prediction interface."""

    def __call__(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]: ...


class SpatialUpsampler(Protocol):
    """Spatial latent upsampler interface."""

    def __call__(self, latent: torch.Tensor) -> torch.Tensor: ...


class LatentStatistics(Protocol):
    """Video-VAE latent normalization bridge used by the learned upsampler."""

    def normalize(self, latent: torch.Tensor) -> torch.Tensor: ...

    def un_normalize(self, latent: torch.Tensor) -> torch.Tensor: ...


class VideoEncoder(Protocol):
    """Video VAE encoder used to build an I2V latent condition."""

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor: ...


class _LazyTextEncoder:
    def __init__(self, loader: Callable[[], TextEncoder], *, release_after_call: bool = True) -> None:
        self._loader = loader
        self._release_after_call = release_after_call
        self._model: TextEncoder | None = None

    def encode(self, prompts: list[str]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        if self._model is None:
            self._model = self._loader()
        try:
            return self._model.encode(prompts)
        finally:
            if self._release_after_call:
                _release_modules(self._model)
                self._model = None

    def release(self) -> None:
        """Release a retained text encoder after the reference path no longer needs it."""
        _release_modules(self._model)
        self._model = None


class _LazyCallable:
    def __init__(self, loader: Callable[[], Callable[..., object]], *, release_after_call: bool = False) -> None:
        self._loader = loader
        self._release_after_call = release_after_call
        self._model: object | None = None

    def resolve(self) -> object:
        if self._model is None:
            self._model = self._loader()
        return self._model

    def release(self) -> None:
        """Release a retained callable checkpoint model."""
        _release_modules(self._model)
        self._model = None

    def __getattr__(self, name: str) -> object:
        """Proxy model attributes required by instrumentation before the first call."""
        return getattr(self.resolve(), name)

    def forward(self, *args: object, **kwargs: object) -> object:
        self.resolve()
        try:
            return self._model(*args, **kwargs)  # type: ignore[operator]
        finally:
            if self._release_after_call:
                _release_modules(self._model)
                self._model = None

    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.forward(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class LTX25ReferenceImageCondition:
    """An I2V condition in output-frame coordinates for Golden capture."""

    image: Image.Image
    frame_idx: int = 0
    strength: float = 1.0
    crf: int | None = None


@dataclass(frozen=True, slots=True)
class LTX25ReferenceRequest:
    """Fixed-shape T2V/I2V request for the faithful pre-stage-splitting path."""

    prompt: str
    seed: int
    height: int
    width: int
    num_frames: int
    frame_rate: float
    images: tuple[LTX25ReferenceImageCondition, ...] = ()

    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0 or self.num_frames <= 0 or self.frame_rate <= 0:
            raise ValueError("height, width, num_frames, and frame_rate must be positive")
        if self.height % 64 or self.width % 64:
            raise ValueError("LTX-2.5 distilled two-stage generation requires height and width divisible by 64")
        for condition in self.images:
            if condition.frame_idx < 0:
                raise ValueError(f"image frame_idx must be non-negative, got {condition.frame_idx}")
            if not 0.0 <= condition.strength <= 1.0:
                raise ValueError(f"image strength must be in [0, 1], got {condition.strength}")


@dataclass(frozen=True, slots=True)
class LTX25ReferenceComponents:
    """Already-loaded modules required by the monolithic diffusion reference path."""

    text_encoder: TextEncoder
    embeddings_processor: EmbeddingsProcessor
    transformer: X0Transformer
    spatial_upsampler: SpatialUpsampler
    latent_statistics: LatentStatistics
    video_encoder: VideoEncoder | None = None


@dataclass(slots=True)
class LTX25ReferenceTrace:
    """Intermediate tensors used for golden-artifact comparison."""

    video_context: torch.Tensor
    audio_context: torch.Tensor
    context_attention_mask: torch.Tensor
    artifacts: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LTX25ReferenceResult:
    """Final unpatchified stage states, decoder RNG state, and comparison trace."""

    stage1_video: LatentState
    stage1_audio: LatentState
    stage2_video: LatentState
    stage2_audio: LatentState
    decoder_generator_state: torch.Tensor
    trace: LTX25ReferenceTrace


def _noised_state(state: LatentState, generator: torch.Generator, noise_scale: float) -> LatentState:
    """Match upstream GaussianNoiser's float32 lerp and original latent dtype."""
    noise = torch.randn(
        *state.latent.shape,
        device=state.latent.device,
        dtype=state.latent.dtype,
        generator=generator,
    )
    latent = torch.lerp(state.latent.float(), noise.float(), noise_scale)
    latent = torch.lerp(state.clean_latent.float(), latent, state.denoise_mask)
    return replace(state, latent=latent.to(state.latent.dtype))


def _modality(state: LatentState, context: torch.Tensor, sigma: torch.Tensor) -> Modality:
    """Construct the upstream SimpleDenoiser modality without guidance or conditioning masks."""
    batch_size = state.latent.shape[0]
    expanded_sigma = sigma.expand(batch_size)
    return Modality(
        latent=state.latent,
        sigma=expanded_sigma,
        timesteps=state.denoise_mask * expanded_sigma.view(batch_size, 1, 1),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
        keyframes_mask=state.keyframes_mask,
    )


def _post_process(denoised: torch.Tensor, state: LatentState) -> torch.Tensor:
    return (denoised * state.denoise_mask + state.clean_latent.float() * (1 - state.denoise_mask)).to(denoised.dtype)


def _deterministic_euler_step(
    sample: torch.Tensor,
    denoised: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    """Match the upstream EulerDiffusionStep rounding points."""
    sigma = sigmas[step_index]
    velocity = ((sample.float() - denoised.float()) / sigma.to(torch.float32).item()).to(sample.dtype)
    return (sample.float() + velocity.float() * (sigmas[step_index + 1] - sigma)).to(sample.dtype)


class LTX25DistilledReference:
    """Faithful two-stage T2V diffusion path used to establish upstream parity."""

    def __init__(
        self,
        components: LTX25ReferenceComponents,
        *,
        device: torch.device,
        dtype: torch.dtype,
        capture_prompt_intermediates: bool = False,
        video_encoder_path: str | Path | None = None,
        offload: Literal["none", "cpu"] = "cpu",
    ) -> None:
        self.components = components
        self.device = device
        self.dtype = dtype
        self.capture_prompt_intermediates = capture_prompt_intermediates
        self.video_encoder_path = Path(video_encoder_path) if video_encoder_path is not None else None
        self.offload = offload

    @classmethod
    def from_model_root(
        cls,
        model_root: str,
        *,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        video_vae: Literal["diff", "conv"] = "diff",
        capture_prompt_intermediates: bool = False,
        offload: Literal["none", "cpu"] = "cpu",
    ) -> "LTX25DistilledReference":
        """Load the exact modules needed by the monolithic T2V parity path."""
        from telefuser.models.ltx25.embeddings import LTX25EmbeddingsProcessor
        from telefuser.models.ltx25.gemma4 import LTX25Gemma4TextEncoder
        from telefuser.models.ltx25.spatial_upsampler import LTX25SpatialUpsampler, load_video_latent_statistics
        from telefuser.models.ltx25.transformer import LTX25AVTransformer

        resolved_device = torch.device(device)
        paths = LTX25ModelPaths.from_model_root(model_root)
        video_vae_path = paths.video_vae_path if video_vae == "diff" else paths.conv_video_vae_path
        components = LTX25ReferenceComponents(
            text_encoder=_LazyTextEncoder(
                lambda: LTX25Gemma4TextEncoder.from_checkpoint(
                    paths.text_encoder_path, device=resolved_device, torch_dtype=dtype
                ),
                release_after_call=offload == "cpu",
            ),
            embeddings_processor=_LazyCallable(
                lambda: LTX25EmbeddingsProcessor.from_checkpoints(
                    paths.transformer_path, paths.text_encoder_path, device=resolved_device, torch_dtype=dtype
                ),
                release_after_call=offload == "cpu",
            ),
            transformer=_LazyCallable(
                lambda: LTX25AVTransformer.from_checkpoint(
                    paths.transformer_path, device=resolved_device, torch_dtype=dtype
                )
            ),
            spatial_upsampler=_LazyCallable(
                lambda: LTX25SpatialUpsampler.from_checkpoint(
                    paths.spatial_upsampler_path, device=resolved_device, torch_dtype=dtype
                ),
                release_after_call=offload == "cpu",
            ),
            latent_statistics=load_video_latent_statistics(video_vae_path).to(device=resolved_device, dtype=dtype),
        )
        return cls(
            components,
            device=resolved_device,
            dtype=dtype,
            capture_prompt_intermediates=capture_prompt_intermediates,
            video_encoder_path=video_vae_path,
            offload=offload,
        )

    @torch.inference_mode()
    def generate(self, request: LTX25ReferenceRequest) -> LTX25ReferenceResult:
        """Run the upstream-equivalent T2V/I2V diffusion and return unpatchified stage states."""
        request.validate()
        contexts, prompt_artifacts = self._encode_prompt(request.prompt)
        trace = LTX25ReferenceTrace(
            video_context=contexts.video_encoding.detach().clone(),
            audio_context=contexts.audio_encoding.detach().clone(),
            context_attention_mask=contexts.attention_mask.detach().clone(),
            artifacts=prompt_artifacts,
        )
        generator = torch.Generator(device=self.device).manual_seed(request.seed)

        stage1_video_tools, stage1_audio_tools = self._tools(request, half_resolution=True)
        stage1_video = self._apply_image_conditions(
            stage1_video_tools.create_initial_state(self.device, self.dtype),
            stage1_video_tools,
            request.images,
            request.height // 2,
            request.width // 2,
        )
        stage1_video = _noised_state(stage1_video, generator, 1.0)
        stage1_audio = _noised_state(stage1_audio_tools.create_initial_state(self.device, self.dtype), generator, 1.0)
        stage1_video, stage1_audio = self._sample_stage(
            stage_name="stage1",
            video=stage1_video,
            audio=stage1_audio,
            video_context=contexts.video_encoding,
            audio_context=contexts.audio_encoding,
            sigmas=distilled_sigmas(1, device=self.device),
            ancestral=True,
            seed=request.seed,
            trace=trace,
        )
        stage1_video = stage1_video_tools.unpatchify(stage1_video)
        stage1_audio = stage1_audio_tools.unpatchify(stage1_audio)

        upsampled = self.components.latent_statistics.normalize(
            self.components.spatial_upsampler(self.components.latent_statistics.un_normalize(stage1_video.latent[:1]))
        )
        trace.artifacts["upsampler_input"] = stage1_video.latent.detach().clone()
        trace.artifacts["upsampler_output"] = upsampled.detach().clone()
        stage2_video_tools, stage2_audio_tools = self._tools(request, half_resolution=False)
        stage2_sigmas = distilled_sigmas(2, device=self.device)
        stage2_video = self._apply_image_conditions(
            stage2_video_tools.create_initial_state(self.device, self.dtype, initial_latent=upsampled),
            stage2_video_tools,
            request.images,
            request.height,
            request.width,
        )
        stage2_video = _noised_state(
            stage2_video,
            generator,
            float(stage2_sigmas[0]),
        )
        stage2_audio = _noised_state(
            stage2_audio_tools.create_initial_state(self.device, self.dtype, initial_latent=stage1_audio.latent),
            generator,
            float(stage2_sigmas[0]),
        )
        stage2_video, stage2_audio = self._sample_stage(
            stage_name="stage2",
            video=stage2_video,
            audio=stage2_audio,
            video_context=contexts.video_encoding,
            audio_context=contexts.audio_encoding,
            sigmas=stage2_sigmas,
            ancestral=False,
            seed=request.seed,
            trace=trace,
        )
        stage2_video = stage2_video_tools.unpatchify(stage2_video)
        stage2_audio = stage2_audio_tools.unpatchify(stage2_audio)
        trace.artifacts["stage1_video_latent"] = stage1_video.latent.detach().clone()
        trace.artifacts["stage1_audio_latent"] = stage1_audio.latent.detach().clone()
        trace.artifacts["stage2_video_latent"] = stage2_video.latent.detach().clone()
        trace.artifacts["stage2_audio_latent"] = stage2_audio.latent.detach().clone()
        return LTX25ReferenceResult(
            stage1_video,
            stage1_audio,
            stage2_video,
            stage2_audio,
            generator.get_state(),
            trace,
        )

    def _encode_prompt(self, prompt: str) -> tuple[LTX25EmbeddingsProcessorOutput, dict[str, torch.Tensor]]:
        hidden_states, token_ids, attention_mask = self.components.text_encoder.encode([prompt])
        artifacts: dict[str, torch.Tensor] = {}
        if self.capture_prompt_intermediates:
            processor = self.components.embeddings_processor
            if isinstance(processor, _LazyCallable):
                processor = processor.resolve()  # type: ignore[assignment]
            feature_extractor = getattr(processor, "feature_extractor", None)
            if feature_extractor is not None:
                video_features, audio_features = feature_extractor(hidden_states, attention_mask)
                artifacts["video_features"] = video_features.detach().cpu()
                artifacts["audio_features"] = audio_features.detach().cpu()
            artifacts = {
                "gemma_token_ids": token_ids.detach().cpu(),
                "gemma_attention_mask": attention_mask.detach().cpu(),
                **{
                    f"gemma_hidden_state_{index}": hidden_state.detach().cpu()
                    for index, hidden_state in enumerate(hidden_states)
                },
                **artifacts,
            }
        return self.components.embeddings_processor(hidden_states, attention_mask), artifacts

    def _tools(
        self, request: LTX25ReferenceRequest, *, half_resolution: bool
    ) -> tuple[VideoLatentTools, AudioLatentTools]:
        height = request.height // 2 if half_resolution else request.height
        width = request.width // 2 if half_resolution else request.width
        pixel_shape = VideoPixelShape(1, request.num_frames, height, width, request.frame_rate)
        video_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
        audio_shape = AudioLatentShape.from_duration(1, request.num_frames / request.frame_rate)
        return (
            VideoLatentTools(VideoLatentPatchifier(patch_size=1), video_shape, request.frame_rate),
            AudioLatentTools(AudioPatchifier(patch_size=1), audio_shape),
        )

    def _apply_image_conditions(
        self,
        state: LatentState,
        tools: VideoLatentTools,
        conditions: Sequence[LTX25ReferenceImageCondition],
        height: int,
        width: int,
    ) -> LatentState:
        if not conditions:
            return state
        encoder, release_after = self._video_encoder()
        try:
            for condition in conditions:
                pixels = preprocess_ltx25_image(
                    condition.image,
                    height,
                    width,
                    self._image_crf(condition),
                    device=self.device,
                    dtype=self.dtype,
                )
                encoded = encoder(pixels)
                conditioning = (
                    VideoConditionByLatentIndex(encoded, condition.strength, 0)
                    if condition.frame_idx == 0
                    else VideoConditionByKeyframeIndex(encoded, condition.frame_idx, condition.strength)
                )
                state = conditioning.apply_to(state, tools)
            return state
        finally:
            if release_after and self.offload == "cpu":
                encoder.to("cpu")  # type: ignore[union-attr]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _image_crf(self, condition: LTX25ReferenceImageCondition) -> int:
        if condition.crf is not None:
            return condition.crf
        return default_image_crf(self.video_encoder_path)

    def _video_encoder(self) -> tuple[VideoEncoder, bool]:
        if self.components.video_encoder is not None:
            return self.components.video_encoder, False
        if self.video_encoder_path is None:
            raise RuntimeError("I2V reference generation requires a video encoder or video_encoder_path")
        from telefuser.models.ltx25.video_encoder import LTX25VideoEncoder

        return (
            LTX25VideoEncoder.from_checkpoint(self.video_encoder_path, device=self.device, torch_dtype=self.dtype),
            True,
        )

    def _sample_stage(  # noqa: PLR0913
        self,
        *,
        stage_name: str,
        video: LatentState,
        audio: LatentState,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        sigmas: torch.Tensor,
        ancestral: bool,
        seed: int,
        trace: LTX25ReferenceTrace,
    ) -> tuple[LatentState, LatentState]:
        trace.artifacts[f"{stage_name}_initial_video_noise"] = video.latent.detach().clone()
        trace.artifacts[f"{stage_name}_initial_audio_noise"] = audio.latent.detach().clone()
        stepper = LTX25EulerAncestralStep() if ancestral else None
        ancestral_generator = ancestral_noise_generator(seed, self.device) if ancestral else None

        for step_index in range(len(sigmas) - 1):
            denoised_video, denoised_audio = self.components.transformer(
                _modality(video, video_context, sigmas[step_index]),
                _modality(audio, audio_context, sigmas[step_index]),
                BatchedPerturbationConfig.empty(video.latent.shape[0]),
            )
            if denoised_video is None or denoised_audio is None:
                raise RuntimeError("LTX-2.5 joint T2V reference requires both video and audio x0 predictions")
            denoised_video = _post_process(denoised_video, video)
            denoised_audio = _post_process(denoised_audio, audio)
            trace.artifacts[f"{stage_name}_step{step_index}_video_x0"] = denoised_video.detach().clone()
            trace.artifacts[f"{stage_name}_step{step_index}_audio_x0"] = denoised_audio.detach().clone()

            if bool(sigmas[step_index + 1] == 0):
                if stepper is None:
                    trace.artifacts[f"{stage_name}_step{step_index}_updated_video_latent"] = (
                        denoised_video.detach().clone()
                    )
                    trace.artifacts[f"{stage_name}_step{step_index}_updated_audio_latent"] = (
                        denoised_audio.detach().clone()
                    )
                video = replace(video, latent=denoised_video.to(self.dtype))
                audio = replace(audio, latent=denoised_audio.to(self.dtype))
                continue
            if stepper is None:
                video_updated = _deterministic_euler_step(video.latent, denoised_video, sigmas, step_index)
                audio_updated = _deterministic_euler_step(audio.latent, denoised_audio, sigmas, step_index)
                trace.artifacts[f"{stage_name}_step{step_index}_updated_video_latent"] = video_updated.detach().clone()
                trace.artifacts[f"{stage_name}_step{step_index}_updated_audio_latent"] = audio_updated.detach().clone()
                video = replace(video, latent=video_updated)
                audio = replace(audio, latent=audio_updated)
                continue

            assert ancestral_generator is not None
            video_noise = torch.randn(
                video.latent.shape,
                generator=ancestral_generator,
                dtype=video.latent.dtype,
                device=video.latent.device,
            )
            audio_noise = torch.randn(
                audio.latent.shape,
                generator=ancestral_generator,
                dtype=audio.latent.dtype,
                device=audio.latent.device,
            )
            trace.artifacts[f"{stage_name}_step{step_index}_ancestral_noise_video"] = video_noise.detach().clone()
            trace.artifacts[f"{stage_name}_step{step_index}_ancestral_noise_audio"] = audio_noise.detach().clone()
            video_updated = stepper.step(video.latent.float(), denoised_video, sigmas, step_index, video_noise)
            audio_updated = stepper.step(audio.latent.float(), denoised_audio, sigmas, step_index, audio_noise)
            trace.artifacts[f"{stage_name}_step{step_index}_updated_video_latent"] = video_updated.detach().clone()
            trace.artifacts[f"{stage_name}_step{step_index}_updated_audio_latent"] = audio_updated.detach().clone()
            video = replace(video, latent=_post_process(video_updated, video).to(self.dtype))
            audio = replace(audio, latent=_post_process(audio_updated, audio).to(self.dtype))
        return video, audio


__all__ = [
    "LTX25DistilledReference",
    "LTX25ReferenceComponents",
    "LTX25ReferenceImageCondition",
    "LTX25ReferenceRequest",
    "LTX25ReferenceResult",
    "LTX25ReferenceTrace",
]
