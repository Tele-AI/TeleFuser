# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 local T2VA, FL2VA, and Ref2VA pipeline."""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoProcessor

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.worker import ParallelWorker, WorkerTensorChannel

from .data import (
    minimax_h3_validate_canonical_request,
    minimax_h3_validate_reference_media_facts,
)
from .denoising import MiniMaxH3DenoisingStage
from .material_io import (
    MiniMaxH3MaterialFacts,
    minimax_h3_localize_material,
    minimax_h3_probe_material,
)
from .resolved_plan import (
    MiniMaxH3ResolvedPlan,
    minimax_h3_resolve_plan,
    minimax_h3_resolve_spatial_shape,
)
from .text_encoding import MiniMaxH3TextEncodingStage
from .vae import MiniMaxH3AudioVAEStage, MiniMaxH3PreparedCondition, MiniMaxH3VideoVAEStage


def _fp32_runtime_config() -> ModelRuntimeConfig:
    return ModelRuntimeConfig(torch_dtype=torch.float32)


@dataclass
class MiniMaxH3PipelineConfig:
    processor_path: str
    text_encoder_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    video_vae_config: ModelRuntimeConfig = field(default_factory=_fp32_runtime_config)
    audio_vae_config: ModelRuntimeConfig = field(default_factory=_fp32_runtime_config)
    num_inference_steps: int = 50

    def __post_init__(self) -> None:
        if not self.processor_path:
            raise ValueError("processor_path is required")
        if self.num_inference_steps < 2:
            raise ValueError("num_inference_steps must be at least 2")


@dataclass(frozen=True)
class MiniMaxH3Generation:
    video: torch.Tensor
    audio: torch.Tensor
    video_fps: int
    audio_sample_rate: int
    plan: MiniMaxH3ResolvedPlan
    packed_sequence_length: int
    runtime_metrics: dict[str, float | int]


class MiniMaxH3Pipeline(BasePipeline):
    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.config: MiniMaxH3PipelineConfig | None = None
        self.text_stage: MiniMaxH3TextEncodingStage | ParallelWorker | None = None
        self.video_vae_stage: MiniMaxH3VideoVAEStage | ParallelWorker | None = None
        self.audio_vae_stage: MiniMaxH3AudioVAEStage | None = None
        self.denoising_stage: MiniMaxH3DenoisingStage | ParallelWorker | None = None
        self._worker_tensor_channels: list[WorkerTensorChannel] = []
        self._uses_direct_text_handoff = False
        self._uses_direct_visual_handoff = False
        self._uses_direct_video_latent_handoff = False

    def init(self, module_manager: ModuleManager, config: MiniMaxH3PipelineConfig) -> None:
        self.config = config
        processor = AutoProcessor.from_pretrained(
            config.processor_path,
            local_files_only=True,
            trust_remote_code=False,
        )
        text_stage = MiniMaxH3TextEncodingStage(
            module_manager,
            config.text_encoder_config,
            processor=processor,
        )
        video_vae_stage = MiniMaxH3VideoVAEStage(
            module_manager,
            config.video_vae_config,
        )
        self.audio_vae_stage = MiniMaxH3AudioVAEStage(module_manager, config.audio_vae_config)
        denoising_stage = MiniMaxH3DenoisingStage(module_manager, config.dit_config)

        standalone_workers = not dist.is_initialized()
        text_parallel = standalone_workers and config.text_encoder_config.parallel_config.world_size > 1
        video_vae_parallel = standalone_workers and config.video_vae_config.parallel_config.world_size > 1
        denoising_parallel = standalone_workers and config.dit_config.parallel_config.world_size > 1
        text_to_denoise_channel = None
        visual_to_denoise_channel = None
        denoise_to_video_vae_channel = None
        if denoising_parallel:
            denoise_world_size = config.dit_config.parallel_config.world_size
            if text_parallel:
                text_to_denoise_channel = WorkerTensorChannel(
                    denoise_world_size,
                    timeout=max(
                        config.text_encoder_config.parallel_config.timeout,
                        config.dit_config.parallel_config.timeout,
                    ),
                )
                self._worker_tensor_channels.append(text_to_denoise_channel)
            if video_vae_parallel:
                visual_to_denoise_channel = WorkerTensorChannel(
                    denoise_world_size,
                    timeout=max(
                        config.video_vae_config.parallel_config.timeout,
                        config.dit_config.parallel_config.timeout,
                    ),
                )
                denoise_to_video_vae_channel = WorkerTensorChannel(
                    config.video_vae_config.parallel_config.world_size,
                    timeout=max(
                        config.video_vae_config.parallel_config.timeout,
                        config.dit_config.parallel_config.timeout,
                    ),
                )
                self._worker_tensor_channels.extend((visual_to_denoise_channel, denoise_to_video_vae_channel))

        if text_parallel:
            self.text_stage = ParallelWorker(
                text_stage,
                tensor_output_channel=text_to_denoise_channel,
                tensor_output_methods=("encode_for_denoising",) if text_to_denoise_channel is not None else (),
            )
        else:
            self.text_stage = text_stage
            if dist.is_initialized():
                text_stage.parallel_models()
        if video_vae_parallel:
            self.video_vae_stage = ParallelWorker(
                video_vae_stage,
                tensor_output_channel=visual_to_denoise_channel,
                tensor_output_methods=("encode_visual_for_denoising",) if visual_to_denoise_channel is not None else (),
                tensor_input_channels=(denoise_to_video_vae_channel,)
                if denoise_to_video_vae_channel is not None
                else (),
            )
        else:
            self.video_vae_stage = video_vae_stage
            if dist.is_initialized():
                video_vae_stage.parallel_models()
        if denoising_parallel:
            self.denoising_stage = ParallelWorker(
                denoising_stage,
                tensor_output_channel=denoise_to_video_vae_channel,
                tensor_output_methods=("denoise_for_video_vae",) if denoise_to_video_vae_channel is not None else (),
                tensor_input_channels=tuple(
                    channel for channel in (text_to_denoise_channel, visual_to_denoise_channel) if channel is not None
                ),
            )
        else:
            self.denoising_stage = denoising_stage
            if dist.is_initialized():
                denoising_stage.parallel_models()
        self._uses_direct_text_handoff = text_to_denoise_channel is not None
        self._uses_direct_visual_handoff = visual_to_denoise_channel is not None
        self._uses_direct_video_latent_handoff = denoise_to_video_vae_channel is not None
        self._model_info = module_manager.get_model_info()

    def _get_stages(self) -> list[object]:
        return [
            stage
            for stage in (self.text_stage, self.video_vae_stage, self.audio_vae_stage, self.denoising_stage)
            if stage is not None
        ]

    def stop(self) -> None:
        # Release the two producer-to-DiT IPC mappings before their producer
        # workers exit. The DiT-to-VAE channel makes fully consumer-first
        # shutdown impossible without a worker-level release command.
        stages = (self.denoising_stage, self.text_stage, self.video_vae_stage, self.audio_vae_stage)
        for stage in stages:
            if stage is None:
                continue
            close = getattr(stage, "close", None)
            if callable(close):
                close()
        for channel in self._worker_tensor_channels:
            channel.close()

    @staticmethod
    def _resolve_stage_result(value: Any) -> Any:
        return value() if callable(value) else value

    def _synchronize(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "npu":
            torch.npu.synchronize(device)

    def _timed_call(self, function: Any, /, *args: Any, **kwargs: Any) -> tuple[Any, float]:
        self._synchronize()
        started = time.perf_counter()
        value = function(*args, **kwargs)
        value = self._resolve_stage_result(value)
        self._synchronize()
        return value, time.perf_counter() - started

    @staticmethod
    def _resolve_deferred_plan(
        canonical: dict[str, Any],
        facts: dict[int, MiniMaxH3MaterialFacts],
    ) -> tuple[dict[str, Any], MiniMaxH3ResolvedPlan]:
        MiniMaxH3Pipeline._validate_reference_start_times(canonical, facts)
        if canonical["target"].get("duration_seconds") is None:
            duration_sources = [
                (index, condition)
                for index, condition in enumerate(canonical["conditions"])
                if condition["type"] in {"audio", "video", "video_audio"} and facts[index].has_audio
            ]
            if len(duration_sources) != 1:
                raise ValueError(
                    "audio-derived target duration requires exactly one probed "
                    f"condition with an audio stream, got {len(duration_sources)}"
                )
            index, condition = duration_sources[0]
            duration = facts[index].audio_duration_seconds
            if duration is None:
                raise ValueError(f"conditions[{index}] has no positive probed audio duration")
            effective = duration - float(condition.get("start_time_seconds", 0.0))
            canonical = minimax_h3_validate_canonical_request(
                task=canonical["task"],
                prompt=canonical["prompt"],
                conditions=canonical["conditions"],
                target={**canonical["target"], "duration_seconds": effective},
                flow_shift=canonical.get("flow_shift"),
                audio_flow_shift=canonical.get("audio_flow_shift"),
                seed=canonical.get("seed"),
            )
        plan = minimax_h3_resolve_plan(canonical)
        if plan.shape.get("geometry") == "deferred":
            first = plan.materials[0]
            item_facts = facts[int(first.condition_index)]
            if item_facts.width is None or item_facts.height is None:
                raise ValueError("deferred FL2VA geometry requires image width and height")
            shape = dict(plan.shape)
            shape.update(
                minimax_h3_resolve_spatial_shape(
                    width=item_facts.width,
                    height=item_facts.height,
                )
            )
            plan = replace(plan, shape=shape)
        if plan.shape.get("geometry") != "resolved_v2":
            raise ValueError("MiniMax H3 target geometry must resolve before model execution")
        return canonical, plan

    @staticmethod
    def _validate_reference_start_times(
        canonical: dict[str, Any],
        facts: dict[int, MiniMaxH3MaterialFacts],
    ) -> None:
        for index, condition in enumerate(canonical["conditions"]):
            start = float(condition.get("start_time_seconds", 0.0))
            if start == 0.0:
                continue
            item_facts = facts[index]
            video_duration = item_facts.video_duration_seconds
            if video_duration is None or start >= video_duration:
                raise ValueError(f"conditions[{index}].start_time_seconds must be less than the video duration")
            if item_facts.has_audio:
                audio_duration = item_facts.audio_duration_seconds
                if audio_duration is None or start >= audio_duration:
                    raise ValueError(
                        f"conditions[{index}].start_time_seconds must be less than the soundtrack duration"
                    )

    @staticmethod
    def _condition_labels(
        prepared: list[MiniMaxH3PreparedCondition],
    ) -> list[tuple[str, int]]:
        counters = {"image": 0, "audio": 0, "video": 0}
        labels: list[tuple[str, int]] = []
        for condition in prepared:
            if condition.kind == "image":
                counters["image"] += 1
                labels.append(("image", counters["image"]))
            elif condition.kind == "audio":
                counters["audio"] += 1
                labels.append(("audio", counters["audio"]))
            elif condition.kind in {"video", "video_audio"}:
                if condition.has_audio:
                    counters["audio"] += 1
                    labels.append(("audio", counters["audio"]))
                counters["video"] += 1
                labels.append(("video", counters["video"]))
        return labels

    @staticmethod
    def _condition_transport_payload(condition: MiniMaxH3PreparedCondition) -> dict[str, Any]:
        return condition.as_denoising_payload()

    @torch.inference_mode()
    def __call__(
        self,
        *,
        task: str,
        prompt: str,
        conditions: list[dict[str, Any]] | None,
        target: dict[str, Any],
        seed: int | None = None,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
        num_inference_steps: int | None = None,
    ) -> MiniMaxH3Generation:
        if (
            self.config is None
            or self.text_stage is None
            or self.video_vae_stage is None
            or self.audio_vae_stage is None
            or self.denoising_stage is None
        ):
            raise RuntimeError("MiniMaxH3Pipeline.init must be called before generation")
        self._synchronize()
        pipeline_started = time.perf_counter()
        canonical = minimax_h3_validate_canonical_request(
            task=task,
            prompt=prompt,
            conditions=[] if conditions is None else conditions,
            target=target,
            seed=seed,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
        )
        with ExitStack() as stack:
            media_started = time.perf_counter()
            paths: dict[int, Path] = {}
            facts: dict[int, MiniMaxH3MaterialFacts] = {}
            for index, condition in enumerate(canonical["conditions"]):
                path = stack.enter_context(minimax_h3_localize_material(condition["uri"]))
                paths[index] = path
                facts[index] = minimax_h3_probe_material(path, condition["type"])
            if canonical["task"] == "ref2va":
                video_duration_facts = {
                    index: float(item.video_duration_seconds)
                    for index, item in facts.items()
                    if item.video_duration_seconds is not None
                }
                audio_duration_facts = {
                    index: float(item.audio_duration_seconds)
                    for index, item in facts.items()
                    if item.audio_duration_seconds is not None
                }
                minimax_h3_validate_reference_media_facts(
                    canonical["conditions"],
                    video_duration_seconds_by_condition=video_duration_facts,
                    audio_duration_seconds_by_condition=audio_duration_facts,
                )
            canonical, plan = self._resolve_deferred_plan(canonical, facts)
            prepared = MiniMaxH3VideoVAEStage.prepare_media(plan, paths, facts)
            self._synchronize()
            media_preparation_seconds = time.perf_counter() - media_started
            images = [item.image for item in prepared if item.image is not None]
            videos = [item.video_frames for item in prepared if item.video_frames is not None]
            text_method = (
                self.text_stage.encode_for_denoising if self._uses_direct_text_handoff else self.text_stage.encode
            )
            text, text_encoding_seconds = self._timed_call(
                text_method,
                task=plan.task,
                prompt=plan.prompt,
                images=images,
                videos=videos,
                condition_labels=self._condition_labels(prepared),
            )
            visual_encoding_seconds = 0.0
            condition_payloads: list[dict[str, Any]] | None = None
            if any(item.image is not None or item.video_frames is not None for item in prepared):
                if self._uses_direct_visual_handoff:
                    condition_payloads, visual_encoding_seconds = self._timed_call(
                        self.video_vae_stage.encode_visual_for_denoising, prepared
                    )
                else:
                    prepared, visual_encoding_seconds = self._timed_call(self.video_vae_stage.encode_visual, prepared)
            duration_seconds = float(plan.shape["frame_count"]) / float(plan.shape["fps"])
            audio_encoding_seconds = 0.0
            if any(item.has_audio for item in prepared):
                prepared, audio_encoding_seconds = self._timed_call(
                    self.audio_vae_stage.encode_audio, prepared, paths, facts, duration_seconds
                )
            if self._uses_direct_visual_handoff:
                if condition_payloads is None:
                    condition_payloads = [self._condition_transport_payload(condition) for condition in prepared]
                else:
                    condition_payloads = [
                        {
                            **payload,
                            "audio_rows": condition.audio_rows,
                            "ref_audio_t": condition.ref_audio_t,
                            "has_audio": condition.has_audio,
                        }
                        for payload, condition in zip(condition_payloads, prepared, strict=True)
                    ]
                denoising_conditions: list[MiniMaxH3PreparedCondition] | list[dict[str, Any]] = condition_payloads
            else:
                denoising_conditions = prepared
            steps = self.config.num_inference_steps if num_inference_steps is None else num_inference_steps
            if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
                raise ValueError("num_inference_steps must be an integer of at least 2")
            if self._uses_direct_video_latent_handoff:
                transported, denoising_stage_seconds = self._timed_call(
                    self.denoising_stage.denoise_for_video_vae,
                    plan=plan,
                    text=text,
                    conditions=denoising_conditions,
                    num_inference_steps=steps,
                )
                video_latent = transported["video_latent"]
                denoised = transported["remainder"]
            else:
                denoised, denoising_stage_seconds = self._timed_call(
                    self.denoising_stage.denoise,
                    plan=plan,
                    text=text,
                    conditions=denoising_conditions,
                    num_inference_steps=steps,
                )
                video_latent = denoised.video_latent
            video_device, video_decoding_seconds = self._timed_call(self.video_vae_stage.decode_video, video_latent)
            # Materialize exactly once in the caller. For a parallel VAE worker,
            # video_device arrives through CUDA IPC without routing 1.5 GB of
            # FP32 frames through multiprocessing shared-memory serialization.
            video = video_device.cpu()
            audio, audio_decoding_seconds = self._timed_call(self.audio_vae_stage.decode_audio, denoised.audio_latent)
            pipeline_seconds = time.perf_counter() - pipeline_started
            runtime_metrics = {
                **denoised.runtime_metrics,
                "media_preparation_seconds": media_preparation_seconds,
                "text_encoding_seconds": text_encoding_seconds,
                "visual_encoding_seconds": visual_encoding_seconds,
                "audio_encoding_seconds": audio_encoding_seconds,
                "denoising_stage_seconds": denoising_stage_seconds,
                "video_decoding_seconds": video_decoding_seconds,
                "audio_decoding_seconds": audio_decoding_seconds,
                "pipeline_seconds": pipeline_seconds,
            }
            return MiniMaxH3Generation(
                video=video[:, : int(plan.shape["frame_count"])],
                audio=audio,
                video_fps=24,
                audio_sample_rate=32_000,
                plan=plan,
                packed_sequence_length=int(denoised.packed["seq_len"]),
                runtime_metrics=runtime_metrics,
            )


__all__ = [
    "MiniMaxH3Generation",
    "MiniMaxH3Pipeline",
    "MiniMaxH3PipelineConfig",
]
