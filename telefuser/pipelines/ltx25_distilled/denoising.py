"""Two-phase distilled denoising for LTX-2.5."""

from __future__ import annotations

import torch

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import ModelRuntimeConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed import create_device_mesh_from_config
from telefuser.distributed.fsdp import shard_model_fsdp2_inference
from telefuser.models.ltx25 import LTX25AVTransformer
from telefuser.models.ltx25.sampler import LTX25_STAGE1_DISTILLED_SIGMAS, LTX25_STAGE2_DISTILLED_SIGMAS
from telefuser.offload import AsyncOffloadManager
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger

from .core import LTX25SimpleDenoiser, euler_ancestral_denoising_loop, euler_denoising_loop
from .latent import LatentState

LatentStateInput = LatentState | dict[str, torch.Tensor | None]


def _restore_latent_state(state: LatentStateInput) -> LatentState:
    if isinstance(state, LatentState):
        return state
    return LatentState(
        latent=state["latent"],
        denoise_mask=state["denoise_mask"],
        positions=state["positions"],
        clean_latent=state["clean_latent"],
        attention_mask=state.get("attention_mask"),
        keyframes_mask=state.get("keyframes_mask"),
    )


class LTX25DenoisingStage(BaseStage):
    def __init__(self, module_manager: ModuleManager, config: ModelRuntimeConfig) -> None:
        super().__init__("ltx25_denoising", config)
        self.transformer: LTX25AVTransformer = module_manager.fetch_module("ltx25_transformer")
        self.transformer.set_attention_config(config.attention_config)
        self.model_names = ["transformer"]
        self.empty_cache_after_call = False
        self.offload_manager: AsyncOffloadManager | None = None
        if config.offload_config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD:
            self.offload_manager = AsyncOffloadManager(
                self.transformer.velocity_model.transformer_blocks,
                device=self.device,
                pin_cpu_memory=config.offload_config.pin_cpu_memory,
                offload_ratio=config.offload_config.offload_ratio,
                prefetch_size=config.offload_config.prefetch_size,
            )
            self.transformer.to(device=self.device, dtype=self.torch_dtype)
            self.onload_models_flag = True

    def parallel_models(self) -> None:
        """Configure Ulysses SP and optional block-level FSDP2 for denoising."""
        parallel_config = self.model_runtime_config.parallel_config
        unsupported = {
            "dp_degree": parallel_config.dp_degree,
            "cfg_degree": parallel_config.cfg_degree,
            "sp_ring_degree": parallel_config.sp_ring_degree,
            "pp_degree": parallel_config.pp_degree,
            "tp_degree": parallel_config.tp_degree,
        }
        invalid = {name: degree for name, degree in unsupported.items() if degree != 1}
        if invalid:
            raise NotImplementedError(f"LTX-2.5 does not support these parallel degrees: {invalid}")
        device_mesh = create_device_mesh_from_config(parallel_config, device_type=self.device.type)
        self.transformer.set_attention_config(self.model_runtime_config.attention_config)
        if parallel_config.sp_ulysses_degree > 1:
            self.transformer.enable_usp(device_mesh)
            logger.info(f"enabled LTX-2.5 Ulysses SP degree={parallel_config.sp_ulysses_degree}")
        if parallel_config.enable_fsdp:
            if self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
                raise ValueError("LTX-2.5 FSDP inference cannot be combined with model CPU offload")
            logger.info(f"enabled LTX-2.5 block FSDP2 for {self.name}")
            self.transformer.velocity_model = shard_model_fsdp2_inference(
                module=self.transformer.velocity_model,
                device_mesh=device_mesh,
                wrap_module_names=self.transformer.get_fsdp_module_names(),
            )
            self.onload_models_flag = True
            current_platform.empty_cache()

    def _onload(self) -> None:
        if not self.onload_models_flag:
            self.transformer.to(self.device)
            self.onload_models_flag = True

    def _offload_between_phases(self) -> None:
        if self.model_runtime_config.offload_config.offload_type == WeightOffloadType.MODEL_CPU_OFFLOAD:
            self.transformer.cpu()
            self.onload_models_flag = False

    @torch.inference_mode()
    def denoise_stage1(
        self,
        video: LatentStateInput,
        audio: LatentStateInput,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        *,
        noise_seed: int,
    ) -> tuple[LatentState, LatentState]:
        self._onload()
        video = _restore_latent_state(video)
        audio = _restore_latent_state(audio)
        result = euler_ancestral_denoising_loop(
            torch.tensor(LTX25_STAGE1_DISTILLED_SIGMAS, device=self.device),
            video,
            audio,
            self.transformer,
            LTX25SimpleDenoiser(video_context, audio_context),
            noise_seed=noise_seed,
            model_dtype=self.torch_dtype,
        )
        self._offload_between_phases()
        if result[0] is None or result[1] is None:
            raise RuntimeError("LTX-2.5 stage-one denoising requires video and audio outputs")
        if self.model_runtime_config.parallel_config.world_size > 1:
            return result[0].clone(), result[1].clone()
        return result[0], result[1]

    @torch.inference_mode()
    def denoise_stage2(
        self,
        video: LatentStateInput,
        audio: LatentStateInput,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
    ) -> tuple[LatentState, LatentState]:
        self._onload()
        video = _restore_latent_state(video)
        audio = _restore_latent_state(audio)
        result = euler_denoising_loop(
            torch.tensor(LTX25_STAGE2_DISTILLED_SIGMAS, device=self.device),
            video,
            audio,
            self.transformer,
            LTX25SimpleDenoiser(video_context, audio_context),
            model_dtype=self.torch_dtype,
        )
        if result[0] is None or result[1] is None:
            raise RuntimeError("LTX-2.5 stage-two denoising requires video and audio outputs")
        if self.model_runtime_config.parallel_config.world_size > 1:
            return result[0].clone(), result[1].clone()
        return result[0], result[1]

    def finish_request(self) -> None:
        if self.offload_manager is not None:
            self.offload_manager.release_all()
        elif self.model_runtime_config.offload_config.offload_type != WeightOffloadType.NO_CPU_OFFLOAD:
            self.transformer.cpu()
            self.onload_models_flag = False


__all__ = ["LTX25DenoisingStage"]
