from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_ulysses_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.models.lingbot_world_fast_dit import LingBotWorldFastDiT
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.logging import logger


def _select_timesteps(
    scheduler: FlowUniPCMultistepScheduler,
    indices: tuple[int, ...],
    shift: float,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    if not indices:
        raise ValueError("timestep indices must not be empty")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise ValueError(f"timestep indices must be integers, got {indices!r}")
    if any(index < 0 or index >= num_train_timesteps for index in indices):
        raise ValueError(f"timestep indices must be in [0, {num_train_timesteps}), got {indices!r}")
    if tuple(sorted(indices)) != indices or len(set(indices)) != len(indices):
        raise ValueError(f"timestep indices must be strictly increasing, got {indices!r}")

    scheduler.set_timesteps(num_train_timesteps, shift=shift)
    if max(indices) >= len(scheduler.timesteps):
        raise ValueError(f"timestep index exceeds scheduler output: {indices!r}")
    return scheduler.timesteps[list(indices)].clone()


@dataclass
class _DenoisingCacheState:
    scheduler: FlowUniPCMultistepScheduler
    timesteps: torch.Tensor
    self_kv_cache: list[dict[str, torch.Tensor | int]]
    crossattn_cache: list[dict[str, torch.Tensor | bool]]
    generator: torch.Generator


class LingBotWorldFastDenoisingStage(BaseStage):
    """Chunk-level denoising stage with worker-local persistent KV caches."""

    def __init__(
        self,
        name: str,
        dit_model: LingBotWorldFastDiT,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.dit = dit_model
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.model_names = ["dit"]
        self._cache_registry: dict[int, _DenoisingCacheState] = {}

    def parallel_models(self) -> None:
        """Configure Ulysses SP and optional FSDP inside a ParallelWorker."""
        parallel_config = self.model_runtime_config.parallel_config
        self.dit.device_mesh = create_device_mesh_from_config(parallel_config)
        self.dit.set_attention_config(self.model_runtime_config.attention_config)
        if parallel_config.sp_ulysses_degree > 1:
            self.dit.enable_usp(self.dit.device_mesh)
        if parallel_config.enable_fsdp:
            logger.info(f"Enabling FSDP for {self.name}")
            self.dit = shard_model(
                module=self.dit,
                device_id=self.device,
                wrap_module_names=self.dit.get_fsdp_module_names(),
                param_dtype=self.torch_dtype,
                reduce_dtype=self.torch_dtype,
                buffer_dtype=self.torch_dtype,
            )
            self.onload_models_flag = True

    def _init_self_kv_cache(
        self,
        batch_size: int,
        kv_size: int,
    ) -> list[dict[str, torch.Tensor | int]]:
        head_dim = self.dit.dim // self.dit.num_heads
        ulysses_world_size = get_ulysses_world_size(getattr(self.dit, "device_mesh", None))
        num_heads = self.dit.num_heads
        if ulysses_world_size > 1:
            num_heads = (num_heads + ulysses_world_size - 1) // ulysses_world_size
        shape = (batch_size, kv_size, num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "global_end_index": 0,
                "local_end_index": 0,
            }
            for _ in range(self.dit.num_layers)
        ]

    def _init_crossattn_cache(
        self,
        batch_size: int,
        max_sequence_length: int,
    ) -> list[dict[str, torch.Tensor | bool]]:
        head_dim = self.dit.dim // self.dit.num_heads
        shape = (batch_size, max_sequence_length, self.dit.num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "is_init": False,
            }
            for _ in range(self.dit.num_layers)
        ]

    @with_model_offload(["dit"])
    def initialize_cache(
        self,
        cache_handle: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
        sample_shift: float,
        generator_state: list[int],
        timestep_indices: tuple[int, ...] = (0, 179, 358, 679),
    ) -> bool:
        """Atomically register session-scoped KV, scheduler, and RNG state."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"Cache handle {cache_handle} is already registered")

        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        timesteps = _select_timesteps(scheduler, tuple(timestep_indices), sample_shift)
        generator = torch.Generator(device=self.device)
        generator.set_state(torch.tensor(generator_state, dtype=torch.uint8))
        state = _DenoisingCacheState(
            scheduler=scheduler,
            timesteps=timesteps,
            self_kv_cache=self._init_self_kv_cache(batch_size, kv_size),
            crossattn_cache=self._init_crossattn_cache(batch_size, max_sequence_length),
            generator=generator,
        )
        self._cache_registry[cache_handle] = state
        return True

    @staticmethod
    def _convert_flow_pred_to_x0(
        flow_pred: torch.Tensor,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
    ) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps - timestep.double()).abs())
        sigma_t = sigmas[timestep_id].reshape(-1)
        while sigma_t.ndim < xt.ndim:
            sigma_t = sigma_t.unsqueeze(-1)
        x0 = xt - sigma_t * flow_pred
        return x0.to(original_dtype)

    def denoise_chunk(
        self,
        latent_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        prompt_emb: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
        control_chunk: torch.Tensor | None,
        self_kv_cache: list[dict[str, torch.Tensor | int]],
        crossattn_cache: list[dict[str, torch.Tensor | bool]],
        current_start: int,
        max_attention_size: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        current_latent = latent_chunk
        for timestep_idx in range(len(timesteps)):
            schedule_timestep = timesteps[timestep_idx].view(1).to(device=current_latent.device)
            model_timestep = schedule_timestep.to(dtype=torch.float32)
            with torch.amp.autocast(
                current_latent.device.type,
                dtype=self.torch_dtype,
                enabled=current_latent.device.type == "cuda",
            ):
                noise_pred = self.dit(
                    x=current_latent.to(dtype=self.torch_dtype),
                    timestep=model_timestep,
                    context=prompt_emb,
                    y=condition_chunk,
                    control_tensor=control_chunk,
                    kv_cache=self_kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    max_attention_size=max_attention_size,
                )
            x0 = self._convert_flow_pred_to_x0(noise_pred, current_latent, schedule_timestep[0], scheduler)
            if timestep_idx < len(timesteps) - 1:
                next_timestep = timesteps[timestep_idx + 1].view(1).to(device=x0.device)
                noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                current_latent = scheduler.add_noise(x0, noise, next_timestep)
            else:
                current_latent = x0

        logger.debug("LingBotWorldFast chunk denoised")
        return current_latent

    @with_model_offload(["dit"])
    def denoise_and_update_cache(
        self,
        cache_handle: int,
        latent_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        prompt_emb: torch.Tensor,
        control_chunk: torch.Tensor | None,
        current_start: int,
        max_attention_size: int,
        chunk_id: int | None = None,
        return_profile: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
        """Denoise a chunk and commit its clean KV state inside each worker."""
        try:
            state = self._cache_registry[cache_handle]
        except KeyError as exc:
            raise KeyError(f"Unknown cache handle {cache_handle}") from exc

        device = torch.device(self.device)
        use_cuda_events = return_profile and device.type == "cuda" and torch.cuda.is_available()
        profile: dict[str, object] = {"chunk_id": chunk_id} if return_profile else {}
        denoise_start_event = denoise_end_event = None
        kv_start_event = kv_end_event = None
        if use_cuda_events:
            with torch.cuda.device(device):
                denoise_start_event = torch.cuda.Event(enable_timing=True)
                denoise_end_event = torch.cuda.Event(enable_timing=True)
                kv_start_event = torch.cuda.Event(enable_timing=True)
                kv_end_event = torch.cuda.Event(enable_timing=True)
                denoise_start_event.record()

        denoise_start_ns = time.perf_counter_ns()
        denoised = self.denoise_chunk(
            latent_chunk=latent_chunk,
            condition_chunk=condition_chunk,
            prompt_emb=prompt_emb,
            timesteps=state.timesteps,
            scheduler=state.scheduler,
            control_chunk=control_chunk,
            self_kv_cache=state.self_kv_cache,
            crossattn_cache=state.crossattn_cache,
            current_start=current_start,
            max_attention_size=max_attention_size,
            generator=state.generator,
        )
        denoise_end_ns = time.perf_counter_ns()
        if use_cuda_events and denoise_end_event is not None and kv_start_event is not None:
            denoise_end_event.record()
            kv_start_event.record()

        kv_start_ns = time.perf_counter_ns()
        with torch.amp.autocast(
            self.device.type,
            dtype=self.torch_dtype,
            enabled=self.device.type == "cuda",
        ):
            self.dit(
                x=denoised.to(dtype=self.torch_dtype),
                timestep=torch.zeros((1,), dtype=torch.float32, device=self.device),
                context=prompt_emb,
                y=condition_chunk,
                control_tensor=control_chunk,
                kv_cache=state.self_kv_cache,
                crossattn_cache=state.crossattn_cache,
                current_start=current_start,
                max_attention_size=max_attention_size,
            )
        kv_end_ns = time.perf_counter_ns()

        if return_profile:
            profile.update(
                {
                    "dit_denoise_ms": (denoise_end_ns - denoise_start_ns) / 1_000_000.0,
                    "kv_update_ms": (kv_end_ns - kv_start_ns) / 1_000_000.0,
                    "dit_total_ms": (kv_end_ns - denoise_start_ns) / 1_000_000.0,
                }
            )
            if use_cuda_events and all(
                event is not None for event in (denoise_start_event, denoise_end_event, kv_start_event, kv_end_event)
            ):
                assert denoise_start_event is not None
                assert denoise_end_event is not None
                assert kv_start_event is not None
                assert kv_end_event is not None
                kv_end_event.record()
                kv_end_event.synchronize()
                profile.update(
                    {
                        "dit_denoise_gpu_ms": denoise_start_event.elapsed_time(denoise_end_event),
                        "kv_update_gpu_ms": kv_start_event.elapsed_time(kv_end_event),
                        "dit_total_gpu_ms": denoise_start_event.elapsed_time(kv_end_event),
                    }
                )
            if chunk_id is not None:
                logger.info(
                    "lingbot_async_vae dit_end "
                    f"chunk_id={chunk_id} dit_total_ms={profile.get('dit_total_ms'):.3f} "
                    f"kv_update_ms={profile.get('kv_update_ms'):.3f} "
                    f"dit_total_gpu_ms={profile.get('dit_total_gpu_ms')} "
                    f"kv_update_gpu_ms={profile.get('kv_update_gpu_ms')}"
                )
            return denoised, profile
        return denoised

    def has_cache(self, cache_handle: int) -> bool:
        """Return whether this worker owns the requested cache handle."""
        return cache_handle in self._cache_registry

    def list_cache_handles(self) -> tuple[int, ...]:
        """Return registered cache handles for diagnostics and tests."""
        return tuple(sorted(self._cache_registry))

    def release_cache(self, cache_handle: int) -> bool:
        """Idempotently release worker-local state for one generation session."""
        return self._cache_registry.pop(cache_handle, None) is not None
