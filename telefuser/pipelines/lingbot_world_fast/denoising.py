from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import torch

from telefuser.cache.session_memory import SessionSlotPool
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config, get_ulysses_world_size
from telefuser.distributed.fsdp import shard_model
from telefuser.models.lingbot_world_fast_dit import LingBotWorldFastDiT
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.logging import logger

from .vae_stage import LingBotWorldFastVAEDecodeStage


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
    crossattn_cache: list[dict[str, torch.Tensor | bool | int]]
    generator: torch.Generator
    noise_generator: torch.Generator
    noise_shape: tuple[int, int, int, int, int]
    prompt_emb: torch.Tensor | None = None
    image_condition_latent: torch.Tensor | None = None
    pool_slot: int | None = None
    projected_context_key: tuple[object, ...] | None = None
    prepared_control_key: tuple[object, ...] | None = None
    prepared_control_is_sharded: bool = False
    session_input_cache: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DenoisingCachePoolProfile:
    """Fixed cache-pool dimensions and storage size for one worker rank."""

    capacity: int
    batch_size: int
    kv_size: int
    max_sequence_length: int
    bytes_per_session: int
    allocated_bytes: int


class _DenoisingCachePool:
    """Preallocated rank-local KV storage split into reusable session slots."""

    def __init__(
        self,
        *,
        capacity: int,
        num_layers: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
        num_heads: int,
        local_num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.kv_size = kv_size
        self.max_sequence_length = max_sequence_length
        self._slots = SessionSlotPool(capacity, name="LingBot KV cache")

        self.self_k = torch.empty(
            (capacity, num_layers, batch_size, kv_size, local_num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.self_v = torch.empty_like(self.self_k)
        self.cross_k = torch.empty(
            (capacity, num_layers, batch_size, max_sequence_length, num_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        self.cross_v = torch.empty_like(self.cross_k)
        self.cursors = torch.zeros((capacity, num_layers, 2), dtype=torch.int64, device=device)

    @property
    def allocated_bytes(self) -> int:
        tensors = (self.self_k, self.self_v, self.cross_k, self.cross_v, self.cursors)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    @property
    def bytes_per_session(self) -> int:
        return self.allocated_bytes // self.capacity

    def acquire(
        self,
    ) -> tuple[int, list[dict[str, torch.Tensor | int]], list[dict[str, torch.Tensor | bool | int]]]:
        acquired = self.try_acquire()
        if acquired is None:
            raise RuntimeError(f"LingBot KV cache pool is full (capacity={self.capacity})")
        return acquired

    def try_acquire(
        self,
    ) -> tuple[int, list[dict[str, torch.Tensor | int]], list[dict[str, torch.Tensor | bool | int]]] | None:
        """Acquire reusable KV views without raising on capacity exhaustion."""
        slot = self._slots.try_acquire()
        if slot is None:
            return None
        self.cursors[slot].zero_()
        self_cache = [
            {
                "k": self.self_k[slot, layer],
                "v": self.self_v[slot, layer],
                "global_end_index": self.cursors[slot, layer, 0],
                "local_end_index": self.cursors[slot, layer, 1],
                "host_indices": {"global_end_index": 0, "local_end_index": 0},
            }
            for layer in range(self.num_layers)
        ]
        cross_cache = [
            {
                "k": self.cross_k[slot, layer],
                "v": self.cross_v[slot, layer],
                "is_init": False,
                "sequence_length": 0,
            }
            for layer in range(self.num_layers)
        ]
        return slot, self_cache, cross_cache

    def release(self, slot: int) -> None:
        self._slots.release(slot)


class LingBotWorldFastDenoisingStage(BaseStage):
    """Chunk-level denoising stage with worker-local persistent KV caches."""

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ) -> None:
        super().__init__(name, model_runtime_config)
        self.dit: LingBotWorldFastDiT = module_manager.fetch_module("lingbot_world_fast_dit")
        if self.dit is None:
            raise ValueError("LingBot denoising stage requires a loaded lingbot_world_fast_dit module")
        self.dit.set_attention_config(model_runtime_config.attention_config)
        self.model_names = ["dit"]
        # This worker alternates persistent DiT and VAE decode calls. Keeping the
        # allocator cache avoids driver allocations between chunks; cached blocks
        # remain reclaimable and do not change model or session-cache capacity.
        self.empty_cache_after_call = False
        self._cache_registry: dict[int, _DenoisingCacheState] = {}
        self._cache_pool: _DenoisingCachePool | None = None
        self._observed_condition_bytes = 0
        self._vae_decode_stage: LingBotWorldFastVAEDecodeStage | None = None
        self._pending_vae_decode_latents: dict[int, deque[torch.Tensor]] = {}
        if model_runtime_config.parallel_config.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info(f"Enabling torch.compile for {self.name}")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

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
        if self.model_runtime_config.compile_config.enabled:
            logger.info(f"Enabling torch.compile for {self.name}")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())
        if self._vae_decode_stage is not None:
            self._vae_decode_stage.device = self.device
            self._vae_decode_stage.model_runtime_config.device_id = self.device.index or 0
            self._vae_decode_stage.parallel_models()

    def attach_vae_decode_stage(self, stage: LingBotWorldFastVAEDecodeStage) -> None:
        """Attach a VAE decoder that shares this worker's process group and CUDA context."""
        if self._vae_decode_stage is not None:
            raise RuntimeError("a VAE decode stage is already attached")
        self._vae_decode_stage = stage

    def _require_vae_decode_stage(self) -> LingBotWorldFastVAEDecodeStage:
        if self._vae_decode_stage is None:
            raise RuntimeError("no VAE decode stage is attached")
        return self._vae_decode_stage

    def reset_vae_decode_device_memory_peak(self) -> bool:
        return self._require_vae_decode_stage().reset_device_memory_peak()

    def vae_decode_device_memory_snapshots(self) -> list[dict[str, int | str]]:
        return self._require_vae_decode_stage().device_memory_snapshots()

    def estimate_vae_decode_session_cache_bytes(self) -> int:
        return self._require_vae_decode_stage().estimate_session_cache_bytes()

    def observed_vae_decode_session_cache_bytes(self) -> int:
        return self._require_vae_decode_stage().observed_session_cache_bytes()

    def configure_vae_decode_cache_pool(self, capacity: int):
        return self._require_vae_decode_stage().configure_cache_pool(capacity)

    def initialize_vae_decode_cache(self, cache_handle: int) -> bool:
        return self._require_vae_decode_stage().initialize_cache(cache_handle)

    def decode_chunk(
        self,
        cache_handle: int,
        latents: torch.Tensor | None,
        is_first_clip: bool,
        is_last_clip: bool,
        _benchmark_profile: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        pending_latents = self._pending_vae_decode_latents.get(cache_handle)
        local_latents = pending_latents.popleft() if pending_latents else None
        decode_latents = local_latents if local_latents is not None else latents
        if decode_latents is None:
            raise RuntimeError(f"No worker-local VAE latent is available for cache handle {cache_handle}")
        if pending_latents is not None and not pending_latents:
            self._pending_vae_decode_latents.pop(cache_handle)
        profile_kwargs = {"_benchmark_profile": True} if _benchmark_profile else {}
        return self._require_vae_decode_stage().decode_chunk(
            cache_handle,
            decode_latents,
            is_first_clip,
            is_last_clip,
            **profile_kwargs,
        )

    def release_vae_decode_cache(self, cache_handle: int) -> bool:
        return self._require_vae_decode_stage().release_cache(cache_handle)

    def _init_self_kv_cache(
        self,
        batch_size: int,
        kv_size: int,
    ) -> list[dict[str, torch.Tensor | int]]:
        head_dim = self.dit.dim // self.dit.num_heads
        device_mesh = getattr(self.dit, "device_mesh", None)
        ulysses_world_size = get_ulysses_world_size(device_mesh)
        num_heads = self.dit.num_heads
        if ulysses_world_size > 1:
            if num_heads % ulysses_world_size:
                raise ValueError(
                    f"LingBot Ulysses SP requires {num_heads} attention heads to be divisible "
                    f"by degree {ulysses_world_size}"
                )
            num_heads //= ulysses_world_size

        shape = (batch_size, kv_size, num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                # These cursors cross actor/FSDP call boundaries.  Keep their
                # storage inside the cache so shallow argument copies retain
                # the updates made by the attention block.
                "global_end_index": torch.zeros((), dtype=torch.int64, device=self.device),
                "local_end_index": torch.zeros((), dtype=torch.int64, device=self.device),
                "host_indices": {"global_end_index": 0, "local_end_index": 0},
            }
            for _ in range(self.dit.num_layers)
        ]

    def _init_crossattn_cache(
        self,
        batch_size: int,
        max_sequence_length: int,
    ) -> list[dict[str, torch.Tensor | bool | int]]:
        head_dim = self.dit.dim // self.dit.num_heads
        shape = (batch_size, max_sequence_length, self.dit.num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "v": torch.zeros(shape, dtype=self.torch_dtype, device=self.device),
                "is_init": False,
                "sequence_length": 0,
            }
            for _ in range(self.dit.num_layers)
        ]

    def _cache_dimensions(self) -> tuple[int, int, int]:
        head_dim = self.dit.dim // self.dit.num_heads
        device_mesh = getattr(self.dit, "device_mesh", None)
        ulysses_world_size = get_ulysses_world_size(device_mesh)
        if self.dit.num_heads % ulysses_world_size:
            raise ValueError(
                f"LingBot attention heads {self.dit.num_heads} are not divisible by SP degree {ulysses_world_size}"
            )
        return head_dim, self.dit.num_heads, self.dit.num_heads // ulysses_world_size

    def estimate_session_cache_bytes(self, batch_size: int, kv_size: int, max_sequence_length: int) -> int:
        """Return exact persistent DiT KV bytes for one session on this rank."""
        head_dim, num_heads, local_num_heads = self._cache_dimensions()
        element_size = torch.empty((), dtype=self.torch_dtype).element_size()
        self_kv = 2 * self.dit.num_layers * batch_size * kv_size * local_num_heads * head_dim * element_size
        cross_kv = 2 * self.dit.num_layers * batch_size * max_sequence_length * num_heads * head_dim * element_size
        cursors = self.dit.num_layers * 2 * torch.empty((), dtype=torch.int64).element_size()
        return self_kv + cross_kv + cursors + getattr(self, "_observed_condition_bytes", 0)

    def configure_cache_pool(
        self,
        capacity: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
    ) -> DenoisingCachePoolProfile:
        """Allocate all persistent DiT KV slots before accepting sessions."""
        if capacity < 1:
            raise ValueError(f"cache pool capacity must be positive, got {capacity}")
        if self._cache_registry:
            raise RuntimeError("cannot configure the LingBot KV cache pool while sessions are active")
        existing = getattr(self, "_cache_pool", None)
        if existing is not None:
            profile = DenoisingCachePoolProfile(
                capacity=existing.capacity,
                batch_size=existing.batch_size,
                kv_size=existing.kv_size,
                max_sequence_length=existing.max_sequence_length,
                bytes_per_session=existing.bytes_per_session,
                allocated_bytes=existing.allocated_bytes,
            )
            requested = (capacity, batch_size, kv_size, max_sequence_length)
            current = (profile.capacity, profile.batch_size, profile.kv_size, profile.max_sequence_length)
            if requested != current:
                raise RuntimeError(f"LingBot KV cache pool is already configured as {current}, requested {requested}")
            return profile

        head_dim, num_heads, local_num_heads = self._cache_dimensions()
        pool = _DenoisingCachePool(
            capacity=capacity,
            num_layers=self.dit.num_layers,
            batch_size=batch_size,
            kv_size=kv_size,
            max_sequence_length=max_sequence_length,
            num_heads=num_heads,
            local_num_heads=local_num_heads,
            head_dim=head_dim,
            dtype=self.torch_dtype,
            device=self.device,
        )
        self._cache_pool = pool
        return DenoisingCachePoolProfile(
            capacity=capacity,
            batch_size=batch_size,
            kv_size=kv_size,
            max_sequence_length=max_sequence_length,
            bytes_per_session=pool.bytes_per_session,
            allocated_bytes=pool.allocated_bytes,
        )

    @with_model_offload(["dit"])
    def initialize_cache(
        self,
        cache_handle: int,
        batch_size: int,
        kv_size: int,
        max_sequence_length: int,
        sample_shift: float,
        generator_state: list[int],
        noise_generator_state: list[int],
        noise_shape: tuple[int, int, int, int, int],
        prompt_emb: torch.Tensor | None = None,
        image_condition: dict[str, object] | None = None,
        timestep_indices: tuple[int, ...] = (0, 179, 358, 679),
    ) -> bool:
        """Atomically register session-scoped KV, scheduler, and RNG state."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"Cache handle {cache_handle} is already registered")

        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
        timesteps = _select_timesteps(scheduler, tuple(timestep_indices), sample_shift).to(self.device)
        generator = torch.Generator(device=self.device)
        generator.set_state(torch.tensor(generator_state, dtype=torch.uint8))
        noise_generator = torch.Generator(device=self.device)
        noise_generator.set_state(torch.tensor(noise_generator_state, dtype=torch.uint8))
        pool = getattr(self, "_cache_pool", None)
        pool_slot = None
        try:
            if pool is None:
                self_kv_cache = self._init_self_kv_cache(batch_size, kv_size)
                crossattn_cache = self._init_crossattn_cache(batch_size, max_sequence_length)
            else:
                if (
                    batch_size != pool.batch_size
                    or kv_size > pool.kv_size
                    or max_sequence_length > pool.max_sequence_length
                ):
                    raise ValueError(
                        "Session cache dimensions exceed the fixed LingBot cache profile: "
                        f"requested={(batch_size, kv_size, max_sequence_length)}, "
                        f"configured={(pool.batch_size, pool.kv_size, pool.max_sequence_length)}"
                    )
                acquired = pool.try_acquire()
                if acquired is None:
                    return False
                pool_slot, self_kv_cache, crossattn_cache = acquired
            state = _DenoisingCacheState(
                scheduler=scheduler,
                timesteps=timesteps,
                self_kv_cache=self_kv_cache,
                crossattn_cache=crossattn_cache,
                generator=generator,
                noise_generator=noise_generator,
                noise_shape=noise_shape,
                prompt_emb=prompt_emb,
                pool_slot=pool_slot,
            )
            if image_condition is not None:
                self._resolve_image_condition(
                    state,
                    image_condition,
                    device=self.device,
                    dtype=self.torch_dtype,
                )
        except Exception:
            if pool is not None and pool_slot is not None:
                pool.release(pool_slot)
            raise
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

    def _resolve_image_condition(
        self,
        state: _DenoisingCacheState,
        condition_chunk: torch.Tensor | dict[str, object],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize one condition chunk from a session-resident image latent."""
        if isinstance(condition_chunk, torch.Tensor):
            return condition_chunk
        if not isinstance(condition_chunk, dict):
            raise TypeError(
                f"LingBot image condition must be a tensor or mapping, got {type(condition_chunk).__name__}"
            )

        chunk_index = condition_chunk.get("chunk_index")
        chunk_size = condition_chunk.get("chunk_size")
        if not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError(f"LingBot image condition has invalid chunk_index {chunk_index!r}")
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError(f"LingBot image condition has invalid chunk_size {chunk_size!r}")

        exported = condition_chunk.get("latent_condition")
        if exported is not None:
            if not isinstance(exported, torch.Tensor) or exported.ndim != 4 or exported.shape[1] < 1:
                raise ValueError("LingBot image condition latent must have shape (channels, frames, height, width)")
            state.image_condition_latent = exported.to(device=device, dtype=dtype)
            condition_bytes = state.image_condition_latent.numel() * state.image_condition_latent.element_size()
            self._observed_condition_bytes = max(getattr(self, "_observed_condition_bytes", 0), condition_bytes)
        latent_condition = state.image_condition_latent
        if latent_condition is None:
            raise RuntimeError("LingBot image condition metadata arrived before the session latent")

        start = chunk_index * chunk_size
        available = latent_condition[:, start : start + chunk_size]
        if available.shape[1] < chunk_size:
            tail = latent_condition[:, -1:].expand(-1, chunk_size - available.shape[1], -1, -1)
            available = torch.cat([available, tail], dim=1)
        mask = torch.zeros(
            (4, chunk_size, available.shape[2], available.shape[3]),
            dtype=available.dtype,
            device=available.device,
        )
        if chunk_index == 0:
            mask[:, 0] = 1
        return torch.cat([mask, available], dim=0).unsqueeze(0)

    @staticmethod
    def _build_i2v_model_input_writer(
        latent_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Allocate one I2V input buffer and overwrite only its latent channels."""
        batch, latent_channels, frames, height, width = latent_chunk.shape
        condition = condition_chunk.to(device=latent_chunk.device, dtype=target_dtype)
        model_input = torch.empty(
            (batch, latent_channels + condition.shape[1], frames, height, width),
            dtype=target_dtype,
            device=latent_chunk.device,
        )
        model_input[:, latent_channels:].copy_(condition)

        def write(current_latent: torch.Tensor) -> torch.Tensor:
            model_input[:, :latent_channels].copy_(current_latent)
            return model_input

        return write

    def denoise_chunk(
        self,
        latent_chunk: torch.Tensor,
        condition_chunk: torch.Tensor,
        prompt_emb: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: FlowUniPCMultistepScheduler,
        control_chunk: torch.Tensor | None,
        self_kv_cache: list[dict[str, torch.Tensor | int]],
        crossattn_cache: list[dict[str, torch.Tensor | bool | int]],
        current_start: int,
        max_attention_size: int,
        generator: torch.Generator | None = None,
        session_input_cache: dict[str, object] | None = None,
        prepared_control_is_sharded: bool = False,
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor] | None = None,
        benchmark_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] | None = None,
    ) -> torch.Tensor:
        current_latent = latent_chunk
        for timestep_idx in range(len(timesteps)):
            step_start = step_end = None
            if benchmark_events is not None:
                step_start = torch.cuda.Event(enable_timing=True)
                step_end = torch.cuda.Event(enable_timing=True)
                step_start.record()
            schedule_timestep = timesteps[timestep_idx].view(1)
            model_timestep = schedule_timestep.to(dtype=torch.float32)
            with torch.amp.autocast(
                current_latent.device.type,
                dtype=self.torch_dtype,
                enabled=current_latent.device.type == "cuda",
            ):
                if prepare_model_input is None:
                    model_input = current_latent.to(dtype=self.torch_dtype)
                    model_condition = condition_chunk
                else:
                    model_input = prepare_model_input(current_latent)
                    model_condition = None
                noise_pred = self.dit(
                    x=model_input,
                    timestep=model_timestep,
                    context=prompt_emb,
                    y=model_condition,
                    control_tensor=control_chunk,
                    kv_cache=self_kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start,
                    max_attention_size=max_attention_size,
                    _session_input_cache=session_input_cache,
                    _prepared_control_is_sharded=prepared_control_is_sharded or timestep_idx > 0,
                )
            if noise_pred is None:
                raise RuntimeError("LingBot DMD forward unexpectedly returned no prediction")
            x0 = self._convert_flow_pred_to_x0(noise_pred, current_latent, schedule_timestep[0], scheduler)
            if timestep_idx < len(timesteps) - 1:
                next_timestep = timesteps[timestep_idx + 1].view(1)
                noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
                current_latent = scheduler.add_noise(x0, noise, next_timestep)
            else:
                current_latent = x0
            if benchmark_events is not None:
                step_end.record()
                benchmark_events.append((step_start, step_end))

        logger.debug("LingBotWorldFast chunk denoised")
        return current_latent

    def _next_noise_chunk(self, state: _DenoisingCacheState) -> torch.Tensor:
        """Generate the replicated pre-Ulysses input noise for one causal chunk."""
        return torch.randn(
            state.noise_shape,
            generator=state.noise_generator,
            device=self.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _tensor_cache_key(tensor: torch.Tensor) -> tuple[object, ...]:
        """Return mutation-sensitive identity facts without retaining a CPU copy."""
        return (
            id(tensor),
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor._version,
        )

    def _prepare_session_inputs(
        self,
        state: _DenoisingCacheState,
        cache_handle: int,
        prompt_emb: torch.Tensor,
        control_chunk: torch.Tensor | None,
    ) -> None:
        parameter = next(self.dit.parameters())
        text_key = (*self._tensor_cache_key(prompt_emb), id(parameter), parameter.dtype, parameter.device)
        if state.projected_context_key != text_key:
            state.projected_context_key = text_key
            state.session_input_cache.pop("projected_context", None)
        if control_chunk is None:
            state.prepared_control_key = None
            state.prepared_control_is_sharded = False
            state.session_input_cache.pop("prepared_control", None)
            return
        key = (cache_handle, *self._tensor_cache_key(control_chunk))
        if state.prepared_control_key == key:
            return
        state.prepared_control_key = key
        state.prepared_control_is_sharded = False
        state.session_input_cache.pop("prepared_control", None)

    @with_model_offload(["dit"])
    def denoise_and_update_cache(
        self,
        cache_handle: int,
        condition_chunk: torch.Tensor | dict[str, object],
        prompt_emb: torch.Tensor | None,
        control_chunk: torch.Tensor | None,
        current_start: int,
        max_attention_size: int,
        _local_vae_handoff: bool = False,
        _benchmark_profile: bool = False,
    ) -> torch.Tensor | None | tuple[torch.Tensor | None, dict[str, object]]:
        """Denoise a chunk and commit its clean KV state inside each worker."""
        try:
            state = self._cache_registry[cache_handle]
        except KeyError as exc:
            raise KeyError(f"Unknown cache handle {cache_handle}") from exc
        session_prompt_emb = state.prompt_emb if state.prompt_emb is not None else prompt_emb
        if session_prompt_emb is None:
            raise ValueError("LingBot denoising requires a session prompt embedding")
        self._prepare_session_inputs(state, cache_handle, session_prompt_emb, control_chunk)
        try:
            latent_chunk = self._next_noise_chunk(state)
            condition_chunk = self._resolve_image_condition(
                state,
                condition_chunk,
                device=latent_chunk.device,
                dtype=self.torch_dtype,
            )
            prepare_model_input = self._build_i2v_model_input_writer(
                latent_chunk,
                condition_chunk,
                self.torch_dtype,
            )
            benchmark_events = [] if _benchmark_profile and self.device.type == "cuda" else None
            denoise_span_start = None
            if benchmark_events is not None:
                denoise_span_start = torch.cuda.Event(enable_timing=True)
                denoise_span_start.record()
            denoised = self.denoise_chunk(
                latent_chunk=latent_chunk,
                condition_chunk=condition_chunk,
                prompt_emb=session_prompt_emb,
                timesteps=state.timesteps,
                scheduler=state.scheduler,
                control_chunk=control_chunk,
                self_kv_cache=state.self_kv_cache,
                crossattn_cache=state.crossattn_cache,
                current_start=current_start,
                max_attention_size=max_attention_size,
                generator=state.generator,
                session_input_cache=state.session_input_cache,
                prepared_control_is_sharded=state.prepared_control_is_sharded,
                prepare_model_input=prepare_model_input,
                benchmark_events=benchmark_events,
            )
            state.prepared_control_is_sharded = control_chunk is not None
            clean_start = clean_end = None
            if benchmark_events is not None:
                clean_start = torch.cuda.Event(enable_timing=True)
                clean_end = torch.cuda.Event(enable_timing=True)
                clean_start.record()
            with torch.amp.autocast(
                self.device.type,
                dtype=self.torch_dtype,
                enabled=self.device.type == "cuda",
            ):
                self.dit(
                    x=prepare_model_input(denoised),
                    timestep=torch.zeros((1,), dtype=torch.float32, device=self.device),
                    context=session_prompt_emb,
                    y=None,
                    control_tensor=control_chunk,
                    kv_cache=state.self_kv_cache,
                    crossattn_cache=state.crossattn_cache,
                    current_start=current_start,
                    max_attention_size=max_attention_size,
                    _session_input_cache=state.session_input_cache,
                    _prepared_control_is_sharded=state.prepared_control_is_sharded,
                    update_cache_only=True,
                )
            profile = None
            if benchmark_events is not None:
                clean_end.record()
                clean_end.synchronize()
                profile = {
                    "dmd_step_seconds": [start.elapsed_time(end) / 1000.0 for start, end in benchmark_events],
                    "clean_kv_seconds": clean_start.elapsed_time(clean_end) / 1000.0,
                    "denoise_gpu_span_seconds": denoise_span_start.elapsed_time(clean_end) / 1000.0,
                }
            if _local_vae_handoff and self._vae_decode_stage is not None:
                self._pending_vae_decode_latents.setdefault(cache_handle, deque()).append(denoised)
                result = None
            else:
                result = denoised
            return (result, profile) if profile is not None else result
        finally:
            # Camera tensors are immutable only within this chunk and can be large.
            state.prepared_control_key = None
            state.prepared_control_is_sharded = False
            state.session_input_cache.pop("prepared_control", None)

    def advance_noise(self, cache_handle: int) -> bool:
        """Advance the actor-owned noise RNG for a decode-only cache hit."""
        try:
            state = self._cache_registry[cache_handle]
        except KeyError as exc:
            raise KeyError(f"Unknown cache handle {cache_handle}") from exc
        self._next_noise_chunk(state)
        return True

    def has_cache(self, cache_handle: int) -> bool:
        """Return whether this worker owns the requested cache handle."""
        return cache_handle in self._cache_registry

    def list_cache_handles(self) -> tuple[int, ...]:
        """Return registered cache handles for diagnostics and tests."""
        return tuple(sorted(self._cache_registry))

    def release_cache(self, cache_handle: int) -> bool:
        """Idempotently release worker-local state for one generation session."""
        state = self._cache_registry.pop(cache_handle, None)
        pending_latents = getattr(self, "_pending_vae_decode_latents", None)
        if pending_latents is not None:
            pending_latents.pop(cache_handle, None)
        if state is None:
            return False
        pool = getattr(self, "_cache_pool", None)
        if pool is not None and state.pool_slot is not None:
            pool.release(state.pool_slot)
        return True
