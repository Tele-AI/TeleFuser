"""Session-safe VAE stage for the LingBot realtime pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from telefuser.cache.session_memory import SessionSlotPool
from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.wan_video_vae import (
    WanVideoVAE,
    WanVideoVAEStreamingDecodeState,
    WanVideoVAEStreamingEncodeState,
    _convert_conv3d_to_channels_last_3d,
    _enable_spatial_parallel_decode,
)


def _cache_tensor_bytes(cache: list[object]) -> int:
    """Return allocator bytes retained by tensor entries in a causal VAE cache."""
    return sum(item.numel() * item.element_size() for item in cache if isinstance(item, torch.Tensor))


_CACHE_POOL_HEADROOM_NUMERATOR = 11
_CACHE_POOL_HEADROOM_DENOMINATOR = 10


@dataclass(frozen=True)
class VAECachePoolProfile:
    """Fixed VAE causal-cache storage allocated for one worker."""

    capacity: int
    tensor_entries: int
    bytes_per_session: int
    allocated_bytes: int


class _VAECachePool:
    """Flat per-entry storage that supports reusable views with different spatial shapes."""

    def __init__(
        self,
        *,
        capacity: int,
        layout: dict[int, tuple[torch.dtype, int]],
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self._slots = SessionSlotPool(capacity, name="LingBot VAE cache")
        self._storage = {
            index: torch.empty((capacity, max_numel), dtype=dtype, device=device)
            for index, (dtype, max_numel) in layout.items()
        }

    @property
    def allocated_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._storage.values())

    @property
    def bytes_per_session(self) -> int:
        return self.allocated_bytes // self.capacity

    @property
    def tensor_entries(self) -> int:
        return len(self._storage)

    def acquire(self) -> int:
        return self._slots.acquire()

    def try_acquire(self) -> int | None:
        """Acquire a storage slot without raising on capacity exhaustion."""
        return self._slots.try_acquire()

    def stabilize(self, cache: list[object], slot: int) -> None:
        for index, item in enumerate(cache):
            if not isinstance(item, torch.Tensor):
                continue
            storage = self._storage.get(index)
            if storage is None:
                raise RuntimeError(f"LingBot VAE cache entry {index} was not observed during warmup")
            if item.dtype != storage.dtype:
                raise RuntimeError(
                    f"LingBot VAE cache entry {index} changed dtype from {storage.dtype} to {item.dtype}"
                )
            if item.numel() > storage.shape[1]:
                raise RuntimeError(
                    f"LingBot VAE cache entry {index} exceeds its fixed profile: "
                    f"requested={item.numel()}, configured={storage.shape[1]}"
                )
            target = storage[slot, : item.numel()].view(item.shape)
            target.copy_(item)
            cache[index] = target

    def release(self, slot: int) -> None:
        self._slots.release(slot)


@dataclass
class _VAEEncodeCacheState:
    """Worker-local condition image and causal encoder cache for one session."""

    condition_image: torch.Tensor | None
    encoder_state: WanVideoVAEStreamingEncodeState = field(default_factory=WanVideoVAEStreamingEncodeState)
    latent_condition: torch.Tensor | None = None
    latent_condition_exported: bool = False
    pool_slot: int | None = None


class LingBotWorldFastVAEEncodeStage(BaseStage):
    """Run condition encoding in a VAE worker independent from video decoding."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.vae: WanVideoVAE = module_manager.fetch_module("wan_video_vae")
        if self.vae is None:
            raise ValueError("LingBot VAE encode stage requires a loaded wan_video_vae module")
        self.model_names = ["vae"]
        # Condition chunks reuse the same bounded shapes for the session lifetime.
        # Retain allocator blocks instead of forcing a driver allocation per call.
        self.empty_cache_after_call = False
        self._cache_registry: dict[int, _VAEEncodeCacheState] = {}
        self._observed_session_cache_bytes = 0
        self._observed_condition_bytes = 0
        self._cache_layout: dict[int, tuple[torch.dtype, int]] = {}
        self._cache_pool: _VAECachePool | None = None

    def _observe_cache(self, cache: list[object]) -> None:
        self._observed_session_cache_bytes = max(self._observed_session_cache_bytes, _cache_tensor_bytes(cache))
        for index, item in enumerate(cache):
            if not isinstance(item, torch.Tensor):
                continue
            current = self._cache_layout.get(index)
            if current is not None and current[0] != item.dtype:
                raise RuntimeError(f"VAE encode cache entry {index} changed dtype from {current[0]} to {item.dtype}")
            self._cache_layout[index] = (item.dtype, max(current[1] if current is not None else 0, item.numel()))

    def estimate_session_cache_bytes(self) -> int:
        """Return VAE encoder bytes for one fixed slot including shape headroom."""
        causal_cache_bytes = sum(
            (
                (numel * _CACHE_POOL_HEADROOM_NUMERATOR + _CACHE_POOL_HEADROOM_DENOMINATOR - 1)
                // _CACHE_POOL_HEADROOM_DENOMINATOR
            )
            * torch.empty((), dtype=dtype).element_size()
            for dtype, numel in self._cache_layout.values()
        )
        return causal_cache_bytes + getattr(self, "_observed_condition_bytes", 0)

    def configure_cache_pool(self, capacity: int) -> VAECachePoolProfile:
        """Allocate all persistent encoder-cache slots before accepting sessions."""
        if capacity < 1:
            raise ValueError(f"cache pool capacity must be positive, got {capacity}")
        if self._cache_registry:
            raise RuntimeError("cannot configure the LingBot VAE encode cache pool while sessions are active")
        existing = self._cache_pool
        if existing is not None:
            if existing.capacity != capacity:
                raise RuntimeError(
                    f"LingBot VAE encode cache pool has capacity {existing.capacity}, requested {capacity}"
                )
            return VAECachePoolProfile(
                capacity=existing.capacity,
                tensor_entries=existing.tensor_entries,
                bytes_per_session=existing.bytes_per_session,
                allocated_bytes=existing.allocated_bytes,
            )
        layout = {
            index: (
                dtype,
                (numel * _CACHE_POOL_HEADROOM_NUMERATOR + _CACHE_POOL_HEADROOM_DENOMINATOR - 1)
                // _CACHE_POOL_HEADROOM_DENOMINATOR,
            )
            for index, (dtype, numel) in self._cache_layout.items()
        }
        pool = _VAECachePool(capacity=capacity, layout=layout, device=self.device)
        self._cache_pool = pool
        return VAECachePoolProfile(
            capacity=capacity,
            tensor_entries=pool.tensor_entries,
            bytes_per_session=pool.bytes_per_session,
            allocated_bytes=pool.allocated_bytes,
        )

    def initialize_cache(self, cache_handle: int, condition_image: torch.Tensor) -> bool:
        """Register the encoder state for one session."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"VAE encode cache handle {cache_handle} is already registered")
        pool_slot = self._cache_pool.try_acquire() if self._cache_pool is not None else None
        if self._cache_pool is not None and pool_slot is None:
            return False
        self._cache_registry[cache_handle] = _VAEEncodeCacheState(
            condition_image=condition_image,
            pool_slot=pool_slot,
        )
        return True

    @with_model_offload(["vae"])
    def encode_condition_chunk(
        self,
        cache_handle: int,
        chunk_index: int,
        chunk_count: int,
        chunk_size: int,
        height: int,
        width: int,
        output_dtype: torch.dtype,
    ) -> dict[str, object]:
        """Encode the bounded image prefix once and export it once per session."""
        state = self._cache_registry[cache_handle]
        if state.latent_condition is None:
            if state.condition_image is None:
                raise RuntimeError("The first condition request requires the session image tensor")
            target_latent_frames = chunk_count * chunk_size
            encoded_latent_frames = min(target_latent_frames, 16)
            video = torch.zeros(
                (3, 1 + 4 * (encoded_latent_frames - 1), height, width),
                device=self.device,
                dtype=self.torch_dtype,
            )
            video[:, 0] = state.condition_image
            latent = self.vae.cached_encode_withflag(
                video,
                device=self.device,
                is_first_clip=True,
                is_last_clip=True,
                encode_state=state.encoder_state,
            )
            if latent.shape[1] != encoded_latent_frames:
                raise RuntimeError(
                    f"VAE condition prefix has {latent.shape[1]} latent frames, expected {encoded_latent_frames}"
                )
            state.latent_condition = latent.to(dtype=output_dtype)
            self._observed_condition_bytes = max(
                getattr(self, "_observed_condition_bytes", 0),
                state.latent_condition.numel() * state.latent_condition.element_size(),
            )
            state.condition_image = None

        exported = None
        if not state.latent_condition_exported:
            exported = state.latent_condition
            state.latent_condition_exported = True
        return {
            "chunk_index": chunk_index,
            "chunk_size": chunk_size,
            "latent_condition": exported,
        }

    def release_cache(self, cache_handle: int) -> bool:
        """Release encoder state for one session."""
        state = self._cache_registry.pop(cache_handle, None)
        if state is None:
            return False
        if self._cache_pool is not None and state.pool_slot is not None:
            self._cache_pool.release(state.pool_slot)
        return True

    def observed_session_cache_bytes(self) -> int:
        """Return the largest causal encoder cache observed during warmup."""
        return self._observed_session_cache_bytes


@dataclass
class _VAEDecodeCacheState:
    """Worker-local causal decoder cache for one session."""

    decoder_state: WanVideoVAEStreamingDecodeState = field(default_factory=WanVideoVAEStreamingDecodeState)
    pool_slot: int | None = None
    output_buffer: torch.Tensor | None = None
    exported_output_buffer: torch.Tensor | None = None


def _copy_frames_to_shared_cpu(state: _VAEDecodeCacheState, frames: torch.Tensor) -> torch.Tensor:
    converted = ((frames + 1) * 127.5).clamp_(0, 255).to(dtype=torch.uint8)
    output = state.output_buffer
    if output is None or output.shape != converted.shape:
        output = torch.empty(converted.shape, dtype=torch.uint8, device="cpu").share_memory_()
        state.output_buffer = output
    output.copy_(converted)
    return output


class LingBotWorldFastVAEDecodeStage(BaseStage):
    """Run video decoding in a VAE worker independent from condition encoding."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        self.vae: WanVideoVAE = module_manager.fetch_module("wan_video_vae")
        if self.vae is None:
            raise ValueError("LingBot VAE decode stage requires a loaded wan_video_vae module")
        self.model_names = ["vae"]
        self._cache_registry: dict[int, _VAEDecodeCacheState] = {}
        self._observed_session_cache_bytes = 0
        self._cache_layout: dict[int, tuple[torch.dtype, int]] = {}
        self._cache_pool: _VAECachePool | None = None

    def parallel_models(self) -> None:
        """Shard decoder feature maps across the configured VAE worker ranks."""
        _enable_spatial_parallel_decode(self.vae)
        _convert_conv3d_to_channels_last_3d(self.vae.model.decoder)

    def _observe_cache(self, cache: list[object]) -> None:
        self._observed_session_cache_bytes = max(self._observed_session_cache_bytes, _cache_tensor_bytes(cache))
        for index, item in enumerate(cache):
            if not isinstance(item, torch.Tensor):
                continue
            current = self._cache_layout.get(index)
            if current is not None and current[0] != item.dtype:
                raise RuntimeError(f"VAE decode cache entry {index} changed dtype from {current[0]} to {item.dtype}")
            self._cache_layout[index] = (item.dtype, max(current[1] if current is not None else 0, item.numel()))

    def estimate_session_cache_bytes(self) -> int:
        """Return VAE decoder bytes for one fixed slot including shape headroom."""
        return sum(
            (
                (numel * _CACHE_POOL_HEADROOM_NUMERATOR + _CACHE_POOL_HEADROOM_DENOMINATOR - 1)
                // _CACHE_POOL_HEADROOM_DENOMINATOR
            )
            * torch.empty((), dtype=dtype).element_size()
            for dtype, numel in self._cache_layout.values()
        )

    def configure_cache_pool(self, capacity: int) -> VAECachePoolProfile:
        """Allocate all persistent decoder-cache slots before accepting sessions."""
        if capacity < 1:
            raise ValueError(f"cache pool capacity must be positive, got {capacity}")
        if self._cache_registry:
            raise RuntimeError("cannot configure the LingBot VAE decode cache pool while sessions are active")
        if not self._cache_layout:
            raise RuntimeError("cannot configure the LingBot VAE decode cache pool before warmup")
        existing = self._cache_pool
        if existing is not None:
            if existing.capacity != capacity:
                raise RuntimeError(
                    f"LingBot VAE decode cache pool has capacity {existing.capacity}, requested {capacity}"
                )
            return VAECachePoolProfile(
                capacity=existing.capacity,
                tensor_entries=existing.tensor_entries,
                bytes_per_session=existing.bytes_per_session,
                allocated_bytes=existing.allocated_bytes,
            )
        layout = {
            index: (
                dtype,
                (numel * _CACHE_POOL_HEADROOM_NUMERATOR + _CACHE_POOL_HEADROOM_DENOMINATOR - 1)
                // _CACHE_POOL_HEADROOM_DENOMINATOR,
            )
            for index, (dtype, numel) in self._cache_layout.items()
        }
        pool = _VAECachePool(capacity=capacity, layout=layout, device=self.device)
        self._cache_pool = pool
        return VAECachePoolProfile(
            capacity=capacity,
            tensor_entries=pool.tensor_entries,
            bytes_per_session=pool.bytes_per_session,
            allocated_bytes=pool.allocated_bytes,
        )

    def initialize_cache(self, cache_handle: int) -> bool:
        """Register the decoder state for one session."""
        if cache_handle in self._cache_registry:
            raise ValueError(f"VAE decode cache handle {cache_handle} is already registered")
        pool_slot = self._cache_pool.try_acquire() if self._cache_pool is not None else None
        if self._cache_pool is not None and pool_slot is None:
            return False
        self._cache_registry[cache_handle] = _VAEDecodeCacheState(pool_slot=pool_slot)
        return True

    @with_model_offload(["vae"])
    def decode_chunk(
        self,
        cache_handle: int,
        latents: torch.Tensor,
        is_first_clip: bool,
        is_last_clip: bool,
        _benchmark_profile: bool = False,
    ) -> torch.Tensor | None | tuple[torch.Tensor | None, dict[str, float]]:
        """Decode one latent chunk and return CPU frame tensors."""
        state = self._cache_registry[cache_handle]
        decode_start = decode_end = None
        if _benchmark_profile and self.device.type == "cuda":
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            decode_start.record()
        frames = self.vae.cached_decode_withflag(
            latents,
            device=self.device,
            is_first_clip=is_first_clip,
            is_last_clip=is_last_clip,
            decode_state=state.decoder_state,
        )
        profile = None
        if decode_end is not None:
            decode_end.record()
            decode_end.synchronize()
            profile = {"vae_decode_gpu_seconds": decode_start.elapsed_time(decode_end) / 1000.0}
        self._observe_cache(state.decoder_state.feat_cache)
        if self._cache_pool is not None and state.pool_slot is not None:
            self._cache_pool.stabilize(state.decoder_state.feat_cache, state.pool_slot)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                result = None
            else:
                transfer_start = time.perf_counter()
                shared_result = _copy_frames_to_shared_cpu(state, frames)
                if state.exported_output_buffer is shared_result:
                    result = None
                else:
                    state.exported_output_buffer = shared_result
                    result = shared_result
                if profile is not None:
                    profile["gpu_to_cpu_seconds"] = time.perf_counter() - transfer_start
        else:
            transfer_start = time.perf_counter()
            result = frames.cpu()
            if profile is not None:
                profile["gpu_to_cpu_seconds"] = time.perf_counter() - transfer_start
        return (result, profile) if profile is not None else result

    def release_cache(self, cache_handle: int) -> bool:
        """Release decoder state for one session."""
        state = self._cache_registry.pop(cache_handle, None)
        if state is None:
            return False
        if self._cache_pool is not None and state.pool_slot is not None:
            self._cache_pool.release(state.pool_slot)
        return True

    def observed_session_cache_bytes(self) -> int:
        """Return the largest causal decoder cache observed during warmup."""
        return self._observed_session_cache_bytes
