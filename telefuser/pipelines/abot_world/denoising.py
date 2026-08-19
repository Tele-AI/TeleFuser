"""Causal long-forcing denoising stage for ABot-World."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.abot_world_dit import ABotWorldDiT
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.utils.logging import logger

_CUDA_GRAPH_WARMUP_ITERATIONS = 1


@dataclass
class _CudaGraphSlot:
    """One replayable fixed-shape DiT call and its static output."""

    graph: torch.cuda.CUDAGraph
    output: torch.Tensor


@dataclass
class _BatchedCudaGraphState:
    """Persistent B=2/3 cache arena and graph for one ordered session cohort.

    Each member session's K/V tensors are rebound to a row view into this
    arena after the initial successful capture. Subsequent singleton work on
    a member therefore updates the same storage, while a later replay of the
    exact cohort can directly use the contiguous batched cache without a
    multi-gigabyte collate/scatter copy.
    """

    session_ids: tuple[str, ...]
    self_cache: list[dict[str, Any]]
    cross_cache: list[dict[str, Any]]
    graph: "_ABotSteadyCudaGraph"

    def matches_members(
        self,
        session_ids: Sequence[str],
        self_caches: Sequence[Sequence[dict[str, Any]]],
        cross_caches: Sequence[Sequence[dict[str, Any]]],
    ) -> bool:
        if tuple(session_ids) != self.session_ids:
            return False
        for row, (member_self, member_cross) in enumerate(zip(self_caches, cross_caches, strict=True)):
            if len(member_self) != len(self.self_cache) or len(member_cross) != len(self.cross_cache):
                return False
            for source, arena in zip(member_self, self.self_cache, strict=True):
                if source["k"].data_ptr() != arena["k"][row : row + 1].data_ptr():
                    return False
                if source["v"].data_ptr() != arena["v"][row : row + 1].data_ptr():
                    return False
            for source, arena in zip(member_cross, self.cross_cache, strict=True):
                if source["k"].data_ptr() != arena["k"][row : row + 1].data_ptr():
                    return False
                if source["v"].data_ptr() != arena["v"][row : row + 1].data_ptr():
                    return False
        return True


class _ABotSteadyCudaGraph:
    """CUDA-Graph state for one full-window Relative-RoPE ABot session.

    The graph owns only persistent input/scratch tensors; the session's KV and
    cross-attention caches deliberately stay in place. That keeps graph replay
    compatible with the existing retained-session lifecycle without copying a
    multi-gigabyte KV cache for every continuation.
    """

    def __init__(
        self,
        dit: ABotWorldDiT,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        *,
        torch_dtype: torch.dtype,
    ) -> None:
        self.dit = dit
        self.device = latent.device
        self.torch_dtype = torch_dtype
        self.frames = latent.shape[2]
        self.frame_tokens = (latent.shape[-2] // dit.patch_size[1]) * (latent.shape[-1] // dit.patch_size[2])
        self.static_x = torch.empty_like(latent)
        self.static_action = torch.empty_like(action_context)
        self.static_timestep = torch.empty((latent.shape[0], self.frames), dtype=torch.float32, device=self.device)
        self.static_context = prompt_emb.detach().clone()
        self.current_end = torch.empty(1, dtype=torch.long, device=self.device)
        # CUDA Graph capture must use an already-warmed non-default stream.
        # Keeping the stream on this graph state makes the warmup and capture
        # execute on the same device/stream pair even in a process-NCCL
        # worker where the assigned CUDA device is not logical device zero.
        # CPU construction remains supported by lightweight unit tests.
        self.capture_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        capacity = self_cache[0]["k"].shape[1]
        sink_tokens = dit.sink_size * self.frame_tokens
        rolled_tokens = capacity - sink_tokens - latent.shape[2] * self.frame_tokens
        if rolled_tokens < 0:
            raise ValueError("ABot CUDA Graph block does not fit in its rolling cache tail")
        scratch_shape = (
            latent.shape[0],
            rolled_tokens,
            dit.num_heads,
            dit.dim // dit.num_heads,
        )
        self.roll_scratch_k = torch.empty(scratch_shape, dtype=latent.dtype, device=self.device)
        self.roll_scratch_v = torch.empty_like(self.roll_scratch_k)
        self.entry: _CudaGraphSlot | None = None
        self.refinement: _CudaGraphSlot | None = None
        self._self_cache_signature = self._cache_signature(self_cache)
        self._cross_cache_signature = self._cache_signature(cross_cache)

    @staticmethod
    def _cache_signature(caches: Sequence[dict[str, Any]]) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        return tuple(
            (
                int(layer["k"].data_ptr()),
                int(layer["v"].data_ptr()),
                tuple(layer["k"].shape),
            )
            for layer in caches
        )

    def matches(
        self,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: Sequence[dict[str, Any]],
        cross_cache: Sequence[dict[str, Any]],
    ) -> bool:
        return (
            self.entry is not None
            and self.refinement is not None
            and tuple(latent.shape) == tuple(self.static_x.shape)
            and latent.dtype == self.static_x.dtype
            and tuple(prompt_emb.shape) == tuple(self.static_context.shape)
            and prompt_emb.dtype == self.static_context.dtype
            and tuple(action_context.shape) == tuple(self.static_action.shape)
            and action_context.dtype == self.static_action.dtype
            and self._cache_signature(self_cache) == self._self_cache_signature
            and self._cache_signature(cross_cache) == self._cross_cache_signature
        )

    @staticmethod
    def backup_caches(caches: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value.detach().clone() if isinstance(value, torch.Tensor) else value for key, value in layer.items()}
            for layer in caches
        ]

    @staticmethod
    def restore_caches(caches: Sequence[dict[str, Any]], backup: Sequence[dict[str, Any]]) -> None:
        for current, saved in zip(caches, backup, strict=True):
            for key, saved_value in saved.items():
                current_value = current[key]
                if isinstance(saved_value, torch.Tensor):
                    if not isinstance(current_value, torch.Tensor):
                        raise RuntimeError("ABot CUDA Graph cache metadata changed during capture")
                    current_value.copy_(saved_value)
                else:
                    current[key] = saved_value

    def _set_inputs(
        self,
        latent: torch.Tensor,
        action_context: torch.Tensor,
        timestep: torch.Tensor | None,
        *,
        current_end: int,
    ) -> None:
        self.static_x.copy_(latent)
        self.static_action.copy_(action_context)
        if timestep is None:
            self.static_timestep.zero_()
        else:
            self.static_timestep.copy_(timestep.reshape(1, 1).expand_as(self.static_timestep))
        self.current_end.fill_(current_end)

    def _capture_slot(
        self,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        *,
        update_cache: bool,
    ) -> _CudaGraphSlot:
        graph = torch.cuda.CUDAGraph()
        assert self.capture_stream is not None
        # The caller may have restored cache contents on the current stream
        # after warmup. Make the explicit capture stream observe that work
        # before entering capture; a graph may not contain a cross-stream
        # dependency established while capture is active.
        with torch.cuda.device(self.device):
            self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.graph(
                graph,
                stream=self.capture_stream,
                capture_error_mode="thread_local",
            ):
                with torch.autocast(self.device.type, dtype=self.torch_dtype, enabled=self.device.type == "cuda"):
                    output = self.dit.forward_steady_state(
                        x=self.static_x,
                        timestep=self.static_timestep,
                        context=self.static_context,
                        act_context=self.static_action,
                        kv_cache=self_cache,
                        crossattn_cache=cross_cache,
                        current_end=self.current_end,
                        roll_scratch_k=self.roll_scratch_k,
                        roll_scratch_v=self.roll_scratch_v,
                        update_cache=update_cache,
                    )
        return _CudaGraphSlot(graph=graph, output=output)

    def warmup(
        self,
        stage: "ABotWorldDenoisingStage",
        latent: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        *,
        current_start: int,
        scheduler: FlowMatchScheduler,
    ) -> None:
        """Warm both fixed DiT call shapes on the eventual capture stream.

        ``forward_steady_state`` is deliberately distinct from the public
        eager forward that fills the causal KV window. Its first SDPA/kernel
        plan creation must therefore happen *outside* graph capture. This
        method is allowed to mutate the supplied caches; callers use private
        batched arenas or restore their B=1 cache backup before capture.
        """
        timesteps = stage._official_denoising_timesteps(scheduler).to(device=self.device)
        current_end = (current_start + self.frames) * self.frame_tokens
        assert self.capture_stream is not None

        def warm_slot(timestep: torch.Tensor, *, update_cache: bool) -> None:
            self._set_inputs(latent, action_context, timestep, current_end=current_end)
            with torch.autocast(self.device.type, dtype=self.torch_dtype, enabled=self.device.type == "cuda"):
                self.dit.forward_steady_state(
                    x=self.static_x,
                    timestep=self.static_timestep,
                    context=self.static_context,
                    act_context=self.static_action,
                    kv_cache=self_cache,
                    crossattn_cache=cross_cache,
                    current_end=self.current_end,
                    roll_scratch_k=self.roll_scratch_k,
                    roll_scratch_v=self.roll_scratch_v,
                    update_cache=update_cache,
                )

        with torch.cuda.device(self.device):
            self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(self.capture_stream):
                for _ in range(_CUDA_GRAPH_WARMUP_ITERATIONS):
                    warm_slot(timesteps[0], update_cache=True)
                    warm_slot(timesteps[1], update_cache=False)
            self.capture_stream.synchronize()

    @staticmethod
    def _draw_noise(
        current: torch.Tensor,
        generator: torch.Generator | Sequence[torch.Generator],
    ) -> torch.Tensor:
        if isinstance(generator, Sequence):
            if len(generator) != current.shape[0]:
                raise ValueError("ABot CUDA Graph batch needs one generator per session")
            return torch.cat(
                [
                    torch.randn(
                        (1, *current.shape[1:]),
                        generator=item_generator,
                        dtype=current.dtype,
                        device=current.device,
                    )
                    for item_generator in generator
                ],
                dim=0,
            )
        return torch.randn(current.shape, generator=generator, dtype=current.dtype, device=current.device)

    def run(
        self,
        stage: "ABotWorldDenoisingStage",
        latent: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        *,
        current_start: int,
        generator: torch.Generator | Sequence[torch.Generator],
        scheduler: FlowMatchScheduler,
        capture: bool,
    ) -> tuple[torch.Tensor, int]:
        """Execute the four fixed sampler calls plus the context-cache update."""
        if capture and (self.entry is not None or self.refinement is not None):
            raise RuntimeError("ABot CUDA Graph capture was attempted twice for one session")
        if not capture and (self.entry is None or self.refinement is None):
            raise RuntimeError("ABot CUDA Graph replay was requested before capture")
        timesteps = stage._official_denoising_timesteps(scheduler).to(device=self.device)
        current_end = (current_start + self.frames) * self.frame_tokens
        current = latent
        replays = 0
        for index, current_timestep in enumerate(timesteps):
            self._set_inputs(current, action_context, current_timestep, current_end=current_end)
            if index == 0:
                if capture:
                    self.entry = self._capture_slot(self_cache, cross_cache, update_cache=True)
                    # CUDA stream capture records operations but does not run
                    # them. Replay once before consuming the static output or
                    # relying on its externally-owned KV-cache writes.
                    self.entry.graph.replay()
                    replays += 1
                else:
                    assert self.entry is not None
                    self.entry.graph.replay()
                    replays += 1
                assert self.entry is not None
                flow_prediction = self.entry.output
            elif index == 1 and capture:
                self.refinement = self._capture_slot(self_cache, cross_cache, update_cache=False)
                self.refinement.graph.replay()
                replays += 1
                flow_prediction = self.refinement.output
            else:
                assert self.refinement is not None
                self.refinement.graph.replay()
                replays += 1
                flow_prediction = self.refinement.output
            x0 = stage._x0_prediction(flow_prediction, current, self.static_timestep, scheduler)
            if index < len(timesteps) - 1:
                current = scheduler.add_noise(x0, self._draw_noise(x0, generator), timesteps[index + 1])
            else:
                # The cache-only context pass is dynamic, so it does not
                # overwrite the refinement graph's output buffer. Retain the
                # original x0 layout: Conv3D may select a layout-sensitive
                # kernel for the final cache write.
                current = x0
        # The final forward exists only to commit the denoised x0 into the
        # retained KV cache. Keep it on the original dynamic path. Its input
        # must not alias a graph slot's static output: although the values are
        # equal, reuse of that storage can alter the cache-write behavior of
        # the subsequent eager DiT call. The small x0 clone is therefore a
        # correctness boundary; the four denoising calls remain graphed.
        context_input = current.clone()
        self.static_timestep.zero_()
        # Match _denoise_block exactly: its final cache-only public forward is
        # deliberately outside the sampler autocast scope. Changing that
        # precision scope leaves the generated x0 intact but changes retained
        # tail K/V values, which then diverge on the next continuation.
        self.dit(
            x=context_input.to(dtype=self.torch_dtype),
            timestep=self.static_timestep,
            context=self.static_context,
            act_context=action_context,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            current_start=current_start * self.frame_tokens,
        )
        return current, replays


class ABotWorldDenoisingStage(BaseStage):
    """Run ABot's published four-step, x0-prediction causal sampler on one GPU."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        dit = module_manager.fetch_module("abot_world_dit")
        if dit is None or not isinstance(dit, ABotWorldDiT):
            raise ValueError("ABot-World requires a loaded abot_world_dit module")
        self.dit = dit
        self.model_names = ["dit"]
        self._cuda_graph_enabled = False
        self._cuda_graph_states: dict[str, _ABotSteadyCudaGraph] = {}
        self._cuda_graph_batch_states: dict[tuple[str, ...], _BatchedCudaGraphState] = {}
        self._cuda_graph_captures = 0
        self._cuda_graph_replays = 0
        self._cuda_graph_capture_failures = 0
        self._cuda_graph_capture_disabled = False
        self._last_cuda_graph_metrics: dict[str, int] = {
            "cuda_graph_enabled": 0,
            "cuda_graph_eligible": 0,
            "cuda_graph_captured": 0,
            "cuda_graph_replays": 0,
            "cuda_graph_fallback": 0,
            "cuda_graph_batch_size": 0,
            "cuda_graph_batched": 0,
        }

    def configure_cuda_graph(self, enabled: bool) -> None:
        """Enable the experimental fixed-shape CUDA Graph continuation path."""
        self._cuda_graph_enabled = bool(enabled)
        self._cuda_graph_capture_disabled = False
        self._cuda_graph_states.clear()
        self._cuda_graph_batch_states.clear()

    def release_cuda_graph(self, session_id: str) -> None:
        """Drop a graph whose cache pointers are about to move or be released."""
        self._cuda_graph_states.pop(session_id, None)
        for cohort_key in tuple(self._cuda_graph_batch_states):
            if session_id in cohort_key:
                self._cuda_graph_batch_states.pop(cohort_key, None)

    def cuda_graph_metrics(self) -> dict[str, int]:
        """Return process-local, low-cardinality CUDA Graph accounting."""
        return {
            "enabled": int(self._cuda_graph_enabled),
            "resident_sessions": len(self._cuda_graph_states),
            "resident_batch_cohorts": len(self._cuda_graph_batch_states),
            "resident_batched_sessions": sum(
                len(state.session_ids) for state in self._cuda_graph_batch_states.values()
            ),
            "captures": self._cuda_graph_captures,
            "replays": self._cuda_graph_replays,
            "capture_failures": self._cuda_graph_capture_failures,
            "capture_disabled": int(self._cuda_graph_capture_disabled),
        }

    def last_cuda_graph_metrics(self) -> dict[str, int]:
        """Return graph status for the most recently generated interactive block."""
        return dict(self._last_cuda_graph_metrics)

    def record_cuda_graph_not_used(self) -> None:
        """Mark a non-eligible batch without exposing a stale previous hit."""
        self._set_cuda_graph_last_metrics(eligible=False)

    def _record_cuda_graph_capture_failure(self) -> None:
        """Disable graph capture for this worker after an unsafe capture."""
        self._cuda_graph_capture_failures += 1
        self._cuda_graph_capture_disabled = True
        self._cuda_graph_states.clear()
        self._cuda_graph_batch_states.clear()

    def _cuda_graph_backend_is_supported(self) -> bool:
        """Return whether the active attention backend passed graph parity.

        The public SageAttention wrappers allocate and quantize temporary
        tensors on every call. Their current SM90 implementation can enter a
        CUDA graph without an exception but does not yet replay equivalently,
        so keep Sage eager-only until it has a static-buffer adapter.
        """
        blocked = {
            AttnImplType.SAGE_ATTN_2_8_8,
            AttnImplType.SAGE_ATTN_2_8_16,
            AttnImplType.SAGE_ATTN_2_8_8_SM90,
        }
        return self.dit.blocks[0].self_attn.attention_config.attn_impl not in blocked

    def _set_cuda_graph_last_metrics(
        self,
        *,
        eligible: bool,
        captured: bool = False,
        replays: int = 0,
        fallback: bool = False,
        batch_size: int = 0,
        batched: bool = False,
    ) -> None:
        self._last_cuda_graph_metrics = {
            "cuda_graph_enabled": int(self._cuda_graph_enabled),
            "cuda_graph_eligible": int(eligible),
            "cuda_graph_captured": int(captured),
            "cuda_graph_replays": replays,
            "cuda_graph_fallback": int(fallback),
            "cuda_graph_batch_size": batch_size,
            "cuda_graph_batched": int(batched),
        }

    def parallel_models(self) -> None:
        if self.model_runtime_config.parallel_config.world_size != 1:
            raise ValueError("ABot-World initial integration supports exactly one DiT GPU")
        self.dit.set_attention_config(self.model_runtime_config.attention_config)

    def _new_cache(self, batch_size: int, height: int, width: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        frame_tokens = (height // self.dit.patch_size[1]) * (width // self.dit.patch_size[2])
        # The fixed window includes both the retained sink and rolling tail,
        # matching LingBot-World v2: 6 + 12 = 18 latent frames by default.
        kv_size = self.dit.local_attn_size * frame_tokens
        dtype = self.torch_dtype
        self_cache: list[dict[str, Any]] = []
        cross_cache: list[dict[str, Any]] = []
        head_dim = self.dit.dim // self.dit.num_heads
        for _ in range(self.dit.num_layers):
            self_cache.append(
                {
                    "k": torch.zeros(
                        (batch_size, kv_size, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "v": torch.zeros(
                        (batch_size, kv_size, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "global_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
                    "local_end_index": torch.zeros(1, dtype=torch.long, device=self.device),
                }
            )
            cross_cache.append(
                {
                    "k": torch.zeros(
                        (batch_size, self.dit.text_len, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "v": torch.zeros(
                        (batch_size, self.dit.text_len, self.dit.num_heads, head_dim), dtype=dtype, device=self.device
                    ),
                    "is_init": False,
                    "sequence_length": 0,
                }
            )
        return self_cache, cross_cache

    @staticmethod
    def _x0_prediction(
        flow_prediction: torch.Tensor,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        scheduler: FlowMatchScheduler,
    ) -> torch.Tensor:
        if timestep.shape != (latent.shape[0], latent.shape[2]):
            raise ValueError("ABot timestep must have one value per latent frame")
        flat_timestep = timestep.flatten().float()
        sigma_index = (
            (scheduler.timesteps.to(flat_timestep.device).unsqueeze(0) - flat_timestep.unsqueeze(1)).abs().argmin(dim=1)
        )
        sigma = scheduler.sigmas.to(device=latent.device, dtype=torch.float64)[sigma_index]
        sigma = sigma.view(latent.shape[0], 1, latent.shape[2], 1, 1)
        return (latent.double() - sigma * flow_prediction.double()).to(flow_prediction.dtype)

    @staticmethod
    def _scheduler() -> FlowMatchScheduler:
        scheduler = FlowMatchScheduler(template="Wan")
        # ABot's official wrapper creates the full Wan training schedule, then
        # uses these exact four training timesteps.
        scheduler.set_timesteps(1000, training=True, shift=5.0)
        return scheduler

    @staticmethod
    def _official_denoising_timesteps(scheduler: FlowMatchScheduler) -> torch.Tensor:
        """Return the four warped training times from ABot's published config."""
        # ``warp_denoising_step: true`` indexes the 1,000-step shifted Wan
        # schedule with ``1000 - [1000, 750, 500, 250]``.
        return scheduler.timesteps[torch.tensor((0, 250, 500, 750), dtype=torch.long)]

    def _is_cuda_graph_eligible(
        self,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: Sequence[dict[str, Any]],
        cross_cache: Sequence[dict[str, Any]],
        *,
        current_start: int,
        generator: torch.Generator,
    ) -> bool:
        """Check static continuation invariants only before first capture."""
        if (
            not self._cuda_graph_enabled
            or self._cuda_graph_capture_disabled
            or torch.device(self.device).type != "cuda"
        ):
            return False
        if not self._cuda_graph_backend_is_supported():
            return False
        if latent.shape[0] != 1 or latent.shape[2] != 3 or action_context.shape[0] != 1:
            return False
        if not self.dit.use_relative_rope or not isinstance(generator, torch.Generator):
            return False
        if len(self_cache) != self.dit.num_layers or len(cross_cache) != self.dit.num_layers:
            return False
        frame_tokens = (latent.shape[-2] // self.dit.patch_size[1]) * (latent.shape[-1] // self.dit.patch_size[2])
        expected_global_end = current_start * frame_tokens
        capacity = self.dit.local_attn_size * frame_tokens
        if self.dit.local_attn_size <= latent.shape[2] or self.dit.sink_size < 0:
            return False
        for self_layer, cross_layer in zip(self_cache, cross_cache, strict=True):
            if self_layer["k"].shape[1] != capacity or self_layer["v"].shape != self_layer["k"].shape:
                return False
            if int(self_layer["local_end_index"].item()) != capacity:
                return False
            if int(self_layer["global_end_index"].item()) != expected_global_end:
                return False
            if not bool(cross_layer["is_init"]) or int(cross_layer["sequence_length"]) != prompt_emb.shape[1]:
                return False
        return True

    def _is_cuda_graph_batched_eligible(
        self,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_caches: Sequence[Sequence[dict[str, Any]]],
        cross_caches: Sequence[Sequence[dict[str, Any]]],
        *,
        current_starts: Sequence[int],
        generators: Sequence[torch.Generator],
    ) -> bool:
        """Check B=2/3 fixed-window Relative-RoPE continuation invariants."""
        batch_size = latent.shape[0]
        if (
            not self._cuda_graph_enabled
            or self._cuda_graph_capture_disabled
            or torch.device(self.device).type != "cuda"
        ):
            return False
        if not self._cuda_graph_backend_is_supported():
            return False
        if batch_size not in {2, 3} or latent.shape[2] != 3 or action_context.shape[0] != batch_size:
            return False
        if prompt_emb.shape[0] != batch_size or not self.dit.use_relative_rope:
            return False
        if len(self_caches) != batch_size or len(cross_caches) != batch_size:
            return False
        if len(current_starts) != batch_size or len(generators) != batch_size:
            return False
        # The graph owns one internal cursor tensor.  Relative-RoPE permits
        # generic eager batches at different global positions, but a graph
        # cohort must stay position-aligned until it owns per-row cursors.
        if len(set(current_starts)) != 1:
            return False
        if not all(isinstance(generator, torch.Generator) for generator in generators):
            return False
        frame_tokens = (latent.shape[-2] // self.dit.patch_size[1]) * (latent.shape[-1] // self.dit.patch_size[2])
        capacity = self.dit.local_attn_size * frame_tokens
        if self.dit.local_attn_size <= latent.shape[2] or self.dit.sink_size < 0:
            return False
        for current_start, self_cache, cross_cache in zip(current_starts, self_caches, cross_caches, strict=True):
            if len(self_cache) != self.dit.num_layers or len(cross_cache) != self.dit.num_layers:
                return False
            expected_global_end = current_start * frame_tokens
            for self_layer, cross_layer in zip(self_cache, cross_cache, strict=True):
                if self_layer["k"].shape[0] != 1 or self_layer["k"].shape[1] != capacity:
                    return False
                if self_layer["v"].shape != self_layer["k"].shape:
                    return False
                if int(self_layer["local_end_index"].item()) != capacity:
                    return False
                if int(self_layer["global_end_index"].item()) != expected_global_end:
                    return False
                if not bool(cross_layer["is_init"]) or int(cross_layer["sequence_length"]) != prompt_emb.shape[1]:
                    return False
        return True

    def _drop_conflicting_batched_cuda_graphs(self, cohort_key: tuple[str, ...]) -> None:
        members = set(cohort_key)
        for existing_key in tuple(self._cuda_graph_batch_states):
            if existing_key != cohort_key and members.intersection(existing_key):
                self._cuda_graph_batch_states.pop(existing_key, None)

    def _create_batched_cuda_graph_state(
        self,
        session_ids: tuple[str, ...],
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_caches: Sequence[Sequence[dict[str, Any]]],
        cross_caches: Sequence[Sequence[dict[str, Any]]],
        *,
        current_starts: Sequence[int],
    ) -> _BatchedCudaGraphState:
        """Create an unbound persistent cache arena for one graph cohort."""
        batch_size = len(session_ids)
        arena_self, arena_cross = self._new_cache(batch_size, latent.shape[-2], latent.shape[-1])
        for row, (source_self, source_cross) in enumerate(zip(self_caches, cross_caches, strict=True)):
            for source, arena in zip(source_self, arena_self, strict=True):
                arena["k"][row : row + 1].copy_(source["k"])
                arena["v"][row : row + 1].copy_(source["v"])
            for source, arena in zip(source_cross, arena_cross, strict=True):
                arena["k"][row : row + 1].copy_(source["k"])
                arena["v"][row : row + 1].copy_(source["v"])
        frame_tokens = (latent.shape[-2] // self.dit.patch_size[1]) * (latent.shape[-1] // self.dit.patch_size[2])
        capacity = self.dit.local_attn_size * frame_tokens
        for self_layer, cross_layer in zip(arena_self, arena_cross, strict=True):
            self_layer["global_end_index"].fill_(current_starts[0] * frame_tokens)
            self_layer["local_end_index"].fill_(capacity)
            cross_layer["is_init"] = True
            cross_layer["sequence_length"] = prompt_emb.shape[1]
        return _BatchedCudaGraphState(
            session_ids=session_ids,
            self_cache=arena_self,
            cross_cache=arena_cross,
            graph=_ABotSteadyCudaGraph(
                self.dit,
                latent,
                prompt_emb,
                action_context,
                arena_self,
                arena_cross,
                torch_dtype=self.torch_dtype,
            ),
        )

    @staticmethod
    def _bind_batched_cache_arena(
        state: _BatchedCudaGraphState,
        self_caches: Sequence[Sequence[dict[str, Any]]],
        cross_caches: Sequence[Sequence[dict[str, Any]]],
    ) -> None:
        """Make each member cache's K/V tensors a view of its arena row."""
        for row, (member_self, member_cross) in enumerate(zip(self_caches, cross_caches, strict=True)):
            for source, arena in zip(member_self, state.self_cache, strict=True):
                source["k"] = arena["k"][row : row + 1]
                source["v"] = arena["v"][row : row + 1]
            for source, arena in zip(member_cross, state.cross_cache, strict=True):
                source["k"] = arena["k"][row : row + 1]
                source["v"] = arena["v"][row : row + 1]

    def _advance_batched_cache_cursors(
        self,
        self_caches: Sequence[Sequence[dict[str, Any]]],
        *,
        current_starts: Sequence[int],
        latent: torch.Tensor,
    ) -> None:
        """Update independent per-session cursor metadata after arena replay."""
        frame_tokens = (latent.shape[-2] // self.dit.patch_size[1]) * (latent.shape[-1] // self.dit.patch_size[2])
        capacity = self.dit.local_attn_size * frame_tokens
        for current_start, self_cache in zip(current_starts, self_caches, strict=True):
            expected_global_end = (current_start + latent.shape[2]) * frame_tokens
            for layer in self_cache:
                layer["global_end_index"].fill_(expected_global_end)
                layer["local_end_index"].fill_(capacity)

    def denoise_interactive_blocks(
        self,
        *,
        session_ids: Sequence[str],
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_caches: Sequence[Sequence[dict[str, Any]]],
        cross_caches: Sequence[Sequence[dict[str, Any]]],
        current_starts: Sequence[int],
        generators: Sequence[torch.Generator],
        scheduler: FlowMatchScheduler,
    ) -> torch.Tensor | None:
        """Run an exact B=2/3 cohort through a persistent CUDA-Graph arena.

        ``None`` means the caller must use its existing generic eager batch
        path. The arena is only bound to sessions after its first capture has
        successfully produced a real continuation, so a failed capture cannot
        corrupt their retained caches.
        """
        if not self._is_cuda_graph_batched_eligible(
            latent,
            prompt_emb,
            action_context,
            self_caches,
            cross_caches,
            current_starts=current_starts,
            generators=generators,
        ):
            self._set_cuda_graph_last_metrics(eligible=False)
            return None

        cohort_key = tuple(session_ids)
        state = self._cuda_graph_batch_states.get(cohort_key)
        if state is not None and (
            not state.matches_members(session_ids, self_caches, cross_caches)
            or not state.graph.matches(latent, prompt_emb, action_context, state.self_cache, state.cross_cache)
        ):
            self._cuda_graph_batch_states.pop(cohort_key, None)
            state = None

        if state is not None:
            output, replays = state.graph.run(
                self,
                latent,
                action_context,
                state.self_cache,
                state.cross_cache,
                current_start=current_starts[0],
                generator=generators,
                scheduler=scheduler,
                capture=False,
            )
            self._advance_batched_cache_cursors(self_caches, current_starts=current_starts, latent=latent)
            self._cuda_graph_replays += replays
            self._set_cuda_graph_last_metrics(
                eligible=True,
                replays=replays,
                batch_size=latent.shape[0],
                batched=True,
            )
            return output

        self._drop_conflicting_batched_cuda_graphs(cohort_key)
        generator_states = [generator.get_state().clone() for generator in generators]
        try:
            # The cohort's arena is private until a successful capture. This
            # avoids cloning multi-gigabyte session KV caches just to recover
            # from an unsupported graph backend.
            torch.cuda.synchronize(self.device)
            captured = self._create_batched_cuda_graph_state(
                cohort_key,
                latent,
                prompt_emb,
                action_context,
                self_caches,
                cross_caches,
                current_starts=current_starts,
            )
            captured.graph.warmup(
                self,
                latent,
                action_context,
                captured.self_cache,
                captured.cross_cache,
                current_start=current_starts[0],
                scheduler=scheduler,
            )
            for row, (source_self, source_cross) in enumerate(zip(self_caches, cross_caches, strict=True)):
                for source, arena in zip(source_self, captured.self_cache, strict=True):
                    arena["k"][row : row + 1].copy_(source["k"])
                    arena["v"][row : row + 1].copy_(source["v"])
                for source, arena in zip(source_cross, captured.cross_cache, strict=True):
                    arena["k"][row : row + 1].copy_(source["k"])
                    arena["v"][row : row + 1].copy_(source["v"])
            frame_tokens = (latent.shape[-2] // self.dit.patch_size[1]) * (latent.shape[-1] // self.dit.patch_size[2])
            capacity = self.dit.local_attn_size * frame_tokens
            for self_layer, cross_layer in zip(captured.self_cache, captured.cross_cache, strict=True):
                self_layer["global_end_index"].fill_(current_starts[0] * frame_tokens)
                self_layer["local_end_index"].fill_(capacity)
                cross_layer["is_init"] = True
                cross_layer["sequence_length"] = prompt_emb.shape[1]
            output, replays = captured.graph.run(
                self,
                latent,
                action_context,
                captured.self_cache,
                captured.cross_cache,
                current_start=current_starts[0],
                generator=generators,
                scheduler=scheduler,
                capture=True,
            )
        except (RuntimeError, ValueError) as exc:
            for generator, saved_state in zip(generators, generator_states, strict=True):
                generator.set_state(saved_state)
            self._record_cuda_graph_capture_failure()
            self._set_cuda_graph_last_metrics(
                eligible=True,
                fallback=True,
                batch_size=latent.shape[0],
                batched=True,
            )
            logger.warning("ABot CUDA Graph capture for cohort {} failed; using eager: {}", cohort_key, exc)
            return None

        for session_id in cohort_key:
            self._cuda_graph_states.pop(session_id, None)
        self._bind_batched_cache_arena(captured, self_caches, cross_caches)
        self._advance_batched_cache_cursors(self_caches, current_starts=current_starts, latent=latent)
        self._cuda_graph_batch_states[cohort_key] = captured
        self._cuda_graph_captures += 1
        self._cuda_graph_replays += replays
        self._set_cuda_graph_last_metrics(
            eligible=True,
            captured=True,
            replays=replays,
            batch_size=latent.shape[0],
            batched=True,
        )
        return output

    def denoise_interactive_block(
        self,
        *,
        session_id: str,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        current_start: int,
        generator: torch.Generator,
        scheduler: FlowMatchScheduler,
    ) -> torch.Tensor:
        """Use a graph replay for an eligible retained-session continuation.

        The first eligible chunk captures the two static DiT call shapes while
        producing a valid real result. Capture has a full cache/RNG rollback;
        therefore an unsupported PyTorch or attention backend safely falls
        back to the existing eager implementation.
        """
        state = self._cuda_graph_states.get(session_id)
        if state is not None and state.matches(latent, prompt_emb, action_context, self_cache, cross_cache):
            output, replays = state.run(
                self,
                latent,
                action_context,
                self_cache,
                cross_cache,
                current_start=current_start,
                generator=generator,
                scheduler=scheduler,
                capture=False,
            )
            self._cuda_graph_replays += replays
            self._set_cuda_graph_last_metrics(eligible=True, replays=replays)
            return output
        if state is not None:
            self.release_cuda_graph(session_id)
        if not self._is_cuda_graph_eligible(
            latent,
            prompt_emb,
            action_context,
            self_cache,
            cross_cache,
            current_start=current_start,
            generator=generator,
        ):
            self._set_cuda_graph_last_metrics(eligible=False)
            return self._denoise_block(
                latent,
                prompt_emb,
                action_context,
                None,
                self_cache,
                cross_cache,
                current_start,
                generator,
                scheduler,
            )

        self_backup: list[dict[str, Any]] | None = None
        cross_backup: list[dict[str, Any]] | None = None
        generator_state = generator.get_state().clone()
        try:
            # Capture is a rare event. Synchronizing here prevents a previous
            # eager chunk from leaving a lazy kernel initialization on the
            # capture stream; normal replays remain fully asynchronous.
            torch.cuda.synchronize(self.device)
            self_backup = _ABotSteadyCudaGraph.backup_caches(self_cache)
            cross_backup = _ABotSteadyCudaGraph.backup_caches(cross_cache)
            captured = _ABotSteadyCudaGraph(
                self.dit,
                latent,
                prompt_emb,
                action_context,
                self_cache,
                cross_cache,
                torch_dtype=self.torch_dtype,
            )
            captured.warmup(
                self,
                latent,
                action_context,
                self_cache,
                cross_cache,
                current_start=current_start,
                scheduler=scheduler,
            )
            _ABotSteadyCudaGraph.restore_caches(self_cache, self_backup)
            _ABotSteadyCudaGraph.restore_caches(cross_cache, cross_backup)
            output, replays = captured.run(
                self,
                latent,
                action_context,
                self_cache,
                cross_cache,
                current_start=current_start,
                generator=generator,
                scheduler=scheduler,
                capture=True,
            )
        except (RuntimeError, ValueError) as exc:
            if self_backup is not None:
                _ABotSteadyCudaGraph.restore_caches(self_cache, self_backup)
            if cross_backup is not None:
                _ABotSteadyCudaGraph.restore_caches(cross_cache, cross_backup)
            generator.set_state(generator_state)
            self._record_cuda_graph_capture_failure()
            self._set_cuda_graph_last_metrics(eligible=True, fallback=True)
            logger.warning("ABot CUDA Graph capture for session {} failed; falling back to eager: {}", session_id, exc)
            return self._denoise_block(
                latent,
                prompt_emb,
                action_context,
                None,
                self_cache,
                cross_cache,
                current_start,
                generator,
                scheduler,
            )
        self._cuda_graph_states[session_id] = captured
        self._cuda_graph_captures += 1
        self._cuda_graph_replays += replays
        self._set_cuda_graph_last_metrics(eligible=True, captured=True, replays=replays)
        return output

    def _denoise_block(
        self,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        first_frame_latent: torch.Tensor | None,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        current_start: int,
        generator: torch.Generator | Sequence[torch.Generator],
        scheduler: FlowMatchScheduler,
    ) -> torch.Tensor:
        current = latent
        batch, _, frames, height, width = current.shape
        replace_first = current_start == 0 and first_frame_latent is not None
        if replace_first:
            current = current.clone()
            current[:, :, :1].copy_(first_frame_latent)
        frame_tokens = (height // self.dit.patch_size[1]) * (width // self.dit.patch_size[2])
        timesteps = self._official_denoising_timesteps(scheduler).to(device=self.device)
        for index, current_timestep in enumerate(timesteps):
            timestep = torch.full((batch, frames), current_timestep, dtype=timesteps.dtype, device=self.device)
            if replace_first:
                timestep[:, 0] = 0
            with torch.autocast(self.device.type, dtype=self.torch_dtype, enabled=self.device.type == "cuda"):
                flow_prediction = self.dit(
                    x=current.to(dtype=self.torch_dtype),
                    timestep=timestep,
                    context=prompt_emb,
                    act_context=action_context,
                    kv_cache=self_cache,
                    crossattn_cache=cross_cache,
                    current_start=current_start * frame_tokens,
                )
            x0 = self._x0_prediction(flow_prediction, current, timestep, scheduler)
            if index < len(timesteps) - 1:
                if isinstance(generator, Sequence):
                    if len(generator) != x0.shape[0]:
                        raise ValueError("ABot batched denoising requires one generator per session")
                    noise = torch.cat(
                        [
                            torch.randn(
                                (1, *x0.shape[1:]),
                                generator=item_generator,
                                dtype=x0.dtype,
                                device=self.device,
                            )
                            for item_generator in generator
                        ],
                        dim=0,
                    )
                else:
                    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype, device=self.device)
                current = scheduler.add_noise(x0, noise, timesteps[index + 1])
            else:
                current = x0
            if replace_first:
                current[:, :, :1].copy_(first_frame_latent)
        context_timestep = torch.zeros_like(timestep)
        self.dit(
            x=current.to(dtype=self.torch_dtype),
            timestep=context_timestep,
            context=prompt_emb,
            act_context=action_context,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            current_start=current_start * frame_tokens,
        )
        return current

    @with_model_offload(["dit"])
    @torch.inference_mode()
    def process(
        self,
        noise: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        first_frame_latent: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        """Generate a ``1 mod 3`` latent-frame video from a starting image."""
        if noise.ndim != 5 or noise.shape[2] < 1 or (noise.shape[2] - 1) % 3:
            raise ValueError("ABot latent frame count must be positive and equal to 1 mod 3")
        if action_context.shape[:3] != (noise.shape[0], 32, noise.shape[2]):
            raise ValueError("ABot action context must be [batch, 32, latent_frames, height, width]")
        if first_frame_latent.shape != noise[:, :, :1].shape:
            raise ValueError("ABot starting-image latent must be [batch, 48, 1, latent_height, latent_width]")
        self.dit.set_causal_attention_window(self.dit.local_attn_size, self.dit.sink_size)
        self_cache, cross_cache = self._new_cache(noise.shape[0], noise.shape[-2], noise.shape[-1])
        scheduler = self._scheduler()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        output = []
        for start in range(0, noise.shape[2], 3):
            frames = 1 if start == 0 else 3
            block = self._denoise_block(
                noise[:, :, start : start + frames].to(device=self.device, dtype=self.torch_dtype),
                prompt_emb.to(device=self.device, dtype=self.torch_dtype),
                action_context[:, :, start : start + frames].to(device=self.device, dtype=self.torch_dtype),
                first_frame_latent.to(device=self.device, dtype=self.torch_dtype) if start == 0 else None,
                self_cache,
                cross_cache,
                start,
                generator,
                scheduler,
            )
            output.append(block)
        return torch.cat(output, dim=2)
