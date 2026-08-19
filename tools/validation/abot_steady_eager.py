"""Benchmark-only eager executor for ABot's fixed-window continuation path.

This module intentionally does not change production serving behavior.  It
temporarily replaces one pipeline instance's continuation entry points so an
offline benchmark can distinguish the cost of the fixed-window
``forward_steady_state`` path from the additional benefit of CUDA Graph
replay. The hook accepts a compatible native microbatch, not merely B=1.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import MethodType
from typing import Any

import torch


def _has_one_generator_per_batch_item(generator: torch.Generator | Sequence[torch.Generator], batch_size: int) -> bool:
    """Match the eager sampler's one-generator-per-session RNG contract."""
    if isinstance(generator, torch.Generator):
        return batch_size == 1
    return (
        isinstance(generator, Sequence)
        and len(generator) == batch_size
        and all(isinstance(item, torch.Generator) for item in generator)
    )


def _cache_cursor_matches(value: Any, expected: int) -> bool:
    """Accept singleton or per-item cursor tensors only when all values agree."""
    return isinstance(value, torch.Tensor) and value.numel() > 0 and bool(torch.all(value == expected).item())


def _is_steady_eager_eligible(
    stage: Any,
    *,
    latent: torch.Tensor,
    prompt_emb: torch.Tensor,
    action_context: torch.Tensor,
    self_cache: list[dict[str, Any]],
    cross_cache: list[dict[str, Any]],
    current_start: int,
    generator: torch.Generator | Sequence[torch.Generator],
) -> bool:
    """Check fixed-window invariants for one compatible native microbatch.

    This is deliberately stricter than the legacy batch path. A false result
    is safe: the hook delegates to the original dynamic eager implementation,
    and the benchmark reports that no static eager observation was made.
    """
    dit = stage.dit
    if latent.ndim != 5 or latent.shape[0] < 1 or latent.shape[2] != 3:
        return False
    batch_size = latent.shape[0]
    if prompt_emb.ndim != 3 or prompt_emb.shape[0] != batch_size:
        return False
    if action_context.ndim != 5 or action_context.shape[0] != batch_size:
        return False
    if not dit.use_relative_rope or not _has_one_generator_per_batch_item(generator, batch_size):
        return False
    if len(self_cache) != dit.num_layers or len(cross_cache) != dit.num_layers:
        return False
    if dit.local_attn_size <= latent.shape[2] or dit.sink_size < 0:
        return False
    frame_tokens = (latent.shape[-2] // dit.patch_size[1]) * (latent.shape[-1] // dit.patch_size[2])
    expected_global_end = current_start * frame_tokens
    capacity = dit.local_attn_size * frame_tokens
    for self_layer, cross_layer in zip(self_cache, cross_cache, strict=True):
        key = self_layer.get("k")
        value = self_layer.get("v")
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            return False
        if key.shape[0] != batch_size or key.shape[1] != capacity or value.shape != key.shape:
            return False
        if not _cache_cursor_matches(self_layer.get("local_end_index"), capacity):
            return False
        if not _cache_cursor_matches(self_layer.get("global_end_index"), expected_global_end):
            return False
        cross_key = cross_layer.get("k")
        cross_value = cross_layer.get("v")
        if not isinstance(cross_key, torch.Tensor) or not isinstance(cross_value, torch.Tensor):
            return False
        if cross_key.shape[0] != batch_size or cross_value.shape != cross_key.shape:
            return False
        if not bool(cross_layer["is_init"]) or int(cross_layer["sequence_length"]) != prompt_emb.shape[1]:
            return False

    return True


class _SteadyEagerRunner:
    """Persistent scratch allocations for one fixed-shape native microbatch."""

    def __init__(self, stage: Any, latent: torch.Tensor, self_cache: list[dict[str, Any]]) -> None:
        dit = stage.dit
        self.shape = tuple(latent.shape)
        self.dtype = latent.dtype
        self.device = latent.device
        self.frames = latent.shape[2]
        self.frame_tokens = (latent.shape[-2] // dit.patch_size[1]) * (latent.shape[-1] // dit.patch_size[2])
        rolled_tokens = (dit.local_attn_size - dit.sink_size - self.frames) * self.frame_tokens
        if rolled_tokens < 0:
            raise ValueError("ABot steady eager block does not fit in the rolling cache tail")
        scratch_shape = (latent.shape[0], rolled_tokens, dit.num_heads, dit.dim // dit.num_heads)
        self.roll_scratch_k = torch.empty(scratch_shape, dtype=latent.dtype, device=latent.device)
        self.roll_scratch_v = torch.empty_like(self.roll_scratch_k)
        cursor = self_cache[0]["global_end_index"]
        if not isinstance(cursor, torch.Tensor):
            raise TypeError("ABot steady eager cache cursor must be a tensor")
        self.current_end = torch.empty_like(cursor, dtype=torch.long, device=latent.device)

    def matches(self, latent: torch.Tensor, self_cache: list[dict[str, Any]]) -> bool:
        cursor = self_cache[0].get("global_end_index")
        return (
            tuple(latent.shape) == self.shape
            and latent.dtype == self.dtype
            and latent.device == self.device
            and isinstance(cursor, torch.Tensor)
            and tuple(cursor.shape) == tuple(self.current_end.shape)
        )

    @staticmethod
    def _draw_noise(
        current: torch.Tensor,
        generator: torch.Generator | Sequence[torch.Generator],
    ) -> torch.Tensor:
        if isinstance(generator, torch.Generator):
            return torch.randn(current.shape, generator=generator, dtype=current.dtype, device=current.device)
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

    def run(
        self,
        stage: Any,
        *,
        latent: torch.Tensor,
        prompt_emb: torch.Tensor,
        action_context: torch.Tensor,
        self_cache: list[dict[str, Any]],
        cross_cache: list[dict[str, Any]],
        current_start: int,
        generator: torch.Generator | Sequence[torch.Generator],
        scheduler: Any,
    ) -> torch.Tensor:
        """Mirror ``_denoise_block`` using eager steady-state DiT calls."""
        timesteps = stage._official_denoising_timesteps(scheduler).to(device=self.device)
        self.current_end.fill_((current_start + self.frames) * self.frame_tokens)
        current = latent
        for index, current_timestep in enumerate(timesteps):
            timestep = torch.full(
                (latent.shape[0], self.frames),
                current_timestep,
                dtype=timesteps.dtype,
                device=self.device,
            )
            with torch.autocast(self.device.type, dtype=stage.torch_dtype, enabled=self.device.type == "cuda"):
                flow_prediction = stage.dit.forward_steady_state(
                    x=current.to(dtype=stage.torch_dtype),
                    timestep=timestep,
                    context=prompt_emb,
                    act_context=action_context,
                    kv_cache=self_cache,
                    crossattn_cache=cross_cache,
                    current_end=self.current_end,
                    roll_scratch_k=self.roll_scratch_k,
                    roll_scratch_v=self.roll_scratch_v,
                    update_cache=index == 0,
                )
            x0 = stage._x0_prediction(flow_prediction, current, timestep, scheduler)
            if index < len(timesteps) - 1:
                noise = self._draw_noise(x0, generator)
                current = scheduler.add_noise(x0, noise, timesteps[index + 1])
            else:
                current = x0
        # Keep the benchmark-only steady path faithful to _denoise_block:
        # final cache-only context update runs outside sampler autocast.
        stage.dit(
            x=current.to(dtype=stage.torch_dtype),
            timestep=torch.zeros_like(timestep),
            context=prompt_emb,
            act_context=action_context,
            kv_cache=self_cache,
            crossattn_cache=cross_cache,
            current_start=current_start * self.frame_tokens,
        )
        return current


class SteadyEagerHook:
    """Temporarily route an offline benchmark pipeline to steady-state eager.

    Pre-full-window chunks retain the original dynamic method.  The hook only
    runs for the fixed B=1, LF=3, Relative-RoPE continuation shape; it records
    every substitution so callers cannot mislabel a legacy fallback as a
    steady-state eager result.
    """

    def __init__(self, stage: Any) -> None:
        self.stage = stage
        self._original = stage.denoise_interactive_block
        self._runners: dict[str, _SteadyEagerRunner] = {}
        self._installed = False
        self._measurement_active = False
        self.total_steady_calls = 0
        self.measured_steady_calls = 0
        self.total_legacy_calls = 0

    def install(self) -> None:
        if self._installed:
            return

        def dispatch(
            stage: Any,
            *,
            session_id: str,
            latent: torch.Tensor,
            prompt_emb: torch.Tensor,
            action_context: torch.Tensor,
            self_cache: list[dict[str, Any]],
            cross_cache: list[dict[str, Any]],
            current_start: int,
            generator: torch.Generator,
            scheduler: Any,
        ) -> torch.Tensor:
            if not _is_steady_eager_eligible(
                stage,
                latent=latent,
                prompt_emb=prompt_emb,
                action_context=action_context,
                self_cache=self_cache,
                cross_cache=cross_cache,
                current_start=current_start,
                generator=generator,
            ):
                self.total_legacy_calls += 1
                return self._original(
                    session_id=session_id,
                    latent=latent,
                    prompt_emb=prompt_emb,
                    action_context=action_context,
                    self_cache=self_cache,
                    cross_cache=cross_cache,
                    current_start=current_start,
                    generator=generator,
                    scheduler=scheduler,
                )
            runner = self._runners.get(session_id)
            if runner is None or not runner.matches(latent, self_cache):
                runner = _SteadyEagerRunner(stage, latent, self_cache)
                self._runners[session_id] = runner
            output = runner.run(
                stage,
                latent=latent,
                prompt_emb=prompt_emb,
                action_context=action_context,
                self_cache=self_cache,
                cross_cache=cross_cache,
                current_start=current_start,
                generator=generator,
                scheduler=scheduler,
            )
            self.total_steady_calls += 1
            if self._measurement_active:
                self.measured_steady_calls += 1
            set_last_metrics = getattr(stage, "_set_cuda_graph_last_metrics", None)
            if callable(set_last_metrics):
                set_last_metrics(eligible=True)
            return output

        self.stage.denoise_interactive_block = MethodType(dispatch, self.stage)
        self._installed = True

    def begin_measurement(self) -> None:
        self.measured_steady_calls = 0
        self._measurement_active = True

    def runtime_metrics(self) -> dict[str, int]:
        return {
            "installed": int(self._installed),
            "steady_calls_total": self.total_steady_calls,
            "steady_calls_measured": self.measured_steady_calls,
            "legacy_calls_total": self.total_legacy_calls,
        }

    def close(self) -> None:
        if self._installed:
            self.stage.denoise_interactive_block = self._original
            self._installed = False
        self._runners.clear()


def install_steady_eager_hook(pipeline: Any) -> SteadyEagerHook:
    """Install a scoped benchmark-only steady-state eager route on ``pipeline``."""
    hook = SteadyEagerHook(pipeline.denoise_stage)
    hook.install()
    return hook
