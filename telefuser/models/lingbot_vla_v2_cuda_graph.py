"""CUDA Graph runner for the fixed-shape LingBot-VLA v2 denoising loop."""

from __future__ import annotations

from threading import Lock
from typing import Any
from weakref import proxy

import torch

PastKeyValues = dict[int, dict[str, torch.Tensor]]


def _clone_past_key_values(past_key_values: PastKeyValues) -> PastKeyValues:
    return {
        layer_idx: {
            "key_states": layer_cache["key_states"].detach().clone(),
            "value_states": layer_cache["value_states"].detach().clone(),
        }
        for layer_idx, layer_cache in past_key_values.items()
    }


def _copy_tensor(name: str, target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
        raise ValueError(
            f"LingBot-VLA v2 CUDA Graph input {name} changed from "
            f"{tuple(target.shape)}/{target.dtype}/{target.device} to "
            f"{tuple(source.shape)}/{source.dtype}/{source.device}"
        )
    target.copy_(source)


class LingBotVlaV2DenoisingCudaGraph:
    """Capture and replay all fixed flow-matching steps as one CUDA Graph."""

    def __init__(self, model: Any, *, warmup_runs: int = 1) -> None:
        if warmup_runs < 1:
            raise ValueError("CUDA Graph warmup_runs must be positive")
        self.model = proxy(model)
        self.warmup_runs = warmup_runs
        self._lock = Lock()
        self._completion_event: torch.cuda.Event | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.output: torch.Tensor | None = None
        self.state: torch.Tensor | None = None
        self.prefix_pad_masks: torch.Tensor | None = None
        self.prefix_position_ids: torch.Tensor | None = None
        self.noise: torch.Tensor | None = None
        self.dt: torch.Tensor | None = None
        self.initial_time: torch.Tensor | None = None
        self.past_key_values: PastKeyValues | None = None

    @property
    def ready(self) -> bool:
        return self.graph is not None

    def _denoise(self) -> torch.Tensor:
        assert self.state is not None
        assert self.prefix_pad_masks is not None
        assert self.prefix_position_ids is not None
        assert self.noise is not None
        assert self.dt is not None
        assert self.initial_time is not None
        assert self.past_key_values is not None

        batch_size = self.state.shape[0]
        time = self.initial_time.clone()
        x_t = self.noise.clone()
        for _ in range(int(self.model.config.num_steps)):
            v_t = self.model.predict_velocity(
                self.state,
                self.prefix_pad_masks,
                self.past_key_values,
                x_t,
                time.expand(batch_size),
                prefix_position_ids=self.prefix_position_ids,
            )
            x_t.add_(self.dt * v_t)
            time.add_(self.dt)
        return x_t

    def _capture(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: PastKeyValues,
        noise: torch.Tensor,
        prefix_position_ids: torch.Tensor,
    ) -> None:
        if state.device.type != "cuda":
            raise ValueError("LingBot-VLA v2 CUDA Graph requires CUDA inputs")
        self.state = state.detach().clone()
        self.prefix_pad_masks = prefix_pad_masks.detach().clone()
        self.prefix_position_ids = prefix_position_ids.detach().clone()
        self.noise = noise.detach().clone()
        self.dt = torch.full((), -1.0 / self.model.config.num_steps, dtype=state.dtype, device=state.device)
        self.initial_time = torch.ones((), dtype=state.dtype, device=state.device)
        self.past_key_values = _clone_past_key_values(past_key_values)

        device = state.device
        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(self.warmup_runs):
                self._denoise()
        current_stream.wait_stream(capture_stream)
        current_stream.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=capture_stream):
            self.output = self._denoise()
        current_stream.wait_stream(capture_stream)

    def _copy_inputs(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: PastKeyValues,
        noise: torch.Tensor,
        prefix_position_ids: torch.Tensor,
    ) -> None:
        assert self.state is not None
        assert self.prefix_pad_masks is not None
        assert self.prefix_position_ids is not None
        assert self.noise is not None
        assert self.past_key_values is not None

        _copy_tensor("state", self.state, state)
        _copy_tensor("prefix_pad_masks", self.prefix_pad_masks, prefix_pad_masks)
        _copy_tensor("prefix_position_ids", self.prefix_position_ids, prefix_position_ids)
        _copy_tensor("noise", self.noise, noise)
        if past_key_values is self.past_key_values:
            return
        if past_key_values.keys() != self.past_key_values.keys():
            raise ValueError("LingBot-VLA v2 CUDA Graph KV cache layers changed")
        for layer_idx, layer_cache in past_key_values.items():
            static_cache = self.past_key_values[layer_idx]
            _copy_tensor(
                f"past_key_values[{layer_idx}].key_states", static_cache["key_states"], layer_cache["key_states"]
            )
            _copy_tensor(
                f"past_key_values[{layer_idx}].value_states",
                static_cache["value_states"],
                layer_cache["value_states"],
            )

    def run(
        self,
        state: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        past_key_values: PastKeyValues,
        noise: torch.Tensor,
        prefix_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Capture lazily, then replay with static inputs and return an owned output."""
        with self._lock:
            current_stream = torch.cuda.current_stream(state.device)
            if self._completion_event is not None:
                current_stream.wait_event(self._completion_event)
            if self.graph is None:
                self._capture(state, prefix_pad_masks, past_key_values, noise, prefix_position_ids)
            else:
                self._copy_inputs(state, prefix_pad_masks, past_key_values, noise, prefix_position_ids)
            assert self.graph is not None
            assert self.output is not None
            self.graph.replay()
            result = self.output.clone()
            if self._completion_event is None:
                self._completion_event = torch.cuda.Event()
            self._completion_event.record(current_stream)
            return result
