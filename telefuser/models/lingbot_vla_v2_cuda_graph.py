"""CUDA Graph runner for the fixed-shape LingBot-VLA v2 denoising loop."""

from __future__ import annotations

from threading import Lock
from typing import Any
from weakref import ReferenceType, proxy, ref

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
    if target is not source:
        target.copy_(source)


class LingBotVlaV2PrefixCudaGraph:
    """Capture fixed-shape visual/language prefix encoding and KV construction."""

    def __init__(self, model: Any, *, warmup_runs: int = 1) -> None:
        if warmup_runs < 1:
            raise ValueError("CUDA Graph warmup_runs must be positive")
        self.model = proxy(model)
        self.warmup_runs = warmup_runs
        self.graph: torch.cuda.CUDAGraph | None = None
        self._completion_event: torch.cuda.Event | None = None
        self.images: torch.Tensor | None = None
        self.img_masks: torch.Tensor | None = None
        self.lang_tokens: torch.Tensor | None = None
        self.lang_masks: torch.Tensor | None = None
        self.image_grid_thw: torch.Tensor | None = None
        self._grid_signature: tuple[int, ...] | None = None
        self._img_mask_signature: tuple[int, ...] | None = None
        self._lang_mask_signature: tuple[int, ...] | None = None
        self._layout_source_refs: tuple[tuple[ReferenceType[torch.Tensor], int | None], ...] | None = None
        self.prefix_pad_masks: torch.Tensor | None = None
        self.prefix_position_ids: torch.Tensor | None = None
        self.past_key_values: PastKeyValues | None = None

    @property
    def ready(self) -> bool:
        return self.graph is not None

    def _forward(self) -> tuple[torch.Tensor, torch.Tensor, PastKeyValues]:
        assert self.images is not None
        assert self.img_masks is not None
        assert self.lang_tokens is not None
        assert self.lang_masks is not None
        assert self.image_grid_thw is not None
        return self.model.build_prefix_cache(
            self.images,
            self.img_masks,
            self.lang_tokens,
            self.lang_masks,
            self.image_grid_thw,
        )

    @staticmethod
    def _signature(image_grid_thw: torch.Tensor) -> tuple[int, ...]:
        return tuple(image_grid_thw.detach().to(device="cpu").reshape(-1).tolist())

    @staticmethod
    def _version(tensor: torch.Tensor) -> int | None:
        try:
            return tensor._version
        except RuntimeError:
            return None

    @classmethod
    def _source_ref(cls, tensor: torch.Tensor) -> tuple[ReferenceType[torch.Tensor], int | None]:
        return ref(tensor), cls._version(tensor)

    def _layout_sources_unchanged(self, tensors: tuple[torch.Tensor, ...]) -> bool:
        if self._layout_source_refs is None:
            return False
        return all(
            source_version is not None and source_ref() is tensor and source_version == self._version(tensor)
            for (source_ref, source_version), tensor in zip(self._layout_source_refs, tensors, strict=True)
        )

    def _capture(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        if images.device.type != "cuda":
            raise ValueError("LingBot-VLA v2 prefix CUDA Graph requires CUDA inputs")
        self.images = images.detach().clone()
        self.img_masks = img_masks.detach().clone()
        self.lang_tokens = lang_tokens.detach().clone()
        self.lang_masks = lang_masks.detach().clone()
        self.image_grid_thw = image_grid_thw.detach().clone()
        self._grid_signature = self._signature(image_grid_thw)
        self._img_mask_signature = self._signature(img_masks)
        self._lang_mask_signature = self._signature(lang_masks)
        self._layout_source_refs = tuple(self._source_ref(tensor) for tensor in (img_masks, lang_masks, image_grid_thw))

        device = images.device
        current_stream = torch.cuda.current_stream(device)
        capture_stream = torch.cuda.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.cuda.stream(capture_stream):
            for _ in range(self.warmup_runs):
                self._forward()
        current_stream.wait_stream(capture_stream)
        current_stream.synchronize()
        self.model.set_prefix_cuda_graph_capture(True)
        try:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=capture_stream):
                self.prefix_pad_masks, self.prefix_position_ids, self.past_key_values = self._forward()
        except Exception:
            self.graph = None
            self.model.set_prefix_cuda_graph_capture(False)
            raise
        current_stream.wait_stream(capture_stream)

    def _copy_inputs(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        assert self.images is not None
        assert self.img_masks is not None
        assert self.lang_tokens is not None
        assert self.lang_masks is not None
        assert self.image_grid_thw is not None
        layout_sources = (img_masks, lang_masks, image_grid_thw)
        if not self._layout_sources_unchanged(layout_sources):
            if self._signature(image_grid_thw) != self._grid_signature:
                raise ValueError("LingBot-VLA v2 prefix CUDA Graph image_grid_thw values changed")
            if self._signature(img_masks) != self._img_mask_signature:
                raise ValueError("LingBot-VLA v2 prefix CUDA Graph img_masks values changed")
            if self._signature(lang_masks) != self._lang_mask_signature:
                raise ValueError("LingBot-VLA v2 prefix CUDA Graph lang_masks values changed")
            self._layout_source_refs = tuple(self._source_ref(tensor) for tensor in layout_sources)
        _copy_tensor("images", self.images, images)
        _copy_tensor("lang_tokens", self.lang_tokens, lang_tokens)

    def run(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, PastKeyValues]:
        """Capture lazily, then replay into stable prefix and KV output buffers."""
        current_stream = torch.cuda.current_stream(images.device)
        if self._completion_event is not None:
            current_stream.wait_event(self._completion_event)
        if self.graph is None:
            self._capture(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
        else:
            self._copy_inputs(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
        assert self.graph is not None
        assert self.prefix_pad_masks is not None
        assert self.prefix_position_ids is not None
        assert self.past_key_values is not None
        self.graph.replay()
        if self._completion_event is None:
            self._completion_event = torch.cuda.Event()
        self._completion_event.record(current_stream)
        return self.prefix_pad_masks, self.prefix_position_ids, self.past_key_values


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
        self._uses_static_prefix = False

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
        *,
        static_prefix: bool,
    ) -> None:
        if state.device.type != "cuda":
            raise ValueError("LingBot-VLA v2 CUDA Graph requires CUDA inputs")
        self.state = state.detach().clone()
        self._uses_static_prefix = static_prefix
        self.prefix_pad_masks = prefix_pad_masks if static_prefix else prefix_pad_masks.detach().clone()
        self.prefix_position_ids = prefix_position_ids if static_prefix else prefix_position_ids.detach().clone()
        self.noise = noise.detach().clone()
        self.dt = torch.full((), -1.0 / self.model.config.num_steps, dtype=state.dtype, device=state.device)
        self.initial_time = torch.ones((), dtype=state.dtype, device=state.device)
        self.past_key_values = past_key_values if static_prefix else _clone_past_key_values(past_key_values)

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
        *,
        static_prefix: bool,
    ) -> None:
        assert self.state is not None
        assert self.prefix_pad_masks is not None
        assert self.prefix_position_ids is not None
        assert self.noise is not None
        assert self.past_key_values is not None

        _copy_tensor("state", self.state, state)
        if static_prefix != self._uses_static_prefix:
            raise ValueError("LingBot-VLA v2 CUDA Graph static prefix mode changed")
        if static_prefix:
            if (
                prefix_pad_masks is not self.prefix_pad_masks
                or prefix_position_ids is not self.prefix_position_ids
                or past_key_values is not self.past_key_values
            ):
                raise ValueError("LingBot-VLA v2 CUDA Graph static prefix buffers changed identity")
        else:
            _copy_tensor("prefix_pad_masks", self.prefix_pad_masks, prefix_pad_masks)
            _copy_tensor("prefix_position_ids", self.prefix_position_ids, prefix_position_ids)
        _copy_tensor("noise", self.noise, noise)
        if static_prefix or past_key_values is self.past_key_values:
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
        *,
        static_prefix: bool = False,
    ) -> torch.Tensor:
        """Capture lazily, then replay with static inputs and return an owned output."""
        with self._lock:
            current_stream = torch.cuda.current_stream(state.device)
            if self._completion_event is not None:
                current_stream.wait_event(self._completion_event)
            if self.graph is None:
                self._capture(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    noise,
                    prefix_position_ids,
                    static_prefix=static_prefix,
                )
            else:
                self._copy_inputs(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    noise,
                    prefix_position_ids,
                    static_prefix=static_prefix,
                )
            assert self.graph is not None
            assert self.output is not None
            self.graph.replay()
            result = self.output.clone()
            if self._completion_event is None:
                self._completion_event = torch.cuda.Event()
            self._completion_event.record(current_stream)
            return result


class LingBotVlaV2CudaGraphs:
    """Coordinate prefix and denoising graphs over one shared static-buffer transaction."""

    def __init__(self, model: Any) -> None:
        self._lock = Lock()
        self._completion_event: torch.cuda.Event | None = None
        self.prefix = LingBotVlaV2PrefixCudaGraph(model)
        self.denoising = LingBotVlaV2DenoisingCudaGraph(model)

    @property
    def ready(self) -> bool:
        return self.prefix.ready and self.denoising.ready

    def run(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Replay prefix first, then denoise from its stable KV buffers."""
        with self._lock:
            current_stream = torch.cuda.current_stream(state.device)
            if self._completion_event is not None:
                current_stream.wait_event(self._completion_event)
            prefix_pad_masks, prefix_position_ids, past_key_values = self.prefix.run(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                image_grid_thw,
            )
            result = self.denoising.run(
                state,
                prefix_pad_masks,
                past_key_values,
                noise,
                prefix_position_ids,
                static_prefix=True,
            )
            if self._completion_event is None:
                self._completion_event = torch.cuda.Event()
            self._completion_event.record(current_stream)
            return result
