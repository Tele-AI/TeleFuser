"""Bounded activation-only CUDA graph replay for inference attention."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


class FP8SolAttentionReplay:
    """Replay one local attention shape with fresh Q/K/V and independently owned outputs.

    A runner belongs to one serial inference worker. It captures no model weights,
    distributed collectives, prompts or completed generation results.
    """

    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._inputs: tuple[torch.Tensor, ...] = ()
        self._output: torch.Tensor | None = None

    def run(
        self,
        operation: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key: tuple[Any, ...],
    ) -> torch.Tensor:
        if not q.is_cuda or torch.is_grad_enabled() or torch.cuda.is_current_stream_capturing():
            return operation(q, k, v)
        signature = (
            q.shape,
            q.dtype,
            q.device,
            k.shape,
            k.dtype,
            v.shape,
            v.dtype,
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            *key,
        )
        if self._key != signature:
            self._graph, self._inputs, self._output, self._key = None, (), None, None
            inputs = tuple(t.contiguous().clone() for t in (q, k, v))
            current = torch.cuda.current_stream(q.device)
            stream = torch.cuda.Stream(device=q.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    operation(*inputs)
            current.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = operation(*inputs)
            current.wait_stream(stream)
            self._graph, self._inputs, self._output, self._key = graph, inputs, output, signature
        for static, fresh in zip(self._inputs, (q, k, v), strict=True):
            static.copy_(fresh)
        self._graph.replay()
        return self._output.clone()
