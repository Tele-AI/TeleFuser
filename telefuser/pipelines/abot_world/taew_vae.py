"""Official TAeW2.2 lightweight streaming decoder stage for ABot-World."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.taew2_2 import TAEHV, StreamingTAEHV, TWorkItem

# This is deliberately a numeric enum: stage metrics are forwarded through
# process boundaries and exported as low-cardinality Prometheus facts.  Keep it
# separate from the DiT scheduler batch size, which can be greater than one
# even when causal TAeW state requires serial decode calls.
_TAEW_DECODE_SINGLETON = 0
_TAEW_DECODE_SYNCHRONIZED_BATCH = 1
_TAEW_DECODE_SERIAL_FALLBACK = 2


@dataclass
class ABotWorldTAEWDecodeState:
    """Session-owned TAeW streaming queues and temporal MemBlock state."""

    stream: StreamingTAEHV


class ABotWorldTAEWDecodeStage(BaseStage):
    """Decode ABot latent chunks with the official TAeW2.2 streaming decoder."""

    def __init__(self, name: str, module_manager: ModuleManager, model_runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, model_runtime_config)
        taew = module_manager.fetch_module("abot_world_taew_decoder")
        if taew is None or not isinstance(taew, TAEHV):
            raise ValueError("ABot-World requires a loaded abot_world_taew_decoder module")
        self.taew = taew
        self.model_names = ["taew"]
        self._last_decode_metrics: dict[str, int] = self._empty_decode_metrics()

    @staticmethod
    def _empty_decode_metrics() -> dict[str, int]:
        return {
            "taew_decode_items": 0,
            "taew_decode_batch_size": 0,
            "taew_decode_invocations": 0,
            "taew_decode_mode": _TAEW_DECODE_SINGLETON,
        }

    def last_decode_metrics(self) -> dict[str, int]:
        """Return facts for the most recently successful TAeW decode.

        ``taew_decode_mode`` is numeric so it remains low-cardinality across
        worker IPC and Prometheus export: ``0`` = singleton, ``1`` = one
        synchronized native LightVAE batch, and ``2`` = safe serial fallback.
        ``taew_decode_batch_size`` is the *effective native decoder* batch
        size, not the enclosing DiT scheduler batch size.
        """
        return dict(getattr(self, "_last_decode_metrics", self._empty_decode_metrics()))

    def create_decode_state(self) -> ABotWorldTAEWDecodeState:
        """Create an isolated stream state while sharing immutable decoder weights."""
        return ABotWorldTAEWDecodeState(stream=StreamingTAEHV(self.taew))

    def snapshot_decode_state(self, state: ABotWorldTAEWDecodeState) -> dict[str, Any]:
        """Clone decode-only streaming state to a CPU-owned tensor tree.

        ABot only calls :meth:`StreamingTAEHV.decode`, so its continuation is
        fully determined by the decoder queue, temporal decoder memory, and
        decoded-frame counter. Encoder-side state intentionally does not belong
        to an ABot session snapshot.
        """
        return self._decode_state_tensor_tree(state, device="cpu", clone_tensors=True)

    def export_decode_state_for_nccl(self, state: ABotWorldTAEWDecodeState) -> dict[str, Any]:
        """Return a tensor-tree payload whose leaves remain on the source GPU.

        The returned structure contains only scalars, mappings, sequences, and
        tensors, so ``flatten_tensor_tree`` can describe it for direct NCCL
        transfer without materializing a CPU copy.
        """
        return self._decode_state_tensor_tree(state, device=None, clone_tensors=False)

    def restore_decode_state(
        self,
        snapshot: Mapping[str, Any],
        *,
        direct_device_tensors: bool = False,
    ) -> ABotWorldTAEWDecodeState:
        """Restore a decoder stream from a CPU or direct-NCCL tensor tree."""
        target_device = None if direct_device_tensors else self.device
        tree = self._normalise_decode_state_tree(
            snapshot,
            device=target_device,
            clone_tensors=not direct_device_tensors,
        )
        if direct_device_tensors:
            expected_device = torch.device(self.device)
            if any(not self._matches_device(tensor.device, expected_device) for tensor in self._iter_tensors(tree)):
                raise ValueError("NCCL TAeW migration tensors must already reside on the target decoder device")
        state = self.create_decode_state()
        self._apply_decode_state_tensor_tree(state, tree)
        return state

    def move_decode_state(self, state: ABotWorldTAEWDecodeState, device: torch.device | str) -> None:
        """Move every session-owned decoder tensor while retaining causal state."""
        tree = self._decode_state_tensor_tree(state, device=device, clone_tensors=False)
        self._apply_decode_state_tensor_tree(state, tree)

    @staticmethod
    def clear_decode_state(state: ABotWorldTAEWDecodeState) -> None:
        """Release queued tensors and temporal memory for a closed session."""
        state.stream.reset()

    @with_model_offload(["taew"])
    @torch.inference_mode()
    def warmup_first_frame(self, state: ABotWorldTAEWDecodeState, first_frame_latent: torch.Tensor) -> None:
        """Populate official TAeW temporal memory from the conditioning latent."""
        state.stream.reset()
        latent = first_frame_latent.permute(0, 2, 1, 3, 4).to(self.device, dtype=self.torch_dtype)
        state.stream.decode(latent)

    def decode_chunk(self, latents: torch.Tensor, state: ABotWorldTAEWDecodeState) -> torch.Tensor:
        """Decode one causal latent chunk to RGB frames in [-1, 1]."""
        return self.decode_chunks(latents, [state])

    @with_model_offload(["taew"])
    @torch.inference_mode()
    def decode_chunks(
        self,
        latents: torch.Tensor,
        states: Sequence[ABotWorldTAEWDecodeState],
    ) -> torch.Tensor:
        """Decode synchronized session chunks in one LightVAE batch when safe.

        State is merged only when its causal queues and temporal-memory layout
        are identical. Incompatible states retain correct per-session decoding
        instead of forcing an invalid batch.
        """
        return self._decode_chunks_impl(latents, states)

    def _decode_chunks_impl(
        self,
        latents: torch.Tensor,
        states: Sequence[ABotWorldTAEWDecodeState],
    ) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f"TAeW decode expects BCTHW latents, got shape {tuple(latents.shape)}")
        if not states or latents.shape[0] != len(states):
            raise ValueError("TAeW decode states must be non-empty and match the latent batch size")
        decoder_latents = latents.permute(0, 2, 1, 3, 4).to(self.device, dtype=self.torch_dtype)
        item_count = len(states)
        if item_count == 1:
            decoded = states[0].stream.decode(decoder_latents)
            decode_metrics = {
                "taew_decode_items": item_count,
                "taew_decode_batch_size": 1,
                "taew_decode_invocations": 1,
                "taew_decode_mode": _TAEW_DECODE_SINGLETON,
            }
        elif self._states_are_batch_compatible(states):
            decoded = self._decode_synchronized_batch(decoder_latents, states)
            decode_metrics = {
                "taew_decode_items": item_count,
                "taew_decode_batch_size": item_count,
                "taew_decode_invocations": 1,
                "taew_decode_mode": _TAEW_DECODE_SYNCHRONIZED_BATCH,
            }
        else:
            decoded = self._decode_serial_batch(decoder_latents, states)
            decode_metrics = {
                "taew_decode_items": item_count,
                "taew_decode_batch_size": 1,
                "taew_decode_invocations": item_count,
                "taew_decode_mode": _TAEW_DECODE_SERIAL_FALLBACK,
            }
        # Only commit the telemetry after the native calls have completed.  A
        # failed decode must not be misreported as a successful batch/fallback.
        self._last_decode_metrics = decode_metrics
        if decoded is None:
            return latents.new_empty((latents.shape[0], 0, 3, 0, 0))
        return decoded.mul(2).sub(1).clamp(-1, 1).permute(0, 2, 1, 3, 4).contiguous()

    def _decode_synchronized_batch(
        self,
        decoder_latents: torch.Tensor,
        states: Sequence[ABotWorldTAEWDecodeState],
    ) -> torch.Tensor | None:
        combined = self._combine_decode_states(states)
        decoded = combined.stream.decode(decoder_latents)
        self._scatter_decode_state(combined, states)
        return decoded

    @staticmethod
    def _decode_serial_batch(
        decoder_latents: torch.Tensor,
        states: Sequence[ABotWorldTAEWDecodeState],
    ) -> torch.Tensor | None:
        decoded_parts = [state.stream.decode(decoder_latents[index : index + 1]) for index, state in enumerate(states)]
        if all(item is None for item in decoded_parts):
            return None
        if any(item is None for item in decoded_parts):
            raise RuntimeError("TAeW decoder states produced incompatible output frame counts")
        return torch.cat([item for item in decoded_parts if item is not None], dim=0)

    @classmethod
    def _states_are_batch_compatible(cls, states: Sequence[ABotWorldTAEWDecodeState]) -> bool:
        reference = cls._decode_state_batch_signature(states[0])
        return all(cls._decode_state_batch_signature(state) == reference for state in states[1:])

    @classmethod
    def _decode_state_batch_signature(cls, state: ABotWorldTAEWDecodeState) -> tuple[Any, ...]:
        stream = state.stream
        return (
            int(stream.n_frames_decoded),
            tuple(
                (int(item.block_index), cls._state_value_batch_signature(item.input_tensor))
                for item in stream.decoder_work_queue
            ),
            cls._state_value_batch_signature(stream.decoder_memory),
        )

    @classmethod
    def _state_value_batch_signature(cls, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim < 1 or value.shape[0] != 1:
                return ("invalid-tensor", tuple(value.shape), str(value.dtype), str(value.device))
            return ("tensor", tuple(value.shape[1:]), str(value.dtype), str(value.device))
        if value is None:
            return ("none",)
        if isinstance(value, list):
            return ("list", tuple(cls._state_value_batch_signature(item) for item in value))
        if isinstance(value, tuple):
            return ("tuple", tuple(cls._state_value_batch_signature(item) for item in value))
        return ("value", value)

    def _combine_decode_states(self, states: Sequence[ABotWorldTAEWDecodeState]) -> ABotWorldTAEWDecodeState:
        combined = self.create_decode_state()
        reference = states[0].stream
        combined.stream.decoder_work_queue = [
            TWorkItem(
                torch.cat([state.stream.decoder_work_queue[index].input_tensor for state in states], dim=0),
                int(item.block_index),
            )
            for index, item in enumerate(reference.decoder_work_queue)
        ]
        combined.stream.decoder_memory = self._collate_state_values([state.stream.decoder_memory for state in states])
        combined.stream.n_frames_decoded = int(reference.n_frames_decoded)
        return combined

    @classmethod
    def _collate_state_values(cls, values: Sequence[Any]) -> Any:
        first = values[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(list(values), dim=0)
        if first is None:
            return None
        if isinstance(first, list):
            return [cls._collate_state_values([value[index] for value in values]) for index in range(len(first))]
        if isinstance(first, tuple):
            return tuple(cls._collate_state_values([value[index] for value in values]) for index in range(len(first)))
        if all(value == first for value in values[1:]):
            return first
        raise ValueError("TAeW decoder states cannot be collated")

    @classmethod
    def _scatter_decode_state(
        cls,
        combined: ABotWorldTAEWDecodeState,
        states: Sequence[ABotWorldTAEWDecodeState],
    ) -> None:
        batch_size = len(states)
        stream = combined.stream
        queue_items = [
            [
                TWorkItem(
                    item.input_tensor[index : index + 1].detach().clone(),
                    int(item.block_index),
                )
                for item in stream.decoder_work_queue
            ]
            for index in range(batch_size)
        ]
        memory_items = cls._split_state_value(stream.decoder_memory, batch_size)
        for index, state in enumerate(states):
            state.stream.decoder_work_queue = queue_items[index]
            state.stream.decoder_memory = memory_items[index]
            state.stream.n_frames_decoded = int(stream.n_frames_decoded)

    @classmethod
    def _split_state_value(cls, value: Any, batch_size: int) -> list[Any]:
        if isinstance(value, torch.Tensor):
            if value.ndim < 1 or value.shape[0] != batch_size:
                raise ValueError("Batched TAeW decoder tensor has an invalid leading batch dimension")
            return [value[index : index + 1].detach().clone() for index in range(batch_size)]
        if value is None:
            return [None] * batch_size
        if isinstance(value, list):
            children = [cls._split_state_value(item, batch_size) for item in value]
            return [[child[index] for child in children] for index in range(batch_size)]
        if isinstance(value, tuple):
            children = [cls._split_state_value(item, batch_size) for item in value]
            return [tuple(child[index] for child in children) for index in range(batch_size)]
        return [value] * batch_size

    def _decode_state_tensor_tree(
        self,
        state: ABotWorldTAEWDecodeState,
        *,
        device: torch.device | str | None,
        clone_tensors: bool,
    ) -> dict[str, Any]:
        stream = state.stream
        return {
            "decoder_work_queue": [
                {
                    "input_tensor": self._copy_tensor_tree(
                        item.input_tensor,
                        device=device,
                        clone_tensors=clone_tensors,
                    ),
                    "block_index": int(item.block_index),
                }
                for item in stream.decoder_work_queue
            ],
            "decoder_memory": self._copy_tensor_tree(
                stream.decoder_memory,
                device=device,
                clone_tensors=clone_tensors,
            ),
            "n_frames_decoded": int(stream.n_frames_decoded),
        }

    def _normalise_decode_state_tree(
        self,
        snapshot: Mapping[str, Any],
        *,
        device: torch.device | str | None,
        clone_tensors: bool,
    ) -> dict[str, Any]:
        required = {"decoder_work_queue", "decoder_memory", "n_frames_decoded"}
        missing = required.difference(snapshot)
        if missing:
            raise ValueError(f"TAeW decode-state snapshot is missing fields: {sorted(missing)}")
        work_queue = snapshot["decoder_work_queue"]
        if not isinstance(work_queue, (list, tuple)):
            raise TypeError("TAeW decoder_work_queue snapshot must be a sequence")
        restored_queue: list[dict[str, Any]] = []
        for item in work_queue:
            if not isinstance(item, Mapping):
                raise TypeError("TAeW decoder_work_queue entries must be mappings")
            if "input_tensor" not in item or "block_index" not in item:
                raise ValueError("TAeW decoder_work_queue entry is incomplete")
            input_tensor = item["input_tensor"]
            if not isinstance(input_tensor, torch.Tensor):
                raise TypeError("TAeW decoder_work_queue input_tensor must be a tensor")
            block_index = int(item["block_index"])
            if not 0 <= block_index <= len(self.taew.decoder):
                raise ValueError(f"TAeW decoder work item has invalid block index {block_index}")
            restored_queue.append(
                {
                    "input_tensor": self._copy_tensor_tree(
                        input_tensor,
                        device=device,
                        clone_tensors=clone_tensors,
                    ),
                    "block_index": block_index,
                }
            )
        decoder_memory = snapshot["decoder_memory"]
        if not isinstance(decoder_memory, (list, tuple)):
            raise TypeError("TAeW decoder_memory snapshot must be a sequence")
        if len(decoder_memory) != len(self.taew.decoder):
            raise ValueError("TAeW decoder_memory snapshot does not match the loaded decoder architecture")
        n_frames_decoded = int(snapshot["n_frames_decoded"])
        if n_frames_decoded < 0:
            raise ValueError("TAeW n_frames_decoded must be non-negative")
        return {
            "decoder_work_queue": restored_queue,
            "decoder_memory": self._copy_tensor_tree(
                list(decoder_memory),
                device=device,
                clone_tensors=clone_tensors,
            ),
            "n_frames_decoded": n_frames_decoded,
        }

    @staticmethod
    def _apply_decode_state_tensor_tree(state: ABotWorldTAEWDecodeState, tree: Mapping[str, Any]) -> None:
        stream = state.stream
        stream.decoder_work_queue = [
            TWorkItem(item["input_tensor"], int(item["block_index"])) for item in tree["decoder_work_queue"]
        ]
        stream.decoder_memory = list(tree["decoder_memory"])
        stream.n_frames_decoded = int(tree["n_frames_decoded"])

    @classmethod
    def _copy_tensor_tree(
        cls,
        value: Any,
        *,
        device: torch.device | str | None,
        clone_tensors: bool,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach()
            if device is not None:
                tensor = tensor.to(device)
            return tensor.clone() if clone_tensors else tensor
        if isinstance(value, list):
            return [cls._copy_tensor_tree(item, device=device, clone_tensors=clone_tensors) for item in value]
        if isinstance(value, tuple):
            return tuple(cls._copy_tensor_tree(item, device=device, clone_tensors=clone_tensors) for item in value)
        if isinstance(value, dict):
            return {
                key: cls._copy_tensor_tree(item, device=device, clone_tensors=clone_tensors)
                for key, item in value.items()
            }
        if value is None or isinstance(value, (bool, float, int, str)):
            return value
        raise TypeError(f"Unsupported TAeW decoder state value: {type(value)!r}")

    @classmethod
    def _iter_tensors(cls, value: Any):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from cls._iter_tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._iter_tensors(item)

    @staticmethod
    def _matches_device(actual: torch.device, expected: torch.device) -> bool:
        return actual.type == expected.type and (expected.index is None or actual.index == expected.index)
