"""Official TAeW2.2 lightweight streaming decoder stage for ABot-World."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.taew2_2 import StreamingTAEHV, TAEHV


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

    def create_decode_state(self) -> ABotWorldTAEWDecodeState:
        """Create an isolated stream state while sharing immutable decoder weights."""
        return ABotWorldTAEWDecodeState(stream=StreamingTAEHV(self.taew))

    @with_model_offload(["taew"])
    @torch.inference_mode()
    def warmup_first_frame(self, state: ABotWorldTAEWDecodeState, first_frame_latent: torch.Tensor) -> None:
        """Populate official TAeW temporal memory from the conditioning latent."""
        state.stream.reset()
        latent = first_frame_latent.permute(0, 2, 1, 3, 4).to(self.device, dtype=self.torch_dtype)
        state.stream.decode(latent)

    @with_model_offload(["taew"])
    @torch.inference_mode()
    def decode_chunk(self, latents: torch.Tensor, state: ABotWorldTAEWDecodeState) -> torch.Tensor:
        """Decode one causal latent chunk to RGB frames in [-1, 1]."""
        decoded = state.stream.decode(latents.permute(0, 2, 1, 3, 4).to(self.device, dtype=self.torch_dtype))
        if decoded is None:
            return latents.new_empty((latents.shape[0], 0, 3, 0, 0))
        return decoded.mul(2).sub(1).clamp(-1, 1).permute(0, 2, 1, 3, 4).contiguous()
