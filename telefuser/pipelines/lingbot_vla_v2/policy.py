"""TeleFuser stage for LingBot-VLA v2 flow-matching action inference."""

from __future__ import annotations

from typing import Any

import torch

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.metrics import with_metrics

from .data import LingBotVlaV2Inputs


class LingBotVlaV2PolicyStage(BaseStage):
    """Run Qwen3-VL prefix encoding and the complete 10-step action sampler."""

    def __init__(self, name: str, module_manager: ModuleManager, runtime_config: ModelRuntimeConfig) -> None:
        super().__init__(name, runtime_config)
        self.policy = module_manager.fetch_module("lingbot_vla_v2")
        if self.policy is None:
            raise RuntimeError("ModuleManager does not contain 'lingbot_vla_v2'")
        self.model_names = ["policy"]
        self._validate_parallelism()

    def _validate_parallelism(self) -> None:
        parallel_config = self.model_runtime_config.parallel_config
        if getattr(parallel_config, "world_size", 1) != 1:
            raise ValueError("LingBot-VLA v2 currently supports one GPU per pipeline replica")

    @with_model_offload(["policy"])
    @torch.inference_mode()
    @with_metrics
    def process(
        self,
        inputs: LingBotVlaV2Inputs,
        seed: int | None = None,
        stop_event: Any | None = None,
    ) -> torch.Tensor:
        """Return a CPU float32 normalized action chunk with shape ``[1, H, 55]``."""
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("LingBot-VLA v2 inference cancelled")
        device = self.device
        dtype = self.torch_dtype
        tensors = {
            "images": inputs.images.to(device=device, dtype=dtype),
            "img_masks": inputs.img_masks.to(device=device),
            "lang_tokens": inputs.lang_tokens.to(device=device),
            "lang_masks": inputs.lang_masks.to(device=device),
            "state": inputs.state.to(device=device, dtype=dtype),
            "image_grid_thw": inputs.image_grid_thw.to(device=device, dtype=torch.long),
        }
        noise = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            config = self.policy.config
            noise = torch.randn(
                tensors["state"].shape[0],
                int(config.n_action_steps),
                int(config.max_action_dim),
                device=device,
                dtype=dtype,
                generator=generator,
            )
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("LingBot-VLA v2 inference cancelled")
        actions = self.policy.sample_actions(**tensors, noise=noise, stop_event=stop_event)
        if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
            raise RuntimeError(f"LingBot-VLA v2 policy returned an invalid action tensor: {type(actions)!r}")
        config = self.policy.config
        expected_shape = (
            tensors["state"].shape[0],
            int(config.n_action_steps),
            int(config.max_action_dim),
        )
        if tuple(actions.shape) != expected_shape:
            raise RuntimeError(
                f"LingBot-VLA v2 policy returned shape {tuple(actions.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(actions).all():
            raise RuntimeError("LingBot-VLA v2 policy returned non-finite actions")
        return actions.detach().to(device="cpu", dtype=torch.float32)
