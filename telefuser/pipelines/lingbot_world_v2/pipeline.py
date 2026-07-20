"""LingBot-World v2 causal-fast facade."""

from __future__ import annotations

from dataclasses import dataclass

from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_world_fast.pipeline import (
    LingBotWorldFastPipeline,
    LingBotWorldFastPipelineConfig,
)


@dataclass
class LingBotWorldV2PipelineConfig(LingBotWorldFastPipelineConfig):
    """Configuration for the public LingBot-World v2 causal-fast checkpoint."""

    control_type: str = "cam"
    local_attn_size: int = 18
    sink_size: int = 6
    timestep_indices: tuple[int, ...] = (0, 250, 500, 750)


class LingBotWorldV2Pipeline(LingBotWorldFastPipeline):
    """Thin v2 facade over the shared LingBot causal-fast engine."""

    def init(self, module_manager: ModuleManager, config: LingBotWorldV2PipelineConfig) -> None:
        if config.control_type != "cam":
            raise ValueError("LingBot-World v2 causal-fast supports camera control only")
        super().init(module_manager, config)
