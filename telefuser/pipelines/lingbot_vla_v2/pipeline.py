"""BasePipeline integration for LingBot-VLA v2 base-model inference."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager

from .data import LingBotVlaV2InputProcessor, LingBotVlaV2Inputs, LingBotVlaV2Observation
from .policy import LingBotVlaV2PolicyStage
from .robot_profile import ROBOTWIN_CAMERA_KEYS, RobotWinProfile


@dataclass
class LingBotVlaV2PipelineConfig:
    """Runtime configuration for one LingBot-VLA v2 pipeline replica."""

    policy_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    robot_profile: RobotWinProfile = field(default_factory=RobotWinProfile.default)
    image_size: int = 256
    enable_metrics: bool = False
    cuda_graph: bool = False


@dataclass(frozen=True)
class LingBotVlaV2CanonicalActionChunk:
    """Normalized canonical actions produced by the base checkpoint."""

    canonical_normalized_actions: torch.Tensor
    horizon: int
    action_dim: int
    checkpoint_variant: str = "base"
    policy_verified: bool = False
    verification_status: str = "unverified_official_6b_base"


class LingBotVlaV2Pipeline(BasePipeline):
    """Single-replica LingBot-VLA v2 canonical action SDK."""

    # The service owns one fixed-shape resident policy. Per-request GC and
    # allocator cache eviction add latency without releasing model weights.
    clear_memory_after_call = False

    def _get_stages(self) -> list:
        return [self.policy_stage]

    def init(self, module_manager: ModuleManager, config: LingBotVlaV2PipelineConfig) -> None:
        self._model_info = module_manager.get_model_info()
        self.config = config
        policy = module_manager.fetch_module("lingbot_vla_v2")
        processor = module_manager.fetch_module("lingbot_vla_v2_processor")
        if policy is None or processor is None:
            raise RuntimeError("LingBot-VLA v2 requires policy and lingbot_vla_v2_processor modules")
        self.input_processor = LingBotVlaV2InputProcessor(
            processor,
            policy.config,
            config.robot_profile,
            image_size=config.image_size,
        )
        self.policy_stage = LingBotVlaV2PolicyStage("policy", module_manager, config.policy_config)
        if config.cuda_graph:
            if config.policy_config.device_type != "cuda":
                raise ValueError("LingBot-VLA v2 CUDA Graph requires a CUDA policy")
            if not hasattr(policy, "set_cuda_graph_enabled"):
                raise TypeError("LingBot-VLA v2 policy does not support CUDA Graph execution")
            policy.set_cuda_graph_enabled(True)
        if config.enable_metrics:
            self.enable_metrics()

    @torch.inference_mode()
    def predict(
        self,
        inputs: LingBotVlaV2Inputs,
        seed: int | None = None,
    ) -> LingBotVlaV2CanonicalActionChunk:
        """Run prepared tensors and return normalized canonical actions."""
        actions = self.policy_stage.process(inputs, seed=seed)
        if actions.shape[0] != 1:
            raise RuntimeError(f"LingBot-VLA v2 pipeline expects batch size 1, got {actions.shape[0]}")
        canonical_actions = actions[0]
        policy_config = self.policy_stage.policy.config
        return LingBotVlaV2CanonicalActionChunk(
            canonical_normalized_actions=canonical_actions,
            horizon=int(canonical_actions.shape[0]),
            action_dim=int(canonical_actions.shape[1]),
            checkpoint_variant=str(getattr(policy_config, "checkpoint_variant", "base")),
            policy_verified=bool(getattr(policy_config, "policy_verified", False)),
            verification_status=str(getattr(policy_config, "verification_status", "unverified_official_6b_base")),
        )

    @torch.inference_mode()
    def __call__(
        self,
        observation: LingBotVlaV2Observation,
        seed: int | None = None,
    ) -> LingBotVlaV2CanonicalActionChunk:
        """Predict one normalized canonical action chunk."""
        return self.predict(self.input_processor.prepare(observation), seed=seed)

    def prepare_for_inference(self) -> None:
        """Move the policy to its target device before the service becomes ready."""
        if not self.policy_stage.onload_models_flag:
            self.policy_stage.onload_models()
            self.policy_stage.onload_models_flag = True

    @torch.inference_mode()
    def warmup(self) -> None:
        """Initialize fixed-shape CUDA kernels before accepting service requests."""
        self.prepare_for_inference()
        image_size = self.input_processor.image_size
        image = torch.zeros(3, image_size, image_size, dtype=torch.uint8)
        self(
            LingBotVlaV2Observation(
                task="warm up the policy",
                state=[0.0] * 14,
                images={key: image for key in ROBOTWIN_CAMERA_KEYS},
            ),
            seed=0,
        )

    def close(self) -> None:
        """Release policy device memory."""
        if hasattr(self, "policy_stage"):
            if hasattr(self.policy_stage.policy, "set_cuda_graph_enabled"):
                self.policy_stage.policy.set_cuda_graph_enabled(False)
            self.policy_stage.offload_models()
            self.policy_stage.onload_models_flag = False
