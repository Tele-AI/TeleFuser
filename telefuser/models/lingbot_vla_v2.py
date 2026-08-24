"""Native LingBot-VLA v2 policy and flow-matching implementation.

Adapted from the Apache-2.0 licensed LingBot-VLA v2 implementation.
"""

# Copyright 2026 Robbyant Team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Literal

import einops
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import AutoConfig, PreTrainedModel, PretrainedConfig
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import is_flash_attn_available
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
from transformers.utils import logging

from telefuser.core.config import QuantConfig, QuantKernelBackend, QuantType
from telefuser.models.lingbot_vla_v2_alignment import TaskTokenDepthHead
from telefuser.models.lingbot_vla_v2_attention import (
    block_suffix_to_fv_,
    build_block_mask,
    create_sinusoidal_pos_embedding,
    flex_attention_forward,
    flex_attention_with_block_mask,
    make_att_2d_masks,
    our_eager_attention_forward,
    prefix_query_segments,
    prefix_query_token_spans,
)
from telefuser.models.lingbot_vla_v2_cuda_graph import LingBotVlaV2CudaGraphs
from telefuser.models.lingbot_vla_v2_loader import LingBotVlaV2StateDictConverter
from telefuser.models.lingbot_vla_v2_moe import (
    FixQwen2RMSNorm,
    Qwen2ForCausalLM,
    Qwen2FusedExperts,
    Qwen2TokenMoeBlock,
)
from telefuser.models.lingbot_vla_v2_quantization import (
    LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZATION_MANIFEST_SHA256,
    LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT,
    build_lingbot_vla_v2_linear_manifest,
    finalize_lingbot_vla_v2_quantization_identity,
)
from telefuser.models.lingbot_vla_v2_qwen import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
)

try:
    from dinov3.hub.backbones import dinov3_vitb16
except ImportError:
    dinov3_vitb16 = None

logger = logging.get_logger(__name__)


# Quantize the standard Qwen text/vision blocks and action-expert attention.
# The fused 3-D MoE weights and action/state heads intentionally remain BF16.
LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES = (
    "qwenvl.model.language_model.layers.",
    "qwenvl.model.visual.blocks.",
    "self_attn.",
)
LINGBOT_VLA_V2_REQUIRED_SKIP_MODULES = (
    "action_in_proj",
    "action_out_proj",
    "action_time_mlp",
    "state_proj",
    "lm_head",
)


class LingbotVLAConfig(PretrainedConfig):
    """Configuration class for Lingbot-VLA.
    This is the configuration class to store the configuration of a [`Lingbot-VLA`].
    """

    model_type = "lingbotvla"
    is_composition = True

    def __init__(
        self,
        vlm_repo_id: str | None = None,
        expert_vision_path: str | None = None,
        tokenizer_path: str | None = None,
        post_training: bool = False,
        adanorm_time: bool = False,
        split_gate_liner: bool = False,
        nosplit_gate_liner: bool = False,
        separate_time_proj: bool = False,
        final_norm_adanorm: bool = False,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        freeze_vision_encoder: bool = False,
        incremental_training: bool = False,
        depth_incremental_training: bool = False,
        reinit_mismatched_weights: bool = False,
        action_dim: int = 14,
        max_action_dim: int = 14,
        max_state_dim: int = 14,
        chunk_size: int = 50,
        vlm_causal: bool = False,
        tokenizer_max_length: int = 48,
        loss_type: str = "fm",
        norm_qkv: bool = False,
        align_params: dict[str, Any] | None = None,
        use_compile: bool = False,
        use_moe: bool = False,
        token_moe_layers: list[int] | None = None,
        token_num_experts: int = 32,
        token_top_k: int = 1,
        token_moe_intermediate_size: int = 256,
        token_shared_intermediate_size: int = 256,
        bias_update_speed: float = 0.001,
        sequence_wise_loss_coeff: float = 0.001,
        sequence_wise_mode: str = "per_sequence",
        router_z_loss_coeff: float = 0.0,
        router_activation: str = "softmax",
        routed_scaling_factor: float = 1.0,
        use_shared_expert_gate: bool = True,
        moe_implementation: Literal["eager", "fused"] | None = None,
        use_robby_moe_kernel: bool = False,
        split_fused_experts_from_decoder_fsdp: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 16,
        action_num_key_value_heads: int = 2,
        action_head_dim: int = 128,
        action_fp32: bool = False,
        use_qwen3_chat_template: bool = False,
        return_image_grid_thw: bool = False,
        qwen3vl_use_vision_boundaries: bool = False,
        precompute_grid_thw: bool = False,
        use_qwen3_fixed_grid_cache: bool = False,
        use_lm_head: bool = False,
        vocab_size: int = 0,
        vit_attn_implementation: str = "flash_attention_2",
        attention_implementation: str = "flex",
        train_expert_only: bool = False,
        train_state_proj: bool = True,
        **kwargs,
    ):
        super().__init__()
        if moe_implementation is None:
            moe_implementation = kwargs.pop("_moe_implementation", None)
        self.architectures = ["LingbotVlaPolicy"]
        self.train_state_proj = train_state_proj
        self.train_expert_only = train_expert_only
        self.use_cache = False
        self.attention_implementation = attention_implementation
        self.num_steps = 10
        self.n_obs_steps = 1

        if split_gate_liner and nosplit_gate_liner:
            raise ValueError("split_gate_liner and nosplit_gate_liner cannot both be True")

        self.vlm_repo_id = vlm_repo_id
        self.expert_vision_path = expert_vision_path
        self.tokenizer_path = tokenizer_path
        self.post_training = post_training
        self.adanorm_time = adanorm_time
        self.split_gate_liner = split_gate_liner
        self.nosplit_gate_liner = nosplit_gate_liner
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.incremental_training = incremental_training
        self.depth_incremental_training = depth_incremental_training
        self.reinit_mismatched_weights = reinit_mismatched_weights
        self.norm_qkv = norm_qkv
        self.use_compile = use_compile
        self.loss_type = loss_type
        self.separate_time_proj = separate_time_proj
        self.final_norm_adanorm = final_norm_adanorm
        self.freeze_vision_encoder = freeze_vision_encoder
        self.tokenizer_max_length = tokenizer_max_length
        self.action_dim = action_dim
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.n_action_steps = chunk_size
        self.vlm_causal = vlm_causal
        self.align_params = align_params
        self.use_moe = use_moe
        if self.use_moe:
            self.token_moe_layers = token_moe_layers
            self.token_num_experts = token_num_experts
            self.token_top_k = token_top_k
            self.token_moe_intermediate_size = token_moe_intermediate_size
            self.token_shared_intermediate_size = token_shared_intermediate_size
        self.bias_update_speed = bias_update_speed
        self.sequence_wise_loss_coeff = sequence_wise_loss_coeff
        self.sequence_wise_mode = sequence_wise_mode
        self.router_z_loss_coeff = router_z_loss_coeff
        self.router_activation = router_activation
        self.routed_scaling_factor = routed_scaling_factor
        self.use_shared_expert_gate = use_shared_expert_gate
        self.moe_implementation = moe_implementation
        self.use_robby_moe_kernel = use_robby_moe_kernel
        if moe_implementation is not None:
            if moe_implementation not in ("eager", "fused"):
                raise ValueError(f"Invalid moe_implementation: {moe_implementation}")
            self._moe_implementation = moe_implementation
        self.split_fused_experts_from_decoder_fsdp = split_fused_experts_from_decoder_fsdp
        self.expert_hidden_size = expert_hidden_size
        self.expert_intermediate_size = expert_intermediate_size
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        self.action_fp32 = action_fp32
        self.use_qwen3_chat_template = use_qwen3_chat_template
        self.return_image_grid_thw = return_image_grid_thw
        self.qwen3vl_use_vision_boundaries = qwen3vl_use_vision_boundaries
        self.precompute_grid_thw = precompute_grid_thw
        self.use_qwen3_fixed_grid_cache = use_qwen3_fixed_grid_cache
        self.use_lm_head = use_lm_head
        if vocab_size == 0:
            if vlm_repo_id and "paligemma" in vlm_repo_id.lower():
                self.vocab_size = 257216
            elif vlm_repo_id and "qwen" in vlm_repo_id.lower():
                self.vocab_size = 151936
            else:
                self.vocab_size = 257152
        else:
            self.vocab_size = vocab_size
        self.vit_attn_implementation = vit_attn_implementation


class LingbotVLAV2Config(LingbotVLAConfig):
    def __init__(self, **kwargs):
        kwargs.setdefault("attention_implementation", "flex_cached")
        kwargs.setdefault("vit_attn_implementation", "flash_attention_2")
        kwargs.setdefault("action_num_attention_heads", 32)
        kwargs.setdefault("action_num_key_value_heads", 8)
        kwargs.setdefault("action_head_dim", 128)
        kwargs.setdefault("expert_hidden_size", 768)
        kwargs.setdefault("use_qwen3_chat_template", True)
        kwargs.setdefault("return_image_grid_thw", True)
        kwargs.setdefault("qwen3vl_use_vision_boundaries", True)
        kwargs.setdefault("use_qwen3_fixed_grid_cache", True)
        super().__init__(**kwargs)
        self.architectures = ["LingbotVlaV2Policy"]
        self.vlm_family = "qwen3_vl"


ConfigClass = [LingbotVLAConfig, LingbotVLAV2Config]
__all__ = ["LingbotVLAConfig", "LingbotVLAV2Config"]


class AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(torch.float32)
        return hidden_states.to(input_dtype)


class FixAdaRMSNorm(AdaRMSNorm):
    def forward(self, hidden_states, cond):
        return super().forward(hidden_states, cond.float())


def replace_lnorm_with_adanorm(module, hidden_size, cond_dim, final_norm_adanorm):
    for name, child in module.named_children():
        if isinstance(child, Qwen2RMSNorm) and "q_layernorm" not in name and "k_layernorm" not in name:
            setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
        elif (
            final_norm_adanorm
            and isinstance(child, FixQwen2RMSNorm)
            and "q_layernorm" not in name
            and "k_layernorm" not in name
        ):
            setattr(module, name, FixAdaRMSNorm(hidden_size, cond_dim))
        else:
            replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)


class FlowMatchingBase(nn.Module):
    def init_depth_heads(self, config):
        self.llm_image_token_size = config["llm"]["image_token_size"]
        self.llm_image_input_size = config["llm"]["image_input_size"]
        self.depth_token_size = config["depth"]["token_size"]
        self.depth_input_size = config["depth"]["input_size"]
        self.align_type = config.get("mode", None)
        self.model_type = config["depth"]["model_type"]
        if self.align_type != "query":
            raise ValueError(f"Only query depth alignment is supported, got {self.align_type!r}.")
        if self.model_type != "MoRGBD":
            raise ValueError(f"Only MoRGBD depth distillation is supported, got {self.model_type!r}.")
        self.use_future_depth = (config.get("depth") or {}).get("use_future_depth", False)
        self.block_future_depth_to_action = (config.get("depth") or {}).get("block_future_depth_to_action", False)
        self.detach_future_depth_image_feats = bool((config.get("depth") or {}).get("detach_future_image_feats", False))
        self.use_future_video = bool(config.get("use_future_video", False))
        self.use_future_video_patch = False
        self.use_current_video_patch = False
        self.use_current_shared_task_proj = False
        self.use_future_video_cls = False
        self.use_shared_future_task_proj = False
        self.future_video_share_future_depth_query = False
        self.num_task_tokens = config["num_task_tokens"]
        if config["depth"]["num_backbone_tokens"] % self.num_task_tokens != 0:
            raise ValueError("depth.num_backbone_tokens must be divisible by num_task_tokens")
        self.depth_align_embs = nn.Parameter(
            torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
        )

        self.depth_align_head = TaskTokenDepthHead(config["depth"], llm_hidden_size=config["llm"]["dim_out"]).to(
            dtype=torch.bfloat16
        )

        if self.use_future_depth:
            self.future_depth_align_embs = nn.Parameter(
                torch.randn(config["depth"]["num_backbone_tokens"], config["llm"]["dim_out"])
            )

            self.future_depth_align_head = TaskTokenDepthHead(
                config["depth"], llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)

    def init_video_heads(self, config):
        if self.align_type != "query":
            raise ValueError("future-video alignment is only supported for query align mode.")

        video_config = dict(config.get("depth", {}))
        video_config.update(config.get("video", {}))
        required_keys = ("num_backbone_tokens", "dim_out", "num_layers", "num_heads", "dim_head", "ff_mult")
        missing = [key for key in required_keys if key not in video_config]
        if missing:
            raise ValueError(f"video align config missing required keys: {missing}")
        self.use_future_video_patch = bool(video_config.get("use_patch_loss", True))
        self.use_current_video_patch = bool(video_config.get("use_current_patch_loss", False))
        if self.use_current_video_patch and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_current_patch_loss=True requires align_params.video.use_patch_loss=True."
            )
        self.use_current_shared_task_proj = bool(
            video_config.get("use_current_shared_task_proj", self.use_current_video_patch)
        )
        if self.use_current_shared_task_proj and not self.use_current_video_patch:
            raise ValueError(
                "align_params.video.use_current_shared_task_proj=True requires "
                "align_params.video.use_current_patch_loss=True."
            )
        self.use_future_video_cls = bool(video_config.get("use_cls_loss", False))
        self.future_video_share_future_depth_query = bool(video_config.get("share_future_depth_query", False))
        self.use_shared_future_task_proj = bool(video_config.get("use_shared_future_task_proj", False))
        if self.use_shared_future_task_proj and not self.use_future_video_patch:
            raise ValueError(
                "align_params.video.use_shared_future_task_proj=True requires align_params.video.use_patch_loss=True."
            )
        if self.use_shared_future_task_proj and not self.future_video_share_future_depth_query:
            raise ValueError(
                "align_params.video.use_shared_future_task_proj=True requires "
                "align_params.video.share_future_depth_query=True."
            )
        if self.future_video_share_future_depth_query:
            if not self.use_future_depth:
                raise ValueError(
                    "align_params.video.share_future_depth_query=True requires "
                    "align_params.depth.use_future_depth=True."
                )
            if int(video_config["num_backbone_tokens"]) != int(config["depth"]["num_backbone_tokens"]):
                raise ValueError(
                    "future-video shared query requires video.num_backbone_tokens to match depth.num_backbone_tokens."
                )

        self.block_suffix_to_future_video = bool(video_config.get("block_suffix_to_future_video", False))
        self.future_video_context_mode = str(video_config.get("context_mode", "img_query")).lower()
        if self.future_video_context_mode not in ("img_query", "query_only"):
            raise ValueError(
                "future-video context_mode must be 'img_query' or 'query_only', "
                f"got {self.future_video_context_mode!r}."
            )
        if self.use_future_video_patch:
            if self.use_current_video_patch:
                self.current_video_align_embs = nn.Parameter(
                    torch.randn(video_config["num_backbone_tokens"], config["llm"]["dim_out"])
                )
                if self.use_current_shared_task_proj:
                    self.current_shared_task_proj = nn.Linear(
                        config["llm"]["dim_out"] * 2,
                        config["llm"]["dim_out"],
                    )
                self.current_video_align_head = TaskTokenDepthHead(
                    video_config, llm_hidden_size=config["llm"]["dim_out"]
                ).to(dtype=torch.bfloat16)

            if not self.future_video_share_future_depth_query or self.use_shared_future_task_proj:
                self.future_video_align_embs = nn.Parameter(
                    torch.randn(video_config["num_backbone_tokens"], config["llm"]["dim_out"])
                )
            if self.use_shared_future_task_proj:
                self.future_shared_task_proj = nn.Linear(
                    config["llm"]["dim_out"] * 2,
                    config["llm"]["dim_out"],
                )
            self.future_video_align_head = TaskTokenDepthHead(
                video_config, llm_hidden_size=config["llm"]["dim_out"]
            ).to(dtype=torch.bfloat16)

        if self.use_future_video_cls:
            self.future_video_cls_align_emb = nn.Embedding(1, config["llm"]["dim_out"])
            self.future_video_cls_head = nn.Sequential(
                nn.LayerNorm(config["llm"]["dim_out"]),
                nn.Linear(config["llm"]["dim_out"], video_config["dim_out"]),
            ).to(dtype=torch.bfloat16)

    def _future_depth_token_count(self):
        return self.num_task_tokens if getattr(self, "use_future_depth", False) else 0

    def _future_video_own_token_count(self):
        if not getattr(self, "use_future_video", False):
            return 0
        count = 1 if getattr(self, "use_future_video_cls", False) else 0
        if getattr(self, "use_future_video_patch", True) and not getattr(
            self, "future_video_share_future_depth_query", False
        ):
            count += self.num_task_tokens
        return count

    def _future_video_own_span(self, hidden_states):
        own_count = self._future_video_own_token_count()
        future_depth_count = self._future_depth_token_count()
        end = hidden_states.shape[1] - future_depth_count
        start = end - own_count
        return start, end

    def _future_depth_task_tokens(self, hidden_states):
        if not getattr(self, "use_future_depth", False):
            raise ValueError("future-depth query tokens are not enabled.")
        return hidden_states[:, -self.num_task_tokens :, :]

    def _future_video_cls_task_tokens(self, hidden_states):
        if not getattr(self, "use_future_video_cls", False):
            return None
        start, _ = self._future_video_own_span(hidden_states)
        return hidden_states[:, start : start + 1, :]

    def _future_video_patch_task_tokens(self, hidden_states):
        if getattr(self, "future_video_share_future_depth_query", False):
            return self._future_depth_task_tokens(hidden_states)
        start, end = self._future_video_own_span(hidden_states)
        if getattr(self, "use_future_video_cls", False):
            start += 1
        return hidden_states[:, start:end, :]

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        chunk_size = self.llm_image_token_size * self.llm_image_token_size
        image_token_len = chunk_size + (2 if getattr(self.config, "qwen3vl_use_vision_boundaries", False) else 0)
        if getattr(self, "use_future_depth", False):
            start = num_images * image_token_len
            return hidden_states[:, start : start + self.num_task_tokens, :]
        end = hidden_states.shape[1] - self._future_video_own_token_count()
        start = end - self.num_task_tokens
        return hidden_states[:, start:end, :]

    def _future_video_query_span(self, prefix_len):
        if not getattr(self, "use_future_video", False):
            return prefix_len, prefix_len
        future_depth_count = self._future_depth_token_count()
        own_count = self._future_video_own_token_count()
        end = prefix_len - future_depth_count
        return end - own_count, end

    def _block_suffix_to_future_video_(self, att_2d_masks, suffix_row_start, prefix_len):
        start, end = self._future_video_query_span(prefix_len)
        if end <= start:
            return att_2d_masks
        att_2d_masks[:, suffix_row_start:, start:end] = False
        return att_2d_masks

    def _block_suffix_to_future_video_if_enabled_(
        self,
        att_2d_masks,
        suffix_row_start,
        prefix_len,
    ):
        if not getattr(self, "block_suffix_to_future_video", False):
            return att_2d_masks
        return self._block_suffix_to_future_video_(
            att_2d_masks,
            suffix_row_start=suffix_row_start,
            prefix_len=prefix_len,
        )

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2FusedExperts):
            module.initializer_range = std
            module.reset_parameters()
        reset_post_init = getattr(module, "_reset_post_init_parameters", None)
        if reset_post_init is not None:
            reset_post_init()

    @staticmethod
    def _fp32_linear(module, x):
        """Compute linear layer in fp32 regardless of module's current parameter dtype."""
        return F.linear(x.float(), module.weight.float(), module.bias.float() if module.bias is not None else None)

    def embed_suffix(
        self, state, noisy_actions, timestep
    ):  # (torch.Size([state_bs, 32]), torch.Size([1, state_bs*50, 32]), torch.Size([1]))
        bsize = state.shape[0]  # state_bs = img_bs
        device = state.device
        dtype = state.dtype
        _fp32 = getattr(self.config, "action_fp32", False)
        # embed state
        state_emb = self._fp32_linear(self.state_proj, state) if _fp32 else self.state_proj(state)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(  # 1, 1024
            timestep,  # torch.Size([1]))
            self.config.proj_width,  # 1024
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb_ori = time_emb

        # Fuse timestep + action information using an MLP
        action_emb = (
            self._fp32_linear(self.action_in_proj, noisy_actions) if _fp32 else self.action_in_proj(noisy_actions)
        )  # torch.Size([1, state_bs*50, 1024])
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])  # [1, 1024] -> [1, state_bs*50, 1024]
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)  # [1, state_bs*50, 2048]

        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_in, action_time_emb)
            if _fp32
            else self.action_time_mlp_in(action_time_emb)
        )
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = (
            self._fp32_linear(self.action_time_mlp_out, action_time_emb)
            if _fp32
            else self.action_time_mlp_out(action_time_emb)
        )  # [1, state_bs*50, 1024]
        action_time_dim = action_time_emb.shape[1]

        embs = torch.cat([state_emb[:, None], action_time_emb], dim=1)
        pad_masks = torch.ones((bsize, action_time_dim + 1), device=device, dtype=torch.bool)

        # Set attention masks for suffix tokens so that prefix tokens cannot attend to suffix tokens.
        # And state token cannot attend action tokens.
        # Action tokens use a bidirectional attention.
        att_masks = torch.zeros((bsize, action_time_dim + 1), device=device, dtype=torch.bool)
        att_masks[:, :2] = True

        return time_emb_ori, embs, pad_masks, att_masks


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
        vocab_size: int = 0,
        use_lm_head: bool = False,
        attention_implementation: str = "flex_cached",
        tokenizer_path: str | None = None,
        enable_expert_vision: bool = False,
        expert_vision_type: str | None = None,
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        self.tokenizer_path = tokenizer_path
        self.enable_expert_vision = enable_expert_vision
        self.expert_vision_type = expert_vision_type
        self.vocab_size = vocab_size
        self.use_lm_head = use_lm_head
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim
        num_layers = 36

        self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=expert_hidden_size,
            head_dim=action_head_dim,
            initializer_range=0.02,
            intermediate_size=expert_intermediate_size,
            max_position_embeddings=32768,
            max_window_layers=21,
            model_type="qwen2",
            num_attention_heads=action_num_attention_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=action_num_key_value_heads,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="5.14.1",
            use_cache=use_cache,
            use_sliding_window=False,
            vocab_size=151936,
        )
        print(
            "=====Action Expert V2 init "
            f"{num_layers} Layers, hidden={expert_hidden_size}, "
            f"q_heads={action_num_attention_heads}, kv_heads={action_num_key_value_heads}, "
            f"head_dim={action_head_dim}.====="
        )
        super().__init__(**kwargs)


def _resolve_qwen_attention_implementations(
    vision_implementation: str,
    *,
    flash_attention_available: bool,
) -> tuple[str, str]:
    """Resolve Qwen text and vision backends without changing action attention semantics."""
    base_implementation = "flash_attention_2" if flash_attention_available else "eager"
    if vision_implementation == "flash_attention_2" and not flash_attention_available:
        vision_implementation = "sdpa"
    return base_implementation, vision_implementation


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, eval=False):
        super().__init__(config=config)
        self.config = config
        vlm_config = AutoConfig.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        if self.config.vocab_size not in (0, 257152):
            vlm_config.text_config.vocab_size = self.config.vocab_size
        flash_attention_available = is_flash_attn_available()
        base_attn_implementation, vision_attn_implementation = _resolve_qwen_attention_implementations(
            self.config.vit_attn_implementation,
            flash_attention_available=flash_attention_available,
        )
        if self.config.vit_attn_implementation == "flash_attention_2" and not flash_attention_available:
            logger.warning_once("flash-attn is unavailable; using SDPA attention for Qwen3-VL vision")
        vlm_config._attn_implementation = base_attn_implementation
        vlm_config.text_config._attn_implementation = base_attn_implementation
        vlm_config.vision_config._attn_implementation = vision_attn_implementation
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)
        if self.config.use_lm_head:
            self.qwenvl.tie_weights()

        self.config.qwen_expert_config._attn_implementation = base_attn_implementation
        self.qwen_expert = Qwen2ForCausalLM._from_config(self.config.qwen_expert_config, eval=eval)

        if getattr(self.config, "adanorm_time", False):
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )

        self._install_moe_blocks()
        self.pos_embeds = None
        self.position_embeddings = None
        self.cu_seqlens = None
        self.visual_split_sizes = None
        self.visual_max_seqlen = None
        self.visual_sequence_lengths = None
        self._cached_image_grid_signature = None
        self._cuda_graph_fixed_grid = False
        self._cached_visual_pos_indices = None

        del self.qwen_expert.model.embed_tokens
        if self.config.enable_expert_vision:
            if dinov3_vitb16 is None:
                raise ImportError("dinov3 is required when enable_expert_vision=True")
            if "dinov3_vitb16" in self.config.expert_vision_type:
                self.expert_visual = dinov3_vitb16(pretrained=False)
            self.expert_visual_mlp = nn.Sequential(
                nn.Linear(self.expert_visual.embed_dim, self.expert_visual.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.expert_visual.embed_dim * 2, self.config.qwen_expert_config.hidden_size),
            )

        self.attention_interface = self.get_attention_interface()

    def _apply(self, fn):
        super()._apply(fn)
        for name in ("pos_embeds", "position_embeddings", "cu_seqlens"):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                setattr(self, name, fn(value))
            elif isinstance(value, tuple):
                setattr(
                    self,
                    name,
                    tuple(fn(item) if isinstance(item, torch.Tensor) else item for item in value),
                )
        return self

    def _install_moe_blocks(self):
        if not getattr(self.config, "use_moe", False):
            return
        bias_update_speed = getattr(self.config, "bias_update_speed", 0.001)
        hidden_size = self.config.qwen_expert_config.hidden_size
        token_moe_layers = getattr(self.config, "token_moe_layers", None) or []

        _moe_impl = getattr(self.config, "_moe_implementation", None)

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=getattr(self.config, "token_num_experts", 32),
                num_experts_per_tok=getattr(self.config, "token_top_k", 1),
                norm_topk_prob=True,
                hidden_size=hidden_size,
                moe_intermediate_size=getattr(self.config, "token_moe_intermediate_size", 256),
                shared_expert_intermediate_size=getattr(self.config, "token_shared_intermediate_size", 256),
                output_router_logits=False,
            )
            token_config.bias_update_speed = bias_update_speed
            token_config._moe_implementation = _moe_impl
            token_config.router_activation = getattr(self.config, "router_activation", "softmax")
            token_config.routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.0)
            token_config.use_shared_expert_gate = getattr(self.config, "use_shared_expert_gate", True)
            token_config.use_robby_moe_kernel = getattr(self.config, "use_robby_moe_kernel", False)
            for idx in token_moe_layers:
                self.qwen_expert.model.layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        precompute_grid_thw = getattr(self.config, "precompute_grid_thw", False)
        if getattr(self, "_cuda_graph_fixed_grid", False):
            if self._cached_image_grid_signature is None:
                raise RuntimeError("Qwen3-VL fixed-grid CUDA Graph cache was enabled before grid preprocessing")
            grid_signature = self._cached_image_grid_signature
            cache_miss = False
        else:
            grid_signature = tuple(image_grid_thw.detach().to(device="cpu").reshape(-1).tolist())
            cache_miss = self.position_embeddings is None or self._cached_image_grid_signature != grid_signature
        if precompute_grid_thw and cache_miss:
            (
                self.pos_embeds,
                self.position_embeddings,
                self.cu_seqlens,
                self.visual_split_sizes,
                self.visual_max_seqlen,
            ) = self.qwenvl.visual.preprcess_grid_thw(grid_thw=image_grid_thw)
            if self.pos_embeds is None:
                self.pos_embeds = self.qwenvl.visual.fast_pos_embed_interpolate(image_grid_thw)
            self.visual_sequence_lengths = tuple((self.cu_seqlens[1:] - self.cu_seqlens[:-1]).tolist())
            self._cached_image_grid_signature = grid_signature
        image_embeds, deepstack_image_embeds = self.qwenvl.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            pos_embeds=self.pos_embeds,
            position_embeddings=self.position_embeddings,
            cu_seqlens=self.cu_seqlens,
            max_seqlen=self.visual_max_seqlen,
            sequence_lengths=self.visual_sequence_lengths,
        )
        split_sizes = self.visual_split_sizes
        if split_sizes is None:
            split_sizes = (image_grid_thw.prod(-1) // self.qwenvl.visual.spatial_merge_size**2).tolist()
        image_chunks = list(torch.split(image_embeds, split_sizes))
        deepstack_chunks = [
            list(torch.split(deepstack_embeds, split_sizes)) for deepstack_embeds in deepstack_image_embeds
        ]
        image_embeds = torch.stack(image_chunks, dim=0)
        deepstack_image_embeds = [torch.stack(chunks, dim=0) for chunks in deepstack_chunks]
        return image_embeds, deepstack_image_embeds

    def embed_image(self, image: torch.Tensor, image_grid_thw: torch.LongTensor):
        return self.get_image_features(
            image,
            image_grid_thw=image_grid_thw,
        )

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(self, token_id: int, batch: int, count: int, device, dtype):
        weight = self.qwenvl.model.language_model.embed_tokens.weight
        emb = weight[token_id].to(device=device, dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def build_prefix_position_ids(self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids

    def apply_mrope(self, query_states, key_states, position_ids):
        position_embeddings = self.qwenvl.model.language_model.rotary_emb(query_states, position_ids)
        return apply_rotary_pos_emb(query_states, key_states, *position_embeddings, unsqueeze_dim=2)

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)
        return key_states, value_states, past_key_values

    def _apply_deepstack(self, hidden_states, layer_idx, visual_pos_masks, deepstack_visual_embeds):
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            visual_pos_indices = self._cached_visual_pos_indices
            if visual_pos_indices is None:
                visual_pos_indices = torch.nonzero(visual_pos_masks.reshape(-1), as_tuple=False).flatten()
            visual_embeds = deepstack_visual_embeds[layer_idx].to(hidden_states.device, hidden_states.dtype)
            hidden_states.reshape(-1, hidden_states.shape[-1]).index_add_(
                0,
                visual_pos_indices,
                visual_embeds,
            )
        return hidden_states

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        vlm_position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: list[torch.FloatTensor] | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ):
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers
        action_num_layers = self.config.qwen_expert_config.num_hidden_layers
        router_logits_list = []

        if action_num_layers != num_layers:
            raise ValueError(
                "Action expert and VLM must have the same number of layers "
                f"(got action={action_num_layers}, vlm={num_layers})"
            )

        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                if i == 1:
                    q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True, ada_cond=ada_cond)
                else:
                    q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
                query_states.append(q.float())
                key_states.append(k.float())
                value_states.append(v.float())

            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)
            query_states, key_states = self.apply_mrope(query_states, key_states, position_ids)
            key_states, value_states, past_key_values = self.handle_kv_cache(
                key_states,
                value_states,
                layer_idx,
                past_key_values=past_key_values,
                use_cache=use_cache,
                fill_kv_cache=fill_kv_cache,
            )
            if self.config.attention_implementation == "flex_cached":
                if layer_idx == 0:
                    _full_len = query_states.shape[1]
                    _full_block_mask = build_block_mask(
                        attention_mask,
                        self.qwenvl.config.text_config.num_attention_heads,
                        _full_len,
                        _full_len,
                    )
                att_output = flex_attention_with_block_mask(
                    query_states, key_states, value_states, _full_block_mask, query_states.shape[1]
                )
            else:
                att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                end = start + hidden_states.shape[1]
                if i == 1:
                    out_emb, router_logits = models[i].layers[layer_idx](
                        hidden_states,
                        att_output,
                        start,
                        end,
                        output_atten=True,
                        ada_cond=ada_cond,
                    )
                    if router_logits is not None:
                        router_logits_list.append(router_logits)
                else:
                    out_emb = models[i].layers[layer_idx](hidden_states, att_output, start, end, output_atten=True)
                    out_emb = self._apply_deepstack(out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds)
                outputs_embeds.append(out_emb)
                start = end
            inputs_embeds = outputs_embeds

        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb, _ = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values, router_logits_list

    def get_attention_interface(self):
        if self.config.attention_implementation == "flex":
            print("=====Using Flex Attn=====")
            return flex_attention_forward
        if self.config.attention_implementation == "flex_cached":
            print("=====Using Flex Cached (prebuilt BlockMask) Attn=====")
            return flex_attention_forward
        if self.config.attention_implementation == "eager":
            print("=====Using Eager Attn=====")
            return our_eager_attention_forward
        raise ValueError(f"Invalid attention implementation: {self.config.attention_implementation}")


class FlowMatchingV2(FlowMatchingBase):
    def __init__(self, config, eval):
        nn.Module.__init__(self)
        self.config = config
        self._cuda_graph_enabled = False
        self._cuda_graph_runner: LingBotVlaV2CudaGraphs | None = None
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            vocab_size=getattr(self.config, "vocab_size", 0),
            use_lm_head=getattr(self.config, "use_lm_head", False),
            attention_implementation=self.config.attention_implementation,
            tokenizer_path=self.config.tokenizer_path,
            enable_expert_vision=self.config.enable_expert_vision,
            expert_vision_type=self.config.expert_vision_type,
            use_cache=getattr(self.config, "use_cache", True),
            expert_hidden_size=getattr(self.config, "expert_hidden_size", 768),
            expert_intermediate_size=getattr(self.config, "expert_intermediate_size", 2752),
            action_num_attention_heads=getattr(self.config, "action_num_attention_heads", 32),
            action_num_key_value_heads=getattr(self.config, "action_num_key_value_heads", 8),
            action_head_dim=getattr(self.config, "action_head_dim", 128),
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "precompute_grid_thw",
            "vit_attn_implementation",
            "use_moe",
            "bias_update_speed",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
            "use_robby_moe_kernel",
            "_moe_implementation",
        ]:
            if hasattr(config, name):
                setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, eval)
        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size
        self.config.initializer_range = getattr(qwenvl_with_export_config.qwen_expert_config, "initializer_range", None)

        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        self.config.align_params = getattr(self.config, "align_params", None) or {}
        if self.config.align_params != {}:
            self.steps = 0
            self.use_depth_align = True
            self.init_depth_heads(self.config.align_params)
            self.use_future_video = self.config.align_params.get("use_future_video", False)
            if self.use_future_video:
                self.init_video_heads(self.config.align_params)
        else:
            self.use_depth_align = False
            self.use_future_video = False
            self.use_future_video_patch = False
            self.use_current_video_patch = False
            self.use_current_shared_task_proj = False
            self.use_future_video_cls = False
            self.use_shared_future_task_proj = False
            self.future_video_share_future_depth_query = False
            self.block_future_depth_to_action = False
        self._cached_prefix_position_ids = None
        self._cached_deepstack_indices = None

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
    ):
        if image_grid_thw is None:
            raise ValueError("LingbotVlaV2Policy requires image_grid_thw from the Qwen3-VL image processor.")
        bsize = images.shape[0]
        device = images.device
        if images.ndim == 3:
            bsize = 1
            num_images = images.shape[0]
        else:
            num_images = images.shape[1] if images.ndim >= 4 else 1
        if images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if image_grid_thw.ndim == 3:
            flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        else:
            flat_grid_thw = image_grid_thw

        img_emb, deepstack_embs = self.qwenvl_with_expert.embed_image(
            images,
            flat_grid_thw,
        )
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images)
        deepstack_embs = [einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images) for x in deepstack_embs]
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id

        if getattr(self.config, "qwen3vl_use_vision_boundaries", True):
            start_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
            )
            end_emb = self.qwenvl_with_expert.embed_special_token(
                cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
            )
            img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
            image_token_len = num_patch + 2
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = torch.zeros_like(image_pad_masks)
            image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )
            fake_image_ids[:, :, 0] = cfg.vision_start_token_id
            fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        else:
            img_chunks = img_emb
            image_token_len = num_patch
            image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
            image_visual_masks = image_pad_masks
            fake_image_ids = torch.full(
                (bsize, num_images, image_token_len),
                visual_token_id,
                dtype=torch.long,
                device=device,
            )

        img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        visual_pos_masks = einops.rearrange(image_visual_masks, "b n l -> b (n l)")
        fake_image_ids = einops.rearrange(fake_image_ids, "b n l -> b (n l)")

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(dtype=embed_dtype)

        if self.use_depth_align and self.align_type == "query":

            def _get_align_tokens(tokens):
                tk_weights = tokens.view(self.num_task_tokens, tokens.shape[0] // self.num_task_tokens, tokens.shape[1])
                tk_weights = tk_weights.mean(dim=1)
                return tk_weights

            align_pad_masks = torch.ones(bsize, self.num_task_tokens, device=device, dtype=lang_masks.dtype)
            fake_align_ids = torch.full(
                (bsize, self.num_task_tokens), cfg.text_config.eos_token_id, dtype=torch.long, device=device
            )

            current_task = _get_align_tokens(self.depth_align_embs)
            if (
                getattr(self, "use_future_video", False)
                and getattr(self, "use_current_video_patch", False)
                and getattr(self, "use_current_shared_task_proj", False)
            ):
                current_video_task = _get_align_tokens(self.current_video_align_embs)
                current_task = self.current_shared_task_proj(torch.cat([current_task, current_video_task], dim=-1))
            align_embs = current_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)
            parts = [img_emb]
            masks = [image_pad_masks]
            input_ids = [fake_image_ids]
            visual_masks = [visual_pos_masks]

            def _append(
                tokens,
                token_masks,
                token_ids,
                token_visual_masks=None,
            ):
                parts.append(tokens)
                masks.append(token_masks)
                input_ids.append(token_ids)
                if token_visual_masks is None:
                    token_visual_masks = torch.zeros_like(token_masks)
                visual_masks.append(token_visual_masks)

            future_align_embs = None
            if self.use_future_depth:
                future_task = _get_align_tokens(self.future_depth_align_embs)
                if (
                    getattr(self, "use_future_video", False)
                    and getattr(self, "use_future_video_patch", True)
                    and getattr(self, "future_video_share_future_depth_query", False)
                    and getattr(self, "use_shared_future_task_proj", False)
                ):
                    future_video_task = _get_align_tokens(self.future_video_align_embs)
                    future_task = self.future_shared_task_proj(torch.cat([future_task, future_video_task], dim=-1))
                future_align_embs = future_task.repeat(img_emb.size(0), 1, 1).to(img_emb.device, img_emb.dtype)

            if (
                not self.use_future_depth
                and getattr(self, "use_future_video", False)
                and getattr(self, "future_video_share_future_depth_query", False)
            ):
                raise ValueError("share_future_depth_query=True requires depth.use_future_depth=True.")

            for segment_name in prefix_query_segments(
                use_depth_align=True,
                use_future_depth=self.use_future_depth,
                use_future_video=getattr(self, "use_future_video", False),
                use_future_video_cls=getattr(self, "use_future_video_cls", False),
                use_future_video_patch=getattr(self, "use_future_video_patch", True),
                future_video_share_future_depth_query=getattr(
                    self,
                    "future_video_share_future_depth_query",
                    False,
                ),
            ):
                if segment_name == "language":
                    _append(
                        lang_emb,
                        lang_masks,
                        lang_tokens.to(device),
                    )
                elif segment_name == "current_depth":
                    _append(align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_video_cls":
                    future_video_cls_align_emb = self.future_video_cls_align_emb.weight.repeat(
                        img_emb.size(0), 1, 1
                    ).to(img_emb.device, img_emb.dtype)
                    cls_align_pad_masks = torch.ones(
                        bsize,
                        1,
                        device=device,
                        dtype=lang_masks.dtype,
                    )
                    fake_cls_align_ids = torch.full(
                        (bsize, 1),
                        cfg.text_config.eos_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    _append(future_video_cls_align_emb, cls_align_pad_masks, fake_cls_align_ids)
                elif segment_name == "future_video":
                    future_video_align_embs = (
                        _get_align_tokens(self.future_video_align_embs)
                        .repeat(img_emb.size(0), 1, 1)
                        .to(img_emb.device, img_emb.dtype)
                    )
                    _append(future_video_align_embs, align_pad_masks, fake_align_ids)
                elif segment_name == "future_depth":
                    _append(future_align_embs, align_pad_masks, fake_align_ids)
                else:
                    raise ValueError(f"Unsupported prefix query segment: {segment_name}")

            embs = torch.cat(parts, dim=1)
            pad_masks = torch.cat(masks, dim=1)
            prefix_input_ids = torch.cat(input_ids, dim=1)
            full_visual_pos_masks = torch.cat(visual_masks, dim=1)
        else:
            embs = torch.cat([img_emb, lang_emb], dim=1)
            pad_masks = torch.cat([image_pad_masks, lang_masks], dim=1)
            prefix_input_ids = torch.cat([fake_image_ids, lang_tokens.to(device)], dim=1)
            full_visual_pos_masks = torch.cat([visual_pos_masks, torch.zeros_like(lang_masks)], dim=1)

        if getattr(self.config, "vlm_causal", False):
            att_masks = torch.ones((bsize, embs.shape[1]), device=device, dtype=torch.bool)
        else:
            att_masks = torch.zeros((bsize, embs.shape[1]), device=device, dtype=torch.bool)

        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        fixed_prefix_layout = getattr(self.qwenvl_with_expert, "_cuda_graph_fixed_grid", False)
        if fixed_prefix_layout:
            if (
                self._cached_prefix_position_ids is None
                or self._cached_deepstack_indices is None
                or self.qwenvl_with_expert._cached_visual_pos_indices is None
            ):
                raise RuntimeError("LingBot-VLA v2 fixed prefix layout was enabled before prefix preprocessing")
            prefix_position_ids = self._cached_prefix_position_ids
            deepstack_indices = self._cached_deepstack_indices
        else:
            flat_img_masks = einops.rearrange(img_masks, "b n -> (b n)")
            active_image_indices = torch.nonzero(flat_img_masks, as_tuple=False).flatten()
            if active_image_indices.numel() == 0:
                active_image_indices = torch.zeros(1, dtype=torch.long, device=flat_grid_thw.device)
            rope_grid_thw = flat_grid_thw.index_select(0, active_image_indices)
            prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
                prefix_input_ids,
                pad_masks.long(),
                image_grid_thw=rope_grid_thw,
                video_grid_thw=None,
            )
            deepstack_indices = torch.nonzero(img_visual_only.reshape(-1), as_tuple=False).flatten()
            self._cached_prefix_position_ids = prefix_position_ids
            self._cached_deepstack_indices = deepstack_indices
            self.qwenvl_with_expert._cached_visual_pos_indices = torch.nonzero(
                full_visual_pos_masks.reshape(-1), as_tuple=False
            ).flatten()

        filtered_deepstack = []
        for deepstack in deepstack_embs:
            filtered_deepstack.append(deepstack.reshape(-1, deepstack.shape[-1]).index_select(0, deepstack_indices))

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            filtered_deepstack,
        )
        return result

    def _build_full_position_ids(self, prefix_position_ids, prefix_pad_masks, suffix_pad_masks):
        valid_prefix_pos = prefix_position_ids.masked_fill(~prefix_pad_masks.unsqueeze(0), 0)
        prefix_offsets = valid_prefix_pos.amax(dim=(0, 2)) + 1
        suffix_1d = prefix_offsets[:, None] + torch.cumsum(suffix_pad_masks.long(), dim=1) - 1
        suffix_1d = suffix_1d.masked_fill(~suffix_pad_masks, 1)
        suffix_position_ids = suffix_1d.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([prefix_position_ids, suffix_position_ids], dim=-1)

    def _current_depth_task_tokens(self, hidden_states, num_images=3):
        query_spans = prefix_query_token_spans(
            prefix_len=hidden_states.shape[1],
            num_task_tokens=self.num_task_tokens,
            use_depth_align=True,
            use_future_depth=getattr(self, "use_future_depth", False),
            use_future_video=getattr(self, "use_future_video", False),
            use_future_video_cls=getattr(self, "use_future_video_cls", False),
            use_future_video_patch=getattr(self, "use_future_video_patch", True),
            future_video_share_future_depth_query=getattr(
                self,
                "future_video_share_future_depth_query",
                False,
            ),
        )
        start, end = query_spans["current_depth"]
        return hidden_states[:, start:end, :]

    def forward(self, *args, **kwargs):
        """Reject the upstream training API in the inference-only model."""
        del args, kwargs
        raise RuntimeError("LingBot-VLA v2 is inference-only; use sample_actions()")

    def set_cuda_graph_enabled(self, enabled: bool) -> None:
        """Enable lazy capture of fixed-shape prefix encoding and denoising."""
        self._cuda_graph_enabled = bool(enabled)
        if not self._cuda_graph_enabled:
            self._cuda_graph_runner = None
            self.set_prefix_cuda_graph_capture(False)
            self._cached_prefix_position_ids = None
            self._cached_deepstack_indices = None
            self.qwenvl_with_expert._cached_visual_pos_indices = None

    def set_prefix_cuda_graph_capture(self, enabled: bool) -> None:
        """Skip host grid-signature work while replaying a validated fixed prefix graph."""
        self.qwenvl_with_expert._cuda_graph_fixed_grid = bool(enabled)

    @property
    def cuda_graph_enabled(self) -> bool:
        return self._cuda_graph_enabled

    @property
    def cuda_graph_ready(self) -> bool:
        return self._cuda_graph_runner is not None and self._cuda_graph_runner.ready

    def build_prefix_cache(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        """Encode the multimodal prefix and construct the action expert KV cache."""
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        _, past_key_values, _ = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            vlm_position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        if past_key_values is None:
            raise RuntimeError("LingBot-VLA v2 prefix encoding did not produce a KV cache")
        return prefix_pad_masks, prefix_position_ids, past_key_values

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_grid_thw=None,
    ) -> Tensor:
        """Do a full Qwen3-VL inference forward and compute the action."""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        if self._cuda_graph_enabled:
            if device.type != "cuda":
                raise RuntimeError("LingBot-VLA v2 CUDA Graph requires CUDA inference")
            if image_grid_thw is None:
                raise RuntimeError("LingBot-VLA v2 prefix CUDA Graph requires image_grid_thw")
            if getattr(self, "_use_compile_predict_velocity", False):
                raise RuntimeError("LingBot-VLA v2 CUDA Graph and torch.compile cannot be enabled together")
            if self._cuda_graph_runner is None:
                self._cuda_graph_runner = LingBotVlaV2CudaGraphs(self)
            return self._cuda_graph_runner.run(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise,
                image_grid_thw,
            )

        prefix_pad_masks, prefix_position_ids, past_key_values = self.build_prefix_cache(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw,
        )

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        predict_velocity_fn = self.predict_velocity
        if getattr(self, "_use_compile_predict_velocity", False):
            predict_velocity_fn = getattr(self, "_compiled_predict_velocity", None)
            if predict_velocity_fn is None:
                predict_velocity_fn = torch.compile(
                    self.predict_velocity,
                    fullgraph=False,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
                self._compiled_predict_velocity = predict_velocity_fn

        for _ in range(int(self.config.num_steps)):
            expanded_time = time.expand(bsize)
            v_t = predict_velocity_fn(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                prefix_position_ids=prefix_position_ids,
            )

            x_t += dt * v_t
            time += dt
        logger.debug(
            "Denoised actions in %d steps",
            self.config.num_steps,
        )
        return x_t

    def predict_velocity(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        prefix_position_ids=None,
    ):
        """Predict velocity at time t using cached Qwen3-VL prefix states."""
        if prefix_position_ids is None:
            raise ValueError("FlowMatchingV2.predict_velocity requires Qwen3-VL prefix_position_ids.")

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state,
            x_t,
            timestep,
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_len,
            prefix_len,
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        if self.block_future_depth_to_action:
            # Query rows here are all suffix (state/action), so row start is 0.
            full_att_2d_masks = block_suffix_to_fv_(
                full_att_2d_masks,
                suffix_row_start=0,
                prefix_len=prefix_len,
                num_task_tokens=self.num_task_tokens,
            )
        full_att_2d_masks = self._block_suffix_to_future_video_if_enabled_(
            full_att_2d_masks,
            suffix_row_start=0,
            prefix_len=prefix_len,
        )

        full_position_ids = self._build_full_position_ids(
            prefix_position_ids,
            prefix_pad_masks,
            suffix_pad_masks,
        )
        position_ids = full_position_ids[:, :, -suffix_len:]

        outputs_embeds, _, _ = self.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            ada_cond=time_embs if getattr(self.config, "adanorm_time", False) else None,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        if getattr(self.config, "action_fp32", False):
            v_t = self._fp32_linear(self.action_out_proj, suffix_out)
        else:
            if suffix_out.dtype != self.action_out_proj.weight.dtype:
                suffix_out = suffix_out.to(self.action_out_proj.weight.dtype)
            v_t = self.action_out_proj(suffix_out)
        return v_t


class LingbotVlaV2Policy(PreTrainedModel):
    config_class = LingbotVLAV2Config
    name = "torch_lingbot_vla_v2"
    _no_split_modules = ["Qwen2DecoderLayer", "FixQwen2RMSNorm", "FixAdaRMSNorm"]

    def __init__(self, config: LingbotVLAV2Config, eval: bool = True):
        if not eval:
            raise ValueError("LingBot-VLA v2 only supports inference mode")
        super().__init__(config)
        self.config = config
        self.model = FlowMatchingV2(config, eval)
        if not getattr(self.config, "use_lm_head", False):
            del self.model.qwenvl_with_expert.qwenvl.lm_head
        del self.model.qwenvl_with_expert.qwen_expert.lm_head
        self.requires_grad_(False)
        self.eval()
        self.reset()

    def reset(self):
        return None

    def set_cuda_graph_enabled(self, enabled: bool) -> None:
        """Configure fixed-step CUDA Graph execution for this policy."""
        self.model.set_cuda_graph_enabled(enabled)

    @property
    def cuda_graph_ready(self) -> bool:
        return self.model.cuda_graph_ready

    def forward(self, *args, **kwargs):
        """Reject the upstream training API in the inference-only model."""
        del args, kwargs
        raise RuntimeError("LingBot-VLA v2 is inference-only; use sample_actions()")

    def sample_actions(self, *args, **kwargs) -> Tensor:
        return self.model.sample_actions(*args, **kwargs)


ModelClass = LingbotVlaV2Policy

__all__ = [
    "LingbotVlaV2Policy",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLTextModel",
    "Qwen3VLPreTrainedModel",
    "Qwen2ForCausalLM",
]


class LingBotVlaV2Model(LingbotVlaV2Policy):
    """TeleFuser-native entry point preserving official checkpoint key names."""

    name = "lingbot_vla_v2"

    def __init__(self, config, eval=True):
        super().__init__(config=config, eval=eval)

    @staticmethod
    def state_dict_converter(**kwargs):
        return LingBotVlaV2StateDictConverter(**kwargs)

    def enable_quant(self, quant_config: QuantConfig) -> None:
        """Apply supported online quantization without changing action or MoE heads."""
        if not isinstance(quant_config, QuantConfig):
            raise TypeError("LingBot-VLA v2 online quantization requires QuantConfig")
        if not quant_config.enabled:
            return

        existing_quant_type = getattr(self, "quant_type", None)
        if existing_quant_type is not None:
            requested_backend = quant_config.kernel_backend
            if requested_backend == QuantKernelBackend.AUTO:
                requested_backend = {
                    QuantType.TORCHAO_FP8: QuantKernelBackend.TORCHAO,
                    QuantType.FP8: QuantKernelBackend.TF_KERNEL,
                    QuantType.BNB_NF4: QuantKernelBackend.BITSANDBYTES,
                }.get(quant_config.quant_type, requested_backend)
            if (
                existing_quant_type == quant_config.quant_type
                and getattr(self, "quant_kernel_backend", None) == requested_backend
            ):
                return
            raise RuntimeError(
                "LingBot-VLA v2 is already quantized as "
                f"{existing_quant_type}/{getattr(self, 'quant_kernel_backend', None)}, cannot apply "
                f"{quant_config.quant_type}/{requested_backend}"
            )

        if quant_config.quant_type == QuantType.FP8 and quant_config.kernel_backend == QuantKernelBackend.CUTLASS:
            self._enable_fused_fp8_graph()
            return

        profiles = {
            QuantType.TORCHAO_FP8: "torchao-fp8",
            QuantType.FP8: "tf-kernel-fp8",
            QuantType.BNB_NF4: "bnb-nf4",
        }
        effective_backends = {
            QuantType.TORCHAO_FP8: QuantKernelBackend.TORCHAO,
            QuantType.FP8: QuantKernelBackend.TF_KERNEL,
            QuantType.BNB_NF4: QuantKernelBackend.BITSANDBYTES,
        }
        if quant_config.quant_type not in profiles:
            raise ValueError(f"LingBot-VLA v2 does not support online quantization type {quant_config.quant_type.name}")

        include_names = quant_config.quantize_modules or LINGBOT_VLA_V2_DEFAULT_QUANTIZE_MODULES
        exclude_names = tuple(dict.fromkeys((*quant_config.skip_modules, *LINGBOT_VLA_V2_REQUIRED_SKIP_MODULES)))
        manifest = build_lingbot_vla_v2_linear_manifest(
            self,
            include_names=include_names,
            exclude_names=exclude_names,
        )
        selected_count = int(manifest["selected_count"])
        if selected_count == 0:
            raise RuntimeError("LingBot-VLA v2 online quantization did not select any Linear layers")
        frozen_official_profile = (
            getattr(getattr(self, "config", None), "checkpoint_variant", None) == "base"
            and quant_config.quantize_modules is None
            and quant_config.skip_modules == QuantConfig().skip_modules
        )
        if frozen_official_profile:
            manifest_sha256 = str(manifest["manifest_sha256"])
            if (
                selected_count != LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT
                or manifest_sha256 != LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZATION_MANIFEST_SHA256
            ):
                raise RuntimeError(
                    "LingBot-VLA v2 official 6B quantization manifest changed: "
                    f"expected count={LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZED_LINEAR_COUNT} "
                    f"sha256={LINGBOT_VLA_V2_OFFICIAL_6B_QUANTIZATION_MANIFEST_SHA256}, "
                    f"got count={selected_count} sha256={manifest_sha256}"
                )

        if quant_config.quant_type == QuantType.TORCHAO_FP8:
            if quant_config.kernel_backend not in (QuantKernelBackend.AUTO, QuantKernelBackend.TORCHAO):
                raise ValueError(
                    f"LingBot-VLA v2 TorchAO FP8 requires the TorchAO backend; got {quant_config.kernel_backend.name}"
                )
            from telefuser.ops.torchao_fp8_linear import replace_linear_layers_with_torchao_fp8

            replaced = replace_linear_layers_with_torchao_fp8(
                self,
                include_names=include_names,
                exclude_names=exclude_names,
            )
            self.torchao_fp8_replaced_linear = replaced
        elif quant_config.quant_type == QuantType.BNB_NF4:
            if quant_config.kernel_backend not in (QuantKernelBackend.AUTO, QuantKernelBackend.BITSANDBYTES):
                raise ValueError(
                    f"LingBot-VLA v2 BNB NF4 requires the bitsandbytes backend; got {quant_config.kernel_backend.name}"
                )
            from telefuser.ops.bnb_nf4_linear import replace_linear_layers_with_bnb_nf4

            replaced = replace_linear_layers_with_bnb_nf4(
                self,
                compute_dtype=torch.bfloat16,
                include_names=include_names,
                exclude_names=exclude_names,
            )
            self.bnb_nf4_replaced_linear = replaced
        elif quant_config.quant_type == QuantType.FP8:
            if quant_config.kernel_backend not in (QuantKernelBackend.AUTO, QuantKernelBackend.TF_KERNEL):
                raise ValueError(
                    "LingBot-VLA v2 FP8 online quantization requires the tf-kernel backend; "
                    f"got {quant_config.kernel_backend.name}"
                )
            from telefuser.ops.fp8_gemm import FP8GemmOptions, count_linear_layers, enable_fp8_gemm

            def module_filter(name: str, _module: nn.Module) -> bool:
                return any(token in name for token in include_names) and not any(
                    token and token in name for token in exclude_names
                )

            replaced = count_linear_layers(self, module_filter=module_filter)
            enable_fp8_gemm(
                self,
                options=FP8GemmOptions(
                    fp16_weight_storage="keep" if quant_config.keep_fp16_weight else "discard",
                    materialize_fp8_on_wrap=True,
                ),
                module_filter=module_filter,
            )
            self.tf_kernel_fp8_replaced_linear = replaced
        if replaced != selected_count:
            raise RuntimeError(
                f"LingBot-VLA v2 quantization selected {selected_count} Linear layers but converted {replaced}"
            )
        self.quant_type = quant_config.quant_type
        self.quant_kernel_backend = effective_backends[quant_config.quant_type]
        finalize_lingbot_vla_v2_quantization_identity(
            self,
            profile=profiles[quant_config.quant_type],
            quant_type=quant_config.quant_type.name,
            kernel_backend=effective_backends[quant_config.quant_type].name,
            manifest=manifest,
        )
        logger.info(
            "LingBot-VLA v2 %s converted %d selected Linear layers (manifest %s); "
            "fused MoE and action heads remain BF16",
            quant_config.quant_type.name,
            replaced,
            manifest["manifest_sha256"],
        )

    def _enable_fused_fp8_graph(self) -> None:
        """Quantize only the ten-step action path with graph-safe FP8 kernels."""
        if not next(self.parameters()).is_cuda:
            raise RuntimeError("LingBot-VLA v2 fused FP8 graph requires CUDA-resident model weights")
        if torch.cuda.get_device_capability(next(self.parameters()).device) < (9, 0):
            raise RuntimeError("LingBot-VLA v2 fused FP8 graph requires Hopper or newer CUDA hardware")

        action_layers = self.model.qwenvl_with_expert.qwen_expert.model.layers
        include_names = tuple(
            token
            for layer_idx in range(len(action_layers))
            for token in (
                f"qwenvl_with_expert.qwen_expert.model.layers.{layer_idx}.self_attn.",
                f"qwenvl_with_expert.qwen_expert.model.layers.{layer_idx}.mlp.shared_expert.",
            )
        ) + ("action_time_mlp_",)
        exclude_names: tuple[str, ...] = ()
        manifest = build_lingbot_vla_v2_linear_manifest(
            self,
            include_names=include_names,
            exclude_names=exclude_names,
        )

        from telefuser.ops.graph_fp8_linear import replace_linear_layers_with_graph_fp8

        def module_filter(name: str, module: nn.Linear) -> bool:
            selected = any(token in name for token in include_names) and not any(
                token in name for token in exclude_names
            )
            return selected and module.in_features % 16 == 0 and module.out_features % 16 == 0

        replaced = replace_linear_layers_with_graph_fp8(self, module_filter=module_filter)
        selected_count = int(manifest["selected_count"])
        if replaced != selected_count:
            raise RuntimeError(
                f"LingBot-VLA v2 fused FP8 graph selected {selected_count} Linear layers but converted {replaced}"
            )

        fused_expert_layers = 0
        for layer in action_layers:
            mlp = getattr(layer, "mlp", None)
            experts = getattr(mlp, "experts", None)
            if not isinstance(experts, Qwen2FusedExperts):
                raise RuntimeError("LingBot-VLA v2 fused FP8 graph requires fused expert storage")
            experts.enable_graph_fp8()
            fused_expert_layers += 1

        self.graph_fp8_replaced_linear = replaced
        self.graph_fp8_fused_expert_layers = fused_expert_layers
        self.quant_type = QuantType.FP8
        self.quant_kernel_backend = QuantKernelBackend.CUTLASS
        identity = finalize_lingbot_vla_v2_quantization_identity(
            self,
            profile="fused-fp8-graph",
            quant_type=QuantType.FP8.name,
            kernel_backend=QuantKernelBackend.CUTLASS.name,
            manifest=manifest,
        )
        identity["implementation"].update(
            {
                "fused_expert_layers": fused_expert_layers,
                "fused_expert_weight_dtype": "float8_e4m3fn",
                "bf16_expert_weights_retained": False,
            }
        )
        self._lingbot_vla_v2_quantization_identity = identity
        logger.info(
            "LingBot-VLA v2 fused FP8 graph converted %d action Linear layers and %d routed MoE layers",
            replaced,
            fused_expert_layers,
        )
