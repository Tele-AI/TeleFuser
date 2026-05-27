from __future__ import annotations

import torch
from transformers import Qwen3Config, Qwen3Model

from telefuser.core.base_model import BaseModel


class ZImageTextEncoder(BaseModel):
    """Text encoder for Z-Image using Qwen3."""

    def __init__(self):
        super().__init__()
        config = Qwen3Config(
            **{
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 2560,
                "initializer_range": 0.02,
                "intermediate_size": 9728,
                "max_position_embeddings": 40960,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06,
                "rope_scaling": None,
                "rope_theta": 1000000,
                "sliding_window": None,
                "tie_word_embeddings": True,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936,
            }
        )
        self.model = Qwen3Model(config)
        self.config = config

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        output_hidden_states: bool = True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return outputs.hidden_states

    @staticmethod
    def state_dict_converter():
        return ZImageTextEncoderStateDictConverter()


class ZImageTextEncoderStateDictConverter:
    """State dict converter for Z-Image text encoder."""

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return state_dict


# --- Model registry: hash-based detection ---
from telefuser.core.model_registry import register_model_config

register_model_config(
    None, "0f050f62a88876fea6eae0a18dac5a2e", ["zimage_text_encoder"], [ZImageTextEncoder], "diffusers"
)
