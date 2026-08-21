"""LTX-2.5 isolated AV-transformer architecture tests."""

from __future__ import annotations

import torch

from telefuser.core.config import AttentionConfig, AttnImplType
from telefuser.models.ltx25.transformer import (
    Attention,
    LTX25AVTransformer,
    Modality,
    ltx25_transformer_key_to_model_key,
)


def _small_ltx25_config() -> dict:
    return {
        "transformer": {
            "num_layers": 48,
            "rope_type": "split",
            "apply_gated_attention": True,
            "ff_bias": False,
            "caption_proj_before_connector": True,
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "num_vector_embeds": None,
            "dropout": 0.0,
            "num_embeds_ada_norm": 1000,
            "use_linear_projection": False,
            "only_cross_attention": False,
            "cross_attention_norm": True,
            "double_self_attention": False,
            "upcast_attention": False,
            "standardization_norm": "rms_norm",
            "norm_elementwise_affine": False,
            "qk_norm": "rms_norm",
            "positional_embedding_type": "rope",
            "use_audio_video_cross_attention": True,
            "share_ff": False,
            "av_cross_ada_norm": True,
            "use_middle_indices_grid": True,
            "num_attention_heads": 1,
            "attention_head_dim": 4,
            "in_channels": 4,
            "out_channels": 4,
            "cross_attention_dim": 4,
            "audio_num_attention_heads": 1,
            "audio_attention_head_dim": 4,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_cross_attention_dim": 4,
            "norm_eps": 1e-6,
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [20, 32, 32],
            "audio_positional_embedding_max_pos": [20],
            "timestep_scale_multiplier": 1000,
            "av_ca_timestep_scale_multiplier": 1000.0,
            "frequencies_precision": "float64",
            "cross_attention_adaln": True,
            "use_keyframes_abs_pos_embedding": True,
        }
    }


def test_ltx25_transformer_builds_metadata_architecture_with_asymmetric_ff_biases() -> None:
    with torch.device("meta"):
        model = LTX25AVTransformer(_small_ltx25_config())
    keys = set(model.state_dict())
    assert "velocity_model.transformer_blocks.0.ff.net.0.proj.bias" not in keys
    assert "velocity_model.transformer_blocks.0.audio_ff.net.0.proj.bias" in keys
    assert "velocity_model.keyframes_abs_pos_embedding" in keys


def test_transformer_split_checkpoint_mapping_excludes_embedding_processor_weights() -> None:
    assert (
        ltx25_transformer_key_to_model_key("model.diffusion_model.proj_out.weight") == "velocity_model.proj_out.weight"
    )
    assert (
        ltx25_transformer_key_to_model_key("model.diffusion_model.video_embeddings_connector.learnable_registers")
        is None
    )
    assert ltx25_transformer_key_to_model_key("unrelated.weight") is None


def test_ltx25_transformer_exposes_the_runtime_attention_override() -> None:
    with torch.device("meta"):
        model = LTX25AVTransformer(_small_ltx25_config())

    model.set_attention_config(AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))
    assert Attention.attention_config.attn_impl is AttnImplType.TORCH_SDPA


def test_cross_attention_scale_shift_uses_per_token_timesteps() -> None:
    model = LTX25AVTransformer(_small_ltx25_config())
    video = Modality(
        latent=torch.randn(1, 3, 4),
        sigma=torch.tensor([1.0]),
        timesteps=torch.ones(1, 3),
        positions=torch.zeros(1, 3, 3, 2),
        context=torch.randn(1, 2, 4),
    )
    audio = Modality(
        latent=torch.randn(1, 2, 4),
        sigma=torch.tensor([1.0]),
        timesteps=torch.ones(1, 2),
        positions=torch.zeros(1, 1, 2, 2),
        context=torch.randn(1, 2, 4),
    )

    args = model.velocity_model.audio_args_preprocessor.prepare(audio, video)

    assert args.cross_scale_shift_timestep is not None
    assert args.cross_scale_shift_timestep.shape == (1, 2, 16)
