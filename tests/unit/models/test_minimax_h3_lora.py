from __future__ import annotations

import torch
from safetensors.torch import save_file

from telefuser.core.config import LoraConfig
from telefuser.models.minimax_h3_dit import MiniMaxH3DiT, MiniMaxH3DiTConfig
from telefuser.models.minimax_h3_lora import MiniMaxH3LoraAdapter


def _small_model() -> MiniMaxH3DiT:
    return MiniMaxH3DiT(
        MiniMaxH3DiTConfig(
            hidden_size=8,
            num_layers=1,
            token_refiner_num_layers=1,
            num_attention_heads=2,
            attention_head_dim=4,
            ffn_hidden_size=12,
            latents_dim=2,
            audio_latents_dim=4,
            patch_size=(1, 2, 2),
            text_dim=6,
            timestep_input_dim=4,
            time_embed_hidden_size=8,
            time_embed_dim=4,
            rope_inv_freq_len=0,
        )
    ).eval()


def test_h3_turbo_lora_merges_diffusers_attention_and_ffn_keys(tmp_path) -> None:
    model = _small_model()
    qkv_before = model.blocks[0].attn.qkv_proj.weight.detach().clone()
    fc2_before = model.blocks[0].mlp.fc2.weight.detach().clone()
    refiner_fc2_before = model.token_refiner.blocks[0].mlp.fc2.weight.detach().clone()
    weights = {
        "base_model.model.transformer_blocks.0.attn.to_q.lora_A.default.weight": torch.ones(2, 8),
        "base_model.model.transformer_blocks.0.attn.to_q.lora_B.default.weight": torch.ones(8, 2),
        "base_model.model.transformer_blocks.0.attn.to_k.lora_A.default.weight": torch.ones(2, 8),
        "base_model.model.transformer_blocks.0.attn.to_k.lora_B.default.weight": torch.ones(8, 2),
        "base_model.model.transformer_blocks.0.ff.net.2.lora_A.default.weight": torch.ones(2, 12),
        "base_model.model.transformer_blocks.0.ff.net.2.lora_B.default.weight": torch.ones(8, 2),
        "base_model.model.token_refiner.refiner_blocks.0.ff.net.2.lora_A.default.weight": torch.ones(2, 12),
        "base_model.model.token_refiner.refiner_blocks.0.ff.net.2.lora_B.default.weight": torch.ones(8, 2),
    }
    path = tmp_path / "turbo.safetensors"
    save_file(weights, str(path))

    assert MiniMaxH3LoraAdapter.apply(model, [LoraConfig(path=str(path), strength=0.5)]) == 4

    scale = 0.5 * 128 / 2
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight[:8], qkv_before[:8] + scale * 2)
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight[8:16], qkv_before[8:16] + scale * 2)
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight[16:], qkv_before[16:])
    torch.testing.assert_close(model.blocks[0].mlp.fc2.weight, fc2_before + scale * 2)
    torch.testing.assert_close(model.token_refiner.blocks[0].mlp.fc2.weight, refiner_fc2_before + scale * 2)


def test_h3_turbo_lora_rejects_missing_pairs(tmp_path) -> None:
    path = tmp_path / "incomplete.safetensors"
    save_file(
        {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 8)},
        str(path),
    )

    try:
        MiniMaxH3LoraAdapter.apply(_small_model(), [LoraConfig(path=str(path))])
    except ValueError as exc:
        assert "incomplete LoRA pairs" in str(exc)
    else:
        raise AssertionError("incomplete H3 LoRA should fail")


def test_h3_fastvideo_hybrid_adapter_merges_low_rank_and_exact_deltas(tmp_path) -> None:
    model = _small_model()
    qkv_before = model.blocks[0].attn.qkv_proj.weight.detach().clone()
    patch_before = model.video_patch_proj.weight.detach().clone()
    condition_bias_before = model.condition_proj.bias.detach().clone()
    adaln_bias_before = model.blocks[0].adaln_proj.linear.bias.detach().clone()
    q_norm_before = model.blocks[0].attn.q_norm.weight.detach().clone()
    weights = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones(2, 8),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.ones(8, 2),
        "proj_in.diff": torch.ones_like(patch_before),
        "context_embedder.diff_b": torch.ones_like(condition_bias_before),
        "transformer_blocks.0.adaln_proj.linear.diff_b": torch.ones_like(adaln_bias_before),
        "transformer_blocks.0.attn.norm_q.diff": torch.ones_like(q_norm_before),
    }
    adapter_dir = tmp_path / "dense-datafree"
    adapter_dir.mkdir()
    save_file(
        weights,
        str(adapter_dir / "adapter_model.safetensors"),
        metadata={"format": "fastvideo-lora-v2", "rank": "2"},
    )

    assert MiniMaxH3LoraAdapter.apply(model, [LoraConfig(path=str(adapter_dir), strength=0.5)]) == 5

    # FastVideo's hybrid format is W += B @ A, with no implicit alpha/rank scale.
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight[:8], qkv_before[:8] + 1)
    torch.testing.assert_close(model.blocks[0].attn.qkv_proj.weight[8:], qkv_before[8:])
    torch.testing.assert_close(model.video_patch_proj.weight, patch_before + 0.5)
    torch.testing.assert_close(model.condition_proj.bias, condition_bias_before + 0.5)
    torch.testing.assert_close(model.blocks[0].adaln_proj.linear.bias, adaln_bias_before + 0.5)
    torch.testing.assert_close(model.blocks[0].attn.q_norm.weight, q_norm_before + 0.5)


def test_h3_fastvideo_vsa_adapter_rejects_missing_gate_backend(tmp_path) -> None:
    path = tmp_path / "vsa.safetensors"
    save_file(
        {"transformer_blocks.0.attn.to_gate_compress.set_weight": torch.ones(8, 8)},
        str(path),
        metadata={"format": "fastvideo-lora-v2"},
    )

    try:
        MiniMaxH3LoraAdapter.apply(_small_model(), [LoraConfig(path=str(path))])
    except ValueError as exc:
        assert "VSA compression-gate" in str(exc)
        assert "dense FastH3 adapter" in str(exc)
    else:
        raise AssertionError("VSA-only FastH3 adapter should fail without its gate backend")
