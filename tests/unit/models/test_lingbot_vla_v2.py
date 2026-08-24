from types import SimpleNamespace

import pytest
import torch

from telefuser.models import lingbot_vla_v2_loader
from telefuser.models.lingbot_vla_v2 import (
    LingbotVlaV2Policy,
    QwenvlWithExpertV2Model,
    _resolve_qwen_attention_implementations,
)
from telefuser.models.lingbot_vla_v2_moe import Qwen2ForCausalLM
from telefuser.models.lingbot_vla_v2_qwen import Qwen3VLForConditionalGeneration


class _Visual:
    spatial_merge_size = 1

    def __init__(self) -> None:
        self.preprocess_calls = 0

    def preprcess_grid_thw(self, grid_thw: torch.Tensor):
        self.preprocess_calls += 1
        token_count = int(grid_thw.prod(dim=-1).sum())
        position_embeddings = (torch.zeros(token_count, 2), torch.ones(token_count, 2))
        cu_seqlens = torch.tensor([0, token_count], dtype=torch.int32)
        split_sizes = grid_thw.prod(dim=-1).tolist()
        return None, position_embeddings, cu_seqlens, split_sizes, token_count

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor):
        token_count = int(grid_thw.prod(dim=-1).sum())
        return torch.zeros(token_count, 3)

    def __call__(self, pixel_values: torch.Tensor, **kwargs):
        del kwargs
        embeddings = torch.zeros(pixel_values.shape[0], 3)
        return embeddings, [embeddings.clone()]


def _model(visual: _Visual) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(precompute_grid_thw=True),
        qwenvl=SimpleNamespace(visual=visual),
        pos_embeds=None,
        position_embeddings=None,
        cu_seqlens=None,
        visual_split_sizes=None,
        visual_max_seqlen=None,
        visual_sequence_lengths=None,
        _cached_image_grid_signature=None,
    )


def test_image_grid_cache_is_reused_and_invalidated_by_grid_shape() -> None:
    visual = _Visual()
    model = _model(visual)
    first_grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    second_grid = torch.tensor([[1, 1, 2], [1, 1, 2]])

    first = QwenvlWithExpertV2Model.get_image_features(model, torch.zeros(8, 6), first_grid)
    repeated = QwenvlWithExpertV2Model.get_image_features(model, torch.zeros(8, 6), first_grid.clone())
    changed = QwenvlWithExpertV2Model.get_image_features(model, torch.zeros(4, 6), second_grid)

    assert visual.preprocess_calls == 2
    assert first[0].shape == repeated[0].shape == (2, 4, 3)
    assert changed[0].shape == (2, 2, 3)


def test_qwen_vision_falls_back_to_sdpa_without_changing_text_backend() -> None:
    assert _resolve_qwen_attention_implementations(
        "flash_attention_2",
        flash_attention_available=False,
    ) == ("eager", "sdpa")
    assert _resolve_qwen_attention_implementations(
        "flash_attention_2",
        flash_attention_available=True,
    ) == ("flash_attention_2", "flash_attention_2")
    assert _resolve_qwen_attention_implementations(
        "eager",
        flash_attention_available=False,
    ) == ("eager", "eager")


def test_policy_rejects_training_entrypoints() -> None:
    with pytest.raises(RuntimeError, match="inference-only"):
        LingbotVlaV2Policy.forward(None)
    with pytest.raises(ValueError, match="only supports inference mode"):
        LingbotVlaV2Policy.__init__(None, SimpleNamespace(), eval=False)

    assert "get_optim_params" not in LingbotVlaV2Policy.__dict__
    assert "get_parallel_plan" not in LingbotVlaV2Policy.__dict__


def test_loader_does_not_expose_training_loss_helpers() -> None:
    assert not hasattr(lingbot_vla_v2_loader, "triton_sequence_wise_balance_loss")
    assert not hasattr(lingbot_vla_v2_loader, "triton_load_balancing_loss_func")


def test_custom_qwen_models_use_transformers_5_tied_weight_mappings() -> None:
    assert Qwen3VLForConditionalGeneration._tied_weights_keys == {
        "lm_head.weight": "model.language_model.embed_tokens.weight"
    }
    assert Qwen2ForCausalLM._tied_weights_keys == {"lm_head.weight": "model.embed_tokens.weight"}
