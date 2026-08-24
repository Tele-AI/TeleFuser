from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from telefuser.core.config import SparseAttentionConfig
from telefuser.pipelines.wan_video.single_dit_denoising import SingleDitDenoisingStage


def _stage_with_mock_dit() -> SingleDitDenoisingStage:
    stage = SingleDitDenoisingStage.__new__(SingleDitDenoisingStage)
    stage.dit = MagicMock()
    return stage


def test_stage_enables_sol_attention_on_local_dit() -> None:
    stage = _stage_with_mock_dit()
    sparse_config = SparseAttentionConfig(sparse_impl="sol", sol_fp8=True)

    stage.enable_sparse_attention(height=480, width=832, num_frames=81, sparse_config=sparse_config)

    stage.dit.enable_sol_attention.assert_called_once_with(
        height=480,
        width=832,
        num_frames=81,
        sparse_config=sparse_config,
    )


def test_stage_translates_radial_attention_config() -> None:
    stage = _stage_with_mock_dit()
    sparse_config = SparseAttentionConfig(
        sparse_impl="radial",
        dense_layers=3,
        dense_timesteps=7,
        decay_factor=0.8,
        use_sage_attention=True,
    )

    stage.enable_sparse_attention(height=64, width=96, num_frames=17, sparse_config=sparse_config)

    stage.dit.enable_radial_attention.assert_called_once_with(
        height=64,
        width=96,
        num_frames=17,
        dense_layers=3,
        dense_timesteps=7,
        decay_factor=0.8,
        use_sage_attention=True,
    )


def test_stage_rejects_unsupported_sparse_attention() -> None:
    stage = _stage_with_mock_dit()
    sparse_config = SimpleNamespace(sparse_impl="local")

    with pytest.raises(ValueError, match="Unsupported Wan sparse attention: local"):
        stage.enable_sparse_attention(height=64, width=96, num_frames=17, sparse_config=sparse_config)
