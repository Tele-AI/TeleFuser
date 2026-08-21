"""LTX-2.5 split-checkpoint metadata contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from telefuser.models.ltx25.checkpoint import (
    LTX25CheckpointMetadata as Metadata,
)
from telefuser.models.ltx25.checkpoint import (
    LTX25ModelPaths,
    parse_model_version,
    validate_gemma_source_checkpoint,
)
from telefuser.models.ltx25.gemma4 import _cast_checkpoint_tensor, gemma4_checkpoint_key_to_model_key


def test_model_version_parsing_normalizes_prerelease_separators() -> None:
    assert parse_model_version("2.5") == (2, 5)
    assert parse_model_version("2.5-rc1") == (2, 5)
    assert parse_model_version("2.5.rc1") == (2, 5)
    assert parse_model_version(None) == ()


def test_split_layout_rejects_missing_required_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="transformer"):
        LTX25ModelPaths.from_model_root(tmp_path)


def test_gemma_source_checkpoint_requires_matching_gemma_version() -> None:
    metadata = Metadata(
        path=Path("transformer.safetensors"),
        size_bytes=1,
        sha256=None,
        tensor_count=1,
        metadata={"gemma_source_checkpoint": {"gemma_version": 4}},
        config={},
        model_version=(2, 5),
    )
    validate_gemma_source_checkpoint(metadata, {"gemma_version": 4})
    with pytest.raises(ValueError, match="Gemma version mismatch"):
        validate_gemma_source_checkpoint(metadata, {"gemma_version": 3})


def test_gemma4_comfy_flat_checkpoint_keys_map_to_unified_model() -> None:
    assert gemma4_checkpoint_key_to_model_key("model.layers.0.self_attn.q_proj.weight") == (
        "model.model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert gemma4_checkpoint_key_to_model_key("model.embed_tokens.weight") == (
        "model.model.language_model.embed_tokens.weight"
    )
    assert gemma4_checkpoint_key_to_model_key("audio_projector.embedding_projection.weight") == (
        "model.model.embed_audio.embedding_projection.weight"
    )
    assert gemma4_checkpoint_key_to_model_key("tokenizer_json") is None


def test_gemma_checkpoint_cast_matches_upstream_builder_policy() -> None:
    vector = torch.ones(2, dtype=torch.float32)
    scalar = torch.ones((), dtype=torch.float32)
    integer = torch.ones(2, dtype=torch.int64)

    assert _cast_checkpoint_tensor(vector, torch.bfloat16).dtype is torch.bfloat16
    assert _cast_checkpoint_tensor(scalar, torch.bfloat16).dtype is torch.float32
    assert _cast_checkpoint_tensor(integer, torch.bfloat16).dtype is torch.int64
