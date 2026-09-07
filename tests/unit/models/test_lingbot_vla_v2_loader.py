from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from telefuser.core.config import QuantConfig, QuantType
from telefuser.models.lingbot_vla_v2_loader import (
    build_official_6b_config,
    load_lingbot_vla_v2,
    resolve_lingbot_vla_v2_shards,
    validate_official_6b_checkpoint,
)


def test_resolve_lingbot_vla_v2_shards_uses_index_manifest(tmp_path) -> None:
    shard_names = ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"]
    for name in shard_names:
        (tmp_path / name).write_bytes(b"")
    index = {
        "weight_map": {
            "layer.0": shard_names[0],
            "layer.1": shard_names[1],
            "layer.2": shard_names[0],
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    resolved = resolve_lingbot_vla_v2_shards(tmp_path)

    assert resolved == [str(tmp_path / name) for name in sorted(shard_names)]


def test_resolve_lingbot_vla_v2_shards_rejects_missing_files(tmp_path) -> None:
    index = {"weight_map": {"layer.0": "missing.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint shards"):
        resolve_lingbot_vla_v2_shards(tmp_path)


def test_validate_official_6b_checkpoint_accepts_expected_gate_shapes() -> None:
    prefix = "model.qwenvl_with_expert.qwen_expert.model.layers"
    state_dict = {
        f"{prefix}.0.mlp.experts.gate_proj": SimpleNamespace(shape=(32, 512, 768)),
        f"{prefix}.35.mlp.experts.gate_proj": SimpleNamespace(shape=(32, 512, 768)),
    }

    validate_official_6b_checkpoint(state_dict)


def test_validate_official_6b_checkpoint_rejects_wrong_shape() -> None:
    prefix = "model.qwenvl_with_expert.qwen_expert.model.layers"
    state_dict = {
        f"{prefix}.0.mlp.experts.gate_proj": SimpleNamespace(shape=(1, 2, 3)),
        f"{prefix}.35.mlp.experts.gate_proj": SimpleNamespace(shape=(32, 512, 768)),
    }

    with pytest.raises(ValueError, match="Unexpected shape"):
        validate_official_6b_checkpoint(state_dict)


def test_build_official_6b_config_rejects_non_base_variant(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported LingBot-VLA v2 checkpoint variant"):
        build_official_6b_config(tmp_path, checkpoint_variant="robotwin")


def test_public_loader_routes_official_shards_through_module_manager(tmp_path, monkeypatch) -> None:
    shard_names = ["model-00002-of-00002.safetensors", "model-00001-of-00002.safetensors"]
    for name in shard_names:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": shard_names[0], "layer.1": shard_names[1]}}),
        encoding="utf-8",
    )

    fake_model_class = type("FakeLingBotVlaV2Model", (), {})
    monkeypatch.setitem(
        sys.modules,
        "telefuser.models.lingbot_vla_v2",
        SimpleNamespace(LingBotVlaV2Model=fake_model_class),
    )

    class _RecordingManager:
        def __init__(self) -> None:
            self.load_kwargs = None

        def load_model(self, file_path, **kwargs) -> None:
            self.load_kwargs = {"file_path": file_path, **kwargs}

        def fetch_module(self, name: str):
            return SimpleNamespace(name=name)

    manager = _RecordingManager()

    loaded = load_lingbot_vla_v2(
        manager,
        tmp_path,
        tmp_path / "qwen3vl",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )

    assert loaded.name == "lingbot_vla_v2"
    assert manager.load_kwargs == {
        "file_path": [str(tmp_path / name) for name in sorted(shard_names)],
        "device": "cpu",
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "name": "lingbot_vla_v2",
        "model_class": fake_model_class,
        "model_resource": "official",
        "converter_kwargs": {
            "qwen3vl_path": str(tmp_path / "qwen3vl"),
            "checkpoint_variant": "base",
            "checkpoint_path": str(tmp_path),
        },
        "quant_config": None,
    }

    quant_config = QuantConfig(enabled=True, quant_type=QuantType.TORCHAO_FP8)
    load_lingbot_vla_v2(
        manager,
        tmp_path,
        tmp_path / "qwen3vl",
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        quant_config=quant_config,
    )
    assert manager.load_kwargs["device"] == "cuda:0"
    assert manager.load_kwargs["quant_config"] is quant_config
