"""Unit tests for config serialization utilities."""

import pytest
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, OffloadConfig, ParallelConfig
from telefuser.core.config_serializer import serialize_config, serialize_value


def test_serialize_value_none():
    """Test serialization of None value."""
    assert serialize_value(None) is None


def test_serialize_value_torch_dtype():
    """Test serialization of torch dtype."""
    assert serialize_value(torch.float16) == "float16"
    assert serialize_value(torch.bfloat16) == "bfloat16"
    assert serialize_value(torch.float32) == "float32"


def test_serialize_value_enum():
    """Test serialization of Enum values."""
    assert serialize_value(AttnImplType.TORCH_SDPA) == "TORCH_SDPA"
    assert serialize_value(AttnImplType.FLASH_ATTN_2) == "FLASH_ATTN_2"


def test_serialize_value_list():
    """Test serialization of list values."""
    assert serialize_value([1, 2, 3]) == [1, 2, 3]
    assert serialize_value([AttnImplType.TORCH_SDPA, AttnImplType.FLASH_ATTN_2]) == ["TORCH_SDPA", "FLASH_ATTN_2"]


def test_serialize_value_dict():
    """Test serialization of dict values."""
    result = serialize_value({"dtype": torch.bfloat16, "impl": AttnImplType.TORCH_SDPA})
    assert result == {"dtype": "bfloat16", "impl": "TORCH_SDPA"}


def test_serialize_config_parallel_config():
    """Test serialization of ParallelConfig."""
    config = ParallelConfig(device_ids=[0, 1], dp_degree=2, sp_ulysses_degree=1)
    result = serialize_config(config)

    assert result["device_ids"] == [0, 1]
    assert result["dp_degree"] == 2
    assert result["sp_ulysses_degree"] == 1
    assert result["enable_fsdp"] is False
    assert result["worker_intra_op_threads"] == 1


def test_serialize_config_offload_config():
    """Test serialization of OffloadConfig."""
    config = OffloadConfig(pin_cpu_memory=True, offload_ratio=0.8)
    result = serialize_config(config)

    assert result["pin_cpu_memory"] is True
    assert result["offload_ratio"] == 0.8
    assert result["offload_type"] == "NO_CPU_OFFLOAD"


def test_serialize_config_attention_config():
    """Test serialization of AttentionConfig."""
    config = AttentionConfig(attn_impl=AttnImplType.FLASH_ATTN_2, dropout=0.1)
    result = serialize_config(config)

    assert result["attn_impl"] == "FLASH_ATTN_2"
    assert result["dropout"] == 0.1
    assert result["is_causal"] is False


def test_serialize_config_nested():
    """Test serialization of nested config (ModelRuntimeConfig)."""
    config = ModelRuntimeConfig(
        torch_dtype=torch.bfloat16,
        device_id=0,
        attention_config=AttentionConfig(attn_impl=AttnImplType.TORCH_SDPA),
        offload_config=OffloadConfig(pin_cpu_memory=True),
        parallel_config=ParallelConfig(device_ids=[0]),
    )
    result = serialize_config(config)

    assert result["torch_dtype"] == "bfloat16"
    assert result["device_id"] == 0
    assert result["attention_config"]["attn_impl"] == "TORCH_SDPA"
    assert result["offload_config"]["pin_cpu_memory"] is True
    assert result["parallel_config"]["device_ids"] == [0]


def test_serialize_config_non_dataclass():
    """Test serialization of non-dataclass returns empty dict."""
    result = serialize_config("not a dataclass")
    assert result == {}


def test_serialize_value_unknown_type():
    """Test serialization of unknown type returns string representation."""

    class CustomClass:
        def __str__(self):
            return "CustomClass()"

    result = serialize_value(CustomClass())
    assert result == "CustomClass()"
