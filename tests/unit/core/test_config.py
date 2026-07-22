"""Tests for core configuration classes."""

import pytest
import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    LoraConfig,
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    RayConfig,
    RayGPUConfig,
    WeightOffloadType,
)


class TestDataclassBasics:
    """Basic tests for dataclass functionality using parametrization."""

    @pytest.mark.parametrize(
        "config_class,expected_defaults",
        [
            (RayGPUConfig, {"num_gpus": 0, "memory_limit": 0.8}),
            (RayConfig, {"ray_address": None, "num_cpus": 8, "memory_gb": 32}),
            (LoraConfig, {"path": "", "strength": 1.0}),
        ],
    )
    def test_default_values(self, config_class, expected_defaults):
        """Test that dataclasses have correct default values."""
        config = config_class()
        for key, value in expected_defaults.items():
            assert getattr(config, key) == value

    @pytest.mark.parametrize(
        "config_class,field_name,custom_value",
        [
            (RayGPUConfig, "num_gpus", 4),
            (RayGPUConfig, "memory_limit", 0.95),
            (LoraConfig, "path", "/path/to/lora.safetensors"),
            (LoraConfig, "strength", 0.5),
        ],
    )
    def test_custom_values(self, config_class, field_name, custom_value):
        """Test that dataclasses accept custom values."""
        kwargs = {field_name: custom_value}
        config = config_class(**kwargs)
        assert getattr(config, field_name) == custom_value


class TestRayConfig:
    """Test RayConfig dataclass with nested config."""

    def test_nested_gpu_config(self):
        """Test that RayConfig properly handles nested RayGPUConfig."""
        gpu_config = RayGPUConfig(num_gpus=4)
        config = RayConfig(ray_address="ray://localhost:10001", gpu_config=gpu_config, num_cpus=16, memory_gb=64)
        assert config.gpu_config.num_gpus == 4


class TestParallelConfig:
    """Test ParallelConfig dataclass with complex validation logic."""

    def test_default_world_size(self):
        """Default world_size should be 1."""
        config = ParallelConfig()
        assert config.world_size == 1
        assert config.worker_intra_op_threads == 1

    def test_worker_intra_op_threads_must_be_positive(self):
        """Distributed worker CPU thread count must be positive."""
        config = ParallelConfig(worker_intra_op_threads=0)
        with pytest.raises(ValueError, match="worker_intra_op_threads must be positive"):
            config.validate()

    @pytest.mark.parametrize(
        "device_ids,degrees,expected_world_size",
        [
            ([0, 1], {"dp_degree": 2}, 2),
            ([0, 1, 2, 3], {"dp_degree": 2, "tp_degree": 2}, 4),
            (list(range(8)), {"dp_degree": 2, "cfg_degree": 2, "sp_ulysses_degree": 2}, 8),
        ],
    )
    def test_world_size_calculation(self, device_ids, degrees, expected_world_size):
        """Test world_size calculation with various parallelism configs."""
        config = ParallelConfig(device_ids=device_ids, **degrees)
        assert config.world_size == expected_world_size

    @pytest.mark.parametrize(
        "device_ids,degrees",
        [
            ([0, 1], {"dp_degree": 4}),  # 2 GPUs but DP needs 4
            ([0, 1, 2, 3], {"dp_degree": 2, "tp_degree": 4}),  # 4 GPUs but need 8
        ],
    )
    def test_validation_failure(self, device_ids, degrees):
        """Test that validation fails when device count doesn't match degrees."""
        config = ParallelConfig(device_ids=device_ids, **degrees)
        with pytest.raises(RuntimeError, match="device num .* and world size .* not match"):
            config.validate()

    def test_validation_success(self):
        """Test validation passes with matching config."""
        config = ParallelConfig(device_ids=[0, 1, 2, 3], dp_degree=2, tp_degree=2)
        config.validate()  # Should not raise


class TestOffloadConfig:
    """Test OffloadConfig dataclass."""

    def test_default_offload_type(self):
        """Default offload type should be NO_CPU_OFFLOAD."""
        config = OffloadConfig()
        assert config.offload_type == WeightOffloadType.NO_CPU_OFFLOAD

    def test_custom_offload_config(self):
        """Test custom offload configuration."""
        config = OffloadConfig(
            offload_type=WeightOffloadType.ASYNC_CPU_OFFLOAD,
            pin_cpu_memory=False,
            offload_ratio=0.5,
        )
        assert config.offload_type == WeightOffloadType.ASYNC_CPU_OFFLOAD
        assert config.pin_cpu_memory is False
        assert config.offload_ratio == 0.5


class TestModelRuntimeConfig:
    """Test ModelRuntimeConfig dataclass."""

    def test_default_configuration(self):
        """Test default runtime configuration."""
        config = ModelRuntimeConfig()
        assert config.torch_dtype == torch.bfloat16
        assert config.attention_config.attn_impl == AttnImplType.TORCH_SDPA
        assert config.compile_config.enabled is False
        assert isinstance(config.offload_config, OffloadConfig)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_custom_dtype(self, dtype):
        """Test various dtype configurations."""
        config = ModelRuntimeConfig(torch_dtype=dtype)
        assert config.torch_dtype == dtype

    @pytest.mark.parametrize(
        "attn_impl",
        [
            AttnImplType.FLASH_ATTN_2,
            AttnImplType.FLASH_ATTN_3,
            AttnImplType.TORCH_CUDNN,
        ],
    )
    def test_custom_attention_impl(self, attn_impl):
        """Test various attention implementations via attention_config."""
        attention_config = AttentionConfig.dense_attention(attn_impl)
        config = ModelRuntimeConfig(attention_config=attention_config)
        assert config.attention_config.attn_impl == attn_impl

    def test_with_lora_configs(self):
        """Test configuration with LoRA adapters."""
        loras = [
            LoraConfig(path="/path/lora1", strength=0.5),
            LoraConfig(path="/path/lora2", strength=0.7),
        ]
        config = ModelRuntimeConfig(lora_configs=loras)
        assert len(config.lora_configs) == 2
        assert config.lora_configs[0].strength == 0.5
