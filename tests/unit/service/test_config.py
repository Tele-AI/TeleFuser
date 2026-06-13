from __future__ import annotations

from telefuser.service.core.config import ServerConfig


def test_effective_max_concurrent_tasks_default_is_one() -> None:
    config = ServerConfig(max_concurrent_tasks=8)

    assert config.max_concurrent_tasks == 8
    assert config.effective_max_concurrent_tasks == 1  # num_replicas defaults to 1


def test_effective_max_concurrent_tasks_equals_num_replicas() -> None:
    config = ServerConfig(num_replicas=4)

    assert config.effective_max_concurrent_tasks == 4


def test_server_config_ignores_unknown_fields_for_forward_compatibility() -> None:
    config = ServerConfig(max_queue_size=12, future_option="ignored")

    assert config.max_queue_size == 12
    assert not hasattr(config, "future_option")
