from __future__ import annotations

from telefuser.service.core.config import ServerConfig
from telefuser.service.core.container import ServiceContainer


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


def test_artifact_retention_defaults_are_configured() -> None:
    config = ServerConfig()

    assert config.artifact_storage_backend == "local"
    assert config.artifact_local_root is None
    assert config.effective_artifact_local_root == config.cache_dir
    assert config.artifact_retention_seconds == 7 * 24 * 60 * 60
    assert config.artifact_tmp_retention_seconds == 60 * 60
    assert config.artifact_cleanup_interval_seconds == 60 * 60
    assert config.artifact_max_total_bytes == 0
    assert config.artifact_max_task_bytes == 0
    assert config.artifact_preserve_failed_outputs is False


def test_artifact_local_root_can_override_cache_dir() -> None:
    config = ServerConfig(cache_dir="cache-a", artifact_local_root="artifact-a")

    assert config.effective_artifact_local_root == "artifact-a"


def test_container_uses_artifact_local_root_when_cache_dir_is_not_explicit(tmp_path) -> None:
    config = ServerConfig(cache_dir=str(tmp_path / "cache"), artifact_local_root=str(tmp_path / "artifacts"))
    container = ServiceContainer.create(config=config)

    file_service = container.initialize_file_service()

    assert file_service.cache_dir == (tmp_path / "artifacts").resolve()


def test_container_explicit_cache_dir_takes_precedence_over_artifact_local_root(tmp_path) -> None:
    config = ServerConfig(cache_dir=str(tmp_path / "cache"), artifact_local_root=str(tmp_path / "artifacts"))
    container = ServiceContainer.create(config=config, cache_dir=tmp_path / "explicit")

    file_service = container.initialize_file_service()

    assert file_service.cache_dir == (tmp_path / "explicit").resolve()


def test_container_rejects_unimplemented_artifact_backend(tmp_path) -> None:
    config = ServerConfig(artifact_storage_backend="s3", artifact_local_root=str(tmp_path / "artifacts"))
    container = ServiceContainer.create(config=config)

    try:
        container.initialize_file_service()
    except RuntimeError as exc:
        assert "Only the local artifact backend is implemented" in str(exc)
    else:
        raise AssertionError("Expected S3 artifact backend to be rejected")
