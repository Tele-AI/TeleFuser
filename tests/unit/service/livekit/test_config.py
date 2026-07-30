from __future__ import annotations

import pytest

from telefuser.service.livekit.config import LiveKitServeConfig


def test_worker_gpu_groups_default_to_empty_groups() -> None:
    config = LiveKitServeConfig(num_workers=2)

    assert config.worker_gpu_groups() == [[], []]


def test_worker_gpu_groups_parse_semicolon_map() -> None:
    config = LiveKitServeConfig(num_workers=2, worker_gpu_map="0,1;2,3")

    assert config.worker_gpu_groups() == [["0", "1"], ["2", "3"]]


def test_worker_gpu_groups_reject_wrong_group_count() -> None:
    config = LiveKitServeConfig(num_workers=2, worker_gpu_map="0,1")

    with pytest.raises(ValueError, match="worker groups"):
        config.worker_gpu_groups()


def test_require_livekit_credentials_reports_missing_fields() -> None:
    config = LiveKitServeConfig(livekit_url="wss://example.livekit.cloud")

    with pytest.raises(ValueError, match="livekit_api_key, livekit_api_secret"):
        config.require_livekit_credentials()


def test_multi_session_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TELEFUSER_LIVEKIT_MAX_SESSIONS_PER_WORKER", "3")
    monkeypatch.setenv("TELEFUSER_LIVEKIT_CONTROL_IDLE_TIMEOUT", "7.5")

    config = LiveKitServeConfig()

    assert config.max_sessions_per_worker == 3
    assert config.control_idle_timeout == 7.5
