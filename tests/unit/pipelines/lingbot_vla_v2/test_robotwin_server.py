from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from click.testing import CliRunner
from fastapi.testclient import TestClient

from examples.lingbot_vla_v2 import lingbot_vla_v2_robotwin_server as server
from telefuser.pipelines.lingbot_vla_v2 import (
    ROBOTWIN_CAMERA_KEYS,
    LingBotVlaV2CanonicalActionChunk,
    RobotWinProfile,
)


def _profile() -> RobotWinProfile:
    return RobotWinProfile(
        {
            "observation.state.arm.position": {"q01": [0.0] * 12, "q99": [2.0] * 12},
            "observation.state.effector.position": {"q01": [-1.0] * 2, "q99": [1.0] * 2},
            "action.arm.position": {"q01": [0.0] * 12, "q99": [2.0] * 12},
            "action.effector.position": {"q01": [-1.0] * 2, "q99": [1.0] * 2},
        }
    )


class _Pipeline:
    def __init__(self, profile: RobotWinProfile) -> None:
        self.config = SimpleNamespace(robot_profile=profile)
        self.observations: list[Any] = []
        self.seeds: list[int | None] = []
        self.closed = False

    def __call__(self, observation, seed: int | None = None) -> LingBotVlaV2CanonicalActionChunk:
        self.observations.append(observation)
        self.seeds.append(seed)
        return LingBotVlaV2CanonicalActionChunk(
            canonical_normalized_actions=torch.zeros(50, 55),
            horizon=50,
            action_dim=55,
        )

    def close(self) -> None:
        self.closed = True


def _observation() -> dict[str, Any]:
    images = {key: np.zeros((8, 8, 3), dtype=np.uint8) for key in ROBOTWIN_CAMERA_KEYS}
    return {
        **images,
        "observation.state": np.zeros(14, dtype=np.float32),
        "task": "pick up the block",
    }


def test_message_codec_matches_upstream_numpy_contract() -> None:
    payload = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
        "state": np.float32(1.5),
    }

    decoded = server.unpack_message(server.pack_message(payload))

    assert np.array_equal(decoded["image"], payload["image"])
    assert decoded["image"].dtype == np.uint8
    assert decoded["state"] == np.float32(1.5)


def test_message_codec_rejects_unsafe_object_arrays() -> None:
    with pytest.raises(ValueError, match="unsupported NumPy dtype"):
        server.pack_message({"value": np.asarray([object()], dtype=object)})


def test_adapter_maps_observation_and_returns_robotwin_chunk() -> None:
    profile = _profile()
    pipeline = _Pipeline(profile)
    adapter = server.RobotWinPolicyAdapter(pipeline, use_length=8)

    request = {**_observation(), "seed": 7, "request_id": "request-1", "episode_id": "episode-1"}
    result = adapter.infer(request)

    assert result["action"].shape == (8, 14)
    assert result["action"].dtype == np.float32
    assert np.allclose(result["action"][:, [6, 13]], 0.0, atol=1e-6)
    assert np.allclose(np.delete(result["action"], [6, 13], axis=1), 1.0000005)
    assert result["policy_verified"] is False
    assert result["seed"] == 7
    assert result["request_id"] == "request-1"
    assert result["episode_id"] == "episode-1"
    assert result["server_timing"]["lock_wait_ms"] >= 0
    assert result["server_timing"]["pipeline_ms"] >= 0
    assert result["server_timing"]["action_mapping_ms"] >= 0
    assert result["server_timing"]["adapter_total_ms"] >= 0
    assert len(pipeline.observations) == 1
    assert pipeline.seeds == [7]
    observation = pipeline.observations[0]
    assert observation.task == "pick up the block"
    assert set(observation.images) == set(ROBOTWIN_CAMERA_KEYS)


def test_adapter_reset_does_not_run_or_reload_policy() -> None:
    pipeline = _Pipeline(_profile())
    adapter = server.RobotWinPolicyAdapter(pipeline)

    assert adapter.infer({"reset": True, "robo_name": "robotwin"}) == {"action": None}
    assert pipeline.observations == []
    with pytest.raises(ValueError, match="runtime checkpoint switching"):
        adapter.infer({"reset": True, "path_to_pi_model": "/different/checkpoint"})


@pytest.mark.parametrize("field", ["request_id", "episode_id"])
def test_adapter_rejects_invalid_trace_fields(field: str) -> None:
    adapter = server.RobotWinPolicyAdapter(_Pipeline(_profile()))

    with pytest.raises(ValueError, match=field):
        adapter.infer({**_observation(), field: ""})


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_adapter_rejects_non_integer_seed(seed: Any) -> None:
    adapter = server.RobotWinPolicyAdapter(_Pipeline(_profile()))

    with pytest.raises(ValueError, match="seed must be an integer"):
        adapter.infer({**_observation(), "seed": seed})


def test_adapter_rejects_missing_observation_fields() -> None:
    adapter = server.RobotWinPolicyAdapter(_Pipeline(_profile()))

    with pytest.raises(ValueError, match="missing fields"):
        adapter.infer({"task": "pick up the block"})


def test_websocket_is_persistent_and_uses_upstream_response_fields() -> None:
    pipeline = _Pipeline(_profile())
    adapter = server.RobotWinPolicyAdapter(pipeline, use_length=3)

    with TestClient(server.create_robotwin_app(adapter)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        with client.websocket_connect("/") as websocket:
            metadata = server.unpack_message(websocket.receive_bytes())
            assert metadata["robot_profile"] == "robotwin"
            assert metadata["protocol_version"] == "1.0"
            assert metadata["action_type"] == "absolute_qpos"
            assert metadata["action_horizon"] == 3
            assert metadata["action_dim"] == 14
            assert metadata["action_dtype"] == "float32"
            assert metadata["action_order"] == list(server.ROBOTWIN_ACTION_ORDER)
            assert metadata["max_request_bytes"] == server.ROBOTWIN_MAX_REQUEST_BYTES

            websocket.send_bytes(
                server.pack_message(
                    {**_observation(), "seed": 11, "request_id": "request-1", "episode_id": "episode-1"}
                )
            )
            response = server.unpack_message(websocket.receive_bytes())
            assert response["action"].shape == (3, 14)
            assert response["seed"] == 11
            assert response["request_id"] == "request-1"
            assert response["episode_id"] == "episode-1"
            assert response["server_timing"]["decode_ms"] >= 0
            assert response["server_timing"]["infer_ms"] >= 0
            assert response["server_timing"]["pipeline_ms"] >= 0
            assert response["server_timing"]["action_mapping_ms"] >= 0

            websocket.send_bytes(
                server.pack_message(
                    {
                        "reset": True,
                        "robo_name": "robotwin",
                        "request_id": "reset-1",
                        "episode_id": "episode-1",
                    }
                )
            )
            reset_response = server.unpack_message(websocket.receive_bytes())
            assert reset_response["action"] is None
            assert reset_response["request_id"] == "reset-1"
            assert reset_response["episode_id"] == "episode-1"
            assert reset_response["server_timing"]["prev_total_ms"] >= 0


def test_cli_exposes_isolated_robotwin_server_options() -> None:
    result = CliRunner().invoke(server.main, ["--help"])

    assert result.exit_code == 0
    assert "--model-root" in result.output
    assert "--qwen3vl-root" in result.output
    assert "--use-length" in result.output
    assert "--cuda-graph" in result.output


def test_h100_policy_process_disables_only_cudnn_sdpa(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA H100 80GB HBM3")
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", lambda enabled: calls.append(("cudnn", enabled)))
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", lambda enabled: calls.append(("flash", enabled)))
    monkeypatch.setattr(torch.backends.cuda, "enable_math_sdp", lambda enabled: calls.append(("math", enabled)))
    monkeypatch.setattr(
        torch.backends.cuda,
        "enable_mem_efficient_sdp",
        lambda enabled: calls.append(("mem_efficient", enabled)),
    )

    server._configure_h100_sdpa_backends("cuda:0")

    assert calls == [("cudnn", False), ("flash", True), ("math", True), ("mem_efficient", True)]


def test_non_h100_policy_process_preserves_sdpa_backends(monkeypatch) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA RTX 4090")
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", calls.append)

    server._configure_h100_sdpa_backends("cuda:0")

    assert calls == []
