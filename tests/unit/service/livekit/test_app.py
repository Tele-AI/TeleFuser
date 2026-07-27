from __future__ import annotations

from telefuser.service.livekit.app import create_livekit_app
from telefuser.service.livekit.config import LiveKitServeConfig
from telefuser.service.livekit.runtime import LiveKitServeRuntime
from tests.unit.openai._asgi_test_client import ASGITestClient


class FakeTokenService:
    def create_token(self, *, identity: str, room_name: str, role: str, **kwargs: object) -> str:
        return f"{role}:{identity}:{room_name}"


class FakeWorkerPool:
    async def start(self, *, skip_validation: bool = False) -> None:
        return None

    def start_session(self, record) -> None:
        return None

    async def stop_session(self, session_id: str) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _make_runtime(*, num_workers: int = 1, queue_size: int = 0) -> LiveKitServeRuntime:
    config = LiveKitServeConfig(
        livekit_url="wss://livekit.example",
        livekit_api_key="key",
        livekit_api_secret="secret",
        num_workers=num_workers,
        queue_size=queue_size,
    )
    return LiveKitServeRuntime(
        config=config, pipeline_file="pipeline.py", token_service=FakeTokenService(), worker_pool=FakeWorkerPool()
    )


def test_livekit_session_lifecycle_routes() -> None:
    runtime = _make_runtime()
    app = create_livekit_app(runtime)

    with ASGITestClient(app) as client:
        create_resp = client.post(
            "/v1/stream/sessions",
            json={"identity": "controller-1", "config": {"fps": 16}},
        )
        assert create_resp.status_code == 200
        created = create_resp.json()
        assert created["status"] == "assigned"
        assert created["worker_id"] == "worker-0"
        assert created["token"].startswith("controller:controller-1:")

        session_id = created["session_id"]
        viewer_resp = client.post(
            f"/v1/stream/sessions/{session_id}/tokens",
            json={"identity": "viewer-1"},
        )
        assert viewer_resp.status_code == 200
        assert viewer_resp.json()["token"].startswith("viewer:viewer-1:")

        status_resp = client.get(f"/v1/stream/sessions/{session_id}")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "assigned"

        delete_resp = client.delete(f"/v1/stream/sessions/{session_id}")
        assert delete_resp.status_code == 200
        assert delete_resp.json() == {"session_id": session_id, "status": "closed"}


def test_livekit_session_queue_returns_accepted() -> None:
    runtime = _make_runtime(queue_size=1)
    app = create_livekit_app(runtime)

    with ASGITestClient(app) as client:
        first_resp = client.post("/v1/stream/sessions", json={"identity": "controller-1"})
        second_resp = client.post("/v1/stream/sessions", json={"identity": "controller-2"})

    assert first_resp.status_code == 200
    assert second_resp.status_code == 202
    assert second_resp.json()["status"] == "queued"
    assert second_resp.json()["queue_position"] == 1


def test_livekit_session_rejects_when_busy() -> None:
    runtime = _make_runtime(queue_size=0)
    app = create_livekit_app(runtime)

    with ASGITestClient(app) as client:
        first_resp = client.post("/v1/stream/sessions", json={"identity": "controller-1"})
        second_resp = client.post("/v1/stream/sessions", json={"identity": "controller-2"})

    assert first_resp.status_code == 200
    assert second_resp.status_code == 429


def test_livekit_health_and_service_metadata_routes() -> None:
    runtime = _make_runtime()
    app = create_livekit_app(runtime)

    with ASGITestClient(app) as client:
        health = client.get("/v1/stream/health")
        metadata = client.get("/v1/service/metadata")

    assert health.status_code == 200
    assert health.json()["workers_total"] == 1
    assert metadata.status_code == 200
    assert metadata.json()["service_type"] == "stream"
    assert metadata.json()["transport"] == "livekit"
