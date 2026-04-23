"""Integration tests for stream API endpoints."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _make_offer() -> dict:
    """Create a bidirectional SDP offer with a DataChannel."""
    pytest.importorskip("aiortc")
    from aiortc import RTCPeerConnection

    async def _create():
        pc = RTCPeerConnection()
        pc.createDataChannel("telefuser")
        pc.addTransceiver("video", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        sdp = pc.localDescription.sdp
        sdp_type = pc.localDescription.type
        await pc.close()
        return {"sdp": sdp, "type": sdp_type}

    return asyncio.run(_create())


def _make_server_push_offer() -> dict:
    """Create a server-push SDP offer (no DataChannel)."""
    pytest.importorskip("aiortc")
    from aiortc import RTCPeerConnection

    async def _create():
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        sdp = pc.localDescription.sdp
        sdp_type = pc.localDescription.type
        await pc.close()
        return {"sdp": sdp, "type": sdp_type}

    return asyncio.run(_create())


# ---------------------------------------------------------------------------
# Bidirectional (session management) tests — via WebRTC offer
# ---------------------------------------------------------------------------


class TestBidirectionalSessions:
    """Tests for bidirectional session management via WebRTC offer endpoint."""

    @pytest.fixture(autouse=True)
    def _skip_without_aiortc(self):
        pytest.importorskip("aiortc")

    def test_create_bidirectional_session_via_offer(self, bidirectional_client):
        offer = _make_offer()
        body = {**offer, "task": "s2v", "config": {"fps": 24}}
        resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["type"] == "answer"

    def test_close_bidirectional_session(self, bidirectional_client):
        offer = _make_offer()
        body = {**offer, "task": "s2v"}
        create_resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        session_id = create_resp.json()["session_id"]

        resp = bidirectional_client.delete(f"/v1/stream/webrtc/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"

    def test_close_via_stream_sessions_alias(self, bidirectional_client):
        """DELETE /v1/stream/sessions/{id} should close both pipeline and WebRTC sessions."""
        offer = _make_offer()
        body = {**offer, "task": "s2v"}
        create_resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        session_id = create_resp.json()["session_id"]

        resp = bidirectional_client.delete(f"/v1/stream/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"

        status_resp = bidirectional_client.get(f"/v1/stream/sessions/{session_id}/status")
        assert status_resp.json()["status"] == "unknown"


# ---------------------------------------------------------------------------
# Service status / metadata tests
# ---------------------------------------------------------------------------


class TestStreamServiceStatus:
    """Tests for service health / metadata with stream endpoints."""

    def test_service_status_with_stream(self, server_push_client):
        resp = server_push_client.get("/v1/service/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "service_status" in data

    def test_health_with_stream(self, server_push_client):
        resp = server_push_client.get("/v1/service/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_health_reports_stream_readiness(self, server_push_client):
        resp = server_push_client.get("/v1/service/health")
        data = resp.json()
        assert data["stream_ready"] is True
        assert data["stream_mode"] == "server_push"

    def test_metadata_returns_stream_info_without_inference_service(self, server_push_client):
        resp = server_push_client.get("/v1/service/metadata")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service_type"] == "stream"
        assert data["stream_mode"] == "server_push"
        assert data["runner"] == "StreamPipelineService"

    def test_metrics_json_includes_webrtc_stats(self, server_push_client):
        resp = server_push_client.get("/v1/service/metrics/json")
        assert resp.status_code == 200
        data = resp.json()
        assert "webrtc" in data

    def test_close_server_push_via_stream_sessions_alias(self, server_push_client):
        """DELETE /v1/stream/sessions/{id} should close server-push WebRTC sessions."""
        pytest.importorskip("aiortc")
        offer = _make_server_push_offer()
        body = {**offer, "task": "t2v", "prompt": "test"}
        create_resp = server_push_client.post("/v1/stream/webrtc/offer", json=body)
        assert create_resp.status_code == 200
        session_id = create_resp.json()["session_id"]

        resp = server_push_client.delete(f"/v1/stream/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"

        status_resp = server_push_client.get(f"/v1/stream/sessions/{session_id}/status")
        assert status_resp.json()["status"] == "unknown"


# ---------------------------------------------------------------------------
# Session status tests
# ---------------------------------------------------------------------------


class TestSessionStatusLookup:
    """Session status should query both stream service and WebRTC session manager."""

    @pytest.fixture(autouse=True)
    def _skip_without_aiortc(self):
        pytest.importorskip("aiortc")

    def test_active_session_returns_active(self, bidirectional_client):
        offer = _make_offer()
        body = {**offer, "task": "s2v"}
        create_resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        session_id = create_resp.json()["session_id"]

        resp = bidirectional_client.get(f"/v1/stream/sessions/{session_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "active"

    def test_unknown_session_returns_unknown(self, bidirectional_client):
        resp = bidirectional_client.get("/v1/stream/sessions/nonexistent/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown"

    def test_closed_session_returns_unknown(self, bidirectional_client):
        offer = _make_offer()
        body = {**offer, "task": "s2v"}
        create_resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        session_id = create_resp.json()["session_id"]
        bidirectional_client.delete(f"/v1/stream/webrtc/{session_id}")

        resp = bidirectional_client.get(f"/v1/stream/sessions/{session_id}/status")
        assert resp.json()["status"] == "unknown"
