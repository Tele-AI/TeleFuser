"""Integration tests for WebRTC signaling endpoints."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
aiortc = pytest.importorskip("aiortc")

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sdp_offer(include_audio: bool = True) -> dict:
    """Create a minimal SDP offer using aiortc."""
    from aiortc import RTCPeerConnection

    async def _create():
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        if include_audio:
            pc.addTransceiver("audio", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        sdp = pc.localDescription.sdp
        sdp_type = pc.localDescription.type
        await pc.close()
        return {"sdp": sdp, "type": sdp_type}

    return asyncio.run(_create())


def _make_bidirectional_sdp_offer(include_audio: bool = True) -> dict:
    """Create an SDP offer with a DataChannel (for bidirectional mode)."""
    from aiortc import RTCPeerConnection

    async def _create():
        pc = RTCPeerConnection()
        pc.createDataChannel("telefuser")
        pc.addTransceiver("video", direction="recvonly")
        if include_audio:
            pc.addTransceiver("audio", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        sdp = pc.localDescription.sdp
        sdp_type = pc.localDescription.type
        await pc.close()
        return {"sdp": sdp, "type": sdp_type}

    return asyncio.run(_create())


# ---------------------------------------------------------------------------
# Server-push tests (unchanged)
# ---------------------------------------------------------------------------


class TestWebRTCOffer:
    """Tests for POST /v1/stream/webrtc/offer (server-push mode)."""

    def test_offer_returns_sdp_answer(self, server_push_client):
        offer = _make_sdp_offer()
        body = {**offer, "task": "t2v", "prompt": "a sunset"}
        resp = server_push_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "sdp" in data
        assert data["type"] == "answer"
        assert "session_id" in data

    def test_offer_rejects_when_service_not_running(self):
        from telefuser.service.api.api_server import ApiServer
        from telefuser.service.core.task_manager import TaskManager

        server = ApiServer(max_queue_size=10, task_manager=TaskManager(), enable_openai_api=False)
        with TestClient(server.app) as client:
            offer = _make_sdp_offer()
            body = {**offer, "task": "t2v", "prompt": "test"}
            resp = client.post("/v1/stream/webrtc/offer", json=body)
            assert resp.status_code == 503


class TestWebRTCSession:
    """Tests for DELETE /v1/stream/webrtc/{session_id}."""

    def test_close_existing_session(self, server_push_client):
        offer = _make_sdp_offer()
        body = {**offer, "task": "t2v", "prompt": "test"}
        create_resp = server_push_client.post("/v1/stream/webrtc/offer", json=body)
        session_id = create_resp.json()["session_id"]

        resp = server_push_client.delete(f"/v1/stream/webrtc/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"

    def test_close_nonexistent_session(self, server_push_client):
        resp = server_push_client.delete("/v1/stream/webrtc/nonexistent")
        assert resp.status_code == 404


class TestWebRTCAudio:
    """Tests for WebRTC audio track support."""

    def test_offer_with_audio_and_audio_chunks(self, audio_server_push_client):
        offer = _make_sdp_offer(include_audio=True)
        body = {**offer, "task": "t2v", "prompt": "audio test"}
        resp = audio_server_push_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "sdp" in data
        assert "m=audio" in data["sdp"]

    def test_offer_with_audio_but_no_audio_chunks(self, server_push_client):
        """Backwards compat: client offers audio but pipeline has no audio data."""
        offer = _make_sdp_offer(include_audio=True)
        body = {**offer, "task": "t2v", "prompt": "no audio"}
        resp = server_push_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200

    def test_offer_without_audio_transceiver(self, audio_server_push_client):
        """Video-only client: no audio transceiver in SDP offer."""
        offer = _make_sdp_offer(include_audio=False)
        body = {**offer, "task": "t2v", "prompt": "video only"}
        resp = audio_server_push_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "m=audio" not in data["sdp"]


# ---------------------------------------------------------------------------
# Bidirectional WebRTC tests
# ---------------------------------------------------------------------------


class TestWebRTCBidirectional:
    """Tests for WebRTC bidirectional mode via POST /v1/stream/webrtc/offer."""

    def test_offer_creates_bidirectional_session(self, bidirectional_client):
        offer = _make_bidirectional_sdp_offer()
        body = {**offer, "task": "s2v", "prompt": "test", "config": {"fps": 24}}
        resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "sdp" in data
        assert data["type"] == "answer"
        assert "session_id" in data

    def test_offer_bidirectional_without_audio(self, bidirectional_client):
        offer = _make_bidirectional_sdp_offer(include_audio=False)
        body = {**offer, "task": "s2v", "prompt": "no audio"}
        resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "m=audio" not in data["sdp"]

    def test_close_bidirectional_session(self, bidirectional_client):
        offer = _make_bidirectional_sdp_offer()
        body = {**offer, "task": "s2v", "prompt": "test"}
        create_resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        session_id = create_resp.json()["session_id"]

        resp = bidirectional_client.delete(f"/v1/stream/webrtc/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "closed"

    def test_close_nonexistent_bidirectional(self, bidirectional_client):
        resp = bidirectional_client.delete("/v1/stream/webrtc/nonexistent")
        assert resp.status_code == 404

    def test_sdp_failure_rolls_back_pipeline_session(self, bidirectional_client):
        """Offer with invalid SDP should not leave orphaned pipeline sessions."""
        body = {"sdp": "invalid-sdp", "type": "offer", "task": "s2v", "prompt": "test"}
        resp = bidirectional_client.post("/v1/stream/webrtc/offer", json=body)
        assert resp.status_code == 400
