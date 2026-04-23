"""Stream routes: session status and lifecycle endpoints.

Bidirectional streaming uses WebRTC (DataChannel + media tracks).

Endpoints:
    DELETE /v1/stream/sessions/{session_id}      – close session
    GET    /v1/stream/sessions/{session_id}/status
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

if TYPE_CHECKING:
    from ..api_server import ApiServer


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


class StreamRoutes:
    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

    def _require_service(self):
        svc = self.api.stream_service
        if svc is None or not svc.is_running:
            raise HTTPException(status_code=503, detail="Stream service is not running")
        return svc

    async def close_session(self, session_id: str) -> dict:
        svc = self._require_service()
        pipeline_closed = False
        try:
            svc.close_session(session_id)
            pipeline_closed = True
        except Exception:
            pass
        webrtc_closed = False
        webrtc_routes = self.api._webrtc_routes
        if webrtc_routes is not None:
            webrtc_closed = await webrtc_routes._session_manager.close_session(session_id)
        if not pipeline_closed and not webrtc_closed:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return {"session_id": session_id, "status": "closed"}

    async def session_status(self, session_id: str) -> dict:
        task = self.api.task_manager.get_task_status(session_id)
        if task:
            return task

        svc = self.api.stream_service
        if svc is not None and svc.is_running and svc.has_session(session_id):
            return {"session_id": session_id, "status": "active", "stream_mode": svc.stream_mode}

        webrtc_routes = self.api._webrtc_routes
        if webrtc_routes is not None and webrtc_routes._session_manager.has_session(session_id):
            return {"session_id": session_id, "status": "active", "stream_mode": svc.stream_mode if svc else "unknown"}

        return {"session_id": session_id, "status": "unknown"}


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_router(api_server: ApiServer) -> APIRouter:
    router = APIRouter(prefix="/v1/stream", tags=["stream"])
    routes = StreamRoutes(api_server)

    @router.delete("/sessions/{session_id}", summary="Close session")
    async def close_session(session_id: str):
        return await routes.close_session(session_id)

    @router.get("/sessions/{session_id}/status", summary="Session status")
    async def session_status(session_id: str):
        return await routes.session_status(session_id)

    return router


def setup_routes(api_server: ApiServer) -> APIRouter:
    return create_router(api_server)
