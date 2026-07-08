"""WebRTC signaling routes for SDP offer/answer exchange.

Supports both stream modes:

* **server_push** – output-only tracks, no DataChannel.
* **bidirectional** – client-created DataChannel + optional media tracks.

Endpoints:
    POST   /v1/stream/webrtc/offer            – SDP offer → answer
    DELETE /v1/stream/webrtc/{session_id}      – close session
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from telefuser.utils.logging import logger

from ...core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL, STREAM_MODE_SERVER_PUSH
from ..stream_schema import WebRTCOfferRequest, WebRTCOfferResponse

if TYPE_CHECKING:
    from ..api_server import ApiServer


class WebRTCRoutes:
    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

        from aiortc import RTCConfiguration, RTCIceServer

        from ...webrtc.session_manager import WebRTCSessionManager

        config = api_server.server_config
        ice_servers: list[RTCIceServer] = []
        for url in config.stun_servers:
            ice_servers.append(RTCIceServer(urls=url))
        if config.turn_server:
            ice_servers.append(
                RTCIceServer(
                    urls=config.turn_server,
                    username=config.turn_username or "",
                    credential=config.turn_credential or "",
                )
            )

        configuration = RTCConfiguration(iceServers=ice_servers) if ice_servers else None

        self._session_manager = WebRTCSessionManager(
            max_sessions=config.webrtc_max_sessions,
            configuration=configuration,
        )

    async def handle_offer(self, request: WebRTCOfferRequest) -> WebRTCOfferResponse:
        svc = self.api.stream_service
        if svc is None or not svc.is_running:
            raise HTTPException(status_code=503, detail="Stream service is not running")

        if svc.stream_mode == STREAM_MODE_SERVER_PUSH:
            return await self._handle_server_push(svc, request)
        elif svc.stream_mode == STREAM_MODE_BIDIRECTIONAL:
            return await self._handle_bidirectional(svc, request)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown stream mode: {svc.stream_mode}")

    async def _handle_server_push(self, svc, request: WebRTCOfferRequest) -> WebRTCOfferResponse:
        session_id = request.session_id
        task_data = request.model_dump(exclude={"sdp", "type"})
        task_data["task_id"] = session_id

        generator = svc.stream_task(task_data)

        try:
            answer_sdp, answer_type = await self._session_manager.create_session(
                session_id=session_id,
                offer_sdp=request.sdp,
                offer_type=request.type,
                generator=generator,
                fps=request.fps or 24,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"SDP negotiation failed: {exc}")

        return WebRTCOfferResponse(
            session_id=session_id,
            sdp=answer_sdp,
            type=answer_type,
        )

    async def _handle_bidirectional(self, svc, request: WebRTCOfferRequest) -> WebRTCOfferResponse:
        session_id = request.session_id
        config = {**request.model_dump(exclude={"sdp", "type"}), "session_id": session_id}

        pipeline_session_id = svc.create_session(config)

        try:
            output_gen = svc.pull_chunks(pipeline_session_id)
            answer_sdp, answer_type = await self._session_manager.create_bidirectional_session(
                session_id=pipeline_session_id,
                offer_sdp=request.sdp,
                offer_type=request.type,
                output_generator=output_gen,
                on_input=lambda sid, chunk: svc.push_chunk(sid, chunk),
                on_close=lambda sid: svc.close_session(sid),
                fps=request.fps or 24,
            )
        except RuntimeError as exc:
            try:
                svc.close_session(pipeline_session_id)
            except Exception:
                pass
            raise HTTPException(status_code=503, detail=str(exc))
        except Exception as exc:
            try:
                svc.close_session(pipeline_session_id)
            except Exception:
                pass
            raise HTTPException(status_code=400, detail=f"SDP negotiation failed: {exc}")

        return WebRTCOfferResponse(
            session_id=pipeline_session_id,
            sdp=answer_sdp,
            type=answer_type,
        )

    async def close_session(self, session_id: str) -> dict:
        closed = await self._session_manager.close_session(session_id, reason="webrtc_session_delete")
        if not closed:
            raise HTTPException(status_code=404, detail=f"WebRTC session {session_id} not found")
        return {"session_id": session_id, "status": "closed"}

    async def cleanup(self) -> None:
        await self._session_manager.close_all()


def create_router(api_server: ApiServer) -> APIRouter:
    router = APIRouter(prefix="/v1/stream/webrtc", tags=["webrtc"])
    routes = WebRTCRoutes(api_server)

    api_server._webrtc_routes = routes

    @router.post("/offer", response_model=WebRTCOfferResponse, summary="WebRTC SDP offer/answer signaling")
    async def webrtc_offer(request: WebRTCOfferRequest):
        return await routes.handle_offer(request)

    @router.delete("/{session_id}", summary="Close WebRTC session")
    async def close_webrtc(session_id: str):
        return await routes.close_session(session_id)

    return router


def setup_routes(api_server: ApiServer) -> APIRouter:
    return create_router(api_server)
