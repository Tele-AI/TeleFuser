"""FastAPI app for the LiveKit serving entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from telefuser.metrics import get_service_metrics

from .runtime import LiveKitServeRuntime, session_record_to_response
from .schemas import (
    LiveKitHealthResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionDeleteResponse,
    SessionStatusResponse,
    SessionTokenRequest,
    SessionTokenResponse,
)
from .token_service import LiveKitDependencyError


def create_livekit_app(runtime: LiveKitServeRuntime) -> FastAPI:
    """Create the LiveKit-backed stream HTTP app."""

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await runtime.start()
            yield
        finally:
            await runtime.aclose()

    app = FastAPI(
        title="TeleFuser Stream API",
        description="LiveKit-backed real-time TeleFuser stream API.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime.config.cors_allow_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/stream/sessions", response_model=SessionCreateResponse)
    async def create_session(request: SessionCreateRequest):
        try:
            result = runtime.create_session(request)
        except LiveKitDependencyError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        admission = result.admission
        if admission.status == "rejected":
            detail = admission.reason or "no_capacity"
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail)

        body = SessionCreateResponse(
            session_id=result.record.session_id,
            room=result.record.room_name,
            livekit_url=runtime.config.livekit_url,
            token=result.token,
            worker_id=result.record.worker_id,
            status=result.record.status,
            expires_at=result.record.expires_at,
            queue_position=admission.queue_position,
        )
        if admission.status == "queued":
            return JSONResponse(status_code=status.HTTP_202_ACCEPTED, content=body.model_dump())
        return body

    @app.post("/v1/stream/sessions/{session_id}/tokens", response_model=SessionTokenResponse)
    async def create_token(session_id: str, request: SessionTokenRequest) -> SessionTokenResponse:
        try:
            record, token = runtime.create_viewer_token(session_id, request)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        except LiveKitDependencyError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

        return SessionTokenResponse(
            session_id=record.session_id,
            room=record.room_name,
            livekit_url=runtime.config.livekit_url,
            token=token,
            role="viewer",
        )

    @app.get("/v1/stream/sessions/{session_id}", response_model=SessionStatusResponse)
    async def get_session(session_id: str) -> SessionStatusResponse:
        try:
            return runtime.get_session_response(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.delete("/v1/stream/sessions/{session_id}", response_model=SessionDeleteResponse)
    async def delete_session(session_id: str) -> SessionDeleteResponse:
        try:
            record = await runtime.delete_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return SessionDeleteResponse(session_id=record.session_id, status=record.status)

    @app.get("/v1/stream/health", response_model=LiveKitHealthResponse)
    async def livekit_health() -> LiveKitHealthResponse:
        return runtime.health()

    @app.get("/v1/service/health")
    async def service_health() -> dict:
        health = runtime.health()
        return {
            "status": health.status,
            "ready": runtime.is_ready and health.status != "unhealthy",
            "service_type": "stream",
            "transport": "livekit",
            **health.model_dump(),
        }

    @app.get("/v1/service/ready")
    async def service_ready() -> JSONResponse:
        health = runtime.health()
        ready = runtime.is_ready and health.status != "unhealthy"
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "ready" if ready else "not_ready",
                "ready": ready,
                "service_type": "stream",
                "transport": "livekit",
                **health.model_dump(),
            },
        )

    @app.get("/v1/service/metadata")
    async def service_metadata() -> dict:
        return runtime.metadata()

    @app.get("/v1/service/metrics")
    async def service_metrics() -> Response:
        service_metrics_obj = get_service_metrics()
        return Response(
            content=service_metrics_obj.get_prometheus_format(),
            media_type="text/plain; charset=utf-8",
        )

    @app.get("/v1/service/metrics/json")
    async def service_metrics_json() -> dict:
        service_metrics_obj = get_service_metrics()
        health = runtime.health()
        return {
            "uptime_seconds": service_metrics_obj.service_uptime.value,
            "service_type": "stream",
            "transport": "livekit",
            "livekit": health.model_dump(),
        }

    return app
