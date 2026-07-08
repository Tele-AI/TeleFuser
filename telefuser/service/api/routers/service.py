"""
Service Routes for TeleFuser API

Provides service health, metadata, and metrics endpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import JSONResponse

from telefuser.metrics import get_service_metrics

if TYPE_CHECKING:
    from ..api_server import ApiServer

router = APIRouter(prefix="/v1/service", tags=["service"])


class ServiceRoutes:
    """Service route handlers with dependency injection."""

    def __init__(self, api_server: ApiServer) -> None:
        self.api = api_server

    async def get_status(self) -> dict:
        """Get service status."""
        status = self.api.task_manager.get_service_status()
        status["effective_max_concurrent_tasks"] = self.api.task_manager.max_concurrent_processing
        status["configured_max_concurrent_tasks"] = self.api.configured_max_concurrent_tasks
        pool_status = self._pipeline_pool_status()
        if pool_status is not None:
            status["execution_mode"] = "concurrent_pipeline_pool"
            status["pool"] = pool_status
        else:
            status["execution_mode"] = "serial_single_pipeline"
        webrtc_stats = self._webrtc_session_stats()
        status.update(webrtc_stats)
        if webrtc_stats.get("webrtc_active_sessions", 0) > 0 and status.get("service_status") == "idle":
            status["service_status"] = "active"
        return status

    def _pipeline_pool_status(self) -> list[dict] | None:
        """Return pipeline pool status when the inference service exposes a real pool."""
        if self.api.inference_service is None:
            return None
        pool_status_fn = getattr(self.api.inference_service, "pool_status", None)
        if not callable(pool_status_fn):
            return None
        pool_status = pool_status_fn()
        if not isinstance(pool_status, list) or not all(isinstance(replica, dict) for replica in pool_status):
            return None
        return pool_status

    def _webrtc_session_stats(self) -> dict:
        """Return WebRTC session stats if available."""
        routes = self.api._webrtc_routes
        if routes is None:
            return {}
        return routes._session_manager.session_stats()

    async def get_metadata(self) -> dict:
        """Get service metadata."""
        if self.api.inference_service is not None:
            metadata = self.api.inference_service.server_metadata()
            metadata["service_effective_max_concurrent_tasks"] = self.api.max_concurrent_tasks
            metadata["service_configured_max_concurrent_tasks"] = self.api.configured_max_concurrent_tasks
            metadata["max_queue_size"] = self.api.max_queue_size
            return metadata

        if self.api.stream_service is not None:
            metadata = self.api.stream_service.server_metadata()
            metadata["max_queue_size"] = self.api.max_queue_size
            metadata.update(self._webrtc_session_stats())
            return metadata

        raise HTTPException(status_code=503, detail="No service is initialized")

    async def health_check(self) -> dict:
        """Liveness endpoint for monitoring."""
        from datetime import datetime, timezone

        status = {
            "status": "healthy",
            "ready": self._is_ready(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
        }

        if self.api.inference_service:
            status["pipeline_ready"] = self.api.inference_service.is_running

        if self.api.stream_service:
            status["stream_ready"] = self.api.stream_service.is_running
            status["stream_mode"] = self.api.stream_service.stream_mode
            status.update(self._webrtc_session_stats())

        return status

    def _is_ready(self) -> bool:
        """Return whether the initialized service can currently accept work."""
        if self.api.inference_service is not None:
            pool_status = self._pipeline_pool_status()
            if pool_status is not None:
                return any(replica.get("status") != "dead" for replica in pool_status)
            return bool(getattr(self.api.inference_service, "is_running", False))

        if self.api.stream_service is not None:
            return bool(getattr(self.api.stream_service, "is_running", False))

        return False

    async def readiness_check(self) -> JSONResponse:
        """Readiness endpoint for load balancers."""
        body = await self.health_check()
        ready = bool(body.get("ready", False))
        body["status"] = "ready" if ready else "not_ready"
        return JSONResponse(
            status_code=status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body,
        )


def create_router(api_server: ApiServer) -> APIRouter:
    """Create a new router with fresh routes for the given ApiServer instance."""
    new_router = APIRouter(prefix="/v1/service", tags=["service"])
    routes = ServiceRoutes(api_server)

    @new_router.get("/status", response_model=dict)
    async def get_service_status() -> dict:
        return await routes.get_status()

    @new_router.get("/metadata", response_model=dict)
    async def get_service_metadata() -> dict:
        return await routes.get_metadata()

    @new_router.get("/health")
    async def health_check() -> dict:
        return await routes.health_check()

    @new_router.get("/ready")
    async def readiness_check() -> JSONResponse:
        return await routes.readiness_check()

    @new_router.get("/metrics", summary="Get Prometheus Metrics")
    async def get_metrics_endpoint() -> Response:
        """Get Prometheus-compatible metrics."""
        service_metrics = get_service_metrics()
        return Response(
            content=service_metrics.get_prometheus_format(),
            media_type="text/plain; charset=utf-8",
        )

    @new_router.get("/metrics/json", summary="Get Metrics (JSON)")
    async def get_metrics_json() -> dict:
        """Get metrics in JSON format."""
        service_metrics = get_service_metrics()
        registry = service_metrics.registry
        result = {
            "uptime_seconds": service_metrics.service_uptime.value,
            "tasks": {
                "created": service_metrics.tasks_created.value,
                "completed": service_metrics.tasks_completed.value,
                "failed": service_metrics.tasks_failed.value,
                "cancelled": service_metrics.tasks_cancelled.value,
            },
            "queue": {
                "size": service_metrics.queue_size.value,
                "pending": service_metrics.queue_pending.value,
                "processing": service_metrics.queue_processing.value,
            },
            "metrics_count": len(registry.list_metrics()),
            "registered_stages": registry.list_stages(),
        }
        result["webrtc"] = routes._webrtc_session_stats()
        return result

    return new_router


def setup_routes(api_server: ApiServer) -> APIRouter:
    """Setup service routes."""
    return create_router(api_server)
