"""
Middleware for TeleFuser API

Provides rate limiting, logging, metrics collection, and other cross-cutting concerns.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from telefuser.utils.logging import logger

from ..core.config import server_config

if TYPE_CHECKING:
    from telefuser.metrics import MetricRegistry


class RateLimitErrorResponse:
    """Structured response for rate limit errors."""

    def __init__(
        self,
        request_id: str,
        limit: int,
        window: int,
        retry_after: int,
        message: str | None = None,
    ):
        self.request_id = request_id
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        self.message = message or "Rate limit exceeded"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": self.message,
                "request_id": self.request_id,
                "details": {
                    "limit": self.limit,
                    "window_seconds": self.window,
                    "retry_after_seconds": self.retry_after,
                    "friendly_message": (
                        f"You have made too many requests. "
                        f"Please wait {self.retry_after} seconds before trying again. "
                        f"Current limit: {self.limit} requests per {self.window} seconds."
                    ),
                    "suggestions": [
                        "Reduce the frequency of your requests",
                        "Consider using bulk operations if available",
                        "Contact support if you need higher rate limits",
                    ],
                },
            }
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware based on client IP with sliding window.

    Operates as a whitelist: only requests whose path starts with one of
    ``limited_paths`` are counted and gated. Everything else (status polls,
    health checks, file downloads, stream endpoints) passes through untouched.

    Uses in-memory storage. For production with multiple instances,
    consider using Redis-based rate limiting with fastapi-limiter.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int | None = None,
        window_size: int | None = None,
        limited_paths: list[str] | None = None,
    ):
        super().__init__(app)
        config = server_config
        self.requests_per_minute = requests_per_minute or config.rate_limit_requests_per_minute
        self.window_size = window_size or config.rate_limit_window_size
        self.limited_paths = tuple(limited_paths if limited_paths is not None else config.rate_limit_paths)

        self._clients: dict[str, list[float]] = {}

    def _is_limited(self, path: str) -> bool:
        return any(path.startswith(prefix) for prefix in self.limited_paths)

    async def dispatch(self, request: Request, call_next: callable) -> Response:
        """Process request with rate limiting."""
        request_id = str(uuid.uuid4())[:8]

        if request.method == "OPTIONS" or not self._is_limited(request.url.path):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        if not server_config.enable_rate_limit:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response

        client_id = self._get_client_id(request)

        if not self._allow_request(client_id):
            logger.warning(f"Rate limit exceeded for {client_id} (request_id: {request_id})")

            error_response = RateLimitErrorResponse(
                request_id=request_id,
                limit=self.requests_per_minute,
                window=self.window_size,
                retry_after=self.window_size,
                message="Too many requests",
            )

            raise HTTPException(
                status_code=429,
                detail=error_response.to_dict(),
                headers={
                    "Retry-After": str(self.window_size),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Window": str(self.window_size),
                    "X-Request-ID": request_id,
                },
            )

        response = await call_next(request)

        remaining = self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Window"] = str(self.window_size)
        response.headers["X-Request-ID"] = request_id

        return response

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        if request.client:
            return request.client.host

        return "unknown"

    def _allow_request(self, client_id: str) -> bool:
        """Check if request should be allowed based on rate limits."""
        now = time.time()

        # Clean old entries periodically to prevent memory leak
        if len(self._clients) > 10000:
            self._clean_old_entries(now)

        if client_id not in self._clients:
            self._clients[client_id] = []

        requests = self._clients[client_id]

        # Remove requests outside the window
        cutoff = now - self.window_size
        requests[:] = [r for r in requests if r > cutoff]

        if len(requests) >= self.requests_per_minute:
            return False

        requests.append(now)
        return True

    def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        if client_id not in self._clients:
            return self.requests_per_minute

        now = time.time()
        cutoff = now - self.window_size
        requests = [r for r in self._clients[client_id] if r > cutoff]

        return self.requests_per_minute - len(requests)

    def _clean_old_entries(self, now: float) -> None:
        """Clean old entries to prevent memory leak."""
        cutoff = now - self.window_size
        self._clients = {k: v for k, v in self._clients.items() if v and v[-1] > cutoff}


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""

    async def dispatch(self, request: Request, call_next: callable) -> Response:
        """Log request and response."""
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

        logger.debug(f"[{request_id}] {request.method} {request.url.path}")

        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({duration:.3f}s)")

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for HTTP request metrics collection.

    Collects metrics for each HTTP request including:
    - Request count by method, path, and status
    - Request duration histogram
    - In-flight request gauge
    """

    def __init__(
        self,
        app: ASGIApp,
        registry: MetricRegistry | None = None,
        exempt_paths: list[str] | None = None,
    ) -> None:
        """Initialize metrics middleware.

        Args:
            app: The ASGI application.
            registry: Optional metrics registry. Uses global registry if not provided.
            exempt_paths: Paths to exclude from metrics collection.
        """
        super().__init__(app)
        self._registry = registry
        self._exempt_paths = exempt_paths or ["/health", "/metrics", "/v1/service/health"]

        # Metrics will be lazily initialized with the registry
        self._initialized = False
        self._http_requests_total = None
        self._http_request_duration = None
        self._http_requests_inflight = None

    def _ensure_initialized(self) -> bool:
        """Ensure metrics are initialized."""
        if self._initialized:
            return self._registry is not None

        if self._registry is None:
            try:
                from telefuser.metrics import get_metrics_registry

                self._registry = get_metrics_registry()
            except ImportError:
                return False

        # Initialize metrics
        self._http_requests_total = self._registry.counter(
            "http_requests_total",
            "Total number of HTTP requests",
        )

        self._http_request_duration = self._registry.histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
        )

        self._http_requests_inflight = self._registry.gauge(
            "http_requests_inflight",
            "Number of HTTP requests currently being processed",
        )

        self._initialized = True
        return True

    async def dispatch(self, request: Request, call_next: callable) -> Response:
        """Process request with metrics collection."""
        if not self._ensure_initialized():
            return await call_next(request)

        # Skip metrics for exempt paths
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        # Track in-flight requests
        self._http_requests_inflight.inc()

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.perf_counter() - start_time

            # Record metrics
            self._http_request_duration.observe(duration)
            self._http_requests_inflight.dec()


def setup_middleware(
    app: FastAPI,
    enable_rate_limit: bool = True,
    enable_logging: bool = True,
    enable_metrics: bool = True,
    metrics_registry: MetricRegistry | None = None,
) -> None:
    """Setup middleware for FastAPI app.

    Args:
        app: The FastAPI application.
        enable_rate_limit: Whether to enable rate limiting.
        enable_logging: Whether to enable request logging.
        enable_metrics: Whether to enable metrics collection.
        metrics_registry: Optional metrics registry for metrics middleware.
    """
    if enable_logging:
        app.add_middleware(LoggingMiddleware)

    if enable_metrics:
        app.add_middleware(
            MetricsMiddleware,
            registry=metrics_registry,
        )

    if enable_rate_limit and server_config.enable_rate_limit:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=server_config.rate_limit_requests_per_minute,
            window_size=server_config.rate_limit_window_size,
            limited_paths=server_config.rate_limit_paths,
        )
