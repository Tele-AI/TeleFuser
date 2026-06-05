"""
Tests for middleware components

Tests rate limiting, logging, and other middleware functionality.
"""

# Set environment before imports
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

os.environ["TELEFUSER_SECURITY_LEVEL"] = "NONE"

from telefuser.service.api.middleware import (
    LoggingMiddleware,
    RateLimitMiddleware,
    setup_middleware,
)
from telefuser.service.core.config import ServerConfig, server_config


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    def test_rate_limit_allows_requests_under_limit(self):
        """Test that requests under the limit are allowed."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        # Create middleware instance
        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=100,
            window_size=60,
            limited_paths=["/test"],
        )
        # Clear any state
        middleware._clients = {}

        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            return await middleware.dispatch(request, call_next)

        client = TestClient(app)

        # Should allow 5 requests
        for i in range(5):
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json() == {"status": "ok"}

    def test_rate_limit_only_applies_to_limited_paths(self):
        """Test that paths outside the whitelist are not rate limited."""
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "healthy"}

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        # Only /test is rate limited; /health passes through untouched.
        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=10,
            window_size=60,
            limited_paths=["/test"],
        )
        middleware._clients = {}

        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            return await middleware.dispatch(request, call_next)

        client = TestClient(app)

        # /health is not in limited_paths, so it should never be rate limited.
        for i in range(20):
            response = client.get("/health")
            assert response.status_code == 200, f"Health check failed on request {i + 1}"

    def test_rate_limit_headers_present(self):
        """Test that rate limit headers are included in responses."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=60,
            window_size=60,
            limited_paths=["/test"],
        )
        middleware._clients = {}

        @app.middleware("http")
        async def rate_limit_middleware(request, call_next):
            return await middleware.dispatch(request, call_next)

        client = TestClient(app)

        response = client.get("/test")
        assert response.status_code == 200

        # Check headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Window" in response.headers
        assert response.headers["X-RateLimit-Limit"] == "60"

    def test_rate_limit_get_client_id_from_forwarded_for(self):
        """Test client identification from X-Forwarded-For header."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=60,
            window_size=60,
            limited_paths=["/test"],
        )

        # Test client ID extraction
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [(b"x-forwarded-for", b"192.168.1.1")],
            "client": ("127.0.0.1", 12345),
        }
        request = Request(scope)

        client_id = middleware._get_client_id(request)
        assert client_id == "192.168.1.1"

    def test_rate_limit_get_client_id_from_client(self):
        """Test client identification from direct client IP."""
        app = FastAPI()

        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=60,
            window_size=60,
            limited_paths=["/test"],
        )

        # Test client ID extraction without X-Forwarded-For
        from starlette.requests import Request

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [],
            "client": ("10.0.0.1", 12345),
        }
        request = Request(scope)

        client_id = middleware._get_client_id(request)
        assert client_id == "10.0.0.1"

    def test_rate_limit_allow_request_logic(self):
        """Test the rate limit allow_request logic directly."""
        app = FastAPI()

        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=3,
            window_size=60,
            limited_paths=["/test"],
        )
        middleware._clients = {}

        # First 3 requests should be allowed (window cap)
        assert middleware._allow_request("client1") is True
        assert middleware._allow_request("client1") is True
        assert middleware._allow_request("client1") is True

        # Fourth request should be rejected since cap is reached
        assert middleware._allow_request("client1") is False

        # Different client should have its own counter
        assert middleware._allow_request("client2") is True

    def test_rate_limit_prefix_match(self):
        """Test that limited_paths is matched as a prefix, not an exact path."""
        app = FastAPI()

        middleware = RateLimitMiddleware(
            app,
            requests_per_minute=60,
            window_size=60,
            limited_paths=["/v1/tasks/create", "/v1/images"],
        )

        assert middleware._is_limited("/v1/tasks/create") is True
        assert middleware._is_limited("/v1/images/generations") is True
        assert middleware._is_limited("/v1/tasks/123/status") is False
        assert middleware._is_limited("/v1/files/download/x.mp4") is False
        assert middleware._is_limited("/v1/service/health") is False

    def test_rate_limit_error_response_structure(self):
        """Test RateLimitErrorResponse structure."""
        from telefuser.service.api.middleware import RateLimitErrorResponse

        error_response = RateLimitErrorResponse(
            request_id="abc123",
            limit=60,
            window=60,
            retry_after=60,
            message="Too many requests",
        )

        data = error_response.to_dict()

        assert "error" in data
        assert data["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert data["error"]["message"] == "Too many requests"
        assert data["error"]["request_id"] == "abc123"
        assert "details" in data["error"]
        assert "friendly_message" in data["error"]["details"]
        assert "suggestions" in data["error"]["details"]
        assert len(data["error"]["details"]["suggestions"]) > 0


class TestLoggingMiddleware:
    """Test logging middleware."""

    @patch("telefuser.service.api.middleware.logger")
    def test_logging_middleware_logs_requests(self, mock_logger):
        """Test that requests are logged."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(LoggingMiddleware)

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        # Verify logger was called
        assert mock_logger.debug.called
        assert mock_logger.info.called


class TestSetupMiddleware:
    """Test middleware setup function."""

    def test_setup_middleware_with_logging_only(self):
        """Test setup with only logging middleware."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        setup_middleware(app, enable_rate_limit=False, enable_logging=True)

        client = TestClient(app)

        # Requests should succeed
        response = client.get("/test")
        assert response.status_code == 200

    def test_setup_middleware_no_middleware(self):
        """Test setup with no middleware."""
        app = FastAPI()

        @app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        setup_middleware(app, enable_rate_limit=False, enable_logging=False)

        client = TestClient(app)

        # Requests should succeed
        response = client.get("/test")
        assert response.status_code == 200


class TestConfigIntegration:
    """Test integration with ServerConfig."""

    def test_rate_limit_config_values(self):
        """Test that rate limit config values are correct."""
        config = ServerConfig(
            enable_rate_limit=True,
            rate_limit_requests_per_minute=50,
            rate_limit_window_size=30,
            rate_limit_paths=["/v1/tasks/create"],
        )

        assert config.enable_rate_limit is True
        assert config.rate_limit_requests_per_minute == 50
        assert config.rate_limit_window_size == 30
        assert "/v1/tasks/create" in config.rate_limit_paths

    def test_rate_limit_can_be_disabled_via_config(self):
        """Test that rate limiting can be disabled via config."""
        config = ServerConfig(enable_rate_limit=False)

        assert config.enable_rate_limit is False

    def test_default_rate_limit_config(self):
        """Test default rate limit configuration."""
        config = ServerConfig()

        assert config.enable_rate_limit is True
        assert config.rate_limit_requests_per_minute == 60
        assert config.rate_limit_window_size == 60
        assert "/v1/tasks/create" in config.rate_limit_paths
        assert "/v1/tasks/form" in config.rate_limit_paths

    def test_rate_limit_config_validation(self):
        """Test rate limit config validation."""
        # Invalid requests_per_minute (too low)
        with pytest.raises(ValueError):
            ServerConfig(rate_limit_requests_per_minute=5)

        # Invalid window_size (too low)
        with pytest.raises(ValueError):
            ServerConfig(rate_limit_window_size=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
