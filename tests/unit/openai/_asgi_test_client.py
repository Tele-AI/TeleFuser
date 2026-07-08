from __future__ import annotations

import asyncio
from typing import Any

import httpx


class ASGITestClient:
    """Small sync wrapper around httpx ASGITransport for route unit tests."""

    __test__ = False

    def __init__(self, app) -> None:
        self.app = app

    def __enter__(self) -> ASGITestClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        async def _request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(_request())

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)
