from __future__ import annotations

import sys
import types

from telefuser.service.livekit.token_service import LiveKitTokenService


def test_token_service_builds_viewer_grants(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeVideoGrants:
        def __init__(self, **kwargs) -> None:
            captured["grants"] = kwargs

    class FakeAccessToken:
        def __init__(self, api_key, api_secret) -> None:
            captured["api_key"] = api_key
            captured["api_secret"] = api_secret

        def with_identity(self, identity):
            captured["identity"] = identity
            return self

        def with_name(self, name):
            captured["name"] = name
            return self

        def with_grants(self, grants):
            captured["grant_obj"] = grants
            return self

        def with_ttl(self, ttl):
            captured["ttl"] = ttl.total_seconds()
            return self

        def to_jwt(self):
            return "jwt-token"

    fake_api = types.SimpleNamespace(AccessToken=FakeAccessToken, VideoGrants=FakeVideoGrants)
    monkeypatch.setitem(sys.modules, "livekit", types.SimpleNamespace(api=fake_api))

    token = LiveKitTokenService(api_key="key", api_secret="secret", token_ttl=123).create_token(
        identity="viewer-1",
        room_name="room-1",
        role="viewer",
    )

    assert token == "jwt-token"
    assert captured["api_key"] == "key"
    assert captured["api_secret"] == "secret"
    assert captured["identity"] == "viewer-1"
    assert captured["ttl"] == 123
    assert captured["grants"] == {
        "room_join": True,
        "room": "room-1",
        "can_publish": False,
        "can_publish_data": False,
        "can_subscribe": True,
    }


def test_token_service_builds_worker_grants(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeVideoGrants:
        def __init__(self, **kwargs) -> None:
            captured["grants"] = kwargs

    class FakeAccessToken:
        def __init__(self, api_key, api_secret) -> None:
            return None

        def with_identity(self, identity):
            return self

        def with_name(self, name):
            return self

        def with_grants(self, grants):
            return self

        def with_ttl(self, ttl):
            return self

        def to_jwt(self):
            return "worker-token"

    fake_api = types.SimpleNamespace(AccessToken=FakeAccessToken, VideoGrants=FakeVideoGrants)
    monkeypatch.setitem(sys.modules, "livekit", types.SimpleNamespace(api=fake_api))

    token = LiveKitTokenService(api_key="key", api_secret="secret", token_ttl=60).create_token(
        identity="worker-0",
        room_name="room-1",
        role="worker",
    )

    assert token == "worker-token"
    assert captured["grants"] == {
        "room_join": True,
        "room": "room-1",
        "can_publish": True,
        "can_publish_data": True,
        "can_subscribe": False,
    }
