"""LiveKit token generation."""

from __future__ import annotations

import datetime
from typing import Any

from .schemas import LiveKitTokenRole


class LiveKitDependencyError(RuntimeError):
    """Raised when LiveKit SDK dependencies are unavailable."""


class LiveKitTokenService:
    """Generate LiveKit JWTs with TeleFuser role-specific grants."""

    def __init__(self, *, api_key: str, api_secret: str, token_ttl: int) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.token_ttl = token_ttl

    def create_token(
        self,
        *,
        identity: str,
        room_name: str,
        role: LiveKitTokenRole,
        name: str | None = None,
        metadata: str | None = None,
    ) -> str:
        """Create a LiveKit join token for a scoped TeleFuser role."""
        api = self._load_livekit_api()
        grants = self._create_video_grants(api, room_name=room_name, role=role)

        token = api.AccessToken(self.api_key, self.api_secret)
        token = token.with_identity(identity)
        token = token.with_name(name or identity)
        token = token.with_grants(grants)
        token = token.with_ttl(datetime.timedelta(seconds=self.token_ttl))
        if metadata is not None:
            token = token.with_metadata(metadata)
        return token.to_jwt()

    @staticmethod
    def _load_livekit_api() -> Any:
        try:
            from livekit import api
        except ModuleNotFoundError as exc:
            raise LiveKitDependencyError(
                "LiveKit Python SDK is required for `telefuser stream-serve`. "
                "Install the declared TeleFuser runtime dependencies before starting this service."
            ) from exc
        return api

    @staticmethod
    def _create_video_grants(api: Any, *, room_name: str, role: LiveKitTokenRole) -> Any:
        kwargs: dict[str, Any] = {
            "room_join": True,
            "room": room_name,
        }
        if role == "controller":
            kwargs.update(can_publish=False, can_publish_data=True, can_subscribe=True)
        elif role == "viewer":
            kwargs.update(can_publish=False, can_publish_data=False, can_subscribe=True)
        elif role == "worker":
            kwargs.update(can_publish=True, can_publish_data=True, can_subscribe=False)
        elif role == "admin":
            kwargs.update(can_publish=True, can_publish_data=True, can_subscribe=True, room_admin=True)
        else:
            raise ValueError(f"Unsupported LiveKit role: {role}")
        return api.VideoGrants(**kwargs)
