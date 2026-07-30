"""Configuration for the LiveKit-backed ``telefuser stream-serve`` command."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveKitServeConfig(BaseSettings):
    """Runtime configuration for the LiveKit serving entrypoint."""

    model_config = SettingsConfigDict(
        env_prefix="TELEFUSER_LIVEKIT_",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="HTTP bind host")
    port: int = Field(default=8088, ge=1, le=65535, description="HTTP bind port")

    livekit_url: str = Field(default="", description="LiveKit server URL")
    livekit_api_key: str = Field(default="", description="LiveKit API key")
    livekit_api_secret: str = Field(default="", description="LiveKit API secret")

    num_workers: int = Field(default=1, ge=1, le=64, description="Number of TeleFuser LiveKit workers")
    max_sessions_per_worker: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Maximum retained sessions per model worker",
    )
    worker_gpu_map: str | None = Field(
        default=None,
        description="Semicolon-separated worker GPU groups, for example '0,1;2,3'",
    )
    worker_mode: Literal["in-process", "process"] = Field(
        default="in-process",
        description="Worker isolation mode",
    )

    queue_size: int = Field(default=0, ge=0, le=10000, description="Maximum queued sessions")
    control_idle_timeout: float = Field(
        default=10.0,
        gt=0,
        description="Seconds without control activity before a LingBot execution lease may yield",
    )
    session_timeout: int = Field(default=1800, ge=1, description="Maximum session lifetime in seconds")
    token_ttl: int = Field(default=3600, ge=1, description="LiveKit token TTL in seconds")
    controller_timeout: int = Field(
        default=60,
        ge=0,
        description="Seconds to keep a session after controller disconnect",
    )
    room_empty_timeout: int = Field(
        default=30,
        ge=0,
        description="Seconds to keep a session after the LiveKit room becomes empty",
    )
    role_mode: Literal["single-controller"] = Field(default="single-controller")

    default_fps: int = Field(default=16, ge=1, le=120, description="Default output video FPS")
    max_data_message_bytes: int = Field(default=12 * 1024, ge=1024, description="Maximum accepted data message size")
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS origins for the LiveKit serve API",
    )

    @field_validator("worker_gpu_map")
    @classmethod
    def validate_worker_gpu_map(cls: type[LiveKitServeConfig], value: str | None) -> str | None:
        """Normalize empty GPU maps to ``None``."""
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    def require_livekit_credentials(self) -> None:
        """Raise when the minimum LiveKit connection settings are missing."""
        missing = []
        if not self.livekit_url:
            missing.append("livekit_url")
        if not self.livekit_api_key:
            missing.append("livekit_api_key")
        if not self.livekit_api_secret:
            missing.append("livekit_api_secret")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing LiveKit configuration: {joined}")

    def worker_gpu_groups(self) -> list[list[str]]:
        """Return one GPU-id group per configured worker."""
        if self.worker_gpu_map is None:
            return [[] for _ in range(self.num_workers)]

        groups = [
            [gpu.strip() for gpu in group.split(",") if gpu.strip()]
            for group in self.worker_gpu_map.split(";")
            if group.strip()
        ]
        if len(groups) != self.num_workers:
            raise ValueError(f"worker_gpu_map defines {len(groups)} worker groups, but num_workers={self.num_workers}")
        return groups
