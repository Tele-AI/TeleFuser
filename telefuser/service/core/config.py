"""Service configuration with security settings."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..security.security_validator import SecurityLevel


class ServerConfig(BaseSettings):
    """Server configuration including security settings.

    Uses Pydantic for validation and environment variable support.
    """

    model_config = SettingsConfigDict(
        env_prefix="TELEFUSER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Task settings
    task_timeout: int = Field(default=1200, ge=60, le=3600, description="Task timeout in seconds")
    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Deprecated compatibility field. A single ppl instance is currently executed serially, "
            "so effective task concurrency is forced to 1. Use max_queue_size to control the maximum "
            "number of queued plus running tasks."
        ),
    )
    max_queue_size: int = Field(default=10, ge=1, le=1000, description="Maximum queue size")

    # Task cleanup settings
    cleanup_keep_count: int = Field(
        default=1000, ge=100, le=10000, description="Number of completed tasks to keep in memory"
    )

    cancel_timeout: float = Field(default=5.0, ge=1.0, le=30.0, description="Timeout for task cancellation")

    processing_lock_timeout: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Timeout for acquiring processing lock"
    )

    # Cache settings
    cache_dir: str = Field(default="work_dirs/server_cache", description="Cache directory path")

    enable_latent_cache: bool | None = Field(
        default=None,
        description="Latent cache service override; when unset, the pipeline CACHE_CONFIG value is used",
    )
    cache_mode: Literal["read_write", "read_only", "write_only"] | None = Field(
        default=None,
        description="Latent cache mode override; when unset, the pipeline CACHE_CONFIG value is used",
    )

    # Security settings
    security_level: SecurityLevel = Field(default=SecurityLevel.STRICT, description="Security validation level")

    max_ppl_file_size: int = Field(
        default=1024 * 1024,  # 1MB
        ge=1024,
        le=10 * 1024 * 1024,  # 10MB
        description="Maximum pipeline file size in bytes",
    )

    allow_unsafe_pipelines: bool = Field(default=False, description="Allow unsafe pipelines with warnings")

    strict_validation: bool = Field(default=True, description="Validation errors prevent startup")

    # SSL/TLS settings
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    ssl_cert_path: str | None = Field(default=None, description="Path to custom SSL certificate")

    # API settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")

    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Log level")

    enable_access_log: bool = Field(default=True, description="Enable access logging")

    # File service settings
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        ge=1 * 1024 * 1024,  # 1MB
        le=1024 * 1024 * 1024,  # 1GB
        description="Maximum file download size",
    )
    artifact_retention_seconds: int = Field(
        default=7 * 24 * 60 * 60,
        ge=0,
        description="Maximum retention time for terminal task artifacts. Zero disables TTL cleanup.",
    )
    artifact_tmp_retention_seconds: int = Field(
        default=60 * 60,
        ge=0,
        description="Maximum retention time for temporary .part files. Zero disables tmp cleanup.",
    )
    artifact_cleanup_interval_seconds: int = Field(
        default=60 * 60,
        ge=60,
        description="Suggested interval for periodic artifact cleanup.",
    )
    artifact_max_total_bytes: int = Field(
        default=0,
        ge=0,
        description="Maximum local artifact bytes. Zero disables capacity cleanup.",
    )
    artifact_max_task_bytes: int = Field(
        default=0,
        ge=0,
        description="Maximum local artifact bytes per task. Zero disables per-task capacity checks.",
    )
    artifact_preserve_failed_outputs: bool = Field(
        default=False,
        description="Whether failed task output directories should be preserved until normal retention expiry.",
    )

    # Rate limiting settings
    enable_rate_limit: bool = Field(default=True, description="Enable rate limiting middleware")

    rate_limit_requests_per_minute: int = Field(
        default=60, ge=10, le=10000, description="Maximum requests per minute per client"
    )

    rate_limit_window_size: int = Field(default=60, ge=10, le=3600, description="Rate limiting window size in seconds")

    rate_limit_paths: list = Field(
        default_factory=lambda: [
            "/v1/tasks/create",
            "/v1/tasks/form",
            "/v1/images/generations",
            "/v1/videos/generations",
        ],
        description=(
            "Path prefixes subject to rate limiting (whitelist). "
            "Requests whose URL path does not start with any of these are not rate limited."
        ),
    )

    # Metrics settings
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")

    enable_gpu_metrics: bool = Field(default=True, description="Enable GPU metrics collection")

    enable_stage_metrics: bool = Field(default=True, description="Enable stage-level metrics collection")

    gpu_metrics_interval: float = Field(
        default=5.0, ge=1.0, le=60.0, description="Interval for GPU metrics collection in seconds"
    )

    metrics_path: str = Field(default="/v1/service/metrics", description="HTTP path for Prometheus metrics endpoint")

    metrics_namespace: str = Field(default="telefuser", description="Namespace prefix for all metrics")

    gpu_platform: Literal["nvidia", "amd", "auto"] = Field(
        default="auto", description="GPU platform for metrics collection"
    )

    # Stream settings
    webrtc_max_sessions: int = Field(default=10, ge=1, le=100, description="Maximum concurrent WebRTC sessions")

    # Pipeline replication settings
    num_replicas: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of independent pipeline replicas for concurrent serving.",
    )

    # WebRTC ICE settings (for public network deployment)
    stun_servers: list[str] = Field(
        default_factory=lambda: ["stun:stun.l.google.com:19302"],
        description="STUN server URLs (e.g. stun:stun.l.google.com:19302)",
    )
    turn_server: str | None = Field(default=None, description="TURN server URL (e.g. turn:your-domain.com:3478)")
    turn_username: str | None = Field(default=None, description="TURN server username")
    turn_credential: str | None = Field(default=None, description="TURN server credential")

    @field_validator("port")
    @classmethod
    def validate_port(cls: type[ServerConfig], v: int) -> int:
        """Validate port number - warn if using privileged ports."""
        if v < 1024 and v != 0:
            import warnings

            warnings.warn(
                f"Port {v} is a privileged port. Running on ports < 1024 requires root privileges.",
                UserWarning,
            )
        return v

    @field_validator("security_level", mode="before")
    @classmethod
    def validate_security_level(cls: type[ServerConfig], v: SecurityLevel | str) -> SecurityLevel:
        """Validate security level from string."""
        if isinstance(v, str):
            try:
                return SecurityLevel[v.upper()]
            except KeyError:
                raise ValueError(f"Invalid security level: {v}")
        return v

    def validate(self) -> bool:
        """Validate the entire configuration."""
        try:
            ServerConfig.model_validate(self.model_dump())
            return True
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")

    @property
    def effective_max_concurrent_tasks(self) -> int:
        """Effective task concurrency — equals num_replicas (one slot per replica)."""
        return self.num_replicas

    def resolve_replica_device_ids(self, parallelism: int) -> list[list[str]]:
        """Compute device groups for DP replicas.

        Precedence: --parallelism determines HOW MANY devices to use.
        The platform's device control env var (if set) provides the physical device mapping.
        Device tokens are treated as opaque strings (supports UUIDs, MIG).
        """
        from telefuser.platforms import current_platform

        env_var = current_platform.device_control_env_var
        cvd = os.environ.get(env_var)
        if cvd is not None and cvd.strip():
            all_visible = [d.strip() for d in cvd.split(",") if d.strip()]
        else:
            all_visible = [str(i) for i in range(parallelism)]

        if len(all_visible) < parallelism:
            raise ValueError(f"--parallelism={parallelism} but only {len(all_visible)} GPUs visible ({env_var}={cvd})")

        selected = all_visible[:parallelism]

        if parallelism % self.num_replicas != 0:
            raise ValueError(f"--parallelism ({parallelism}) must be divisible by --num-replicas ({self.num_replicas})")

        gpus_per_replica = parallelism // self.num_replicas
        return [selected[i * gpus_per_replica : (i + 1) * gpus_per_replica] for i in range(self.num_replicas)]


# Global server configuration instance
server_config = ServerConfig()


def configure_security(
    level: SecurityLevel = SecurityLevel.STRICT,
    allow_unsafe: bool = False,
    max_file_size: int = 1024 * 1024,
    custom_blocked_patterns: list[str] | None = None,
) -> None:
    """Configure security settings for the server."""
    global server_config
    server_config.security_level = level
    server_config.allow_unsafe_pipelines = allow_unsafe
    server_config.max_ppl_file_size = max_file_size
    if custom_blocked_patterns:
        # Note: blocked_patterns would need to be added to ServerConfig if needed
        pass

    import logging

    logging.info(f"Security configured: level={level.name}, allow_unsafe={allow_unsafe}")


def load_config_from_env() -> None:
    """Load configuration from environment variables."""
    global server_config
    server_config = ServerConfig()

    import logging

    logging.info("Configuration loaded from environment")
