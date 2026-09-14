"""Standalone generic VLA WebSocket service backed by TeleFuser replicas."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from telefuser.service.core.config import ServerConfig
from telefuser.service.core.pipeline_pool import PipelinePool
from telefuser.service.security.security_validator import SecurityLevel
from telefuser.service.vla_session import create_pipeline_pool_vla_session_app

DEFAULT_PIPELINE_FILE = Path(__file__).with_name("lingbot_vla_v2_native_service.py")


def create_app(
    *,
    pipeline_file: str | Path = DEFAULT_PIPELINE_FILE,
    parallelism: int = 1,
    num_replicas: int = 1,
    skip_validation: bool = False,
) -> FastAPI:
    """Load LingBot replicas and create the standalone generic VLA app."""
    config = ServerConfig(num_replicas=num_replicas, security_level=SecurityLevel.NONE)
    replica_device_ids = config.resolve_replica_device_ids(parallelism)
    pool = PipelinePool(
        num_replicas=num_replicas,
        replica_device_ids=replica_device_ids,
        security_level_name=config.security_level.name,
        config=config,
    )
    started = pool.start_all(
        ppl_file=str(Path(pipeline_file).expanduser().resolve()),
        parallelism_per_replica=len(replica_device_ids[0]),
        task="vla_action",
        skip_validation=skip_validation,
        vla_provider_factory="get_vla_provider",
    )
    if not started:
        raise RuntimeError("failed to start LingBot-VLA v2 pipeline replicas")
    return create_pipeline_pool_vla_session_app(pool)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--num-replicas", type=int, default=1)
    parser.add_argument("--pipeline-file", type=Path, default=DEFAULT_PIPELINE_FILE)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(
        pipeline_file=args.pipeline_file,
        parallelism=args.parallelism,
        num_replicas=args.num_replicas,
        skip_validation=args.skip_validation,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
