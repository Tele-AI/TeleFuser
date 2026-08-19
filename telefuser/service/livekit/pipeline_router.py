"""Chunk-boundary routing and two-phase migration for TurboServe stream sessions."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any

from telefuser.utils.logging import logger

from .pipeline_adapter import LiveKitPipelineAdapter
from .turboserve import TurboServeOwnership, TurboServeOwnershipTable


class TurboServePipelineRouter:
    """Keep LiveKit transport stable while model-session ownership moves between workers."""

    def __init__(self, backends: dict[str, LiveKitPipelineAdapter]) -> None:
        if not backends:
            raise ValueError("TurboServe router requires at least one backend")
        self._backends = dict(backends)
        self._routes: dict[str, str] = {}
        self._ownership = TurboServeOwnershipTable()
        self._pending_chunks: dict[str, list[dict[str, Any]]] = {}
        self._migration_cleanup_failures = 0
        self._lock = threading.RLock()

    def worker_view(self, worker_id: str, *, gpu_ids: list[str] | None = None) -> TurboServeWorkerPipelineView:
        if worker_id not in self._backends:
            raise KeyError(worker_id)
        return TurboServeWorkerPipelineView(self, worker_id, gpu_ids=gpu_ids)

    def create_session(self, worker_id: str, config: dict[str, Any]) -> str:
        backend = self._backends[worker_id]
        pipeline_session_id = backend.create_session(config)
        try:
            with self._lock:
                if pipeline_session_id in self._routes:
                    raise ValueError(f"Pipeline session {pipeline_session_id!r} is already routed")
                self._ownership.register(pipeline_session_id, worker_id)
                self._routes[pipeline_session_id] = worker_id
        except Exception:
            backend.close_session(pipeline_session_id)
            raise
        return pipeline_session_id

    def push_chunk(self, pipeline_session_id: str, chunk: dict[str, Any]) -> None:
        with self._lock:
            pending = self._pending_chunks.get(pipeline_session_id)
            if pending is not None:
                pending.append(dict(chunk))
                return
            worker_id = self._routes.get(pipeline_session_id)
        if worker_id is None:
            raise KeyError(f"Unknown routed pipeline session {pipeline_session_id!r}")
        self._backends[worker_id].push_chunk(pipeline_session_id, chunk)

    async def pull_chunks(self, pipeline_session_id: str) -> AsyncGenerator[dict, None]:
        """Continue on a new backend after the source generator closes during commit."""
        while True:
            with self._lock:
                worker_id = self._routes.get(pipeline_session_id)
            if worker_id is None:
                return
            backend = self._backends[worker_id]
            async for chunk in backend.pull_chunks(pipeline_session_id):
                yield chunk
            with self._lock:
                next_worker_id = self._routes.get(pipeline_session_id)
            if next_worker_id is None or next_worker_id == worker_id:
                return

    def enable_publisher_frame_tracking(self, pipeline_session_id: str) -> bool:
        return self._backend_for(pipeline_session_id).enable_publisher_frame_tracking(pipeline_session_id)

    def report_publisher_frame_progress(
        self,
        pipeline_session_id: str,
        *,
        event: str,
        frames_delta: int,
        sequence: int,
        observed_monotonic_seconds: float,
    ) -> bool:
        return self._backend_for(pipeline_session_id).report_publisher_frame_progress(
            pipeline_session_id,
            event=event,
            frames_delta=frames_delta,
            sequence=sequence,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    def close_session(self, pipeline_session_id: str) -> None:
        with self._lock:
            worker_id = self._routes.pop(pipeline_session_id, None)
            if worker_id is None:
                return
            self._ownership.release(pipeline_session_id)
        self._backends[worker_id].close_session(pipeline_session_id)

    def migrate_session(self, pipeline_session_id: str, target_worker_id: str) -> TurboServeOwnership:
        """Atomically switch ownership while buffering controls received during transfer."""
        with self._lock:
            source_worker_id = self._routes[pipeline_session_id]
            if source_worker_id == target_worker_id:
                return self._ownership.owner(pipeline_session_id)
            if target_worker_id not in self._backends:
                raise KeyError(target_worker_id)
            token = self._ownership.prepare_migration(
                pipeline_session_id,
                source_worker_id,
                target_worker_id,
            )
            self._pending_chunks[pipeline_session_id] = []

        imported = False
        prepared = False
        source = None
        target = None
        try:
            source = self._migration_service(source_worker_id)
            target = self._migration_service(target_worker_id)
            bundle = source.prepare_migration(pipeline_session_id)
            prepared = True
            target.import_migration(
                bundle,
                owner_worker_id=target_worker_id,
                ownership_epoch=token.source_epoch + 1,
            )
            enable_tracking = getattr(target, "enable_publisher_frame_tracking", None)
            if callable(enable_tracking):
                enable_tracking(pipeline_session_id)
            imported = True
        except Exception:
            if imported and target is not None:
                target.close_session(pipeline_session_id)
            if prepared and source is not None:
                source.abort_migration(pipeline_session_id)
            with self._lock:
                pending = self._pending_chunks.pop(pipeline_session_id, [])
                self._ownership.abort_migration(token)
            for chunk in pending:
                self._backends[source_worker_id].push_chunk(pipeline_session_id, chunk)
            raise

        with self._lock:
            ownership = self._ownership.commit_migration(token)
            self._routes[pipeline_session_id] = target_worker_id
            pending = self._pending_chunks.pop(pipeline_session_id)
        for chunk in pending:
            self._backends[target_worker_id].push_chunk(pipeline_session_id, chunk)

        try:
            source.commit_migration(pipeline_session_id)
        except Exception as exc:
            # Ownership is already committed. Preserve the live target and report a source cleanup leak.
            with self._lock:
                self._migration_cleanup_failures += 1
            logger.warning(
                f"TurboServe source cleanup failed after committed migration: "
                f"session={pipeline_session_id} source={source_worker_id} error={exc}"
            )
        return ownership

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            routes = dict(self._routes)
            ownership = {session_id: asdict(self._ownership.owner(session_id)) for session_id in routes}
        retained_by_worker = {worker_id: 0 for worker_id in self._backends}
        for worker_id in routes.values():
            retained_by_worker[worker_id] += 1
        runtime_metrics: dict[str, dict[str, float | int | str]] = {}
        for worker_id, backend in self._backends.items():
            metrics = getattr(backend, "runtime_metrics", None)
            if not callable(metrics):
                continue
            try:
                value = metrics()
            except Exception as exc:
                logger.warning(f"Unable to read TurboServe runtime metrics: worker={worker_id} error={exc}")
                continue
            if value is not None:
                runtime_metrics[worker_id] = dict(value)
        return {
            "routes": routes,
            "ownership": ownership,
            "retained_sessions_by_worker": retained_by_worker,
            "worker_runtime_metrics": runtime_metrics,
            "migration_supported": True,
            "migration_cleanup_failures": self._migration_cleanup_failures,
        }

    def _backend_for(self, pipeline_session_id: str) -> LiveKitPipelineAdapter:
        with self._lock:
            worker_id = self._routes.get(pipeline_session_id)
        if worker_id is None:
            raise KeyError(f"Unknown routed pipeline session {pipeline_session_id!r}")
        return self._backends[worker_id]

    def _migration_service(self, worker_id: str) -> object:
        backend = self._backends[worker_id]
        stream_service = getattr(backend, "stream_service", None)
        service = getattr(stream_service, "service", None)
        required = ("prepare_migration", "import_migration", "commit_migration", "abort_migration", "close_session")
        if service is None or any(not callable(getattr(service, name, None)) for name in required):
            raise RuntimeError(f"Worker {worker_id} pipeline does not implement TurboServe migration")
        return service


class TurboServeWorkerPipelineView:
    """Worker-scoped facade backed by a shared session ownership router."""

    def __init__(
        self,
        router: TurboServePipelineRouter,
        worker_id: str,
        *,
        gpu_ids: list[str] | None = None,
    ) -> None:
        self._router = router
        self.worker_id = worker_id
        self.gpu_ids = list(gpu_ids) if gpu_ids is not None else None

    @property
    def _backend(self) -> LiveKitPipelineAdapter:
        return self._router._backends[self.worker_id]

    @property
    def stream_mode(self) -> str | None:
        return self._backend.stream_mode

    def start(
        self,
        pipeline_file: str,
        *,
        skip_validation: bool = False,
        gpu_num: int = 1,
        gpu_ids: list[str] | None = None,
    ) -> None:
        assigned = self.gpu_ids if gpu_ids is None else gpu_ids
        self._backend.start(
            pipeline_file,
            skip_validation=skip_validation,
            gpu_num=gpu_num,
            gpu_ids=assigned,
        )

    async def aclose(self) -> None:
        await self._backend.aclose()

    def configure_session_capacity(self, max_sessions: int | None) -> dict[str, object] | None:
        return self._backend.configure_session_capacity(max_sessions)

    def create_session(self, config: dict[str, Any]) -> str:
        return self._router.create_session(self.worker_id, config)

    def push_chunk(self, pipeline_session_id: str, chunk: dict[str, Any]) -> None:
        self._router.push_chunk(pipeline_session_id, chunk)

    async def pull_chunks(self, pipeline_session_id: str) -> AsyncGenerator[dict, None]:
        async for chunk in self._router.pull_chunks(pipeline_session_id):
            yield chunk

    def enable_publisher_frame_tracking(self, pipeline_session_id: str) -> bool:
        return self._router.enable_publisher_frame_tracking(pipeline_session_id)

    def report_publisher_frame_progress(
        self,
        pipeline_session_id: str,
        *,
        event: str,
        frames_delta: int,
        sequence: int,
        observed_monotonic_seconds: float,
    ) -> bool:
        return self._router.report_publisher_frame_progress(
            pipeline_session_id,
            event=event,
            frames_delta=frames_delta,
            sequence=sequence,
            observed_monotonic_seconds=observed_monotonic_seconds,
        )

    def close_session(self, pipeline_session_id: str) -> None:
        self._router.close_session(pipeline_session_id)

    async def stream_task(self, config: dict[str, Any]) -> AsyncGenerator[dict, None]:
        async for chunk in self._backend.stream_task(config):
            yield chunk


class TurboServeRoutedWorkerPoolMixin:
    """Small protocol helper used only for runtime feature detection."""

    router: TurboServePipelineRouter

    async def migrate_session(self, pipeline_session_id: str, target_worker_id: str) -> TurboServeOwnership:
        return await asyncio.to_thread(self.router.migrate_session, pipeline_session_id, target_worker_id)
