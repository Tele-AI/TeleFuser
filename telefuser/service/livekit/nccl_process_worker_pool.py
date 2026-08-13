"""Process-isolated ABot model workers with NCCL state migration.

The parent retains LiveKit transport ownership. Child processes retain model
state, so a committed migration changes only the model route, not the room.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from typing import Any

import torch
import torch.distributed as dist

from telefuser.service.core.stream_pipeline_service import STREAM_MODE_BIDIRECTIONAL
from telefuser.service.security.security_validator import SecurityLevel
from telefuser.utils.logging import logger

from .nccl_transfer import allocate_tensor_tree_leaves, transfer_tensor_leaves_nccl
from .pipeline_adapter import LiveKitPipelineAdapter
from .process_worker_pool import ProcessLiveKitWorkerPool, ProcessWorkerSpec, _close_queue
from .session_registry import SessionRecord
from .token_service import LiveKitTokenService
from .turboserve import TurboServeOwnership, TurboServeOwnershipTable
from .worker import LiveKitWorker


class _ProcessPipelineAdapter:
    stream_mode = STREAM_MODE_BIDIRECTIONAL

    def __init__(self, pool: "NCCLProcessLiveKitWorkerPool", initial_worker_id: str) -> None:
        self._pool = pool
        self._initial_worker_id = initial_worker_id

    def create_session(self, config: dict) -> str:
        session_id = str(config["session_id"])
        self._pool.create_model_session(self._initial_worker_id, session_id, config)
        return session_id

    def push_chunk(self, session_id: str, chunk: dict) -> None:
        self._pool.push_model_chunk(session_id, chunk)

    async def pull_chunks(self, session_id: str):
        async for chunk in self._pool.pull_model_chunks(session_id):
            yield chunk

    def close_session(self, session_id: str) -> None:
        self._pool.close_model_session(session_id)


class _ParentTransportSink:
    def __init__(self, pool: "NCCLProcessLiveKitWorkerPool") -> None:
        self.pool = pool

    def on_worker_status(self, worker_id: str, status: str) -> None:
        del worker_id, status

    def on_worker_capacity(self, worker_id: str, capacity: int, profile: dict[str, object] | None = None) -> None:
        self.pool._event_sink.on_worker_capacity(worker_id, capacity, profile)

    def on_session_status(self, session_id: str, status: str, error: str | None = None) -> None:
        self.pool._event_sink.on_session_status(session_id, status, error)

    def on_pipeline_session(self, session_id: str, pipeline_session_id: str) -> None:
        self.pool._event_sink.on_pipeline_session(session_id, pipeline_session_id)

    def on_session_finished(self, worker_id: str, session_id: str, error: str | None = None) -> None:
        self.pool._transport_finished(session_id)
        self.pool._event_sink.on_session_finished(worker_id, session_id, error)


class NCCLProcessLiveKitWorkerPool(ProcessLiveKitWorkerPool):
    """TurboServe-compatible parent transport / GPU model-process pool."""

    def __init__(self, specs: list[ProcessWorkerSpec], **kwargs: Any) -> None:
        super().__init__(specs, **kwargs)
        self._worker_target = _nccl_model_worker_main
        self._ownership = TurboServeOwnershipTable()
        self._model_outputs: dict[str, asyncio.Queue[dict | None]] = {}
        self._transport_workers: dict[str, LiveKitWorker] = {}
        self._transport_tasks: dict[str, asyncio.Task[None]] = {}
        self._migrating_controls: dict[str, list[dict]] = {}
        self._worker_runtime_metrics: dict[str, dict[str, float | int]] = {}
        self._session_runtime_metrics: dict[str, dict[str, float | int]] = {}
        self._migration_total_ms: list[float] = []
        self._nccl_ranks: dict[str, int] = {}
        self._migration_lock = asyncio.Lock()

    async def start(self, *, skip_validation: bool = False) -> None:
        await super().start(skip_validation=skip_validation)
        if len(self._active_workers) > 1:
            await self._init_nccl()

    async def scale_to(self, target_workers: int) -> int:
        """Rebuild the static NCCL communicator around a new replica set."""
        async with self._migration_lock:
            if len(self._active_workers) == target_workers:
                return target_workers
            if self._nccl_ranks:
                await asyncio.gather(
                    *(self._request(worker_id, "nccl_destroy") for worker_id in self._nccl_ranks),
                    return_exceptions=True,
                )
                self._nccl_ranks.clear()
            actual = await super().scale_to(target_workers)
            if actual > 1:
                await self._init_nccl()
            return actual

    def start_session(self, record: SessionRecord) -> None:
        if record.worker_id is None or record.worker_id not in self._active_workers:
            raise RuntimeError("Model worker is not active")
        runner = LiveKitWorker(
            worker_id=record.worker_id,
            config=self._config,
            pipeline_file=self._pipeline_file,
            token_service=LiveKitTokenService(
                api_key=self._config.livekit_api_key,
                api_secret=self._config.livekit_api_secret,
                token_ttl=self._config.token_ttl,
            ),
            event_sink=_ParentTransportSink(self),
            pipeline_adapter=_ProcessPipelineAdapter(self, record.worker_id),
        )
        task = asyncio.create_task(runner.run_session(record), name=f"livekit-transport-{record.session_id}")
        self._transport_workers[record.session_id] = runner
        self._transport_tasks[record.session_id] = task
        task.add_done_callback(lambda done, sid=record.session_id: self._transport_task_done(sid, done))

    async def stop_session(self, session_id: str) -> None:
        runner = self._transport_workers.get(session_id)
        task = self._transport_tasks.get(session_id)
        if runner is not None:
            await runner.stop_session(session_id)
        if task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(task), timeout=15.0)

    def create_model_session(self, worker_id: str, session_id: str, config: dict) -> None:
        self._model_outputs[session_id] = asyncio.Queue()
        self._pipeline_routes[session_id] = worker_id
        self._session_workers[session_id] = worker_id
        self._ownership.register(session_id, worker_id)
        self._send(worker_id, {"type": "model_create", "session_id": session_id, "config": dict(config)})

    def push_model_chunk(self, session_id: str, chunk: dict) -> None:
        if session_id in self._migrating_controls:
            self._migrating_controls[session_id].append(dict(chunk))
            return
        self._send(self._pipeline_routes[session_id], {"type": "model_push", "session_id": session_id, "chunk": dict(chunk)})

    def close_model_session(self, session_id: str) -> None:
        worker_id = self._pipeline_routes.pop(session_id, None)
        self._session_workers.pop(session_id, None)
        self._ownership.release(session_id)
        self._migrating_controls.pop(session_id, None)
        self._session_runtime_metrics.pop(session_id, None)
        if worker_id in self._active_workers:
            self._send(worker_id, {"type": "model_close", "session_id": session_id})
        if (output := self._model_outputs.pop(session_id, None)) is not None:
            output.put_nowait(None)

    async def pull_model_chunks(self, session_id: str):
        output = self._model_outputs[session_id]
        while (payload := await output.get()) is not None:
            yield payload

    async def migrate_session(self, pipeline_session_id: str, target_worker_id: str) -> TurboServeOwnership:
        async with self._migration_lock:
            source_worker_id = self._pipeline_routes[pipeline_session_id]
            if source_worker_id == target_worker_id:
                return self._ownership.owner(pipeline_session_id)
            if source_worker_id not in self._nccl_ranks or target_worker_id not in self._nccl_ranks:
                raise RuntimeError("NCCL migration requires initialized source and target workers")
            token = self._ownership.prepare_migration(pipeline_session_id, source_worker_id, target_worker_id)
            self._migrating_controls[pipeline_session_id] = []
            started = time.monotonic()
            try:
                await asyncio.gather(
                    self._request(source_worker_id, "scheduler_pause", timeout=300.0),
                    self._request(target_worker_id, "scheduler_pause", timeout=300.0),
                )
                exported = await self._request(source_worker_id, "nccl_export", session_id=pipeline_session_id, transfer_id=token.token_id, timeout=300.0)
                metadata = dict(exported["result"])
                await self._request(target_worker_id, "nccl_prepare_recv", transfer_id=token.token_id, metadata=metadata, source_rank=self._nccl_ranks[source_worker_id], owner_worker_id=target_worker_id, ownership_epoch=token.source_epoch + 1, timeout=300.0)
                await asyncio.gather(
                    self._request(source_worker_id, "nccl_send", transfer_id=token.token_id, target_rank=self._nccl_ranks[target_worker_id], timeout=300.0),
                    self._request(target_worker_id, "nccl_recv", transfer_id=token.token_id, source_rank=self._nccl_ranks[source_worker_id], timeout=300.0),
                )
                await self._request(source_worker_id, "nccl_commit_source", session_id=pipeline_session_id, timeout=300.0)
                ownership = self._ownership.commit_migration(token)
            except Exception:
                with contextlib.suppress(Exception):
                    await self._request(target_worker_id, "nccl_discard", transfer_id=token.token_id, session_id=pipeline_session_id)
                with contextlib.suppress(Exception):
                    await self._request(source_worker_id, "nccl_abort_source", session_id=pipeline_session_id, transfer_id=token.token_id)
                self._ownership.abort_migration(token)
                for chunk in self._migrating_controls.pop(pipeline_session_id, []):
                    self._send(source_worker_id, {"type": "model_push", "session_id": pipeline_session_id, "chunk": chunk})
                raise
                await asyncio.gather(
                    self._request(source_worker_id, "scheduler_resume"),
                    self._request(target_worker_id, "scheduler_resume"),
                    return_exceptions=True,
                )
            self._pipeline_routes[pipeline_session_id] = target_worker_id
            self._session_workers[pipeline_session_id] = target_worker_id
            for chunk in self._migrating_controls.pop(pipeline_session_id, []):
                self._send(target_worker_id, {"type": "model_push", "session_id": pipeline_session_id, "chunk": chunk})
            self._migration_total_ms.append((time.monotonic() - started) * 1000.0)
            await asyncio.gather(
                self._request(source_worker_id, "scheduler_resume"),
                self._request(target_worker_id, "scheduler_resume"),
                return_exceptions=True,
            )
            return ownership

    def turboserve_snapshot(self) -> dict[str, object]:
        snapshot = super().turboserve_snapshot()
        snapshot.update({
            "migration_supported": bool(self._nccl_ranks),
            "migration_backend": "process_nccl" if self._nccl_ranks else None,
            "nccl_ranks": dict(self._nccl_ranks),
            "worker_runtime_metrics": {worker_id: dict(self._worker_runtime_metrics.get(worker_id, {})) for worker_id in self._specs},
            "session_runtime_metrics": dict(self._session_runtime_metrics),
            "migration_calibration": {"average_total_ms": sum(self._migration_total_ms) / len(self._migration_total_ms) if self._migration_total_ms else 0.0},
        })
        return snapshot

    async def aclose(self) -> None:
        for session_id in tuple(self._transport_workers):
            with contextlib.suppress(Exception):
                await self.stop_session(session_id)
        if self._nccl_ranks:
            await asyncio.gather(*(self._request(worker_id, "nccl_destroy") for worker_id in self._nccl_ranks), return_exceptions=True)
            self._nccl_ranks.clear()
        await super().aclose()

    async def _init_nccl(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        workers = sorted(self._active_workers)
        await asyncio.gather(*(self._request(worker_id, "nccl_init", rank=rank, world_size=len(workers), init_method=f"tcp://127.0.0.1:{port}") for rank, worker_id in enumerate(workers)))
        self._nccl_ranks = {worker_id: rank for rank, worker_id in enumerate(workers)}

    def _transport_finished(self, session_id: str) -> None:
        self.close_model_session(session_id)

    def _transport_task_done(self, session_id: str, task: asyncio.Task[None]) -> None:
        self._transport_tasks.pop(session_id, None)
        self._transport_workers.pop(session_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.warning("LiveKit transport failed: session=%s error=%s", session_id, task.exception())

    def _dispatch_event(self, event: dict[str, Any]) -> None:
        if event.get("type") == "model_output":
            metrics = event.get("runtime_metrics")
            if isinstance(metrics, dict):
                self._worker_runtime_metrics[event["worker_id"]] = {key: value for key, value in metrics.items() if isinstance(value, int | float)}
            session_metrics = event.get("session_runtime_metrics")
            if isinstance(session_metrics, dict):
                self._session_runtime_metrics[event["session_id"]] = {key: value for key, value in session_metrics.items() if isinstance(value, int | float)}
            if (output := self._model_outputs.get(event["session_id"])) is not None:
                output.put_nowait(event["payload"])
            return
        super()._dispatch_event(event)


def _nccl_model_worker_main(spec: ProcessWorkerSpec, config_values: dict[str, Any], pipeline_file: str, skip_validation: bool, security_name: str | None, commands: Any, events: Any) -> None:
    try:
        asyncio.run(_run_nccl_model_worker(spec, pipeline_file, skip_validation, security_name, commands, events))
    finally:
        _close_queue(commands, join=False)
        _close_queue(events)


async def _run_nccl_model_worker(spec: ProcessWorkerSpec, pipeline_file: str, skip_validation: bool, security_name: str | None, commands: Any, events: Any) -> None:
    if not spec.gpu_ids:
        raise RuntimeError("process-nccl requires one CUDA GPU per worker")
    torch.cuda.set_device(int(spec.gpu_ids[0]))
    adapter = LiveKitPipelineAdapter(security_level=SecurityLevel[security_name] if security_name else None)
    adapter.start(pipeline_file, skip_validation=skip_validation, gpu_num=1, gpu_ids=spec.gpu_ids)
    if adapter.stream_mode != STREAM_MODE_BIDIRECTIONAL:
        raise RuntimeError("process-nccl requires a bidirectional pipeline")
    profile = adapter.configure_session_capacity(None)
    events.put({"type": "worker_capacity", "worker_id": spec.worker_id, "capacity": int((profile or {}).get("effective_capacity", 1)), "profile": profile})
    events.put({"type": "worker_status", "worker_id": spec.worker_id, "status": "idle"})
    events.put({"type": "worker_ready", "worker_id": spec.worker_id})
    service = adapter.stream_service.service
    outputs: dict[str, asyncio.Task[None]] = {}
    outgoing: dict[str, dict[tuple[Any, ...], torch.Tensor]] = {}
    incoming: dict[str, tuple[dict[str, Any], dict[tuple[Any, ...], torch.Tensor], str, int]] = {}

    async def pump(session_id: str) -> None:
        async for payload in adapter.pull_chunks(session_id):
            events.put({"type": "model_output", "worker_id": spec.worker_id, "session_id": session_id, "payload": payload, "runtime_metrics": adapter.runtime_metrics() or {}, "session_runtime_metrics": service.runtime_metrics(session_id)})

    async def result(request_id: str | None, value: Any = True, error: Exception | None = None) -> None:
        if request_id is not None:
            events.put({"type": "command_result", "worker_id": spec.worker_id, "request_id": request_id, "result": value, "error": repr(error) if error else None})

    try:
        while True:
            command = await asyncio.to_thread(commands.get)
            request_id, kind = command.get("request_id"), command["type"]
            try:
                if kind == "model_create":
                    session_id = adapter.create_session(command["config"])
                    outputs[session_id] = asyncio.create_task(pump(session_id))
                elif kind == "model_push":
                    adapter.push_chunk(command["session_id"], command["chunk"])
                elif kind == "model_close":
                    adapter.close_session(command["session_id"])
                    if (task := outputs.pop(command["session_id"], None)):
                        task.cancel()
                elif kind == "nccl_init":
                    await asyncio.to_thread(dist.init_process_group, "nccl", init_method=command["init_method"], rank=command["rank"], world_size=command["world_size"])
                elif kind == "scheduler_pause":
                    await asyncio.to_thread(service.pause_scheduler)
                elif kind == "scheduler_resume":
                    service.resume_scheduler()
                elif kind == "nccl_export":
                    metadata = service.prepare_migration_nccl_metadata(command["session_id"])
                    outgoing[command["transfer_id"]] = metadata.pop("_nccl_tensor_leaves")
                    await result(request_id, metadata)
                    continue
                elif kind == "nccl_prepare_recv":
                    metadata = command["metadata"]
                    leaves = allocate_tensor_tree_leaves(metadata["tensor_manifest"], torch.device(f"cuda:{spec.gpu_ids[0]}"))
                    incoming[command["transfer_id"]] = (metadata, leaves, command["owner_worker_id"], command["ownership_epoch"])
                elif kind == "nccl_send":
                    transfer_tensor_leaves_nccl(outgoing.pop(command["transfer_id"]), peer_rank=command["target_rank"], send=True)
                elif kind == "nccl_recv":
                    metadata, leaves, owner, epoch = incoming.pop(command["transfer_id"])
                    transfer_tensor_leaves_nccl(leaves, peer_rank=command["source_rank"], send=False)
                    session_id = service.import_migration_nccl(metadata, leaves, owner_worker_id=owner, ownership_epoch=epoch)
                    outputs[session_id] = asyncio.create_task(pump(session_id))
                elif kind == "nccl_commit_source":
                    service.commit_migration(command["session_id"])
                    if (task := outputs.pop(command["session_id"], None)):
                        task.cancel()
                elif kind == "nccl_abort_source":
                    service.abort_migration(command["session_id"])
                    outgoing.pop(command.get("transfer_id", ""), None)
                elif kind == "nccl_discard":
                    incoming.pop(command["transfer_id"], None)
                    if service.has_session(command["session_id"]):
                        service.close_session(command["session_id"])
                elif kind == "nccl_destroy":
                    if dist.is_initialized():
                        dist.destroy_process_group()
                elif kind == "shutdown":
                    break
                else:
                    raise ValueError(f"Unknown process-nccl command {kind!r}")
                await result(request_id)
            except Exception as exc:
                await result(request_id, error=exc)
    finally:
        for task in outputs.values():
            task.cancel()
        await asyncio.gather(*outputs.values(), return_exceptions=True)
        if dist.is_initialized():
            with contextlib.suppress(Exception):
                dist.destroy_process_group()
        await adapter.aclose()
