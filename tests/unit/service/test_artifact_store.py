from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from telefuser.service.api.api_server import ApiServer
from telefuser.service.core.config import ServerConfig
from telefuser.service.core.artifact_store import ArtifactStore
from telefuser.service.core.file_service import FileService
from telefuser.service.core.task_manager import TaskManager
from telefuser.service_types import MediaType


class _AsyncUpload:
    def __init__(self, filename: str, chunks: list[bytes]) -> None:
        self.filename = filename
        self._chunks = list(chunks)

    async def read(self, size: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_artifact_store_creates_task_scoped_output_paths(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    output = store.output_path("result.png", media_type=MediaType.IMAGE, task_id="task-123")

    assert output == tmp_path / "tasks" / "task-123" / "outputs" / "images" / "result.png"
    assert output.parent.exists()


def test_artifact_store_rejects_unsafe_task_output_paths(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="Absolute paths"):
        store.output_path("/tmp/escape.png", media_type=MediaType.IMAGE, task_id="task-123")

    with pytest.raises(ValueError, match="escapes"):
        store.output_path("../escape.png", media_type=MediaType.IMAGE, task_id="task-123")

    with pytest.raises(ValueError, match="Invalid task_id"):
        store.output_path("result.png", media_type=MediaType.IMAGE, task_id="../bad")


def test_artifact_store_resolves_task_scoped_outputs_for_download(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    output = store.output_path("result.mp4", media_type=MediaType.VIDEO, task_id="task-123")
    output.write_bytes(b"video")

    assert store.resolve_output_file("result.mp4") == output
    assert store.resolve_output_file(output) == output


def test_file_service_uses_task_scoped_outputs_when_task_id_is_provided(tmp_path: Path) -> None:
    files = FileService(tmp_path)

    output = files.get_output_path("result.png", media_type=MediaType.IMAGE, task_id="task-123")

    assert output == tmp_path / "tasks" / "task-123" / "outputs" / "images" / "result.png"


def test_file_service_saves_upload_stream_to_media_input_dir(tmp_path: Path) -> None:
    files = FileService(tmp_path)
    upload = _AsyncUpload("cat.png", [b"ca", b"t"])

    output = asyncio.run(files.save_upload_file(upload, media_type=MediaType.IMAGE, prefix="input"))

    assert output.parent == files.input_image_dir
    assert output.name.startswith("input_")
    assert output.suffix == ".png"
    assert output.read_bytes() == b"cat"
    assert not list(output.parent.glob("*.part"))


def test_file_service_upload_stream_enforces_max_file_size(tmp_path: Path) -> None:
    files = FileService(tmp_path, max_file_size=2)
    upload = _AsyncUpload("cat.png", [b"cat"])

    with pytest.raises(ValueError, match="File too large"):
        asyncio.run(files.save_upload_file(upload, media_type=MediaType.IMAGE, prefix="input"))

    assert not list(files.input_image_dir.glob("*.part"))


def test_artifact_store_cleanup_removes_only_expired_terminal_tasks(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    old_output = store.output_path("old.mp4", media_type=MediaType.VIDEO, task_id="old-task")
    active_output = store.output_path("active.mp4", media_type=MediaType.VIDEO, task_id="active-task")
    fresh_output = store.output_path("fresh.mp4", media_type=MediaType.VIDEO, task_id="fresh-task")
    old_output.write_bytes(b"old")
    active_output.write_bytes(b"active")
    fresh_output.write_bytes(b"fresh")
    now = datetime.now()

    result = store.cleanup(
        active_task_ids={"active-task"},
        terminal_task_end_times={
            "old-task": now - timedelta(seconds=120),
            "active-task": now - timedelta(seconds=120),
            "fresh-task": now,
        },
        retention_seconds=60,
        tmp_retention_seconds=0,
        now=now,
    )

    assert result["removed_task_ids"] == ["old-task"]
    assert not old_output.exists()
    assert active_output.exists()
    assert fresh_output.exists()


def test_artifact_store_cleanup_removes_expired_part_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    part_file = store.task_tmp_dir("task-123") / "upload.part"
    part_file.write_bytes(b"partial")
    old_ts = (datetime.now() - timedelta(seconds=120)).timestamp()
    part_file.touch()
    import os

    os.utime(part_file, (old_ts, old_ts))

    result = store.cleanup(
        active_task_ids={"task-123"},
        terminal_task_end_times={},
        retention_seconds=0,
        tmp_retention_seconds=60,
    )

    assert result["removed_tmp_files"] == 1
    assert not part_file.exists()


def test_artifact_store_cleanup_removes_oversized_terminal_tasks(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    oversized_output = store.output_path("big.mp4", media_type=MediaType.VIDEO, task_id="oversized-task")
    active_output = store.output_path("active.mp4", media_type=MediaType.VIDEO, task_id="active-task")
    small_output = store.output_path("small.mp4", media_type=MediaType.VIDEO, task_id="small-task")
    oversized_output.write_bytes(b"12345")
    active_output.write_bytes(b"12345")
    small_output.write_bytes(b"12")
    now = datetime.now()

    result = store.cleanup(
        active_task_ids={"active-task"},
        terminal_task_end_times={
            "oversized-task": now,
            "active-task": now,
            "small-task": now,
        },
        retention_seconds=0,
        tmp_retention_seconds=0,
        max_task_bytes=4,
        now=now,
    )

    assert result["removed_task_ids"] == ["oversized-task"]
    assert not oversized_output.exists()
    assert active_output.exists()
    assert small_output.exists()


def test_file_service_passes_max_task_bytes_to_artifact_cleanup(tmp_path: Path) -> None:
    files = FileService(tmp_path, artifact_max_task_bytes=3)
    output = files.get_output_path("big.mp4", media_type=MediaType.VIDEO, task_id="task-123")
    output.write_bytes(b"1234")

    result = files.cleanup_artifacts(
        active_task_ids=set(),
        terminal_task_end_times={"task-123": datetime.now()},
    )

    assert result["removed_task_ids"] == ["task-123"]
    assert not output.exists()


def test_api_server_runs_artifact_cleanup_from_task_snapshot(tmp_path: Path) -> None:
    task_manager = TaskManager()
    config = ServerConfig(artifact_retention_seconds=1, artifact_tmp_retention_seconds=0)
    server = ApiServer(task_manager=task_manager, config=config, enable_openai_api=False)
    server.initialize_services(tmp_path, Mock())

    task_id = task_manager.create_task(SimpleNamespace(output_path="old.mp4"))
    output = server.file_service.get_output_path("old.mp4", media_type=MediaType.VIDEO, task_id=task_id)
    output.write_bytes(b"old")
    task_manager.complete_task(task_id)
    task = task_manager.get_task(task_id)

    result = server.run_artifact_cleanup(now=task.end_time + timedelta(seconds=2))

    assert result["removed_task_ids"] == [task_id]
    assert not output.exists()


def test_api_server_artifact_cleanup_loop_starts_and_stops(tmp_path: Path) -> None:
    async def _exercise() -> None:
        server = ApiServer(task_manager=TaskManager(), config=ServerConfig(), enable_openai_api=False)
        server.file_service = FileService(tmp_path)

        await server.ensure_artifact_cleanup_running()

        assert server._artifact_cleanup_task is not None
        assert not server._artifact_cleanup_task.done()

        await server.cleanup()

        assert server._artifact_cleanup_task is None

    asyncio.run(_exercise())
