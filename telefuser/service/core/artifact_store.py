"""Local artifact storage for service inputs and outputs."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from telefuser.service_types import MediaType
from telefuser.utils.logging import logger


def _to_timestamp(value: datetime | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    return float(value)


class ArtifactStore:
    """Path-safe local artifact store.

    The store is intentionally local-only for now. It centralizes root checks
    and task-scoped output paths without claiming remote persistence semantics.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.legacy_input_image_dir = self.root / "inputs" / "imgs"
        self.legacy_input_video_dir = self.root / "inputs" / "videos"
        self.legacy_input_audio_dir = self.root / "inputs" / "audios"
        self.legacy_output_dir = self.root / "outputs"
        self.legacy_output_video_dir = self.legacy_output_dir / "videos"
        self.legacy_output_image_dir = self.legacy_output_dir / "images"
        self.tasks_dir = self.root / "tasks"

        for directory in (
            self.legacy_input_image_dir,
            self.legacy_input_video_dir,
            self.legacy_input_audio_dir,
            self.legacy_output_video_dir,
            self.legacy_output_image_dir,
            self.tasks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _media_dir_name(media_type: MediaType | str) -> str:
        return "images" if media_type == MediaType.IMAGE or str(media_type) == MediaType.IMAGE.value else "videos"

    def _resolve_under(self, root: Path, relative_path: str | Path) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise ValueError("Absolute paths are not allowed")

        resolved_root = root.resolve()
        resolved_path = (resolved_root / path).resolve()
        if not self._is_relative_to(resolved_path, resolved_root):
            raise ValueError("Path escapes the service cache directory")
        return resolved_path

    def _validate_task_id(self, task_id: str) -> str:
        if not task_id or any(char in task_id for char in ("/", "\\")) or task_id in {".", ".."}:
            raise ValueError("Invalid task_id for artifact path")
        return task_id

    def task_root(self, task_id: str) -> Path:
        return self._resolve_under(self.tasks_dir, self._validate_task_id(task_id))

    def task_input_dir(self, task_id: str, media_type: MediaType | str) -> Path:
        media_dir = self._media_dir_name(media_type)
        path = self.task_root(task_id) / "inputs" / media_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_output_dir(self, task_id: str, media_type: MediaType | str) -> Path:
        media_dir = self._media_dir_name(media_type)
        path = self.task_root(task_id) / "outputs" / media_dir
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_tmp_dir(self, task_id: str) -> Path:
        path = self.task_root(task_id) / "tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def output_path(
        self,
        output_path: str | Path,
        *,
        media_type: MediaType | str = MediaType.VIDEO,
        task_id: str | None = None,
    ) -> Path:
        if task_id:
            return self._resolve_under(self.task_output_dir(task_id, media_type), output_path)

        root = self.legacy_output_image_dir if self._media_dir_name(media_type) == "images" else self.legacy_output_video_dir
        return self._resolve_under(root, output_path)

    def resolve_output_file(self, file_path: str | Path) -> Path:
        """Resolve a downloadable output artifact under legacy or task output roots."""
        path = Path(file_path)
        if path.is_absolute():
            resolved_path = path.resolve()
            if self._is_allowed_output_path(resolved_path):
                return resolved_path
            raise ValueError("Access to this file is not allowed")

        for root in (self.legacy_output_video_dir, self.legacy_output_image_dir, self.legacy_output_dir):
            candidate = self._resolve_under(root, path)
            if candidate.exists():
                return candidate

        for task_root in self.tasks_dir.glob("*"):
            if not task_root.is_dir():
                continue
            for media_dir in ("videos", "images"):
                candidate = self._resolve_under(task_root / "outputs" / media_dir, path)
                if candidate.exists():
                    return candidate

        return self._resolve_under(self.legacy_output_dir, path)

    def _is_allowed_output_path(self, resolved_path: Path) -> bool:
        if self._is_relative_to(resolved_path, self.legacy_output_dir.resolve()):
            return True

        tasks_root = self.tasks_dir.resolve()
        if not self._is_relative_to(resolved_path, tasks_root):
            return False
        relative = resolved_path.relative_to(tasks_root)
        return len(relative.parts) >= 4 and relative.parts[1] == "outputs" and relative.parts[2] in {"images", "videos"}

    def cleanup(
        self,
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_end_times: Mapping[str, datetime | float | int | None],
        retention_seconds: int,
        tmp_retention_seconds: int,
        max_total_bytes: int = 0,
        now: datetime | float | int | None = None,
    ) -> dict[str, Any]:
        """Clean expired local artifacts without touching active task directories."""
        now_ts = _to_timestamp(now) or datetime.now().timestamp()
        removed_task_ids: list[str] = []
        removed_tmp_files = 0
        errors: list[str] = []

        removed_tmp_files += self._cleanup_tmp_files(tmp_retention_seconds=tmp_retention_seconds, now_ts=now_ts)

        if retention_seconds > 0:
            for task_id, end_time in terminal_task_end_times.items():
                if task_id in active_task_ids:
                    continue
                end_ts = _to_timestamp(end_time)
                if end_ts is None or now_ts - end_ts < retention_seconds:
                    continue
                task_dir = self.task_root(task_id)
                if not task_dir.exists():
                    continue
                try:
                    shutil.rmtree(task_dir)
                    removed_task_ids.append(task_id)
                except Exception as exc:
                    message = f"Failed to remove artifact directory for task {task_id}: {exc}"
                    logger.warning(message)
                    errors.append(message)

        if max_total_bytes > 0:
            removed_task_ids.extend(
                self._cleanup_capacity(
                    active_task_ids=active_task_ids,
                    terminal_task_end_times=terminal_task_end_times,
                    max_total_bytes=max_total_bytes,
                )
            )

        return {
            "removed_task_ids": removed_task_ids,
            "removed_tmp_files": removed_tmp_files,
            "errors": errors,
        }

    def _cleanup_tmp_files(self, *, tmp_retention_seconds: int, now_ts: float) -> int:
        if tmp_retention_seconds <= 0:
            return 0

        removed = 0
        for part_file in self.root.rglob("*.part"):
            try:
                if now_ts - part_file.stat().st_mtime < tmp_retention_seconds:
                    continue
                part_file.unlink()
                removed += 1
            except FileNotFoundError:
                continue
            except Exception as exc:
                logger.warning(f"Failed to remove temporary artifact {part_file}: {exc}")
        return removed

    def _cleanup_capacity(
        self,
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_end_times: Mapping[str, datetime | float | int | None],
        max_total_bytes: int,
    ) -> list[str]:
        current_size = self._tree_size(self.root)
        if current_size <= max_total_bytes:
            return []

        candidates = [
            (task_id, _to_timestamp(end_time) or 0.0, self.task_root(task_id))
            for task_id, end_time in terminal_task_end_times.items()
            if task_id not in active_task_ids and self.task_root(task_id).exists()
        ]
        candidates.sort(key=lambda item: item[1])

        removed: list[str] = []
        for task_id, _, task_dir in candidates:
            if current_size <= max_total_bytes:
                break
            task_size = self._tree_size(task_dir)
            try:
                shutil.rmtree(task_dir)
                current_size -= task_size
                removed.append(task_id)
            except Exception as exc:
                logger.warning(f"Failed to remove artifact directory for task {task_id}: {exc}")
        return removed

    @staticmethod
    def _tree_size(root: Path) -> int:
        if not root.exists():
            return 0

        total = 0
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total
