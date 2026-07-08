"""Local artifact storage for service inputs and outputs."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from telefuser.service_types import MediaType
from telefuser.utils.logging import logger

LOCAL_ARTIFACT_ID_PREFIX = "local:"


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

    @staticmethod
    def _utc_timestamp(value: float) -> str:
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")

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

        root = (
            self.legacy_output_image_dir
            if self._media_dir_name(media_type) == "images"
            else self.legacy_output_video_dir
        )
        return self._resolve_under(root, output_path)

    def resolve_output_file(self, file_path: str | Path) -> Path:
        """Resolve a downloadable output artifact under legacy or task output roots."""
        raw_path = str(file_path)
        if raw_path.startswith(LOCAL_ARTIFACT_ID_PREFIX):
            return self.resolve_artifact_id(raw_path)

        path = Path(file_path)
        if path.is_absolute():
            resolved_path = path.resolve()
            if self._is_allowed_output_path(resolved_path):
                return resolved_path
            raise ValueError("Access to this file is not allowed")

        root_relative_candidate = self._resolve_under(self.root, path)
        if root_relative_candidate.exists() and self._is_allowed_output_path(root_relative_candidate.resolve()):
            return root_relative_candidate

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

    def artifact_id_for_path(self, file_path: str | Path) -> str:
        """Return a stable local artifact id for a path under the artifact root."""
        raw_path = str(file_path)
        if raw_path.startswith(LOCAL_ARTIFACT_ID_PREFIX):
            resolved_path = self.resolve_artifact_id(raw_path)
        else:
            resolved_path = Path(file_path).resolve()

        if not self._is_relative_to(resolved_path, self.root):
            raise ValueError("Artifact path is outside the service cache directory")
        if not self._is_allowed_output_path(resolved_path):
            raise ValueError("Artifact path is not a downloadable output")
        return f"{LOCAL_ARTIFACT_ID_PREFIX}{resolved_path.relative_to(self.root).as_posix()}"

    def resolve_artifact_id(self, artifact_id: str) -> Path:
        """Resolve a local artifact id back to an allowed output artifact path."""
        if not artifact_id.startswith(LOCAL_ARTIFACT_ID_PREFIX):
            raise ValueError("Unsupported artifact backend")

        relative_path = artifact_id[len(LOCAL_ARTIFACT_ID_PREFIX) :]
        if not relative_path:
            raise ValueError("Artifact id is empty")

        resolved_path = self._resolve_under(self.root, relative_path)
        if not self._is_allowed_output_path(resolved_path):
            raise ValueError("Access to this artifact is not allowed")
        return resolved_path

    def artifact_metadata(
        self,
        file_path: str | Path,
        *,
        task_id: str | None = None,
        media_type: MediaType | str | None = None,
    ) -> dict[str, Any]:
        """Return local artifact metadata without exposing remote-storage semantics."""
        resolved_path = self.resolve_output_file(file_path)
        stat = resolved_path.stat() if resolved_path.exists() else None
        media_value = media_type.value if isinstance(media_type, MediaType) else media_type

        return {
            "artifact_id": self.artifact_id_for_path(resolved_path),
            "backend": "local",
            "relative_path": resolved_path.relative_to(self.root).as_posix(),
            "task_id": task_id,
            "media_type": media_value,
            "filename": resolved_path.name,
            "size_bytes": stat.st_size if stat is not None else None,
            "created_at": self._utc_timestamp(stat.st_ctime) if stat is not None else None,
            "modified_at": self._utc_timestamp(stat.st_mtime) if stat is not None else None,
        }

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
        terminal_task_statuses: Mapping[str, str] | None = None,
        retention_seconds: int,
        tmp_retention_seconds: int,
        persistence_mode: str = "persistent",
        preserve_failed_outputs: bool = False,
        max_total_bytes: int = 0,
        max_task_bytes: int = 0,
        now: datetime | float | int | None = None,
    ) -> dict[str, Any]:
        """Clean expired local artifacts without touching active task directories."""
        now_ts = _to_timestamp(now) or datetime.now().timestamp()
        removed_task_ids: list[str] = []
        removed_tmp_files = 0
        errors: list[str] = []
        protected_task_ids = self._protected_terminal_task_ids(
            active_task_ids=active_task_ids,
            terminal_task_statuses=terminal_task_statuses,
            preserve_failed_outputs=preserve_failed_outputs,
        )

        removed_tmp_files += self._cleanup_tmp_files(tmp_retention_seconds=tmp_retention_seconds, now_ts=now_ts)

        if persistence_mode not in {"persistent", "ephemeral"}:
            raise ValueError(f"Unsupported artifact persistence mode: {persistence_mode}")

        if persistence_mode == "ephemeral":
            removed_task_ids.extend(
                self._cleanup_terminal_tasks(
                    active_task_ids=active_task_ids,
                    terminal_task_end_times=terminal_task_end_times,
                    protected_task_ids=protected_task_ids,
                )
            )
        elif retention_seconds > 0:
            for task_id, end_time in terminal_task_end_times.items():
                if task_id in active_task_ids or task_id in protected_task_ids:
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

        if max_task_bytes > 0:
            removed_task_ids.extend(
                self._cleanup_oversized_tasks(
                    active_task_ids=active_task_ids,
                    terminal_task_end_times=terminal_task_end_times,
                    max_task_bytes=max_task_bytes,
                    already_removed=set(removed_task_ids),
                    protected_task_ids=protected_task_ids,
                )
            )

        if max_total_bytes > 0:
            removed_task_ids.extend(
                self._cleanup_capacity(
                    active_task_ids=active_task_ids,
                    terminal_task_end_times=terminal_task_end_times,
                    max_total_bytes=max_total_bytes,
                    protected_task_ids=protected_task_ids,
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

    @staticmethod
    def _protected_terminal_task_ids(
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_statuses: Mapping[str, str] | None,
        preserve_failed_outputs: bool,
    ) -> set[str]:
        if not preserve_failed_outputs or not terminal_task_statuses:
            return set()
        return {
            task_id
            for task_id, status in terminal_task_statuses.items()
            if task_id not in active_task_ids and str(status).lower() == "failed"
        }

    def _cleanup_terminal_tasks(
        self,
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_end_times: Mapping[str, datetime | float | int | None],
        protected_task_ids: set[str],
    ) -> list[str]:
        removed: list[str] = []
        for task_id in terminal_task_end_times:
            if task_id in active_task_ids or task_id in protected_task_ids:
                continue
            task_dir = self.task_root(task_id)
            if not task_dir.exists():
                continue
            try:
                shutil.rmtree(task_dir)
                removed.append(task_id)
            except Exception as exc:
                logger.warning(f"Failed to remove artifact directory for task {task_id}: {exc}")
        return removed

    def _cleanup_capacity(
        self,
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_end_times: Mapping[str, datetime | float | int | None],
        max_total_bytes: int,
        protected_task_ids: set[str],
    ) -> list[str]:
        current_size = self._tree_size(self.root)
        if current_size <= max_total_bytes:
            return []

        candidates = [
            (task_id, _to_timestamp(end_time) or 0.0, self.task_root(task_id))
            for task_id, end_time in terminal_task_end_times.items()
            if task_id not in active_task_ids and task_id not in protected_task_ids and self.task_root(task_id).exists()
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

    def _cleanup_oversized_tasks(
        self,
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_end_times: Mapping[str, datetime | float | int | None],
        max_task_bytes: int,
        already_removed: set[str],
        protected_task_ids: set[str],
    ) -> list[str]:
        removed: list[str] = []
        for task_id in terminal_task_end_times:
            if task_id in active_task_ids or task_id in already_removed or task_id in protected_task_ids:
                continue

            task_dir = self.task_root(task_id)
            if not task_dir.exists() or self._tree_size(task_dir) <= max_task_bytes:
                continue
            try:
                shutil.rmtree(task_dir)
                removed.append(task_id)
            except Exception as exc:
                logger.warning(f"Failed to remove oversized artifact directory for task {task_id}: {exc}")
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
