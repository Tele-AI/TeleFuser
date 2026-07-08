"""File service for downloading and managing media files."""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from telefuser.service_types import MediaType
from telefuser.utils.logging import logger

from .artifact_store import ArtifactStore


class FileService:
    """Service for downloading and caching media files from URLs."""

    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(
        self,
        cache_dir: Path,
        max_file_size: int | None = None,
        *,
        verify_ssl: bool = True,
        ssl_cert_path: str | None = None,
        artifact_retention_seconds: int = 7 * 24 * 60 * 60,
        artifact_tmp_retention_seconds: int = 60 * 60,
        artifact_persistence_mode: str = "persistent",
        artifact_preserve_failed_outputs: bool = False,
        artifact_max_total_bytes: int = 0,
        artifact_max_task_bytes: int = 0,
    ) -> None:
        self.artifact_store = ArtifactStore(cache_dir)
        self.cache_dir = self.artifact_store.root
        self.input_image_dir = self.artifact_store.legacy_input_image_dir
        self.input_video_dir = self.artifact_store.legacy_input_video_dir
        self.input_audio_dir = self.artifact_store.legacy_input_audio_dir
        self.output_dir = self.artifact_store.legacy_output_dir
        self.output_video_dir = self.artifact_store.legacy_output_video_dir
        self.output_image_dir = self.artifact_store.legacy_output_image_dir

        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE
        self.verify_ssl = verify_ssl
        self.ssl_cert_path = ssl_cert_path
        self.artifact_retention_seconds = artifact_retention_seconds
        self.artifact_tmp_retention_seconds = artifact_tmp_retention_seconds
        self.artifact_persistence_mode = artifact_persistence_mode
        self.artifact_preserve_failed_outputs = artifact_preserve_failed_outputs
        self.artifact_max_total_bytes = artifact_max_total_bytes
        self.artifact_max_task_bytes = artifact_max_task_bytes

        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_retry_delay = 10.0

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling."""
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
                limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0)
                verify = self.ssl_cert_path if self.ssl_cert_path else self.verify_ssl
                self._http_client = httpx.AsyncClient(
                    verify=verify, timeout=timeout, limits=limits, follow_redirects=True
                )
            return self._http_client

    async def _download_file_with_retry(
        self,
        url: str,
        destination: Path,
        max_size: int,
        max_retries: int | None = None,
    ) -> None:
        """Download a remote file with retries and bounded streaming writes."""
        if max_retries is None:
            max_retries = self.max_retries

        last_exception = None
        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                client = await self._get_http_client()
                async with client.stream("GET", url) as response:
                    if response.status_code == 200:
                        content_length = response.headers.get("content-length")
                        if content_length is not None and int(content_length) > max_size:
                            raise ValueError(
                                f"File too large: {content_length} bytes, max: {max_size} bytes"
                            )
                        await self._write_response_stream(response, destination, max_size)
                        return
                    if response.status_code >= 500:
                        logger.warning(
                            f"Server error {response.status_code} for {url}, attempt {attempt + 1}/{max_retries}"
                        )
                        last_exception = httpx.HTTPStatusError(
                            f"Server returned {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    else:
                        raise httpx.HTTPStatusError(
                            f"Client error {response.status_code}",
                            request=response.request,
                            response=response,
                        )

            except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                logger.warning(f"Connection error for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                last_exception = e
            except httpx.HTTPStatusError as e:
                if e.response and e.response.status_code < 500:
                    raise
                last_exception = e
            except Exception as e:
                logger.error(f"Unexpected error downloading {url}: {str(e)}")
                last_exception = e

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self.max_retry_delay)

        error_msg = f"All {max_retries} connection attempts failed for {url}"
        if last_exception:
            error_msg += f": {str(last_exception)}"
        raise httpx.ConnectError(error_msg)

    async def _write_response_stream(self, response: httpx.Response, destination: Path, max_size: int) -> None:
        """Write response bytes through a .part file and atomically publish on success."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        part_path = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
        bytes_written = 0
        try:
            with open(part_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                    bytes_written += len(chunk)
                    if bytes_written > max_size:
                        raise ValueError(f"File too large: {bytes_written} bytes, max: {max_size} bytes")
                    f.write(chunk)
            os.replace(part_path, destination)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _safe_basename(raw_name: str, fallback: str) -> str:
        name = Path(raw_name).name
        return name or fallback

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        return ArtifactStore._is_relative_to(path, root)

    def _resolve_under(self, root: Path, relative_path: str | Path) -> Path:
        return self.artifact_store._resolve_under(root, relative_path)

    async def download_image(self, image_url: str, max_size: int | None = None) -> Path:
        """Download image with retry logic and proper error handling."""
        max_size = max_size or self.max_file_size

        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {image_url}")

            image_name = self._safe_basename(parsed_url.path, f"{uuid.uuid4()}.jpg")
            image_path = self._resolve_under(self.input_image_dir, image_name)
            await self._download_file_with_retry(image_url, image_path, max_size)

            logger.info(f"Successfully downloaded image from {image_url} to {image_path}")
            return image_path

        except httpx.ConnectError as e:
            logger.error(f"Connection error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Failed to connect to {image_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Download timeout for {image_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"HTTP error for {image_url}: {str(e)}")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading image from {image_url}: {str(e)}")
            raise ValueError(f"Failed to download image from {image_url}: {str(e)}")

    async def download_audio(self, audio_url: str, max_size: int | None = None) -> Path:
        """Download audio with retry logic and proper error handling."""
        max_size = max_size or self.max_file_size

        try:
            parsed_url = urlparse(audio_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {audio_url}")

            audio_name = self._safe_basename(parsed_url.path, f"{uuid.uuid4()}.mp3")
            audio_path = self._resolve_under(self.input_audio_dir, audio_name)
            await self._download_file_with_retry(audio_url, audio_path, max_size)

            logger.info(f"Successfully downloaded audio from {audio_url} to {audio_path}")
            return audio_path

        except httpx.ConnectError as e:
            logger.error(f"Connection error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Failed to connect to {audio_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Download timeout for {audio_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"HTTP error for {audio_url}: {str(e)}")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading audio from {audio_url}: {str(e)}")
            raise ValueError(f"Failed to download audio from {audio_url}: {str(e)}")

    async def download_video(self, video_url: str, max_size: int | None = None) -> Path:
        """Download video with retry logic and proper error handling."""
        max_size = max_size or self.max_file_size

        try:
            parsed_url = urlparse(video_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {video_url}")

            video_name = self._safe_basename(parsed_url.path, f"{uuid.uuid4()}.mp4")
            video_path = self._resolve_under(self.input_video_dir, video_name)
            await self._download_file_with_retry(video_url, video_path, max_size)

            logger.info(f"Successfully downloaded video from {video_url} to {video_path}")
            return video_path

        except httpx.ConnectError as e:
            logger.error(f"Connection error downloading video from {video_url}: {str(e)}")
            raise ValueError(f"Failed to connect to {video_url}: {str(e)}")
        except httpx.TimeoutException as e:
            logger.error(f"Timeout downloading video from {video_url}: {str(e)}")
            raise ValueError(f"Download timeout for {video_url}: {str(e)}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading video from {video_url}: {str(e)}")
            raise ValueError(f"HTTP error for {video_url}: {str(e)}")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error downloading video from {video_url}: {str(e)}")
            raise ValueError(f"Failed to download video from {video_url}: {str(e)}")

    def save_uploaded_file(self, file_content: bytes, filename: str) -> Path:
        """Save uploaded file content to disk."""
        if len(file_content) > self.max_file_size:
            raise ValueError(f"File too large: {len(file_content)} bytes, max: {self.max_file_size} bytes")
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self._resolve_under(self.input_image_dir, unique_filename)

        part_path = file_path.with_name(f"{file_path.name}.{uuid.uuid4().hex}.part")
        try:
            with open(part_path, "wb") as f:
                f.write(file_content)
            os.replace(part_path, file_path)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

        return file_path

    async def save_upload_file(
        self,
        file: Any,
        *,
        media_type: MediaType | str,
        prefix: str = "upload",
        fallback_filename: str = "upload.bin",
    ) -> Path:
        """Save an async upload stream through a bounded .part file write."""
        original_name = self._safe_basename(getattr(file, "filename", None) or fallback_filename, fallback_filename)
        suffix = Path(original_name).suffix
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        file_path = self._resolve_under(self._input_dir_for_media(media_type), filename)
        await self._write_upload_stream(file, file_path, self.max_file_size)
        return file_path

    def _input_dir_for_media(self, media_type: MediaType | str) -> Path:
        media_value = str(media_type)
        if media_value == MediaType.IMAGE.value:
            return self.input_image_dir
        if media_value == MediaType.VIDEO.value:
            return self.input_video_dir
        if media_value == "audio":
            return self.input_audio_dir
        raise ValueError(f"Unsupported media type: {media_type}")

    async def _write_upload_stream(self, file: Any, destination: Path, max_size: int) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        part_path = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
        bytes_written = 0
        try:
            with open(part_path, "wb") as buffer:
                while True:
                    chunk = await self._read_upload_chunk(file)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > max_size:
                        raise ValueError(f"File too large: {bytes_written} bytes, max: {max_size} bytes")
                    buffer.write(chunk)
            os.replace(part_path, destination)
        except Exception:
            part_path.unlink(missing_ok=True)
            raise

    @staticmethod
    async def _read_upload_chunk(file: Any) -> bytes:
        reader = getattr(file, "read")
        chunk = reader(1024 * 1024)
        if inspect.isawaitable(chunk):
            chunk = await chunk
        return chunk

    def get_output_path(
        self,
        output_path: str,
        media_type: MediaType | str = MediaType.VIDEO,
        *,
        task_id: str | None = None,
    ) -> Path:
        """Get the full output path for a file."""
        return self.artifact_store.output_path(output_path, media_type=media_type, task_id=task_id)

    def resolve_output_file(self, file_path: str | Path) -> Path:
        """Resolve a downloadable output file within the configured cache root."""
        return self.artifact_store.resolve_output_file(file_path)

    def artifact_id_for_path(self, file_path: str | Path) -> str:
        """Return a stable local artifact id for a file under the artifact root."""
        return self.artifact_store.artifact_id_for_path(file_path)

    def resolve_artifact_id(self, artifact_id: str) -> Path:
        """Resolve a local artifact id to an output file path."""
        return self.artifact_store.resolve_artifact_id(artifact_id)

    def artifact_metadata(
        self,
        file_path: str | Path,
        *,
        task_id: str | None = None,
        media_type: MediaType | str | None = None,
    ) -> dict[str, Any]:
        """Return local artifact metadata for an output file."""
        return self.artifact_store.artifact_metadata(file_path, task_id=task_id, media_type=media_type)

    def cleanup_artifacts(
        self,
        *,
        active_task_ids: set[str] | frozenset[str],
        terminal_task_end_times: Mapping[str, datetime | float | int | None],
        terminal_task_statuses: Mapping[str, str] | None = None,
        now: datetime | float | int | None = None,
    ) -> dict[str, Any]:
        """Clean expired local artifacts according to configured retention settings."""
        return self.artifact_store.cleanup(
            active_task_ids=active_task_ids,
            terminal_task_end_times=terminal_task_end_times,
            terminal_task_statuses=terminal_task_statuses,
            retention_seconds=self.artifact_retention_seconds,
            tmp_retention_seconds=self.artifact_tmp_retention_seconds,
            persistence_mode=self.artifact_persistence_mode,
            preserve_failed_outputs=self.artifact_preserve_failed_outputs,
            max_total_bytes=self.artifact_max_total_bytes,
            max_task_bytes=self.artifact_max_task_bytes,
            now=now,
        )

    async def cleanup(self) -> None:
        """Cleanup resources including HTTP client."""
        async with self._client_lock:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
                self._http_client = None
