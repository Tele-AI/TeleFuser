"""File service for downloading and managing media files."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from telefuser.service_types import MediaType
from telefuser.utils.logging import logger


class FileService:
    """Service for downloading and caching media files from URLs."""

    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(self, cache_dir: Path, max_file_size: int | None = None) -> None:
        self.cache_dir = cache_dir
        self.input_image_dir = cache_dir / "inputs" / "imgs"
        self.input_video_dir = cache_dir / "inputs" / "videos"
        self.input_audio_dir = cache_dir / "inputs" / "audios"
        self.output_dir = cache_dir / "outputs"
        self.output_video_dir = self.output_dir / "videos"
        self.output_image_dir = self.output_dir / "images"

        self.max_file_size = max_file_size or self.DEFAULT_MAX_FILE_SIZE

        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

        self.max_retries = 3
        self.retry_delay = 1.0
        self.max_retry_delay = 10.0

        for directory in [
            self.input_image_dir,
            self.output_dir,
            self.output_video_dir,
            self.output_image_dir,
            self.input_audio_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling."""
        from .config import server_config

        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)
                limits = httpx.Limits(max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0)
                verify = server_config.ssl_cert_path if server_config.ssl_cert_path else server_config.verify_ssl
                self._http_client = httpx.AsyncClient(
                    verify=verify, timeout=timeout, limits=limits, follow_redirects=True
                )
            return self._http_client

    async def _download_with_retry(self, url: str, max_retries: int | None = None) -> httpx.Response:
        """Download with exponential backoff retry logic."""
        if max_retries is None:
            max_retries = self.max_retries

        last_exception = None
        retry_delay = self.retry_delay

        for attempt in range(max_retries):
            try:
                client = await self._get_http_client()
                response = await client.get(url)

                if response.status_code == 200:
                    return response
                elif response.status_code >= 500:
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

    async def download_image(self, image_url: str, max_size: int | None = None) -> Path:
        """Download image with retry logic and proper error handling."""
        max_size = max_size or self.max_file_size

        try:
            parsed_url = urlparse(image_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL format: {image_url}")

            response = await self._download_with_retry(image_url)

            content_length = len(response.content)
            if content_length > max_size:
                raise ValueError(f"Image too large: {content_length} bytes, max: {max_size} bytes")

            image_name = Path(parsed_url.path).name
            if not image_name:
                image_name = f"{uuid.uuid4()}.jpg"

            image_path = self.input_image_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)

            with open(image_path, "wb") as f:
                f.write(response.content)

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

            response = await self._download_with_retry(audio_url)

            content_length = len(response.content)
            if content_length > max_size:
                raise ValueError(f"Audio too large: {content_length} bytes, max: {max_size} bytes")

            audio_name = Path(parsed_url.path).name
            if not audio_name:
                audio_name = f"{uuid.uuid4()}.mp3"

            audio_path = self.input_audio_dir / audio_name
            audio_path.parent.mkdir(parents=True, exist_ok=True)

            with open(audio_path, "wb") as f:
                f.write(response.content)

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

            response = await self._download_with_retry(video_url)

            content_length = len(response.content)
            if content_length > max_size:
                raise ValueError(f"Video too large: {content_length} bytes, max: {max_size} bytes")

            video_name = Path(parsed_url.path).name
            if not video_name:
                video_name = f"{uuid.uuid4()}.mp4"

            video_path = self.input_video_dir / video_name
            video_path.parent.mkdir(parents=True, exist_ok=True)

            with open(video_path, "wb") as f:
                f.write(response.content)

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
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = self.input_image_dir / unique_filename

        with open(file_path, "wb") as f:
            f.write(file_content)

        return file_path

    def get_output_path(self, output_path: str, media_type: MediaType | str = MediaType.VIDEO) -> Path:
        """Get the full output path for a file."""
        path = Path(output_path)
        if path.is_absolute():
            return path

        if media_type == MediaType.IMAGE:
            return self.output_image_dir / output_path
        else:
            return self.output_video_dir / output_path

    async def cleanup(self) -> None:
        """Cleanup resources including HTTP client."""
        async with self._client_lock:
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
                self._http_client = None
