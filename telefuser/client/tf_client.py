"""TeleFuser Standalone API Client — single file, zero internal dependencies.

Drop this file into any project (or import via ``from telefuser.client import TFClient``).
The file has no dependency on the rest of ``telefuser``.

Only requires: requests (pip install requests)
Optional:     tqdm    (pip install tqdm, for progress bars in multi-server mode)

Quick start — Text-to-Video (T2V):
    from telefuser.client import TFClient, TASK_T2V

    client = TFClient("http://localhost:8000")
    resp = client.create_t2v_task(prompt="A sunset over the ocean")
    task_id = resp["task_id"]
    status = client.wait_for_completion(task_id)       # raises on failure/timeout
    client.download_result(task_id, "sunset.mp4")

VC chain workflow (Video Continue — cold start → continuation):
    client = TFClient("http://localhost:8000")

    # Step 1: Cold start — first chunk, only first_image_path
    resp1 = client.create_vc_task(
        prompt="A woman standing by a window, warm sunlight, gentle smile.",
        first_image_path="/path/to/ref_image.png",
    )
    client.wait_for_completion(resp1["task_id"])
    client.download_result(resp1["task_id"], "chunk_1.mp4")

    # Step 2: Continuation — pass downloaded local video as ref_video_path
    resp2 = client.create_vc_task(
        prompt="She turns around, walking deeper into the room.",
        first_image_path="/path/to/ref_image.png",
        ref_video_path="chunk_1.mp4",       # local file, auto base64-encoded
    )
    client.wait_for_completion(resp2["task_id"])
    client.download_result(resp2["task_id"], "chunk_2.mp4")

    # Step 3: Another continuation
    resp3 = client.create_vc_task(
        prompt="She walks to the window and looks outside.",
        first_image_path="/path/to/ref_image.png",
        ref_video_path="chunk_2.mp4",
    )
    client.wait_for_completion(resp3["task_id"])
    client.download_result(resp3["task_id"], "chunk_3.mp4")

Multi-server async processing:
    from telefuser.client import get_available_urls, process_tasks_async

    urls = ["http://gpu1:8000", "http://gpu2:8000", "http://gpu3:8000"]
    available = get_available_urls(urls)
    messages = [
        {"task": "t2v", "prompt": "Scene 1", "seed": 1},
        {"task": "t2v", "prompt": "Scene 2", "seed": 2},
    ]
    process_tasks_async(messages, available)

Task types:
    TASK_T2V  = "t2v"   — Text-to-Video
    TASK_I2V  = "i2v"   — Image-to-Video
    TASK_FL2V = "fl2v"  — First-Last-frame-to-Video
    TASK_VC   = "vc"    — Video Continue (streaming continuation)
    TASK_T2I  = "t2i"   — Text-to-Image
    TASK_I2I  = "i2i"   — Image-to-Image
    TASK_S2V  = "s2v"   — Sketch-to-Video
    TASK_VSR  = "vsr"   — Video Super-Resolution
    TASK_EDIT = "edit"  — Image Editing
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ── Constants: Task Types ───────────────────────────────────────────────────

TASK_T2V = "t2v"
TASK_I2V = "i2v"
TASK_FL2V = "fl2v"
TASK_VC = "vc"
TASK_T2I = "t2i"
TASK_I2I = "i2i"
TASK_S2V = "s2v"
TASK_VSR = "vsr"
TASK_EDIT = "edit"

VIDEO_TASKS = (TASK_T2V, TASK_I2V, TASK_FL2V, TASK_VC, TASK_S2V, TASK_VSR)
IMAGE_TASKS = (TASK_T2I, TASK_I2I, TASK_EDIT)
VALID_TASK_TYPES = (TASK_T2V, TASK_I2V, TASK_FL2V, TASK_VC, TASK_T2I, TASK_I2I, TASK_S2V, TASK_VSR, TASK_EDIT)

# ── Constants: Aspect Ratios ────────────────────────────────────────────────

AR_16_9 = "16:9"
AR_9_16 = "9:16"
AR_4_3 = "4:3"
AR_3_4 = "3:4"
AR_1_1 = "1:1"
AR_2_3 = "2:3"
AR_3_2 = "3:2"

VALID_ASPECT_RATIOS = (AR_16_9, AR_9_16, AR_4_3, AR_3_4, AR_1_1, AR_2_3, AR_3_2)

# ── Constants: Output Formats ───────────────────────────────────────────────

FORMAT_PNG = "png"
FORMAT_JPG = "jpg"
FORMAT_JPEG = "jpeg"
FORMAT_WEBP = "webp"

VALID_OUTPUT_FORMATS = (FORMAT_PNG, FORMAT_JPG, FORMAT_JPEG, FORMAT_WEBP)

# ── Constants: Task Statuses ────────────────────────────────────────────────

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


# ── Exceptions ──────────────────────────────────────────────────────────────


class TeleFuserError(Exception):
    """Base exception for TeleFuser client errors."""


class TaskCreationError(TeleFuserError):
    """Raised when task creation fails (HTTP error from /v1/tasks/create)."""


class TaskFailedError(TeleFuserError):
    """Raised when a task reaches 'failed' or 'cancelled' status."""


class TaskTimeoutError(TeleFuserError):
    """Raised when wait_for_completion exceeds the timeout."""


# ── TFClient ────────────────────────────────────────────────────────────────


class TFClient:
    """Client for interacting with the TeleFuser API server.

    Uses requests.Session with trust_env=False to bypass proxy interference
    on localhost connections.

    Example:
        client = TFClient("http://localhost:8000")
        resp = client.create_t2v_task(prompt="A sunset over the ocean")
        status = client.wait_for_completion(resp["task_id"])
        client.download_result(resp["task_id"], "output.mp4")
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.trust_env = False  # Avoid proxy interference on localhost

    def close(self) -> None:
        """Close the underlying requests session."""
        self._session.close()

    # ── File encoding helpers ───────────────────────────────────────────────

    def file_to_base64(self, file_path: str) -> str:
        """Convert a local file to a base64-encoded string.

        Args:
            file_path: Path to a local file.

        Returns:
            Base64-encoded string of the file contents.
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _encode_file_input(self, path: str) -> str:
        """Encode file input: pass through URLs, base64-encode local files.

        Args:
            path: HTTP/HTTPS URL (passed through) or local file path (base64-encoded).

        Returns:
            URL string or base64-encoded data string.
        """
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return self.file_to_base64(path)

    # ── Generic task creation ────────────────────────────────────────────────

    def create_task(self, task_type: str, **params: Any) -> Dict[str, Any]:
        """Create a task of any supported type with arbitrary parameters.

        Low-level entry point. Specific methods (create_t2v_task, etc.)
        delegate here after encoding file inputs and setting defaults.

        Args:
            task_type: One of VALID_TASK_TYPES or a custom task name string.
            **params: Arbitrary parameters passed directly to the API.
                Common params: prompt, seed, resolution, negative_prompt,
                aspect_ratio, target_video_length, output_format,
                first_image_path, last_image_path, ref_video_path.

        Returns:
            Dict with task_id, task_status, output_path.

        Raises:
            TeleFuserError: On HTTP errors from the server.

        Example:
            resp = client.create_task(TASK_T2V, prompt="Sunset", seed=42)
            resp = client.create_task("custom_task", prompt="...", my_param="value")
        """
        # Encode file-like parameters if they are local paths
        for key in ("first_image_path", "last_image_path", "ref_video_path"):
            if key in params and params[key] and not isinstance(params[key], list):
                val = str(params[key])
                if not val.startswith(("http://", "https://")):
                    params[key] = self.file_to_base64(val)

        payload = {"task": task_type, **params}
        try:
            response = self._session.post(
                f"{self.base_url}/v1/tasks/create",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as e:
            raise TaskCreationError(f"Task creation failed (HTTP {e.response.status_code}): {e.response.text}") from e
        except requests.RequestException as e:
            raise TaskCreationError(f"Task creation request failed: {e}") from e

    # ── Video task creation methods ──────────────────────────────────────────

    def create_t2v_task(
        self,
        prompt: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: str = AR_16_9,
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create a Text-to-Video (T2V) generation task.

        Example:
            resp = client.create_t2v_task(
                prompt="A sunset over the ocean",
                resolution="720p",
                aspect_ratio="16:9",
            )

        Args:
            prompt: Text description of the video to generate.
            resolution: Target video resolution (e.g., "720p", "1080p", "480p").
            seed: Random seed for reproducibility.
            negative_prompt: Text describing what to avoid in the video.
            aspect_ratio: Video aspect ratio (one of VALID_ASPECT_RATIOS).
            video_length: Target video length in seconds.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        return self.create_task(
            TASK_T2V,
            prompt=prompt,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            target_video_length=video_length,
        )

    def create_i2v_task(
        self,
        prompt: str,
        first_image_path: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create an Image-to-Video (I2V) generation task.

        Example:
            resp = client.create_i2v_task(
                prompt="A woman dancing in the rain",
                first_image_path="/path/to/photo.jpg",
            )

        Args:
            prompt: Text description of the video to generate.
            first_image_path: Reference image (local file, URL, or server path).
                Local files are base64-encoded; URLs pass through directly.
            resolution: Target video resolution.
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt text.
            video_length: Target video length in seconds.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        image_input = self._encode_file_input(first_image_path)
        return self.create_task(
            TASK_I2V,
            prompt=prompt,
            first_image_path=image_input,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            target_video_length=video_length,
        )

    def create_fl2v_task(
        self,
        prompt: str,
        first_image_path: str,
        last_image_path: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create a First-Last-frame-to-Video (FL2V) generation task.

        Example:
            resp = client.create_fl2v_task(
                prompt="Transition from sunrise to sunset",
                first_image_path="/path/to/start_frame.png",
                last_image_path="/path/to/end_frame.png",
            )

        Args:
            prompt: Text description of the video to generate.
            first_image_path: First frame image (local file or URL).
            last_image_path: Last frame image (local file or URL).
            resolution: Target video resolution.
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt text.
            video_length: Target video length in seconds.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        first_image = self._encode_file_input(first_image_path)
        last_image = self._encode_file_input(last_image_path)
        return self.create_task(
            TASK_FL2V,
            prompt=prompt,
            first_image_path=first_image,
            last_image_path=last_image,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            target_video_length=video_length,
        )

    def create_vc_task(
        self,
        prompt: str,
        first_image_path: str,
        ref_video_path: str = "",
        seed: int = 42,
        negative_prompt: str = "",
    ) -> Dict[str, Any]:
        """Create a Video Continue (VC) task — streaming chunk-by-chunk generation.

        Cold start (first chunk): provide only first_image_path.
            resp = client.create_vc_task(
                prompt="A woman standing by a window, warm sunlight.",
                first_image_path="/path/to/ref_image.png",
            )
            client.wait_for_completion(resp["task_id"])
            client.download_result(resp["task_id"], "chunk_1.mp4")

        Continuation (subsequent chunks): pass downloaded local video as ref_video_path.
            resp2 = client.create_vc_task(
                prompt="She turns around, walking deeper into the room.",
                first_image_path="/path/to/ref_image.png",
                ref_video_path="chunk_1.mp4",       # local file, auto base64-encoded
            )
            client.wait_for_completion(resp2["task_id"])
            client.download_result(resp2["task_id"], "chunk_2.mp4")

        Args:
            prompt: Positive guidance text for this chunk.
            first_image_path: Memory anchor image (required every call).
                Local files are base64-encoded; URLs pass through directly.
            ref_video_path: Previous chunk output (local file or URL).
                Empty for cold start. When non-empty, generates a continuation chunk.
                Must be a client-local file — download the previous output first,
                then pass the local path here (auto base64-encoded on upload).
            seed: Random seed for reproducibility.
            negative_prompt: Additional negative prompt text.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        image_input = self._encode_file_input(first_image_path)
        params: Dict[str, Any] = {
            "prompt": prompt,
            "first_image_path": image_input,
            "seed": seed,
            "negative_prompt": negative_prompt,
        }
        if ref_video_path:
            params["ref_video_path"] = self._encode_file_input(ref_video_path)
        return self.create_task(TASK_VC, **params)

    def create_s2v_task(
        self,
        prompt: str,
        sketch_path: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create a Sketch-to-Video (S2V) generation task.

        Example:
            resp = client.create_s2v_task(
                prompt="A bird flying across the sky",
                sketch_path="/path/to/sketch.png",
            )

        Args:
            prompt: Text description of the video to generate.
            sketch_path: Sketch image path (local file or URL).
                Mapped to first_image_path internally.
            resolution: Target video resolution.
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt text.
            video_length: Target video length in seconds.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        sketch_input = self._encode_file_input(sketch_path)
        return self.create_task(
            TASK_S2V,
            prompt=prompt,
            first_image_path=sketch_input,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            target_video_length=video_length,
        )

    def create_vsr_task(
        self,
        prompt: str,
        ref_video_path: str,
        resolution: str = "1080p",
        seed: int = 42,
        negative_prompt: str = "",
    ) -> Dict[str, Any]:
        """Create a Video Super-Resolution (VSR) task.

        Example:
            resp = client.create_vsr_task(
                prompt="Upscale to HD",
                ref_video_path="/path/to/low_quality.mp4",
                resolution="1080p",
            )

        Args:
            prompt: Text description or quality guidance.
            ref_video_path: Low-resolution input video (local file or URL).
                Local files are base64-encoded; URLs pass through directly.
            resolution: Target output resolution (e.g., "1080p", "2k").
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt text.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        video_input = self._encode_file_input(ref_video_path)
        return self.create_task(
            TASK_VSR,
            prompt=prompt,
            ref_video_path=video_input,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
        )

    # ── Image task creation methods ──────────────────────────────────────────

    def create_t2i_task(
        self,
        prompt: str,
        resolution: str = "1024x1024",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: str = AR_1_1,
        output_format: str = FORMAT_PNG,
    ) -> Dict[str, Any]:
        """Create a Text-to-Image (T2I) generation task.

        Example:
            resp = client.create_t2i_task(
                prompt="A beautiful landscape painting",
                resolution="1024x1024",
                output_format="png",
            )

        Args:
            prompt: Text description of the image to generate.
            resolution: Target image resolution (e.g., "1024x1024", "1024x768").
            seed: Random seed for reproducibility.
            negative_prompt: Text describing what to avoid in the image.
            aspect_ratio: Image aspect ratio (one of VALID_ASPECT_RATIOS).
            output_format: Output image format (one of VALID_OUTPUT_FORMATS).

        Returns:
            Dict with task_id, task_status, output_path.
        """
        return self.create_task(
            TASK_T2I,
            prompt=prompt,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
        )

    def create_i2i_task(
        self,
        prompt: str,
        image_path: str,
        resolution: str = "1024x1024",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: str = AR_1_1,
        output_format: str = FORMAT_PNG,
    ) -> Dict[str, Any]:
        """Create an Image-to-Image (I2I) generation task.

        Example:
            resp = client.create_i2i_task(
                prompt="Make it look like a watercolor painting",
                image_path="/path/to/input.png",
            )

        Args:
            prompt: Text description of the desired transformation.
            image_path: Path to input image (local file or URL).
                Local files are base64-encoded; URLs pass through directly.
            resolution: Target image resolution.
            seed: Random seed for reproducibility.
            negative_prompt: Text describing what to avoid.
            aspect_ratio: Image aspect ratio.
            output_format: Output image format.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        image_input = self._encode_file_input(image_path)
        return self.create_task(
            TASK_I2I,
            prompt=prompt,
            first_image_path=image_input,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
        )

    def create_edit_task(
        self,
        prompt: str,
        image_path: str,
        resolution: str = "1024x1024",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: str = AR_1_1,
        output_format: str = FORMAT_PNG,
    ) -> Dict[str, Any]:
        """Create an Image Editing (Edit) task.

        Example:
            resp = client.create_edit_task(
                prompt="Add sunglasses to the person",
                image_path="/path/to/face.png",
            )

        Args:
            prompt: Text description of the desired edit.
            image_path: Path to input image (local file or URL).
                Local files are base64-encoded; URLs pass through directly.
            resolution: Target image resolution.
            seed: Random seed for reproducibility.
            negative_prompt: Negative prompt text.
            aspect_ratio: Image aspect ratio.
            output_format: Output image format.

        Returns:
            Dict with task_id, task_status, output_path.
        """
        image_input = self._encode_file_input(image_path)
        return self.create_task(
            TASK_EDIT,
            prompt=prompt,
            first_image_path=image_input,
            seed=seed,
            resolution=resolution,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
        )

    # ── Task lifecycle methods ───────────────────────────────────────────────

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Query the status of a task.

        Args:
            task_id: The task ID returned by a create_*_task method.

        Returns:
            Dict with task_id, status, output_path, error, etc.

        Example:
            status = client.get_task_status("ABCD-1234-EFGH-5678")
            print(status["status"])  # "pending", "processing", "completed", "failed"
        """
        response = self._session.get(
            f"{self.base_url}/v1/tasks/{task_id}/status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        task_id: str,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """Wait for a task to reach a terminal state (completed, failed, or cancelled).

        Polls get_task_status at the given interval until the task finishes
        or the timeout elapses.

        Args:
            task_id: The task ID to monitor.
            timeout: Maximum seconds to wait (default 300).
            poll_interval: Seconds between status checks (default 2).

        Returns:
            Final status dict from get_task_status (contains output_path on success).

        Raises:
            TaskFailedError: If task status is 'failed' or 'cancelled'.
            TaskTimeoutError: If timeout elapses without reaching a terminal state.

        Example:
            status = client.wait_for_completion(task_id, timeout=300)
            print(status["output_path"])  # completed → has output_path
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            task_status = status.get("status") or status.get("task_status")

            if task_status == STATUS_COMPLETED:
                return status
            elif task_status == STATUS_FAILED:
                raise TaskFailedError(f"Task {task_id} failed: {status.get('error', 'unknown')}")
            elif task_status == STATUS_CANCELLED:
                raise TaskFailedError(f"Task {task_id} was cancelled")

            time.sleep(poll_interval)

        raise TaskTimeoutError(f"Task {task_id} timed out after {timeout}s")

    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a running or pending task.

        Args:
            task_id: The task ID to cancel.

        Returns:
            Dict with stop_status and reason.

        Example:
            result = client.cancel_task("ABCD-1234-EFGH-5678")
            print(result["stop_status"])  # "success" or "do_nothing"
        """
        response = self._session.delete(
            f"{self.base_url}/v1/tasks/{task_id}",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    # ── Result retrieval ──────────────────────────────────────────────────────

    def download_result(self, task_id: str, output_path: str) -> bool:
        """Download the result file (video or image) to a local path.

        Args:
            task_id: The task ID whose result to download.
            output_path: Local file path to save the result.

        Returns:
            True if download succeeded, False if task has no output_path.

        Example:
            client.download_result(task_id, "output.mp4")
        """
        status = self.get_task_status(task_id)
        if not status or "output_path" not in status:
            return False

        file_name = status["output_path"].split("/")[-1]
        response = self._session.get(
            f"{self.base_url}/v1/files/download/{file_name}",
            stream=True,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True

        return False

    # ── Service info methods ─────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Check if the server is healthy and the pipeline is ready.

        Returns:
            True if server is healthy, False otherwise.

        Example:
            if not client.health_check():
                print("Server not ready")
        """
        try:
            response = self._session.get(
                f"{self.base_url}/v1/service/health",
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("pipeline_ready", False)
            return False
        except requests.RequestException:
            return False

    def get_service_status(self) -> Dict[str, Any]:
        """Get the current service status (idle/active, current task info).

        Returns:
            Dict with service_status, current_task, pending_count, etc.

        Example:
            status = client.get_service_status()
            print(status["service_status"])  # "idle" or "active"
        """
        response = self._session.get(
            f"{self.base_url}/v1/service/status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_service_metadata(self) -> Dict[str, Any]:
        """Get pipeline metadata (supported tasks, parameter contracts).

        Returns:
            Dict with pipeline_name, supported_tasks, task_contracts, etc.

        Example:
            meta = client.get_service_metadata()
            print(meta["supported_tasks"])  # ["vc", "t2v", ...]
        """
        response = self._session.get(
            f"{self.base_url}/v1/service/metadata",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_queue_status(self) -> Dict[str, Any]:
        """Get the current task queue status.

        Returns:
            Dict with is_processing, current_task, pending_count, queue_size, etc.

        Example:
            queue = client.get_queue_status()
            print(f"Pending: {queue['pending_count']}, Active: {queue['active_count']}")
        """
        response = self._session.get(
            f"{self.base_url}/v1/tasks/queue/status",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()


# ── Multi-Server Utilities ──────────────────────────────────────────────────


def send_and_monitor_task(
    url: str,
    message: Dict[str, Any],
    task_index: int,
    complete_bar: Optional[Any] = None,
    complete_lock: Optional[threading.Lock] = None,
) -> bool:
    """Send a task to a server and monitor until completion.

    Args:
        url: Server base URL.
        message: Task payload dict (same format as create_task params).
        task_index: Index of the task for logging.
        complete_bar: Optional tqdm progress bar instance.
        complete_lock: Optional threading.Lock for thread-safe progress updates.

    Returns:
        True if task completed successfully, False otherwise.

    Example:
        send_and_monitor_task(
            "http://gpu1:8000",
            {"task": "t2v", "prompt": "Scene 1", "seed": 42},
            task_index=0,
        )
    """
    try:
        response = requests.post(f"{url}/v1/tasks/create", json=message)
        response_data: Dict[str, Any] = response.json()
        task_id: Optional[str] = response_data.get("task_id")

        if not task_id:
            logger.error("No task_id received from %s", url)
            return False

        while True:
            try:
                status_response = requests.get(f"{url}/v1/tasks/{task_id}/status")
                status_data: Dict[str, Any] = status_response.json()
                task_status: Optional[str] = status_data.get("status")

                if task_status == STATUS_COMPLETED:
                    logger.info("Task %d completed: %s", task_index + 1, response_data)
                    if complete_bar and complete_lock:
                        with complete_lock:
                            complete_bar.update(1)
                    return True
                elif task_status == STATUS_FAILED:
                    logger.error("Task %d (task_id: %s) failed", task_index + 1, task_id)
                    if complete_bar and complete_lock:
                        with complete_lock:
                            complete_bar.update(1)
                    return False
                else:
                    time.sleep(0.5)
            except Exception as e:
                logger.error("Failed to check status for task_id %s: %s", task_id, e)
                time.sleep(0.5)

    except Exception as e:
        logger.error("Failed to send task to %s: %s", url, e)
        return False


def get_available_urls(urls: List[str]) -> Optional[List[str]]:
    """Check which server URLs are reachable and return the list.

    Args:
        urls: List of server URLs to check.

    Returns:
        List of available URLs, or None if none are reachable.

    Example:
        available = get_available_urls(["http://gpu1:8000", "http://gpu2:8000"])
    """
    available_urls: List[str] = []
    for url in urls:
        try:
            requests.get(f"{url}/v1/service/status", timeout=5)
            available_urls.append(url)
        except Exception:
            continue

    if not available_urls:
        logger.error("No available URLs found.")
        return None

    logger.info("Available URLs: %s", available_urls)
    return available_urls


def find_idle_server(available_urls: List[str]) -> str:
    """Find an idle server from available URLs by checking service status.

    Blocks until an idle server is found. Polls every 3 seconds.

    Args:
        available_urls: List of reachable server URLs.

    Returns:
        URL of an idle server.

    Example:
        idle_url = find_idle_server(["http://gpu1:8000", "http://gpu2:8000"])
    """
    while True:
        for url in available_urls:
            try:
                response = requests.get(f"{url}/v1/service/status", timeout=5).json()
                if response.get("service_status") == "idle":
                    return url
            except Exception:
                continue
        time.sleep(3)


def process_tasks_async(
    messages: List[Dict[str, Any]],
    available_urls: List[str],
    show_progress: bool = True,
) -> bool:
    """Process a list of tasks across multiple servers concurrently.

    Each task is assigned to the first idle server found. Tasks run in
    separate threads; progress is shown via tqdm (if installed).

    Args:
        messages: List of task payload dicts.
        available_urls: List of reachable server URLs.
        show_progress: Whether to show a progress bar (requires tqdm).

    Returns:
        True if all tasks were dispatched, False if no servers available.

    Example:
        urls = get_available_urls(["http://gpu1:8000", "http://gpu2:8000"])
        messages = [
            {"task": "t2v", "prompt": "Scene 1", "seed": 1},
            {"task": "t2v", "prompt": "Scene 2", "seed": 2},
        ]
        process_tasks_async(messages, urls)
    """
    if not available_urls:
        logger.error("No available servers to process tasks.")
        return False

    active_threads: List[threading.Thread] = []
    logger.info("Sending %d tasks to available servers...", len(messages))

    complete_bar: Optional[Any] = None
    complete_lock: Optional[threading.Lock] = None
    if show_progress:
        if tqdm is not None:
            complete_bar = tqdm(total=len(messages), desc="Completing tasks")
            complete_lock = threading.Lock()
        else:
            logger.info("tqdm not installed — using simple print-based progress.")

    completed = 0
    total = len(messages)

    for idx, message in enumerate(messages):
        server_url = find_idle_server(available_urls)

        thread = threading.Thread(
            target=send_and_monitor_task,
            args=(server_url, message, idx, complete_bar, complete_lock),
        )
        thread.daemon = False
        thread.start()
        active_threads.append(thread)
        time.sleep(0.5)

    for thread in active_threads:
        thread.join()

    if complete_bar:
        complete_bar.close()

    if tqdm is None and show_progress:
        # Simple fallback progress reporting
        completed = sum(1 for t in active_threads if t.is_alive() is False)
        logger.info("Tasks processed: %d / %d", completed, total)

    logger.info("All tasks processing completed!")
    return True
