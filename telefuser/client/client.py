"""TeleFuser API Client.

Provides synchronous client for the TeleFuser native REST API.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Dict

import requests

from telefuser.service_types import AspectRatio, OutputFormat, TaskType
from telefuser.utils.logging import logger


class TAPClient:
    """Client for interacting with the TeleFuser API server."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url: str = base_url

    def file_to_base64(self, file_path: str) -> str:
        """Convert a file to base64 encoded string."""
        with open(file_path, "rb") as f:
            file_data = f.read()
        return base64.b64encode(file_data).decode("utf-8")

    def create_t2v_task(
        self,
        prompt: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: AspectRatio | str = AspectRatio.RATIO_16_9,
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create a text-to-video generation task.

        Args:
            prompt: Text description of the video to generate
            resolution: Target video resolution (e.g., "720p", "1080p")
            seed: Random seed for reproducibility
            negative_prompt: Text describing what to avoid in the video
            aspect_ratio: Video aspect ratio (e.g., "16:9", "9:16")
            video_length: Target video length in seconds

        Returns:
            API response containing task_id and status
        """
        payload: Dict[str, Any] = {
            "task": TaskType.T2V,
            "prompt": prompt,
            "seed": seed,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "target_video_length": video_length,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/create", json=payload)
        return response.json()

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
        """Create first-last image to video task."""
        # Support multiple image input formats
        if first_image_path.startswith("http"):
            # URL format
            first_image = first_image_path
            last_image = last_image_path
        else:
            # Local file to base64
            first_image = self.file_to_base64(first_image_path)
            last_image = self.file_to_base64(last_image_path)

        payload: Dict[str, Any] = {
            "task": TaskType.FL2V,
            "prompt": prompt,
            "first_image_path": first_image,
            "last_image_path": last_image,
            "seed": seed,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "target_video_length": video_length,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/create", json=payload)
        return response.json()

    def create_vc_task(
        self,
        prompt: str,
        ref_video_path: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create video continue task."""
        # Support multiple image input formats
        if ref_video_path.startswith("http"):
            # URL format
            video_input = ref_video_path
        else:
            # Local file to base64
            video_input = self.file_to_base64(ref_video_path)

        payload: Dict[str, Any] = {
            "task": TaskType.VC,
            "prompt": prompt,
            "ref_video_path": video_input,
            "seed": seed,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "target_video_length": video_length,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/create", json=payload)
        return response.json()

    def create_i2v_task(
        self,
        prompt: str,
        first_image_path: str,
        resolution: str = "720p",
        seed: int = 42,
        negative_prompt: str = "",
        video_length: int = 5,
    ) -> Dict[str, Any]:
        """Create i2v task."""
        # Support multiple image input formats
        if first_image_path.startswith("http"):
            # URL format
            image_input = first_image_path
        else:
            # Local file to base64
            image_input = self.file_to_base64(first_image_path)

        payload: Dict[str, Any] = {
            "task": TaskType.I2V,
            "prompt": prompt,
            "first_image_path": image_input,
            "seed": seed,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "target_video_length": video_length,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/create", json=payload)
        return response.json()

    def create_t2i_task(
        self,
        prompt: str,
        resolution: str = "1024x1024",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: AspectRatio | str = AspectRatio.RATIO_1_1,
        output_format: OutputFormat | str = OutputFormat.PNG,
    ) -> Dict[str, Any]:
        """Create a text-to-image generation task.

        Args:
            prompt: Text description of the image to generate
            resolution: Target image resolution (e.g., "1024x1024", "1024x768")
            seed: Random seed for reproducibility
            negative_prompt: Text describing what to avoid in the image
            aspect_ratio: Image aspect ratio (e.g., "1:1", "16:9", "4:3")
            output_format: Output image format ("png", "jpg", "webp")

        Returns:
            API response containing task_id and status
        """
        payload: Dict[str, Any] = {
            "task": TaskType.T2I,
            "prompt": prompt,
            "seed": seed,
            "resolution": resolution,  # Reuse field for compatibility
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/create", json=payload)
        return response.json()

    def create_i2i_task(
        self,
        prompt: str,
        image_path: str,
        resolution: str = "1024x1024",
        seed: int = 42,
        negative_prompt: str = "",
        aspect_ratio: AspectRatio | str = AspectRatio.RATIO_1_1,
        output_format: OutputFormat | str = OutputFormat.PNG,
    ) -> Dict[str, Any]:
        """Create an image-to-image generation task.

        Args:
            prompt: Text description of the image to generate
            image_path: Path to input image (local file or URL)
            resolution: Target image resolution (e.g., "1024x1024")
            seed: Random seed for reproducibility
            negative_prompt: Text describing what to avoid
            aspect_ratio: Image aspect ratio
            output_format: Output image format

        Returns:
            API response containing task_id and status
        """
        # Support multiple image input formats
        if image_path.startswith("http"):
            image_input = image_path
        else:
            image_input = self.file_to_base64(image_path)

        payload: Dict[str, Any] = {
            "task": TaskType.I2I,
            "prompt": prompt,
            "first_image_path": image_input,
            "seed": seed,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "output_format": output_format,
        }

        response = requests.post(f"{self.base_url}/v1/tasks/create", json=payload)
        return response.json()

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Query task status."""
        response = requests.get(f"{self.base_url}/v1/tasks/{task_id}/status")
        return response.json()

    def wait_for_completion(self, task_id: str, timeout: int = 300) -> bool:
        """Wait for task completion."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            if status.get("status") == "completed":
                return True
            elif status.get("status") == "failed":
                logger.warning(f"task {task_id} status failed")
                return False
            time.sleep(2)
        logger.warning(f"task {task_id} status timeout with th {timeout}s")
        return False

    def download_result(self, task_id: str, output_path: str) -> bool:
        """Download result file (video or image).

        Args:
            task_id: The task ID
            output_path: Local path to save the file

        Returns:
            True if download successful, False otherwise
        """
        # Try the new files/download endpoint first
        status = self.get_task_status(task_id)
        if status and "output_path" in status:
            file_name = status["output_path"].split("/")[-1]
            response = requests.get(f"{self.base_url}/v1/files/download/{file_name}", stream=True)
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True

        return False
