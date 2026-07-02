from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import random
import signal
import string
import sys
import time
from datetime import datetime
from types import FrameType, ModuleType
from typing import Any, Callable

import psutil
import torch
from PIL import Image
from pydantic import BaseModel

from telefuser.utils.logging import logger


def import_function_from_file(file_path: str, function_name: str) -> Callable[..., Any]:
    """Import a specific function from a Python file at the specified path."""
    module_name = file_path.split("/")[-1].split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def generate_task_id() -> str:
    """Generate a random task ID in format XXXX-XXXX-XXXX-XXXX-XXXX.

    Does not modify the global random state. Uses time-based entropy for uniqueness.
    """
    original_state = random.getstate()

    try:
        characters = string.ascii_uppercase + string.digits
        local_random = random.Random(time.perf_counter_ns())

        groups = []
        for _ in range(5):
            time_mix = int(datetime.now().timestamp())
            local_random.seed(time_mix + local_random.getstate()[1][0] + time.perf_counter_ns())
            groups.append("".join(local_random.choices(characters, k=4)))

        return "-".join(groups)

    finally:
        random.setstate(original_state)


class ProcessManager:
    """Process management utilities for cleanup."""

    @staticmethod
    def kill_all_related_processes() -> None:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception as e:
                logger.info(f"Failed to kill child process {child.pid}: {e}")
        try:
            current_process.kill()
        except Exception as e:
            logger.info(f"Failed to kill main process: {e}")

    @staticmethod
    def signal_handler(sig: int, frame: FrameType | None) -> None:
        logger.info("\nReceived Ctrl+C, shutting down all related processes...")
        ProcessManager.kill_all_related_processes()
        sys.exit(0)

    @staticmethod
    def register_signal_handler() -> None:
        signal.signal(signal.SIGINT, ProcessManager.signal_handler)


class TaskStatusMessage(BaseModel):
    """Task status message model."""

    task_id: str


class TensorTransporter:
    """Utility for serializing/deserializing tensors via base64."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def to_device(self, data: Any, device: str | torch.device) -> Any:
        """Recursively move data to specified device."""
        if isinstance(data, dict):
            return {key: self.to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item, device) for item in data]
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            return data

    def prepare_tensor(self, data: Any) -> str:
        """Serialize tensor to base64 string."""
        self.buffer.seek(0)
        self.buffer.truncate()
        torch.save(self.to_device(data, "cpu"), self.buffer)
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_tensor(self, tensor_base64: str, device: str | torch.device = "cuda") -> Any:
        """Load tensor from base64 string."""
        tensor_bytes = base64.b64decode(tensor_base64)
        with io.BytesIO(tensor_bytes) as buffer:
            return self.to_device(torch.load(buffer), device)


class ImageTransporter:
    """Utility for serializing/deserializing images via base64."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()

    def prepare_image(self, image: Image.Image) -> str:
        """Serialize PIL image to base64 PNG string."""
        self.buffer.seek(0)
        self.buffer.truncate()
        image.save(self.buffer, format="PNG")
        return base64.b64encode(self.buffer.getvalue()).decode("utf-8")

    def load_image(self, image_base64: bytes) -> Image.Image:
        """Load PIL image from base64 string."""
        image_bytes = base64.b64decode(image_base64)
        with io.BytesIO(image_bytes) as buffer:
            return Image.open(buffer).convert("RGB")
