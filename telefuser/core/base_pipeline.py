"""Base pipeline for multimodal generation."""

from __future__ import annotations

import json
from abc import ABC
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

from telefuser.utils.logging import logger

if TYPE_CHECKING:
    from telefuser.metrics import StageMetricsManager


class BasePipeline(ABC):
    """Base pipeline for generation tasks."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Wrap init method to print config after initialization
        if "init" in cls.__dict__:
            original_init = cls.__dict__["init"]

            @wraps(original_init)
            def wrapped_init(self, *args, **kwargs):
                result = original_init(self, *args, **kwargs)
                if hasattr(self, "config"):
                    self._print_config_banner()
                return result

            cls.init = wrapped_init

        # Wrap __call__ method to print parameters and reset timing registry
        if "__call__" in cls.__dict__:
            original_call = cls.__dict__["__call__"]

            @wraps(original_call)
            def wrapped_call(self, *args, **kwargs):
                from telefuser.utils.profiler import reset_timing_registry

                reset_timing_registry()
                self._print_call_banner(args, kwargs)
                try:
                    return original_call(self, *args, **kwargs)
                finally:
                    if getattr(self, "clear_memory_after_call", True):
                        # Clear GPU memory after pipeline execution.
                        import gc

                        from telefuser.platforms import current_platform

                        gc.collect()
                        current_platform.empty_cache()

            cls.__call__ = wrapped_call

    # ANSI color codes for banner formatting
    _ANSI_CYAN = "\033[36m"
    _ANSI_GREEN = "\033[32m"
    _ANSI_YELLOW = "\033[33m"
    _ANSI_BLUE = "\033[34m"
    _ANSI_DIM = "\033[2m"
    _ANSI_BOLD = "\033[1m"
    _ANSI_RESET = "\033[0m"

    def _get_config_defaults(self, config: Any) -> dict[str, Any]:
        """Get default values for config fields.

        Args:
            config: Config object (typically a dataclass)

        Returns:
            Dict mapping field names to their default values
        """
        defaults = {}
        if hasattr(config, "__dataclass_fields__"):
            from dataclasses import MISSING, fields

            for field in fields(config):
                if field.default is not MISSING:
                    defaults[field.name] = field.default
                elif field.default_factory is not MISSING:
                    defaults[field.name] = field.default_factory()
                else:
                    # No default value - treat as always changed
                    defaults[field.name] = None
        return defaults

    def _print_config_banner(self) -> None:
        """Print pipeline config initialization banner with formatted output."""
        SEP = f"{self._ANSI_DIM}─{'─' * 50}─{self._ANSI_RESET}"

        lines = [
            SEP,
            f"{self._ANSI_BOLD}{self._ANSI_CYAN}Pipeline Config{self._ANSI_RESET}  "
            f"{self._ANSI_DIM}{self.__class__.__name__}{self._ANSI_RESET}",
        ]

        # Format config fields, only showing values that differ from defaults
        config = self.config
        defaults = self._get_config_defaults(config)

        if hasattr(config, "__dataclass_fields__"):
            # Dataclass config - format each field
            from dataclasses import asdict

            config_dict = asdict(config)
            changed_count = 0
            for key, value in config_dict.items():
                default_value = defaults.get(key)
                is_changed = default_value is None or value != default_value
                if is_changed:
                    changed_count += 1
                    formatted_value = self._format_config_value(value)
                    lines.append(f"  {self._ANSI_DIM}{key}:{self._ANSI_RESET} {formatted_value}")

            if changed_count == 0:
                lines.append(f"  {self._ANSI_DIM}(all defaults){self._ANSI_RESET}")
        else:
            # Non-dataclass config - try to format as dict
            try:
                config_dict = dict(config) if hasattr(config, "items") else {}
                for key, value in config_dict.items():
                    formatted_value = self._format_config_value(value)
                    lines.append(f"  {self._ANSI_DIM}{key}:{self._ANSI_RESET} {formatted_value}")
            except Exception:
                lines.append(f"  {config}")

        lines.append(SEP)

        # Print to stderr for visibility (like logging.py)
        import sys

        print("\n".join(lines), file=sys.stderr)

    def _format_config_value(self, value: Any) -> str:
        """Format a config value for display.

        Args:
            value: Config value to format

        Returns:
            Formatted string representation
        """
        if isinstance(value, str):
            return f"{self._ANSI_GREEN}{value}{self._ANSI_RESET}"
        elif isinstance(value, bool):
            color = self._ANSI_GREEN if value else self._ANSI_YELLOW
            return f"{color}{value}{self._ANSI_RESET}"
        elif isinstance(value, (int, float)):
            return f"{self._ANSI_BLUE}{value}{self._ANSI_RESET}"
        elif isinstance(value, dict):
            # Nested dict - show key count
            return f"{self._ANSI_DIM}dict({len(value)} items){self._ANSI_RESET}"
        elif isinstance(value, list):
            return f"{self._ANSI_DIM}list({len(value)} items){self._ANSI_RESET}"
        elif hasattr(value, "__class__"):
            return f"{self._ANSI_DIM}{value.__class__.__name__}{self._ANSI_RESET}"
        return str(value)

    def _print_call_banner(self, args: tuple, kwargs: dict) -> None:
        """Print pipeline __call__ parameters banner with formatted output."""
        SEP = f"{self._ANSI_DIM}─{'─' * 50}─{self._ANSI_RESET}"

        lines = [
            SEP,
            f"{self._ANSI_BOLD}{self._ANSI_YELLOW}Pipeline Call{self._ANSI_RESET}  "
            f"{self._ANSI_DIM}{self.__class__.__name__}{self._ANSI_RESET}",
        ]

        # Format kwargs (primary parameters for pipeline calls)
        if kwargs:
            for key, value in kwargs.items():
                formatted_value = self._format_call_value(value)
                lines.append(f"  {self._ANSI_DIM}{key}:{self._ANSI_RESET} {formatted_value}")

        # Format positional args (if any)
        if args:
            display_args = args[1:] if args and getattr(args[0], "__class__", None) else args
            if display_args:
                lines.append(f"  {self._ANSI_DIM}args:{self._ANSI_RESET} {self._format_call_value(display_args)}")

        if not kwargs and not args:
            lines.append(f"  {self._ANSI_DIM}(no parameters){self._ANSI_RESET}")

        lines.append(SEP)

        import sys

        print("\n".join(lines), file=sys.stderr)

    def _format_call_value(self, value: Any) -> str:
        """Format a __call__ parameter value for display.

        Args:
            value: Parameter value to format

        Returns:
            Formatted string representation
        """
        if isinstance(value, str):
            # Truncate long strings
            if len(value) > 80:
                return (
                    f"{self._ANSI_GREEN}{value[:80]}...{self._ANSI_RESET} "
                    f"{self._ANSI_DIM}({len(value)} chars){self._ANSI_RESET}"
                )
            return f"{self._ANSI_GREEN}{value}{self._ANSI_RESET}"
        elif isinstance(value, bool):
            color = self._ANSI_GREEN if value else self._ANSI_YELLOW
            return f"{color}{value}{self._ANSI_RESET}"
        elif isinstance(value, (int, float)):
            return f"{self._ANSI_BLUE}{value}{self._ANSI_RESET}"
        elif isinstance(value, torch.Tensor):
            return (
                f"{self._ANSI_CYAN}Tensor{self._ANSI_RESET}("
                f"{self._ANSI_DIM}shape={list(value.shape)}, dtype={value.dtype}{self._ANSI_RESET})"
            )
        elif isinstance(value, Image.Image):
            return f"{self._ANSI_CYAN}PIL.Image{self._ANSI_RESET}({self._ANSI_DIM}size={value.size}{self._ANSI_RESET})"
        elif isinstance(value, list):
            if len(value) > 5:
                return f"{self._ANSI_DIM}list({len(value)} items){self._ANSI_RESET}"
            formatted_items = [self._format_call_value(v) for v in value]
            return f"[{', '.join(formatted_items)}]"
        elif isinstance(value, tuple):
            if len(value) > 5:
                return f"{self._ANSI_DIM}tuple({len(value)} items){self._ANSI_RESET}"
            formatted_items = [self._format_call_value(v) for v in value]
            return f"({', '.join(formatted_items)})"
        elif isinstance(value, dict):
            if len(value) > 8:
                return f"{self._ANSI_DIM}dict({len(value)} items){self._ANSI_RESET}"
            # Recurse so inner Tensor/Image show as shape summary, not raw dump.
            formatted_items = [f"{k}={self._format_call_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(formatted_items) + "}"
        elif value is None:
            return f"{self._ANSI_DIM}None{self._ANSI_RESET}"
        return str(value)

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype) -> None:
        self.device = device
        self.torch_dtype = torch_dtype
        # VAE typically requires dimensions divisible by 16
        self.height_division_factor = 16
        self.width_division_factor = 16
        self._metrics_manager: StageMetricsManager | None = None

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection.

        Subclasses should override this method to return their stages.

        Returns:
            List of stage instances that support metrics collection.
        """
        return []

    def enable_metrics(self) -> None:
        """Enable metrics collection for all pipeline stages.

        This method enables metrics tracking for stages returned by _get_stages().

        Example:
            >>> pipeline = MyPipeline.from_pretrained(...)
            >>> pipeline.enable_metrics()
            >>> # Run inference with metrics enabled
            >>> result = pipeline(...)
            >>> # Get metrics
            >>> print(pipeline.get_prometheus_metrics())
        """
        if self._metrics_manager is not None:
            logger.warning("Metrics already enabled, skipping")
            return

        from telefuser.metrics import StageMetricsManager

        self._metrics_manager = StageMetricsManager()
        stages = self._get_stages()

        if stages:
            self._metrics_manager.enable_all_stages(stages)
            logger.info(f"Enabled metrics for stages: {self._metrics_manager.enabled_stages}")
        else:
            logger.warning("No stages available for metrics collection")

    def disable_metrics(self) -> None:
        """Disable metrics collection for all pipeline stages.

        Example:
            >>> pipeline.disable_metrics()
        """
        if self._metrics_manager is None:
            return

        self._metrics_manager.disable_all_stages()
        self._metrics_manager = None
        logger.info("Disabled metrics for all stages")

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus text exposition format.

        Returns:
            String containing all metrics in Prometheus format.

        Example:
            >>> metrics = pipeline.get_prometheus_metrics()
            >>> print(metrics)
            # HELP stage_vae_duration_seconds Execution duration
            # TYPE stage_vae_duration_seconds histogram
            ...

        Raises:
            RuntimeError: If metrics are not enabled.
        """
        if self._metrics_manager is None:
            raise RuntimeError("Metrics not enabled. Call enable_metrics() first.")
        return self._metrics_manager.registry.get_prometheus_format()

    @property
    def metrics_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self._metrics_manager is not None

    def check_resize_height_width(self, height: int, width: int) -> tuple[int, int]:
        """Ensure dimensions are divisible by division factors."""
        if height % self.height_division_factor != 0:
            factor = self.height_division_factor
            height = ((height + factor - 1) // factor) * factor
            print(f"Height rounded up to {height}")
        if width % self.width_division_factor != 0:
            factor = self.width_division_factor
            width = ((width + factor - 1) // factor) * factor
            print(f"Width rounded up to {width}")
        return height, width

    def preprocess_image(self, image: Image.Image, height: int | None = None, width: int | None = None) -> torch.Tensor:
        """Preprocess PIL image to tensor."""
        if height is not None and width is not None:
            if height != image.size[1] or width != image.size[0]:
                image = image.resize((width, height), Image.LANCZOS)
        return torch.Tensor(np.array(image, dtype=np.float32) * (2 / 255) - 1).permute(2, 0, 1).unsqueeze(0)

    def preprocess_images(
        self, images: Sequence[Image.Image], height: int | None = None, width: int | None = None
    ) -> list[torch.Tensor]:
        """Preprocess multiple images."""
        return [self.preprocess_image(image, height, width) for image in images]

    def tensor2video(
        self, frames: torch.Tensor, height: int | None = None, width: int | None = None
    ) -> list[Image.Image]:
        """Convert tensor to list of PIL images."""
        if height is not None and width is not None:
            if height != frames.shape[2] or width != frames.shape[3]:
                logger.info(f"Resizing video to {width}x{height}")
                # Bicubic antialias resize does not support bf16 inputs in PyTorch.
                resize_dtype = frames.dtype
                resize_frames = frames.float() if frames.dtype == torch.bfloat16 else frames
                frames = F.interpolate(
                    resize_frames, size=(height, width), mode="bicubic", align_corners=False, antialias=True
                )
                frames = frames.to(resize_dtype)
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    def generate_noise(
        self, shape: Sequence[int], seed: int | None = None, device: str = "cpu", dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Generate random noise."""
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)

    def dump_config(self, path: str | Path | None = None) -> dict[str, Any]:
        """Dump pipeline configuration to dict or file.

        Captures model definition (Layer 1) and inference algorithm
        parameters (Layer 2) for reproducibility and debugging.

        Args:
            path: Optional file path to save config (JSON format)

        Returns:
            Configuration dictionary with model and inference settings

        Example:
            >>> pipeline = MyPipeline.from_pretrained(...)
            >>> config = pipeline.dump_config()
            >>> pipeline.dump_config("output/config.json")
        """
        from .config_serializer import serialize_config

        config: dict[str, Any] = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "pipeline_type": self.__class__.__name__,
            "device": str(self.device),
            "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            "layer1_model_definition": {
                "models": self._model_info if hasattr(self, "_model_info") else [],
            },
            "layer2_inference_config": serialize_config(self.config) if hasattr(self, "config") else {},
        }

        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Config dumped to {path}")

        return config
