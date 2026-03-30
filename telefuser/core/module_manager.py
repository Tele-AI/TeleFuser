"""Model loading and management utilities with automatic type detection."""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from telefuser.platforms import current_platform
from telefuser.utils.hf_utils import load_module_from_huggingface
from telefuser.utils.logging import logger
from telefuser.utils.model_weight import hash_state_dict_keys, init_weights_on_device, load_state_dict

from .model_config import model_loader_configs


def load_model_from_single_file(
    state_dict: dict[str, torch.Tensor],
    model_names: list[str],
    model_classes: list[type[nn.Module]],
    model_resource: str,
    torch_dtype: torch.dtype,
    device: str | torch.device,
    low_cpu_mem_usage: bool = False,
) -> tuple[list[str], list[nn.Module]]:
    """Load models from state dict with format conversion.

    Args:
        state_dict: Raw state dict from checkpoint file
        model_names: Names to assign to loaded models
        model_classes: Model classes to instantiate
        model_resource: Source format ("official" or "diffusers")
        torch_dtype: Target dtype for model weights
        device: Target device for model
        low_cpu_mem_usage: If True, keep weights in CPU memory until moved to device

    Returns:
        Tuple of (model_names, loaded_models)
    """
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        logger.info(f"Loading {model_name} ({model_class.__name__})")
        converter = model_class.state_dict_converter()

        # Convert state dict from source format
        if model_resource == "official":
            state_dict_results = converter.from_official(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = converter.from_diffusers(state_dict)
        else:
            raise ValueError(f"Unknown model_resource: {model_resource}")

        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}

        # Initialize model on meta device for low memory usage
        with init_weights_on_device("meta"):
            model = model_class(**extra_kwargs)

        # Enable quantization if needed
        if torch_dtype == torch.float8_e4m3fn:
            model.enable_quant(torch_dtype)
        if hasattr(model, "eval"):
            model = model.eval()
        model.requires_grad_(False)

        # Clone to CPU if not using low memory mode
        if not low_cpu_mem_usage:
            model_state_dict = {k: v.to("cpu").clone() for k, v in model_state_dict.items()}

        # Load weights and move to target device/dtype
        model.load_state_dict(model_state_dict, assign=True)
        if torch_dtype != torch.float8_e4m3fn:
            model = model.to(dtype=torch_dtype)
        model = model.to(device)

        loaded_model_names.append(model_name)
        loaded_models.append(model)

    return loaded_model_names, loaded_models


class ModelDetectorFromSingleFile:
    """Detect model type from state dict hash for automatic loading."""

    def __init__(self, configs: list = model_loader_configs) -> None:
        self.keys_hash_with_shape_dict: dict[str, tuple] = {}
        self.keys_hash_dict: dict[str, tuple] = {}
        for metadata in configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self,
        keys_hash: str | None,
        keys_hash_with_shape: str,
        model_names: list[str],
        model_classes: list[type],
        model_resource: str,
    ) -> None:
        """Register model metadata for detection by hash."""
        self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)
        if keys_hash is not None:
            self.keys_hash_dict[keys_hash] = (model_names, model_classes, model_resource)

    def match(self, file_path: str = "", state_dict: dict | None = None) -> bool:
        """Check if file matches known model patterns by hash."""
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if not state_dict:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            return True
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        return keys_hash in self.keys_hash_dict

    def load(
        self,
        file_path: str = "",
        state_dict: dict | None = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        low_cpu_mem_usage: bool = False,
    ) -> tuple[list[str], list[nn.Module]]:
        """Load model from file with automatic type detection."""
        if not state_dict:
            state_dict = load_state_dict(file_path)

        # Try matching with shape hash first (more specific)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape in self.keys_hash_with_shape_dict:
            model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
            return load_model_from_single_file(
                state_dict, model_names, model_classes, model_resource, torch_dtype, device, low_cpu_mem_usage
            )

        # Fall back to key-only hash
        keys_hash = hash_state_dict_keys(state_dict, with_shape=False)
        if keys_hash in self.keys_hash_dict:
            model_names, model_classes, model_resource = self.keys_hash_dict[keys_hash]
            return load_model_from_single_file(
                state_dict, model_names, model_classes, model_resource, torch_dtype, device, low_cpu_mem_usage
            )

        return [], []


class ModuleManager:
    """Manage loaded models and provide unified fetch interface."""

    def __init__(self, torch_dtype: torch.dtype = torch.float16, device: str | None = None) -> None:
        self.torch_dtype = torch_dtype
        self.device = device or current_platform.device_type
        self.modules: list[nn.Module] = []
        self.module_paths: list[str] = []
        self.module_names: list[str] = []
        self.model_detectors = [ModelDetectorFromSingleFile(model_loader_configs)]

    def load_models(
        self,
        file_path_list: str | list[str],
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        for file_path in file_path_list:
            self.load_model(file_path, device, torch_dtype, low_cpu_mem_usage)

    def load_model(
        self,
        file_path: str | list[str],
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        low_cpu_mem_usage: bool = False,
        name: str | None = None,
    ) -> None:
        """Load model from file path with automatic type detection.

        Args:
            file_path: Path to model checkpoint file(s)
            device: Target device (default: self.device)
            torch_dtype: Target dtype (default: self.torch_dtype)
            low_cpu_mem_usage: Keep weights in CPU memory until moved to device
            name: Override model name (default: None, uses name from model_config)
        """
        device = device or self.device
        torch_dtype = torch_dtype or self.torch_dtype

        # Merge state dicts if multiple files provided
        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = {}

        logger.info(f"Loading model from {file_path}")
        for detector in self.model_detectors:
            if detector.match(file_path, state_dict):
                model_names, models = detector.load(file_path, state_dict, device, torch_dtype, low_cpu_mem_usage)
                # Override model name if specified
                if name is not None:
                    model_names = [name] * len(models)
                for model_name, model in zip(model_names, models):
                    self.modules.append(model)
                    self.module_paths.append(file_path)
                    self.module_names.append(model_name)
                logger.info(f"Loaded models: {model_names}")
                break

    def fetch_module(
        self,
        model_name: str,
        file_path: str | None = None,
        require_model_path: bool = False,
        index: int | None = None,
    ) -> Any:
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.modules, self.module_paths, self.module_names):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)
        if len(fetched_models) == 0:
            logger.info(f"No {model_name} models available.")
            return None
        if len(fetched_models) == 1:
            logger.info(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        else:
            if index is None:
                model = fetched_models[0]
                path = fetched_model_paths[0]
                logger.info(
                    f"More than one {model_name} models are loaded in model "
                    f"manager: {fetched_model_paths}. Using {model_name} from "
                    f"{fetched_model_paths[0]}."
                )
            elif isinstance(index, int):
                model = fetched_models[:index]
                path = fetched_model_paths[:index]
                logger.info(
                    f"More than one {model_name} models are loaded in model "
                    f"manager: {fetched_model_paths}. Using {model_name} from "
                    f"{fetched_model_paths[:index]}."
                )
            else:
                model = fetched_models
                path = fetched_model_paths
                logger.info(
                    f"More than one {model_name} models are loaded in model "
                    f"manager: {fetched_model_paths}. Using {model_name} from "
                    f"{fetched_model_paths}."
                )
        if require_model_path:
            return model, path
        else:
            return model

    def load_from_huggingface(
        self,
        module_path: str,
        module_source: str = "transformers",
        module_name: str | None = None,
        module_class: type | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> None:
        """Load module from HuggingFace Hub."""
        device = device or self.device
        torch_dtype = torch_dtype or self.torch_dtype
        module, path, name = load_module_from_huggingface(
            module_path, module_source, module_name, module_class, device, torch_dtype, **kwargs
        )
        self.modules.append(module)
        self.module_paths.append(path)
        self.module_names.append(name)
        logger.info(f"Loaded {name} from {path}")

    def add_module(
        self,
        module: nn.Module,
        name: str,
        path: str = "manual",
    ) -> None:
        """Add an already-initialized module directly to the manager.

        This allows bypassing the automatic registration mechanism and
        adding modules that were initialized externally (e.g., text encoders,
        image encoders loaded from HuggingFace).

        Args:
            module: The nn.Module instance to add
            name: The name to register the module under
            path: Optional source path for tracking (default: "manual")
        """
        self.modules.append(module)
        self.module_paths.append(path)
        self.module_names.append(name)
        logger.info(f"Added module '{name}' to manager (source: {path})")

    def remove_module(self, name: str) -> bool:
        """Remove a module by name from the manager.

        Args:
            name: The name of the module to remove

        Returns:
            True if module was found and removed, False otherwise
        """
        indices_to_remove = []
        for i, module_name in enumerate(self.module_names):
            if module_name == name:
                indices_to_remove.append(i)

        for i in reversed(indices_to_remove):
            self.modules.pop(i)
            self.module_paths.pop(i)
            self.module_names.pop(i)

        if indices_to_remove:
            logger.info(f"Removed module '{name}' from manager")
            return True
        else:
            logger.warning(f"Module '{name}' not found in manager")
            return False

    def get_model_info(self) -> list[dict]:
        """Get information about loaded models for config dump.

        Returns:
            List of dicts with model name, path, and class name
        """
        return [
            {
                "name": name,
                "path": path if isinstance(path, str) else list(path),
                "class": model.__class__.__name__,
            }
            for name, path, model in zip(self.module_names, self.module_paths, self.modules)
        ]
