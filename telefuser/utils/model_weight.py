"""State dict loading utilities with memory-efficient initialization."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import torch
from safetensors import safe_open


@contextmanager
@torch.no_grad()
def init_weights_on_device(
    device: torch.device | str = torch.device("meta"), include_buffers: bool = False
) -> Generator[None, None, None]:
    """Context manager to initialize weights on specified device (default: meta).

    Useful for low-memory model initialization before loading state dict.
    """
    old_register_parameter = torch.nn.Module.register_parameter
    old_register_buffer = torch.nn.Module.register_buffer if include_buffers else None

    def register_empty_parameter(module: torch.nn.Module, name: str, param: torch.nn.Parameter | None) -> None:
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__.copy()
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(
        module: torch.nn.Module, name: str, buffer: torch.Tensor | None, persistent: bool = True
    ) -> None:
        if old_register_buffer:
            old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    tensor_constructors = (
        {name: getattr(torch, name) for name in ["empty", "zeros", "ones", "full"]} if include_buffers else {}
    )

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers and old_register_buffer:
            torch.nn.Module.register_buffer = register_empty_buffer
        for name in tensor_constructors:
            setattr(torch, name, patch_tensor_constructor(getattr(torch, name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers and old_register_buffer:
            torch.nn.Module.register_buffer = old_register_buffer
        for name, fn in tensor_constructors.items():
            setattr(torch, name, fn)


def load_state_dict(file_path: str, torch_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Load state dict from file (safetensors or bin)."""
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype)
    return load_state_dict_from_bin(file_path, torch_dtype)


def load_state_dict_from_safetensors(file_path: str, torch_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Load state dict from safetensors file."""
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            state_dict[k] = tensor.to(torch_dtype) if torch_dtype else tensor
    return state_dict


def load_state_dict_from_bin(file_path: str, torch_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Load state dict from PyTorch bin file."""
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype:
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor):
                state_dict[key] = state_dict[key].to(torch_dtype)
    return state_dict


def load_state_dict_from_folder(file_path: str, torch_dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Load and merge state dicts from folder containing multiple checkpoint files."""
    state_dict = {}
    extensions = ["safetensors", "bin", "ckpt", "pth", "pt"]
    for file_name in os.listdir(file_path):
        ext = file_name.split(".")[-1] if "." in file_name else ""
        if ext in extensions:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype))
    return state_dict


def hash_state_dict_keys(state_dict: dict[str, Any], with_shape: bool = True) -> str:
    """Generate MD5 hash from state dict keys for model identification."""
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape)
    return hashlib.md5(keys_str.encode("utf-8")).hexdigest()


def convert_state_dict_keys_to_single_str(state_dict: dict[str, Any], with_shape: bool = True) -> str:
    """Convert state dict keys to sorted string for hashing."""
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor) and with_shape:
                shape = "_".join(map(str, value.shape))
                keys.append(f"{key}:{shape}")
            keys.append(key)
        elif isinstance(value, dict):
            keys.append(f"{key}|{convert_state_dict_keys_to_single_str(value, with_shape)}")
    keys.sort()
    return ",".join(keys)
