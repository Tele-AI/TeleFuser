"""Sequential CPU offloading for memory-constrained inference.

Wraps individual layers to move parameters between CPU and GPU on demand.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from telefuser.utils.model_weight import init_weights_on_device


def cast_to(
    weight: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device | str,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Cast tensor to target dtype/device with optional pinned memory."""
    if pin_memory and device == "cuda":
        # Pin in CPU first, then move to CUDA for faster transfer
        cpu_weight = weight.detach().cpu().pin_memory()
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(cpu_weight, non_blocking=True)
    else:
        r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight)
    return r


class AutoTorchModule(torch.nn.Module):
    """Base class for auto-offloading modules."""

    def __init__(self) -> None:
        super().__init__()

    def check_free_vram(self) -> bool:
        """Check if GPU has available memory below limit."""
        gpu_mem_state = torch.cuda.mem_get_info(self.computation_device)
        used_memory = (gpu_mem_state[1] - gpu_mem_state[0]) / (1024**3)
        return used_memory < self.vram_limit

    def offload(self) -> None:
        """Move module to offload device."""
        if self.state != 0:
            self.to(dtype=self.offload_dtype, device=self.offload_device)
            self.state = 0

    def onload(self) -> None:
        """Move module to onload device."""
        if self.state != 1:
            self.to(dtype=self.onload_dtype, device=self.onload_device)
            self.state = 1

    def keep(self) -> None:
        """Move module to computation device."""
        if self.state != 2:
            self.to(dtype=self.computation_dtype, device=self.computation_device)
            self.state = 2


class AutoWrappedModule(AutoTorchModule):
    """Wrapper for entire modules with auto-offloading."""

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype: torch.dtype,
        offload_device: torch.device | str,
        onload_dtype: torch.dtype,
        onload_device: torch.device | str,
        computation_dtype: torch.dtype,
        computation_device: torch.device | str,
        vram_limit: float | None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.module = module.to(dtype=offload_dtype, device=offload_device)
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward with automatic device management."""
        if self.state == 2:
            module = self.module
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                module = self.module
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                module = self.module
            else:
                module = copy.deepcopy(self.module).to(dtype=self.computation_dtype, device=self.computation_device)
        return module(*args, **kwargs)


class WanAutoCastLayerNorm(torch.nn.LayerNorm, AutoTorchModule):
    """LayerNorm with auto-offloading support."""

    def __init__(
        self,
        module: torch.nn.Module,
        offload_dtype: torch.dtype,
        offload_device: torch.device | str,
        onload_dtype: torch.dtype,
        onload_device: torch.device | str,
        computation_dtype: torch.dtype,
        computation_device: torch.device | str,
        vram_limit: float | None,
        **kwargs: Any,
    ) -> None:
        # Support both torch.nn.LayerNorm (normalized_shape) and custom LayerNorm (weight.shape)
        normalized_shape = getattr(module, "normalized_shape", None)
        if normalized_shape is None:
            # For custom LayerNorm, get dimension from weight shape
            weight = getattr(module, "weight", None)
            if weight is not None:
                normalized_shape = weight.shape[0]
            else:
                # Fallback: check bias shape for elementwise_affine=False with bias
                bias = getattr(module, "bias", None)
                if bias is not None:
                    normalized_shape = bias.shape[0]
                else:
                    # For elementwise_affine=False case, use a placeholder
                    # The actual normalized_shape will be inferred from input during forward
                    normalized_shape = 1  # placeholder, will be overridden by input shape
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                normalized_shape,
                eps=module.eps,
                elementwise_affine=module.elementwise_affine,
                bias=module.bias is not None,
                dtype=offload_dtype,
                device=offload_device,
            )
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward with automatic weight loading."""
        if self.state == 2:
            weight, bias = self.weight, self.bias
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                weight, bias = self.weight, self.bias
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                weight, bias = self.weight, self.bias
            else:
                weight = (
                    None
                    if self.weight is None
                    else cast_to(self.weight, self.computation_dtype, self.computation_device)
                )
                bias = (
                    None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
                )
        # Use input shape for normalized_shape when weight is None (elementwise_affine=False)
        normalized_shape = (x.shape[-1],) if weight is None else self.normalized_shape
        with torch.amp.autocast(device_type=x.device.type):
            x = torch.nn.functional.layer_norm(x.float(), normalized_shape, weight, bias, self.eps).type_as(x)
        return x


class AutoWrappedLinear(torch.nn.Linear, AutoTorchModule):
    """Linear layer with auto-offloading support."""

    def __init__(
        self,
        module: torch.nn.Linear,
        offload_dtype: torch.dtype,
        offload_device: torch.device | str,
        onload_dtype: torch.dtype,
        onload_device: torch.device | str,
        computation_dtype: torch.dtype,
        computation_device: torch.device | str,
        vram_limit: float | None,
        name: str = "",
        **kwargs: Any,
    ) -> None:
        with init_weights_on_device(device=torch.device("meta")):
            super().__init__(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                dtype=offload_dtype,
                device=offload_device,
            )
        self.weight = module.weight
        self.bias = module.bias
        self.offload_dtype = offload_dtype
        self.offload_device = offload_device
        self.onload_dtype = onload_dtype
        self.onload_device = onload_device
        self.computation_dtype = computation_dtype
        self.computation_device = computation_device
        self.vram_limit = vram_limit
        self.state = 0
        self.name = name
        self.lora_merger = None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Forward with automatic weight loading."""
        # VRAM management
        if self.state == 2:
            weight, bias = self.weight, self.bias
        else:
            if self.onload_dtype == self.computation_dtype and self.onload_device == self.computation_device:
                weight, bias = self.weight, self.bias
            elif self.vram_limit is not None and self.check_free_vram():
                self.keep()
                weight, bias = self.weight, self.bias
            else:
                weight = cast_to(self.weight, self.computation_dtype, self.computation_device)
                bias = (
                    None if self.bias is None else cast_to(self.bias, self.computation_dtype, self.computation_device)
                )

        # Linear forward
        out = torch.nn.functional.linear(x, weight, bias)
        return out


def enable_sequential_cpu_offload_recursively(
    model: torch.nn.Module,
    module_map: dict[type, type],
    module_config: dict[str, Any],
    max_num_param: int | None = None,
    overflow_module_config: dict[str, Any] | None = None,
    total_num_param: int = 0,
    vram_limit: float | None = None,
    name_prefix: str = "",
) -> int:
    """Recursively wrap modules for sequential CPU offloading.

    Args:
        model: Model to wrap
        module_map: Mapping from source module types to target wrappers
        module_config: Configuration for wrapped modules
        max_num_param: Max parameters before using overflow config
        overflow_module_config: Alternative config for large modules
        total_num_param: Running count of wrapped parameters
        vram_limit: VRAM limit in GB
        name_prefix: Module name prefix for nesting
    """
    for name, module in model.named_children():
        layer_name = name if name_prefix == "" else name_prefix + "." + name
        for source_module, target_module in module_map.items():
            if isinstance(module, source_module):
                num_param = sum(p.numel() for p in module.parameters())
                if max_num_param is not None and total_num_param + num_param > max_num_param:
                    module_config_ = overflow_module_config
                else:
                    module_config_ = module_config
                module_ = target_module(module, **module_config_, vram_limit=vram_limit, name=layer_name)
                setattr(model, name, module_)
                total_num_param += num_param
                break
        else:
            total_num_param = enable_sequential_cpu_offload_recursively(
                module,
                module_map,
                module_config,
                max_num_param,
                overflow_module_config,
                total_num_param,
                vram_limit=vram_limit,
                name_prefix=layer_name,
            )
    return total_num_param


def enable_sequential_cpu_offload(
    model: torch.nn.Module,
    module_map: dict[type, type],
    module_config: dict[str, Any],
    max_num_param: int | None = None,
    overflow_module_config: dict[str, Any] | None = None,
    vram_limit: float | None = None,
) -> None:
    """Enable sequential CPU offloading for a model.

    Args:
        model: Model to enable offloading for
        module_map: Mapping from source module types to target wrappers
        module_config: Configuration for wrapped modules
        max_num_param: Max parameters before using overflow config
        overflow_module_config: Alternative config for large modules
        vram_limit: VRAM limit in GB
    """
    enable_sequential_cpu_offload_recursively(
        model,
        module_map,
        module_config,
        max_num_param,
        overflow_module_config,
        total_num_param=0,
        vram_limit=vram_limit,
    )
    model.sequential_cpu_offload_enabled = True
