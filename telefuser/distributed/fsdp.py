"""FSDP (Fully Sharded Data Parallel) utilities.

Provides model sharding using PyTorch's FSDP for data parallel training.
Supports both FSDP1 and FSDP2 APIs with automatic wrapping policies.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

import torch
from torch import nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.tensor import DeviceMesh


def shard_model(
    module: nn.Module,
    device_id: int,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    wrap_module_names: list[str] | None = None,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.bfloat16,
    buffer_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = False,
    ignored_states: Iterable[nn.Parameter] | None = None,
) -> FSDP:
    """Shard model using FSDP1.

    Args:
        module: Model to shard
        device_id: Target device ID
        sharding_strategy: FSDP sharding strategy
        wrap_module_names: Module names to wrap as FSDP units
        param_dtype: Parameter dtype
        reduce_dtype: Gradient reduction dtype
        buffer_dtype: Buffer dtype
        cpu_offload: Whether to offload parameters to CPU
        ignored_states: Parameters to retain as unsharded, replicated state.

    Returns:
        FSDP-wrapped module
    """
    wrap_module_names = wrap_module_names or []
    mixed_precision = MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        cast_root_forward_inputs=False,  # Prevent FSDP from converting inputs
        cast_forward_inputs=False,
    )

    def wrap_fn(m: nn.Module) -> bool:
        """Check if module should be wrapped as FSDP unit."""
        for name in wrap_module_names:
            submodule = getattr(module, name)
            if isinstance(submodule, nn.ModuleList) and m in submodule:
                return True
            elif not isinstance(submodule, nn.ModuleList) and m is submodule:
                return True
        return False

    return FSDP(
        module,
        device_id=device_id,
        sharding_strategy=sharding_strategy,
        use_orig_params=False,
        mixed_precision=mixed_precision,
        forward_prefetch=True,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=wrap_fn),
        cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
        ignored_states=ignored_states,
    )


def shard_model_fsdp2(
    module: nn.Module,
    device_id: int,  # Kept for API compatibility, but uses DeviceMesh internally
    wrap_module_names: list[str] | None = None,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.bfloat16,
    buffer_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Shard model using FSDP2 (composable API).

    FSDP2 uses DeviceMesh for sharding configuration and provides better
    composability with other parallel strategies.

    Args:
        module: Model to shard
        device_id: Target device ID (for API compatibility)
        wrap_module_names: Module names to wrap as FSDP units
        param_dtype: Parameter dtype
        reduce_dtype: Gradient reduction dtype
        buffer_dtype: Buffer dtype

    Returns:
        Module with FSDP2 applied
    """
    wrap_module_names = wrap_module_names or []

    # Create 1D device mesh for data parallelism
    world_size = torch.distributed.get_world_size()
    device_mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dp",))

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )

    def _wrap_fn(m: nn.Module, name: str = "") -> bool:
        """Check if module matches wrap criteria by name."""
        for target_name in wrap_module_names:
            if target_name == name or target_name in name:
                return True
        return False

    def _is_container_module(m: nn.Module) -> bool:
        """Check if module is a container (no custom forward)."""
        container_types = (nn.ModuleList, nn.ModuleDict, nn.Sequential, nn.ParameterList, nn.ParameterDict)
        if isinstance(m, container_types):
            return True
        # Check for default forward (no custom implementation)
        if hasattr(m, "forward") and type(m).forward == nn.Module.forward:
            return True
        return False

    # Apply FSDP2 recursively in bottom-up order
    def _apply_fsdp_recursive(current_module: nn.Module, module_path: str = "") -> None:
        for name, child_module in current_module.named_children():
            child_path = f"{module_path}.{name}" if module_path else name
            _apply_fsdp_recursive(child_module, child_path)

            # Wrap non-container modules matching criteria
            if _wrap_fn(child_module, child_path) and not _is_container_module(child_module):
                fully_shard(child_module, mesh=device_mesh, mp_policy=mp_policy)

    _apply_fsdp_recursive(module)

    # Finally wrap root module
    fully_shard(module, mesh=device_mesh, mp_policy=mp_policy)

    return module


def shard_model_fsdp2_inference(
    module: nn.Module,
    device_mesh: DeviceMesh,
    wrap_module_names: list[str],
    ignored_states: Iterable[nn.Parameter] | None = None,
) -> nn.Module:
    """Apply source-style composable FSDP2 for inference-only model sharding."""
    ignored_params = set(ignored_states or ())
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        for submodule in module.modules():
            for name, buffer in tuple(submodule.named_buffers(recurse=False)):
                if buffer is not None and buffer.device != device:
                    submodule._buffers[name] = buffer.to(device=device)
        for parameter in ignored_params:
            if parameter.device != device:
                parameter.data = parameter.data.to(device=device)

    wrapped: set[nn.Module] = set()
    for name in wrap_module_names:
        target = getattr(module, name)
        targets = target if isinstance(target, nn.ModuleList) else (target,)
        for child in targets:
            child_ignored_params = {parameter for parameter in child.parameters() if parameter in ignored_params}
            fully_shard(child, mesh=device_mesh, ignored_params=child_ignored_params)
            wrapped.add(child)
    fully_shard(module, mesh=device_mesh, ignored_params=ignored_params)
    return module
