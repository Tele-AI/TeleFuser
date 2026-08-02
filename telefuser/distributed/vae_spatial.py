"""Internal height-sharded helpers for causal VAE decoding."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _spatial_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _spatial_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _height_memory_format(tensor: torch.Tensor) -> torch.memory_format:
    if tensor.dim() == 5 and tensor.stride(1) == 1:
        return torch.channels_last_3d
    if tensor.dim() == 4 and tensor.stride(1) == 1:
        return torch.channels_last
    return torch.contiguous_format


def _split_height(tensor: torch.Tensor) -> torch.Tensor:
    world_size = _spatial_world_size()
    if world_size == 1:
        return tensor
    shard = torch.tensor_split(tensor, world_size, dim=-2)[_spatial_rank()]
    return shard.contiguous(memory_format=_height_memory_format(tensor))


def _gather_height_sizes(tensor: torch.Tensor) -> list[int]:
    world_size = _spatial_world_size()
    if world_size == 1:
        return [tensor.shape[-2]]
    local_height = torch.tensor([tensor.shape[-2]], dtype=torch.int64, device=tensor.device)
    gathered = [torch.empty_like(local_height) for _ in range(world_size)]
    dist.all_gather(gathered, local_height)
    return [int(height.item()) for height in gathered]


def _gather_height(tensor: torch.Tensor) -> torch.Tensor:
    world_size = _spatial_world_size()
    if world_size == 1:
        return tensor
    heights = _gather_height_sizes(tensor)
    max_height = max(heights)
    if tensor.shape[-2] < max_height:
        shape = list(tensor.shape)
        shape[-2] = max_height - tensor.shape[-2]
        tensor = torch.cat([tensor, tensor.new_zeros(shape)], dim=-2)
    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat([shard[..., :height, :] for shard, height in zip(gathered, heights)], dim=-2)


def _ensure_halo_buffer(buffer: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    memory_format = _height_memory_format(reference)
    if (
        buffer is None
        or buffer.shape != reference.shape
        or buffer.dtype != reference.dtype
        or buffer.device != reference.device
        or not buffer.is_contiguous(memory_format=memory_format)
    ):
        return torch.empty(reference.shape, dtype=reference.dtype, device=reference.device, memory_format=memory_format)
    return buffer


def _exchange_height_halo(module: nn.Module, tensor: torch.Tensor, halo_size: int) -> torch.Tensor:
    world_size = _spatial_world_size()
    if world_size == 1 or halo_size == 0:
        return tensor
    rank = _spatial_rank()
    top = tensor[..., :halo_size, :]
    bottom = tensor[..., -halo_size:, :]
    module._halo_recv_top = _ensure_halo_buffer(module._halo_recv_top, top)
    module._halo_recv_bottom = _ensure_halo_buffer(module._halo_recv_bottom, bottom)

    operations = []
    if rank > 0:
        operations.extend(
            [
                dist.P2POp(dist.irecv, module._halo_recv_top, rank - 1),
                dist.P2POp(dist.isend, top.contiguous(memory_format=_height_memory_format(top)), rank - 1),
            ]
        )
    if rank < world_size - 1:
        operations.extend(
            [
                dist.P2POp(dist.isend, bottom.contiguous(memory_format=_height_memory_format(bottom)), rank + 1),
                dist.P2POp(dist.irecv, module._halo_recv_bottom, rank + 1),
            ]
        )
    for request in dist.batch_isend_irecv(operations):
        request.wait()

    if rank == 0:
        module._halo_recv_top.zero_()
    if rank == world_size - 1:
        module._halo_recv_bottom.zero_()
    return torch.cat([module._halo_recv_top, tensor, module._halo_recv_bottom], dim=-2)


def _spatial_causal_conv3d_forward(
    module: nn.Conv3d,
    tensor: torch.Tensor,
    cache_tensor: torch.Tensor | None,
) -> torch.Tensor:
    padding = list(module._spatial_padding)
    if cache_tensor is not None and padding[4] > 0:
        cache_tensor = cache_tensor.to(tensor.device)
        tensor = torch.cat([cache_tensor, tensor], dim=2)
        padding[4] -= cache_tensor.shape[2]
    if any(padding):
        tensor = F.pad(tensor, padding)
    tensor = _exchange_height_halo(module, tensor, module._height_halo_size)
    tensor = tensor.contiguous(memory_format=torch.channels_last_3d)
    return F.conv3d(
        tensor,
        module.weight,
        module.bias,
        module.stride,
        (0, 0, 0),
        module.dilation,
        module.groups,
    )


class _SpatialParallelConv2d(nn.Conv2d):
    """Conv2d over a height shard with one halo exchange per invocation."""

    def __init__(self, source: nn.Conv2d) -> None:
        if source.stride[0] != 1:
            raise ValueError("VAE spatial decode only supports stride-one Conv2d layers")
        if source.padding_mode != "zeros":
            raise ValueError("VAE spatial decode only supports zero-padded Conv2d layers")
        super().__init__(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            stride=source.stride,
            padding=0,
            dilation=source.dilation,
            groups=source.groups,
            bias=source.bias is not None,
            padding_mode=source.padding_mode,
            device=source.weight.device,
            dtype=source.weight.dtype,
        )
        self.weight = source.weight
        self.bias = source.bias
        self._height_halo_size = source.dilation[0] * (source.kernel_size[0] - 1) // 2
        if source.padding[0] != self._height_halo_size:
            raise ValueError("VAE spatial Conv2d requires symmetric height padding")
        self._width_padding = source.padding[1]
        self._halo_recv_top: torch.Tensor | None = None
        self._halo_recv_bottom: torch.Tensor | None = None
        self.train(source.training)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._width_padding:
            tensor = F.pad(tensor, (self._width_padding, self._width_padding, 0, 0))
        tensor = _exchange_height_halo(self, tensor, self._height_halo_size)
        return F.conv2d(tensor, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)
