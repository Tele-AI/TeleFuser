"""Pipeline Parallel P2P Communication Module.

This module provides point-to-point communication utilities for Pipeline Parallelism,
enabling efficient communication between pipeline stages.

Pipeline Parallelism Algorithm:
1. Model is split across multiple stages (each GPU holds a subset of layers)
2. Stage 0 processes input, sends intermediate activations to Stage 1
3. Each stage receives from previous, processes, sends to next
4. Final stage produces output

Communication Pattern:
- Each stage sends to next rank, receives from previous rank
- Uses async send/recv for communication overlapping
- Supports send_recv for simultaneous communication

References:
- GPipe: https://arxiv.org/abs/1811.06965
- PipeDream: https://arxiv.org/abs/1806.03377
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from telefuser.platforms import current_platform
from telefuser.utils.logging import logger


class PipelineP2PComm:
    """Pipeline Parallel P2P communication manager.

    Manages point-to-point communication between pipeline stages.
    Each stage communicates only with its immediate neighbors (previous and next).

    Topology for N stages:
    - Stage 0 (first): receives from None, sends to Stage 1
    - Stage i: receives from Stage i-1, sends to Stage i+1
    - Stage N-1 (last): receives from Stage N-2, sends to None
    """

    def __init__(self, process_group: dist.ProcessGroup | None):
        self._process_group = process_group
        self._ops: list[dist.P2POp] = []
        self._reqs: list[dist.Work] | None = None

        if process_group is not None:
            self.rank = dist.get_rank(process_group)
            self.world_size = dist.get_world_size(process_group)
            self.is_first_stage = self.rank == 0
            self.is_last_stage = self.rank == self.world_size - 1

            # Get global ranks for P2P communication
            self.send_dst = dist.get_global_rank(process_group, (self.rank + 1) % self.world_size)
            self.recv_src = dist.get_global_rank(process_group, (self.rank - 1) % self.world_size)
        else:
            # Single GPU fallback
            self.rank = 0
            self.world_size = 1
            self.is_first_stage = True
            self.is_last_stage = True
            self.send_dst = 0
            self.recv_src = 0

    def _clear_state(self) -> None:
        """Clear internal state for next iteration."""
        self._reqs = None
        self._ops = []

    def send(self, tensor: torch.Tensor, async_op: bool = True) -> dist.Work | None:
        """Send tensor to the next pipeline stage.

        Args:
            tensor: Tensor to send (will be made contiguous)
            async_op: Whether to use async send (default True)

        Returns:
            dist.Work object if async_op=True, None otherwise
        """
        if self.is_last_stage:
            logger.warning("Last stage has no next stage to send to")
            return None

        tensor = tensor.contiguous()
        if async_op:
            return dist.isend(tensor, self.send_dst, group=self._process_group)
        else:
            dist.send(tensor, self.send_dst, group=self._process_group)
            return None

    def recv(
        self, buffer: torch.Tensor | None = None, shape: tuple | None = None, async_op: bool = True
    ) -> tuple[torch.Tensor, dist.Work | None]:
        """Receive tensor from the previous pipeline stage.

        Args:
            buffer: Optional pre-allocated buffer for receiving
            shape: Shape of tensor to receive (required if buffer is None)
            async_op: Whether to use async recv (default True)

        Returns:
            Tuple of (received tensor, dist.Work object if async_op=True else None)
        """
        if self.is_first_stage:
            raise RuntimeError("First stage has no previous stage to receive from")

        if buffer is None:
            if shape is None:
                raise ValueError("Either buffer or shape must be provided")
            buffer = torch.empty(shape, dtype=torch.float16, device=current_platform.device_type)

        buffer = buffer.contiguous()
        if async_op:
            work = dist.irecv(buffer, self.recv_src, group=self._process_group)
            return buffer, work
        else:
            dist.recv(buffer, self.recv_src, group=self._process_group)
            return buffer, None

    def send_recv(
        self,
        send_tensor: torch.Tensor,
        recv_buffer: torch.Tensor | None = None,
        recv_shape: tuple | None = None,
    ) -> torch.Tensor:
        """Send and receive tensors simultaneously.

        Overlaps send and recv operations for better efficiency.
        Uses batch_isend_irecv for optimized communication.

        Args:
            send_tensor: Tensor to send to next stage
            recv_buffer: Optional pre-allocated buffer for receiving
            recv_shape: Shape of tensor to receive (required if buffer is None)

        Returns:
            Received tensor from previous stage
        """
        ops = []

        # Setup send operation (if not last stage)
        if not self.is_last_stage:
            send_tensor = send_tensor.contiguous()
            send_op = dist.P2POp(dist.isend, send_tensor, self.send_dst, group=self._process_group)
            ops.append(send_op)

        # Setup receive operation (if not first stage)
        if not self.is_first_stage:
            if recv_buffer is None:
                if recv_shape is None:
                    raise ValueError("Either recv_buffer or recv_shape must be provided")
                recv_buffer = torch.empty(recv_shape, dtype=send_tensor.dtype, device=send_tensor.device)
            recv_buffer = recv_buffer.contiguous()
            recv_op = dist.P2POp(dist.irecv, recv_buffer, self.recv_src, group=self._process_group)
            ops.append(recv_op)

        # Execute all operations
        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        return recv_buffer

    def queue_send(self, tensor: torch.Tensor) -> None:
        """Queue a send operation for batch execution.

        Use with queue_recv() and commit() for overlapping multiple communications.

        Args:
            tensor: Tensor to send
        """
        if self.is_last_stage:
            return
        tensor = tensor.contiguous()
        send_op = dist.P2POp(dist.isend, tensor, self.send_dst, group=self._process_group)
        self._ops.append(send_op)

    def queue_recv(self, buffer: torch.Tensor) -> None:
        """Queue a receive operation for batch execution.

        Args:
            buffer: Buffer to receive data into
        """
        if self.is_first_stage:
            return
        buffer = buffer.contiguous()
        recv_op = dist.P2POp(dist.irecv, buffer, self.recv_src, group=self._process_group)
        self._ops.append(recv_op)

    def commit(self) -> None:
        """Execute all queued P2P operations as a batch."""
        if self._reqs is not None:
            raise RuntimeError("commit() called twice without wait()")
        if not self._ops:
            return
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self) -> None:
        """Wait for all pending operations to complete."""
        if self._reqs is None:
            return
        for req in self._reqs:
            req.wait()
        self._clear_state()

    def get_stage_indices(self, num_layers: int) -> tuple[int, int]:
        """Get the start and end layer indices for this pipeline stage.

        Distributes layers evenly across pipeline stages.

        Args:
            num_layers: Total number of layers in the model

        Returns:
            Tuple of (start_idx, end_idx) for this stage
        """
        layers_per_stage = num_layers // self.world_size
        remainder = num_layers % self.world_size

        # Distribute remainder layers to first 'remainder' stages
        if self.rank < remainder:
            start_idx = self.rank * (layers_per_stage + 1)
            end_idx = start_idx + layers_per_stage + 1
        else:
            start_idx = remainder * (layers_per_stage + 1) + (self.rank - remainder) * layers_per_stage
            end_idx = start_idx + layers_per_stage

        return start_idx, end_idx

    # ========== Convenience methods for latent communication ==========

    def send_latent(self, tensor: torch.Tensor) -> None:
        """Send latent tensor to the next pipeline stage.

        This is a convenience method that wraps send() for latent tensors.
        Blocks until send is complete.

        Args:
            tensor: Latent tensor to send
        """
        if self.is_last_stage:
            logger.warning("send_latent: Last stage has no next stage to send to")
            return

        tensor = tensor.contiguous()
        work = dist.isend(tensor, self.send_dst, group=self._process_group)
        work.wait()

    def recv_latent(self, shape: tuple | None = None, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Receive latent tensor from the previous pipeline stage.

        This is a convenience method that wraps recv() for latent tensors.
        Blocks until receive is complete.

        Args:
            shape: Shape of tensor to receive (can be inferred from cached shape)
            dtype: Data type of the tensor (default: bfloat16)

        Returns:
            Received latent tensor
        """
        if self.is_first_stage:
            raise RuntimeError("recv_latent: First stage has no previous stage to receive from")

        if shape is None:
            raise ValueError("recv_latent: shape must be provided")

        buffer = torch.empty(shape, dtype=dtype, device=current_platform.device_type)
        buffer = buffer.contiguous()
        work = dist.irecv(buffer, self.recv_src, group=self._process_group)
        work.wait()
        return buffer

    def send_latent_async(self, tensor: torch.Tensor) -> dist.Work | None:
        """Send latent tensor asynchronously to the next pipeline stage.

        Args:
            tensor: Latent tensor to send

        Returns:
            dist.Work object to wait on
        """
        if self.is_last_stage:
            return None

        tensor = tensor.contiguous()
        return dist.isend(tensor, self.send_dst, group=self._process_group)

    def recv_latent_async(self, shape: tuple, dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, dist.Work]:
        """Receive latent tensor asynchronously from the previous pipeline stage.

        Args:
            shape: Shape of tensor to receive
            dtype: Data type of the tensor

        Returns:
            Tuple of (buffer tensor, work object)
        """
        if self.is_first_stage:
            raise RuntimeError("recv_latent_async: First stage has no previous stage to receive from")

        buffer = torch.empty(shape, dtype=dtype, device=current_platform.device_type)
        buffer = buffer.contiguous()
        work = dist.irecv(buffer, self.recv_src, group=self._process_group)
        return buffer, work
