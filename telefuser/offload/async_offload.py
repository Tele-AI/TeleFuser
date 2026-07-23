"""Async layerwise CPU offloading with prefetching.

Adapted from SGLang for efficient transformer layer offloading.
Provides non-blocking H2D transfers and GPU buffer pooling.
"""

from __future__ import annotations

import math
from itertools import chain
from typing import Any, Dict, List, Set, Tuple

import torch

from telefuser.utils.logging import logger


class AsyncOffloadManager:
    """Layerwise CPU offload manager with async prefetching.

    Offloads transformer layer weights to CPU and prefetches them
    asynchronously using a dedicated CUDA stream.

    Typical usage:
    - Construct with transformer blocks
    - Call initialize() to offload and prefetch layer 0
    - During forward, prefetch_layer() for next layer and
      release_layer() for finished layer
    """

    def __init__(
        self,
        layers: torch.nn.ModuleList,
        device: torch.device | None = None,
        *,
        enabled: bool = True,
        pin_cpu_memory: bool = True,
        offload_ratio: float = 1,
        prefetch_size: int = 1,
        lazy_gpu_cache: bool = False,
    ) -> None:
        self.layers = layers
        self.num_layers = len(layers)
        self.pin_cpu_memory = pin_cpu_memory
        self.prefetch_size = min(max(1, prefetch_size), self.num_layers)
        self.num_resident_layers = min(
            max(self.prefetch_size, int(self.num_layers * (1 - offload_ratio))), self.num_layers
        )
        self.lazy_gpu_cache = lazy_gpu_cache

        self.enabled = bool(enabled and torch.cuda.is_available())
        if not self.enabled:
            return
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.copy_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

        # layer_idx -> {dtype: consolidated_pinned_cpu_tensor}
        self._consolidated_cpu_weights: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        # layer_idx -> {name: {dtype, offset, numel, shape}}
        self._weight_metadata: Dict[int, Dict[str, Dict[str, Any]]] = {}
        # layer indices that are already in gpu
        self._gpu_layers: Set[int] = set()
        # layer_idx -> torch.cuda.Event for fine-grained sync
        self._prefetch_events: Dict[int, torch.cuda.Event] = {}

        # Double buffer management for GPU memory reuse
        self._gpu_buffer_pool: Dict[torch.dtype, List[torch.Tensor]] = {}
        self._gpu_buffer_sizes: Dict[torch.dtype, List[int]] = {}
        self._layer_to_gpu_buffer: Dict[int, Dict[torch.dtype, torch.Tensor]] = {}
        self._gpu_buffer_ready_events: Dict[int, torch.cuda.Event] = {}

        self._named_parameters: Dict[str, torch.nn.Parameter] = {}
        self._named_buffers: Dict[str, torch.Tensor] = {}
        self._forward_hooks: List[Any] = []

        self._initialize()

    @torch.compiler.disable
    def _initialize(self) -> None:
        """Initialize offloading: consolidate weights and setup buffers."""
        if not self.enabled:
            return

        # Collect parameters and buffers from all layers
        self._named_parameters = {}
        self._named_buffers = {}

        for layer_idx, layer in enumerate(self.layers):
            for name, param in layer.named_parameters():
                full_name = f"layers.{layer_idx}.{name}"
                self._named_parameters[full_name] = param
            for name, buffer in layer.named_buffers():
                full_name = f"layers.{layer_idx}.{name}"
                self._named_buffers[full_name] = buffer

        # 1. collect and group tensors by layer and dtype
        layer_groups: Dict[int, Dict[torch.dtype, List[Tuple[str, torch.Tensor]]]] = {}

        for name, tensor in chain(self._named_parameters.items(), self._named_buffers.items()):
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "layers":
                try:
                    layer_idx = int(parts[1])
                    if layer_idx < self.num_layers:
                        layer_groups.setdefault(layer_idx, {}).setdefault(tensor.dtype, []).append((name, tensor))
                except ValueError:
                    continue

        # 2. concat and offload (in pinned memory)
        for layer_idx, dtype_to_params in layer_groups.items():
            self._consolidated_cpu_weights[layer_idx] = {}
            self._weight_metadata[layer_idx] = {}

            for dtype, weights in dtype_to_params.items():
                total_numel = sum(t.numel() for _, t in weights)

                cpu_buffer = torch.empty(total_numel, dtype=dtype, pin_memory=self.pin_cpu_memory)

                current_offset = 0
                for name, weight in weights:
                    numel = weight.numel()
                    cpu_buffer[current_offset : current_offset + numel].copy_(weight.flatten())
                    self._weight_metadata[layer_idx][name] = {
                        "dtype": dtype,
                        "offset": current_offset,
                        "numel": numel,
                        "shape": weight.shape,
                    }

                    weight.data = torch.empty((1,), device=self.device, dtype=dtype)

                    current_offset += numel

                self._consolidated_cpu_weights[layer_idx][dtype] = cpu_buffer

        # Initialize GPU buffer pool
        if not self.lazy_gpu_cache:
            self._initialize_gpu_buffer_pool()

        # prefetch the first layer for warm-up
        self.prepare_for_next_req(non_blocking=False)

        self.register_forward_hooks()
        logger.info(
            f"LayerwiseOffloadManager initialized with num prefetched layer: {self.prefetch_size}, "
            f"resident layers: {self.num_resident_layers} total num layers: {self.num_layers}"
        )

    def _initialize_gpu_buffer_pool(self) -> None:
        """Initialize GPU buffer pool with double buffering strategy."""
        if not self.enabled or self.device is None or self.device.type != "cuda":
            return

        buffer_sizes_by_dtype: Dict[torch.dtype, int] = {}

        for layer_idx in range(self.num_layers):
            if layer_idx in self._consolidated_cpu_weights:
                for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
                    current_max = buffer_sizes_by_dtype.get(dtype, 0)
                    buffer_sizes_by_dtype[dtype] = max(current_max, cpu_buffer.numel())

        for dtype, max_size in buffer_sizes_by_dtype.items():
            if max_size > 0:
                self._gpu_buffer_pool[dtype] = []
                self._gpu_buffer_sizes[dtype] = []

                for _ in range(2 * self.prefetch_size):
                    buffer = torch.empty(max_size, dtype=dtype, device=self.device)
                    self._gpu_buffer_pool[dtype].append(buffer)
                    self._gpu_buffer_sizes[dtype].append(max_size)

    def _ensure_gpu_buffer_pool_initialized(self) -> None:
        """Ensure GPU buffer pool is initialized (for lazy loading mode)."""
        if not self._gpu_buffer_pool:
            self._initialize_gpu_buffer_pool()

    def allocate_gpu_cache(self) -> None:
        """Manually allocate GPU cache (useful when lazy_gpu_cache=True)."""
        if not self.enabled:
            return
        self._initialize_gpu_buffer_pool()

    def cleanup_gpu_cache(self) -> None:
        """Release GPU cache to reduce VRAM usage."""
        if hasattr(self, "_gpu_buffer_pool"):
            self._cleanup_gpu_buffer_pool()

    def prepare_for_next_req(self, non_blocking: bool = True) -> None:
        """Prepare for the next round of denoising loop with prefetching."""
        for i in range(self.num_resident_layers):
            self.prefetch_layer(i, non_blocking=non_blocking)
        if not non_blocking and self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

    def get_target_with_name(self, name: str) -> torch.Tensor:
        """Get the target model weight/buffer to be replaced."""
        if name in self._named_parameters:
            target = self._named_parameters[name]
        else:
            target = self._named_buffers[name]
        return target

    @torch.compiler.disable
    def prefetch_layer(self, layer_idx: int, non_blocking: bool = True) -> None:
        """Prefetch layer weights from CPU to GPU.

        Non-blocking: Does NOT wait on current compute stream, allowing
        prefetch to start immediately during compute kernel execution.
        Consumer syncs via wait_event when the layer is needed.
        """
        if not self.enabled or self.device is None or self.copy_stream is None:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self._gpu_layers:
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return

        self._ensure_gpu_buffer_pool_initialized()

        gpu_buffers: Dict[torch.dtype, torch.Tensor] = {}
        with torch.cuda.stream(self.copy_stream):
            for dtype, cpu_buffer in self._consolidated_cpu_weights[layer_idx].items():
                if layer_idx < self.num_resident_layers:
                    gpu_buffer = torch.empty(cpu_buffer.shape, dtype=dtype, device=self.device)
                else:
                    gpu_buffer, ready_event = self._get_gpu_buffer(dtype, cpu_buffer.numel())
                    if gpu_buffer is None:
                        gpu_buffer = torch.empty(cpu_buffer.shape, dtype=dtype, device=self.device)
                    elif ready_event is not None:
                        self.copy_stream.wait_event(ready_event)

                if gpu_buffer.numel() == cpu_buffer.numel():
                    gpu_buffer.copy_(cpu_buffer, non_blocking=non_blocking)
                else:
                    gpu_buffer_slice = gpu_buffer[: cpu_buffer.numel()]
                    gpu_buffer_slice.copy_(cpu_buffer, non_blocking=non_blocking)
                gpu_buffers[dtype] = gpu_buffer

                if layer_idx >= self.num_resident_layers:
                    self._layer_to_gpu_buffer.setdefault(layer_idx, {})[dtype] = gpu_buffer

        event = torch.cuda.Event()
        event.record(self.copy_stream)
        self._prefetch_events[layer_idx] = event

        for name, meta in self._weight_metadata[layer_idx].items():
            dtype = meta["dtype"]
            gpu_buffer = gpu_buffers[dtype]

            target = self.get_target_with_name(name)
            target.data = gpu_buffer[meta["offset"] : meta["offset"] + meta["numel"]].view(meta["shape"])

        self._gpu_layers.add(layer_idx)

    @torch.compiler.disable
    def release_layer(self, layer_idx: int) -> None:
        """Release layer weights from GPU."""
        if not self.enabled or self.device is None:
            return

        self._prefetch_events.pop(layer_idx, None)

        if layer_idx < self.num_resident_layers:
            return

        if layer_idx not in self._gpu_layers:
            return

        if layer_idx >= self.num_resident_layers and layer_idx in self._layer_to_gpu_buffer:
            for dtype, gpu_buffer in self._layer_to_gpu_buffer[layer_idx].items():
                compute_done = torch.cuda.Event()
                compute_done.record(torch.cuda.current_stream(self.device))
                self._return_gpu_buffer(dtype, gpu_buffer, ready_event=compute_done)
            del self._layer_to_gpu_buffer[layer_idx]

        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            target.data = torch.empty((1,), device=self.device, dtype=meta["dtype"])

        self._gpu_layers.discard(layer_idx)

    def _get_gpu_buffer(
        self, dtype: torch.dtype, required_size: int
    ) -> tuple[torch.Tensor | None, torch.cuda.Event | None]:
        """Get a GPU buffer and the event that makes it safe to overwrite."""
        if dtype not in self._gpu_buffer_pool:
            return None, None

        for i, (buffer, size) in enumerate(zip(self._gpu_buffer_pool[dtype], self._gpu_buffer_sizes[dtype])):
            if size >= required_size:
                buffer = self._gpu_buffer_pool[dtype].pop(i)
                self._gpu_buffer_sizes[dtype].pop(i)
                return buffer, self._gpu_buffer_ready_events.pop(id(buffer), None)

        return None, None

    def _return_gpu_buffer(
        self, dtype: torch.dtype, buffer: torch.Tensor, ready_event: torch.cuda.Event | None = None
    ) -> None:
        """Return a GPU buffer to the pool after its compute-stream use completes."""
        if dtype not in self._gpu_buffer_pool:
            self._gpu_buffer_pool[dtype] = []
            self._gpu_buffer_sizes[dtype] = []

        self._gpu_buffer_pool[dtype].append(buffer)
        if ready_event is not None:
            self._gpu_buffer_ready_events[id(buffer)] = ready_event
        self._gpu_buffer_sizes[dtype].append(buffer.numel())

    @torch.compiler.disable
    def release_all(self) -> None:
        """Release all non-resident layers."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            if layer_idx >= self.num_resident_layers:
                self.release_layer(layer_idx)

        for layer_idx in list(self._layer_to_gpu_buffer.keys()):
            if layer_idx not in self._gpu_layers:
                del self._layer_to_gpu_buffer[layer_idx]

    @torch.compiler.disable
    def load_all_layers(self) -> None:
        """Load all layers from CPU to GPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in range(self.num_layers):
            if layer_idx not in self._gpu_layers:
                self.prefetch_layer(layer_idx, non_blocking=False)

    @torch.compiler.disable
    def sync_layer_to_cpu(self, layer_idx: int) -> None:
        """Sync a layer's weights from GPU back to CPU."""
        if not self.enabled or layer_idx not in self._gpu_layers:
            return
        if layer_idx not in self._consolidated_cpu_weights:
            return

        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for name, meta in self._weight_metadata.get(layer_idx, {}).items():
            target = self.get_target_with_name(name)
            gpu_weight = target.data.flatten().cpu()

            dtype = meta["dtype"]
            cpu_buffer = self._consolidated_cpu_weights[layer_idx][dtype]
            offset = meta["offset"]
            numel = meta["numel"]
            cpu_buffer[offset : offset + numel].copy_(gpu_weight)

    @torch.compiler.disable
    def sync_all_layers_to_cpu(self) -> None:
        """Sync all loaded layers' weights from GPU back to CPU."""
        if not self.enabled or self.device is None:
            return
        if self.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(self.copy_stream)

        for layer_idx in list(self._gpu_layers):
            if layer_idx >= self.num_resident_layers:
                self.sync_layer_to_cpu(layer_idx)

    def register_forward_hooks(self) -> None:
        """Register forward hooks for automatic prefetch/release."""
        if not self.enabled:
            return

        def make_pre_hook(i: int):
            def hook(module: torch.nn.Module, input: tuple[Any, ...]) -> None:
                if i in self._prefetch_events:
                    torch.cuda.current_stream().wait_event(self._prefetch_events[i])

                if i % self.prefetch_size == 0:
                    for j in range(i + self.prefetch_size, i + 2 * self.prefetch_size):
                        layer_to_prefetch = j % self.num_layers
                        self.prefetch_layer(layer_to_prefetch, non_blocking=True)

            return hook

        def make_post_hook(i: int):
            def hook(module: torch.nn.Module, input: tuple[Any, ...], output: Any) -> None:
                self.release_layer(i)

            return hook

        self._forward_hooks.clear()
        for i, layer in enumerate(self.layers):
            pre_hook_handle = layer.register_forward_pre_hook(make_pre_hook(i))
            post_hook_handle = layer.register_forward_hook(make_post_hook(i))
            self._forward_hooks.extend([pre_hook_handle, post_hook_handle])

    def remove_forward_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook_handle in self._forward_hooks:
            hook_handle.remove()
        self._forward_hooks.clear()

    def disable_offload(self) -> None:
        """Disable layerwise offload: load all layers to GPU and remove hooks."""
        if self.enabled:
            self.remove_forward_hooks()
            self.load_all_layers()
            self._cleanup_gpu_buffer_pool()

    def _cleanup_gpu_buffer_pool(self) -> None:
        """Clean up GPU buffer pool."""
        self._gpu_buffer_pool.clear()
        self._gpu_buffer_sizes.clear()
        self._layer_to_gpu_buffer.clear()
        self._gpu_buffer_ready_events.clear()

    def enable_offload(self) -> None:
        """Re-enable layerwise offload."""
        if self.enabled:
            self.sync_all_layers_to_cpu()
            self.release_all()
            self.register_forward_hooks()
