"""Parallel worker using multiprocessing for multi-GPU execution.

Spawns separate processes for distributed execution across multiple GPUs
with proper process group initialization.
"""

from __future__ import annotations

import gc
import os
import signal
import threading
import time
from collections.abc import Callable, Collection
from datetime import timedelta
from functools import partial
from multiprocessing.queues import SimpleQueue
from queue import Empty
from typing import TYPE_CHECKING, Any, BinaryIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing.spawn import ProcessContext

from telefuser.core.base_stage import BaseStage
from telefuser.platforms import BasePlatform, CpuPlatform, current_platform, requires_host_tensor_queue, resolve_device
from telefuser.platforms.hccl import reserve_hccl_ports
from telefuser.utils.logging import logger
from telefuser.utils.system import PortAllocator

from .tensor_channel import WorkerTensorChannel

if TYPE_CHECKING:
    from telefuser.metrics import StageMetricContext


_DISCARD_TENSOR_REFS = "__telefuser_discard_tensor_refs__"


def _raise_nofile_limit(target: int = 65536) -> None:
    """Raise soft RLIMIT_NOFILE toward target so multi-stage spawn does not exhaust 1024 container FDs."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = target if hard == resource.RLIM_INFINITY else min(hard, target)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (ImportError, OSError, ValueError):
        pass


def _worker_platform(stage: BaseStage) -> BasePlatform:
    if stage.device_type == "cpu" and current_platform.device_type != "cpu":
        return CpuPlatform()
    return current_platform


def to_device(data: Any, device: str | torch.device, *, preserve_tuples: bool = False) -> Any:
    """Move nested inputs; legacy callers receive lists for positional argument tuples."""
    if isinstance(data, dict):
        return {k: to_device(v, device, preserve_tuples=preserve_tuples) for k, v in data.items()}
    if isinstance(data, (tuple, list)):
        items = [to_device(v, device, preserve_tuples=preserve_tuples) for v in data]
        return tuple(items) if preserve_tuples and isinstance(data, tuple) else items
    if isinstance(data, torch.Tensor):
        tensor = data.to(device)
        if tensor.device.type == "cpu" and not tensor.is_shared():
            tensor.share_memory_()
        return tensor
    return data


def _worker_loop(
    rank: int,
    world_size: int,
    queue_in: list[SimpleQueue],
    queue_out: SimpleQueue,
    stage: BaseStage,
    master_port: int,
    tensor_output_channel: WorkerTensorChannel | None = None,
    tensor_input_channels: tuple[WorkerTensorChannel, ...] = (),
    hccl_if_base_port: int | None = None,
    *,
    rank_offset: int = 0,
) -> None:
    """Worker process main loop.

    Initializes distributed process group and processes tasks.
    """
    from telefuser.utils.profiler import mark_as_worker_process

    rank += rank_offset
    _raise_nofile_limit()
    mark_as_worker_process()
    previous_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    platform = _worker_platform(stage)
    args = None
    kwargs = None
    stage_inputs = None
    y = None
    try:
        parallel_config = stage.model_runtime_config.parallel_config
        # Avoid host-wide launch pools in every spawned CUDA worker, including
        # single-rank workers such as the LingBot condition encoder.
        torch.set_num_threads(parallel_config.worker_intra_op_threads)
        device = stage.device
        host_queue = requires_host_tensor_queue(platform.device_type)
        if world_size > 1:
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = str(master_port)
            if hccl_if_base_port is not None:
                os.environ["HCCL_IF_BASE_PORT"] = str(hccl_if_base_port)
                os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")

        if host_queue or world_size > 1:
            ids = parallel_config.device_ids
            device_id = rank if world_size > 1 else device.index or 0
            if ids is not None:
                device_id = ids[rank]
            device = torch.device(platform.device_type, device_id)
            stage.model_runtime_config.device_id = device_id
            stage.device = device
            if platform.device_type != "cpu":
                platform.set_device(device)
            logger.info(f"Worker {stage.name} rank {rank}/{world_size} bound to {device}")
        if world_size > 1:
            if platform.device_type == "cuda":
                # torch.distributed.all_reduce does not free the input tensor until
                # the synchronization point. This causes the memory usage to grow
                # as the number of all_reduce calls increases. This env var disables
                # this behavior.
                # Related issue:
                # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
                os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            timeout = timedelta(seconds=600)
            dist.init_process_group(
                backend=platform.dist_backend,
                init_method="env://",
                timeout=timeout,
                world_size=world_size,
                rank=rank,
                device_id=device if device.type != "cpu" else None,
            )
            stage.parallel_models()

        while True:
            data = queue_in[rank].get()
            if len(data) == 3:
                name, args, kwargs = data
                transport_output = False
            else:
                name, args, kwargs, transport_output = data
            del data
            if name == "exit":
                logger.info(f"parallel worker {stage.name} on rank {rank} exits")
                break
            if name == _DISCARD_TENSOR_REFS:
                discarded = 0
                for channel in tensor_input_channels:
                    if channel.contains_ref(args):
                        discarded += channel.discard(args, rank=rank)
                if world_size > 1:
                    dist.barrier()
                if world_size == 1 or rank == 0:
                    queue_out.put(discarded)
                continue
            if not hasattr(stage, name):
                raise AttributeError(f'{stage.__class__.__name__} has no attribute "{name}"')
            stage_inputs = (args, kwargs)
            for channel in tensor_input_channels:
                if channel.contains_ref(stage_inputs):
                    stage_inputs = channel.receive(stage_inputs, rank=rank, device=device)
            args, kwargs = stage_inputs
            kwargs = to_device(kwargs, device, preserve_tuples=host_queue)
            args = to_device(args, device, preserve_tuples=host_queue)
            with torch.no_grad():
                if not isinstance(args, (list, tuple)):
                    args = [args]
                y = getattr(stage, name)(*args, **kwargs)
            del kwargs, args
            if world_size > 1 and rank != 0:
                y = None
            if not host_queue and getattr(stage, "empty_cache_after_call", True):
                platform.empty_cache()
            if transport_output and tensor_output_channel is not None and (world_size == 1 or rank == 0):
                y = tensor_output_channel.send(y)
            # Always output results when world_size=1
            if world_size == 1 or rank == 0:
                if host_queue:
                    y = to_device(y, "cpu", preserve_tuples=True)
                queue_out.put(y)
                y = None
            if host_queue and getattr(stage, "empty_cache_after_call", True):
                platform.empty_cache()
    except Exception as e:
        import traceback

        traceback.print_exc()
        logger.error(f"Error in worker loop (rank {rank}): {e}")
        message = f"Parallel worker rank {rank} failed with {type(e).__name__}: {e}"
        try:
            worker_error = type(e)(message)
        except Exception:
            worker_error = RuntimeError(message)
        queue_out.put(worker_error)
    finally:
        args = None
        kwargs = None
        stage_inputs = None
        y = None
        if platform.device_type != "cpu":
            platform.synchronize()
            platform.ipc_collect()
        gc.collect()
        platform.empty_cache()
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        signal.signal(signal.SIGINT, previous_sigint_handler)


def _spawn_hccl_group(world_size: int, arguments: tuple) -> ProcessContext:
    """Keep handles to earlier ranks if spawning a later rank fails in the parent."""
    contexts = []
    try:
        for rank in range(world_size):
            context = mp.spawn(partial(_worker_loop, rank_offset=rank), args=arguments, nprocs=1, join=False)
            contexts.append(context)
        return ProcessContext(
            [context.processes[0] for context in contexts],
            [context.error_files[0] for context in contexts],
        )
    except BaseException:
        for context in contexts:
            process = context.processes[0]
            if process.is_alive():
                process.kill()
            process.join()
        raise


class ParallelWorker:
    """Multi-process worker for distributed execution across multiple GPUs."""

    _hccl_lock: BinaryIO | None = None
    _master_port: int | None = None
    ctx = None

    def __init__(
        self,
        stage: BaseStage,
        *,
        tensor_output_channel: WorkerTensorChannel | None = None,
        tensor_output_methods: Collection[str] = (),
        tensor_input_channels: Collection[WorkerTensorChannel] = (),
    ) -> None:
        parallel_config = stage.model_runtime_config.parallel_config
        parallel_config.validate()
        _raise_nofile_limit()

        self._stage: BaseStage = stage  # Keep reference to wrapped stage for metrics proxying
        self.world_size: int = parallel_config.world_size
        platform = _worker_platform(stage)
        self.device: str = platform.device_type
        self.device_ids: list[int] = parallel_config.device_ids
        if self.device_ids is None:
            self.device_ids = list(range(self.world_size))

        if stage.device_type in ("npu", "cuda") and stage.device_type != platform.device_type:
            resolve_device(stage.device)
        if self.device == "npu":
            if parallel_config.device_ids is None:
                self.device_ids = [stage.device.index or 0]
            if len(set(self.device_ids)) != len(self.device_ids):
                raise ValueError(f"NPU worker ranks require distinct device IDs, got {self.device_ids}")
            for idx in self.device_ids:
                resolve_device(f"npu:{idx}")

        self.name: str = f"Parallel Worker {stage.name}"
        # Queue transport requires CPU marshalling on platforms without reliable device IPC (e.g. NPU).
        self.queue_with_cpu: bool = parallel_config.queue_with_cpu or requires_host_tensor_queue(self.device)
        self.timeout: int = parallel_config.timeout
        self._lifecycle_lock = threading.Lock()
        self._failed = False
        self._closed = False
        self._failure_reason: str | None = None
        self.tensor_output_channel = tensor_output_channel
        self.tensor_output_methods = frozenset(tensor_output_methods)
        self.tensor_input_channels = tuple(tensor_input_channels)
        if self.tensor_output_channel is None and self.tensor_output_methods:
            raise ValueError("tensor_output_methods require a tensor_output_channel")
        if self.tensor_output_channel is not None:
            if not self.tensor_output_methods:
                raise ValueError("tensor_output_channel requires at least one tensor_output_method")
            self.tensor_output_channel.bind_producer()
        for channel in self.tensor_input_channels:
            channel.bind_consumer(self.world_size)

        # Use spawn to start processes regardless of world_size
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None or current_method != "spawn":
            try:
                mp.set_start_method("spawn", force=True)
            except RuntimeError as e:
                raise RuntimeError("Failed to set start method to spawn:", e)

        spawn_ctx = mp.get_context("spawn")
        # Queue uses a background feeder thread for every process. These messages
        # only carry small commands and shared-tensor handles, so synchronous
        # SimpleQueue writes avoid feeder scheduling tails between GPU stages.
        self.queue_in: list[SimpleQueue] = [spawn_ctx.SimpleQueue() for _ in range(self.world_size)]
        self.queue_out: SimpleQueue = spawn_ctx.SimpleQueue()

        try:
            self._master_port = PortAllocator().get_free_port_in_interval()
            hccl_port = None
            if self.world_size > 1 and platform.dist_backend == "hccl":
                hccl_port, self._hccl_lock = reserve_hccl_ports()
            logger.info(f"{self.name}: world_size={self.world_size}, master_port={self._master_port}, hccl={hccl_port}")
            arguments = (
                self.world_size,
                self.queue_in,
                self.queue_out,
                stage,
                self._master_port,
                self.tensor_output_channel,
                self.tensor_input_channels,
                hccl_port,
            )
            if hccl_port is not None:
                self.ctx = _spawn_hccl_group(self.world_size, arguments)
            else:
                self.ctx = mp.spawn(_worker_loop, args=arguments, nprocs=self.world_size, join=False)
        except BaseException:
            self.close()
            raise

    def _release_ports(self) -> None:
        with self._lifecycle_lock:
            if self._hccl_lock is not None:
                self._hccl_lock.close()
                self._hccl_lock = None
            if self._master_port is not None:
                PortAllocator.release_port(self._master_port)
                self._master_port = None

    @property
    def failed(self) -> bool:
        """Return whether this worker group encountered an unrecoverable failure."""
        return self._failed

    @property
    def closed(self) -> bool:
        """Return whether this worker group has been deterministically closed."""
        return self._closed

    @property
    def failure_reason(self) -> str | None:
        """Return the first failure that made this worker group unusable."""
        return self._failure_reason

    def _ensure_usable(self) -> None:
        if self._closed:
            raise RuntimeError(f"ParallelWorker:{self.name} is closed")
        if self._failed:
            raise RuntimeError(f"ParallelWorker:{self.name} has failed: {self._failure_reason}")

    def _terminate_processes(self) -> None:
        if self.ctx is None:
            return
        for process in self.ctx.processes:
            try:
                if process.is_alive():
                    process.kill()
                process.join(timeout=2)
            except Exception as exc:
                logger.warning(f"Failed to terminate {self.name} process: {exc}")

    def _mark_failed(self, reason: str) -> None:
        with self._lifecycle_lock:
            if self._failed:
                return
            self._failed = True
            self._failure_reason = reason
        logger.error(f"ParallelWorker:{self.name} marked failed: {reason}")
        self._terminate_processes()
        self._release_ports()

    def _wait_result(self, method_name: str) -> Any:
        try:
            # SimpleQueue does not expose a timeout argument. Its reader is the
            # underlying multiprocessing Connection and provides the same bounded
            # wait without reintroducing a feeder or polling thread.
            if not self.queue_out._reader.poll(self.timeout):
                raise Empty
            result = self.queue_out.get()
        except Empty as exc:
            reason = f"{method_name} timeout after {self.timeout} seconds"
            self._mark_failed(reason)
            raise RuntimeError(f"ParallelWorker:{self.name} {reason}") from exc
        except Exception as exc:
            reason = f"{method_name} result queue failed: {exc}"
            self._mark_failed(reason)
            raise RuntimeError(f"ParallelWorker:{self.name} {reason}") from exc
        if isinstance(result, Exception):
            reason = f"{method_name} failed: {result}"
            self._mark_failed(reason)
            raise RuntimeError(f"ParallelWorker:{self.name} {reason}") from result
        return result

    def enable_metrics(self, registry: Any | None = None) -> None:
        """Enable metrics collection on the wrapped stage.

        Note: Metrics are collected in worker subprocesses but aggregated
        metrics won't be visible in the main process. For distributed
        execution, consider using external metrics collection (e.g., Prometheus).
        """
        self._stage.enable_metrics(registry)

    def disable_metrics(self) -> None:
        """Disable metrics collection on the wrapped stage."""
        self._stage.disable_metrics()

    @property
    def _metrics_hook(self) -> StageMetricContext | None:
        """Proxy metrics hook from wrapped stage."""
        return self._stage._metrics_hook

    @_metrics_hook.setter
    def _metrics_hook(self, value: StageMetricContext | None) -> None:
        """Proxy metrics hook setter to wrapped stage."""
        self._stage._metrics_hook = value

    def put_data(self, data: Any) -> None:
        """Send data to all worker processes."""
        self._ensure_usable()
        if self.queue_with_cpu:
            data = to_device(data, "cpu", preserve_tuples=requires_host_tensor_queue(self.device))
        for q in self.queue_in:
            q.put(data)

    def discard_tensor_refs(self, value: Any, *, sync: bool = False) -> int | Callable[[], int]:
        """Release direct-channel tensors that will not be passed to the stage."""
        self._ensure_usable()
        self.put_data([_DISCARD_TENSOR_REFS, (value,), {}, False])

        def wait() -> int:
            return int(self._wait_result(_DISCARD_TENSOR_REFS))

        return wait() if sync else wait

    def __call__(self, *args: Any, **kwargs: Any) -> Any | Callable[[], Any]:
        """Submit __call__ task to all workers."""
        self._ensure_usable()
        sync = kwargs.pop("sync", False)
        transport_output = kwargs.pop("_tensor_transport", "__call__" in self.tensor_output_methods)
        data = ["__call__", args, kwargs, transport_output]
        self.put_data(data)

        def wait() -> Any:
            return self._wait_result("__call__")

        if sync:
            return wait()
        else:
            return wait

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """Submit arbitrary method call to all workers."""

        def wrapped_func(*args: Any, **kwargs: Any) -> Any | Callable[[], Any]:
            self._ensure_usable()
            sync = kwargs.pop("sync", False)
            transport_output = kwargs.pop("_tensor_transport", name in self.tensor_output_methods)
            data = [name, args, kwargs, transport_output]
            self.put_data(data)

            hook = self._metrics_hook
            if hook is not None:
                hook.enter()

            def wait() -> Any:
                start_time = time.perf_counter()
                success = False
                try:
                    result = self._wait_result(name)
                    success = True
                    logger.info(f"ParallelWorker:{self.name} {name} done")
                    return result
                finally:
                    if hook is not None:
                        duration = time.perf_counter() - start_time
                        hook.record_execution(duration, success=success)
                        hook.exit()

            if sync:
                return wait()
            else:
                return wait

        return wrapped_func

    def close(self) -> None:
        """Idempotently stop all ranks and close multiprocessing queues."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True

        if self.ctx is not None:
            if not self._failed:
                for queue in self.queue_in:
                    try:
                        queue.put(["exit", None, None])
                    except Exception as exc:
                        logger.warning(f"Failed to send exit to {self.name}: {exc}")
            for process in self.ctx.processes:
                try:
                    process.join(timeout=10)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=2)
                except Exception as exc:
                    logger.warning(f"Failed to join {self.name} process: {exc}")
        for queue in self.queue_in:
            try:
                queue.close()
            except Exception:
                pass
        try:
            self.queue_out.close()
        except Exception:
            pass
        self._release_ports()

    def __del__(self) -> None:
        """Best-effort fallback; callers should use close explicitly."""
        try:
            self.close()
        except Exception:
            pass
