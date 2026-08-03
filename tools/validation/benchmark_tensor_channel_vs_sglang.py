#!/usr/bin/env python3
"""Compare TeleFuser direct tensor handoff with SGLang's CUDA IPC pool."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.multiprocessing as mp

from telefuser.worker import WorkerTensorChannel


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported dtype: {name}")
    return value


def _telefuser_producer(
    channel: WorkerTensorChannel,
    commands: Any,
    references: Any,
    ready: Any,
    shape: tuple[int, ...],
    dtype_name: str,
    device_index: int,
) -> None:
    torch.cuda.set_device(device_index)
    source = torch.ones(shape, dtype=_dtype(dtype_name), device=f"cuda:{device_index}")
    torch.cuda.synchronize(device_index)
    ready.put(None)
    while commands.get() is not None:
        references.put(channel.send(source))


def _telefuser_consumer(
    channel: WorkerTensorChannel,
    commands: Any,
    completed: Any,
    ready: Any,
    device_index: int,
) -> None:
    torch.cuda.set_device(device_index)
    torch.empty(1, device=f"cuda:{device_index}")
    torch.cuda.synchronize(device_index)
    ready.put(None)
    try:
        while True:
            reference = commands.get()
            if reference is None:
                break
            output = channel.receive(reference, rank=0, device=f"cuda:{device_index}")
            torch.cuda.synchronize(device_index)
            completed.put((tuple(output.shape), output.numel() * output.element_size()))
    finally:
        channel.release_local_cuda_ipc()


def _sglang_producer(
    commands: Any,
    references: Any,
    ready: Any,
    shape: tuple[int, ...],
    dtype_name: str,
    device_index: int,
) -> None:
    import sglang.srt.utils.cuda_ipc_transport_utils as transport

    transport.get_server_args = lambda: SimpleNamespace(tp_size=1)
    torch.cuda.set_device(device_index)
    source = torch.ones(shape, dtype=_dtype(dtype_name), device=f"cuda:{device_index}")
    source_bytes = source.view(torch.int8).view(-1)
    pool = transport.MmItemMemoryPool(source_bytes.numel(), recycle_interval=60, base_gpu_id=device_index)
    torch.cuda.synchronize(device_index)
    ready.put(None)
    try:
        while commands.get() is not None:
            with pool._lock:
                pool.recycle_chunks()
                pool.merge_chunks()
            sync_meta, pool_slice, byte_offset = pool.return_a_slice_tensor_with_flag(source)
            if pool_slice is None:
                raise RuntimeError("SGLang IPC pool did not recycle its single benchmark slot")
            pool_slice.copy_(source_bytes, non_blocking=True)
            references.put(
                transport.CudaIpcTensorTransportProxy(
                    data=pool_slice,
                    info_data=source,
                    sync_buffer_meta=sync_meta,
                    pool_ipc_handle=pool._pool_ipc_handle,
                    pool_byte_offset=byte_offset,
                    pool_device_index=pool._pool_device_index,
                )
            )
    finally:
        pool.shutdown()
        pool.clear_sync_flag_list()


def _sglang_consumer(commands: Any, completed: Any, ready: Any, device_index: int) -> None:
    from sglang.srt.utils.cuda_ipc_transport_utils import _pool_handle_cache_clear

    torch.cuda.set_device(device_index)
    torch.empty(1, device=f"cuda:{device_index}")
    torch.cuda.synchronize(device_index)
    ready.put(None)
    try:
        while True:
            proxy = commands.get()
            if proxy is None:
                break
            output = proxy.reconstruct_on_target_device(device_index, consumer_count=1)
            torch.cuda.synchronize(device_index)
            completed.put((tuple(output.shape), output.numel() * output.element_size()))
    finally:
        _pool_handle_cache_clear()


def _run_path(
    producer_target: Any,
    consumer_target: Any,
    *,
    shape: tuple[int, ...],
    dtype_name: str,
    source_device: int,
    target_device: int,
    warmup: int,
    iterations: int,
    channel: WorkerTensorChannel | None = None,
) -> list[float]:
    context = mp.get_context("spawn")
    producer_commands = context.SimpleQueue()
    consumer_commands = context.SimpleQueue()
    references = context.SimpleQueue()
    completed = context.SimpleQueue()
    ready = context.SimpleQueue()
    common_producer_args = (producer_commands, references, ready, shape, dtype_name, source_device)
    producer_args = (channel, *common_producer_args) if channel is not None else common_producer_args
    common_consumer_args = (consumer_commands, completed, ready, target_device)
    consumer_args = (channel, *common_consumer_args) if channel is not None else common_consumer_args
    producer = context.Process(target=producer_target, args=producer_args)
    consumer = context.Process(target=consumer_target, args=consumer_args)
    producer.start()
    consumer.start()
    expected_nbytes = math.prod(shape) * torch.empty((), dtype=_dtype(dtype_name)).element_size()
    timings = []
    try:
        ready.get()
        ready.get()
        for index in range(warmup + iterations):
            started_at = time.perf_counter_ns()
            producer_commands.put(True)
            reference = references.get()
            consumer_commands.put(reference)
            output_shape, output_nbytes = completed.get()
            elapsed_ms = (time.perf_counter_ns() - started_at) / 1_000_000
            if output_shape != shape or output_nbytes != expected_nbytes:
                raise RuntimeError(
                    f"Transport returned shape={output_shape}, nbytes={output_nbytes}; "
                    f"expected shape={shape}, nbytes={expected_nbytes}"
                )
            if index >= warmup:
                timings.append(elapsed_ms)
    finally:
        consumer_commands.put(None)
        consumer.join(timeout=30)
        producer_commands.put(None)
        producer.join(timeout=30)
        for process in (consumer, producer):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for queue in (producer_commands, consumer_commands, references, completed, ready):
            queue.close()
        if channel is not None:
            channel.close()
    if producer.exitcode != 0 or consumer.exitcode != 0:
        raise RuntimeError(f"Transport workers failed: producer={producer.exitcode}, consumer={consumer.exitcode}")
    return timings


def _summary(timings: list[float], nbytes: int, copy_count: int) -> dict[str, float | int]:
    ordered = sorted(timings)
    p50_ms = statistics.median(ordered)
    p95_ms = ordered[math.ceil(0.95 * len(ordered)) - 1]
    return {
        "p50_ms": round(p50_ms, 4),
        "p95_ms": round(p95_ms, 4),
        "mean_ms": round(statistics.mean(ordered), 4),
        "logical_gib_per_second_at_p50": round(nbytes / (p50_ms / 1000) / (1024**3), 3),
        "device_copy_count": copy_count,
        "device_bytes_per_transfer": nbytes * copy_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="1,16,4,60,104")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--target-device", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-p50-ratio", type=float, default=1.05)
    parser.add_argument("--max-p95-ratio", type=float, default=1.10)
    parser.add_argument("--p95-jitter-ms", type=float, default=0.05)
    parser.add_argument(
        "--sglang-python",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "work_dirs" / "sglang" / "python",
    )
    args = parser.parse_args()
    shape = tuple(int(value) for value in args.shape.split(","))
    sglang_python = str(args.sglang_python.resolve())
    sys.path.append(sglang_python)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (sglang_python, os.environ.get("PYTHONPATH"))))
    nbytes = math.prod(shape) * torch.empty((), dtype=_dtype(args.dtype)).element_size()

    telefuser_timings = _run_path(
        _telefuser_producer,
        _telefuser_consumer,
        shape=shape,
        dtype_name=args.dtype,
        source_device=args.source_device,
        target_device=args.target_device,
        warmup=args.warmup,
        iterations=args.iterations,
        channel=WorkerTensorChannel(consumer_world_size=1, timeout=30),
    )
    sglang_timings = _run_path(
        _sglang_producer,
        _sglang_consumer,
        shape=shape,
        dtype_name=args.dtype,
        source_device=args.source_device,
        target_device=args.target_device,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    telefuser = _summary(telefuser_timings, nbytes, copy_count=2)
    sglang = _summary(sglang_timings, nbytes, copy_count=2)
    p50_ratio = float(telefuser["p50_ms"]) / float(sglang["p50_ms"])
    p95_ratio = float(telefuser["p95_ms"]) / float(sglang["p95_ms"])
    p95_limit_ms = max(
        float(sglang["p95_ms"]) * args.max_p95_ratio,
        float(sglang["p95_ms"]) + args.p95_jitter_ms,
    )
    result = {
        "shape": shape,
        "dtype": args.dtype,
        "bytes": nbytes,
        "source_device": args.source_device,
        "target_device": args.target_device,
        "iterations": args.iterations,
        "telefuser": telefuser,
        "sglang_cuda_ipc_pool": sglang,
        "telefuser_to_sglang_p50_ratio": round(p50_ratio, 4),
        "telefuser_to_sglang_p95_ratio": round(p95_ratio, 4),
        "p95_limit_ms": round(p95_limit_ms, 4),
        "passes": p50_ratio <= args.max_p50_ratio and float(telefuser["p95_ms"]) <= p95_limit_ms,
    }
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(
            "TeleFuser latency ratio exceeds the allowed limit: "
            f"p50={p50_ratio:.3f}/{args.max_p50_ratio:.3f}, "
            f"p95_ms={float(telefuser['p95_ms']):.4f}/{p95_limit_ms:.4f}"
        )


if __name__ == "__main__":
    main()
