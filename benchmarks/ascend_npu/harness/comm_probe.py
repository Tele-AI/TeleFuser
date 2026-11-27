#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate G0 (Container): Verify logical NPU mapping, fused NPU ops, HCCL collectives, and host queues."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from importlib.metadata import version
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from telefuser.platforms import current_platform


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _hccl_worker(rank: int, world_size: int, master_port: int, hccl_base_port: int, result_q: Any) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HCCL_IF_BASE_PORT"] = str(hccl_base_port)
    torch.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)
    try:
        dev = torch.device(f"npu:{rank}")
        val = torch.full((1024,), float(rank + 1), dtype=torch.float32, device=dev)
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) / 2.0
        assert torch.allclose(val, torch.full_like(val, expected)), f"all_reduce mismatch on rank {rank}"

        elems = ((32 * 1024 * 1024) // world_size) * world_size
        inp = torch.ones((elems,), dtype=torch.bfloat16, device=dev) * (rank + 1)
        out = torch.empty_like(inp)
        for _ in range(3):
            dist.all_to_all_single(out, inp)
        torch.npu.synchronize()
        iters = 10
        t0 = time.perf_counter()
        for _ in range(iters):
            dist.all_to_all_single(out, inp)
        torch.npu.synchronize()
        algbw_gbps = (elems * 2 * iters) / (time.perf_counter() - t0) / 1e9

        cpu_shared = out[:256].to("cpu").share_memory_()
        if rank == 0:
            result_q.put({"algbw_gbps": round(algbw_gbps, 2), "sample_sum": float(cpu_shared.float().sum().item())})
    finally:
        dist.destroy_process_group()


def main() -> None:
    """Run container-level Ascend NPU runtime and HCCL collective probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-cards", type=int, required=True)
    parser.add_argument("--hccl-base-port", type=int, default=61600)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    vis_env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    assert current_platform.device_type == "npu", f"Expected npu, got {current_platform.device_type}"
    assert current_platform.dist_backend == "hccl", f"Expected hccl, got {current_platform.dist_backend}"
    assert torch.npu.is_available(), "torch.npu.is_available() is False"
    actual = torch.npu.device_count()
    assert actual == args.expected_cards, f"Visible count {actual} != {args.expected_cards} ({vis_env!r})"

    devices_ok = []
    for idx in range(actual):
        x = torch.randn((16, 64, 128), dtype=torch.bfloat16, device=f"npu:{idx}")
        w = torch.ones((128,), dtype=torch.bfloat16, device=f"npu:{idx}")
        y, _ = torch_npu.npu_rms_norm(x, w, 1e-6)
        assert torch.isfinite(y.float()).all().item(), f"Non-finite npu_rms_norm on npu:{idx}"
        devices_ok.append({"logical_id": idx, "name": torch.npu.get_device_name(idx)})

    comm_result: dict[str, Any] = {"algbw_gbps": None}
    if actual > 1:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        mp.spawn(_hccl_worker, args=(actual, _pick_free_port(), args.hccl_base_port, queue), nprocs=actual, join=True)
        comm_result = queue.get(timeout=10)

    report = {
        "gate": "G0_PASS",
        "expected_cards": args.expected_cards,
        "telefuser_version": version("telefuser"),
        "torch_version": torch.__version__,
        "torch_npu_version": version("torch-npu"),
        "cann_version": os.environ.get("CANN_VERSION", "unknown"),
        "devices": devices_ok,
        "hccl_all_to_all": comm_result,
    }
    txt = json.dumps(report, indent=2)
    if args.out:
        Path_out = args.out
        with open(Path_out, "w", encoding="utf-8") as out_file:
            out_file.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
