"""Inspect GPU capacity and estimate a safe LingBot-VLA v2 replica plan."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _query_gpus() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        index, name, total, used, free = fields
        gpus.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": float(total),
                "memory_used_mib": float(used),
                "memory_free_mib": float(free),
            }
        )
    return gpus


def _checkpoint_bytes(model_root: Path) -> int | None:
    candidates = sorted(model_root.rglob("*.safetensors.index.json"))
    if not candidates:
        return None
    index_path = candidates[0]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return None
    total = 0
    for shard in sorted(set(weight_map.values())):
        shard_path = index_path.parent / str(shard)
        if not shard_path.is_file():
            return None
        total += shard_path.stat().st_size
    return total


def build_report(model_root: Path, *, replica_memory_mib: float | None = None) -> dict[str, Any]:
    gpus = _query_gpus()
    checkpoint_bytes = _checkpoint_bytes(model_root)
    report: dict[str, Any] = {
        "schema_version": 1,
        "model": "lingbot-vla-v2-6b",
        "model_root": str(model_root.resolve()),
        "checkpoint_bytes_on_disk": checkpoint_bytes,
        "visible_gpus": gpus,
        "visible_gpu_count": len(gpus),
        "visible_total_memory_mib": sum(float(gpu["memory_total_mib"]) for gpu in gpus),
        "visible_free_memory_mib": sum(float(gpu["memory_free_mib"]) for gpu in gpus),
        "parallelism": {
            "single_gpu_replica_supported": True,
            "model_parallel_supported_by_current_pipeline": False,
            "request_level_replica_supported": True,
            "recommended_current_mode": "one full replica per GPU",
            "fsdp_or_tensor_parallel": (
                "requires a separate parity-preserving implementation and is not enabled by this tool"
            ),
        },
    }
    if replica_memory_mib is not None:
        usable = [gpu for gpu in gpus if gpu["memory_free_mib"] >= replica_memory_mib]
        report["replica_memory_mib"] = replica_memory_mib
        report["estimated_fit_replicas"] = len(usable)
        report["replica_fit_gpu_indexes"] = [gpu["index"] for gpu in usable]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--replica-memory-mib", type=float, help="Measured resident memory of one full replica.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replica_memory_mib is not None and args.replica_memory_mib <= 0:
        parser.error("--replica-memory-mib must be positive")
    report = build_report(args.model_root, replica_memory_mib=args.replica_memory_mib)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
