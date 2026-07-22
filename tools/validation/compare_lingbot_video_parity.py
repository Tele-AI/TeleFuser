"""Compare a TeleFuser replay against a captured LingBot-Video reference.

The comparator is intentionally framework-agnostic: captures contain tensor
metadata and optional ``.pt`` tensors, so this tool can be used before the
TeleFuser pipeline is wired into the serving stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_metadata(root: Path) -> dict[str, Any]:
    path = root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Capture metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    ref_raw = reference.detach().cpu()
    got_raw = candidate.detach().cpu()
    ref = ref_raw.float()
    got = got_raw.float()
    delta = (got - ref).abs()
    ref_norm = ref.reshape(-1).norm()
    cosine = (
        torch.nn.functional.cosine_similarity(ref.reshape(1, -1), got.reshape(1, -1)).item() if ref.numel() else 1.0
    )
    return {
        "shape_match": True,
        "dtype_match": reference.dtype == candidate.dtype,
        "reference_dtype": str(reference.dtype).removeprefix("torch."),
        "candidate_dtype": str(candidate.dtype).removeprefix("torch."),
        "exact_mismatch_count": int(torch.count_nonzero(ref_raw != got_raw).item()),
        "is_discrete": not reference.is_floating_point() and not candidate.is_floating_point(),
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "relative_l2": float(delta.reshape(-1).norm().div(ref_norm.clamp_min(1e-12)).item()),
        "cosine_similarity": float(max(-1.0, min(1.0, cosine))),
        "nan_count": int(torch.isnan(got).sum().item()),
        "inf_count": int(torch.isinf(got).sum().item()),
    }


def compare_captures(reference_root: Path, candidate_root: Path) -> dict[str, Any]:
    """Compare tensors with matching relative paths under two capture roots."""
    reference = _load_metadata(reference_root)
    candidate = _load_metadata(candidate_root)
    names = sorted(set(reference.get("tensors", {})) & set(candidate.get("tensors", {})))
    tensors: dict[str, Any] = {}
    for name in names:
        ref_path = reference_root / reference["tensors"][name]["path"]
        got_path = candidate_root / candidate["tensors"][name]["path"]
        if ref_path.is_file() and got_path.is_file():
            ref_tensor = torch.load(ref_path, map_location="cpu", weights_only=True)
            candidate_tensor = torch.load(got_path, map_location="cpu", weights_only=True)
            tensors[name] = _tensor_metrics(ref_tensor, candidate_tensor)
    return {
        "schema_version": 2,
        "reference": str(reference_root),
        "candidate": str(candidate_root),
        "matched_tensors": len(tensors),
        "missing_from_candidate": sorted(set(reference.get("tensors", {})) - set(candidate.get("tensors", {}))),
        "unexpected_in_candidate": sorted(set(candidate.get("tensors", {})) - set(reference.get("tensors", {}))),
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_captures(args.reference, args.candidate)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
