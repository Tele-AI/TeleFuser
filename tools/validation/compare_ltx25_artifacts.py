"""Compare an LTX-2.5 TeleFuser capture against an upstream golden manifest."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

_EXACT_ARTIFACT_PARTS = ("token", "mask", "noise")
_DEFAULT_COSINE = 0.9999
_DEFAULT_NRMSE = 1e-3
_RGB_PSNR_MIN = 40.0
_RGB_SSIM_MIN = 0.99
_WAVEFORM_SI_SDR_MIN = 40.0
_UPSTREAM_DIAGNOSTIC_ARTIFACTS = {"audio_features", "gemma_attention_mask", "gemma_token_ids", "video_features"}


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "capture_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"LTX-2.5 capture manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(root: Path, descriptor: dict[str, Any]) -> torch.Tensor:
    path = root / descriptor["path"]
    return torch.load(path, map_location="cpu", weights_only=True)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        return {"shape_match": False}
    reference_float = reference.double().reshape(-1)
    candidate_float = candidate.double().reshape(-1)
    delta = candidate_float - reference_float
    reference_norm = torch.linalg.vector_norm(reference_float)
    cosine = (
        1.0
        if reference_norm == 0 and torch.linalg.vector_norm(candidate_float) == 0
        else float(torch.nn.functional.cosine_similarity(reference_float, candidate_float, dim=0))
    )
    nrmse = float(torch.sqrt(torch.mean(delta.square())) / reference_float.square().mean().sqrt().clamp_min(1e-12))
    return {
        "shape_match": True,
        "dtype_match": reference.dtype == candidate.dtype,
        "exact": bool(torch.equal(reference, candidate)),
        "cosine": cosine,
        "nrmse": nrmse,
        "max_abs_error": float(delta.abs().max()),
    }


def _is_exact_contract(name: str) -> bool:
    return any(part in name for part in _EXACT_ARTIFACT_PARTS)


def _is_upstream_diagnostic_artifact(name: str) -> bool:
    """Return whether an artifact is emitted only by the upstream recorder diagnostics."""
    return name in _UPSTREAM_DIAGNOSTIC_ARTIFACTS or name.startswith("gemma_hidden_state_")


def _decoded_quality(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = candidate.float() - reference.float()
    if name == "decoded_rgb":
        rmse = torch.sqrt(torch.mean(delta.square())).item()
        return {
            "psnr": 20.0 * math.log10(1.0 / max(rmse, 1e-12)),
            "ssim": _structural_similarity(reference, candidate),
        }
    reference_flat = reference.double().reshape(-1)
    candidate_flat = candidate.double().reshape(-1)
    scale = torch.dot(candidate_flat, reference_flat) / torch.dot(reference_flat, reference_flat).clamp_min(1e-12)
    target = scale * reference_flat
    noise = candidate_flat - target
    return {"si_sdr": 10.0 * math.log10(float(torch.dot(target, target) / torch.dot(noise, noise).clamp_min(1e-12)))}


def _passes_decoded_quality(name: str, quality: dict[str, float]) -> bool:
    if name == "decoded_rgb":
        return quality["psnr"] >= _RGB_PSNR_MIN and quality["ssim"] >= _RGB_SSIM_MIN
    return quality["si_sdr"] >= _WAVEFORM_SI_SDR_MIN


def _structural_similarity(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compute SSIM for decoded RGB, using local windows when image layout is available."""
    reference_float = reference.float()
    candidate_float = candidate.float()
    if reference_float.ndim == 4 and reference_float.shape[-1] in {1, 3, 4}:
        reference_image = reference_float.permute(0, 3, 1, 2)
        candidate_image = candidate_float.permute(0, 3, 1, 2)
        window_size = min(11, reference_image.shape[-2], reference_image.shape[-1])
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 3:
            coords = torch.arange(window_size, dtype=reference_image.dtype, device=reference_image.device)
            coords = coords - (window_size - 1) / 2
            gaussian = torch.exp(-(coords.square()) / (2 * 1.5**2))
            window = (gaussian[:, None] * gaussian[None, :]) / gaussian.sum().square()
            window = window.expand(reference_image.shape[1], 1, window_size, window_size)
            groups = reference_image.shape[1]
            mu_reference = torch.nn.functional.conv2d(reference_image, window, padding=window_size // 2, groups=groups)
            mu_candidate = torch.nn.functional.conv2d(candidate_image, window, padding=window_size // 2, groups=groups)
            sigma_reference = (
                torch.nn.functional.conv2d(reference_image.square(), window, padding=window_size // 2, groups=groups)
                - mu_reference.square()
            )
            sigma_candidate = (
                torch.nn.functional.conv2d(candidate_image.square(), window, padding=window_size // 2, groups=groups)
                - mu_candidate.square()
            )
            covariance = (
                torch.nn.functional.conv2d(
                    reference_image * candidate_image, window, padding=window_size // 2, groups=groups
                )
                - mu_reference * mu_candidate
            )
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu_reference * mu_candidate + c1) * (2 * covariance + c2)) / (
                (mu_reference.square() + mu_candidate.square() + c1) * (sigma_reference + sigma_candidate + c2)
            )
            return float(ssim.mean())

    reference_flat = reference_float.reshape(-1)
    candidate_flat = candidate_float.reshape(-1)
    mean_reference = reference_flat.mean()
    mean_candidate = candidate_flat.mean()
    variance_reference = reference_flat.var(unbiased=False)
    variance_candidate = candidate_flat.var(unbiased=False)
    covariance = ((reference_flat - mean_reference) * (candidate_flat - mean_candidate)).mean()
    c1, c2 = 0.01**2, 0.03**2
    return float(
        ((2 * mean_reference * mean_candidate + c1) * (2 * covariance + c2))
        / ((mean_reference.square() + mean_candidate.square() + c1) * (variance_reference + variance_candidate + c2))
    )


def compare_captures(
    golden_root: Path,
    candidate_root: Path,
    *,
    cosine_threshold: float = _DEFAULT_COSINE,
    nrmse_threshold: float = _DEFAULT_NRMSE,
) -> dict[str, Any]:
    """Compare shared artifact tensors and validate frozen request/checkpoint contracts."""
    golden = _load_manifest(golden_root)
    candidate = _load_manifest(candidate_root)
    golden_artifacts = golden.get("artifacts", {})
    candidate_artifacts = candidate.get("artifacts", {})
    if not isinstance(golden_artifacts, dict) or not isinstance(candidate_artifacts, dict):
        raise ValueError("LTX-2.5 manifests must contain an artifacts object")

    results: dict[str, Any] = {}
    failures: list[str] = []
    for name in sorted(set(golden_artifacts) & set(candidate_artifacts)):
        metrics = _metrics(
            _load_tensor(golden_root, golden_artifacts[name]), _load_tensor(candidate_root, candidate_artifacts[name])
        )
        exact_contract = _is_exact_contract(name)
        decoded = name in {"decoded_rgb", "decoded_waveform"}
        passed = bool(metrics.get("shape_match"))
        if exact_contract:
            passed = passed and bool(metrics.get("exact"))
        elif decoded and passed:
            quality = _decoded_quality(
                name,
                _load_tensor(golden_root, golden_artifacts[name]),
                _load_tensor(candidate_root, candidate_artifacts[name]),
            )
            metrics.update(quality)
            passed = _passes_decoded_quality(name, quality)
        elif passed:
            passed = bool(metrics["exact"]) or bool(
                metrics["cosine"] >= cosine_threshold and metrics["nrmse"] <= nrmse_threshold
            )
        metrics["exact_contract"] = exact_contract
        metrics["passed"] = passed
        results[name] = metrics
        if not passed:
            failures.append(name)

    golden_request = golden.get("request", {})
    candidate_request = candidate.get("request", {})
    request_match = golden_request == candidate_request
    if not request_match:
        failures.append("request")
    checkpoint_match = golden.get("checkpoints", {}) == candidate.get("checkpoints", {})
    if not checkpoint_match:
        failures.append("checkpoints")
    audio_match = golden.get("audio", {}) == candidate.get("audio", {})
    if not audio_match:
        failures.append("audio")
    missing_from_candidate = sorted(set(golden_artifacts) - set(candidate_artifacts))
    missing_required_from_candidate = [
        name for name in missing_from_candidate if not _is_upstream_diagnostic_artifact(name)
    ]
    unexpected_in_candidate = sorted(set(candidate_artifacts) - set(golden_artifacts))
    artifact_set_match = set(golden_artifacts) == set(candidate_artifacts)
    artifact_contract_match = not missing_required_from_candidate and not unexpected_in_candidate
    if not artifact_contract_match:
        failures.append("artifact_set")
    return {
        "golden": str(golden_root),
        "candidate": str(candidate_root),
        "shared_artifacts": sorted(results),
        "missing_from_candidate": missing_from_candidate,
        "missing_required_from_candidate": missing_required_from_candidate,
        "unexpected_in_candidate": unexpected_in_candidate,
        "request_match": request_match,
        "checkpoint_match": checkpoint_match,
        "audio_match": audio_match,
        "artifact_set_match": artifact_set_match,
        "artifact_contract_match": artifact_contract_match,
        "tensors": results,
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    """Compare two LTX-2.5 capture directories and emit a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("golden", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--cosine-threshold", type=float, default=_DEFAULT_COSINE)
    parser.add_argument("--nrmse-threshold", type=float, default=_DEFAULT_NRMSE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare_captures(
        args.golden,
        args.candidate,
        cosine_threshold=args.cosine_threshold,
        nrmse_threshold=args.nrmse_threshold,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
