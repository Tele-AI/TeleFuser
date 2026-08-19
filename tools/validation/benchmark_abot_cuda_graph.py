"""A/B benchmark for ABot-World's steady-state CUDA Graph continuation path.

The benchmark deliberately bypasses the serving scheduler: it measures a
single compatible microbatch after the causal KV cache has reached its fixed
window.  This makes a CUDA-Graph on/off comparison reproducible and avoids
mistaking scheduler queueing for model-runtime speedup.

``eager`` and ``cuda_graph`` accept B=1/2/3 compatible microbatches. For each
CUDA-Graph B>1 point, this tool requires explicit runtime evidence that one
batched graph replay occurred; it never treats B independent singleton graphs
as a native B=2/3 result. ``steady_eager`` remains a benchmark-only static
DiT control path and reports unsupported rather than silently falling back
when its requested native batch is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from PIL import Image

_GRAPH_ENV = "TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"
_MODES = ("eager", "steady_eager", "cuda_graph")


def _load_example_loader(mode: str) -> Any:
    """Load the example loader after selecting the graph environment flag."""
    loader_path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location(f"abot_cuda_graph_loader_{mode}", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot example loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_steady_eager_hook() -> Any:
    """Load the sibling benchmark-only hook without relying on PYTHONPATH."""
    hook_path = Path(__file__).with_name("abot_steady_eager.py")
    spec = importlib.util.spec_from_file_location("abot_steady_eager_benchmark_hook", hook_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load steady-eager benchmark hook: {hook_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.install_steady_eager_hook


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(modes).difference(_MODES))
    if not modes or invalid:
        raise argparse.ArgumentTypeError(f"modes must be a comma-separated subset of {','.join(_MODES)}; got {value!r}")
    return modes


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch sizes must be positive integers") from exc
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive integers")
    return values


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _numeric_stage_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    values_by_key: dict[str, list[float]] = {}
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, Real) and not isinstance(value, bool):
                values_by_key.setdefault(str(key), []).append(float(value))
    return {key: _summary(values) for key, values in sorted(values_by_key.items())}


def _graph_metrics_from_runtime(pipeline: Any) -> dict[str, Any]:
    """Best-effort extraction; core code owns the exact CUDA-Graph counters."""
    result: dict[str, Any] = {}
    candidates = [
        ("pipeline", pipeline),
        ("denoise_stage", getattr(pipeline, "denoise_stage", None)),
    ]
    for owner_name, owner in candidates:
        if owner is None:
            continue
        for attr in ("cuda_graph_metrics", "cuda_graph_runtime_metrics", "graph_runtime_metrics"):
            callback = getattr(owner, attr, None)
            if not callable(callback):
                continue
            try:
                metrics = callback()
            except Exception as exc:  # A diagnostic must never hide a benchmark result.
                result[f"{owner_name}.{attr}_error"] = f"{type(exc).__name__}: {exc}"
                continue
            if isinstance(metrics, Mapping):
                result[f"{owner_name}.{attr}"] = _json_safe(metrics)
    return result


def _graph_replay_observed(stage_samples: Sequence[Mapping[str, Any]], runtime_metrics: Mapping[str, Any]) -> bool:
    """Return true only for an explicit replay/used counter, never by inference."""

    def walk(value: Any, graph_context: bool = False) -> bool:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key).lower()
                item_graph_context = graph_context or "graph" in key
                if isinstance(item, (Mapping, list, tuple)) and walk(item, item_graph_context):
                    return True
                if not item_graph_context:
                    continue
                if isinstance(item, str) and item.lower() in {"true", "yes", "replay", "replayed", "used", "hit"}:
                    return True
                if not any(token in key for token in ("replay", "used", "hit")):
                    continue
                if isinstance(item, bool) and item:
                    return True
                if isinstance(item, Real) and item > 0:
                    return True
            return False
        if isinstance(value, (list, tuple)):
            return any(walk(item, graph_context) for item in value)
        return False

    return any(walk(sample) for sample in stage_samples) or walk(runtime_metrics)


_TAEW_DECODE_MODE_NAMES = {
    0: "singleton",
    1: "synchronized_native_batch",
    2: "serial_fallback",
}


def _as_optional_int(value: Any) -> int | None:
    """Return a strict integer metric, excluding booleans and opaque values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Real) and float(value).is_integer():
        return int(value)
    return None


def _normalise_graph_mode(value: Any) -> str | None:
    if isinstance(value, str):
        normalised = value.strip().lower()
        return normalised or None
    return None


def _graph_batch_verification(
    stage_samples: Sequence[Mapping[str, Any]],
    requested_batch_size: int,
) -> dict[str, Any]:
    """Prove that measured Graph chunks were one native B-sized replay.

    A positive replay counter alone is insufficient for B>1: it could describe
    one or more singleton graph slots. B>1 therefore requires a B-sized graph
    metric and, where exported, its explicit native-batched marker. Older core
    revisions can instead export cuda_graph_mode=batched.
    """

    chunks: list[dict[str, Any]] = []
    for sample in stage_samples:
        scheduler_batch_size = _as_optional_int(sample.get("batch_size"))
        graph_batch_size = _as_optional_int(sample.get("cuda_graph_batch_size"))
        graph_mode = _normalise_graph_mode(sample.get("cuda_graph_mode"))
        graph_batched = _as_optional_int(sample.get("cuda_graph_batched"))
        replay_count = _as_optional_int(sample.get("cuda_graph_replays")) or 0
        fallback_count = _as_optional_int(sample.get("cuda_graph_fallback")) or 0
        enabled = _as_optional_int(sample.get("cuda_graph_enabled")) == 1
        eligible = _as_optional_int(sample.get("cuda_graph_eligible")) == 1
        scheduler_batch_matches = scheduler_batch_size == requested_batch_size
        if requested_batch_size == 1:
            graph_batch_matches = True
        elif graph_batched is not None:
            graph_batch_matches = graph_batch_size == requested_batch_size and graph_batched == 1
        else:
            graph_batch_matches = graph_mode == "batched" or graph_batch_size == requested_batch_size
        chunks.append(
            {
                "scheduler_batch_size": scheduler_batch_size,
                "cuda_graph_batch_size": graph_batch_size,
                "cuda_graph_mode": graph_mode,
                "cuda_graph_batched": graph_batched,
                "cuda_graph_replays": replay_count,
                "cuda_graph_fallback": fallback_count,
                "cuda_graph_enabled": enabled,
                "cuda_graph_eligible": eligible,
                "scheduler_batch_matches": scheduler_batch_matches,
                "graph_batch_matches": graph_batch_matches,
                "replay_observed": replay_count > 0,
                "fallback_observed": fallback_count > 0,
                "verified": scheduler_batch_matches
                and graph_batch_matches
                and enabled
                and eligible
                and replay_count > 0
                and fallback_count == 0,
            }
        )
    return {
        "requested_batch_size": requested_batch_size,
        "measured_chunks": len(chunks),
        "all_scheduler_batches_exact": bool(chunks) and all(item["scheduler_batch_matches"] for item in chunks),
        "all_graph_batches_native": bool(chunks) and all(item["graph_batch_matches"] for item in chunks),
        "all_chunks_replayed": bool(chunks) and all(item["replay_observed"] for item in chunks),
        "fallback_observed": any(item["fallback_observed"] for item in chunks),
        "verified": bool(chunks) and all(item["verified"] for item in chunks),
        "chunks": chunks,
    }


def _taew_batch_verification(
    stage_samples: Sequence[Mapping[str, Any]],
    requested_batch_size: int,
) -> dict[str, Any]:
    """Report actual LightVAE/TAeW decode behavior, including serial fallback."""
    expected_mode = 0 if requested_batch_size == 1 else 1
    chunks: list[dict[str, Any]] = []
    for sample in stage_samples:
        mode = _as_optional_int(sample.get("taew_decode_mode"))
        effective_batch_size = _as_optional_int(sample.get("taew_decode_batch_size"))
        invocations = _as_optional_int(sample.get("taew_decode_invocations"))
        items = _as_optional_int(sample.get("taew_decode_items"))
        verified = (
            mode == expected_mode
            and effective_batch_size == requested_batch_size
            and invocations == 1
            and items == requested_batch_size
        )
        chunks.append(
            {
                "mode": mode,
                "mode_name": _TAEW_DECODE_MODE_NAMES.get(mode, "unreported"),
                "effective_batch_size": effective_batch_size,
                "invocations": invocations,
                "items": items,
                "native_batch_verified": verified,
            }
        )
    return {
        "requested_batch_size": requested_batch_size,
        "expected_mode": expected_mode,
        "expected_mode_name": _TAEW_DECODE_MODE_NAMES[expected_mode],
        "measured_chunks": len(chunks),
        "reported": bool(chunks) and all(item["mode"] is not None for item in chunks),
        "verified": bool(chunks) and all(item["native_batch_verified"] for item in chunks),
        "fallback_observed": any(item["mode"] == 2 for item in chunks),
        "chunks": chunks,
    }


class _NativeBatchProbe:
    """Observe Python model entry points in a benchmark-only pipeline.

    This validates eager/static-eager DiT calls. CUDA Graph replays deliberately
    bypass Python after capture, so graph-native batching is instead proved by
    _graph_batch_verification and core-reported batch facts.
    """

    def __init__(self, pipeline: Any) -> None:
        self._patches: list[tuple[Any, str, Any]] = []
        self._measurement_active = False
        self._calls: dict[str, list[int | None]] = {
            "dit_dynamic": [],
            "dit_steady_state": [],
            "taew_decode": [],
        }
        self._install_on_pipeline(pipeline)

    @staticmethod
    def _batch_size(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> int | None:
        value = kwargs.get("x")
        if value is None:
            value = kwargs.get("latents")
        if value is None and args:
            value = args[0]
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
        return None

    def _wrap(self, owner: Any, attribute: str, bucket: str) -> None:
        original = getattr(owner, attribute, None)
        if not callable(original):
            return

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self._measurement_active:
                self._calls[bucket].append(self._batch_size(args, kwargs))
            return original(*args, **kwargs)

        try:
            setattr(owner, attribute, wrapped)
        except (AttributeError, TypeError):
            return
        self._patches.append((owner, attribute, original))

    def _install_on_pipeline(self, pipeline: Any) -> None:
        denoise_stage = getattr(pipeline, "denoise_stage", None)
        dit = getattr(denoise_stage, "dit", None)
        if dit is not None:
            self._wrap(dit, "forward", "dit_dynamic")
            self._wrap(dit, "forward_steady_state", "dit_steady_state")
        taew_stage = getattr(pipeline, "taew_decode_stage", None)
        if taew_stage is not None:
            self._wrap(taew_stage, "decode_chunks", "taew_decode")

    def begin_measurement(self) -> None:
        for values in self._calls.values():
            values.clear()
        self._measurement_active = True

    def metrics(self) -> dict[str, Any]:
        return {
            "installed_wrappers": len(self._patches),
            "dit_dynamic_batch_sizes_measured": list(self._calls["dit_dynamic"]),
            "dit_steady_state_batch_sizes_measured": list(self._calls["dit_steady_state"]),
            "taew_decode_input_batch_sizes_measured": list(self._calls["taew_decode"]),
        }

    def close(self) -> None:
        for owner, attribute, original in reversed(self._patches):
            setattr(owner, attribute, original)
        self._patches.clear()


def _dit_batch_verification(
    probe_metrics: Mapping[str, Any],
    requested_batch_size: int,
    *,
    cuda_graph: bool,
    graph_verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate Python-observed eager batching from CUDA-Graph batch evidence."""
    dynamic = [item for item in probe_metrics.get("dit_dynamic_batch_sizes_measured", []) if item is not None]
    steady = [item for item in probe_metrics.get("dit_steady_state_batch_sizes_measured", []) if item is not None]
    observed = dynamic + steady
    if cuda_graph:
        verified = bool(graph_verification.get("verified"))
        evidence = "core_cuda_graph_batch_metrics"
    else:
        verified = bool(observed) and all(item == requested_batch_size for item in observed)
        evidence = "python_dit_entrypoint_probe"
    return {
        "requested_batch_size": requested_batch_size,
        "cuda_graph": cuda_graph,
        "evidence": evidence,
        "dynamic_calls": dynamic,
        "steady_state_calls": steady,
        "verified": verified,
    }


def _warmup_chunks_for_steady_state(pipeline: Any, control_latent_frames: int, requested: int) -> tuple[int, int]:
    """Return (effective warmups, automatic minimum) after the first chunk.

    An initial chunk advances the session by ``control_latent_frames``.  To
    exercise the fixed-window continuation graph once, fill the local window
    and issue one more continuation chunk.  This keeps graph capture outside
    the measured samples.
    """
    dit = getattr(getattr(pipeline, "denoise_stage", None), "dit", None)
    local_frames = int(getattr(dit, "local_attn_size", 0))
    fill_chunks = max(0, math.ceil(max(0, local_frames - control_latent_frames) / control_latent_frames))
    automatic_minimum = fill_chunks + 1
    return max(requested, automatic_minimum), automatic_minimum


def _make_pipeline(args: argparse.Namespace, mode: str) -> Any:
    os.environ[_GRAPH_ENV] = "1" if mode == "cuda_graph" else "0"
    loader = _load_example_loader(mode)
    # Import only after setting the environment flag.  The core implementation
    # reads the flag while constructing a fresh pipeline for every A/B point.
    from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline

    return loader.get_pipeline(
        model_root=args.model_root,
        device_id=args.device_id,
        pipeline_class=ABotWorldInteractivePipeline,
    )


def _run_point(args: argparse.Namespace, mode: str, batch_size: int, image: Image.Image) -> dict[str, Any]:
    graph_requested = mode == "cuda_graph"
    steady_eager_requested = mode == "steady_eager"
    if steady_eager_requested and batch_size != 1:
        return {
            "mode": mode,
            "cuda_graph_requested": graph_requested,
            "steady_eager_requested": steady_eager_requested,
            "batch": batch_size,
            "status": "unsupported",
            "reason": (
                "steady_eager is a B=1-only static DiT control path; use eager and cuda_graph to benchmark "
                "native B=2/3 continuation batches."
            ),
        }

    batch_probe: _NativeBatchProbe | None = None
    pipeline = None
    sessions: list[Any] = []
    steady_eager_hook: Any | None = None
    try:
        pipeline = _make_pipeline(args, mode)
        device = torch.device(pipeline.device)
        if device.type != "cuda":
            raise RuntimeError(f"CUDA Graph benchmark requires a CUDA pipeline, got {pipeline.device!r}")
        pipeline.preload_models()
        batch_probe = _NativeBatchProbe(pipeline)
        if steady_eager_requested:
            steady_eager_hook = _load_steady_eager_hook()(pipeline)
        for index in range(batch_size):
            sessions.append(
                pipeline.create_interactive_session(
                    image,
                    args.prompt,
                    seed=args.seed + index,
                    session_id=f"cuda-graph-{mode}-b{batch_size}-s{index}",
                )
            )
        controls = [{"W": True} for _ in sessions]
        expected_frames = 4 * args.control_latent_frames

        # First block is deliberately outside steady-state timing.
        first = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=args.control_latent_frames)
        if any(len(frames) != expected_frames for frames in first):
            raise RuntimeError(f"first block emitted unexpected frame counts: {[len(frames) for frames in first]}")

        effective_warmups, automatic_warmup_minimum = _warmup_chunks_for_steady_state(
            pipeline, args.control_latent_frames, args.warmup_chunks
        )
        warmup_stage_samples: list[dict[str, Any]] = []
        for _ in range(effective_warmups):
            frames = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=args.control_latent_frames)
            if any(len(item) != expected_frames for item in frames):
                raise RuntimeError(f"warmup emitted unexpected frame counts: {[len(item) for item in frames]}")
            warmup_stage_samples.append(_json_safe(pipeline.last_stage_metrics()))

        if steady_eager_hook is not None:
            steady_eager_hook.begin_measurement()
        if batch_probe is not None:
            batch_probe.begin_measurement()

        torch.cuda.synchronize(device)
        allocated_before = int(torch.cuda.memory_allocated(device))
        reserved_before = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        wall_samples: list[float] = []
        stage_samples: list[dict[str, Any]] = []
        for _ in range(args.repeats):
            torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            frames = pipeline.generate_next_blocks(sessions, controls, control_latent_frames=args.control_latent_frames)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started_at
            if any(len(item) != expected_frames for item in frames):
                raise RuntimeError(f"sample emitted unexpected frame counts: {[len(item) for item in frames]}")
            wall_samples.append(elapsed)
            stage_samples.append(_json_safe(pipeline.last_stage_metrics()))

        runtime_graph_metrics = _graph_metrics_from_runtime(pipeline)
        graph_verification = _graph_batch_verification(stage_samples, batch_size)
        graph_replay_observed = graph_requested and bool(graph_verification["verified"])
        steady_eager_metrics = steady_eager_hook.runtime_metrics() if steady_eager_hook is not None else {}
        steady_eager_observed = bool(steady_eager_metrics.get("steady_calls_measured", 0))
        probe_metrics = batch_probe.metrics() if batch_probe is not None else {}
        dit_batch_verification = _dit_batch_verification(
            probe_metrics,
            batch_size,
            cuda_graph=graph_requested,
            graph_verification=graph_verification,
        )
        taew_batch_verification = _taew_batch_verification(stage_samples, batch_size)
        native_microbatch_verified = bool(dit_batch_verification["verified"]) and bool(
            taew_batch_verification["verified"]
        )
        wall = _summary(wall_samples)
        stage_summary = _numeric_stage_summary(stage_samples)
        chunk_seconds = wall["mean"]
        result = {
            "mode": mode,
            "execution_path": {
                "eager": "legacy_dynamic",
                "steady_eager": "steady_state_eager_benchmark_hook",
                "cuda_graph": "steady_state_cuda_graph",
            }[mode],
            "cuda_graph_requested": graph_requested,
            "steady_eager_requested": steady_eager_requested,
            "steady_eager_observed": steady_eager_observed,
            "steady_eager_metrics": steady_eager_metrics,
            "batch": batch_size,
            "status": "ok",
            "device": str(device),
            "control_latent_frames": args.control_latent_frames,
            "frames_per_session_per_chunk": expected_frames,
            "repeats": args.repeats,
            "warmup_chunks_requested": args.warmup_chunks,
            "warmup_chunks_effective": effective_warmups,
            "warmup_chunks_steady_state_minimum": automatic_warmup_minimum,
            "chunk_wall_seconds": wall,
            "chunk_time_seconds": chunk_seconds,
            "aggregate_fps": (expected_frames * batch_size / chunk_seconds) if chunk_seconds else 0.0,
            "fps_per_session": (expected_frames / chunk_seconds) if chunk_seconds else 0.0,
            "stage_seconds": stage_summary,
            "stage_samples": stage_samples,
            "warmup_stage_samples": warmup_stage_samples,
            "runtime_graph_metrics": runtime_graph_metrics,
            "cuda_graph_verification": graph_verification,
            "cuda_graph_replay_observed": graph_replay_observed,
            "native_batch_probe": probe_metrics,
            "dit_batch_verification": dit_batch_verification,
            "taew_batch_verification": taew_batch_verification,
            "native_microbatch_verified": native_microbatch_verified,
            "gpu_memory": {
                "allocated_before_measured_bytes": allocated_before,
                "reserved_before_measured_bytes": reserved_before,
                "peak_allocated_measured_bytes": int(torch.cuda.max_memory_allocated(device)),
                "allocated_after_measured_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_after_measured_bytes": int(torch.cuda.memory_reserved(device)),
            },
        }
        if graph_requested and args.require_graph_replay and not graph_replay_observed:
            result["status"] = "graph_unverified"
            result["error"] = (
                "No verified measured CUDA-Graph replay was observed for the requested batch. "
                "For B>1 this requires batch_size=B, replay>0, fallback=0, and "
                "cuda_graph_mode=batched or cuda_graph_batch_size=B on every measured chunk."
            )
        if steady_eager_requested and not steady_eager_observed:
            result["status"] = "steady_eager_unverified"
            result["error"] = (
                "No measured full-window forward_steady_state eager invocation was observed. "
                "This prevents a legacy dynamic fallback from being reported as steady eager."
            )
        if args.require_native_batch and result["status"] == "ok" and not native_microbatch_verified:
            result["status"] = "native_batch_unverified"
            result["error"] = (
                "The requested scheduler batch did not prove both native DiT and synchronized LightVAE batching. "
                "Inspect dit_batch_verification and taew_batch_verification for the exact fallback."
            )
        return result
    except torch.OutOfMemoryError as exc:
        return {
            "mode": mode,
            "cuda_graph_requested": graph_requested,
            "steady_eager_requested": steady_eager_requested,
            "batch": batch_size,
            "status": "oom",
            "error": str(exc).splitlines()[0],
        }
    except Exception as exc:
        return {
            "mode": mode,
            "cuda_graph_requested": graph_requested,
            "steady_eager_requested": steady_eager_requested,
            "batch": batch_size,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if steady_eager_hook is not None:
            try:
                steady_eager_hook.close()
            except Exception:
                pass
        if batch_probe is not None:
            try:
                batch_probe.close()
            except Exception:
                pass
        if pipeline is not None:
            for session in sessions:
                try:
                    pipeline.close_interactive_session(session)
                except Exception:
                    pass
            try:
                pipeline.close()
            except Exception:
                pass
        del sessions
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _display(value: Any, digits: int = 3) -> str:
    if isinstance(value, Real) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    return str(value if value is not None else "")


def _observed_fact(verification: Mapping[str, Any], key: str) -> str:
    """Render distinct per-chunk evidence values compactly for CSV/Markdown."""
    chunks = verification.get("chunks", [])
    if not isinstance(chunks, Sequence):
        return ""
    values = sorted({str(item.get(key)) for item in chunks if isinstance(item, Mapping) and item.get(key) is not None})
    return ",".join(values)


def _write_outputs(output_dir: Path, results: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(_json_safe({"arguments": vars(args), "results": list(results)}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fields = [
        "mode",
        "execution_path",
        "batch",
        "status",
        "chunk_time_seconds",
        "aggregate_fps",
        "fps_per_session",
        "denoise_seconds_mean",
        "vae_decode_seconds_mean",
        "postprocess_seconds_mean",
        "steady_eager_observed",
        "cuda_graph_replay_observed",
        "cuda_graph_batch_verified",
        "cuda_graph_mode",
        "cuda_graph_batch_size",
        "dit_native_batch_verified",
        "taew_native_batch_verified",
        "taew_decode_mode",
        "taew_effective_batch_size",
        "native_microbatch_verified",
        "error",
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        stage = result.get("stage_seconds", {}) if isinstance(result, Mapping) else {}
        graph = result.get("cuda_graph_verification", {}) if isinstance(result, Mapping) else {}
        dit = result.get("dit_batch_verification", {}) if isinstance(result, Mapping) else {}
        taew = result.get("taew_batch_verification", {}) if isinstance(result, Mapping) else {}
        rows.append(
            {
                "mode": result.get("mode"),
                "execution_path": result.get("execution_path"),
                "batch": result.get("batch"),
                "status": result.get("status"),
                "chunk_time_seconds": result.get("chunk_time_seconds"),
                "aggregate_fps": result.get("aggregate_fps"),
                "fps_per_session": result.get("fps_per_session"),
                "denoise_seconds_mean": stage.get("denoise_seconds", {}).get("mean"),
                "vae_decode_seconds_mean": stage.get("vae_decode_seconds", {}).get("mean"),
                "postprocess_seconds_mean": stage.get("postprocess_seconds", {}).get("mean"),
                "steady_eager_observed": result.get("steady_eager_observed"),
                "cuda_graph_replay_observed": result.get("cuda_graph_replay_observed"),
                "cuda_graph_batch_verified": graph.get("verified") if isinstance(graph, Mapping) else None,
                "cuda_graph_mode": _observed_fact(graph, "cuda_graph_mode") if isinstance(graph, Mapping) else "",
                "cuda_graph_batch_size": _observed_fact(graph, "cuda_graph_batch_size")
                if isinstance(graph, Mapping)
                else "",
                "dit_native_batch_verified": dit.get("verified") if isinstance(dit, Mapping) else None,
                "taew_native_batch_verified": taew.get("verified") if isinstance(taew, Mapping) else None,
                "taew_decode_mode": _observed_fact(taew, "mode_name") if isinstance(taew, Mapping) else "",
                "taew_effective_batch_size": _observed_fact(taew, "effective_batch_size")
                if isinstance(taew, Mapping)
                else "",
                "native_microbatch_verified": result.get("native_microbatch_verified"),
                "error": result.get("error") or result.get("reason"),
            }
        )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# ABot steady-state path comparison",
        "",
        (
            "`steady_eager` is benchmark-only and invokes the same full-window "
            "`forward_steady_state` path without CUDA Graph capture. Its row is valid only "
            "when `steady_eager_observed` is `True`; `cuda_graph` is valid only when "
            "`cuda_graph_replay_observed` is `True`. For B>1, the CSV also records native-DiT, "
            "native-LightVAE, and graph-batch evidence; a scheduler batch is not treated as native merely "
            "because it contains more than one session."
        ),
        "",
        (
            "| Mode | Path | B | Status | Chunk (s) | Aggregate FPS | FPS/session | DiT (s) | "
            "VAE decode (s) | Postprocess (s) | Steady eager | Graph replay | Graph B native | "
            "DiT native | LightVAE native |"
        ),
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            (
                "| {mode} | {path} | {batch} | {status} | {chunk} | {aggregate} | {session} | "
                "{denoise} | {vae} | {post} | {steady} | {replay} | {graph_batch} | {dit_batch} | {taew_batch} |"
            ).format(
                mode=_display(row["mode"]),
                path=_display(row["execution_path"]),
                batch=_display(row["batch"], 0),
                status=_display(row["status"]),
                chunk=_display(row["chunk_time_seconds"]),
                aggregate=_display(row["aggregate_fps"], 2),
                session=_display(row["fps_per_session"], 2),
                denoise=_display(row["denoise_seconds_mean"]),
                vae=_display(row["vae_decode_seconds_mean"]),
                post=_display(row["postprocess_seconds_mean"]),
                steady=_display(row["steady_eager_observed"]),
                replay=_display(row["cuda_graph_replay_observed"]),
                graph_batch=_display(row["cuda_graph_batch_verified"]),
                dit_batch=_display(row["dit_native_batch_verified"]),
                taew_batch=_display(row["taew_native_batch_verified"]),
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--modes", type=_parse_modes, default=["eager", "steady_eager", "cuda_graph"])
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=[1])
    parser.add_argument("--control-latent-frames", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--warmup-chunks", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Logical CUDA device after CUDA_VISIBLE_DEVICES remapping (normally 0).",
    )
    parser.add_argument(
        "--require-graph-replay",
        action="store_true",
        help="Mark graph results invalid unless core metrics explicitly report a graph replay/use.",
    )
    parser.add_argument(
        "--require-native-batch",
        action="store_true",
        help=("Mark a point invalid unless DiT and LightVAE both prove one native requested-size batch."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the planned A/B sweep without loading a model.",
    )
    args = parser.parse_args()
    if args.warmup_chunks < 0:
        parser.error("warmup-chunks must be non-negative")
    if args.repeats < 1:
        parser.error("repeats must be positive")
    return args


def main() -> None:
    args = _parse_args()
    plan = {
        "modes": args.modes,
        "batch_sizes": args.batch_sizes,
        "graph_environment_variable": _GRAPH_ENV,
        "graph_environment_values": {"eager": "0", "steady_eager": "0", "cuda_graph": "1"},
        "device_id": args.device_id,
        "cuda_graph_batch_sizes": [1, 2, 3],
        "steady_eager_batch_sizes": [1],
        "B_gt_1_graph_gate": (
            "batch_size=B, cuda_graph_batch_size=B plus cuda_graph_batched=1 when exported "
            "(or cuda_graph_mode=batched on older cores), replay>0, fallback=0"
        ),
        "B_gt_1_native_batch_gate": "native DiT evidence and TAeW synchronized-native-batch evidence",
        "steady_eager": "benchmark-only forward_steady_state execution without CUDA Graph capture",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    image = Image.open(args.image).convert("RGB")
    results: list[dict[str, Any]] = []
    for mode in args.modes:
        for batch_size in args.batch_sizes:
            print(f"running mode={mode} batch={batch_size}", flush=True)
            results.append(_run_point(args, mode, batch_size, image))
            _write_outputs(args.output_dir, results, args)
    _write_outputs(args.output_dir, results, args)
    print(json.dumps(_json_safe(results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
