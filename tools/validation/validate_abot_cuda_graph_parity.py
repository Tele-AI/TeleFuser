"""Validate ABot-World's real CUDA-Graph continuation against eager output.

This is deliberately a correctness tool, not a throughput benchmark.  It
loads one interactive pipeline, creates two independent B=1 retained
sessions with the same seed, and advances both through the normal eager path
until their causal KV windows are full.  One subsequent continuation uses the
experimental CUDA-Graph path while its twin stays eager.  The resulting
rendered frames are compared byte-for-byte and by RGB absolute pixel error.

The graph result is accepted only when the serving stage explicitly reports a
capture and at least one graph replay without a fallback.  This avoids
mistaking the safety fallback for a graph correctness result.

Example (GPU 3 remapped to logical CUDA device 0)::

    CUDA_VISIBLE_DEVICES=3 \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/validate_abot_cuda_graph_parity.py \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir results/validation/abot_cuda_graph_parity_gpu3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageChops

_GRAPH_ENV = "TELEFUSER_ABOT_CUDA_GRAPH_ENABLED"
_ACTION_KEYS = ("W", "A", "S", "D", "I", "J", "K", "L")


def _load_example_loader() -> Any:
    loader_path = Path(__file__).resolve().parents[2] / "examples/abot_world/_loader.py"
    spec = importlib.util.spec_from_file_location("abot_cuda_graph_parity_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot example loader: {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_action_keys(value: str) -> dict[str, bool]:
    keys = [item.strip().upper() for item in value.split(",") if item.strip()]
    if len(keys) == 1 and keys[0] in {"NONE", "IDLE"}:
        return {}
    unknown = sorted(set(keys).difference(_ACTION_KEYS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown ABot action keys: {', '.join(unknown)}")
    return {key: True for key in keys}


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


def _frame_hash(frame: Image.Image) -> str:
    """Hash RGB pixels together with dimensions, independent of PIL metadata."""
    rgb = frame.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB:{rgb.width}x{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _sequence_hash(frames: Sequence[Image.Image]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(bytes.fromhex(_frame_hash(frame)))
    return digest.hexdigest()


def _compare_frames(graph_frames: Sequence[Image.Image], eager_frames: Sequence[Image.Image]) -> dict[str, Any]:
    """Compare rendered RGB frames without adding a NumPy dependency.

    ``PIL.ImageChops`` calculates the per-value absolute difference in C; the
    histogram then gives exact maximum/mean differences over all RGB values.
    """
    graph_hashes = [_frame_hash(frame) for frame in graph_frames]
    eager_hashes = [_frame_hash(frame) for frame in eager_frames]
    details: list[dict[str, Any]] = []
    total_absolute_difference = 0
    total_values = 0
    nonzero_values = 0
    maximum_absolute_difference = 0
    comparable = len(graph_frames) == len(eager_frames)

    for index, (graph_frame, eager_frame) in enumerate(zip(graph_frames, eager_frames, strict=False)):
        graph_rgb = graph_frame.convert("RGB")
        eager_rgb = eager_frame.convert("RGB")
        identical_shape = graph_rgb.size == eager_rgb.size
        frame_maximum = None
        frame_mean = None
        frame_nonzero = None
        if identical_shape:
            histogram = ImageChops.difference(graph_rgb, eager_rgb).histogram()
            channel_total = 0
            channel_values = 0
            channel_nonzero = 0
            channel_maximum = 0
            for channel_index in range(3):
                channel_histogram = histogram[channel_index * 256 : (channel_index + 1) * 256]
                channel_total += sum(value * count for value, count in enumerate(channel_histogram))
                channel_values += sum(channel_histogram)
                channel_nonzero += sum(channel_histogram[1:])
                channel_maximum = max(
                    channel_maximum, max((value for value, count in enumerate(channel_histogram) if count), default=0)
                )
            total_absolute_difference += channel_total
            total_values += channel_values
            nonzero_values += channel_nonzero
            maximum_absolute_difference = max(maximum_absolute_difference, channel_maximum)
            frame_maximum = channel_maximum
            frame_mean = (channel_total / channel_values) if channel_values else 0.0
            frame_nonzero = channel_nonzero
        else:
            comparable = False
        details.append(
            {
                "frame_index": index,
                "graph_sha256": graph_hashes[index],
                "eager_sha256": eager_hashes[index],
                "hash_equal": graph_hashes[index] == eager_hashes[index],
                "graph_size": list(graph_rgb.size),
                "eager_size": list(eager_rgb.size),
                "max_abs_rgb_difference": frame_maximum,
                "mean_abs_rgb_difference": frame_mean,
                "nonzero_rgb_values": frame_nonzero,
            }
        )

    return {
        "comparable": comparable,
        "frame_count_graph": len(graph_frames),
        "frame_count_eager": len(eager_frames),
        "sequence_sha256_graph": _sequence_hash(graph_frames),
        "sequence_sha256_eager": _sequence_hash(eager_frames),
        "all_frame_hashes_equal": graph_hashes == eager_hashes,
        "max_abs_rgb_difference": maximum_absolute_difference if comparable else None,
        "mean_abs_rgb_difference": (total_absolute_difference / total_values) if total_values else None,
        "nonzero_rgb_values": nonzero_values if comparable else None,
        "total_rgb_values": total_values if comparable else None,
        "frames": details,
    }


def _cache_readiness(session: Any, pipeline: Any) -> dict[str, Any]:
    """Return only compact metadata proving that a session reached full KV."""
    dit = pipeline.denoise_stage.dit
    latent_height, latent_width = session.first_frame_latent.shape[-2:]
    frame_tokens = (latent_height // dit.patch_size[1]) * (latent_width // dit.patch_size[2])
    expected_capacity = dit.local_attn_size * frame_tokens
    expected_global_end = session.next_latent_frame * frame_tokens
    local_ends = [int(layer["local_end_index"].item()) for layer in session.self_cache]
    global_ends = [int(layer["global_end_index"].item()) for layer in session.self_cache]
    cache_capacities = [int(layer["k"].shape[1]) for layer in session.self_cache]
    cross_initialized = sum(bool(layer["is_init"]) for layer in session.cross_cache)
    cross_lengths = sorted({int(layer["sequence_length"]) for layer in session.cross_cache})
    ready = (
        bool(session.self_cache)
        and all(value == expected_capacity for value in local_ends)
        and all(value == expected_global_end for value in global_ends)
        and all(value == expected_capacity for value in cache_capacities)
        and cross_initialized == len(session.cross_cache)
    )
    return {
        "ready": ready,
        "next_latent_frame": int(session.next_latent_frame),
        "frame_tokens": frame_tokens,
        "local_attn_size_frames": int(dit.local_attn_size),
        "expected_capacity_tokens": expected_capacity,
        "expected_global_end_tokens": expected_global_end,
        "self_cache_layers": len(session.self_cache),
        "unique_local_end_tokens": sorted(set(local_ends)),
        "unique_global_end_tokens": sorted(set(global_ends)),
        "unique_capacity_tokens": sorted(set(cache_capacities)),
        "cross_attention_initialized_layers": cross_initialized,
        "cross_attention_sequence_lengths": cross_lengths,
    }


def _required_warmup_chunks(local_attn_size: int, control_latent_frames: int, extra_chunks: int) -> int:
    if local_attn_size < 1:
        raise ValueError("ABot local attention window must be positive")
    if control_latent_frames < 1:
        raise ValueError("control_latent_frames must be positive")
    if extra_chunks < 0:
        raise ValueError("extra warmup chunks must be non-negative")
    return math.ceil(local_attn_size / control_latent_frames) + extra_chunks


def _graph_verified(metrics: Mapping[str, Any]) -> dict[str, Any]:
    captured = int(metrics.get("cuda_graph_captured", 0)) > 0
    replays = int(metrics.get("cuda_graph_replays", 0)) > 0
    fallback = int(metrics.get("cuda_graph_fallback", 0)) > 0
    eligible = int(metrics.get("cuda_graph_eligible", 0)) > 0
    enabled = int(metrics.get("cuda_graph_enabled", 0)) > 0
    return {
        "enabled": enabled,
        "eligible": eligible,
        "captured": captured,
        "replay_observed": replays,
        "fallback_observed": fallback,
        "verified": enabled and eligible and captured and replays and not fallback,
    }


def _make_pipeline(args: argparse.Namespace) -> Any:
    # Warmup has to be eager for both sessions.  The stage is toggled later
    # rather than loading two model copies on the same GPU.
    # The CUDA Graph capture stream follows the *current* CUDA device. Set the
    # CVD-local device before any model load so this remains correct under
    # CUDA_VISIBLE_DEVICES=4,5,6,7 with --device-id 1, as in a process-NCCL
    # worker.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph parity validation requires CUDA, but torch.cuda.is_available() is false")
    visible_count = int(torch.cuda.device_count())
    if not 0 <= args.device_id < visible_count:
        raise RuntimeError(f"--device-id {args.device_id} is out of range for {visible_count} visible CUDA device(s)")
    torch.cuda.set_device(args.device_id)
    if int(torch.cuda.current_device()) != args.device_id:
        raise RuntimeError(f"failed to select requested logical CUDA device {args.device_id}")
    original = os.environ.get(_GRAPH_ENV)
    os.environ[_GRAPH_ENV] = "0"
    try:
        loader = _load_example_loader()
        from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline

        return loader.get_pipeline(
            model_root=args.model_root,
            device_id=args.device_id,
            pipeline_class=ABotWorldInteractivePipeline,
        )
    finally:
        if original is None:
            os.environ.pop(_GRAPH_ENV, None)
        else:
            os.environ[_GRAPH_ENV] = original


def _run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph parity validation requires CUDA, but torch.cuda.is_available() is false")
    image = Image.open(args.image).convert("RGB")
    pipeline = None
    graph_session = None
    eager_session = None
    try:
        pipeline = _make_pipeline(args)
        device = torch.device(pipeline.device)
        if device.type != "cuda":
            raise RuntimeError(f"CUDA Graph parity validation requires a CUDA pipeline, got {pipeline.device!r}")
        pipeline.preload_models()
        pipeline.denoise_stage.configure_cuda_graph(False)
        graph_session = pipeline.create_interactive_session(
            image,
            args.prompt,
            seed=args.seed,
            session_id="cuda-graph-parity-graph",
        )
        eager_session = pipeline.create_interactive_session(
            image,
            args.prompt,
            seed=args.seed,
            session_id="cuda-graph-parity-eager",
        )

        local_attn_size = int(pipeline.denoise_stage.dit.local_attn_size)
        warmup_chunks = _required_warmup_chunks(
            local_attn_size,
            args.control_latent_frames,
            args.extra_warmup_chunks,
        )
        warmup_hash_equal: list[bool] = []
        for _ in range(warmup_chunks):
            graph_warm_frames = pipeline.generate_next_block(
                graph_session,
                args.action_keys,
                control_latent_frames=args.control_latent_frames,
            )
            eager_warm_frames = pipeline.generate_next_block(
                eager_session,
                args.action_keys,
                control_latent_frames=args.control_latent_frames,
            )
            warmup_hash_equal.append(_sequence_hash(graph_warm_frames) == _sequence_hash(eager_warm_frames))
        torch.cuda.synchronize(device)
        graph_warmup_cache = _cache_readiness(graph_session, pipeline)
        eager_warmup_cache = _cache_readiness(eager_session, pipeline)
        warmup_equivalent = bool(warmup_hash_equal) and all(warmup_hash_equal)
        warmup_ready = graph_warmup_cache["ready"] and eager_warmup_cache["ready"]

        # Capture/replay graph continuation from exactly the same full-window
        # session state as the eager continuation below.
        pipeline.denoise_stage.configure_cuda_graph(True)
        graph_frames = pipeline.generate_next_block(
            graph_session,
            args.action_keys,
            control_latent_frames=args.control_latent_frames,
        )
        torch.cuda.synchronize(device)
        graph_stage_metrics = dict(pipeline.last_stage_metrics())
        graph_runtime_metrics = dict(pipeline.denoise_stage.cuda_graph_metrics())
        graph_verification = _graph_verified(graph_stage_metrics)

        # Clear graph ownership before running the twin, so this call cannot
        # accidentally use an existing graph slot.
        pipeline.denoise_stage.configure_cuda_graph(False)
        eager_frames = pipeline.generate_next_block(
            eager_session,
            args.action_keys,
            control_latent_frames=args.control_latent_frames,
        )
        torch.cuda.synchronize(device)
        eager_stage_metrics = dict(pipeline.last_stage_metrics())
        comparison = _compare_frames(graph_frames, eager_frames)
        within_pixel_tolerance = (
            comparison["comparable"]
            and comparison["max_abs_rgb_difference"] is not None
            and comparison["mean_abs_rgb_difference"] is not None
            and comparison["max_abs_rgb_difference"] <= args.max_abs_rgb_difference
            and comparison["mean_abs_rgb_difference"] <= args.mean_abs_rgb_difference
        )
        if not warmup_ready or not warmup_equivalent:
            status = "invalid_warmup"
        elif not graph_verification["verified"]:
            status = "graph_unverified"
        elif not within_pixel_tolerance:
            status = "pixel_mismatch"
        else:
            status = "pass"
        return {
            "status": status,
            "device": str(device),
            "control_latent_frames": args.control_latent_frames,
            "actions": args.action_keys,
            "seed": args.seed,
            "warmup": {
                "chunks": warmup_chunks,
                "extra_chunks": args.extra_warmup_chunks,
                "per_chunk_output_hash_equal": warmup_hash_equal,
                "all_output_hashes_equal": warmup_equivalent,
                "graph_session_cache": graph_warmup_cache,
                "eager_session_cache": eager_warmup_cache,
            },
            "graph_continuation": {
                "stage_metrics": graph_stage_metrics,
                "runtime_metrics": graph_runtime_metrics,
                "verification": graph_verification,
            },
            "eager_continuation": {"stage_metrics": eager_stage_metrics},
            "comparison": comparison,
            "pixel_tolerance": {
                "max_abs_rgb_difference": args.max_abs_rgb_difference,
                "mean_abs_rgb_difference": args.mean_abs_rgb_difference,
                "within_tolerance": within_pixel_tolerance,
            },
        }
    finally:
        if pipeline is not None:
            if graph_session is not None:
                try:
                    pipeline.close_interactive_session(graph_session)
                except Exception:
                    pass
            if eager_session is not None:
                try:
                    pipeline.close_interactive_session(eager_session)
                except Exception:
                    pass
            try:
                pipeline.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_results(output_dir: Path, result: Mapping[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arguments": _json_safe(vars(args)), "result": _json_safe(result)}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    comparison = result.get("comparison", {})
    graph = result.get("graph_continuation", {})
    verification = graph.get("verification", {}) if isinstance(graph, Mapping) else {}
    warmup = result.get("warmup", {})
    tolerance = result.get("pixel_tolerance", {})
    graph_cache_ready = warmup.get("graph_session_cache", {}).get("ready", False)
    eager_cache_ready = warmup.get("eager_session_cache", {}).get("ready", False)
    graph_frame_count = comparison.get("frame_count_graph", "")
    eager_frame_count = comparison.get("frame_count_eager", "")
    maximum_tolerance = tolerance.get("max_abs_rgb_difference", "")
    mean_tolerance = tolerance.get("mean_abs_rgb_difference", "")
    lines = [
        "# ABot CUDA Graph continuation parity",
        "",
        (
            "This validates one B=1 retained-session continuation after two same-seed sessions "
            "were advanced eagerly to a full KV window."
        ),
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Status | {result.get('status', 'error')} |",
        f"| Device | {result.get('device', '')} |",
        f"| Eager warmup chunks/session | {warmup.get('chunks', '')} |",
        f"| Warmup outputs identical | {warmup.get('all_output_hashes_equal', False)} |",
        f"| Full KV ready (graph / eager) | {graph_cache_ready} / {eager_cache_ready} |",
        f"| Graph capture observed | {verification.get('captured', False)} |",
        f"| Graph replay observed | {verification.get('replay_observed', False)} |",
        f"| Graph fallback observed | {verification.get('fallback_observed', False)} |",
        f"| Frame count (graph / eager) | {graph_frame_count} / {eager_frame_count} |",
        f"| All frame SHA-256 equal | {comparison.get('all_frame_hashes_equal', False)} |",
        f"| Max abs RGB difference | {comparison.get('max_abs_rgb_difference', '')} |",
        f"| Mean abs RGB difference | {comparison.get('mean_abs_rgb_difference', '')} |",
        f"| Accepted max / mean tolerance | {maximum_tolerance} / {mean_tolerance} |",
        "",
        (
            "The full per-frame SHA-256 values, pixel differences, and CUDA-Graph stage/runtime metrics "
            "are in `results.json`."
        ),
    ]
    if result.get("error"):
        lines.extend(["", "## Error", "", f"`{result['error']}`"])
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--action-keys", type=_parse_action_keys, default={"W": True})
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-latent-frames", type=int, choices=(3,), default=3)
    parser.add_argument("--extra-warmup-chunks", type=int, default=0)
    parser.add_argument("--max-abs-rgb-difference", type=int, default=0)
    parser.add_argument("--mean-abs-rgb-difference", type=float, default=0.0)
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Logical CUDA device after CUDA_VISIBLE_DEVICES remapping (normally 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the fixed B=1 validation plan without loading a model or requiring paths.",
    )
    args = parser.parse_args()
    if args.extra_warmup_chunks < 0:
        parser.error("--extra-warmup-chunks must be non-negative")
    if args.max_abs_rgb_difference < 0 or args.mean_abs_rgb_difference < 0:
        parser.error("pixel-difference tolerances must be non-negative")
    if not args.dry_run:
        if args.model_root is None or args.image is None or args.output_dir is None:
            parser.error("--model-root, --image, and --output-dir are required unless --dry-run is used")
        if not args.model_root.is_dir():
            parser.error(f"model root does not exist: {args.model_root}")
        if not args.image.is_file():
            parser.error(f"image does not exist: {args.image}")
    return args


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "batch_size": 1,
                    "sessions": ["graph", "eager"],
                    "warmup": "both sessions eager until the 18-latent-frame KV window is full",
                    "continuation": "one CUDA-Graph-enabled session versus one eager session",
                    "graph_verification": "requires explicit capture and replay metrics with no fallback",
                    "pixel_comparison": "RGB SHA-256 plus maximum and mean absolute per-channel difference",
                    "control_latent_frames": args.control_latent_frames,
                    "extra_warmup_chunks": args.extra_warmup_chunks,
                    "device_id": args.device_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    assert args.output_dir is not None
    try:
        result = _run_validation(args)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_results(args.output_dir, result, args)
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    if result.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
