"""Strict B=2/B=3 ABot CUDA-Graph batch-continuation parity validation.

This correctness tool drives the public interactive batching API twice.  For
each batch lane it creates an eager twin and a candidate CUDA-Graph twin with
the same per-session seed, warms both *batched* groups eagerly until their
causal KV windows are full, and then runs two B=2 or B=3 continuations through
the ordinary eager and candidate graph-mode paths. The first candidate chunk
must capture; the second must reuse that graph. It verifies every retained
session state, the DiT latents handed to LightVAE, and decoded RGB frames.

The result is deliberately rejected unless the candidate path reports a real
capture and replay with no fallback.  On a revision that has not implemented
batched graphs yet, this tool therefore exits ``graph_unverified`` rather than
mistaking the regular eager fallback for a graph result.

Example (physical GPU 3 remapped to logical CUDA device 0)::

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/validate_abot_cuda_graph_batch_parity.py \\
      --batch-size 2 \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir results/validation/abot_cuda_graph_batch2_parity_gpu3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _load_base_validator() -> Any:
    """Load the B=1 validator's model-loading and pixel-comparison helpers."""
    path = Path(__file__).with_name("validate_abot_cuda_graph_parity.py")
    spec = importlib.util.spec_from_file_location("abot_cuda_graph_batch_parity_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ABot base parity validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_exactness(left: Any, right: Any) -> dict[str, Any]:
    """Strictly compare an on-device retained-state tensor tree.

    No cache is copied to CPU: ``torch.equal`` checks each CUDA leaf directly,
    which keeps this useful for the multi-GB KV state of B=2/B=3 sessions.
    """
    tensor_leaves = 0
    checked_leaves = 0
    mismatches: list[str] = []

    def visit(lhs: Any, rhs: Any, path: str) -> bool:
        nonlocal checked_leaves, tensor_leaves
        checked_leaves += 1
        if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
            tensor_leaves += 1
            if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
                mismatches.append(f"{path}: tensor/non-tensor type mismatch")
                return False
            if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
                mismatches.append(
                    f"{path}: tensor metadata differs "
                    f"({tuple(lhs.shape)}, {lhs.dtype}, {lhs.device}) != "
                    f"({tuple(rhs.shape)}, {rhs.dtype}, {rhs.device})"
                )
                return False
            if not bool(torch.equal(lhs, rhs)):
                mismatches.append(f"{path}: tensor values differ")
                return False
            return True
        if isinstance(lhs, Mapping) or isinstance(rhs, Mapping):
            if not isinstance(lhs, Mapping) or not isinstance(rhs, Mapping) or set(lhs) != set(rhs):
                mismatches.append(f"{path}: mapping keys/type differ")
                return False
            return all(visit(lhs[key], rhs[key], f"{path}.{key}") for key in sorted(lhs, key=str))
        if isinstance(lhs, (list, tuple)) or isinstance(rhs, (list, tuple)):
            if not isinstance(lhs, (list, tuple)) or not isinstance(rhs, (list, tuple)) or len(lhs) != len(rhs):
                mismatches.append(f"{path}: sequence length/type differs")
                return False
            return all(
                visit(item_lhs, item_rhs, f"{path}[{index}]")
                for index, (item_lhs, item_rhs) in enumerate(zip(lhs, rhs, strict=True))
            )
        if lhs != rhs:
            mismatches.append(f"{path}: {lhs!r} != {rhs!r}")
            return False
        return True

    exact = visit(left, right, "session")
    return {
        "exact": exact,
        "checked_leaves": checked_leaves,
        "tensor_leaves": tensor_leaves,
        "mismatches": mismatches[:20],
        "mismatch_count_at_least": len(mismatches),
    }


def _session_state_tree(session: Any, pipeline: Any) -> dict[str, Any]:
    """Return every stateful object that may affect a later interactive chunk."""
    if session.taew_decode_state is None:
        raise RuntimeError("ABot session is missing its TAeW decode state")
    return {
        "prompt_emb": session.prompt_emb,
        "first_frame_latent": session.first_frame_latent,
        "self_cache": session.self_cache,
        "cross_cache": session.cross_cache,
        "generator_state": session.generator.get_state(),
        "wan_decode_state": {
            "feat_cache": session.vae_decode_state.feat_cache,
            "feat_idx": session.vae_decode_state.feat_idx,
        },
        "taew_decode_state": pipeline.taew_decode_stage.export_decode_state_for_nccl(session.taew_decode_state),
        "next_latent_frame": session.next_latent_frame,
        "emitted_frames": session.emitted_frames,
    }


def _compare_tensor(left: torch.Tensor | None, right: torch.Tensor | None) -> dict[str, Any]:
    """Compare captured DiT output latents exactly without moving them to CPU."""
    if left is None or right is None:
        return {
            "comparable": False,
            "exact": False,
            "left_captured": left is not None,
            "right_captured": right is not None,
        }
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return {
            "comparable": False,
            "exact": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "left_device": str(left.device),
            "right_device": str(right.device),
        }
    difference = (left.float() - right.float()).abs()
    return {
        "comparable": True,
        "exact": bool(torch.equal(left, right)),
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "device": str(left.device),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
    }


class _DecodeLatentCapture:
    """Record the exact DiT output consumed by the public LightVAE batch call."""

    def __init__(self, decode_stage: Any) -> None:
        self._decode_stage = decode_stage
        self._original: Any = None
        self.latents: torch.Tensor | None = None

    def __enter__(self) -> "_DecodeLatentCapture":
        self._original = self._decode_stage.decode_chunks

        def capture(latents: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            # Enqueue the clone before decode mutates any session-owned stream
            # state. A later synchronize makes the captured tensor concrete.
            self.latents = latents.detach().clone()
            return self._original(latents, *args, **kwargs)

        self._decode_stage.decode_chunks = capture
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._decode_stage.decode_chunks = self._original


def _parse_actions(raw: str, batch_size: int, base: Any) -> list[dict[str, bool]]:
    """Parse semicolon-delimited action sets, one deterministic set per lane."""
    items = [item.strip() for item in raw.split(";") if item.strip()]
    if len(items) < batch_size:
        raise ValueError(
            f"--session-actions supplies {len(items)} action sets, but batch size {batch_size} needs one per session"
        )
    return [base._parse_action_keys(item) for item in items[:batch_size]]


def _all_exact(items: Sequence[Mapping[str, Any]]) -> bool:
    return bool(items) and all(bool(item.get("exact", False)) for item in items)


def _all_pixel_valid(items: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> bool:
    return bool(items) and all(
        bool(item.get("comparable"))
        and item.get("max_abs_rgb_difference") is not None
        and item.get("mean_abs_rgb_difference") is not None
        and item["max_abs_rgb_difference"] <= args.max_abs_rgb_difference
        and item["mean_abs_rgb_difference"] <= args.mean_abs_rgb_difference
        for item in items
    )


def _attach_batch_graph_evidence(graph: dict[str, Any], metrics: Mapping[str, Any], batch_size: int) -> dict[str, Any]:
    """Prove this was one native B=N graph, not N singleton graph calls."""
    observed_batch_size = metrics.get("batch_size")
    observed_graph_batch_size = metrics.get("cuda_graph_batch_size")
    graph["expected_batch_size"] = batch_size
    graph["observed_batch_size"] = observed_batch_size
    graph["batch_size_matches"] = observed_batch_size == batch_size
    graph["observed_cuda_graph_batch_size"] = observed_graph_batch_size
    graph["cuda_graph_batch_size_matches"] = observed_graph_batch_size == batch_size
    graph["cuda_graph_batched"] = bool(int(metrics.get("cuda_graph_batched", 0)))
    graph["actual_batched_graph"] = bool(
        graph["batch_size_matches"] and graph["cuda_graph_batch_size_matches"] and graph["cuda_graph_batched"]
    )
    return graph


def _batch_graph_replay_verified(metrics: Mapping[str, Any], batch_size: int, base: Any) -> dict[str, Any]:
    """Require a reuse call to replay a graph after its prior capture."""
    graph = _attach_batch_graph_evidence(base._graph_verified(metrics), metrics, batch_size)
    graph["verified"] = bool(
        graph["enabled"]
        and graph["eligible"]
        and graph["replay_observed"]
        and not graph["fallback_observed"]
        and graph["actual_batched_graph"]
    )
    return graph


@torch.inference_mode()
def _run_public_batch(
    pipeline: Any,
    sessions: Sequence[Any],
    actions: Sequence[Mapping[str, bool]],
    *,
    control_latent_frames: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run one public batch and retain the exact latent passed to LightVAE."""
    with _DecodeLatentCapture(pipeline.taew_decode_stage) as capture:
        frames = pipeline.generate_next_blocks(
            sessions,
            actions,
            control_latent_frames=control_latent_frames,
        )
    torch.cuda.synchronize(device)
    return {
        "frames": frames,
        "latents": capture.latents,
        "stage_metrics": dict(pipeline.last_stage_metrics()),
    }


def _batch_graph_verified(metrics: Mapping[str, Any], batch_size: int, base: Any) -> dict[str, Any]:
    """Require proof that this B>1 request captured one native B=N graph."""
    graph = _attach_batch_graph_evidence(base._graph_verified(metrics), metrics, batch_size)
    graph["verified"] = bool(graph["verified"] and graph["actual_batched_graph"])
    return graph


@torch.inference_mode()
def _run_validation(args: argparse.Namespace, base: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("batched CUDA Graph parity validation requires CUDA")
    image = Image.open(args.image).convert("RGB")
    pipeline = None
    graph_sessions: list[Any] = []
    eager_sessions: list[Any] = []
    try:
        pipeline = base._make_pipeline(args)
        device = torch.device(pipeline.device)
        if device.type != "cuda":
            raise RuntimeError(
                f"batched CUDA Graph parity validation requires a CUDA pipeline, got {pipeline.device!r}"
            )
        pipeline.preload_models()
        pipeline.denoise_stage.configure_cuda_graph(False)
        actions = _parse_actions(args.session_actions, args.batch_size, base)
        seeds = [args.seed + index * 9973 for index in range(args.batch_size)]
        for index, seed in enumerate(seeds):
            graph_sessions.append(
                pipeline.create_interactive_session(
                    image,
                    args.prompt,
                    seed=seed,
                    session_id=f"cuda-graph-batch-{args.batch_size}-candidate-{index}",
                )
            )
            eager_sessions.append(
                pipeline.create_interactive_session(
                    image,
                    args.prompt,
                    seed=seed,
                    session_id=f"cuda-graph-batch-{args.batch_size}-eager-{index}",
                )
            )

        local_attn_size = int(pipeline.denoise_stage.dit.local_attn_size)
        warmup_chunks = base._required_warmup_chunks(
            local_attn_size,
            args.control_latent_frames,
            args.extra_warmup_chunks,
        )
        warmup_per_session_hash_equal: list[list[bool]] = []
        for _ in range(warmup_chunks):
            candidate_frames = pipeline.generate_next_blocks(
                graph_sessions,
                actions,
                control_latent_frames=args.control_latent_frames,
            )
            eager_frames = pipeline.generate_next_blocks(
                eager_sessions,
                actions,
                control_latent_frames=args.control_latent_frames,
            )
            warmup_per_session_hash_equal.append(
                [
                    base._sequence_hash(candidate) == base._sequence_hash(eager)
                    for candidate, eager in zip(candidate_frames, eager_frames, strict=True)
                ]
            )
        torch.cuda.synchronize(device)

        pre_continuation: list[dict[str, Any]] = []
        for index, (candidate, eager) in enumerate(zip(graph_sessions, eager_sessions, strict=True)):
            pre_continuation.append(
                {
                    "session_index": index,
                    "candidate_session_id": candidate.session_id,
                    "eager_session_id": eager.session_id,
                    "candidate_cache": base._cache_readiness(candidate, pipeline),
                    "eager_cache": base._cache_readiness(eager, pipeline),
                    "state_exact": _tree_exactness(
                        _session_state_tree(candidate, pipeline),
                        _session_state_tree(eager, pipeline),
                    ),
                }
            )

        # First advance the eager twins twice while graph execution is disabled.
        # Then run the candidate cohort twice without a mode switch in-between:
        # capture on round one and a real persistent-slot replay on round two.
        pipeline.denoise_stage.configure_cuda_graph(False)
        eager_capture_round = _run_public_batch(
            pipeline,
            eager_sessions,
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        eager_replay_round = _run_public_batch(
            pipeline,
            eager_sessions,
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )

        pipeline.denoise_stage.configure_cuda_graph(True)
        graph_capture_round = _run_public_batch(
            pipeline,
            graph_sessions,
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        graph_runtime_after_capture = dict(pipeline.denoise_stage.cuda_graph_metrics())
        graph_replay_round = _run_public_batch(
            pipeline,
            graph_sessions,
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        graph_runtime_after_replay = dict(pipeline.denoise_stage.cuda_graph_metrics())
        graph_capture_verification = _batch_graph_verified(graph_capture_round["stage_metrics"], args.batch_size, base)
        graph_replay_verification = _batch_graph_replay_verified(
            graph_replay_round["stage_metrics"], args.batch_size, base
        )
        pipeline.denoise_stage.configure_cuda_graph(False)

        continuation_rounds = (
            ("capture_continuation", graph_capture_round, eager_capture_round),
            ("replay_continuation", graph_replay_round, eager_replay_round),
        )

        per_session: list[dict[str, Any]] = []
        for index, (candidate, eager) in enumerate(zip(graph_sessions, eager_sessions, strict=True)):
            session_rounds: list[dict[str, Any]] = []
            for round_name, candidate_round, eager_round in continuation_rounds:
                candidate_latents = candidate_round["latents"]
                eager_latents = eager_round["latents"]
                graph_latent = candidate_latents[index : index + 1] if candidate_latents is not None else None
                eager_latent = eager_latents[index : index + 1] if eager_latents is not None else None
                session_rounds.append(
                    {
                        "round": round_name,
                        "latent_comparison": _compare_tensor(graph_latent, eager_latent),
                        "rgb_comparison": base._compare_frames(
                            candidate_round["frames"][index],
                            eager_round["frames"][index],
                        ),
                    }
                )
            per_session.append(
                {
                    "session_index": index,
                    "seed": seeds[index],
                    "actions": actions[index],
                    "candidate_session_id": candidate.session_id,
                    "eager_session_id": eager.session_id,
                    "continuations": session_rounds,
                    "post_replay_state_exact": _tree_exactness(
                        _session_state_tree(candidate, pipeline),
                        _session_state_tree(eager, pipeline),
                    ),
                }
            )

        precondition_valid = all(
            bool(item["candidate_cache"]["ready"])
            and bool(item["eager_cache"]["ready"])
            and bool(item["state_exact"]["exact"])
            for item in pre_continuation
        ) and all(all(item) for item in warmup_per_session_hash_equal)
        round_comparisons = [round_item for item in per_session for round_item in item["continuations"]]
        latent_valid = _all_exact([item["latent_comparison"] for item in round_comparisons])
        state_valid = _all_exact([item["post_replay_state_exact"] for item in per_session])
        pixels_valid = _all_pixel_valid([item["rgb_comparison"] for item in round_comparisons], args)
        if not precondition_valid:
            status = "invalid_warmup"
        elif not graph_capture_verification["verified"] or not graph_replay_verification["verified"]:
            status = "graph_unverified"
        elif not latent_valid:
            status = "latent_mismatch"
        elif not state_valid:
            status = "state_mismatch"
        elif not pixels_valid:
            status = "pixel_mismatch"
        else:
            status = "pass"
        return {
            "status": status,
            "device": str(device),
            "batch_size": args.batch_size,
            "control_latent_frames": args.control_latent_frames,
            "session_actions": actions,
            "session_seeds": seeds,
            "warmup": {
                "chunks": warmup_chunks,
                "extra_chunks": args.extra_warmup_chunks,
                "per_chunk_per_session_output_hash_equal": warmup_per_session_hash_equal,
                "all_output_hashes_equal": all(all(item) for item in warmup_per_session_hash_equal),
                "pre_continuation_sessions": pre_continuation,
            },
            "candidate_graph_continuations": {
                "capture_continuation": {
                    "stage_metrics": graph_capture_round["stage_metrics"],
                    "runtime_metrics": graph_runtime_after_capture,
                    "verification": graph_capture_verification,
                    "captured_latent_batch_shape": (
                        list(graph_capture_round["latents"].shape)
                        if graph_capture_round["latents"] is not None
                        else None
                    ),
                },
                "replay_continuation": {
                    "stage_metrics": graph_replay_round["stage_metrics"],
                    "runtime_metrics": graph_runtime_after_replay,
                    "verification": graph_replay_verification,
                    "captured_latent_batch_shape": (
                        list(graph_replay_round["latents"].shape) if graph_replay_round["latents"] is not None else None
                    ),
                },
            },
            "ordinary_batched_eager_continuations": {
                "capture_continuation": {
                    "stage_metrics": eager_capture_round["stage_metrics"],
                    "captured_latent_batch_shape": (
                        list(eager_capture_round["latents"].shape)
                        if eager_capture_round["latents"] is not None
                        else None
                    ),
                },
                "replay_continuation": {
                    "stage_metrics": eager_replay_round["stage_metrics"],
                    "captured_latent_batch_shape": (
                        list(eager_replay_round["latents"].shape) if eager_replay_round["latents"] is not None else None
                    ),
                },
            },
            "per_session": per_session,
            "pixel_tolerance": {
                "max_abs_rgb_difference": args.max_abs_rgb_difference,
                "mean_abs_rgb_difference": args.mean_abs_rgb_difference,
                "within_tolerance": pixels_valid,
            },
        }
    finally:
        if pipeline is not None:
            for session in [*graph_sessions, *eager_sessions]:
                try:
                    pipeline.close_interactive_session(session)
                except Exception:
                    pass
            try:
                pipeline.close()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_results(output_dir: Path, result: Mapping[str, Any], args: argparse.Namespace, base: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arguments": base._json_safe(vars(args)), "result": base._json_safe(result)}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    graph = result.get("candidate_graph_continuations", {})
    capture = graph.get("capture_continuation", {}) if isinstance(graph, Mapping) else {}
    replay = graph.get("replay_continuation", {}) if isinstance(graph, Mapping) else {}
    capture_verification = capture.get("verification", {}) if isinstance(capture, Mapping) else {}
    replay_verification = replay.get("verification", {}) if isinstance(replay, Mapping) else {}
    lines = [
        f"# ABot CUDA Graph B={args.batch_size} continuation parity",
        "",
        (
            "Two same-seed B=N retained-session groups are warmed through the ordinary public batched path "
            "until every KV window is full. The eager and candidate groups then run two B=N continuations: "
            "candidate capture followed by a persistent graph replay."
        ),
        "",
        (
            "| Session | Round | Candidate / eager RGB sequence SHA-256 | Latent exact | RGB exact | "
            "Max RGB abs diff | Final state exact |"
        ),
        "| ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    for item in result.get("per_session", []):
        state = item.get("post_replay_state_exact", {})
        for round_item in item.get("continuations", []):
            latent = round_item.get("latent_comparison", {})
            rgb = round_item.get("rgb_comparison", {})
            lines.append(
                f"| {item.get('session_index', '')} | {round_item.get('round', '')} | "
                f"{rgb.get('sequence_sha256_graph', '')} / {rgb.get('sequence_sha256_eager', '')} | "
                f"{latent.get('exact', False)} | {rgb.get('all_frame_hashes_equal', False)} | "
                f"{rgb.get('max_abs_rgb_difference', '')} | {state.get('exact', False)} |"
            )
    lines.extend(
        [
            "",
            f"Status: `{result.get('status', 'error')}`.",
            "",
            f"Capture chunk capture/replay/fallback: `{capture_verification.get('captured', False)}` / "
            f"`{capture_verification.get('replay_observed', False)}` / "
            f"`{capture_verification.get('fallback_observed', False)}`.",
            "",
            f"Reuse chunk replay/fallback: `{replay_verification.get('replay_observed', False)}` / "
            f"`{replay_verification.get('fallback_observed', False)}`.",
            "",
            (
                "A passing result requires capture on the first B=N request and an actual no-fallback reuse replay "
                "on the second, plus full-KV paired warmup and exact per-session state/latent/RGB checks. "
                "`results.json` contains all per-frame and per-session SHA-256 values."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(base: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--batch-size", type=int, choices=(2, 3), required=False, default=2)
    parser.add_argument(
        "--session-actions",
        default="W;A;S",
        help="Semicolon-separated action sets for lanes 0..B-1, e.g. 'W;A;S'; NONE means idle.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Lane i uses seed + 9973*i in both paired groups.")
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
    parser.add_argument("--dry-run", action="store_true", help="Print the validation plan without loading a model.")
    args = parser.parse_args()
    if args.extra_warmup_chunks < 0:
        parser.error("--extra-warmup-chunks must be non-negative")
    if args.max_abs_rgb_difference < 0 or args.mean_abs_rgb_difference < 0:
        parser.error("pixel-difference tolerances must be non-negative")
    try:
        _parse_actions(args.session_actions, args.batch_size, base)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    if not args.dry_run:
        if args.model_root is None or args.image is None or args.output_dir is None:
            parser.error("--model-root, --image, and --output-dir are required unless --dry-run is used")
        if not args.model_root.is_dir():
            parser.error(f"model root does not exist: {args.model_root}")
        if not args.image.is_file():
            parser.error(f"image does not exist: {args.image}")
    return args


def main() -> None:
    base = _load_base_validator()
    args = _parse_args(base)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "batch_size": args.batch_size,
                    "paired_groups": ["candidate_cuda_graph", "ordinary_batched_eager"],
                    "session_actions": _parse_actions(args.session_actions, args.batch_size, base),
                    "warmup": "both B=N groups batched eagerly until all per-session KV windows are full",
                    "candidate_rounds": ["capture_continuation", "persistent_replay_continuation"],
                    "candidate_gate": (
                        "requires native cuda_graph_batch_size=B, cuda_graph_batched=1, capture/replay, and no fallback"
                    ),
                    "comparisons_per_session": ["pre/post retained state", "DiT latent", "RGB frame SHA-256/pixels"],
                    "control_latent_frames": args.control_latent_frames,
                    "pixel_tolerance": {
                        "max_abs_rgb_difference": args.max_abs_rgb_difference,
                        "mean_abs_rgb_difference": args.mean_abs_rgb_difference,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    assert args.output_dir is not None
    try:
        result = _run_validation(args, base)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_results(args.output_dir, result, args, base)
    print(json.dumps(base._json_safe(result), indent=2, sort_keys=True))
    if result.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
