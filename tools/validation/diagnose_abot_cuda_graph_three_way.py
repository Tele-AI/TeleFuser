"""Separate ABot CUDA-Graph mismatch sources on a real GPU.

This diagnostic starts three identical B=1 retained sessions and warms all of
them through the ordinary eager interactive path until the 18-latent-frame KV
window is full.  It then compares one continuation through:

1. the regular dynamic eager DiT path;
2. ``ABotWorldDiT.forward_steady_state`` invoked eagerly (no CUDA capture);
3. the captured/replayed CUDA-Graph wrapper.

It compares the initial random latent, retained state, final latent, and
decoded RGB frames.  Thus it distinguishes a static-model semantic error from
a CUDA-Graph wrapper/capture error without modifying serving code.

Example (GPU 3 remapped to CUDA device 0)::

    CUDA_VISIBLE_DEVICES=3 \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/diagnose_abot_cuda_graph_three_way.py \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir results/validation/abot_cuda_graph_three_way_gpu3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def _load_base_validator() -> Any:
    """Reuse the two-way tool's loader, cache, and pixel-comparison helpers."""
    path = Path(__file__).with_name("validate_abot_cuda_graph_parity.py")
    spec = importlib.util.spec_from_file_location("abot_cuda_graph_parity_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the base parity validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree_exactness(left: Any, right: Any) -> dict[str, Any]:
    """Compare a session-state tree without copying its multi-GB KV cache to CPU."""
    tensor_leaves = 0
    checked_leaves = 0
    mismatches: list[str] = []

    def visit(lhs: Any, rhs: Any, path: str) -> bool:
        nonlocal tensor_leaves, checked_leaves
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


def _compare_tensor(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return {
            "comparable": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
            "left_device": str(left.device),
            "right_device": str(right.device),
            "exact": False,
        }
    difference = (left.float() - right.float()).abs()
    return {
        "comparable": True,
        "shape": list(left.shape),
        "dtype": str(left.dtype),
        "device": str(left.device),
        "exact": bool(torch.equal(left, right)),
        "max_abs_difference": float(difference.max().item()),
        "mean_abs_difference": float(difference.mean().item()),
    }


def _named_frame_comparison(
    base: Any, left_name: str, left: Sequence[Image.Image], right_name: str, right: Sequence[Image.Image]
) -> dict[str, Any]:
    """Turn the base tool's generic pixel report into an explicitly named pair."""
    raw = base._compare_frames(left, right)
    frames = [
        {
            "frame_index": item["frame_index"],
            f"{left_name}_sha256": item["graph_sha256"],
            f"{right_name}_sha256": item["eager_sha256"],
            "hash_equal": item["hash_equal"],
            f"{left_name}_size": item["graph_size"],
            f"{right_name}_size": item["eager_size"],
            "max_abs_rgb_difference": item["max_abs_rgb_difference"],
            "mean_abs_rgb_difference": item["mean_abs_rgb_difference"],
            "nonzero_rgb_values": item["nonzero_rgb_values"],
        }
        for item in raw["frames"]
    ]
    return {
        "left": left_name,
        "right": right_name,
        "comparable": raw["comparable"],
        f"frame_count_{left_name}": raw["frame_count_graph"],
        f"frame_count_{right_name}": raw["frame_count_eager"],
        f"sequence_sha256_{left_name}": raw["sequence_sha256_graph"],
        f"sequence_sha256_{right_name}": raw["sequence_sha256_eager"],
        "all_frame_hashes_equal": raw["all_frame_hashes_equal"],
        "max_abs_rgb_difference": raw["max_abs_rgb_difference"],
        "mean_abs_rgb_difference": raw["mean_abs_rgb_difference"],
        "nonzero_rgb_values": raw["nonzero_rgb_values"],
        "total_rgb_values": raw["total_rgb_values"],
        "frames": frames,
    }


def _within_pixel_tolerance(comparison: Mapping[str, Any], args: argparse.Namespace) -> bool:
    maximum = comparison.get("max_abs_rgb_difference")
    mean = comparison.get("mean_abs_rgb_difference")
    return bool(
        comparison.get("comparable")
        and maximum is not None
        and mean is not None
        and maximum <= args.max_abs_rgb_difference
        and mean <= args.mean_abs_rgb_difference
    )


@torch.inference_mode()
def _prepare_continuation_input(
    pipeline: Any, session: Any, actions: Mapping[str, bool], frames: int
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = session.first_frame_latent.shape
    noise = torch.randn(
        (1, shape[1], frames, shape[3], shape[4]),
        generator=session.generator,
        device=pipeline.device,
        dtype=torch.float32,
    ).to(dtype=pipeline.torch_dtype)
    action_context = pipeline.build_action_context(
        actions,
        latent_frames=frames,
        height=pipeline.config.height,
        width=pipeline.config.width,
        device=pipeline.device,
        dtype=pipeline.torch_dtype,
    )
    return noise, action_context


@torch.inference_mode()
def _steady_state_eager_denoise(
    pipeline: Any,
    session: Any,
    latent: torch.Tensor,
    action_context: torch.Tensor,
) -> torch.Tensor:
    """Execute exactly the graph wrapper's static DiT calls, but without capture."""
    stage = pipeline.denoise_stage
    dit = stage.dit
    frames = latent.shape[2]
    frame_tokens = (latent.shape[-2] // dit.patch_size[1]) * (latent.shape[-1] // dit.patch_size[2])
    capacity = session.self_cache[0]["k"].shape[1]
    sink_tokens = dit.sink_size * frame_tokens
    rolled_tokens = capacity - sink_tokens - frames * frame_tokens
    if rolled_tokens < 0:
        raise ValueError("fixed ABot block does not fit in the rolling cache tail")
    scratch_shape = (latent.shape[0], rolled_tokens, dit.num_heads, dit.dim // dit.num_heads)
    roll_scratch_k = torch.empty(scratch_shape, dtype=latent.dtype, device=latent.device)
    roll_scratch_v = torch.empty_like(roll_scratch_k)
    current_end = torch.tensor(
        [(session.next_latent_frame + frames) * frame_tokens],
        dtype=torch.long,
        device=latent.device,
    )
    timesteps = stage._official_denoising_timesteps(session.scheduler).to(device=latent.device)
    current = latent
    for index, timestep_value in enumerate(timesteps):
        timestep = torch.full((1, frames), timestep_value, dtype=timesteps.dtype, device=latent.device)
        with torch.autocast(latent.device.type, dtype=pipeline.torch_dtype, enabled=latent.device.type == "cuda"):
            flow_prediction = dit.forward_steady_state(
                x=current.to(dtype=pipeline.torch_dtype),
                timestep=timestep,
                context=session.prompt_emb,
                act_context=action_context,
                kv_cache=session.self_cache,
                crossattn_cache=session.cross_cache,
                current_end=current_end,
                roll_scratch_k=roll_scratch_k,
                roll_scratch_v=roll_scratch_v,
                update_cache=index == 0,
            )
        x0 = stage._x0_prediction(flow_prediction, current, timestep, session.scheduler)
        if index < len(timesteps) - 1:
            noise = torch.randn(x0.shape, generator=session.generator, dtype=x0.dtype, device=latent.device)
            current = session.scheduler.add_noise(x0, noise, timesteps[index + 1])
        else:
            # Mirror _ABotSteadyCudaGraph.run(): its refinement graph's output
            # buffer is reused by the context-cache replay, so it explicitly
            # preserves the terminal latent first.
            current = x0.clone()
    with torch.autocast(latent.device.type, dtype=pipeline.torch_dtype, enabled=latent.device.type == "cuda"):
        dit.forward_steady_state(
            x=current.to(dtype=pipeline.torch_dtype),
            timestep=torch.zeros_like(timestep),
            context=session.prompt_emb,
            act_context=action_context,
            kv_cache=session.self_cache,
            crossattn_cache=session.cross_cache,
            current_end=current_end,
            roll_scratch_k=roll_scratch_k,
            roll_scratch_v=roll_scratch_v,
            update_cache=False,
        )
    return current


@torch.inference_mode()
def _decode_and_advance(pipeline: Any, session: Any, latents: torch.Tensor) -> list[Image.Image]:
    if session.taew_decode_state is None:
        raise RuntimeError("ABot session is missing its TAeW decode state")
    decoded = pipeline.taew_decode_stage.decode_chunks(latents, [session.taew_decode_state])
    frames = pipeline.tensor2video(decoded[0])
    session.next_latent_frame += latents.shape[2]
    session.emitted_frames += len(frames)
    return frames


def _timed(device: torch.device, callback: Any) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    result = callback()
    torch.cuda.synchronize(device)
    return result, time.perf_counter() - started_at


@torch.inference_mode()
def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("three-way CUDA Graph diagnostic requires CUDA")
    base = _load_base_validator()
    image = Image.open(args.image).convert("RGB")
    pipeline = None
    sessions: dict[str, Any] = {}
    try:
        pipeline = base._make_pipeline(args)
        device = torch.device(pipeline.device)
        if device.type != "cuda":
            raise RuntimeError(f"three-way CUDA Graph diagnostic requires CUDA, got {pipeline.device!r}")
        pipeline.preload_models()
        pipeline.denoise_stage.configure_cuda_graph(False)
        roles = ("regular_eager", "steady_state_eager", "cuda_graph")
        for role in roles:
            sessions[role] = pipeline.create_interactive_session(
                image,
                args.prompt,
                seed=args.seed,
                session_id=f"cuda-graph-three-way-{role}",
            )

        local_attn_size = int(pipeline.denoise_stage.dit.local_attn_size)
        warmup_chunks = math.ceil(local_attn_size / args.control_latent_frames) + args.extra_warmup_chunks
        warmup_hashes_equal: list[bool] = []
        for _ in range(warmup_chunks):
            warm_frames = {
                role: pipeline.generate_next_block(
                    sessions[role],
                    args.action_keys,
                    control_latent_frames=args.control_latent_frames,
                )
                for role in roles
            }
            reference = base._sequence_hash(warm_frames["regular_eager"])
            warmup_hashes_equal.append(all(base._sequence_hash(warm_frames[role]) == reference for role in roles[1:]))
        torch.cuda.synchronize(device)

        cache_readiness = {role: base._cache_readiness(session, pipeline) for role, session in sessions.items()}
        regular_tree = _session_state_tree(sessions["regular_eager"], pipeline)
        state_equivalence = {
            "regular_vs_steady_state": _tree_exactness(
                regular_tree,
                _session_state_tree(sessions["steady_state_eager"], pipeline),
            ),
            "regular_vs_cuda_graph": _tree_exactness(
                regular_tree,
                _session_state_tree(sessions["cuda_graph"], pipeline),
            ),
        }

        inputs = {
            role: _prepare_continuation_input(
                pipeline,
                sessions[role],
                args.action_keys,
                args.control_latent_frames,
            )
            for role in roles
        }
        input_equivalence = {
            "regular_vs_steady_state": _compare_tensor(inputs["regular_eager"][0], inputs["steady_state_eager"][0]),
            "regular_vs_cuda_graph": _compare_tensor(inputs["regular_eager"][0], inputs["cuda_graph"][0]),
        }

        stage = pipeline.denoise_stage
        pipeline.denoise_stage.configure_cuda_graph(False)
        regular_latent, regular_denoise_seconds = _timed(
            device,
            lambda: stage._denoise_block(
                inputs["regular_eager"][0],
                sessions["regular_eager"].prompt_emb,
                inputs["regular_eager"][1],
                None,
                sessions["regular_eager"].self_cache,
                sessions["regular_eager"].cross_cache,
                sessions["regular_eager"].next_latent_frame,
                sessions["regular_eager"].generator,
                sessions["regular_eager"].scheduler,
            ),
        )
        steady_latent, steady_denoise_seconds = _timed(
            device,
            lambda: _steady_state_eager_denoise(
                pipeline,
                sessions["steady_state_eager"],
                inputs["steady_state_eager"][0],
                inputs["steady_state_eager"][1],
            ),
        )
        pipeline.denoise_stage.configure_cuda_graph(True)
        graph_latent, graph_denoise_seconds = _timed(
            device,
            lambda: stage.denoise_interactive_block(
                session_id=sessions["cuda_graph"].session_id,
                latent=inputs["cuda_graph"][0],
                prompt_emb=sessions["cuda_graph"].prompt_emb,
                action_context=inputs["cuda_graph"][1],
                self_cache=sessions["cuda_graph"].self_cache,
                cross_cache=sessions["cuda_graph"].cross_cache,
                current_start=sessions["cuda_graph"].next_latent_frame,
                generator=sessions["cuda_graph"].generator,
                scheduler=sessions["cuda_graph"].scheduler,
            ),
        )
        graph_last_metrics = dict(stage.last_cuda_graph_metrics())
        graph_runtime_metrics = dict(stage.cuda_graph_metrics())
        graph_verification = base._graph_verified(graph_last_metrics)
        pipeline.denoise_stage.configure_cuda_graph(False)

        latent_comparisons = {
            "regular_vs_steady_state": _compare_tensor(regular_latent, steady_latent),
            "steady_state_vs_cuda_graph": _compare_tensor(steady_latent, graph_latent),
            "regular_vs_cuda_graph": _compare_tensor(regular_latent, graph_latent),
        }
        regular_frames, regular_decode_seconds = _timed(
            device,
            lambda: _decode_and_advance(pipeline, sessions["regular_eager"], regular_latent),
        )
        steady_frames, steady_decode_seconds = _timed(
            device,
            lambda: _decode_and_advance(pipeline, sessions["steady_state_eager"], steady_latent),
        )
        graph_frames, graph_decode_seconds = _timed(
            device,
            lambda: _decode_and_advance(pipeline, sessions["cuda_graph"], graph_latent),
        )
        frame_comparisons = {
            "regular_vs_steady_state": _named_frame_comparison(
                base, "regular_eager", regular_frames, "steady_state_eager", steady_frames
            ),
            "steady_state_vs_cuda_graph": _named_frame_comparison(
                base, "steady_state_eager", steady_frames, "cuda_graph", graph_frames
            ),
            "regular_vs_cuda_graph": _named_frame_comparison(
                base, "regular_eager", regular_frames, "cuda_graph", graph_frames
            ),
        }
        warmup_valid = all(warmup_hashes_equal) and all(item["ready"] for item in cache_readiness.values())
        state_valid = all(item["exact"] for item in state_equivalence.values()) and all(
            item["exact"] for item in input_equivalence.values()
        )
        pixels_valid = all(_within_pixel_tolerance(item, args) for item in frame_comparisons.values())
        latents_valid = all(item["exact"] for item in latent_comparisons.values())
        if not warmup_valid or not state_valid:
            status = "invalid_precondition"
        elif not graph_verification["verified"]:
            status = "graph_unverified"
        elif not latent_comparisons["regular_vs_steady_state"]["exact"] or not _within_pixel_tolerance(
            frame_comparisons["regular_vs_steady_state"], args
        ):
            status = "steady_state_model_mismatch"
        elif not latent_comparisons["steady_state_vs_cuda_graph"]["exact"] or not _within_pixel_tolerance(
            frame_comparisons["steady_state_vs_cuda_graph"], args
        ):
            status = "cuda_graph_wrapper_mismatch"
        elif not latents_valid or not pixels_valid:
            status = "cuda_graph_parity_mismatch"
        else:
            status = "pass"
        return {
            "status": status,
            "diagnostic": "regular eager vs eager steady-state vs captured CUDA Graph",
            "device": str(device),
            "seed": args.seed,
            "actions": args.action_keys,
            "control_latent_frames": args.control_latent_frames,
            "warmup": {
                "chunks": warmup_chunks,
                "per_chunk_all_three_hashes_equal": warmup_hashes_equal,
                "all_three_outputs_equal": all(warmup_hashes_equal),
                "cache_readiness": cache_readiness,
            },
            "state_equivalence_before_continuation": state_equivalence,
            "initial_noise_equivalence": input_equivalence,
            "continuation_timings_seconds": {
                "regular_eager": {"denoise": regular_denoise_seconds, "decode": regular_decode_seconds},
                "steady_state_eager": {"denoise": steady_denoise_seconds, "decode": steady_decode_seconds},
                "cuda_graph": {"denoise": graph_denoise_seconds, "decode": graph_decode_seconds},
            },
            "cuda_graph": {
                "last_metrics": graph_last_metrics,
                "runtime_metrics": graph_runtime_metrics,
                "verification": graph_verification,
            },
            "latent_comparisons": latent_comparisons,
            "frame_comparisons": frame_comparisons,
            "pixel_tolerance": {
                "max_abs_rgb_difference": args.max_abs_rgb_difference,
                "mean_abs_rgb_difference": args.mean_abs_rgb_difference,
            },
        }
    finally:
        if pipeline is not None:
            for session in sessions.values():
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


def _write_output(output_dir: Path, args: argparse.Namespace, result: Mapping[str, Any], base: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(
            {"arguments": base._json_safe(vars(args)), "result": base._json_safe(result)}, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    graph = result.get("cuda_graph", {})
    verification = graph.get("verification", {}) if isinstance(graph, Mapping) else {}
    lines = [
        "# ABot CUDA Graph three-way diagnostic",
        "",
        (
            "Three same-seed B=1 sessions were warmed eagerly to a full KV window, then run through "
            "regular eager, static eager, and captured CUDA-Graph continuations."
        ),
        "",
        "| Pair / fact | Exact RGB hashes | Max RGB abs diff | Mean RGB abs diff |",
        "| --- | --- | ---: | ---: |",
    ]
    for name, comparison in result.get("frame_comparisons", {}).items():
        lines.append(
            f"| {name} | {comparison.get('all_frame_hashes_equal', False)} | "
            f"{comparison.get('max_abs_rgb_difference', '')} | {comparison.get('mean_abs_rgb_difference', '')} |"
        )
    lines.extend(
        [
            "",
            f"Status: `{result.get('status', 'error')}`.",
            "",
            f"Graph capture/replay/fallback: `{verification.get('captured', False)}` / "
            f"`{verification.get('replay_observed', False)}` / `{verification.get('fallback_observed', False)}`.",
            "",
            (
                "`results.json` includes exact pre-continuation session-state checks, initial-noise checks, "
                "latent comparisons, and per-frame hashes."
            ),
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    base = _load_base_validator()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--action-keys", type=base._parse_action_keys, default={"W": True})
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-latent-frames", type=int, choices=(3,), default=3)
    parser.add_argument("--extra-warmup-chunks", type=int, default=0)
    parser.add_argument("--max-abs-rgb-difference", type=int, default=0)
    parser.add_argument("--mean-abs-rgb-difference", type=float, default=0.0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
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
                    "sessions": ["regular_eager", "steady_state_eager", "cuda_graph"],
                    "warmup": "all three sessions eager until the 18-latent-frame KV window is full",
                    "continuation_paths": [
                        "regular dynamic eager",
                        "forward_steady_state eager",
                        "captured CUDA Graph",
                    ],
                    "comparisons": ["state", "initial noise", "final latent", "decoded RGB pixels/hashes"],
                    "zero_tolerance": {
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
    base = _load_base_validator()
    try:
        result = _run(args)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_output(args.output_dir, args, result, base)
    print(json.dumps(base._json_safe(result), indent=2, sort_keys=True))
    if result.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
