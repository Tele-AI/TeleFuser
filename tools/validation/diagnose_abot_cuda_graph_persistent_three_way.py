"""Isolate persistent CUDA-Graph replay errors for ABot B=1, B=2, and B=3.

This correctness diagnostic starts three same-seed retained-session cohorts:

* ordinary public eager;
* persistent-static eager, using the same fixed static tensors and, for B=2/3,
  the same persistent KV arena layout as the graph path, but calling
  forward_steady_state normally rather than replaying a CUDA Graph;
* ordinary public CUDA-Graph capture then persistent replay.

Every cohort is warmed through the public eager path until the causal KV
window is full.  It then executes exactly two continuation chunks.  After
each chunk, it compares every lane's DiT latent, rendered RGB frames, and the
complete retained state (KV, cross-cache, RNG, decoder state, and counters).

The first comparison identifies a static-model or capture error.  The second
comparison is the persistent-replay test that a capture-only parity check
misses.  B=2/3 additionally prove that the graph metrics describe one native
batched graph rather than singleton graphs.

Example:

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/diagnose_abot_cuda_graph_persistent_three_way.py \\
      --batch-size 1 \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir results/validation/abot_cuda_graph_b1_persistent_three_way

Process-NCCL logical-device smoke (four physical devices visible to one
worker process, using local cuda:1 / physical GPU 5)::

    CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONPATH=$PWD \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/diagnose_abot_cuda_graph_persistent_three_way.py \\
      --batch-size 2 --device-id 1 \\
      --expected-cuda-visible-devices 4,5,6,7 \\
      --expected-visible-device-count 4 --nccl-single-rank \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir results/validation/abot_cuda_graph_b2_nccl_visible4567_device1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image


def _load_base_validator() -> Any:
    path = Path(__file__).with_name("validate_abot_cuda_graph_parity.py")
    spec = importlib.util.spec_from_file_location("abot_cuda_graph_persistent_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load base CUDA-Graph validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_cuda_visible_devices(value: str | None) -> list[str] | None:
    """Return the process-visible device tokens without touching CUDA."""
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _expected_visible_devices(value: str | None) -> list[str] | None:
    """Validate the optional exact CVD mapping requested by the operator."""
    if value is None:
        return None
    devices = _split_cuda_visible_devices(value)
    if not devices:
        raise ValueError("--expected-cuda-visible-devices must contain at least one device token")
    return devices


def _device_context_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Validate CVD/local-index assumptions before loading a CUDA model.

    Process-NCCL workers receive *logical* GPU ids. For example, a process
    launched with ``CUDA_VISIBLE_DEVICES=4,5,6,7`` must use ``device_id=1``
    to select physical GPU 5; passing ``5`` would be invalid in that process.
    """
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = _split_cuda_visible_devices(raw_visible)
    expected = _expected_visible_devices(args.expected_cuda_visible_devices)
    if expected is not None and visible != expected:
        observed = "<unset>" if visible is None else ",".join(visible)
        raise ValueError(
            "CUDA_VISIBLE_DEVICES does not match --expected-cuda-visible-devices: "
            f"expected {','.join(expected)!r}, observed {observed!r}"
        )
    if args.expected_visible_device_count is not None and visible is not None:
        if len(visible) != args.expected_visible_device_count:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES count does not match --expected-visible-device-count: "
                f"expected {args.expected_visible_device_count}, observed {len(visible)}"
            )
    if visible is not None and not 0 <= args.device_id < len(visible):
        raise ValueError(f"--device-id {args.device_id} is not a logical index in CUDA_VISIBLE_DEVICES={raw_visible!r}")
    return {
        "cuda_visible_devices": raw_visible,
        "visible_device_tokens": visible,
        "expected_cuda_visible_devices": expected,
        "expected_visible_device_count": args.expected_visible_device_count,
        "logical_device_requested": args.device_id,
        "physical_device_token_for_logical_device": (
            visible[args.device_id] if visible is not None and 0 <= args.device_id < len(visible) else None
        ),
    }


def _activate_cuda_device_context(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Select the requested local CUDA device before any pipeline/graph work.

    ``torch.cuda.graph`` uses the current CUDA device for its capture stream.
    This explicit selection mirrors ``_run_nccl_model_worker`` and avoids a
    standalone validator accidentally capturing on logical cuda:0 while its
    pipeline tensors reside on logical cuda:1.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("persistent CUDA-Graph diagnostic requires CUDA")
    runtime_count = int(torch.cuda.device_count())
    if args.expected_visible_device_count is not None and runtime_count != args.expected_visible_device_count:
        raise RuntimeError(
            "torch.cuda.device_count() does not match --expected-visible-device-count: "
            f"expected {args.expected_visible_device_count}, observed {runtime_count}"
        )
    if not 0 <= args.device_id < runtime_count:
        raise RuntimeError(f"--device-id {args.device_id} is out of range for {runtime_count} visible CUDA device(s)")
    # Do not call current_device before this: that can initialize CUDA on the
    # wrong default lane and would not reproduce a process-NCCL worker.
    torch.cuda.set_device(args.device_id)
    current = int(torch.cuda.current_device())
    if current != args.device_id:
        raise RuntimeError(f"failed to select logical cuda:{args.device_id}; current device is cuda:{current}")
    properties = torch.cuda.get_device_properties(args.device_id)
    return {
        **dict(plan),
        "runtime_visible_device_count": runtime_count,
        "logical_device_current_after_set": current,
        "selected_device": f"cuda:{args.device_id}",
        "selected_device_name": properties.name,
        "selected_device_capability": [int(properties.major), int(properties.minor)],
    }


def _free_loopback_port() -> int:
    """Allocate a short-lived loopback rendezvous port for a rank-0 group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_single_rank_nccl(args: argparse.Namespace, device: torch.device) -> tuple[dict[str, Any], bool]:
    """Optionally reproduce the worker-local initialized-NCCL context.

    It is intentionally world-size one: CUDA Graph parity needs the same
    process-local current-device and communicator initialization ordering as a
    process-NCCL model worker, not a second 25-GB replica or a migration test.
    """
    report: dict[str, Any] = {
        "requested": bool(args.nccl_single_rank),
        "initialized_by_tool": False,
        "already_initialized": bool(dist.is_available() and dist.is_initialized()),
    }
    if not args.nccl_single_rank:
        return report, False
    if not dist.is_available() or not dist.is_nccl_available():
        raise RuntimeError("--nccl-single-rank requires a PyTorch build with NCCL support")
    if dist.is_initialized():
        raise RuntimeError("--nccl-single-rank refuses to reuse or destroy an existing process group")
    port = _free_loopback_port()
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=args.nccl_init_timeout_seconds),
            device_id=device,
        )
    except Exception:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    if not dist.is_initialized() or str(dist.get_backend()) != "nccl":
        if dist.is_initialized():
            dist.destroy_process_group()
        raise RuntimeError("single-rank NCCL initialization did not produce an NCCL process group")
    report.update(
        {
            "initialized_by_tool": True,
            "backend": str(dist.get_backend()),
            "rank": int(dist.get_rank()),
            "world_size": int(dist.get_world_size()),
            "current_cuda_device": int(torch.cuda.current_device()),
        }
    )
    return report, True


def _tree_exactness(left: Any, right: Any) -> dict[str, Any]:
    """Compare a retained-state tree directly on device without CPU KV copies."""
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
                visit(lhs_item, rhs_item, f"{path}[{index}]")
                for index, (lhs_item, rhs_item) in enumerate(zip(lhs, rhs, strict=True))
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
        "mismatch_count_at_least": len(mismatches),
        "mismatches": mismatches[:20],
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


def _compare_tensor(left: torch.Tensor | None, right: torch.Tensor | None) -> dict[str, Any]:
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
    """Record the exact denoised tensor consumed by public LightVAE decode."""

    def __init__(self, decode_stage: Any) -> None:
        self._decode_stage = decode_stage
        self._original: Any = None
        self.latents: torch.Tensor | None = None

    def __enter__(self) -> "_DecodeLatentCapture":
        self._original = self._decode_stage.decode_chunks

        def capture(latents: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            self.latents = latents.detach().clone()
            return self._original(latents, *args, **kwargs)

        self._decode_stage.decode_chunks = capture
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._decode_stage.decode_chunks = self._original


def _parse_actions(raw: str, batch_size: int, base: Any) -> list[dict[str, bool]]:
    items = [item.strip() for item in raw.split(";") if item.strip()]
    if len(items) < batch_size:
        raise ValueError(
            f"--session-actions supplies {len(items)} action sets, but batch size {batch_size} needs one per lane"
        )
    return [base._parse_action_keys(item) for item in items[:batch_size]]


@torch.inference_mode()
def _run_public(
    pipeline: Any,
    sessions: Sequence[Any],
    actions: Sequence[Mapping[str, bool]],
    *,
    control_latent_frames: int,
    device: torch.device,
) -> dict[str, Any]:
    with _DecodeLatentCapture(pipeline.taew_decode_stage) as capture:
        frames = pipeline.generate_next_blocks(sessions, actions, control_latent_frames=control_latent_frames)
    torch.cuda.synchronize(device)
    return {
        "latents": capture.latents,
        "frames": frames,
        "stage_metrics": dict(pipeline.last_stage_metrics()),
    }


@torch.inference_mode()
def _prepare_inputs(
    pipeline: Any,
    sessions: Sequence[Any],
    actions: Sequence[Mapping[str, bool]],
    frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror public generate_next_blocks input construction byte-for-byte."""
    noises: list[torch.Tensor] = []
    action_contexts: list[torch.Tensor] = []
    for session, session_actions in zip(sessions, actions, strict=True):
        shape = session.first_frame_latent.shape
        noises.append(
            torch.randn(
                (1, shape[1], frames, shape[3], shape[4]),
                generator=session.generator,
                device=pipeline.device,
                dtype=torch.float32,
            )
        )
        action_contexts.append(
            pipeline.build_action_context(
                session_actions,
                latent_frames=frames,
                height=pipeline.config.height,
                width=pipeline.config.width,
                device=pipeline.device,
                dtype=pipeline.torch_dtype,
            )
        )
    return (
        torch.cat(noises, dim=0).to(dtype=pipeline.torch_dtype),
        torch.cat([session.prompt_emb for session in sessions], dim=0),
        torch.cat(action_contexts, dim=0),
    )


def _static_state(
    pipeline: Any,
    sessions: Sequence[Any],
    latent: torch.Tensor,
    prompt_emb: torch.Tensor,
    action_context: torch.Tensor,
) -> dict[str, Any]:
    """Allocate a persistent static control with graph-equivalent storage."""
    stage = pipeline.denoise_stage
    if len(sessions) == 1:
        from telefuser.pipelines.abot_world.denoising import _ABotSteadyCudaGraph

        session = sessions[0]
        graph = _ABotSteadyCudaGraph(
            stage.dit,
            latent,
            prompt_emb,
            action_context,
            session.self_cache,
            session.cross_cache,
            torch_dtype=pipeline.torch_dtype,
        )
        return {
            "graph": graph,
            "self_cache": session.self_cache,
            "cross_cache": session.cross_cache,
            "arena_state": None,
        }
    arena = stage._create_batched_cuda_graph_state(
        tuple(session.session_id for session in sessions),
        latent,
        prompt_emb,
        action_context,
        [session.self_cache for session in sessions],
        [session.cross_cache for session in sessions],
        current_starts=[session.next_latent_frame for session in sessions],
    )
    return {
        "graph": arena.graph,
        "self_cache": arena.self_cache,
        "cross_cache": arena.cross_cache,
        "arena_state": arena,
    }


@torch.inference_mode()
def _static_denoise(
    pipeline: Any,
    state: Mapping[str, Any],
    latent: torch.Tensor,
    action_context: torch.Tensor,
    *,
    current_start: int,
    generators: Sequence[torch.Generator],
    scheduler: Any,
) -> torch.Tensor:
    """Run graph-equivalent persistent buffers through eager forward_steady_state."""
    stage = pipeline.denoise_stage
    graph = state["graph"]
    current_end = (current_start + graph.frames) * graph.frame_tokens
    timesteps = stage._official_denoising_timesteps(scheduler).to(device=latent.device)
    generator: torch.Generator | Sequence[torch.Generator]
    generator = generators[0] if len(generators) == 1 else generators
    current = latent
    for index, current_timestep in enumerate(timesteps):
        graph._set_inputs(current, action_context, current_timestep, current_end=current_end)
        with torch.autocast(latent.device.type, dtype=pipeline.torch_dtype, enabled=latent.device.type == "cuda"):
            flow_prediction = stage.dit.forward_steady_state(
                x=graph.static_x,
                timestep=graph.static_timestep,
                context=graph.static_context,
                act_context=graph.static_action,
                kv_cache=state["self_cache"],
                crossattn_cache=state["cross_cache"],
                current_end=graph.current_end,
                roll_scratch_k=graph.roll_scratch_k,
                roll_scratch_v=graph.roll_scratch_v,
                update_cache=index == 0,
            )
        x0 = stage._x0_prediction(flow_prediction, current, graph.static_timestep, scheduler)
        if index < len(timesteps) - 1:
            current = scheduler.add_noise(x0, graph._draw_noise(x0, generator), timesteps[index + 1])
        else:
            current = x0
    # Mirror the production graph path: the final dynamic cache-only call
    # receives independent x0 storage rather than a static graph output view.
    context_input = current.clone()
    # Match the public eager cache-only call: it is outside the sampler's
    # autocast scope.
    stage.dit(
        x=context_input.to(dtype=pipeline.torch_dtype),
        timestep=torch.zeros_like(graph.static_timestep),
        context=graph.static_context,
        act_context=action_context,
        kv_cache=state["self_cache"],
        crossattn_cache=state["cross_cache"],
        current_start=current_start * graph.frame_tokens,
    )
    return current


@torch.inference_mode()
def _run_static(
    pipeline: Any,
    sessions: Sequence[Any],
    actions: Sequence[Mapping[str, bool]],
    *,
    control_latent_frames: int,
    state: dict[str, Any] | None,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latent, prompt_emb, action_context = _prepare_inputs(pipeline, sessions, actions, control_latent_frames)
    starts = [session.next_latent_frame for session in sessions]
    if len(set(starts)) != 1:
        raise RuntimeError("persistent static control requires aligned continuation positions")
    if state is None:
        state = _static_state(pipeline, sessions, latent, prompt_emb, action_context)
    latents = _static_denoise(
        pipeline,
        state,
        latent,
        action_context,
        current_start=starts[0],
        generators=[session.generator for session in sessions],
        scheduler=sessions[0].scheduler,
    )
    arena = state["arena_state"]
    if arena is not None:
        pipeline.denoise_stage._bind_batched_cache_arena(
            arena,
            [session.self_cache for session in sessions],
            [session.cross_cache for session in sessions],
        )
        pipeline.denoise_stage._advance_batched_cache_cursors(
            [session.self_cache for session in sessions],
            current_starts=starts,
            latent=latents,
        )
    if any(session.taew_decode_state is None for session in sessions):
        raise RuntimeError("ABot session is missing its TAeW decode state")
    decoded = pipeline.taew_decode_stage.decode_chunks(latents, [session.taew_decode_state for session in sessions])
    frames = []
    for index, session in enumerate(sessions):
        session_frames = pipeline.tensor2video(decoded[index])
        session.next_latent_frame += control_latent_frames
        session.emitted_frames += len(session_frames)
        frames.append(session_frames)
    torch.cuda.synchronize(device)
    return {"latents": latents.detach().clone(), "frames": frames}, state


def _graph_evidence(metrics: Mapping[str, Any], batch_size: int, *, capture: bool, base: Any) -> dict[str, Any]:
    graph = dict(base._graph_verified(metrics))
    observed_batch_size = metrics.get("batch_size")
    graph_batch_size = metrics.get("cuda_graph_batch_size")
    graph_batched = bool(int(metrics.get("cuda_graph_batched", 0)))
    batch_valid = batch_size == 1 or (
        observed_batch_size == batch_size and graph_batch_size == batch_size and graph_batched
    )
    graph["expected_batch_size"] = batch_size
    graph["observed_batch_size"] = observed_batch_size
    graph["observed_cuda_graph_batch_size"] = graph_batch_size
    graph["cuda_graph_batched"] = graph_batched
    graph["native_batch_valid"] = batch_valid
    graph["verified"] = bool(
        graph["enabled"]
        and graph["eligible"]
        and graph["replay_observed"]
        and not graph["fallback_observed"]
        and (graph["captured"] if capture else True)
        and batch_valid
    )
    return graph


def _round_comparisons(
    base: Any,
    regular: Mapping[str, Any],
    static: Mapping[str, Any],
    graph: Mapping[str, Any],
    regular_sessions: Sequence[Any],
    static_sessions: Sequence[Any],
    graph_sessions: Sequence[Any],
    pipeline: Any,
) -> dict[str, Any]:
    lanes: list[dict[str, Any]] = []
    for index in range(len(regular_sessions)):
        regular_latent = regular["latents"][index : index + 1]
        static_latent = static["latents"][index : index + 1]
        graph_latent = graph["latents"][index : index + 1] if graph["latents"] is not None else None
        lanes.append(
            {
                "lane": index,
                "regular_vs_static": {
                    "latent": _compare_tensor(regular_latent, static_latent),
                    "rgb": base._compare_frames(regular["frames"][index], static["frames"][index]),
                    "state": _tree_exactness(
                        _session_state_tree(regular_sessions[index], pipeline),
                        _session_state_tree(static_sessions[index], pipeline),
                    ),
                },
                "static_vs_graph": {
                    "latent": _compare_tensor(static_latent, graph_latent),
                    "rgb": base._compare_frames(static["frames"][index], graph["frames"][index]),
                    "state": _tree_exactness(
                        _session_state_tree(static_sessions[index], pipeline),
                        _session_state_tree(graph_sessions[index], pipeline),
                    ),
                },
                "regular_vs_graph": {
                    "latent": _compare_tensor(regular_latent, graph_latent),
                    "rgb": base._compare_frames(regular["frames"][index], graph["frames"][index]),
                    "state": _tree_exactness(
                        _session_state_tree(regular_sessions[index], pipeline),
                        _session_state_tree(graph_sessions[index], pipeline),
                    ),
                },
            }
        )
    return {"lanes": lanes}


def _pair_exact(round_report: Mapping[str, Any], pair: str) -> bool:
    for lane in round_report.get("lanes", []):
        comparison = lane.get(pair, {})
        latent = comparison.get("latent", {})
        rgb = comparison.get("rgb", {})
        state = comparison.get("state", {})
        if not (latent.get("exact") and rgb.get("all_frame_hashes_equal") and state.get("exact")):
            return False
    return bool(round_report.get("lanes"))


@torch.inference_mode()
def _run(args: argparse.Namespace, base: Any) -> dict[str, Any]:
    device_context = _activate_cuda_device_context(args, _device_context_plan(args))
    image = Image.open(args.image).convert("RGB")
    pipeline = None
    nccl_context: dict[str, Any] = {
        "requested": bool(args.nccl_single_rank),
        "initialized_by_tool": False,
    }
    nccl_owned = False
    cohorts: dict[str, list[Any]] = {"regular": [], "static": [], "graph": []}
    try:
        pipeline = base._make_pipeline(args)
        device = torch.device(pipeline.device)
        expected_device = torch.device("cuda", args.device_id)
        if device != expected_device:
            raise RuntimeError(
                "pipeline device does not match the requested logical CUDA device: "
                f"expected {expected_device}, got {pipeline.device!r}"
            )
        pipeline.preload_models()
        if int(torch.cuda.current_device()) != args.device_id:
            raise RuntimeError(
                "pipeline preload changed the current CUDA device: "
                f"expected cuda:{args.device_id}, got cuda:{torch.cuda.current_device()}"
            )
        nccl_context, nccl_owned = _start_single_rank_nccl(args, device)
        pipeline.denoise_stage.configure_cuda_graph(False)
        actions = _parse_actions(args.session_actions, args.batch_size, base)
        seeds = [args.seed + 9973 * index for index in range(args.batch_size)]
        for role, sessions in cohorts.items():
            for index, seed in enumerate(seeds):
                sessions.append(
                    pipeline.create_interactive_session(
                        image,
                        args.prompt,
                        seed=seed,
                        session_id=f"persistent-{role}-b{args.batch_size}-{index}",
                    )
                )

        warmup_chunks = base._required_warmup_chunks(
            int(pipeline.denoise_stage.dit.local_attn_size),
            args.control_latent_frames,
            args.extra_warmup_chunks,
        )
        warmup_hashes: list[list[bool]] = []
        for _ in range(warmup_chunks):
            outputs = {
                role: pipeline.generate_next_blocks(
                    sessions,
                    actions,
                    control_latent_frames=args.control_latent_frames,
                )
                for role, sessions in cohorts.items()
            }
            warmup_hashes.append(
                [
                    base._sequence_hash(outputs["regular"][lane])
                    == base._sequence_hash(outputs["static"][lane])
                    == base._sequence_hash(outputs["graph"][lane])
                    for lane in range(args.batch_size)
                ]
            )
        torch.cuda.synchronize(device)
        warmup_states = {
            role: [
                {
                    "cache": base._cache_readiness(session, pipeline),
                    "regular_state_exact": _tree_exactness(
                        _session_state_tree(cohorts["regular"][index], pipeline),
                        _session_state_tree(session, pipeline),
                    ),
                }
                for index, session in enumerate(sessions)
            ]
            for role, sessions in cohorts.items()
        }
        warmup_valid = all(all(chunk) for chunk in warmup_hashes) and all(
            item["cache"]["ready"] and item["regular_state_exact"]["exact"]
            for role_items in warmup_states.values()
            for item in role_items
        )

        pipeline.denoise_stage.configure_cuda_graph(False)
        regular_first = _run_public(
            pipeline,
            cohorts["regular"],
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        static_first, static_state = _run_static(
            pipeline,
            cohorts["static"],
            actions,
            control_latent_frames=args.control_latent_frames,
            state=None,
            device=device,
        )

        pipeline.denoise_stage.configure_cuda_graph(True)
        graph_first = _run_public(
            pipeline,
            cohorts["graph"],
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        graph_runtime_after_first = dict(pipeline.denoise_stage.cuda_graph_metrics())
        # All three cohorts are still at continuation one here. Compare now;
        # later state is mutable and cannot stand in for this checkpoint.
        first = _round_comparisons(
            base,
            regular_first,
            static_first,
            graph_first,
            cohorts["regular"],
            cohorts["static"],
            cohorts["graph"],
            pipeline,
        )
        # Avoid configure_cuda_graph(False): it deliberately clears resident
        # graphs. This diagnostic-only flag toggle preserves graph cohort state.
        stage = pipeline.denoise_stage
        stage._cuda_graph_enabled = False
        regular_second = _run_public(
            pipeline,
            cohorts["regular"],
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        static_second, static_state = _run_static(
            pipeline,
            cohorts["static"],
            actions,
            control_latent_frames=args.control_latent_frames,
            state=static_state,
            device=device,
        )
        stage._cuda_graph_enabled = True
        graph_second = _run_public(
            pipeline,
            cohorts["graph"],
            actions,
            control_latent_frames=args.control_latent_frames,
            device=device,
        )
        graph_runtime_after_second = dict(pipeline.denoise_stage.cuda_graph_metrics())
        first_evidence = _graph_evidence(graph_first["stage_metrics"], args.batch_size, capture=True, base=base)
        second_evidence = _graph_evidence(graph_second["stage_metrics"], args.batch_size, capture=False, base=base)

        second = _round_comparisons(
            base,
            regular_second,
            static_second,
            graph_second,
            cohorts["regular"],
            cohorts["static"],
            cohorts["graph"],
            pipeline,
        )
        static_first_exact = _pair_exact(first, "regular_vs_static")
        graph_first_exact = _pair_exact(first, "static_vs_graph")
        static_second_exact = _pair_exact(second, "regular_vs_static")
        graph_second_exact = _pair_exact(second, "static_vs_graph")
        if not warmup_valid:
            status = "invalid_warmup"
        elif not first_evidence["verified"] or not second_evidence["verified"]:
            status = "graph_unverified"
        elif not static_first_exact:
            status = "static_eager_first_continuation_mismatch"
        elif not graph_first_exact:
            status = "cuda_graph_capture_mismatch"
        elif not static_second_exact:
            status = "persistent_static_eager_mismatch"
        elif not graph_second_exact:
            status = "cuda_graph_persistent_replay_mismatch"
        else:
            status = "pass"
        return {
            "status": status,
            "diagnostic": "ordinary eager vs persistent-static eager vs CUDA Graph across two continuations",
            "device": str(device),
            "execution_context": device_context,
            "nccl_context": nccl_context,
            "batch_size": args.batch_size,
            "control_latent_frames": args.control_latent_frames,
            "session_actions": actions,
            "session_seeds": seeds,
            "warmup": {
                "chunks": warmup_chunks,
                "per_chunk_per_lane_hash_equal": warmup_hashes,
                "cohort_state": warmup_states,
                "valid": warmup_valid,
            },
            "cuda_graph": {
                "first_continuation": {
                    "stage_metrics": graph_first["stage_metrics"],
                    "runtime_metrics": graph_runtime_after_first,
                    "verification": first_evidence,
                },
                "second_continuation": {
                    "stage_metrics": graph_second["stage_metrics"],
                    "runtime_metrics": graph_runtime_after_second,
                    "verification": second_evidence,
                },
            },
            "rounds": {
                "first_continuation": first,
                "second_continuation_persistent_replay": second,
            },
            "classification": {
                "regular_vs_static_first_exact": static_first_exact,
                "static_vs_graph_first_exact": graph_first_exact,
                "regular_vs_static_second_exact": static_second_exact,
                "static_vs_graph_second_exact": graph_second_exact,
            },
        }
    finally:
        if pipeline is not None:
            for sessions in cohorts.values():
                for session in sessions:
                    try:
                        pipeline.close_interactive_session(session)
                    except Exception:
                        pass
            try:
                pipeline.close()
            except Exception:
                pass
        if nccl_owned and dist.is_initialized():
            dist.destroy_process_group()
        elif args.nccl_single_rank and dist.is_initialized():
            raise RuntimeError("single-rank NCCL diagnostic left an unowned process group initialized")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_results(output_dir: Path, result: Mapping[str, Any], args: argparse.Namespace, base: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"arguments": base._json_safe(vars(args)), "result": base._json_safe(result)}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    classification = result.get("classification", {})
    graph = result.get("cuda_graph", {})
    first_graph = graph.get("first_continuation", {}) if isinstance(graph, Mapping) else {}
    second_graph = graph.get("second_continuation", {}) if isinstance(graph, Mapping) else {}
    first_verification = first_graph.get("verification", {}) if isinstance(first_graph, Mapping) else {}
    second_verification = second_graph.get("verification", {}) if isinstance(second_graph, Mapping) else {}
    execution_context = result.get("execution_context", {})
    if not isinstance(execution_context, Mapping):
        execution_context = {}
    nccl_context = result.get("nccl_context", {})
    if not isinstance(nccl_context, Mapping):
        nccl_context = {}
    visible_tokens = execution_context.get("visible_device_tokens")
    visible_text = ",".join(str(item) for item in visible_tokens) if isinstance(visible_tokens, list) else "<unset>"
    logical = execution_context.get("logical_device_requested", "")
    physical = execution_context.get("physical_device_token_for_logical_device", "")
    current = execution_context.get("logical_device_current_after_set", "")
    nccl_summary = (
        f"requested={nccl_context.get('requested', False)}, "
        f"initialized={nccl_context.get('initialized_by_tool', False)}, "
        f"backend={nccl_context.get('backend', '')}, "
        f"rank/world={nccl_context.get('rank', '')}/{nccl_context.get('world_size', '')}"
    )
    lines = [
        f"# ABot B={args.batch_size} persistent CUDA-Graph three-way diagnostic",
        "",
        (
            "All three same-seed cohorts were eagerly warmed to full KV, then each ran two continuation chunks. "
            "The static control uses the graph's fixed buffers and, at B=2/3, its persistent KV arena but executes "
            "forward_steady_state eagerly."
        ),
        "",
        "| Process-NCCL compatibility context | Value |",
        "| --- | --- |",
        f"| CUDA_VISIBLE_DEVICES | `{visible_text}` |",
        f"| Selected logical CUDA / mapped physical token | `cuda:{logical}` / `{physical}` |",
        f"| Current logical CUDA after selection | `cuda:{current}` |",
        f"| Runtime visible CUDA device count | {execution_context.get('runtime_visible_device_count', '')} |",
        f"| Single-rank NCCL | {nccl_summary} |",
        "",
        "| Check | First continuation | Second continuation / persistent replay |",
        "| --- | --- | --- |",
        (
            "| ordinary eager == persistent-static eager | "
            f"{classification.get('regular_vs_static_first_exact', False)} | "
            f"{classification.get('regular_vs_static_second_exact', False)} |"
        ),
        (
            "| persistent-static eager == CUDA Graph | "
            f"{classification.get('static_vs_graph_first_exact', False)} | "
            f"{classification.get('static_vs_graph_second_exact', False)} |"
        ),
        (
            "| CUDA Graph verified | "
            f"{first_verification.get('verified', False)} | {second_verification.get('verified', False)} |"
        ),
        "",
        f"Status: {result.get('status', 'error')}.",
        "",
        (
            "results.json includes per-lane strict latent, RGB SHA-256/pixel, and complete retained-state "
            "comparisons after both continuations."
        ),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(base: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--session-actions", default="W;A;S")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-latent-frames", type=int, choices=(3,), default=3)
    parser.add_argument("--extra-warmup-chunks", type=int, default=0)
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Logical CUDA index after CUDA_VISIBLE_DEVICES remapping (not a physical GPU id).",
    )
    parser.add_argument(
        "--expected-cuda-visible-devices",
        default=None,
        help="Require this exact comma-separated CVD mapping, e.g. '4,5,6,7'.",
    )
    parser.add_argument(
        "--expected-visible-device-count",
        type=int,
        default=None,
        help="Require this number of CUDA-visible logical devices at runtime.",
    )
    parser.add_argument(
        "--nccl-single-rank",
        action="store_true",
        help="Initialize a world-size-one NCCL group after model preload, mirroring a process-NCCL worker.",
    )
    parser.add_argument(
        "--nccl-init-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for the optional single-rank NCCL initialization.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.device_id < 0:
        parser.error("--device-id must be non-negative")
    if args.expected_visible_device_count is not None and args.expected_visible_device_count < 1:
        parser.error("--expected-visible-device-count must be positive")
    if args.nccl_init_timeout_seconds <= 0:
        parser.error("--nccl-init-timeout-seconds must be positive")
    if args.extra_warmup_chunks < 0:
        parser.error("--extra-warmup-chunks must be non-negative")
    try:
        _parse_actions(args.session_actions, args.batch_size, base)
        _device_context_plan(args)
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
                    "cuda_device_context_plan": _device_context_plan(args),
                    "nccl_single_rank_requested": args.nccl_single_rank,
                    "nccl_ordering": (
                        "logical device is selected before pipeline load; optional NCCL initializes after preload"
                    ),
                    "cohorts": ["ordinary_eager", "persistent_static_eager", "cuda_graph"],
                    "continuations": ["first_capture", "second_persistent_replay"],
                    "comparisons_after_each": ["per-lane latent", "RGB frame hashes", "full retained state"],
                    "B2_graph_gate": (
                        "batch_size=B, cuda_graph_batch_size=B, cuda_graph_batched=1, replay>0, fallback=0"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    assert args.output_dir is not None
    try:
        result = _run(args, base)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_results(args.output_dir, result, args, base)
    print(json.dumps(base._json_safe(result), indent=2, sort_keys=True))
    if result.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
