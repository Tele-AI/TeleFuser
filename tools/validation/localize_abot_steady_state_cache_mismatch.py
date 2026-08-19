"""Locate ABot B=1 dynamic-versus-steady-state KV cache divergence.

This is a tools-only, single-continuation diagnostic. It warms ordinary and
steady-state twins to the same full causal KV window, then interleaves each
ordinary DiT denoising call with its persistent-static counterpart. After each
pair it performs an exact on-device comparison of layer-0 K/V and cache
cursors. It separately compares the final dynamic context-cache update and
records the value/layout (shape, stride, memory format) of its x0, prompt, and
action inputs. Token intervals are only materialized after an exact K/V check
fails, so the tool does not retain multi-GB cache snapshots.

Example:

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/localize_abot_steady_state_cache_mismatch.py \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir results/validation/abot_b1_steady_cache_localization_gpu3
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


def _load_module(filename: str, module_name: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load diagnostic helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _intervals(indices: torch.Tensor) -> list[list[int]]:
    """Turn a compact CPU token-index tensor into half-open intervals."""
    values = [int(value) for value in indices.to(device="cpu").tolist()]
    if not values:
        return []
    result: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append([start, previous + 1])
            start = value
        previous = value
    result.append([start, previous + 1])
    return result


def _changed_token_intervals(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Strictly identify which KV token rows differ, without copying full KV."""
    if left.shape != right.shape or left.dtype != right.dtype or left.device != right.device:
        return {
            "comparable": False,
            "equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    exact = bool(torch.equal(left, right))
    if exact:
        return {
            "comparable": True,
            "equal": True,
            "changed_token_count": 0,
            "first_changed_token": None,
            "last_changed_token": None,
            "changed_token_intervals": [],
        }
    changed = torch.any(left != right, dim=(0, 2, 3))
    indices = torch.nonzero(changed, as_tuple=False).flatten()
    return {
        "comparable": True,
        "equal": False,
        "changed_token_count": int(indices.numel()),
        "first_changed_token": int(indices[0].item()) if indices.numel() else None,
        "last_changed_token": int(indices[-1].item()) if indices.numel() else None,
        "changed_token_intervals": _intervals(indices),
    }


def _exact_layer_cache_comparison(dynamic: Mapping[str, Any], steady: Mapping[str, Any]) -> dict[str, Any]:
    """Strict layer-cache equality plus ranges only when an exact check fails."""
    k = _changed_token_intervals(dynamic["k"], steady["k"])
    v = _changed_token_intervals(dynamic["v"], steady["v"])
    dynamic_global = int(dynamic["global_end_index"].item())
    steady_global = int(steady["global_end_index"].item())
    dynamic_local = int(dynamic["local_end_index"].item())
    steady_local = int(steady["local_end_index"].item())
    return {
        "exact": bool(k["equal"] and v["equal"] and dynamic_global == steady_global and dynamic_local == steady_local),
        "k": k,
        "v": v,
        "dynamic_global_end_index": dynamic_global,
        "steady_global_end_index": steady_global,
        "dynamic_local_end_index": dynamic_local,
        "steady_local_end_index": steady_local,
    }


def _tensor_layout(value: torch.Tensor) -> dict[str, Any]:
    """Capture the tensor facts that can change a CUDA kernel choice."""
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "is_contiguous": bool(value.is_contiguous()),
        "is_channels_last_3d": bool(value.is_contiguous(memory_format=torch.channels_last_3d)),
    }


def _tensor_pair(dynamic: torch.Tensor, steady: torch.Tensor) -> dict[str, Any]:
    return {
        "values_exact": bool(torch.equal(dynamic, steady)),
        "dynamic_layout": _tensor_layout(dynamic),
        "steady_layout": _tensor_layout(steady),
    }


def _fingerprint(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Produce collision-resistant, per-token K/V summaries for one layer."""

    def summarise(value: torch.Tensor) -> torch.Tensor:
        # Reductions operate on the full KV layer but only materialize a tiny
        # [tokens, 3] result. Avoid float()/square() full-cache temporaries:
        # two B=1 retained sessions are already substantial GPU residents.
        detached = value.detach()
        return torch.stack(
            (
                detached.sum(dim=(0, 2, 3), dtype=torch.float32),
                detached.amin(dim=(0, 2, 3)).float(),
                detached.amax(dim=(0, 2, 3)).float(),
            ),
            dim=1,
        ).cpu()

    return {
        "k": summarise(cache["k"]),
        "v": summarise(cache["v"]),
        "global_end_index": int(cache["global_end_index"].item()),
        "local_end_index": int(cache["local_end_index"].item()),
    }


def _fingerprint_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    def changed(value_name: str) -> dict[str, Any]:
        lhs = left[value_name]
        rhs = right[value_name]
        if lhs.shape != rhs.shape:
            return {"comparable": False, "changed_token_intervals": []}
        rows = torch.any(lhs != rhs, dim=1)
        indices = torch.nonzero(rows, as_tuple=False).flatten()
        return {
            "comparable": True,
            "changed_token_count": int(indices.numel()),
            "first_changed_token": int(indices[0].item()) if indices.numel() else None,
            "changed_token_intervals": _intervals(indices),
        }

    return {
        "k_fingerprint": changed("k"),
        "v_fingerprint": changed("v"),
        "global_end_index": {
            "dynamic": left["global_end_index"],
            "steady_state": right["global_end_index"],
            "equal": left["global_end_index"] == right["global_end_index"],
        },
        "local_end_index": {
            "dynamic": left["local_end_index"],
            "steady_state": right["local_end_index"],
            "equal": left["local_end_index"] == right["local_end_index"],
        },
    }


class _LayerZeroCallProbe:
    """Fingerprint layer-0 K/V after every dynamic or steady DiT invocation."""

    def __init__(self, dit: Any, layer: int) -> None:
        self._dit = dit
        self._layer = layer
        self._dynamic_original: Any = None
        self._steady_original: Any = None
        self.dynamic: list[dict[str, Any]] = []
        self.steady: list[dict[str, Any]] = []

    def __enter__(self) -> "_LayerZeroCallProbe":
        self._dynamic_original = self._dit.forward
        self._steady_original = self._dit.forward_steady_state

        def dynamic(*args: Any, **kwargs: Any) -> Any:
            output = self._dynamic_original(*args, **kwargs)
            cache = kwargs["kv_cache"][self._layer]
            self.dynamic.append(_fingerprint(cache))
            return output

        def steady(*args: Any, **kwargs: Any) -> Any:
            output = self._steady_original(*args, **kwargs)
            cache = kwargs["kv_cache"][self._layer]
            self.steady.append(_fingerprint(cache))
            return output

        self._dit.forward = dynamic
        self._dit.forward_steady_state = steady
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._dit.forward = self._dynamic_original
        self._dit.forward_steady_state = self._steady_original


def _call_reports(probe: _LayerZeroCallProbe) -> list[dict[str, Any]]:
    count = max(len(probe.dynamic), len(probe.steady))
    reports: list[dict[str, Any]] = []
    for index in range(count):
        dynamic = probe.dynamic[index] if index < len(probe.dynamic) else None
        steady = probe.steady[index] if index < len(probe.steady) else None
        reports.append(
            {
                "call_index": index,
                "phase": f"denoise_step_{index}" if index < 4 else "context_cache_update",
                "dynamic_seen": dynamic is not None,
                "steady_state_seen": steady is not None,
                "fingerprint_comparison": (
                    _fingerprint_difference(dynamic, steady) if dynamic is not None and steady is not None else None
                ),
            }
        )
    return reports


def _cache_layout(pipeline: Any, session: Any, frames: int) -> dict[str, int]:
    dit = pipeline.denoise_stage.dit
    height, width = session.first_frame_latent.shape[-2:]
    frame_tokens = (height // dit.patch_size[1]) * (width // dit.patch_size[2])
    capacity = int(session.self_cache[0]["k"].shape[1])
    sink_tokens = int(dit.sink_size) * frame_tokens
    return {
        "frame_tokens": frame_tokens,
        "continuation_tokens": frames * frame_tokens,
        "capacity_tokens": capacity,
        "sink_tokens": sink_tokens,
        "rolling_tail_start": sink_tokens,
        "new_block_tail_start": capacity - frames * frame_tokens,
    }


@torch.inference_mode()
def _run(args: argparse.Namespace, base: Any, control: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("cache localization requires CUDA")
    image = Image.open(args.image).convert("RGB")
    pipeline = None
    dynamic_session = None
    static_session = None
    try:
        pipeline = base._make_pipeline(args)
        device = torch.device(pipeline.device)
        if device.type != "cuda":
            raise RuntimeError(f"cache localization requires CUDA, got {pipeline.device!r}")
        pipeline.preload_models()
        pipeline.denoise_stage.configure_cuda_graph(False)
        actions = base._parse_action_keys(args.action_keys)
        dynamic_session = pipeline.create_interactive_session(
            image, args.prompt, seed=args.seed, session_id="cache-localization-dynamic"
        )
        static_session = pipeline.create_interactive_session(
            image, args.prompt, seed=args.seed, session_id="cache-localization-steady"
        )
        warmup_chunks = base._required_warmup_chunks(
            int(pipeline.denoise_stage.dit.local_attn_size),
            args.control_latent_frames,
            args.extra_warmup_chunks,
        )
        warmup_equal: list[bool] = []
        for _ in range(warmup_chunks):
            dynamic_frames = pipeline.generate_next_block(
                dynamic_session, actions, control_latent_frames=args.control_latent_frames
            )
            static_frames = pipeline.generate_next_block(
                static_session, actions, control_latent_frames=args.control_latent_frames
            )
            warmup_equal.append(base._sequence_hash(dynamic_frames) == base._sequence_hash(static_frames))
        torch.cuda.synchronize(device)
        warmup_state_exact = control._tree_exactness(
            control._session_state_tree(dynamic_session, pipeline),
            control._session_state_tree(static_session, pipeline),
        )
        dynamic_latent, dynamic_prompt, dynamic_action = control._prepare_inputs(
            pipeline, [dynamic_session], [actions], args.control_latent_frames
        )
        static_latent, static_prompt, static_action = control._prepare_inputs(
            pipeline, [static_session], [actions], args.control_latent_frames
        )
        input_exact = control._compare_tensor(dynamic_latent, static_latent)
        stage = pipeline.denoise_stage
        static_state = control._static_state(pipeline, [static_session], static_latent, static_prompt, static_action)
        graph = static_state["graph"]
        current_start = dynamic_session.next_latent_frame
        if current_start != static_session.next_latent_frame:
            raise RuntimeError("dynamic and steady sessions must share a continuation position")
        timesteps = stage._official_denoising_timesteps(dynamic_session.scheduler).to(device=device)
        dynamic_current = dynamic_latent
        steady_current = static_latent
        step_reports: list[dict[str, Any]] = []
        for index, current_timestep in enumerate(timesteps):
            dynamic_timestep = torch.full(
                (1, args.control_latent_frames),
                current_timestep,
                dtype=timesteps.dtype,
                device=device,
            )
            with torch.autocast(device.type, dtype=pipeline.torch_dtype, enabled=device.type == "cuda"):
                dynamic_prediction = stage.dit(
                    x=dynamic_current.to(dtype=pipeline.torch_dtype),
                    timestep=dynamic_timestep,
                    context=dynamic_prompt,
                    act_context=dynamic_action,
                    kv_cache=dynamic_session.self_cache,
                    crossattn_cache=dynamic_session.cross_cache,
                    current_start=current_start * graph.frame_tokens,
                )
            graph._set_inputs(
                steady_current,
                static_action,
                current_timestep,
                current_end=(current_start + args.control_latent_frames) * graph.frame_tokens,
            )
            with torch.autocast(device.type, dtype=pipeline.torch_dtype, enabled=device.type == "cuda"):
                steady_prediction = stage.dit.forward_steady_state(
                    x=graph.static_x,
                    timestep=graph.static_timestep,
                    context=graph.static_context,
                    act_context=graph.static_action,
                    kv_cache=static_state["self_cache"],
                    crossattn_cache=static_state["cross_cache"],
                    current_end=graph.current_end,
                    roll_scratch_k=graph.roll_scratch_k,
                    roll_scratch_v=graph.roll_scratch_v,
                    update_cache=index == 0,
                )
            torch.cuda.synchronize(device)
            layer_cache = _exact_layer_cache_comparison(
                dynamic_session.self_cache[args.layer], static_session.self_cache[args.layer]
            )
            dynamic_x0 = stage._x0_prediction(
                dynamic_prediction, dynamic_current, dynamic_timestep, dynamic_session.scheduler
            )
            steady_x0 = stage._x0_prediction(
                steady_prediction, steady_current, graph.static_timestep, static_session.scheduler
            )
            step_reports.append(
                {
                    "call_index": index,
                    "phase": f"denoise_step_{index}",
                    "flow_prediction": _tensor_pair(dynamic_prediction, steady_prediction),
                    "x0": _tensor_pair(dynamic_x0, steady_x0),
                    "layer0_cache": layer_cache,
                }
            )
            if index < len(timesteps) - 1:
                dynamic_noise = torch.randn(
                    dynamic_x0.shape,
                    generator=dynamic_session.generator,
                    dtype=dynamic_x0.dtype,
                    device=device,
                )
                steady_noise = torch.randn(
                    steady_x0.shape,
                    generator=static_session.generator,
                    dtype=steady_x0.dtype,
                    device=device,
                )
                dynamic_current = dynamic_session.scheduler.add_noise(dynamic_x0, dynamic_noise, timesteps[index + 1])
                steady_current = static_session.scheduler.add_noise(steady_x0, steady_noise, timesteps[index + 1])
            else:
                dynamic_current = dynamic_x0
                steady_current = steady_x0.clone()

        dynamic_context_timestep = torch.zeros_like(dynamic_timestep)
        graph.static_timestep.zero_()
        dynamic_context_input = dynamic_current.to(dtype=pipeline.torch_dtype)
        steady_context_input = steady_current.to(dtype=pipeline.torch_dtype)
        final_dynamic_context_inputs = {
            "x0": _tensor_pair(dynamic_context_input, steady_context_input),
            "timestep": _tensor_pair(dynamic_context_timestep, graph.static_timestep),
            "prompt": _tensor_pair(dynamic_prompt, graph.static_context),
            "action": _tensor_pair(dynamic_action, static_action),
        }
        # The public _denoise_block commits final x0 outside its sampler
        # autocast scope; mirror that precision boundary on both sides.
        stage.dit(
            x=dynamic_context_input,
            timestep=dynamic_context_timestep,
            context=dynamic_prompt,
            act_context=dynamic_action,
            kv_cache=dynamic_session.self_cache,
            crossattn_cache=dynamic_session.cross_cache,
            current_start=current_start * graph.frame_tokens,
        )
        stage.dit(
            x=steady_context_input,
            timestep=graph.static_timestep,
            context=graph.static_context,
            act_context=static_action,
            kv_cache=static_state["self_cache"],
            crossattn_cache=static_state["cross_cache"],
            current_start=current_start * graph.frame_tokens,
        )
        torch.cuda.synchronize(device)
        context_cache = _exact_layer_cache_comparison(
            dynamic_session.self_cache[args.layer], static_session.self_cache[args.layer]
        )
        output = _tensor_pair(dynamic_current, steady_current)
        first_exact_cache_divergence = next(
            (item["call_index"] for item in step_reports if not item["layer0_cache"]["exact"]),
            4 if not context_cache["exact"] else None,
        )
        all_step_caches_exact = all(item["layer0_cache"]["exact"] for item in step_reports)
        status = (
            "pass"
            if output["values_exact"] and all_step_caches_exact and context_cache["exact"]
            else "cache_mismatch_localized"
        )
        return {
            "status": status,
            "scope": {
                "public_denoise_block_exercised": False,
                "dynamic_control": "handwritten per-DiT-call loop",
                "steady_control": "handwritten forward_steady_state loop",
                "execution_order": "dynamic_then_steady_per_denoise_step",
                "cache_coverage": f"layer {args.layer} only",
                "note": (
                    "This localizer isolates individual DiT/cache writes; it is not a parity proof for "
                    "generate_next_block -> denoise_interactive_block -> _denoise_block. "
                    "Use validate_abot_public_vs_steady_state.py for that public-path check."
                ),
            },
            "device": str(device),
            "layer": args.layer,
            "control_latent_frames": args.control_latent_frames,
            "warmup": {
                "chunks": warmup_chunks,
                "per_chunk_frames_equal": warmup_equal,
                "state_exact": warmup_state_exact,
            },
            "continuation_input_exact": input_exact,
            "continuation_output_exact": output,
            "cache_layout": _cache_layout(pipeline, dynamic_session, args.control_latent_frames),
            "per_denoise_step_layer0_exact": step_reports,
            "all_denoise_step_caches_exact": all_step_caches_exact,
            "first_exact_cache_divergence_call": first_exact_cache_divergence,
            "final_dynamic_context_inputs": final_dynamic_context_inputs,
            "final_layer_cache_exact": context_cache,
        }
    finally:
        if pipeline is not None:
            for session in (dynamic_session, static_session):
                if session is not None:
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


def _write_output(output_dir: Path, result: Mapping[str, Any], args: argparse.Namespace, base: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps({"arguments": base._json_safe(vars(args)), "result": base._json_safe(result)}, indent=2) + "\n",
        encoding="utf-8",
    )
    final = result.get("final_layer_cache_exact", {})
    lines = [
        "# ABot steady-state cache mismatch localization",
        "",
        f"Status: {result.get('status', 'error')}.",
        "",
        "Scope: handwritten dynamic and steady-state DiT loops, interleaved per denoising step; "
        "this is not the public `_denoise_block` parity check.",
        "",
        f"First exact layer-0 cache divergence call: {result.get('first_exact_cache_divergence_call')}.",
        "",
        f"Final K changed intervals: {final.get('k', {}).get('changed_token_intervals', [])}.",
        f"Final V changed intervals: {final.get('v', {}).get('changed_token_intervals', [])}.",
        "",
        "Call indices 0..3 are denoising calls; index 4 is the context-cache update.",
        "results.json contains exact per-step K/V comparisons, context-input layouts, and cursor values.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args(base: Any) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prompt", default="A smooth first-person exploration through a vivid natural landscape.")
    parser.add_argument("--action-keys", default="W")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-latent-frames", type=int, choices=(3,), default=3)
    parser.add_argument("--extra-warmup-chunks", type=int, default=0)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.extra_warmup_chunks < 0:
        parser.error("--extra-warmup-chunks must be non-negative")
    try:
        base._parse_action_keys(args.action_keys)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.layer < 0:
        parser.error("--layer must be non-negative")
    if not args.dry_run:
        if args.model_root is None or args.image is None or args.output_dir is None:
            parser.error("--model-root, --image, and --output-dir are required unless --dry-run is used")
        if not args.model_root.is_dir():
            parser.error(f"model root does not exist: {args.model_root}")
        if not args.image.is_file():
            parser.error(f"image does not exist: {args.image}")
    return args


def main() -> None:
    base = _load_module("validate_abot_cuda_graph_parity.py", "abot_cache_localization_base")
    control = _load_module("diagnose_abot_cuda_graph_persistent_three_way.py", "abot_cache_localization_control")
    args = _parse_args(base)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "batch_size": 1,
                    "layer": args.layer,
                    "calls": ["denoise_step_0", "denoise_step_1", "denoise_step_2", "denoise_step_3", "context"],
                    "output": ["first exact cache divergence", "per-step K/V token intervals", "context-input layouts"],
                },
                indent=2,
            )
        )
        return
    assert args.output_dir is not None
    try:
        result = _run(args, base, control)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_output(args.output_dir, result, args, base)
    print(json.dumps(base._json_safe(result), indent=2))
    if result.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
