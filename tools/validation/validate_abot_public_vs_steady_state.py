"""Compare ABot's real public B=1 continuation with handwritten steady state.

This is a narrow diagnostic for reconciling two different validation scopes:

* the public path, ``generate_next_block -> denoise_interactive_block ->
  _denoise_block`` with CUDA Graph explicitly disabled; and
* the graph-shaped eager control, which invokes ``forward_steady_state`` with
  the persistent static tensors/scratch buffers but never captures a graph.

Two equal-seed sessions are warmed through public eager generation until their
KV windows are full.  The tool then hooks the *actual* public ``_denoise_block``
call for one continuation, records its exact input contract, and runs the
steady-state control for the twin.  It checks the public call inputs, sampled
latent, self/cross cache plus RNG state, rendered frames, and complete retained
session state.

``--static-state-timing after_public`` mirrors the three-way diagnostic's
allocator/order.  ``before_public`` creates the steady static buffers before
the public continuation and is useful when contrasting that outcome with the
per-step-interleaved localizer.  Neither mode is a CUDA-Graph test.

Example (physical GPU 3 mapped to logical CUDA device 0)::

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD \\
    /public/fanyk1/lwb/envs/telefuser_sage291/bin/python \\
    tools/validation/validate_abot_public_vs_steady_state.py \\
      --model-root /public/fanyk1/lwb/model_zoo/ABot-World-0-5B-LF \\
      --image /public/fanyk1/lwb/ABot-World/web_client/datasets/images/84b90ad568b693d2.png \\
      --output-dir /public/fanyk1/lwb/results/validation/abot_public_vs_steady_gpu3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections.abc import Mapping
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


def _tensor_layout(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "is_contiguous": bool(value.is_contiguous()),
        "is_channels_last_3d": bool(value.is_contiguous(memory_format=torch.channels_last_3d)),
    }


def _tensor_contract(left: torch.Tensor | None, right: torch.Tensor | None) -> dict[str, Any]:
    """Compare values and layout, deliberately not allocator-specific pointers."""
    if left is None or right is None:
        return {
            "comparable": False,
            "values_exact": False,
            "left_present": left is not None,
            "right_present": right is not None,
        }
    comparable = left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
    return {
        "comparable": comparable,
        "values_exact": bool(torch.equal(left, right)) if comparable else False,
        "left_layout": _tensor_layout(left),
        "right_layout": _tensor_layout(right),
    }


def _sampling_state_tree(session: Any) -> dict[str, Any]:
    """The state that must agree before VAE decoding and counter updates."""
    return {
        "prompt_emb": session.prompt_emb,
        "first_frame_latent": session.first_frame_latent,
        "self_cache": session.self_cache,
        "cross_cache": session.cross_cache,
        "generator_state": session.generator.get_state(),
    }


def _cache_cursor_contract(caches: list[dict[str, Any]]) -> list[dict[str, int]]:
    return [
        {
            "global_end_index": int(layer["global_end_index"].item()),
            "local_end_index": int(layer["local_end_index"].item()),
        }
        for layer in caches
    ]


class _PublicBlockProbe:
    """Capture the real singleton public ``_denoise_block`` contract once."""

    def __init__(self, stage: Any) -> None:
        self._stage = stage
        self._original: Any = None
        self.call_count = 0
        self.inputs: dict[str, Any] | None = None
        self.output: torch.Tensor | None = None

    def __enter__(self) -> "_PublicBlockProbe":
        self._original = self._stage._denoise_block

        def wrapped(
            latent: torch.Tensor,
            prompt_emb: torch.Tensor,
            action_context: torch.Tensor,
            first_frame_latent: torch.Tensor | None,
            self_cache: list[dict[str, Any]],
            cross_cache: list[dict[str, Any]],
            current_start: int,
            generator: torch.Generator,
            scheduler: Any,
        ) -> torch.Tensor:
            self.call_count += 1
            if self.call_count != 1:
                raise RuntimeError("expected exactly one public _denoise_block call for one continuation")
            self.inputs = {
                "latent": latent,
                "prompt_emb": prompt_emb,
                "action_context": action_context,
                "first_frame_is_none": first_frame_latent is None,
                "current_start": int(current_start),
                "generator_state_after_input_draw": generator.get_state().clone(),
                "self_cache_cursors_before": _cache_cursor_contract(self_cache),
                "cross_cache_initialized_before": [bool(layer["is_init"]) for layer in cross_cache],
                "scheduler_type": type(scheduler).__name__,
            }
            output = self._original(
                latent,
                prompt_emb,
                action_context,
                first_frame_latent,
                self_cache,
                cross_cache,
                current_start,
                generator,
                scheduler,
            )
            self.output = output.detach().clone()
            return output

        self._stage._denoise_block = wrapped
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stage._denoise_block = self._original


def _prepare_steady_control(
    pipeline: Any,
    control: Any,
    session: Any,
    actions: Mapping[str, bool],
    frames: int,
) -> dict[str, Any]:
    latent, prompt_emb, action_context = control._prepare_inputs(pipeline, [session], [actions], frames)
    return {
        "latent": latent,
        "prompt_emb": prompt_emb,
        "action_context": action_context,
        "generator_state_after_input_draw": session.generator.get_state().clone(),
        "current_start": int(session.next_latent_frame),
        "state": control._static_state(pipeline, [session], latent, prompt_emb, action_context),
    }


def _finish_steady_lifecycle(pipeline: Any, session: Any, latents: torch.Tensor, frames: int) -> list[Image.Image]:
    if session.taew_decode_state is None:
        raise RuntimeError("ABot session is missing its TAeW decode state")
    decoded = pipeline.taew_decode_stage.decode_chunks(latents, [session.taew_decode_state])
    output = pipeline.tensor2video(decoded[0])
    session.next_latent_frame += frames
    session.emitted_frames += len(output)
    return output


def _contracts_all_exact(contracts: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        bool(contract.get("comparable")) and bool(contract.get("values_exact")) for contract in contracts.values()
    )


@torch.inference_mode()
def _run(args: argparse.Namespace, base: Any, control: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("public-vs-steady diagnostic requires CUDA")
    image = Image.open(args.image).convert("RGB")
    pipeline = None
    public_session = None
    steady_session = None
    try:
        pipeline = base._make_pipeline(args)
        device = torch.device(pipeline.device)
        if device.type != "cuda":
            raise RuntimeError(f"public-vs-steady diagnostic requires CUDA, got {pipeline.device!r}")
        pipeline.preload_models()
        stage = pipeline.denoise_stage
        stage.configure_cuda_graph(False)
        actions = base._parse_action_keys(args.action_keys)
        public_session = pipeline.create_interactive_session(
            image, args.prompt, seed=args.seed, session_id="public-vs-steady-public"
        )
        steady_session = pipeline.create_interactive_session(
            image, args.prompt, seed=args.seed, session_id="public-vs-steady-steady"
        )
        warmup_chunks = base._required_warmup_chunks(
            int(stage.dit.local_attn_size), args.control_latent_frames, args.extra_warmup_chunks
        )
        warmup_hashes: list[bool] = []
        for _ in range(warmup_chunks):
            public_frames = pipeline.generate_next_block(
                public_session, actions, control_latent_frames=args.control_latent_frames
            )
            steady_frames = pipeline.generate_next_block(
                steady_session, actions, control_latent_frames=args.control_latent_frames
            )
            warmup_hashes.append(base._sequence_hash(public_frames) == base._sequence_hash(steady_frames))
        torch.cuda.synchronize(device)
        warmup_state = control._tree_exactness(
            control._session_state_tree(public_session, pipeline),
            control._session_state_tree(steady_session, pipeline),
        )
        warmup_ready = {
            "public": base._cache_readiness(public_session, pipeline),
            "steady": base._cache_readiness(steady_session, pipeline),
        }
        if public_session.next_latent_frame != steady_session.next_latent_frame:
            raise RuntimeError("same-seed sessions did not reach the same continuation position")

        steady_control: dict[str, Any] | None = None
        if args.static_state_timing == "before_public":
            steady_control = _prepare_steady_control(
                pipeline, control, steady_session, actions, args.control_latent_frames
            )
        with _PublicBlockProbe(stage) as probe:
            public_frames = pipeline.generate_next_block(
                public_session, actions, control_latent_frames=args.control_latent_frames
            )
        if probe.inputs is None or probe.output is None:
            raise RuntimeError("public continuation did not enter _denoise_block")
        if args.static_state_timing == "after_public":
            steady_control = _prepare_steady_control(
                pipeline, control, steady_session, actions, args.control_latent_frames
            )
        assert steady_control is not None

        public_inputs = probe.inputs
        input_contracts = {
            "latent": _tensor_contract(public_inputs["latent"], steady_control["latent"]),
            "prompt_emb": _tensor_contract(public_inputs["prompt_emb"], steady_control["prompt_emb"]),
            "action_context": _tensor_contract(public_inputs["action_context"], steady_control["action_context"]),
            "generator_state_after_input_draw": _tensor_contract(
                public_inputs["generator_state_after_input_draw"],
                steady_control["generator_state_after_input_draw"],
            ),
        }
        static_start = int(steady_control["current_start"])
        if static_start != int(public_inputs["current_start"]):
            raise RuntimeError("public and steady controls disagree on current_start")
        steady_latent = control._static_denoise(
            pipeline,
            steady_control["state"],
            steady_control["latent"],
            steady_control["action_context"],
            current_start=static_start,
            generators=[steady_session.generator],
            scheduler=steady_session.scheduler,
        )
        torch.cuda.synchronize(device)
        sampling_state = control._tree_exactness(
            _sampling_state_tree(public_session), _sampling_state_tree(steady_session)
        )
        latent_contract = _tensor_contract(probe.output, steady_latent)
        steady_frames = _finish_steady_lifecycle(pipeline, steady_session, steady_latent, args.control_latent_frames)
        torch.cuda.synchronize(device)
        rgb = base._compare_frames(public_frames, steady_frames)
        full_state = control._tree_exactness(
            control._session_state_tree(public_session, pipeline),
            control._session_state_tree(steady_session, pipeline),
        )
        public_metrics = dict(pipeline.last_stage_metrics())
        public_path_valid = (
            probe.call_count == 1
            and bool(public_inputs["first_frame_is_none"])
            and int(public_metrics.get("cuda_graph_enabled", 0)) == 0
            and int(public_metrics.get("cuda_graph_replays", 0)) == 0
            and int(public_metrics.get("cuda_graph_captured", 0)) == 0
        )
        exact = (
            all(warmup_hashes)
            and bool(warmup_state["exact"])
            and bool(warmup_ready["public"]["ready"])
            and bool(warmup_ready["steady"]["ready"])
            and public_path_valid
            and _contracts_all_exact(input_contracts)
            and bool(latent_contract["values_exact"])
            and bool(sampling_state["exact"])
            and bool(rgb["all_frame_hashes_equal"])
            and bool(full_state["exact"])
        )
        return {
            "status": "pass" if exact else "mismatch",
            "scope": {
                "public_path": "generate_next_block -> denoise_interactive_block -> _denoise_block",
                "steady_control": "handwritten forward_steady_state with graph-shaped static buffers",
                "continuations": 1,
                "cuda_graph_enabled": False,
                "static_state_timing": args.static_state_timing,
            },
            "device": str(device),
            "warmup": {
                "chunks": warmup_chunks,
                "per_chunk_frame_hash_equal": warmup_hashes,
                "public_ready": warmup_ready["public"],
                "steady_ready": warmup_ready["steady"],
                "complete_state_exact": warmup_state,
            },
            "public_block": {
                "call_count": probe.call_count,
                "first_frame_is_none": public_inputs["first_frame_is_none"],
                "current_start": public_inputs["current_start"],
                "self_cache_cursors_before": public_inputs["self_cache_cursors_before"],
                "cross_cache_initialized_before": public_inputs["cross_cache_initialized_before"],
                "scheduler_type": public_inputs["scheduler_type"],
                "stage_metrics": public_metrics,
                "valid_eager_public_path": public_path_valid,
            },
            "input_contracts": input_contracts,
            "post_denoise": {
                "latent": latent_contract,
                "sampling_state_exact": sampling_state,
            },
            "post_lifecycle": {
                "rgb": rgb,
                "complete_state_exact": full_state,
            },
        }
    finally:
        if pipeline is not None:
            for session in (public_session, steady_session):
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
    payload = {"arguments": base._json_safe(vars(args)), "result": base._json_safe(result)}
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    post_denoise = result.get("post_denoise", {})
    post_lifecycle = result.get("post_lifecycle", {})
    lines = [
        "# ABot public `_denoise_block` vs steady-state control",
        "",
        f"Status: {result.get('status', 'error')}.",
        "",
        "The public side was observed through `generate_next_block -> denoise_interactive_block -> "
        "_denoise_block` with CUDA Graph disabled.",
        "",
        f"Post-denoise latent exact: {post_denoise.get('latent', {}).get('values_exact', False)}.",
        f"Post-denoise sampling state exact: {post_denoise.get('sampling_state_exact', {}).get('exact', False)}.",
        f"Rendered RGB exact: {post_lifecycle.get('rgb', {}).get('all_frame_hashes_equal', False)}.",
        f"Full lifecycle state exact: {post_lifecycle.get('complete_state_exact', {}).get('exact', False)}.",
        "",
        "results.json records actual public `_denoise_block` input values/layouts, generator position, "
        "cache cursors, and strict retained-state comparisons.",
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
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--static-state-timing", choices=("after_public", "before_public"), default="after_public")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.extra_warmup_chunks < 0:
        parser.error("--extra-warmup-chunks must be non-negative")
    try:
        base._parse_action_keys(args.action_keys)
    except argparse.ArgumentTypeError as exc:
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
    base = _load_module("validate_abot_cuda_graph_parity.py", "abot_public_vs_steady_base")
    control = _load_module("diagnose_abot_cuda_graph_persistent_three_way.py", "abot_public_vs_steady_control")
    args = _parse_args(base)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "public_path": "generate_next_block -> denoise_interactive_block -> _denoise_block",
                    "steady_control": "forward_steady_state with graph-shaped static buffers",
                    "cuda_graph": "disabled",
                    "continuations": 1,
                    "comparisons": [
                        "warmup full retained state",
                        "actual public input values/layouts and generator position",
                        "post-denoise latent and sampling state",
                        "rendered RGB and complete lifecycle state",
                    ],
                    "static_state_timing": args.static_state_timing,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    assert args.output_dir is not None
    try:
        result = _run(args, base, control)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    _write_output(args.output_dir, result, args, base)
    print(json.dumps(base._json_safe(result), indent=2, sort_keys=True))
    if result.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
