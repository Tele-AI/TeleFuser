"""Capture reproducible LingBot-Video upstream reference artifacts.

This tool deliberately executes the checked-out upstream Diffusers implementation.
It does not copy or modify upstream source files. The captured artifacts are intended
to remain under ``work_dirs/`` and provide the numerical oracle for later phases.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_model

DEFAULT_UPSTREAM_ROOT = Path("work_dirs/lingbot-video-master")
DEFAULT_CASE_MANIFEST = "assets/cases/manifest.json"
DEFAULT_DENSE_MODEL_DIR = Path("/hhb-data/aigc/model_zoo/lingbot/lingbot-video-dense-1.3b")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: Path, *, include_contents: bool) -> str:
    """Return a stable fingerprint without loading checkpoint tensors into memory."""
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        stat = path.stat()
        digest.update(f"{relative}\0{stat.st_size}\0".encode())
        if include_contents:
            digest.update(_sha256_file(path).encode())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
    finite = torch.isfinite(cpu)
    summary: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype).removeprefix("torch."),
        "numel": cpu.numel(),
        "nan_count": int(torch.isnan(cpu).sum().item()) if cpu.is_floating_point() else 0,
        "inf_count": int(torch.isinf(cpu).sum().item()) if cpu.is_floating_point() else 0,
        "sha256": hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }
    if cpu.is_floating_point() and finite.any():
        values = cpu[finite].float()
        summary.update(
            min=float(values.min().item()),
            max=float(values.max().item()),
            mean=float(values.mean().item()),
            std=float(values.std(unbiased=False).item()),
        )
    return summary


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    sample = getattr(value, "sample", None)
    return sample if isinstance(sample, torch.Tensor) else None


@contextlib.contextmanager
def _upstream_import_path(root: Path) -> Iterator[None]:
    """Temporarily expose the upstream checkout without editing it."""
    root = root.resolve()
    if not (root / "lingbot_video" / "__init__.py").is_file():
        raise FileNotFoundError(f"LingBot-Video package not found under {root}")
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))


def _import_upstream(root: Path) -> ModuleType:
    with _upstream_import_path(root):
        return importlib.import_module("lingbot_video.runner")


def _sample_cases(upstream_root: Path, modes: list[str], case_name: str | None) -> list[dict[str, Any]]:
    manifest_path = upstream_root / DEFAULT_CASE_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    for mode in modes:
        matches = [
            item
            for item in manifest["examples"]
            if item["mode"] == mode and (case_name is None or item["name"] == case_name)
        ]
        if not matches:
            requested = "all cases" if case_name is None else f"case {case_name!r}"
            raise ValueError(f"No {mode} {requested} in {manifest_path}")
        selected.extend(matches)
    return selected


def _selected_step_indices(step_count: int, trace: str) -> set[int]:
    if trace == "full":
        return set(range(step_count))
    if trace == "sampled" and step_count:
        return {0, step_count // 2, step_count - 1}
    return set()


@dataclass
class ArtifactRecorder:
    root: Path
    trace: str
    tensor_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0

    def __post_init__(self) -> None:
        (self.root / "tensors").mkdir(parents=True, exist_ok=True)

    @property
    def selected_steps(self) -> set[int]:
        return _selected_step_indices(self.step_count, self.trace)

    def record_tensor(self, name: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        safe_name = name.replace("/", "_").replace(" ", "_")
        path = self.root / "tensors" / f"{safe_name}.pt"
        cpu = tensor.detach().cpu().clone(memory_format=torch.contiguous_format)
        torch.save(cpu, path)
        metadata = _tensor_summary(cpu)
        metadata["path"] = str(path.relative_to(self.root))
        self.tensor_metadata[safe_name] = metadata

    def event(self, name: str, **data: Any) -> None:
        self.events.append({"name": name, **data})


class PipelineTrace:
    """Install reversible hooks around stable upstream pipeline boundaries."""

    def __init__(self, pipe: Any, recorder: ArtifactRecorder):
        self.pipe = pipe
        self.recorder = recorder
        self._restore: list[Callable[[], None]] = []
        self._prompt_call = 0
        self._transformer_call = 0
        self._step = 0

    def _replace_method(self, obj: Any, name: str, wrapper: Callable[..., Any]) -> None:
        original = getattr(obj, name)
        setattr(obj, name, wrapper(original))
        self._restore.append(lambda: setattr(obj, name, original))

    def __enter__(self) -> "PipelineTrace":
        self._replace_method(self.pipe, "_build_prompt_inputs", self._wrap_prompt_inputs)
        self._replace_method(self.pipe, "encode_prompt", self._wrap_encode_prompt)
        self._replace_method(self.pipe, "prepare_latents", self._wrap_prepare_latents)
        self._replace_method(self.pipe.scheduler, "set_timesteps", self._wrap_set_timesteps)
        self._replace_method(self.pipe.scheduler, "step", self._wrap_scheduler_step)
        self._replace_method(self.pipe.transformer, "forward", self._wrap_transformer_forward)
        if hasattr(self.pipe, "encode_image_latent"):
            self._replace_method(self.pipe, "encode_image_latent", self._wrap_encode_image_latent)
        if hasattr(self.pipe, "preprocess_image"):
            self._replace_method(self.pipe, "preprocess_image", self._wrap_preprocess_image)
        if getattr(self.pipe, "vae", None) is not None:
            self._replace_method(self.pipe.vae, "decode", self._wrap_vae_decode)
        return self

    def __exit__(self, *_: Any) -> None:
        for restore in reversed(self._restore):
            restore()

    def _wrap_prompt_inputs(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            inputs = original(*args, **kwargs)
            for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
                value = inputs.get(name) if hasattr(inputs, "get") else None
                if value is not None:
                    self.recorder.record_tensor(f"prompt_inputs_{self._prompt_call}_{name}", value)
            return inputs

        return wrapped

    def _wrap_encode_prompt(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            call = self._prompt_call
            self._prompt_call += 1
            output = original(*args, **kwargs)
            if isinstance(output, tuple):
                self.recorder.record_tensor(f"prompt_{call}_embeds", output[0])
                self.recorder.record_tensor(f"prompt_{call}_mask", output[1])
            self.recorder.event("encode_prompt", call=call)
            return output

        return wrapped

    def _wrap_prepare_latents(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            self.recorder.record_tensor("initial_latents", output)
            return output

        return wrapped

    def _wrap_set_timesteps(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            timesteps = getattr(self.pipe.scheduler, "timesteps", None)
            sigmas = getattr(self.pipe.scheduler, "sigmas", None)
            if isinstance(timesteps, torch.Tensor):
                self.recorder.step_count = int(timesteps.numel())
                self.recorder.record_tensor("scheduler_timesteps", timesteps)
            if isinstance(sigmas, torch.Tensor):
                self.recorder.record_tensor("scheduler_sigmas", sigmas)
            return output

        return wrapped

    def _wrap_transformer_forward(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            call = self._transformer_call
            self._transformer_call += 1
            if self._step in self.recorder.selected_steps:
                self.recorder.record_tensor(f"step_{self._step}_transformer_{call}_latent", args[0])
                self.recorder.record_tensor(f"step_{self._step}_transformer_{call}_timestep", args[1])
                self.recorder.record_tensor(f"step_{self._step}_transformer_{call}_prompt", args[2])
            output = original(*args, **kwargs)
            if self._step in self.recorder.selected_steps:
                self.recorder.record_tensor(f"step_{self._step}_transformer_{call}_noise", output)
            return output

        return wrapped

    def _wrap_scheduler_step(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(
            model_output: torch.Tensor, timestep: torch.Tensor, sample: torch.Tensor, *args: Any, **kwargs: Any
        ) -> Any:
            selected = self._step in self.recorder.selected_steps
            if selected:
                self.recorder.record_tensor(f"step_{self._step}_noise_prediction", model_output)
                self.recorder.record_tensor(f"step_{self._step}_latent_before", sample)
                self.recorder.record_tensor(f"step_{self._step}_timestep", timestep)
            output = original(model_output, timestep, sample, *args, **kwargs)
            if selected:
                self.recorder.record_tensor(f"step_{self._step}_latent_after", output)
            self._step += 1
            return output

        return wrapped

    def _wrap_encode_image_latent(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            self.recorder.record_tensor("ti2v_condition_latent", output)
            return output

        return wrapped

    def _wrap_preprocess_image(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            output = original(*args, **kwargs)
            self.recorder.record_tensor("ti2v_preprocessed_image", output)
            return output

        return wrapped

    def _wrap_vae_decode(self, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.recorder.record_tensor("vae_decode_input", args[0])
            output = original(*args, **kwargs)
            self.recorder.record_tensor("vae_decode_output", output)
            return output

        return wrapped


def _generator_state(generator: torch.Generator) -> dict[str, Any]:
    return {
        "device": str(generator.device),
        "state_sha256": hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
    }


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version(), "torch": torch.__version__}
    for name in ("diffusers", "transformers"):
        try:
            module = importlib.import_module(name)
            result[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            result[name] = "not-installed"
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_value) + "\n", encoding="utf-8")


def _load_reference_pipeline(
    runner: ModuleType, model_dir: Path, dtype_map: dict[str, torch.dtype], mode: str, transformer_subfolder: str
) -> Any:
    """Load upstream components with a safetensors fallback when accelerate is absent."""
    try:
        return runner._load_diffusers_pipe(model_dir, dtype_map, mode=mode, transformer_subfolder=transformer_subfolder)
    except ValueError as exc:
        if "low_cpu_mem_usage" not in str(exc) or "keep_in_fp32_modules" not in str(exc):
            raise
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

    config = json.loads((model_dir / transformer_subfolder / "config.json").read_text(encoding="utf-8"))
    fields = (
        "patch_size",
        "in_channels",
        "out_channels",
        "hidden_size",
        "num_attention_heads",
        "depth",
        "intermediate_size",
        "text_dim",
        "freq_dim",
        "norm_eps",
        "rope_theta",
        "axes_dims",
        "qkv_bias",
        "out_bias",
        "patch_embed_bias",
        "timestep_mlp_bias",
    )
    device = runner._default_device()
    transformer = LingBotVideoTransformer3DModel(**{field: config[field] for field in fields}).to(
        device, dtype_map.get("transformer", dtype_map["default"])
    )
    load_model(
        transformer,
        model_dir / transformer_subfolder / "diffusion_pytorch_model.safetensors",
        strict=True,
        device=str(device),
    )
    pipeline_class = runner._pipeline_class_for_mode(mode)
    with runner._patch_qwen3vl_from_pretrained():
        pipeline = pipeline_class.from_pretrained(
            str(model_dir), transformer=transformer, trust_remote_code=True, torch_dtype=dtype_map
        )
    return pipeline.to(device)


def _capture_once(
    *,
    runner: ModuleType,
    upstream_root: Path,
    model_dir: Path,
    case: dict[str, Any],
    destination: Path,
    args: argparse.Namespace,
    repeat: int,
) -> dict[str, Any]:
    prompt_path = upstream_root / case["prompt_json"]
    sample = runner._load_prompt_sample(prompt_path)
    prompt = runner._caption_from_sample(sample)
    mode = case["mode"]
    num_frames = 1 if mode == "t2i" else args.num_frames
    image_path = upstream_root / case["image"] if "image" in case else None
    dtype_map = {
        "default": runner._parse_dtype(args.default_dtype),
        "transformer": runner._parse_dtype(args.transformer_dtype),
        "text_encoder": runner._parse_dtype(args.text_encoder_dtype),
        "vae": runner._parse_dtype(args.vae_dtype),
    }
    pipe = _load_reference_pipeline(
        runner, model_dir.resolve(), dtype_map, mode=mode, transformer_subfolder=args.transformer_subfolder
    )
    device = runner._default_device()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    recorder = ArtifactRecorder(destination, args.trace)
    rng_before = _generator_state(generator)
    started = time.perf_counter()
    call_args: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": runner.DEFAULT_NEGATIVE_PROMPT_IMAGE if mode == "t2i" else runner.DEFAULT_NEGATIVE_PROMPT,
        "height": args.height,
        "width": args.width,
        "num_frames": num_frames,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "generator": generator,
        "output_type": "np",
        "batch_cfg": False,
        "null_cond_clone_zero": False,
    }
    if image_path is not None:
        call_args["image"] = Image.open(image_path).convert("RGB")
    with PipelineTrace(pipe, recorder):
        result = pipe(**call_args)
    frames = result.frames if hasattr(result, "frames") else result[0]
    frames_array = np.asarray(frames[0] if isinstance(frames, list) else frames)
    np.save(destination / "frames.npy", frames_array)
    metadata = {
        "capture_schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "kind": "lingbot_video_numerical_oracle",
        "mode": mode,
        "case": case["name"],
        "repeat": repeat,
        "upstream_root": str(upstream_root.resolve()),
        "upstream_source_fingerprint": _tree_fingerprint(upstream_root, include_contents=True),
        "checkpoint_manifest_fingerprint": _tree_fingerprint(model_dir, include_contents=False),
        "prompt_json": str(prompt_path),
        "prompt_json_sha256": _sha256_file(prompt_path),
        "image": str(image_path) if image_path is not None else None,
        "image_sha256": _sha256_file(image_path) if image_path is not None else None,
        "sampling": {
            key: value
            for key, value in call_args.items()
            if key not in {"generator", "image", "prompt", "negative_prompt"}
        },
        "seed": args.seed,
        "prompt": prompt,
        "negative_prompt": call_args["negative_prompt"],
        "rng_before": rng_before,
        "rng_after": _generator_state(generator),
        "trace": args.trace,
        "events": recorder.events,
        "tensors": recorder.tensor_metadata,
        "frames": {
            "path": "frames.npy",
            "shape": list(frames_array.shape),
            "dtype": str(frames_array.dtype),
            "sha256": hashlib.sha256(frames_array.tobytes()).hexdigest(),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "versions": _versions(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "device": str(device),
            "qwen_attn_implementation": os.environ.get("LINGBOT_QWEN_ATTN_IMPLEMENTATION"),
        },
    }
    _write_json(destination / "metadata.json", metadata)
    return metadata


def _repeatability_report(root: Path, captures: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for capture in captures:
        grouped.setdefault((capture["mode"], capture["case"]), []).append(capture)
    report: dict[str, Any] = {"schema_version": 2, "groups": {}}
    for (mode, case), values in grouped.items():
        frame_hashes = [item["frames"]["sha256"] for item in values]
        shared_names = set(values[0]["tensors"])
        for value in values[1:]:
            shared_names.intersection_update(value["tensors"])
        tensor_hashes = {name: [value["tensors"][name]["sha256"] for value in values] for name in sorted(shared_names)}
        mismatched_tensors = [name for name, hashes in tensor_hashes.items() if len(set(hashes)) != 1]
        report["groups"][f"{mode}/{case}"] = {
            "runs": len(values),
            "exact_frame_hash_match": len(set(frame_hashes)) == 1,
            "frame_hashes": frame_hashes,
            "exact_tensor_hash_match": not mismatched_tensors,
            "shared_tensor_count": len(tensor_hashes),
            "mismatched_tensor_hashes": mismatched_tensors,
            "elapsed_seconds": [item["elapsed_seconds"] for item in values],
        }
    _write_json(root / "repeatability.json", report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_DENSE_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/lingbot_video_reference"))
    parser.add_argument("--all-cases", action="store_true", help="Capture every bundled case for the selected modes.")
    parser.add_argument("--mode", choices=["t2i", "t2v", "ti2v"], action="append")
    parser.add_argument("--case", default="example_1")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace", choices=["none", "sampled", "full"], default="full")
    parser.add_argument("--default-dtype", default="bf16")
    parser.add_argument("--transformer-dtype", default="bf16")
    parser.add_argument("--text-encoder-dtype", default="bf16")
    parser.add_argument("--vae-dtype", default="fp32")
    parser.add_argument("--transformer-subfolder", default="transformer")
    parser.add_argument("--qwen-attn-implementation", default="sdpa")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        parser.error("--num-frames must be 1 or 4n+1")
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must be multiples of 16")
    return args


def main() -> None:
    args = _parse_args()
    upstream_root = args.upstream_root.resolve()
    model_dir = args.model_dir.resolve()
    modes = args.mode or ["t2i", "t2v", "ti2v"]
    cases = _sample_cases(upstream_root, modes, None if args.all_cases else args.case)
    run_manifest = {
        "schema_version": 1,
        "upstream_root": str(upstream_root),
        "all_cases": args.all_cases,
        "selected_cases": [{"mode": item["mode"], "name": item["name"]} for item in cases],
        "model_dir": str(model_dir),
        "modes": modes,
        "case": args.case,
        "dry_run": args.dry_run,
        "arguments": vars(args),
    }
    if args.dry_run:
        _write_json(args.output_dir / "capture_manifest.json", run_manifest)
        print(json.dumps(run_manifest, indent=2, default=_json_value))
        return
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Dense checkpoint is not available: {model_dir}")
    os.environ["LINGBOT_QWEN_ATTN_IMPLEMENTATION"] = args.qwen_attn_implementation
    os.environ["LINGBOT_QUIET_PROGRESS"] = "1"
    runner = _import_upstream(upstream_root)
    captures: list[dict[str, Any]] = []
    for case in cases:
        for repeat in range(args.repeats):
            destination = args.output_dir / case["mode"] / case["name"] / f"run-{repeat:02d}"
            destination.mkdir(parents=True, exist_ok=False)
            print(f"capturing {case['mode']}/{case['name']} run={repeat}", flush=True)
            captures.append(
                _capture_once(
                    runner=runner,
                    upstream_root=upstream_root,
                    model_dir=model_dir,
                    case=case,
                    destination=destination,
                    args=args,
                    repeat=repeat,
                )
            )
    _write_json(args.output_dir / "capture_manifest.json", run_manifest)
    _repeatability_report(args.output_dir, captures)


if __name__ == "__main__":
    main()
