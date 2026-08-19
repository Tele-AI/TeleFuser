"""Capture a layered TeleFuser LingBot-VLA v2 regression artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import transformers

from telefuser.models.lingbot_vla_v2_loader import resolve_lingbot_vla_v2_shards
from telefuser.models.lingbot_vla_v2_quantization import lingbot_vla_v2_quantization_identity
from telefuser.pipelines.lingbot_vla_v2 import (
    ROBOTWIN_CAMERA_KEYS,
    LingBotVlaV2Observation,
    LingBotVlaV2Pipeline,
)
from telefuser.pipelines.lingbot_vla_v2.runtime import get_lingbot_vla_v2_pipeline

ARTIFACT_SCHEMA_VERSION = 1


def _sha256_file(path: Path, digest: Any | None = None) -> str:
    result = hashlib.sha256() if digest is None else digest
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _manifest_sha256(paths: Sequence[Path], *, include_contents: bool) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=lambda item: item.name):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        if include_contents:
            _sha256_file(path, digest)
    return digest.hexdigest()


def _processor_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix != ".safetensors")


def _input_sha256(task: str, state: Sequence[float], image_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    canonical = json.dumps({"task": task, "state": list(state)}, sort_keys=True, separators=(",", ":"))
    digest.update(canonical.encode("utf-8"))
    for path in image_paths:
        digest.update(path.name.encode("utf-8"))
        _sha256_file(path, digest)
    return digest.hexdigest()


def _git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class TensorCapture:
    """Own CPU snapshots and their original tensor contracts."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.array_metadata: dict[str, dict[str, object]] = {}

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if key in self.arrays:
            raise ValueError(f"Duplicate capture key: {key}")
        snapshot = tensor.detach().cpu().clone()
        stored = snapshot.float() if snapshot.is_floating_point() else snapshot
        array = stored.numpy()
        self.arrays[key] = array
        self.array_metadata[key] = {
            "shape": list(snapshot.shape),
            "original_dtype": str(snapshot.dtype).removeprefix("torch."),
            "stored_dtype": str(array.dtype),
        }


class VelocityTrace:
    """Capture the state around each call to the flow-matching velocity model."""

    def __init__(self, capture: TensorCapture) -> None:
        self.capture = capture
        self.step = 0

    def record(self, original: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) < 5:
            raise RuntimeError("LingBot-VLA v2 predict_velocity trace received an unexpected call signature")
        x_t = args[3]
        timestep = args[4]
        suffix = f"{self.step:02d}"
        if self.step == 0:
            self.capture.add("initial_noise", x_t)
        self.capture.add(f"timestep_step_{suffix}", timestep)
        self.capture.add(f"x_t_step_{suffix}", x_t)
        velocity = original(*args, **kwargs)
        self.capture.add(f"velocity_step_{suffix}", velocity)
        self.step += 1
        return velocity


@contextmanager
def trace_predict_velocity(flow_model: Any, capture: TensorCapture) -> Iterator[VelocityTrace]:
    """Temporarily trace one model instance without changing global classes."""
    original = flow_model.predict_velocity
    had_instance_override = "predict_velocity" in vars(flow_model)
    previous_override = vars(flow_model).get("predict_velocity")
    compile_enabled = bool(getattr(flow_model, "_use_compile_predict_velocity", False))
    trace = VelocityTrace(capture)

    flow_model._use_compile_predict_velocity = False
    flow_model.predict_velocity = lambda *args, **kwargs: trace.record(original, *args, **kwargs)
    try:
        yield trace
    finally:
        if had_instance_override:
            flow_model.predict_velocity = previous_override
        else:
            del flow_model.predict_velocity
        flow_model._use_compile_predict_velocity = compile_enabled


def _build_pipeline(
    model_root: Path,
    qwen3vl_root: Path,
    device: str,
    quantization: str | None,
) -> LingBotVlaV2Pipeline:
    return get_lingbot_vla_v2_pipeline(
        str(model_root),
        str(qwen3vl_root),
        device=device,
        quantization=quantization,
    )


def capture_artifact(
    *,
    model_root: Path,
    qwen3vl_root: Path,
    image_paths: Sequence[Path],
    task: str,
    state: Sequence[float],
    seed: int,
    output: Path,
    device: str,
    full_checkpoint_hash: bool,
    deterministic_moe: bool,
    quantization: str | None = None,
) -> tuple[Path, Path]:
    if len(image_paths) != len(ROBOTWIN_CAMERA_KEYS):
        raise ValueError(f"expected {len(ROBOTWIN_CAMERA_KEYS)} camera paths, got {len(image_paths)}")
    output = output.with_suffix(".npz")
    metadata_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(model_root, qwen3vl_root, device, quantization)
    if deterministic_moe:
        for module in pipeline.policy_stage.policy.modules():
            if hasattr(module, "_use_robby_moe_kernel"):
                module._use_robby_moe_kernel = False
    capture = TensorCapture()
    try:
        observation = LingBotVlaV2Observation(
            task=task,
            state=state,
            images=dict(zip(ROBOTWIN_CAMERA_KEYS, image_paths, strict=True)),
        )
        inputs = pipeline.input_processor.prepare(observation)
        for key in ("images", "img_masks", "image_grid_thw", "lang_tokens", "lang_masks", "state"):
            capture.add(key, getattr(inputs, key))

        flow_model = pipeline.policy_stage.policy.model
        with trace_predict_velocity(flow_model, capture) as trace:
            chunk = pipeline.predict(inputs, seed=seed)
        capture.add("canonical_normalized_actions", chunk.canonical_normalized_actions)

        expected_steps = int(flow_model.config.num_steps)
        if trace.step != expected_steps:
            raise RuntimeError(f"captured {trace.step} denoising steps, expected {expected_steps}")

        target_device = torch.device(device)
        checkpoint_paths = [Path(path) for path in resolve_lingbot_vla_v2_shards(model_root)]
        metadata = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_kind": "telefuser_regression",
            "telefuser_commit": _git_commit(),
            "checkpoint_manifest_sha256": _manifest_sha256(
                checkpoint_paths,
                include_contents=full_checkpoint_hash,
            ),
            "checkpoint_hash_mode": "full_sha256" if full_checkpoint_hash else "filename_and_size",
            "norm_stats_sha256": _sha256_file(
                Path(__file__).resolve().parents[2]
                / "telefuser/pipelines/lingbot_vla_v2/assets/robotwin_norm_stats.json"
            ),
            "processor_manifest_sha256": _manifest_sha256(
                _processor_files(qwen3vl_root),
                include_contents=True,
            ),
            "input_sha256": _input_sha256(task, state, image_paths),
            "seed": seed,
            "num_steps": trace.step,
            "torch_dtype": str(pipeline.torch_dtype).removeprefix("torch."),
            "attention_backend": str(flow_model.config.attention_implementation),
            "moe_backend": "deterministic_torch_reference" if deterministic_moe else "upstream_triton",
            "device": str(target_device),
            "device_name": torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "quantization": lingbot_vla_v2_quantization_identity(pipeline.policy_stage.policy),
            "arrays": capture.array_metadata,
        }
        np.savez(output, **capture.arrays)
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        pipeline.close()

    return output, metadata_path


def _parse_state(value: str) -> list[float]:
    try:
        state = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("state-json must be valid JSON") from error
    if not isinstance(state, list) or len(state) != 14 or any(isinstance(item, bool) for item in state):
        raise argparse.ArgumentTypeError("state-json must be a 14-element numeric JSON list")
    try:
        return [float(item) for item in state]
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("state-json must contain only numeric values") from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--qwen3vl-root", required=True, type=Path)
    parser.add_argument("--camera-high", required=True, type=Path)
    parser.add_argument("--camera-left-wrist", required=True, type=Path)
    parser.add_argument("--camera-right-wrist", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--state-json", required=True, type=_parse_state)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantization", choices=("torchao-fp8", "tf-kernel-fp8", "bnb-nf4"))
    parser.add_argument(
        "--full-checkpoint-hash",
        action="store_true",
        help="Hash all checkpoint bytes instead of the faster filename-and-size manifest",
    )
    parser.add_argument(
        "--deterministic-moe",
        action="store_true",
        help="Disable the upstream atomic Triton MoE kernel for bitwise cross-process parity",
    )
    args = parser.parse_args()

    paths = (
        args.model_root,
        args.qwen3vl_root,
        args.camera_high,
        args.camera_left_wrist,
        args.camera_right_wrist,
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        parser.error(f"input paths do not exist: {missing}")

    artifact, metadata = capture_artifact(
        model_root=args.model_root,
        qwen3vl_root=args.qwen3vl_root,
        image_paths=(args.camera_high, args.camera_left_wrist, args.camera_right_wrist),
        task=args.task,
        state=args.state_json,
        seed=args.seed,
        output=args.output,
        device=args.device,
        full_checkpoint_hash=args.full_checkpoint_hash,
        deterministic_moe=args.deterministic_moe,
        quantization=args.quantization,
    )
    print(f"Saved LingBot-VLA v2 capture: {artifact}")
    print(f"Saved LingBot-VLA v2 metadata: {metadata}")


if __name__ == "__main__":
    main()
