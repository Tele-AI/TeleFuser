"""Capture a layered artifact from the fixed official LingBot-VLA v2 checkout.

Run this script with the dedicated upstream uv environment and with the fixed
upstream checkout as ``--upstream-root``. The official code forces
FlashAttention during construction. For reproducible comparison on the local
PyTorch 2.11 stack, this runner intercepts model construction inside this
process only and selects the eager attention implementation used by TeleFuser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from accelerate import init_empty_weights
from safetensors.torch import load_file
from torchvision.transforms.v2 import Resize
from transformers import AutoConfig, AutoProcessor, PreTrainedModel

ARTIFACT_SCHEMA_VERSION = 2
UPSTREAM_COMMIT = "be27333c9b5f2663b0ec33f069dd7dfd67fa32b5"
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


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


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError(f"Upstream checkout must be clean, got:\n{status.stdout}")
    return completed.stdout.strip()


def _checkpoint_shards(model_root: Path) -> list[Path]:
    index_path = model_root / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid checkpoint index: {index_path}")
    shards = [model_root / name for name in sorted(set(weight_map.values()))]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
    return shards


class TensorCapture:
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
    def __init__(self, capture: TensorCapture) -> None:
        self.capture = capture
        self.step = 0

    def record(self, original: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        if len(args) < 5:
            raise RuntimeError("Official predict_velocity trace received an unexpected call signature")
        suffix = f"{self.step:02d}"
        x_t, timestep = args[3], args[4]
        if self.step == 0:
            self.capture.add("initial_noise", x_t)
        self.capture.add(f"timestep_step_{suffix}", timestep)
        self.capture.add(f"x_t_step_{suffix}", x_t)
        velocity = original(*args, **kwargs)
        self.capture.add(f"velocity_step_{suffix}", velocity)
        self.step += 1
        return velocity


@contextmanager
def _trace_predict_velocity(flow_model: Any, capture: TensorCapture) -> Iterator[VelocityTrace]:
    original = flow_model.predict_velocity
    trace = VelocityTrace(capture)
    flow_model.predict_velocity = lambda *args, **kwargs: trace.record(original, *args, **kwargs)
    try:
        yield trace
    finally:
        del flow_model.predict_velocity


def _force_eager(config: Any) -> None:
    for current in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if current is not None:
            current._attn_implementation = "eager"


@contextmanager
def _eager_construction() -> Iterator[None]:
    """Override the upstream hard-coded FA2 selection in this process only."""
    original = PreTrainedModel._from_config.__func__

    def from_config(cls: type[PreTrainedModel], config: Any, **kwargs: Any) -> PreTrainedModel:
        _force_eager(config)
        return original(cls, config, **kwargs)

    PreTrainedModel._from_config = classmethod(from_config)
    try:
        yield
    finally:
        PreTrainedModel._from_config = classmethod(original)


def _official_model_values(qwen3vl_root: Path) -> dict[str, Any]:
    # Base-6B values are documented in the fixed upstream Training_Config.md.
    return {
        "post_training": False,
        "adanorm_time": True,
        "moe_implementation": "fused",
        "use_robby_moe_kernel": False,
        "attention_implementation": "eager",
        "vit_attn_implementation": "eager",
        "precompute_grid_thw": True,
        "vlm_causal": True,
        "use_moe": True,
        "token_moe_layers": list(range(36)),
        "token_num_experts": 32,
        "token_top_k": 4,
        "token_moe_intermediate_size": 512,
        "token_shared_intermediate_size": 704,
        "bias_update_speed": 0.0,
        "sequence_wise_mode": "per_sequence",
        "sequence_wise_loss_coeff": 1e-3,
        "router_z_loss_coeff": 1e-4,
        "router_activation": "sigmoid",
        "routed_scaling_factor": 4.0,
        "use_shared_expert_gate": False,
        "freeze_vision_encoder": False,
        "tokenizer_max_length": 72,
        "loss_type": "L1_fm",
        "action_dim": 55,
        "max_action_dim": 55,
        "max_state_dim": 55,
        "tokenizer_path": str(qwen3vl_root),
        "align_params": {
            "mode": "query",
            "num_task_tokens": 8,
            "depth_loss_weight": 0.004,
            "future_depth_loss_weight": 0.004,
            "use_future_video": True,
            "llm": {"dim_out": 2560, "image_token_size": 8, "image_input_size": 224},
            "depth": {
                "model_type": "MoRGBD",
                "num_layers": 1,
                "num_heads": 4,
                "dim_head": 32,
                "ff_mult": 1,
                "num_backbone_tokens": 256,
                "token_size": 16,
                "dim_out": 1024,
                "input_size": 224,
                "use_future_depth": True,
                "block_future_depth_to_action": True,
                "future_depth_head_type": "resampler",
                "detach_future_image_feats": True,
            },
            "video": {
                "attention_mode": "flex_block_causal",
                "input_size": 256,
                "block_suffix_to_future_video": True,
                "share_future_depth_query": True,
                "use_shared_future_task_proj": True,
                "use_current_shared_task_proj": True,
                "num_future_frames": 1,
                "use_warmup_frame": True,
                "effective_fps": 1.0,
                "n_blocks": 1,
                "cls_pool": "last",
                "detach_image_feats": True,
                "num_layers": 1,
                "num_heads": 4,
                "dim_head": 32,
                "ff_mult": 1,
                "num_backbone_tokens": 256,
                "dim_out": 1024,
                "future_video_loss_weight": 0.004,
                "use_smooth_l1_loss": False,
                "use_mse_loss": True,
                "mse_loss_weight": 1.0,
                "use_patch_loss": True,
                "use_current_patch_loss": True,
                "use_cosine_loss": False,
                "cosine_loss_weight": 0.2,
                "use_cls_loss": False,
                "cls_loss_type": "mse",
                "cls_loss_weight": 0.2,
            },
        },
    }


def _build_config(qwen3vl_root: Path) -> Any:
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config

    config = LingbotVLAV2Config(**_official_model_values(qwen3vl_root))
    qwen_config = AutoConfig.from_pretrained(str(qwen3vl_root), local_files_only=True)
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
    ):
        if hasattr(qwen_config.text_config, key):
            setattr(config, key, getattr(qwen_config.text_config, key))
    config.vision_config = qwen_config.vision_config
    config.use_cache = True
    return config


def _load_official_model(model_root: Path, config: Any, device: torch.device) -> Any:
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch

    apply_lingbot_qwen3_vl_patch()
    with _eager_construction(), init_empty_weights():
        model = LingbotVlaV2Policy(config, eval=True)

    index = json.loads((model_root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    checkpoint_keys = set(index["weight_map"])
    model_keys = set(model.state_dict())
    if checkpoint_keys != model_keys:
        raise RuntimeError(
            "Official model/checkpoint key mismatch: "
            f"missing={sorted(model_keys - checkpoint_keys)[:10]}, "
            f"unexpected={sorted(checkpoint_keys - model_keys)[:10]}"
        )

    for shard in _checkpoint_shards(model_root):
        model.load_state_dict(load_file(shard, device="cpu"), strict=False, assign=True)
    unmaterialized = [name for name, tensor in model.state_dict().items() if tensor.is_meta]
    if unmaterialized:
        raise RuntimeError(f"Official checkpoint left meta tensors: {unmaterialized[:10]}")
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB")).copy()
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _prepare_inputs(
    upstream_root: Path,
    qwen3vl_root: Path,
    config: Any,
    image_paths: Sequence[Path],
    task: str,
    state: Sequence[float],
) -> dict[str, torch.Tensor]:
    from lingbotvla.data.vla_data.utils import FeatureTransform

    processor = AutoProcessor.from_pretrained(str(qwen3vl_root), local_files_only=True, padding_side="right")
    data_config = SimpleNamespace(
        joints=["{'arm.position': 14}", "{'end.position': 14}", "{'effector.position': 2}"],
        cameras=["camera_top", "camera_wrist_left", "camera_wrist_right"],
        norm_type=[
            "{'arm.position': 'bounds_99_woclip'}",
            "{'end.position': 'bounds_99_woclip'}",
            "{'effector.position': 'bounds_99_woclip'}",
        ],
    )
    transform = FeatureTransform(
        upstream_root / "configs/robot_configs/robotwin.yaml",
        data_config,
        config,
        processor,
        chunk_size=config.chunk_size,
        norm_stats_path=upstream_root / "assets/norm_stats/robotwin.json",
    )
    resize = Resize((256, 256), antialias=True)
    item: dict[str, Any] = {"observation.state": torch.tensor(state, dtype=torch.float32), "task": task}
    for key, path in zip(CAMERA_KEYS, image_paths, strict=True):
        item[key] = resize(_load_rgb(path).to(dtype=torch.float32))
    prepared = transform.apply(item, policy_eval=True)
    return {
        "images": prepared["images"].unsqueeze(0),
        "img_masks": prepared["img_masks"].unsqueeze(0),
        "image_grid_thw": prepared["image_grid_thw"].unsqueeze(0),
        "lang_tokens": prepared["lang_tokens"].unsqueeze(0),
        "lang_masks": prepared["lang_masks"].unsqueeze(0),
        "state": prepared["state"].unsqueeze(0),
    }


def capture_artifact(
    *,
    upstream_root: Path,
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
) -> tuple[Path, Path]:
    commit = _git_commit(upstream_root)
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected upstream commit {UPSTREAM_COMMIT}, got {commit}")
    if len(image_paths) != len(CAMERA_KEYS):
        raise ValueError(f"expected {len(CAMERA_KEYS)} camera paths, got {len(image_paths)}")

    sys.path.insert(0, str(upstream_root))
    output = output.with_suffix(".npz")
    metadata_path = output.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError("Official 6B parity capture currently requires CUDA")

    config = _build_config(qwen3vl_root)
    inputs = _prepare_inputs(upstream_root, qwen3vl_root, config, image_paths, task, state)
    capture = TensorCapture()
    for key, tensor in inputs.items():
        capture.add(key, tensor)

    model = _load_official_model(model_root, config, target_device)
    if deterministic_moe:
        import lingbotvla.models.vla.lingbot_vla.qwen2_action_expert as qwen2_action_expert

        qwen2_action_expert.robby_moe_forward = None

        def deterministic_forward(experts, module, num_experts, routing_weights, selected_experts, hidden_states):
            del module
            output = torch.zeros_like(hidden_states)
            for expert_id in range(num_experts):
                routes = (selected_experts == expert_id).nonzero(as_tuple=False)
                if routes.numel() == 0:
                    continue
                token_ids, route_ids = routes[:, 0], routes[:, 1]
                expert_input = hidden_states.index_select(0, token_ids)
                gate = torch.nn.functional.linear(expert_input, experts.gate_proj[expert_id])
                up = torch.nn.functional.linear(expert_input, experts.up_proj[expert_id])
                intermediate = torch.nn.functional.silu(gate) * up
                expert_output = torch.nn.functional.linear(intermediate, experts.down_proj[expert_id])
                weights = routing_weights[token_ids, route_ids].unsqueeze(-1)
                output.index_add_(0, token_ids, expert_output * weights)
            return output

        for module in model.modules():
            if module.__class__.__name__ == "Qwen2FusedExperts":
                module.forward = MethodType(deterministic_forward, module)
    tensors = {
        "images": inputs["images"].to(device=target_device, dtype=torch.bfloat16),
        "img_masks": inputs["img_masks"].to(device=target_device),
        "lang_tokens": inputs["lang_tokens"].to(device=target_device),
        "lang_masks": inputs["lang_masks"].to(device=target_device),
        "state": inputs["state"].to(device=target_device, dtype=torch.bfloat16),
        "image_grid_thw": inputs["image_grid_thw"].to(device=target_device, dtype=torch.long),
    }
    generator = torch.Generator(device=target_device).manual_seed(seed)
    noise = torch.randn(
        1,
        int(config.n_action_steps),
        int(config.max_action_dim),
        device=target_device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    with torch.inference_mode(), _trace_predict_velocity(model.model, capture) as trace:
        actions = model.sample_actions(**tensors, noise=noise)
    capture.add("canonical_normalized_actions", actions.squeeze(0).to(device="cpu", dtype=torch.float32))
    if trace.step != int(config.num_steps):
        raise RuntimeError(f"captured {trace.step} denoising steps, expected {config.num_steps}")

    checkpoint_paths = _checkpoint_shards(model_root)
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "official_upstream_common_eager",
        "upstream_commit": commit,
        "upstream_attention_override": "process_local_pretrained_model_from_config_intercept",
        "checkpoint_manifest_sha256": _manifest_sha256(checkpoint_paths, include_contents=full_checkpoint_hash),
        "checkpoint_hash_mode": "full_sha256" if full_checkpoint_hash else "filename_and_size",
        "norm_stats_sha256": _sha256_file(upstream_root / "assets/norm_stats/robotwin.json"),
        "processor_manifest_sha256": _manifest_sha256(_processor_files(qwen3vl_root), include_contents=True),
        "input_sha256": _input_sha256(task, state, image_paths),
        "seed": seed,
        "num_steps": trace.step,
        "torch_dtype": "bfloat16",
        "attention_backend": "eager",
        "vision_attention_backend": "eager",
        "moe_backend": "deterministic_torch_reference" if deterministic_moe else "upstream_triton",
        "device": str(target_device),
        "device_name": torch.cuda.get_device_name(target_device),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "arrays": capture.array_metadata,
    }
    np.savez(output, **capture.arrays)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
    parser.add_argument("--upstream-root", required=True, type=Path)
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
    parser.add_argument("--full-checkpoint-hash", action="store_true")
    parser.add_argument(
        "--deterministic-moe",
        action="store_true",
        help="Disable the upstream atomic Triton MoE kernel for bitwise cross-process parity",
    )
    args = parser.parse_args()

    paths = (
        args.upstream_root,
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
        upstream_root=args.upstream_root.resolve(),
        model_root=args.model_root.resolve(),
        qwen3vl_root=args.qwen3vl_root.resolve(),
        image_paths=(args.camera_high, args.camera_left_wrist, args.camera_right_wrist),
        task=args.task,
        state=args.state_json,
        seed=args.seed,
        output=args.output,
        device=args.device,
        full_checkpoint_hash=args.full_checkpoint_hash,
        deterministic_moe=args.deterministic_moe,
    )
    print(f"Saved official LingBot-VLA v2 capture: {artifact}")
    print(f"Saved official LingBot-VLA v2 metadata: {metadata}")


if __name__ == "__main__":
    main()
