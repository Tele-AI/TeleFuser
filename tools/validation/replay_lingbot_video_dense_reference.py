"""Replay a captured Dense LingBot-Video oracle through native TeleFuser stages.

The upstream capture supplies prompt embeddings, initial latents, and optional
TI2V condition latents. This isolates native DiT, scheduler, condition, and
VAE-decode parity from text-encoder and RNG differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_video.data import preprocess_ti2v_image
from telefuser.pipelines.lingbot_video.denoising import LingBotVideoDenoisingStage, reinject_ti2v_condition
from telefuser.pipelines.lingbot_video.text_encoding import LingBotVideoTextEncodingStage
from telefuser.pipelines.lingbot_video.vae import (
    LingBotVideoVAEDecodeStage,
    LingBotVideoVAEEncodeStage,
    denormalize_latent,
    first_frame_condition_mask,
)
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler


def tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int | bool]:
    """Return L0/L1 metrics for one captured tensor pair."""
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_numel": reference.numel(),
            "candidate_numel": candidate.numel(),
        }
    ref = reference.float().cpu()
    got = candidate.float().cpu()
    delta = (got - ref).abs()
    return {
        "shape_match": True,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm().div(ref.norm().clamp_min(1e-12)).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()),
        "exact_mismatch_count": int(torch.count_nonzero(reference.cpu() != candidate.cpu()).item()),
    }


def _load_tensor(reference_dir: Path, metadata: dict[str, Any], name: str) -> torch.Tensor:
    entry = metadata["tensors"].get(name)
    if entry is None:
        raise KeyError(f"capture does not contain required tensor: {name}")
    return torch.load(reference_dir / entry["path"], map_location="cpu", weights_only=True)


def _record(
    results: dict[str, dict[str, float | int | bool]], name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> None:
    results[name] = tensor_metrics(reference, candidate)


def discover_reference_dirs(reference_root: Path) -> list[Path]:
    """Return captured-run directories under a reference artifact root."""
    references = sorted(path.parent for path in reference_root.rglob("metadata.json"))
    if not references:
        raise FileNotFoundError(f"No capture metadata found under {reference_root}")
    return references


def exact_replay_failures(report: dict[str, Any]) -> list[str]:
    """Return metric paths that do not meet the exact numerical-oracle gate."""
    reports = report.get("reports", [report])
    failures: list[str] = []
    for item_index, item in enumerate(reports):
        metrics = item.get("metrics", {})
        for name, value in metrics.items():
            if not value.get("shape_match", False) or value.get("exact_mismatch_count") != 0:
                failures.append(f"report[{item_index}].{name}")
    return failures


class _RecordingProcessor:
    """Proxy a Hugging Face processor while retaining its pre-device input tensors."""

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._processor(*args, **kwargs)
        self.calls.append(
            {
                name: value.detach().cpu().clone(memory_format=torch.contiguous_format)
                for name in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
                if (value := inputs.get(name)) is not None
            }
        )
        return inputs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)


def replay_text(
    reference_dir: Path,
    model_dir: Path,
    *,
    validate_ti2v_vae: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Replay source prompt construction and Qwen3-VL encoding for one capture."""
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Text replay requires transformers with Qwen3-VL support") from exc

    metadata = json.loads((reference_dir / "metadata.json").read_text(encoding="utf-8"))
    device = torch.device("cuda")
    runtime_config = ModelRuntimeConfig(device_type="cuda", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_dir / "processor")
    text_encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir / "text_encoder", dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    module_manager = ModuleManager(device="cuda", torch_dtype=torch.bfloat16)
    module_manager.add_module(text_encoder, name="text_encoder")
    module_manager.add_module(processor, name="processor")
    stage = LingBotVideoTextEncodingStage("text_encoder", module_manager, runtime_config)
    stage._crop_prefix_length()
    recording_processor = _RecordingProcessor(processor)
    stage.processor = recording_processor

    images: list[Image.Image] | None = None
    results: dict[str, dict[str, float | int | bool]] = {}
    if metadata["mode"] == "ti2v":
        image_path = Path(metadata["image"])
        raw_image = (
            torch.from_numpy(np.array(Image.open(image_path).convert("RGB"), copy=True)).permute(2, 0, 1).unsqueeze(0)
        )
        sampling = metadata["sampling"]
        condition_pixels = preprocess_ti2v_image(raw_image, height=sampling["height"], width=sampling["width"])
        _record(
            results,
            "ti2v_preprocessed_image",
            _load_tensor(reference_dir, metadata, "ti2v_preprocessed_image"),
            condition_pixels,
        )
        images = [stage.prepare_ti2v_vlm_image(condition_pixels)]
        if validate_ti2v_vae:
            try:
                from diffusers import AutoencoderKLWan
            except ImportError as exc:
                raise RuntimeError("TI2V VAE replay requires diffusers AutoencoderKLWan") from exc
            vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
            module_manager.add_module(vae, name="vae")
            vae_encode = LingBotVideoVAEEncodeStage("vae_encode", module_manager, runtime_config)
            generator = torch.Generator(device=device).manual_seed(metadata.get("seed", seed))
            _record(
                results,
                "ti2v_condition_latent",
                _load_tensor(reference_dir, metadata, "ti2v_condition_latent"),
                vae_encode.encode(condition_pixels, generator=generator),
            )

    for call, prompt in enumerate((metadata["prompt"], metadata["negative_prompt"])):
        embeddings, mask = stage.encode(prompt, images=images)
        _record(
            results, f"prompt_{call}_embeds", _load_tensor(reference_dir, metadata, f"prompt_{call}_embeds"), embeddings
        )
        _record(results, f"prompt_{call}_mask", _load_tensor(reference_dir, metadata, f"prompt_{call}_mask"), mask)
        inputs = recording_processor.calls[call]
        for name, candidate in inputs.items():
            reference_name = f"prompt_inputs_{call + 1}_{name}"
            if reference_name in metadata["tensors"]:
                _record(results, reference_name, _load_tensor(reference_dir, metadata, reference_name), candidate)
    return {
        "reference_dir": str(reference_dir),
        "mode": metadata["mode"],
        "case": metadata["case"],
        "metrics": results,
    }


def _load_replay_runtime(
    model_dir: Path,
) -> tuple[ModelRuntimeConfig, LingBotVideoDenoisingStage, Any]:
    """Load the Dense DiT and VAE once for one or more capture replays."""
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise RuntimeError("Dense replay requires diffusers AutoencoderKLWan") from exc
    device = torch.device("cuda")
    runtime_config = ModelRuntimeConfig(device_type="cuda", torch_dtype=torch.bfloat16)
    module_manager = ModuleManager(device="cuda", torch_dtype=torch.bfloat16)
    module_manager.load_model(str(model_dir / "transformer"), name="transformer")
    transformer = module_manager.fetch_module("transformer")
    if transformer is None:
        raise RuntimeError(f"Unable to load LingBot-Video transformer from {model_dir / 'transformer'}")
    transformer.promote_stability_layers_to_fp32()
    denoising = LingBotVideoDenoisingStage("transformer", module_manager, runtime_config)
    vae = AutoencoderKLWan.from_pretrained(model_dir / "vae", torch_dtype=torch.float32).to(device).eval()
    return runtime_config, denoising, vae


def _replay_with_runtime(
    reference_dir: Path,
    model_dir: Path,
    *,
    runtime_config: ModelRuntimeConfig,
    denoising: LingBotVideoDenoisingStage,
    vae: Any,
) -> dict[str, Any]:
    """Replay one capture using already-loaded Dense and VAE components."""
    metadata = json.loads((reference_dir / "metadata.json").read_text(encoding="utf-8"))
    sampling = metadata["sampling"]
    device = torch.device("cuda")
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(model_dir / "scheduler")
    scheduler.set_timesteps(sampling["num_inference_steps"], device=device, shift=sampling["shift"])
    results: dict[str, dict[str, float | int | bool]] = {}
    _record(
        results,
        "scheduler_timesteps",
        _load_tensor(reference_dir, metadata, "scheduler_timesteps"),
        scheduler.timesteps,
    )
    _record(results, "scheduler_sigmas", _load_tensor(reference_dir, metadata, "scheduler_sigmas"), scheduler.sigmas)

    current = _load_tensor(reference_dir, metadata, "initial_latents").to(device)
    positive = _load_tensor(reference_dir, metadata, "prompt_0_embeds").to(device)
    positive_mask = _load_tensor(reference_dir, metadata, "prompt_0_mask").to(device)
    negative = _load_tensor(reference_dir, metadata, "prompt_1_embeds").to(device)
    negative_mask = _load_tensor(reference_dir, metadata, "prompt_1_mask").to(device)
    condition = None
    condition_mask = None
    if metadata["mode"] == "ti2v":
        condition = _load_tensor(reference_dir, metadata, "ti2v_condition_latent").to(device)
        condition_mask = first_frame_condition_mask(current.shape[2], device=device)

    for index, timestep in enumerate(scheduler.timesteps):
        if condition is not None and condition_mask is not None:
            current = reinject_ti2v_condition(current, condition, condition_mask)
        timestep_batch = timestep.expand(current.shape[0])
        noise = denoising.predict_noise_with_cfg(
            current,
            timestep_batch,
            positive,
            negative,
            positive_mask,
            negative_mask,
            float(sampling["guidance_scale"]),
        )
        prefix = f"step_{index}"
        if f"{prefix}_noise_prediction" in metadata["tensors"]:
            _record(
                results,
                f"{prefix}_noise_prediction",
                _load_tensor(reference_dir, metadata, f"{prefix}_noise_prediction"),
                noise,
            )
        current = scheduler.step(noise, timestep, current)
        if f"{prefix}_latent_after" in metadata["tensors"]:
            _record(
                results,
                f"{prefix}_latent_after",
                _load_tensor(reference_dir, metadata, f"{prefix}_latent_after"),
                current,
            )
    if condition is not None and condition_mask is not None:
        current = reinject_ti2v_condition(current, condition, condition_mask)

    raw_latent = denormalize_latent(
        current,
        torch.tensor(vae.config.latents_mean, device=device),
        torch.tensor(vae.config.latents_std, device=device),
    )
    _record(results, "vae_decode_input", _load_tensor(reference_dir, metadata, "vae_decode_input"), raw_latent)
    vae_manager = ModuleManager(device="cuda", torch_dtype=torch.bfloat16)
    vae_manager.add_module(vae, name="vae")
    decoder = LingBotVideoVAEDecodeStage("vae_decode", vae_manager, runtime_config)
    frames = decoder.decode(current)
    captured_frames = torch.from_numpy(np.load(reference_dir / metadata["frames"]["path"]))
    candidate_frames = frames.permute(0, 2, 3, 4, 1)[0].cpu()
    _record(results, "decoded_frames", captured_frames, candidate_frames)
    return {
        "reference_dir": str(reference_dir),
        "mode": metadata["mode"],
        "case": metadata["case"],
        "metrics": results,
    }


def replay(reference_dir: Path, model_dir: Path) -> dict[str, Any]:
    """Replay a one-GPU Dense capture and return its native stage parity report."""
    runtime_config, denoising, vae = _load_replay_runtime(model_dir)
    return _replay_with_runtime(
        reference_dir,
        model_dir,
        runtime_config=runtime_config,
        denoising=denoising,
        vae=vae,
    )


def replay_many(reference_dirs: list[Path], model_dir: Path) -> dict[str, Any]:
    """Replay multiple Dense captures while retaining one DiT/VAE runtime."""
    if not reference_dirs:
        raise ValueError("reference_dirs must not be empty")
    runtime_config, denoising, vae = _load_replay_runtime(model_dir)
    reports = [
        _replay_with_runtime(
            reference_dir,
            model_dir,
            runtime_config=runtime_config,
            denoising=denoising,
            vae=vae,
        )
        for reference_dir in reference_dirs
    ]
    return {
        "reference_count": len(reports),
        "reference_dirs": [str(reference_dir) for reference_dir in reference_dirs],
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--reference-dir", type=Path)
    reference_group.add_argument(
        "--reference-root",
        type=Path,
        help="Replay every captured run below this artifact root while retaining one Dense/VAE runtime.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("/hhb-data/aigc/model_zoo/lingbot/lingbot-video-dense-1.3b")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-text",
        action="store_true",
        help="Validate source prompt construction and Qwen3-VL embeddings instead of DiT/VAE replay.",
    )
    parser.add_argument(
        "--validate-ti2v-vae",
        action="store_true",
        help="Also compare the sampled TI2V VAE condition latent; requires the capture seed.",
    )
    parser.add_argument(
        "--assert-exact",
        action="store_true",
        help="Exit nonzero unless every recorded tensor has an exact shape and value match.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Fallback seed for captures created before seed metadata.")
    args = parser.parse_args()
    reference_dirs = (
        [args.reference_dir] if args.reference_dir is not None else discover_reference_dirs(args.reference_root)
    )
    if args.validate_text:
        reports = [
            replay_text(
                reference_dir,
                args.model_dir,
                validate_ti2v_vae=args.validate_ti2v_vae,
                seed=args.seed,
            )
            for reference_dir in reference_dirs
        ]
        report: dict[str, Any] = reports[0] if len(reports) == 1 else {"reports": reports}
    elif len(reference_dirs) == 1:
        report = replay(reference_dirs[0], args.model_dir)
    else:
        report = replay_many(reference_dirs, args.model_dir)
    if args.assert_exact:
        failures = exact_replay_failures(report)
        if failures:
            raise SystemExit(f"Exact replay gate failed: {', '.join(failures)}")
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
