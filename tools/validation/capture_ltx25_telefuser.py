"""Capture TeleFuser LTX-2.5 distilled T2V stage artifacts for golden comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

try:
    from ltx25_capture_utils import deterministic_audio_kernels
except ModuleNotFoundError:  # Supports runpy-based capture wrappers from the repository root.
    from tools.validation.ltx25_capture_utils import deterministic_audio_kernels

from telefuser.models.ltx25.checkpoint import LTX25ModelPaths, inspect_model_pack
from telefuser.models.ltx25.sampler import distilled_sigmas
from telefuser.pipelines.ltx25_distilled.reference import (
    LTX25DistilledReference,
    LTX25ReferenceImageCondition,
    LTX25ReferenceRequest,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tiling_config_metadata(tiling_config: Any) -> dict[str, dict[str, int]]:
    """Serialize resolved DiffVAE tile geometry for a reproducible capture manifest."""
    return {
        axis: {"tile_size": config.tile_size, "overlap": config.overlap}
        for axis, config in (
            ("frames", tiling_config.frames),
            ("height", tiling_config.height),
            ("width", tiling_config.width),
        )
    }


def _save_tensor(output_dir: Path, name: str, tensor: torch.Tensor) -> dict[str, Any]:
    path = output_dir / f"{name}.pt"
    value = tensor.detach().cpu().contiguous()
    torch.save(value, path)
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _diffvae_attention_backends(decoder: torch.nn.Module) -> list[str | None]:
    """Return the NATTEN backend identities used by a loaded DiffVAE decoder."""
    backends = {module.natten_backend for module in decoder.modules() if hasattr(module, "natten_backend")}
    return sorted(backends, key=lambda backend: backend or "")


def _capture_block_outputs(
    transformer: torch.nn.Module,
    output_dir: Path,
    artifacts: dict[str, Any],
    prefix: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Capture post-block video and audio states for one denoiser invocation."""
    velocity_model = getattr(transformer, "velocity_model")
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for index, block in enumerate(velocity_model.transformer_blocks):

        def save_output(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: tuple[Any, Any],
            *,
            block_index: int = index,
        ) -> None:
            video, audio = output
            if video is not None:
                name = f"{prefix}_block{block_index}_video"
                artifacts[name] = _save_tensor(output_dir, name, video.x)
            if audio is not None:
                name = f"{prefix}_block{block_index}_audio"
                artifacts[name] = _save_tensor(output_dir, name, audio.x)

        handles.append(block.register_forward_hook(save_output))
    return handles


def _release_modules(*modules: object, offload: str) -> None:
    """Release reference-path modules before loading the decoders."""
    if offload == "cpu":
        for module in modules:
            release = getattr(module, "release", None)
            if callable(release):
                release()
            elif isinstance(module, torch.nn.Module):
                module.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def _trajectories(artifacts: dict[str, Any], device: torch.device) -> dict[str, list[dict[str, Any]]]:
    """Build the upstream-compatible per-step diffusion index from saved tensors."""
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for stage_name, stage_number in (("stage1", 1), ("stage2", 2)):
        rows: list[dict[str, Any]] = []
        for step_index, sigma in enumerate(distilled_sigmas(stage_number, device=device)[:-1]):
            row: dict[str, Any] = {"index": step_index, "sigma": float(sigma)}
            for modality in ("video", "audio"):
                updated = f"{stage_name}_step{step_index}_updated_{modality}_latent"
                if updated in artifacts:
                    row[f"updated_{modality}_latent"] = updated
                ancestral_noise = f"{stage_name}_step{step_index}_ancestral_noise_{modality}"
                if ancestral_noise in artifacts:
                    row[f"ancestral_noise_{modality}"] = ancestral_noise
            rows.append(row)
        trajectories[stage_name] = rows
    return trajectories


@torch.inference_mode()
def main() -> None:
    """Run the isolated TeleFuser T2V reference path and save stage-boundary artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--frame-rate", type=float, default=24.0)
    parser.add_argument("--video-vae", choices=("diff", "conv"), default="diff")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-frame-index", type=int, default=0)
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--offload", choices=("none", "cpu"), default="cpu")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capture-prompt-intermediates", action="store_true")
    parser.add_argument("--capture-stage2-step2-blocks", action="store_true")
    parser.add_argument("--capture-decoded", action="store_true")
    parser.add_argument(
        "--deterministic-audio",
        action="store_true",
        help="Use deterministic CUDA kernels only while capturing decoded audio.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"LTX-2.5 conditioning image does not exist: {args.image}")
    images = (
        ()
        if args.image is None
        else (
            LTX25ReferenceImageCondition(
                Image.open(args.image).convert("RGB"),
                frame_idx=args.image_frame_index,
                strength=args.image_strength,
            ),
        )
    )
    request = LTX25ReferenceRequest(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=images,
    )
    runner = LTX25DistilledReference.from_model_root(
        str(args.model_root),
        device=args.device,
        video_vae=args.video_vae,
        capture_prompt_intermediates=args.capture_prompt_intermediates,
        offload=args.offload,
    )
    block_artifacts: dict[str, Any] = {}
    if args.capture_stage2_step2_blocks:
        transformer = runner.components.transformer
        original_forward = transformer.forward
        calls = 0

        def traced_forward(*forward_args: Any, **forward_kwargs: Any) -> Any:
            nonlocal calls
            handles: list[torch.utils.hooks.RemovableHandle] = []
            if calls == 10:
                handles = _capture_block_outputs(transformer, args.output_dir, block_artifacts, "stage2_step2")
            try:
                return original_forward(*forward_args, **forward_kwargs)
            finally:
                for handle in handles:
                    handle.remove()
                calls += 1

        transformer.forward = traced_forward  # type: ignore[method-assign]
        try:
            with torch.inference_mode():
                result = runner.generate(request)
        finally:
            transformer.forward = original_forward  # type: ignore[method-assign]
    else:
        with torch.inference_mode():
            result = runner.generate(request)

    decoded_artifacts: dict[str, Any] = {}
    resolved_video_tiling: dict[str, dict[str, int]] | None = None
    resolved_diffvae_attention_backends: list[str | None] | None = None
    resolved_diffvae_optimization: str | None = None
    if args.capture_decoded:
        from telefuser.models.ltx25 import (
            DiffusionVideoDecoder,
            LTX25ConvVideoVAE,
            load_ltx25_audio_decoder_and_vocoder,
        )

        _release_modules(
            runner.components.text_encoder,
            runner.components.embeddings_processor,
            runner.components.transformer,
            runner.components.spatial_upsampler,
            runner.components.latent_statistics,
            offload=args.offload,
        )
        decoder_generator = torch.Generator(device=runner.device)
        decoder_generator.set_state(result.decoder_generator_state)
        paths = LTX25ModelPaths.from_model_root(args.model_root)
        if args.video_vae == "diff":
            video_decoder = DiffusionVideoDecoder.from_checkpoint(
                paths.video_vae_path,
                device=runner.device,
                torch_dtype=runner.dtype,
            )
            tiling_config = video_decoder.recommended_tiling_config(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
            )
            resolved_video_tiling = _tiling_config_metadata(tiling_config)
            resolved_diffvae_attention_backends = _diffvae_attention_backends(video_decoder)
            resolved_diffvae_optimization = "chunked_compile" if video_decoder.mark_dynamic_shapes else "chunked_eager"
            decoded_rgb = torch.cat(
                list(
                    video_decoder.decode_video(
                        result.stage2_video.latent, tiling_config=tiling_config, generator=decoder_generator
                    )
                ),
                dim=0,
            )
        else:
            video_decoder = LTX25ConvVideoVAE.from_checkpoint(
                paths.conv_video_vae_path,
                device=runner.device,
                torch_dtype=runner.dtype,
            )
            decoded_rgb = torch.cat(
                [
                    chunk[0].permute(1, 2, 3, 0).add(1).mul(0.5).clamp(0, 1)
                    for chunk in video_decoder.decode(result.stage2_video.latent, generator=decoder_generator)
                ],
                dim=0,
            )
        decoded_artifacts["video_decoder_generator_state"] = _save_tensor(
            args.output_dir, "video_decoder_generator_state", result.decoder_generator_state
        )
        decoded_artifacts["video_decoder_input_latent"] = _save_tensor(
            args.output_dir, "video_decoder_input_latent", result.stage2_video.latent
        )
        decoded_artifacts["decoded_rgb"] = _save_tensor(args.output_dir, "decoded_rgb", decoded_rgb)
        _release_modules(video_decoder, offload=args.offload)

        audio_decoder, vocoder = load_ltx25_audio_decoder_and_vocoder(
            paths.audio_vae_path,
            device=runner.device,
            torch_dtype=runner.dtype,
        )
        with deterministic_audio_kernels(args.deterministic_audio):
            waveform = vocoder(audio_decoder(result.stage2_audio.latent)).squeeze(0).float()
        decoded_artifacts["decoded_waveform"] = _save_tensor(args.output_dir, "decoded_waveform", waveform)
        _release_modules(audio_decoder, vocoder, offload=args.offload)

    artifacts = {
        name: _save_tensor(args.output_dir, name, value) for name, value in sorted(result.trace.artifacts.items())
    }
    artifacts.update(block_artifacts)
    artifacts.update(decoded_artifacts)
    artifacts["video_context"] = _save_tensor(args.output_dir, "video_context", result.trace.video_context)
    artifacts["audio_context"] = _save_tensor(args.output_dir, "audio_context", result.trace.audio_context)
    artifacts["context_attention_mask"] = _save_tensor(
        args.output_dir, "context_attention_mask", result.trace.context_attention_mask
    )
    manifest = {
        "runtime": {
            "torch": torch.__version__,
            "device": str(args.device),
            "dtype": "bfloat16",
            "deterministic_audio": args.deterministic_audio,
            "tiling": "auto",
            "resolved_video_tiling": resolved_video_tiling,
            "resolved_diffvae_attention_backends": resolved_diffvae_attention_backends,
            "resolved_diffvae_optimization": resolved_diffvae_optimization,
        },
        "request": {
            "prompt": request.prompt,
            "seed": request.seed,
            "height": request.height,
            "width": request.width,
            "num_frames": request.num_frames,
            "resolved_num_frames": request.num_frames,
            "frame_rate": request.frame_rate,
            "video_vae": args.video_vae,
            "offload": args.offload,
            "dtype": "bfloat16",
            "prompt_normalization": [request.prompt.strip()],
            "image": (
                None
                if args.image is None
                else {
                    "path": str(args.image.resolve()),
                    "sha256": _sha256(args.image),
                    "frame_index": args.image_frame_index,
                    "strength": args.image_strength,
                }
            ),
        },
        "checkpoints": {
            name: metadata.as_dict()
            for name, metadata in inspect_model_pack(args.model_root, include_sha256=True).items()
        },
        "audio": {"sample_rate": 48000},
        "artifacts": artifacts,
        "trajectories": _trajectories(artifacts, runner.device),
    }
    (args.output_dir / "capture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
