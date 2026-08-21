"""Capture LTX-2.5 distilled upstream stage-boundary Golden artifacts.

Run this tool with the pinned upstream source on ``PYTHONPATH``.  The reference
runtime must use the same PyTorch and Transformers versions as TeleFuser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

try:
    from ltx25_capture_utils import deterministic_audio_kernels, mp4_container_metadata
except ModuleNotFoundError:  # Supports runpy-based capture wrappers from the repository root.
    from tools.validation.ltx25_capture_utils import deterministic_audio_kernels, mp4_container_metadata

from telefuser.models.ltx25.checkpoint import inspect_model_pack


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_package_version(package: str) -> str | None:
    """Return an installed package version without making it a capture dependency."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None


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


class _StageRecorder:
    """Record initial, per-step, and final upstream diffusion-stage tensors."""

    def __init__(
        self,
        stage: Any,
        output_dir: Path,
        artifacts: dict[str, Any],
        trajectories: dict[str, list[dict[str, Any]]],
        capture_stage2_step2_blocks: bool,
    ) -> None:
        self._stage = stage
        self._output_dir = output_dir
        self._artifacts = artifacts
        self._trajectories = trajectories
        self._capture_stage2_step2_blocks = capture_stage2_step2_blocks
        self._calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        from ltx_core.components.diffusion_steps import EulerDiffusionStep  # type: ignore[import-not-found]

        stage_name = f"stage{self._calls + 1}"
        original_denoiser = kwargs["denoiser"]
        stepper = kwargs.get("stepper") or EulerDiffusionStep()
        kwargs["stepper"] = stepper
        trajectories = self._trajectories.setdefault(stage_name, [])

        def record_denoiser(
            transformer: Any,
            video_state: Any,
            audio_state: Any,
            sigmas: torch.Tensor,
            step_index: int,
        ) -> tuple[Any, Any]:
            if step_index == 0:
                if video_state is not None:
                    self._artifacts[f"{stage_name}_initial_video_noise"] = _save_tensor(
                        self._output_dir, f"{stage_name}_initial_video_noise", video_state.latent
                    )
                if audio_state is not None:
                    self._artifacts[f"{stage_name}_initial_audio_noise"] = _save_tensor(
                        self._output_dir, f"{stage_name}_initial_audio_noise", audio_state.latent
                    )
            handles: list[torch.utils.hooks.RemovableHandle] = []
            if self._capture_stage2_step2_blocks and stage_name == "stage2" and step_index == 2:
                handles = _capture_block_outputs(transformer, self._output_dir, self._artifacts, "stage2_step2")
            try:
                video_result, audio_result = original_denoiser(
                    transformer, video_state, audio_state, sigmas, step_index
                )
            finally:
                for handle in handles:
                    handle.remove()
            row: dict[str, Any] = {"index": step_index, "sigma": float(sigmas[step_index])}
            if video_result is not None and video_result.denoised is not None:
                artifact = f"{stage_name}_step{step_index}_video_x0"
                self._artifacts[artifact] = _save_tensor(self._output_dir, artifact, video_result.denoised)
            if audio_result is not None and audio_result.denoised is not None:
                artifact = f"{stage_name}_step{step_index}_audio_x0"
                self._artifacts[artifact] = _save_tensor(self._output_dir, artifact, audio_result.denoised)
            trajectories.append(row)
            return video_result, audio_result

        original_step = stepper.step

        def record_step(*step_args: Any, **step_kwargs: Any) -> torch.Tensor:
            step_index = step_kwargs["step_index"] if "step_index" in step_kwargs else step_args[3]
            noise = step_kwargs.get("noise")
            if noise is not None:
                # The upstream loop draws video before audio from one generator. Recording here
                # preserves both the actual tensor and that call order without changing RNG.
                modality = "video" if "ancestral_noise_video" not in trajectories[step_index] else "audio"
                artifact = f"{stage_name}_step{step_index}_ancestral_noise_{modality}"
                self._artifacts[artifact] = _save_tensor(self._output_dir, artifact, noise)
                trajectories[step_index][f"ancestral_noise_{modality}"] = artifact
            updated = original_step(*step_args, **step_kwargs)
            modality = "video" if "updated_video_latent" not in trajectories[step_index] else "audio"
            artifact = f"{stage_name}_step{step_index}_updated_{modality}_latent"
            self._artifacts[artifact] = _save_tensor(self._output_dir, artifact, updated)
            trajectories[step_index][f"updated_{modality}_latent"] = artifact
            return updated

        stepper.step = record_step
        kwargs["denoiser"] = record_denoiser
        video, audio = self._stage(*args, **kwargs)
        self._artifacts[f"{stage_name}_video_latent"] = _save_tensor(
            self._output_dir, f"{stage_name}_video_latent", video.latent
        )
        self._artifacts[f"{stage_name}_audio_latent"] = _save_tensor(
            self._output_dir, f"{stage_name}_audio_latent", audio.latent
        )
        self._calls += 1
        return video, audio

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stage, name)


class _UpsamplerRecorder:
    """Record the exact upsampler input and output used between denoising stages."""

    def __init__(self, upsampler: Any, output_dir: Path, artifacts: dict[str, Any]) -> None:
        self._upsampler = upsampler
        self._output_dir = output_dir
        self._artifacts = artifacts

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        self._artifacts["upsampler_input"] = _save_tensor(self._output_dir, "upsampler_input", latent)
        output = self._upsampler(latent)
        self._artifacts["upsampler_output"] = _save_tensor(self._output_dir, "upsampler_output", output)
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._upsampler, name)


class _VideoDecoderRecorder:
    """Record the exact generator state entering the upstream video decoder."""

    def __init__(self, decoder: Any, output_dir: Path, artifacts: dict[str, Any]) -> None:
        self._decoder = decoder
        self._output_dir = output_dir
        self._artifacts = artifacts

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        latent = kwargs.get("latent") if "latent" in kwargs else args[0]
        if isinstance(latent, torch.Tensor):
            self._artifacts["video_decoder_input_latent"] = _save_tensor(
                self._output_dir, "video_decoder_input_latent", latent
            )
        generator = kwargs.get("generator")
        if generator is None and len(args) >= 3:
            generator = args[2]
        if generator is not None:
            self._artifacts["video_decoder_generator_state"] = _save_tensor(
                self._output_dir,
                "video_decoder_generator_state",
                generator.get_state(),
            )
        return self._decoder(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._decoder, name)


class _PromptRecorder:
    """Record raw Gemma inputs/states and final connectors without changing lifecycle."""

    def __init__(self, encoder: Any, output_dir: Path, artifacts: dict[str, Any]) -> None:
        self._encoder = encoder
        self._output_dir = output_dir
        self._artifacts = artifacts
        self.normalized_prompts: list[str] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from ltx_core.text_encoders.gemma.embeddings_processor import (
            EmbeddingsProcessor,  # type: ignore[import-not-found]
        )
        from ltx_core.text_encoders.gemma.encoders.base_encoder import (
            LTXGemmaTextEncoder,  # type: ignore[import-not-found]
        )

        original_encode = LTXGemmaTextEncoder.encode
        original_process_hidden_states = EmbeddingsProcessor.process_hidden_states

        def record_encode(text_encoder: Any, prompts: list[str]) -> Any:
            if len(prompts) != 1:
                raise ValueError(f"Golden capture expects one prompt, got {len(prompts)}")
            self.normalized_prompts = [prompt.strip() for prompt in prompts]
            pairs = text_encoder.tokenizer.tokenize_with_weights(prompts[0])["gemma"]
            token_ids = torch.tensor([[token for token, _ in pairs]], device=text_encoder.model.device)
            raw_mask = torch.tensor([[weight for _, weight in pairs]], device=text_encoder.model.device)
            self._artifacts["gemma_token_ids"] = _save_tensor(self._output_dir, "gemma_token_ids", token_ids)
            self._artifacts["gemma_attention_mask"] = _save_tensor(self._output_dir, "gemma_attention_mask", raw_mask)
            raw_outputs = original_encode(text_encoder, prompts)
            hidden_states, attention_mask = raw_outputs[0]
            if not torch.equal(raw_mask, attention_mask):
                raise ValueError("Golden capture token mask differs from the encoder's raw attention mask")
            for index, hidden_state in enumerate(hidden_states):
                artifact = f"gemma_hidden_state_{index}"
                self._artifacts[artifact] = _save_tensor(self._output_dir, artifact, hidden_state)
            return raw_outputs

        def record_process_hidden_states(
            processor: Any,
            hidden_states: tuple[torch.Tensor, ...],
            attention_mask: torch.Tensor,
            padding_side: str = "left",
        ) -> Any:
            video_features, audio_features = processor.feature_extractor(hidden_states, attention_mask, padding_side)
            self._artifacts["video_features"] = _save_tensor(self._output_dir, "video_features", video_features)
            self._artifacts["audio_features"] = _save_tensor(self._output_dir, "audio_features", audio_features)
            return original_process_hidden_states(processor, hidden_states, attention_mask, padding_side)

        LTXGemmaTextEncoder.encode = record_encode
        EmbeddingsProcessor.process_hidden_states = record_process_hidden_states
        try:
            outputs = self._encoder(*args, **kwargs)
        finally:
            LTXGemmaTextEncoder.encode = original_encode
            EmbeddingsProcessor.process_hidden_states = original_process_hidden_states
        if len(outputs) != 1:
            raise ValueError(f"Golden capture expects one prompt, got {len(outputs)}")
        output = outputs[0]
        self._artifacts["video_context"] = _save_tensor(self._output_dir, "video_context", output.video_encoding)
        if output.audio_encoding is not None:
            self._artifacts["audio_context"] = _save_tensor(self._output_dir, "audio_context", output.audio_encoding)
        self._artifacts["context_attention_mask"] = _save_tensor(
            self._output_dir, "context_attention_mask", output.attention_mask
        )
        return outputs

    def __getattr__(self, name: str) -> Any:
        return getattr(self._encoder, name)


def _upstream_commit(upstream_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    """Run one upstream T2V case and persist uncompressed reference artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
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
    parser.add_argument("--offload", choices=("cpu", "none"), default="cpu")
    parser.add_argument(
        "--attention-backend",
        choices=("automatic", "pytorch"),
        default="automatic",
        help="Pin upstream transformer attention for a matched fidelity capture.",
    )
    parser.add_argument("--capture-stage2-step2-blocks", action="store_true")
    parser.add_argument(
        "--diffvae-optimization",
        choices=("chunked_eager", "chunked_compile"),
        default="chunked_eager",
        help="Select the upstream DiffVAE decoder recipe for a matched capture.",
    )
    parser.add_argument(
        "--diffvae-natten-backend",
        choices=("automatic", "cutlass-fna"),
        default="automatic",
        help="Pin upstream DiffVAE NATTEN for a matched capture.",
    )
    parser.add_argument(
        "--deterministic-audio",
        action="store_true",
        help="Use deterministic CUDA kernels only while capturing decoded audio.",
    )
    args = parser.parse_args()

    # Upstream imports remain local to this entry point so ordinary TeleFuser imports never depend on it.
    import ltx_pipelines.utils.blocks as pipeline_blocks  # type: ignore[import-not-found]
    from ltx_core.loader.module_ops import ModuleOps  # type: ignore[import-not-found]
    from ltx_core.model.video_vae.transformer.compiling import (
        configure_natten_backend,  # type: ignore[import-not-found]
    )
    from ltx_core.model.video_vae.transformer.config import DiffVAEMode  # type: ignore[import-not-found]
    from ltx_pipelines.distilled import DistilledPipeline  # type: ignore[import-not-found]
    from ltx_pipelines.utils.args import ImageConditioningInput  # type: ignore[import-not-found]
    from ltx_pipelines.utils.media_io.encode import encode_video  # type: ignore[import-not-found]
    from ltx_pipelines.utils.model_paths import ModelPaths  # type: ignore[import-not-found]
    from ltx_pipelines.utils.types import OffloadMode  # type: ignore[import-not-found]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"LTX-2.5 conditioning image does not exist: {args.image}")
    artifacts: dict[str, Any] = {}
    trajectories: dict[str, list[dict[str, Any]]] = {}
    paths = ModelPaths.from_split(
        transformer_path=str(args.model_root / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        text_encoder_path=str(args.model_root / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        video_vae_path=str(
            args.model_root
            / "vae"
            / (
                "ltx-2.5-video-vae-bf16.safetensors"
                if args.video_vae == "diff"
                else "ltx-2.5-video-vae-conv-bf16.safetensors"
            )
        ),
        audio_vae_path=str(args.model_root / "vae/ltx-2.5-audio-vae-bf16.safetensors"),
        duration_head_path=str(args.model_root / "model_patches/ltx-2.5-duration-head-bf16.safetensors"),
    )
    pipeline = DistilledPipeline(
        model_paths=paths,
        spatial_upsampler_path=str(
            args.model_root / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
        ),
        loras=(),
        device=torch.device("cuda"),
        offload_mode=OffloadMode(args.offload),
        diffvae_optimization=DiffVAEMode(args.diffvae_optimization),
    )
    if args.attention_backend == "pytorch":
        from ltx_core.model.transformer.attention import AttentionFunction  # type: ignore[import-not-found]

        pipeline.stage = pipeline.stage.with_attention(AttentionFunction.PYTORCH)
    prompt_recorder = _PromptRecorder(pipeline.prompt_encoder, args.output_dir, artifacts)
    pipeline.prompt_encoder = prompt_recorder
    pipeline.stage = _StageRecorder(
        pipeline.stage,
        args.output_dir,
        artifacts,
        trajectories,
        args.capture_stage2_step2_blocks,
    )
    if args.diffvae_natten_backend == "cutlass-fna":
        decoder_builder = pipeline.video_decoder._decoder_builder

        def pin_cutlass_fna(model: torch.nn.Module) -> torch.nn.Module:
            configure_natten_backend(model, "cutlass-fna")
            return model

        pipeline.video_decoder._decoder_builder = decoder_builder.with_module_ops(
            (*decoder_builder.module_ops, ModuleOps("capture_cutlass_fna", lambda _model: True, pin_cutlass_fna))
        )
        resolved_diffvae_attention_backends: list[str | None] = ["cutlass-fna"]
    else:
        resolved_diffvae_attention_backends = [None]
    pipeline.upsampler = _UpsamplerRecorder(pipeline.upsampler, args.output_dir, artifacts)
    pipeline.video_decoder = _VideoDecoderRecorder(pipeline.video_decoder, args.output_dir, artifacts)
    original_decode_audio = pipeline_blocks.vae_decode_audio

    def decode_audio(*decode_args: Any, **decode_kwargs: Any) -> Any:
        with deterministic_audio_kernels(args.deterministic_audio):
            return original_decode_audio(*decode_args, **decode_kwargs)

    pipeline_blocks.vae_decode_audio = decode_audio
    try:
        with torch.inference_mode():
            images = (
                []
                if args.image is None
                else [
                    ImageConditioningInput(
                        path=str(args.image.resolve()),
                        frame_idx=args.image_frame_index,
                        strength=args.image_strength,
                    )
                ]
            )
            video, audio, resolved_frames, resolved_tiling = pipeline(
                prompt=args.prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                frame_rate=args.frame_rate,
                images=images,
                num_frames=args.num_frames,
            )
            rgb = torch.cat(list(video), dim=0)
    finally:
        pipeline_blocks.vae_decode_audio = original_decode_audio
    artifacts["decoded_rgb"] = _save_tensor(args.output_dir, "decoded_rgb", rgb)
    artifacts["decoded_waveform"] = _save_tensor(args.output_dir, "decoded_waveform", audio.waveform)
    mp4_path = args.output_dir / "decoded.mp4"
    encode_video(rgb, int(args.frame_rate), audio, str(mp4_path), video_chunks_number=1)
    components = inspect_model_pack(args.model_root, include_sha256=True)
    manifest = {
        "upstream_commit": _upstream_commit(args.upstream_root),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "natten": _optional_package_version("natten"),
            "attention_backend": args.attention_backend,
            "deterministic_audio": args.deterministic_audio,
            "compile": False,
            "tiling": "auto",
            "resolved_video_tiling": None if resolved_tiling is None else asdict(resolved_tiling),
            "diffvae_optimization": args.diffvae_optimization,
            "resolved_diffvae_attention_backends": resolved_diffvae_attention_backends,
        },
        "request": {
            "prompt": args.prompt,
            "seed": args.seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "resolved_num_frames": resolved_frames,
            "frame_rate": args.frame_rate,
            "video_vae": args.video_vae,
            "offload": args.offload,
            "dtype": "bfloat16",
            "prompt_normalization": prompt_recorder.normalized_prompts,
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
        "checkpoints": {name: value.as_dict() for name, value in components.items()},
        "audio": {"sample_rate": audio.sampling_rate},
        "container": mp4_container_metadata(mp4_path),
        "artifacts": artifacts,
        "trajectories": trajectories,
    }
    manifest_path = args.output_dir / "capture_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "artifacts": sorted(artifacts)}, sort_keys=True))


if __name__ == "__main__":
    main()
