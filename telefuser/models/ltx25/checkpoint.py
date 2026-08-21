"""LTX-2.5 split-checkpoint discovery and metadata validation.

This module owns the LTX-2.5 file layout.  It is intentionally independent from
the legacy monolithic LTX checkpoint loader and does not instantiate model
classes; architecture construction must consume the returned metadata rather
than copy constants from LTX-2.3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

LTX25_COMPONENT_NAMES = (
    "ltx25_transformer",
    "ltx25_gemma4",
    "ltx25_embeddings_processor",
    "ltx25_video_encoder",
    "ltx25_video_decoder",
    "ltx25_audio_decoder",
    "ltx25_vocoder",
    "ltx25_spatial_upsampler",
    "ltx25_duration_head",
)

_DEFAULT_PATHS = {
    "transformer_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae_path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "conv_video_vae_path": "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler_path": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "duration_head_path": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
}


def parse_model_version(version: str | None) -> tuple[int, ...]:
    """Parse an LTX metadata version into comparable numeric components."""
    if not version:
        return ()
    values: list[int] = []
    for part in version.replace("-", ".").split("."):
        if not part.isdigit():
            break
        values.append(int(part))
    return tuple(values)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"LTX-2.5 {label} checkpoint does not exist: {path}")
    return path


@dataclass(frozen=True, slots=True)
class LTX25ModelPaths:
    """Resolved LTX-2.5 split-checkpoint paths."""

    transformer_path: Path
    text_encoder_path: Path
    video_vae_path: Path
    conv_video_vae_path: Path
    audio_vae_path: Path
    spatial_upsampler_path: Path
    duration_head_path: Path

    @classmethod
    def from_model_root(cls, model_root: str | Path) -> "LTX25ModelPaths":
        """Resolve the official BF16 LTX-2.5 distilled model-pack layout."""
        root = Path(model_root).expanduser().resolve()
        paths = {
            name: _require_file(root / relative_path, name.removesuffix("_path"))
            for name, relative_path in _DEFAULT_PATHS.items()
        }
        return cls(**paths)

    def as_dict(self) -> dict[str, str]:
        """Return absolute paths in a JSON-compatible representation."""
        return {name: str(path) for name, path in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class LTX25CheckpointMetadata:
    """Provenance and unmodified safetensors metadata for one component."""

    path: Path
    size_bytes: int
    sha256: str | None
    tensor_count: int
    metadata: dict[str, Any]
    config: dict[str, Any]
    model_version: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "tensor_count": self.tensor_count,
            "metadata": self.metadata,
            "config": self.config,
            "model_version": list(self.model_version),
        }


def _decode_metadata_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a checkpoint without loading tensor payloads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_checkpoint(path: str | Path, *, include_sha256: bool = False) -> LTX25CheckpointMetadata:
    """Read metadata and key count without materializing checkpoint tensors."""
    resolved = _require_file(Path(path).expanduser().resolve(), "component")
    with safe_open(resolved, framework="pt", device="cpu") as checkpoint:
        raw_metadata = checkpoint.metadata() or {}
        metadata = {key: _decode_metadata_value(value) for key, value in raw_metadata.items()}
        tensor_count = len(checkpoint.keys())
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        raise ValueError(f"LTX-2.5 checkpoint config metadata must be an object: {resolved}")
    metadata_version = metadata.get("model_version")
    model_version = parse_model_version(metadata_version if isinstance(metadata_version, str) else None)
    return LTX25CheckpointMetadata(
        path=resolved,
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved) if include_sha256 else None,
        tensor_count=tensor_count,
        metadata=metadata,
        config=config,
        model_version=model_version,
    )


def validate_gemma_source_checkpoint(
    transformer_metadata: LTX25CheckpointMetadata,
    gemma_config: dict[str, Any],
) -> None:
    """Validate the Gemma4 version declared by an LTX-2.5 transformer checkpoint."""
    source = transformer_metadata.metadata.get("gemma_source_checkpoint")
    if not isinstance(source, dict):
        raise ValueError(
            f"LTX-2.5 transformer {transformer_metadata.path} is missing gemma_source_checkpoint metadata."
        )
    expected = source.get("gemma_version")
    actual = gemma_config.get("gemma_version")
    if expected != actual:
        raise ValueError(
            "Gemma version mismatch: transformer metadata expects "
            f"gemma_version={expected!r}, but the Gemma config declares {actual!r}."
        )


def inspect_model_pack(model_root: str | Path, *, include_sha256: bool = False) -> dict[str, LTX25CheckpointMetadata]:
    """Inspect every checkpoint required by the LTX-2.5 distilled reference path."""
    paths = LTX25ModelPaths.from_model_root(model_root)
    return {
        name.removesuffix("_path"): inspect_checkpoint(path, include_sha256=include_sha256)
        for name, path in asdict(paths).items()
    }
