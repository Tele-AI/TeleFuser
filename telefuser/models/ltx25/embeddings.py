"""LTX-2.5 Gemma feature extraction and audio/video embeddings connectors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors import safe_open

from telefuser.core.config import AttentionConfig, AttnImplType

from .checkpoint import inspect_checkpoint
from .gemma4 import LTX25GemmaAssets
from .transformer import (
    Attention,
    FeedForward,
    LTXRopeType,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
    rms_norm,
)

_TRANSFORMER_PREFIX = "model.diffusion_model."
_VIDEO_CONNECTOR_PREFIX = _TRANSFORMER_PREFIX + "video_embeddings_connector."
_AUDIO_CONNECTOR_PREFIX = _TRANSFORMER_PREFIX + "audio_embeddings_connector."
_TEXT_PROJECTION_PREFIX = "text_embedding_projection."


class LTX25EmbeddingsProcessorOutput(NamedTuple):
    """Conditioning tensors consumed by the video and audio diffusion paths."""

    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


def _right_pad_order(additive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    binary = (additive_mask[:, 0, 0, :] >= 0).to(torch.int32)
    indices = torch.argsort(binary, dim=-1, descending=True, stable=True)
    ordered = torch.gather(binary, 1, indices)
    mask = (ordered.to(additive_mask.dtype) - 1) * torch.finfo(additive_mask.dtype).max
    return indices, mask[:, None, None, :]


def _apply_order(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(features, 1, indices.unsqueeze(-1).expand_as(features))


def _additive_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (mask.to(torch.int64) - 1).to(dtype).reshape(mask.shape[0], 1, 1, mask.shape[-1]) * torch.finfo(dtype).max


class LTX25FeatureExtractor(torch.nn.Module):
    """LTX-2.5 per-token Gemma RMS feature extraction and dual projections."""

    def __init__(self, hidden_size: int, num_hidden_layers: int, video_dim: int, audio_dim: int) -> None:
        super().__init__()
        self.embedding_dim = hidden_size
        self.flat_dim = hidden_size * (num_hidden_layers + 1)
        self.video_aggregate_embed = torch.nn.Linear(self.flat_dim, video_dim, bias=True)
        self.audio_aggregate_embed = torch.nn.Linear(self.flat_dim, audio_dim, bias=True)

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.stack(hidden_states, dim=-1)
        variance = torch.mean(encoded.square(), dim=2, keepdim=True)
        normalized = encoded * torch.rsqrt(variance + 1e-6)
        normalized = normalized.reshape(encoded.shape[0], encoded.shape[1], -1).to(encoded.dtype)
        normalized = torch.where(attention_mask.bool().unsqueeze(-1), normalized, torch.zeros_like(normalized))
        video = self.video_aggregate_embed(
            normalized * math.sqrt(self.video_aggregate_embed.out_features / self.embedding_dim)
        )
        audio = self.audio_aggregate_embed(
            normalized * math.sqrt(self.audio_aggregate_embed.out_features / self.embedding_dim)
        )
        return video, audio


class _LTX25ConnectorBlock(torch.nn.Module):
    """Pre-norm RoPE attention and feed-forward block used by each connector."""

    def __init__(self, dim: int, heads: int, dim_head: int, rope_type: LTXRopeType, gated_attention: bool) -> None:
        super().__init__()
        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=gated_attention,
        )
        # Connector feed-forward layers retain the checkpoint's biases.
        self.ff = FeedForward(dim, dim_out=dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor,
        positional_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(
            rms_norm(hidden_states), mask=additive_attention_mask, pe=positional_embeddings
        )
        return hidden_states + self.ff(rms_norm(hidden_states))


class LTX25EmbeddingsConnector(torch.nn.Module):
    """LTX-2.5 learned-register, 1D text-conditioning connector."""

    def __init__(
        self,
        *,
        attention_head_dim: int,
        num_attention_heads: int,
        num_layers: int,
        positional_embedding_max_pos: list[int],
        rope_type: LTXRopeType,
        double_precision_rope: bool,
        apply_gated_attention: bool,
        num_learnable_registers: int,
    ) -> None:
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.num_learnable_registers = num_learnable_registers
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _LTX25ConnectorBlock(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    rope_type,
                    apply_gated_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.learnable_registers = torch.nn.Parameter(torch.empty(num_learnable_registers, self.inner_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = hidden_states.shape
        if sequence_length % self.num_learnable_registers:
            raise ValueError(
                "LTX-2.5 connector token length must be divisible by its learnable-register count: "
                f"{sequence_length} % {self.num_learnable_registers}"
            )
        registers = self.learnable_registers.repeat(sequence_length // self.num_learnable_registers, 1)
        registers = registers.to(hidden_states).unsqueeze(0).expand(batch_size, -1, -1)
        valid = (additive_attention_mask[:, 0, 0, :].unsqueeze(-1) >= 0).to(hidden_states.dtype)
        hidden_states = valid * hidden_states + (1 - valid) * registers
        connector_mask = torch.zeros_like(additive_attention_mask)
        indices = torch.arange(sequence_length, dtype=torch.float32, device=hidden_states.device)
        indices = indices[None, None, :].expand(batch_size, -1, -1)
        generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        positional_embeddings = precompute_freqs_cis(
            indices,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=10000.0,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=generator,
        )
        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, connector_mask, positional_embeddings)
        return rms_norm(hidden_states), connector_mask


class LTX25EmbeddingsProcessor(torch.nn.Module):
    """Feature extraction and dual connectors loaded from LTX-2.5 split checkpoints."""

    def __init__(self, transformer_config: dict[str, Any], gemma_config: dict[str, Any]) -> None:
        super().__init__()
        text_config = gemma_config.get("text_config")
        if not isinstance(text_config, dict):
            raise ValueError("LTX-2.5 Gemma config is missing object text_config")
        hidden_size = text_config.get("hidden_size")
        num_hidden_layers = text_config.get("num_hidden_layers")
        if not isinstance(hidden_size, int) or not isinstance(num_hidden_layers, int):
            raise ValueError("LTX-2.5 Gemma text_config is missing hidden_size or num_hidden_layers")
        rope_type = LTXRopeType(transformer_config["rope_type"])
        double_precision_rope = transformer_config.get("frequencies_precision") == "float64"
        common = {
            "num_layers": transformer_config["connector_num_layers"],
            "positional_embedding_max_pos": transformer_config["connector_positional_embedding_max_pos"],
            "rope_type": rope_type,
            "double_precision_rope": double_precision_rope,
            "apply_gated_attention": transformer_config["connector_apply_gated_attention"],
            "num_learnable_registers": transformer_config["connector_num_learnable_registers"],
        }
        video_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
        audio_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
        self.feature_extractor = LTX25FeatureExtractor(hidden_size, num_hidden_layers, video_dim, audio_dim)
        self.video_connector = LTX25EmbeddingsConnector(
            attention_head_dim=transformer_config["connector_attention_head_dim"],
            num_attention_heads=transformer_config["connector_num_attention_heads"],
            **common,
        )
        self.audio_connector = LTX25EmbeddingsConnector(
            attention_head_dim=transformer_config["audio_connector_attention_head_dim"],
            num_attention_heads=transformer_config["audio_connector_num_attention_heads"],
            **common,
        )

    def forward(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> LTX25EmbeddingsProcessorOutput:
        # Upstream connector blocks use PyTorch's native SDPA priority path.
        # Keep that exact baseline locally while leaving the denoiser's public
        # attention dispatch configurable by its runtime configuration.
        native_attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
        connector_attentions = [
            *(block.attn1 for block in self.video_connector.transformer_1d_blocks),
            *(block.attn1 for block in self.audio_connector.transformer_1d_blocks),
        ]
        original_attention_configs = [attention.attention_config for attention in connector_attentions]
        for attention in connector_attentions:
            attention.attention_config = native_attention_config
        if any(attention.attention_config is not native_attention_config for attention in connector_attentions):
            raise RuntimeError("LTX-2.5 connector attention configuration was not applied")
        try:
            video_features, audio_features = self.feature_extractor(hidden_states, attention_mask)
            additive_mask = _additive_mask(attention_mask, video_features.dtype)
            order, connector_mask = _right_pad_order(additive_mask)
            video_encoding, output_mask = self.video_connector(_apply_order(video_features, order), connector_mask)
            audio_encoding, _ = self.audio_connector(_apply_order(audio_features, order), connector_mask)
            binary_mask = (output_mask[:, 0, 0, :] >= 0).to(torch.int64)
            return LTX25EmbeddingsProcessorOutput(
                video_encoding * binary_mask.unsqueeze(-1), audio_encoding, binary_mask
            )
        finally:
            for attention, original_attention_config in zip(
                connector_attentions, original_attention_configs, strict=True
            ):
                attention.attention_config = original_attention_config

    @classmethod
    def from_checkpoints(
        cls,
        transformer_path: str | Path,
        text_encoder_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LTX25EmbeddingsProcessor":
        """Construct and strictly load connector weights from the official split files."""
        transformer = inspect_checkpoint(transformer_path)
        assets = LTX25GemmaAssets.load(text_encoder_path)
        transformer_config = transformer.config.get("transformer")
        if not isinstance(transformer_config, dict):
            raise ValueError("LTX-2.5 transformer checkpoint is missing object transformer config")
        with torch.device("meta"):
            model = cls(transformer_config, assets.config)
        unexpected, missing = embeddings_checkpoint_key_coverage(
            transformer.path, assets.checkpoint_path, set(model.state_dict())
        )
        if unexpected or missing:
            raise ValueError(
                "LTX-2.5 embeddings checkpoint coverage mismatch: "
                f"unexpected={sorted(unexpected)[:5]}, missing={sorted(missing)[:5]}"
            )
        state_dict = _load_embedding_state_dict(transformer.path, assets.checkpoint_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=True, assign=True)
        if missing or unexpected:
            raise ValueError(f"LTX-2.5 embeddings load mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        return model.to(device=device, dtype=torch_dtype).eval()


def _embedding_target_key(path: Path, key: str) -> str | None:
    if path.name.startswith("ltx-2.5-22b"):
        if key.startswith(_VIDEO_CONNECTOR_PREFIX):
            return "video_connector." + key.removeprefix(_VIDEO_CONNECTOR_PREFIX)
        if key.startswith(_AUDIO_CONNECTOR_PREFIX):
            return "audio_connector." + key.removeprefix(_AUDIO_CONNECTOR_PREFIX)
    if key.startswith(_TEXT_PROJECTION_PREFIX):
        return "feature_extractor." + key.removeprefix(_TEXT_PROJECTION_PREFIX)
    return None


def _load_embedding_state_dict(transformer_path: Path, text_encoder_path: Path) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    for path in (transformer_path, text_encoder_path):
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                target = _embedding_target_key(path, key)
                if target is not None:
                    state_dict[target] = checkpoint.get_tensor(key)
    return state_dict


def embeddings_checkpoint_key_coverage(
    transformer_path: str | Path,
    text_encoder_path: str | Path,
    model_keys: set[str],
) -> tuple[set[str], set[str]]:
    """Return unexplained source keys and missing model keys without materializing weights."""
    mapped: set[str] = set()
    for path in (Path(transformer_path), Path(text_encoder_path)):
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            mapped.update(
                target for key in checkpoint.keys() if (target := _embedding_target_key(path, key)) is not None
            )
    return mapped - model_keys, model_keys - mapped
