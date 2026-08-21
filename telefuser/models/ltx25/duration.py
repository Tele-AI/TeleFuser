"""LTX-2.5 DurationHead checkpoint loader and frame-grid resolution."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .checkpoint import inspect_checkpoint


class LTX25AttentionPooler(nn.Module):
    """Learned-query cross-attention pooler used by the DurationHead."""

    def __init__(self, hidden_dim: int, num_queries: int, num_heads: int) -> None:
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_queries, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        return pooled


class LTX25DurationHead(nn.Module):
    """Predict a duration in seconds from LTX-2.5 connector output tokens."""

    def __init__(
        self,
        *,
        video_cross_attention_dim: int,
        audio_cross_attention_dim: int,
        pooler_hidden_dim: int = 256,
        num_queries: int = 1,
        num_pooler_heads: int = 4,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.video_input_proj = nn.Linear(video_cross_attention_dim, pooler_hidden_dim)
        self.video_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)
        self.audio_input_proj = nn.Linear(audio_cross_attention_dim, pooler_hidden_dim)
        self.audio_modality_emb = nn.Parameter(torch.randn(pooler_hidden_dim) * 0.02)
        self.attention_pooler = LTX25AttentionPooler(pooler_hidden_dim, num_queries, num_pooler_heads)
        self.mlp_hidden = nn.Linear(pooler_hidden_dim * num_queries, mlp_hidden)
        self.mlp_out = nn.Linear(mlp_hidden, 1)

    def forward(
        self,
        video_tokens: torch.Tensor | None = None,
        audio_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a positive duration prediction in seconds for every batch item."""
        if video_tokens is None and audio_tokens is None:
            raise ValueError("LTX25DurationHead requires video_tokens or audio_tokens")
        groups: list[torch.Tensor] = []
        if video_tokens is not None:
            groups.append(self.video_input_proj(video_tokens) + self.video_modality_emb)
        if audio_tokens is not None:
            groups.append(self.audio_input_proj(audio_tokens) + self.audio_modality_emb)
        pooled = self.attention_pooler(torch.cat(groups, dim=1)).flatten(1)
        hidden = torch.nn.functional.gelu(self.mlp_hidden(pooled), approximate="tanh")
        return self.mlp_out(hidden).squeeze(-1).exp()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "LTX25DurationHead":
        """Construct and strictly load the standalone LTX-2.5 DurationHead."""
        metadata = inspect_checkpoint(checkpoint_path)
        transformer = metadata.config.get("transformer", {})
        duration = metadata.config.get("duration_head", {})
        if not isinstance(transformer, dict) or not isinstance(duration, dict):
            raise ValueError("LTX-2.5 DurationHead metadata must contain transformer and duration_head objects")
        with torch.device("meta"):
            model = cls(
                video_cross_attention_dim=int(transformer.get("cross_attention_dim", 4096)),
                audio_cross_attention_dim=int(transformer.get("audio_cross_attention_dim", 2048)),
                pooler_hidden_dim=int(duration.get("pooler_hidden_dim", 256)),
                num_queries=int(duration.get("num_queries", 1)),
                num_pooler_heads=int(duration.get("num_pooler_heads", 4)),
                mlp_hidden=int(duration.get("mlp_hidden", 256)),
            )
        unexpected, missing = ltx25_duration_checkpoint_key_coverage(checkpoint_path, set(model.state_dict()))
        if unexpected or missing:
            raise ValueError(
                "LTX-2.5 DurationHead checkpoint coverage mismatch: "
                f"unexpected={sorted(unexpected)[:5]}, missing={sorted(missing)[:5]}"
            )
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            state_dict = {
                key.removeprefix("duration_head."): checkpoint.get_tensor(key)
                for key in checkpoint.keys()
                if key.startswith("duration_head.")
            }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True, assign=True)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"LTX-2.5 DurationHead load mismatch: missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return model.to(device=device, dtype=torch_dtype).eval()


def ltx25_duration_checkpoint_key_coverage(
    checkpoint_path: str | Path, model_keys: set[str]
) -> tuple[set[str], set[str]]:
    """Return unexplained source keys and missing isolated DurationHead keys."""
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        mapped = {key.removeprefix("duration_head.") for key in checkpoint.keys() if key.startswith("duration_head.")}
    return mapped - model_keys, model_keys - mapped


def seconds_to_num_frames(
    seconds: float,
    *,
    frame_rate: float,
    min_seconds: float = 1.0,
    max_seconds: float = 20.0,
) -> int:
    """Clamp a duration then snap it upward to LTX's causal ``8k + 1`` frame grid."""
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    min_frames = round(min_seconds * frame_rate)
    max_frames = round(max_seconds * frame_rate)
    raw_frames = min(max(round(seconds * frame_rate), min_frames), max_frames)
    frames = ((raw_frames - 1) // 8) * 8 + 1
    if frames < min_frames:
        frames = min(((min_frames - 1 + 7) // 8) * 8 + 1, max_frames)
    return frames


__all__ = [
    "LTX25DurationHead",
    "ltx25_duration_checkpoint_key_coverage",
    "seconds_to_num_frames",
]
