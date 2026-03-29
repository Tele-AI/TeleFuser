from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, NamedTuple

import torch
from einops import rearrange
from torch import nn
from transformers import (
    AutoTokenizer,
    Gemma3Config,
    Gemma3ForConditionalGeneration,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from telefuser.core.base_model import BaseModel

from .ltx_dit import (
    Attention,
    FeedForward,
    LTXRopeType,
    find_matching_file,
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    get_ltx23_dev_transformer_config,
    precompute_freqs_cis,
    rms_norm,
)


@dataclass
class Gemma3RopeScaling:
    factor: float = 8.0
    rope_type: str = "linear"


@dataclass
class Gemma3TextConfig:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_logit_softcapping: float | None = None
    cache_implementation: str = "hybrid"
    final_logit_softcapping: float | None = None
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 3840
    initializer_range: float = 0.02
    intermediate_size: int = 15360
    max_position_embeddings: int = 131072
    model_type: str = "gemma3_text"
    num_attention_heads: int = 16
    num_hidden_layers: int = 48
    num_key_value_heads: int = 8
    query_pre_attn_scalar: int = 256
    rms_norm_eps: float = 1e-06
    rope_local_base_freq: int = 10000
    rope_scaling: Gemma3RopeScaling = field(default_factory=Gemma3RopeScaling)
    rope_theta: int = 1000000
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    torch_dtype: str = "float32"
    use_cache: bool = True
    vocab_size: int = 262208


@dataclass
class Gemma3VisionConfig:
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    image_size: int = 896
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-06
    model_type: str = "siglip_vision_model"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    torch_dtype: str = "float32"
    vision_use_head: bool = False


@dataclass
class Gemma3ConfigData:
    architectures: list[str] = field(default_factory=lambda: ["Gemma3ForConditionalGeneration"])
    boi_token_index: int = 255999
    eoi_token_index: int = 256000
    eos_token_id: list[int] = field(default_factory=lambda: [1, 106])
    image_token_index: int = 262144
    initializer_range: float = 0.02
    mm_tokens_per_image: int = 256
    model_type: str = "gemma3"
    text_config: Gemma3TextConfig = field(default_factory=Gemma3TextConfig)
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    vision_config: Gemma3VisionConfig = field(default_factory=Gemma3VisionConfig)

    def to_dict(self) -> dict:
        return asdict(self)


GEMMA3_CONFIG_FOR_LTX = Gemma3ConfigData()


class LTXVGemmaTokenizer:
    """
    Tokenizer wrapper for Gemma models compatible with LTXV processes.
    This class wraps HuggingFace's `AutoTokenizer` for use with Gemma text encoders,
    ensuring correct settings and output formatting for downstream consumption.
    """

    def __init__(self, tokenizer_path: str, max_length: int = 256):
        """
        Initialize the tokenizer.
        Args:
            tokenizer_path (str): Path to the pretrained tokenizer files or model directory.
            max_length (int, optional): Max sequence length for encoding. Defaults to 256.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, model_max_length=max_length
        )
        # Gemma expects left padding for chat-style prompts; for plain text it doesn't matter much.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False) -> dict[str, list[tuple[int, int]]]:
        """
        Tokenize the given text and return token IDs and attention weights.
        Args:
            text (str): The input string to tokenize.
            return_word_ids (bool, optional): If True, includes the token's position (index) in the output tuples.
                                              If False (default), omits the indices.
        Returns:
            dict[str, list[tuple[int, int]]] OR dict[str, list[tuple[int, int, int]]]:
                A dictionary with a "gemma" key mapping to:
                    - a list of (token_id, attention_mask) tuples if return_word_ids is False;
                    - a list of (token_id, attention_mask, index) tuples if return_word_ids is True.
        Example:
            >>> tokenizer = LTXVGemmaTokenizer("path/to/tokenizer", max_length=8)
            >>> tokenizer.tokenize_with_weights("hello world")
            {'gemma': [(1234, 1), (5678, 1), (2, 0), ...]}
        """
        text = text.strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask
        tuples = [
            (token_id, attn, i) for i, (token_id, attn) in enumerate(zip(input_ids[0], attention_mask[0], strict=True))
        ]
        out = {"gemma": tuples}

        if not return_word_ids:
            # Return only (token_id, attention_mask) pairs, omitting token position
            out = {k: [(t, w) for t, w, _ in v] for k, v in out.items()}

        return out


# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


def norm_and_concat_per_token_rms(
    encoded_text: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token RMSNorm normalization for V2 models.
    Args:
        encoded_text: [B, T, D, L]
        attention_mask: [B, T] binary mask
    Returns:
        [B, T, D*L] normalized tensor with padding zeroed out.
    """
    B, T, D, L = encoded_text.shape  # noqa: N806
    variance = torch.mean(encoded_text**2, dim=2, keepdim=True)  # [B,T,1,L]
    normed = encoded_text * torch.rsqrt(variance + 1e-6)
    normed = normed.reshape(B, T, D * L)
    mask_3d = attention_mask.bool().unsqueeze(-1)  # [B, T, 1]
    return torch.where(mask_3d, normed, torch.zeros_like(normed))


def _rescale_norm(x: torch.Tensor, target_dim: int, source_dim: int) -> torch.Tensor:
    """Rescale normalization: x * sqrt(target_dim / source_dim)."""
    return x * math.sqrt(target_dim / source_dim)


# ---------------------------------------------------------------------------
# Feature extractor variants
# ---------------------------------------------------------------------------


class FeatureExtractorV1(nn.Module):
    """19B: per-segment norm -> aggregate_embed -> 3840"""

    def __init__(self, aggregate_embed: nn.Module, is_av: bool = False):
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.is_av = is_av

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "left"
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        dtype = encoded.dtype
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(encoded, sequence_lengths, padding_side)
        features = self.aggregate_embed(normed.to(dtype))
        if self.is_av:
            return features, features
        return features, None


class FeatureExtractorV2(nn.Module):
    """22B: per-token RMS norm -> rescale -> dual aggregate embeds"""

    def __init__(
        self,
        video_aggregate_embed: nn.Linear,
        embedding_dim: int,
        audio_aggregate_embed: nn.Linear | None = None,
    ):
        super().__init__()
        self.video_aggregate_embed = video_aggregate_embed
        self.audio_aggregate_embed = audio_aggregate_embed
        self.embedding_dim = embedding_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        normed = norm_and_concat_per_token_rms(encoded, attention_mask)
        normed = normed.to(encoded.dtype)
        v_dim = self.video_aggregate_embed.out_features
        video = self.video_aggregate_embed(_rescale_norm(normed, v_dim, self.embedding_dim))
        audio = None
        if self.audio_aggregate_embed is not None:
            a_dim = self.audio_aggregate_embed.out_features
            audio = self.audio_aggregate_embed(_rescale_norm(normed, a_dim, self.embedding_dim))
        return video, audio


class _BasicTransformerBlock1D(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        apply_gated_attention: bool = False,
    ):
        super().__init__()

        self.attn1 = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            rope_type=rope_type,
            apply_gated_attention=apply_gated_attention,
        )

        self.ff = FeedForward(
            dim,
            dim_out=dim,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.

        # 1. Normalization Before Self-Attention
        norm_hidden_states = rms_norm(hidden_states)

        norm_hidden_states = norm_hidden_states.squeeze(1)

        # 2. Self-Attention
        attn_output = self.attn1(norm_hidden_states, mask=attention_mask, pe=pe)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Normalization before Feed-Forward
        norm_hidden_states = rms_norm(hidden_states)

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class Embeddings1DConnector(torch.nn.Module):
    """
    Embeddings1DConnector applies a 1D transformer-based processing to sequential embeddings (e.g., for video, audio, or
    other modalities). It supports rotary positional encoding (rope), optional causal temporal positioning, and can
    substitute padded positions with learnable registers. The module is highly configurable for head size, number of
    layers, and register usage.
    Args:
        attention_head_dim (int): Dimension of each attention head (default=128).
        num_attention_heads (int): Number of attention heads (default=30).
        num_layers (int): Number of transformer layers (default=2).
        positional_embedding_theta (float): Scaling factor for position embedding (default=10000.0).
        positional_embedding_max_pos (list[int] | None): Max positions for positional embeddings (default=[1]).
        causal_temporal_positioning (bool): If True, uses causal attention (default=False).
        num_learnable_registers (int | None): Number of learnable registers to replace padded tokens. If None, disables
            register replacement. (default=128)
        rope_type (LTXRopeType): The RoPE variant to use (default=DEFAULT_ROPE_TYPE).
        double_precision_rope (bool): Use double precision rope calculation (default=False).
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        attention_head_dim: int = 128,
        num_attention_heads: int = 30,
        num_layers: int = 2,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        causal_temporal_positioning: bool = False,
        num_learnable_registers: int | None = 128,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.inner_dim = num_attention_heads * attention_head_dim
        self.causal_temporal_positioning = causal_temporal_positioning
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = (
            positional_embedding_max_pos if positional_embedding_max_pos is not None else [1]
        )
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.transformer_1d_blocks = torch.nn.ModuleList(
            [
                _BasicTransformerBlock1D(
                    dim=self.inner_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    rope_type=rope_type,
                    apply_gated_attention=apply_gated_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.num_learnable_registers = num_learnable_registers
        if self.num_learnable_registers:
            self.learnable_registers = torch.nn.Parameter(
                torch.rand(self.num_learnable_registers, self.inner_dim, dtype=torch.bfloat16) * 2.0 - 1.0
            )

    def _replace_padded_with_learnable_registers(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert hidden_states.shape[1] % self.num_learnable_registers == 0, (
            f"Hidden states sequence length {hidden_states.shape[1]} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}."
        )

        num_registers_duplications = hidden_states.shape[1] // self.num_learnable_registers
        learnable_registers = torch.tile(self.learnable_registers, (num_registers_duplications, 1))
        attention_mask_binary = (attention_mask.squeeze(1).squeeze(1).unsqueeze(-1) >= -9000.0).int()

        non_zero_hidden_states = hidden_states[:, attention_mask_binary.squeeze().bool(), :]
        non_zero_nums = non_zero_hidden_states.shape[1]
        pad_length = hidden_states.shape[1] - non_zero_nums
        adjusted_hidden_states = torch.nn.functional.pad(non_zero_hidden_states, pad=(0, 0, 0, pad_length), value=0)
        flipped_mask = torch.flip(attention_mask_binary, dims=[1])
        hidden_states = flipped_mask * adjusted_hidden_states + (1 - flipped_mask) * learnable_registers

        attention_mask = torch.full_like(
            attention_mask,
            0.0,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return hidden_states, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Embeddings1DConnector.
        Args:
            hidden_states (torch.Tensor): Input tensor of embeddings (shape [batch, seq_len, feature_dim]).
            attention_mask (torch.Tensor|None): Optional mask for valid tokens (shape compatible with hidden_states).
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed features and the corresponding (possibly modified) mask.
        """
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(hidden_states, attention_mask)

        indices_grid = torch.arange(hidden_states.shape[1], dtype=torch.float32, device=hidden_states.device)
        indices_grid = indices_grid[None, None, :]
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        freqs_cis = precompute_freqs_cis(
            indices_grid=indices_grid,
            dim=self.inner_dim,
            out_dtype=hidden_states.dtype,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            num_attention_heads=self.num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, pe=freqs_cis)

        hidden_states = rms_norm(hidden_states)

        return hidden_states, attention_mask


class Embeddings1DConnectorConfigurator:
    """Configurator for video embeddings connector."""

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        transformer_config = config.get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "interleaved"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

        # Video connector dimensions
        num_attention_heads = transformer_config.get("connector_num_attention_heads", 30)
        attention_head_dim = transformer_config.get("connector_attention_head_dim", 128)
        num_layers = transformer_config.get("connector_num_layers", 2)

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
        )
        return connector


class AudioEmbeddings1DConnectorConfigurator:
    """Configurator for audio embeddings connector with separate dimension settings."""

    @classmethod
    def from_config(cls, config: dict) -> Embeddings1DConnector:
        transformer_config = config.get("transformer", {})
        rope_type = LTXRopeType(transformer_config.get("rope_type", "interleaved"))
        double_precision_rope = transformer_config.get("frequencies_precision", False) == "float64"
        pe_max_pos = transformer_config.get("connector_positional_embedding_max_pos", [1])

        # Audio connector dimensions - fall back to video connector config for backwards compatibility
        num_attention_heads = transformer_config.get(
            "audio_connector_num_attention_heads",
            transformer_config.get("connector_num_attention_heads", 30),
        )
        attention_head_dim = transformer_config.get(
            "audio_connector_attention_head_dim",
            transformer_config.get("connector_attention_head_dim", 128),
        )
        num_layers = transformer_config.get(
            "audio_connector_num_layers",
            transformer_config.get("connector_num_layers", 2),
        )

        connector = Embeddings1DConnector(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            positional_embedding_max_pos=pe_max_pos,
            rope_type=rope_type,
            double_precision_rope=double_precision_rope,
            apply_gated_attention=transformer_config.get("connector_apply_gated_attention", False),
        )
        return connector


class EmbeddingsProcessorOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


def convert_to_additive_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert binary attention mask to additive form for transformer masking."""
    return (attention_mask.to(torch.int64) - 1).to(dtype).reshape(
        (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
    ) * torch.finfo(dtype).max


def _to_binary_mask(encoded: torch.Tensor, encoded_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert connector output mask to binary mask and apply to encoded tensor."""
    binary_mask = (encoded_mask < 0.000001).to(torch.int64)
    binary_mask = binary_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
    encoded = encoded * binary_mask
    return encoded, binary_mask


class EmbeddingsProcessor(nn.Module):
    """Wraps feature extractor + video connector + optional audio connector.
    Can operate in two modes:
    1. create_embeddings(): Takes pre-computed features + additive mask (backward compat, used by trainer)
    2. process_hidden_states(): Takes raw Gemma hidden states, runs feature extraction + connectors
    """

    def __init__(
        self,
        *,
        feature_extractor: nn.Module | None = None,
        video_connector: Embeddings1DConnector,
        audio_connector: Embeddings1DConnector | None = None,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.video_connector = video_connector
        self.audio_connector = audio_connector

    def create_embeddings(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor | None,
        additive_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.audio_connector is not None and audio_features is None:
            raise ValueError("Audio connector is configured but no audio features were provided.")
        if self.audio_connector is None and audio_features is not None:
            raise ValueError("Audio features were provided but no audio connector is configured.")

        video_encoded, video_mask = self.video_connector(video_features, additive_attention_mask)
        video_encoded, binary_mask = _to_binary_mask(video_encoded, video_mask)

        audio_encoded = None
        if self.audio_connector is not None:
            audio_encoded, _ = self.audio_connector(audio_features, additive_attention_mask)

        return video_encoded, audio_encoded, binary_mask.squeeze(-1)

    def process_hidden_states(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
        padding_side: str = "left",
    ) -> EmbeddingsProcessorOutput:
        """Full pipeline: feature extraction -> connectors -> final embeddings.
        Args:
            hidden_states: Raw Gemma hidden states (tuple of tensors per layer).
            attention_mask: Binary attention mask [B, seq_len].
            padding_side: Padding side used during tokenization.
        Returns:
            EmbeddingsProcessorOutput with video_encoding, audio_encoding, and attention_mask.
        """
        if self.feature_extractor is None:
            raise ValueError("feature_extractor is required for process_hidden_states()")

        video_feats, audio_feats = self.feature_extractor(hidden_states, attention_mask, padding_side)
        additive_mask = convert_to_additive_mask(attention_mask, video_feats.dtype)
        video_enc, audio_enc, binary_mask = self.create_embeddings(video_feats, audio_feats, additive_mask)
        return EmbeddingsProcessorOutput(video_enc, audio_enc, binary_mask)


class GemmaTextEncoder(BaseModel):
    """Pure Gemma text encoder.

    This module only consumes token ids / attention masks and returns raw hidden states.
    Tokenization and any chat-template / processor logic is handled by pipeline stages
    (Wan-style responsibility split).
    """

    def __init__(
        self,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if model is None:
            gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
            model = Gemma3ForConditionalGeneration(gemma_config)
        self.model = model
        self._dtype = dtype

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True, assign: bool = False):
        """Load weights and register runtime-only Gemma buffers required by LTX stages."""
        incompatible_keys = super().load_state_dict(state_dict, strict=strict, assign=assign)
        create_and_populate(self)
        return incompatible_keys

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype | None = None,
        device_map: str | None = None,
        local_files_only: bool = True,
        **kwargs,
    ) -> GemmaTextEncoder:
        """Load Gemma from a HuggingFace-style root and wrap it as a TeleFuser text encoder."""
        model = Gemma3ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            **kwargs,
        )
        module = cls(model=model, dtype=torch_dtype or next(model.parameters()).dtype)
        module = create_and_populate(module)
        module.eval()
        module.requires_grad_(False)
        if device_map not in {None, "auto"}:
            module = module.to(device_map)
        return module

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        """Run the Gemma language model and return per-layer hidden states."""
        if self.model is None:
            raise ValueError("GemmaTextEncoder.model is not initialized.")
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return outputs.hidden_states

    def enable_sequential_cpu_offload(
        self,
        device: torch.device | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        """Enable sequential CPU offload for the underlying Gemma model."""
        if self.model is None:
            raise ValueError("GemmaTextEncoder.model is not initialized.")

        from telefuser.offload import (
            AutoWrappedLinear,
            AutoWrappedModule,
            enable_sequential_cpu_offload,
        )

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch_dtype is None:
            torch_dtype = self._dtype

        dtype = next(iter(self.model.parameters())).dtype
        enable_sequential_cpu_offload(
            self.model,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=torch_dtype,
                computation_device=device,
            ),
        )

    @staticmethod
    def state_dict_converter():
        return GemmaTextEncoderStateDictConverter()


class GemmaTextEncoderStateDictConverter:
    """State dict converter for GemmaTextEncoder.

    Note: LTX Gemma weights are typically loaded via HuggingFace `from_pretrained`.
    This converter exists to keep the model API aligned with other TeleFuser models.
    """

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._convert(state_dict)

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self._convert(state_dict)

    @staticmethod
    def _convert(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Convert common Gemma checkpoint key layouts into the `GemmaTextEncoder` wrapper layout."""
        if not state_dict:
            return state_dict

        converted: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            # Some exports contain a typo in the prefix.
            key = key.replace("lanuage_model", "language_model")
            # Some exports wrap the language model in an extra `.model` module.
            key = key.replace("language_model.model.", "language_model.")

            # `GemmaTextEncoder` wraps `Gemma3ForConditionalGeneration` as `self.model`, so the full prefix is:
            # - `model.model.*` for the inner transformer/vision modules
            # - `model.lm_head.*` for the output head
            if key.startswith(("model.model.", "model.lm_head.")):
                converted_key = key
            elif key.startswith(("model.", "lm_head.")):
                converted_key = f"model.{key}"
            else:
                converted_key = f"model.model.{key}"

            converted[converted_key] = value

        # Some Gemma exports omit `lm_head.weight` when weights are tied to `embed_tokens.weight`.
        # HuggingFace models still expose an `lm_head` parameter, so we synthesize it to keep strict loading.
        if "model.lm_head.weight" not in converted:
            tie_candidates = (
                "model.model.language_model.embed_tokens.weight",
                "model.model.embed_tokens.weight",
            )
            for candidate in tie_candidates:
                if candidate in converted:
                    converted["model.lm_head.weight"] = converted[candidate]
                    break

        return converted


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    """Deprecated: tokenizer/processor are handled in pipeline stages.

    Kept for backward compatibility with older loading code paths.
    """
    _ = gemma_root
    return ()


class ModuleOps(NamedTuple):
    name: str
    matcher: Callable[[torch.nn.Module], bool]
    mutator: Callable[[torch.nn.Module], torch.nn.Module]


class GemmaTextEncoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> GemmaTextEncoder:  # noqa: ARG003
        from telefuser.utils.model_weight import init_weights_on_device

        gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        with init_weights_on_device("meta"):
            model = Gemma3ForConditionalGeneration(gemma_config)

        return GemmaTextEncoder(model=model)


class EmbeddingsProcessorConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> EmbeddingsProcessor:
        transformer_config = config.get("transformer", {})

        # Create video embeddings connector (always needed)
        video_connector = Embeddings1DConnectorConfigurator.from_config(config)

        # Create audio embeddings connector
        audio_connector = AudioEmbeddings1DConnectorConfigurator.from_config(config)

        # Create feature extractor
        feature_extractor = _create_feature_extractor(transformer_config)

        return EmbeddingsProcessor(
            video_connector=video_connector,
            audio_connector=audio_connector,
            feature_extractor=feature_extractor,
        )


_V2_EXPECTED_CONFIG = {
    "caption_proj_before_connector": True,
    "caption_projection_first_linear": False,
    "caption_proj_input_norm": False,
    "caption_projection_second_linear": False,
}


def _create_feature_extractor(transformer_config: dict) -> torch.nn.Module:
    """Select and create the appropriate feature extractor based on config.
    Detection logic:
    - V1: V2 config keys absent -> projection lives in transformer
    - V2: V2 config keys present with exact expected values -> per-token RMS norm with dual aggregate embeds
    - Anything else: NotImplementedError (config drift)
    """
    gemma_text_config = GEMMA3_CONFIG_FOR_LTX.text_config
    embedding_dim = gemma_text_config.hidden_size
    num_layers = gemma_text_config.num_hidden_layers + 1  # +1 for the embedding layer
    flat_dim = embedding_dim * num_layers

    overlapping_keys = transformer_config.keys() & _V2_EXPECTED_CONFIG.keys()
    if not overlapping_keys:
        aggregate_embed = torch.nn.Linear(flat_dim, embedding_dim, bias=False)
        return FeatureExtractorV1(aggregate_embed=aggregate_embed, is_av=True)

    missing_keys = _V2_EXPECTED_CONFIG.keys() - overlapping_keys
    if missing_keys:
        raise NotImplementedError("Partial V2 config - missing keys: " + ", ".join(sorted(missing_keys)))

    unexpected_value_keys = {k for k in overlapping_keys if transformer_config[k] != _V2_EXPECTED_CONFIG[k]}
    if unexpected_value_keys:
        raise NotImplementedError(
            "Unknown config: "
            + ", ".join(
                f"{k}={transformer_config[k]!r} (expected {_V2_EXPECTED_CONFIG[k]!r})" for k in unexpected_value_keys
            )
        )

    video_inner_dim = transformer_config["num_attention_heads"] * transformer_config["attention_head_dim"]
    audio_inner_dim = transformer_config["audio_num_attention_heads"] * transformer_config["audio_attention_head_dim"]
    return FeatureExtractorV2(
        video_aggregate_embed=torch.nn.Linear(flat_dim, video_inner_dim, bias=True),
        embedding_dim=embedding_dim,
        audio_aggregate_embed=torch.nn.Linear(flat_dim, audio_inner_dim, bias=True),
    )


def create_and_populate(module: GemmaTextEncoder) -> GemmaTextEncoder:
    model = module.model
    v_model = model.model.vision_tower.vision_model
    l_model = model.model.language_model

    config = model.config.text_config
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    base = config.rope_local_base_freq
    local_rope_freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim))
    inv_freqs, _ = ROPE_INIT_FUNCTIONS[config.rope_scaling["rope_type"]](config)

    positions_length = len(v_model.embeddings.position_ids[0])
    position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
    v_model.embeddings.register_buffer("position_ids", position_ids)
    embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale)
    l_model.rotary_emb_local.register_buffer("inv_freq", local_rope_freqs)
    l_model.rotary_emb.register_buffer("inv_freq", inv_freqs)

    return module


class LTXEmbeddingsProcessor(EmbeddingsProcessor):
    """Registered embeddings processor for the LTX 2.3 dev checkpoint."""

    def __init__(self) -> None:
        config = get_ltx23_dev_transformer_config()
        transformer_config = config["transformer"]
        super().__init__(
            feature_extractor=_create_feature_extractor(transformer_config),
            video_connector=Embeddings1DConnectorConfigurator.from_config(config),
            audio_connector=AudioEmbeddings1DConnectorConfigurator.from_config(config),
        )

    @staticmethod
    def state_dict_converter():
        return LTXEmbeddingsProcessorStateDictConverter()


class LTXEmbeddingsProcessorStateDictConverter:
    """Convert the shared LTX checkpoint into the embeddings processor layout."""

    _KEY_PREFIX_MAP = {
        "text_embedding_projection.aggregate_embed.": "feature_extractor.aggregate_embed.",
        "text_embedding_projection.video_aggregate_embed.": "feature_extractor.video_aggregate_embed.",
        "text_embedding_projection.audio_aggregate_embed.": "feature_extractor.audio_aggregate_embed.",
        "model.diffusion_model.video_embeddings_connector.": "video_connector.",
        "model.diffusion_model.audio_embeddings_connector.": "audio_connector.",
        # Some checkpoint exports use singular connector names.
        "model.diffusion_model.video_embedding_connector.": "video_connector.",
        "model.diffusion_model.audio_embedding_connector.": "audio_connector.",
    }

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        converted_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            for prefix, target_prefix in self._KEY_PREFIX_MAP.items():
                if key.startswith(prefix):
                    converted_state_dict[f"{target_prefix}{key.removeprefix(prefix)}"] = value
                    break
        return converted_state_dict

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("LTXEmbeddingsProcessor only supports the civitai-style single-file checkpoint.")


GEMMA_MODEL_OPS = ModuleOps(
    name="GemmaModel",
    matcher=lambda module: hasattr(module, "model") and isinstance(module.model, Gemma3ForConditionalGeneration),
    mutator=create_and_populate,
)

LTXTextEncoder = GemmaTextEncoder

__all__ = [
    "EmbeddingsProcessor",
    "EmbeddingsProcessorConfigurator",
    "EmbeddingsProcessorOutput",
    "GEMMA_MODEL_OPS",
    "LTXEmbeddingsProcessorStateDictConverter",
    "GemmaTextEncoder",
    "GemmaTextEncoderConfigurator",
    "LTXEmbeddingsProcessor",
    "LTXTextEncoder",
    "module_ops_from_gemma_root",
]
