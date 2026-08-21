"""Gemma4 Unified text-encoder loading for the LTX-2.5 packed checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

PACKED_GEMMA_CONFIG_KEY = "gemma_config"
PACKED_TOKENIZER_KEY = "tokenizer_json"
PACKED_ASSET_PREFIX = "hf_asset__"
GEMMA_MAX_LENGTH = 1024
_CASTABLE_FLOAT_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32, torch.float64})


def _initialize_gemma4_unified_buffers(model: torch.nn.Module) -> None:
    """Recompute non-persistent Gemma4 buffers after meta construction.

    The packed checkpoint stores model parameters only.  Upstream rebuilds the
    per-layer rotary frequencies and embedding scale after construction, rather
    than relying on the meta-initialized values inherited from Transformers.
    """
    language_model = model.model.language_model
    config = model.config.text_config
    rotary_embedding = language_model.rotary_emb
    for layer_type in dict.fromkeys(config.layer_types):
        rope_params = config.rope_parameters[layer_type]
        if rope_params is None:
            continue
        rope_type = rope_params["rope_type"]
        if rope_type == "default":
            inv_freq, attention_scaling = rotary_embedding.compute_default_rope_parameters(
                config, layer_type=layer_type
            )
        else:
            kwargs = {"layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                kwargs["head_dim_key"] = "global_head_dim"
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](config, **kwargs)
        rotary_embedding.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
        rotary_embedding.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
        setattr(rotary_embedding, f"{layer_type}_attention_scaling", attention_scaling)

    language_model.embed_tokens.register_buffer(
        "embed_scale", torch.tensor(config.hidden_size**0.5, device="cpu"), persistent=False
    )
    if hasattr(language_model, "embed_tokens_per_layer"):
        language_model.embed_tokens_per_layer.register_buffer(
            "embed_scale", torch.tensor(config.hidden_size_per_layer_input**0.5, device="cpu"), persistent=False
        )


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Decode a packed byte tensor without depending on its signedness."""
    array = tensor.detach().cpu().numpy()
    return array.tobytes() if array.dtype == np.uint8 else array.astype(np.uint8).tobytes()


def _cast_checkpoint_tensor(tensor: torch.Tensor, torch_dtype: torch.dtype) -> torch.Tensor:
    """Match the upstream builder's pre-assignment floating-point cast policy."""
    if tensor.dtype not in _CASTABLE_FLOAT_DTYPES:
        return tensor
    # Scalar float32 values are checkpoint scales, not model weights.  Upstream
    # keeps them in float32 to avoid losing scale precision.
    if tensor.ndim == 0 and tensor.dtype == torch.float32:
        return tensor
    return tensor.to(dtype=torch_dtype)


@dataclass(frozen=True, slots=True)
class LTX25GemmaAssets:
    """Hugging Face assets embedded in an LTX-2.5 Gemma4 checkpoint."""

    checkpoint_path: Path
    config: dict[str, Any]
    tokenizer_json: bytes
    sidecars: dict[str, bytes]

    @classmethod
    def load(cls, checkpoint_path: str | Path) -> "LTX25GemmaAssets":
        """Read embedded assets without materializing model weights."""
        path = Path(checkpoint_path).expanduser().resolve()
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            serialized_config = metadata.get(PACKED_GEMMA_CONFIG_KEY)
            if serialized_config is None:
                raise ValueError(f"LTX-2.5 Gemma checkpoint is missing {PACKED_GEMMA_CONFIG_KEY!r}: {path}")
            config = json.loads(serialized_config)
            keys = set(checkpoint.keys())
            if PACKED_TOKENIZER_KEY not in keys:
                raise ValueError(f"LTX-2.5 Gemma checkpoint is missing {PACKED_TOKENIZER_KEY!r}: {path}")
            tokenizer_json = _tensor_to_bytes(checkpoint.get_tensor(PACKED_TOKENIZER_KEY))
            sidecars = {
                key.removeprefix(PACKED_ASSET_PREFIX): _tensor_to_bytes(checkpoint.get_tensor(key))
                for key in keys
                if key.startswith(PACKED_ASSET_PREFIX)
            }
        if config.get("model_type") != "gemma4_unified":
            raise ValueError(
                f"LTX-2.5 requires model_type='gemma4_unified', got {config.get('model_type')!r} from {path}"
            )
        if "tokenizer_config.json" not in sidecars:
            raise ValueError(f"LTX-2.5 Gemma checkpoint is missing tokenizer_config.json: {path}")
        return cls(path, config, tokenizer_json, sidecars)

    def build_config(self) -> Any:
        """Build the registered Transformers configuration from packed JSON."""
        model_type = self.config.get("model_type")
        try:
            return CONFIG_MAPPING[model_type].from_dict(self.config)
        except KeyError as exc:
            raise ValueError(f"Unsupported packed Gemma model_type={model_type!r}") from exc

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        """Build the LTX Gemma tokenizer with upstream packed sidecar settings."""
        tokenizer_config = json.loads(self.sidecars["tokenizer_config.json"])
        excluded = {
            "tokenizer_class",
            "auto_map",
            "model_max_length",
            "backend",
            "is_local",
            "local_files_only",
            "processor_class",
            "added_tokens_decoder",
        }
        kwargs = {key: value for key, value in tokenizer_config.items() if key not in excluded}
        template = self.sidecars.get("chat_template.jinja")
        if template is not None:
            kwargs.setdefault("chat_template", template.decode())
        return PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_buffer(self.tokenizer_json),
            model_max_length=GEMMA_MAX_LENGTH,
            **kwargs,
        )


class LTX25GemmaTokenizer:
    """LTX's 1024-token, left-padded Gemma encoding adapter."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = GEMMA_MAX_LENGTH
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize prompts using the upstream BOS insertion and left-padding contract."""
        bos_token_id = self.tokenizer.bos_token_id
        if bos_token_id is None:
            raise ValueError("LTX-2.5 Gemma tokenizer is missing bos_token_id")
        token_ids: list[list[int]] = []
        for prompt in prompts:
            encoded = self.tokenizer(
                prompt.strip(),
                padding=False,
                truncation=True,
                max_length=GEMMA_MAX_LENGTH,
                return_tensors="pt",
            )
            values = encoded.input_ids[0].tolist()
            if not values or values[0] != bos_token_id:
                values = [bos_token_id, *values][:GEMMA_MAX_LENGTH]
            token_ids.append(values)
        padded = self.tokenizer.pad(
            {"input_ids": token_ids},
            padding="max_length",
            max_length=GEMMA_MAX_LENGTH,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return padded.input_ids.to(device), padded.attention_mask.to(device)


def gemma4_checkpoint_key_to_model_key(key: str) -> str | None:
    """Map packed Comfy-flat Gemma4 keys into ``LTX25Gemma4TextEncoder`` keys."""
    if key == PACKED_TOKENIZER_KEY or key.startswith(PACKED_ASSET_PREFIX):
        return None
    if key.startswith("model.layers."):
        return "model.model.language_model." + key.removeprefix("model.")
    if key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
        return "model.model.language_model." + key.removeprefix("model.")
    if key.startswith("vision_model."):
        return "model.model.embed_vision." + key.removeprefix("vision_model.")
    if key.startswith("multi_modal_projector.embedding_projection."):
        suffix = key.removeprefix("multi_modal_projector.embedding_projection.")
        return "model.model.embed_vision.multimodal_embedder.embedding_projection." + suffix
    if key.startswith("audio_projector."):
        return "model.model.embed_audio." + key.removeprefix("audio_projector.")
    return None


def gemma4_checkpoint_key_coverage(checkpoint_path: str | Path, model_keys: set[str]) -> tuple[set[str], set[str]]:
    """Return unexpected mapped checkpoint keys and missing non-tied model keys."""
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        source_keys = checkpoint.keys()
    mapped = {target for key in source_keys if (target := gemma4_checkpoint_key_to_model_key(key)) is not None}
    expected = set(model_keys)
    expected.discard("model.lm_head.weight")
    return mapped - expected, expected - mapped


class LTX25Gemma4TextEncoder(torch.nn.Module):
    """Gemma4 Unified wrapper that exposes LTX's raw hidden-state encoding path."""

    def __init__(self, model: torch.nn.Module, tokenizer: LTX25GemmaTokenizer) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        strict: bool = True,
    ) -> "LTX25Gemma4TextEncoder":
        """Construct and assign the packed Gemma4 checkpoint with explicit key coverage."""
        assets = LTX25GemmaAssets.load(checkpoint_path)
        config = assets.build_config()
        # Match the upstream Gemma4 configurator: parameters and transient
        # buffers originate on meta, then the packed checkpoint assigns weights.
        with torch.device("meta"):
            model = AutoModelForImageTextToText.from_config(config)
        _initialize_gemma4_unified_buffers(model)
        wrapper = cls(model, LTX25GemmaTokenizer(assets.build_tokenizer()))
        unexpected, missing = gemma4_checkpoint_key_coverage(checkpoint_path, set(wrapper.state_dict()))
        if unexpected or missing:
            raise ValueError(
                "LTX-2.5 Gemma checkpoint coverage mismatch: "
                f"unexpected={sorted(unexpected)[:5]}, missing={sorted(missing)[:5]}"
            )
        source = load_file(str(assets.checkpoint_path), device="cpu")
        state_dict = {
            target: _cast_checkpoint_tensor(tensor, torch_dtype)
            for key, tensor in source.items()
            if (target := gemma4_checkpoint_key_to_model_key(key)) is not None
        }
        embed_key = "model.model.language_model.embed_tokens.weight"
        state_dict["model.lm_head.weight"] = state_dict[embed_key]
        missing_keys, unexpected_keys = wrapper.load_state_dict(state_dict, strict=strict, assign=True)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"LTX-2.5 Gemma load mismatch: missing={missing_keys[:5]}, unexpected={unexpected_keys[:5]}"
            )
        # The upstream builder casts checkpoint weights before assignment and
        # then transfers the module without a dtype conversion.  In particular,
        # Gemma's non-persistent rotary buffers remain float32.
        return wrapper.to(device=device).eval()

    @torch.inference_mode()
    def encode(self, prompts: list[str]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        """Return the raw Gemma hidden states, token IDs, and binary attention mask."""
        if not prompts:
            raise ValueError("LTX-2.5 Gemma encode requires at least one prompt")
        device = next(self.model.parameters()).device
        input_ids, attention_mask = self.tokenizer.encode(prompts, device)
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return tuple(outputs.hidden_states), input_ids, attention_mask
