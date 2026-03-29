"""Simplified LoRA (Low-Rank Adaptation) loader with support for multiple format patterns."""

from __future__ import annotations

import re

import torch
from torch import nn

from telefuser.utils.logging import logger
from telefuser.utils.model_weight import load_state_dict

# Simplified pattern definitions as tuples (up_suffix, down_suffix, mid_suffix)
LORA_PATTERNS = {
    "standard": (".lora_up.weight", ".lora_down.weight", ".lora_mid.weight"),
    "diffusers": ("_lora.up.weight", "_lora.down.weight", None),
    "diffusers_v2": (".lora_B.weight", ".lora_A.weight", None),
    "diffusers_v3": (".lora.up.weight", ".lora.down.weight", None),
    "mochi": (".lora_B", ".lora_A", None),
    "transformers": (
        ".lora_linear_layer.up.weight",
        ".lora_linear_layer.down.weight",
        None,
    ),
    "qwen": (".lora_B.default.weight", ".lora_A.default.weight", None),
}

# Diff patterns for direct addition style LoRA
DIFF_PATTERNS = [
    (".diff", ".weight"),
    (".diff_b", ".bias"),
    (".diff_m", ".modulation"),
]

# Common prefixes to remove from model keys
COMMON_PREFIXES = ["diffusion_model.", "model.", "unet."]


class LoRALoader:
    """Simplified LoRA loader that applies weights to model weights."""

    def __init__(self, key_mapping_rules: list[tuple[str, str]] | None = None):
        """
        Args:
            key_mapping_rules: Optional list of (pattern, replacement) regex rules for key mapping
        """
        self.key_mapping_rules = key_mapping_rules or []
        self._compile_rules()

    def _compile_rules(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_rules = [(re.compile(pattern), replacement) for pattern, replacement in self.key_mapping_rules]

    def _apply_key_mapping(self, key: str) -> str:
        """Apply key mapping rules to a key."""
        for pattern, replacement in self.compiled_rules:
            key = pattern.sub(replacement, key)
        return key

    def _detect_lora_format(self, key: str) -> tuple[str, str] | None:
        """Detect LoRA format and return (format_name, up_suffix) if found."""
        for format_name, (up_suffix, down_suffix, _) in LORA_PATTERNS.items():
            if key.endswith(up_suffix):
                return format_name, up_suffix
        return None

    def _extract_base_key(self, key: str, suffix: str) -> str | None:
        """Extract base key by removing the detected suffix."""
        if key.endswith(suffix):
            return key[: -len(suffix)]
        return None

    def _get_model_key(self, base_key: str, suffix_to_add: str = ".weight") -> str | None:
        """Extract the model weight key from LoRA key."""
        # For Qwen models, keep transformer_blocks prefix
        if base_key.startswith("transformer_blocks.") and len(base_key.split(".")) > 1:
            if base_key.split(".")[1].isdigit():
                model_key = base_key + suffix_to_add
            else:
                model_key = self._remove_prefixes(base_key) + suffix_to_add
        else:
            model_key = self._remove_prefixes(base_key) + suffix_to_add

        # Apply key mapping rules if provided
        if self.compiled_rules:
            model_key = self._apply_key_mapping(model_key)

        return model_key

    @staticmethod
    def _remove_prefixes(key: str) -> str:
        """Remove common model prefixes from a key."""
        for prefix in COMMON_PREFIXES:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    def extract_lora_alphas(self, lora_weights: dict) -> dict:
        """Extract LoRA alpha values from the state dict."""
        lora_alphas = {}
        for key in lora_weights.keys():
            if key.endswith(".alpha"):
                base_key = key[:-6]  # Remove .alpha
                lora_alphas[base_key] = lora_weights[key].item()
        return lora_alphas

    def extract_lora_pairs(self, lora_weights: dict) -> dict[str, dict]:
        """Extract all LoRA pairs from the state dict."""
        lora_alphas = self.extract_lora_alphas(lora_weights)
        lora_pairs = {}

        for key in lora_weights.keys():
            # Skip alpha parameters
            if key.endswith(".alpha"):
                continue

            # Detect format
            format_detected = self._detect_lora_format(key)
            if format_detected is None:
                continue

            format_name, up_suffix = format_detected
            up_suffix, down_suffix, mid_suffix = LORA_PATTERNS[format_name]

            # Extract base key
            base_key = self._extract_base_key(key, up_suffix)
            if base_key is None:
                continue

            # Check if down weight exists
            down_key = base_key + down_suffix
            if down_key not in lora_weights:
                continue

            # Check for mid weight
            mid_key = None
            if mid_suffix:
                mid_key = base_key + mid_suffix
                if mid_key not in lora_weights:
                    mid_key = None

            # Get alpha value
            alpha = lora_alphas.get(base_key, None)

            # Get model key
            model_key = self._get_model_key(base_key, ".weight")
            if model_key is None:
                logger.warning(f"Failed to extract model key from LoRA key: {key}")
                continue

            lora_pairs[model_key] = {
                "format": format_name,
                "base_key": base_key,
                "up_key": key,
                "down_key": down_key,
                "mid_key": mid_key,
                "alpha": alpha,
            }

        return lora_pairs

    def extract_lora_diffs(self, lora_weights: dict) -> dict[str, dict]:
        """Extract diff-style LoRA weights."""
        lora_diffs = {}

        for key in lora_weights.keys():
            for check_suffix, add_suffix in DIFF_PATTERNS:
                if key.endswith(check_suffix):
                    base_key = key[: -len(check_suffix)]
                    model_key = self._get_model_key(base_key, add_suffix)

                    if model_key:
                        lora_diffs[model_key] = {
                            "diff_key": key,
                            "type": check_suffix,
                        }
                    break

        return lora_diffs

    def apply_lora(
        self,
        model_weights: dict[str, torch.Tensor] | nn.Module,
        lora_weights: dict[str, torch.Tensor] | str,
        alpha: float = None,
        strength: float = 1.0,
    ) -> int:
        """Apply LoRA weights to model weights.

        Args:
            model_weights: The model weights dictionary or module
            lora_weights: The LoRA weights dictionary or file path
            alpha: Global alpha scaling factor
            strength: Additional strength factor for LoRA deltas

        Returns:
            Number of LoRA weights successfully applied
        """
        # Load weights if paths are provided
        if isinstance(lora_weights, str):
            lora_weights = load_state_dict(lora_weights)
        if isinstance(model_weights, nn.Module):
            weight_dict = model_weights.state_dict()
        else:
            weight_dict = model_weights

        # Extract LoRA pairs and diffs
        lora_pairs = self.extract_lora_pairs(lora_weights)
        lora_diffs = self.extract_lora_diffs(lora_weights)

        applied_count = 0
        used_lora_keys = set()

        # Apply LoRA pairs (matrix multiplication)
        for model_key, pair_info in lora_pairs.items():
            if model_key not in weight_dict:
                logger.debug(f"Model key not found: {model_key}")
                continue

            param = weight_dict[model_key]
            up_key = pair_info["up_key"]
            down_key = pair_info["down_key"]

            # Track used keys
            used_lora_keys.add(up_key)
            used_lora_keys.add(down_key)
            if pair_info["mid_key"]:
                used_lora_keys.add(pair_info["mid_key"])

            try:
                lora_up = lora_weights[up_key].to(param.device, param.dtype)
                lora_down = lora_weights[down_key].to(param.device, param.dtype)

                # Calculate LoRA scale
                if pair_info["alpha"]:
                    lora_scale = pair_info["alpha"] / lora_down.shape[0]
                elif alpha is not None:
                    lora_scale = alpha / lora_down.shape[0]
                else:
                    lora_scale = 1

                # Apply matrix multiplication for 2D tensors
                if len(lora_down.shape) == 2 and len(lora_up.shape) == 2:
                    lora_delta = torch.mm(lora_up, lora_down) * lora_scale
                    if strength is not None:
                        lora_delta = lora_delta * float(strength)

                    param.data += lora_delta
                    applied_count += 1
                else:
                    logger.warning(f"Unexpected LoRA shape for {model_key}: down={lora_down.shape}, up={lora_up.shape}")

            except Exception as e:
                logger.warning(f"Failed to apply LoRA pair for {model_key}: {e}")

        # Apply diff weights (direct addition)
        for model_key, diff_info in lora_diffs.items():
            if model_key not in weight_dict:
                logger.debug(f"Model key not found for diff: {model_key}")
                continue

            param = weight_dict[model_key]
            diff_key = diff_info["diff_key"]

            # Track used keys
            used_lora_keys.add(diff_key)

            try:
                lora_diff = lora_weights[diff_key].to(param.device, param.dtype)
                scale_factor = (alpha if alpha is not None else 1) * (strength if strength is not None else 1.0)
                param.data += lora_diff * scale_factor
                applied_count += 1
            except Exception as e:
                logger.warning(f"Failed to apply LoRA diff for {model_key}: {e}")

        # Warn about unused keys
        all_lora_keys = set(k for k in lora_weights.keys() if not k.endswith(".alpha"))
        unused_lora_keys = all_lora_keys - used_lora_keys

        if unused_lora_keys:
            logger.warning(f"Found {len(unused_lora_keys)} unused LoRA weights:")
            for key in list(unused_lora_keys)[:10]:
                logger.warning(f"  Unused: {key}")
            if len(unused_lora_keys) > 10:
                logger.warning(f"  ... and {len(unused_lora_keys) - 10} more")

        logger.info(f"Applied {applied_count} LoRA weight adjustments")

        if applied_count == 0 and (lora_pairs or lora_diffs):
            logger.error("No LoRA weights were applied! Check for key name mismatches.")

        del lora_weights
        if isinstance(model_weights, nn.Module):
            model_weights.load_state_dict(weight_dict, strict=True)
        return applied_count
