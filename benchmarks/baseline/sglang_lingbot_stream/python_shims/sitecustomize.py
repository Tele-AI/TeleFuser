from __future__ import annotations

import os


def _patch_transformers_utils() -> None:
    try:
        import transformers.utils as transformers_utils
    except Exception:
        return

    defaults = {
        "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
    }
    for name, value in defaults.items():
        if not hasattr(transformers_utils, name):
            setattr(transformers_utils, name, value)


_patch_transformers_utils()


def _patch_sglang_cutedsl_norm_fallback() -> None:
    enabled = os.environ.get("SGLANG_LINGBOT_NATIVE_NORM_FALLBACK", "").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    try:
        import torch
        from sglang.multimodal_gen.runtime.layers import layernorm
    except Exception:
        return

    def align_modulation(value, target):
        while value.dim() > target.dim():
            squeezed = False
            for dim in range(1, value.dim() - 1):
                if value.shape[dim] == 1:
                    value = value.squeeze(dim)
                    squeezed = True
                    break
            if not squeezed:
                break
        while value.dim() < target.dim():
            value = value.unsqueeze(1)
        return value

    def apply_scale_shift(normalized, scale, shift):
        scale = align_modulation(scale, normalized)
        shift = align_modulation(shift, normalized)
        return normalized * (1 + scale) + shift

    def scale_residual_norm_scale_shift_cuda(self, residual, x, gate, shift, scale):
        if isinstance(gate, int):
            if gate != 1:
                raise ValueError(
                    f"Only gate value of 1 is supported for int type, but got {gate}"
                )
            residual_output = residual + x
        elif isinstance(gate, torch.Tensor):
            if gate.dim() == 4:
                num_frames = gate.shape[1]
                frame_seqlen = x.shape[1] // num_frames
                gated = x.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * gate
                residual_output = residual + gated.flatten(1, 2)
            else:
                residual_output = residual + x * gate
        else:
            raise ValueError(f"Gate type {type(gate)} not supported")

        normalized = self.norm(residual_output)
        modulated = apply_scale_shift(normalized, scale, shift)
        return modulated, residual_output

    def norm_scale_shift_cuda(self, x, shift, scale):
        normalized = self.norm(x)
        return apply_scale_shift(normalized, scale, shift).to(x.dtype)

    def norm_tanh_mul_add_cuda(self, x, scale, shift):
        return (self.norm(x) * torch.tanh(scale) + shift).to(x.dtype)

    scale_residual_class = getattr(layernorm, "_ScaleResidualNormScaleShift", None)
    if scale_residual_class is not None:
        scale_residual_class.forward_cuda = scale_residual_norm_scale_shift_cuda

    norm_scale_class = getattr(layernorm, "_NormScaleShift", None)
    if norm_scale_class is not None:
        norm_scale_class.forward_cuda = norm_scale_shift_cuda

    norm_tanh_class = getattr(layernorm, "_NormTanhMulAdd", None)
    if norm_tanh_class is not None:
        norm_tanh_class.forward_cuda = norm_tanh_mul_add_cuda

    def apply_rmsnorm_tanh_mul_add_native(x, gate, residual, norm):
        return residual + torch.tanh(gate) * norm(x)

    layernorm.apply_rmsnorm_tanh_mul_add = apply_rmsnorm_tanh_mul_add_native


_patch_sglang_cutedsl_norm_fallback()


def _patch_sglang_flashinfer_rope_fallback() -> None:
    enabled = os.environ.get("SGLANG_LINGBOT_DISABLE_FLASHINFER_ROPE", "").lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return

    try:
        from sglang.multimodal_gen.runtime.layers.rotary_embedding import utils
    except Exception:
        return

    utils._flashinfer_apply_rope_inplace = None
    utils.flashinfer_apply_rope_inplace = None


_patch_sglang_flashinfer_rope_fallback()


def _patch_lingbot_optional_components() -> None:
    try:
        from sglang.multimodal_gen.runtime.pipelines.lingbot_world_causal_dmd_pipeline import (
            LingBotWorldCausalDMDPipeline,
        )
    except Exception:
        return

    LingBotWorldCausalDMDPipeline._required_config_modules = [
        module
        for module in LingBotWorldCausalDMDPipeline._required_config_modules
        if module not in {"image_encoder", "image_processor"}
    ]


_patch_lingbot_optional_components()
