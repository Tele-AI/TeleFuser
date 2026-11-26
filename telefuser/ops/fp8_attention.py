"""Block-scaled FP8 activation helpers for attention boundaries."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

FP8_ATTENTION_BLOCK_SIZE = 64


def _parallel_sequence_mean(x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split long reductions across SMs; preserve FP32 accumulation and input layout."""
    import triton

    from telefuser.kernel.triton.fp8_attention import _sequence_mean_finish_kernel, _sequence_mean_partials_kernel

    batch, tokens, heads, dims = x.shape
    chunks = triton.cdiv(tokens, 256)
    partials = torch.empty((chunks, batch * heads, dims), device=x.device, dtype=torch.float32)
    y_partials = torch.empty_like(partials) if y is not None else partials
    output = torch.empty((batch, heads, dims), device=x.device, dtype=torch.float32)
    y_output = torch.empty_like(output) if y is not None else output
    other = x if y is None else y
    _sequence_mean_partials_kernel[(chunks, batch * heads, triton.cdiv(dims, 32))](
        x,
        other,
        partials,
        y_partials,
        tokens,
        heads,
        dims,
        *x.stride(),
        *other.stride(),
        y is not None,
        y is None and x.stride(1) == 1,
        256,
        32,
        num_warps=4,
    )
    _sequence_mean_finish_kernel[(batch * heads, triton.cdiv(dims, 32))](
        partials,
        y_partials,
        output,
        y_output,
        tokens,
        chunks,
        batch * heads,
        dims,
        y is not None,
        32,
        num_warps=4,
    )
    return output, y_output


def _sequence_mean(x: torch.Tensor) -> torch.Tensor:
    """Reduce BTHD sequence means in FP32 without materializing an FP32 copy."""

    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("attention smoothing expects a floating-point [B, T, H, D] tensor")
    if not x.is_cuda:
        return x.float().mean(dim=1)
    if x.shape[1] >= 1024:
        return _parallel_sequence_mean(x)[0]
    import triton

    from telefuser.kernel.triton.fp8_attention import _sequence_mean_kernel

    batch, tokens, heads, head_dim = x.shape
    output = torch.empty((batch, heads, head_dim), device=x.device, dtype=torch.float32)
    block_tokens = 128
    block_dim = 32
    _sequence_mean_kernel[(batch * heads, triton.cdiv(head_dim, block_dim))](
        x,
        output,
        tokens,
        heads,
        head_dim,
        *x.stride(),
        block_tokens,
        block_dim,
        num_warps=4,
        num_stages=1,
    )
    return output


def _sequence_means(k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce matching K/V sequence means in one GPU kernel launch."""

    if k.shape != v.shape or k.ndim != 4 or not (k.is_floating_point() and v.is_floating_point()):
        raise ValueError("K/V attention smoothing expects matching floating-point [B, T, H, D] tensors")
    if k.device != v.device:
        raise ValueError("K and V must be on the same device")
    if not k.is_cuda:
        return k.float().mean(dim=1), v.float().mean(dim=1)
    if k.shape[1] >= 1024:
        return _parallel_sequence_mean(k, v)
    import triton

    from telefuser.kernel.triton.fp8_attention import _sequence_mean_kv_kernel

    batch, tokens, heads, head_dim = k.shape
    k_output = torch.empty((batch, heads, head_dim), device=k.device, dtype=torch.float32)
    v_output = torch.empty_like(k_output)
    block_tokens = 128
    block_dim = 32
    _sequence_mean_kv_kernel[(batch * heads, triton.cdiv(head_dim, block_dim))](
        k,
        v,
        k_output,
        v_output,
        tokens,
        heads,
        head_dim,
        *k.stride(),
        *v.stride(),
        block_tokens,
        block_dim,
        num_warps=4,
        num_stages=1,
    )
    return k_output, v_output


def _quantize_fp8_qkv_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    k_mean: torch.Tensor | None = None,
    v_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused SM90 QKV quantization with optional sequence-mean shifts."""

    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must share shape [B, T, H, D]")
    if not (q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("fused FP8 QKV quantization requires contiguous CUDA tensors")
    batch, tokens, heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError("fused FP8 QKV quantization requires head dimension 128")
    import triton

    from telefuser.kernel.triton.fp8_attention import _quantize_qkv_fp8_stage1, _quantize_qkv_fp8_stage2_v

    blocks = triton.cdiv(tokens, FP8_ATTENTION_BLOCK_SIZE)
    q_out = torch.empty(q.shape, device=q.device, dtype=torch.float8_e4m3fn)
    k_out = torch.empty_like(q_out)
    v_storage = torch.empty((batch, heads, head_dim, tokens), device=q.device, dtype=torch.float8_e4m3fn)
    q_scale = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    k_scale = torch.ones_like(q_scale)
    v_scale = torch.zeros((batch, heads, head_dim), device=q.device, dtype=torch.float32)
    dummy_mean = torch.empty((1,), device=q.device, dtype=torch.float32)
    grid = (blocks, batch * heads)
    _quantize_qkv_fp8_stage1[grid](
        q,
        k,
        v,
        dummy_mean if k_mean is None else k_mean,
        dummy_mean if v_mean is None else v_mean,
        q_out,
        k_out,
        q_scale,
        k_scale,
        v_scale,
        tokens,
        heads,
        head_dim,
        FP8_ATTENTION_BLOCK_SIZE,
        smooth_k=k_mean is not None,
        smooth_v=v_mean is not None,
        num_warps=8,
        num_stages=1,
    )
    _quantize_qkv_fp8_stage2_v[grid](
        v,
        dummy_mean if v_mean is None else v_mean,
        v_storage,
        v_scale,
        tokens,
        heads,
        head_dim,
        FP8_ATTENTION_BLOCK_SIZE,
        smooth_v=v_mean is not None,
        num_warps=8,
        num_stages=1,
    )
    v_out = v_storage.permute(0, 3, 1, 2)
    return q_out, k_out, v_out, q_scale, k_scale, v_scale


def quantize_fp8_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused SM90 block-scaled Q/K and layout-aware V-channel E4M3 quantization."""

    return _quantize_fp8_qkv_sm90(q, k, v)


def quantize_fp8_per_block(
    x: torch.Tensor,
    block_size: int = FP8_ATTENTION_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [B, T, H, D] tensor with one E4M3 scale per block/head."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("FP8 attention quantization expects a floating-point [B, T, H, D] tensor")
    batch, tokens, heads, head_dim = x.shape
    blocks = (tokens + block_size - 1) // block_size
    padded_tokens = blocks * block_size
    padded = F.pad(x, (0, 0, 0, 0, 0, padded_tokens - tokens))
    blocked = padded.reshape(batch, blocks, block_size, heads, head_dim)
    scale = blocked.detach().abs().amax(dim=(2, 4)).float().clamp_min(1e-6) / 448.0
    quantized = (blocked / scale.to(x.dtype)[:, :, None, :, None]).to(torch.float8_e4m3fn)
    return quantized.reshape(batch, padded_tokens, heads, head_dim)[:, :tokens].contiguous(), scale


def dequantize_fp8_per_block(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    block_size: int = FP8_ATTENTION_BLOCK_SIZE,
) -> torch.Tensor:
    """Restore block-scaled FP8 activations to ``dtype``."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    token_scale = scale.repeat_interleave(block_size, dim=1)[:, : x.shape[1]]
    return x.to(dtype) * token_scale.to(dtype).unsqueeze(-1)


def dequantize_fp8_per_token(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore token-scaled FP8 [B, T, H, D] activations."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    if scale.shape[0] != x.shape[0] or scale.shape[1] < x.shape[1] or scale.shape[2] != x.shape[2]:
        raise ValueError("scale must have shape [B, padded_T, H] with padded_T >= T")
    return x.to(dtype) * scale[:, : x.shape[1]].to(dtype).unsqueeze(-1)


def quantize_fp8_per_channel(
    x: torch.Tensor,
    *,
    token_contiguous: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BTHD with one E4M3 scale per head/channel.

    ``token_contiguous`` stores the same BTHD view over B,H,D,T-contiguous
    backing memory, matching the SM90 K-major PV WGMMA operand.
    """
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("FP8 attention quantization expects a floating-point [B, T, H, D] tensor")
    scale = x.detach().abs().amax(dim=1).float().clamp_min(1e-6) / 448.0
    quantized = (x / scale.to(x.dtype).unsqueeze(1)).to(torch.float8_e4m3fn)
    if token_contiguous:
        quantized = quantized.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    else:
        quantized = quantized.contiguous()
    return quantized, scale.contiguous()


def dequantize_fp8_per_channel(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore channel-scaled FP8 [B, T, H, D] activations."""
    if x.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn activations")
    if scale.shape != (x.shape[0], x.shape[2], x.shape[3]):
        raise ValueError("scale must have shape [B, H, D]")
    return x.to(dtype) * scale.to(dtype).unsqueeze(1)


def quantize_fp8_qkv_smoothed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    smoothing: Literal["none", "k", "kv"] = "kv",
    correct_v_bias: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Quantize QKV after attention-equivalent sequence-mean smoothing.

    Subtracting one sequence mean from every key shifts each query's logits by
    a constant, leaving softmax probabilities unchanged. Centering values is
    also equivalent when their mean is added back to the attention output.
    ``correct_v_bias`` additionally removes the mean error introduced when the
    centered values are rounded to E4M3.
    """

    if smoothing not in ("none", "k", "kv"):
        raise ValueError("FP8 attention smoothing must be 'none', 'k', or 'kv'")
    if correct_v_bias and smoothing != "kv":
        raise ValueError("FP8 value bias correction requires smoothing='kv'")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must share shape [B, T, H, D]")

    if smoothing == "kv":
        k_mean, v_mean = _sequence_means(k, v)
    else:
        k_mean = _sequence_mean(k) if smoothing == "k" else None
        v_mean = None
    native_sm90 = q.is_cuda and torch.cuda.get_device_capability(q.device) == (9, 0)
    if native_sm90:
        quantized = _quantize_fp8_qkv_sm90(q, k, v, k_mean=k_mean, v_mean=v_mean)
    else:
        centered_k = k if k_mean is None else k - k_mean.to(k.dtype).unsqueeze(1)
        centered_v = v if v_mean is None else v - v_mean.to(v.dtype).unsqueeze(1)
        q_out, q_scale = quantize_fp8_per_block(q)
        k_out, k_scale = quantize_fp8_per_block(centered_k.contiguous())
        v_out, v_scale = quantize_fp8_per_block(centered_v.contiguous())
        quantized = q_out, k_out, v_out, q_scale, k_scale, v_scale

    q_out, k_out, v_out, q_scale, k_scale, v_scale = quantized
    output_correction = None
    if v_mean is not None:
        output_correction = v_mean
        if correct_v_bias:
            if v_scale.shape == (v.shape[0], v.shape[2], v.shape[3]):
                quantized_v_mean = _sequence_mean(v_out) * v_scale
            else:
                restored_v = dequantize_fp8_per_block(v_out, v_scale, torch.bfloat16)
                quantized_v_mean = _sequence_mean(restored_v)
            output_correction = output_correction - quantized_v_mean
        output_correction = output_correction.unsqueeze(1).contiguous()
    return q_out, k_out, v_out, q_scale, k_scale, v_scale, output_correction


def apply_fp8_attention_output_correction(output: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
    """Add an FP32 per-head value correction and store back in output dtype."""

    if output.ndim != 4 or correction.shape != (output.shape[0], 1, output.shape[2], output.shape[3]):
        raise ValueError("FP8 attention correction must have shape [B, 1, H, D] for a [B, T, H, D] output")
    if correction.dtype != torch.float32 or correction.device != output.device:
        raise ValueError("FP8 attention correction must be FP32 on the output device")
    if not output.is_cuda:
        return (output.float() + correction).to(output.dtype)
    if not output.is_contiguous() or not correction.is_contiguous():
        raise ValueError("FP8 attention output and correction must be contiguous")
    import triton

    from telefuser.kernel.triton.fp8_attention import _add_output_correction_kernel

    batch, tokens, heads, head_dim = output.shape
    elements = output.numel()
    block = 256
    _add_output_correction_kernel[(triton.cdiv(elements, block),)](
        output,
        correction,
        elements,
        tokens,
        heads,
        head_dim,
        block,
        num_warps=4,
        num_stages=1,
    )
    return output


def merge_fp8_attention_prefix(
    dense_prefix: torch.Tensor,
    output: torch.Tensor,
    correction: torch.Tensor,
) -> torch.Tensor:
    """Merge a dense prefix with a corrected sparse suffix in one output pass."""

    if dense_prefix.ndim != 4 or output.ndim != 4:
        raise ValueError("attention prefix merge expects [B, T, H, D] tensors")
    if dense_prefix.shape[0] != output.shape[0] or dense_prefix.shape[2:] != output.shape[2:]:
        raise ValueError("dense prefix and sparse output must share batch, head, and head-dimension shapes")
    prefix_tokens = dense_prefix.shape[1]
    if not 0 < prefix_tokens <= output.shape[1]:
        raise ValueError("dense attention prefix must be non-empty and no longer than the sparse output")
    expected_correction = (output.shape[0], 1, output.shape[2], output.shape[3])
    if correction.shape != expected_correction or correction.dtype != torch.float32:
        raise ValueError("FP8 attention correction must be FP32 with shape [B, 1, H, D]")
    if dense_prefix.device != output.device or correction.device != output.device:
        raise ValueError("dense prefix, sparse output, and correction must be on the same device")
    if dense_prefix.dtype != output.dtype:
        raise ValueError("dense prefix and sparse output must have the same dtype")
    if not output.is_cuda:
        corrected_suffix = (output[:, prefix_tokens:].float() + correction).to(output.dtype)
        return torch.cat((dense_prefix, corrected_suffix), dim=1)
    if not (dense_prefix.is_contiguous() and output.is_contiguous() and correction.is_contiguous()):
        raise ValueError("FP8 attention prefix merge inputs must be contiguous")

    import triton

    from telefuser.kernel.triton.fp8_attention import _merge_prefix_output_correction_kernel

    merged = torch.empty_like(output)
    batch, tokens, heads, head_dim = output.shape
    elements = output.numel()
    block = 256
    _merge_prefix_output_correction_kernel[(triton.cdiv(elements, block),)](
        dense_prefix,
        output,
        correction,
        merged,
        elements,
        prefix_tokens,
        tokens,
        heads,
        head_dim,
        block,
        num_warps=4,
        num_stages=1,
    )
    return merged


__all__ = [
    "FP8_ATTENTION_BLOCK_SIZE",
    "apply_fp8_attention_output_correction",
    "dequantize_fp8_per_block",
    "dequantize_fp8_per_channel",
    "dequantize_fp8_per_token",
    "quantize_fp8_qkv",
    "quantize_fp8_qkv_smoothed",
    "merge_fp8_attention_prefix",
    "quantize_fp8_per_block",
    "quantize_fp8_per_channel",
]
