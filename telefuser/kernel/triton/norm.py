"""Triton kernels for LayerNorm and RMSNorm operations.

Adapted from sglang diffusion kernels for TeleFuser.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


def triton_autotune_configs():
    """Return autotune_configs with valid warp count for current device."""
    max_threads_per_block = 1024
    warp_size = 32
    if torch.cuda.is_available():
        try:
            warp_size = torch.cuda.get_device_properties(torch.cuda.current_device()).warp_size
        except RuntimeError:
            # Importing this module must not require an initialized CUDA driver.
            warp_size = 32
    return [
        triton.Config({}, num_warps=warp_count)
        for warp_count in [1, 2, 4, 8, 16, 32]
        if warp_count * warp_size <= max_threads_per_block
    ]


@triton.autotune(
    configs=triton_autotune_configs(),
    key=[
        "N",
        "HAS_RESIDUAL",
        "STORE_RESIDUAL_OUT",
        "IS_RMS_NORM",
        "HAS_BIAS",
        "HAS_WEIGHT",
        "HAS_X1",
        "HAS_W1",
        "HAS_B1",
    ],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X,
    Y,
    W,
    B,
    RESIDUAL,
    X1,
    W1,
    B1,
    Y1,
    RESIDUAL_OUT,
    ROWSCALE,
    SEEDS,
    DROPOUT_MASK,
    DROPOUT_MASK1,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    stride_x1_row,
    stride_y1_row,
    M,
    N,
    eps,
    dropout_p,
    zero_centered_weight,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_DROPOUT: tl.constexpr,
    STORE_DROPOUT_MASK: tl.constexpr,
    HAS_ROWSCALE: tl.constexpr,
    HAS_X1: tl.constexpr,
    HAS_W1: tl.constexpr,
    HAS_B1: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    if HAS_X1:
        X1 += row * stride_x1_row
    if HAS_W1:
        Y1 += row * stride_y1_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_ROWSCALE:
        rowscale = tl.load(ROWSCALE + row).to(tl.float32)
        x *= rowscale
    if HAS_DROPOUT:
        keep_mask = tl.rand(tl.load(SEEDS + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
        x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)
        if STORE_DROPOUT_MASK:
            tl.store(DROPOUT_MASK + row * N + cols, keep_mask, mask=cols < N)
    if HAS_X1:
        x1 = tl.load(X1 + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_ROWSCALE:
            rowscale = tl.load(ROWSCALE + M + row).to(tl.float32)
            x1 *= rowscale
        if HAS_DROPOUT:
            keep_mask = tl.rand(tl.load(SEEDS + M + row).to(tl.uint32), cols, n_rounds=7) > dropout_p
            x1 = tl.where(keep_mask, x1 / (1.0 - dropout_p), 0.0)
            if STORE_DROPOUT_MASK:
                tl.store(DROPOUT_MASK1 + row * N + cols, keep_mask, mask=cols < N)
        x += x1
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        if zero_centered_weight:
            w += 1.0
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    if HAS_WEIGHT:
        y = x_hat * w + b if HAS_BIAS else x_hat * w
    else:
        y = x_hat + b if HAS_BIAS else x_hat
    tl.store(Y + cols, y, mask=mask)
    if HAS_W1:
        w1 = tl.load(W1 + cols, mask=mask).to(tl.float32)
        if zero_centered_weight:
            w1 += 1.0
        if HAS_B1:
            b1 = tl.load(B1 + cols, mask=mask).to(tl.float32)
        y1 = x_hat * w1 + b1 if HAS_B1 else x_hat * w1
        tl.store(Y1 + cols, y1, mask=mask)


def _layer_norm_fwd_impl(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    out: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    x1: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[torch.Tensor] = None,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
    residual_out: Optional[torch.Tensor] = None,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    if x1 is not None:
        assert x1.shape == x.shape
        assert rowscale is None
        assert x1.stride(-1) == 1
    if weight1 is not None:
        assert weight1.shape == (N,)
        assert weight1.stride(-1) == 1
    if bias1 is not None:
        assert bias1.shape == (N,)
        assert bias1.stride(-1) == 1
    if rowscale is not None:
        assert rowscale.is_contiguous()
        assert rowscale.shape == (M,)
    assert out.shape == x.shape
    assert out.stride(-1) == 1
    if residual_out is not None:
        assert residual_out.shape == x.shape
        assert residual_out.stride(-1) == 1
    if weight1 is not None:
        y1 = torch.empty_like(out)
        assert y1.stride(-1) == 1
    else:
        y1 = None
    mean = torch.empty((M,), dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    if dropout_p > 0.0:
        seeds = torch.randint(2**32, (M if x1 is None else 2 * M,), device=x.device, dtype=torch.int64)
    else:
        seeds = None
    if return_dropout_mask and dropout_p > 0.0:
        dropout_mask = torch.empty(M, N, device=x.device, dtype=torch.bool)
        if x1 is not None:
            dropout_mask1 = torch.empty(M, N, device=x.device, dtype=torch.bool)
        else:
            dropout_mask1 = None
    else:
        dropout_mask, dropout_mask1 = None, None
    # 65536 bytes = 64KB, Triton's max fused operation size for a single kernel block
    # This is limited by shared memory and register file constraints on most GPUs
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(_layer_norm_fwd_1pass_kernel)[(M,)](
            x,
            out,
            weight if weight is not None else x,
            bias,
            residual,
            x1,
            weight1,
            bias1,
            y1,
            residual_out,
            rowscale,
            seeds,
            dropout_mask,
            dropout_mask1,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            x1.stride(0) if x1 is not None else 0,
            y1.stride(0) if y1 is not None else 0,
            M,
            N,
            eps,
            dropout_p,
            int(zero_centered_weight),
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            weight is not None,
            bias is not None,
            dropout_p > 0.0,
            dropout_mask is not None,
            rowscale is not None,
            HAS_X1=x1 is not None,
            HAS_W1=weight1 is not None,
            HAS_B1=bias1 is not None,
        )
    return y1, mean, rstd, seeds, dropout_mask, dropout_mask1


def _layer_norm_fwd(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    residual: Optional[torch.Tensor] = None,
    x1: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    zero_centered_weight: bool = False,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
    out: Optional[torch.Tensor] = None,
    residual_out: Optional[torch.Tensor] = None,
):
    if out is None:
        out = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    if residual is not None:
        residual_dtype = residual.dtype
    if residual_out is None and (
        residual is not None
        or (residual_dtype is not None and residual_dtype != x.dtype)
        or dropout_p > 0.0
        or rowscale is not None
        or x1 is not None
    ):
        residual_out = torch.empty_like(x, dtype=residual_dtype if residual_dtype is not None else x.dtype)
    else:
        residual_out = None
    y1, mean, rstd, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd_impl(
        x,
        weight,
        bias,
        eps,
        out,
        residual=residual,
        x1=x1,
        weight1=weight1,
        bias1=bias1,
        dropout_p=dropout_p,
        rowscale=rowscale,
        zero_centered_weight=zero_centered_weight,
        is_rms_norm=is_rms_norm,
        return_dropout_mask=return_dropout_mask,
        residual_out=residual_out,
    )
    if residual_out is None:
        residual_out = x
    return out, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1


class LayerNormFn:
    """LayerNorm function with optional residual, dropout, and parallel LayerNorm support."""

    @staticmethod
    def forward(
        x,
        weight,
        bias,
        residual=None,
        x1=None,
        weight1=None,
        bias1=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        zero_centered_weight=False,
        is_rms_norm=False,
        return_dropout_mask=False,
        out_dtype=None,
        out=None,
        residual_out=None,
    ):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1]).contiguous()
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1]).contiguous()
        if x1 is not None:
            assert x1.shape == x_shape_og
            assert rowscale is None, "rowscale is not supported with parallel LayerNorm"
            x1 = x1.reshape(-1, x1.shape[-1]).contiguous()
        if weight is not None:
            weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()
        if weight1 is not None:
            weight1 = weight1.contiguous()
        if bias1 is not None:
            bias1 = bias1.contiguous()
        if rowscale is not None:
            rowscale = rowscale.reshape(-1).contiguous()
        residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else None)
        if out is not None:
            out = out.reshape(-1, out.shape[-1])
        if residual_out is not None:
            residual_out = residual_out.reshape(-1, residual_out.shape[-1])
        y, y1, mean, rstd, residual_out, seeds, dropout_mask, dropout_mask1 = _layer_norm_fwd(
            x,
            weight,
            bias,
            eps,
            residual,
            x1,
            weight1,
            bias1,
            dropout_p=dropout_p,
            rowscale=rowscale,
            out_dtype=out_dtype,
            residual_dtype=residual_dtype,
            zero_centered_weight=zero_centered_weight,
            is_rms_norm=is_rms_norm,
            return_dropout_mask=return_dropout_mask,
            out=out,
            residual_out=residual_out,
        )
        y = y.reshape(x_shape_og)
        if residual is not None:
            residual_out = residual_out.reshape(x_shape_og)
            return y, residual_out
        return y


def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    x1=None,
    weight1=None,
    bias1=None,
    eps=1e-6,
    dropout_p=0.0,
    rowscale=None,
    prenorm=False,
    residual_in_fp32=False,
    zero_centered_weight=False,
    is_rms_norm=False,
    return_dropout_mask=False,
    out_dtype=None,
    out=None,
    residual_out=None,
):
    """Fused LayerNorm with optional residual, dropout, and parallel LayerNorm.

    Args:
        x: Input tensor of shape [*, hidden_size]
        weight: Weight tensor of shape [hidden_size]
        bias: Bias tensor of shape [hidden_size]
        residual: Optional residual tensor to add before normalization
        x1: Optional second input for parallel LayerNorm
        weight1: Optional second weight for parallel LayerNorm
        bias1: Optional second bias for parallel LayerNorm
        eps: Epsilon for numerical stability
        dropout_p: Dropout probability
        rowscale: Optional per-row scale factor
        prenorm: If True, return residual_out as well
        residual_in_fp32: Cast residual to fp32
        zero_centered_weight: If True, add 1.0 to weight
        is_rms_norm: If True, use RMSNorm instead of LayerNorm
        return_dropout_mask: If True, return dropout mask
        out_dtype: Output dtype
        out: Optional output tensor
        residual_out: Optional residual output tensor

    Returns:
        Normalized tensor, or tuple with residual_out if prenorm=True
    """
    return LayerNormFn.forward(
        x,
        weight,
        bias,
        residual,
        x1,
        weight1,
        bias1,
        eps,
        dropout_p,
        rowscale,
        prenorm,
        residual_in_fp32,
        zero_centered_weight,
        is_rms_norm,
        return_dropout_mask,
        out_dtype,
        out,
        residual_out,
    )


@triton.jit
def _norm_infer_kernel(
    X,
    Y,
    W,
    B,
    stride_x_row,
    stride_y_row,
    M,
    N,
    eps,
    IS_RMS_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Inference kernel for LayerNorm/RMSNorm without storing mean/rstd."""
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
        y = x_hat * w
    else:
        y = x_hat
    if HAS_BIAS:
        b = tl.load(B + cols, mask=cols < N, other=0.0).to(tl.float32)
        y += b
    tl.store(Y + cols, y, mask=cols < N)


def _norm_infer_kernel_call(
    x: torch.Tensor,
    out: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    M: int,
    N: int,
    eps: float,
    is_rms_norm: bool,
    has_weight: bool,
    has_bias: bool,
) -> None:
    """Call the norm_infer Triton kernel."""
    BLOCK_N = min(65536 // x.element_size(), triton.next_power_of_2(N))
    num_warps = min(max(BLOCK_N // 256, 1), 8)
    _norm_infer_kernel[(M,)](
        x,
        out,
        weight,
        bias,
        x.stride(0),
        out.stride(0),
        M,
        N,
        eps,
        IS_RMS_NORM=is_rms_norm,
        HAS_WEIGHT=has_weight,
        HAS_BIAS=has_bias,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
    )


def norm_infer(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    eps: float,
    is_rms_norm: bool = False,
    out: Optional[torch.Tensor] = None,
):
    """Inference-optimized LayerNorm/RMSNorm without storing intermediate stats.

    Args:
        x: Input tensor of shape [M, N] or [*, hidden_size]
        weight: Weight tensor of shape [hidden_size]
        bias: Bias tensor of shape [hidden_size]
        eps: Epsilon for numerical stability
        is_rms_norm: If True, use RMSNorm instead of LayerNorm
        out: Optional output tensor

    Returns:
        Normalized tensor
    """
    shape = x.shape
    x = x.contiguous()
    x_2d = x.view(-1, shape[-1])
    M, N = x_2d.shape
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.shape == (N,)
        assert bias.stride(-1) == 1
    if out is None:
        out = torch.empty_like(x_2d)
    else:
        out = out.view(-1, shape[-1])
    # 65536 bytes = 64KB, Triton's max fused operation size for a single kernel block
    # This is limited by shared memory and register file constraints on most GPUs
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    with torch.cuda.device(x_2d.device):
        _norm_infer_kernel_call(
            x_2d,
            out,
            weight if weight is not None else x_2d,
            bias if bias is not None else x_2d,
            M,
            N,
            eps,
            is_rms_norm,
            weight is not None,
            bias is not None,
        )
    return out.view(shape)


# =============================================================================
# Tiled RMSNorm (Single-pass optimization)
# =============================================================================


@triton.jit
def _rms_norm_tiled_onepass(
    y_ptr,
    x_ptr,
    w_ptr,
    SEQ: tl.constexpr,
    DIM: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_SEQ: tl.constexpr,
    BLOCK_SIZE_DIM: tl.constexpr,
):
    """Single-pass RMSNorm kernel with tiled processing.

    Args:
        y_ptr: Output tensor pointer
        x_ptr: Input tensor pointer
        w_ptr: Weight tensor pointer
        SEQ: Sequence length (compile-time constant)
        DIM: Hidden dimension (compile-time constant)
        EPS: Epsilon for numerical stability
        BLOCK_SIZE_SEQ: Block size for sequence dimension
        BLOCK_SIZE_DIM: Block size for hidden dimension
    """
    seq_blk_id = tl.program_id(0)
    seq_id = seq_blk_id * BLOCK_SIZE_SEQ

    seq_offset = seq_id + tl.arange(0, BLOCK_SIZE_SEQ)[:, None]
    s_mask = seq_offset < SEQ
    d_offset = tl.arange(0, BLOCK_SIZE_DIM)[None, :]
    d_mask = d_offset < DIM
    y_blk = y_ptr + seq_offset * DIM + d_offset
    x_blk = x_ptr + seq_offset * DIM + d_offset
    mask = s_mask & d_mask

    x = tl.load(x_blk, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=1, keep_dims=True) / DIM
    rstd = tl.math.rsqrt(mean_square + EPS)
    w = tl.load(w_ptr + d_offset, mask=d_mask)
    tl.store(y_blk, x * rstd * w, mask=mask)


def _triton_one_pass_rms_norm(
    x_view: torch.Tensor,
    y_view: torch.Tensor,
    w: torch.Tensor,
    S: int,
    D: int,
    eps: float,
    block_size_dim: int,
    block_size_seq: int,
) -> None:
    """Internal implementation for one-pass RMSNorm."""
    grid = (triton.cdiv(S, block_size_seq),)
    torch.library.wrap_triton(_rms_norm_tiled_onepass)[grid](
        y_view,
        x_view,
        w,
        S,
        D,
        eps,
        BLOCK_SIZE_DIM=block_size_dim,
        BLOCK_SIZE_SEQ=block_size_seq,
    )


def triton_one_pass_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """Single-pass RMSNorm using Triton.

    This is an optimized RMSNorm implementation that processes the entire tensor
    in a single pass with tiled processing for better memory access patterns.

    Args:
        x: Input tensor of shape [*, hidden_size]
        w: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor with same shape as input
    """
    shape = x.shape
    x = x.contiguous()
    w = w.contiguous()
    y = torch.empty_like(x)
    x_view = x.reshape(-1, shape[-1])
    y_view = y.reshape(-1, shape[-1])
    S, D = x_view.shape

    # Tuning: target ~512 rows per block for good occupancy, capped at 16 blocks max
    # Larger block counts increase kernel launch overhead without much benefit
    BLOCK_SIZE_SEQ = min(16, triton.next_power_of_2(max(1, S // 512)))

    with torch.cuda.device(x.device):
        _triton_one_pass_rms_norm(
            x_view,
            y_view,
            w,
            S,
            D,
            eps,
            triton.next_power_of_2(D),
            BLOCK_SIZE_SEQ,
        )
    return y


# =============================================================================
# Fused Add + RMSNorm
# =============================================================================


@triton.jit
def _fused_add_rms_norm_kernel(
    Y,
    X,
    W,
    Residual,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused Add + RMSNorm kernel.

    Args:
        Y: Output tensor pointer
        X: Input tensor pointer
        W: Weight tensor pointer
        Residual: Residual tensor pointer (input and output)
        stride: Stride for moving to next row
        N: Hidden dimension size
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size for processing
    """
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    Residual += row * stride

    cols = tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    residual = tl.load(Residual + cols, mask=cols < N, other=0.0).to(tl.float32)

    x = x + residual
    tl.store(Residual + cols, x, mask=cols < N)

    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=cols < N, other=1.0).to(tl.float32)
    y = x * rstd * w

    tl.store(Y + cols, y, mask=cols < N)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Add + RMSNorm operation.

    Computes: residual = x + residual, y = rms_norm(residual, weight)
    This is more efficient than separate add and normalization.

    Args:
        x: Input tensor of shape [*, hidden_size]
        residual: Residual tensor (will be modified in-place)
        weight: Weight tensor of shape [hidden_size]
        eps: Epsilon for numerical stability

    Returns:
        Tuple of (normalized output, updated residual)
    """
    shape = x.shape
    x = x.contiguous()
    weight = weight.contiguous()
    residual = residual.contiguous()

    y = torch.empty_like(x)

    x_2d = x.view(-1, shape[-1])
    y_2d = y.view(-1, shape[-1])
    residual_2d = residual.view(-1, shape[-1])
    M, N = x_2d.shape

    # 65536 bytes = 64KB, Triton's max fused operation size for a single kernel block
    # This is limited by shared memory and register file constraints on most GPUs
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))

    if N > BLOCK_SIZE:
        raise RuntimeError(f"This RMSNorm doesn't support feature dim >= {BLOCK_SIZE}.")

    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    _fused_add_rms_norm_kernel[(M,)](
        y_2d,
        x_2d,
        weight,
        residual_2d,
        x_2d.stride(0),
        N,
        eps,
        BLOCK_SIZE,
        num_warps=num_warps,
    )

    return y, residual
