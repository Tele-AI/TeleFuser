"""Public Sol-Attn interface."""

from __future__ import annotations

import functools

import torch

BLOCK_SIZE = 64
_CUTE_BACKENDS = {
    (9, 0): "cute_sm90",
    (10, 0): "cute_sm100",
    (12, 0): "cute_sm120",
}
_compiled = {}


def _is_token_contiguous_bthd(x: torch.Tensor) -> bool:
    batch, tokens, heads, head_dim = x.shape
    return x.stride() == (
        heads * head_dim * tokens,
        1,
        head_dim * tokens,
        tokens,
    )


def _to_token_contiguous_bthd(x: torch.Tensor) -> torch.Tensor:
    if _is_token_contiguous_bthd(x):
        return x
    return x.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)


def _validate_inputs(
    q,
    k,
    v,
    thresh_type,
    sink_tokens=0,
    sink_start=None,
):
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must share shape [B, T, H, 128]")
    if q.shape[1] == 0 or q.shape[3] != 128:
        raise ValueError("Sol-Attn requires T > 0 and head dimension 128")
    if any(x.dtype not in (torch.bfloat16, torch.float8_e4m3fn) for x in (q, k, v)):
        raise TypeError("q, k, and v must use torch.bfloat16 or torch.float8_e4m3fn")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    v_layout_valid = v.is_contiguous() or (v.dtype == torch.float8_e4m3fn and _is_token_contiguous_bthd(v))
    if not (q.is_contiguous() and k.is_contiguous() and v_layout_valid):
        raise ValueError("q and k must be contiguous BTHD; FP8 v may also be token-contiguous BTHD")
    if thresh_type not in ("diag", "exact"):
        raise ValueError("thresh_type must be 'diag' or 'exact'")
    if not isinstance(sink_tokens, int):
        raise TypeError("sink_tokens must be an integer")
    if not 0 <= sink_tokens <= q.shape[1]:
        raise ValueError("sink_tokens must be in [0, T]")
    if sink_start is not None:
        if not isinstance(sink_start, int):
            raise TypeError("sink_start must be an integer or None")
        if not 0 <= sink_start <= q.shape[1]:
            raise ValueError("sink_start must be in [0, T]")
        if sink_start + sink_tokens > q.shape[1]:
            raise ValueError("sink_start + sink_tokens must be <= T")

    return tuple(torch.cuda.get_device_capability(q.device))


@functools.lru_cache(maxsize=1)
def _cute_runtime_available() -> bool:
    """Whether the optional CuTe DSL runtime can be imported."""

    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _backend_for_arch(
    arch: tuple[int, int],
    *,
    cute_available: bool | None = None,
) -> str:
    """Select CuTe when specialized and available, otherwise Triton."""

    if arch[0] < 8:
        raise RuntimeError(f"Sol-Attn requires an NVIDIA GPU with compute capability >= 8.0; got SM{arch[0]}{arch[1]}")
    cute_backend = _CUTE_BACKENDS.get(arch)
    if cute_backend is not None:
        available = _cute_runtime_available() if cute_available is None else cute_available
        if available:
            return cute_backend
    return "triton"


def _validate_cute(arch, tokens, kv_splits):
    if arch != (9, 0) and kv_splits != 1:
        raise ValueError("kv_splits=2/4 is currently available on SM90 only")
    route_groups = ((tokens + 63) // 64 + 63) // 64
    if kv_splits > route_groups:
        raise ValueError("each KV split must contain at least one N64 route group")


def _stream(device):
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)


def _to_cute_tensors(tensors):
    from .common import to_cute_tensor

    return [to_cute_tensor(x) for x in tensors]


def _sink_block_range(tokens, sink_start, sink_tokens):
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if not sink_tokens:
        return blocks, blocks
    start = tokens - sink_tokens if sink_start is None else sink_start
    return (
        start // BLOCK_SIZE,
        (start + sink_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE,
    )


def _compile_sm90(
    key,
    tensors,
    scale,
    tokens,
    kv_splits,
    sink_range,
    stream,
    fp8_inputs,
):
    import cutlass.cute as cute

    from .sm90 import make_kernel

    operator = make_kernel(tokens, kv_splits, fp8_inputs=fp8_inputs)
    args = _to_cute_tensors(tensors)
    compiled = cute.compile(
        operator,
        *args,
        scale,
        sink_range,
        tokens,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _compile_sm100(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
):
    import cutlass.cute as cute

    from .sm100 import forward

    args = _to_cute_tensors(tensors)
    compiled = cute.compile(
        forward,
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _compile_sm120(
    key,
    tensors,
    scale,
    sink_start_block,
    sink_end_block,
    stream,
):
    import cutlass.cute as cute

    from .sm120 import make_kernel

    operator = make_kernel()
    args = _to_cute_tensors(tensors)
    compiled = cute.compile(
        operator,
        *args,
        scale,
        sink_start_block,
        sink_end_block,
        stream=stream,
        options="--enable-tvm-ffi",
    )
    _compiled[key] = compiled
    return compiled, args


def _sol_attn_cute(
    q,
    k,
    v,
    *,
    arch,
    scale,
    tau,
    thresh_type,
    kv_splits,
    sink_tokens,
    sink_start,
    q_scale=None,
    k_scale=None,
    v_scale=None,
):
    batch, tokens, heads, _ = q.shape
    fp8_inputs = q.dtype == torch.float8_e4m3fn

    with torch.cuda.device(q.device):
        if fp8_inputs:
            from .triton_ref.preprocess import prepare_sm90_fp8

            kc, vc, threshold, kc_scale = prepare_sm90_fp8(
                q,
                k,
                v,
                scale=scale,
                tau=tau,
                thresh_type=thresh_type,
                tokens=tokens,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            from .preprocess import prepare

            kc, vc, threshold = prepare(
                q,
                k,
                v,
                scale=scale,
                tau=tau,
                thresh_type=thresh_type,
            )
            dummy_scale = torch.ones((1,), device=q.device, dtype=torch.float32)
            q_scale = k_scale = v_scale = kc_scale = dummy_scale
        if fp8_inputs and tokens % BLOCK_SIZE:
            padded_tokens = ((tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
            q_padded = torch.zeros(
                (batch, padded_tokens, heads, q.shape[-1]),
                device=q.device,
                dtype=q.dtype,
            )
            k_padded = torch.zeros_like(q_padded)
            q_padded[:, :tokens].copy_(q)
            k_padded[:, :tokens].copy_(k)
            v_storage = torch.zeros(
                (batch, heads, v.shape[-1], padded_tokens),
                device=v.device,
                dtype=v.dtype,
            )
            v_storage[..., :tokens].copy_(v.permute(0, 2, 3, 1))
            q, k = q_padded, k_padded
            v = v_storage.permute(0, 3, 1, 2)
        output = torch.empty(v.shape, device=v.device, dtype=torch.bfloat16)
        lse = torch.empty(
            (batch, q.shape[1], heads),
            device=q.device,
            dtype=torch.float32,
        )
        stream = _stream(q.device)
        key = (q.device.index, arch, batch, tokens, heads, kv_splits, q.dtype)

        if arch == (9, 0):
            if sink_tokens:
                sink_start_block, sink_end_block = _sink_block_range(
                    tokens,
                    sink_start,
                    sink_tokens,
                )
                sink_range = sink_start_block | (sink_end_block << 16)
            else:
                sink_range = 0
            tensors = [q, k, v, output, kc, vc, threshold, lse, q_scale, k_scale, v_scale, kc_scale]
            if kv_splits > 1:
                tensors.extend(
                    [
                        torch.empty(
                            (batch, q.shape[1], kv_splits * heads, 128),
                            device=q.device,
                            dtype=torch.bfloat16,
                        ),
                        torch.empty(
                            (batch, q.shape[1], kv_splits * heads),
                            device=q.device,
                            dtype=torch.float32,
                        ),
                    ]
                )
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm90(
                    key,
                    tensors,
                    scale,
                    tokens,
                    kv_splits,
                    sink_range,
                    stream,
                    fp8_inputs,
                )
            else:
                args = _to_cute_tensors(tensors)
            compiled(
                *args,
                scale,
                sink_range,
                tokens,
                stream=stream,
            )
        elif arch == (10, 0):
            sink_start_block, sink_end_block = _sink_block_range(
                tokens,
                sink_start,
                sink_tokens,
            )
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm100(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                )
            else:
                args = _to_cute_tensors(tensors)
            compiled(
                *args,
                scale,
                sink_start_block,
                sink_end_block,
                stream=stream,
            )
        else:
            sink_start_block, sink_end_block = _sink_block_range(
                tokens,
                sink_start,
                sink_tokens,
            )
            tensors = [q, k, v, output, kc, vc, threshold, lse]
            compiled = _compiled.get(key)
            if compiled is None:
                compiled, args = _compile_sm120(
                    key,
                    tensors,
                    scale,
                    sink_start_block,
                    sink_end_block,
                    stream,
                )
            else:
                args = _to_cute_tensors(tensors)
            compiled(
                *args,
                scale,
                sink_start_block,
                sink_end_block,
                stream=stream,
            )
    return output[:, :tokens]


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    tau: float = 1.0,
    thresh_type: str = "diag",
    kv_splits: int = 1,
    sink_tokens: int = 0,
    sink_start: int | None = None,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute noncausal Sol-Attn for contiguous BF16 or FP8 BTHD tensors.

    ``sink_start`` and ``sink_tokens`` keep every KV block overlapping the
    corresponding contiguous token range exact for all queries. Omitting
    ``sink_start`` places the range at the token suffix.
    """

    fp8_inputs = any(x.dtype == torch.float8_e4m3fn for x in (q, k, v))
    if fp8_inputs:
        if kv_splits not in (1, 2, 4):
            raise ValueError("kv_splits must be 1, 2, or 4")
        if not all(x.dtype == torch.float8_e4m3fn for x in (q, k, v)):
            raise TypeError("q, k, and v must all use the same dtype")
        if any(scale is None for scale in (q_scale, k_scale, v_scale)):
            raise ValueError("FP8 Sol-Attn requires q_scale, k_scale, and v_scale")
        _validate_inputs(q, k, v, thresh_type, sink_tokens, sink_start)
        arch = tuple(torch.cuda.get_device_capability(q.device))
        native_sm90_fp8 = arch == (9, 0) and _cute_runtime_available()
        blocks = (q.shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
        expected_scale_shape = (
            (q.shape[0], blocks * BLOCK_SIZE, q.shape[2])
            if native_sm90_fp8
            else (q.shape[0], blocks, q.shape[2])
        )
        for name, tensor in (("q_scale", q_scale), ("k_scale", k_scale)):
            if tensor.shape != expected_scale_shape or tensor.device != q.device:
                raise ValueError(f"{name} must have shape {expected_scale_shape} on the Q/K/V device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if native_sm90_fp8:
            expected_v_scale_shape = (q.shape[0], q.shape[2], q.shape[3])
            if v_scale.shape != expected_v_scale_shape or v_scale.device != q.device:
                raise ValueError("SM90 FP8 Sol-Attn requires v_scale with shape [B, H, D]")
            if not v_scale.is_contiguous():
                raise ValueError("v_scale must be contiguous")
            v = _to_token_contiguous_bthd(v)
            scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
            return _sol_attn_cute(
                q,
                k,
                v,
                arch=arch,
                scale=scale,
                tau=float(tau),
                thresh_type=thresh_type,
                kv_splits=kv_splits,
                sink_tokens=sink_tokens,
                sink_start=sink_start,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        if v_scale.shape != expected_scale_shape or v_scale.device != q.device:
            raise ValueError(f"v_scale must have shape {expected_scale_shape} on the Q/K/V device")
        if not v_scale.is_contiguous():
            raise ValueError("v_scale must be contiguous")
        from .triton_ref import sol_attn as triton_sol_attn

        return triton_sol_attn(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    arch = _validate_inputs(
        q,
        k,
        v,
        thresh_type,
        sink_tokens,
        sink_start,
    )
    if kv_splits not in (1, 2, 4):
        raise ValueError("kv_splits must be 1, 2, or 4")
    backend = _backend_for_arch(arch)
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    tau = float(tau)

    if backend == "triton":
        if kv_splits != 1:
            raise ValueError("kv_splits=2/4 is currently available on SM90 only")
        from .triton_ref import sol_attn as triton_sol_attn

        return triton_sol_attn(
            q,
            k,
            v,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
        )

    _validate_cute(arch, q.shape[1], kv_splits)
    return _sol_attn_cute(
        q,
        k,
        v,
        arch=arch,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        kv_splits=kv_splits,
        sink_tokens=sink_tokens,
        sink_start=sink_start,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )


__all__ = ["sol_attn"]
