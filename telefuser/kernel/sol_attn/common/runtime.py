"""Small host helpers shared by the architecture backends."""

from cutlass.cute.runtime import from_dlpack


def to_cute_tensor(tensor):
    leading_dim = tensor.ndim - 1
    if tensor.stride(leading_dim) != 1:
        leading_dim = next(i for i, stride in enumerate(tensor.stride()) if stride == 1)
    return from_dlpack(
        tensor,
        assumed_align=16,
        enable_tvm_ffi=True,
    ).mark_layout_dynamic(leading_dim=leading_dim)


__all__ = ["to_cute_tensor"]
