# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tiny utility to enable tf-kernel FP8 GEMM (W8A8) for arbitrary PyTorch models.

What it does
- Replaces nn.Linear modules with a drop-in module that:
  - quantizes activations dynamically per forward call
  - quantizes weights lazily on first CUDA forward (and caches them)
  - dispatches per-token quantization and scaled GEMM through tf-kernel

Notes
- CUDA-only fast path; CPU (and unsupported cases) automatically fall back to
  the original nn.Linear.
- Output of tf-kernel FP8 GEMM is fp16/bf16. If your input is fp32, you can either
  keep fp32 (fallback) or enable casting to fp16/bf16 for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.nn as nn

try:
    import tf_kernel
except ImportError:
    tf_kernel = None


@dataclass(frozen=True)
class FP8GemmOptions:
    # If True, non-fp16/bf16 inputs will be cast to fp16 for the FP8 GEMM path.
    # If False, non-fp16/bf16 inputs will fall back to the original nn.Linear.
    cast_inputs: bool = True

    # If True, the output will be cast back to the original input dtype when
    # we cast inputs for the fast path.
    cast_output_back: bool = True

    # What to do with the original (FP16/BF16) weights after wrapping.
    #
    # - "keep": keep original weights inside the wrapped module (default).
    # - "cpu_offload": move original weights to CPU to save GPU VRAM; keep them
    #   for potential CPU fallback and/or re-quantization.
    # - "discard": do not keep original weights after FP8 weights are
    #   materialized (lowest steady-state memory). In this mode, CPU fallback
    #   is not available and weights cannot be re-quantized if the FP8 cache is
    #   invalidated.
    fp16_weight_storage: Literal["keep", "cpu_offload", "discard"] = "discard"

    # If True, try to quantize weights immediately while wrapping (only works
    # when the original nn.Linear weights are already on CUDA). This enables
    # discarding/offloading FP16 weights right away, instead of waiting for the
    # first forward pass.
    materialize_fp8_on_wrap: bool = True


class FP8Linear(nn.Module):
    """Drop-in replacement for nn.Linear that uses tf-kernel FP8 GEMM."""

    def __init__(self, linear: nn.Linear, *, options: FP8GemmOptions):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear)}")
        if tf_kernel is None:
            raise ImportError("tf-kernel is required to enable FP8 GEMM")

        if options.fp16_weight_storage not in ("keep", "cpu_offload", "discard"):
            raise ValueError(
                f"invalid fp16_weight_storage={options.fp16_weight_storage!r}; "
                "expected one of {'keep','cpu_offload','discard'}"
            )
        if options.fp16_weight_storage == "discard" and not options.cast_inputs:
            # Without FP16 weights, we cannot fall back for non-fp16/bf16 inputs.
            raise ValueError(
                "fp16_weight_storage='discard' requires cast_inputs=True "
                "(otherwise non-fp16/bf16 inputs would need FP16 fallback)."
            )

        # Keep the original nn.Linear module only in "keep" mode.
        self.linear: Optional[nn.Linear] = linear if options.fp16_weight_storage == "keep" else None
        self.options = options

        # Optional CPU copies for fallback and/or re-quantization.
        self._fp16_weight_cpu: Optional[torch.Tensor] = None  # [N, K], fp16
        self._fp16_bias_cpu: Optional[torch.Tensor] = None  # [N], fp16

        # Bias for the fast path when we are not keeping the original Linear.
        # (In "keep" mode we rely on self.linear.bias.)
        self.bias: Optional[nn.Parameter] = None
        if options.fp16_weight_storage != "keep":
            self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None
            # Stash FP16 weights on CPU to immediately free GPU VRAM. We keep
            # them until FP8 weights are materialized, then optionally discard.
            self._fp16_weight_cpu = linear.weight.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            if linear.bias is not None:
                self._fp16_bias_cpu = linear.bias.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()

        self._tf_kernel = tf_kernel

        # Lazy weight cache (per-device). Register these as non-persistent
        # buffers so module.to()/cpu()/cuda() also migrates the FP8 cache.
        self.register_buffer("_fp8_weight", None, persistent=False)  # [K, N] view
        self.register_buffer("_fp8_weight_scale", None, persistent=False)  # scalar or vec
        self._weight_cache_device: Optional[torch.device] = None

        # Track when weights change (best-effort) in "keep" mode.
        # Users can also call invalidate_weight_cache() explicitly after weight updates.
        self._last_weight_version: Optional[int] = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, *, options: FP8GemmOptions) -> "FP8Linear":
        # In "keep" mode, we keep the original Linear module instance so
        # state_dict stays natural (weights/bias remain at linear.weight / linear.bias).
        return cls(linear, options=options)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]

        if self.linear is not None:
            src_weight = self.linear.weight.detach()
            src_bias = self.linear.bias.detach() if self.linear.bias is not None else None
        elif self._fp16_weight_cpu is not None:
            src_weight = self._fp16_weight_cpu.detach()
            src_bias = self._fp16_bias_cpu.detach() if self._fp16_bias_cpu is not None else None
        else:
            raise RuntimeError("FP8Linear cannot be deep-copied without an FP16 weight source.")

        linear = nn.Linear(
            in_features=src_weight.shape[1],
            out_features=src_weight.shape[0],
            bias=src_bias is not None,
            device=src_weight.device,
            dtype=src_weight.dtype,
        )
        linear.weight.data.copy_(src_weight)
        if src_bias is not None:
            linear.bias.data.copy_(src_bias)

        cloned = FP8Linear(linear, options=self.options)
        memo[id(self)] = cloned

        if self._fp16_weight_cpu is not None:
            cloned._fp16_weight_cpu = self._fp16_weight_cpu.detach().clone()
        if self._fp16_bias_cpu is not None:
            cloned._fp16_bias_cpu = self._fp16_bias_cpu.detach().clone()

        if self._fp8_weight is not None:
            cloned._fp8_weight = self._fp8_weight.detach().clone()
        if self._fp8_weight_scale is not None:
            cloned._fp8_weight_scale = self._fp8_weight_scale.detach().clone()

        cloned._weight_cache_device = self._weight_cache_device
        cloned._last_weight_version = self._last_weight_version
        return cloned

    def invalidate_weight_cache(self) -> None:
        self._fp8_weight = None
        self._fp8_weight_scale = None
        self._weight_cache_device = None
        self._last_weight_version = None

    def _cached_fp8_device(self) -> Optional[torch.device]:
        if self._fp8_weight is None or self._fp8_weight_scale is None:
            return None
        if self._fp8_weight.device != self._fp8_weight_scale.device:
            return None
        return self._fp8_weight.device

    def materialize_fp8_weight(self, device: torch.device) -> None:
        """Force FP8 weight materialization on the given device."""
        self._maybe_requantize_weight(device)

    def _maybe_requantize_weight(self, device: torch.device) -> None:
        # Detect weight changes (best-effort) and/or device changes.
        cache_device = self._cached_fp8_device()
        version: Optional[int] = None
        if self.linear is not None:
            weight = self.linear.weight
            v = getattr(weight, "_version", None)
            version = v if isinstance(v, int) else None
            if (
                self._fp8_weight is not None
                and self._fp8_weight_scale is not None
                and cache_device == device
                and (version is None or version == self._last_weight_version)
            ):
                return
        else:
            if self._fp8_weight is not None and self._fp8_weight_scale is not None and cache_device == device:
                return

        # Quantize each output row, then pass the transposed [K, N] view to the
        # scaled GEMM kernel.
        if self.linear is not None:
            w_src = self.linear.weight.detach()
        elif self._fp16_weight_cpu is not None:
            w_src = self._fp16_weight_cpu
        else:
            raise RuntimeError(
                "FP8Linear has no FP16 weight source available to (re)quantize. "
                "This can happen if fp16_weight_storage='discard' and the FP8 cache was "
                "invalidated."
            )

        w_n_k = w_src.to(device=device, dtype=torch.bfloat16, non_blocking=True).contiguous()

        qweight_n_k = torch.empty_like(w_n_k, dtype=torch.float8_e4m3fn)
        w_scale = torch.empty((w_n_k.shape[0], 1), dtype=torch.float32, device=device)
        self._tf_kernel.tf_per_token_quant_fp8(w_n_k, qweight_n_k, w_scale)
        self._fp8_weight = qweight_n_k.t()
        self._fp8_weight_scale = w_scale
        self._weight_cache_device = self._cached_fp8_device()
        self._last_weight_version = version

        # If requested, discard FP16 weights once FP8 is materialized.
        if self.options.fp16_weight_storage == "discard":
            self._fp16_weight_cpu = None
            self._fp16_bias_cpu = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CPU / non-CUDA: fall back.
        if not x.is_cuda:
            if self.linear is not None:
                return self.linear(x)
            if self._fp16_weight_cpu is not None:
                bias = self._fp16_bias_cpu
                return torch.nn.functional.linear(
                    x,
                    self._fp16_weight_cpu.to(dtype=x.dtype),  # type: ignore[arg-type]
                    bias.to(dtype=x.dtype) if bias is not None else None,
                )
            raise RuntimeError(
                "FP8Linear cannot run on CPU because FP16 weights are not kept. "
                "Use fp16_weight_storage='cpu_offload' (or 'keep') for CPU fallback."
            )

        if x.dtype not in (torch.float16, torch.bfloat16) and not self.options.cast_inputs:
            if self.linear is not None:
                return self.linear(x)
            if self._fp16_weight_cpu is not None:
                weight = self._fp16_weight_cpu.to(device=x.device, dtype=x.dtype)
                bias = self._fp16_bias_cpu
                bias = bias.to(device=x.device, dtype=x.dtype) if bias is not None else None
                return torch.nn.functional.linear(x, weight, bias)
            raise RuntimeError("cast_inputs=False requires FP16 weights for fallback, but they were discarded.")

        x_fp, in_dtype, out_dtype = self._prepare_cuda_input(x)
        x_shape = x_fp.shape
        x_2d = x_fp.reshape(-1, x_shape[-1]).contiguous()
        qinput = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
        input_scale = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x_fp.device)
        self._tf_kernel.tf_per_token_quant_fp8(x_2d, qinput, input_scale)
        return self._forward_quantized(qinput, input_scale, x_shape, in_dtype, out_dtype)

    def _prepare_cuda_input(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.dtype, torch.dtype]:
        """Cast an activation to a dtype supported by tf-kernel FP8 GEMM."""
        in_dtype = x.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            x_fp = x.to(torch.bfloat16)
            out_dtype = torch.bfloat16
        else:
            x_fp = x
            out_dtype = in_dtype
        return x_fp, in_dtype, out_dtype

    def _forward_quantized(
        self,
        qinput: torch.Tensor,
        input_scale: torch.Tensor,
        x_shape: torch.Size,
        in_dtype: torch.dtype,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run this Linear using an already quantized shared activation."""
        self._maybe_requantize_weight(qinput.device)

        if self.linear is not None:
            bias = self.linear.bias
        else:
            bias = self.bias
        if bias is not None:
            if bias.device != qinput.device:
                bias = bias.to(device=qinput.device, non_blocking=True)
            if bias.dtype != out_dtype:
                bias = bias.to(dtype=out_dtype)

        y = self._tf_kernel.fp8_scaled_mm(
            qinput,
            self._fp8_weight,
            input_scale,
            self._fp8_weight_scale,
            out_dtype,
            bias,
        )
        y = y.reshape(*x_shape[:-1], y.shape[-1])

        if self.options.cast_inputs and self.options.cast_output_back and y.dtype != in_dtype:
            return y.to(in_dtype)
        return y


def fp8_linear_forward_many(linears: tuple[FP8Linear, ...], x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Reuse one dynamic activation quantization across compatible FP8 Linears."""
    if not linears:
        return ()
    first = linears[0]
    unsupported_no_cast = x.dtype not in (torch.float16, torch.bfloat16) and not first.options.cast_inputs
    if not x.is_cuda or unsupported_no_cast or any(linear.options != first.options for linear in linears[1:]):
        return tuple(linear(x) for linear in linears)

    x_fp, in_dtype, out_dtype = first._prepare_cuda_input(x)
    x_shape = x_fp.shape
    x_2d = x_fp.reshape(-1, x_shape[-1]).contiguous()
    qinput = torch.empty_like(x_2d, dtype=torch.float8_e4m3fn)
    input_scale = torch.empty((x_2d.shape[0], 1), dtype=torch.float32, device=x_fp.device)
    first._tf_kernel.tf_per_token_quant_fp8(x_2d, qinput, input_scale)
    return tuple(linear._forward_quantized(qinput, input_scale, x_shape, in_dtype, out_dtype) for linear in linears)


def enable_fp8_gemm(
    model: nn.Module,
    *,
    options: FP8GemmOptions = FP8GemmOptions(),
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
    inplace: bool = True,
) -> nn.Module:
    """
    Replace nn.Linear modules in a model with FP8Linear to accelerate GEMMs.

    Args:
        model: Any torch.nn.Module.
        options: FP8GemmOptions controlling casting / fallback behavior.
        module_filter: Optional predicate (name, module) -> bool to decide
            whether to wrap a given module. If None, wraps all nn.Linear.
        inplace: If True, modifies model in-place and returns it.

    Returns:
        The modified model (same object if inplace=True).
    """
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    def should_wrap(name: str, m: nn.Module) -> bool:
        if not isinstance(m, nn.Linear):
            return False
        if module_filter is None:
            return True
        return bool(module_filter(name, m))

    def _recurse(prefix: str, parent: nn.Module) -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if should_wrap(full_name, child):
                fp8_mod = FP8Linear.from_linear(child, options=options)
                # Optionally materialize immediately while the original weight is
                # already on CUDA, so we can discard/offload FP16 weights right away.
                if options.materialize_fp8_on_wrap and child.weight.is_cuda:
                    fp8_mod.materialize_fp8_weight(child.weight.device)
                setattr(parent, child_name, fp8_mod)
            else:
                _recurse(full_name, child)

    _recurse("", model)
    return model


def count_linear_layers(
    model: nn.Module,
    *,
    module_filter: Optional[Callable[[str, nn.Module], bool]] = None,
) -> int:
    """Count the ``nn.Linear`` modules that ``enable_fp8_gemm`` would wrap."""

    def _count(prefix: str, parent: nn.Module) -> int:
        count = 0
        for child_name, child in parent.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if module_filter is None or module_filter(full_name, child):
                    count += 1
            else:
                count += _count(full_name, child)
        return count

    return _count("", model)
