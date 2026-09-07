"""Benchmark FP8 Sol-Attn with Ulysses-local head counts on Hopper GPUs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from telefuser.kernel.sol_attn import sol_attn
from telefuser.ops.fp8_attention import quantize_fp8_qkv


@dataclass(frozen=True)
class Result:
    tokens: int
    global_heads: int
    sp_degree: int
    local_heads: int
    tau: float
    threshold_type: str
    kv_splits: int
    quantize_ms: float
    kernel_ms: float
    total_ms: float
    reference_ms: float | None
    cosine: float | None
    mean_abs_error: float | None


def _csv(value: str, cast: Callable[[str], object]) -> list:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _time_cuda(call: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _benchmark_case(
    *,
    tokens: int,
    global_heads: int,
    sp_degree: int,
    tau: float,
    threshold_type: str,
    kv_splits: int,
    warmup: int,
    repeats: int,
    include_reference: bool,
) -> Result:
    if global_heads % sp_degree:
        raise ValueError(f"global heads ({global_heads}) must divide SP degree ({sp_degree})")
    local_heads = global_heads // sp_degree
    q = torch.randn((1, tokens, local_heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    q_fp8, k_fp8, v_fp8, q_scale, k_scale, v_scale = quantize_fp8_qkv(q, k, v)

    def run_kernel():
        return sol_attn(
            q_fp8,
            k_fp8,
            v_fp8,
            tau=tau,
            thresh_type=threshold_type,
            kv_splits=kv_splits,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
        )

    # Compile before timing. CuTe compilation is cached per shape and split.
    output = run_kernel()
    quantize_ms = _time_cuda(lambda: quantize_fp8_qkv(q, k, v), warmup, repeats)
    kernel_ms = _time_cuda(run_kernel, warmup, repeats)

    def run_total():
        q8, k8, v8, qs, ks, vs = quantize_fp8_qkv(q, k, v)
        return sol_attn(
            q8,
            k8,
            v8,
            tau=tau,
            thresh_type=threshold_type,
            kv_splits=kv_splits,
            q_scale=qs,
            k_scale=ks,
            v_scale=vs,
        )

    total_ms = _time_cuda(run_total, warmup, repeats)
    reference_ms = cosine = mean_abs_error = None
    if include_reference:

        def run_reference():
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))

        reference = run_reference().transpose(1, 2)
        reference_ms = _time_cuda(run_reference, warmup, repeats)
        output = output.float()
        reference = reference.float()
        cosine = F.cosine_similarity(output.flatten(), reference.flatten(), dim=0).item()
        mean_abs_error = (output - reference).abs().mean().item()

    return Result(
        tokens=tokens,
        global_heads=global_heads,
        sp_degree=sp_degree,
        local_heads=local_heads,
        tau=tau,
        threshold_type=threshold_type,
        kv_splits=kv_splits,
        quantize_ms=quantize_ms,
        kernel_ms=kernel_ms,
        total_ms=total_ms,
        reference_ms=reference_ms,
        cosine=cosine,
        mean_abs_error=mean_abs_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="8192")
    parser.add_argument("--global-heads", type=int, default=40)
    parser.add_argument("--sp-degrees", default="1,2,4")
    parser.add_argument("--taus", default="1.0")
    parser.add_argument("--threshold-types", default="diag")
    parser.add_argument("--kv-splits", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("this benchmark requires an NVIDIA Hopper (SM90) GPU")
    torch.manual_seed(0)
    results = [
        _benchmark_case(
            tokens=tokens,
            global_heads=args.global_heads,
            sp_degree=sp_degree,
            tau=tau,
            threshold_type=threshold_type,
            kv_splits=kv_splits,
            warmup=args.warmup,
            repeats=args.repeats,
            include_reference=args.include_reference,
        )
        for tokens in _csv(args.tokens, int)
        for sp_degree in _csv(args.sp_degrees, int)
        for tau in _csv(args.taus, float)
        for threshold_type in _csv(args.threshold_types, str)
        for kv_splits in _csv(args.kv_splits, int)
    ]
    payload = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": [asdict(result) for result in results],
    }
    serialized = json.dumps(payload, indent=2)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
