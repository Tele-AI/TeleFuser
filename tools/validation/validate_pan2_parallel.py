# SPDX-License-Identifier: Apache-2.0
"""Validate actual tiny-model distributed and quantized execution against dense execution.

Launch with torchrun. These are numerical integration fixtures, not full-checkpoint
performance or generated-image quality benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from telefuser.core.config import AttentionConfig, ParallelConfig, QuantConfig, QuantKernelBackend, QuantType
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.models.pan2_dit import PAN2DiT
from telefuser.ops.attention import attention_impl
from telefuser.ops.fp8_gemm import FP8Linear

_original_warning = attention_impl.logger.warning


def _strict_attention_warning(message: str, *args: object, **kwargs: object) -> None:
    if "Sol-Attn execution failed" in message or ("SOL_ATTN" in message and "falling back" in message):
        detail = message % args if args else message
        raise AssertionError(f"Requested Sol-Attn did not execute: {detail}")
    _original_warning(message, *args, **kwargs)


attention_impl.logger.warning = _strict_attention_warning


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pan2"], default="pan2")
    parser.add_argument("--sp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--sol", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--sol-fp8", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    if world != args.sp * args.tp:
        raise ValueError("world size must equal SP times TP")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    try:
        torch.manual_seed(2026)
        generator = torch.Generator(device="cpu").manual_seed(71)
        model = PAN2DiT(
            in_channels=5,
            out_channels=2,
            num_attention_heads=4,
            attention_head_dim=128,
            num_layers=2,
            num_refiner_layers=1,
            text_embed_dim=32,
            rope_axes_dim=(16, 56, 56),
            mlp_ratio=2,
            context_mlp_ratio=2,
            refiner_mlp_ratio=2,
        )
        model.to(device=device, dtype=torch.bfloat16).eval()
        height, width = (16, 32) if args.sol else (6, 6)
        inputs = dict(
            hidden_states=torch.randn(1, 5, 3, height, width, generator=generator).to(device, dtype=torch.bfloat16),
            timestep=torch.tensor([500.0], device=device),
            encoder_hidden_states=torch.randn(1, 5, 32, generator=generator).to(device, dtype=torch.bfloat16),
            sparse_step_index=3,
        )

        def call():
            return model(**inputs).sample

        with torch.inference_mode():
            expected = call().float()
        mesh = create_device_mesh_from_config(
            ParallelConfig(device_ids=list(range(world)), tp_degree=args.tp, sp_ulysses_degree=args.sp)
        )
        if args.tp > 1:
            model.enable_tp(mesh)
        if args.sp > 1:
            model.enable_usp(mesh)
        if args.sol:
            model.set_attention_config(
                AttentionConfig.sol_attention(
                    dense_timesteps=0, dense_layers=0, tau=1.0, threshold_type="exact", sol_fp8=args.sol_fp8
                )
            )
        if args.fp8:
            model.enable_quant(
                QuantConfig(enabled=True, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL)
            )
        with torch.inference_mode():
            actual = call().float()
        if args.offload:
            for _ in range(2):
                model.offload_device(pin_memory=False)
                model.onload_device(device, non_blocking=False)
                with torch.inference_mode():
                    repeated = call().float()
                torch.testing.assert_close(repeated, actual, rtol=0, atol=0)
        torch.cuda.synchronize(device)
        error = actual - expected
        relative = float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-08))
        fp8_modules = [m for m in model.modules() if isinstance(m, FP8Linear)]
        cached = sum((m._fp8_weight is not None and m._fp8_weight.dtype == torch.float8_e4m3fn for m in fp8_modules))
        if args.fp8 and (not cached):
            raise AssertionError("FP8 requested but no FP8 weight cache was materialized")
        limit = 0.15 if args.sol else 0.06 if args.fp8 else 0.02
        if not torch.isfinite(actual).all() or relative > limit:
            raise AssertionError(f"numerical error {relative:.6f} exceeds fixture tolerance {limit}")
        record = dict(
            model=args.model,
            repeated_offload=args.offload,
            sp=args.sp,
            tp=args.tp,
            sol=args.sol,
            fp8=args.fp8,
            sol_fp8=args.sol_fp8,
            relative_l2=relative,
            max_abs=float(error.abs().max()),
            output_shape=list(actual.shape),
            fp8_modules=len(fp8_modules),
            materialized_fp8_modules=cached,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            scope="tiny-model numerical validation, not full-checkpoint quality",
        )
        records = [None] * world
        dist.all_gather_object(records, record)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
