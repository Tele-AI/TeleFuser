# SPDX-License-Identifier: Apache-2.0
"""Compare repeated ParallelWorker sampling with a single-device tiny-model baseline.

This exercises request transport, conditioning, all sampling steps and CPU offload.
It does not load released weights or measure generated-image/video quality.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from telefuser.core.base_model import BaseModel
from telefuser.core.config import (
    AttentionConfig,
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
    WeightOffloadType,
)
from telefuser.models.minimax_h3_dit import MiniMaxH3DiTConfig
from telefuser.models.minimax_h3_image_dit import MiniMaxH3ImageDiT
from telefuser.models.pan2_dit import PAN2DiT
from telefuser.pipelines.minimax_h3_image.stages import MiniMaxH3ImageDenoisingStage
from telefuser.pipelines.pan2.stages import PAN2DenoisingStage
from telefuser.worker import ParallelWorker


class FixtureManager:
    """Only supply the already constructed random model; do not simulate checkpoint loading."""

    def __init__(self, model: BaseModel) -> None:
        self.model = model

    def fetch_module(self, name: str) -> BaseModel:
        return self.model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pan2", "h3-image"], required=True)
    parser.add_argument("--scheduler-config", type=Path, help="Released PAN2 scheduler_config.json")
    parser.add_argument("--device-ids", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--sp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--sol", action="store_true")
    parser.add_argument("--sol-fp8", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.model == "pan2" and args.scheduler_config is None:
        parser.error("PAN2 requires --scheduler-config")
    if args.sol_fp8 and not args.sol:
        parser.error("--sol-fp8 requires --sol")
    parallel = ParallelConfig(device_ids=args.device_ids, sp_ulysses_degree=args.sp, tp_degree=args.tp)
    parallel.validate()
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device_ids[0])
    torch.manual_seed(87)
    offload = OffloadConfig(offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD, pin_cpu_memory=False)
    baseline_config = ModelRuntimeConfig(device_id=args.device_ids[0], offload_config=offload)
    runtime = ModelRuntimeConfig(
        device_id=args.device_ids[0],
        offload_config=offload,
        parallel_config=parallel,
        quant_config=QuantConfig(
            enabled=args.fp8, quant_type=QuantType.FP8, kernel_backend=QuantKernelBackend.TF_KERNEL
        ),
        attention_config=AttentionConfig.sol_attention(
            dense_timesteps=0, dense_layers=0, tau=1.0, threshold_type="exact", sol_fp8=args.sol_fp8
        )
        if args.sol
        else AttentionConfig(),
    )
    if args.model == "h3-image":
        model = MiniMaxH3ImageDiT(
            MiniMaxH3DiTConfig(
                hidden_size=256,
                num_layers=2,
                token_refiner_num_layers=1,
                num_attention_heads=4,
                attention_head_dim=128,
                ffn_hidden_size=512,
                latents_dim=2,
                text_dim=32,
                time_embed_hidden_size=256,
                time_embed_dim=128,
                rope_inv_freq_len=16,
            )
        )
        inputs = [
            {"hidden_states": torch.randn(5, 32), "token_tags": torch.ones(5, dtype=torch.long)},
            [torch.randn(1, 2, 4, 4)],
            256,
            512,
            torch.Generator().manual_seed(71).get_state(),
        ]

        def build(m: BaseModel, config: ModelRuntimeConfig) -> MiniMaxH3ImageDenoisingStage:
            return MiniMaxH3ImageDenoisingStage(
                FixtureManager(m), config, [1.0, 0.967523694, 0.887548506, 0.75, 0.532773912, 0.232008874, 0.0]
            )
    else:
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
        ).to(dtype=torch.bfloat16)
        inputs = [torch.randn(1, 5, 32), torch.randn(1, 3, 32), torch.randn(1, 2, 1, 8, 16), 128, 256, 5, 3, 2.5, 71]

        def build(m: BaseModel, config: ModelRuntimeConfig) -> PAN2DenoisingStage:
            return PAN2DenoisingStage(FixtureManager(m), config, str(args.scheduler_config))

    worker_stage = build(copy.deepcopy(model), runtime)
    expected = build(model, baseline_config).denoise(*inputs)
    worker = ParallelWorker(worker_stage)
    try:
        actual = worker.denoise(*inputs)()
        repeated = worker.denoise(*inputs)()
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        relative = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
        if not torch.isfinite(actual).all() or relative > 0.1:
            raise AssertionError(f"Sampling numerical fixture relative error {relative} exceeds .1")
        record = dict(
            model=args.model,
            sp=args.sp,
            tp=args.tp,
            fp8=args.fp8,
            sol=args.sol,
            sol_fp8=args.sol_fp8,
            repeated_worker_requests_identical=True,
            relative_l2=relative,
            shape=list(actual.shape),
            scope="tiny-model sampling fixture; not released-checkpoint quality or performance",
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2))
        print(json.dumps(record), flush=True)
    finally:
        worker.close()


if __name__ == "__main__":
    main()
