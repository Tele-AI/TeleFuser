"""Validate LingBot-Video Ulysses sequence parallelism against an eager model."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from telefuser.core.config import ModelRuntimeConfig, OffloadConfig, ParallelConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.lingbot_video_dit import LingBotVideoTransformer3DModel
from telefuser.pipelines.lingbot_video.denoising import LingBotVideoDenoisingStage, transformer_timestep


def _build_model(device: torch.device) -> LingBotVideoTransformer3DModel:
    """Create a compact deterministic DiT whose joint stream needs SP padding."""
    torch.manual_seed(7)
    return (
        LingBotVideoTransformer3DModel(
            patch_size=(1, 2, 2),
            in_channels=2,
            out_channels=2,
            hidden_size=16,
            num_attention_heads=4,
            depth=2,
            intermediate_size=32,
            text_dim=8,
            freq_dim=8,
            axes_dims=(2, 2, 0),
        )
        .to(device=device, dtype=torch.bfloat16)
        .eval()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--fsdp", action="store_true", help="Also apply TeleFuser block FSDP.")
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    try:
        world_size = dist.get_world_size()
        if world_size < 2 or 4 % world_size:
            raise ValueError("SP parity requires a world size that divides four")
        model = _build_model(device)
        # LingBot checkpoints intentionally preserve these modulation tables in FP32.
        # Exercise the FSDP ignored-state path with the same mixed-dtype layout.
        for module in model.modules():
            if hasattr(module, "scale_shift_table"):
                module.scale_shift_table.data = module.scale_shift_table.data.float()
        torch.manual_seed(11)
        latent = torch.randn(1, 2, 1, 2, 8, device=device, dtype=torch.bfloat16)
        prompt = torch.randn(1, 3, 8, device=device, dtype=torch.bfloat16)
        negative_prompt = torch.randn(1, 3, 8, device=device, dtype=torch.bfloat16)
        prompt_mask = torch.tensor([[True, True, False]], device=device)
        timestep = torch.tensor([125.0], device=device)
        with torch.inference_mode():
            model_timestep = transformer_timestep(timestep, torch.bfloat16)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                eager_positive = model(latent, model_timestep, prompt, prompt_mask)
                eager_negative = model(latent, model_timestep, negative_prompt, prompt_mask)
            eager = eager_negative.float() + 3.0 * (eager_positive.float() - eager_negative.float())
            if args.fsdp:
                runtime_config = ModelRuntimeConfig(
                    device_type="cuda",
                    device_id=local_rank,
                    torch_dtype=torch.bfloat16,
                    offload_config=OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD),
                    parallel_config=ParallelConfig(
                        device_ids=list(range(world_size)),
                        sp_ulysses_degree=world_size,
                        enable_fsdp=True,
                    ),
                )
                module_manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
                module_manager.add_module(model, name="transformer")
                stage = LingBotVideoDenoisingStage("sp4-test", module_manager, runtime_config, batch_cfg=True)
                stage.parallel_models()
                distributed = stage.predict_noise_with_cfg(
                    latent,
                    timestep,
                    prompt,
                    negative_prompt,
                    positive_attention_mask=prompt_mask,
                    negative_attention_mask=prompt_mask,
                    guidance_scale=3.0,
                )
            else:
                model.set_ulysses_group(dist.group.WORLD)
                positive = model(latent, model_timestep, prompt, prompt_mask)
                negative = model(latent, model_timestep, negative_prompt, prompt_mask)
                distributed = negative.float() + 3.0 * (positive.float() - negative.float())
        delta = (distributed.float() - eager.float()).abs()
        metrics = torch.tensor(
            [delta.max(), delta.mean(), delta.norm() / eager.float().norm().clamp_min(1e-12)], device=device
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        report = {
            "fsdp": args.fsdp,
            "batch_cfg": args.fsdp,
            "world_size": world_size,
            "joint_tokens": 7,
            "ulysses_padding_tokens": 1 if world_size == 4 else 0,
            "fp32_unsharded_parameters": sum(parameter.dtype == torch.float32 for parameter in model.parameters()),
            "max_abs": float(metrics[0].item()),
            "mean_abs": float(metrics[1].item()),
            "relative_l2": float(metrics[2].item()),
        }
        if args.output and dist.get_rank() == 0:
            with open(args.output, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
        if dist.get_rank() == 0:
            print(json.dumps(report, sort_keys=True))
        if report["max_abs"] != 0.0:
            raise AssertionError(f"Ulysses SP parity failed: {report}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
