"""Compare upstream and TeleFuser refiner low-noise sampling with injected tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.lingbot_video.denoising import LingBotVideoDenoisingStage
from telefuser.pipelines.lingbot_video.refiner import (
    LingBotVideoRefinerStage,
    compute_refiner_sigmas,
    prepare_refiner_latent,
)
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler


class _Encode:
    def encode(self, values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        del generator
        return values


class _Decode:
    def decode(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = reference.float() - candidate.float()
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(delta.norm() / reference.float().norm().clamp_min(1e-12)),
        "cosine": float(
            torch.nn.functional.cosine_similarity(reference.float().flatten(), candidate.float().flatten(), dim=0)
        ),
    }


def exact_parity_failures(metrics: dict[str, float]) -> list[str]:
    """Return numerical fields that do not meet the zero-drift oracle gate."""
    return [name for name in ("max_abs", "mean_abs", "relative_l2") if metrics[name] != 0.0]


def _load_upstream():
    from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

    config = json.loads((DIRECTORY / "config.json").read_text())
    model = LingBotVideoTransformer3DModel(**{key: value for key, value in config.items() if not key.startswith("_")})
    model = model.to(torch.bfloat16).to(DEVICE).eval()
    index = json.loads((DIRECTORY / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    expected, found, unexpected = set(model.state_dict()), set(), set()
    for shard in sorted(set(index.values())):
        values = load_file(DIRECTORY / shard, device=str(DEVICE))
        unexpected.update(model.load_state_dict(values, strict=False).unexpected_keys)
        found.update(values)
        del values
    if expected - found or unexpected:
        raise RuntimeError("upstream checkpoint key mismatch")
    return model


def main() -> None:
    global ROOT, DIRECTORY, DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--refiner-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-exact", action="store_true", help="Exit nonzero unless the oracle replay is exact.")
    args = parser.parse_args()
    ROOT = args.model_root
    DIRECTORY = args.refiner_dir or ROOT / "refiner"
    DEVICE = torch.device(args.device)
    from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline
    from lingbot_video.scheduling_flow_unipc import FlowUniPCMultistepScheduler as UpstreamScheduler

    torch.manual_seed(args.seed)
    x_up = torch.randn(1, 16, 1, 8, 8)
    noise = torch.randn_like(x_up)
    condition = torch.randn(1, 16, 1, 8, 8)
    positive = torch.randn(1, 3, 2560, dtype=torch.bfloat16)
    negative = torch.randn(1, 3, 2560, dtype=torch.bfloat16)
    mask = torch.ones(1, 3, dtype=torch.long)
    sigmas = compute_refiner_sigmas(
        sigma_max=1.0, sigma_min=0.0, num_inference_steps=2, shift=3.0, t_thresh=0.25, tail_steps=1
    )
    initial = prepare_refiner_latent(x_up, noise, 0.25)

    upstream = _load_upstream()
    upstream_scheduler = UpstreamScheduler.from_pretrained(ROOT / "scheduler")
    source_pipeline = LingBotVideoPipeline(upstream, None, None, None, upstream_scheduler)
    with torch.no_grad():
        reference = (
            source_pipeline(
                "",
                height=64,
                width=64,
                num_frames=1,
                guidance_scale=3.0,
                num_inference_steps=2,
                shift=3.0,
                latents=initial.to(DEVICE),
                cond_latent=condition.to(DEVICE),
                prompt_embeds=positive.to(DEVICE),
                prompt_mask=mask.to(DEVICE),
                negative_prompt_embeds=negative.to(DEVICE),
                negative_prompt_mask=mask.to(DEVICE),
                output_type="latent",
                t_thresh=0.25,
                refiner_sigma_tail_steps=1,
            )
            .frames.float()
            .cpu()
        )
    upstream.to("cpu")
    source_pipeline.transformer = None
    del source_pipeline, upstream
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    module_manager = ModuleManager(device=str(DEVICE), torch_dtype=torch.bfloat16)
    module_manager.load_model(str(DIRECTORY), name="transformer")
    local = module_manager.fetch_module("transformer")
    if local is None:
        raise RuntimeError(f"Unable to load LingBot-Video transformer from {DIRECTORY}")
    local.promote_stability_layers_to_fp32()
    local_scheduler = FlowUniPCMultistepScheduler.from_pretrained(ROOT / "scheduler")
    stage = LingBotVideoRefinerStage(
        denoising_stage=LingBotVideoDenoisingStage(
            "refiner", module_manager, ModelRuntimeConfig(device_type="cuda", torch_dtype=torch.bfloat16)
        ),
        vae_encode_stage=_Encode(),
        vae_decode_stage=_Decode(),
        scheduler=local_scheduler,
    )
    with torch.no_grad():
        candidate = (
            stage.refine(
                x_up.to(DEVICE),
                positive.to(DEVICE),
                negative.to(DEVICE),
                mask.to(DEVICE),
                mask.to(DEVICE),
                num_inference_steps=2,
                guidance_scale=3.0,
                shift=3.0,
                t_thresh=0.25,
                tail_steps=1,
                clean_first_frame=condition.to(DEVICE),
                noise=noise.to(DEVICE),
            )
            .float()
            .cpu()
        )
    metrics = _metrics(reference, candidate)
    if args.assert_exact:
        failures = exact_parity_failures(metrics)
        if failures:
            raise SystemExit(f"Exact refiner parity gate failed: {', '.join(failures)}")
    payload = json.dumps({"sigmas": sigmas.tolist(), "metrics": metrics}, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
