"""Compare an upstream Dense DiT forward with TeleFuser on one GPU.

Set ``PYTHONPATH=work_dirs/lingbot-video-master`` so the upstream package is
available without modifying its checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model

from telefuser.core.module_manager import ModuleManager


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    delta = reference.float() - candidate.float()
    return {
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float((delta.norm() / reference.float().norm().clamp_min(1e-12)).item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                reference.float().flatten(), candidate.float().flatten(), dim=0
            ).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--text-length", type=int, default=3)
    args = parser.parse_args()
    try:
        from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel as UpstreamTransformer
    except ImportError as exc:
        raise RuntimeError("Set PYTHONPATH to the upstream LingBot-Video checkout") from exc

    config = json.loads((args.transformer_dir / "config.json").read_text(encoding="utf-8"))
    fields = (
        "patch_size",
        "in_channels",
        "out_channels",
        "hidden_size",
        "num_attention_heads",
        "depth",
        "intermediate_size",
        "text_dim",
        "freq_dim",
        "norm_eps",
        "rope_theta",
        "axes_dims",
        "qkv_bias",
        "out_bias",
        "patch_embed_bias",
        "timestep_mlp_bias",
    )
    device = torch.device("cuda")
    upstream = UpstreamTransformer(**{field: config[field] for field in fields}).to(device, torch.bfloat16).eval()
    load_model(upstream, args.transformer_dir / "diffusion_pytorch_model.safetensors", strict=True, device=str(device))
    module_manager = ModuleManager(device=str(device), torch_dtype=torch.bfloat16)
    module_manager.load_model(str(args.transformer_dir), name="transformer")
    telefuser = module_manager.fetch_module("transformer")
    if telefuser is None:
        raise RuntimeError(f"Unable to load LingBot-Video transformer from {args.transformer_dir}")
    telefuser.promote_stability_layers_to_fp32()
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 16, 1, args.height, args.width, device=device, dtype=torch.bfloat16)
    text = torch.randn(1, args.text_length, config["text_dim"], device=device, dtype=torch.bfloat16)
    timestep = torch.tensor([500], device=device)
    with torch.no_grad():
        reference = upstream(latent, timestep, text, return_dict=False)[0]
        candidate = telefuser(latent, timestep, text)
    print(json.dumps(_metrics(reference, candidate), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
