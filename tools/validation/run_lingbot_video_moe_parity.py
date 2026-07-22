"""Compare upstream and TeleFuser MoE/refiner forwards without dual model residency.

Set ``PYTHONPATH=work_dirs/lingbot-video-master`` so the upstream package is
available without modifying its checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from telefuser.pipelines.lingbot_video.loading import load_lingbot_video_moe_transformer


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


def exact_parity_failures(metrics: dict[str, float]) -> list[str]:
    """Return numerical fields that do not meet the zero-drift oracle gate."""
    return [name for name in ("max_abs", "mean_abs", "relative_l2") if metrics[name] != 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--text-length", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Optional JSON metrics destination.")
    parser.add_argument("--assert-exact", action="store_true", help="Exit nonzero unless the oracle replay is exact.")
    args = parser.parse_args()
    try:
        from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel as UpstreamTransformer
    except ImportError as exc:
        raise RuntimeError("Set PYTHONPATH to the upstream LingBot-Video checkout") from exc

    directory = args.transformer_dir
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    config_fields = {name: value for name, value in config.items() if not name.startswith("_")}
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    latent = torch.randn(1, 16, 1, args.height, args.width, dtype=torch.bfloat16)
    text = torch.randn(1, args.text_length, config["text_dim"], dtype=torch.bfloat16)
    timestep = torch.tensor([500])

    upstream = UpstreamTransformer(**config_fields).to(torch.bfloat16).to(device).eval()
    index = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8"))[
        "weight_map"
    ]
    expected = set(upstream.state_dict())
    found: set[str] = set()
    unexpected: set[str] = set()
    for shard in sorted(set(index.values())):
        shard_state = load_file(directory / shard, device=str(device))
        unexpected.update(upstream.load_state_dict(shard_state, strict=False).unexpected_keys)
        found.update(shard_state)
        del shard_state
    missing = expected - found
    if missing or unexpected:
        raise RuntimeError(
            f"Upstream MoE checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    with torch.no_grad():
        reference = (
            upstream(latent.to(device), timestep.to(device), text.to(device), return_dict=False)[0].float().cpu()
        )
    del upstream
    torch.cuda.empty_cache()

    telefuser = load_lingbot_video_moe_transformer(directory, device=device, torch_dtype=torch.bfloat16)
    with torch.no_grad():
        candidate = telefuser(latent.to(device), timestep.to(device), text.to(device)).float().cpu()
    metrics = _metrics(reference, candidate)
    if args.assert_exact:
        failures = exact_parity_failures(metrics)
        if failures:
            raise SystemExit(f"Exact MoE parity gate failed: {', '.join(failures)}")
    payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
