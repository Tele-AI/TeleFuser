from __future__ import annotations

from pathlib import Path

import torch

from tools.validation.inspect_lingbot_video_checkpoint import build_load_report


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList((torch.nn.Linear(2, 3), torch.nn.Linear(3, 2)))
        self.head = torch.nn.Linear(2, 1, bias=False)


def test_load_report_covers_keys_and_parameter_layout() -> None:
    model = _TinyModel()
    keys = set(model.state_dict())
    config = {
        "patch_size": [1, 2, 2],
        "in_channels": 16,
        "out_channels": 16,
        "hidden_size": 8,
        "num_attention_heads": 2,
        "depth": 2,
        "intermediate_size": 12,
        "text_dim": 10,
        "freq_dim": 4,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "axes_dims": [2, 2, 4],
        "qkv_bias": True,
        "out_bias": True,
        "patch_embed_bias": True,
        "timestep_mlp_bias": True,
    }

    report = build_load_report(
        model,
        checkpoint_dir=Path("checkpoint"),
        variant="dense",
        config=config,
        available_keys=keys,
    )

    coverage = report["checkpoint_key_coverage"]
    parameters = report["parameters"]
    assert coverage["coverage"] == 1.0
    assert coverage["missing_keys"] == []
    assert coverage["unexpected_keys"] == []
    assert parameters["total_parameter_count"] == sum(parameter.numel() for parameter in model.parameters())
    assert parameters["parameter_count_by_block"] == {"0": 9, "1": 8}
    assert parameters["parameter_count_by_component"]["head"] == 2
