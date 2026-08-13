from __future__ import annotations

import torch
import pytest

from telefuser.service.livekit.nccl_transfer import flatten_tensor_tree, rebuild_tensor_tree
from telefuser.service.livekit.config import LiveKitServeConfig


def test_tensor_manifest_round_trip_preserves_nested_structure() -> None:
    source = {
        "cache": [{"k": torch.ones((1, 2)), "cursor": 3}],
        "latent": torch.zeros((1, 3)),
        "flags": (True, None),
    }
    skeleton, manifest, leaves = flatten_tensor_tree(source)

    restored = rebuild_tensor_tree(skeleton, leaves)

    assert len(manifest) == 2
    assert restored["cache"][0]["cursor"] == 3
    assert restored["flags"] == (True, None)
    assert torch.equal(restored["cache"][0]["k"], source["cache"][0]["k"])


def test_process_nccl_requires_fixed_two_gpu_group() -> None:
    config = LiveKitServeConfig(
        worker_mode="process-nccl",
        num_workers=2,
        worker_gpu_map="0;1",
    )

    assert config.worker_mode == "process-nccl"

    autoscaling = LiveKitServeConfig(
        worker_mode="process-nccl",
        num_workers=2,
        worker_gpu_map="0;1",
        autoscaling_enabled=True,
        queue_size=1,
    )
    assert autoscaling.autoscaling_enabled is True
