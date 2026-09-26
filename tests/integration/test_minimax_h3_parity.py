from __future__ import annotations

from pathlib import Path

import pytest
import torch

try:
    from safetensors import safe_open
except ImportError:
    pytest.skip("safetensors is required for MiniMax H3 checkpoint tests", allow_module_level=True)

try:
    from telefuser.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
    from telefuser.models.minimax_h3_dit import MiniMaxH3DiT
    from telefuser.models.minimax_h3_encoder import MiniMaxH3Encoder
    from telefuser.models.minimax_h3_video_vae import MiniMaxH3VideoVAE
except ImportError as exc:
    pytest.skip(f"MiniMax H3 model dependencies are unavailable: {exc}", allow_module_level=True)


def _metadata_state(paths: list[Path]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shape = tuple(handle.get_slice(name).get_shape())
                state[name] = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    return state


def _assert_exact_contract(
    model_class: type[torch.nn.Module],
    checkpoint_paths: list[Path],
    config_path: Path,
) -> None:
    official = _metadata_state(checkpoint_paths)
    converter = model_class.state_dict_converter(config_path=config_path)
    converted, kwargs = converter.from_official(official)
    with torch.device("meta"):
        model = model_class(**kwargs)
    expected = model.state_dict()
    assert set(converted) == set(expected)
    mismatches = {
        name: (tuple(converted[name].shape), tuple(expected[name].shape))
        for name in expected
        if converted[name].shape != expected[name].shape
    }
    assert not mismatches


@pytest.mark.filesystem
@pytest.mark.parametrize("partition", ["FL2VA", "Ref2VA"])
def test_original_partition_checkpoint_contracts_are_exact(partition: str, pytestconfig: pytest.Config) -> None:
    model_root = pytestconfig.getoption("--minimax-h3-model-root")
    if not model_root:
        pytest.skip("pass --minimax-h3-model-root to run MiniMax H3 checkpoint tests")
    root = Path(model_root).expanduser() / partition
    if not root.is_dir():
        pytest.skip(f"MiniMax H3 checkpoint partition does not exist: {root}")
    _assert_exact_contract(
        MiniMaxH3DiT,
        sorted((root / "transformer").glob("model-*.safetensors")),
        root / "transformer" / "config.json",
    )
    _assert_exact_contract(
        MiniMaxH3Encoder,
        sorted((root / "text_encoder").glob("model-*.safetensors")),
        root / "text_encoder",
    )
    _assert_exact_contract(
        MiniMaxH3VideoVAE,
        [root / "video_vae" / "source" / "model.safetensors"],
        root / "video_vae",
    )
    _assert_exact_contract(
        MiniMaxH3AudioVAE,
        [root / "audio_vae" / "model.safetensors"],
        root / "audio_vae",
    )
