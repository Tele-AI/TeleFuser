"""PAN2 architecture, checkpoint and precision contracts."""

import json

import pytest
import torch

try:
    from telefuser.models.pan2_dit import PAN2DiT
except ImportError as exc:
    pytest.skip(f"Optional attention dependencies unavailable: {exc}", allow_module_level=True)


def tiny_model() -> PAN2DiT:
    return PAN2DiT(
        in_channels=5,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_refiner_layers=1,
        text_embed_dim=16,
        rope_axes_dim=(4, 6, 6),
        mlp_ratio=2,
        context_mlp_ratio=2,
        refiner_mlp_ratio=2,
    )


def test_released_pan2_meta_shapes() -> None:
    """Verify released pan2 meta shapes."""
    with torch.device("meta"):
        model = PAN2DiT()
    assert len(model.transformer_blocks) == 44
    assert len(model.context_refiner) == 4
    assert model.x_embedder.weight.shape == (4096, 97, 1, 2, 2)
    assert model.proj_out.weight.shape == (192, 4096)


def test_precision_does_not_round_modulation_weights() -> None:
    """Verify precision does not round modulation weights."""
    model = tiny_model()
    original = model.transformer_blocks[0].norm1.linear.weight.detach().clone()
    model.to(dtype=torch.bfloat16)
    assert model.x_embedder.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.transformer_blocks[0].norm1.linear.weight, original, rtol=0, atol=0)
    assert model.time_embedder.linear_1.weight.dtype == torch.float32


def test_tiny_pan2_reference_when_installed() -> None:
    """Verify tiny pan2 reference when installed."""
    try:
        from diffusers.models.transformers.transformer_pan2 import PAN2Transformer3DModel
    except ImportError:
        pytest.skip("Independent reference requires Diffusers with PAN2 support")
    torch.manual_seed(81)
    model = tiny_model().eval()
    reference = PAN2Transformer3DModel.from_config(dict(model.config)).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    args = dict(
        hidden_states=torch.randn(1, 5, 2, 6, 6),
        timestep=torch.tensor([500.0]),
        encoder_hidden_states=torch.randn(1, 5, 16),
    )
    with torch.inference_mode():
        torch.testing.assert_close(model(**args).sample, reference(**args).sample, rtol=0, atol=0)


def test_converter_uses_checkpoint_architecture(tmp_path) -> None:
    """Verify converter uses checkpoint architecture."""
    model = tiny_model()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(model.config)))
    weights, kwargs = model.state_dict_converter(path).from_diffusers(model.state_dict())
    loaded = PAN2DiT(**kwargs)
    loaded.load_state_dict(weights, strict=True)
    assert loaded.config.out_channels == 2


def test_device_only_move_does_not_clone_modulation_weights(monkeypatch) -> None:
    """Verify device only move does not clone modulation weights."""
    model = tiny_model()

    def reject_clone(*args, **kwargs):
        raise AssertionError("Device-only moves must not duplicate FP32 modulation weights")

    monkeypatch.setattr(torch.Tensor, "clone", reject_clone)
    model.to("cpu")
