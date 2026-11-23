"""Retain non-persistent rotary buffers exactly across runtime dtype/device casts."""

import pytest
import torch

try:
    from telefuser.core.base_model import BaseModel
    from telefuser.models.pan2_conditioning import PAN2TextEncoder
except ImportError as exc:
    pytest.skip(f"Optional conditioning dependencies unavailable: {exc}", allow_module_level=True)


@pytest.mark.parametrize("model_class", [PAN2TextEncoder])
def test_conditioner_cast_preserves_rotary_frequencies(model_class) -> None:
    model = model_class.__new__(model_class)
    BaseModel.__init__(model)
    model.model = torch.nn.Linear(4, 4)
    frequency = torch.tensor([0.123456789, 0.000123456789], dtype=torch.float32)
    model.model.register_buffer("inv_freq", frequency.clone(), persistent=False)
    model.model.register_buffer("original_inv_freq", frequency.clone(), persistent=False)
    model.to(dtype=torch.bfloat16)
    assert model.model.weight.dtype == torch.bfloat16
    torch.testing.assert_close(model.model.inv_freq, frequency, rtol=0, atol=0)
    torch.testing.assert_close(model.model.original_inv_freq, frequency, rtol=0, atol=0)
