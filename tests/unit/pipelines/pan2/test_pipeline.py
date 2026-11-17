import pytest
import torch

from telefuser.pipelines.pan2.pipeline import PAN2PipelineConfig, _quantize_nvfp4


def test_pan2_defaults_are_single_card_safe():
    config = PAN2PipelineConfig()
    assert config.model_id == "IFM/PAN2"
    assert config.cpu_offload is True
    assert config.nvfp4 is False


@pytest.mark.gpu
def test_nvfp4_linear_smoke():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("NVFP4 requires Blackwell or newer")
    module = torch.nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
    _quantize_nvfp4(module, use_triton_kernel=False)
    output = module(torch.randn(2, 32, device="cuda", dtype=torch.bfloat16))
    assert output.shape == (2, 32)
    assert output.dtype == torch.bfloat16
