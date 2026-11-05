import pytest
import torch
import torch.nn.functional as F

from telefuser.ops.fp8_attention import (
    _quantize_fp8_qkv_sm90,
    _sequence_mean,
    apply_fp8_attention_output_correction,
    dequantize_fp8_per_channel,
    merge_fp8_attention_prefix,
    quantize_fp8_qkv_smoothed,
)


def test_kv_sequence_mean_smoothing_is_attention_equivalent() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 3, 7, 8)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    key_mean = key.mean(dim=2, keepdim=True)
    value_mean = value.mean(dim=2, keepdim=True)

    reference = F.scaled_dot_product_attention(query, key, value)
    smoothed = F.scaled_dot_product_attention(query, key - key_mean, value - value_mean) + value_mean

    torch.testing.assert_close(smoothed, reference, atol=2e-6, rtol=2e-6)


@pytest.mark.gpu
def test_fp8_value_bias_correction_restores_original_sequence_mean_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FP8 attention smoothing kernel test requires H100")
    torch.manual_seed(0)
    shape = (1, 130, 2, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    _, _, value_fp8, _, _, value_scale, correction = quantize_fp8_qkv_smoothed(
        query,
        key,
        value,
        smoothing="kv",
        correct_v_bias=True,
    )

    assert correction is not None
    restored = dequantize_fp8_per_channel(value_fp8, value_scale, torch.float32) + correction
    torch.testing.assert_close(
        restored.mean(dim=1),
        value.float().mean(dim=1),
        atol=2e-6,
        rtol=2e-6,
    )


@pytest.mark.gpu
def test_fused_kv_statistics_match_separate_reductions_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FP8 attention smoothing kernel test requires H100")
    torch.manual_seed(1)
    shape = (1, 130, 2, 128)
    query = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    key_mean = _sequence_mean(key)
    value_mean = _sequence_mean(value)
    expected = _quantize_fp8_qkv_sm90(query, key, value, k_mean=key_mean, v_mean=value_mean)
    expected_correction = (value_mean - _sequence_mean(expected[2]) * expected[5]).unsqueeze(1).contiguous()

    *actual, actual_correction = quantize_fp8_qkv_smoothed(
        query,
        key,
        value,
        smoothing="kv",
        correct_v_bias=True,
    )

    assert all(torch.equal(before, after) for before, after in zip(expected, actual, strict=True))
    assert torch.equal(expected_correction, actual_correction)


@pytest.mark.gpu
def test_fp8_output_correction_matches_fp32_add_without_allocation_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FP8 attention correction kernel test requires H100")
    torch.manual_seed(0)
    output = torch.randn((1, 130, 2, 128), device="cuda", dtype=torch.bfloat16)
    correction = torch.randn((1, 1, 2, 128), device="cuda", dtype=torch.float32)
    expected = (output.float() + correction).to(output.dtype)
    pointer = output.data_ptr()

    actual = apply_fp8_attention_output_correction(output, correction)

    assert actual.data_ptr() == pointer
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.gpu
def test_fused_prefix_merge_matches_correction_then_cat_on_h100() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FP8 attention prefix merge kernel test requires H100")
    torch.manual_seed(2)
    output = torch.randn((1, 130, 2, 128), device="cuda", dtype=torch.bfloat16)
    prefix = torch.randn((1, 17, 2, 128), device="cuda", dtype=torch.bfloat16)
    correction = torch.randn((1, 1, 2, 128), device="cuda", dtype=torch.float32)
    corrected_suffix = (output[:, prefix.shape[1] :].float() + correction).to(output.dtype)
    expected = torch.cat((prefix, corrected_suffix), dim=1)

    actual = merge_fp8_attention_prefix(prefix, output, correction)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_value_bias_correction_requires_value_smoothing() -> None:
    query = torch.randn(1, 64, 1, 128)

    with pytest.raises(ValueError, match="requires smoothing='kv'"):
        quantize_fp8_qkv_smoothed(query, query, query, smoothing="k", correct_v_bias=True)
