"""Tests for normalization operations."""

import pytest
import torch
import torch.nn as nn

from telefuser.ops.normalization import (
    AdaLayerNormContinuous,
    LayerNorm,
    RMSNorm,
    _fused_add_layer_norm_scale_shift,
    _fused_layer_norm_scale_shift,
)


def test_fused_layer_norm_scale_shift_matches_native() -> None:
    x = torch.randn(2, 5, 16)
    scale = torch.randn(2, 1, 16)
    shift = torch.randn(2, 1, 16)

    output = _fused_layer_norm_scale_shift(x, scale, shift)
    expected = torch.nn.functional.layer_norm(x, (16,)) * (1 + scale) + shift

    torch.testing.assert_close(output, expected, atol=5e-5, rtol=1e-5)


def test_fused_add_layer_norm_scale_shift_matches_native() -> None:
    residual = torch.randn(2, 5, 16)
    x = torch.randn_like(residual)
    scale = torch.randn(2, 1, 16)
    shift = torch.randn(2, 1, 16)

    output, residual_out = _fused_add_layer_norm_scale_shift(residual, x, scale, shift)
    expected_residual = residual + x
    expected = torch.nn.functional.layer_norm(expected_residual, (16,)) * (1 + scale) + shift

    torch.testing.assert_close(residual_out, expected_residual)
    torch.testing.assert_close(output, expected, atol=5e-5, rtol=1e-5)


class TestRMSNorm:
    """Test RMSNorm layer."""

    @pytest.mark.parametrize(
        "elementwise_affine,bias",
        [
            (True, False),  # Default: affine with weight only
            (True, True),  # Affine with weight and bias
            (False, False),  # No affine transformation
        ],
    )
    def test_initialization(self, elementwise_affine, bias):
        """Test various initialization configurations."""
        norm = RMSNorm(dim=64, eps=1e-6, elementwise_affine=elementwise_affine, bias=bias)

        assert norm.eps == 1e-6
        assert norm.elementwise_affine == elementwise_affine

        if elementwise_affine:
            assert norm.weight is not None
            assert norm.weight.shape == torch.Size([64])
            # Weight should be initialized to ones
            assert torch.allclose(norm.weight, torch.ones(64))
            if bias:
                assert norm.bias is not None
                assert norm.bias.shape == torch.Size([64])
                # Bias should be initialized to zeros
                assert torch.allclose(norm.bias, torch.zeros(64))
        else:
            assert norm.weight is None
            assert norm.bias is None

    def test_forward_shape_preservation(self):
        """Test that forward preserves input shape across various input shapes."""
        norm = RMSNorm(dim=64, eps=1e-6)

        test_cases = [
            torch.randn(2, 10, 64),  # 3D
            torch.randn(4, 64),  # 2D
            torch.randn(1, 2, 3, 64),  # 4D
        ]

        for x in test_cases:
            output = norm(x)
            assert output.shape == x.shape

    def test_forward_normalization_effect(self):
        """Test that normalization actually normalizes the input."""
        norm = RMSNorm(dim=64, eps=1e-6, elementwise_affine=False)
        # Create tensor with varying magnitudes
        x = torch.randn(2, 10, 64) * 10.0

        output = norm(x)

        # Output should be different from input
        assert not torch.allclose(output, x)

        # RMS should be approximately 1 (without affine)
        rms = torch.sqrt(output.pow(2).mean(dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    @pytest.mark.parametrize(
        "input_dtype,expected_dtype",
        [
            (torch.float32, torch.float16),  # Input float32, weight is half
            (torch.float64, torch.float64),  # No affine, preserve input dtype
        ],
    )
    def test_dtype_handling(self, input_dtype, expected_dtype):
        """Test dtype handling in forward pass."""
        norm = RMSNorm(dim=64, eps=1e-6, elementwise_affine=True)
        if expected_dtype == torch.float16:
            norm.weight = nn.Parameter(norm.weight.half())

        x = torch.randn(2, 10, 64, dtype=input_dtype)
        output = norm(x)

        assert output.dtype == expected_dtype

    def test_edge_cases(self):
        """Test edge cases: all-ones input and zeros input."""
        norm = RMSNorm(dim=64, eps=1e-6, elementwise_affine=False)

        # All ones - output should be ones (RMS of all-ones is 1)
        x_ones = torch.ones(2, 10, 64)
        output_ones = norm(x_ones)
        assert torch.allclose(output_ones, x_ones, atol=1e-5)

        # All zeros - should not produce NaN
        x_zeros = torch.zeros(2, 10, 64)
        output_zeros = norm(x_zeros)
        assert not torch.isnan(output_zeros).any()


class TestAdaLayerNormContinuous:
    """Test AdaLayerNormContinuous layer."""

    @pytest.mark.parametrize(
        "norm_type,expected_norm_class",
        [
            ("layer_norm", LayerNorm),
            ("rms_norm", RMSNorm),
        ],
    )
    def test_initialization_with_norm_types(self, norm_type, expected_norm_class):
        """Test initialization with different norm types."""
        layer = AdaLayerNormContinuous(
            embedding_dim=512,
            conditioning_embedding_dim=256,
            norm_type=norm_type,
        )

        assert isinstance(layer.silu, nn.SiLU)
        assert isinstance(layer.linear, nn.Linear)
        assert layer.linear.in_features == 256
        assert layer.linear.out_features == 1024  # 512 * 2
        assert isinstance(layer.norm, expected_norm_class)

    def test_init_invalid_norm_type(self):
        """Test initialization with invalid norm type."""
        with pytest.raises(ValueError, match="unknown norm_type"):
            AdaLayerNormContinuous(
                embedding_dim=512,
                conditioning_embedding_dim=256,
                norm_type="invalid_norm",
            )

    def test_forward_shape_preservation(self):
        """Test forward pass shape preservation."""
        layer = AdaLayerNormContinuous(embedding_dim=64, conditioning_embedding_dim=32)

        x = torch.randn(2, 10, 64)
        conditioning = torch.randn(2, 32)

        output = layer(x, conditioning)

        assert output.shape == x.shape

    def test_forward_modulation_effect(self):
        """Test that conditioning actually modulates the output."""
        layer = AdaLayerNormContinuous(embedding_dim=64, conditioning_embedding_dim=32)

        x = torch.randn(2, 10, 64)
        cond1 = torch.randn(2, 32)
        cond2 = torch.randn(2, 32)

        output1 = layer(x, cond1)
        output2 = layer(x, cond2)

        # Output should be different from input
        assert not torch.allclose(output1, x)

        # Different conditioning should produce different outputs
        assert not torch.allclose(output1, output2)

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_configuration(self, bias):
        """Test with and without bias in linear layer."""
        layer = AdaLayerNormContinuous(
            embedding_dim=64,
            conditioning_embedding_dim=32,
            bias=bias,
        )

        if bias:
            assert layer.linear.bias is not None
        else:
            assert layer.linear.bias is None

        x = torch.randn(2, 10, 64)
        conditioning = torch.randn(2, 32)

        output = layer(x, conditioning)
        assert output.shape == x.shape
