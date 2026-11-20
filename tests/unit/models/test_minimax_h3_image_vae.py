"""Image VAE numerical equivalence and cast-sensitive encoder regression."""

import pytest
import torch

try:
    from telefuser.models.minimax_h3_image_vae import MiniMaxH3ImageVAE
except ImportError as exc:
    pytest.skip(f"Optional model dependencies unavailable: {exc}", allow_module_level=True)


def tiny_vae() -> MiniMaxH3ImageVAE:
    return MiniMaxH3ImageVAE(
        latent_channels=2,
        block_out_channels=(8, 8),
        layers_per_block=1,
        spatial_downsample_factors=(2, 2),
        temporal_downsample_factors=(1, 1),
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
        latents_mean=(0.0, 0.0),
        latents_std=(1.0, 1.0),
    )


def test_cast_preserves_encoder_and_decoder_norms_exactly() -> None:
    model = tiny_vae()
    original = model.encoder.conv_in.weight.detach().clone()
    original_rope = model.decoder.rope.inv_freq.clone()
    model.to(dtype=torch.bfloat16)
    torch.testing.assert_close(model.encoder.conv_in.weight, original, rtol=0, atol=0)
    assert model.decoder.proj_in.weight.dtype == torch.bfloat16
    assert model.decoder.rope.inv_freq.dtype == torch.float32
    torch.testing.assert_close(model.decoder.rope.inv_freq, original_rope, rtol=0, atol=0)
    assert all(p.dtype == torch.float32 for n, p in model.named_parameters() if ".norm1." in n or ".scale1." in n)


def test_reference_encoder_decoder_and_tiling() -> None:
    try:
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import AutoencoderKLMiniMaxH3Image
    except ImportError:
        pytest.skip("Independent reference requires Diffusers with H3-Image support")
    model = tiny_vae().eval()
    reference = AutoencoderKLMiniMaxH3Image.from_config(dict(model.config)).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    pixels, latent = torch.randn(1, 3, 32, 32), torch.randn(1, 2, 8, 8)
    # Force multiple overlapping tiles to test stitching, not only the untiled path.
    model.enable_tiling(
        tile_sample_min_height=24,
        tile_sample_min_width=24,
        tile_sample_min_overlap_height=8,
        tile_sample_min_overlap_width=8,
    )
    reference.enable_tiling(
        tile_sample_min_height=24,
        tile_sample_min_width=24,
        tile_sample_min_overlap_height=8,
        tile_sample_min_overlap_width=8,
    )
    with torch.inference_mode():
        torch.testing.assert_close(
            model.encode(pixels).latent_dist.parameters, reference.encode(pixels).latent_dist.parameters, rtol=0, atol=0
        )
        torch.testing.assert_close(model.decode(latent).sample, reference.decode(latent).sample, rtol=0, atol=0)
