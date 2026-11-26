"""Image VAE numerical equivalence and cast-sensitive encoder regression."""

from pathlib import Path

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
    """Verify cast preserves encoder and decoder norms exactly."""
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
    """Verify reference encoder decoder and tiling."""
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decoder_graph_replays_new_inputs_without_aliasing(dtype: torch.dtype) -> None:
    """Compare real decoder graph replays with eager execution and invalidate on moves."""
    model = tiny_vae().eval().to(device="cuda", dtype=dtype)
    first = torch.randn(1, 2, 8, 8, device="cuda", dtype=dtype)
    second = torch.randn_like(first)
    with torch.inference_mode():
        expected_first = model.decode(first).sample
        expected_second = model.decode(second).sample
        model.enable_cuda_graph_decode()
        actual_first = model.decode(first).sample
        saved_first = actual_first.clone()
        actual_second = model.decode(second).sample
        assert model._decode_graph is not None
        # More tiles than streams exercise replay reuse and output ownership.
        tiles = [torch.randn_like(first) for _ in range(8)]
        expected_tiles = [model.decoder(model.post_quant_conv(tile)) for tile in tiles]
        actual_tiles = model._decode_tiles(tiles)
        assert len(model._decode_graph[1]) == 6
        tolerance = 3e-5 if dtype == torch.float32 else 2e-2
        for actual, expected in zip(actual_tiles, expected_tiles):
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(actual_first, expected_first, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(actual_second, expected_second, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(actual_first, saved_first, rtol=0, atol=0)
        model.to("cpu")
        assert model._decode_graph is None


def _parallel_tile_worker(rank: int, init_file: str) -> None:
    from datetime import timedelta

    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        world_size=2,
        rank=rank,
        timeout=timedelta(seconds=60),
        device_id=torch.device("cuda", rank),
    )
    try:
        torch.manual_seed(71)
        device = torch.device("cuda", rank)
        model = tiny_vae().eval().to(device)
        model.enable_tiling(
            tile_sample_min_height=24,
            tile_sample_min_width=24,
            tile_sample_min_overlap_height=8,
            tile_sample_min_overlap_width=8,
        )
        pairs = []
        # Three tiles exercise padding; one tile leaves the second rank idle.
        for height, width in ((48, 24), (8, 12)):
            pixels = torch.randn(1, 3, height, width, device=device)
            latent = torch.randn(1, 2, height // 4, width // 4, device=device)
            with torch.inference_mode():
                pairs.append((pixels, latent, model.encode(pixels).latent_dist.parameters, model.decode(latent).sample))
        model.enable_parallel_tiling(dist.group.WORLD)
        model.enable_cuda_graph_decode()
        actual = []
        with torch.inference_mode():
            for pixels, latent, _, _ in pairs:
                actual.append((model.encode(pixels).latent_dist.parameters, model.decode(latent).sample))
        for (_, _, expected_encode, expected_decode), (encoded, decoded) in zip(pairs, actual):
            torch.testing.assert_close(encoded, expected_encode, rtol=3e-5, atol=3e-5)
            torch.testing.assert_close(decoded, expected_decode, rtol=3e-5, atol=3e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_parallel_tiles_match_serial_with_uneven_and_idle_ranks(tmp_path: Path) -> None:
    """Check real encode/decode tiles, graph replay and exact reconstruction order."""
    torch.multiprocessing.spawn(_parallel_tile_worker, args=(str(tmp_path / "vae-group"),), nprocs=2, join=True)
