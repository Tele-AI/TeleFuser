import json
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

try:
    from telefuser.models.minimax_h3_dit import MiniMaxH3DiTConfig
    from telefuser.models.minimax_h3_image_dit import MiniMaxH3ImageDiT, MiniMaxH3ImagePDDProjection
except ImportError as exc:
    pytest.skip(f"Optional attention dependencies are unavailable: {exc}", allow_module_level=True)


def small_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )


def packed_image_inputs() -> dict:
    return {
        "x": torch.randn(1, 6, 8),
        "img_position_ids": torch.zeros(1, 6, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(6, dtype=torch.long),
        "update_mask": torch.ones(4, dtype=torch.bool),
        "token_tags": torch.tensor([1, 1, 0, 0, 0, 0]),
        "prompt_embeds": torch.randn(2, 16),
        "img_pos_info": {"position_ids": torch.arange(2, 6)},
        "text_pos_info": {"position_ids": torch.arange(2)},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(2, 6)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 6], dtype=torch.int32)},
    }


def test_image_checkpoint_has_no_audio_parameters() -> None:
    with torch.device("meta"):
        model = MiniMaxH3ImageDiT()
    state = model.state_dict()
    assert not any("audio" in key for key in state)
    assert state["final_layer.video_out.weight"].shape == (6, 96, 5376)
    assert state["final_layer.video_out.weight"].dtype == torch.float32
    assert state["blocks.0.adaln_proj.linear.weight"].shape[0] == 18 * 5376


def test_pdd_step_selects_its_own_released_projection() -> None:
    projection = MiniMaxH3ImagePDDProjection(8, 4, 6)
    inputs = torch.randn(5, 8)
    for index in range(6):
        projection.step_index = index
        torch.testing.assert_close(
            projection(inputs), F.linear(inputs, projection.weight[index], projection.bias[index])
        )


def test_image_forward_has_no_audio_requirement_and_uses_pdd_head() -> None:
    torch.manual_seed(3)
    model = MiniMaxH3ImageDiT(small_config()).eval()
    inputs = packed_image_inputs()
    with torch.inference_mode():
        first = model(pdd_step_index=0, **inputs)
        last = model(pdd_step_index=5, **inputs)
    assert first.shape == (4, 8)
    assert torch.isfinite(first).all()
    assert not torch.equal(first, last)
    with pytest.raises(ValueError, match="PDD step index"):
        model(pdd_step_index=6, **inputs)
    with pytest.raises(ValueError, match="does not accept audio"):
        model(pdd_step_index=0, audio_x=torch.ones(1, 1, 32), **inputs)


def test_official_dit_and_fused_heads_load_strictly(tmp_path) -> None:
    config = small_config()
    model = MiniMaxH3ImageDiT(config)
    state = dict(model.state_dict())
    state["video_out.weight"] = state.pop("final_layer.video_out.weight")
    state["video_out.bias"] = state.pop("final_layer.video_out.bias")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(asdict(config)))
    converted, kwargs = model.state_dict_converter(path).from_official(state)
    clone = MiniMaxH3ImageDiT(**kwargs)
    clone.load_state_dict(converted, strict=True)
    torch.testing.assert_close(clone.final_layer.video_out.weight, model.final_layer.video_out.weight)
    with pytest.raises(ValueError, match="pdd_fused_heads"):
        model.state_dict_converter(path).from_official(
            {k: v for k, v in state.items() if not k.startswith("video_out")}
        )


def test_complete_six_head_edit_sampler_matches_independent_reference(tmp_path) -> None:
    try:
        from diffusers import MiniMaxH3Scheduler
        from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel
        from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
            build_image_packed_sequence,
            patchify_video_latents,
        )
    except ImportError:
        pytest.skip("Independent image sampling reference requires recent Diffusers")
    from telefuser.core.config import ModelRuntimeConfig
    from telefuser.pipelines.minimax_h3_image.stages import MiniMaxH3ImageDenoisingStage

    torch.manual_seed(73)
    config = small_config()
    reference = MiniMaxH3ImageTransformer2DModel(
        hidden_size=32,
        num_layers=2,
        num_refiner_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_dim=64,
        in_channels=2,
        text_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=1,
    ).eval()
    preserved = {
        k: v.clone()
        for k, v in reference.state_dict().items()
        if k.startswith(("proj_in.", "time_embedder.", "pdd_proj_out."))
    }
    reference.to(dtype=torch.bfloat16)
    for name, value in preserved.items():
        reference.get_parameter(name).data = value.float()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    model = MiniMaxH3ImageDiT(config).eval()
    converted, _ = model.state_dict_converter(config_path).from_diffusers(reference.state_dict())
    model.load_state_dict(converted, strict=True)
    sigmas = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]

    class Manager:
        def fetch_module(self, name: str) -> MiniMaxH3ImageDiT:
            return model

    stage = MiniMaxH3ImageDenoisingStage(Manager(), ModelRuntimeConfig(device_type="cpu"), sigmas)
    text = {"hidden_states": torch.randn(2, 16, dtype=torch.bfloat16), "token_tags": torch.ones(2, dtype=torch.long)}
    image = torch.randn(1, 2, 4, 4)
    actual = stage.denoise(text, [image], 64, 64, torch.Generator().manual_seed(71).get_state())
    generator = torch.Generator().manual_seed(71)
    target = patchify_video_latents(torch.randn(1, 2, 1, 4, 4, generator=generator), (1, 2, 2))
    conditioned = 0.999 * image + (1.0 - 0.999) * torch.randn(image.shape, generator=generator)
    rows = torch.cat([patchify_video_latents(conditioned.unsqueeze(2), (1, 2, 2)), target])
    pos, tags, image_ids, text_ids, nref = build_image_packed_sequence(text["token_tags"], [(4, 4)], 4, 4, (1, 2, 2), 0)
    timesteps = torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32)
    scheduler = MiniMaxH3Scheduler(euler_update_form="velocity", shift_provided_sigmas=False)
    scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]))
    with torch.inference_mode():
        for step, timestep in enumerate(scheduler.timesteps):
            row_t = torch.full((len(tags),), float(timestep))
            row_t[image_ids[:nref]] = 0.999
            unique, inverse = torch.unique(row_t, sorted=True, return_inverse=True)
            velocity = reference(
                hidden_states=rows[None],
                encoder_hidden_states=text["hidden_states"][None],
                timestep=unique,
                timestep_indices=inverse,
                token_tags=tags,
                position_ids=pos,
                image_indices=image_ids,
                text_indices=text_ids,
                pdd_step_index=step,
            ).sample[0]
            rows[nref:] = scheduler.step(velocity[nref:], timestep, rows[nref:], return_dict=False)[0]
    expected = rows[nref:].reshape(1, 1, 2, 2, 2, 1, 2, 2).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 2, 1, 4, 4)
    torch.testing.assert_close(actual, expected[:, :, 0], rtol=0, atol=0)
