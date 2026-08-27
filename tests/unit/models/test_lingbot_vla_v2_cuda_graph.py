from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from telefuser.models.lingbot_vla_v2_cuda_graph import LingBotVlaV2CudaGraphs, LingBotVlaV2DenoisingCudaGraph


class _CudaGraphVelocityModel:
    config = SimpleNamespace(num_steps=10)

    def __init__(self) -> None:
        self.prefix_builds = 0

    def build_prefix_cache(self, images, _img_masks, lang_tokens, lang_masks, _image_grid_thw):
        self.prefix_builds += 1
        scale = images.mean() + lang_tokens.float().mean()
        positions = lang_tokens.clone()
        cache = {
            0: {
                "key_states": torch.ones((1, 2), device=images.device) * scale,
                "value_states": torch.zeros((1, 2), device=images.device),
            }
        }
        return lang_masks.clone(), positions, cache

    def predict_velocity(self, state, _prefix_pad_masks, past_key_values, x_t, timestep, **_kwargs):
        scale = past_key_values[0]["key_states"].mean()
        return x_t * 0.125 + state + scale + timestep.view(-1, 1, 1) * 0.01


@pytest.mark.gpu
def test_denoising_cuda_graph_replays_all_steps_with_new_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graph test requires CUDA")

    device = torch.device("cuda:0")
    model = _CudaGraphVelocityModel()
    state = torch.full((1, 2, 3), 0.2, device=device)
    noise = torch.ones((1, 2, 3), device=device)
    masks = torch.ones((1, 4), dtype=torch.bool, device=device)
    positions = torch.zeros((1, 4), dtype=torch.long, device=device)
    cache = {
        0: {"key_states": torch.full((1, 2), 0.3, device=device), "value_states": torch.zeros((1, 2), device=device)}
    }

    def eager() -> torch.Tensor:
        dt = torch.full((), -0.1, device=device)
        time = torch.ones((), device=device)
        result = noise.clone()
        for _ in range(10):
            velocity = model.predict_velocity(state, masks, cache, result, time.expand(1))
            result.add_(dt * velocity)
            time.add_(dt)
        return result

    runner = LingBotVlaV2DenoisingCudaGraph(model)
    expected = eager()
    actual = runner.run(state, masks, cache, noise, positions)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)

    state.fill_(-0.1)
    cache[0]["key_states"].fill_(0.6)
    expected_replay = eager()
    actual_replay = runner.run(state, masks, cache, noise, positions)
    torch.cuda.synchronize(device)
    assert torch.equal(actual_replay, expected_replay)


@pytest.mark.gpu
@torch.inference_mode()
def test_cuda_graph_rebuilds_dynamic_prefix_and_replays_denoising() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA Graph test requires CUDA")

    device = torch.device("cuda:0")
    model = _CudaGraphVelocityModel()
    images = torch.full((2, 3), 0.25, device=device)
    img_masks = torch.ones((1, 2), dtype=torch.bool, device=device)
    lang_tokens = torch.tensor([[1, 2, 3, 4]], device=device)
    lang_masks = torch.ones_like(lang_tokens, dtype=torch.bool)
    grid = torch.tensor([[1, 2, 2]], device=device)
    state = torch.full((1, 2, 3), 0.2, device=device)
    noise = torch.ones((1, 2, 3), device=device)

    def eager() -> torch.Tensor:
        masks, positions, cache = model.build_prefix_cache(images, img_masks, lang_tokens, lang_masks, grid)
        del positions
        dt = torch.full((), -0.1, device=device)
        time = torch.ones((), device=device)
        result = noise.clone()
        for _ in range(10):
            velocity = model.predict_velocity(state, masks, cache, result, time.expand(1))
            result.add_(dt * velocity)
            time.add_(dt)
        return result

    runner = LingBotVlaV2CudaGraphs(model)
    expected = eager()
    actual = runner.run(images, img_masks, lang_tokens, lang_masks, state, noise, grid)
    torch.cuda.synchronize(device)
    assert torch.equal(actual, expected)
    assert model.prefix_builds == 2

    images.fill_(0.5)
    lang_tokens.add_(1)
    lang_masks[:, :2] = False
    state.fill_(-0.1)
    expected_replay = eager()
    actual_replay = runner.run(images, img_masks, lang_tokens, lang_masks, state, noise, grid)
    torch.cuda.synchronize(device)
    assert torch.equal(actual_replay, expected_replay)
    assert model.prefix_builds == 4
