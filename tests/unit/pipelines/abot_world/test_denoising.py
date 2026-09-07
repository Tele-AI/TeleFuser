from __future__ import annotations

from types import MethodType

import torch

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.models.abot_world_dit import ABotWorldDiT
from telefuser.pipelines.abot_world.denoising import ABotWorldDenoisingStage, _ABotSteadyCudaGraph


def _stage_with_recording_dit() -> tuple[ABotWorldDenoisingStage, list[torch.Tensor]]:
    dit = ABotWorldDiT(
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=8,
        text_dim=16,
        out_dim=4,
        num_heads=4,
        num_layers=2,
        downscale_factor_control_adapter=2,
    )
    manager = ModuleManager(torch_dtype=torch.float32, device="cpu")
    manager.add_module(dit, "abot_world_dit")
    stage = ABotWorldDenoisingStage(
        "abot-world-test",
        manager,
        ModelRuntimeConfig(device_type="cpu", torch_dtype=torch.float32),
    )
    stage.parallel_models()
    observed_timesteps: list[torch.Tensor] = []

    def zero_flow_prediction(model: ABotWorldDiT, **kwargs: object) -> torch.Tensor:
        del model
        observed_timesteps.append(kwargs["timestep"].detach().clone())
        return torch.zeros_like(kwargs["x"])

    dit.forward = MethodType(zero_flow_prediction, dit)
    return stage, observed_timesteps


def test_official_four_step_schedule_matches_warped_wan_training_indices() -> None:
    scheduler = ABotWorldDenoisingStage._scheduler()

    actual = ABotWorldDenoisingStage._official_denoising_timesteps(scheduler)

    torch.testing.assert_close(actual, torch.tensor([1000.0, 937.5, 833.3333, 625.0]), rtol=1e-4, atol=1e-4)


def test_x0_prediction_uses_the_scheduler_sigma_for_each_frame() -> None:
    scheduler = ABotWorldDenoisingStage._scheduler()
    timestep = ABotWorldDenoisingStage._official_denoising_timesteps(scheduler)[1].reshape(1, 1)
    latent = torch.full((1, 1, 1, 1, 1), 4.0)
    flow_prediction = torch.full_like(latent, 2.0)

    actual = ABotWorldDenoisingStage._x0_prediction(flow_prediction, latent, timestep, scheduler)

    torch.testing.assert_close(actual, torch.full_like(latent, 2.125))


def test_denoising_block_runs_four_model_updates_then_issues_context_cache_update() -> None:
    stage, observed_timesteps = _stage_with_recording_dit()
    self_cache, cross_cache = stage._new_cache(batch_size=1, height=8, width=8)
    scheduler = stage._scheduler()
    generator = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn(1, 4, 3, 8, 8, generator=generator)

    output = stage._denoise_block(
        latent=noise,
        prompt_emb=torch.randn(1, 4, 16),
        action_context=torch.randn(1, 32, 3, 16, 16),
        first_frame_latent=None,
        self_cache=self_cache,
        cross_cache=cross_cache,
        current_start=3,
        generator=generator,
        scheduler=scheduler,
    )

    expected = ABotWorldDenoisingStage._official_denoising_timesteps(scheduler)
    assert output.shape == noise.shape
    assert len(observed_timesteps) == 5
    for observed, timestep in zip(observed_timesteps[:4], expected, strict=True):
        torch.testing.assert_close(observed, torch.full((1, 3), timestep))
    assert torch.equal(observed_timesteps[-1], torch.zeros(1, 3))


def test_interactive_cuda_graph_wrapper_falls_back_cleanly_on_cpu() -> None:
    stage, observed_timesteps = _stage_with_recording_dit()
    stage.configure_cuda_graph(True)
    self_cache, cross_cache = stage._new_cache(batch_size=1, height=8, width=8)
    scheduler = stage._scheduler()
    generator = torch.Generator(device="cpu").manual_seed(43)
    noise = torch.randn(1, 4, 3, 8, 8, generator=generator)

    output = stage.denoise_interactive_block(
        session_id="cpu-fallback",
        latent=noise,
        prompt_emb=torch.randn(1, 4, 16),
        action_context=torch.randn(1, 32, 3, 16, 16),
        self_cache=self_cache,
        cross_cache=cross_cache,
        current_start=3,
        generator=generator,
        scheduler=scheduler,
    )

    assert output.shape == noise.shape
    assert len(observed_timesteps) == 5
    assert stage.last_cuda_graph_metrics() == {
        "cuda_graph_enabled": 1,
        "cuda_graph_eligible": 0,
        "cuda_graph_captured": 0,
        "cuda_graph_replays": 0,
        "cuda_graph_fallback": 0,
        "cuda_graph_batch_size": 0,
        "cuda_graph_batched": 0,
    }
    assert stage.cuda_graph_metrics()["replays"] == 0


def test_sage_attention_is_explicitly_eager_only_until_graph_parity_is_available() -> None:
    stage, _ = _stage_with_recording_dit()
    stage.dit.set_attention_config(AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90))

    assert not stage._cuda_graph_backend_is_supported()


def test_cuda_graph_capture_replays_slots_before_consuming_static_outputs() -> None:
    """Capture records work; it must be replayed before sampler state advances."""
    stage, _ = _stage_with_recording_dit()
    scheduler = stage._scheduler()
    generator = torch.Generator(device="cpu").manual_seed(71)
    latent = torch.randn(1, 4, 3, 8, 8, generator=generator)
    action_context = torch.randn(1, 32, 3, 16, 16, generator=generator)

    class FakeGraph:
        def __init__(self) -> None:
            self.replays = 0

        def replay(self) -> None:
            self.replays += 1

    entry_graph = FakeGraph()
    refinement_graph = FakeGraph()
    state = object.__new__(_ABotSteadyCudaGraph)
    state.device = torch.device("cpu")
    state.torch_dtype = stage.torch_dtype
    state.dit = stage.dit
    state.frames = latent.shape[2]
    state.frame_tokens = 16
    state.static_x = torch.empty_like(latent)
    state.static_action = torch.empty_like(action_context)
    state.static_timestep = torch.empty((1, latent.shape[2]), dtype=torch.float32)
    state.static_context = torch.empty(1, 4, 16)
    state.current_end = torch.empty(1, dtype=torch.long)
    state.entry = None
    state.refinement = None

    def capture_slot(
        _state: _ABotSteadyCudaGraph,
        _self_cache: list[dict[str, object]],
        _cross_cache: list[dict[str, object]],
        *,
        update_cache: bool,
    ) -> object:
        graph = entry_graph if update_cache else refinement_graph
        return type("Slot", (), {"graph": graph, "output": torch.zeros_like(latent)})()

    state._capture_slot = MethodType(capture_slot, state)
    output, replays = state.run(
        stage,
        latent,
        action_context,
        [{}],
        [{}],
        current_start=3,
        generator=generator,
        scheduler=scheduler,
        capture=True,
    )

    assert output.shape == latent.shape
    assert replays == 4
    assert entry_graph.replays == 1
    assert refinement_graph.replays == 3


def test_batched_cuda_graph_arena_binds_rows_and_keeps_independent_cursors() -> None:
    """The B=2 arena owns K/V while each retained session keeps its cursor."""
    stage, _ = _stage_with_recording_dit()
    self_caches = []
    cross_caches = []
    for start in (18, 21):
        self_cache, cross_cache = stage._new_cache(batch_size=1, height=8, width=8)
        for self_layer, cross_layer in zip(self_cache, cross_cache, strict=True):
            self_layer["local_end_index"].fill_(stage.dit.local_attn_size * 16)
            self_layer["global_end_index"].fill_(start * 16)
            cross_layer["is_init"] = True
            cross_layer["sequence_length"] = 4
        self_caches.append(self_cache)
        cross_caches.append(cross_cache)

    latent = torch.randn(2, 4, 3, 8, 8)
    prompt_emb = torch.randn(2, 4, 16)
    action_context = torch.randn(2, 32, 3, 16, 16)
    state = stage._create_batched_cuda_graph_state(
        ("a", "b"),
        latent,
        prompt_emb,
        action_context,
        self_caches,
        cross_caches,
        current_starts=(18, 21),
    )

    stage._bind_batched_cache_arena(state, self_caches, cross_caches)
    assert state.matches_members(("a", "b"), self_caches, cross_caches)
    for row, (self_cache, cross_cache) in enumerate(zip(self_caches, cross_caches, strict=True)):
        assert self_cache[0]["k"].data_ptr() == state.self_cache[0]["k"][row : row + 1].data_ptr()
        assert cross_cache[0]["v"].data_ptr() == state.cross_cache[0]["v"][row : row + 1].data_ptr()

    stage._advance_batched_cache_cursors(self_caches, current_starts=(18, 21), latent=latent)

    assert int(self_caches[0][0]["global_end_index"].item()) == 21 * 16
    assert int(self_caches[1][0]["global_end_index"].item()) == 24 * 16
    assert int(self_caches[0][0]["local_end_index"].item()) == stage.dit.local_attn_size * 16
    assert int(self_caches[1][0]["local_end_index"].item()) == stage.dit.local_attn_size * 16
