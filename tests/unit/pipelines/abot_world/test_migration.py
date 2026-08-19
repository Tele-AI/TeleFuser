from __future__ import annotations

import threading
from collections import deque
from types import MethodType, SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from telefuser.models.taew2_2 import TAEHV, TWorkItem
from telefuser.models.wan22_video_vae import Wan22VideoVAEStreamingDecodeState
from telefuser.pipelines.abot_world.interactive import (
    ABotWorldInteractivePipeline,
    ABotWorldInteractiveSession,
    ABotWorldSessionLifecycle,
)
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService
from telefuser.pipelines.abot_world.taew_vae import ABotWorldTAEWDecodeStage
from telefuser.service.livekit.nccl_transfer import flatten_tensor_tree, rebuild_tensor_tree


class _TinyTAEW(nn.Module):
    """Minimal decoder topology used only to exercise session-state transport."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Identity())
        self.decoder = nn.Sequential(*(nn.Identity() for _ in range(4)))


def _taew_stage() -> ABotWorldTAEWDecodeStage:
    stage = object.__new__(ABotWorldTAEWDecodeStage)
    stage.taew = _TinyTAEW()
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32
    return stage


def _taew_state(stage: ABotWorldTAEWDecodeStage) -> Any:
    state = stage.create_decode_state()
    state.stream.decoder_work_queue = [TWorkItem(torch.tensor([[[[8.0]]]]), 2)]
    state.stream.decoder_memory[1] = torch.tensor([[[[9.0]]]])
    state.stream.decoder_memory[2] = [torch.tensor([[[[10.0]]]])]
    state.stream.n_frames_decoded = 11
    return state


def _assert_taew_state(state: Any) -> None:
    assert state.stream.decoder_work_queue[0].block_index == 2
    assert state.stream.decoder_work_queue[0].input_tensor.item() == 8
    assert state.stream.decoder_memory[1].item() == 9
    assert state.stream.decoder_memory[2][0].item() == 10
    assert state.stream.n_frames_decoded == 11


def test_session_snapshot_round_trip_preserves_causal_and_rng_state() -> None:
    source = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    source.denoise_stage = SimpleNamespace(_scheduler=lambda: object())
    source.taew_decode_stage = _taew_stage()
    generator = torch.Generator(device="cpu").manual_seed(123)
    session = ABotWorldInteractiveSession(
        session_id="migrating",
        prompt_emb=torch.tensor([1.0]),
        first_frame_latent=torch.tensor([2.0]),
        self_cache=[
            {
                "k": torch.tensor([[[[3.0]]]]),
                "v": torch.tensor([[[[4.0]]]]),
                "global_end_index": torch.tensor([12]),
                "local_end_index": torch.tensor([6]),
            }
        ],
        cross_cache=[
            {
                "k": torch.tensor([[[[5.0]]]]),
                "v": torch.tensor([[[[6.0]]]]),
                "is_init": True,
                "sequence_length": 1,
            }
        ],
        scheduler=object(),
        generator=generator,
        vae_decode_state=Wan22VideoVAEStreamingDecodeState(feat_cache=[torch.tensor([7.0])]),
        taew_decode_state=_taew_state(source.taew_decode_stage),
        next_latent_frame=12,
        emitted_frames=45,
        ownership_epoch=4,
    )
    source._interactive_sessions[session.session_id] = session
    generator_state = generator.get_state()
    expected_next_random = torch.randn(1, generator=generator)
    generator.set_state(generator_state)

    snapshot = source.snapshot_interactive_session(session)
    assert snapshot.taew_decode_state["decoder_work_queue"][0]["input_tensor"].device.type == "cpu"
    session.taew_decode_state.stream.decoder_work_queue[0].input_tensor.fill_(99)
    assert snapshot.taew_decode_state["decoder_work_queue"][0]["input_tensor"].item() == 8
    source.close_interactive_session(session)
    target = ABotWorldInteractivePipeline(device="cpu", torch_dtype=torch.float32)
    target.denoise_stage = SimpleNamespace(_scheduler=lambda: object())
    target.taew_decode_stage = _taew_stage()
    restored = target.restore_interactive_snapshot(snapshot, owner_worker_id="gpu-1")

    assert restored.lifecycle == ABotWorldSessionLifecycle.READY
    assert restored.owner_worker_id == "gpu-1"
    assert restored.ownership_epoch == 5
    assert restored.next_latent_frame == 12
    assert restored.emitted_frames == 45
    assert restored.self_cache[0]["k"].item() == 3
    assert restored.cross_cache[0]["v"].item() == 6
    assert restored.vae_decode_state.feat_cache[0].item() == 7
    assert restored.taew_decode_state is not None
    _assert_taew_state(restored.taew_decode_state)
    target.suspend_interactive_session(restored)
    assert restored.taew_decode_state.stream.decoder_memory[1].device.type == "cpu"
    target.restore_interactive_session(restored)
    _assert_taew_state(restored.taew_decode_state)
    assert torch.equal(torch.randn(1, generator=restored.generator), expected_next_random)
    target.close_interactive_session(restored)
    assert restored.taew_decode_state is None


def _real_taew_stage() -> ABotWorldTAEWDecodeStage:
    model = TAEHV(
        checkpoint_path=None,
        encoder_time_downscale=(False, False, False),
        decoder_time_upscale=(False, False, False),
        decoder_space_upscale=(False, False, False),
        latent_channels=2,
    ).eval()
    stage = object.__new__(ABotWorldTAEWDecodeStage)
    stage.taew = model
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32
    return stage


def test_taew_batched_decode_matches_independent_streams() -> None:
    stage = _real_taew_stage()
    serial_states = [stage.create_decode_state(), stage.create_decode_state()]
    batched_states = [stage.create_decode_state(), stage.create_decode_state()]
    first = [torch.randn(1, 2, 1, 1, 1), torch.randn(1, 2, 1, 1, 1)]
    continuation = [torch.randn(1, 2, 3, 1, 1), torch.randn(1, 2, 3, 1, 1)]

    expected_first = torch.cat(
        [stage._decode_chunks_impl(latents, [state]) for latents, state in zip(first, serial_states)],
        dim=0,
    )
    actual_first = stage._decode_chunks_impl(torch.cat(first), batched_states)
    torch.testing.assert_close(actual_first, expected_first)

    expected_continuation = torch.cat(
        [stage._decode_chunks_impl(latents, [state]) for latents, state in zip(continuation, serial_states)],
        dim=0,
    )
    actual_continuation = stage._decode_chunks_impl(torch.cat(continuation), batched_states)
    torch.testing.assert_close(actual_continuation, expected_continuation)
    assert [state.stream.n_frames_decoded for state in batched_states] == [
        state.stream.n_frames_decoded for state in serial_states
    ]


def test_taew_snapshot_restore_preserves_real_causal_decode() -> None:
    source_stage = _real_taew_stage()
    original = source_stage.create_decode_state()
    transferred = source_stage.create_decode_state()
    first = torch.randn(1, 2, 1, 1, 1)
    continuation = torch.randn(1, 2, 3, 1, 1)

    source_stage._decode_chunks_impl(first, [original])
    source_stage._decode_chunks_impl(first, [transferred])
    snapshot = source_stage.snapshot_decode_state(transferred)
    restored = source_stage.restore_decode_state(snapshot)

    expected = source_stage._decode_chunks_impl(continuation, [original])
    actual = source_stage._decode_chunks_impl(continuation, [restored])
    torch.testing.assert_close(actual, expected)


def test_taew_decoder_state_nccl_tensor_tree_round_trip() -> None:
    source_stage = _taew_stage()
    source_state = _taew_state(source_stage)

    payload = source_stage.export_decode_state_for_nccl(source_state)
    skeleton, manifest, leaves = flatten_tensor_tree(payload)
    assert manifest
    assert any(path[:2] == ("decoder_work_queue", 0) for path in leaves)
    target_leaves = {path: tensor.clone() for path, tensor in leaves.items()}
    rebuilt = rebuild_tensor_tree(skeleton, target_leaves)

    target_stage = _taew_stage()
    restored = target_stage.restore_decode_state(rebuilt, direct_device_tensors=True)
    _assert_taew_state(restored)
    source_state.stream.decoder_memory[1].fill_(99)
    assert restored.stream.decoder_memory[1].item() == 9


def test_nccl_migration_metadata_includes_taew_decoder_state() -> None:
    stage = _taew_stage()
    generator = torch.Generator(device="cpu").manual_seed(7)
    session = SimpleNamespace(
        prompt_emb=torch.tensor([1.0]),
        first_frame_latent=torch.tensor([2.0]),
        self_cache=[],
        cross_cache=[],
        vae_decode_state=Wan22VideoVAEStreamingDecodeState(),
        taew_decode_state=_taew_state(stage),
        generator=generator,
        next_latent_frame=3,
        emitted_frames=12,
        ownership_epoch=2,
    )
    service = object.__new__(ABotWorldLiveKitService)
    service.pipeline = SimpleNamespace(taew_decode_stage=stage)
    service._quiesce_migration = lambda session_id, timeout: SimpleNamespace(
        pipeline_session=session,
        config={"fps": 12},
        controls={"W"},
        control_idle_timeout=10.0,
        last_control_at=1.0,
        next_chunk_index=1,
        next_playout_deadline=2.0,
    )

    metadata = service.prepare_migration_nccl_metadata("migrating", timeout=1)
    payload = rebuild_tensor_tree(metadata["tensor_skeleton"], metadata["_nccl_tensor_leaves"])

    assert metadata["state_bytes"] > 0
    assert "taew_decode_state" in payload
    restored = stage.restore_decode_state(payload["taew_decode_state"], direct_device_tensors=True)
    _assert_taew_state(restored)


def test_import_migration_nccl_restores_taew_state_without_cpu_copy() -> None:
    source_stage = _taew_stage()
    source_state = _taew_state(source_stage)
    payload = {
        "prompt_emb": torch.tensor([1.0]),
        "first_frame_latent": torch.tensor([2.0]),
        "self_cache": [],
        "cross_cache": [],
        "vae_feat_cache": [],
        "taew_decode_state": source_stage.export_decode_state_for_nccl(source_state),
    }
    skeleton, _, leaves = flatten_tensor_tree(payload)
    target_leaves = {path: tensor.clone() for path, tensor in leaves.items()}
    restored_pipeline_session = SimpleNamespace(session_id="migrating")
    observed: dict[str, Any] = {}

    def restore_interactive_device_snapshot(snapshot, **kwargs):
        observed["snapshot"] = snapshot
        observed["kwargs"] = kwargs
        return restored_pipeline_session

    pipeline = SimpleNamespace(
        restore_interactive_device_snapshot=restore_interactive_device_snapshot,
        close_interactive_session=lambda session: None,
    )
    service = object.__new__(ABotWorldLiveKitService)
    service.pipeline = pipeline
    service._ensure_scheduler_started = lambda: None
    service._capacity_profile = {"effective_capacity": 1}
    service._sessions = {}
    service._round_robin_order = deque()
    service._scheduler_condition = threading.Condition(threading.RLock())
    service.output_queue_size = 1

    session_id = service.import_migration_nccl(
        {
            "session_id": "migrating",
            "tensor_skeleton": skeleton,
            "vae_feat_idx": [],
            "generator_state": torch.Generator(device="cpu").get_state(),
            "next_latent_frame": 3,
            "emitted_frames": 12,
            "ownership_epoch": 2,
            "config": {"fps": 12},
            "controls": ["W"],
            "control_idle_timeout": 10.0,
            "last_control_at": 1.0,
            "next_chunk_index": 1,
            "next_playout_deadline": 2.0,
        },
        target_leaves,
        owner_worker_id="gpu-1",
        ownership_epoch=3,
    )

    assert session_id == "migrating"
    snapshot = observed["snapshot"]
    assert observed["kwargs"] == {"owner_worker_id": "gpu-1", "ownership_epoch": 3}
    restored = source_stage.restore_decode_state(snapshot.taew_decode_state, direct_device_tensors=True)
    _assert_taew_state(restored)
    assert service._sessions[session_id].pipeline_session is restored_pipeline_session


class _FakeTAEWStream:
    """CPU-only stream double that exposes compatibility and call count."""

    def __init__(self, cursor: int) -> None:
        self.decoder_work_queue: list[object] = []
        self.decoder_memory = None
        self.n_frames_decoded = cursor
        self.decode_shapes: list[tuple[int, ...]] = []

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_shapes.append(tuple(latents.shape))
        return latents


def _fake_taew_telemetry_stage() -> ABotWorldTAEWDecodeStage:
    stage = object.__new__(ABotWorldTAEWDecodeStage)
    stage.device = torch.device("cpu")
    stage.torch_dtype = torch.float32
    stage.synchronized_decode_shapes = []

    def synchronized_decode(self, decoder_latents, states):
        del states
        self.synchronized_decode_shapes.append(tuple(decoder_latents.shape))
        return decoder_latents

    stage._decode_synchronized_batch = MethodType(synchronized_decode, stage)
    return stage


def test_taew_decode_telemetry_distinguishes_native_batch_from_serial_fallback() -> None:
    stage = _fake_taew_telemetry_stage()
    latents = torch.ones(2, 1, 3, 1, 1)

    synchronized_states = [
        SimpleNamespace(stream=_FakeTAEWStream(cursor=0)),
        SimpleNamespace(stream=_FakeTAEWStream(cursor=0)),
    ]
    synchronized = stage._decode_chunks_impl(latents, synchronized_states)

    assert synchronized.shape == latents.shape
    assert stage.synchronized_decode_shapes == [(2, 3, 1, 1, 1)]
    assert [state.stream.decode_shapes for state in synchronized_states] == [[], []]
    assert stage.last_decode_metrics() == {
        "taew_decode_items": 2,
        "taew_decode_batch_size": 2,
        "taew_decode_invocations": 1,
        "taew_decode_mode": 1,
    }

    fallback_states = [
        SimpleNamespace(stream=_FakeTAEWStream(cursor=0)),
        SimpleNamespace(stream=_FakeTAEWStream(cursor=1)),
    ]
    fallback = stage._decode_chunks_impl(latents, fallback_states)

    assert fallback.shape == latents.shape
    assert [state.stream.decode_shapes for state in fallback_states] == [
        [(1, 3, 1, 1, 1)],
        [(1, 3, 1, 1, 1)],
    ]
    assert stage.last_decode_metrics() == {
        "taew_decode_items": 2,
        "taew_decode_batch_size": 1,
        "taew_decode_invocations": 2,
        "taew_decode_mode": 2,
    }
