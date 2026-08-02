from unittest.mock import MagicMock

import torch
from PIL import Image

from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline, _CoLocatedVAEDecodeWorker
from telefuser.pipelines.lingbot_world_fast.session import (
    LingBotWorldFastGenerationSession,
    LingBotWorldFastSessionConfig,
    LingBotWorldFastSessionStatus,
)
from telefuser.worker.parallel_worker import ParallelWorker


def _session() -> LingBotWorldFastGenerationSession:
    return LingBotWorldFastGenerationSession(
        config=LingBotWorldFastSessionConfig(prompt="test", image=Image.new("RGB", (8, 8))),
        latent_f=1,
        chunk_size=1,
        cache_handle=7,
    )


def test_release_session_is_idempotent() -> None:
    pipeline = LingBotWorldFastPipeline(device="cpu")
    pipeline.denoise_stage = MagicMock()
    session = _session()

    pipeline.release_session(session)
    pipeline.release_session(session)

    pipeline.denoise_stage.release_cache.assert_called_once_with(7)
    assert session.cache_handle is None
    assert session.status == LingBotWorldFastSessionStatus.RELEASED


def test_pipeline_close_delegates_to_parallel_worker() -> None:
    pipeline = LingBotWorldFastPipeline(device="cpu")
    worker = object.__new__(ParallelWorker)
    worker.close = MagicMock()
    pipeline.denoise_stage = worker

    pipeline.close()

    worker.close.assert_called_once_with()


def test_colocated_vae_decode_worker_maps_lifecycle_calls() -> None:
    worker = MagicMock()
    worker.name = "denoise"
    proxy = _CoLocatedVAEDecodeWorker(worker)

    assert proxy.uses_local_latent_handoff is True

    proxy.reset_device_memory_peak()
    proxy.device_memory_snapshots()
    proxy.estimate_session_cache_bytes()
    proxy.observed_session_cache_bytes()
    proxy.configure_cache_pool(3)
    proxy.initialize_cache(7, sync=True)
    proxy.decode_chunk(7, "latent")
    proxy.release_cache(7, sync=True)

    worker.reset_vae_decode_device_memory_peak.assert_called_once_with(sync=True)
    worker.vae_decode_device_memory_snapshots.assert_called_once_with(sync=True)
    worker.estimate_vae_decode_session_cache_bytes.assert_called_once_with(sync=True)
    worker.observed_vae_decode_session_cache_bytes.assert_called_once_with(sync=True)
    worker.configure_vae_decode_cache_pool.assert_called_once_with(3, sync=True)
    worker.initialize_vae_decode_cache.assert_called_once_with(7, sync=True)
    worker.decode_chunk.assert_called_once_with(7, "latent")
    worker.release_vae_decode_cache.assert_called_once_with(7, sync=True)


def test_colocated_vae_decode_worker_reuses_registered_output_buffer() -> None:
    worker = MagicMock()
    worker.name = "denoise"
    first = torch.ones(3, 4, 2, 2, dtype=torch.uint8).share_memory_()
    worker.decode_chunk.side_effect = [
        lambda: (first, {"chunk": 0}),
        lambda: (None, {"chunk": 1}),
    ]
    worker.release_vae_decode_cache.return_value = True
    proxy = _CoLocatedVAEDecodeWorker(worker)

    first_output, first_profile = proxy.decode_chunk(7, "latent")()
    second_output, second_profile = proxy.decode_chunk(7, "latent")()
    proxy.release_cache(7, sync=True)

    assert first_output.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert second_output.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    assert first_profile == {"chunk": 0}
    assert second_profile == {"chunk": 1}
    assert proxy._output_buffers == {}


def test_colocated_vae_decode_worker_replaces_resized_output_buffer() -> None:
    worker = MagicMock()
    worker.name = "denoise"
    first = torch.ones(3, 13, 2, 2, dtype=torch.uint8).share_memory_()
    resized = torch.ones(3, 16, 2, 2, dtype=torch.uint8).share_memory_()
    worker.decode_chunk.side_effect = [lambda: first, lambda: resized, lambda: None]
    proxy = _CoLocatedVAEDecodeWorker(worker)

    assert proxy.decode_chunk(7, "latent")().shape[1] == 13
    assert proxy.decode_chunk(7, "latent")().shape[1] == 16
    assert proxy.decode_chunk(7, "latent")().shape[1] == 16
