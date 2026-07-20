from unittest.mock import MagicMock

from PIL import Image

from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline
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
