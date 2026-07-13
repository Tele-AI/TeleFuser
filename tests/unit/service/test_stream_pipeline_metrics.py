from __future__ import annotations

from types import SimpleNamespace

from telefuser.service.core import stream_pipeline_service as stream_module
from telefuser.service.core.stream_pipeline_service import StreamPipelineService


class _FakeServerPushService:
    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    async def serve(self, request):
        if False:
            yield request


def test_stream_service_metadata_exposes_startup_performance(
    monkeypatch,
) -> None:
    service = _FakeServerPushService()
    module = SimpleNamespace(get_service=lambda: service)
    monkeypatch.setattr(
        stream_module,
        "load_pipeline_module",
        lambda *args, **kwargs: (module, "fake_module"),
    )
    monkeypatch.setattr(stream_module, "visible_cuda_devices", lambda: ())
    monkeypatch.setattr(
        stream_module,
        "finish_runtime_measurement",
        lambda measurement: {"seconds": 4.0, "memory": []},
    )
    monkeypatch.setattr(
        stream_module,
        "collect_runtime_environment",
        lambda *args, **kwargs: {"torch_version": "test"},
    )
    pipeline_service = StreamPipelineService()

    assert pipeline_service.start_service("fake.py", skip_validation=True) is True
    metadata = pipeline_service.server_metadata()

    assert metadata["performance"]["phases"] == [{"name": "pipeline_init", "seconds": 4.0, "memory": []}]
    assert metadata["environment"] == {"torch_version": "test"}
