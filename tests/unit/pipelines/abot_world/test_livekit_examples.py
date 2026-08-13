from __future__ import annotations

import pytest

from examples.abot_world import abot_world_livekit as browser_example
from examples.abot_world import abot_world_livekit_service as service_example
from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline
from telefuser.pipelines.abot_world.service import ABotWorldLiveKitService


def test_livekit_service_entrypoint_builds_single_gpu_abot_service(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = object()
    captured: dict[str, object] = {}

    def fake_get_pipeline(**kwargs: object) -> object:
        captured.update(kwargs)
        return pipeline

    monkeypatch.setattr(service_example, "get_pipeline", fake_get_pipeline)
    service = service_example.get_service(gpu_num=1, gpu_ids=["3"])

    assert isinstance(service, ABotWorldLiveKitService)
    assert service.pipeline is pipeline
    assert captured == {"device_id": 3, "pipeline_class": ABotWorldInteractivePipeline}
    assert service.default_fps == 8
    assert service.default_session_config["fps"] == 8
    assert service.default_session_config["control_latent_frames"] == 2
    assert service.default_session_config["seed"] == 42
    assert service.default_session_config["prompt"] == service_example.DEFAULT_PROMPT
    assert str(service.default_session_config["image_path"]).endswith("84b90ad568b693d2.png")


def test_livekit_service_entrypoint_rejects_non_numeric_gpu_id() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        service_example.get_service(gpu_num=1, gpu_ids=["GPU-deadbeef"])


@pytest.mark.parametrize("gpu_num", [0, 2])
def test_livekit_service_entrypoint_rejects_unsupported_gpu_counts(gpu_num: int) -> None:
    with pytest.raises(ValueError, match="exactly one GPU"):
        service_example.get_service(gpu_num=gpu_num)


def test_livekit_browser_wrapper_sets_abot_defaults_before_shared_main(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, str] = {}

    def fake_main() -> None:
        observed["image_path"] = browser_example.livekit_bidirectional_demo.DEFAULT_IMAGE_PATH
        observed["prompt"] = browser_example.livekit_bidirectional_demo.DEFAULT_PROMPT

    monkeypatch.setattr(browser_example.livekit_bidirectional_demo, "main", fake_main)
    browser_example.main()

    assert observed["image_path"].endswith("84b90ad568b693d2.png")
    assert observed["prompt"] == browser_example.DEFAULT_PROMPT
