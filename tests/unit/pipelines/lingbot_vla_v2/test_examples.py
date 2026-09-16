from __future__ import annotations

from click.testing import CliRunner

from examples.lingbot_vla_v2 import (
    lingbot_vla_v2_inference,
    lingbot_vla_v2_native_service,
    lingbot_vla_v2_vla_server,
)


def test_direct_inference_cli_exposes_cuda_graph() -> None:
    result = CliRunner().invoke(lingbot_vla_v2_inference.main, ["--help"])

    assert result.exit_code == 0
    assert "--cuda-graph" in result.output


def test_direct_inference_forwards_cuda_graph(monkeypatch) -> None:
    captured: dict[str, object] = {}
    pipeline = object()

    def fake_get_pipeline(model_root: str, qwen3vl_root: str, **kwargs: object) -> object:
        captured.update({"model_root": model_root, "qwen3vl_root": qwen3vl_root, **kwargs})
        return pipeline

    monkeypatch.setattr(lingbot_vla_v2_inference, "get_lingbot_vla_v2_pipeline", fake_get_pipeline)

    result = lingbot_vla_v2_inference.get_pipeline(
        "model",
        "qwen",
        device="cuda:1",
        quantization="fused-fp8-graph",
        cuda_graph=True,
    )

    assert result is pipeline
    assert captured == {
        "model_root": "model",
        "qwen3vl_root": "qwen",
        "device": "cuda:1",
        "quantization": "fused-fp8-graph",
        "cuda_graph": True,
    }


def test_native_service_forwards_cuda_graph(monkeypatch) -> None:
    captured: dict[str, object] = {}
    pipeline = object()

    def fake_get_pipeline(model_root: str, qwen3vl_root: str, **kwargs: object) -> object:
        captured.update({"model_root": model_root, "qwen3vl_root": qwen3vl_root, **kwargs})
        return pipeline

    monkeypatch.setattr(lingbot_vla_v2_native_service, "get_lingbot_vla_v2_pipeline", fake_get_pipeline)
    monkeypatch.setitem(lingbot_vla_v2_native_service.PPL_CONFIG, "quantization", "fused-fp8-graph")
    monkeypatch.setitem(lingbot_vla_v2_native_service.PPL_CONFIG, "cuda_graph", True)

    result = lingbot_vla_v2_native_service.get_pipeline()

    assert result is pipeline
    assert captured["warmup"] is True
    assert captured["quantization"] == "fused-fp8-graph"
    assert captured["cuda_graph"] is True


def test_native_service_exposes_worker_local_vla_provider() -> None:
    provider = lingbot_vla_v2_native_service.get_vla_provider(object())
    assert provider.metadata() == {
        "model_ids": ["lingbot-vla-v2"],
        "embodiment_ids": ["robotwin"],
    }


def test_generic_vla_server_starts_pool_with_optional_provider(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    class FakePool:
        def __init__(self, **kwargs: object) -> None:
            captured["pool"] = kwargs

        def start_all(self, **kwargs: object) -> bool:
            captured["start"] = kwargs
            return True

    monkeypatch.setattr(lingbot_vla_v2_vla_server, "PipelinePool", FakePool)
    monkeypatch.setattr(
        lingbot_vla_v2_vla_server,
        "create_pipeline_pool_vla_session_app",
        lambda pool: sentinel,
    )
    app = lingbot_vla_v2_vla_server.create_app(parallelism=1, num_replicas=1)

    assert app is sentinel
    start = captured["start"]
    assert isinstance(start, dict)
    assert start["vla_provider_factory"] == "get_vla_provider"
    assert start["task"] == "vla_action"
