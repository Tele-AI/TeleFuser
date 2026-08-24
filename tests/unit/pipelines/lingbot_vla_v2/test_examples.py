from __future__ import annotations

from click.testing import CliRunner

from examples.lingbot_vla_v2 import lingbot_vla_v2_inference, lingbot_vla_v2_native_service


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
