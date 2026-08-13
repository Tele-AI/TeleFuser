from __future__ import annotations

import base64
import math

import orjson
import pytest
from aiperf.common.models import TextResponse
from aiperf.plugin import plugins
from telefuser_aiperf import register_plugins
from telefuser_aiperf.vla_structured import (
    TeleFuserStructuredHttpTransport,
    TeleFuserVlaStructuredEndpoint,
    VlaActionResponseData,
    build_task_status_url,
    build_vla_payload,
    summarize_action_result,
)


def _action_result(value: float = 0.25) -> dict:
    return {
        "canonical_normalized_actions": [[value] * 55 for _ in range(50)],
        "horizon": 50,
        "action_dim": 55,
        "checkpoint_variant": "base",
        "policy_verified": False,
        "verification_status": "unverified_official_6b_base",
    }


def test_registration_uses_aiperf_endpoint_and_transport_plugins() -> None:
    register_plugins(replace=True)

    endpoint_class = plugins.get_class("endpoint", "telefuser_vla_structured")
    transport_class = plugins.get_class("transport", "telefuser_structured_http")

    assert endpoint_class is TeleFuserVlaStructuredEndpoint
    assert transport_class is TeleFuserStructuredHttpTransport
    assert plugins.get_endpoint_metadata("telefuser_vla_structured").requires_polling is True

    from aiperf.metrics.metric_registry import MetricRegistry

    assert MetricRegistry.get_class("vla_inference_time").__name__ == "VlaInferenceTimeMetric"
    assert MetricRegistry.get_class("vla_peak_memory").__name__ == "VlaPeakMemoryMetric"


def test_build_vla_payload_reuses_inline_image_for_three_cameras() -> None:
    encoded = base64.b64encode(b"image bytes").decode()

    payload = build_vla_payload(
        "pick up the object",
        f"data:image/png;base64,{encoded}",
        extra={"state": [0.0] * 14, "seed": 7},
    )

    assert payload["task"] == "vla_action"
    assert payload["camera_high"] == encoded
    assert payload["camera_left_wrist"] == encoded
    assert payload["camera_right_wrist"] == encoded
    assert payload["state"] == [0.0] * 14


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"state": [0.0] * 13}, "exactly 14"),
        ({"state": [0.0] * 13 + [math.nan]}, "finite"),
        ({"state": [0.0] * 14, "unknown": 1}, "Unsupported"),
    ],
)
def test_build_vla_payload_rejects_contract_drift(extra: dict, match: str) -> None:
    encoded = base64.b64encode(b"image bytes").decode()

    with pytest.raises(ValueError, match=match):
        build_vla_payload("instruction", encoded, extra=extra)


def test_summarize_action_result_validates_shape_and_omits_full_actions() -> None:
    summary = summarize_action_result(_action_result())

    assert summary["horizon"] == 50
    assert summary["action_dim"] == 55
    assert summary["value_count"] == 2750
    assert summary["minimum"] == 0.25
    assert len(summary["sha256_float64_le"]) == 64
    assert "canonical_normalized_actions" not in summary


def test_summarize_action_result_rejects_non_finite_action() -> None:
    result = _action_result()
    result["canonical_normalized_actions"][0][0] = math.inf

    with pytest.raises(ValueError, match="finite"):
        summarize_action_result(result)


def test_endpoint_parses_bounded_completed_response() -> None:
    endpoint = object.__new__(TeleFuserVlaStructuredEndpoint)
    summary = summarize_action_result(_action_result())
    response = TextResponse(
        perf_ns=123,
        content_type="application/json",
        text=orjson.dumps(
            {
                "task_id": "task-1",
                "status": "completed",
                "inference_time_s": 0.65,
                "peak_memory_mb": None,
                "action_summary": summary,
            }
        ).decode(),
    )

    parsed = endpoint.parse_response(response)

    assert parsed is not None
    assert isinstance(parsed.data, VlaActionResponseData)
    assert parsed.data.task_id == "task-1"
    assert parsed.data.value_count == 2750
    assert parsed.metadata == {"media_type": "structured"}


def test_task_status_url_uses_native_structured_route() -> None:
    assert (
        build_task_status_url("http://127.0.0.1:18080/v1/tasks/structured", "task id")
        == "http://127.0.0.1:18080/v1/tasks/task%20id/status"
    )
