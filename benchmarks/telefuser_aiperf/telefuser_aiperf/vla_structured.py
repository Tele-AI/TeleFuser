"""AIPerf endpoint and HTTP polling transport for TeleFuser VLA actions."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import math
import struct
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import orjson
from aiperf.common.enums import MetricFlags, MetricSizeUnit, MetricTimeUnit
from aiperf.common.exceptions import NoMetricValue, NotInitializedError
from aiperf.common.models import (
    BaseResponseData,
    ErrorDetails,
    InferenceServerResponse,
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
    TextResponse,
)
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.aiohttp_transport import AioHttpTransport

_RESULT_FIELDS = frozenset(
    {
        "canonical_normalized_actions",
        "horizon",
        "action_dim",
        "checkpoint_variant",
        "policy_verified",
        "verification_status",
    }
)
_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_EXPECTED_HORIZON = 50
_EXPECTED_ACTION_DIM = 55
_POLL_INTERVAL_SECONDS = 0.05


@dataclass(slots=True)
class VlaActionResponseData(BaseResponseData):
    """Validated summary of one VLA action chunk."""

    task_id: str
    horizon: int
    action_dim: int
    value_count: int
    sha256_float64_le: str
    checkpoint_variant: str
    policy_verified: bool
    verification_status: str
    inference_time_s: float | None = None
    peak_memory_mb: float | None = None


class VlaInferenceTimeMetric(BaseRecordMetric[float]):
    """Expose server-measured VLA inference time to AIPerf."""

    tag = "vla_inference_time"
    header = "VLA Inference Time"
    short_header = "VLA Inference"
    unit = MetricTimeUnit.SECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    display_order = 310
    flags = MetricFlags.NONE

    def _parse_record(self, record: ParsedResponseRecord, record_metrics: MetricRecordDict) -> float:
        del record_metrics
        for response in reversed(record.responses):
            if isinstance(response.data, VlaActionResponseData) and response.data.inference_time_s is not None:
                return response.data.inference_time_s
        raise NoMetricValue("VLA inference time is not available in the structured response.")


class VlaPeakMemoryMetric(BaseRecordMetric[float]):
    """Expose server-measured peak accelerator memory to AIPerf."""

    tag = "vla_peak_memory"
    header = "VLA Peak Memory"
    short_header = "VLA Peak Memory"
    unit = MetricSizeUnit.MEGABYTES
    display_order = 311
    flags = MetricFlags.NONE

    def _parse_record(self, record: ParsedResponseRecord, record_metrics: MetricRecordDict) -> float:
        del record_metrics
        for response in reversed(record.responses):
            if isinstance(response.data, VlaActionResponseData) and response.data.peak_memory_mb is not None:
                return response.data.peak_memory_mb
        raise NoMetricValue("VLA peak memory is not available in the structured response.")


def _finite_number(value: Any, *, name: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    return normalized


def validate_state(state: Any) -> list[float]:
    """Validate the canonical 14-dimensional VLA state vector."""
    if not isinstance(state, list) or len(state) != 14:
        raise ValueError("VLA state must contain exactly 14 values")
    return [float(_finite_number(value, name="VLA state value")) for value in state]


def image_content_to_base64(content: str) -> str:
    """Normalize an AIPerf image data URL to raw validated base64."""
    if content.lower().startswith(("http://", "https://")):
        raise ValueError("TeleFuser VLA camera inputs must be inline image data, not URLs")
    encoded = content
    if content.startswith("data:"):
        try:
            header, encoded = content.split(",", 1)
        except ValueError as error:
            raise ValueError("VLA image data URL is missing a comma") from error
        if ";base64" not in header.lower() or not header.lower().startswith("data:image/"):
            raise ValueError("VLA camera input must be a base64 image data URL")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("VLA camera input is not valid base64") from error
    if not decoded:
        raise ValueError("VLA camera input is empty")
    return encoded


def build_vla_payload(
    instruction: str,
    image_content: str,
    *,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the stable TeleFuser structured action request body."""
    if not instruction.strip():
        raise ValueError("VLA instruction must not be empty")
    parameters = dict(extra or {})
    unsupported = sorted(set(parameters).difference({"state", "seed"}))
    if unsupported:
        raise ValueError(f"Unsupported VLA request fields: {', '.join(unsupported)}")
    if "state" not in parameters:
        raise ValueError("VLA dataset entry must provide state in extra")
    state = validate_state(parameters["state"])
    seed = parameters.get("seed", 7)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("VLA seed must be an integer")
    image_base64 = image_content_to_base64(image_content)
    return {
        "task": "vla_action",
        "instruction": instruction,
        "state": state,
        "camera_high": image_base64,
        "camera_left_wrist": image_base64,
        "camera_right_wrist": image_base64,
        "seed": seed,
    }


def summarize_action_result(result: Any) -> dict[str, Any]:
    """Validate a 50 x 55 normalized action chunk and return bounded facts."""
    if not isinstance(result, dict) or set(result) != set(_RESULT_FIELDS):
        observed = sorted(result) if isinstance(result, dict) else type(result).__name__
        raise ValueError(f"VLA result fields changed: {observed}")
    actions = result.get("canonical_normalized_actions")
    if not isinstance(actions, list) or len(actions) != _EXPECTED_HORIZON:
        raise ValueError(f"VLA action horizon must be {_EXPECTED_HORIZON}")
    if result.get("horizon") != _EXPECTED_HORIZON or result.get("action_dim") != _EXPECTED_ACTION_DIM:
        raise ValueError("VLA action dimension metadata changed")

    digest = hashlib.sha256()
    value_count = 0
    minimum = math.inf
    maximum = -math.inf
    for row_index, row in enumerate(actions):
        if not isinstance(row, list) or len(row) != _EXPECTED_ACTION_DIM:
            raise ValueError(f"VLA action row {row_index} must contain {_EXPECTED_ACTION_DIM} values")
        for raw_value in row:
            value = float(_finite_number(raw_value, name="VLA action value"))
            digest.update(struct.pack("<d", value))
            value_count += 1
            minimum = min(minimum, value)
            maximum = max(maximum, value)

    checkpoint_variant = result.get("checkpoint_variant")
    verification_status = result.get("verification_status")
    policy_verified = result.get("policy_verified")
    if not isinstance(checkpoint_variant, str) or not checkpoint_variant:
        raise ValueError("VLA checkpoint_variant must be a non-empty string")
    if not isinstance(verification_status, str) or not verification_status:
        raise ValueError("VLA verification_status must be a non-empty string")
    if not isinstance(policy_verified, bool):
        raise ValueError("VLA policy_verified must be boolean")
    return {
        "horizon": _EXPECTED_HORIZON,
        "action_dim": _EXPECTED_ACTION_DIM,
        "value_count": value_count,
        "minimum": minimum,
        "maximum": maximum,
        "sha256_float64_le": digest.hexdigest(),
        "checkpoint_variant": checkpoint_variant,
        "policy_verified": policy_verified,
        "verification_status": verification_status,
    }


def build_task_status_url(submit_url: str, task_id: str) -> str:
    """Build the native task status URL on the same origin as submission."""
    parsed = urlsplit(submit_url)
    path = f"/v1/tasks/{quote(task_id, safe='')}/status"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


class TeleFuserVlaStructuredEndpoint(BaseEndpoint):
    """Format and parse TeleFuser's native VLA structured API."""

    def format_payload(self, request_info: RequestInfo) -> dict[str, Any]:
        if not request_info.turns:
            raise ValueError("TeleFuser VLA endpoint requires one dataset turn")
        turn = request_info.turns[-1]
        if not turn.texts or not turn.texts[0].contents:
            raise ValueError("TeleFuser VLA endpoint requires one instruction")
        if not turn.images or not turn.images[0].contents:
            raise ValueError("TeleFuser VLA endpoint requires one camera image")
        merged_extra = dict(request_info.model_endpoint.endpoint.extra or {})
        merged_extra.update(turn.extra_body or {})
        return build_vla_payload(
            turn.texts[0].contents[0],
            turn.images[0].contents[0],
            extra=merged_extra,
        )

    def parse_response(self, response: InferenceServerResponse) -> ParsedResponse | None:
        body = response.get_json()
        if not isinstance(body, dict) or body.get("status") != "completed":
            return None
        summary = body.get("action_summary")
        if not isinstance(summary, dict):
            raise ValueError("completed VLA benchmark response has no action_summary")
        data = VlaActionResponseData(
            task_id=str(body["task_id"]),
            horizon=int(summary["horizon"]),
            action_dim=int(summary["action_dim"]),
            value_count=int(summary["value_count"]),
            sha256_float64_le=str(summary["sha256_float64_le"]),
            checkpoint_variant=str(summary["checkpoint_variant"]),
            policy_verified=bool(summary["policy_verified"]),
            verification_status=str(summary["verification_status"]),
            inference_time_s=body.get("inference_time_s"),
            peak_memory_mb=body.get("peak_memory_mb"),
        )
        return ParsedResponse(perf_ns=response.perf_ns, data=data, metadata={"media_type": "structured"})


class TeleFuserStructuredHttpTransport(AioHttpTransport):
    """HTTP JSON transport for TeleFuser submit/poll structured tasks."""

    @classmethod
    def metadata(cls) -> TransportMetadata:
        return TransportMetadata(transport_type="telefuser_structured_http", url_schemes=[])

    @staticmethod
    def _parse_json_record(record: RequestRecord, context: str) -> tuple[dict[str, Any], TextResponse] | ErrorDetails:
        if record.error:
            return record.error
        if not record.responses or not isinstance(record.responses[0], TextResponse):
            return ErrorDetails(type="VlaStructuredError", message=f"No JSON response from {context}", code=500)
        response = record.responses[0]
        try:
            body = orjson.loads(response.text)
        except orjson.JSONDecodeError:
            return ErrorDetails(type="VlaStructuredError", message=f"Invalid JSON from {context}", code=500)
        if not isinstance(body, dict):
            return ErrorDetails(type="VlaStructuredError", message=f"Non-object JSON from {context}", code=500)
        return body, response

    async def send_request(
        self,
        request_info: RequestInfo,
        payload: dict[str, Any],
        *,
        first_token_callback: Any = None,
    ) -> RequestRecord:
        """Submit one action task, poll terminal state, and retain bounded facts."""
        del first_token_callback
        if self.aiohttp_client is None:
            raise NotInitializedError("AioHttpClient not initialized")
        start_ns = time.perf_counter_ns()
        headers = self.build_headers(request_info)
        responses: list[TextResponse] = []

        def make_record(error: ErrorDetails | None = None, status: int | None = None) -> RequestRecord:
            return RequestRecord(
                request_info=request_info,
                request_headers=headers,
                start_perf_ns=start_ns,
                end_perf_ns=time.perf_counter_ns(),
                responses=responses,
                error=error,
                status=status,
            )

        try:
            submit_url = self.build_url(request_info)
            submitted = await self.aiohttp_client.post_request(submit_url, orjson.dumps(payload), headers)
            parsed_submit = self._parse_json_record(submitted, "VLA task submission")
            if isinstance(parsed_submit, ErrorDetails):
                return make_record(error=parsed_submit, status=submitted.status)
            submit_body, submit_response = parsed_submit
            task_id = submit_body.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                return make_record(
                    error=ErrorDetails(
                        type="VlaStructuredError",
                        message="VLA submission returned no task_id",
                        code=500,
                    )
                )
            responses.append(
                TextResponse(
                    perf_ns=submit_response.perf_ns,
                    text=orjson.dumps({"task_id": task_id, "status": "pending"}).decode(),
                    content_type="application/json",
                )
            )

            status_url = build_task_status_url(submit_url, task_id)
            timeout = request_info.model_endpoint.endpoint.timeout
            deadline = time.monotonic() + timeout if timeout > 0 else math.inf
            while time.monotonic() < deadline:
                polled = await self.aiohttp_client.get_request(status_url, headers)
                parsed_poll = self._parse_json_record(polled, "VLA task status")
                if isinstance(parsed_poll, ErrorDetails):
                    return make_record(error=parsed_poll, status=polled.status)
                status_body, status_response = parsed_poll
                status = status_body.get("status") or status_body.get("task_status")
                if status not in _TERMINAL_STATUSES:
                    await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                    continue
                if status != "completed":
                    return make_record(
                        error=ErrorDetails(
                            type="VlaStructuredError",
                            message=f"VLA task {task_id} ended with {status}: {status_body.get('error')}",
                            code=500,
                        ),
                        status=polled.status,
                    )
                action_summary = summarize_action_result(status_body.get("result"))
                bounded_status = {
                    "task_id": task_id,
                    "status": "completed",
                    "inference_time_s": _finite_number(
                        status_body.get("inference_time_s"), name="inference_time_s", allow_none=True
                    ),
                    "peak_memory_mb": _finite_number(
                        status_body.get("peak_memory_mb"), name="peak_memory_mb", allow_none=True
                    ),
                    "action_summary": action_summary,
                }
                responses.append(
                    TextResponse(
                        perf_ns=status_response.perf_ns,
                        text=orjson.dumps(bounded_status).decode(),
                        content_type="application/json",
                    )
                )
                return make_record(status=200)
            return make_record(
                error=ErrorDetails(
                    type="TimeoutError",
                    message=f"VLA task {task_id} timed out after {timeout:g}s",
                    code=504,
                ),
                status=504,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return make_record(error=ErrorDetails.from_exception(error))


ENDPOINT_METADATA = {
    "endpoint_path": "/v1/tasks/structured",
    "supports_streaming": False,
    "tokenizes_input": False,
    "produces_tokens": False,
    "supports_images": True,
    "requires_polling": True,
    "requires_form_data": False,
    "metrics_title": "TeleFuser VLA Structured Metrics",
    "service_kind": "telefuser_vla",
}

TRANSPORT_METADATA = {
    "transport_type": "telefuser_structured_http",
    "url_schemes": [],
}


__all__ = [
    "ENDPOINT_METADATA",
    "TRANSPORT_METADATA",
    "TeleFuserStructuredHttpTransport",
    "TeleFuserVlaStructuredEndpoint",
    "VlaActionResponseData",
    "build_task_status_url",
    "build_vla_payload",
    "image_content_to_base64",
    "summarize_action_result",
    "validate_state",
]
