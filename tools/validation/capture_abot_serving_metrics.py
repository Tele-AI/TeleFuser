#!/usr/bin/env python3
"""Persist TeleFuser ABot serving metrics when Prometheus is unavailable.

This collector samples the public LiveKit service endpoints only. It is useful
on experiment nodes without Docker/DCGM/Grafana and intentionally uses a direct
urllib opener so inherited HTTP proxy environment variables cannot intercept
loopback requests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import OpenerDirector, ProxyHandler, Request, build_opener

_SCHEMA_VERSION = 1
_USER_AGENT = "TeleFuserServingMetricsCapture/1.0"


class CaptureError(RuntimeError):
    """Raised when one HTTP endpoint cannot be captured."""


@dataclass(frozen=True)
class CaptureConfig:
    """Validated command-line configuration for one capture run."""

    server_url: str
    duration_seconds: float
    interval_seconds: float
    timeout_seconds: float
    output_dir: Path


def _positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _server_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError("must be an http(s) URL with a host")
    if parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError("must not include a query string or fragment")
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> CaptureConfig:
    """Parse the standalone collector interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", type=_server_url, default="http://127.0.0.1:8088")
    parser.add_argument("--duration", type=_positive_seconds, required=True, help="Capture duration in seconds.")
    parser.add_argument("--interval", type=_positive_seconds, default=1.0, help="Sample interval in seconds.")
    parser.add_argument("--timeout", type=_positive_seconds, default=3.0, help="Per-request timeout in seconds.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory for capture artifacts.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    return CaptureConfig(
        server_url=args.server_url,
        duration_seconds=args.duration,
        interval_seconds=args.interval,
        timeout_seconds=args.timeout,
        output_dir=output_dir,
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_direct_opener() -> OpenerDirector:
    """Build an opener that ignores HTTP(S)_PROXY and all other proxy variables."""

    return build_opener(ProxyHandler({}))


def _request_text(opener: OpenerDirector, url: str, timeout_seconds: float) -> str:
    request = Request(url, headers={"Accept": "application/json, text/plain; q=0.9", "User-Agent": _USER_AGENT})
    try:
        with opener.open(request, timeout=timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(charset)
    except (HTTPError, URLError, OSError, TimeoutError, UnicodeDecodeError) as exc:
        raise CaptureError(f"{type(exc).__name__}: {exc}") from exc


def _prepare_output_dir(output_dir: Path) -> tuple[Path, Path, Path]:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    jsonl_path = output_dir / "serving-metrics.jsonl"
    prometheus_dir = output_dir / "prometheus"
    if manifest_path.exists() or jsonl_path.exists():
        raise ValueError(f"Refusing to overwrite an existing capture artifact in {output_dir}")
    if prometheus_dir.exists() and not prometheus_dir.is_dir():
        raise ValueError(f"Prometheus artifact path is not a directory: {prometheus_dir}")
    if prometheus_dir.exists() and any(prometheus_dir.iterdir()):
        raise ValueError(f"Refusing to reuse non-empty Prometheus artifact directory: {prometheus_dir}")
    prometheus_dir.mkdir(exist_ok=True)
    return manifest_path, jsonl_path, prometheus_dir


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_error(record: dict[str, Any], endpoint: str, error: Exception) -> None:
    record["errors"].append({"endpoint": endpoint, "error": f"{type(error).__name__}: {error}"})


def _capture_once(
    *,
    config: CaptureConfig,
    opener: OpenerDirector,
    sequence: int,
    started_monotonic: float,
    prometheus_dir: Path,
) -> dict[str, Any]:
    prometheus_url = f"{config.server_url}/metrics"
    json_url = f"{config.server_url}/v1/service/metrics/json"
    record: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "sequence": sequence,
        "observed_at_utc": _utc_timestamp(),
        "offset_seconds": round(time.monotonic() - started_monotonic, 6),
        "prometheus": {"url": prometheus_url},
        "serving": {"url": json_url},
        "errors": [],
    }

    try:
        prometheus_text = _request_text(opener, prometheus_url, config.timeout_seconds)
        prometheus_name = f"{sequence:06d}.prom"
        (prometheus_dir / prometheus_name).write_text(prometheus_text, encoding="utf-8")
        record["prometheus"].update(
            {
                "path": str(Path("prometheus") / prometheus_name),
                "bytes": len(prometheus_text.encode("utf-8")),
            }
        )
    except CaptureError as exc:
        _record_error(record, "metrics", exc)

    try:
        json_text = _request_text(opener, json_url, config.timeout_seconds)
        response = json.loads(json_text)
        if not isinstance(response, dict) or not isinstance(response.get("serving"), dict):
            raise CaptureError("response does not contain an object-valued 'serving' field")
        record["serving"]["snapshot"] = response["serving"]
    except (CaptureError, json.JSONDecodeError) as exc:
        _record_error(record, "metrics_json", exc)

    return record


def capture(config: CaptureConfig) -> dict[str, Any]:
    """Capture service metrics and always write a terminal manifest after start."""

    manifest_path, jsonl_path, prometheus_dir = _prepare_output_dir(config.output_dir)
    started_monotonic = time.monotonic()
    started_at_utc = _utc_timestamp()
    deadline = started_monotonic + config.duration_seconds
    statistics = {
        "attempted": 0,
        "prometheus_saved": 0,
        "serving_json_saved": 0,
        "complete": 0,
        "errors": 0,
    }
    status = "completed"
    failure: Exception | None = None
    opener = build_direct_opener()

    try:
        with jsonl_path.open("x", encoding="utf-8") as jsonl_file:
            while True:
                if statistics["attempted"] and time.monotonic() >= deadline:
                    break
                sequence = statistics["attempted"] + 1
                record = _capture_once(
                    config=config,
                    opener=opener,
                    sequence=sequence,
                    started_monotonic=started_monotonic,
                    prometheus_dir=prometheus_dir,
                )
                jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
                jsonl_file.flush()

                statistics["attempted"] += 1
                prometheus_saved = "path" in record["prometheus"]
                serving_json_saved = "snapshot" in record["serving"]
                statistics["prometheus_saved"] += int(prometheus_saved)
                statistics["serving_json_saved"] += int(serving_json_saved)
                statistics["complete"] += int(prometheus_saved and serving_json_saved)
                statistics["errors"] += len(record["errors"])

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(config.interval_seconds, remaining))
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception as exc:
        status = "failed"
        failure = exc
    finally:
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "status": status,
            "started_at_utc": started_at_utc,
            "completed_at_utc": _utc_timestamp(),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
            "configuration": {
                "server_url": config.server_url,
                "duration_seconds": config.duration_seconds,
                "interval_seconds": config.interval_seconds,
                "timeout_seconds": config.timeout_seconds,
                "proxy_mode": "direct_no_proxy",
            },
            "samples": statistics,
            "artifacts": {
                "serving_metrics_jsonl": jsonl_path.name,
                "prometheus_snapshots_directory": prometheus_dir.name,
            },
        }
        _write_json(manifest_path, manifest)

    if failure is not None:
        raise failure
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collector and report the durable artifact path."""

    try:
        config = parse_args(argv)
        manifest = capture(config)
    except (CaptureError, OSError, ValueError) as exc:
        print(f"Metrics capture failed: {exc}", file=sys.stderr)
        return 2

    samples = manifest["samples"]
    print(
        f"Metrics capture {manifest['status']}: {samples['complete']}/{samples['attempted']} complete samples; "
        f"manifest: {config.output_dir / 'manifest.json'}"
    )
    if manifest["status"] == "interrupted":
        return 130
    return 0 if samples["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
