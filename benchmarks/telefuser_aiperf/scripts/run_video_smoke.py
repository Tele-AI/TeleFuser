#!/usr/bin/env python3
"""Run a dependency-light OpenAI video API smoke benchmark."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--health-path", default="/v1/service/health")
    parser.add_argument(
        "--prompt",
        default="The character turns slightly toward the camera and breathes naturally, cinematic motion, consistent identity",
    )
    parser.add_argument("--reference-url", default="examples/data/101235-video-720_0.png")
    parser.add_argument("--model", default="telefuser-wan21-i2v-480p")
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--poll-timeout-s", type=float, default=3600.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    parser.add_argument("--artifacts-dir", default="artifacts/video_smoke")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _read_json(request: urllib.request.Request, timeout_s: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {request.full_url}: {exc.reason}") from exc
    payload = json.loads(body.decode("utf-8")) if body else {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {request.full_url}, got {type(payload).__name__}")
    return payload


def _get_json(url: str, timeout_s: float = 30.0) -> dict[str, Any]:
    return _read_json(urllib.request.Request(url, method="GET"), timeout_s)


def _post_form(url: str, fields: dict[str, Any], timeout_s: float = 30.0) -> dict[str, Any]:
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    return _read_json(request, timeout_s)


def _download(url: str, output_path: Path, timeout_s: float = 300.0) -> int:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc
    output_path.write_bytes(data)
    return len(data)


def _reference_path(raw_path: str) -> str:
    path = Path(raw_path)
    if path.exists():
        return str(path.resolve())
    return raw_path


def main() -> None:
    args = _build_parser().parse_args()
    artifacts_dir = Path(args.artifacts_dir) / time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    health = _get_json(_url(args.server_url, args.health_path))
    submit_started = time.perf_counter()
    create_response = _post_form(
        _url(args.server_url, "/v1/videos"),
        {
            "prompt": args.prompt,
            "reference_url": _reference_path(args.reference_url),
            "model": args.model,
            "size": args.size,
            "seconds": args.seconds,
            "seed": args.seed,
            "negative_prompt": args.negative_prompt,
        },
    )
    submit_latency_ms = (time.perf_counter() - submit_started) * 1000.0
    video_id = create_response.get("id")
    if not isinstance(video_id, str) or not video_id:
        raise RuntimeError(f"Create video response did not include an id: {create_response}")

    deadline = time.perf_counter() + args.poll_timeout_s
    statuses: list[dict[str, Any]] = []
    final_status: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        status = _get_json(_url(args.server_url, f"/v1/videos/{video_id}"))
        statuses.append(status)
        if status.get("status") in TERMINAL_STATUSES:
            final_status = status
            break
        time.sleep(args.poll_interval_s)
    if final_status is None:
        raise TimeoutError(f"Timed out waiting for video {video_id} after {args.poll_timeout_s}s")

    output_bytes = 0
    output_path = None
    if args.download and final_status.get("status") == "completed":
        output_path = artifacts_dir / f"{video_id}.mp4"
        output_bytes = _download(_url(args.server_url, f"/v1/videos/{video_id}/content"), output_path)
        if output_bytes <= 0:
            raise RuntimeError(f"Downloaded empty video content for {video_id}")

    summary = {
        "server_url": args.server_url,
        "health": health,
        "request": {
            "prompt": args.prompt,
            "reference_url": _reference_path(args.reference_url),
            "model": args.model,
            "size": args.size,
            "seconds": args.seconds,
            "seed": args.seed,
        },
        "create_response": create_response,
        "final_status": final_status,
        "status_samples": statuses,
        "metrics": {
            "submit_latency_ms": submit_latency_ms,
            "end_to_end_latency_ms": (time.perf_counter() - started) * 1000.0,
            "poll_count": len(statuses),
            "output_bytes": output_bytes,
        },
        "output_path": str(output_path) if output_path is not None else None,
    }
    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Video smoke status: {final_status.get('status')}")
    print(f"Video id: {video_id}")
    print(f"Artifacts: {artifacts_dir}")
    print(f"Summary JSON: {summary_path}")
    if final_status.get("status") != "completed":
        raise RuntimeError(f"Video smoke did not complete successfully: {final_status}")


if __name__ == "__main__":
    main()
