from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tools.validation import capture_abot_serving_metrics as capture


class _MetricsHandler(BaseHTTPRequestHandler):
    paths: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        type(self).paths.append(self.path)
        if self.path == "/metrics":
            body = b'telefuser_serving_sessions{state="active"} 4\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        elif self.path == "/v1/service/metrics/json":
            body = json.dumps(
                {
                    "serving": {
                        "summary": {"sessions": {"active": 4}},
                        "counters": {'telefuser_serving_chunks_total{result="processed"}': 12},
                    }
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            body = b"not found"
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def test_capture_writes_prometheus_jsonl_and_manifest_without_proxy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _MetricsHandler.paths = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")

    try:
        output_dir = tmp_path / "metrics"
        config = capture.CaptureConfig(
            server_url=f"http://127.0.0.1:{server.server_port}",
            duration_seconds=0.04,
            interval_seconds=0.01,
            timeout_seconds=1.0,
            output_dir=output_dir,
        )
        manifest = capture.capture(config)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    assert manifest["status"] == "completed"
    assert manifest["configuration"]["proxy_mode"] == "direct_no_proxy"
    assert manifest["samples"]["complete"] >= 1
    assert _MetricsHandler.paths.count("/metrics") >= 1
    assert _MetricsHandler.paths.count("/v1/service/metrics/json") >= 1

    records = [
        json.loads(line) for line in (output_dir / "serving-metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == manifest["samples"]["attempted"]
    assert records[0]["serving"]["snapshot"]["summary"]["sessions"]["active"] == 4
    first_prometheus = output_dir / records[0]["prometheus"]["path"]
    assert "telefuser_serving_sessions" in first_prometheus.read_text(encoding="utf-8")

    stored_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert stored_manifest == manifest
