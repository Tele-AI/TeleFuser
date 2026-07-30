"""LingBot-World-Fast LiveKit control demo.

The page reuses the shared control UI asset to keep the prompt, image, controls,
and telemetry behavior consistent across interactive examples.

Usage:
    # 1. Start a LiveKit server and export its URL/key/secret.
    # 2. Start TeleFuser:
    telefuser stream-serve examples/lingbot/lingbot_world_fast_image_to_video_h100.py --skip-validation
    # 3. Start this browser client:
    python examples/stream_server/livekit_bidirectional_demo.py --server-url http://localhost:8088
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import runpy
import threading
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

DEFAULT_SERVER_URL = "http://localhost:8088"
DEFAULT_PORT = 8092
LIVEKIT_CLIENT_URL = "https://cdn.jsdelivr.net/npm/livekit-client@2.21.0/dist/livekit-client.umd.min.js"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTROL_DEMO_UI_PATH = _PROJECT_ROOT / "examples" / "stream_server" / "_control_demo_ui.py"


def _shared_demo_parts() -> tuple[str, str, str, str, str]:
    namespace = runpy.run_path(str(_CONTROL_DEMO_UI_PATH))
    template = str(namespace["HTML_TEMPLATE"])
    script = template.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    shell = template.split("<script>", 1)[0]
    utilities = script[script.index("function $(id)") : script.index("function describeCandidate")]
    controls = script[script.index("function setControlActive") :]
    return shell, utilities, controls, str(namespace["DEFAULT_IMAGE_PATH"]), str(namespace["DEFAULT_PROMPT"])


_HTML_SHELL, _UTILITIES, _CONTROLS, DEFAULT_IMAGE_PATH, DEFAULT_PROMPT = _shared_demo_parts()

_LIVEKIT_SCRIPT = r"""
const SERVER_URL = __SERVER_URL__;
const DEFAULT_IMAGE_PATH = __DEFAULT_IMAGE_PATH__;
const DEFAULT_PROMPT = __PROMPT__;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
const CONTROL_HEARTBEAT_MS = 1000;
const CONTROL_TOPIC = "tf.control";
const STATUS_TOPIC = "tf.status";
const METRICS_TOPIC = "tf.metrics";
const TURN_RTC_CONFIG = {
  iceServers: [{
    urls: ["turn:127.0.0.1:3478?transport=tcp"],
    username: "livekit-demo",
    credential: "livekit-demo-password",
  }],
  iceTransportPolicy: "relay",
};

let room = null;
let dc = null;
let sessionId = null;
let cleaning = false;
const pressedControls = new Set();
const keyToControl = {
  ArrowUp: "w",
  ArrowDown: "s",
  ArrowLeft: "a",
  ArrowRight: "d",
  KeyW: "w",
  KeyA: "a",
  KeyS: "s",
  KeyD: "d",
  KeyI: "i",
  KeyJ: "j",
  KeyK: "k",
  KeyL: "l",
};

__UTILITIES__

function createDataSender(activeRoom) {
  return {
    readyState: "open",
    send(text) {
      const bytes = new TextEncoder().encode(text);
      activeRoom.localParticipant.publishData(bytes, { reliable: true, topic: CONTROL_TOPIC }).catch(error => {
        setStatus("Control send failed: " + error.message);
      });
    },
  };
}

function handleStatusMessage(payload, topic) {
  if (topic && topic !== STATUS_TOPIC && topic !== METRICS_TOPIC) return;
  const text = new TextDecoder().decode(payload);
  log("in", text);
  try {
    const msg = JSON.parse(text);
    let data = msg.data || msg;
    if (data.data && typeof data.data === "object") data = { ...data, ...data.data };
    updateTelemetry(data.stream_progress, {
      ...(data.runtime_metrics || {}),
      output_cadence_seconds: data.output_cadence_seconds,
      pipeline_residence_seconds: data.pipeline_residence_seconds,
      applied_control_latency_seconds: data.applied_control_latency_seconds,
      chunk_elapsed_seconds: data.chunk_elapsed_seconds,
      control_to_chunk_seconds: data.control_to_chunk_seconds,
    });
    if (msg.error || data.error) {
      setStatus("Server error: " + (msg.error || data.error));
    } else if (data.stage) {
      const suffix = data.index !== undefined ? " #" + data.index : "";
      const controls = data.controls ? " [" + data.controls.join(",") + "]" : "";
      setStatus("Server: " + data.stage + suffix + controls);
    } else if (msg.type === "done") {
      setStatus("Done.");
    }
  } catch (error) {
    setStatus("Invalid status message: " + error.message);
  }
}

__CONTROLS__

$("connect").onclick = async () => {
  const prompt = $("prompt").value.trim();
  const imageFile = $("image-file").files[0];
  if (!prompt) {
    setStatus("Prompt is required.");
    return;
  }
  if (imageFile && !imageFile.type.startsWith("image/")) {
    setStatus("Please select an image file.");
    return;
  }
  if (imageFile && imageFile.size > MAX_IMAGE_BYTES) {
    setStatus("Image must not exceed 10 MiB.");
    return;
  }

  setStatus("Creating LiveKit session...");
  $("connect").disabled = true;
  try {
    const image = imageFile ? await readFileAsDataUrl(imageFile) : null;
    const requestBody = {
      identity: "controller-" + crypto.randomUUID(),
      prompt,
      config: image ? { image } : {},
    };
    if (!image) requestBody.image_path = DEFAULT_IMAGE_PATH;

    const response = await fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/sessions",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      },
      30000,
    );
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || response.statusText);
    }

    const created = await response.json();
    sessionId = created.session_id;
    room = new LivekitClient.Room({ adaptiveStream: false, dynacast: false });
    room.on(LivekitClient.RoomEvent.TrackSubscribed, track => {
      if (track.kind === LivekitClient.Track.Kind.Video) track.attach($("output-video"));
    });
    room.on(LivekitClient.RoomEvent.TrackUnsubscribed, track => {
      track.detach($("output-video"));
    });
    room.on(LivekitClient.RoomEvent.DataReceived, (payload, _participant, _kind, topic) => {
      handleStatusMessage(payload, topic);
    });
    room.on(LivekitClient.RoomEvent.Reconnecting, () => setStatus("LiveKit reconnecting..."));
    room.on(LivekitClient.RoomEvent.Reconnected, () => setStatus("LiveKit reconnected."));
    room.on(LivekitClient.RoomEvent.Disconnected, () => {
      if (!cleaning) {
        setStatus("LiveKit disconnected.");
        cleanup();
      }
    });

    setStatus("Joining LiveKit room...");
    await room.connect(created.livekit_url, created.token, { rtcConfig: TURN_RTC_CONFIG });
    dc = createDataSender(room);
    setStatus("Connected. Waiting for LingBot output...");
    $("stop").style.display = "inline-block";
  } catch (error) {
    setStatus("Error: " + error.message);
    await cleanup();
  }
};

$("stop").onclick = async () => {
  setStatus("Stopping...");
  releaseAllControls(true);
  if (dc && dc.readyState === "open") {
    const msg = JSON.stringify({ type: "stop" });
    dc.send(msg);
    log("out", msg);
  }
  await cleanup();
};

async function cleanup() {
  if (cleaning) return;
  cleaning = true;
  releaseAllControls(false);
  dc = null;

  const closingSessionId = sessionId;
  sessionId = null;
  if (closingSessionId) {
    await fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/sessions/" + closingSessionId,
      { method: "DELETE" },
      5000,
    ).catch(() => {});
  }

  const closingRoom = room;
  room = null;
  if (closingRoom) await closingRoom.disconnect().catch(() => {});
  $("output-video").srcObject = null;
  $("connect").disabled = false;
  $("stop").style.display = "none";
  setTimeout(() => { cleaning = false; }, 0);
}

window.addEventListener("pagehide", () => {
  if (!sessionId) return;
  fetch(SERVER_URL + "/v1/stream/sessions/" + sessionId, { method: "DELETE", keepalive: true }).catch(() => {});
});

$("prompt").value = DEFAULT_PROMPT;
"""


def _render_html(server_url: str) -> str:
    script = (
        _LIVEKIT_SCRIPT.replace("__SERVER_URL__", json.dumps(server_url))
        .replace("__DEFAULT_IMAGE_PATH__", json.dumps(DEFAULT_IMAGE_PATH))
        .replace("__PROMPT__", json.dumps(DEFAULT_PROMPT))
        .replace("__UTILITIES__", _UTILITIES)
        .replace("__CONTROLS__", _CONTROLS)
    )
    return f'{_HTML_SHELL}<script src="{LIVEKIT_CLIENT_URL}"></script>\n<script>{script}</script>\n</body>\n</html>'


def main() -> None:
    parser = argparse.ArgumentParser(description="LingBot-World-Fast LiveKit control demo")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="LiveKit API server base URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local HTTP server port")
    parser.add_argument(
        "--proxy-backend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Proxy /v1/stream/* via this demo server",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    server_url_for_browser = "" if args.proxy_backend else args.server_url
    html = _render_html(server_url_for_browser)

    class Handler(http.server.BaseHTTPRequestHandler):
        _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def _is_backend_request(self) -> bool:
            return bool(args.proxy_backend) and self.path.startswith("/v1/stream/")

        def _proxy(self) -> None:
            url = f"{args.server_url.rstrip('/')}{self.path}"
            content_length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_length) if content_length else None
            headers = {"Content-Type": self.headers["Content-Type"]} if self.headers.get("Content-Type") else {}
            request = urllib.request.Request(url, data=body, headers=headers, method=self.command)
            try:
                with self._opener.open(request, timeout=30) as response:
                    response_body = response.read()
                    response_status = getattr(response, "status", 200)
                    response_type = response.headers.get("Content-Type", "application/octet-stream")
            except urllib.error.HTTPError as exc:
                response_body = exc.read()
                response_status = exc.code
                response_type = exc.headers.get("Content-Type", "application/json")
            except Exception as exc:
                response_body = json.dumps({"detail": f"Demo proxy error: {exc}"}).encode()
                response_status = 502
                response_type = "application/json"
            self.send_response(response_status)
            self.send_header("Content-Type", response_type)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def do_GET(self) -> None:
            if self._is_backend_request():
                self._proxy()
                return
            if self.path == "/default-image":
                body = Path(DEFAULT_IMAGE_PATH).read_bytes()
                content_type = "image/jpeg"
            else:
                body = html.encode()
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self._proxy() if self._is_backend_request() else self.send_error(404)

        def do_DELETE(self) -> None:
            self._proxy() if self._is_backend_request() else self.send_error(404)

        def log_message(self, format: str, *_args: object) -> None:
            return None

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Serving LingBot-World-Fast LiveKit demo at {url}")
    print(f"LiveKit API server: {args.server_url}")
    print(f"LiveKit JS client: {LIVEKIT_CLIENT_URL}")
    if args.proxy_backend:
        print("Proxy: enabled for the TeleFuser API")
        print("VS Code forwarding required: TCP 8092 (page), 7880 (LiveKit), and 3478 (TURN)")
    print("Press Ctrl+C to stop.\n")
    if not args.no_open:
        threading.Timer(0.5, functools.partial(webbrowser.open, url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
