"""WebRTC arrow overlay client demo.

Connects to a bidirectional ``ArrowOverlayService`` and sends keyboard
arrow-key events over a DataChannel.  The server overlays a D-pad HUD
on the video and streams it back via a WebRTC media track.

Usage:
    # 1. Start the server:
    telefuser stream-serve examples/stream_server/stream_arrow_overlay.py -p 8088 --skip-validation

    # 2. Start this client:
    python examples/stream_server/webrtc_arrow_overlay_demo.py --server-url http://localhost:8088

    # 3. Click Connect, then press arrow keys.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
import webbrowser

DEFAULT_SERVER_URL = "http://localhost:8088"
DEFAULT_PORT = 8092

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TeleFuser Arrow Overlay Demo</title>
<style>
  body {{ font-family: sans-serif; max-width: 720px; margin: 40px auto; padding: 0 20px;
         background: #1a1a2e; color: #e0e0e0; }}
  h2 {{ color: #fff; }}
  video {{ width: 100%; max-height: 480px; background: #000; border-radius: 8px;
           border: 2px solid #333; }}
  .controls {{ display: flex; gap: 10px; margin: 16px 0; align-items: center; }}
  button {{ padding: 8px 20px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; }}
  #connect {{ background: #2563eb; color: #fff; }}
  #connect:disabled {{ background: #555; cursor: default; }}
  #stop {{ background: #dc2626; color: #fff; display: none; }}
  #status {{ color: #aaa; font-size: 13px; margin: 8px 0; }}

  /* D-pad indicator */
  .dpad {{ display: grid; grid-template-columns: 50px 50px 50px; grid-template-rows: 50px 50px 50px;
           gap: 4px; justify-content: center; margin: 16px 0; user-select: none; }}
  .dpad-btn {{ display: flex; align-items: center; justify-content: center;
               background: #333; border-radius: 6px; font-size: 22px; color: #666;
               transition: background 0.08s, color 0.08s; }}
  .dpad-btn.active {{ background: #16a34a; color: #fff; }}
  .dpad-center {{ background: #222; border-radius: 6px; }}

  #messages {{ border: 1px solid #333; border-radius: 4px; padding: 8px; max-height: 140px;
               overflow-y: auto; font-family: monospace; font-size: 11px; background: #111; }}
  .msg {{ margin: 2px 0; }}
  .msg-out {{ color: #4ade80; }}
  .msg-in {{ color: #60a5fa; }}
  .hint {{ color: #888; font-size: 13px; margin: 8px 0; }}
</style>
</head>
<body>
<h2>Arrow Overlay Demo</h2>
<video id="video" autoplay playsinline muted></video>

<div class="controls">
  <button id="connect">Connect</button>
  <button id="stop">Stop</button>
</div>
<div id="status">Ready. Click Connect then press arrow keys.</div>
<p class="hint">Use keyboard arrow keys. The server overlays a D-pad HUD on the video.</p>

<div class="dpad">
  <div></div>
  <div class="dpad-btn" id="k-up">&uarr;</div>
  <div></div>
  <div class="dpad-btn" id="k-left">&larr;</div>
  <div class="dpad-center"></div>
  <div class="dpad-btn" id="k-right">&rarr;</div>
  <div></div>
  <div class="dpad-btn" id="k-down">&darr;</div>
  <div></div>
</div>

<h4 style="font-size:13px; color:#888; margin:12px 0 4px;">DataChannel Log</h4>
<div id="messages"></div>

<script>
const SERVER = "{server_url}";
let pc = null, dc = null, sessionId = null;

const keyMap = {{
  ArrowUp: "k-up", ArrowDown: "k-down",
  ArrowLeft: "k-left", ArrowRight: "k-right"
}};

function log(dir, text) {{
  const el = document.getElementById("messages");
  const cls = dir === "out" ? "msg-out" : "msg-in";
  const pfx = dir === "out" ? ">>" : "<<";
  const t = text.length > 160 ? text.slice(0, 160) + "..." : text;
  el.innerHTML += '<div class="msg ' + cls + '">' + pfx + " " + t + "</div>";
  el.scrollTop = el.scrollHeight;
}}

// --- keyboard ---
const pressed = new Set();
document.addEventListener("keydown", (e) => {{
  if (!(e.key in keyMap)) return;
  e.preventDefault();
  if (pressed.has(e.key)) return;
  pressed.add(e.key);
  document.getElementById(keyMap[e.key]).classList.add("active");
  if (dc && dc.readyState === "open") {{
    const msg = JSON.stringify({{ type: "control", key: e.key, action: "press" }});
    dc.send(msg);
    log("out", msg);
  }}
}});
document.addEventListener("keyup", (e) => {{
  if (!(e.key in keyMap)) return;
  pressed.delete(e.key);
  document.getElementById(keyMap[e.key]).classList.remove("active");
  if (dc && dc.readyState === "open") {{
    const msg = JSON.stringify({{ type: "control", key: e.key, action: "release" }});
    dc.send(msg);
    log("out", msg);
  }}
}});

// --- connect ---
document.getElementById("connect").onclick = async () => {{
  document.getElementById("status").textContent = "Connecting...";
  document.getElementById("connect").disabled = true;
  try {{
    pc = new RTCPeerConnection();
    dc = pc.createDataChannel("telefuser");
    dc.onopen = () => {{
      document.getElementById("status").textContent = "Connected. Press arrow keys!";
    }};
    dc.onmessage = (evt) => {{ log("in", evt.data); }};
    dc.onclose = () => {{
      document.getElementById("status").textContent = "DataChannel closed.";
    }};

    pc.addTransceiver("video", {{ direction: "recvonly" }});

    pc.ontrack = (evt) => {{
      if (evt.track.kind === "video") {{
        document.getElementById("video").srcObject = evt.streams[0];
      }}
    }};
    pc.onconnectionstatechange = () => {{
      if (!pc) return;
      const s = pc.connectionState;
      if (s === "connected") document.getElementById("stop").style.display = "inline-block";
      if (s === "failed" || s === "closed") {{ cleanup(); }}
    }};

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch(SERVER + "/v1/stream/webrtc/offer", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        task: "arrow_overlay",
        config: {{ fps: 24 }},
      }}),
    }});
    if (!resp.ok) {{ throw new Error((await resp.json()).detail || resp.statusText); }}

    const answer = await resp.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(new RTCSessionDescription({{ sdp: answer.sdp, type: answer.type }}));
  }} catch (e) {{
    document.getElementById("status").textContent = "Error: " + e.message;
    cleanup();
  }}
}};

document.getElementById("stop").onclick = async () => {{
  document.getElementById("status").textContent = "Stopping...";
  if (dc && dc.readyState === "open") {{
    dc.send(JSON.stringify({{ type: "stop" }}));
  }}
  // Send DELETE first so server cleans up the pipeline session
  if (sessionId) {{
    await fetch(SERVER + "/v1/stream/webrtc/" + sessionId, {{ method: "DELETE" }}).catch(() => {{}});
  }}
  cleanup();
}};

let _cleaning = false;
function cleanup() {{
  if (_cleaning) return;
  _cleaning = true;
  if (pc) {{
    try {{ pc.close(); }} catch(e) {{}}
    pc = null; dc = null;
  }}
  if (sessionId) {{
    fetch(SERVER + "/v1/stream/webrtc/" + sessionId, {{ method: "DELETE" }}).catch(() => {{}});
    sessionId = null;
  }}
  document.getElementById("video").srcObject = null;
  document.getElementById("connect").disabled = false;
  document.getElementById("stop").style.display = "none";
  pressed.clear();
  document.querySelectorAll(".dpad-btn").forEach(b => b.classList.remove("active"));
  _cleaning = false;
}}
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleFuser arrow overlay WebRTC client demo")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Stream server base URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local HTTP server port")
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()

    html = HTML_TEMPLATE.format(server_url=args.server_url)

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))

        def log_message(self, format: str, *_args: object) -> None:
            pass

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Serving arrow overlay demo at {url}")
    print(f"Stream server: {args.server_url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_open:
        threading.Timer(0.5, functools.partial(webbrowser.open, url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
