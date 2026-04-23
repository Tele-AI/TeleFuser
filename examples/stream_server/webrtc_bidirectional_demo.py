"""WebRTC bidirectional client demo.

Demonstrates the full-duplex WebRTC protocol:

* Client creates a DataChannel ("telefuser") for JSON control messages.
* Client optionally captures camera/microphone and sends via media tracks.
* Server sends generated video/audio via media tracks and metadata via DataChannel.

Usage:
    # 1. Start the stream server with a bidirectional pipeline:
    telefuser stream-serve examples/stream_server/my_bidirectional_pipeline.py -p 8088 --skip-validation

    # 2. Start this client (opens browser):
    python examples/stream_server/webrtc_bidirectional_demo.py --server-url http://localhost:8088

    # 3. Enter a prompt, optionally enable camera, click Connect.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import threading
import webbrowser

DEFAULT_SERVER_URL = "http://localhost:8088"
DEFAULT_PORT = 8091

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>TeleFuser WebRTC Bidirectional Demo</title>
<style>
  body {{ font-family: sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; }}
  .video-row {{ display: flex; gap: 16px; margin: 16px 0; }}
  .video-box {{ flex: 1; }}
  .video-box h3 {{ margin: 0 0 8px; font-size: 14px; color: #666; }}
  video {{ width: 100%; max-height: 360px; background: #000; border-radius: 8px; }}
  .controls {{ display: flex; gap: 10px; margin: 16px 0; align-items: center; flex-wrap: wrap; }}
  input[type=text] {{ flex: 1; min-width: 200px; padding: 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }}
  button {{ padding: 8px 20px; font-size: 14px; border: none; border-radius: 4px; cursor: pointer; }}
  #connect {{ background: #2563eb; color: #fff; }}
  #connect:disabled {{ background: #94a3b8; cursor: default; }}
  #stop {{ background: #dc2626; color: #fff; display: none; }}
  #send {{ background: #16a34a; color: #fff; display: none; }}
  #unmute {{ background: #7c3aed; color: #fff; display: none; }}
  label {{ font-size: 14px; cursor: pointer; }}
  #status {{ color: #666; font-size: 13px; margin: 8px 0; }}
  #messages {{ border: 1px solid #e5e7eb; border-radius: 4px; padding: 8px; max-height: 200px;
               overflow-y: auto; font-family: monospace; font-size: 12px; background: #f9fafb; }}
  .msg {{ margin: 2px 0; }}
  .msg-in {{ color: #2563eb; }}
  .msg-out {{ color: #16a34a; }}
</style>
</head>
<body>
<h2>TeleFuser WebRTC Bidirectional Demo</h2>

<div class="video-row">
  <div class="video-box">
    <h3>Server Output</h3>
    <video id="output-video" autoplay playsinline muted></video>
  </div>
  <div class="video-box">
    <h3>Camera Input (optional)</h3>
    <video id="input-video" autoplay playsinline muted></video>
  </div>
</div>

<div class="controls">
  <input id="prompt" type="text" placeholder="Enter a prompt..." value="a dog running">
  <label><input type="checkbox" id="use-camera"> Camera</label>
  <label><input type="checkbox" id="use-mic"> Mic</label>
  <button id="connect">Connect</button>
  <button id="send">Send Prompt</button>
  <button id="stop">Stop</button>
  <button id="unmute">Unmute Output</button>
</div>
<div id="status">Ready.</div>
<h3 style="font-size: 14px; color: #666; margin: 16px 0 4px;">DataChannel Messages</h3>
<div id="messages"></div>

<script>
const SERVER_URL = "{server_url}";
let pc = null;
let dc = null;
let sessionId = null;
let localStream = null;

function log(dir, text) {{
  const el = document.getElementById("messages");
  const cls = dir === "in" ? "msg-in" : "msg-out";
  const prefix = dir === "in" ? "<<" : ">>";
  const truncated = text.length > 200 ? text.slice(0, 200) + "..." : text;
  el.innerHTML += '<div class="msg ' + cls + '">' + prefix + " " + truncated + "</div>";
  el.scrollTop = el.scrollHeight;
}}

document.getElementById("connect").onclick = async () => {{
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;

  document.getElementById("status").textContent = "Connecting...";
  document.getElementById("connect").disabled = true;

  try {{
    pc = new RTCPeerConnection();

    // 1. Create DataChannel (client-created, server reuses)
    dc = pc.createDataChannel("telefuser");
    dc.onopen = () => {{
      document.getElementById("status").textContent = "DataChannel open. Sending prompt...";
      const msg = JSON.stringify({{ type: "control", prompt: prompt }});
      dc.send(msg);
      log("out", msg);
      document.getElementById("send").style.display = "inline-block";
    }};
    dc.onmessage = (evt) => {{
      log("in", evt.data);
    }};
    dc.onclose = () => {{
      document.getElementById("status").textContent = "DataChannel closed.";
    }};

    // 2. Optionally add camera/mic tracks
    const useCamera = document.getElementById("use-camera").checked;
    const useMic = document.getElementById("use-mic").checked;
    if (useCamera || useMic) {{
      const constraints = {{ video: useCamera, audio: useMic }};
      localStream = await navigator.mediaDevices.getUserMedia(constraints);
      localStream.getTracks().forEach(t => pc.addTrack(t, localStream));
      if (useCamera) {{
        document.getElementById("input-video").srcObject = localStream;
      }}
    }}

    // 3. Add recvonly transceivers for server output
    pc.addTransceiver("video", {{ direction: "recvonly" }});
    pc.addTransceiver("audio", {{ direction: "recvonly" }});

    // 4. Handle incoming server tracks
    pc.ontrack = (evt) => {{
      if (evt.track.kind === "video") {{
        document.getElementById("output-video").srcObject = evt.streams[0];
        document.getElementById("status").textContent = "Streaming...";
        document.getElementById("unmute").style.display = "inline-block";
      }}
    }};

    pc.onconnectionstatechange = () => {{
      if (!pc) return;
      const state = pc.connectionState;
      if (state === "failed" || state === "closed") {{
        document.getElementById("status").textContent = "Connection " + state;
        cleanup();
      }} else if (state === "connected") {{
        document.getElementById("stop").style.display = "inline-block";
      }}
    }};

    // 5. SDP offer/answer exchange
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const resp = await fetch(SERVER_URL + "/v1/stream/webrtc/offer", {{
      method: "POST",
      headers: {{ "Content-Type": "application/json" }},
      body: JSON.stringify({{
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
        task: "bidirectional",
        prompt: prompt,
        config: {{ fps: 24 }},
      }}),
    }});

    if (!resp.ok) {{
      const err = await resp.json();
      throw new Error(err.detail || resp.statusText);
    }}

    const answer = await resp.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(new RTCSessionDescription({{
      sdp: answer.sdp,
      type: answer.type,
    }}));

  }} catch (e) {{
    document.getElementById("status").textContent = "Error: " + e.message;
    cleanup();
  }}
}};

document.getElementById("send").onclick = () => {{
  if (!dc || dc.readyState !== "open") return;
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;
  const msg = JSON.stringify({{ type: "control", prompt: prompt }});
  dc.send(msg);
  log("out", msg);
}};

document.getElementById("stop").onclick = async () => {{
  if (dc && dc.readyState === "open") {{
    const msg = JSON.stringify({{ type: "stop" }});
    dc.send(msg);
    log("out", msg);
  }}
  document.getElementById("status").textContent = "Stopped.";
  if (sessionId) {{
    await fetch(SERVER_URL + "/v1/stream/webrtc/" + sessionId, {{ method: "DELETE" }}).catch(() => {{}});
  }}
  cleanup();
}};

document.getElementById("unmute").onclick = () => {{
  const video = document.getElementById("output-video");
  video.muted = !video.muted;
  document.getElementById("unmute").textContent = video.muted ? "Unmute Output" : "Mute Output";
}};

let _cleaning = false;
function cleanup() {{
  if (_cleaning) return;
  _cleaning = true;
  if (localStream) {{
    localStream.getTracks().forEach(t => t.stop());
    localStream = null;
    document.getElementById("input-video").srcObject = null;
  }}
  if (pc) {{
    try {{ pc.close(); }} catch(e) {{}}
    pc = null;
    dc = null;
  }}
  if (sessionId) {{
    fetch(SERVER_URL + "/v1/stream/webrtc/" + sessionId, {{ method: "DELETE" }}).catch(() => {{}});
    sessionId = null;
  }}
  document.getElementById("output-video").srcObject = null;
  document.getElementById("connect").disabled = false;
  document.getElementById("stop").style.display = "none";
  document.getElementById("send").style.display = "none";
  document.getElementById("unmute").style.display = "none";
  document.getElementById("unmute").textContent = "Unmute Output";
  _cleaning = false;
}}
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleFuser WebRTC bidirectional client demo")
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
    print(f"Serving WebRTC bidirectional demo at {url}")
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
