"""LingBot-World-Fast WebRTC control demo.

Demonstrates the LingBot-World-Fast bidirectional WebRTC protocol:

* Client creates a DataChannel ("telefuser") for JSON control messages.
* Client sends prompt and direction controls.
* Server sends generated video via media tracks and metadata via DataChannel.

Usage:
    # 1. Start the LingBot stream server:
    telefuser stream-serve examples/stream_server/stream_lingbot_world_fast.py -p 8088 --skip-validation

    # 2. Start this client (opens browser):
    python examples/stream_server/webrtc_bidirectional_demo.py --server-url http://localhost:8088 --image-path /path/to/input.png

    # 3. Enter a prompt, click Connect, then use arrow keys or the D-pad.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import threading
import urllib.error
import urllib.request
import webbrowser

DEFAULT_SERVER_URL = "http://localhost:8088"
DEFAULT_PORT = 8091
DEFAULT_SAMPLE_SHIFT = 10.0
MAX_GENERATION_SECONDS = 20.0
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
    "mountains under a bright blue sky with drifting white clouds. Gentle ripples reflect the tree and sky."
)

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
const RTC_CONFIG = {rtc_config};
const IMAGE_PATH = {image_path};
const REQUEST_OPTIONS = {request_options};
const ICE_GATHER_TIMEOUT_MS = {ice_gather_timeout_ms};
let pc = null;
let dc = null;
let sessionId = null;
let localStream = null;

async function fetchJsonWithTimeout(url, options, timeoutMs) {{
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {{
    const resp = await fetch(url, {{ ...options, signal: controller.signal }});
    return resp;
  }} finally {{
    clearTimeout(timeout);
  }}
}}

function log(dir, text) {{
  const el = document.getElementById("messages");
  const cls = dir === "in" ? "msg-in" : "msg-out";
  const prefix = dir === "in" ? "<<" : ">>";
  const truncated = text.length > 200 ? text.slice(0, 200) + "..." : text;
  el.innerHTML += '<div class="msg ' + cls + '">' + prefix + " " + truncated + "</div>";
  el.scrollTop = el.scrollHeight;
}}

function setStatus(text) {{
  document.getElementById("status").textContent = text;
}}

function describeCandidate(candidateStr) {{
  if (!candidateStr) return "";
  const parts = candidateStr.split(" ");
  const typIdx = parts.indexOf("typ");
  const typ = typIdx !== -1 ? parts[typIdx + 1] : "";
  const proto = parts.length > 2 ? parts[2] : "";
  return (typ ? (" typ=" + typ) : "") + (proto ? (" proto=" + proto) : "");
}}

async function waitForIceGathering(pc, timeoutMs) {{
  if (pc.iceGatheringState === "complete") return true;
  return await Promise.race([
    new Promise(resolve => {{
      const onStateChange = () => {{
        if (pc.iceGatheringState === "complete") {{
          pc.removeEventListener("icegatheringstatechange", onStateChange);
          resolve(true);
        }}
      }};
      pc.addEventListener("icegatheringstatechange", onStateChange);
    }}),
    new Promise(resolve => setTimeout(() => resolve(false), timeoutMs)),
  ]);
}}

document.getElementById("connect").onclick = async () => {{
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return;

  setStatus("Connecting...");
  document.getElementById("connect").disabled = true;

  try {{
    pc = new RTCPeerConnection(RTC_CONFIG);
    let iceCandidateCount = 0;
    let relayCandidateCount = 0;

    pc.oniceconnectionstatechange = () => {{
      if (!pc) return;
      log("in", "iceConnectionState=" + pc.iceConnectionState);
      if (pc.iceConnectionState === "failed") {{
        setStatus("ICE failed (check TURN reachability/ports).");
      }}
    }};
    pc.onicegatheringstatechange = () => {{
      if (!pc) return;
      log("in", "iceGatheringState=" + pc.iceGatheringState);
    }};
    pc.onicecandidateerror = (evt) => {{
      const msg = "iceCandidateError " + (evt.errorText || "") + " " + (evt.url || "");
      log("in", msg);
    }};
    pc.onicecandidate = (evt) => {{
      if (evt.candidate && evt.candidate.candidate) {{
        iceCandidateCount += 1;
        if (evt.candidate.candidate.includes(" typ relay ")) {{
          relayCandidateCount += 1;
        }}
        log("in", "iceCandidate" + describeCandidate(evt.candidate.candidate));
      }}
    }};

    // 1. Create DataChannel (client-created, server reuses)
    dc = pc.createDataChannel("telefuser");
    dc.onopen = () => {{
      setStatus("DataChannel open. Sending prompt...");
      const msg = JSON.stringify({{ type: "control", prompt: prompt }});
      dc.send(msg);
      log("out", msg);
      document.getElementById("send").style.display = "inline-block";
    }};
    dc.onmessage = (evt) => {{
      log("in", evt.data);
      try {{
        const msg = JSON.parse(evt.data);
        const data = msg.data || msg;
        if (data.error) {{
          setStatus("Server error: " + data.error);
        }} else if (data.stage) {{
          const suffix = data.index !== undefined ? " #" + data.index : "";
          setStatus("Server: " + data.stage + suffix);
        }} else if (msg.type === "done") {{
          setStatus("Done.");
        }}
      }} catch (e) {{}}
    }};
    dc.onclose = () => {{
      if (!_cleaning) {{
        setStatus("DataChannel closed.");
      }}
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
        setStatus("Streaming...");
        document.getElementById("unmute").style.display = "inline-block";
      }}
    }};

    pc.onconnectionstatechange = () => {{
      if (!pc) return;
      const state = pc.connectionState;
      if (state === "failed" || state === "closed") {{
        setStatus("Connection " + state);
        cleanup();
      }} else if (state === "connected") {{
        document.getElementById("stop").style.display = "inline-block";
      }}
    }};

    // 5. SDP offer/answer exchange
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    setStatus("Gathering ICE...");
    const iceComplete = await waitForIceGathering(pc, ICE_GATHER_TIMEOUT_MS);
    log("in", "iceGatheringComplete=" + iceComplete + " candidates=" + iceCandidateCount + " relay=" + relayCandidateCount);
    if (RTC_CONFIG.iceTransportPolicy === "relay" && relayCandidateCount === 0) {{
      throw new Error("No TURN relay candidate gathered. Check TURN URL, credentials, and opened TURN ports.");
    }}
    setStatus("Sending offer...");

    const requestOptions = Object.assign({{}}, REQUEST_OPTIONS);
    const requestFps = requestOptions.fps || 24;
    const requestBody = {{
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      task: "bidirectional",
      prompt: prompt,
      fps: requestFps,
      config: Object.assign({{}}, requestOptions),
      ...requestOptions,
    }};
    if (IMAGE_PATH) {{
      requestBody.image_path = IMAGE_PATH;
    }}

    const resp = await fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/webrtc/offer",
      {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(requestBody),
      }},
      30000,
    );

    if (!resp.ok) {{
      const err = await resp.json().catch(() => ({{}}));
      throw new Error(err.detail || resp.statusText);
    }}

    const answer = await resp.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(new RTCSessionDescription({{
      sdp: answer.sdp,
      type: answer.type,
    }}));

  }} catch (e) {{
    setStatus("Error: " + e.message);
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
  setStatus("Stopped.");
  if (sessionId) {{
    await fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/webrtc/" + sessionId,
      {{ method: "DELETE" }},
      5000,
    ).catch(() => {{}});
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
    fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/webrtc/" + sessionId,
      {{ method: "DELETE" }},
      5000,
    ).catch(() => {{}});
    sessionId = null;
  }}
  document.getElementById("output-video").srcObject = null;
  document.getElementById("connect").disabled = false;
  document.getElementById("stop").style.display = "none";
  document.getElementById("send").style.display = "none";
  document.getElementById("unmute").style.display = "none";
  document.getElementById("unmute").textContent = "Unmute Output";
  setTimeout(() => {{ _cleaning = false; }}, 0);
}}
</script>
</body>
</html>"""


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LingBot-World-Fast WebRTC Demo</title>
<style>
  :root {
    color-scheme: light;
    --bg: #f5f7fb;
    --panel: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --line: #d8dee9;
    --blue: #1d4ed8;
    --blue-soft: #dbeafe;
    --green: #15803d;
    --red: #b91c1c;
    --ink: #0f172a;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
  }
  main {
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 20px 36px;
  }
  h1 {
    margin: 0 0 16px;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 0;
  }
  .workspace {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 340px;
    gap: 18px;
    align-items: start;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
  }
  .video-panel {
    overflow: hidden;
  }
  .video-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px;
    border-bottom: 1px solid var(--line);
  }
  .video-head h2,
  .control-panel h2,
  .log-panel h2 {
    margin: 0;
    font-size: 14px;
    font-weight: 650;
    color: var(--ink);
  }
  #status {
    color: var(--muted);
    font-size: 12px;
    text-align: right;
  }
  video {
    display: block;
    width: 100%;
    aspect-ratio: 16 / 9;
    max-height: 640px;
    background: #000;
  }
  .side {
    display: grid;
    gap: 14px;
  }
  .control-panel,
  .log-panel {
    padding: 14px;
  }
  .field {
    display: grid;
    gap: 6px;
    margin-top: 12px;
  }
  label {
    color: #374151;
    font-size: 12px;
    font-weight: 600;
  }
  input,
  select {
    width: 100%;
    min-height: 36px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 13px;
    background: #fff;
    color: var(--text);
  }
  textarea {
    width: 100%;
    min-height: 72px;
    resize: vertical;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px 9px;
    font-size: 13px;
    font-family: inherit;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
  }
  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
  }
  button {
    min-height: 36px;
    padding: 7px 14px;
    border: 0;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 650;
  }
  button:disabled {
    cursor: default;
    opacity: 0.55;
  }
  #connect {
    background: var(--blue);
    color: #fff;
  }
  #stop {
    display: none;
    background: var(--red);
    color: #fff;
  }
  #reset-control {
    background: #e2e8f0;
    color: #0f172a;
  }
  .dpad {
    display: grid;
    grid-template-columns: 44px 44px 44px;
    grid-template-rows: 44px 44px 44px;
    gap: 6px;
    justify-content: center;
    margin: 8px 0 2px;
    user-select: none;
  }
  .dpad button {
    width: 44px;
    height: 44px;
    padding: 0;
    border: 1px solid #cbd5e1;
    background: #f8fafc;
    color: #475569;
    font-size: 23px;
    line-height: 1;
  }
  .dpad button.active {
    background: var(--blue-soft);
    border-color: #60a5fa;
    color: var(--blue);
  }
  .dpad .empty {
    width: 44px;
    height: 44px;
  }
  .control-pads {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 14px;
  }
  .control-pad {
    padding: 9px 4px 6px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #f8fafc;
  }
  .control-pad h3 {
    margin: 0;
    color: #475569;
    font-size: 11px;
    text-align: center;
  }
  .field-help {
    margin-top: 4px;
    color: var(--muted);
    font-size: 11px;
  }
  #messages {
    height: 210px;
    overflow-y: auto;
    margin-top: 10px;
    padding: 8px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    background: #f8fafc;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 11px;
  }
  .msg {
    margin: 0 0 4px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }
  .msg-in { color: #1d4ed8; }
  .msg-out { color: #15803d; }
  @media (max-width: 900px) {
    main { padding: 18px 12px 28px; }
    .workspace { grid-template-columns: 1fr; }
    #status { text-align: left; }
    .video-head { align-items: flex-start; flex-direction: column; }
  }
</style>
</head>
<body>
<main>
  <h1>LingBot-World-Fast WebRTC Demo</h1>
  <div class="workspace">
    <section class="panel video-panel">
      <div class="video-head">
        <h2>Server Output</h2>
        <div id="status">Ready.</div>
      </div>
      <video id="output-video" autoplay playsinline muted></video>
    </section>

    <aside class="side">
      <section class="panel control-panel">
        <h2>Inputs</h2>
        <div class="field">
          <label for="prompt">Prompt</label>
          <textarea id="prompt"></textarea>
        </div>
        <div class="field">
          <label for="image-path">Server image path</label>
          <input id="image-path" type="text" placeholder="/path/on/remote/server.png">
        </div>
        <div class="grid">
          <div class="field">
            <label for="duration-seconds">Duration (seconds)</label>
            <input id="duration-seconds" type="number" min="0.5" max="20" step="0.5">
          </div>
          <div class="field">
            <label for="frame-num">Generated frames</label>
            <input id="frame-num" type="number" readonly>
          </div>
          <div class="field">
            <label for="fps">FPS</label>
            <input id="fps" type="number" min="1" step="1" readonly>
          </div>
          <div class="field">
            <label for="chunk-size">Chunk size</label>
            <input id="chunk-size" type="number" min="1" step="1" readonly>
          </div>
          <div class="field">
            <label for="seed">Seed</label>
            <input id="seed" type="number" step="1">
          </div>
        </div>
        <div id="duration-help" class="field-help"></div>
        <div class="grid">
          <div class="field">
            <label for="sample-shift">Sample shift</label>
            <input id="sample-shift" type="number" step="0.1">
          </div>
          <div class="field">
            <label for="control-mode">Control mode</label>
            <select id="control-mode">
              <option value="cam">cam</option>
              <option value="act">act</option>
            </select>
          </div>
        </div>
        <div class="field">
          <label for="action-path">Control file path</label>
          <input id="action-path" type="text" placeholder="Optional poses/action directory">
        </div>
        <div class="grid">
          <div class="field">
            <label for="control-move-step">Move step</label>
            <input id="control-move-step" type="number" min="0" step="0.01">
          </div>
          <div class="field">
            <label for="control-yaw-step">Yaw step</label>
            <input id="control-yaw-step" type="number" min="0" step="0.5">
          </div>
          <div class="field">
            <label for="control-lateral-step">Lateral step</label>
            <input id="control-lateral-step" type="number" min="0" step="0.01">
          </div>
        </div>
        <div class="field">
          <label><input id="show-control-hud" type="checkbox"> Control HUD</label>
        </div>

        <div class="actions">
          <button id="connect">Connect</button>
          <button id="stop">Stop</button>
          <button id="reset-control">Reset Control</button>
        </div>

        <div class="control-pads">
          <div class="control-pad">
            <h3>Move · WASD / Arrows</h3>
            <div class="dpad" aria-label="Translation controls">
              <div class="empty"></div>
              <button id="ctrl-forward" data-control="w" title="Move forward">↑</button>
              <div class="empty"></div>
              <button id="ctrl-strafe-left" data-control="a" title="Strafe left">←</button>
              <div class="empty"></div>
              <button id="ctrl-strafe-right" data-control="d" title="Strafe right">→</button>
              <div class="empty"></div>
              <button id="ctrl-backward" data-control="s" title="Move backward">↓</button>
              <div class="empty"></div>
            </div>
          </div>
          <div class="control-pad">
            <h3>Rotate · IJKL</h3>
            <div class="dpad" aria-label="Rotation controls">
              <div class="empty"></div>
              <button id="ctrl-pitch-up" data-control="i" title="Pitch up">↑</button>
              <div class="empty"></div>
              <button id="ctrl-yaw-left" data-control="j" title="Yaw left">↶</button>
              <div class="empty"></div>
              <button id="ctrl-yaw-right" data-control="l" title="Yaw right">↷</button>
              <div class="empty"></div>
              <button id="ctrl-pitch-down" data-control="k" title="Pitch down">↓</button>
              <div class="empty"></div>
            </div>
          </div>
        </div>
      </section>

      <section class="panel log-panel">
        <h2>DataChannel Messages</h2>
        <div id="messages"></div>
      </section>
    </aside>
  </div>
</main>

<script>
const SERVER_URL = __SERVER_URL__;
const RTC_CONFIG = __RTC_CONFIG__;
const DEFAULT_IMAGE_PATH = __IMAGE_PATH__;
const DEFAULT_PROMPT = __PROMPT__;
const DEFAULT_OPTIONS = __REQUEST_OPTIONS__;
const ICE_GATHER_TIMEOUT_MS = __ICE_GATHER_TIMEOUT_MS__;
const MAX_GENERATION_SECONDS = __MAX_GENERATION_SECONDS__;

let pc = null;
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

function $(id) {
  return document.getElementById(id);
}

function setStatus(text) {
  $("status").textContent = text;
}

function log(dir, text) {
  const el = $("messages");
  const row = document.createElement("div");
  row.className = "msg " + (dir === "in" ? "msg-in" : "msg-out");
  const prefix = dir === "in" ? "<<" : ">>";
  const value = String(text);
  row.textContent = prefix + " " + (value.length > 240 ? value.slice(0, 240) + "..." : value);
  el.appendChild(row);
  el.scrollTop = el.scrollHeight;
}

function numberValue(id, fallback) {
  const raw = $(id).value;
  if (raw === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function frameNumForDuration(durationSeconds, fps, chunkSize) {
  const duration = Math.min(MAX_GENERATION_SECONDS, Math.max(0.5, durationSeconds));
  const targetFrames = Math.floor(duration * fps) + 1;
  const targetLatentFrames = Math.floor((targetFrames - 1) / 4) + 1;
  const completeLatentFrames = Math.max(chunkSize, Math.floor(targetLatentFrames / chunkSize) * chunkSize);
  return 4 * (completeLatentFrames - 1) + 1;
}

function updateFrameNum() {
  const fps = Math.max(1, numberValue("fps", DEFAULT_OPTIONS.fps ?? 16));
  const chunkSize = Math.max(1, numberValue("chunk-size", DEFAULT_OPTIONS.chunk_size ?? 3));
  const duration = Math.min(
    MAX_GENERATION_SECONDS,
    Math.max(0.5, numberValue("duration-seconds", 5.0)),
  );
  $("duration-seconds").value = duration;
  const frameNum = frameNumForDuration(duration, fps, chunkSize);
  $("frame-num").value = frameNum;
  const actualDuration = (frameNum - 1) / fps;
  $("duration-help").textContent =
    "Actual duration: " + actualDuration.toFixed(2) + " s · maximum: " + MAX_GENERATION_SECONDS + " s";
}

function fillDefaults() {
  $("prompt").value = DEFAULT_PROMPT;
  $("image-path").value = DEFAULT_IMAGE_PATH || "";
  $("fps").value = DEFAULT_OPTIONS.fps ?? 16;
  $("chunk-size").value = DEFAULT_OPTIONS.chunk_size ?? 3;
  $("duration-seconds").max = MAX_GENERATION_SECONDS;
  $("duration-seconds").value = ((DEFAULT_OPTIONS.frame_num ?? 81) - 1) / (DEFAULT_OPTIONS.fps ?? 16);
  $("sample-shift").value = DEFAULT_OPTIONS.sample_shift ?? 10.0;
  $("seed").value = DEFAULT_OPTIONS.seed ?? 42;
  $("control-mode").value = DEFAULT_OPTIONS.control_mode || "cam";
  $("action-path").value = DEFAULT_OPTIONS.action_path || "";
  $("control-move-step").value = DEFAULT_OPTIONS.control_move_step ?? 0.05;
  $("control-yaw-step").value = DEFAULT_OPTIONS.control_yaw_step_degrees ?? 2.0;
  $("control-lateral-step").value = DEFAULT_OPTIONS.control_lateral_step ?? 0.05;
  $("show-control-hud").checked = DEFAULT_OPTIONS.show_control_hud ?? true;
  updateFrameNum();
}

async function fetchJsonWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function describeCandidate(candidateStr) {
  if (!candidateStr) return "";
  const parts = candidateStr.split(" ");
  const typIdx = parts.indexOf("typ");
  const typ = typIdx !== -1 ? parts[typIdx + 1] : "";
  const proto = parts.length > 2 ? parts[2] : "";
  return (typ ? (" typ=" + typ) : "") + (proto ? (" proto=" + proto) : "");
}

async function waitForIceGathering(peer, timeoutMs) {
  if (peer.iceGatheringState === "complete") return true;
  return await Promise.race([
    new Promise(resolve => {
      const onStateChange = () => {
        if (peer.iceGatheringState === "complete") {
          peer.removeEventListener("icegatheringstatechange", onStateChange);
          resolve(true);
        }
      };
      peer.addEventListener("icegatheringstatechange", onStateChange);
    }),
    new Promise(resolve => setTimeout(() => resolve(false), timeoutMs)),
  ]);
}

function requestOptionsFromForm() {
  updateFrameNum();
  const options = {
    ...DEFAULT_OPTIONS,
    fps: numberValue("fps", DEFAULT_OPTIONS.fps ?? 16),
    frame_num: numberValue("frame-num", DEFAULT_OPTIONS.frame_num ?? 81),
    chunk_size: numberValue("chunk-size", DEFAULT_OPTIONS.chunk_size ?? 3),
    sample_shift: numberValue("sample-shift", DEFAULT_OPTIONS.sample_shift ?? 10.0),
    seed: numberValue("seed", DEFAULT_OPTIONS.seed ?? 42),
    control_mode: $("control-mode").value,
    control_move_step: numberValue("control-move-step", DEFAULT_OPTIONS.control_move_step ?? 0.05),
    control_yaw_step_degrees: numberValue("control-yaw-step", DEFAULT_OPTIONS.control_yaw_step_degrees ?? 2.0),
    control_lateral_step: numberValue("control-lateral-step", DEFAULT_OPTIONS.control_lateral_step ?? 0.05),
    control_pitch_step_degrees: DEFAULT_OPTIONS.control_pitch_step_degrees ?? 2.0,
    control_pitch_limit_degrees: DEFAULT_OPTIONS.control_pitch_limit_degrees ?? 85.0,
    show_control_hud: $("show-control-hud").checked,
  };
  const actionPath = $("action-path").value.trim();
  if (actionPath) {
    options.action_path = actionPath;
  } else {
    delete options.action_path;
  }
  return options;
}

$("duration-seconds").addEventListener("input", updateFrameNum);
$("fps").addEventListener("input", updateFrameNum);
$("chunk-size").addEventListener("input", updateFrameNum);

function setControlActive(control, active) {
  const btn = document.querySelector('[data-control="' + control + '"]');
  if (btn) btn.classList.toggle("active", active);
}

function sendControl(control, eventName) {
  if (!control) return;
  if (eventName === "press") {
    if (pressedControls.has(control)) return;
    pressedControls.add(control);
    setControlActive(control, true);
  } else {
    if (!pressedControls.has(control)) return;
    pressedControls.delete(control);
    setControlActive(control, false);
  }
  if (dc && dc.readyState === "open") {
    const msg = JSON.stringify({ type: "control", control, event: eventName });
    dc.send(msg);
    log("out", msg);
  }
}

function releaseAllControls(sendMessages = true) {
  for (const control of Array.from(pressedControls)) {
    pressedControls.delete(control);
    setControlActive(control, false);
    if (sendMessages && dc && dc.readyState === "open") {
      const msg = JSON.stringify({ type: "control", control, event: "release" });
      dc.send(msg);
      log("out", msg);
    }
  }
}

document.querySelectorAll("[data-control]").forEach(btn => {
  const control = btn.dataset.control;
  btn.addEventListener("pointerdown", evt => {
    evt.preventDefault();
    btn.setPointerCapture(evt.pointerId);
    sendControl(control, "press");
  });
  btn.addEventListener("pointerup", evt => {
    evt.preventDefault();
    sendControl(control, "release");
  });
  btn.addEventListener("pointercancel", () => sendControl(control, "release"));
  btn.addEventListener("lostpointercapture", () => sendControl(control, "release"));
});

document.addEventListener("keydown", evt => {
  const control = keyToControl[evt.key] || keyToControl[evt.code];
  if (!control) return;
  evt.preventDefault();
  sendControl(control, "press");
});

document.addEventListener("keyup", evt => {
  const control = keyToControl[evt.key] || keyToControl[evt.code];
  if (!control) return;
  evt.preventDefault();
  sendControl(control, "release");
});

$("reset-control").onclick = () => {
  releaseAllControls(false);
  if (dc && dc.readyState === "open") {
    const msg = JSON.stringify({ type: "control", control: "up", event: "reset" });
    dc.send(msg);
    log("out", msg);
  }
};

$("connect").onclick = async () => {
  const prompt = $("prompt").value.trim();
  const imagePath = $("image-path").value.trim();
  if (!prompt) {
    setStatus("Prompt is required.");
    return;
  }
  if (!imagePath) {
    setStatus("Server image path is required.");
    return;
  }

  setStatus("Connecting...");
  $("connect").disabled = true;

  try {
    pc = new RTCPeerConnection(RTC_CONFIG);
    let iceCandidateCount = 0;
    let relayCandidateCount = 0;

    pc.oniceconnectionstatechange = () => {
      if (!pc) return;
      log("in", "iceConnectionState=" + pc.iceConnectionState);
      if (pc.iceConnectionState === "failed") {
        setStatus("ICE failed.");
      }
    };
    pc.onicegatheringstatechange = () => {
      if (pc) log("in", "iceGatheringState=" + pc.iceGatheringState);
    };
    pc.onicecandidateerror = evt => {
      log("in", "iceCandidateError " + (evt.errorText || "") + " " + (evt.url || ""));
    };
    pc.onicecandidate = evt => {
      if (evt.candidate && evt.candidate.candidate) {
        iceCandidateCount += 1;
        if (evt.candidate.candidate.includes(" typ relay ")) relayCandidateCount += 1;
        log("in", "iceCandidate" + describeCandidate(evt.candidate.candidate));
      }
    };

    dc = pc.createDataChannel("telefuser");
    dc.onopen = () => {
      setStatus("Connected. Waiting for LingBot output...");
      $("stop").style.display = "inline-block";
    };
    dc.onmessage = evt => {
      log("in", evt.data);
      try {
        const msg = JSON.parse(evt.data);
        const data = msg.data || msg;
        if (data.error) {
          setStatus("Server error: " + data.error);
        } else if (data.stage) {
          const suffix = data.index !== undefined ? " #" + data.index : "";
          const controls = data.controls ? " [" + data.controls.join(",") + "]" : "";
          setStatus("Server: " + data.stage + suffix + controls);
        } else if (msg.type === "done") {
          setStatus("Done.");
        }
      } catch (e) {}
    };
    dc.onclose = () => {
      if (!cleaning) setStatus("DataChannel closed.");
    };

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.ontrack = evt => {
      if (evt.track.kind === "video") {
        $("output-video").srcObject = evt.streams[0];
      }
    };
    pc.onconnectionstatechange = () => {
      if (!pc) return;
      const state = pc.connectionState;
      if (state === "failed" || state === "closed") {
        setStatus("Connection " + state);
        cleanup();
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    setStatus("Gathering ICE...");
    const iceComplete = await waitForIceGathering(pc, ICE_GATHER_TIMEOUT_MS);
    log("in", "iceGatheringComplete=" + iceComplete + " candidates=" + iceCandidateCount + " relay=" + relayCandidateCount);
    if (RTC_CONFIG.iceTransportPolicy === "relay" && relayCandidateCount === 0) {
      throw new Error("No TURN relay candidate gathered. Check TURN URL, credentials, and forwarded/opened TURN ports.");
    }

    setStatus("Sending offer...");
    const requestOptions = requestOptionsFromForm();
    const requestBody = {
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      task: "bidirectional",
      prompt,
      fps: requestOptions.fps,
      image_path: imagePath,
      config: { ...requestOptions },
      ...requestOptions,
    };
    const resp = await fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/webrtc/offer",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      },
      30000,
    );
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || resp.statusText);
    }

    const answer = await resp.json();
    sessionId = answer.session_id;
    await pc.setRemoteDescription(new RTCSessionDescription({
      sdp: answer.sdp,
      type: answer.type,
    }));
  } catch (e) {
    setStatus("Error: " + e.message);
    cleanup();
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
  if (sessionId) {
    await fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/webrtc/" + sessionId,
      { method: "DELETE" },
      5000,
    ).catch(() => {});
  }
  cleanup();
};

function cleanup() {
  if (cleaning) return;
  cleaning = true;
  releaseAllControls(false);
  if (pc) {
    try { pc.close(); } catch (e) {}
    pc = null;
    dc = null;
  }
  if (sessionId) {
    fetchJsonWithTimeout(
      SERVER_URL + "/v1/stream/webrtc/" + sessionId,
      { method: "DELETE" },
      5000,
    ).catch(() => {});
    sessionId = null;
  }
  $("output-video").srcObject = null;
  $("connect").disabled = false;
  $("stop").style.display = "none";
  setTimeout(() => { cleaning = false; }, 0);
}

fillDefaults();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="LingBot-World-Fast WebRTC control demo")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Stream server base URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local HTTP server port")
    parser.add_argument("--image-path", default="", help="Server-side image path for pipelines that require one")
    parser.add_argument("--fps", type=int, default=16, help="Output WebRTC FPS and pipeline FPS")
    parser.add_argument("--frame-num", type=int, default=81, help="Requested LingBot frame count")
    parser.add_argument("--chunk-size", type=int, default=3, help="LingBot latent chunk size")
    parser.add_argument("--sample-shift", type=float, default=DEFAULT_SAMPLE_SHIFT, help="LingBot sampler shift")
    parser.add_argument("--seed", type=int, default=42, help="LingBot random seed")
    parser.add_argument("--max-attention-size", type=int, default=None, help="Optional LingBot max attention size")
    parser.add_argument("--max-sequence-length", type=int, default=512, help="LingBot max text sequence length")
    parser.add_argument("--control-mode", default="cam", choices=("cam", "act"), help="LingBot control mode")
    parser.add_argument("--action-path", default="", help="Optional LingBot camera/action control directory")
    parser.add_argument("--control-move-step", type=float, default=0.05, help="LingBot video-frame move step")
    parser.add_argument(
        "--control-yaw-step-degrees",
        type=float,
        default=2.0,
        help="LingBot yaw step per video frame",
    )
    parser.add_argument(
        "--control-lateral-step",
        type=float,
        default=0.05,
        help="LingBot video-frame lateral strafe step",
    )
    parser.add_argument(
        "--control-pitch-step-degrees",
        type=float,
        default=2.0,
        help="LingBot pitch step per video frame",
    )
    parser.add_argument(
        "--control-pitch-limit-degrees",
        type=float,
        default=85.0,
        help="LingBot absolute pitch limit",
    )
    parser.add_argument(
        "--show-control-hud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overlay an active-control HUD on chunks that consumed direction controls",
    )
    parser.add_argument(
        "--ice-gather-timeout-ms",
        type=int,
        default=10000,
        help="How long the browser waits for ICE candidates before sending the SDP offer",
    )
    parser.add_argument(
        "--proxy-backend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Proxy /v1/stream/webrtc/* via this demo server to --server-url (recommended for VS Code Remote port forwarding)",
    )
    parser.add_argument(
        "--turn-url",
        default=os.environ.get("TELEFUSER_TURN_SERVER", ""),
        help="TURN server URL for browser WebRTC ICE, e.g. turn:localhost:3478?transport=tcp",
    )
    parser.add_argument(
        "--turn-username",
        default=os.environ.get("TELEFUSER_TURN_USERNAME", ""),
        help="TURN username; defaults to TELEFUSER_TURN_USERNAME",
    )
    parser.add_argument(
        "--turn-credential",
        default=os.environ.get("TELEFUSER_TURN_CREDENTIAL", ""),
        help="TURN credential; defaults to TELEFUSER_TURN_CREDENTIAL",
    )
    parser.add_argument(
        "--force-turn-relay",
        action="store_true",
        help="Force WebRTC to use relay candidates only. Useful when testing through SSH port forwarding.",
    )
    parser.add_argument("--no-open", action="store_true", help="Don't open browser automatically")
    args = parser.parse_args()
    if args.image_path and not os.path.exists(args.image_path):
        parser.error(f"--image-path does not exist on the server: {args.image_path}")

    rtc_config: dict[str, object] = {}
    if args.turn_url:
        turn_server: dict[str, str] = {"urls": args.turn_url}
        if args.turn_username:
            turn_server["username"] = args.turn_username
        if args.turn_credential:
            turn_server["credential"] = args.turn_credential
        rtc_config["iceServers"] = [turn_server]
    if args.force_turn_relay:
        rtc_config["iceTransportPolicy"] = "relay"

    request_options: dict[str, object] = {
        "fps": args.fps,
        "frame_num": args.frame_num,
        "chunk_size": args.chunk_size,
        "sample_shift": args.sample_shift,
        "seed": args.seed,
        "max_sequence_length": args.max_sequence_length,
        "control_mode": args.control_mode,
        "control_move_step": args.control_move_step,
        "control_yaw_step_degrees": args.control_yaw_step_degrees,
        "control_lateral_step": args.control_lateral_step,
        "control_pitch_step_degrees": args.control_pitch_step_degrees,
        "control_pitch_limit_degrees": args.control_pitch_limit_degrees,
        "show_control_hud": args.show_control_hud,
    }
    if args.max_attention_size is not None:
        request_options["max_attention_size"] = args.max_attention_size
    if args.action_path:
        request_options["action_path"] = args.action_path

    # When proxying, the browser should call the demo origin (no separate port forward needed for --server-url).
    server_url_for_browser = "" if args.proxy_backend else args.server_url

    html = (
        HTML_TEMPLATE.replace("__SERVER_URL__", json.dumps(server_url_for_browser))
        .replace("__RTC_CONFIG__", json.dumps(rtc_config))
        .replace("__IMAGE_PATH__", json.dumps(args.image_path))
        .replace("__PROMPT__", json.dumps(DEFAULT_PROMPT))
        .replace("__REQUEST_OPTIONS__", json.dumps(request_options))
        .replace("__ICE_GATHER_TIMEOUT_MS__", str(args.ice_gather_timeout_ms))
        .replace("__MAX_GENERATION_SECONDS__", str(MAX_GENERATION_SECONDS))
    )

    class Handler(http.server.BaseHTTPRequestHandler):
        _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def _proxy_backend(self) -> bool:
            return bool(args.proxy_backend) and self.path.startswith("/v1/stream/webrtc/")

        def _proxy(self) -> None:
            backend = args.server_url.rstrip("/")
            url = f"{backend}{self.path}"
            content_len = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(content_len) if content_len > 0 else None

            headers: dict[str, str] = {}
            content_type = self.headers.get("Content-Type")
            if content_type:
                headers["Content-Type"] = content_type

            req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
            try:
                with self._opener.open(req, timeout=30) as resp:
                    resp_body = resp.read()
                    status = getattr(resp, "status", 200)
                    resp_headers = resp.headers
            except urllib.error.HTTPError as exc:
                status = exc.code
                resp_headers = exc.headers
                resp_body = exc.read()
            except Exception as exc:
                resp_body = json.dumps({"detail": f"Demo proxy error: {exc}"}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                return

            self.send_response(status)
            if resp_headers.get("Content-Type"):
                self.send_header("Content-Type", resp_headers.get("Content-Type"))
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

        def do_GET(self) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self._proxy_backend():
                self._proxy()
                return
            self.send_error(404)

        def do_DELETE(self) -> None:
            if self._proxy_backend():
                self._proxy()
                return
            self.send_error(404)

        def log_message(self, format: str, *_args: object) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"Serving LingBot-World-Fast WebRTC demo at {url}")
    print(f"Stream server: {args.server_url}")
    if args.image_path:
        print(f"Image path: {args.image_path}")
    print(f"Request options: {json.dumps(request_options)}")
    print(f"ICE gather timeout: {args.ice_gather_timeout_ms} ms")
    if args.proxy_backend:
        print("Proxy: enabled (browser will call this demo origin; no separate port forward needed for --server-url)")
        if args.turn_url and "localhost" in args.turn_url:
            print(
                "TURN uses localhost. This is valid only when local port 3478 is forwarded to the remote TURN server; "
                "otherwise use the remote public IP/DNS or disable --force-turn-relay."
            )
    if rtc_config:
        print(f"WebRTC config: {json.dumps(rtc_config)}")
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
