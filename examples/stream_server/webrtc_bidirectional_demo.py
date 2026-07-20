"""LingBot-World-Fast WebRTC control demo.

Demonstrates the LingBot-World-Fast bidirectional WebRTC protocol:

* Client creates a DataChannel ("telefuser") for JSON control messages.
* Client sends prompt and direction controls.
* Server sends generated video via media tracks and metadata via DataChannel.

Usage:
    # 1. Start the LingBot stream server:
    telefuser stream-serve examples/stream_server/stream_lingbot_world_fast.py -p 8088 --skip-validation

    # 2. Start this client (opens browser):
    python examples/stream_server/webrtc_bidirectional_demo.py --server-url http://localhost:8088

    # 3. Select an image, enter a prompt, click Connect, then use arrow keys or the D-pad.
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
from pathlib import Path

DEFAULT_SERVER_URL = "http://localhost:8088"
DEFAULT_PORT = 8091
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = str(_PROJECT_ROOT / "examples" / "data" / "lingbot_world_fast" / "image.jpg")
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
    "mountains under a bright blue sky with drifting white clouds. Gentle ripples reflect the tree and sky."
)

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
  input[type="file"] {
    width: 100%;
    color: #374151;
    font-size: 12px;
  }
  .image-preview {
    display: grid;
    gap: 5px;
    margin-top: 2px;
  }
  .image-preview img {
    display: block;
    width: 100%;
    height: 140px;
    object-fit: cover;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #f8fafc;
  }
  .image-preview span {
    color: var(--muted);
    font-size: 11px;
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
  .telemetry-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
  }
  .telemetry-item {
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 8px;
    background: var(--panel-soft);
  }
  .telemetry-item span {
    display: block;
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }
  .telemetry-item output {
    display: block;
    margin-top: 3px;
    color: var(--ink);
    font-size: 14px;
    font-weight: 650;
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
      <div class="telemetry-grid" aria-label="LingBot server telemetry">
        <div class="telemetry-item"><span>Server limit</span><output id="telemetry-service-limit">--</output></div>
        <div class="telemetry-item"><span>Target video</span><output id="telemetry-target-duration">--</output></div>
        <div class="telemetry-item"><span>Generated video</span><output id="telemetry-generated-duration">--</output></div>
        <div class="telemetry-item"><span>Frames / chunks</span><output id="telemetry-progress">--</output></div>
        <div class="telemetry-item"><span>Chunk / control latency</span><output id="telemetry-latency">--</output></div>
        <div class="telemetry-item"><span>Queue / dropped video</span><output id="telemetry-queue">--</output></div>
      </div>
    </section>

    <aside class="side">
      <section class="panel control-panel">
        <h2>Inputs</h2>
        <div class="field">
          <label for="prompt">Prompt</label>
          <textarea id="prompt"></textarea>
        </div>
        <div class="field">
          <label for="image-file">Initial image (optional)</label>
          <input id="image-file" type="file" accept="image/*">
          <div class="image-preview">
            <img id="image-preview" src="/default-image" alt="Initial image preview">
            <span id="image-preview-label">Default image</span>
          </div>
        </div>

        <div class="actions">
          <button id="connect">Connect</button>
          <button id="stop">Stop</button>
          <button id="reset-control">Release Controls</button>
          <button id="reset-pose">Reset Camera Pose</button>
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
const DEFAULT_IMAGE_PATH = __DEFAULT_IMAGE_PATH__;
const DEFAULT_PROMPT = __PROMPT__;
const ICE_GATHER_TIMEOUT_MS = __ICE_GATHER_TIMEOUT_MS__;
const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

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

function formatSeconds(value) {
  const seconds = Number(value);
  return Number.isFinite(seconds) ? seconds.toFixed(2) + " s" : "--";
}

function setTelemetry(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function updateTelemetry(progress, metrics) {
  if (progress) {
    setTelemetry("telemetry-service-limit", formatSeconds(progress.service_max_duration_seconds));
    setTelemetry("telemetry-target-duration", formatSeconds(progress.target_duration_seconds));
    setTelemetry("telemetry-generated-duration", formatSeconds(progress.generated_duration_seconds));
    const frames = progress.generated_frames ?? 0;
    const targetFrames = progress.target_frames ?? "--";
    const chunks = progress.completed_chunks ?? 0;
    const totalChunks = progress.total_chunks ?? "--";
    setTelemetry("telemetry-progress", frames + "/" + targetFrames + " · " + chunks + "/" + totalChunks);
  }
  if (metrics) {
    const chunk = formatSeconds(metrics.chunk_elapsed_seconds);
    const control = formatSeconds(metrics.control_to_chunk_seconds);
    setTelemetry("telemetry-latency", chunk + " / " + control);
    const depth = metrics.output_queue_high_watermark ?? 0;
    const dropped = metrics.dropped_video_payloads ?? 0;
    setTelemetry("telemetry-queue", depth + " high-water · " + dropped + " dropped");
  }
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

async function fetchJsonWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read the selected image."));
    reader.readAsDataURL(file);
  });
}

let imagePreviewObjectUrl = null;

$("image-file").addEventListener("change", () => {
  if (imagePreviewObjectUrl) {
    URL.revokeObjectURL(imagePreviewObjectUrl);
    imagePreviewObjectUrl = null;
  }
  const file = $("image-file").files[0];
  if (file && file.type.startsWith("image/")) {
    imagePreviewObjectUrl = URL.createObjectURL(file);
    $("image-preview").src = imagePreviewObjectUrl;
    $("image-preview-label").textContent = file.name;
    return;
  }
  $("image-preview").src = "/default-image";
  $("image-preview-label").textContent = "Default image";
});

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

function setControlActive(control, active) {
  const btn = document.querySelector('[data-control="' + control + '"]');
  if (btn) btn.classList.toggle("active", active);
}

function sendControlState() {
  if (!dc || dc.readyState !== "open") return;
  const msg = JSON.stringify({ type: "control_state", controls: Array.from(pressedControls).sort() });
  dc.send(msg);
  log("out", msg);
}

function setControlPressed(control, active) {
  if (!control) return;
  if (active) {
    if (pressedControls.has(control)) return;
    pressedControls.add(control);
    setControlActive(control, true);
  } else {
    if (!pressedControls.has(control)) return;
    pressedControls.delete(control);
    setControlActive(control, false);
  }
  sendControlState();
}

function releaseAllControls(sendMessages = true) {
  const hadPressedControls = pressedControls.size > 0;
  for (const control of Array.from(pressedControls)) {
    pressedControls.delete(control);
    setControlActive(control, false);
  }
  if (sendMessages && hadPressedControls) sendControlState();
}

document.querySelectorAll("[data-control]").forEach(btn => {
  const control = btn.dataset.control;
  btn.addEventListener("pointerdown", evt => {
    evt.preventDefault();
    btn.setPointerCapture(evt.pointerId);
    setControlPressed(control, true);
  });
  btn.addEventListener("pointerup", evt => {
    evt.preventDefault();
    setControlPressed(control, false);
  });
  btn.addEventListener("pointercancel", () => setControlPressed(control, false));
  btn.addEventListener("lostpointercapture", () => setControlPressed(control, false));
});

document.addEventListener("keydown", evt => {
  const control = keyToControl[evt.key] || keyToControl[evt.code];
  if (!control) return;
  evt.preventDefault();
  setControlPressed(control, true);
});

document.addEventListener("keyup", evt => {
  const control = keyToControl[evt.key] || keyToControl[evt.code];
  if (!control) return;
  evt.preventDefault();
  setControlPressed(control, false);
});

window.addEventListener("blur", () => releaseAllControls(true));
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseAllControls(true);
});
window.addEventListener("pagehide", () => {
  releaseAllControls(true);
  if (dc && dc.readyState === "open") dc.send(JSON.stringify({ type: "stop" }));
});

$("reset-control").onclick = () => {
  releaseAllControls(false);
  if (dc && dc.readyState === "open") {
    const msg = JSON.stringify({ type: "control", control: "up", event: "reset" });
    dc.send(msg);
    log("out", msg);
  }
};

$("reset-pose").onclick = () => {
  releaseAllControls(false);
  if (dc && dc.readyState === "open") {
    const msg = JSON.stringify({ type: "control", control: "up", event: "reset_pose" });
    dc.send(msg);
    log("out", msg);
  }
};

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

  setStatus("Connecting...");
  $("connect").disabled = true;

  try {
    const image = imageFile ? await readFileAsDataUrl(imageFile) : null;
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
        updateTelemetry(data.stream_progress, {
          ...(data.runtime_metrics || {}),
          chunk_elapsed_seconds: data.chunk_elapsed_seconds,
          control_to_chunk_seconds: data.control_to_chunk_seconds,
        });
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
    const requestBody = {
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
      task: "bidirectional",
      prompt,
    };
    if (image) {
      requestBody.image = image;
    } else {
      requestBody.image_path = DEFAULT_IMAGE_PATH;
    }
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

$("stop").onclick = () => {
  setStatus("Stopping...");
  releaseAllControls(true);
  if (dc && dc.readyState === "open") {
    const msg = JSON.stringify({ type: "stop" });
    dc.send(msg);
    log("out", msg);
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

$("prompt").value = DEFAULT_PROMPT;
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="LingBot-World-Fast WebRTC control demo")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Stream server base URL")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local HTTP server port")
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

    # When proxying, the browser should call the demo origin (no separate port forward needed for --server-url).
    server_url_for_browser = "" if args.proxy_backend else args.server_url

    html = (
        HTML_TEMPLATE.replace("__SERVER_URL__", json.dumps(server_url_for_browser))
        .replace("__RTC_CONFIG__", json.dumps(rtc_config))
        .replace("__DEFAULT_IMAGE_PATH__", json.dumps(DEFAULT_IMAGE_PATH))
        .replace("__PROMPT__", json.dumps(DEFAULT_PROMPT))
        .replace("__ICE_GATHER_TIMEOUT_MS__", str(args.ice_gather_timeout_ms))
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
            if self.path == "/default-image":
                body = Path(DEFAULT_IMAGE_PATH).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
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
