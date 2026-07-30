"""Shared browser layout and camera-control UI fragments for stream demos."""

from __future__ import annotations

import json
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
<title>LingBot-World-Fast LiveKit Demo</title>
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
  <h1>LingBot-World-Fast LiveKit Demo</h1>
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
        <div class="telemetry-item"><span>Output cadence</span><output id="telemetry-cadence">--</output></div>
        <div class="telemetry-item"><span>Pipeline residence</span><output id="telemetry-residence">--</output></div>
        <div class="telemetry-item"><span>Applied control latency</span><output id="telemetry-control">--</output></div>
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
        <h2>LiveKit Messages</h2>
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
const CONTROL_HEARTBEAT_MS = 1000;

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
    const cadence = metrics.output_cadence_seconds;
    const residence = metrics.pipeline_residence_seconds ?? metrics.chunk_elapsed_seconds;
    const control = metrics.applied_control_latency_seconds ?? metrics.control_to_chunk_seconds;
    if (cadence !== null && cadence !== undefined) {
      setTelemetry("telemetry-cadence", formatSeconds(cadence));
    }
    if (residence !== null && residence !== undefined) {
      setTelemetry("telemetry-residence", formatSeconds(residence));
    }
    if (control !== null && control !== undefined) {
      setTelemetry("telemetry-control", formatSeconds(control));
    }
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

function sendControlState(logMessage = true) {
  if (!dc || dc.readyState !== "open") return;
  const msg = JSON.stringify({ type: "control_state", controls: Array.from(pressedControls).sort() });
  dc.send(msg);
  if (logMessage) log("out", msg);
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
setInterval(() => {
  if (pressedControls.size > 0) sendControlState(false);
}, CONTROL_HEARTBEAT_MS);
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

</script>
</body>
</html>"""
