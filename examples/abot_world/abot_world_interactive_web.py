"""Direct-loaded, LingBot-style browser controller for ABot-World.

The page intentionally uses a small native HTTP server instead of a component
framework: control buttons must reflect keyboard state immediately, while the
GPU worker keeps one causal ABot session alive.  Press Connect once, then hold
WASD/arrow keys for movement or IJKL for camera rotation; a new causal block
is scheduled as soon as the previous one completes.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

try:
    from ._loader import DEFAULT_PROMPT, get_pipeline
except ImportError:  # Supports direct execution from examples/abot_world.
    from _loader import DEFAULT_PROMPT, get_pipeline

from telefuser.pipelines.abot_world.interactive import ABotWorldInteractivePipeline, ABotWorldInteractiveSession
from telefuser.utils.video import save_video

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_OFFICIAL_SAMPLE = _PROJECT_ROOT.parent / "ABot-World" / "web_client" / "datasets" / "images" / "84b90ad568b693d2.png"
_OUTPUT_DIR = _PROJECT_ROOT / "work_dirs" / "abot_world_interactive"
_ACTION_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")
_DEFAULT_OUTPUT_QUEUE_SIZE = 4
_PULL_TIMEOUT_SECONDS = 15.0


class InteractiveRuntime:
    """Own one causal ABot session plus a LingBot-style output queue."""

    def __init__(
        self,
        pipeline: ABotWorldInteractivePipeline,
        fps: int,
        control_latent_frames: int,
        output_queue_size: int = _DEFAULT_OUTPUT_QUEUE_SIZE,
    ) -> None:
        if control_latent_frames not in {1, 2, 3}:
            raise ValueError("control_latent_frames must be 1, 2, or 3")
        if output_queue_size <= 0:
            raise ValueError("output_queue_size must be positive")
        self.pipeline = pipeline
        self.fps = fps
        self.control_latent_frames = control_latent_frames
        self.output_queue_size = output_queue_size
        self.lock = threading.RLock()
        self.session: ABotWorldInteractiveSession | None = None
        self.frames: list[Image.Image] = []
        self.chunk_index = 0
        self.version = 0
        self._encoded_blocks: dict[int, list[bytes]] = {}
        self._current_frame_urls: list[str] = []
        self._output_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=output_queue_size)
        self._control_event = threading.Event()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._active_controls: dict[str, bool] = {}
        self._control_revision = 0
        self._worker_error: str | None = None
        self._queue_high_watermark = 0
        self._dropped_video_blocks = 0
        self._dropped_video_frames = 0
        self._producer_backpressure_events = 0
        self._producer_backpressure_seconds = 0.0
        self.latest_frame_path = _OUTPUT_DIR / "current_frame.jpg"
        self.video_path = _OUTPUT_DIR / "current_session.mp4"

    @staticmethod
    def _actions(raw_controls: object) -> dict[str, bool]:
        if raw_controls is None:
            controls: list[object] = []
        elif isinstance(raw_controls, list):
            controls = raw_controls
        else:
            raise ValueError("controls must be an array")
        pressed = {str(control).upper() for control in controls}
        unknown = pressed.difference(_ACTION_ORDER)
        if unknown:
            raise ValueError("Unsupported controls: " + ", ".join(sorted(unknown)))
        return {control: True for control in pressed}

    def _queue_metrics(self) -> dict[str, int | float]:
        return {
            "queued_chunks": self._output_queue.qsize(),
            "queue_capacity": self.output_queue_size,
            "queue_high_watermark": self._queue_high_watermark,
            "dropped_video_blocks": self._dropped_video_blocks,
            "dropped_video_frames": self._dropped_video_frames,
            "producer_backpressure_events": self._producer_backpressure_events,
            "producer_backpressure_seconds": round(self._producer_backpressure_seconds, 3),
        }

    def _clear_output_queue(self) -> None:
        while True:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                return

    def _cache_block_frames(self, block: list[Image.Image]) -> None:
        if not block:
            raise RuntimeError("No decoded ABot frames are available")
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.version += 1
        encoded: list[bytes] = []
        for frame in block:
            buffer = BytesIO()
            frame.save(buffer, format="JPEG", quality=92)
            encoded.append(buffer.getvalue())
        self._encoded_blocks[self.version] = encoded
        for stale_version in sorted(self._encoded_blocks)[:-12]:
            del self._encoded_blocks[stale_version]
        block[-1].save(self.latest_frame_path, format="JPEG", quality=92)
        self._current_frame_urls = [f"/api/block-frame/{self.version}/{index}" for index in range(len(encoded))]

    def _result(
        self,
        *,
        new_frames: int,
        controls: dict[str, bool],
        status: str,
        event_type: str,
        control_revision: int,
    ) -> dict[str, Any]:
        return {
            "type": event_type,
            "chunk": self.chunk_index,
            "new_frames": new_frames,
            "total_frames": len(self.frames),
            "controls": sorted(controls),
            "control_revision": control_revision,
            "frame_url": f"/api/frame?v={self.version}",
            "frame_urls": list(self._current_frame_urls),
            "status": status,
            "queue": self._queue_metrics(),
        }

    def _enqueue_video_output(self, payload: dict[str, Any]) -> bool:
        """Block the producer when the browser has not drained its FIFO.

        This is deliberate backpressure: normal streaming never discards a
        generated ABot block merely because the browser is temporarily ahead
        of its configured playback clock.
        """
        blocked_started_at: float | None = None
        while not self._stop_event.is_set():
            try:
                self._output_queue.put(payload, timeout=0.1)
                break
            except queue.Full:
                if blocked_started_at is None:
                    blocked_started_at = time.monotonic()
                    with self.lock:
                        self._producer_backpressure_events += 1
                continue
        else:
            return False

        if blocked_started_at is not None:
            with self.lock:
                self._producer_backpressure_seconds += time.monotonic() - blocked_started_at
        with self.lock:
            self._queue_high_watermark = max(self._queue_high_watermark, self._output_queue.qsize())
        return True

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._control_event.wait(timeout=0.25)
            if self._stop_event.is_set():
                return
            with self.lock:
                session = self.session
                controls = dict(self._active_controls)
                revision = self._control_revision
            if session is None:
                return
            if not controls:
                continue
            try:
                block = self.pipeline.generate_next_block(session, controls, self.control_latent_frames)
            except Exception as error:
                with self.lock:
                    self._worker_error = str(error)
                return
            with self.lock:
                if session is not self.session:
                    return
                self.frames.extend(block)
                self.chunk_index += 1
                self._cache_block_frames(block)
                payload = self._result(
                    new_frames=len(block),
                    controls=controls,
                    status="Causal block generated by the background producer.",
                    event_type="chunk",
                    control_revision=revision,
                )
                should_publish = not self._stop_event.is_set()
            if should_publish and not self._enqueue_video_output(payload):
                return

    def _stop_worker(self) -> None:
        with self.lock:
            self._stop_event.set()
            self._control_event.set()
            worker = self._worker
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=30.0)
        if worker is not None and worker.is_alive():
            raise RuntimeError("Timed out while stopping the ABot background worker")
        with self.lock:
            if self._worker is worker:
                self._worker = None

    def _start_worker_locked(self) -> None:
        self._stop_event.clear()
        worker = threading.Thread(target=self._worker_loop, daemon=True, name="abot-world-producer")
        self._worker = worker
        worker.start()

    def start(self, image_path: str, prompt: str, seed: int, raw_controls: object) -> dict[str, Any]:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Image does not exist on the server: {path}")
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        controls = self._actions(raw_controls)
        self._stop_worker()
        with self.lock:
            if self.session is not None:
                self.pipeline.close_interactive_session(self.session)
            self._clear_output_queue()
            self._encoded_blocks.clear()
            self._current_frame_urls = []
            self.frames = []
            self.chunk_index = 0
            self.version = 0
            self._queue_high_watermark = 0
            self._dropped_video_blocks = 0
            self._dropped_video_frames = 0
            self._producer_backpressure_events = 0
            self._producer_backpressure_seconds = 0.0
            self._worker_error = None
            self._control_revision = 0
            self._active_controls = controls
            with Image.open(path) as source:
                image = source.convert("RGB")
            self.session = self.pipeline.create_interactive_session(image, prompt.strip(), seed=seed)
            # Session creation prepares causal state only. An explicit
            # non-empty control snapshot must drive the first DiT block.
            _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            image.save(self.latest_frame_path, format="JPEG", quality=92)
            result = self._result(
                new_frames=0,
                controls=controls,
                status="Causal session ready; waiting for control input.",
                event_type="preview",
                control_revision=self._control_revision,
            )
            self._start_worker_locked()
            if controls:
                self._control_event.set()
            return result

    def set_controls(self, raw_controls: object, raw_revision: object = None) -> dict[str, Any]:
        controls = self._actions(raw_controls)
        revision = None if raw_revision is None else int(raw_revision)
        with self.lock:
            if self.session is None:
                raise RuntimeError("No active session; press Connect first")
            if revision is not None and revision < self._control_revision:
                return {
                    "type": "status",
                    "stage": "stale_control_ignored",
                    "controls": sorted(self._active_controls),
                    "control_revision": self._control_revision,
                    "queue": self._queue_metrics(),
                }
            self._active_controls = controls
            self._control_revision = self._control_revision + 1 if revision is None else revision
            result = {
                "type": "status",
                "stage": "control_state",
                "controls": sorted(controls),
                "control_revision": self._control_revision,
                "queue": self._queue_metrics(),
            }
        if controls:
            self._control_event.set()
        else:
            self._control_event.clear()
        return result

    def pull_output(self, timeout_seconds: float = _PULL_TIMEOUT_SECONDS) -> dict[str, Any]:
        timeout = min(max(float(timeout_seconds), 0.0), _PULL_TIMEOUT_SECONDS)
        try:
            payload = dict(self._output_queue.get(timeout=timeout))
        except queue.Empty:
            with self.lock:
                stage = "waiting_for_chunk" if self._active_controls else "waiting_for_input"
                return {
                    "type": "status",
                    "stage": stage,
                    "controls": sorted(self._active_controls),
                    "control_revision": self._control_revision,
                    "queue": self._queue_metrics(),
                    "worker_error": self._worker_error,
                }
        with self.lock:
            payload["queue"] = self._queue_metrics()
            return payload

    def next(self, raw_controls: object) -> dict[str, Any]:
        self.set_controls(raw_controls)
        return self.pull_output()

    def stop(self) -> dict[str, Any]:
        self._stop_worker()
        with self.lock:
            if self.session is not None:
                self.pipeline.close_interactive_session(self.session)
                self.session = None
            self._active_controls = {}
            self._clear_output_queue()
            if self.frames:
                _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                save_video(self.frames, str(self.video_path), fps=self.fps, quality=8)
            return {
                "status": "Stopped. The output queue was drained; GPU model weights remain loaded.",
                "video_url": "/api/video" if self.video_path.is_file() else None,
                "queue": self._queue_metrics(),
            }

    def frame_bytes(self) -> bytes | None:
        with self.lock:
            return self.latest_frame_path.read_bytes() if self.latest_frame_path.is_file() else None

    def block_frame_bytes(self, version: int, index: int) -> bytes | None:
        with self.lock:
            block = self._encoded_blocks.get(version)
            if block is None or not 0 <= index < len(block):
                return None
            return block[index]

    def video_bytes(self) -> bytes | None:
        with self.lock:
            return self.video_path.read_bytes() if self.video_path.is_file() else None


_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ABot-World Single-GPU Controller</title>
  <style>
    :root { --bg:#f5f7fb; --panel:#fff; --text:#111827; --muted:#6b7280; --line:#d8dee9; --blue:#1d4ed8; --blue-soft:#dbeafe; --red:#b91c1c; --ink:#0f172a; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
    main { max-width:1180px; margin:0 auto; padding:28px 20px 36px; }
    h1 { margin:0 0 16px; font-size:24px; }
    .workspace { display:grid; grid-template-columns:minmax(0,1fr) 340px; gap:18px; align-items:start; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    .output { overflow:hidden; }
    .output-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-bottom:1px solid var(--line); }
    .output-head h2,.control-panel h2 { margin:0; font-size:14px; color:var(--ink); }
    #status { color:var(--muted); font-size:12px; text-align:right; }
    #output-frame { display:block; width:100%; aspect-ratio:16/9; object-fit:cover; background:#000; }
    .caption { padding:10px 14px; color:var(--muted); font-size:12px; border-top:1px solid var(--line); }
    .control-panel { padding:14px; }
    .field { display:grid; gap:6px; margin-top:12px; }
    label { color:#374151; font-size:12px; font-weight:650; }
    textarea,input { width:100%; border:1px solid #cbd5e1; border-radius:6px; padding:8px 9px; font:13px inherit; }
    textarea { min-height:82px; resize:vertical; }
    .actions { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
    button { min-height:36px; padding:7px 14px; border:0; border-radius:6px; cursor:pointer; font-size:13px; font-weight:650; }
    button:disabled { cursor:default; opacity:.55; }
    #connect { background:var(--blue); color:#fff; }
    #stop { display:none; background:var(--red); color:#fff; }
    #release { background:#e2e8f0; color:#0f172a; }
    .control-pads { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:14px; }
    .control-pad { padding:9px 4px 6px; border:1px solid #e2e8f0; border-radius:6px; background:#f8fafc; }
    .control-pad h3 { margin:0; color:#475569; font-size:11px; text-align:center; }
    .dpad { display:grid; grid-template-columns:44px 44px 44px; grid-template-rows:44px 44px 44px; gap:6px; justify-content:center; margin:8px 0 2px; user-select:none; }
    .dpad button { width:44px; height:44px; padding:0; border:1px solid #cbd5e1; background:#f8fafc; color:#475569; font-size:23px; line-height:1; }
    .dpad button.active { background:var(--blue-soft); border-color:#60a5fa; color:var(--blue); box-shadow:0 0 0 2px #bfdbfe; }
    .empty { width:44px; height:44px; }
    .hint { margin:14px 0 0; color:var(--muted); font-size:11px; line-height:1.45; }
    #download { display:none; margin-top:10px; color:var(--blue); font-size:12px; font-weight:650; }
    @media (max-width:900px) { main{padding:18px 12px 28px}.workspace{grid-template-columns:1fr}.output-head{align-items:flex-start;flex-direction:column}#status{text-align:left} }
  </style>
</head>
<body>
<main>
  <h1>ABot-World Single-GPU Interactive Demo</h1>
  <div class="workspace">
    <section class="panel output">
      <div class="output-head"><h2>Server Output</h2><div id="status">GPU weights are loaded. Ready.</div></div>
      <img id="output-frame" src="/sample-image" alt="Current ABot world frame">
      <div class="caption">Connect starts a persistent causal session. The browser prebuffers one to two blocks, then consumes frames in order at 12 FPS. A full server FIFO pauses the producer instead of dropping frames.</div>
    </section>
    <aside class="panel control-panel">
      <h2>Inputs</h2>
      <div class="field"><label for="prompt">Prompt</label><textarea id="prompt"></textarea></div>
      <div class="field"><label for="image-path">Server image path</label><input id="image-path" type="text"></div>
      <div class="field"><label for="seed">Seed</label><input id="seed" type="number" value="42" step="1"></div>
      <div class="actions"><button id="connect">Connect</button><button id="stop">Stop</button><button id="release">Release Controls</button></div>
      <div class="control-pads">
        <div class="control-pad"><h3>Move · WASD / Arrows</h3><div class="dpad" aria-label="Move controls">
          <div class="empty"></div><button data-control="w" title="Move forward">↑</button><div class="empty"></div>
          <button data-control="a" title="Move left">←</button><div class="empty"></div><button data-control="d" title="Move right">→</button>
          <div class="empty"></div><button data-control="s" title="Move backward">↓</button><div class="empty"></div>
        </div></div>
        <div class="control-pad"><h3>Rotate · IJKL</h3><div class="dpad" aria-label="Camera controls">
          <div class="empty"></div><button data-control="i" title="Pitch up">↑</button><div class="empty"></div>
          <button data-control="j" title="Yaw left">↶</button><div class="empty"></div><button data-control="l" title="Yaw right">↷</button>
          <div class="empty"></div><button data-control="k" title="Pitch down">↓</button><div class="empty"></div>
        </div></div>
      </div>
      <p class="hint">Hold a key or mouse button to light it. Release it to stop that action. While a control is held, the background producer fills a lossless bounded FIFO. With no control held, no idle chunk is sent. The browser waits for one configured playback second of predecoded frames, then consumes them in order at the configured FPS. A full FIFO applies producer backpressure; normal playback never drops generated frames.</p>
      <a id="download" href="/api/video" download="abot_world_session.mp4">Download stopped session video</a>
    </aside>
  </div>
</main>
<script>
const DEFAULT_IMAGE_PATH = __DEFAULT_IMAGE_PATH__;
const DEFAULT_PROMPT = __DEFAULT_PROMPT__;
const pressedControls = new Set();
const keyToControl = { ArrowUp:"w", ArrowDown:"s", ArrowLeft:"a", ArrowRight:"d", KeyW:"w", KeyA:"a", KeyS:"s", KeyD:"d", KeyI:"i", KeyJ:"j", KeyK:"k", KeyL:"l" };
const PLAYBACK_FPS = __PLAYBACK_FPS__;
const FRAME_INTERVAL_MS = 1000 / PLAYBACK_FPS;
const PLAYBACK_JITTER_BUFFER_FRAMES = PLAYBACK_FPS;
const MAX_CLIENT_BUFFERED_FRAMES = 2 * PLAYBACK_FPS;
const playbackQueue = [];
let running = false;
let requestInFlight = false;
let pulling = false;
let playbackActive = false;
let bufferedFrames = 0;
let preloadTail = Promise.resolve();
let playbackGeneration = 0;
let displayedFrameObjectUrl = null;
let controlRevision = 0;
let controlHeartbeat = null;

const byId = id => document.getElementById(id);
byId("prompt").value = DEFAULT_PROMPT;
byId("image-path").value = DEFAULT_IMAGE_PATH;

function setStatus(value) { byId("status").textContent = value; }
function controls() { return Array.from(pressedControls).sort(); }
function setControlActive(control, active) {
  const button = document.querySelector("[data-control=\"" + control + "\"]");
  if (button) button.classList.toggle("active", active);
}
function queueText(result) {
  const queue = result.queue || {};
  if (!Number.isInteger(queue.queued_chunks)) return "";
  let text = " · queue " + queue.queued_chunks + "/" + queue.queue_capacity;
  if (queue.producer_backpressure_events) {
    text += " · producer paused " + queue.producer_backpressure_events + "x";
  }
  return text;
}
function showStatus(result) {
  const controlsText = Array.isArray(result.controls) && result.controls.length ? " [" + result.controls.join(",") + "]" : " [waiting for input]";
  if (result.type === "chunk" || result.type === "preview") {
    setStatus("Server: denoising_chunk #" + result.chunk + controlsText + " · " + result.total_frames + " frames" + queueText(result));
  } else if (result.worker_error) {
    setStatus("Generation stopped: " + result.worker_error);
  }
}
async function callApi(path, body) {
  const response = await fetch(path, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body || {}) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}
function clearPlayback() {
  playbackGeneration += 1;
  while (playbackQueue.length) URL.revokeObjectURL(playbackQueue.shift().objectUrl);
  if (displayedFrameObjectUrl) {
    URL.revokeObjectURL(displayedFrameObjectUrl);
    displayedFrameObjectUrl = null;
  }
  bufferedFrames = 0;
  preloadTail = Promise.resolve();
}
async function preloadFrame(frameUrl) {
  const response = await fetch(frameUrl, { cache:"no-store" });
  if (!response.ok) throw new Error("Unable to fetch generated frame");
  const objectUrl = URL.createObjectURL(await response.blob());
  const image = new Image();
  if ("decode" in image) {
    image.src = objectUrl;
    await image.decode();
  } else {
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("Unable to decode generated frame"));
      image.src = objectUrl;
    });
  }
  return { image, objectUrl };
}
function maybeStartPlayback(generation) {
  if (generation !== playbackGeneration || playbackActive) return;
  if (playbackQueue.length < PLAYBACK_JITTER_BUFFER_FRAMES) return;
  void playQueuedFrames(generation);
}
async function playQueuedFrames(generation) {
  playbackActive = true;
  let nextFrameAt = performance.now();
  while (playbackQueue.length && generation === playbackGeneration) {
    const frame = playbackQueue.shift();
    const now = performance.now();
    // Do not "catch up" by presenting several frames in one browser tick.
    // That would preserve bytes but look like visual frame skipping.
    if (nextFrameAt < now) nextFrameAt = now;
    const previousObjectUrl = displayedFrameObjectUrl;
    byId("output-frame").src = frame.objectUrl;
    displayedFrameObjectUrl = frame.objectUrl;
    bufferedFrames -= 1;
    nextFrameAt += FRAME_INTERVAL_MS;
    if (previousObjectUrl) URL.revokeObjectURL(previousObjectUrl);
    const delay = nextFrameAt - performance.now();
    await new Promise(resolve => window.setTimeout(resolve, delay));
  }
  playbackActive = false;
  if (generation === playbackGeneration) maybeStartPlayback(generation);
}
function enqueueFrames(frameUrls) {
  const generation = playbackGeneration;
  bufferedFrames += frameUrls.length;
  preloadTail = preloadTail
    .then(async () => {
      const frames = await Promise.all(frameUrls.map(preloadFrame));
      if (generation !== playbackGeneration) {
        frames.forEach(frame => URL.revokeObjectURL(frame.objectUrl));
        return;
      }
      playbackQueue.push(...frames);
      maybeStartPlayback(generation);
    })
    .catch(error => {
      if (generation === playbackGeneration) {
        bufferedFrames -= frameUrls.length;
        setStatus("Frame preload failed: " + error.message);
      }
    });
}
function showResult(result) {
  if (Array.isArray(result.frame_urls) && result.frame_urls.length) enqueueFrames(result.frame_urls);
  showStatus(result);
}
function sendControlState() {
  if (!running) return;
  const revision = ++controlRevision;
  void callApi("/api/control", { controls: controls(), revision })
    .then(showStatus)
    .catch(error => { if (running) setStatus("Control update failed: " + error.message); });
}
function setControlPressed(control, active, sync = true) {
  if (active) {
    if (pressedControls.has(control)) return;
    pressedControls.add(control);
  } else {
    if (!pressedControls.has(control)) return;
    pressedControls.delete(control);
  }
  setControlActive(control, active);
  if (sync && running) sendControlState();
}
function releaseAllControls() {
  let changed = false;
  for (const control of Array.from(pressedControls)) {
    setControlPressed(control, false, false);
    changed = true;
  }
  if (changed && running) sendControlState();
}
function startControlHeartbeat() {
  window.clearInterval(controlHeartbeat);
  controlHeartbeat = window.setInterval(() => {
    if (running && pressedControls.size) sendControlState();
  }, 1000);
}
function stopControlHeartbeat() {
  window.clearInterval(controlHeartbeat);
  controlHeartbeat = null;
}
async function pullChunks() {
  if (!running || pulling) return;
  if (bufferedFrames >= MAX_CLIENT_BUFFERED_FRAMES) {
    window.setTimeout(pullChunks, FRAME_INTERVAL_MS);
    return;
  }
  pulling = true;
  try {
    const result = await callApi("/api/pull", { timeout_ms: 15000 });
    if (result.type === "chunk" || result.type === "preview") showResult(result);
    else showStatus(result);
  } catch (error) {
    if (running) {
      running = false;
      stopControlHeartbeat();
      setStatus("Stream stopped: " + error.message);
      byId("connect").disabled = false;
      byId("stop").style.display = "none";
    }
  } finally {
    pulling = false;
    if (running) window.setTimeout(pullChunks, 0);
  }
}

document.querySelectorAll("[data-control]").forEach(button => {
  const control = button.dataset.control;
  button.addEventListener("pointerdown", event => { event.preventDefault(); button.setPointerCapture(event.pointerId); setControlPressed(control, true); });
  button.addEventListener("pointerup", event => { event.preventDefault(); setControlPressed(control, false); });
  button.addEventListener("pointercancel", () => setControlPressed(control, false));
  button.addEventListener("lostpointercapture", () => setControlPressed(control, false));
});
document.addEventListener("keydown", event => {
  if (["INPUT", "TEXTAREA"].includes(event.target.tagName)) return;
  const control = keyToControl[event.code] || keyToControl[event.key];
  if (!control) return;
  event.preventDefault();
  setControlPressed(control, true);
});
document.addEventListener("keyup", event => {
  const control = keyToControl[event.code] || keyToControl[event.key];
  if (!control) return;
  event.preventDefault();
  setControlPressed(control, false);
});
window.addEventListener("blur", releaseAllControls);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseAllControls(); });

byId("connect").onclick = async () => {
  if (running || requestInFlight) return;
  requestInFlight = true;
  clearPlayback();
  controlRevision = 0;
  byId("connect").disabled = true;
  byId("download").style.display = "none";
  setStatus("Encoding image and creating ABot session...");
  try {
    const result = await callApi("/api/start", { image_path: byId("image-path").value.trim(), prompt: byId("prompt").value, seed: Number(byId("seed").value), controls: controls() });
    running = true;
    showResult(result);
    byId("stop").style.display = "inline-block";
    startControlHeartbeat();
    void pullChunks();
  } catch (error) {
    setStatus("Start failed: " + error.message);
    byId("connect").disabled = false;
  } finally {
    requestInFlight = false;
  }
};
byId("stop").onclick = async () => {
  running = false;
  stopControlHeartbeat();
  releaseAllControls();
  setStatus("Stopping session after the active chunk...");
  try {
    const result = await callApi("/api/stop");
    setStatus(result.status + queueText(result));
    if (result.video_url) byId("download").style.display = "inline-block";
  } catch (error) {
    setStatus("Stop failed: " + error.message);
  } finally {
    clearPlayback();
    byId("connect").disabled = false;
    byId("stop").style.display = "none";
  }
};
byId("release").onclick = releaseAllControls;
</script>
</body>
</html>
"""


def _render_html(runtime: InteractiveRuntime) -> bytes:
    return (
        _HTML.replace("__DEFAULT_IMAGE_PATH__", json.dumps(str(_OFFICIAL_SAMPLE)))
        .replace("__DEFAULT_PROMPT__", json.dumps(DEFAULT_PROMPT))
        .replace("__PLAYBACK_FPS__", json.dumps(runtime.fps))
        .encode("utf-8")
    )


def _make_handler(runtime: InteractiveRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: HTTPStatus, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length > 1_000_000:
                raise ValueError("Request body is too large")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("JSON request body must be an object")
            return value

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                self._send(HTTPStatus.OK, _render_html(runtime), "text/html; charset=utf-8")
            elif path == "/sample-image":
                self._send(HTTPStatus.OK, _OFFICIAL_SAMPLE.read_bytes(), "image/jpeg")
            elif path.startswith("/api/block-frame/"):
                parts = path.split("/")
                if len(parts) != 5:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    version, index = int(parts[3]), int(parts[4])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                payload = runtime.block_frame_bytes(version, index)
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "Frame is no longer available")
                else:
                    self._send(HTTPStatus.OK, payload, "image/jpeg")
            elif path == "/api/frame":
                payload = runtime.frame_bytes()
                if payload is None:
                    self.send_error(HTTPStatus.NO_CONTENT)
                else:
                    self._send(HTTPStatus.OK, payload, "image/jpeg")
            elif path == "/api/video":
                payload = runtime.video_bytes()
                if payload is None:
                    self.send_error(HTTPStatus.NOT_FOUND, "No stopped session video is available")
                else:
                    self._send(HTTPStatus.OK, payload, "video/mp4")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/start":
                    result = runtime.start(
                        str(payload.get("image_path") or ""),
                        str(payload.get("prompt") or ""),
                        int(payload.get("seed", 42)),
                        payload.get("controls"),
                    )
                elif path == "/api/control":
                    result = runtime.set_controls(payload.get("controls"), payload.get("revision"))
                elif path == "/api/pull":
                    result = runtime.pull_output(float(payload.get("timeout_ms", 15_000)) / 1000.0)
                elif path == "/api/next":
                    result = runtime.next(payload.get("controls"))
                elif path == "/api/stop":
                    result = runtime.stop()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"detail": str(error)})
                return
            except Exception as error:  # pragma: no cover - protects the local HTTP boundary.
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"detail": str(error)})
                return
            self._send_json(HTTPStatus.OK, result)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct-loaded ABot-World single-GPU interactive controller")
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--latent-frames", type=int, default=31)
    parser.add_argument("--fps", type=int, default=8, help="Playback and downloaded-video FPS; 8 is the real-time target.")
    parser.add_argument(
        "--control-latent-frames",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="Causal latents per control update: 3 matches the official ABot streaming checkpoint; 2 is the 8-FPS experimental target and 1 is experimental.",
    )
    parser.add_argument("--output-queue-size", type=int, default=_DEFAULT_OUTPUT_QUEUE_SIZE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.height % 32 or args.width % 32:
        parser.error("height and width must be divisible by 32")
    if args.latent_frames < 1 or (args.latent_frames - 1) % 3:
        parser.error("latent-frames must be positive and equal to 1 mod 3")
    model_root = args.model_root or str(_PROJECT_ROOT.parent / "model_zoo" / "ABot-World-0-5B-LF")
    print("Loading VAE, T5, and ABot DiT onto GPU before accepting browser requests.", flush=True)
    pipeline = get_pipeline(
        model_root,
        height=args.height,
        width=args.width,
        latent_frames=args.latent_frames,
        pipeline_class=ABotWorldInteractivePipeline,
    )
    pipeline.preload_models()
    runtime = InteractiveRuntime(
        pipeline,
        args.fps,
        args.control_latent_frames,
        output_queue_size=args.output_queue_size,
    )
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(runtime))
    print(f"ABot controller ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping ABot controller.", flush=True)
    finally:
        server.server_close()
        runtime.stop()
        pipeline.close()


if __name__ == "__main__":
    main()
