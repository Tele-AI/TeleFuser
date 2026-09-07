from __future__ import annotations

import threading
import time
from pathlib import Path

from PIL import Image

from examples.abot_world.abot_world_interactive_web import InteractiveRuntime


class _FakePipeline:
    def __init__(self) -> None:
        self.generate_calls = 0
        self.closed_sessions: list[object] = []

    def create_interactive_session(self, image: Image.Image, prompt: str, *, seed: int) -> object:
        assert image.mode == "RGB"
        assert prompt
        return object()

    def close_interactive_session(self, session: object) -> None:
        self.closed_sessions.append(session)

    def generate_next_block(self, session: object, controls: dict[str, bool], control_latent_frames: int) -> list:
        self.generate_calls += 1
        assert controls
        assert control_latent_frames == 3
        return [object()]


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(32, 64, 96)).save(path)


def test_connect_with_empty_controls_does_not_advance_dit(tmp_path: Path) -> None:
    image_path = tmp_path / "initial.png"
    _write_image(image_path)
    pipeline = _FakePipeline()
    runtime = InteractiveRuntime(pipeline, fps=12, control_latent_frames=3, output_queue_size=2)

    result = runtime.start(str(image_path), "test prompt", seed=42, raw_controls=[])
    try:
        assert result["new_frames"] == 0
        assert result["status"] == "Causal session ready; waiting for control input."
        assert pipeline.generate_calls == 0
    finally:
        runtime.stop()


def test_runtime_accepts_two_latent_experimental_chunk(tmp_path: Path) -> None:
    image_path = tmp_path / "initial.png"
    _write_image(image_path)
    runtime = InteractiveRuntime(_FakePipeline(), fps=8, control_latent_frames=2, output_queue_size=2)
    try:
        assert runtime.control_latent_frames == 2
        assert runtime.fps == 8
    finally:
        runtime.stop()


def test_full_fifo_applies_backpressure_without_reordering() -> None:
    pipeline = _FakePipeline()
    runtime = InteractiveRuntime(pipeline, fps=12, control_latent_frames=3, output_queue_size=1)
    first = {"type": "chunk", "index": 1}
    second = {"type": "chunk", "index": 2}
    runtime._output_queue.put(first)

    producer_done = threading.Event()

    def produce() -> None:
        assert runtime._enqueue_video_output(second)
        producer_done.set()

    producer = threading.Thread(target=produce)
    producer.start()
    deadline = time.monotonic() + 1.0
    while runtime._producer_backpressure_events == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runtime._producer_backpressure_events > 0
    assert not producer_done.is_set()

    assert runtime._output_queue.get(timeout=1.0) is first
    producer.join(timeout=1.0)
    assert producer_done.is_set()
    assert runtime._output_queue.get(timeout=1.0) is second
    assert runtime._dropped_video_blocks == 0
    assert runtime._dropped_video_frames == 0
