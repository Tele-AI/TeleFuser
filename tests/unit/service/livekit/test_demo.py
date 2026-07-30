from __future__ import annotations

import runpy
from pathlib import Path


def test_stream_demo_preserves_controls_and_uses_livekit_transport() -> None:
    project_root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(project_root / "examples" / "stream_server" / "livekit_bidirectional_demo.py"))

    html = namespace["_render_html"]("")

    required_fragments = [
        "LingBot-World-Fast LiveKit Demo",
        "livekit-client@2.21.0",
        "RoomEvent.TrackSubscribed",
        "RoomEvent.DataReceived",
        "adaptiveStream: false",
        "dynacast: false",
        'SERVER_URL + "/v1/stream/sessions"',
        'urls: ["turn:127.0.0.1:3478?transport=tcp"]',
        'iceTransportPolicy: "relay"',
        "{ rtcConfig: TURN_RTC_CONFIG }",
        "topic: CONTROL_TOPIC",
        'type: "control_state"',
        "CONTROL_HEARTBEAT_MS = 1000",
        "if (pressedControls.size > 0)",
        'event: "reset"',
        'event: "reset_pose"',
        'type: "stop"',
        "telemetry-progress",
        "telemetry-cadence",
        "output_cadence_seconds: data.output_cadence_seconds",
        "pipeline_residence_seconds: data.pipeline_residence_seconds",
        "applied_control_latency_seconds: data.applied_control_latency_seconds",
    ]
    for fragment in required_fragments:
        assert fragment in html

    assert "__UTILITIES__" not in html
    assert "__CONTROLS__" not in html
