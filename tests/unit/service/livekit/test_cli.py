from __future__ import annotations

from click.testing import CliRunner

from telefuser.entrypoints.cli.main import main


def test_cli_stream_serve_forwards_livekit_options(monkeypatch) -> None:
    captured = {}

    def fake_run_stream_server(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("telefuser.service.livekit.main.run_stream_server", fake_run_stream_server)

    result = CliRunner().invoke(
        main,
        [
            "stream-serve",
            "pipeline.py",
            "--skip-validation",
            "--livekit-url",
            "wss://livekit.example",
            "--livekit-api-key",
            "key",
            "--livekit-api-secret",
            "secret",
            "--num-workers",
            "2",
            "--worker-gpu-map",
            "0;1",
            "--queue-size",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert captured["pipe_path"] == "pipeline.py"
    assert captured["livekit_url"] == "wss://livekit.example"
    assert captured["livekit_api_key"] == "key"
    assert captured["livekit_api_secret"] == "secret"
    assert captured["num_workers"] == 2
    assert captured["worker_gpu_map"] == "0;1"
    assert captured["queue_size"] == 3
    assert captured["skip_validation"] is True
