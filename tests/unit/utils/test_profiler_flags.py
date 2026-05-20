from __future__ import annotations

from telefuser.utils import profiler as profiler_module


class _FakePlatform:
    device_type = "cpu"

    def synchronize(self) -> None:
        pass


class _FakeTorchProfiler:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def export_chrome_trace(self, path: str) -> None:
        self.trace_path = path

    def key_averages(self):
        return []


def test_torch_profiler_options_follow_env(monkeypatch, tmp_path) -> None:
    profile_kwargs: list[dict] = []

    def fake_profile(**kwargs):
        profile_kwargs.append(kwargs)
        return _FakeTorchProfiler(**kwargs)

    monkeypatch.setenv("ENABLE_PROFILER_NAMES", "outer")
    monkeypatch.setenv("TELEFUSER_PROFILER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("TELEFUSER_TORCH_PROFILER_RECORD_SHAPES", "false")
    monkeypatch.setenv("TELEFUSER_TORCH_PROFILER_PROFILE_MEMORY", "false")
    monkeypatch.setenv("TELEFUSER_TORCH_PROFILER_WITH_STACK", "false")
    monkeypatch.setattr(profiler_module, "current_platform", _FakePlatform())
    monkeypatch.setattr(profiler_module.torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(profiler_module, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(profiler_module, "capture_memory_snapshot", lambda: None)

    context = profiler_module.ProfilingContext("outer")
    context.__enter__()
    assert context._profiler is not None
    context._profiler.stop()

    assert profile_kwargs[0]["record_shapes"] is False
    assert profile_kwargs[0]["profile_memory"] is False
    assert profile_kwargs[0]["with_stack"] is False


def test_torch_profiler_options_preserve_existing_defaults(monkeypatch, tmp_path) -> None:
    profile_kwargs: list[dict] = []

    def fake_profile(**kwargs):
        profile_kwargs.append(kwargs)
        return _FakeTorchProfiler(**kwargs)

    monkeypatch.setenv("ENABLE_PROFILER_NAMES", "outer")
    monkeypatch.setenv("TELEFUSER_PROFILER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(profiler_module, "current_platform", _FakePlatform())
    monkeypatch.setattr(profiler_module.torch.profiler, "profile", fake_profile)
    monkeypatch.setattr(profiler_module, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(profiler_module, "capture_memory_snapshot", lambda: None)

    context = profiler_module.ProfilingContext("outer")
    context.__enter__()
    assert context._profiler is not None
    context._profiler.stop()

    assert profile_kwargs[0]["record_shapes"] is True
    assert profile_kwargs[0]["profile_memory"] is True
    assert profile_kwargs[0]["with_stack"] is True
