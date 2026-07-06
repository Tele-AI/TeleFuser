from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from telefuser.entrypoints.cli.main import main
from telefuser.service.core.config import ServerConfig
from telefuser.service.core.container import ServiceContainer


def test_telefuser_pyproject_does_not_vendor_cacheseek_dependencies() -> None:
    pyproject = Path(__file__).resolve().parents[3] / "pyproject.toml"
    text = pyproject.read_text()
    optional_deps = text.split("[project.optional-dependencies]", 1)[1]
    dependency_specs = re.findall(r'^\s*"([^"]+)"', text, flags=re.MULTILINE)

    assert not re.search(r"^cache\s*=", optional_deps, flags=re.MULTILINE)
    for dependency in ("cacheseek", "faiss-cpu", "qwen-vl-utils", "qwen_vl_utils", "sentencepiece", "protobuf"):
        assert all(not spec.startswith(dependency) for spec in dependency_specs)


def test_serve_cli_exposes_latent_cache_options() -> None:
    result = CliRunner().invoke(main, ["serve", "--help"])

    assert result.exit_code == 0
    assert "--enable-latent-cache" in result.output
    assert "--disable-latent-cache" in result.output
    assert "--cache-mode" in result.output


def test_serve_cli_forwards_latent_cache_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("telefuser.service.main.run_server", fake_run_server)

    result = CliRunner().invoke(
        main,
        [
            "serve",
            "pipeline.py",
            "--skip-validation",
            "--enable-latent-cache",
            "--cache-mode",
            "read_only",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "pipe_path": "pipeline.py",
            "task": "i2v",
            "port": 8000,
            "host": "127.0.0.1",
            "cache_dir": "work_dirs/server_cache",
            "parallelism": 1,
            "num_replicas": 1,
            "enable_latent_cache": True,
            "cache_mode": "read_only",
        }
    ]


def test_serve_cli_leaves_latent_cache_unset_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("telefuser.service.main.run_server", fake_run_server)

    result = CliRunner().invoke(main, ["serve", "pipeline.py", "--skip-validation"])

    assert result.exit_code == 0
    assert calls[0]["enable_latent_cache"] is None
    assert calls[0]["cache_mode"] is None


def test_serve_cli_can_disable_pipeline_latent_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    def fake_run_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("telefuser.service.main.run_server", fake_run_server)

    result = CliRunner().invoke(main, ["serve", "pipeline.py", "--skip-validation", "--disable-latent-cache"])

    assert result.exit_code == 0
    assert calls[0]["enable_latent_cache"] is False


def test_cache_disabled_does_not_import_cacheseek(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocked = {"imported": False}

    class BlockCacheseek:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "cacheseek" or fullname.startswith("cacheseek."):
                blocked["imported"] = True
                raise AssertionError("cacheseek should not be imported when latent cache is disabled")
            return None

    finder = BlockCacheseek()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])

    container = ServiceContainer.create(config=ServerConfig(enable_latent_cache=False), cache_dir=tmp_path)

    assert container.initialize_cache_service(pipe_path="pipeline.py") is None
    assert blocked["imported"] is False


def test_pipeline_cache_config_can_disable_cacheseek_import(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    blocked = {"imported": False}
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("CACHE_CONFIG = {'enable_latent_cache': False}\n")

    class BlockCacheseek:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "cacheseek" or fullname.startswith("cacheseek."):
                blocked["imported"] = True
                raise AssertionError("cacheseek should not be imported when CACHE_CONFIG disables latent cache")
            return None

    monkeypatch.setattr(sys, "meta_path", [BlockCacheseek(), *sys.meta_path])
    container = ServiceContainer.create(config=ServerConfig(enable_latent_cache=None), cache_dir=tmp_path)

    assert container.initialize_cache_service(pipe_path=str(pipeline)) is None
    assert blocked["imported"] is False


def test_pipeline_cache_config_can_enable_cacheseek_without_cli_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("CACHE_CONFIG = {'enable_latent_cache': True, 'cache_mode': 'write_only'}\n")
    calls: list[dict] = []
    factory_module = types.ModuleType("cacheseek.adapters.telefuser.cache_factory")

    class FakeFactory:
        @staticmethod
        def create_cache_service(*, ppl_file: str, enable_latent_cache: bool | None, cache_mode: str | None = None):
            calls.append(
                {
                    "ppl_file": ppl_file,
                    "enable_latent_cache": enable_latent_cache,
                    "cache_mode": cache_mode,
                }
            )
            return "service", "adapter"

    factory_module.CacheServiceFactory = FakeFactory
    monkeypatch.setitem(sys.modules, "cacheseek", types.ModuleType("cacheseek"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters", types.ModuleType("cacheseek.adapters"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters.telefuser", types.ModuleType("cacheseek.adapters.telefuser"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters.telefuser.cache_factory", factory_module)
    container = ServiceContainer.create(config=ServerConfig(enable_latent_cache=None), cache_dir=tmp_path)

    assert container.initialize_cache_service(pipe_path=str(pipeline)) == "service"
    assert calls == [
        {
            "ppl_file": str(pipeline),
            "enable_latent_cache": None,
            "cache_mode": None,
        }
    ]


def test_cli_cache_mode_overrides_pipeline_cache_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("CACHE_CONFIG = {'enable_latent_cache': True, 'cache_mode': 'write_only'}\n")
    calls: list[dict] = []
    factory_module = types.ModuleType("cacheseek.adapters.telefuser.cache_factory")

    class FakeFactory:
        @staticmethod
        def create_cache_service(*, ppl_file: str, enable_latent_cache: bool | None, cache_mode: str | None = None):
            calls.append({"enable_latent_cache": enable_latent_cache, "cache_mode": cache_mode})
            return "service", "adapter"

    factory_module.CacheServiceFactory = FakeFactory
    monkeypatch.setitem(sys.modules, "cacheseek", types.ModuleType("cacheseek"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters", types.ModuleType("cacheseek.adapters"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters.telefuser", types.ModuleType("cacheseek.adapters.telefuser"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters.telefuser.cache_factory", factory_module)
    container = ServiceContainer.create(
        config=ServerConfig(enable_latent_cache=None, cache_mode="read_only"),
        cache_dir=tmp_path,
    )

    assert container.initialize_cache_service(pipe_path=str(pipeline)) == "service"
    assert calls == [{"enable_latent_cache": None, "cache_mode": "read_only"}]


def test_cache_enabled_missing_cacheseek_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delitem(sys.modules, "cacheseek", raising=False)

    class MissingCacheseek:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "cacheseek" or fullname.startswith("cacheseek."):
                raise ModuleNotFoundError("No module named 'cacheseek'")
            return None

    monkeypatch.setattr(sys, "meta_path", [MissingCacheseek(), *sys.meta_path])
    container = ServiceContainer.create(config=ServerConfig(enable_latent_cache=True), cache_dir=tmp_path)

    with pytest.raises(RuntimeError, match=r"python -m pip install /path/to/CacheSeek"):
        container.initialize_cache_service(pipe_path="pipeline.py")


def test_cache_enabled_factory_none_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    factory_module = types.ModuleType("cacheseek.adapters.telefuser.cache_factory")

    class FakeFactory:
        @staticmethod
        def create_cache_service(*, ppl_file: str, enable_latent_cache: bool, cache_mode: str | None = None):
            return None

    factory_module.CacheServiceFactory = FakeFactory
    monkeypatch.setitem(sys.modules, "cacheseek", types.ModuleType("cacheseek"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters", types.ModuleType("cacheseek.adapters"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters.telefuser", types.ModuleType("cacheseek.adapters.telefuser"))
    monkeypatch.setitem(sys.modules, "cacheseek.adapters.telefuser.cache_factory", factory_module)

    container = ServiceContainer.create(
        config=ServerConfig(enable_latent_cache=True, cache_mode="write_only"),
        cache_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="failed to initialize"):
        container.initialize_cache_service(pipe_path="pipeline.py")
