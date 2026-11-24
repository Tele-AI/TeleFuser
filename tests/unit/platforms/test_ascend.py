from __future__ import annotations

import builtins
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from telefuser import platforms
from telefuser.platforms.hccl import reserve_hccl_ports


def test_is_npu_available_handles_missing_driver_lib(monkeypatch) -> None:
    real_import = builtins.__import__

    def _fail_with_oserror(name, *args, **kwargs):
        if name == "torch_npu":
            raise OSError("libascend_hal.so: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fail_with_oserror)
    assert platforms._is_npu_available() is False


@pytest.mark.parametrize(("given", "expected"), [("npu", "npu:0"), ("npu:07", "npu:7"), ("rocm:1", "cuda:1")])
def test_normalize_without_vendor_module(given: str, expected: str) -> None:
    assert platforms.normalize_device_name(given) == expected


def test_explicit_devices_fail_before_starting_workers(monkeypatch) -> None:
    with pytest.raises(ValueError):
        platforms.normalize_device_name("npu:-1")
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    with pytest.raises(RuntimeError, match="torch_npu") as failure:
        platforms.resolve_device("npu:1")
    assert isinstance(failure.value.__cause__, ImportError)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(platforms, "current_platform", platforms.CpuPlatform())
    with pytest.raises(RuntimeError, match="detected cpu"):
        platforms.resolve_device("npu:1")
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(device_type="npu", device_name="npu", device_count=lambda: 2)
    )
    with pytest.raises(ValueError, match="2 visible"):
        platforms.resolve_device("npu:2")
    assert platforms.resolve_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("value", ["0", "65521", "bad"])
def test_invalid_hccl_base(monkeypatch, value: str) -> None:
    monkeypatch.setenv("HCCL_IF_BASE_PORT", value)
    with pytest.raises(ValueError, match="HCCL_IF_BASE_PORT"):
        reserve_hccl_ports()


def test_hccl_lease_across_parents_and_shutdown(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    monkeypatch.setenv("HCCL_IF_BASE_PORT", "61001")
    base, lock = reserve_hccl_ports()
    with lock:
        assert base % 16 == 0 and base >= 61001
        code = "from telefuser.platforms.hccl import reserve_hccl_ports; base, lock = reserve_hccl_ports(); print(base)"
        other = subprocess.check_output(
            [sys.executable, "-c", code], text=True, env={**os.environ, "TMPDIR": str(tmp_path)}
        )
        assert int(other.strip()) >= base + 16
    for _ in range(8):
        recycled, lock = reserve_hccl_ports()
        with lock:
            assert recycled == base
    with socket.socket() as occupied:
        occupied.bind(("", base + 7))
        different, lock = reserve_hccl_ports()
        with lock:
            assert different >= base + 16
