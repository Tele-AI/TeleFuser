"""Reserve distinct HCCL TCP port blocks for local multiprocessing groups."""

from __future__ import annotations

import errno
import os
import socket
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO

_BLOCK_SIZE = 16
_LAST_BASE = 65536 - _BLOCK_SIZE


def reserve_hccl_ports() -> tuple[int, BinaryIO]:
    """Hold the returned lock until all ranks exit. Kernel locks survive parent concurrency."""
    import fcntl

    value = os.environ.get("HCCL_IF_BASE_PORT", "60000")
    try:
        start = int(value)
        if not 1024 <= start <= _LAST_BASE:
            raise ValueError
    except ValueError:
        raise ValueError(f"HCCL_IF_BASE_PORT must be between 1024 and {_LAST_BASE}, got {value!r}") from None

    # Use common block boundaries even when two launchers specify different starting ports.
    start += -start % _BLOCK_SIZE
    for base in range(start, _LAST_BASE + 1, _BLOCK_SIZE):
        path = Path(tempfile.gettempdir()) / f"telefuser-hccl-{base}.lock"
        # Read-only descriptors allow launchers with different UIDs to lock an existing file.
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o644)
        except FileExistsError:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        lock = os.fdopen(fd, "rb")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with ExitStack() as probes:
                for port in range(base, base + _BLOCK_SIZE):
                    probe = probes.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
                    probe.bind(("", port))
            return base, lock
        except OSError as exc:
            lock.close()
            if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EADDRINUSE):
                raise
        except BaseException:
            lock.close()
            raise
    raise RuntimeError(f"No free {_BLOCK_SIZE}-port HCCL block at or above {start} (last base {_LAST_BASE})")
