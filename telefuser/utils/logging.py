"""Industrial-grade logging system for TeleFuser.

This module provides a configurable logging system based on loguru, designed for
production deployment scenarios with features like:
- Structured JSON logging for machine parsing
- Rotating file handlers with compression
- Multiple output sinks (console, file, syslog)
- Environment-based configuration
- Log level filtering per module
- Async logging support for high-performance scenarios
- Multi-process logging with PID and Rank info
- Distributed tracing with trace_id/span_id support

Basic Usage:
    >>> from telefuser.utils.logging import logger
    >>> logger.info("Application started")

Configuration via environment variables:
    TELEFUSER_LOG_LEVEL=INFO
    TELEFUSER_LOG_FORMAT=json
    TELEFUSER_LOG_FILE=/var/log/telefuser/app.log
    TELEFUSER_LOG_ROTATION=1 day
    TELEFUSER_LOG_RETENTION=30 days

Programmatic Configuration:
    >>> from telefuser.utils.logging import configure_logging, LogConfig
    >>> configure_logging(LogConfig(level="DEBUG", log_file="/tmp/debug.log"))
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import platform
import socket
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

# Context variables for distributed tracing
_log_context: ContextVar[dict[str, Any]] = ContextVar("log_context", default={})
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
_span_id: ContextVar[str | None] = ContextVar("span_id", default=None)


class LogFormat(Enum):
    """Log output format options."""

    SIMPLE = auto()  # Simple text format
    DETAILED = auto()  # Detailed text with module and line info
    JSON = auto()  # Structured JSON format for machine parsing
    STRUCTURED = auto()  # Key-value structured format


class LogSink(Enum):
    """Log output destinations."""

    CONSOLE = auto()
    FILE = auto()
    SYSLOG = auto()
    BOTH = auto()  # Console + File


@dataclass
class LogConfig:
    """Configuration for the logging system.

    Attributes:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format (simple, detailed, json, structured)
        sink: Where to send logs (console, file, syslog, both)
        log_file: Path to log file (required if sink includes file)
        rotation: When to rotate files (e.g., "100 MB", "1 day", "1 week")
        retention: How long to keep files (e.g., "30 days", "10 files")
        compression: Compression format for rotated files ("gz", "bz2", "xz", None)
        enqueue: Use async logging (recommended for multi-threaded/production)
        backtrace: Include backtrace in error logs
        diagnose: Include diagnostic info in error logs
        serialize: Serialize log record as JSON
        catch: Catch errors during logging to prevent crashes
        colorize: Use colors in console output
        console_stderr: Use stderr for ERROR and above in console
        filter_modules: Dict of module names to log levels for fine-grained control
        extra_fields: Extra fields to include in every log record
        multiprocess: How to handle multi-process logging ("single", "per_process", "per_rank")
        syslog_host: Syslog server host (for SYSLOG sink)
        syslog_port: Syslog server port (for SYSLOG sink)
        syslog_facility: Syslog facility code
        json_include_extra: Whether to include extra fields in JSON output
    """

    level: str = "INFO"
    format: LogFormat = LogFormat.DETAILED
    sink: LogSink = LogSink.CONSOLE
    log_file: str | None = None
    rotation: str = "100 MB"
    retention: str = "30 days"
    compression: str | None = "gz"
    enqueue: bool = True
    backtrace: bool = True
    diagnose: bool = True
    serialize: bool = False
    catch: bool = True
    colorize: bool = True
    console_stderr: bool = True
    filter_modules: dict[str, str] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    multiprocess: str = "single"  # "single", "per_process", "per_rank"
    syslog_host: str = "localhost"
    syslog_port: int = 514
    syslog_facility: str = "user"
    json_include_extra: bool = True


def _get_process_id() -> str:
    """Get current process ID."""
    return str(os.getpid())


def _get_rank() -> str:
    """Get distributed training rank from environment."""
    # Check common environment variables for distributed training
    for env_var in ["RANK", "LOCAL_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK"]:
        rank = os.getenv(env_var)
        if rank is not None:
            return rank
    return "0"


def _generate_trace_id() -> str:
    """Generate a unique trace ID for distributed tracing."""
    return uuid.uuid4().hex[:16]


def _generate_span_id() -> str:
    """Generate a unique span ID for distributed tracing."""
    return uuid.uuid4().hex[:8]


def _resolve_log_file_path(log_file: str, multiprocess: str) -> str:
    """Resolve log file path with multi-process placeholders.

    Supports:
        - {pid}: Process ID
        - {rank}: Distributed training rank
        - {time}: Current timestamp

    Args:
        log_file: Original log file path with optional placeholders
        multiprocess: Multi-process mode ("single", "per_process", "per_rank")

    Returns:
        Resolved log file path
    """
    path = log_file

    # Replace placeholders
    if "{pid}" in path or multiprocess == "per_process":
        if "{pid}" not in path:
            # Auto-add pid before extension
            stem = Path(path).stem
            suffix = Path(path).suffix
            parent = str(Path(path).parent)
            path = os.path.join(parent, f"{stem}_{{pid}}{suffix}")
        path = path.replace("{pid}", _get_process_id())

    if "{rank}" in path or multiprocess == "per_rank":
        if "{rank}" not in path:
            # Auto-add rank before extension
            stem = Path(path).stem
            suffix = Path(path).suffix
            parent = str(Path(path).parent)
            path = os.path.join(parent, f"{stem}_rank{{rank}}{suffix}")
        path = path.replace("{rank}", _get_rank())

    if "{time}" in path:
        path = path.replace("{time}", datetime.now().strftime("%Y%m%d_%H%M%S"))

    return path


def _get_format_string(fmt: LogFormat) -> str:
    """Get log format string based on format type.

    Note: PID, Rank, TraceID, SpanID are injected dynamically via extra fields
    by the filter function.
    """
    formats = {
        LogFormat.SIMPLE: ("{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"),
        LogFormat.DETAILED: (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
            "[pid:{extra[pid]} rank:{extra[rank]}] "
            "{name}:{function}:{line} | {message}"
        ),
        LogFormat.JSON: "{message}",  # JSON handled by loguru's serialize
        LogFormat.STRUCTURED: (
            "time={time:YYYY-MM-DD HH:mm:ss.SSS} level={level} "
            "pid={extra[pid]} rank={extra[rank]} "
            "trace_id={extra[trace_id]} span_id={extra[span_id]} "
            "module={name} func={function} line={line} msg={message}"
        ),
    }
    return formats.get(fmt, formats[LogFormat.DETAILED])


def _load_config_from_env() -> LogConfig:
    """Load configuration from environment variables."""
    fmt_map = {
        "simple": LogFormat.SIMPLE,
        "detailed": LogFormat.DETAILED,
        "json": LogFormat.JSON,
        "structured": LogFormat.STRUCTURED,
    }
    sink_map = {
        "console": LogSink.CONSOLE,
        "file": LogSink.FILE,
        "syslog": LogSink.SYSLOG,
        "both": LogSink.BOTH,
    }

    # Parse extra fields from JSON string
    extra_fields_str = os.getenv("TELEFUSER_LOG_EXTRA_FIELDS", "{}")
    try:
        extra_fields = json.loads(extra_fields_str)
    except json.JSONDecodeError:
        extra_fields = {}

    # Parse filter modules from JSON string
    filter_modules_str = os.getenv("TELEFUSER_LOG_FILTER_MODULES", "{}")
    try:
        filter_modules = json.loads(filter_modules_str)
    except json.JSONDecodeError:
        filter_modules = {}

    return LogConfig(
        level=os.getenv("TELEFUSER_LOG_LEVEL", "INFO").upper(),
        format=fmt_map.get(os.getenv("TELEFUSER_LOG_FORMAT", "detailed").lower(), LogFormat.DETAILED),
        sink=sink_map.get(os.getenv("TELEFUSER_LOG_SINK", "console").lower(), LogSink.CONSOLE),
        log_file=os.getenv("TELEFUSER_LOG_FILE"),
        rotation=os.getenv("TELEFUSER_LOG_ROTATION", "100 MB"),
        retention=os.getenv("TELEFUSER_LOG_RETENTION", "30 days"),
        compression=os.getenv("TELEFUSER_LOG_COMPRESSION", "gz"),
        enqueue=os.getenv("TELEFUSER_LOG_ENQUEUE", "true").lower() == "true",
        backtrace=os.getenv("TELEFUSER_LOG_BACKTRACE", "true").lower() == "true",
        diagnose=os.getenv("TELEFUSER_LOG_DIAGNOSE", "true").lower() == "true",
        serialize=os.getenv("TELEFUSER_LOG_SERIALIZE", "false").lower() == "true",
        filter_modules=filter_modules,
        extra_fields=extra_fields,
        multiprocess=os.getenv("TELEFUSER_LOG_MULTIPROCESS", "single").lower(),
        syslog_host=os.getenv("TELEFUSER_LOG_SYSLOG_HOST", "localhost"),
        syslog_port=int(os.getenv("TELEFUSER_LOG_SYSLOG_PORT", "514")),
        syslog_facility=os.getenv("TELEFUSER_LOG_SYSLOG_FACILITY", "user"),
        json_include_extra=os.getenv("TELEFUSER_LOG_JSON_INCLUDE_EXTRA", "true").lower() == "true",
    )


def _create_filter(config: LogConfig) -> Callable[[Record], bool]:
    """Create a filter function based on config.

    This also injects dynamic fields (pid, rank, trace_id, etc.) into the
    record's extra dict before formatting, ensuring they're available for
    format strings like {extra[pid]}.
    """

    def filter_func(record: Record) -> bool:
        # Inject dynamic fields into extra BEFORE formatting
        record["extra"]["pid"] = _get_process_id()
        record["extra"]["rank"] = _get_rank()
        record["extra"]["trace_id"] = _trace_id.get() or "-"
        record["extra"]["span_id"] = _span_id.get() or "-"

        # Merge context variables
        context = _log_context.get()
        if context:
            record["extra"].update(context)

        # Merge base extra fields from config
        for key, value in config.extra_fields.items():
            record["extra"].setdefault(key, value)

        # Check module-specific filters
        module_name = record.get("name", "") or ""
        for mod_prefix, mod_level in config.filter_modules.items():
            if module_name.startswith(mod_prefix):
                level_no = logger.level(mod_level).no
                return record["level"].no >= level_no
        return True

    return filter_func


_SEP_WIDTH = 50


def _get_startup_info() -> dict[str, Any]:
    """Get system startup information for logging initialization.

    Returns:
        Dict with Python, PyTorch, CUDA, and system info.
    """
    info: dict[str, Any] = {}

    # Python version
    info["Python"] = platform.python_version()

    # System info
    info["System"] = f"{platform.system()} {platform.release()}"
    info["Arch"] = platform.machine()

    # PyTorch and CUDA info (lazy import to avoid issues if torch not installed)
    try:
        import torch

        info["PyTorch"] = torch.__version__
        info["CUDA"] = torch.version.cuda or "N/A"

        if torch.cuda.is_available():
            info["GPU"] = torch.cuda.get_device_name(0)
            info["GPU_Count"] = torch.cuda.device_count()
        else:
            info["GPU"] = "N/A"
            info["GPU_Count"] = 0
    except ImportError:
        info["PyTorch"] = "not installed"
        info["CUDA"] = "N/A"
        info["GPU"] = "N/A"
        info["GPU_Count"] = 0
    except Exception:
        info["PyTorch"] = "error"
        info["CUDA"] = "N/A"
        info["GPU"] = "N/A"
        info["GPU_Count"] = 0

    return info


def _print_log_config(config: LogConfig, startup_info: dict[str, Any]) -> None:
    """Print logging configuration on startup with minimalist style."""
    # ANSI color codes
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    SEP = f"{DIM}─{'─' * _SEP_WIDTH}─{RESET}"

    lines = [
        SEP,
        f"{BOLD}{CYAN}TeleFuser Logging{RESET}  {DIM}initialized{RESET}",
        f"  {GREEN}●{RESET} {config.level}  {BLUE}●{RESET} {config.format.name}  {YELLOW}●{RESET} {config.sink.name}",
    ]

    if config.sink in (LogSink.FILE, LogSink.BOTH) and config.log_file:
        lines.append(f"  {DIM}File:{RESET} {config.log_file}  {DIM}Rotation:{RESET} {config.rotation}")

    if config.sink == LogSink.SYSLOG:
        lines.append(f"  {DIM}Syslog:{RESET} {config.syslog_host}:{config.syslog_port}")

    lines.append(
        f"  {DIM}PID:{RESET} {_get_process_id()}  {DIM}Rank:{RESET} {_get_rank()}  {DIM}Async:{RESET} {config.enqueue}"
    )

    # System info section
    if startup_info:
        sys_info_keys = ["Python", "PyTorch", "CUDA", "System", "Arch", "GPU", "GPU_Count"]
        sys_parts = []
        for k in sys_info_keys:
            if k in startup_info:
                sys_parts.append(f"{DIM}{k}:{RESET} {startup_info[k]}")
        if sys_parts:
            lines.append(f"  {'  '.join(sys_parts[:3])}")
            if len(sys_parts) > 3:
                lines.append(f"  {'  '.join(sys_parts[3:])}")

    lines.append(SEP)

    print("\n".join(lines), file=sys.stderr)


class SyslogHandler:
    """Syslog handler for loguru using UDP."""

    def __init__(self, host: str, port: int, facility: str):
        self.host = host
        self.port = port
        self.facility = facility
        self._socket: socket.socket | None = None
        self._connect()

    def _connect(self) -> None:
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.settimeout(1.0)
        except Exception:
            self._socket = None

    def write(self, message: str) -> None:
        if self._socket is None:
            return
        try:
            # RFC 5424 format: <priority>version timestamp hostname app-name procid msgid message
            priority = 8 + {"user": 1, "local0": 16, "daemon": 3}.get(self.facility, 1)
            hostname = socket.gethostname()
            pid = os.getpid()
            formatted = f"<{priority}>1 {datetime.now().isoformat()} {hostname} telefuser {pid} - {message}"
            self._socket.sendto(formatted.encode("utf-8"), (self.host, self.port))
        except Exception:
            pass

    def close(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None

    def __del__(self) -> None:
        self.close()


def configure_logging(config: LogConfig | None = None, startup_info: dict[str, Any] | None = None) -> None:
    """Configure logging with the given configuration.

    Args:
        config: LogConfig instance. If None, loads from environment variables.
        startup_info: Optional dictionary of startup information to display.

    Example:
        >>> from telefuser.utils.logging import configure_logging, LogConfig, LogFormat
        >>> configure_logging(LogConfig(level="DEBUG", format=LogFormat.JSON))
    """
    if config is None:
        config = _load_config_from_env()

    # Remove all existing handlers
    logger.remove()

    # Create filter function that also injects dynamic fields
    filter_func = _create_filter(config)

    # Determine if we need JSON serialization
    use_json = config.format == LogFormat.JSON
    use_serialize = config.serialize or use_json

    # Console sink
    if config.sink in (LogSink.CONSOLE, LogSink.BOTH):
        logger.add(
            sys.stdout,
            level=config.level,
            format=_get_format_string(config.format),
            colorize=config.colorize,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            serialize=use_serialize,
            catch=config.catch,
            filter=lambda record: filter_func(record) and (record["level"].no < 40 or not config.console_stderr),
        )

        if config.console_stderr:
            logger.add(
                sys.stderr,
                level="ERROR",
                format=_get_format_string(config.format),
                colorize=config.colorize,
                enqueue=config.enqueue,
                backtrace=config.backtrace,
                diagnose=config.diagnose,
                serialize=use_serialize,
                catch=config.catch,
                filter=filter_func,
            )

    # File sink
    if config.sink in (LogSink.FILE, LogSink.BOTH) and config.log_file:
        # Resolve multi-process log file path
        resolved_log_file = _resolve_log_file_path(config.log_file, config.multiprocess)
        log_path = Path(resolved_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            level=config.level,
            format=_get_format_string(config.format),
            rotation=config.rotation,
            retention=config.retention,
            compression=config.compression,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            serialize=use_serialize,
            catch=config.catch,
            filter=filter_func,
        )
        # Update config with resolved path for display
        config.log_file = resolved_log_file

    # Syslog sink
    if config.sink == LogSink.SYSLOG:
        try:
            syslog_handler = SyslogHandler(config.syslog_host, config.syslog_port, config.syslog_facility)
            syslog_format = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "[pid:{extra[pid]} rank:{extra[rank]}] {name}:{function}:{line} | {message}"
            )
            logger.add(
                syslog_handler.write,
                level=config.level,
                format=syslog_format,
                enqueue=config.enqueue,
                catch=config.catch,
                filter=filter_func,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize syslog handler: {e}", file=sys.stderr)

    # Print configuration on startup with system info
    auto_startup_info = _get_startup_info()
    if startup_info:
        auto_startup_info.update(startup_info)
    _print_log_config(config, auto_startup_info)


class LoggingContext:
    """Context manager for temporary log context.

    Values set here will be automatically included in all log messages
    within the context via the filter function.

    Example:
        >>> from telefuser.utils.logging import logger, LoggingContext
        >>> with LoggingContext(request_id="req-123", user_id="user-456"):
        ...     logger.info("Processing request")  # Auto-includes context
    """

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.token: Any = None

    def __enter__(self) -> "LoggingContext":
        current = _log_context.get()
        new_context = {**current, **self.kwargs}
        self.token = _log_context.set(new_context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb  # Unused but required by protocol
        if self.token is not None:
            _log_context.reset(self.token)

    async def __aenter__(self) -> "LoggingContext":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


class TracingContext:
    """Context manager for distributed tracing.

    Sets trace_id and span_id that will be included in all log messages.
    Useful for correlating logs across services in distributed systems.

    Example:
        >>> from telefuser.utils.logging import logger, TracingContext
        >>> with TracingContext(trace_id="abc123", span_id="def456"):
        ...     logger.info("Processing")  # Includes trace_id and span_id
    """

    def __init__(
        self,
        trace_id: str | None = None,
        span_id: str | None = None,
        generate: bool = True,
    ):
        self.trace_id = trace_id or (_generate_trace_id() if generate else None)
        self.span_id = span_id or (_generate_span_id() if generate else None)
        self.trace_token: Any = None
        self.span_token: Any = None

    def __enter__(self) -> "TracingContext":
        if self.trace_id:
            self.trace_token = _trace_id.set(self.trace_id)
        if self.span_id:
            self.span_token = _span_id.set(self.span_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        del exc_type, exc_val, exc_tb  # Unused but required by protocol
        if self.trace_token is not None:
            _trace_id.reset(self.trace_token)
        if self.span_token is not None:
            _span_id.reset(self.span_token)

    async def __aenter__(self) -> "TracingContext":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def get_trace_id() -> str | None:
    """Get the current trace ID from context."""
    return _trace_id.get()


def get_span_id() -> str | None:
    """Get the current span ID from context."""
    return _span_id.get()


def set_trace_id(trace_id: str) -> None:
    """Set the trace ID in the current context."""
    _trace_id.set(trace_id)


def set_span_id(span_id: str) -> None:
    """Set the span ID in the current context."""
    _span_id.set(span_id)


def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time.

    Supports both synchronous and asynchronous functions.

    Example:
        >>> from telefuser.utils.logging import log_execution_time
        >>> @log_execution_time
        ... def my_function():
        ...     pass
        >>>
        >>> @log_execution_time
        ... async def my_async_function():
        ...     pass
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logger.debug(
                    f"Function {func.__name__} executed in {elapsed:.3f}s",
                    function=func.__name__,
                    duration=elapsed,
                )

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.time() - start
                logger.debug(
                    f"Function {func.__name__} executed in {elapsed:.3f}s",
                    function=func.__name__,
                    duration=elapsed,
                )

        return sync_wrapper


def log_exceptions(func: Callable) -> Callable:
    """Decorator to log exceptions with full context.

    Supports both synchronous and asynchronous functions.

    Example:
        >>> from telefuser.utils.logging import log_exceptions
        >>> @log_exceptions
        ... def risky_function():
        ...     raise ValueError("Something went wrong")
        >>>
        >>> @log_exceptions
        ... async def risky_async_function():
        ...     raise ValueError("Something went wrong")
    """
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in {func.__name__}: {e}",
                    function=func.__name__,
                    exception_type=type(e).__name__,
                )
                raise

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.exception(
                    f"Exception in {func.__name__}: {e}",
                    function=func.__name__,
                    exception_type=type(e).__name__,
                )
                raise

        return sync_wrapper


# Auto-configure on import
configure_logging()

# Export symbols
__all__ = [
    "logger",
    "configure_logging",
    "LogConfig",
    "LogFormat",
    "LogSink",
    "LoggingContext",
    "TracingContext",
    "log_execution_time",
    "log_exceptions",
    "get_trace_id",
    "get_span_id",
    "set_trace_id",
    "set_span_id",
]
