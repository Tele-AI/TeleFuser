"""Pipeline service with integrated security validation."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from types import ModuleType
from typing import Any

import torch.multiprocessing as mp

from telefuser.platforms import current_platform
from telefuser.service_types import TaskType
from telefuser.utils.logging import logger

from ..security.security_validator import (
    PipelineSecurityValidator,
    SecurityError,
    SecurityLevel,
)
from .config import ServerConfig, server_config
from .pipeline_contract import load_pipeline_contract
from .pipeline_loader import (
    PipelineValidationConfig,
    load_pipeline_module,
    unload_pipeline_module,
    validate_pipeline_file,
)
from .pipeline_runner import PipelineRunner

mp.set_start_method("spawn", force=True)

_BYTES_PER_MIB = 1024 * 1024


def _nonnegative_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _accelerator_available() -> bool:
    try:
        return bool(current_platform.is_accelerator_available())
    except Exception:
        return False


def _synchronize_accelerator() -> None:
    if not _accelerator_available():
        return
    try:
        current_platform.synchronize()
    except Exception:
        pass


def _reset_peak_memory_stats() -> None:
    if not _accelerator_available():
        return
    try:
        _synchronize_accelerator()
        current_platform.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_memory_mb() -> float | None:
    if not _accelerator_available():
        return None
    try:
        peak_bytes = current_platform.max_memory_allocated()
    except Exception:
        return None
    peak_memory_bytes = _nonnegative_finite_float(peak_bytes)
    return peak_memory_bytes / _BYTES_PER_MIB if peak_memory_bytes is not None else None


def _raw_runtime_metrics(raw: Any) -> tuple[float | None, float | None]:
    if not isinstance(raw, dict):
        return None, None
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    inference_time_s = _nonnegative_finite_float(raw.get("inference_time_s"))
    if inference_time_s is None:
        inference_time_s = _nonnegative_finite_float(metrics.get("inference_time_s"))
    peak_memory_mb = _nonnegative_finite_float(raw.get("peak_memory_mb"))
    if peak_memory_mb is None:
        peak_memory_mb = _nonnegative_finite_float(metrics.get("peak_memory_mb"))
    if peak_memory_mb is None:
        peak_memory_gb = _nonnegative_finite_float(raw.get("peak_memory_gb"))
        if peak_memory_gb is None:
            peak_memory_gb = _nonnegative_finite_float(metrics.get("peak_memory_gb"))
        if peak_memory_gb is not None:
            peak_memory_mb = peak_memory_gb * 1024.0
    return inference_time_s, peak_memory_mb


class PipelineService:
    """Pipeline service with security validation for pipeline configuration files.

    Security features:
    - Static AST analysis for dangerous operations
    - Import restriction and verification
    - Content pattern matching
    - Optional best-effort restricted-load validation
    - Configurable security levels
    """

    def __init__(
        self,
        security_level: SecurityLevel | None = None,
        *,
        config: ServerConfig | None = None,
    ) -> None:
        """Initialize PipelineService."""
        self.server_config = config or server_config
        self.is_running = False
        self.pipeline = None
        self.task: TaskType | None = None
        self.ppl_file: str | None = None
        self.parallelism: int | None = None

        self._module: ModuleType | None = None
        self._module_name: str | None = None
        self._runner: PipelineRunner | None = None
        self._contract = None
        self._declared_contract = False

        self.security_level = security_level or self.server_config.security_level
        self.security_validator = PipelineSecurityValidator(
            security_level=self.security_level,
            max_file_size=self.server_config.max_ppl_file_size,
        )
        self.validation_config = PipelineValidationConfig(
            allow_unsafe_pipelines=self.server_config.allow_unsafe_pipelines,
            strict_validation=self.server_config.strict_validation,
        )

        logger.info(f"PipelineService initialized with security_level={self.security_level.name}")

    def _load_pipeline_module(self, ppl_file: str) -> ModuleType:
        module, name = load_pipeline_module(ppl_file, prefix="telefuser_ppl")
        self._module_name = name
        return module

    def _validate_pipeline_file(self, ppl_file: str) -> bool:
        return validate_pipeline_file(
            ppl_file,
            self.security_level,
            self.security_validator,
            validation_config=self.validation_config,
        )

    def start_pipeline(
        self, ppl_file: str, parallelism: int, task: TaskType | str, skip_validation: bool = False
    ) -> bool:
        """Start the pipeline with security validation."""
        if self.is_running:
            logger.warning("Distributed inference service is already running")
            return True

        try:
            if not skip_validation:
                self._validate_pipeline_file(ppl_file)
            else:
                logger.warning("Skipping security validation for pipeline file")

            self.task = task if isinstance(task, TaskType) else TaskType(task.lower())
            self.ppl_file = ppl_file
            self.parallelism = parallelism

            self._module = self._load_pipeline_module(ppl_file)
            self._contract, self._declared_contract = load_pipeline_contract(
                self._module,
                ppl_file=ppl_file,
                default_task=task,
            )

            if task not in self._contract.supported_tasks:
                raise RuntimeError(
                    f"Pipeline contract for {self._contract.pipeline_name} does not declare support for task '{task}'"
                )

            get_pipeline_name = self._contract.entrypoints.get_pipeline
            run_with_file_name = self._contract.entrypoints.run_with_file
            if not hasattr(self._module, get_pipeline_name):
                raise RuntimeError(f"Pipeline file must define {get_pipeline_name}(parallelism=...)")
            if not hasattr(self._module, run_with_file_name):
                raise RuntimeError(f"Pipeline file must define {run_with_file_name}(...) for service execution")

            get_pipeline = getattr(self._module, get_pipeline_name)
            run_with_file = getattr(self._module, run_with_file_name)

            self.pipeline = get_pipeline(parallelism=parallelism)
            self._runner = PipelineRunner(pipeline=self.pipeline, run_with_file=run_with_file, module=self._module)
            self.is_running = True

            logger.info(f"Pipeline service started with security_level={self.security_level.name}")
            self._log_contract_startup_summary()
            return True

        except SecurityError as e:
            logger.error(f"Security validation failed: {e}")
            return False
        except Exception as e:
            logger.exception(f"Error occurred while starting distributed inference service: {str(e)}")
            asyncio.run(self.aclose())
            return False

    async def aclose(self) -> None:
        """Async close and cleanup resources."""
        if not self.is_running:
            return

        try:
            if self._runner is not None:
                await self._runner.shutdown()
        except Exception as e:
            logger.warning(f"Error during pipeline shutdown: {e}")
        finally:
            self._runner = None
            unload_pipeline_module(self._module_name)
            self._module = None
            self._module_name = None
            self._contract = None
            self._declared_contract = False

            if self.pipeline is not None:
                try:
                    del self.pipeline
                except Exception:
                    pass
                self.pipeline = None

            self.is_running = False

    def stop_pipeline_inference(self) -> None:
        """Stop the pipeline inference service (sync wrapper)."""
        try:
            asyncio.run(self.aclose())
        except RuntimeError:
            # If called from within an event loop, schedule cleanup and return.
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.aclose())
            except Exception:
                self.is_running = False

    async def run_task_with_stop_event(
        self,
        task_data: dict[str, Any],
        stop_event: threading.Event,
        timeout_s: float | None = None,
        output_root: str | None = None,
    ) -> dict[str, Any]:
        """Run a task and return a normalized result dict."""
        if not self.is_running or self.pipeline is None or self._runner is None:
            raise RuntimeError("Pipeline service is not started")

        timeout_s = timeout_s if timeout_s is not None else float(self.server_config.task_timeout)

        _reset_peak_memory_stats()
        inference_started_at = time.perf_counter()
        result = await self._runner.run(
            task_data=task_data,
            stop_event=stop_event,
            timeout_s=timeout_s,
            output_root=output_root,
        )
        _synchronize_accelerator()
        measured_inference_time_s = time.perf_counter() - inference_started_at
        raw_inference_time_s, raw_peak_memory_mb = _raw_runtime_metrics(result.raw)

        return {
            "task_id": task_data.get("task_id", ""),
            "status": result.status.value,
            "output_path": result.output_path or "",
            "message": result.message,
            "raw": result.raw,
            "inference_time_s": (
                raw_inference_time_s if raw_inference_time_s is not None else measured_inference_time_s
            ),
            "peak_memory_mb": raw_peak_memory_mb if raw_peak_memory_mb is not None else _peak_memory_mb(),
        }

    def server_metadata(self) -> dict:
        """Get server metadata."""
        metadata = {
            "pipeline_file": self.ppl_file,
            "parallelism": self.parallelism,
            "task": self.task.value if self.task else None,
            "security_level": self.security_level.name if self.security_level else "NONE",
            "runner": "PipelineRunner",
            "declared_pipeline_contract": self._declared_contract,
        }
        if self._contract is not None:
            metadata.update(self._contract.to_metadata())
        return metadata

    def supported_tasks(self) -> tuple[str, ...]:
        """Return tasks declared by the loaded pipeline contract."""
        if self._contract is None:
            return tuple()
        return self._contract.supported_tasks

    def get_task_contract(self, task: str):
        """Return the task-level contract for a declared task, if available."""
        if self._contract is None:
            return None
        return self._contract.get_task_contract(task)

    def _log_contract_startup_summary(self) -> None:
        """Log a concise startup summary of the active pipeline contract."""
        if self._contract is None:
            return

        pipeline_summary = {
            "pipeline_name": self._contract.pipeline_name,
            "declared_pipeline_contract": self._declared_contract,
            "supported_tasks": list(self._contract.supported_tasks),
            "supported_media_types": list(self._contract.supported_media_types),
            "execution_mode": self._contract.execution_mode,
            "effective_max_concurrent_tasks": self._contract.effective_max_concurrent_tasks,
            "entrypoints": {
                "get_pipeline": self._contract.entrypoints.get_pipeline,
                "run_with_file": self._contract.entrypoints.run_with_file,
            },
            "metadata_endpoint": "/v1/service/metadata",
        }
        logger.info(
            "Pipeline contract startup summary: "
            f"{json.dumps(pipeline_summary, ensure_ascii=True, sort_keys=True, default=str)}"
        )

        for task in self._contract.supported_tasks:
            task_contract = self._contract.get_task_contract(task)
            if task_contract is None:
                continue

            task_summary = {
                "task": task,
                "media_type": task_contract.media_type,
                "required_inputs": list(task_contract.required_inputs),
                "optional_inputs": list(task_contract.optional_inputs),
                "parameters": {
                    name: {
                        "type": parameter.type,
                        "required": parameter.required,
                        "default": parameter.default,
                        "enum": list(parameter.enum),
                    }
                    for name, parameter in task_contract.parameters.items()
                    if parameter.exposed
                },
                "metadata_endpoint": "/v1/service/metadata",
            }
            logger.info(
                "Pipeline task startup summary: "
                f"{json.dumps(task_summary, ensure_ascii=True, sort_keys=True, default=str)}"
            )

    def __del__(self) -> None:
        """Destructor - attempts cleanup but should not be relied upon."""
        self._cleanup_pipeline()

    def _cleanup_pipeline(self) -> None:
        """Safely clean up pipeline resources."""
        if getattr(self, "pipeline", None) is not None:
            try:
                del self.pipeline
                self.pipeline = None
            except Exception:
                pass

    def close(self) -> None:
        """Explicitly close and cleanup resources."""
        self.stop_pipeline_inference()

    def __enter__(self) -> PipelineService:
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """Context manager exit - ensures cleanup."""
        self.close()
        return False  # Don't suppress exceptions


class SecurePipelineService(PipelineService):
    """Pre-configured secure pipeline service with strict validation."""

    def __init__(self) -> None:
        super().__init__(security_level=SecurityLevel.STRICT)

    def start_pipeline(self, ppl_file: str, parallelism: int, task: TaskType | str, **kwargs) -> bool:
        """Start pipeline with strict validation (skip_validation not allowed)."""
        if kwargs.get("skip_validation"):
            raise SecurityError("SecurePipelineService does not allow skip_validation")
        return super().start_pipeline(ppl_file, parallelism, task, skip_validation=False)
