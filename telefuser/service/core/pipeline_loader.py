"""Shared pipeline module loading and security validation utilities.

Used by both PipelineService (request-response) and StreamPipelineService (streaming).
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from telefuser.utils.logging import logger

from ..security.security_validator import (
    PipelineSecurityValidator,
    SecurityError,
    SecurityLevel,
    validate_with_report,
)


@dataclass(frozen=True)
class PipelineValidationConfig:
    """Runtime validation behavior for dynamic pipeline loading."""

    allow_unsafe_pipelines: bool = False
    strict_validation: bool = True


def load_pipeline_module(ppl_file: str, prefix: str = "telefuser_ppl") -> tuple[ModuleType, str]:
    """Load a pipeline module from file, returning (module, module_name).

    The module_name is needed for cleanup via ``unload_pipeline_module``.
    """
    resolved = str(Path(ppl_file).expanduser().resolve())
    module_hash = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:12]
    module_name = f"{prefix}_{module_hash}"

    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {ppl_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, module_name


def unload_pipeline_module(module_name: str | None) -> None:
    """Remove a dynamically-loaded pipeline module from sys.modules."""
    if module_name and module_name in sys.modules:
        del sys.modules[module_name]


def validate_pipeline_file(
    ppl_file: str,
    security_level: SecurityLevel,
    security_validator: PipelineSecurityValidator,
    validation_config: PipelineValidationConfig | None = None,
) -> bool:
    """Validate a pipeline file for security issues.

    Returns True if safe. Raises SecurityError on failure.
    """
    if security_level == SecurityLevel.NONE:
        logger.warning("Security validation is disabled (SecurityLevel.NONE)")
        return True
    validation_config = validation_config or PipelineValidationConfig()

    try:
        logger.info(f"Validating pipeline file: {ppl_file}")
        result = security_validator.validate_file(ppl_file)

        if result.is_safe:
            logger.info(f"Pipeline file '{ppl_file}' passed security validation")
            if result.warnings:
                logger.warning(f"  {len(result.warnings)} warnings found")
                for w in result.warnings[:3]:
                    logger.warning(f"    Line {w.line_number}: {w.description}")
            return True

        report = validate_with_report(ppl_file)
        logger.error(f"Security validation failed:\n{report}")

        critical_count = sum(1 for v in result.violations if v.severity == "critical")
        if critical_count > 0:
            raise SecurityError(f"Pipeline file contains {critical_count} critical security violations.")

        if validation_config.allow_unsafe_pipelines:
            logger.warning("Allowing unsafe pipeline (allow_unsafe_pipelines=True)")
            return True

        raise SecurityError(f"Pipeline file failed validation with {len(result.violations)} violations.")

    except SecurityError:
        raise
    except Exception as e:
        logger.error(f"Unexpected validation error: {e}")
        if validation_config.strict_validation:
            raise SecurityError(f"Validation failed: {e}")
        return True
