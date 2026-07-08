from __future__ import annotations

from telefuser.service.security.security_validator import PipelineSecurityValidator, SecurityLevel


def test_sandbox_level_reports_restricted_load_not_runtime_isolation() -> None:
    validator = PipelineSecurityValidator(security_level=SecurityLevel.SANDBOX)

    result = validator.validate_source("raise RuntimeError('load failed')\n")

    assert result.is_safe is False
    assert result.violations
    assert "restricted-load validation" in result.violations[0].description
    assert "sandboxed load" not in result.violations[0].description
