"""TeleFuser CLI - Command line interface for server management."""

from __future__ import annotations

import click

from telefuser.core.config import TELEFUSER_LOGO
from telefuser.service.security.security_validator import SecurityLevel, validate_with_report
from telefuser.service_types import TaskType


@click.group()
def main():
    """TeleFuser CLI tool for multimodal model inference."""
    print(TELEFUSER_LOGO)
    pass


@main.command()
@click.argument("pipe_path")
@click.option(
    "--task",
    "-t",
    default="i2v",
    type=click.Choice(TaskType.values(), case_sensitive=False),
    help="Task type (t2v, i2v, fl2v, vc, t2i, i2i)",
)
@click.option("--port", "-p", default=8000, type=int, help="Server port")
@click.option("--host", default="127.0.0.1", type=str, help="Server host")
@click.option(
    "--cache-dir",
    "-c",
    default="work_dirs/server_cache",
    type=str,
    help="cache dir for input and output",
)
@click.option("--parallelism", "--gpu-num", "-g", type=int, default=1, help="Number of parallel workers for inference")
@click.option(
    "--security-level",
    type=click.Choice(["none", "basic", "strict", "sandbox"], case_sensitive=False),
    default="strict",
    help="Security validation level for pipeline file (default: strict)",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip security validation (not recommended for production)",
)
@click.option(
    "--validate-only",
    is_flag=True,
    default=False,
    help="Only validate the pipeline file without starting the server",
)
def serve(
    pipe_path: str,
    task: str,
    port: int,
    host: str,
    cache_dir: str,
    parallelism: int,
    security_level: str,
    skip_validation: bool,
    validate_only: bool,
) -> None:
    """Start the TeleFuser API server."""
    from pathlib import Path

    from telefuser.service.main import run_server
    from telefuser.service.security.security_validator import PipelineSecurityValidator, SecurityError

    # Validate pipeline file before starting server
    if not skip_validation and not validate_only:
        level = SecurityLevel[security_level.upper()]
        validator = PipelineSecurityValidator(security_level=level)

        try:
            validator.assert_safe(pipe_path)
        except SecurityError as e:
            click.echo(f"\n❌ Security validation failed:\n{e}", err=True)
            click.echo("\nTo bypass validation (not recommended):", err=True)
            click.echo(f"  telefuser serve  {pipe_path} --skip-validation", err=True)
            click.echo("\nTo see detailed report:", err=True)
            click.echo(f"  telefuser validate {pipe_path}", err=True)
            raise click.Abort()

    # If only validating, show report and exit
    if validate_only:
        report = validate_with_report(pipe_path)
        click.echo(report)
        return

    # Start server
    run_server(pipe_path, TaskType(task.lower()), port, host, cache_dir, parallelism)


@main.command(name="validate")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option(
    "--level",
    type=click.Choice(["none", "basic", "strict", "sandbox"], case_sensitive=False),
    default="strict",
    help="Security validation level",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output in JSON format",
)
def validate_pipeline(
    pipeline_file: str,
    level: str,
    output_json: bool,
) -> None:
    """Validate a pipeline configuration file for security issues."""
    from telefuser.service.security.security_validator import PipelineSecurityValidator, SecurityLevel

    security_level = SecurityLevel[level.upper()]
    validator = PipelineSecurityValidator(security_level=security_level)
    result = validator.validate_file(pipeline_file)

    if output_json:
        import json

        output = {
            "file": pipeline_file,
            "is_safe": result.is_safe,
            "violations": [v.to_dict() for v in result.violations],
            "warnings": [w.to_dict() for w in result.warnings],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        from telefuser.service.security.security_validator import validate_with_report

        report = validate_with_report(pipeline_file)
        click.echo(report)

    # Exit with appropriate code
    if not result.is_safe:
        raise click.Abort()


@main.command(name="scan")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--level",
    type=click.Choice(["none", "basic", "strict", "sandbox"], case_sensitive=False),
    default="strict",
    help="Security validation level",
)
@click.option(
    "--recursive/--no-recursive",
    default=True,
    help="Scan recursively",
)
def scan_pipelines(
    directory: str,
    level: str,
    recursive: bool,
) -> None:
    """Scan a directory for pipeline files and validate them."""
    from pathlib import Path

    from telefuser.service.security.security_validator import PipelineSecurityValidator, SecurityLevel

    security_level = SecurityLevel[level.upper()]
    validator = PipelineSecurityValidator(security_level=security_level)

    dir_path = Path(directory)
    pattern = "**/*.py" if recursive else "*.py"
    py_files = list(dir_path.glob(pattern))

    if not py_files:
        click.echo(f"No Python files found in {directory}")
        return

    click.echo(f"Scanning {len(py_files)} Python files...")
    click.echo("=" * 60)

    unsafe_files = []
    safe_count = 0

    for file_path in py_files:
        result = validator.validate_file(str(file_path))
        status = "✓ SAFE" if result.is_safe else "✗ UNSAFE"
        click.echo(f"{status}: {file_path.relative_to(dir_path)}")

        if result.is_safe:
            safe_count += 1
        else:
            unsafe_files.append((file_path, result))

    click.echo("=" * 60)
    click.echo(f"Summary: {safe_count}/{len(py_files)} files are safe")

    if unsafe_files:
        click.echo(f"\n{len(unsafe_files)} unsafe files found:")
        for file_path, result in unsafe_files:
            click.echo(f"\n{file_path}:")
            for v in result.violations[:3]:
                click.echo(f"  Line {v.line_number}: {v.description}")
        raise click.Abort()


if __name__ == "__main__":
    main()
