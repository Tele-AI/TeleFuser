"""Runtime helpers for applying and validating service task contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fastapi import HTTPException

from telefuser.service.core.pipeline_contract import infer_media_type_for_task


def apply_task_contract_defaults(
    message: Any,
    *,
    task_contract: dict[str, Any] | None,
    explicit_fields: set[str] | None = None,
) -> None:
    """Apply task-parameter defaults from the active contract to a request object."""
    parameters = _get_parameter_contracts(task_contract)
    explicit_fields = set(explicit_fields or ())

    for field_name, metadata in parameters.items():
        if not metadata.get("exposed", True):
            continue
        if field_name in explicit_fields:
            continue
        if "default" not in metadata:
            continue

        default_value = metadata.get("default")
        if default_value is None:
            continue
        _set_message_value(message, field_name, default_value)

    if "output_path" not in explicit_fields:
        output_contract = parameters.get("output_path") or {}
        output_default = output_contract.get("default")
        if output_default in (None, ""):
            default_output_path = _build_default_output_path(message)
            if default_output_path:
                _set_message_value(message, "output_path", default_output_path)


def validate_required_task_parameters(message: Any, *, task_contract: dict[str, Any] | None) -> None:
    """Validate required exposed task parameters after defaults are applied."""
    parameters = _get_parameter_contracts(task_contract)
    missing_parameters = [
        name
        for name, metadata in parameters.items()
        if metadata.get("required")
        and metadata.get("exposed", True)
        and _is_missing_value(getattr(message, name, None))
    ]

    if missing_parameters:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Task '{getattr(message, 'task', '')}' is missing required parameters",
                "required_parameters": [
                    name
                    for name, metadata in parameters.items()
                    if metadata.get("required") and metadata.get("exposed", True)
                ],
                "missing_parameters": missing_parameters,
            },
        )


def map_contract_fields(source_fields: set[str], mapping: dict[str, str]) -> set[str]:
    """Map source-model field names to TaskRequest/task-contract field names."""
    explicit_fields: set[str] = set()
    for source_name in source_fields:
        mapped_name = mapping.get(source_name)
        if mapped_name:
            explicit_fields.add(mapped_name)
    return explicit_fields


def match_task_candidates(
    tasks: Iterable[str],
    *,
    get_task_contract: Callable[[str], dict[str, Any] | None],
    available_inputs: set[str],
    media_type: str | None = None,
) -> list[str]:
    """Return compatible task candidates ordered by required-input specificity."""
    matches: list[tuple[int, int, str]] = []
    for index, task in enumerate(tasks):
        contract = get_task_contract(task) or {}
        contract_media_type = str(contract.get("media_type") or infer_media_type_for_task(task))
        if media_type and contract_media_type != media_type:
            continue

        required_inputs = tuple(contract.get("required_inputs", ()))
        optional_inputs = tuple(contract.get("optional_inputs", ()))
        if not all(input_name in available_inputs for input_name in required_inputs):
            continue

        if available_inputs:
            consumed_inputs = set(required_inputs) | set(optional_inputs)
            if not consumed_inputs.intersection(available_inputs):
                continue

        matches.append((-len(required_inputs), index, str(task)))

    matches.sort()
    return [task for _, _, task in matches]


def _get_parameter_contracts(task_contract: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    raw_parameters = (task_contract or {}).get("parameters") or {}
    return {str(name): dict(metadata) for name, metadata in raw_parameters.items() if isinstance(metadata, dict)}


def _set_message_value(message: Any, field_name: str, value: Any) -> None:
    try:
        setattr(message, field_name, value)
    except Exception:
        object.__setattr__(message, field_name, value)

    fields_set = getattr(message, "model_fields_set", None)
    if isinstance(fields_set, set):
        fields_set.add(field_name)


def _build_default_output_path(message: Any) -> str:
    task_id = getattr(message, "task_id", "")
    task = getattr(message, "task", "")
    if not task_id or not task:
        return ""

    media_type = infer_media_type_for_task(task)
    if media_type == "image":
        output_format = getattr(message, "output_format", "png") or "png"
        return f"{task_id}.{output_format}"
    return f"{task_id}.mp4"


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
