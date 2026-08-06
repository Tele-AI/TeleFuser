"""Structural regression tests for the configured example matrix."""

from __future__ import annotations

import ast
from pathlib import Path

from examples.run_examples import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = PROJECT_ROOT / "examples"
SERVICE_PARITY_EXAMPLES = {
    "wan_video/wan21_14b_image_to_video_480p_service.py",
    "wan_video/wan22_14b_image_to_video_distill_h100.py",
    "lingbot_video/lingbot_video_dense_1_3b.py",
    "lingbot_video/lingbot_video_moe_30b.py",
    "minimax_h3/minimax_h3_fl2va_h100.py",
    "minimax_h3/minimax_h3_fl2va_torchao_fp8_h100.py",
    "minimax_h3/minimax_h3_fl2va_bnb_nf4_h100.py",
    "minimax_h3/minimax_h3_ref2va_h100.py",
}


def _module_symbols(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _declares_service_contract(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        if any(
            isinstance(target, ast.Name) and target.id in {"PIPELINE_CONTRACT", "PIPELINE_MANIFEST"}
            for target in targets
        ):
            return True
    return False


def test_example_regression_registry_has_runnable_entrypoints() -> None:
    config = load_config()

    assert config.pipelines
    for name, pipeline in config.pipelines.items():
        script_path = EXAMPLES_ROOT / pipeline.script
        assert script_path.is_file(), f"{name} references a missing example: {pipeline.script}"
        assert pipeline.output_type in {"image", "video"}, f"{name} has invalid output type: {pipeline.output_type}"
        assert pipeline.gpu_count >= 1, f"{name} must request at least one GPU"
        assert pipeline.timeout_seconds > 0, f"{name} must define a positive timeout"

        symbols = _module_symbols(script_path)
        assert "get_pipeline" in symbols, f"{name} is missing get_pipeline()"
        assert "run" in symbols, f"{name} is missing run()"
        if pipeline.use_run_with_file:
            assert "run_with_file" in symbols, f"{name} is missing configured run_with_file()"


def test_minimax_h3_four_gpu_regression_contract() -> None:
    pipeline = load_config().pipelines["minimax_h3_t2va_4gpu"]

    assert pipeline.script == "minimax_h3/minimax_h3_fl2va_h100.py"
    assert pipeline.gpu_count == 4
    assert pipeline.seed == 0
    assert pipeline.resolution == "768p"
    assert pipeline.target_video_length == 5
    assert pipeline.use_run_with_file is True
    assert pipeline.require_audio is True
    assert pipeline.ppl_config_overrides["num_inference_steps"] == 50


def test_all_declared_service_examples_have_cpu_parity_coverage() -> None:
    declared_contract_examples = {
        path.relative_to(EXAMPLES_ROOT).as_posix()
        for path in EXAMPLES_ROOT.rglob("*.py")
        if _declares_service_contract(path)
    }

    assert declared_contract_examples == SERVICE_PARITY_EXAMPLES
    for script in declared_contract_examples:
        symbols = _module_symbols(EXAMPLES_ROOT / script)
        assert "get_pipeline" in symbols, f"{script} is missing get_pipeline()"
        assert "run_with_file" in symbols, f"{script} is missing run_with_file()"
