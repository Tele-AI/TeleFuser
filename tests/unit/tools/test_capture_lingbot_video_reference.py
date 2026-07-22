from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPOSITORY_ROOT / "tools" / "validation" / "capture_lingbot_video_reference.py"


def test_lingbot_video_reference_capture_dry_run_writes_manifest(tmp_path: Path) -> None:
    output_dir = tmp_path / "reference"
    completed = subprocess.run(
        [
            "python",
            str(TOOL),
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--mode",
            "t2i",
            "--case",
            "example_1",
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert manifest["modes"] == ["t2i"]
    assert manifest["case"] == "example_1"
    assert '"dry_run": true' in completed.stdout


def test_lingbot_video_reference_capture_dry_run_selects_all_mode_cases(tmp_path: Path) -> None:
    output_dir = tmp_path / "all-cases"
    subprocess.run(
        [
            "python",
            str(TOOL),
            "--dry-run",
            "--output-dir",
            str(output_dir),
            "--mode",
            "t2i",
            "--all-cases",
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "capture_manifest.json").read_text(encoding="utf-8"))
    assert manifest["all_cases"] is True
    assert [item["name"] for item in manifest["selected_cases"]] == [f"example_{index}" for index in range(1, 7)]
