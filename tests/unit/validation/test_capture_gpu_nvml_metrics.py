from __future__ import annotations

import pytest

from tools.validation import capture_gpu_nvml_metrics as capture


def test_parse_args_defaults_to_the_four_experiment_gpus(tmp_path) -> None:
    config = capture.parse_args(["--duration", "2", "--output-dir", str(tmp_path / "metrics")])

    assert config.gpu_indices == (0, 1, 2, 3)
    assert config.duration_seconds == 2.0
    assert config.interval_seconds == 1.0


@pytest.mark.parametrize("value", ["", "0,,1", "0,0", "-1", "one"])
def test_gpu_indices_reject_malformed_values(value: str) -> None:
    with pytest.raises(SystemExit):
        capture.parse_args(["--gpu-indices", value, "--duration", "1", "--output-dir", "/tmp/unused"])
