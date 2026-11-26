"""Tests for BF16-reference LTX video quality metrics."""

from __future__ import annotations

import pytest

from tools.validation.compare_ltx_video_quality import parse_metrics


def test_parse_full_reference_metrics() -> None:
    result = parse_metrics(
        "PSNR y:21.1 u:30.0 v:31.0 average:23.399009 min:19.0 max:28.3",
        "SSIM Y:0.87 U:0.97 V:0.97 All:0.908010 (10.36)",
        "\n".join(
            [
                "VIF scale=0 average:0.327015 min:0.1 max:0.4",
                "VIF scale=1 average:0.416468 min:0.2 max:0.5",
                "VIF scale=2 average:0.436655 min:0.2 max:0.5",
                "VIF scale=3 average:0.436058 min:0.2 max:0.5",
            ]
        ),
    )
    assert result["psnr_db"] == 23.399009
    assert result["normalized_rmse"] == pytest.approx(10 ** (-23.399009 / 20))
    assert result["ssim"] == 0.908010
    assert result["vif_mean"] == pytest.approx(0.404049)


def test_parse_rejects_incomplete_metrics() -> None:
    with pytest.raises(ValueError, match="four VIF scales"):
        parse_metrics("PSNR average:20.0", "SSIM All:0.9", "VIF scale=0 average:0.4")
