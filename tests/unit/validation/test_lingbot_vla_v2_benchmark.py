from __future__ import annotations

import argparse

import pytest
from PIL import Image

from tools.validation.benchmark_lingbot_vla_v2_service import (
    encode_image,
    parse_image_sizes,
    percentile,
    summarize,
)


def test_parse_image_sizes_deduplicates_and_preserves_order() -> None:
    assert parse_image_sizes("256x256, 640X480,256x256") == ((256, 256), (640, 480))


@pytest.mark.parametrize("value", ["", "256", "0x256", "axb"])
def test_parse_image_sizes_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_image_sizes(value)


def test_latency_summary_reports_interpolated_percentiles_and_throughput() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    result = summarize(values)

    assert percentile(values, 0.5) == 2.5
    assert result["count"] == 4
    assert result["mean_seconds"] == 2.5
    assert result["p95_seconds"] == pytest.approx(3.85)
    assert result["p99_seconds"] == pytest.approx(3.97)
    assert result["throughput_requests_per_second"] == 0.4


def test_encode_image_reports_decoded_jpeg_size() -> None:
    encoded, encoded_bytes = encode_image(Image.new("RGB", (8, 8)), (32, 24), quality=90)

    assert encoded
    assert encoded_bytes > 0
    assert len(encoded) >= encoded_bytes
