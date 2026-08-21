"""LTX-2.5 image-conditioning preprocessing tests."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from telefuser.pipelines.ltx25_distilled.image import preprocess_ltx25_image


def test_preprocess_ltx25_image_preserves_aspect_before_center_crop() -> None:
    pixels = np.array(
        [
            [[0, 0, 0], [64, 64, 64]],
            [[128, 128, 128], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)

    actual = preprocess_ltx25_image(image, height=2, width=4, crf=0, dtype=torch.float32)
    expected = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).float()
    expected = F.interpolate(expected, size=(4, 4), mode="bilinear", align_corners=False)
    expected = (expected[:, :, 1:3] / 127.5 - 1.0).unsqueeze(2)

    torch.testing.assert_close(actual, expected)
