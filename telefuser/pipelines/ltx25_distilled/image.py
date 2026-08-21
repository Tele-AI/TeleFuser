"""Faithful LTX-2.5 image-conditioning preprocessing."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import av
except ImportError:  # pragma: no cover - exercised by the runtime dependency check.
    av = None

from telefuser.models.ltx25.checkpoint import inspect_checkpoint


def default_image_crf(video_encoder_path: str | Path | None) -> int:
    """Return the upstream default SDR image-conditioning CRF."""
    if video_encoder_path is None:
        return 18
    return 18 if inspect_checkpoint(video_encoder_path).model_version >= (2, 4) else 33


def preprocess_ltx25_image(
    image: Image.Image,
    height: int,
    width: int,
    crf: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Match upstream SDR CRF, crop, resize, and VAE-normalization ordering."""
    pixels = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    if crf and min(pixels.shape[:2]) >= 2:
        if av is None:
            raise RuntimeError("PyAV is required for non-zero LTX-2.5 image conditioning CRF")
        with BytesIO() as encoded:
            container = av.open(encoded, "w", format="mp4")
            try:
                stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
                encoded_height = pixels.shape[0] // 2 * 2
                encoded_width = pixels.shape[1] // 2 * 2
                stream.height, stream.width = encoded_height, encoded_width
                frame = av.VideoFrame.from_ndarray(pixels[:encoded_height, :encoded_width], format="rgb24")
                container.mux(stream.encode(frame.reformat(format="yuv420p")))
                container.mux(stream.encode())
            finally:
                container.close()
            with av.open(BytesIO(encoded.getvalue())) as decoded:
                pixels = next(decoded.decode(video=0)).to_ndarray(format="rgb24")
    # Upstream moves pixels before interpolation. CPU interpolation rounds differently.
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
    source_height, source_width = tensor.shape[-2:]
    scale = max(height / source_height, width / source_width)
    resized_height, resized_width = int(np.ceil(source_height * scale)), int(np.ceil(source_width * scale))
    tensor = F.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    top, left = (resized_height - height) // 2, (resized_width - width) // 2
    return (tensor[:, :, top : top + height, left : left + width] / 127.5 - 1.0).to(dtype=dtype).unsqueeze(2)


__all__ = ["default_image_crf", "preprocess_ltx25_image"]
