# SPDX-License-Identifier: Apache-2.0
"""H3-Image Qwen3-VL conditioner with checkpoint-faithful rotary precision."""

from __future__ import annotations

from typing import Any

import torch

from telefuser.models.minimax_h3_encoder import MiniMaxH3Encoder


class MiniMaxH3ImageEncoder(MiniMaxH3Encoder):
    def to(self, *args: Any, **kwargs: Any) -> MiniMaxH3ImageEncoder:
        # Transformers from_pretrained keeps these non-persistent buffers in FP32.
        preserved = {
            name: buffer.detach().clone()
            for name, buffer in self.named_buffers()
            if name.endswith("inv_freq") and not buffer.is_meta
        }
        result = super().to(*args, **kwargs)
        for name, buffer in result.named_buffers():
            if name.endswith("inv_freq"):
                value = preserved.get(name, buffer)
                buffer.data = value.to(device=buffer.device, dtype=torch.float32)
        return result
