"""
https://github.com/jt-zhang/Sparse_SageAttention_API

Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch


def sparse_sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask_id: torch.Tensor | None = None,
    is_causal: bool = False,
    tensor_layout: str = "HND",
) -> torch.Tensor:
    """Sparse SageAttention with INT8 quantization.

    Args:
        q: Query tensor. HND format: [B, H, L, C]; NHD format: [B, L, H, C]
        k: Key tensor. HND format: [B, H, L, C]; NHD format: [B, L, H, C]
        v: Value tensor. HND format: [B, H, L, C]; NHD format: [B, L, H, C]
        mask_id: Sparse mask ID tensor [B, H, Q_blocks, K_blocks].
        is_causal: Whether to use causal attention.
        tensor_layout: Tensor layout format ("HND" or "NHD").

    Returns:
        Output tensor with same shape and layout as input.
    """
    from telefuser.kernel.triton.quant_per_block import per_block_int8
    from telefuser.kernel.triton.sparse_int8_attn import forward as sparse_sageattn_fwd

    seq_dim = 2 if tensor_layout == "HND" else 1

    if mask_id is None:
        if tensor_layout == "HND":
            mask_id = torch.ones(
                (q.shape[0], q.shape[1], (q.shape[2] + 128 - 1) // 128, (q.shape[3] + 64 - 1) // 64),
                dtype=torch.int8,
                device=q.device,
            )
        else:  # NHD
            mask_id = torch.ones(
                (q.shape[0], q.shape[2], (q.shape[1] + 128 - 1) // 128, (q.shape[3] + 64 - 1) // 64),
                dtype=torch.int8,
                device=q.device,
            )

    output_dtype = q.dtype
    if output_dtype == torch.bfloat16 or output_dtype == torch.float32:
        v = v.to(torch.float16)

    km = k.mean(dim=seq_dim, keepdim=True)
    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k, km=km, tensor_layout=tensor_layout)

    o = sparse_sageattn_fwd(
        q_int8,
        k_int8,
        mask_id,
        v,
        q_scale,
        k_scale,
        is_causal=is_causal,
        tensor_layout=tensor_layout,
        output_dtype=output_dtype,
    )
    return o
