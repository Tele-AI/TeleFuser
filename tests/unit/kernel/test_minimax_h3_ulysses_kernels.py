import pytest
import torch

from telefuser.kernel.triton.ulysses_relayout import (
    merge_ulysses_head_chunk,
    pack_qkv_destination_major,
    pack_qkv_qknorm_rope_destination_major,
)
from telefuser.ops.rotary import apply_qk_norm_rope_neox

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton kernels")


def test_fused_qknorm_rope_pack_is_bit_exact() -> None:
    torch.manual_seed(0)
    rows, heads, head_dim, world_size = 11, 4, 8, 2
    qkv = torch.randn(rows, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(rows, 2, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)

    reference_qkv = qkv.clone()
    query, key, value = reference_qkv.unbind(dim=1)
    query, key = apply_qk_norm_rope_neox(
        query,
        key,
        q_weight,
        k_weight,
        cache,
        eps=1e-5,
    )
    expected = pack_qkv_destination_major(query, key, value, world_size)
    actual = pack_qkv_qknorm_rope_destination_major(
        qkv,
        q_weight,
        k_weight,
        cache,
        world_size,
        1e-5,
    )

    assert torch.equal(actual, expected)


def test_merge_head_chunk_writes_valid_data_and_zero_tail() -> None:
    torch.manual_seed(0)
    world_size, valid_rows, output_rows = 2, 5, 8
    batch, chunk_heads, total_local_heads, head_dim = 1, 1, 2, 8
    received = torch.randn(
        world_size,
        valid_rows,
        batch,
        chunk_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        batch,
        output_rows,
        world_size,
        total_local_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual = merge_ulysses_head_chunk(received, output, local_head_start=1, zero_tail=True)

    expected = torch.empty_like(actual)
    expected.fill_(float("nan"))
    expected[:, :, :, 1].zero_()
    for destination in range(world_size):
        expected[0, :valid_rows, destination, 1].copy_(received[destination, :, 0, 0])
    assert torch.equal(actual[:, :, :, 1], expected[:, :, :, 1])
