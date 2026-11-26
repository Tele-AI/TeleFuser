import pytest
import torch

from telefuser.ops.fp8_attention_replay import FP8SolAttentionReplay


def test_replay_preserves_cpu_autograd() -> None:
    q = torch.randn(2, 3, requires_grad=True)
    k, v = torch.randn_like(q), torch.randn_like(q)
    replay = FP8SolAttentionReplay()
    output = replay.run(lambda a, b, c: a * b + c, q, k, v, key=())
    output.sum().backward()
    torch.testing.assert_close(q.grad, k)


@pytest.mark.gpu
def test_native_sol_replay_refreshes_inputs_and_owns_outputs() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires native SM90 Sol attention")
    pytest.importorskip("cutlass")
    from telefuser.core.config import AttentionConfig
    from telefuser.models.minimax_h3_dit import MiniMaxH3Attention, MiniMaxH3DiTConfig
    from telefuser.ops.attention import SparseAttentionState

    model = MiniMaxH3Attention(
        MiniMaxH3DiTConfig(hidden_size=32, num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=64)
    )
    config = AttentionConfig.sol_attention(dense_timesteps=1, dense_layers=1, sol_fp8=True)
    state = SparseAttentionState(config.sparse_config, None)
    state.update(numeral_timestep=1, layer_idx=1)
    replay = FP8SolAttentionReplay()

    def operation(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return model._attend_live(q, k, v, config, state, 3)

    with torch.inference_mode():
        previous = []
        for tokens in (130, 130, 129):
            q = torch.randn((1, tokens, 2, 128), device="cuda", dtype=torch.bfloat16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            expected = operation(q, k, v)
            actual = replay.run(operation, q, k, v, key=(3, 1.0))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for output, saved in previous:
                torch.testing.assert_close(output, saved, rtol=0, atol=0)
            previous.append((actual, actual.clone()))


@pytest.mark.gpu
@pytest.mark.parametrize("token_contiguous", [False, True])
def test_fused_fp8_padding_preserves_values_layout_and_zero_tails(token_contiguous: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Requires H100 FP8 tensors")
    from telefuser.kernel.sol_attn.triton_ref.preprocess import pad_fp8_qkv

    torch.manual_seed(8)
    q = torch.randn((2, 130, 3, 128), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k, v = q.clone(), q.clone()
    if token_contiguous:
        v = v.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    actual = pad_fp8_qkv(q, k, v, 192)
    for source, padded in zip((q, k, v), actual, strict=True):
        torch.testing.assert_close(padded[:, :130].float(), source.float(), rtol=0, atol=0)
        assert torch.count_nonzero(padded[:, 130:].float()) == 0
    assert actual[0].is_contiguous() and actual[1].is_contiguous()
    assert actual[2].stride(1) == 1 and actual[2].stride(3) == 192
