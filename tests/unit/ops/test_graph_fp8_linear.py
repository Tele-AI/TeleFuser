from __future__ import annotations

import pytest
import torch
from torch import nn

from telefuser.models.lingbot_vla_v2_moe import Qwen2FusedExperts
from telefuser.ops.graph_fp8_linear import GraphFP8Linear
from telefuser.ops.lingbot_vla_v2_moe import robby_moe_forward, robby_moe_forward_fp8


@pytest.mark.gpu
@torch.inference_mode()
def test_graph_fp8_linear_discards_bf16_weight_and_replays_new_input() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Graph FP8 Linear requires Hopper or newer CUDA hardware")

    device = torch.device("cuda:0")
    source = nn.Linear(64, 64, bias=True, device=device, dtype=torch.bfloat16).eval()
    x = torch.randn(16, 64, device=device, dtype=torch.bfloat16)
    reference = source(x)
    linear = GraphFP8Linear(source)
    output = linear(x)
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert not any(parameter.dtype == torch.bfloat16 for parameter in linear.parameters())
    assert torch.nn.functional.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0) > 0.998

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        linear(x)
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = linear(x)

    x.copy_(torch.randn_like(x))
    expected_replay = linear(x)
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(graph_output, expected_replay)


@pytest.mark.gpu
@torch.inference_mode()
def test_graph_fp8_grouped_moe_discards_bf16_weights_and_replays_routing() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Graph FP8 grouped MoE requires Hopper or newer CUDA hardware")

    device = torch.device("cuda:0")
    tokens, hidden, experts, intermediate, top_k = 16, 64, 4, 64, 2
    x = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    routing = torch.softmax(torch.randn(tokens, top_k, device=device), dim=-1).to(torch.bfloat16)
    selected = torch.randint(experts, (tokens, top_k), device=device)
    module = Qwen2FusedExperts(experts, hidden, intermediate).to(device=device, dtype=torch.bfloat16).eval()
    workspace = module._get_robby_moe_workspace(x, top_k)
    reference = robby_moe_forward(
        x,
        routing,
        selected,
        module.gate_proj,
        module.up_proj,
        module.down_proj,
        workspace,
    )
    module.enable_graph_fp8()
    workspace = module._get_robby_moe_workspace(x, top_k)

    def forward() -> torch.Tensor:
        return robby_moe_forward_fp8(
            x,
            routing,
            selected,
            module.gate_proj_fp8,
            module.gate_proj_scale,
            module.up_proj_fp8,
            module.up_proj_scale,
            module.down_proj_fp8,
            module.down_proj_scale,
            workspace,
        )

    output = forward()
    assert not hasattr(module, "gate_proj")
    assert not hasattr(module, "up_proj")
    assert not hasattr(module, "down_proj")
    assert module.gate_proj_fp8.dtype == torch.float8_e4m3fn
    assert torch.nn.functional.cosine_similarity(output.float().flatten(), reference.float().flatten(), dim=0) > 0.995

    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(capture_stream):
        forward()
    torch.cuda.current_stream(device).wait_stream(capture_stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = forward()

    x.copy_(torch.randn_like(x))
    routing.copy_(torch.softmax(torch.randn_like(routing), dim=-1))
    selected.copy_(torch.randint_like(selected, experts))
    expected_replay = forward().clone()
    graph.replay()
    torch.cuda.synchronize(device)
    assert torch.equal(graph_output, expected_replay)
