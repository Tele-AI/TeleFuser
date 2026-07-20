import torch
from safetensors.torch import save_file

from telefuser.core.module_manager import ModuleManager
from telefuser.models.lingbot_world_fast_dit import LingBotWorldFastDiT, LingBotWorldFastDiTStateDictConverter


def _checkpoint(control_dim: int = 6) -> dict[str, torch.Tensor]:
    return {
        "patch_embedding.weight": torch.empty(80, 16, 1, 2, 2),
        "patch_embedding_wancamctrl.weight": torch.empty(80, control_dim * 64 * 4),
        "blocks.0.ffn.0.weight": torch.empty(160, 80),
        "blocks.1.ffn.0.weight": torch.empty(160, 80),
        "text_embedding.0.weight": torch.empty(80, 32),
        "head.head.bias": torch.empty(64),
    }


def test_lingbot_dit_converter_infers_native_checkpoint_shape() -> None:
    checkpoint = _checkpoint()

    weights, config = LingBotWorldFastDiTStateDictConverter().from_official(checkpoint)

    assert weights is checkpoint
    assert config == {
        "patch_size": (1, 2, 2),
        "text_len": 512,
        "in_dim": 16,
        "dim": 80,
        "ffn_dim": 160,
        "freq_dim": 256,
        "text_dim": 32,
        "out_dim": 16,
        "num_heads": 40,
        "num_layers": 2,
        "local_attn_size": -1,
        "sink_size": 0,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "control_type": "cam",
    }


def test_lingbot_dit_cache_window_is_runtime_configuration() -> None:
    dit = LingBotWorldFastDiT(dim=80, ffn_dim=160, text_dim=32, num_heads=40, num_layers=2)

    dit.set_causal_attention_window(local_attn_size=18, sink_size=6)

    assert dit.local_attn_size == 18
    assert dit.sink_size == 6
    assert all(block.self_attn.local_attn_size == 18 for block in dit.blocks)
    assert all(block.self_attn.sink_size == 6 for block in dit.blocks)


def test_module_manager_loads_lingbot_dit_with_native_converter(tmp_path) -> None:
    source = LingBotWorldFastDiT(dim=80, ffn_dim=160, text_dim=32, num_heads=40, num_layers=2)
    checkpoint_path = tmp_path / "diffusion_pytorch_model.safetensors"
    save_file(source.state_dict(), str(checkpoint_path))

    module_manager = ModuleManager(device="cpu", torch_dtype=torch.float32)
    module_manager.load_model(
        str(checkpoint_path),
        name="dit",
        model_class=LingBotWorldFastDiT,
        model_resource="official",
        torch_dtype=torch.float32,
    )

    loaded = module_manager.fetch_module("dit")
    assert isinstance(loaded, LingBotWorldFastDiT)
    assert loaded.control_type == "cam"
    assert loaded.local_attn_size == -1
    torch.testing.assert_close(loaded.patch_embedding.weight, source.patch_embedding.weight)
