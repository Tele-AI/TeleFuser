"""Layerwise loader for the CPU-produced PAN2 NVFP4 block checkpoint."""

from __future__ import annotations

import json
import copy
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers.models.transformers.transformer_pan2 import PAN2Transformer3DModel, PAN2TransformerBlock
from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization import quantize_


def _load_non_block_weights(model: torch.nn.Module, model_dir: Path, device: torch.device) -> None:
    from safetensors import safe_open

    index = json.loads((model_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    for shard in sorted(set(index.values())):
        keys = [key for key, value in index.items() if value == shard and not key.startswith("transformer_blocks.")]
        if not keys:
            continue
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            for key in keys:
                set_module_tensor_to_device(model, key, device, value=handle.get_tensor(key))


def load_layerwise_transformer(model_dir: str | Path, block_dir: str | Path, device: str = "cuda") -> PAN2Transformer3DModel:
    model_dir = Path(model_dir)
    block_dir = Path(block_dir)
    config = PAN2Transformer3DModel.load_config(str(model_dir))
    target = torch.device(device)
    with init_empty_weights():
        model = PAN2Transformer3DModel.from_config(config)
    _load_non_block_weights(model, model_dir, target)
    template = PAN2TransformerBlock(4096, 32, 128, 8.0, 4.0)
    quantize_(template, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True))
    for index, empty_block in enumerate(model.transformer_blocks):
        print(f"restoring block {index}", flush=True)
        state_path = block_dir / f"block-{index:02d}.pt"
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        block = copy.deepcopy(template)
        block.load_state_dict(
            torch.load(state_path, map_location="cpu", weights_only=False), assign=True
        )
        model.transformer_blocks[index] = block
        del empty_block
    del template
    if target.type == "cuda":
        # TorchAO NVFP4 tensors do not implement every tensor hook operation
        # used by Diffusers group offload. Explicit hooks keep only one block
        # on the GPU and use the supported module.to() path.
        for block in model.transformer_blocks:
            def onload(module: torch.nn.Module, _args: tuple[object, ...], d: torch.device = target) -> None:
                module.to(d)

            def offload(module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
                module.to("cpu")
                return output

            block.register_forward_pre_hook(onload)
            block.register_forward_hook(offload)
    return model
