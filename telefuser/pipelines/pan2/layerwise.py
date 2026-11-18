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
                value = handle.get_tensor(key)
                if value.is_floating_point():
                    value = value.to(torch.bfloat16)
                set_module_tensor_to_device(model, key, device, value=value)


def load_layerwise_transformer(model_dir: str | Path, block_dir: str | Path, device: str = "cuda") -> PAN2Transformer3DModel:
    model_dir = Path(model_dir)
    block_dir = Path(block_dir)
    config = PAN2Transformer3DModel.load_config(str(model_dir))
    target = torch.device(device)
    with init_empty_weights():
        model = PAN2Transformer3DModel.from_config(config)
    model = model.to_empty(device="meta").to(dtype=torch.bfloat16)
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
                with torch.inference_mode(False):
                    module.to(d)

            def offload(module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
                with torch.inference_mode(False):
                    module.to("cpu")
                return output

            block.register_forward_pre_hook(onload)
            block.register_forward_hook(offload)

    # The PAN2 checkpoint keeps the input projection in fp32 while the
    # pipeline normally creates BF16 latents. Match the projection dtype so
    # PyTorch selects the CUDA convolution kernel instead of slow_conv3d.
    def cast_inputs(module: torch.nn.Module, args: tuple[object, ...], kwargs: dict[str, object]):
        if len(args) >= 3:
            hidden, timestep, context = args[:3]
            hidden = hidden.to(target, dtype=module.x_embedder.weight.dtype)
            context = context.to(target, dtype=module.context_embedder[1].weight.dtype)
            timestep = timestep.to(target)
            return (hidden, timestep, context, *args[3:]), kwargs
        if "hidden_states" in kwargs:
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = kwargs["hidden_states"].to(target, dtype=module.x_embedder.weight.dtype)
            if "encoder_hidden_states" in kwargs:
                kwargs["encoder_hidden_states"] = kwargs["encoder_hidden_states"].to(
                    target, dtype=module.context_embedder[1].weight.dtype
                )
            if "timestep" in kwargs:
                kwargs["timestep"] = kwargs["timestep"].to(target)
        return args, kwargs

    model.register_forward_pre_hook(cast_inputs, with_kwargs=True)

    def restore_output_device(_module: torch.nn.Module, _args: tuple[object, ...], output: object) -> object:
        if hasattr(output, "sample"):
            output.sample = output.sample.to("cpu")
            return output
        if isinstance(output, torch.Tensor):
            return output.to("cpu")
        return output

    model.register_forward_hook(restore_output_device)
    return model
