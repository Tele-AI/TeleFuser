"""Convert PAN2 transformer blocks to TorchAO NVFP4 one block at a time.

This deliberately avoids loading the 29B transformer as one BF16 module. Each
block is read from the safetensors index, quantized on CPU, and serialized
before the next block is materialized.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from safetensors import safe_open

from diffusers.models.transformers.transformer_pan2 import PAN2TransformerBlock
from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
from torchao.quantization import quantize_


def _block_state(model_dir: Path, index: dict[str, str], block_index: int) -> dict[str, torch.Tensor]:
    prefix = f"transformer_blocks.{block_index}."
    entries = {k: v for k, v in index.items() if k.startswith(prefix)}
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(entries.values())):
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key, filename in entries.items():
                if filename == shard and key in available:
                    state[key[len(prefix) :]] = handle.get_tensor(key)
    if not state:
        raise KeyError(f"No weights found for block {block_index}")
    return state


def convert_block(model_dir: Path, index: dict[str, str], output_dir: Path, block_index: int) -> None:
    output = output_dir / f"block-{block_index:02d}.pt"
    if output.exists():
        print(f"skip block {block_index}: {output}", flush=True)
        return
    state = _block_state(model_dir, index, block_index)
    block = PAN2TransformerBlock(4096, 32, 128, 8.0, 4.0)
    block.load_state_dict(state)
    del state
    quantize_(block, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=False))
    temporary = output.with_suffix(".tmp")
    torch.save(block.state_dict(), temporary)
    temporary.replace(output)
    del block
    gc.collect()
    print(f"converted block {block_index}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-blocks", type=int, default=44)
    args = parser.parse_args()
    index_path = args.model_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"model_dir": str(args.model_dir), "format": "torchao-nvfp4", "num_blocks": args.num_blocks}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for block_index in range(args.num_blocks):
        convert_block(args.model_dir / "transformer", index, args.output_dir, block_index)


if __name__ == "__main__":
    main()
