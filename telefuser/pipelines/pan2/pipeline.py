"""TeleFuser integration for the public IFM/PAN2 Diffusers pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PAN2PipelineConfig:
    model_id: str = "IFM/PAN2"
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    cpu_offload: bool = True
    nvfp4: bool = False
    nvfp4_use_triton: bool = True
    layerwise_nvfp4_dir: str | None = None


def _quantize_nvfp4(module: torch.nn.Module, use_triton_kernel: bool) -> None:
    """Quantize Linear weights and dynamic activations with TorchAO NVFP4."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("PAN2 NVFP4 requires a CUDA Blackwell-or-newer GPU (SM100+)")
    try:
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        from torchao.quantization import quantize_
    except ImportError as exc:
        raise RuntimeError(
            "PAN2 NVFP4 requires torchao with the NVFP4 prototype enabled; install torchao>=0.17"
        ) from exc
    quantize_(module, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=use_triton_kernel))


class PAN2Pipeline:
    """TeleFuser-owned loader that preserves the native Diffusers output."""

    def __init__(self, config: PAN2PipelineConfig):
        self.config = config
        self.pipe: Any | None = None

    @classmethod
    def from_pretrained(cls, config: PAN2PipelineConfig | None = None, **kwargs: Any) -> "PAN2Pipeline":
        config = config or PAN2PipelineConfig()
        try:
            from diffusers import ModularPipeline
        except ImportError as exc:
            raise RuntimeError("PAN2 requires diffusers>=0.43.0") from exc
        runner = cls(config)
        runner.pipe = ModularPipeline.from_pretrained(config.model_id, **kwargs)
        # PAN2 keeps some modules in fp32 and therefore requires accelerate's
        # low-memory loader when loading the sharded checkpoint.
        if config.layerwise_nvfp4_dir:
            # Do not materialize the BF16 transformer; restore its quantized
            # blocks directly and load only the auxiliary pipeline components.
            runner.pipe.load_components(
                names=["text_encoder", "tokenizer", "vae", "video_processor", "scheduler", "guider"],
                dtype=config.torch_dtype,
                low_cpu_mem_usage=True,
            )
            from .layerwise import load_layerwise_transformer

            runner.pipe.transformer = load_layerwise_transformer(
                str(__import__("pathlib").Path(config.model_id) / "transformer"),
                config.layerwise_nvfp4_dir,
                config.device,
            )
        else:
            runner.pipe.load_components(dtype=config.torch_dtype, low_cpu_mem_usage=True)
        # The optional guardrail package is not required for performance
        # benchmarking and is intentionally not a TeleFuser dependency.
        if hasattr(runner.pipe, "disable_safety_checker"):
            runner.pipe.disable_safety_checker()
        if config.layerwise_nvfp4_dir and getattr(runner.pipe, "scheduler", None) is not None:
            scheduler = runner.pipe.scheduler
            original_step = scheduler.step

            def cpu_step(model_output, timestep, sample, *args, **kwargs):
                return original_step(
                    model_output.to("cpu"),
                    timestep.to("cpu") if hasattr(timestep, "to") else timestep,
                    sample.to("cpu"),
                    *args,
                    **kwargs,
                )

            scheduler.step = cpu_step
        if config.nvfp4 and not config.layerwise_nvfp4_dir:
            transformer = getattr(runner.pipe, "transformer", None)
            if transformer is None:
                raise RuntimeError("PAN2 pipeline did not expose a transformer component")
            _quantize_nvfp4(transformer, config.nvfp4_use_triton)
        if config.cpu_offload and not config.layerwise_nvfp4_dir:
            device = torch.device(config.device)
            for component in (getattr(runner.pipe, "transformer", None), getattr(runner.pipe, "text_encoder", None)):
                if component is not None and hasattr(component, "enable_group_offload"):
                    component.enable_group_offload(
                        onload_device=device,
                        offload_device=torch.device("cpu"),
                        offload_type="block_level",
                        num_blocks_per_group=1,
                        use_stream=True,
                    )
            runner.pipe.to("cpu")
        elif not config.layerwise_nvfp4_dir:
            runner.pipe.to(config.device)
        return runner

    def __call__(self, **kwargs: Any) -> Any:
        if self.pipe is None:
            raise RuntimeError("PAN2Pipeline is not loaded; call from_pretrained first")
        return self.pipe(**kwargs)
