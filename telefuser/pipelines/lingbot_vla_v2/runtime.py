"""Runtime construction for single-replica LingBot-VLA v2 inference."""

from __future__ import annotations

import torch
from transformers import AutoProcessor

from telefuser.core.config import ModelRuntimeConfig, QuantConfig, QuantKernelBackend, QuantType
from telefuser.core.module_manager import ModuleManager
from telefuser.models.lingbot_vla_v2_loader import load_lingbot_vla_v2

from .pipeline import LingBotVlaV2Pipeline, LingBotVlaV2PipelineConfig

LINGBOT_VLA_V2_QUANTIZATION_CHOICES = ("torchao-fp8", "tf-kernel-fp8", "bnb-nf4")


def lingbot_vla_v2_quant_config(quantization: str | QuantType | None) -> QuantConfig:
    """Resolve a public LingBot-VLA v2 online-quantization name."""
    if quantization is None:
        return QuantConfig()
    if isinstance(quantization, str):
        normalized = quantization.strip().lower().replace("_", "-")
        names = {
            "torchao-fp8": QuantType.TORCHAO_FP8,
            "tf-kernel-fp8": QuantType.FP8,
            "bnb-nf4": QuantType.BNB_NF4,
        }
        try:
            quant_type = names[normalized]
        except KeyError as exc:
            choices = ", ".join(repr(name) for name in LINGBOT_VLA_V2_QUANTIZATION_CHOICES)
            raise ValueError(f"quantization must be one of {choices}, or None") from exc
    elif isinstance(quantization, QuantType):
        quant_type = quantization
    else:
        raise TypeError("quantization must be a string, QuantType, or None")

    backends = {
        QuantType.TORCHAO_FP8: QuantKernelBackend.TORCHAO,
        QuantType.FP8: QuantKernelBackend.TF_KERNEL,
        QuantType.BNB_NF4: QuantKernelBackend.BITSANDBYTES,
    }
    if quant_type not in backends:
        raise ValueError(f"LingBot-VLA v2 does not support online quantization type {quant_type.name}")
    return QuantConfig(enabled=True, quant_type=quant_type, kernel_backend=backends[quant_type])


def get_lingbot_vla_v2_pipeline(
    model_root: str,
    qwen3vl_root: str,
    device: str = "cuda:0",
    *,
    warmup: bool = False,
    quantization: str | QuantType | None = None,
    cuda_graph: bool = False,
) -> LingBotVlaV2Pipeline:
    """Load one official 6B base checkpoint replica for inference."""
    target_device = torch.device(device)
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} was requested, but CUDA is unavailable")
        device_index = target_device.index or 0
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device_index} is unavailable; visible device count is {torch.cuda.device_count()}"
            )
        target_device = torch.device("cuda", device_index)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    quant_config = lingbot_vla_v2_quant_config(quantization)
    if quant_config.enabled and target_device.type != "cuda":
        raise ValueError("LingBot-VLA v2 online quantization requires a CUDA device")
    if cuda_graph and target_device.type != "cuda":
        raise ValueError("LingBot-VLA v2 CUDA Graph requires a CUDA device")
    if cuda_graph and quant_config.enabled:
        raise ValueError("LingBot-VLA v2 CUDA Graph currently supports only the BF16 profile")
    processor = AutoProcessor.from_pretrained(qwen3vl_root, local_files_only=True, padding_side="right")
    manager = ModuleManager(torch_dtype=dtype, device="cpu")
    manager.add_module(processor, "lingbot_vla_v2_processor", path=qwen3vl_root)
    load_lingbot_vla_v2(
        manager,
        model_root,
        qwen3vl_root,
        torch_dtype=dtype,
        device=target_device if quant_config.enabled else None,
        quant_config=quant_config if quant_config.enabled else None,
    )
    pipeline = LingBotVlaV2Pipeline(device=str(target_device), torch_dtype=dtype)
    pipeline.init(
        manager,
        LingBotVlaV2PipelineConfig(
            cuda_graph=cuda_graph,
            policy_config=ModelRuntimeConfig(
                device_type=target_device.type,
                device_id=target_device.index or 0,
                torch_dtype=dtype,
                quant_config=quant_config,
            ),
        ),
    )
    pipeline.prepare_for_inference()
    if warmup:
        pipeline.warmup()
    return pipeline
