"""Native checkpoint assembly for the LingBot-Video pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, OffloadConfig, WeightOffloadType
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler

from .data import load_lingbot_video_model_config
from .denoising import LingBotVideoDenoisingStage
from .loading import load_lingbot_video_dense_transformer, load_lingbot_video_moe_transformer
from .pipeline import LingBotVideoPipeline, LingBotVideoPipelineConfig
from .refiner import LingBotVideoRefinerStage
from .text_encoding import LingBotVideoTextEncodingStage
from .vae import LingBotVideoVAEDecodeStage, LingBotVideoVAEEncodeStage


def build_lingbot_video_pipeline(
    model_dir: str | Path,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    variant: str = "dense",
    cpu_offload: bool | None = None,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
    shift: float = 3.0,
    attention_config: AttentionConfig | None = None,
) -> LingBotVideoPipeline:
    """Load official checkpoint components and assemble a native pipeline runtime.

    ``variant`` selects a Dense transformer or the MoE base transformer. The
    refiner remains a separately loaded stage so its weights do not need to
    coexist with the base model.
    """
    if cpu_offload is None:
        cpu_offload = variant == "moe"
    if variant not in {"dense", "moe"}:
        raise ValueError("variant must be 'dense' or 'moe'")
    try:
        from diffusers import AutoencoderKLWan
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("LingBot-Video runtime requires diffusers and transformers") from exc

    root = Path(model_dir)
    transformer_dir = root / "transformer"
    runtime_config = ModelRuntimeConfig(
        device_type=torch.device(device).type,
        torch_dtype=torch_dtype,
        attention_config=attention_config or AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
        offload_config=OffloadConfig(
            offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD if cpu_offload else WeightOffloadType.NO_CPU_OFFLOAD
        ),
    )
    load_device = "cpu" if cpu_offload else device
    transformer = (
        load_lingbot_video_dense_transformer(transformer_dir, device=load_device, torch_dtype=torch_dtype)
        if variant == "dense"
        else load_lingbot_video_moe_transformer(transformer_dir, device=load_device, torch_dtype=torch_dtype)
    )
    processor = AutoProcessor.from_pretrained(root / "processor")
    text_encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            root / "text_encoder", dtype=torch_dtype, attn_implementation="sdpa"
        )
        .to(load_device)
        .eval()
    )
    vae = AutoencoderKLWan.from_pretrained(root / "vae", torch_dtype=torch.float32).to(load_device).eval()
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(root / "scheduler")
    pipeline = LingBotVideoPipeline(device=device, torch_dtype=torch_dtype)
    pipeline.model_dir = str(root)
    pipeline.variant = variant
    pipeline.init(
        None,
        LingBotVideoPipelineConfig(
            model=load_lingbot_video_model_config(transformer_dir, variant=variant),
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            shift=shift,
        ),
    )
    pipeline.set_runtime(
        text_stage=LingBotVideoTextEncodingStage("text_encoder", text_encoder, processor, runtime_config),
        denoising_stage=LingBotVideoDenoisingStage("transformer", transformer, runtime_config),
        vae_encode_stage=LingBotVideoVAEEncodeStage("vae_encode", vae, runtime_config),
        vae_decode_stage=LingBotVideoVAEDecodeStage("vae_decode", vae, runtime_config),
        scheduler=scheduler,
    )
    return pipeline


def build_lingbot_video_refiner_stage(
    model_dir: str | Path,
    *,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    cpu_offload: bool = True,
    attention_config: AttentionConfig | None = None,
) -> LingBotVideoRefinerStage:
    """Load the separately checkpointed MoE refiner and its VAE/scheduler stages."""
    try:
        from diffusers import AutoencoderKLWan
    except ImportError as exc:
        raise RuntimeError("LingBot-Video refiner runtime requires diffusers") from exc

    root = Path(model_dir)
    runtime_config = ModelRuntimeConfig(
        device_type=torch.device(device).type,
        torch_dtype=torch_dtype,
        attention_config=attention_config or AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA),
        offload_config=OffloadConfig(
            offload_type=WeightOffloadType.MODEL_CPU_OFFLOAD if cpu_offload else WeightOffloadType.NO_CPU_OFFLOAD
        ),
    )
    load_device = "cpu" if cpu_offload else device
    transformer = load_lingbot_video_moe_transformer(root / "refiner", device=load_device, torch_dtype=torch_dtype)
    vae = AutoencoderKLWan.from_pretrained(root / "vae", torch_dtype=torch.float32).to(load_device).eval()
    scheduler = FlowUniPCMultistepScheduler.from_pretrained(root / "scheduler")
    return LingBotVideoRefinerStage(
        denoising_stage=LingBotVideoDenoisingStage("refiner", transformer, runtime_config),
        vae_encode_stage=LingBotVideoVAEEncodeStage("refiner_vae_encode", vae, runtime_config),
        vae_decode_stage=LingBotVideoVAEDecodeStage("refiner_vae_decode", vae, runtime_config),
        scheduler=scheduler,
    )
