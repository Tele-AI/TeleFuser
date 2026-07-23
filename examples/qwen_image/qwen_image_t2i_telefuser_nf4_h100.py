import os
import time

import click
import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import QwenImagePipeline, QwenImagePipelineConfig
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE
from telefuser.utils.utils import get_example_name


def configure_attention_backends():
    """Avoid cuDNN SDPA plan failures in the Qwen2.5-VL text encoder."""
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="qwen_image_t2i_bnb_nf4",
    model_root=TF_MODEL_ZOO_PATH + "/Qwen-Image-2512",
    dit_path_list=[
        "transformer/diffusion_pytorch_model-00001-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00004-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00005-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00006-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00007-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00008-of-00009.safetensors",
        "transformer/diffusion_pytorch_model-00009-of-00009.safetensors",
    ],
    vae_path_list=["vae/diffusion_pytorch_model.safetensors"],
    text_encoder_path_list=[
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
    ],
    tokenizer_path="tokenizer",
    negative_prompt=(
        "low resolution, low quality, distorted anatomy, malformed fingers, over saturated, "
        "waxy skin, missing facial details, over-smoothed, AI artifacts, messy composition, "
        "blurry text, distorted text"
    ),
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=42,
    sample_solver="euler",
    cfg_scale=1.0,
    num_inference_steps=16,
)


def get_pipeline(model_root=PPL_CONFIG["model_root"]):
    dit_paths = [os.path.join(model_root, p) for p in PPL_CONFIG["dit_path_list"]]
    vae_paths = [os.path.join(model_root, p) for p in PPL_CONFIG["vae_path_list"]]
    text_encoder_paths = [os.path.join(model_root, p) for p in PPL_CONFIG["text_encoder_path_list"]]
    tokenizer_path = os.path.join(model_root, PPL_CONFIG["tokenizer_path"])

    mm = ModuleManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_model(dit_paths, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(vae_paths, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_model(text_encoder_paths, device="cpu", torch_dtype=torch.bfloat16)
    mm.load_from_huggingface(tokenizer_path, "transformers", module_name="tokenizer")

    pipeline = QwenImagePipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = QwenImagePipelineConfig()
    pipe_config.dit_config.attention_config = AttentionConfig.dense_attention(PPL_CONFIG["attn_impl"])
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.NO_CPU_OFFLOAD
    pipe_config.dit_config.quant_config = QuantConfig(
        enabled=True,
        quant_type=QuantType.BNB_NF4,
        kernel_backend=QuantKernelBackend.BITSANDBYTES,
    )
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    pipeline.init(mm, pipe_config)
    return pipeline


def run(
    pipeline: QwenImagePipeline,
    prompt: str,
    aspect_ratio: str = "1:1",
    negative_prompt: str = PPL_CONFIG["negative_prompt"],
    seed: int = PPL_CONFIG["seed"],
    num_inference_steps: int = PPL_CONFIG["num_inference_steps"],
):
    height, width = ASPECT_RATIO_TO_SIZE[aspect_ratio]
    image = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
        rand_device="cpu",
        cfg_scale=PPL_CONFIG["cfg_scale"],
    )
    return image


@click.command()
@click.option("--aspect_ratio", "-ar", default="1:1", help="Image ratio such as 1:1, 16:9", type=str)
@click.option("--prompt", default="A cat playing piano", help="Custom prompt text")
@click.option("--negative_prompt", default=PPL_CONFIG["negative_prompt"], help="Negative prompt")
@click.option("--output", default=get_example_name(__file__, "png"), help="Output image filename")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
@click.option(
    "--num-inference-steps",
    "--num_inference_steps",
    default=PPL_CONFIG["num_inference_steps"],
    type=int,
    help="Number of denoising steps",
)
@click.option("--seed", default=PPL_CONFIG["seed"], type=int, help="Random seed")
def main(aspect_ratio, prompt, negative_prompt, output, model_root, num_inference_steps, seed):
    configure_attention_backends()
    pipeline = get_pipeline(model_root)
    s = time.time()
    images = run(
        pipeline,
        prompt,
        aspect_ratio,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=num_inference_steps,
    )
    print(f"pipe cost {time.time() - s} s")
    for i, image in enumerate(images):
        image.save(output.replace(".png", f"_{i}.png"))
    del pipeline


if __name__ == "__main__":
    main()


