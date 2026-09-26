import os
import time

import click
import torch

from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.qwen_image import (
    QwenImagePipeline,
    QwenImagePipelineConfig,
)
from telefuser.pipelines.qwen_image.qwen_image import ASPECT_RATIO_TO_SIZE
from telefuser.utils.utils import get_example_name

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
DEFAULT_PROMPT = (
    "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and "
    "lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied "
    "in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, "
    "cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at "
    "an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no "
    "staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming "
    "with vivid, fresh, youthful charm."
)
PPL_CONFIG = dict(
    name="qwen_image_t2i",
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
    vae_path_list=[
        "vae/diffusion_pytorch_model.safetensors",
    ],
    text_encoder_path_list=[
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
    ],
    tokenizer_path="tokenizer",
    negative_prompt=(
        "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，"
        "画面具有AI感。构图混乱。文字模糊，扭曲。"
    ),
    attn_impl=AttnImplType.TORCH_SDPA,
    seed=42,
    sample_solver="euler",
    cfg_scale=4.0,
    num_inference_steps=50,
    enable_feature_cache=False,
    model_type="Qwen-Image-2512",
)


def get_pipeline(parallelism=1, model_root=PPL_CONFIG["model_root"]):
    """Load Qwen-Image pipeline.

    Args:
        parallelism: Number of parallel GPUs (REQUIRED)
        model_root: Root directory of model weights (REQUIRED)

    Returns:
        Initialized QwenImagePipeline
    """
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
    pipe_config.dit_config.offload_config.offload_type = WeightOffloadType.ASYNC_CPU_OFFLOAD
    pipe_config.dit_config.offload_config.offload_ratio = 0.5
    pipe_config.sample_solver = PPL_CONFIG["sample_solver"]
    if PPL_CONFIG["enable_feature_cache"]:
        pipe_config.dit_config.feature_cache_config.enabled = True
        pipe_config.dit_config.feature_cache_config.model_type = PPL_CONFIG["model_type"]
    if parallelism > 1:
        pipe_config.dit_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_config.parallel_config.cfg_degree = parallelism
        pipe_config.enable_denoising_parallel = True
        pipe_config.text_encoding_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.text_encoding_config.parallel_config.tp_degree = parallelism
        pipe_config.enable_text_encoding_parallel = True
    pipeline.init(mm, pipe_config)
    return pipeline


def run(
    pipeline: QwenImagePipeline,
    prompt,
    aspect_ratio: str = "16:9",
    negative_prompt=PPL_CONFIG["negative_prompt"],
    seed=PPL_CONFIG["seed"],
):
    width, height = ASPECT_RATIO_TO_SIZE[aspect_ratio]
    image = pipeline(
        prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        seed=seed,
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        rand_device="cuda",
        cfg_scale=PPL_CONFIG["cfg_scale"],
    )
    return image


@click.command()
@click.option("--aspect_ratio", "-ar", default="16:9", help="Image ratio such as 1:1, 16:9", type=str)
@click.option("--gpu_num", default=1, help="Number of GPUs to use", type=int)
@click.option(
    "--prompt",
    default=DEFAULT_PROMPT,
    help="Custom prompt text",
)
@click.option("--negative_prompt", default=PPL_CONFIG["negative_prompt"], help="Negative prompt")
@click.option("--output", default=get_example_name(__file__, "png"), help="Output image filename")
@click.option("--model_root", default=PPL_CONFIG["model_root"], help="Model root directory")
def main(aspect_ratio, gpu_num, prompt, negative_prompt, output, model_root):
    pipeline = get_pipeline(gpu_num, model_root)
    # Warm up
    images = run(pipeline, prompt, aspect_ratio, negative_prompt=negative_prompt)
    # Timing run
    s = time.time()
    images = run(pipeline, prompt, aspect_ratio, negative_prompt=negative_prompt)
    print(f"pipe cost {time.time() - s} s")
    for i, image in enumerate(images):
        image.save(output.replace(".png", f"_{i}.png"))
    del pipeline


if __name__ == "__main__":
    main()
