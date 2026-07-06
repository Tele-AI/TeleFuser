"""Wan2.2 14B Text-to-Video service pipeline without latent cache.

Service-mode counterpart of ``wan22_14b_text_to_video_h100.py`` without latent cache.
Exposes:
- ``get_pipeline`` for service startup
- ``run_with_file`` for TeleFuser PipelineService (must return dict with ``output_path``)
- ``CACHE_CONFIG`` for CacheServiceFactory config overrides

``CACHE_CONFIG["enable_latent_cache"]`` is False here, so the CacheSeek
lookup/resume/save lifecycle is not engaged.
"""

from __future__ import annotations

import os

import torch

from telefuser.core.config import AttentionConfig, AttnImplType, FeatureCacheConfig, WeightOffloadType
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.wan22_video import (
    Wan22VideoPipeline,
    Wan22VideoPipelineConfig,
)
from telefuser.utils.video import get_target_video_size_from_ratio, save_video

TF_MODEL_ZOO_PATH = os.environ.get("TF_MODEL_ZOO_PATH", "model_zoo")
PPL_CONFIG = dict(
    name="wan22_A14B_t2v_service",
    model_root="/storage/model_zoo/Wan2.2-T2V-A14B",
    negative_prompt="Overly saturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, frame, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static frames, cluttered background, three legs, crowded background, walking backwards",
    num_inference_steps=40,
    num_frames=81,
    resolution="720p",
    aspect_ratio="16:9",
    cfg_scale_high=5.0,
    cfg_scale_low=5.0,
    seed=42,
    tiled=False,
    sigma_shift=5.0,
    boundary=0.9,
    sample_solver="euler",
    attn_impl=AttnImplType.TORCH_SDPA,
    dit_high_path_list="high_noise_model/diffusion_pytorch_model-0000*-of-00006.safetensors",
    dit_low_path_list="low_noise_model/diffusion_pytorch_model-0000*-of-00006.safetensors",
    enable_feature_cache_dit_high=True,
    enable_feature_cache_dit_low=True,
    model_type="Wan2_2-T2V-A14B",
    target_fps=16,
)

CACHE_CONFIG = dict(
    enable_latent_cache=False,
    latent_cache_dir=os.getenv("TELEFUSER_LATENT_CACHE_DIR", "./latent_cache/wan22_t2v"),
    # Cache is disabled for this pipeline variant; the remaining values are
    # kept aligned with the cache-enabled sibling for easy toggling.
    cache_mode="write_only",
    # KV store: Fluxon is stubbed in MVP -> use local file backend.
    kv_store_type="local_file",
    # Vector store: Qdrant is stubbed in MVP -> use faiss backend.
    vector_store_type="faiss",
    # Qwen3-VL-Embedding-2B hidden_size=2048. connection.py default is 512 (too small).
    # MUST match encoder output dim or FAISSVectorStore.search raises dim mismatch.
    vector_dim=2048,
    # Steps to snapshot. Aligned with the dataset design notes.
    # (5 mid-to-late steps, no step 0).
    key_steps=[5, 10, 15, 20, 25],
    # Video embedding: required by VideoBasedApproximateCache.save (else rollback).
    video_embedding_enabled=True,
    video_embedding_model_path=os.getenv("QWEN3VL_EMBEDDING_PATH", ""),
    video_embedding_max_frames=16,
    # CacheConfig defaults assume 4 visible GPUs (text=1, video=2, rerank=3).
    # Under CUDA_VISIBLE_DEVICES=2,3 the logical range is 0,1 only -> override.
    # Both encoders colocated on logical 1 (GPU 3) so strategies.py can share
    # a single Qwen3VLEncoder instance for text+video (saves 5GB, see
    # strategies.py video_encoder sharing branch). Reranker takes logical 0
    # (GPU 2, alone with DiT rank 0); shared encoder + DiT rank 1 on logical 1.
    text_embedding_device_id=1,
    video_embedding_device_id=1,
    video_vector_collection="video",
    # Reranker: Qwen3-VL-Reranker-2B is a text-only cross-encoder (score_mm over
    # {query_text, candidate_texts}). Adds ~4GB to logical GPU 1, shared with
    # video_encoder; together they use about 22GB on an 80GB H100.
    rerank_enabled=True,
    rerank_model_path=os.getenv("QWEN3VL_RERANKER_PATH", "/storage/model_zoo/Qwen3-VL-Reranker-2B"),
    # Under parallelism=2 + CVD=2,3, logical 1 already has video_encoder + dit rank 1,
    # putting reranker there too overflows GPU 3 (~80GB H100). Default to logical 0
    # (GPU 2, shared with prompt_encoder + dit rank 0, ~14GB headroom remaining).
    # Override via env TELEFUSER_RERANK_DEVICE_ID when running parallelism=4 etc.
    rerank_device_id=int(os.getenv("TELEFUSER_RERANK_DEVICE_ID", "0")),
    rerank_top_k=5,
    # Used by _determine_skip_step when rerank_enabled=True (rerank score path,
    # strategies.py:361-364). bf16 fp noise gives sim~0.87 for identical prompts
    # via vector, but rerank cross-encoder is usually tighter; 0.85 leaves room.
    rerank_score_threshold=0.85,
)


def _sample_indices(total: int, max_frames: int) -> list[int]:
    if total <= 0:
        return []
    max_frames = max(1, int(max_frames or 1))
    if total <= max_frames:
        return list(range(total))
    step = float(total) / float(max_frames)
    return [min(int(i * step), total - 1) for i in range(max_frames)]


def _sample_video_frames(video_frames, max_frames: int | None = None):
    """Sample representative frames from the output video for embedding."""
    if video_frames is None:
        return []
    if max_frames is None:
        max_frames = CACHE_CONFIG.get("video_embedding_max_frames", 16)
    total = len(video_frames)
    if total <= 0:
        return []
    indices = _sample_indices(total, max_frames)
    return [video_frames[idx] for idx in indices if 0 <= idx < total]


# build_latent_data ppl-file hook removed (kept consistent
# with the cache-enabled sibling file). enable_latent_cache=False here, so the
# adapter path is never engaged anyway, but removing the dead hook keeps the
# two ppl files structurally aligned.


def get_pipeline(parallelism: int = 1, model_root: str | None = None):
    """Build Wan22VideoPipeline for service startup.

    Args:
        parallelism: Number of parallel GPUs (1/2/4/8).
        model_root: Override for ``PPL_CONFIG["model_root"]``.
    """
    ppl_config = PPL_CONFIG
    model_root = model_root or ppl_config["model_root"]

    module_manager = ModuleManager(device="cpu")
    module_manager.load_model(f"{model_root}/Wan2.1_VAE.pth", torch_dtype=torch.bfloat16)
    module_manager.load_model(
        os.path.join(model_root, ppl_config["dit_high_path_list"]),
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_model(
        os.path.join(model_root, ppl_config["dit_low_path_list"]),
        torch_dtype=torch.bfloat16,
    )
    module_manager.load_model(
        f"{model_root}/models_t5_umt5-xxl-enc-bf16.pth",
        torch_dtype=torch.bfloat16,
    )

    pipe = Wan22VideoPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe_config = Wan22VideoPipelineConfig()
    pipe_config.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_high_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_low_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    pipe_config.dit_high_config.attention_config = AttentionConfig.dense_attention(ppl_config["attn_impl"])
    pipe_config.dit_low_config.attention_config = AttentionConfig.dense_attention(ppl_config["attn_impl"])
    pipe_config.sample_solver = ppl_config["sample_solver"]

    if ppl_config.get("enable_feature_cache_dit_high", False):
        pipe_config.dit_high_config.feature_cache_config = FeatureCacheConfig(
            enabled=True, model_type=ppl_config["model_type"]
        )
    if ppl_config.get("enable_feature_cache_dit_low", False):
        pipe_config.dit_low_config.feature_cache_config = FeatureCacheConfig(
            enabled=True, model_type=ppl_config["model_type"]
        )

    if parallelism > 1:
        cfg_scale_high = ppl_config["cfg_scale_high"]
        cfg_scale_low = ppl_config["cfg_scale_low"]
        if cfg_scale_high > 1:
            pipe_config.dit_high_config.parallel_config.cfg_degree = 2
            pipe_config.dit_high_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_high_config.parallel_config.sp_ulysses_degree = parallelism
        if cfg_scale_low > 1:
            pipe_config.dit_low_config.parallel_config.cfg_degree = 2
            pipe_config.dit_low_config.parallel_config.sp_ulysses_degree = parallelism // 2
        else:
            pipe_config.dit_low_config.parallel_config.sp_ulysses_degree = parallelism
        pipe_config.dit_high_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.dit_low_config.parallel_config.device_ids = list(range(parallelism))
        pipe_config.enable_denoising_parallel = True

    pipe.init(module_manager, pipe_config)
    return pipe


def run_with_file(pipeline, **task_data) -> dict:
    """Service entrypoint invoked by PipelineService.

    Returns a dict with ``output_path`` (required) and optionally
    ``latent_payload`` (consumed by task_service's post-inference hook).

    ``**task_data`` is preferred over explicit args: this guarantees
    ``latent_data`` (injected by task_service's pre-inference hook) survives
    ``_select_kwargs`` signature filtering in pipeline_runner.py.
    """
    prompt = task_data["prompt"]
    output_path = task_data["output_path"]
    negative_prompt = task_data.get("negative_prompt", "") or ""
    seed = int(task_data.get("seed", PPL_CONFIG["seed"]))
    resolution = task_data.get("resolution") or PPL_CONFIG["resolution"]
    aspect_ratio = task_data.get("aspect_ratio") or PPL_CONFIG["aspect_ratio"]
    latent_data = task_data.get("latent_data")

    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=16,
        width_division_factor=16,
    )

    result = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}",
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale_high=PPL_CONFIG["cfg_scale_high"],
        cfg_scale_low=PPL_CONFIG["cfg_scale_low"],
        seed=seed,
        tiled=PPL_CONFIG["tiled"],
        height=height,
        width=width,
        sigma_shift=PPL_CONFIG["sigma_shift"],
        boundary=PPL_CONFIG["boundary"],
        latent_data=latent_data,
    )

    latent_payload: dict | None = None
    if isinstance(result, tuple):
        frames, latent_payload = result
    else:
        frames = result

    # Back-fill embedding_video_frames so VideoBasedApproximateCache.save
    # can upsert to vector_store without rolling back the KV write.
    if latent_payload is not None:
        sampled = _sample_video_frames(frames)
        if sampled:
            latent_payload["embedding_video_frames"] = sampled

    save_video(frames, output_path, fps=PPL_CONFIG["target_fps"], quality=6)

    ret: dict = {"output_path": str(output_path)}
    if latent_payload is not None:
        ret["latent_payload"] = latent_payload
    return ret
