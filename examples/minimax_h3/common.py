# SPDX-License-Identifier: Apache-2.0
"""Shared local-checkpoint loader and artifact writer for the MiniMax H3 examples."""

from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

import torch

from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    FeatureCacheConfig,
    ModelRuntimeConfig,
    OffloadConfig,
    ParallelConfig,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
    WeightOffloadType,
)
from telefuser.core.module_manager import ModuleManager
from telefuser.models.minimax_h3_audio_vae import MiniMaxH3AudioVAE
from telefuser.models.minimax_h3_dit import MiniMaxH3DiT
from telefuser.models.minimax_h3_encoder import MiniMaxH3Encoder
from telefuser.models.minimax_h3_video_vae import MiniMaxH3VideoVAE
from telefuser.pipelines.minimax_h3.pipeline import (
    MiniMaxH3Generation,
    MiniMaxH3Pipeline,
    MiniMaxH3PipelineConfig,
)
from telefuser.pipelines.minimax_h3.task_profiles import minimax_h3_task_profile, partition_for_task
from telefuser.pipelines.minimax_h3.time_request import minimax_h3_time_shift_sigmas
from telefuser.utils.audio import save_wav
from telefuser.utils.video import save_video

MINIMAX_H3_EXAMPLE_DATA = Path(__file__).resolve().parents[1] / "data" / "minimax-h3"
MINIMAX_H3_DEFAULT_FL2VA_IMAGE = MINIMAX_H3_EXAMPLE_DATA / "fl2va-reference.png"
MINIMAX_H3_DEFAULT_REF2VA_VIDEO = MINIMAX_H3_EXAMPLE_DATA / "ref2va-reference.mp4"
MINIMAX_H3_DEFAULT_REF2VA_AUDIO = MINIMAX_H3_EXAMPLE_DATA / "ref2va-voice.mp3"
MINIMAX_H3_DEFAULT_REQUEST = MINIMAX_H3_EXAMPLE_DATA / "ref2va.json"

_REQUEST_KEYS = frozenset(
    {
        "task",
        "prompt",
        "conditions",
        "target",
        "seed",
        "flow_shift",
        "audio_flow_shift",
        "num_inference_steps",
    }
)


def load_minimax_h3_request(request_path: str | Path) -> dict[str, Any]:
    """Load a JSON request and resolve relative material paths beside it."""
    path = Path(request_path).expanduser().resolve()
    with path.open(encoding="utf-8") as request_file:
        request = json.load(request_file)
    if not isinstance(request, dict):
        raise ValueError("MiniMax H3 request JSON must contain an object")
    unknown = set(request) - _REQUEST_KEYS
    if unknown:
        raise ValueError(f"MiniMax H3 request JSON has unknown fields: {sorted(unknown)}")
    missing = {"task", "prompt", "target"} - set(request)
    if missing:
        raise ValueError(f"MiniMax H3 request JSON is missing fields: {sorted(missing)}")

    resolved = dict(request)
    conditions = request.get("conditions", [])
    if not isinstance(conditions, list):
        raise ValueError("MiniMax H3 request conditions must be a list")
    resolved_conditions: list[Any] = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise ValueError(f"MiniMax H3 request conditions[{index}] must be an object")
        item = dict(condition)
        uri = item.get("uri")
        if isinstance(uri, str) and uri and not urllib.parse.urlparse(uri).scheme and not Path(uri).is_absolute():
            item["uri"] = str((path.parent / uri).resolve())
        resolved_conditions.append(item)
    resolved["conditions"] = resolved_conditions
    return resolved


def partition_for_minimax_h3_request(request: dict[str, Any]) -> str:
    """Return the original checkpoint partition required by a request."""
    return partition_for_task(request.get("task")).upper()


def run_minimax_h3_request(
    pipeline: MiniMaxH3Pipeline,
    request_path: str | Path,
    *,
    num_inference_steps: int | None = None,
    expected_partition: str | None = None,
) -> MiniMaxH3Generation:
    """Run a JSON request, optionally restricting it to one checkpoint partition."""
    request = load_minimax_h3_request(request_path)
    partition = partition_for_minimax_h3_request(request)
    if expected_partition is not None and partition != expected_partition.upper():
        raise ValueError(f"request requires {partition} checkpoint partition, expected {expected_partition.upper()}")
    if num_inference_steps is not None:
        request["num_inference_steps"] = num_inference_steps
    return pipeline(**request)


def minimax_h3_adaln_cache_timesteps(request: dict[str, Any], *, num_inference_steps: int | None = None) -> list[float]:
    """Return every unique AdaLN timestep needed by a canonical H3 request."""
    profile = minimax_h3_task_profile(str(request["task"]).lower())
    steps = int(request.get("num_inference_steps", 50) if num_inference_steps is None else num_inference_steps)
    video_shift = float(request.get("flow_shift") or profile.default_flow_shift)
    audio_shift = float(request.get("audio_flow_shift") or profile.default_audio_flow_shift)
    video_timesteps = [1.0 - sigma for sigma in minimax_h3_time_shift_sigmas(num_steps=steps, shift_scale=video_shift)]
    audio_timesteps = [1.0 - sigma for sigma in minimax_h3_time_shift_sigmas(num_steps=steps, shift_scale=audio_shift)]
    return sorted({*video_timesteps, *audio_timesteps, 0.999, 1.0})


def _checkpoint_shards(component: Path) -> list[str]:
    shards = sorted(str(path) for path in component.glob("model-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model safetensor shards found in {component}")
    return shards


def minimax_h3_quant_config(quantization: str | QuantType | None) -> QuantConfig:
    """Resolve a public MiniMax H3 online-quantization name to runtime config."""
    if quantization is None:
        return QuantConfig()
    if isinstance(quantization, str):
        normalized = quantization.strip().lower().replace("_", "-")
        names = {
            "torchao-fp8": QuantType.TORCHAO_FP8,
            "bnb-nf4": QuantType.BNB_NF4,
            "tf-kernel-fp8": QuantType.FP8,
        }
        try:
            quant_type = names[normalized]
        except KeyError as exc:
            raise ValueError("quantization must be 'torchao-fp8', 'tf-kernel-fp8', 'bnb-nf4', or None") from exc
    elif isinstance(quantization, QuantType):
        quant_type = quantization
    else:
        raise TypeError("quantization must be a string, QuantType, or None")

    backends = {
        QuantType.TORCHAO_FP8: QuantKernelBackend.TORCHAO,
        QuantType.BNB_NF4: QuantKernelBackend.BITSANDBYTES,
        QuantType.FP8: QuantKernelBackend.TF_KERNEL,
    }
    if quant_type not in backends:
        raise ValueError(f"MiniMax H3 does not support online quantization type {quant_type.name}")
    return QuantConfig(enabled=True, quant_type=quant_type, kernel_backend=backends[quant_type])


def load_minimax_h3_pipeline(
    model_root: str | Path,
    *,
    partition: str,
    device: str = "cuda:0",
    num_inference_steps: int = 50,
    ulysses_degree: int = 1,
    tp_degree: int = 1,
    text_encoder_tp_degree: int | None = None,
    enable_fsdp: bool | None = None,
    attn_impl: AttnImplType | str = AttnImplType.FLASH_ATTN_4,
    feature_cache_config: FeatureCacheConfig | None = None,
    adaln_cache_path: str | Path | None = None,
    online_adaln_cache: bool = False,
    quantization: str | QuantType | None = None,
) -> MiniMaxH3Pipeline:
    if adaln_cache_path is not None and online_adaln_cache:
        raise ValueError("Choose either adaln_cache_path or online_adaln_cache, not both.")

    if partition not in {"FL2VA", "Ref2VA"}:
        raise ValueError("partition must be 'FL2VA' or 'Ref2VA'")
    if ulysses_degree not in {1, 2, 4}:
        raise ValueError("ulysses_degree must be 1, 2, or 4")
    if tp_degree not in {1, 2, 4}:
        raise ValueError("tp_degree must be 1, 2, or 4")
    world_size = ulysses_degree * tp_degree
    if world_size not in {1, 2, 4}:
        raise ValueError("ulysses_degree * tp_degree must be 1, 2, or 4")
    resolved_encoder_tp = world_size if text_encoder_tp_degree is None else text_encoder_tp_degree
    if resolved_encoder_tp not in {1, 2, 4}:
        raise ValueError("text_encoder_tp_degree must be 1, 2, or 4")
    resolved_enable_fsdp = world_size == 4 and tp_degree == 1 if enable_fsdp is None else enable_fsdp
    if resolved_enable_fsdp and (world_size == 1 or tp_degree > 1):
        raise ValueError("enable_fsdp requires multi-GPU sequence parallelism without tensor parallelism")
    if (adaln_cache_path is not None or online_adaln_cache) and resolved_enable_fsdp:
        raise ValueError("AdaLN cache modes do not yet support FSDP deployment.")
    quant_config = minimax_h3_quant_config(quantization)
    if quant_config.enabled and world_size != 1:
        raise ValueError("MiniMax H3 online quantization currently requires a single-GPU profile")
    if quant_config.enabled and resolved_enable_fsdp:
        raise ValueError("MiniMax H3 online quantization cannot be combined with FSDP")
    if isinstance(attn_impl, str):
        try:
            attn_impl = AttnImplType[attn_impl]
        except KeyError as exc:
            raise ValueError(f"unsupported attention implementation: {attn_impl}") from exc
    component_root = Path(model_root) / partition
    if not component_root.is_dir():
        raise FileNotFoundError(f"MiniMax H3 partition not found: {component_root}")
    runtime_device = torch.device(device)
    if quant_config.enabled and runtime_device.type != "cuda":
        raise ValueError("MiniMax H3 online quantization requires a CUDA device")
    use_resident_modules = world_size > 1 or resolved_enable_fsdp
    resident_offload = OffloadConfig(
        offload_type=(
            WeightOffloadType.NO_CPU_OFFLOAD if use_resident_modules else WeightOffloadType.MODEL_CPU_OFFLOAD
        ),
        pin_cpu_memory=False,
    )
    dit_offload = (
        OffloadConfig(offload_type=WeightOffloadType.NO_CPU_OFFLOAD, pin_cpu_memory=False)
        if quant_config.enabled
        else resident_offload
    )
    text_parallel = (
        ParallelConfig(
            device_ids=list(range(resolved_encoder_tp)),
            tp_degree=resolved_encoder_tp,
            timeout=1800,
        )
        if resolved_encoder_tp > 1
        else ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ulysses_degree=world_size,
            enable_fsdp=resolved_enable_fsdp,
            timeout=1800,
        )
    )
    text_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.bfloat16,
        offload_config=resident_offload,
        parallel_config=text_parallel,
    )
    dit_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.bfloat16,
        offload_config=dit_offload,
        attention_config=AttentionConfig.dense_attention(attn_impl),
        feature_cache_config=feature_cache_config or FeatureCacheConfig(),
        quant_config=quant_config,
        parallel_config=ParallelConfig(
            device_ids=list(range(world_size)),
            sp_ulysses_degree=ulysses_degree,
            tp_degree=tp_degree,
            enable_fsdp=resolved_enable_fsdp,
            timeout=1800,
        ),
    )
    video_vae_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.float32,
        offload_config=resident_offload,
        parallel_config=ParallelConfig(
            device_ids=list(range(world_size)),
            tp_degree=world_size,
            timeout=1800,
        ),
    )
    audio_vae_runtime = ModelRuntimeConfig(
        device_type=runtime_device.type,
        device_id=runtime_device.index or 0,
        torch_dtype=torch.float32,
        offload_config=resident_offload,
    )

    manager = ModuleManager(device="cpu", torch_dtype=torch.bfloat16)
    transformer_dir = component_root / "transformer"
    manager.load_model(
        _checkpoint_shards(transformer_dir),
        device="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        name="minimax_h3_transformer",
        model_class=MiniMaxH3DiT,
        converter_kwargs={"config_path": transformer_dir / "config.json"},
    )
    if adaln_cache_path is not None or online_adaln_cache:
        transformer = manager.fetch_module("minimax_h3_transformer")
        if transformer is None:
            raise RuntimeError("MiniMax H3 transformer was not loaded for AdaLN cache activation.")
        if online_adaln_cache:
            transformer.enable_online_adaln_cache(partition=partition)
        else:
            transformer.load_inference_only_adaln(adaln_cache_path, expected_partition=partition)

    encoder_dir = component_root / "text_encoder"
    manager.load_model(
        _checkpoint_shards(encoder_dir),
        device="cpu",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        name="minimax_h3_text_encoder",
        model_class=MiniMaxH3Encoder,
        converter_kwargs={"config_path": encoder_dir},
    )
    video_vae_dir = component_root / "video_vae"
    manager.load_model(
        str(video_vae_dir / "source" / "model.safetensors"),
        device="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        name="minimax_h3_video_vae",
        model_class=MiniMaxH3VideoVAE,
        converter_kwargs={"config_path": video_vae_dir},
    )
    audio_vae_dir = component_root / "audio_vae"
    manager.load_model(
        str(audio_vae_dir / "model.safetensors"),
        device="cpu",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        name="minimax_h3_audio_vae",
        model_class=MiniMaxH3AudioVAE,
        converter_kwargs={"config_path": audio_vae_dir},
    )

    pipeline = MiniMaxH3Pipeline(device=device)
    pipeline.init(
        manager,
        MiniMaxH3PipelineConfig(
            processor_path=str(component_root / "processor"),
            text_encoder_config=text_runtime,
            dit_config=dit_runtime,
            video_vae_config=video_vae_runtime,
            audio_vae_config=audio_vae_runtime,
            num_inference_steps=num_inference_steps,
        ),
    )
    return pipeline


def save_generation(result: MiniMaxH3Generation, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = result.video[0].mul(255).clamp(0, 255).to(torch.uint8)
    waveform = result.audio[0]
    with tempfile.TemporaryDirectory() as directory:
        video_path = Path(directory) / "video.mp4"
        audio_path = Path(directory) / "audio.wav"
        save_wav(waveform, result.audio_sample_rate, str(audio_path))
        save_video(
            frames,
            str(video_path),
            fps=float(result.video_fps),
            quality=6,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(output),
            ],
            check=True,
            capture_output=True,
        )


__all__ = [
    "MINIMAX_H3_DEFAULT_FL2VA_IMAGE",
    "MINIMAX_H3_DEFAULT_REF2VA_AUDIO",
    "MINIMAX_H3_DEFAULT_REF2VA_VIDEO",
    "MINIMAX_H3_DEFAULT_REQUEST",
    "load_minimax_h3_pipeline",
    "load_minimax_h3_request",
    "minimax_h3_adaln_cache_timesteps",
    "minimax_h3_quant_config",
    "partition_for_minimax_h3_request",
    "run_minimax_h3_request",
    "save_generation",
]
