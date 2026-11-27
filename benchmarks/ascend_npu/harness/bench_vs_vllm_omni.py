#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gates G1-G3 (Container): Benchmark TeleFuser / vLLM-Omni, verify media & parity, and emit contact sheets."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image, ImageDraw

WAN22_DEFAULT_PROMPT = "A red kite flying over a green field"
MINIMAX_H3_DEFAULT_PROMPT = (
    "In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; "
    "footsteps crunch in the snow while the creature breathes and softly snorts."
)
WAN22_DEFAULT_NEG = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _sync_npu() -> None:
    if torch.npu.is_available():
        torch.npu.synchronize()


def _ffprobe_check(
    mp4_path: Path,
    *,
    expected_w: int,
    expected_h: int,
    expected_frames: int,
    expect_audio: bool,
) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,sample_rate,channels,duration",
        "-of",
        "json",
        str(mp4_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    vstreams = [s for s in streams if s.get("codec_type") == "video"]
    astreams = [s for s in streams if s.get("codec_type") == "audio"]
    assert vstreams, f"No video stream in {mp4_path}"
    vs = vstreams[0]
    w, h = int(vs["width"]), int(vs["height"])
    frames = int(vs["nb_read_frames"])
    assert (w, h) == (expected_w, expected_h), f"Video size {(w, h)} != {(expected_w, expected_h)}"
    assert frames == expected_frames, f"Frame count {frames} != {expected_frames}"
    if expect_audio:
        assert astreams, f"Expected synchronized audio stream in {mp4_path}, found none"
        vdur = float(vs.get("duration") or (frames / 24.0))
        adur = float(astreams[0].get("duration") or 0.0)
        assert abs(vdur - adur) < 0.08, f"AV sync drift too large: video={vdur:.3f}s audio={adur:.3f}s"
    return {"video_stream": vs, "audio_stream": astreams[0] if astreams else None}


def _save_contact_sheet(frames_rgb: np.ndarray, out_png: Path, title: str) -> None:
    total = frames_rgb.shape[0]
    indices = [0, total // 4, total // 2, (3 * total) // 4, total - 1]
    thumbs = []
    for idx in indices:
        im = Image.fromarray(frames_rgb[idx])
        im.thumbnail((320, 180))
        draw = ImageDraw.Draw(im)
        draw.rectangle([(0, 0), (92, 22)], fill=(0, 0, 0))
        draw.text((6, 4), f"Frame {idx}", fill=(255, 255, 255))
        thumbs.append(im)
    tw, th = thumbs[0].size
    canvas = Image.new("RGB", (tw * len(thumbs), th + 28), color=(18, 22, 30))
    header = ImageDraw.Draw(canvas)
    header.text((8, 6), title, fill=(230, 236, 245))
    for i, im in enumerate(thumbs):
        canvas.paste(im, (i * tw, 28))
    canvas.save(out_png)


def _compute_parity(frames_a: np.ndarray, frames_b: np.ndarray) -> dict[str, float]:
    diff = frames_a.astype(np.float32) - frames_b.astype(np.float32)
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    psnr = 99.0 if mse <= 1e-8 else float(10.0 * np.log10((255.0 * 255.0) / mse))
    return {"mae_255": round(mae, 3), "psnr_db": round(psnr, 2)}


def _compute_audio_cosine(audio_a: np.ndarray, audio_b: np.ndarray) -> float:
    flat_a = audio_a.astype(np.float64).reshape(audio_a.shape[0], -1)
    flat_b = audio_b.astype(np.float64).reshape(audio_b.shape[0], -1)
    min_len = min(flat_a.shape[-1], flat_b.shape[-1])
    va = flat_a[:, :min_len].reshape(-1)
    vb = flat_b[:, :min_len].reshape(-1)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return 0.0
    return round(float(np.dot(va, vb) / denom), 4)


def _write_stereo_wav(waveform_2ch: np.ndarray, sample_rate: int, wav_path: Path) -> None:
    clipped = np.clip(waveform_2ch.T, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).round().astype("<i2")
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(int(waveform_2ch.shape[0]))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def _build_telefuser_wan22(cards: int, model_root: str) -> Any:
    from telefuser.core.config import AttentionConfig, AttnImplType, WeightOffloadType
    from telefuser.core.module_manager import ModuleManager
    from telefuser.pipelines.wan_video.wan22_ti2v import Wan22TI2VPipeline, Wan22TI2VPipelineConfig
    from telefuser.platforms import current_platform, resolve_device

    resolve_device(f"npu:{cards - 1}")
    mm = ModuleManager(device="cpu")
    mm.load_model(os.path.join(model_root, "Wan2.2_VAE.pth"), torch_dtype=torch.bfloat16)
    shards = [os.path.join(model_root, f"diffusion_pytorch_model-0000{i}-of-00003.safetensors") for i in (1, 2, 3)]
    mm.load_model(shards, torch_dtype=torch.bfloat16)
    mm.load_model(os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"), torch_dtype=torch.bfloat16)

    pipe = Wan22TI2VPipeline(device=current_platform.device_type, torch_dtype=torch.bfloat16)
    cfg = Wan22TI2VPipelineConfig()
    cfg.text_encoding_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    cfg.vae_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    cfg.dit_config.offload_config.offload_type = WeightOffloadType.MODEL_CPU_OFFLOAD
    cfg.dit_config.attention_config = AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA)
    cfg.sample_solver = "unipc"
    if cards > 1:
        cfg.dit_config.parallel_config.cfg_degree = 2
        cfg.dit_config.parallel_config.sp_ulysses_degree = cards // 2
        cfg.dit_config.parallel_config.device_ids = list(range(cards))
        cfg.enable_denoising_parallel = True
    pipe.init(mm, cfg)
    return pipe


def _frames_to_rgb_array(video_obj: Any) -> np.ndarray:
    if isinstance(video_obj, np.ndarray):
        arr = video_obj
    elif torch.is_tensor(video_obj):
        arr = video_obj.detach().cpu().numpy()
    elif isinstance(video_obj, list):
        if len(video_obj) == 1 and isinstance(video_obj[0], (np.ndarray, torch.Tensor)):
            first = video_obj[0]
            arr = first.detach().cpu().numpy() if torch.is_tensor(first) else np.asarray(first)
        else:
            arr = np.stack([np.asarray(fr) for fr in video_obj], axis=0)
    else:
        raise TypeError(f"Unsupported video container type: {type(video_obj)}")
    while arr.ndim > 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 3, 0))
    elif arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if np.issubdtype(arr.dtype, np.floating):
        assert np.isfinite(arr).all(), "Detected NaN or Inf in generated frames!"
        if arr.min() < -0.05:
            arr = ((np.clip(arr, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
        else:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _run_telefuser_wan22(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    from telefuser.utils.video import save_video

    t_init0 = time.perf_counter()
    pipe = _build_telefuser_wan22(args.cards, args.model_root)
    init_s = time.perf_counter() - t_init0

    timings: list[float] = []
    hashes: list[str] = []
    last_rgb: np.ndarray | None = None
    for run_idx in range(1 + args.warm_runs):
        _sync_npu()
        t0 = time.perf_counter()
        video = pipe(
            prompt=args.prompt,
            negative_prompt=f" {WAN22_DEFAULT_NEG}",
            num_inference_steps=args.steps,
            num_frames=args.frames,
            cfg_scale=5.0,
            seed=args.seed,
            tiled=False,
            height=args.height,
            width=args.width,
            sigma_shift=5.0,
        )
        _sync_npu()
        elapsed = time.perf_counter() - t0
        rgb = _frames_to_rgb_array(video)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        timings.append(round(elapsed, 3))
        hashes.append(digest)
        last_rgb = rgb
        phase = "cold" if run_idx == 0 else f"warm_{run_idx}"
        print(f"[TeleFuser] run={phase} seconds={elapsed:.3f} rgb_sha256={digest[:16]}", flush=True)

    assert last_rgb is not None
    motion_std = float(np.std(last_rgb[-1].astype(np.float32) - last_rgb[0].astype(np.float32)))
    assert motion_std > 1.0, f"Static/blank video detected (motion_std={motion_std:.4f})"

    mp4_path = out_dir / f"telefuser_wan22_{args.cards}card.mp4"
    save_video([Image.fromarray(frame) for frame in last_rgb], str(mp4_path), fps=24, quality=6)
    npy_path = out_dir / f"telefuser_wan22_{args.cards}card_mid.npy"
    np.save(npy_path, last_rgb[last_rgb.shape[0] // 2])
    warm_med = round(statistics.median(timings[1:] if timings[1:] else timings), 2)
    title = f"TeleFuser Wan2.2-TI2V-5B | {args.cards}x {args.chip_model} | Warm={warm_med:.2f}s"
    _save_contact_sheet(last_rgb, out_dir / f"contact_sheet_telefuser_{args.cards}card.png", title)
    probe = _ffprobe_check(
        mp4_path,
        expected_w=args.width,
        expected_h=args.height,
        expected_frames=args.frames,
        expect_audio=False,
    )
    del pipe
    return {
        "init_s": round(init_s, 2),
        "cold_s": timings[0],
        "warm_runs_s": timings[1:],
        "warm_median_s": warm_med,
        "rgb_hashes": hashes,
        "deterministic_warm": len(set(hashes[1:])) <= 1 if len(hashes) > 1 else True,
        "mp4_path": mp4_path.name,
        "mid_frame_npy": npy_path.name,
        "ffprobe": probe,
    }


def _run_vllm_omni_wan22(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    import imageio.v3 as iio
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni_kwargs: dict[str, Any] = {
        "model": args.omni_model_root,
        "enable_cpu_offload": True,
        "vae_use_tiling": False,
    }
    if args.cards > 1:
        cfg_deg = 2
        sp_deg = max(1, args.cards // 2)
        omni_kwargs["cfg_parallel_size"] = cfg_deg
        omni_kwargs["ulysses_degree"] = sp_deg
        omni_kwargs["ulysses_mode"] = "advanced_uaa" if sp_deg > 1 else "strict"

    t_init0 = time.perf_counter()
    engine = Omni(**omni_kwargs)
    init_s = time.perf_counter() - t_init0

    timings: list[float] = []
    hashes: list[str] = []
    last_rgb: np.ndarray | None = None
    for run_idx in range(1 + args.warm_runs):
        _sync_npu()
        t0 = time.perf_counter()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        sampling = OmniDiffusionSamplingParams(
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
            generator=gen,
        )
        outputs = engine.generate(
            {"prompt": args.prompt, "negative_prompt": f" {WAN22_DEFAULT_NEG}"},
            sampling,
        )
        _sync_npu()
        elapsed = time.perf_counter() - t0
        first_out = outputs[0]
        raw_frames = (
            getattr(first_out, "images", None)
            or getattr(first_out, "frames", None)
            or first_out.request_output[0].images
        )
        rgb = _frames_to_rgb_array(raw_frames)
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        timings.append(round(elapsed, 3))
        hashes.append(digest)
        last_rgb = rgb
        phase = "cold" if run_idx == 0 else f"warm_{run_idx}"
        print(f"[vLLM-Omni] run={phase} seconds={elapsed:.3f} rgb_sha256={digest[:16]}", flush=True)

    assert last_rgb is not None
    mp4_path = out_dir / f"vllm_omni_wan22_{args.cards}card.mp4"
    iio.imwrite(mp4_path, last_rgb, fps=24, codec="libx264")
    npy_path = out_dir / f"vllm_omni_wan22_{args.cards}card_mid.npy"
    np.save(npy_path, last_rgb[last_rgb.shape[0] // 2])
    warm_med = round(statistics.median(timings[1:] if timings[1:] else timings), 2)
    title = f"vLLM-Omni Wan2.2-TI2V-5B | {args.cards}x {args.chip_model} | Warm={warm_med:.2f}s"
    _save_contact_sheet(last_rgb, out_dir / f"contact_sheet_vllm_omni_{args.cards}card.png", title)
    probe = _ffprobe_check(
        mp4_path,
        expected_w=args.width,
        expected_h=args.height,
        expected_frames=args.frames,
        expect_audio=False,
    )
    if hasattr(engine, "close"):
        engine.close()
    return {
        "init_s": round(init_s, 2),
        "cold_s": timings[0],
        "warm_runs_s": timings[1:],
        "warm_median_s": warm_med,
        "rgb_hashes": hashes,
        "deterministic_warm": len(set(hashes[1:])) <= 1 if len(hashes) > 1 else True,
        "mp4_path": mp4_path.name,
        "mid_frame_npy": npy_path.name,
        "ffprobe": probe,
    }


def _run_telefuser_minimax_h3(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    from examples.minimax_h3.common import save_generation
    from examples.minimax_h3.minimax_h3_fl2va_h100 import get_pipeline, run

    t_init0 = time.perf_counter()
    pipe = get_pipeline(
        args.cards,
        args.model_root,
        device="npu:0",
        num_inference_steps=args.steps,
    )
    init_s = time.perf_counter() - t_init0

    timings: list[float] = []
    denoise_timings: list[float] = []
    hashes: list[str] = []
    peak_alloc_mb = 0
    peak_res_mb = 0
    last_gen = None
    last_rgb: np.ndarray | None = None
    last_audio: np.ndarray | None = None
    try:
        for run_idx in range(1 + args.warm_runs):
            _sync_npu()
            t0 = time.perf_counter()
            gen = run(
                pipe,
                prompt=args.prompt,
                seed=args.seed,
                aspect_ratio="16:9",
                target_video_length=args.duration,
                task="t2v",
                mode="t2va",
                flow_shift=12.0,
                audio_flow_shift=3.0,
            )
            _sync_npu()
            elapsed = time.perf_counter() - t0
            rgb = _frames_to_rgb_array(gen.video[0])
            audio_np = gen.audio[0].detach().float().cpu().numpy()
            assert np.isfinite(audio_np).all(), "Detected NaN or Inf in generated audio!"
            digest = hashlib.sha256(rgb.tobytes()).hexdigest()
            rm = gen.runtime_metrics
            denoise_s = float(rm.get("denoising_stage_seconds") or rm.get("denoising_seconds") or 0.0)
            peak_alloc_mb = max(peak_alloc_mb, int(round(float(rm.get("peak_allocated_bytes", 0)) / (1024 * 1024))))
            peak_res_mb = max(peak_res_mb, int(round(float(rm.get("peak_reserved_bytes", 0)) / (1024 * 1024))))
            timings.append(round(elapsed, 3))
            denoise_timings.append(round(denoise_s, 3))
            hashes.append(digest)
            last_gen = gen
            last_rgb = rgb
            last_audio = audio_np
            phase = "cold" if run_idx == 0 else f"warm_{run_idx}"
            print(
                f"[TeleFuser H3] run={phase} total={elapsed:.3f}s denoise={denoise_s:.3f}s "
                f"rgb_sha256={digest[:16]}",
                flush=True,
            )
    finally:
        pipe.stop()

    assert last_gen is not None and last_rgb is not None and last_audio is not None
    motion_std = float(np.std(last_rgb[-1].astype(np.float32) - last_rgb[0].astype(np.float32)))
    assert motion_std > 1.0, f"Static/blank video detected (motion_std={motion_std:.4f})"
    audio_rms = float(np.sqrt(np.mean(np.square(last_audio.astype(np.float64)))))
    assert audio_rms > 1e-4, f"Silent audio detected (audio_rms={audio_rms:.6f})"

    mp4_path = out_dir / f"telefuser_minimax_h3_{args.cards}card.mp4"
    save_generation(last_gen, mp4_path)
    npy_path = out_dir / f"telefuser_minimax_h3_{args.cards}card_mid.npy"
    audio_npy_path = out_dir / f"telefuser_minimax_h3_{args.cards}card_audio.npy"
    np.save(npy_path, last_rgb[last_rgb.shape[0] // 2])
    np.save(audio_npy_path, last_audio)
    warm_med = round(statistics.median(timings[1:] if timings[1:] else timings), 2)
    denoise_med = round(statistics.median(denoise_timings[1:] if denoise_timings[1:] else denoise_timings), 2)
    title = f"TeleFuser MiniMax-H3 T2VA | {args.cards}x {args.chip_model} | Warm={warm_med:.2f}s"
    _save_contact_sheet(last_rgb, out_dir / f"contact_sheet_telefuser_minimax_h3_{args.cards}card.png", title)
    probe = _ffprobe_check(
        mp4_path,
        expected_w=int(last_rgb.shape[2]),
        expected_h=int(last_rgb.shape[1]),
        expected_frames=int(last_rgb.shape[0]),
        expect_audio=True,
    )
    return {
        "init_s": round(init_s, 2),
        "cold_s": timings[0],
        "warm_runs_s": timings[1:],
        "warm_median_s": warm_med,
        "denoise_s": denoise_med,
        "non_denoise_s": round(max(0.0, warm_med - denoise_med), 2),
        "num_frames": int(last_rgb.shape[0]),
        "audio_rms": round(audio_rms, 5),
        "peak_allocated_mb": peak_alloc_mb,
        "peak_reserved_mb": peak_res_mb,
        "rgb_hashes": hashes,
        "deterministic_warm": len(set(hashes[1:])) <= 1 if len(hashes) > 1 else True,
        "mp4_path": mp4_path.name,
        "mid_frame_npy": npy_path.name,
        "audio_npy": audio_npy_path.name,
        "ffprobe": probe,
    }


def _raise_nofile_limit(target: int = 65536) -> None:
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = target if hard == resource.RLIM_INFINITY else min(hard, target)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except (ImportError, OSError, ValueError):
        pass


def _run_vllm_omni_minimax_h3(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    import imageio.v3 as iio
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    _raise_nofile_limit()
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", "4000")
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("MINDIE_SD_FA_TYPE", "ascend_laser_attention")
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")

    omni_kwargs: dict[str, Any] = {
        "model": args.omni_model_root,
        "trust_remote_code": True,
        "task_type": "fl2va",
        "num_gpus": args.cards,
        "ulysses_degree": args.cards,
        "ring_degree": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "text_encoder_tp_size": args.cards,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        "vae_patch_parallel_size": args.cards,
        "enable_distributed_layerwise_offload": True,
        "dlo_use_allgather": args.cards > 1,
        "enable_diffusion_pipeline_profiler": True,
        "diffusion_attention_config": {"default": {"backend": "FLASH_ATTN"}},
        "stage_init_timeout": 1800.0,
        "init_timeout": 1800.0,
    }

    async def _run_async() -> tuple[float, list[float], list[float], list[str], np.ndarray, np.ndarray, float]:
        t_init0 = time.perf_counter()
        engine = AsyncOmni(**omni_kwargs)
        init_elapsed = time.perf_counter() - t_init0
        timings_local: list[float] = []
        denoise_local: list[float] = []
        hashes_local: list[str] = []
        final_rgb: np.ndarray | None = None
        final_audio: np.ndarray | None = None
        peak_mb = 0.0
        try:
            for run_idx in range(1 + args.warm_runs):
                params = copy.deepcopy(engine.default_sampling_params_list)
                diff = params[0]
                diff.width = args.width
                diff.height = args.height
                diff.fps = 24
                diff.num_inference_steps = args.steps
                diff.seed = args.seed
                diff.extra_args = {
                    "task": "t2va",
                    "duration": args.duration,
                    "aspect_ratio": "16:9",
                    "flow_shift": 12.0,
                    "audio_flow_shift": 3.0,
                }
                _sync_npu()
                t0 = time.perf_counter()
                output = None
                async for item in engine.generate(
                    prompt=args.prompt,
                    request_id=f"bench-{run_idx}",
                    sampling_params_list=params,
                ):
                    if item.finished:
                        output = item
                _sync_npu()
                elapsed = time.perf_counter() - t0
                assert output is not None and output.images, f"Empty output on run {run_idx}"
                rgb = _frames_to_rgb_array(output.images[0])
                raw_audio = np.asarray(output.multimodal_output.get("audio"), dtype=np.float32)
                if raw_audio.ndim == 3 and raw_audio.shape[0] == 1:
                    raw_audio = raw_audio[0]
                assert np.isfinite(raw_audio).all(), "Detected NaN/Inf in vLLM-Omni audio!"
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                stage_durs = getattr(output, "stage_durations", None) or {}
                denoise_val = float(
                    stage_durs.get("MiniMaxH3Pipeline.diffuse")
                    or stage_durs.get("diffuse")
                    or stage_durs.get("denoise")
                    or stage_durs.get("denoising")
                    or stage_durs.get("dit")
                    or 0.0
                )
                peak_mb = max(peak_mb, float(getattr(output, "peak_memory_mb", 0.0) or 0.0))
                timings_local.append(round(elapsed, 3))
                denoise_local.append(round(denoise_val, 3))
                hashes_local.append(digest)
                final_rgb = rgb
                final_audio = raw_audio
                phase = "cold" if run_idx == 0 else f"warm_{run_idx}"
                print(
                    f"[vLLM-Omni H3] run={phase} total={elapsed:.3f}s denoise={denoise_val:.3f}s "
                    f"stage_durations={stage_durs} rgb_sha256={digest[:16]}",
                    flush=True,
                )
        finally:
            shutdown = getattr(engine, "shutdown", None) or getattr(engine, "close", None)
            if callable(shutdown):
                res_close = shutdown()
                if asyncio.iscoroutine(res_close):
                    await res_close
        assert final_rgb is not None and final_audio is not None
        return init_elapsed, timings_local, denoise_local, hashes_local, final_rgb, final_audio, peak_mb

    init_s, timings, denoise_timings, hashes, last_rgb, last_audio, peak_mb = asyncio.run(_run_async())
    audio_rms = float(np.sqrt(np.mean(np.square(last_audio.astype(np.float64)))))
    assert audio_rms > 1e-4, f"Silent vLLM-Omni audio detected (audio_rms={audio_rms:.6f})"

    mp4_path = out_dir / f"vllm_omni_minimax_h3_{args.cards}card.mp4"
    with tempfile.TemporaryDirectory() as tmpd:
        v_tmp = Path(tmpd) / "video.mp4"
        a_tmp = Path(tmpd) / "audio.wav"
        iio.imwrite(v_tmp, last_rgb, fps=24, codec="libx264")
        _write_stereo_wav(last_audio, 32000, a_tmp)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(v_tmp),
                "-i",
                str(a_tmp),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                str(mp4_path),
            ],
            check=True,
            capture_output=True,
        )

    npy_path = out_dir / f"vllm_omni_minimax_h3_{args.cards}card_mid.npy"
    audio_npy_path = out_dir / f"vllm_omni_minimax_h3_{args.cards}card_audio.npy"
    np.save(npy_path, last_rgb[last_rgb.shape[0] // 2])
    np.save(audio_npy_path, last_audio)
    warm_med = round(statistics.median(timings[1:] if timings[1:] else timings), 2)
    denoise_med = round(statistics.median(denoise_timings[1:] if denoise_timings[1:] else denoise_timings), 2)
    title = f"vLLM-Omni MiniMax-H3 T2VA | {args.cards}x {args.chip_model} | Warm={warm_med:.2f}s"
    _save_contact_sheet(last_rgb, out_dir / f"contact_sheet_vllm_omni_minimax_h3_{args.cards}card.png", title)
    probe = _ffprobe_check(
        mp4_path,
        expected_w=int(last_rgb.shape[2]),
        expected_h=int(last_rgb.shape[1]),
        expected_frames=int(last_rgb.shape[0]),
        expect_audio=True,
    )
    return {
        "init_s": round(init_s, 2),
        "cold_s": timings[0],
        "warm_runs_s": timings[1:],
        "warm_median_s": warm_med,
        "denoise_s": denoise_med,
        "non_denoise_s": round(max(0.0, warm_med - denoise_med), 2) if denoise_med > 0 else 0.0,
        "num_frames": int(last_rgb.shape[0]),
        "audio_rms": round(audio_rms, 5),
        "reported_peak_memory_mb": round(peak_mb, 1),
        "rgb_hashes": hashes,
        "deterministic_warm": len(set(hashes[1:])) <= 1 if len(hashes) > 1 else True,
        "mp4_path": mp4_path.name,
        "mid_frame_npy": npy_path.name,
        "audio_npy": audio_npy_path.name,
        "ffprobe": probe,
    }


def main() -> None:
    """Parse arguments and run the requested benchmark engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["wan22", "minimax_h3"], default="wan22")
    parser.add_argument("--engine", choices=["telefuser", "vllm_omni"], required=True)
    parser.add_argument("--cards", type=int, required=True)
    parser.add_argument("--chip-model", type=str, default="Ascend 910B2 (64GB)")
    parser.add_argument("--model-root", type=str, default="/models/Wan2.2-TI2V-5B")
    parser.add_argument("--omni-model-root", type=str, default="/models/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--warm-runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--ref-mid-npy", type=str, default="")
    parser.add_argument("--ref-audio-npy", type=str, default="")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.prompt:
        args.prompt = MINIMAX_H3_DEFAULT_PROMPT if args.model == "minimax_h3" else WAN22_DEFAULT_PROMPT
    if args.model == "minimax_h3":
        if args.omni_model_root == "/models/Wan2.2-TI2V-5B-Diffusers":
            args.omni_model_root = args.model_root
        if args.width == 864 and args.height == 480:
            args.width, args.height = 1344, 768
        if args.seed == 42:
            args.seed = 1101

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "minimax_h3":
        res = (
            _run_telefuser_minimax_h3(args, args.out_dir)
            if args.engine == "telefuser"
            else _run_vllm_omni_minimax_h3(args, args.out_dir)
        )
    else:
        res = (
            _run_telefuser_wan22(args, args.out_dir)
            if args.engine == "telefuser"
            else _run_vllm_omni_wan22(args, args.out_dir)
        )

    if args.ref_mid_npy and os.path.isfile(args.ref_mid_npy):
        ref_mid = np.load(args.ref_mid_npy)
        cur_mid = np.load(args.out_dir / res["mid_frame_npy"])
        res["parity_vs_ref"] = _compute_parity(cur_mid, ref_mid)
    if args.ref_audio_npy and os.path.isfile(args.ref_audio_npy) and "audio_npy" in res:
        ref_aud = np.load(args.ref_audio_npy)
        cur_aud = np.load(args.out_dir / res["audio_npy"])
        res["audio_cosine_vs_ref"] = _compute_audio_cosine(cur_aud, ref_aud)

    out_json = args.out_dir / f"{args.engine}_{args.model}_{args.cards}card_metrics.json"
    out_json.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics -> {out_json}")


if __name__ == "__main__":
    main()
