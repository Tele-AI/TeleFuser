"""Benchmark LingBot scheduler latency, placement, and memory for Phase 6."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from telefuser.core.config import AttentionConfig, AttnImplType, ModelRuntimeConfig, ParallelConfig
from telefuser.pipelines.lingbot_world_fast.control import LingBotWorldFastControlBuilder
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline, LingBotWorldFastPipelineConfig
from telefuser.pipelines.lingbot_world_fast.session import LingBotWorldFastSessionConfig

PROFILES = {
    "vae_dedicated": {"encode": 0, "decode": 0, "dit": [1, 2]},
    "vae_split": {"encode": 0, "decode": 1, "dit": [2, 3]},
    "vae_shared": {"encode": 0, "decode": 0, "dit": [0, 1, 2, 3]},
}
DEFAULT_PROMPT = (
    "A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped "
    "mountains under a bright blue sky with drifting white clouds. Gentle ripples reflect the tree and sky."
)


def _gpu_memory_mib() -> dict[int, int]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    memory: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        index, used = line.split(",", maxsplit=1)
        memory[int(index.strip())] = int(used.strip())
    return memory


def _gpu_names() -> dict[int, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        int(index.strip()): name.strip()
        for line in completed.stdout.splitlines()
        for index, name in [line.split(",", maxsplit=1)]
    }


class _PeakMemorySampler:
    def __init__(self, interval_seconds: float = 0.2) -> None:
        self.interval_seconds = interval_seconds
        self.peaks: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="phase6-gpu-memory")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[int, int]:
        self._stop.set()
        self._thread.join()
        return dict(self.peaks)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for device, used in _gpu_memory_mib().items():
                    self.peaks[device] = max(self.peaks.get(device, 0), used)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


def _vae_config(device_id: int) -> ModelRuntimeConfig:
    return ModelRuntimeConfig(
        device_type="cuda",
        device_id=device_id,
        torch_dtype=torch.float32,
        parallel_config=ParallelConfig(device_ids=[device_id]),
    )


def _build_pipeline(profile_name: str, model_root: Path, fast_model_root: Path) -> LingBotWorldFastPipeline:
    profile = PROFILES[profile_name]
    dit_devices = profile["dit"]
    assert isinstance(dit_devices, list)
    pipeline = LingBotWorldFastPipeline(device=f"cuda:{dit_devices[0]}", torch_dtype=torch.bfloat16)
    pipeline.init(
        LingBotWorldFastPipelineConfig(
            checkpoint_dir=str(model_root),
            fast_checkpoint_path=str(fast_model_root),
            vae_config=_vae_config(int(profile["encode"])),
            vae_parallel_config=ParallelConfig(device_ids=[int(profile["encode"])]),
            vae_encode_config=_vae_config(int(profile["encode"])),
            vae_decode_config=_vae_config(int(profile["decode"])),
            text_encoding_config=ModelRuntimeConfig(
                device_type="cuda",
                device_id=dit_devices[0],
                torch_dtype=torch.bfloat16,
            ),
            dit_torch_dtype=torch.bfloat16,
            control_type="cam",
            max_area=480 * 832,
            local_attn_size=18,
            sink_size=0,
            timestep_indices=(0, 179, 358, 679),
            attention_config=AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90),
            parallel_config=ParallelConfig(
                device_ids=dit_devices,
                sp_ulysses_degree=len(dit_devices),
            ),
        )
    )
    return pipeline


def _memory_plateau(samples: list[dict[str, object]], devices: list[int]) -> dict[str, object]:
    warmup_chunks = min(3, max(1, len(samples) // 4))
    steady = samples[warmup_chunks:]
    evaluated = len(samples) >= 20
    by_device: dict[str, dict[str, object]] = {}
    stable = evaluated
    for device in devices:
        values = [int(sample["gpu_memory_mib"].get(str(device), 0)) for sample in steady]
        if not values:
            stable = False
            continue
        memory_range = max(values) - min(values)
        slope = (values[-1] - values[0]) / max(1, len(values) - 1)
        midpoint = max(1, len(values) // 2)
        early_peak = max(values[:midpoint])
        late_peak = max(values[midpoint:]) if values[midpoint:] else early_peak
        peak_growth = late_peak - early_peak
        device_stable = evaluated and peak_growth <= 1024 and slope <= 128
        stable = stable and device_stable if evaluated else False
        by_device[str(device)] = {
            "minimum_mib": min(values),
            "maximum_mib": max(values),
            "range_mib": memory_range,
            "late_peak_growth_mib": peak_growth,
            "endpoint_slope_mib_per_chunk": round(slope, 3),
            "stable": device_stable if evaluated else None,
        }
    return {
        "stable": stable if evaluated else None,
        "evaluated": evaluated,
        "warmup_chunks_excluded": warmup_chunks,
        "criteria": "late peak growth <= 1024 MiB and positive endpoint slope <= 128 MiB/chunk",
        "devices": by_device,
    }


def _relative_stage_timings(timings: tuple[object, ...], origin: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for timing in timings:
        rows.append(
            {
                "sequence_id": timing.sequence_id,
                "inputs_ready_seconds": round(timing.inputs_ready_at - origin, 6),
                "admitted_seconds": round(timing.admitted_at - origin, 6),
                "completed_seconds": round(timing.completed_at - origin, 6),
                "execution_seconds": round(timing.completed_at - timing.admitted_at, 6),
                "dependency_wait_seconds": round(timing.admitted_at - timing.inputs_ready_at, 6),
            }
        )
    return rows


def _run_session(
    pipeline: LingBotWorldFastPipeline,
    image: Image.Image,
    chunks: int,
    devices: list[int],
    timeout: float,
) -> dict[str, object]:
    chunk_size = 3
    frame_num = 4 * (chunks * chunk_size - 1) + 1
    session_config = LingBotWorldFastSessionConfig(
        prompt=DEFAULT_PROMPT,
        image=image,
        control_mode="cam",
        chunk_size=chunk_size,
        frame_num=frame_num,
        frame_policy="strict",
        seed=42,
        show_control_hud=False,
    )
    control_context = pipeline.control_context(session_config)
    control_builder = LingBotWorldFastControlBuilder(control_context)
    identity_poses = np.repeat(np.eye(4, dtype=np.float32)[None], chunk_size, axis=0)
    sampler = _PeakMemorySampler()
    sampler.start()
    wall_started = time.monotonic()
    runtime = None
    try:
        runtime = pipeline._create_initialized_session(session_config)
        streaming_runtime = pipeline._get_streaming_runtime()
        session = streaming_runtime.create_session(runtime)
    except BaseException:
        sampler.stop()
        if runtime is not None:
            pipeline.release_session(runtime)
        raise
    submitted = 0
    completed_chunks = 0
    emitted_frames = 0
    artifact_slot_high_watermark = 0
    memory_by_chunk: list[dict[str, object]] = []
    deadline = time.monotonic() + timeout
    try:
        while completed_chunks < chunks:
            error = streaming_runtime.error(session)
            if error is not None:
                raise RuntimeError("LingBot scheduler failed during Phase 6 benchmark") from error
            while submitted < chunks and streaming_runtime.can_submit_chunk(session):
                control = control_builder.build({"poses": identity_poses})
                if not streaming_runtime.try_submit_chunk(session, submitted, control):
                    raise RuntimeError("Scheduler capacity changed after can_submit_chunk")
                submitted += 1
            stats = streaming_runtime.orchestrator.artifact_stats(session.session_id)
            artifact_slot_high_watermark = max(artifact_slot_high_watermark, stats.slot_count)
            outputs = streaming_runtime.poll_frames(session)
            for sequence_id, frames in outputs:
                if sequence_id != completed_chunks:
                    raise RuntimeError(f"Out-of-order chunk {sequence_id}; expected {completed_chunks}")
                emitted_frames += len(frames)
                completed_chunks += 1
                memory_by_chunk.append(
                    {
                        "chunk": sequence_id,
                        "gpu_memory_mib": {str(key): value for key, value in _gpu_memory_mib().items()},
                    }
                )
                del frames
            if completed_chunks >= chunks:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out after {timeout}s with {completed_chunks}/{chunks} chunks")
            if not outputs:
                streaming_runtime.wait_until_idle(session, timeout=0.05)

        session_metrics = streaming_runtime.session_metrics(session)
        origin = session_metrics.ingress_accepted_at[0][1]
        stage_timings = {
            stage_id: _relative_stage_timings(
                streaming_runtime.orchestrator.stage_timings(session.session_id, stage_id),
                origin,
            )
            for stage_id in ("encode", "denoise", "decode")
        }
        idle_intervals = [asdict(item) for item in streaming_runtime.stage_idle_intervals(session, "denoise")]
        final_stats = asdict(streaming_runtime.orchestrator.artifact_stats(session.session_id))
        diagnostics = asdict(streaming_runtime.orchestrator.diagnostics())
        groups = [asdict(group) for group in streaming_runtime.orchestrator.spec.resource_groups]
        metrics = asdict(session_metrics)
    finally:
        streaming_runtime.close_session(session)
        peak_memory = sampler.stop()

    warmup_proof = []
    denoise_timings = stage_timings["denoise"]
    for previous, current in zip(denoise_timings, denoise_timings[1:]):
        warmup_proof.append(
            {
                "previous_sequence_id": previous["sequence_id"],
                "sequence_id": current["sequence_id"],
                "next_ready_before_previous_completed": (
                    current["inputs_ready_seconds"] <= previous["completed_seconds"]
                ),
            }
        )
    unexplained_idle = [
        interval
        for interval in idle_intervals[2:]
        if interval["idle_seconds"] > 0.001 and interval["reason"] == "scheduler_admission"
    ]
    return {
        "chunks": chunks,
        "frame_num": frame_num,
        "emitted_frames": emitted_frames,
        "wall_seconds": round(time.monotonic() - wall_started, 6),
        "metrics": metrics,
        "stage_timings": stage_timings,
        "dit_idle_intervals": idle_intervals,
        "post_warmup_ready_proof": warmup_proof[2:],
        "unexplained_post_warmup_dit_idle_count": len(unexplained_idle),
        "artifact_slot_high_watermark": artifact_slot_high_watermark,
        "final_artifact_stats": final_stats,
        "scheduler_diagnostics": diagnostics,
        "resource_groups": groups,
        "gpu_peak_memory_mib": {str(key): value for key, value in peak_memory.items()},
        "gpu_memory_by_chunk": memory_by_chunk,
        "memory_plateau": _memory_plateau(memory_by_chunk, devices),
        "session_released": runtime.cache_handle is None,
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_report(paths: list[Path], output: Path) -> None:
    reports = [json.loads(path.read_text()) for path in paths]
    lines = [
        "# Stream Scheduler Phase 6 Test Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Results",
        "",
        "| Profile | Chunks | First output (s) | Chunk p50/p95 (s) | "
        "Control-to-frame p50/p95 (s) | Memory plateau | Unexplained DiT idle |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    phase_passed = True
    for report in reports:
        for run in report["runs"]:
            metrics = run["metrics"]
            period = metrics["chunk_period"]
            control = metrics["control_to_output_latency"]
            diagnostics_ok = not any(run["scheduler_diagnostics"].values())
            ready_ok = all(item["next_ready_before_previous_completed"] for item in run["post_warmup_ready_proof"])
            plateau = run["memory_plateau"]["stable"]
            plateau_ok = run["chunks"] < 20 or plateau is True
            run_passed = (
                run["emitted_frames"] == run["frame_num"]
                and plateau_ok
                and run["unexplained_post_warmup_dit_idle_count"] == 0
                and diagnostics_ok
                and run["session_released"]
                and ready_ok
            )
            phase_passed = phase_passed and run_passed
            lines.append(
                f"| {report['profile']} | {run['chunks']} | {_fmt(metrics['first_output_latency_seconds'])} | "
                f"{_fmt(period['p50_seconds'])}/{_fmt(period['p95_seconds'])} | "
                f"{_fmt(control['p50_seconds'])}/{_fmt(control['p95_seconds'])} | "
                f"{'PASS' if plateau is True else 'FAIL' if plateau is False else 'N/A'} | "
                f"{run['unexplained_post_warmup_dit_idle_count']} |"
            )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"Phase 6 result: **{'PASS' if phase_passed else 'FAIL'}**.",
            "",
            "The 20-chunk memory plateau excludes warm-up and requires late peak growth <= 1,024 MiB "
            "with positive endpoint slope <= 128 MiB per chunk; shorter smoke runs are not gated.",
            "The JSON files contain every stage timestamp, DiT idle attribution, per-chunk memory sample, "
            "resource-group policy, scheduler diagnostic, and lifecycle result.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES))
    parser.add_argument("--chunks", default="5,20", help="Comma-separated chunk counts")
    parser.add_argument("--model-root", type=Path, default=Path("/hhb-data/aigc/model_zoo/Wan2.2-I2V-A14B"))
    parser.add_argument(
        "--fast-model-root",
        type=Path,
        default=Path("/hhb-data/aigc/model_zoo/lingbot/lingbot-world-fast"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("examples/data/lingbot_world_fast/image.jpg"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--render-report", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.render_report:
        output = args.output or Path("work_dirs/stream_scheduler_phase6_report.md")
        _render_report(args.render_report, output)
        print(output)
        return
    if args.profile is None:
        raise ValueError("--profile is required unless --render-report is used")
    profile = PROFILES[args.profile]
    used_devices = sorted({int(profile["encode"]), int(profile["decode"]), *profile["dit"]})
    pipeline = _build_pipeline(args.profile, args.model_root, args.fast_model_root)
    try:
        image = Image.open(args.image).convert("RGB")
        runs = [
            _run_session(pipeline, image, int(chunks), used_devices, args.timeout) for chunks in args.chunks.split(",")
        ]
        report = {
            "profile": args.profile,
            "placement": profile,
            "gpu_names": _gpu_names(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "runs": runs,
            "local_attn_size": 18,
        }
        output = args.output or Path(f"work_dirs/stream_scheduler_phase6_{args.profile}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(output)
    finally:
        pipeline.close()
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
