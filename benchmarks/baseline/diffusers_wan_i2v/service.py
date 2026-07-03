from __future__ import annotations

import base64
import binascii
import io
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video
from transformers import CLIPVisionModel

DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
DEFAULT_OUTPUT_DIR = "artifacts/diffusers_wan_i2v"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010
DEFAULT_GUIDANCE_SCALE = 5.0
DEFAULT_NUM_FRAMES = 81
DEFAULT_NUM_INFERENCE_STEPS = 40
DEFAULT_TARGET_FPS = 16
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 832
DEFAULT_TIMEOUT_SECONDS = 7200.0
VIDEO_STATUS_QUEUED = "queued"
VIDEO_STATUS_GENERATING = "generating"
VIDEO_STATUS_COMPLETED = "completed"
VIDEO_STATUS_FAILED = "failed"
VIDEO_STATUS_CANCELLED = "cancelled"


def _coerce_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _coerce_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


@dataclass
class ServiceConfig:
    model_id: str = os.environ.get("DIFFUSERS_WAN_I2V_MODEL_ID", DEFAULT_MODEL_ID)
    output_dir: Path = Path(os.environ.get("DIFFUSERS_WAN_I2V_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    host: str = os.environ.get("DIFFUSERS_WAN_I2V_HOST", DEFAULT_HOST)
    port: int = _coerce_env_int("DIFFUSERS_WAN_I2V_PORT", DEFAULT_PORT)
    torch_dtype: torch.dtype = torch.bfloat16
    device: str = os.environ.get("DIFFUSERS_WAN_I2V_DEVICE", "cuda")
    guidance_scale: float = _coerce_env_float("DIFFUSERS_WAN_I2V_GUIDANCE_SCALE", DEFAULT_GUIDANCE_SCALE)
    num_frames: int = _coerce_env_int("DIFFUSERS_WAN_I2V_NUM_FRAMES", DEFAULT_NUM_FRAMES)
    num_inference_steps: int = _coerce_env_int(
        "DIFFUSERS_WAN_I2V_NUM_INFERENCE_STEPS", DEFAULT_NUM_INFERENCE_STEPS
    )
    target_fps: int = _coerce_env_int("DIFFUSERS_WAN_I2V_TARGET_FPS", DEFAULT_TARGET_FPS)
    width: int = _coerce_env_int("DIFFUSERS_WAN_I2V_WIDTH", DEFAULT_WIDTH)
    height: int = _coerce_env_int("DIFFUSERS_WAN_I2V_HEIGHT", DEFAULT_HEIGHT)
    timeout_seconds: float = _coerce_env_float("DIFFUSERS_WAN_I2V_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)


@dataclass
class JobRecord:
    job_id: str
    prompt: str
    model: str
    size: str
    seconds: int
    status: str = VIDEO_STATUS_QUEUED
    progress: int = 0
    created_at: int = field(default_factory=lambda: int(time.time()))
    completed_at: int | None = None
    output_path: str | None = None
    error: dict[str, Any] | None = None
    inference_time_s: float | None = None
    peak_memory_mb: float | None = None

    def to_response(self, base_url: str) -> dict[str, Any]:
        url = None
        if self.status == VIDEO_STATUS_COMPLETED:
            url = f"{base_url}/v1/videos/{self.job_id}/content"
        return {
            "id": self.job_id,
            "object": "video",
            "model": self.model,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "size": self.size,
            "seconds": str(self.seconds),
            "quality": "standard",
            "url": url,
            "completed_at": self.completed_at,
            "error": self.error,
            "file_path": self.output_path,
            "inference_time_s": self.inference_time_s,
            "peak_memory_mb": self.peak_memory_mb,
        }


class DiffusersWanService:
    """Standalone async HTTP service for the official Diffusers Wan2.1 I2V 480P pipeline."""

    def __init__(self, config: ServiceConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.input_dir = self.output_dir / "inputs"
        self.video_dir = self.output_dir / "videos"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self) -> WanImageToVideoPipeline:
        image_encoder = CLIPVisionModel.from_pretrained(
            self.config.model_id,
            subfolder="image_encoder",
            torch_dtype=torch.float32,
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.config.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.config.model_id,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=self.config.torch_dtype,
        )
        pipe.to(self.config.device)
        return pipe

    def create_job(self, *, prompt: str, model: str, size: str, seconds: int) -> JobRecord:
        job_id = str(uuid.uuid4())
        job = JobRecord(
            job_id=job_id,
            prompt=prompt,
            model=model or self.config.model_id,
            size=size,
            seconds=seconds,
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def cancel_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in {VIDEO_STATUS_COMPLETED, VIDEO_STATUS_FAILED, VIDEO_STATUS_CANCELLED}:
                return job
            job.status = VIDEO_STATUS_CANCELLED
            job.progress = 0
            job.completed_at = int(time.time())
            job.error = {"message": "Job cancelled by user"}
            return job

    def _resolve_image_from_reference(
        self,
        *,
        input_reference_file: UploadFile | None,
        input_reference_value: str | None,
        reference_url: str | None,
        job_id: str,
    ) -> Image.Image:
        if input_reference_file is not None and input_reference_file.filename:
            content = input_reference_file.file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded input_reference is empty")
            image = Image.open(io.BytesIO(content)).convert("RGB")
            suffix = Path(input_reference_file.filename).suffix or ".png"
            (self.input_dir / f"{job_id}{suffix}").write_bytes(content)
            return image

        ref = input_reference_value or reference_url
        if not ref:
            raise HTTPException(status_code=400, detail="I2V benchmark service requires input_reference or reference_url")

        if ref.startswith("http://") or ref.startswith("https://"):
            raise HTTPException(
                status_code=400,
                detail="Remote HTTP reference_url is not supported by this baseline service; use a local path or upload",
            )

        if ref.startswith("data:"):
            try:
                _, b64_data = ref.split(",", 1)
                image_bytes = base64.b64decode(b64_data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise HTTPException(status_code=400, detail=f"Malformed data URL for input reference: {exc}") from exc
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            (self.input_dir / f"{job_id}.png").write_bytes(image_bytes)
            return image

        path = Path(ref)
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=400, detail=f"Reference image not found: {ref}")
        return Image.open(path).convert("RGB")

    def _run_job(self, job_id: str, image: Image.Image, negative_prompt: str | None, seed: int | None) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        with self._lock:
            if job.status == VIDEO_STATUS_CANCELLED:
                return
            job.status = VIDEO_STATUS_GENERATING
            job.progress = 5

        output_path = self.video_dir / f"{job_id}.mp4"
        start = time.perf_counter()
        peak_memory_mb: float | None = None

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.config.device).manual_seed(int(seed))

            result = self._pipeline(
                image=image,
                prompt=job.prompt,
                negative_prompt=negative_prompt,
                height=self.config.height,
                width=self.config.width,
                num_frames=self.config.num_frames,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=generator,
            )
            frames = result.frames[0]
            export_to_video(frames, str(output_path), fps=self.config.target_fps)

            elapsed = time.perf_counter() - start
            if torch.cuda.is_available():
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            with self._lock:
                if job.status == VIDEO_STATUS_CANCELLED:
                    output_path.unlink(missing_ok=True)
                    return
                job.status = VIDEO_STATUS_COMPLETED
                job.progress = 100
                job.completed_at = int(time.time())
                job.output_path = str(output_path)
                job.inference_time_s = elapsed
                job.peak_memory_mb = peak_memory_mb
        except Exception as exc:
            with self._lock:
                job.status = VIDEO_STATUS_FAILED
                job.progress = 0
                job.completed_at = int(time.time())
                job.error = {"message": str(exc)}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def submit_job(
        self,
        *,
        prompt: str,
        model: str | None,
        size: str | None,
        seconds: int | None,
        seed: int | None,
        negative_prompt: str | None,
        input_reference_file: UploadFile | None,
        input_reference_value: str | None,
        reference_url: str | None,
    ) -> JobRecord:
        job = self.create_job(
            prompt=prompt,
            model=model or self.config.model_id,
            size=size or f"{self.config.width}x{self.config.height}",
            seconds=seconds or max(1, self.config.num_frames // self.config.target_fps),
        )
        image = self._resolve_image_from_reference(
            input_reference_file=input_reference_file,
            input_reference_value=input_reference_value,
            reference_url=reference_url,
            job_id=job.job_id,
        )
        worker = threading.Thread(
            target=self._run_job,
            args=(job.job_id, image, negative_prompt, seed),
            daemon=True,
            name=f"diffusers-wan-job-{job.job_id}",
        )
        worker.start()
        return job


service_config = ServiceConfig()
service = DiffusersWanService(service_config)
app = FastAPI(
    title="Diffusers Wan2.1 I2V Benchmark Service",
    description="Standalone async HTTP baseline for Wan2.1-I2V-14B-480P using official Diffusers.",
    version="0.1.0",
)


@app.get("/v1/service/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "diffusers-wan-i2v",
        "model_id": service_config.model_id,
        "device": service_config.device,
        "workload": {
            "width": service_config.width,
            "height": service_config.height,
            "num_frames": service_config.num_frames,
            "num_inference_steps": service_config.num_inference_steps,
            "guidance_scale": service_config.guidance_scale,
            "fps": service_config.target_fps,
        },
    }


@app.get("/metrics")
async def metrics() -> Response:
    lines = [
        "# HELP diffusers_wan_jobs_total Number of tracked jobs",
        "# TYPE diffusers_wan_jobs_total gauge",
        f"diffusers_wan_jobs_total {len(service.list_jobs())}",
    ]
    completed = sum(1 for job in service.list_jobs() if job.status == VIDEO_STATUS_COMPLETED)
    failed = sum(1 for job in service.list_jobs() if job.status == VIDEO_STATUS_FAILED)
    lines.extend(
        [
            "# HELP diffusers_wan_jobs_completed_total Number of completed jobs",
            "# TYPE diffusers_wan_jobs_completed_total counter",
            f"diffusers_wan_jobs_completed_total {completed}",
            "# HELP diffusers_wan_jobs_failed_total Number of failed jobs",
            "# TYPE diffusers_wan_jobs_failed_total counter",
            f"diffusers_wan_jobs_failed_total {failed}",
        ]
    )
    return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/v1/videos")
async def create_video(
    request: Request,
    prompt: str | None = Form(None),
    input_reference: UploadFile | None = File(None),
    reference_url: str | None = Form(None),
    model: str | None = Form(None),
    seconds: int | None = Form(None),
    size: str | None = Form(None),
    seed: int | None = Form(None),
    negative_prompt: str | None = Form(None),
) -> JSONResponse:
    content_type = request.headers.get("content-type", "").lower()

    if "application/json" in content_type:
        body = await request.json()
        prompt_value = body.get("prompt")
        input_reference_value = body.get("input_reference")
        reference_url_value = body.get("reference_url")
        model_value = body.get("model")
        seconds_value = body.get("seconds")
        size_value = body.get("size")
        seed_value = body.get("seed")
        negative_prompt_value = body.get("negative_prompt")
        upload = None
    else:
        prompt_value = prompt
        input_reference_value = None
        reference_url_value = reference_url
        model_value = model
        seconds_value = seconds
        size_value = size
        seed_value = seed
        negative_prompt_value = negative_prompt
        upload = input_reference

    if not prompt_value:
        raise HTTPException(status_code=422, detail="prompt is required")

    job = service.submit_job(
        prompt=prompt_value,
        model=model_value,
        size=size_value,
        seconds=seconds_value,
        seed=seed_value,
        negative_prompt=negative_prompt_value,
        input_reference_file=upload,
        input_reference_value=input_reference_value,
        reference_url=reference_url_value,
    )
    base_url = str(request.base_url).rstrip("/")
    return JSONResponse(job.to_response(base_url))


@app.get("/v1/videos")
async def list_videos(request: Request) -> dict[str, Any]:
    base_url = str(request.base_url).rstrip("/")
    jobs = sorted(service.list_jobs(), key=lambda item: item.created_at, reverse=True)
    return {
        "object": "list",
        "data": [job.to_response(base_url) for job in jobs],
        "has_more": False,
        "first_id": jobs[0].job_id if jobs else None,
        "last_id": jobs[-1].job_id if jobs else None,
    }


@app.get("/v1/videos/{video_id}")
async def retrieve_video(video_id: str, request: Request) -> dict[str, Any]:
    job = service.get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    base_url = str(request.base_url).rstrip("/")
    return job.to_response(base_url)


@app.delete("/v1/videos/{video_id}")
async def delete_video(video_id: str, request: Request) -> dict[str, Any]:
    job = service.cancel_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    base_url = str(request.base_url).rstrip("/")
    return job.to_response(base_url)


@app.get("/v1/videos/{video_id}/content")
async def get_video_content(video_id: str) -> FileResponse:
    job = service.get_job(video_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    if job.status != VIDEO_STATUS_COMPLETED or not job.output_path:
        raise HTTPException(status_code=400, detail=f"Video {video_id} is not ready")
    path = Path(job.output_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video content not found: {job.output_path}")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


def main() -> None:
    uvicorn.run(app, host=service_config.host, port=service_config.port, log_level="warning")


if __name__ == "__main__":
    main()
