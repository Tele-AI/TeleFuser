from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.orchestrator.artifact_save_stage import ArtifactSaveConfig, ArtifactSaveStage
from telefuser.orchestrator.pipeline_orchestrator import FlexiblePipelineOrchestrator
from telefuser.orchestrator.stage_wrapper import StageConfig
from telefuser.schedulers.flow_match import FlowMatchScheduler
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.logging import logger
from telefuser.utils.video import get_target_image_size
from telefuser.worker.parallel_worker import ParallelWorker
from telefuser.worker.ray_worker import create_ray_worker

from .moe_dit_denoising import MoeDitDenoisingStage
from .text_encoding import TextEncodingStage
from .vae import VAEStage


@dataclass
class AsyncWan22VideoPipelineConfig:
    """Configuration for async Wan2.2 video pipeline with orchestrator support."""

    name: str = ""
    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_high_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_low_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    sample_solver: str = "euler"
    enable_denoising_parallel: bool = False
    enable_vae_parallel: bool = False
    enable_vae_ray: bool = False
    enable_metrics: bool = False


class AsyncWan22VideoPipeline(BasePipeline):
    """Async Wan2.2 video generation pipeline with orchestrator-based execution.

    Supports streaming event-based generation with artifact saving and
    async stage execution for improved throughput.
    """

    def __init__(self, device: str | torch.device, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16
        self.run_name: str = "wan22_video"
        self.default_artifact_fps: int = 16
        self.default_artifact_quality: int = 6

        # Pipeline-owned orchestrator (created lazily on first async start)
        self._orchestrator: FlexiblePipelineOrchestrator | None = None
        self._orchestrator_lock = asyncio.Lock()

        self.artifact_save_stage = ArtifactSaveStage()

    def _get_stages(self) -> list:
        """Get list of pipeline stages for metrics collection."""
        return [self.vae_stage, self.denoise_stage, self.text_encoding_stage]

    def init(self, module_manager: ModuleManager, config: AsyncWan22VideoPipelineConfig):
        """Initialize pipeline stages."""
        self._model_info = module_manager.get_model_info()
        self.config = config
        if getattr(config, "name", ""):
            self.run_name = config.name
        self.vae_stage = VAEStage("vae", module_manager, config.vae_config)
        if config.sample_solver == "euler":
            self.scheduler = FlowMatchScheduler("Wan")
        elif config.sample_solver == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
        else:
            raise NotImplementedError(f"solver {config.sample_solver} is not supported")
        self.denoise_stage = MoeDitDenoisingStage(
            "denoise", module_manager, config.dit_high_config, config.dit_low_config, self.scheduler
        )
        self.text_encoding_stage = TextEncodingStage("text_encoding", module_manager, config.text_encoding_config)
        self.text_encoding_stage = ParallelWorker(self.text_encoding_stage)
        if config.enable_vae_ray:
            logger.info("enable ray actor for vae")
            self.vae_stage = create_ray_worker(self.vae_stage, self.config.enable_vae_parallel)
        elif config.enable_vae_parallel:
            self.vae_stage = ParallelWorker(self.vae_stage)
        if config.enable_denoising_parallel:
            self.denoise_stage = ParallelWorker(self.denoise_stage)

        # Auto-enable metrics if configured
        if config.enable_metrics:
            self.enable_metrics()

    def _get_artifact_output_root(self) -> Path:
        """Get output directory for saved artifacts."""
        root = os.getenv(
            "TELEAI_EXAMPLE_OUTPUT_DIR", "/nvfile-heatstorage/ai_infra/code/jinyx5/lzc/teleai_pipe/outputs"
        )
        return Path(root).expanduser().resolve()

    def normalize_request(
        self,
        *,
        prompt: str,
        input_image: Image.Image | None = None,
        negative_prompt: str = "",
        end_image: Image.Image | None = None,
        seed: int | None = None,
        rand_device: str | None = None,
        height: int = 480,
        width: int = 832,
        resolution: str | None = None,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = False,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        cfg_scale_high: float = 5.0,
        cfg_scale_low: float = 5.0,
        boundary: float = 0.875,
        artifact_fps: int | None = None,
        artifact_quality: int | None = None,
    ) -> dict[str, Any]:
        """Normalize and validate generation request parameters."""
        if input_image is not None and resolution is not None:
            w, h = get_target_image_size(input_image.size[0], input_image.size[1], resolution=resolution)
            width, height = w, h

        height, width = self.check_resize_height_width(height, width)
        # Wan video requires num_frames % 4 == 1
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            logger.info(f"Only `num_frames % 4 == 1` is acceptable; rounded to {num_frames}.")

        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "input_image": input_image,
            "end_image": end_image,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_inference_steps": num_inference_steps,
            "sigma_shift": sigma_shift,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "cfg_scale_high": cfg_scale_high,
            "cfg_scale_low": cfg_scale_low,
            "boundary": boundary,
            "artifact_fps": int(artifact_fps) if artifact_fps is not None else self.default_artifact_fps,
            "artifact_quality": int(artifact_quality)
            if artifact_quality is not None
            else self.default_artifact_quality,
        }

    def get_orchestrator_stage_configs(self):
        """Define orchestrator stage configurations for async execution."""
        use_ray = bool(getattr(self.config, "enable_vae_ray", False))

        def build_text_encoding(context: dict[str, Any], inputs: dict[str, Any]):
            prompt_list = [inputs.get("prompt", "")]
            if float(inputs.get("cfg_scale_high", 1.0)) != 1.0:
                prompt_list.append(inputs.get("negative_prompt", ""))
            return (prompt_list,), {}

        def build_vae_encode(context: dict[str, Any], inputs: dict[str, Any]):
            img = inputs.get("input_image")
            if img is None:
                raise ValueError("input_image is required for wan22 I2V vae_encode stage")

            height = int(inputs["height"])
            width = int(inputs["width"])
            num_frames = int(inputs["num_frames"])

            input_tensor = self.preprocess_image(img, height, width)
            end_img = inputs.get("end_image")
            end_tensor = self.preprocess_image(end_img, height, width) if end_img is not None else None

            tiler_kwargs = {
                "tiled": bool(inputs.get("tiled", False)),
                "tile_size": inputs.get("tile_size", (30, 52)),
                "tile_stride": inputs.get("tile_stride", (15, 26)),
            }

            return ("encode_image", input_tensor, end_tensor, num_frames), tiler_kwargs

        def build_denoise(context: dict[str, Any], inputs: dict[str, Any]):
            height = int(inputs["height"])
            width = int(inputs["width"])
            num_frames = int(inputs["num_frames"])
            num_inference_steps = int(inputs.get("num_inference_steps", 50))

            cfg_scale_high = float(inputs.get("cfg_scale_high", 1.0))
            cfg_scale_low = float(inputs.get("cfg_scale_low", 1.0))
            sigma_shift = float(inputs.get("sigma_shift", 5.0))
            boundary = float(inputs.get("boundary", 0.875))

            rand_device = inputs.get("rand_device") or self.device
            noise = self.generate_noise(
                (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                seed=inputs.get("seed"),
                device=rand_device,
                dtype=torch.float32,
            )
            latents = noise.to(dtype=self.torch_dtype, device=self.device)

            prompt_emb_list = context.get("prompt_emb_list", context.get("text_encoding"))
            if prompt_emb_list is None:
                raise RuntimeError("prompt_emb_list missing from context")
            prompt_emb_posi = prompt_emb_list[0]
            prompt_emb_nega = prompt_emb_list[1] if cfg_scale_high != 1.0 else None

            ref_latent = context.get("ref_latent", context.get("vae_encode"))

            args = (
                latents,
                num_inference_steps,
                ref_latent,
                prompt_emb_posi,
                prompt_emb_nega,
                cfg_scale_high,
                cfg_scale_low,
                sigma_shift,
                boundary,
            )
            return args, {}

        def build_vae_decode(context: dict[str, Any], inputs: dict[str, Any]):
            latents = context.get("denoised_latents", context.get("denoise"))
            if latents is None:
                raise RuntimeError("denoised_latents missing from context")
            tiler_kwargs = {
                "tiled": bool(inputs.get("tiled", False)),
                "tile_size": inputs.get("tile_size", (30, 52)),
                "tile_stride": inputs.get("tile_stride", (15, 26)),
            }
            return ("decode_video", latents), tiler_kwargs

        def postprocess_frames(vae_output: torch.Tensor) -> list[Image.Image]:
            return self.tensor2video(vae_output[0])

        def build_artifact_save(context: dict[str, Any], inputs: dict[str, Any]):
            frames = context.get("frames", context.get("vae_decode"))
            if frames is None:
                raise RuntimeError("frames missing from context")
            rid = str(inputs.get("request_id", "req"))
            cfg = ArtifactSaveConfig(
                output_root=self._get_artifact_output_root(),
                run_name=self.run_name,
                fps=int(inputs.get("artifact_fps", self.default_artifact_fps)),
                quality=int(inputs.get("artifact_quality", self.default_artifact_quality)),
            )
            return (frames,), {"request_id": rid, "config": cfg}

        return [
            StageConfig(
                stage_id=0,
                stage_name="text_encoding",
                pipeline_attr="text_encoding_stage",
                param_builder=build_text_encoding,
                parallel_group="init",
                metadata={"role": "encode_text", "output_key": "prompt_emb_list"},
            ),
            StageConfig(
                stage_id=1,
                stage_name="vae_encode",
                pipeline_attr="vae_stage",
                param_builder=build_vae_encode,
                shared_lock_group="vae",
                parallel_group="init",
                metadata={
                    "role": "encode_ref",
                    "output_key": "ref_latent",
                    "use_ray": use_ray,
                },
            ),
            StageConfig(
                stage_id=2,
                stage_name="denoise",
                pipeline_attr="denoise_stage",
                param_builder=build_denoise,
                metadata={"role": "denoise", "output_key": "denoised_latents"},
            ),
            StageConfig(
                stage_id=3,
                stage_name="vae_decode",
                pipeline_attr="vae_stage",
                param_builder=build_vae_decode,
                result_processor=postprocess_frames,
                shared_lock_group="vae",
                metadata={
                    "role": "decode",
                    "output_key": "frames",
                    "use_ray": use_ray,
                },
            ),
            StageConfig(
                stage_id=4,
                stage_name="artifact_save",
                pipeline_attr="artifact_save_stage",
                param_builder=build_artifact_save,
                metadata={"role": "artifact_save", "output_key": "video_artifact"},
            ),
        ]

    async def astart(self):
        """Start the pipeline orchestrator asynchronously."""
        async with self._orchestrator_lock:
            if self._orchestrator is None:
                self._orchestrator = FlexiblePipelineOrchestrator(
                    pipeline=self,
                    stage_configs=self.get_orchestrator_stage_configs(),
                )
            await self._orchestrator.start()

    async def astop(self):
        """Stop the pipeline orchestrator asynchronously."""
        async with self._orchestrator_lock:
            if self._orchestrator is not None:
                await self._orchestrator.stop()
                self._orchestrator = None

    async def agenerate(
        self,
        *,
        request_id: str,
        include_raw: bool = False,
        **kwargs: Any,
    ):
        """Generate video asynchronously with event streaming."""
        await self.astart()
        assert self._orchestrator is not None

        inputs = self.normalize_request(**kwargs)
        inputs["request_id"] = request_id

        stage_cfg_by_id = {c.stage_id: c for c in self._orchestrator.stage_configs}
        seq = 0

        async for raw in self._orchestrator.generate(request_id=request_id, inputs=inputs):
            seq += 1
            ts_ms = int(time.time() * 1000)

            if raw.get("error"):
                cfg = stage_cfg_by_id.get(raw.get("stage_id", -1))
                stage_meta = {
                    "id": raw.get("stage_id"),
                    "name": raw.get("stage_name"),
                    "role": (cfg.metadata.get("role") if cfg else None),
                }
                evt = {
                    "v": "teleai.v2.event.v1",
                    "type": "error",
                    "request_id": request_id,
                    "seq": seq,
                    "ts_ms": ts_ms,
                    "pipeline": {"pipeline": "wan22_video", "run_name": self.run_name},
                    "error": {"message": raw["error"], "stage": stage_meta},
                }
                if include_raw:
                    evt["raw"] = raw
                yield evt
                return

            if not raw.get("finished", False):
                stage_id = int(raw["stage_id"])
                cfg = stage_cfg_by_id.get(stage_id)
                role = cfg.metadata.get("role") if cfg else None
                output_key = cfg.metadata.get("output_key") if cfg else None
                evt = {
                    "v": "teleai.v2.event.v1",
                    "type": "stage_end",
                    "request_id": request_id,
                    "seq": seq,
                    "ts_ms": ts_ms,
                    "pipeline": {"pipeline": "wan22_video", "run_name": self.run_name},
                    "stage": {
                        "id": stage_id,
                        "name": raw.get("stage_name"),
                        "role": role,
                        "group": (cfg.parallel_group if cfg else None),
                    },
                    "timing": {"duration_ms": float(raw.get("stage_time_ms", 0.0))},
                    "metrics": raw.get("metrics", {}),
                    "payload": {
                        "output_key": output_key,
                        "output_ref": f"context.{output_key}" if output_key else None,
                    },
                }
                if include_raw:
                    evt["raw"] = raw
                yield evt
                continue

            # Finished: produce final event with artifact
            ctx = raw.get("context", {}) or {}
            artifact = ctx.get("video_artifact")
            if artifact is None:
                artifact = raw.get("final_output")

            total_ms = int((raw.get("metrics", {}).get("total_time_s", 0.0)) * 1000)
            evt = {
                "v": "teleai.v2.event.v1",
                "type": "final",
                "request_id": request_id,
                "seq": seq,
                "ts_ms": ts_ms,
                "pipeline": {"pipeline": "wan22_video", "run_name": self.run_name},
                "timing": {"total_ms": total_ms},
                "payload": {
                    "artifacts": [artifact] if artifact is not None else [],
                    "params_snapshot": {
                        "seed": inputs.get("seed"),
                        "steps": inputs.get("num_inference_steps"),
                        "num_frames": inputs.get("num_frames"),
                        "height": inputs.get("height"),
                        "width": inputs.get("width"),
                    },
                },
            }
            if include_raw:
                evt["raw"] = raw
            yield evt
            return

    def __del__(self):
        """Cleanup pipeline stages safely, handling wrapped workers."""
        for attr_name in ["vae_stage", "denoise_stage", "text_encoding_stage"]:
            try:
                stage = getattr(self, attr_name, None)
                if stage is None:
                    continue

                from telefuser.worker.parallel_worker import ParallelWorker

                if isinstance(stage, ParallelWorker):
                    del stage
                    continue

                if hasattr(stage, "__ray_terminate__"):
                    try:
                        stage.__ray_terminate__.remote()
                    except Exception:
                        pass
                    del stage
                    continue

                del stage

            except Exception:
                pass
