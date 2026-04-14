from __future__ import annotations

import torch
from tqdm import tqdm

from telefuser.core.base_stage import BaseStage, with_model_offload
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.metrics import with_metrics
from telefuser.models.flashvsr_dit import FlashVSRModel
from telefuser.utils.logging import logger
from telefuser.utils.profiler import ProfilingContext4Debug


class DitDenoisingStage(BaseStage):
    """Diffusion-based denoising stage for FlashVSR video super-resolution.

    Processes video in chunks with streaming support. First chunk requires 25+ frames
    for initialization; subsequent chunks use 8-frame segments with cached context.
    """

    def __init__(
        self,
        name: str,
        module_manager: ModuleManager,
        model_runtime_config: ModelRuntimeConfig,
    ):
        super().__init__(name, model_runtime_config)
        self.dit: FlashVSRModel = module_manager.fetch_module("flashvsr_dit")
        self.model_names = ["dit"]
        self.cur_process_idx = 0
        self.pre_LQ_video_cache = None

        # Handle torch.compile for single GPU mode
        parallel_cfg = model_runtime_config.parallel_config
        if parallel_cfg.world_size == 1 and model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **model_runtime_config.compile_config.get_compile_kwargs())

    def generate_noise(
        self, shape: tuple, seed: int | None = None, device: str = "cpu", dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Generate random noise tensor with optional seed."""
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise

    def _compute_topk_ratio(self, sparse_ratio: float, height: int, width: int) -> float:
        """Compute top-k ratio based on sparse ratio and image dimensions."""
        return sparse_ratio * 768 * 1280 / (height * width)

    @ProfilingContext4Debug("video proj stream forward")
    def _prepare_video_clips_for_projection(
        self, video_tensor: torch.Tensor, start_frame: int, num_clips: int, clip_length: int = 4, overlap: int = 3
    ) -> torch.Tensor:
        """Prepare video clips for projection by extracting overlapping segments."""
        LQ_latents_list = []

        for inner_idx in range(num_clips):
            # Calculate frame indices for current clip
            start_idx = start_frame + max(0, inner_idx * clip_length - overlap)
            end_idx = start_frame + (inner_idx + 1) * clip_length - overlap

            if end_idx > video_tensor.shape[2]:
                break

            video_clip = video_tensor[:, :, start_idx:end_idx, :, :].to(self.device)

            cur = (
                self.dit.proj_LQ_video_clip(
                    video_clip,
                    tile_size=self.proj_tile_size,
                    tile_stride=self.proj_tile_stride,
                    tile=self.proj_tile,
                )
                if video_tensor is not None
                else None
            )

            if cur is None:
                continue
            LQ_latents_list.append(cur[0])
        LQ_latents = torch.cat(LQ_latents_list, dim=1)

        return LQ_latents

    @ProfilingContext4Debug("denoising chunk")
    def _denoise_chunk(
        self,
        latents: torch.Tensor,
        LQ_latents: torch.Tensor,
        chunk_id: int,
        topk_ratio: float,
        kv_ratio: int,
        local_range: int,
        is_stream: bool = True,
    ) -> torch.Tensor:
        """Denoise a single chunk of latents."""
        cur_latents = latents[:, :, chunk_id * 2 : (chunk_id + 1) * 2, :, :].to(self.device)

        noise_pred_posi = self.dit(
            x=cur_latents,
            LQ_latents=[LQ_latents],
            is_stream=is_stream,
            topk_ratio=topk_ratio,
            kv_ratio=kv_ratio,
            cur_process_idx=self.cur_process_idx,
            local_range=local_range,
            offload_kvcache=False,
        )

        cur_latents = cur_latents - noise_pred_posi
        self.cur_process_idx += 1

        return cur_latents

    @with_metrics
    def process(self, method: str, *args, **kwargs):
        """Generic method dispatcher for different processing operations."""
        if hasattr(self, method):
            return getattr(self, method)(*args, **kwargs)
        else:
            raise NotImplementedError(f"{method} is not supported in flashvsr dit")

    @with_model_offload(["dit"])
    @ProfilingContext4Debug("dit_denoise")
    @torch.inference_mode()
    def denoise(
        self,
        LQ_video: torch.Tensor,
        seed: int,
        rand_device: str,
        height: int,
        width: int,
        sparse_ratio: float = 2.0,
        local_range: int = 9,
        proj_tile: bool = False,
        proj_tile_size: tuple[int, int] = (512, 1024),
        proj_tile_stride: tuple[int, int] = (384, 896),
        kv_ratio: int = 3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process low-quality video through denoising pipeline.

        Uses a two-phase approach: first chunk requires 25+ frames for initialization,
        subsequent chunks are processed in groups of 8 frames with cached context.
        """
        self.proj_tile = proj_tile
        self.proj_tile_size = proj_tile_size
        self.proj_tile_stride = proj_tile_stride
        topk_ratio = self._compute_topk_ratio(sparse_ratio, height, width)

        latents_total = []
        ref_LQ_video_total = []
        is_stream = True

        # First chunk requires at least 25 frames for proper initialization
        if self.cur_process_idx == 0:
            remaining_video = self._process_first_chunk(
                LQ_video,
                seed,
                rand_device,
                topk_ratio,
                local_range,
                kv_ratio,
                is_stream,
                latents_total,
                ref_LQ_video_total,
            )
        else:
            remaining_video = LQ_video

        # Process remaining chunks in 8-frame segments
        self._process_remaining_chunks(
            remaining_video,
            seed,
            rand_device,
            topk_ratio,
            local_range,
            kv_ratio,
            is_stream,
            latents_total,
            ref_LQ_video_total,
        )

        if latents_total:
            latents = torch.cat(latents_total, dim=2).cpu()
            ref_LQ_video = torch.cat(ref_LQ_video_total, dim=2)
        else:
            latents = torch.empty(0)
            ref_LQ_video = torch.empty(0)
        return latents, ref_LQ_video

    def _process_first_chunk(
        self,
        LQ_video: torch.Tensor,
        seed: int,
        rand_device: str,
        topk_ratio: float,
        local_range: int,
        kv_ratio: int,
        is_stream: bool,
        latents_total: list,
        ref_LQ_video_total: list,
    ) -> torch.Tensor:
        """Process the first chunk (requires 25 frames) and return remaining video."""
        if LQ_video.shape[2] < 25:
            raise RuntimeError(f"First chunk video should have more than 25 frames, got {LQ_video.shape[2]}")

        latents = self.generate_noise(
            (1, 16, 6, LQ_video.shape[-2] // 8, LQ_video.shape[-1] // 8),
            seed=seed,
            dtype=self.torch_dtype,
            device=rand_device,
        )

        # Cache last 4 frames for overlap with next chunk
        self.pre_LQ_video_cache = LQ_video[:, :, 21:25]

        LQ_latents = self._prepare_video_clips_for_projection(
            LQ_video, start_frame=0, num_clips=7, clip_length=4, overlap=3
        )

        if LQ_latents is not None:
            cur_latents = latents[:, :, :6, :, :].to(self.device)
            noise_pred_posi = self.dit(
                x=cur_latents,
                LQ_latents=[LQ_latents],
                is_stream=is_stream,
                topk_ratio=topk_ratio,
                kv_ratio=kv_ratio,
                cur_process_idx=self.cur_process_idx,
                local_range=local_range,
                offload_kvcache=False,
            )

            cur_latents = cur_latents - noise_pred_posi
            latents_total.append(cur_latents)

            cur_LQ_frames = LQ_video[:, :, :21]
            ref_LQ_video_total.append(cur_LQ_frames)

            self.cur_process_idx += 1

        return LQ_video[:, :, 25:]

    def _process_remaining_chunks(
        self,
        LQ_video: torch.Tensor,
        seed: int,
        rand_device: str,
        topk_ratio: float,
        local_range: int,
        kv_ratio: int,
        is_stream: bool,
        latents_total: list,
        ref_LQ_video_total: list,
    ) -> None:
        """Process remaining chunks in 8-frame segments with cached context."""
        loop_num = LQ_video.shape[2] // 8

        if loop_num == 0:
            logger.warning("No remaining chunks to process")
            return

        latents = self.generate_noise(
            (1, 16, loop_num * 2, LQ_video.shape[-2] // 8, LQ_video.shape[-1] // 8),
            seed=seed,
            dtype=self.torch_dtype,
            device=rand_device,
        )

        # Combine cached frames with remaining video for temporal continuity
        combined_video = torch.cat([self.pre_LQ_video_cache, LQ_video], dim=2)

        for chunk_id in tqdm(range(loop_num), desc="Denoising chunks"):
            LQ_latents = self._prepare_video_clips_for_projection(
                combined_video, start_frame=4 + chunk_id * 8, num_clips=2, clip_length=4, overlap=0
            )

            cur_latents = self._denoise_chunk(
                latents, LQ_latents, chunk_id, topk_ratio, kv_ratio, local_range, is_stream
            )

            cur_LQ_frames = combined_video[:, :, chunk_id * 8 : (chunk_id + 1) * 8].to(self.device)

            latents_total.append(cur_latents)
            ref_LQ_video_total.append(cur_LQ_frames)

        # Update cache with last 4 frames for next batch
        self.pre_LQ_video_cache = combined_video[:, :, -4:]

    def clean_cache(self) -> None:
        """Clean up cached data and reset processing index."""
        logger.info("clean flashvsr dit cache (LQ Proj cache and kv cache)")
        self.dit.clean_LQ_proj_in_cache()
        self.dit.clear_kv_cache()
        self.cur_process_idx = 0
        self.pre_LQ_video_cache = None

    def parallel_models(self) -> None:
        """Configure parallel processing for the denoising models."""
        parallel_cfg = self.model_runtime_config.parallel_config
        self.dit.device_mesh = create_device_mesh_from_config(parallel_cfg)
        self.dit.LQ_proj_in.set_parallelism(parallel_cfg.world_size)
        if parallel_cfg.sp_ulysses_degree > 1:
            self.dit.enable_usp()

        # Handle torch.compile for distributed mode
        if self.model_runtime_config.compile_config.enabled:
            logger.info("enable torch.compile for dit")
            self.dit = torch.compile(self.dit, **self.model_runtime_config.compile_config.get_compile_kwargs())
