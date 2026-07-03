from __future__ import annotations

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.models.lingbot_world_fast_dit import LingBotWorldFastDiT
from telefuser.models.t5_tokenizer import HuggingfaceTokenizer
from telefuser.models.wan_video_text_encoder import WanTextEncoder
from telefuser.models.wan_video_vae import WanVideoVAE
from telefuser.schedulers.unipc import FlowUniPCMultistepScheduler
from telefuser.utils.logging import logger
from telefuser.utils.model_weight import load_state_dict

from .control import (
    build_action_control_chunk,
    build_camera_control_chunk,
    compute_relative_poses,
    get_ks_transformed,
    interpolate_camera_poses,
    load_action_control_inputs,
    load_camera_control_inputs,
)
from .denoising import LingBotWorldFastDenoisingStage, LingBotWorldFastTimesteps
from .session import LingBotWorldFastRuntimeState, LingBotWorldFastSessionConfig


@dataclass
class LingBotWorldFastPipelineConfig:
    checkpoint_dir: str = ""
    fast_checkpoint_subdir: str = "lingbot_world_fast"
    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_torch_dtype: torch.dtype = torch.bfloat16
    control_type: str = "cam"
    orig_height: int = 480
    orig_width: int = 832
    max_area: int = 480 * 832


class LingBotWorldFastPipeline(BasePipeline):
    """Pipeline wrapper for LingBot-World-Fast chunked causal generation."""

    def __init__(self, device: str, torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.height_division_factor = 16
        self.width_division_factor = 16

    def _get_stages(self) -> list:
        return []

    @staticmethod
    def _notify_progress(
        progress_callback: Callable[..., None] | None,
        stage: str,
        **data: object,
    ) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(stage, **data)
        except Exception as exc:
            logger.warning(f"LingBot progress callback failed at stage={stage}: {exc}")

    def _runtime_device(self, runtime_config: ModelRuntimeConfig) -> torch.device:
        if runtime_config.device_type is None:
            return torch.device(self.device)
        if runtime_config.device_type == "cuda":
            return torch.device(f"cuda:{runtime_config.device_id}")
        return torch.device(runtime_config.device_type)

    def init(self, module_manager, config: LingBotWorldFastPipelineConfig) -> None:
        self.config = config
        checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
        self._model_info = [{"name": "lingbot_world_fast", "path": str(checkpoint_root)}]
        self.text_device = self._runtime_device(config.text_encoding_config)
        self.vae_device = self._runtime_device(config.vae_config)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.load_state_dict(load_state_dict(str(checkpoint_root / "models_t5_umt5-xxl-enc-bf16.pth")))
        self.text_encoder = self.text_encoder.to(device=self.text_device, dtype=self.torch_dtype).eval()
        tokenizer_path = checkpoint_root / "google" / "umt5-xxl"
        self.tokenizer = HuggingfaceTokenizer(str(tokenizer_path), 512, "whitespace")

        vae_state_dict, vae_cfg = WanVideoVAE.state_dict_converter().from_official(
            load_state_dict(str(checkpoint_root / "Wan2.1_VAE.pth"))
        )
        self.vae = WanVideoVAE(**vae_cfg)
        self.vae.load_state_dict(vae_state_dict, strict=False)
        self.vae = self.vae.to(device=self.vae_device, dtype=self.torch_dtype).eval()

        fast_path = Path(config.fast_checkpoint_subdir).expanduser()
        if not fast_path.is_absolute():
            fast_path = (checkpoint_root / fast_path).resolve()
        else:
            fast_path = fast_path.resolve()
        self.dit = LingBotWorldFastDiT.from_pretrained(
            str(fast_path),
            torch_dtype=config.dit_torch_dtype,
            control_type=config.control_type,
            config={"patch_size": (1, 2, 2), "text_len": 512, "control_type": config.control_type},
        ).to(self.device)
        self.dit.eval().requires_grad_(False)

        self.denoise_stage = LingBotWorldFastDenoisingStage(self.dit, torch_dtype=self.torch_dtype)
        self.timesteps = LingBotWorldFastTimesteps()

    @torch.inference_mode()
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.text_device)
        mask = mask.to(self.text_device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb.to(self.device)

    @torch.inference_mode()
    def decode_video_cached(self, latents: torch.Tensor, is_first_clip: bool, is_last_clip: bool) -> torch.Tensor:
        return self.vae.cached_decode_withflag(
            latents,
            device=self.vae_device,
            is_first_clip=is_first_clip,
            is_last_clip=is_last_clip,
        )

    @staticmethod
    def _best_output_size(w: int, h: int, expected_area: int, dw: int = 16, dh: int = 16) -> tuple[int, int]:
        ratio = w / h
        ow = math.sqrt(expected_area * ratio)
        oh = expected_area / ow

        ow1 = int(ow // dw * dw)
        oh1 = int(expected_area / max(ow1, 1) // dh * dh)
        oh2 = int(oh // dh * dh)
        ow2 = int(expected_area / max(oh2, 1) // dw * dw)

        if ow1 <= 0 or oh1 <= 0:
            return max(dw, ow2), max(dh, oh2)
        if ow2 <= 0 or oh2 <= 0:
            return max(dw, ow1), max(dh, oh1)

        ratio1 = ow1 / oh1
        ratio2 = ow2 / oh2
        if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
            return ow1, oh1
        return ow2, oh2

    def _prepare_image_tensor(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        image = image.convert("RGB").resize((width, height), Image.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(array).permute(2, 0, 1).sub_(0.5).div_(0.5).to(self.vae_device, dtype=self.torch_dtype)
        )
        return tensor

    def _encode_condition_video(self, image_tensor: torch.Tensor, frame_num: int) -> torch.Tensor:
        h, w = image_tensor.shape[1:]
        video = torch.cat(
            [
                image_tensor.unsqueeze(1),
                torch.zeros(3, frame_num - 1, h, w, device=image_tensor.device, dtype=image_tensor.dtype),
            ],
            dim=1,
        )
        latent = self.vae.encode([video], device=self.vae_device, tiled=False)[0].to(self.device)

        latent_h = latent.shape[2]
        latent_w = latent.shape[3]
        mask = torch.ones(1, frame_num, latent_h, latent_w, device=self.device, dtype=latent.dtype)
        mask[:, 1:] = 0
        mask = torch.cat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, latent_h, latent_w).transpose(1, 2)[0]
        return torch.cat([mask, latent], dim=0)

    def _init_self_kv_cache(
        self,
        batch_size: int,
        kv_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
    ) -> list[dict[str, torch.Tensor]]:
        head_dim = self.dit.dim // self.dit.num_heads
        shape = (batch_size, kv_size, self.dit.num_heads, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=dtype, device=device),
                "v": torch.zeros(shape, dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
            for _ in range(self.dit.num_layers)
        ]

    def _init_crossattn_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: str | torch.device,
        max_sequence_length: int,
    ) -> list[dict[str, torch.Tensor | bool]]:
        head_dim = self.dit.dim // self.dit.num_heads
        shape = (batch_size, self.dit.num_heads, max_sequence_length, head_dim)
        return [
            {
                "k": torch.zeros(shape, dtype=dtype, device=device),
                "v": torch.zeros(shape, dtype=dtype, device=device),
                "is_init": False,
            }
            for _ in range(self.dit.num_layers)
        ]

    def _populate_session_controls(self, session_config: LingBotWorldFastSessionConfig) -> None:
        if session_config.action_path is None:
            return
        if session_config.poses is not None and session_config.intrinsics is not None:
            if session_config.control_mode != "act" or session_config.action is not None:
                return

        if session_config.control_mode == "act":
            poses, intrinsics, action = load_action_control_inputs(session_config.action_path)
            if session_config.action is None:
                session_config.action = action
        else:
            poses, intrinsics = load_camera_control_inputs(session_config.action_path)

        if session_config.poses is None:
            session_config.poses = poses
        if session_config.intrinsics is None:
            session_config.intrinsics = intrinsics

    @staticmethod
    def _truncate_control_inputs_to_frame_num(
        poses: object,
        intrinsics: object,
        action: object | None,
        frame_num: int,
    ) -> tuple[object, object, object | None, int]:
        requested_frame_num = ((int(frame_num) - 1) // 4) * 4 + 1
        pose_frame_num = ((len(poses) - 1) // 4) * 4 + 1
        effective_frame_num = min(requested_frame_num, pose_frame_num)

        trimmed_poses = poses[:effective_frame_num]
        intrinsics_arr = np.asarray(intrinsics)
        if intrinsics_arr.ndim > 1:
            trimmed_intrinsics = intrinsics[:effective_frame_num]
        else:
            trimmed_intrinsics = intrinsics
        trimmed_action = action[:effective_frame_num] if action is not None else None
        return trimmed_poses, trimmed_intrinsics, trimmed_action, effective_frame_num

    @staticmethod
    def _align_action_frames(action: torch.Tensor, target_frames: int) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == target_frames:
            return action

        sampled = action[::4]
        if sampled.shape[0] >= target_frames:
            return sampled[:target_frames]
        if action.shape[0] == 1:
            return action.repeat(target_frames, 1)
        if action.shape[0] < target_frames:
            raise ValueError(f"Action length {action.shape[0]} is shorter than target latent frames {target_frames}")

        indices = torch.linspace(0, action.shape[0] - 1, target_frames, device=action.device)
        return action.index_select(0, indices.round().long())

    def _prepare_control_chunks(
        self,
        session_config: LingBotWorldFastSessionConfig,
        lat_f: int,
        lat_h: int,
        lat_w: int,
        height: int,
        width: int,
        chunk_size: int,
    ) -> list[torch.Tensor] | None:
        if session_config.poses is None or session_config.intrinsics is None:
            return None

        poses = torch.as_tensor(session_config.poses, dtype=torch.float32)
        intrinsics = torch.as_tensor(session_config.intrinsics, dtype=torch.float32)
        intrinsics = get_ks_transformed(
            intrinsics,
            height_org=self.config.orig_height,
            width_org=self.config.orig_width,
            height_resize=height,
            width_resize=width,
            height_final=height,
            width_final=width,
        )

        len_c2ws = len(poses)
        lat_f_target = int(lat_f - (lat_f % chunk_size))
        poses_interp = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=np.asarray(poses[:, :3, :3]),
            src_trans_vec=np.asarray(poses[:, :3, 3]),
            tgt_indices=np.linspace(0, len_c2ws - 1, lat_f_target),
        )
        poses_rel = compute_relative_poses(poses_interp.to(self.device), framewise=True)
        intrinsics = intrinsics[0].to(self.device).repeat(len(poses_rel), 1)

        if session_config.control_mode == "act" and session_config.action is not None:
            action = torch.as_tensor(session_config.action, dtype=torch.float32, device=self.device)
            action = self._align_action_frames(action, len(poses_rel))
            chunk = build_action_control_chunk(poses_rel, intrinsics, action, lat_h, lat_w, height, width)
        else:
            chunk = build_camera_control_chunk(poses_rel, intrinsics, lat_h, lat_w, height, width)

        return list(chunk.control_tensor.split(chunk_size, dim=2))

    def build_control_override(
        self,
        runtime: LingBotWorldFastRuntimeState,
        chunk: dict,
    ) -> torch.Tensor | None:
        if "control_tensor" in chunk:
            return torch.as_tensor(chunk["control_tensor"], device=self.device, dtype=self.torch_dtype)

        poses = chunk.get("poses")
        intrinsics = chunk.get("intrinsics")
        if poses is None or intrinsics is None:
            return None

        poses_t = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
        intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.device)
        if intrinsics_t.ndim == 1:
            intrinsics_t = intrinsics_t.unsqueeze(0).repeat(poses_t.shape[0], 1)

        intrinsics_t = get_ks_transformed(
            intrinsics_t,
            height_org=self.config.orig_height,
            width_org=self.config.orig_width,
            height_resize=runtime.height,
            width_resize=runtime.width,
            height_final=runtime.height,
            width_final=runtime.width,
        )
        poses_rel = compute_relative_poses(poses_t, framewise=True)

        if chunk.get("action") is not None:
            action_t = torch.as_tensor(chunk["action"], dtype=torch.float32, device=self.device)
            action_t = self._align_action_frames(action_t, len(poses_rel))
            control = build_action_control_chunk(
                poses_rel,
                intrinsics_t,
                action_t,
                runtime.latent_h,
                runtime.latent_w,
                runtime.height,
                runtime.width,
            )
        else:
            control = build_camera_control_chunk(
                poses_rel,
                intrinsics_t,
                runtime.latent_h,
                runtime.latent_w,
                runtime.height,
                runtime.width,
            )
        return control.control_tensor

    @torch.inference_mode()
    def create_runtime(
        self,
        session_config: LingBotWorldFastSessionConfig,
        progress_callback: Callable[..., None] | None = None,
    ) -> LingBotWorldFastRuntimeState:
        self._notify_progress(progress_callback, "loading_controls")
        self._populate_session_controls(session_config)
        self._notify_progress(progress_callback, "encoding_prompt", device=str(self.text_device))
        prompt_emb = self.encode_prompt(session_config.prompt)
        self._notify_progress(progress_callback, "prompt_encoded")

        w0, h0 = session_config.image.size
        width, height = self._best_output_size(w0, h0, self.config.max_area)
        height, width = self.check_resize_height_width(height, width)

        self._notify_progress(progress_callback, "preparing_image", width=width, height=height)
        image_tensor = self._prepare_image_tensor(session_config.image, height, width)

        frame_num = ((session_config.frame_num - 1) // 4) * 4 + 1
        if session_config.poses is not None and session_config.intrinsics is not None:
            session_config.poses, session_config.intrinsics, session_config.action, frame_num = (
                self._truncate_control_inputs_to_frame_num(
                    poses=session_config.poses,
                    intrinsics=session_config.intrinsics,
                    action=session_config.action,
                    frame_num=frame_num,
                )
            )
        self._notify_progress(
            progress_callback,
            "encoding_condition_video",
            frame_num=frame_num,
            device=str(self.vae_device),
        )
        latent_condition = self._encode_condition_video(image_tensor, frame_num)
        self._notify_progress(progress_callback, "condition_video_encoded")

        lat_h = height // 8
        lat_w = width // 8
        lat_f = (frame_num - 1) // 4 + 1
        lat_f = int(lat_f - (lat_f % session_config.chunk_size))
        frame_num = (lat_f - 1) * 4 + 1
        patch_area = self.dit.patch_size[1] * self.dit.patch_size[2]
        frame_tokens = (lat_h * lat_w) // patch_area
        kv_size = frame_tokens * lat_f
        max_seq_len = session_config.chunk_size * frame_tokens
        max_attention_size = (
            kv_size if session_config.max_attention_size is None else int(session_config.max_attention_size)
        )

        self._notify_progress(
            progress_callback,
            "allocating_runtime",
            latent_frames=lat_f,
            latent_height=lat_h,
            latent_width=lat_w,
            total_chunks=math.ceil(lat_f / session_config.chunk_size),
        )
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(session_config.seed))
        noise = torch.randn(
            (1, 16, lat_f, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        noise_chunks = list(noise.split(session_config.chunk_size, dim=2))
        condition_chunks = list(latent_condition.unsqueeze(0).split(session_config.chunk_size, dim=2))
        self._notify_progress(progress_callback, "preparing_controls")
        control_chunks = self._prepare_control_chunks(
            session_config=session_config,
            lat_f=lat_f,
            lat_h=lat_h,
            lat_w=lat_w,
            height=height,
            width=width,
            chunk_size=session_config.chunk_size,
        )

        runtime = LingBotWorldFastRuntimeState(
            prompt_emb=prompt_emb,
            encoded_image_latent=latent_condition,
            noise_chunks=noise_chunks,
            condition_chunks=condition_chunks,
            control_chunks=control_chunks,
            timesteps=torch.empty(0, dtype=torch.int64),
            self_kv_cache=self._init_self_kv_cache(
                batch_size=1,
                kv_size=kv_size,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            crossattn_cache=self._init_crossattn_cache(
                batch_size=1,
                dtype=self.torch_dtype,
                device=self.device,
                max_sequence_length=session_config.max_sequence_length,
            ),
            latent_h=lat_h,
            latent_w=lat_w,
            latent_f=lat_f,
            height=height,
            width=width,
            max_seq_len=max_seq_len,
            frame_tokens=frame_tokens,
            chunk_size=session_config.chunk_size,
            max_attention_size=max_attention_size,
            scheduler=FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False),
            generator=generator,
        )
        runtime.timesteps = self.timesteps.select(runtime.scheduler, session_config.sample_shift)
        self._notify_progress(progress_callback, "runtime_created", width=width, height=height, latent_frames=lat_f)
        logger.info(f"LingBot runtime created: {width}x{height}, latent={lat_f}x{lat_h}x{lat_w}")
        return runtime

    @torch.inference_mode()
    def generate_next_chunk(
        self,
        runtime: LingBotWorldFastRuntimeState,
        control_override: torch.Tensor | None = None,
        progress_callback: Callable[..., None] | None = None,
    ) -> list[Image.Image]:
        if runtime.current_chunk_index >= len(runtime.noise_chunks):
            runtime.active = False
            return []

        idx = runtime.current_chunk_index
        latent_chunk = runtime.noise_chunks[idx]
        condition_chunk = runtime.condition_chunks[idx]
        control_chunk = control_override
        if control_chunk is None and runtime.control_chunks is not None and idx < len(runtime.control_chunks):
            control_chunk = runtime.control_chunks[idx]

        self._notify_progress(progress_callback, "denoising_chunk", index=idx)
        current_start = idx * runtime.chunk_size * runtime.frame_tokens
        denoised = self.denoise_stage.denoise_chunk(
            latent_chunk=latent_chunk,
            condition_chunk=condition_chunk,
            prompt_emb=runtime.prompt_emb,
            timesteps=runtime.timesteps,
            scheduler=runtime.scheduler,
            control_chunk=control_chunk,
            self_kv_cache=runtime.self_kv_cache,
            crossattn_cache=runtime.crossattn_cache,
            current_start=current_start,
            max_attention_size=runtime.max_attention_size,
            generator=runtime.generator,
        )

        self._notify_progress(progress_callback, "updating_cache", index=idx)
        self.dit(
            x=denoised.to(dtype=self.torch_dtype),
            timestep=torch.zeros((1,), dtype=torch.float32, device=self.device),
            context=runtime.prompt_emb,
            y=condition_chunk,
            control_tensor=control_chunk,
            kv_cache=runtime.self_kv_cache,
            crossattn_cache=runtime.crossattn_cache,
            current_start=current_start,
            max_attention_size=runtime.max_attention_size,
        )

        self._notify_progress(progress_callback, "decoding_chunk", index=idx, device=str(self.vae_device))
        frames = self.decode_video_cached(
            denoised,
            is_first_clip=(idx == 0),
            is_last_clip=(idx == len(runtime.noise_chunks) - 1),
        )
        images = self.tensor2video(frames)
        self._notify_progress(progress_callback, "chunk_decoded", index=idx, frames=len(images))
        runtime.current_chunk_index += 1
        runtime.emitted_frames += len(images)
        if runtime.current_chunk_index >= len(runtime.noise_chunks):
            runtime.active = False
        return images

    @staticmethod
    def encode_frames_to_b64(frames: list[Image.Image], quality: int = 85) -> list[str]:
        encoded: list[str] = []
        for frame in frames:
            rgb = np.asarray(frame.convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                encoded.append(base64.b64encode(buf.tobytes()).decode("ascii"))
        return encoded
