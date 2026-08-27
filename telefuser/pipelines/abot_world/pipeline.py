"""Native TeleFuser single-GPU pipeline for ABot-World."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from PIL import Image

from telefuser.core.base_pipeline import BasePipeline
from telefuser.core.config import ModelRuntimeConfig
from telefuser.core.module_manager import ModuleManager
from telefuser.pipelines.wan_video.text_encoding import TextEncodingStage
from telefuser.pipelines.wan_video.vae import VAEStage

from .denoising import ABotWorldDenoisingStage
from .taew_vae import ABotWorldTAEWDecodeStage


@dataclass
class ABotWorldPipelineConfig:
    """Runtime settings for the public ABot-World-0-5B-LF checkpoint."""

    vae_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    text_encoding_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    dit_config: ModelRuntimeConfig = field(default_factory=ModelRuntimeConfig)
    height: int = 480
    width: int = 832
    latent_frames: int = 31
    # Match LingBot-World v2: six fixed sink latents plus a twelve-latent rolling tail.
    local_attn_size: int = 18
    sink_size: int = 6
    # Opt-in only: capture the fixed-shape, steady-state Relative-RoPE DiT
    # continuation path. Dynamic first chunks, cache warmup, VAE and output
    # postprocessing remain eager.
    cuda_graph_enabled: bool = False


class ABotWorldPipeline(BasePipeline):
    """Image-conditioned, action-controlled ABot-World inference on one GPU.

    ``latent_frames`` is the causal Wan latent-frame count (the shipped model
    configuration uses 31).  It must be ``1 mod 3`` because the first image
    context is generated as one block and later blocks contain three latents.
    """

    clear_memory_after_call = False
    _ACTION_ORDER = ("W", "A", "S", "D", "I", "J", "K", "L")

    def __init__(self, device: str | torch.device = "cuda", torch_dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__(device=device, torch_dtype=torch_dtype)
        # VAE spatial compression (16) plus DiT spatial patching (2).
        self.height_division_factor = 32
        self.width_division_factor = 32

    def _get_stages(self) -> list:
        return [self.vae_stage, self.text_encoding_stage, self.denoise_stage, self.taew_decode_stage]

    def init(self, module_manager: ModuleManager, config: ABotWorldPipelineConfig) -> None:
        if config.dit_config.parallel_config.world_size != 1:
            raise ValueError("ABot-World initial integration supports exactly one GPU")
        if config.local_attn_size < 1:
            raise ValueError("local_attn_size must be positive")
        if not 0 <= config.sink_size < config.local_attn_size:
            raise ValueError("sink_size must be non-negative and smaller than local_attn_size")
        if config.latent_frames < 1 or (config.latent_frames - 1) % 3:
            raise ValueError("latent_frames must be positive and equal to 1 mod 3")
        height, width = self.check_resize_height_width(config.height, config.width)
        if (height, width) != (config.height, config.width):
            raise ValueError("ABot height and width must already be divisible by 32")
        self._model_info = module_manager.get_model_info()
        self.config = config
        self.vae_stage = VAEStage("abot_world_vae", module_manager, config.vae_config)
        self.taew_decode_stage = ABotWorldTAEWDecodeStage("abot_world_taew_decode", module_manager, config.vae_config)
        self.text_encoding_stage = TextEncodingStage(
            "abot_world_text_encoding", module_manager, config.text_encoding_config
        )
        self.denoise_stage = ABotWorldDenoisingStage("abot_world_denoise", module_manager, config.dit_config)
        self.denoise_stage.parallel_models()
        self.denoise_stage.configure_cuda_graph(config.cuda_graph_enabled)
        self.denoise_stage.dit.set_causal_attention_window(config.local_attn_size, config.sink_size)

    @classmethod
    def build_action_context(
        cls,
        keys: Mapping[str, bool] | None,
        *,
        latent_frames: int,
        height: int,
        width: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create the official 32-channel action map from WASD/IJKL key state."""
        if latent_frames < 1:
            raise ValueError("latent_frames must be positive")
        key_state = {} if keys is None else keys
        unknown = set(key_state).difference(cls._ACTION_ORDER)
        if unknown:
            raise ValueError(f"Unknown ABot action keys: {sorted(unknown)}")
        values = [float(bool(key_state.get(key, False))) for key in cls._ACTION_ORDER]
        base = torch.tensor(values, device=device, dtype=dtype).view(1, 8, 1, 1, 1)
        return base.expand(1, 8, latent_frames, height, width).repeat_interleave(4, dim=1).contiguous()

    @torch.inference_mode()
    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        actions: Mapping[str, bool] | None = None,
        seed: int = 42,
    ) -> list[Image.Image]:
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        image = image.convert("RGB")
        pixels = self.preprocess_image(image, self.config.height, self.config.width)
        start_latent, _ = self.vae_stage.process(
            "encode_image",
            pixels,
            None,
            1,
            concat_mask=False,
        )
        first_frame_latent = start_latent.unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
        latent_height, latent_width = first_frame_latent.shape[-2:]
        noise = self.generate_noise(
            (1, first_frame_latent.shape[1], self.config.latent_frames, latent_height, latent_width),
            seed=seed,
            device=self.device,
            dtype=torch.float32,
        )
        action_context = self.build_action_context(
            actions,
            latent_frames=self.config.latent_frames,
            height=self.config.height,
            width=self.config.width,
            device=self.device,
            dtype=self.torch_dtype,
        )
        prompt_emb = self.text_encoding_stage.process([prompt])[0]
        latents = self.denoise_stage.process(noise, prompt_emb, action_context, first_frame_latent, seed)
        frames = self.vae_stage.process("decode_video", latents)
        return self.tensor2video(frames[0])

    def close(self) -> None:
        """Release stage references explicitly for scripts that run once."""
        for name in ("vae_stage", "text_encoding_stage", "denoise_stage"):
            if hasattr(self, name):
                getattr(self, name).offload_models()
