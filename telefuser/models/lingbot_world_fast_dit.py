from __future__ import annotations

import math
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig
from telefuser.distributed.device_mesh import get_ulysses_group, get_ulysses_world_size
from telefuser.distributed.parallel_shard import sequence_parallel_shard, sequence_parallel_unshard
from telefuser.distributed.ulysses_comm import ulysses_gather_heads, ulysses_scatter_heads
from telefuser.ops.attention import attention as attn_func
from telefuser.ops.normalization import LayerNorm, RMSNorm
from telefuser.utils.logging import logger
from telefuser.utils.model_weight import init_weights_on_device, load_state_dict

from .wan_video_dit import apply_rotary_emb, precompute_freqs_cis_3d, sinusoidal_embedding_1d


def _cache_index_to_int(value: int | torch.Tensor) -> int:
    if isinstance(value, int):
        return value
    return int(value.item())


def _set_cache_index(cache: dict[str, torch.Tensor | int], name: str, value: int) -> None:
    """Update an eager cursor in place when its storage is a scalar tensor."""
    cursor = cache[name]
    if isinstance(cursor, torch.Tensor):
        cursor.fill_(value)
    else:
        cache[name] = value


class CausalSelfAttention(nn.Module):
    """Causal self-attention with rolling KV cache support."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} is not divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.attention_config = AttentionConfig()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _apply_causal_rope(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        grid_size: tuple[int, int, int],
        start_frame: int,
    ) -> torch.Tensor:
        f, h, w = grid_size
        head_dim = self.head_dim

        f_dim = head_dim - 2 * (head_dim // 3)
        h_dim = head_dim // 3
        w_dim = head_dim // 3
        seq_len = f * h * w

        cos_f, cos_h, cos_w = torch.split(freqs_cos, [f_dim // 2, h_dim // 2, w_dim // 2], dim=-1)
        sin_f, sin_h, sin_w = torch.split(freqs_sin, [f_dim // 2, h_dim // 2, w_dim // 2], dim=-1)

        cos = torch.cat(
            [
                cos_f[start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                cos_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                cos_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        sin = torch.cat(
            [
                sin_f[start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                sin_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                sin_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        roped = apply_rotary_emb(x[:, :seq_len], (cos, sin))
        if x.shape[1] == seq_len:
            return roped
        return torch.cat([roped, x[:, seq_len:]], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        grid_size: tuple[int, int, int],
        kv_cache: dict[str, torch.Tensor | int],
        current_start: int,
        max_attention_size: int,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        q = rearrange(self.norm_q(self.q(x)), "b s (n d) -> b s n d", n=self.num_heads)
        k = rearrange(self.norm_k(self.k(x)), "b s (n d) -> b s n d", n=self.num_heads)
        v = rearrange(self.v(x), "b s (n d) -> b s n d", n=self.num_heads)

        group = get_ulysses_group(device_mesh)
        ulysses_enabled = group is not None and get_ulysses_world_size(device_mesh) > 1
        frame_tokens = grid_size[1] * grid_size[2]
        start_frame = current_start // frame_tokens
        valid_seq_len = math.prod(grid_size)
        if ulysses_enabled:
            q = ulysses_scatter_heads(q, group)()
            k = ulysses_scatter_heads(k, group)()
            v = ulysses_scatter_heads(v, group)()
            padded_seq_len = q.shape[1]
            q = self._apply_causal_rope(q, freqs_cos, freqs_sin, grid_size, start_frame)[:, :valid_seq_len]
            k = self._apply_causal_rope(k, freqs_cos, freqs_sin, grid_size, start_frame)[:, :valid_seq_len]
            v = v[:, :valid_seq_len]
        else:
            q = self._apply_causal_rope(q, freqs_cos, freqs_sin, grid_size, start_frame)
            k = self._apply_causal_rope(k, freqs_cos, freqs_sin, grid_size, start_frame)

        num_new_tokens = q.shape[1]
        current_end = current_start + num_new_tokens
        sink_tokens = self.sink_size * frame_tokens

        cache_k = kv_cache["k"]
        cache_v = kv_cache["v"]
        kv_cache_size = cache_k.shape[1]
        global_end = _cache_index_to_int(kv_cache["global_end_index"])
        local_end = _cache_index_to_int(kv_cache["local_end_index"])

        if self.local_attn_size != -1 and current_end > global_end and num_new_tokens + local_end > kv_cache_size:
            evicted = num_new_tokens + local_end - kv_cache_size
            rolled = local_end - evicted - sink_tokens
            cache_k[:, sink_tokens : sink_tokens + rolled] = cache_k[
                :, sink_tokens + evicted : sink_tokens + evicted + rolled
            ].clone()
            cache_v[:, sink_tokens : sink_tokens + rolled] = cache_v[
                :, sink_tokens + evicted : sink_tokens + evicted + rolled
            ].clone()
            local_end = local_end + current_end - global_end - evicted
        else:
            local_end = local_end + current_end - global_end

        local_start = local_end - num_new_tokens
        cache_k[:, local_start:local_end] = k
        cache_v[:, local_start:local_end] = v

        attn_start = max(0, local_end - max_attention_size)
        k_cache = cache_k[:, attn_start:local_end]
        v_cache = cache_v[:, attn_start:local_end]

        out = attn_func(
            q,
            k_cache,
            v_cache,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )

        _set_cache_index(kv_cache, "global_end_index", current_end)
        _set_cache_index(kv_cache, "local_end_index", local_end)

        if ulysses_enabled:
            pad_len = padded_seq_len - num_new_tokens
            if pad_len > 0:
                out = F.pad(out, (0, 0, 0, 0, 0, pad_len))
            out = ulysses_gather_heads(out, group, num_heads=self.num_heads)()

        out = rearrange(out, "b s n d -> b s (n d)")
        return self.o(out)


class CachedCrossAttention(nn.Module):
    """Cross-attention with persistent cached K/V for static text context."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attention_config = AttentionConfig()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cache: dict[str, torch.Tensor | bool] | None,
    ) -> torch.Tensor:
        q = rearrange(self.norm_q(self.q(x)), "b s (n d) -> b s n d", n=self.num_heads)

        if cache is not None and bool(cache.get("is_init", False)):
            k = cache["k"]
            v = cache["v"]
        else:
            k = rearrange(self.norm_k(self.k(context)), "b s (n d) -> b s n d", n=self.num_heads)
            v = rearrange(self.v(context), "b s (n d) -> b s n d", n=self.num_heads)
            if cache is not None:
                cache["k"] = k
                cache["v"] = v
                cache["is_init"] = True

        out = attn_func(
            q,
            k,
            v,
            attention_config=self.attention_config,
            input_layout="BSND",
            output_layout="BSND",
        )
        out = rearrange(out, "b s n d -> b s (n d)")
        return self.o(out)


class Gate(nn.Module):
    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class LingBotWorldFastBlock(nn.Module):
    """Causal transformer block with optional camera/action injection."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.self_attn = CausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            qk_norm=qk_norm,
            eps=eps,
        )
        self.cross_attn = CachedCrossAttention(dim=dim, num_heads=num_heads, eps=eps)
        self.norm1 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = LayerNorm(dim, eps=eps) if cross_attn_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = Gate()

        self.cam_injector_layer1 = nn.Linear(dim, dim)
        self.cam_injector_layer2 = nn.Linear(dim, dim)
        self.cam_scale_layer = nn.Linear(dim, dim)
        self.cam_shift_layer = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        t_mod: torch.Tensor,
        context: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        grid_size: tuple[int, int, int],
        kv_cache: dict[str, torch.Tensor | int],
        crossattn_cache: dict[str, torch.Tensor | bool],
        current_start: int,
        max_attention_size: int,
        control_tokens: torch.Tensor | None = None,
        device_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        modulation = self.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (modulation.unsqueeze(0) + t_mod).chunk(
            6, dim=2
        )

        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        shift_mlp = shift_mlp.squeeze(2)
        scale_mlp = scale_mlp.squeeze(2)
        gate_mlp = gate_mlp.squeeze(2)

        attn_in = self.norm1(x) * (1 + scale_msa) + shift_msa
        attn_out = self.self_attn(
            attn_in,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            grid_size=grid_size,
            kv_cache=kv_cache,
            current_start=current_start,
            max_attention_size=max_attention_size,
            device_mesh=device_mesh,
        )
        x = self.gate(x, gate_msa, attn_out)

        if control_tokens is not None:
            hidden = self.cam_injector_layer2(F.silu(self.cam_injector_layer1(control_tokens)))
            hidden = hidden + control_tokens
            x = (1.0 + self.cam_scale_layer(hidden)) * x + self.cam_shift_layer(hidden)

        x = x + self.cross_attn(self.norm3(x), context, crossattn_cache)
        mlp_in = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = self.gate(x, gate_mlp, self.ffn(mlp_in))
        return x


class LingBotWorldFastHead(nn.Module):
    """Output projection head."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.norm = LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        modulation = self.modulation.to(dtype=t.dtype, device=t.device)
        shift, scale = (modulation.unsqueeze(0) + t.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
        return self.head(self.norm(x) * (1 + scale) + shift)


class LingBotWorldFastDiT(BaseModel):
    """LingBot-World-Fast causal DiT backbone."""

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 32,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 40,
        num_layers: int = 40,
        local_attn_size: int = -1,
        sink_size: int = 0,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        control_type: str = "cam",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.control_type = control_type
        self.layer_name_list = ["blocks"]

        control_dim = 6 if control_type == "cam" else 7
        control_patch_dim = control_dim * 64 * patch_size[0] * patch_size[1] * patch_size[2]

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_wancamctrl = nn.Linear(control_patch_dim, dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(dim, dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(dim, dim)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList(
            [
                LingBotWorldFastBlock(
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    qk_norm=qk_norm,
                    cross_attn_norm=cross_attn_norm,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = LingBotWorldFastHead(dim=dim, out_dim=out_dim, patch_size=patch_size, eps=eps)

        head_dim = dim // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        freqs = precompute_freqs_cis_3d(head_dim)
        self.freqs_cos = torch.cat([f.real for f in freqs], dim=-1)
        self.freqs_sin = torch.cat([f.imag for f in freqs], dim=-1)
        self.device_mesh: DeviceMesh | None = None
        self.usp_flag = False

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.xavier_uniform_(self.patch_embedding_wancamctrl.weight)
        nn.init.zeros_(self.patch_embedding_wancamctrl.bias)
        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)

    def _build_timestep_embeddings(self, t: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.amp.autocast(t.device.type, dtype=torch.float32, enabled=t.device.type == "cuda"):
            if t.dim() == 1:
                emb_input = sinusoidal_embedding_1d(self.freq_dim, t).float()
                emb = self.time_embedding(emb_input)
                t_mod = self.time_projection(emb).unflatten(1, (6, self.dim)).unsqueeze(1).expand(-1, seq_len, -1, -1)
                return emb.unsqueeze(1).float(), t_mod.float()

            flat = t.flatten()
            emb_input = sinusoidal_embedding_1d(self.freq_dim, flat).float()
            emb = self.time_embedding(emb_input)
            emb = emb.unflatten(0, (t.shape[0], seq_len))
            t_mod = self.time_projection(emb).unflatten(2, (6, self.dim))
            return emb.float(), t_mod.float()

    def _prepare_control_tokens(self, control_tensor: torch.Tensor | None) -> torch.Tensor | None:
        if control_tensor is None:
            return None

        control_tensor = control_tensor.to(
            device=self.patch_embedding_wancamctrl.weight.device,
            dtype=self.patch_embedding_wancamctrl.weight.dtype,
        )
        control_tokens = rearrange(
            control_tensor,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )
        control_tokens = self.patch_embedding_wancamctrl(control_tokens)
        hidden = self.c2ws_hidden_states_layer2(F.silu(self.c2ws_hidden_states_layer1(control_tokens)))
        return control_tokens + hidden

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        return rearrange(x, "b c f h w -> b (f h w) c").contiguous(), grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: tuple[int, int, int]) -> torch.Tensor:
        return rearrange(
            x,
            "b (f h w) (p1 p2 p3 c) -> b c (f p1) (h p2) (w p3)",
            f=grid_size[0],
            h=grid_size[1],
            w=grid_size[2],
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor | None = None,
        control_tensor: torch.Tensor | None = None,
        kv_cache: list[dict[str, torch.Tensor | int]] | None = None,
        crossattn_cache: list[dict[str, torch.Tensor | bool]] | None = None,
        current_start: int = 0,
        max_attention_size: int = 1_000_000,
    ) -> torch.Tensor:
        if y is not None:
            y = y.to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, y], dim=1)

        x, grid_size = self.patchify(x)
        seq_len = x.shape[1]
        t_head, t_mod = self._build_timestep_embeddings(timestep, seq_len)
        context = self.text_embedding(context)
        control_tokens = self._prepare_control_tokens(control_tensor)

        freqs_cos = self.freqs_cos.to(device=x.device)
        freqs_sin = self.freqs_sin.to(device=x.device)
        full_seq_len = seq_len

        if self.usp_flag:
            shard_tensors = [x, t_mod, control_tokens]
            shard_dims = [1, 1, 1]
            if t_head.shape[1] == seq_len:
                shard_tensors.append(t_head)
                shard_dims.append(1)
            sequence_parallel_shard(self.device_mesh, shard_tensors, shard_dims)

        if kv_cache is None or crossattn_cache is None:
            raise ValueError("LingBotWorldFastDiT requires kv_cache and crossattn_cache")

        for idx, block in enumerate(self.blocks):
            x = block(
                x,
                t_mod=t_mod,
                context=context,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                grid_size=grid_size,
                kv_cache=kv_cache[idx],
                crossattn_cache=crossattn_cache[idx],
                current_start=current_start,
                max_attention_size=max_attention_size,
                control_tokens=control_tokens,
                device_mesh=self.device_mesh,
            )

        x = self.head(x, t_head)
        if self.usp_flag:
            (x,) = sequence_parallel_unshard(self.device_mesh, [x], [1], [full_seq_len])
        output = self.unpatchify(x, grid_size)
        return output.float()

    def enable_usp(self, device_mesh: DeviceMesh | None = None) -> None:
        """Enable Ulysses sequence parallelism for LingBot-World-Fast DiT."""
        self.device_mesh = device_mesh if device_mesh is not None else self.device_mesh
        self.usp_flag = get_ulysses_world_size(self.device_mesh) > 1

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        """Set the unified attention backend for self-attention and cross-attention."""
        logger.info(f"LingBot-World-Fast DiT set attention config to {attention_config.attn_impl}")
        for block in self.blocks:
            block.self_attn.attention_config = attention_config
            block.cross_attn.attention_config = attention_config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        control_type: str = "cam",
        config: dict[str, Any] | None = None,
    ) -> "LingBotWorldFastDiT":
        state_path = pretrained_model_name_or_path
        if not str(state_path).endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt")):
            base = pretrained_model_name_or_path.rstrip("/")
            candidate = f"{base}/diffusion_pytorch_model.safetensors"
            if os.path.exists(candidate):
                state_path = candidate
            else:
                index_candidate = f"{candidate}.index.json"
                state_path = index_candidate if os.path.exists(index_candidate) else candidate

        logger.info(f"Loading LingBot-World-Fast DiT from {state_path}")
        state_dict = load_state_dict(state_path)
        state_dict, inferred_config = cls.state_dict_converter().from_official(state_dict)
        if inferred_config["control_type"] != control_type:
            raise ValueError(
                f"Checkpoint control projection is {inferred_config['control_type']!r}, not requested {control_type!r}"
            )

        with init_weights_on_device("meta"):
            model = cls(**inferred_config)

        model.load_state_dict(state_dict, strict=True, assign=True)
        model = model.to(dtype=torch_dtype)
        runtime_config = config or {}
        model.set_causal_attention_window(
            local_attn_size=int(runtime_config.get("local_attn_size", -1)),
            sink_size=int(runtime_config.get("sink_size", 0)),
        )
        return model

    @staticmethod
    def state_dict_converter() -> "LingBotWorldFastDiTStateDictConverter":
        """Return the native LingBot checkpoint converter."""
        return LingBotWorldFastDiTStateDictConverter()

    def set_causal_attention_window(self, local_attn_size: int, sink_size: int) -> None:
        """Configure shape-independent streaming attention policy after loading weights."""
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        for block in self.blocks:
            block.self_attn.local_attn_size = local_attn_size
            block.self_attn.sink_size = sink_size

    def get_fsdp_module_names(self) -> list[str]:
        return ["blocks"]


class LingBotWorldFastDiTStateDictConverter:
    """Build a LingBot causal DiT directly from its native checkpoint tensors."""

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Return native weights and infer constructor arguments from their shapes."""
        patch_size = (1, 2, 2)
        patch_weight = state_dict["patch_embedding.weight"]
        control_patch_weight = state_dict["patch_embedding_wancamctrl.weight"]
        control_patch_volume = math.prod(patch_size) * 64
        inferred_control_dim = int(control_patch_weight.shape[1] // control_patch_volume)
        if inferred_control_dim not in (6, 7):
            raise ValueError(f"Unexpected control patch dim: {inferred_control_dim}")

        checkpoint_control_type = "cam" if inferred_control_dim == 6 else "act"

        inferred_config = {
            "patch_size": patch_size,
            "text_len": 512,
            "in_dim": int(patch_weight.shape[1]),
            "dim": int(patch_weight.shape[0]),
            "ffn_dim": int(state_dict["blocks.0.ffn.0.weight"].shape[0]),
            "freq_dim": 256,
            "text_dim": int(state_dict["text_embedding.0.weight"].shape[1]),
            "out_dim": int(state_dict["head.head.bias"].shape[0] // math.prod(patch_size)),
            "num_heads": 40,
            "num_layers": len({key.split(".")[1] for key in state_dict if key.startswith("blocks.")}),
            "local_attn_size": -1,
            "sink_size": 0,
            "qk_norm": True,
            "cross_attn_norm": True,
            "eps": 1e-6,
            "control_type": checkpoint_control_type,
        }
        return state_dict, inferred_config

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reject unsupported Diffusers-formatted LingBot checkpoints."""
        raise ValueError("LingBot-World-Fast supports native checkpoints only")

    def get_tp_plan(self) -> dict[str, Any]:
        return {}
