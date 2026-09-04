# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 packed multimodal DiT."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from telefuser.core.base_model import BaseModel
from telefuser.core.config import AttentionConfig, AttnImplType, QuantConfig, QuantKernelBackend, QuantType
from telefuser.distributed.collectives import all_gather_cat, all_reduce_sum_
from telefuser.distributed.device_mesh import (
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    get_ulysses_group,
    get_ulysses_world_size,
)
from telefuser.distributed.parallel_shard import sequence_parallel_shard, sequence_parallel_unshard
from telefuser.distributed.ulysses_comm import (
    ulysses_gather_heads_chunk_async,
    ulysses_gather_heads_destination_major,
    ulysses_scatter_heads,
    ulysses_scatter_qkv_qknorm_rope_chunk_async,
)
from telefuser.feature_cache import AdaTaylorCacheCalibrator, NoOpCache
from telefuser.ops import RMSNorm, apply_qk_norm_rope_neox, indexed_gate, indexed_scale_shift, silu_and_mul_reuse_input
from telefuser.ops.attention import SparseAttentionState, attention
from telefuser.ops.fp8_attention import quantize_fp8_per_block, quantize_fp8_qkv
from telefuser.ops.rotary import apply_rotary_emb_neox
from telefuser.utils.logging import logger

MINIMAX_H3_ADALN_MODALITY_NUM = 3
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


def _ulysses_head_chunk_ranges(local_heads: int, chunks: int = 2) -> tuple[tuple[int, int], ...]:
    """Return balanced, non-empty destination-local head ranges."""
    if local_heads <= 0 or chunks <= 0 or chunks > local_heads:
        raise ValueError("invalid Ulysses attention head chunks")
    base, remainder = divmod(local_heads, chunks)
    ranges = []
    start = 0
    for index in range(chunks):
        count = base + (1 if index < remainder else 0)
        ranges.append((start, count))
        start += count
    return tuple(ranges)


def _can_run_fused_ulysses_attention(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    attention_config: AttentionConfig | None,
    sequence_lengths: list[int],
    rope_cos_sin_cache: torch.Tensor | None,
    use_ulysses: bool,
    ulysses_world_size: int,
) -> bool:
    """Check whether the lossless fused MiniMax-H3 Ulysses path is supported."""
    return (
        use_ulysses
        and not torch.compiler.is_compiling()
        and qkv.is_cuda
        and qkv.dtype == torch.bfloat16
        and qkv.ndim == 4
        and qkv.shape[1] == 3
        and qkv.stride(-1) == 1
        and q_weight.is_cuda
        and k_weight.is_cuda
        and q_weight.dtype == k_weight.dtype == qkv.dtype
        and q_weight.is_contiguous()
        and k_weight.is_contiguous()
        and rope_cos_sin_cache is not None
        and rope_cos_sin_cache.is_cuda
        and rope_cos_sin_cache.ndim == 2
        and rope_cos_sin_cache.shape[0] == qkv.shape[0]
        and rope_cos_sin_cache.stride(-1) == 1
        and attention_config is not None
        and attention_config.attn_impl == AttnImplType.FLASH_ATTN_4
        and len(sequence_lengths) == 2
        and sum(sequence_lengths) == qkv.shape[0] * ulysses_world_size
        and sequence_lengths[0] > 0
        and 0 < sequence_lengths[1] < 64
        and sequence_lengths[0] > sequence_lengths[1]
        and qkv.shape[0] % 64 == 0
    )


@dataclass(frozen=True)
class MiniMaxH3DiTConfig:
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str | Path) -> MiniMaxH3DiTConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        values = {key: payload[key] for key in fields if key in payload}
        if "patch_size" in values:
            values["patch_size"] = tuple(int(value) for value in values["patch_size"])
        return cls(**values)

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("MiniMax H3 hidden_size and num_layers must be positive")
        if self.num_attention_heads <= 0 or self.attention_head_dim <= 0:
            raise ValueError("MiniMax H3 attention dimensions must be positive")
        if len(self.patch_size) != 3 or any(value <= 0 for value in self.patch_size):
            raise ValueError("MiniMax H3 patch_size must contain three positive integers")
        if 6 * self.rope_inv_freq_len > self.attention_head_dim:
            raise ValueError("MiniMax H3 rotary dimensions must fit inside attention_head_dim")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        return self.latents_dim * math.prod(self.patch_size)

    @property
    def adaln_out_features(self) -> int:
        return 18 * self.hidden_size

    @property
    def final_adaln_out_features(self) -> int:
        return 2 * self.hidden_size


MINIMAX_H3_ADALN_CACHE_FORMAT_VERSION = 1
MINIMAX_H3_ADALN_CACHE_GENERATOR_VERSION = "telefuser-minimax-h3-adaln-v1"


def _adaln_config_payload(config: MiniMaxH3DiTConfig) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(config), sort_keys=True))


@dataclass
class MiniMaxH3AdaLNCache:
    """Precomputed AdaLN projection outputs for a fixed H3 denoising schedule."""

    timesteps: torch.Tensor
    block_outputs: tuple[torch.Tensor, ...]
    final_output: torch.Tensor
    config: MiniMaxH3DiTConfig
    model_fingerprint: str
    partition: str | None = None
    _device_outputs: dict[str, tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )
    _cpu_positions: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.timesteps = self.timesteps.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.block_outputs = tuple(
            output.detach().to(device="cpu", dtype=torch.bfloat16).contiguous() for output in self.block_outputs
        )
        self.final_output = self.final_output.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if self.timesteps.ndim != 1 or self.timesteps.numel() == 0:
            raise ValueError("AdaLN cache timesteps must be a non-empty rank-1 tensor.")
        if len(self.block_outputs) != self.config.num_layers:
            raise ValueError(f"AdaLN cache has {len(self.block_outputs)} blocks, expected {self.config.num_layers}.")
        expected_rows = self.timesteps.numel()
        expected_block_width = self.config.adaln_out_features
        for index, output in enumerate(self.block_outputs):
            if output.shape != (expected_rows, expected_block_width):
                raise ValueError(
                    f"AdaLN cache block {index} has shape {tuple(output.shape)}, "
                    f"expected {(expected_rows, expected_block_width)}."
                )
        expected_final_shape = (expected_rows, self.config.final_adaln_out_features)
        if self.final_output.shape != expected_final_shape:
            raise ValueError(
                f"AdaLN cache final output has shape {tuple(self.final_output.shape)}, expected {expected_final_shape}."
            )
        self._cpu_positions = {float(value).hex(): index for index, value in enumerate(self.timesteps.tolist())}

    @classmethod
    def from_model(
        cls, model: Any, timesteps: Iterable[float] | torch.Tensor, *, partition: str | None = None
    ) -> MiniMaxH3AdaLNCache:
        if model.tp_flag:
            raise RuntimeError("Build the AdaLN cache before tensor-parallel sharding.")
        if model.time_embedder is None:
            raise RuntimeError("The model is already in inference-only AdaLN mode.")
        unique_timesteps = torch.unique(torch.as_tensor(timesteps, dtype=torch.float32), sorted=True)
        if unique_timesteps.numel() == 0:
            raise ValueError("At least one timestep is required to build an AdaLN cache.")
        model_device = next(model.parameters()).device
        with torch.inference_mode():
            adaln_input = torch.nn.functional.silu(model.time_embedder(unique_timesteps.to(model_device))).to(
                torch.bfloat16
            )
            block_outputs = tuple(
                block.adaln_proj.project_local(adaln_input).detach().cpu().contiguous() for block in model.blocks
            )
            final_output = model.final_layer.adaln_proj.project_local(adaln_input).detach().cpu().contiguous()
        return cls(
            timesteps=unique_timesteps,
            block_outputs=block_outputs,
            final_output=final_output,
            config=model.config,
            model_fingerprint=model.adaln_fingerprint(),
            partition=partition,
        )

    def _cpu_indices_for(self, timesteps: torch.Tensor) -> torch.Tensor:
        indices: list[int] = []
        for timestep in timesteps.detach().to(device="cpu", dtype=torch.float32).tolist():
            index = self._cpu_positions.get(float(timestep).hex())
            if index is None:
                raise ValueError(
                    f"AdaLN cache is missing timestep {timestep:.8g}; rebuild it for the requested denoising schedule."
                )
            indices.append(index)
        return torch.tensor(indices, dtype=torch.long)

    def _outputs_for_device(self, device: torch.device) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        if device.type == "cpu":
            return self.timesteps, self.block_outputs, self.final_output
        key = str(device)
        outputs = self._device_outputs.get(key)
        if outputs is None:
            outputs = (
                self.timesteps.to(device=device, non_blocking=True),
                tuple(output.to(device=device, non_blocking=True) for output in self.block_outputs),
                self.final_output.to(device=device, non_blocking=True),
            )
            self._device_outputs[key] = outputs
        return outputs

    def _indices_for(
        self,
        timesteps: torch.Tensor,
        device: torch.device,
        cached_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if device.type == "cpu":
            return self._cpu_indices_for(timesteps)

        requested = timesteps.detach().to(device=device, dtype=torch.float32)
        indices = torch.searchsorted(cached_timesteps, requested)
        safe_indices = indices.clamp_max(cached_timesteps.numel() - 1)
        matches = (indices < cached_timesteps.numel()) & (cached_timesteps.index_select(0, safe_indices) == requested)
        torch._assert_async(matches.all(), "AdaLN cache is missing a requested timestep.")
        return safe_indices

    def resolve(
        self, model: Any, timesteps: torch.Tensor, device: torch.device
    ) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, torch.Tensor]]:
        cached_timesteps, block_outputs, final_output = self._outputs_for_device(device)
        indices = self._indices_for(timesteps, device, cached_timesteps)
        block_params = tuple(
            block.adaln_proj.split_output(output.index_select(0, indices))
            for block, output in zip(model.blocks, block_outputs, strict=True)
        )
        final_params = model.final_layer.adaln_proj.split_output(final_output.index_select(0, indices))
        return block_params, final_params

    def clear_device_cache(self) -> None:
        self._device_outputs.clear()

    def validate_model(self, model: Any) -> None:
        if _adaln_config_payload(self.config) != _adaln_config_payload(model.config):
            raise ValueError("AdaLN cache configuration does not match the loaded MiniMax H3 model.")
        if self.model_fingerprint != model.adaln_fingerprint():
            raise ValueError("AdaLN cache fingerprint does not match the loaded MiniMax H3 AdaLN weights.")

    def save(self, directory: str | Path) -> None:
        from safetensors.torch import save_file

        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {"timesteps": self.timesteps, "final_output": self.final_output}
        tensors.update({f"block_{index}": output for index, output in enumerate(self.block_outputs)})
        save_file(
            tensors,
            str(root / "adaln.safetensors"),
            metadata={"format_version": str(MINIMAX_H3_ADALN_CACHE_FORMAT_VERSION)},
        )
        manifest = {
            "format_version": MINIMAX_H3_ADALN_CACHE_FORMAT_VERSION,
            "config": _adaln_config_payload(self.config),
            "model_fingerprint": self.model_fingerprint,
            "partition": self.partition,
            "dtype": str(self.final_output.dtype),
            "generator_version": MINIMAX_H3_ADALN_CACHE_GENERATOR_VERSION,
            "tensor_file": "adaln.safetensors",
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, model: Any, *, expected_partition: str | None = None) -> MiniMaxH3AdaLNCache:
        from safetensors.torch import load_file

        root = Path(directory)
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MiniMax H3 AdaLN cache manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != MINIMAX_H3_ADALN_CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported MiniMax H3 AdaLN cache format version.")
        if manifest.get("config") != _adaln_config_payload(model.config):
            raise ValueError("AdaLN cache configuration does not match the loaded MiniMax H3 model.")
        if manifest.get("generator_version") != MINIMAX_H3_ADALN_CACHE_GENERATOR_VERSION:
            raise ValueError("Unsupported MiniMax H3 AdaLN cache generator version.")
        if manifest.get("dtype") != str(torch.bfloat16):
            raise ValueError("AdaLN cache dtype is incompatible with MiniMax H3 BF16 inference.")
        if expected_partition is not None and manifest.get("partition") != expected_partition:
            raise ValueError("AdaLN cache partition does not match the loaded MiniMax H3 checkpoint.")
        if manifest.get("model_fingerprint") != model.adaln_fingerprint():
            raise ValueError("AdaLN cache fingerprint does not match the loaded MiniMax H3 AdaLN weights.")
        tensor_file = root / manifest.get("tensor_file", "adaln.safetensors")
        tensors = load_file(str(tensor_file), device="cpu")
        try:
            block_outputs = tuple(tensors[f"block_{index}"] for index in range(model.config.num_layers))
            return cls(
                timesteps=tensors["timesteps"],
                block_outputs=block_outputs,
                final_output=tensors["final_output"],
                config=model.config,
                model_fingerprint=manifest["model_fingerprint"],
                partition=manifest.get("partition"),
            )
        except KeyError as error:
            raise ValueError(f"MiniMax H3 AdaLN cache is missing tensor {error.args[0]}.") from error


def _rms_norm(size: int, eps: float) -> RMSNorm:
    return RMSNorm(size, eps=eps, dtype=torch.bfloat16)


def _reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    per_group = (heads_per_group + 2) * head_dim
    if weight.shape[0] != num_query_groups * per_group:
        raise ValueError("MiniMax H3 grouped QKV weight has an incompatible output dimension")
    rest = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest)
    q, k, v = torch.split(grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1)
    return torch.cat(
        (
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest),
            k.reshape(num_query_groups * head_dim, *rest),
            v.reshape(num_query_groups * head_dim, *rest),
        ),
        dim=0,
    )


def _replace_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    parameter = module.get_parameter(name)
    setattr(module, name, nn.Parameter(value.contiguous(), requires_grad=parameter.requires_grad))


def _shard_linear_output(
    linear: nn.Linear,
    *,
    rank: int,
    world_size: int,
    sections: tuple[int, ...] | None = None,
) -> None:
    sections = sections or (linear.out_features,)
    if sum(sections) != linear.out_features or any(size % world_size for size in sections):
        raise ValueError(
            f"linear output sections {sections} must sum to {linear.out_features} and divide TP degree {world_size}"
        )
    weight_sections = linear.weight.split(sections, dim=0)
    local_weight = torch.cat(tuple(section.chunk(world_size, dim=0)[rank] for section in weight_sections), dim=0)
    _replace_parameter(linear, "weight", local_weight)
    if linear.bias is not None:
        bias_sections = linear.bias.split(sections, dim=0)
        local_bias = torch.cat(tuple(section.chunk(world_size, dim=0)[rank] for section in bias_sections), dim=0)
        _replace_parameter(linear, "bias", local_bias)
    linear.out_features //= world_size


def _shard_linear_input(linear: nn.Linear, *, rank: int, world_size: int) -> None:
    if linear.in_features % world_size:
        raise ValueError(f"linear input size {linear.in_features} must divide TP degree {world_size}")
    _replace_parameter(linear, "weight", linear.weight.chunk(world_size, dim=1)[rank])
    linear.in_features //= world_size


class MiniMaxH3Rope(nn.Module):
    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        inv_freq = 10000.0 ** (-torch.arange(inv_freq_len, dtype=torch.float32) / inv_freq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        if position_ids.ndim != 3 or position_ids.shape[0] != 1 or position_ids.shape[-1] != 3:
            raise ValueError("MiniMax H3 position_ids must have shape [1, sequence, 3]")
        per_axis = position_ids[0].float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        half = torch.cat(tuple(per_axis.unbind(dim=1)), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = config.timestep_input_dim
        self.proj_in = nn.Linear(
            config.timestep_input_dim,
            config.time_embed_hidden_size,
            dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            config.time_embed_hidden_size,
            config.time_embed_dim,
            dtype=torch.float32,
        )
        self._frequency_cache: dict[torch.device, torch.Tensor] = {}

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        frequencies = self._frequency_cache.get(timestep.device)
        if frequencies is None:
            frequencies = torch.exp(
                -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timestep.device) / half
            )
            self._frequency_cache[timestep.device] = frequencies
        args = timestep.float().reshape(-1, 1) * frequencies.reshape(1, -1)
        embedding = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(embedding)))


class MiniMaxH3Attention(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.attention_head_dim
        self.inner_dim = config.inner_dim
        self.qkv_proj = nn.Linear(config.hidden_size, 3 * self.inner_dim, bias=False, dtype=torch.bfloat16)
        self.q_norm = _rms_norm(self.head_dim, config.qk_norm_eps)
        self.k_norm = _rms_norm(self.head_dim, config.qk_norm_eps)
        self.out_proj = nn.Linear(self.inner_dim, config.hidden_size, bias=False, dtype=torch.bfloat16)
        self.ulysses_group: dist.ProcessGroup | None = None
        self.ulysses_communicator: Any | None = None
        self.tp_group: dist.ProcessGroup | None = None

    def set_ulysses_group(self, group: dist.ProcessGroup | None, communicator: Any | None = None) -> None:
        self.ulysses_group = group
        self.ulysses_communicator = communicator

    def enable_tp(self, group: dist.ProcessGroup, *, rank: int, world_size: int) -> None:
        if self.num_heads % world_size:
            raise ValueError(f"attention heads ({self.num_heads}) must divide TP degree ({world_size})")
        original_inner_dim = self.inner_dim
        _shard_linear_output(
            self.qkv_proj,
            rank=rank,
            world_size=world_size,
            sections=(original_inner_dim, original_inner_dim, original_inner_dim),
        )
        _shard_linear_input(self.out_proj, rank=rank, world_size=world_size)
        self.num_heads //= world_size
        self.inner_dim //= world_size
        self.tp_group = group

    @staticmethod
    def _live_tokens(sequence_lengths: list[int], total_tokens: int) -> int:
        if len(sequence_lengths) == 1 and sequence_lengths[0] == total_tokens:
            return total_tokens
        if (
            len(sequence_lengths) == 2
            and sum(sequence_lengths) == total_tokens
            and sequence_lengths[0] > sequence_lengths[1]
            and 0 < sequence_lengths[1] < 64
            and total_tokens % 64 == 0
        ):
            return sequence_lengths[0]
        raise ValueError("MiniMax H3 optimized attention requires one live sequence with optional trailing padding")

    @staticmethod
    def _is_sol_active(sparse_state: SparseAttentionState | None) -> bool:
        return sparse_state is not None and not sparse_state.should_use_dense()

    @classmethod
    def _prepare_sol_qkv(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sparse_state: SparseAttentionState,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        config = sparse_state.config
        layer_end = config.sol_fp8_layer_end
        fp8_layer_active = (
            cls._is_sol_active(sparse_state)
            and config.sol_fp8
            and sparse_state.layer_idx >= config.sol_fp8_layer_start
            and (layer_end is None or sparse_state.layer_idx < layer_end)
        )
        if not fp8_layer_active:
            return query, key, value, None
        if query.is_cuda and torch.cuda.get_device_capability(query.device) == (9, 0):
            query, key, value, q_scale, k_scale, v_scale = quantize_fp8_qkv(query, key, value)
        else:
            query, q_scale = quantize_fp8_per_block(query)
            key, k_scale = quantize_fp8_per_block(key)
            value, v_scale = quantize_fp8_per_block(value)
        return query, key, value, (q_scale, k_scale, v_scale)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_lengths: list[int],
        rope_cos_sin_cache: torch.Tensor | None,
        attention_config: AttentionConfig | None,
        cu_seqlens: torch.Tensor | None = None,
        sparse_state: SparseAttentionState | None = None,
        prefix_tokens: int = 0,
    ) -> torch.Tensor:
        sequence, _ = hidden.shape
        qkv = self.qkv_proj(hidden).reshape(sequence, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=1)
        group = self.ulysses_group
        use_ulysses = group is not None and dist.get_world_size(group) > 1
        world_size = dist.get_world_size(group) if use_ulysses else 1
        local_heads = self.num_heads // world_size
        use_fused_ulysses = _can_run_fused_ulysses_attention(
            qkv,
            self.q_norm.weight,
            self.k_norm.weight,
            attention_config,
            sequence_lengths,
            rope_cos_sin_cache,
            use_ulysses,
            world_size,
        )
        use_fused_ulysses = (
            use_fused_ulysses and attention_config is not None and attention_config.attention_chunks <= local_heads
        )
        if use_fused_ulysses:
            chunks = attention_config.attention_chunks
            head_ranges = _ulysses_head_chunk_ranges(local_heads, chunks=chunks)
            valid_only = attention_config.ulysses_sequence_mode == "valid_only"
            destination = torch.empty(
                1,
                sequence,
                world_size,
                local_heads,
                self.head_dim,
                dtype=value.dtype,
                device=value.device,
            )
            scatter_waiters = [
                ulysses_scatter_qkv_qknorm_rope_chunk_async(
                    qkv,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    rope_cos_sin_cache,
                    self.q_norm.eps,
                    group,
                    local_head_start=start,
                    local_head_count=count,
                )
                for start, count in head_ranges
            ]
            gather_waiters = []
            for (start, _), scatter_wait in zip(head_ranges, scatter_waiters, strict=True):
                query_chunk, key_chunk, value_chunk = scatter_wait()
                output_chunk = attention(
                    query_chunk,
                    key_chunk,
                    value_chunk,
                    attention_config=attention_config,
                    scale=self.head_dim**-0.5,
                    sequence_lengths=sequence_lengths,
                    cu_seqlens=cu_seqlens,
                    fixed_valid=True,
                    pad_fixed_valid_output=not valid_only,
                )
                gather_waiters.append(
                    ulysses_gather_heads_chunk_async(
                        output_chunk,
                        group,
                        num_heads=self.num_heads,
                        local_head_start=start,
                        destination=destination,
                        zero_tail=valid_only,
                    )
                )
            for gather_wait in gather_waiters:
                gather_wait()
            output = destination.flatten(2, 3)
        else:
            if use_ulysses:
                value_wait = ulysses_scatter_heads(
                    value.unsqueeze(0), group, tag="v", barrier=False, communicator=self.ulysses_communicator
                )
            if rope_cos_sin_cache is not None:
                query, key = apply_qk_norm_rope_neox(
                    query,
                    key,
                    self.q_norm.weight,
                    self.k_norm.weight,
                    rope_cos_sin_cache,
                    eps=self.q_norm.eps,
                )
            else:
                query = self.q_norm(query)
                key = self.k_norm(key)
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            if use_ulysses:
                query_wait = ulysses_scatter_heads(
                    query, group, tag="q", barrier=False, communicator=self.ulysses_communicator
                )
                key_wait = ulysses_scatter_heads(key, group, tag="k", communicator=self.ulysses_communicator)
                query = query_wait()
                key = key_wait()
                value = value_wait()
            optimized_impls = {AttnImplType.SAGE_ATTN_2_8_8_SM90, AttnImplType.SOL_ATTN}
            if attention_config is not None and attention_config.attn_impl in optimized_impls:
                total_tokens = query.shape[1]
                live_tokens = self._live_tokens(sequence_lengths, total_tokens)
                live_query = query[:, :live_tokens].contiguous()
                live_key = key[:, :live_tokens].contiguous()
                live_value = value[:, :live_tokens].contiguous()
                scales = None
                runtime_attention_config = attention_config
                runtime_sparse_state = sparse_state
                if attention_config.attn_impl == AttnImplType.SOL_ATTN:
                    if sparse_state is None:
                        raise RuntimeError("MiniMax H3 Sol-Attn requires sparse runtime state")
                    if not 0 <= prefix_tokens <= live_tokens:
                        raise ValueError("MiniMax H3 Sol-Attn prefix must be within the live packed sequence")
                    sol_query, sol_key, sol_value, scales = self._prepare_sol_qkv(
                        live_query, live_key, live_value, sparse_state
                    )
                    if sparse_state.should_use_dense():
                        runtime_attention_config = AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_4)
                        runtime_sparse_state = None
                else:
                    sol_query, sol_key, sol_value = live_query, live_key, live_value
                live_output = attention(
                    sol_query,
                    sol_key,
                    sol_value,
                    attention_config=runtime_attention_config,
                    sparse_state=runtime_sparse_state,
                    scale=self.head_dim**-0.5,
                    q_scale=None if scales is None else scales[0],
                    k_scale=None if scales is None else scales[1],
                    v_scale=None if scales is None else scales[2],
                    sink_start=0,
                    sink_tokens=prefix_tokens,
                )
                if self._is_sol_active(sparse_state) and prefix_tokens:
                    dense_prefix = F.scaled_dot_product_attention(
                        live_query[:, :prefix_tokens].transpose(1, 2),
                        live_key.transpose(1, 2),
                        live_value.transpose(1, 2),
                        scale=self.head_dim**-0.5,
                    ).transpose(1, 2)
                    live_output = torch.cat((dense_prefix, live_output[:, prefix_tokens:]), dim=1)
                if live_tokens == total_tokens:
                    output = live_output
                else:
                    output = torch.zeros_like(query)
                    output[:, :live_tokens].copy_(live_output)
            else:
                output = attention(
                    query,
                    key,
                    value,
                    attention_config=attention_config,
                    scale=self.head_dim**-0.5,
                    sequence_lengths=sequence_lengths,
                    cu_seqlens=cu_seqlens,
                )
            if use_ulysses:
                output = ulysses_gather_heads_destination_major(output, group, num_heads=self.num_heads)()
        output = self.out_proj(output[0].reshape(sequence, self.inner_dim))
        if self.tp_group is not None:
            all_reduce_sum_((output,), group=self.tp_group)
        return output


class MiniMaxH3MLP(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 2 * config.ffn_hidden_size, bias=False, dtype=torch.bfloat16)
        self.fc2 = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16)
        self.intermediate_size = config.ffn_hidden_size
        self.tp_group: dist.ProcessGroup | None = None

    def enable_tp(self, group: dist.ProcessGroup, *, rank: int, world_size: int) -> None:
        _shard_linear_output(
            self.fc1,
            rank=rank,
            world_size=world_size,
            sections=(self.intermediate_size, self.intermediate_size),
        )
        _shard_linear_input(self.fc2, rank=rank, world_size=world_size)
        self.intermediate_size //= world_size
        self.tp_group = group

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        output = self.fc2(silu_and_mul_reuse_input(self.fc1(hidden)))
        if self.tp_group is not None:
            all_reduce_sum_((output,), group=self.tp_group)
        return output


class MiniMaxH3AdaLNProjection(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig, *, expand_ratio: int, modality_count: int) -> None:
        super().__init__()
        self.expand_ratio = expand_ratio
        self.modality_count = modality_count
        self.hidden_size = config.hidden_size
        self.linear: nn.Linear | None = nn.Linear(
            config.time_embed_dim,
            expand_ratio * modality_count * config.hidden_size,
            dtype=torch.bfloat16,
        )
        self.tp_group: dist.ProcessGroup | None = None
        self.tp_world_size = 1

    def enable_tp(self, group: dist.ProcessGroup, *, rank: int, world_size: int) -> None:
        if self.linear is None:
            raise RuntimeError("MiniMax H3 AdaLN projection weights were released for inference-only execution.")
        _shard_linear_output(self.linear, rank=rank, world_size=world_size)
        self.tp_group = group
        self.tp_world_size = world_size

    def project_local(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            raise RuntimeError("MiniMax H3 AdaLN projection weights were released for inference-only execution.")
        return self.linear(embedding)

    def release_weights(self) -> None:
        self.linear = None

    def split_output(self, output: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = output.reshape(-1, self.expand_ratio * self.hidden_size)
        return tuple(output.chunk(self.expand_ratio, dim=-1))

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.project_local(embedding)
        if self.tp_group is not None:
            output = all_gather_cat(
                output,
                dim=-1,
                group=self.tp_group,
                world_size=self.tp_world_size,
            )
        return self.split_output(output)


def _modulate(
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return indexed_scale_shift(hidden, shift, scale, indices)


def _norm_modulate(
    norm: RMSNorm,
    hidden: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Run the bit-compatible RMSNorm + indexed AdaLN fusion when supported."""
    weight = norm.weight
    use_fused = (
        not torch.compiler.is_compiling()
        and hidden.device.type == "cuda"
        and hidden.dtype == torch.bfloat16
        and hidden.ndim == 2
        and hidden.is_contiguous()
        and weight is not None
        and weight.dtype == shift.dtype == scale.dtype == hidden.dtype
        and weight.is_contiguous()
        and shift.ndim == scale.ndim == 2
        and shift.stride(-1) == scale.stride(-1) == 1
        and indices.ndim == 1
        and indices.device == hidden.device
    )
    if use_fused:
        from telefuser.kernel.triton.indexed_rmsnorm_modulation import (
            indexed_rmsnorm_scale_shift_bf16,
        )

        return indexed_rmsnorm_scale_shift_bf16(
            hidden,
            weight,
            shift,
            scale,
            indices.contiguous(),
            norm.eps,
        )
    return _modulate(norm(hidden), shift, scale, indices)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.norm1 = _rms_norm(config.hidden_size, config.norm_eps)
        self.norm2 = _rms_norm(config.hidden_size, config.norm_eps)
        self.attn = MiniMaxH3Attention(config)
        self.mlp = MiniMaxH3MLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        sequence_lengths: list[int],
        attention_config: AttentionConfig | None,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(
            self.norm1(hidden),
            sequence_lengths=sequence_lengths,
            rope_cos_sin_cache=None,
            attention_config=attention_config,
        )
        return hidden + self.mlp(self.norm2(hidden))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MiniMaxH3TokenRefinerBlock(config) for _ in range(config.token_refiner_num_layers)]
        )
        self.final_norm = _rms_norm(config.hidden_size, config.final_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_config: AttentionConfig | None,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(
                hidden,
                sequence_lengths=[hidden.shape[0]],
                attention_config=attention_config,
            )
        return self.final_norm(hidden)


class MiniMaxH3DiTBlock(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.norm1 = _rms_norm(config.hidden_size, config.norm_eps)
        self.norm2 = _rms_norm(config.hidden_size, config.norm_eps)
        self.attn = MiniMaxH3Attention(config)
        self.mlp = MiniMaxH3MLP(config)
        self.adaln_proj = MiniMaxH3AdaLNProjection(
            config,
            expand_ratio=6,
            modality_count=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor | None,
        combined_indices: torch.Tensor,
        sequence_lengths: list[int],
        rope_cos_sin_cache: torch.Tensor,
        attention_config: AttentionConfig | None,
        cu_seqlens: torch.Tensor | None = None,
        sparse_state: SparseAttentionState | None = None,
        prefix_tokens: int = 0,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params
        residual = hidden
        value = _norm_modulate(self.norm1, hidden, shift_msa, scale_msa, combined_indices)
        value = self.attn(
            value,
            sequence_lengths=sequence_lengths,
            rope_cos_sin_cache=rope_cos_sin_cache,
            attention_config=attention_config,
            cu_seqlens=cu_seqlens,
            sparse_state=sparse_state,
            prefix_tokens=prefix_tokens,
        )
        hidden = indexed_gate(residual, gate_msa, value, combined_indices)
        residual = hidden
        value = _norm_modulate(self.norm2, hidden, shift_mlp, scale_mlp, combined_indices)
        value = self.mlp(value)
        return indexed_gate(residual, gate_mlp, value, combined_indices)


class MiniMaxH3FinalLayer(nn.Module):
    def __init__(self, config: MiniMaxH3DiTConfig) -> None:
        super().__init__()
        self.norm = _rms_norm(config.hidden_size, config.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdaLNProjection(config, expand_ratio=2, modality_count=1)
        self.video_out = nn.Linear(config.hidden_size, config.video_patch_dim, dtype=torch.float32)
        self.audio_out = nn.Linear(config.hidden_size, config.audio_latents_dim, dtype=torch.float32)

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        adaln_input: torch.Tensor | None,
        inverse_indices: torch.Tensor,
        adaln_params: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if adaln_params is None:
            if adaln_input is None:
                raise ValueError("adaln_input is required when no cached AdaLN parameters are supplied.")
            adaln_params = self.adaln_proj(adaln_input)
        shift, scale = adaln_params
        hidden = _norm_modulate(self.norm, hidden, shift, scale, inverse_indices).float()
        return self.video_out(hidden), self.audio_out(hidden)


class MiniMaxH3DiT(BaseModel):
    """Faithful packed DiT baseline for H3-Base with optional Ulysses SP."""

    def __init__(self, config: MiniMaxH3DiTConfig | None = None) -> None:
        super().__init__()
        self.config = config or MiniMaxH3DiTConfig()
        config = self.config
        self.video_patch_proj = nn.Linear(config.video_patch_dim, config.hidden_size, dtype=torch.float32)
        self.audio_patch_proj = nn.Linear(config.audio_latents_dim, config.hidden_size, dtype=torch.float32)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, dtype=torch.bfloat16)
        self.time_embedder = MiniMaxH3TimeEmbedder(config)
        self.rope = MiniMaxH3Rope(config.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(config)
        self.blocks = nn.ModuleList([MiniMaxH3DiTBlock(config) for _ in range(config.num_layers)])
        self.final_layer = MiniMaxH3FinalLayer(config)
        self.layer_name_list = ["blocks"]
        self.device_mesh: Any | None = None
        self.usp_flag = False
        self.tp_flag = False
        self._static_cache_key: Any | None = None
        self._static_prompt: torch.Tensor | None = None
        self._static_rope_cos_sin: torch.Tensor | None = None
        self._static_sequence_lengths: list[int] | None = None
        self._static_cu_seqlens: torch.Tensor | None = None
        self._adaln_cache: MiniMaxH3AdaLNCache | None = None
        self._online_adaln_cache_enabled = False
        self._online_adaln_partition: str | None = None
        self._online_adaln_rows: dict[str, tuple[float, tuple[torch.Tensor, ...], torch.Tensor]] = {}
        self._online_adaln_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._online_adaln_copy_device: torch.device | None = None
        self.sparse_attention_state: SparseAttentionState | None = None

    def set_attention_config(self, attention_config: AttentionConfig) -> None:
        super().set_attention_config(attention_config)
        if attention_config.attn_impl == AttnImplType.SOL_ATTN:
            if attention_config.sparse_config is None:
                raise ValueError("MiniMax H3 Sol-Attn requires sparse attention configuration")
            self.sparse_attention_state = SparseAttentionState(
                config=attention_config.sparse_config,
                mask_map=None,
                model_type="minimax_h3",
            )
        else:
            self.sparse_attention_state = None

    def _token_refiner_attention_config(self) -> AttentionConfig:
        if self.attention_config.is_sparse():
            return AttentionConfig.dense_attention(AttnImplType.FLASH_ATTN_4)
        return self.attention_config

    def adaln_fingerprint(self) -> str:
        if self.time_embedder is None:
            raise RuntimeError("AdaLN weights were released; their fingerprint is unavailable.")
        tensors: list[tuple[str, torch.Tensor]] = list(self.time_embedder.named_parameters(prefix="time_embedder"))
        for index, block in enumerate(self.blocks):
            if block.adaln_proj.linear is None:
                raise RuntimeError("AdaLN weights were released; their fingerprint is unavailable.")
            tensors.extend(block.adaln_proj.linear.named_parameters(prefix=f"blocks.{index}.adaln_proj.linear"))
        if self.final_layer.adaln_proj.linear is None:
            raise RuntimeError("AdaLN weights were released; their fingerprint is unavailable.")
        tensors.extend(self.final_layer.adaln_proj.linear.named_parameters(prefix="final_layer.adaln_proj.linear"))
        digest = hashlib.sha256()
        for name, tensor in tensors:
            value = tensor.detach().to(device="cpu").contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def enable_online_adaln_cache(self, *, partition: str | None = None) -> None:
        if self._adaln_cache is not None:
            raise RuntimeError("MiniMax H3 inference-only AdaLN cache is already enabled.")
        if self._online_adaln_cache_enabled:
            raise RuntimeError("MiniMax H3 online AdaLN cache is already enabled.")
        self._online_adaln_cache_enabled = True
        self._online_adaln_partition = partition
        self._online_adaln_rows.clear()
        self._online_adaln_batches.clear()
        self._online_adaln_copy_device = None

    def _record_online_adaln(
        self, timesteps: torch.Tensor, adaln_input: torch.Tensor
    ) -> tuple[tuple[tuple[torch.Tensor, ...], ...], tuple[torch.Tensor, torch.Tensor]]:
        raw_block_outputs = tuple(block.adaln_proj.project_local(adaln_input) for block in self.blocks)
        raw_final_output = self.final_layer.adaln_proj.project_local(adaln_input)
        if self.tp_flag:
            first_projection = self.blocks[0].adaln_proj
            gathered_block_outputs = all_gather_cat(
                torch.stack(raw_block_outputs),
                dim=-1,
                group=first_projection.tp_group,
                world_size=first_projection.tp_world_size,
            )
            raw_block_outputs = tuple(gathered_block_outputs.unbind(dim=0))
            final_projection = self.final_layer.adaln_proj
            raw_final_output = all_gather_cat(
                raw_final_output,
                dim=-1,
                group=final_projection.tp_group,
                world_size=final_projection.tp_world_size,
            )
        stacked_block_outputs = torch.stack(raw_block_outputs).detach()
        if stacked_block_outputs.device.type == "cuda":
            self._online_adaln_copy_device = stacked_block_outputs.device

            def copy_to_pinned_cpu(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
                output = torch.empty(value.shape, device="cpu", dtype=dtype, pin_memory=True)
                output.copy_(value, non_blocking=True)
                return output

            timesteps_cpu = copy_to_pinned_cpu(timesteps.detach(), torch.float32)
            block_outputs_cpu = copy_to_pinned_cpu(stacked_block_outputs, torch.bfloat16)
            final_output_cpu = copy_to_pinned_cpu(raw_final_output.detach(), torch.bfloat16)
        else:
            timesteps_cpu = timesteps.detach().to(device="cpu", dtype=torch.float32).contiguous()
            block_outputs_cpu = stacked_block_outputs.to(device="cpu", dtype=torch.bfloat16).contiguous()
            final_output_cpu = raw_final_output.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        self._online_adaln_batches.append((timesteps_cpu, block_outputs_cpu, final_output_cpu))
        block_params = tuple(
            block.adaln_proj.split_output(output) for block, output in zip(self.blocks, raw_block_outputs, strict=True)
        )
        final_params = self.final_layer.adaln_proj.split_output(raw_final_output)
        return block_params, final_params

    def finalize_online_adaln_cache(self) -> bool:
        if not self._online_adaln_cache_enabled:
            return self._adaln_cache is not None
        if not self._online_adaln_batches:
            raise RuntimeError("Cannot finalize an online AdaLN cache before any request timestep was observed.")
        if self._online_adaln_copy_device is not None:
            torch.cuda.synchronize(self._online_adaln_copy_device)
        for timesteps, block_outputs, final_output in self._online_adaln_batches:
            for row, timestep in enumerate(timesteps.tolist()):
                key = float(timestep).hex()
                self._online_adaln_rows[key] = (
                    float(timestep),
                    tuple(block_outputs[:, row].unbind(dim=0)),
                    final_output[row],
                )
        rows = [
            self._online_adaln_rows[key]
            for key in sorted(self._online_adaln_rows, key=lambda item: self._online_adaln_rows[item][0])
        ]
        cache = MiniMaxH3AdaLNCache(
            timesteps=torch.tensor([row[0] for row in rows], dtype=torch.float32),
            block_outputs=tuple(
                torch.stack([row[1][block_index] for row in rows]) for block_index in range(self.config.num_layers)
            ),
            final_output=torch.stack([row[2] for row in rows]),
            config=self.config,
            model_fingerprint=self.adaln_fingerprint(),
            partition=self._online_adaln_partition,
        )
        self._activate_inference_only_adaln(cache)
        self._online_adaln_cache_enabled = False
        self._online_adaln_partition = None
        self._online_adaln_rows.clear()
        self._online_adaln_batches.clear()
        self._online_adaln_copy_device = None
        return True

    def prepare_adaln_cache(
        self, timesteps: Iterable[float] | torch.Tensor, *, partition: str | None = None
    ) -> MiniMaxH3AdaLNCache:
        return MiniMaxH3AdaLNCache.from_model(self, timesteps, partition=partition)

    def enable_inference_only_adaln(self, cache: MiniMaxH3AdaLNCache) -> None:
        if self._adaln_cache is not None:
            raise RuntimeError("MiniMax H3 inference-only AdaLN mode is already enabled.")
        cache.validate_model(self)
        self._activate_inference_only_adaln(cache)

    def _activate_inference_only_adaln(self, cache: MiniMaxH3AdaLNCache) -> None:
        self._adaln_cache = cache
        for block in self.blocks:
            block.adaln_proj.release_weights()
        self.final_layer.adaln_proj.release_weights()
        self.time_embedder = None

    def load_inference_only_adaln(self, directory: str | Path, *, expected_partition: str | None = None) -> None:
        if self._adaln_cache is not None:
            raise RuntimeError("MiniMax H3 inference-only AdaLN mode is already enabled.")
        cache = MiniMaxH3AdaLNCache.load(directory, self, expected_partition=expected_partition)
        self._activate_inference_only_adaln(cache)

    def _preserve_fp32_boundaries(self) -> None:
        for name in MINIMAX_H3_FP32_PARAM_NAMES:
            try:
                parameter = self.get_parameter(name)
            except AttributeError:
                continue
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        if self.rope.inv_freq.dtype != torch.float32:
            self.rope.inv_freq.data = self.rope.inv_freq.data.float()

    def to(self, *args: Any, **kwargs: Any) -> MiniMaxH3DiT:
        preserved_parameters: dict[str, torch.Tensor] = {}
        for name in MINIMAX_H3_FP32_PARAM_NAMES:
            try:
                parameter = self.get_parameter(name)
            except AttributeError:
                continue
            if not parameter.is_meta:
                preserved_parameters[name] = parameter.detach().clone()
        preserved_buffers = {
            name: buffer.detach().clone()
            for name in MINIMAX_H3_FP32_BUFFER_NAMES
            if not (buffer := self.get_buffer(name)).is_meta
        }
        result = super().to(*args, **kwargs)
        for name, value in preserved_parameters.items():
            parameter = result.get_parameter(name)
            parameter.data = value.to(device=parameter.device, dtype=torch.float32)
        for name, value in preserved_buffers.items():
            buffer = result.get_buffer(name)
            buffer.data = value.to(device=buffer.device, dtype=torch.float32)
        result._preserve_fp32_boundaries()
        return result

    @staticmethod
    def _position_ids(value: Any, name: str) -> torch.Tensor:
        position_ids = value.get("position_ids") if isinstance(value, dict) else getattr(value, "position_ids", None)
        if position_ids is None:
            raise ValueError(f"{name}.position_ids is required")
        return position_ids.reshape(-1).long()

    @staticmethod
    def _cu_seqlens(packed: Any) -> torch.Tensor:
        cu = packed.get("cu_seqlens_q") if isinstance(packed, dict) else packed.cu_seqlens_q
        if cu is None:
            raise ValueError("packed_seq_params.cu_seqlens_q is required")
        return cu.reshape(-1)

    @classmethod
    def _sequence_lengths(cls, packed: Any) -> list[int]:
        values = [int(value) for value in cls._cu_seqlens(packed).tolist()]
        return [stop - start for start, stop in zip(values[:-1], values[1:], strict=True) if stop > start]

    def _static_inputs(
        self,
        kwargs: dict[str, Any],
        *,
        device: torch.device,
        text_positions: torch.Tensor,
        rope_row_start: int = 0,
        rope_row_stop: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
        cache_key = kwargs.get("static_cache_key")
        if (
            cache_key is not None
            and cache_key == self._static_cache_key
            and self._static_prompt is not None
            and self._static_rope_cos_sin is not None
            and self._static_sequence_lengths is not None
            and self._static_cu_seqlens is not None
        ):
            return (
                self._static_prompt,
                self._static_rope_cos_sin,
                self._static_sequence_lengths,
                self._static_cu_seqlens,
            )

        prompt = kwargs["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
        prompt = self.condition_proj(prompt[: text_positions.numel()])
        prompt = self.token_refiner(prompt, attention_config=self._token_refiner_attention_config())
        rope_position_ids = kwargs["img_position_ids"].to(device)
        rope_position_ids = rope_position_ids[:, rope_row_start:rope_row_stop]
        rope_frequencies = self.rope(rope_position_ids)
        rope_half = rope_frequencies.shape[-1] // 2
        rope_cos_sin_cache = torch.cat(
            (rope_frequencies[..., :rope_half].cos(), rope_frequencies[..., :rope_half].sin()),
            dim=-1,
        ).to(torch.bfloat16)
        sequence_lengths = self._sequence_lengths(kwargs["packed_seq_params"])
        cu_seqlens = self._cu_seqlens(kwargs["packed_seq_params"]).to(device=device, dtype=torch.int32)
        if cache_key is not None:
            self._static_cache_key = cache_key
            self._static_prompt = prompt
            self._static_rope_cos_sin = rope_cos_sin_cache
            self._static_sequence_lengths = sequence_lengths
            self._static_cu_seqlens = cu_seqlens
        return prompt, rope_cos_sin_cache, sequence_lengths, cu_seqlens

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        required = (
            "x",
            "audio_x",
            "img_position_ids",
            "unique_timesteps",
            "inverse_indices",
            "update_mask",
            "prompt_embeds",
            "img_pos_info",
            "audio_pos_info",
            "text_pos_info",
            "img_pos_for_infer_output_info",
            "packed_seq_params",
        )
        missing = [name for name in required if kwargs.get(name) is None]
        if missing:
            raise ValueError(f"MiniMaxH3DiT.forward missing required inputs: {missing}")
        video_state = kwargs["x"]
        audio_state = kwargs["audio_x"]
        if video_state.ndim != 3 or video_state.shape[0] != 1:
            raise ValueError("x must have shape [1, sequence, video_patch_dim]")
        sequence = video_state.shape[1]
        device = video_state.device
        image_positions = self._position_ids(kwargs["img_pos_info"], "img_pos_info").to(device)
        audio_positions = self._position_ids(kwargs["audio_pos_info"], "audio_pos_info").to(device)
        text_positions = self._position_ids(kwargs["text_pos_info"], "text_pos_info").to(device)
        output_positions = self._position_ids(
            kwargs["img_pos_for_infer_output_info"], "img_pos_for_infer_output_info"
        ).to(device)
        sparse_state = self.sparse_attention_state
        prefix_tokens = 0
        if self.attention_config.attn_impl == AttnImplType.SOL_ATTN:
            if sparse_state is None:
                raise RuntimeError("MiniMax H3 Sol-Attn was not initialized through set_attention_config")
            sparse_state.update(numeral_timestep=int(kwargs.get("sparse_step_index", 0)))
            prefix_tokens = int(kwargs.get("sol_prefix_tokens", output_positions.min().item()))

        local_embedding_layout = kwargs.get("local_embedding_layout")
        use_local_embedding = self.usp_flag and local_embedding_layout is not None
        if use_local_embedding:
            row_start = int(local_embedding_layout["row_start"])
            row_stop = int(local_embedding_layout["row_stop"])
            if row_start < 0 or row_stop <= row_start or row_stop > sequence:
                raise ValueError("local_embedding_layout has an invalid packed row range")
        else:
            row_start = 0
            row_stop = sequence

        prompt, rope_cos_sin_cache, sequence_lengths, cu_seqlens = self._static_inputs(
            kwargs,
            device=device,
            text_positions=text_positions,
            rope_row_start=row_start,
            rope_row_stop=row_stop,
        )
        if use_local_embedding:
            hidden = torch.zeros(row_stop - row_start, self.config.hidden_size, device=device, dtype=torch.bfloat16)

            def layout_tensor(name: str) -> torch.Tensor:
                value = local_embedding_layout[name]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"local_embedding_layout.{name} must be a tensor")
                return value.to(device=device, dtype=torch.long)

            text_source_ids = layout_tensor("text_source_ids")
            text_row_ids = layout_tensor("text_row_ids")
            if text_row_ids.numel():
                hidden.index_copy_(0, text_row_ids, prompt.index_select(0, text_source_ids))
            img_global_ids = layout_tensor("img_global_ids")
            img_row_ids = layout_tensor("img_row_ids")
            if img_row_ids.numel():
                video_rows = video_state[0].index_select(0, img_global_ids).float()
                hidden.index_copy_(0, img_row_ids, self.video_patch_proj(video_rows).to(torch.bfloat16))
            audio_global_ids = layout_tensor("audio_global_ids")
            audio_row_ids = layout_tensor("audio_row_ids")
            if audio_row_ids.numel():
                audio_rows = audio_state[0].index_select(0, audio_global_ids).float()
                hidden.index_copy_(0, audio_row_ids, self.audio_patch_proj(audio_rows).to(torch.bfloat16))
        else:
            hidden = torch.zeros(sequence, self.config.hidden_size, device=device, dtype=torch.bfloat16)
            hidden.index_copy_(0, text_positions, prompt)
            video_rows = video_state[0].index_select(0, image_positions).float()
            audio_rows = audio_state[0].index_select(0, audio_positions).float()
            hidden.index_copy_(0, image_positions, self.video_patch_proj(video_rows).to(torch.bfloat16))
            hidden.index_copy_(0, audio_positions, self.audio_patch_proj(audio_rows).to(torch.bfloat16))

        timesteps = kwargs["unique_timesteps"].reshape(-1).to(device)
        adaln_input: torch.Tensor | None = None
        block_adaln_params: tuple[tuple[torch.Tensor, ...], ...] | None = None
        final_adaln_params: tuple[torch.Tensor, torch.Tensor] | None = None
        if self._adaln_cache is None:
            if self.time_embedder is None:
                raise RuntimeError("MiniMax H3 AdaLN weights were released without an inference cache.")
            adaln_input = nn.functional.silu(self.time_embedder(timesteps)).to(torch.bfloat16)
            if self._online_adaln_cache_enabled:
                block_adaln_params, final_adaln_params = self._record_online_adaln(timesteps, adaln_input)
        else:
            block_adaln_params, final_adaln_params = self._adaln_cache.resolve(self, timesteps, device)
        inverse_indices = kwargs["inverse_indices"].reshape(-1).long().to(device)
        if inverse_indices.numel() != sequence:
            raise ValueError("inverse_indices must cover the full packed sequence")
        local_inverse_indices = inverse_indices[row_start:row_stop]
        combined_indices = kwargs.get("block_combined_indices")
        if combined_indices is not None:
            combined_indices = combined_indices.reshape(-1).long().to(device)
            if combined_indices.numel() != row_stop - row_start:
                raise ValueError("block_combined_indices must cover the local packed sequence")
        else:
            token_tags = kwargs.get("block_token_tags")
            if token_tags is None:
                token_tags = kwargs.get("token_tags")
            if token_tags is None:
                raise ValueError("token_tags or block_token_tags is required")
            token_tags = token_tags.reshape(-1).long().to(device).clamp_min(0)
            if token_tags.numel() == sequence:
                token_tags = token_tags[row_start:row_stop]
            if token_tags.numel() != row_stop - row_start:
                raise ValueError("block token tags must cover the local packed sequence")
            combined_indices = token_tags + local_inverse_indices * MINIMAX_H3_ADALN_MODALITY_NUM
        full_sequence = sequence
        if self.usp_flag:
            world_size = get_ulysses_world_size(self.device_mesh)
            if sequence % world_size:
                raise ValueError(
                    f"MiniMax H3 packed sequence length ({sequence}) must be divisible by Ulysses degree ({world_size})"
                )
            if use_local_embedding:
                inverse_indices = local_inverse_indices
            else:
                inverse_indices = inverse_indices.clone()
                rope_cos_sin_cache = rope_cos_sin_cache.clone()
                sequence_parallel_shard(
                    self.device_mesh,
                    [hidden, combined_indices, inverse_indices, rope_cos_sin_cache],
                    [0, 0, 0, 0],
                )
        else:
            inverse_indices = local_inverse_indices
        feature_cache = self.feature_cache
        if feature_cache.should_compute(True):
            # H3's gated residual ops may reuse their input storage. Preserve the
            # pre-block value so cache residuals always span the complete block stack.
            input_hidden = None if isinstance(feature_cache, NoOpCache) else hidden.clone()
            if self.tp_flag and block_adaln_params is None:
                local_adaln = torch.stack([block.adaln_proj.project_local(adaln_input) for block in self.blocks])
                first_projection = self.blocks[0].adaln_proj
                gathered_adaln = all_gather_cat(
                    local_adaln,
                    dim=-1,
                    group=first_projection.tp_group,
                    world_size=first_projection.tp_world_size,
                )
                block_adaln_params = tuple(
                    block.adaln_proj.split_output(output) for block, output in zip(self.blocks, gathered_adaln)
                )
            for index, block in enumerate(self.blocks):
                if sparse_state is not None:
                    sparse_state.update(layer_idx=index)
                hidden = block(
                    hidden,
                    adaln_input=adaln_input,
                    combined_indices=combined_indices,
                    sequence_lengths=sequence_lengths,
                    rope_cos_sin_cache=rope_cos_sin_cache,
                    attention_config=self.attention_config,
                    cu_seqlens=cu_seqlens,
                    sparse_state=sparse_state,
                    prefix_tokens=prefix_tokens,
                    adaln_params=None if block_adaln_params is None else block_adaln_params[index],
                )
            if isinstance(feature_cache, AdaTaylorCacheCalibrator):
                assert input_hidden is not None
                # Video rows greatly outnumber audio rows in H3. Calibrate decisions on
                # audio residuals so the joint hidden cache does not neglect its weaker modality.
                local_audio_positions = (
                    audio_positions[(audio_positions >= row_start) & (audio_positions < row_stop)] - row_start
                )
                calibration_output = hidden.index_select(0, local_audio_positions)
                calibration_input = input_hidden.index_select(0, local_audio_positions)
                feature_cache.update(calibration_output, calibration_input, True)
                # H3 has one joint branch. Mirror it into the legacy CFG slot so the
                # shared parameter format loads unchanged.
                feature_cache.should_compute(False)
                feature_cache.update(calibration_output, calibration_input, False)
            elif input_hidden is not None:
                feature_cache.update(hidden, input_hidden, True)
        else:
            hidden = feature_cache.approximate(hidden, True)
        video_logits, audio_logits = self.final_layer(
            hidden,
            adaln_input=adaln_input,
            adaln_params=final_adaln_params,
            inverse_indices=inverse_indices,
        )
        if self.usp_flag:
            video_logits, audio_logits = sequence_parallel_unshard(
                self.device_mesh,
                [video_logits, audio_logits],
                [0, 0],
                [full_sequence, full_sequence],
            )
        video_logits = video_logits.index_select(0, output_positions)
        audio_logits = audio_logits.index_select(0, audio_positions)
        if not bool(kwargs.get("skip_mask_out_condition", False)):
            video_logits = video_logits * kwargs["update_mask"].reshape(-1, 1).to(video_logits)
            if kwargs.get("update_audio_mask") is not None:
                audio_logits = audio_logits * kwargs["update_audio_mask"].reshape(-1, 1).to(audio_logits)
        return video_logits, audio_logits

    def enable_usp(self, device_mesh: Any | None = None) -> None:
        self.device_mesh = device_mesh if device_mesh is not None else self.device_mesh
        world_size = get_ulysses_world_size(self.device_mesh)
        local_num_heads = self.blocks[0].attn.num_heads
        if local_num_heads % world_size:
            raise ValueError(
                f"MiniMax H3 local attention heads ({local_num_heads}) must be divisible by "
                f"Ulysses degree ({world_size})"
            )
        group = get_ulysses_group(self.device_mesh) if world_size > 1 else None
        self.usp_flag = world_size > 1
        communicator = self._configure_ulysses_communicator(group)
        for block in self.blocks:
            block.attn.set_ulysses_group(group, communicator)

    def enable_quant(self, quant_type: QuantConfig | str | torch.dtype) -> None:
        """Apply supported online quantization to transformer Linear layers."""
        if not isinstance(quant_type, QuantConfig):
            super().enable_quant(quant_type)
            return
        if not quant_type.enabled:
            return

        include_names = quant_type.quantize_modules or ("blocks.",)
        if quant_type.quant_type == QuantType.TORCHAO_FP8:
            from telefuser.ops.torchao_fp8_linear import replace_linear_layers_with_torchao_fp8

            replaced = replace_linear_layers_with_torchao_fp8(
                self,
                include_names=include_names,
                exclude_names=quant_type.skip_modules,
            )
            self.torchao_fp8_replaced_linear = replaced
        elif quant_type.quant_type == QuantType.BNB_NF4:
            from telefuser.ops.bnb_nf4_linear import replace_linear_layers_with_bnb_nf4

            replaced = replace_linear_layers_with_bnb_nf4(
                self,
                compute_dtype=torch.bfloat16,
                include_names=include_names,
                exclude_names=quant_type.skip_modules,
            )
            self.bnb_nf4_replaced_linear = replaced
        elif quant_type.quant_type == QuantType.FP8:
            if quant_type.kernel_backend not in (QuantKernelBackend.AUTO, QuantKernelBackend.TF_KERNEL):
                raise ValueError(
                    "MiniMax H3 FP8 online quantization requires the tf-kernel backend; "
                    f"got {quant_type.kernel_backend.name}"
                )
            from telefuser.ops.fp8_gemm import FP8GemmOptions, count_linear_layers, enable_fp8_gemm

            def module_filter(name: str, _module: nn.Module) -> bool:
                return any(token in name for token in include_names) and not any(
                    token and token in name for token in quant_type.skip_modules
                )

            replaced = count_linear_layers(self, module_filter=module_filter)
            enable_fp8_gemm(
                self,
                options=FP8GemmOptions(
                    fp16_weight_storage="keep" if quant_type.keep_fp16_weight else "discard",
                    materialize_fp8_on_wrap=True,
                ),
                module_filter=module_filter,
            )
            self.tf_kernel_fp8_replaced_linear = replaced
        else:
            raise ValueError(f"MiniMax H3 does not support online quantization type {quant_type.quant_type.name}")

        if replaced == 0:
            raise RuntimeError("MiniMax H3 online quantization did not select any Linear layers")
        self.quant_type = quant_type.quant_type
        logger.info(f"MiniMax H3 {quant_type.quant_type.name} converted {replaced} transformer Linear layers")

    def enable_tp(self, device_mesh: Any | None = None) -> None:
        self.device_mesh = device_mesh if device_mesh is not None else self.device_mesh
        world_size = get_tp_world_size(self.device_mesh)
        if world_size <= 1:
            return
        if self.tp_flag:
            raise RuntimeError("MiniMax H3 tensor parallelism is already enabled")
        group = get_tp_group(self.device_mesh)
        if group is None:
            raise RuntimeError("MiniMax H3 TP requires a tensor-parallel process group")
        rank = get_tp_rank(self.device_mesh)
        if self.config.num_attention_heads % world_size:
            raise ValueError(
                f"MiniMax H3 attention heads ({self.config.num_attention_heads}) must divide TP degree ({world_size})"
            )
        if self.config.ffn_hidden_size % world_size:
            raise ValueError(
                f"MiniMax H3 FFN size ({self.config.ffn_hidden_size}) must divide TP degree ({world_size})"
            )
        for block in self.token_refiner.blocks:
            block.attn.enable_tp(group, rank=rank, world_size=world_size)
            block.mlp.enable_tp(group, rank=rank, world_size=world_size)
        for block in self.blocks:
            block.attn.enable_tp(group, rank=rank, world_size=world_size)
            block.mlp.enable_tp(group, rank=rank, world_size=world_size)
            if self._adaln_cache is None:
                block.adaln_proj.enable_tp(group, rank=rank, world_size=world_size)
        if self._adaln_cache is None:
            self.final_layer.adaln_proj.enable_tp(group, rank=rank, world_size=world_size)
        self.tp_flag = True

    def get_fsdp_module_names(self) -> list[str]:
        return ["blocks"]

    @staticmethod
    def state_dict_converter(config_path: str | Path | None = None) -> MiniMaxH3DiTStateDictConverter:
        return MiniMaxH3DiTStateDictConverter(config_path=config_path)


_BLOCK_INDEX = re.compile(r"^blocks\.(\d+)\.")
_REFINER_INDEX = re.compile(r"^token_refiner\.blocks\.(\d+)\.")


class MiniMaxH3DiTStateDictConverter:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = None if config_path is None else Path(config_path)

    def _config(self, state_dict: dict[str, torch.Tensor]) -> MiniMaxH3DiTConfig:
        if self.config_path is not None:
            return MiniMaxH3DiTConfig.from_json(self.config_path)
        q_norm = state_dict["blocks.0.attn.q_norm.weight"]
        qkv = state_dict["blocks.0.attn.qkv_proj.weight"]
        layers = 1 + max(int(match.group(1)) for key in state_dict if (match := _BLOCK_INDEX.match(key)))
        refiners = 1 + max(int(match.group(1)) for key in state_dict if (match := _REFINER_INDEX.match(key)))
        video_patch_dim = state_dict["video_patch_proj.weight"].shape[1]
        return MiniMaxH3DiTConfig(
            hidden_size=state_dict["video_patch_proj.weight"].shape[0],
            num_layers=layers,
            token_refiner_num_layers=refiners,
            num_attention_heads=qkv.shape[0] // (3 * q_norm.numel()),
            attention_head_dim=q_norm.numel(),
            ffn_hidden_size=state_dict["blocks.0.mlp.fc1.weight"].shape[0] // 2,
            latents_dim=video_patch_dim // 4,
            audio_latents_dim=state_dict["audio_patch_proj.weight"].shape[1],
            text_dim=state_dict["condition_proj.weight"].shape[1],
            timestep_input_dim=state_dict["time_embedder.proj_in.weight"].shape[1],
            time_embed_hidden_size=state_dict["time_embedder.proj_in.weight"].shape[0],
            time_embed_dim=state_dict["time_embedder.proj_out.weight"].shape[0],
            rope_inv_freq_len=state_dict["rope.inv_freq"].numel(),
        )

    def from_official(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        config = self._config(state_dict)
        converted = dict(state_dict)
        for key, value in state_dict.items():
            if key.endswith(".attn.qkv_proj.weight"):
                converted[key] = _reorder_grouped_qkv_to_qkv(
                    value,
                    num_query_groups=config.num_attention_heads,
                    heads_per_group=1,
                    head_dim=config.attention_head_dim,
                )
        return converted, {"config": config}

    def from_diffusers(self, state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        renamed: dict[str, torch.Tensor] = {}
        qkv_parts: dict[str, dict[str, torch.Tensor]] = {}
        direct = {
            "proj_in.": "video_patch_proj.",
            "audio_proj_in.": "audio_patch_proj.",
            "context_embedder.": "condition_proj.",
            "time_embedder.linear_1.": "time_embedder.proj_in.",
            "time_embedder.linear_2.": "time_embedder.proj_out.",
            "norm_out.norm.": "final_layer.norm.",
            "norm_out.linear.": "final_layer.adaln_proj.linear.",
            "proj_out.": "final_layer.video_out.",
            "audio_proj_out.": "final_layer.audio_out.",
        }
        for key, value in state_dict.items():
            target = key
            for source, destination in direct.items():
                if target.startswith(source):
                    target = destination + target[len(source) :]
                    break
            target = target.replace("transformer_blocks.", "blocks.")
            target = target.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
            target = target.replace(".attn.norm_q.", ".attn.q_norm.")
            target = target.replace(".attn.norm_k.", ".attn.k_norm.")
            target = target.replace(".attn.to_out.0.", ".attn.out_proj.")
            target = target.replace(".ff.net.0.proj.", ".mlp.fc1.")
            target = target.replace(".ff.net.2.", ".mlp.fc2.")
            for part in ("q", "k", "v"):
                marker = f".attn.to_{part}."
                if marker in target:
                    prefix, suffix = target.split(marker, 1)
                    qkv_parts.setdefault(f"{prefix}.attn.qkv_proj.{suffix}", {})[part] = value
                    break
            else:
                renamed[target] = value
        for target, parts in qkv_parts.items():
            if set(parts) != {"q", "k", "v"}:
                raise ValueError(f"incomplete Diffusers QKV weights for {target}")
            renamed[target] = torch.cat((parts["q"], parts["k"], parts["v"]), dim=0)
        config = self._config(renamed)
        return renamed, {"config": config}


__all__ = [
    "MINIMAX_H3_ADALN_CACHE_FORMAT_VERSION",
    "MINIMAX_H3_ADALN_CACHE_GENERATOR_VERSION",
    "MiniMaxH3AdaLNCache",
    "MINIMAX_H3_FP32_BUFFER_NAMES",
    "MINIMAX_H3_FP32_PARAM_NAMES",
    "MiniMaxH3DiT",
    "MiniMaxH3DiTConfig",
    "MiniMaxH3DiTStateDictConverter",
    "_reorder_grouped_qkv_to_qkv",
]
