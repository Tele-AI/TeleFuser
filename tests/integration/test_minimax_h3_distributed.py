from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import (
    AttentionConfig,
    AttnImplType,
    ModelRuntimeConfig,
    ParallelConfig,
    QuantConfig,
    QuantKernelBackend,
    QuantType,
)
from telefuser.distributed.device_mesh import create_device_mesh_from_config
from telefuser.models.minimax_h3_dit import MiniMaxH3DiT, MiniMaxH3DiTConfig
from telefuser.worker import ParallelWorker


def _small_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=32,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=4,
        attention_head_dim=8,
        ffn_hidden_size=64,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=32,
        time_embed_dim=16,
        rope_inv_freq_len=1,
    )


def _inputs() -> dict[str, object]:
    sequence = 64
    return {
        "x": torch.randn(1, sequence, 8),
        "audio_x": torch.randn(1, sequence, 2),
        "img_position_ids": torch.zeros(1, sequence, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(sequence, dtype=torch.long),
        "update_mask": torch.ones(32, dtype=torch.bool),
        "update_audio_mask": torch.ones(16, dtype=torch.bool),
        "token_tags": torch.cat(
            (
                torch.ones(16, dtype=torch.long),
                torch.full((16,), 2, dtype=torch.long),
                torch.zeros(32, dtype=torch.long),
            )
        ),
        "prompt_embeds": torch.randn(16, 16),
        "img_pos_info": {"position_ids": torch.arange(32, 64)},
        "audio_pos_info": {"position_ids": torch.arange(16, 32)},
        "text_pos_info": {"position_ids": torch.arange(0, 16)},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(32, 64)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 32, 64], dtype=torch.int32)},
    }


def _sage_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        hidden_size=256,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=8,
        attention_head_dim=128,
        ffn_hidden_size=512,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=(1, 2, 2),
        text_dim=16,
        timestep_input_dim=8,
        time_embed_hidden_size=256,
        time_embed_dim=128,
        rope_inv_freq_len=16,
    )


def _sage_inputs() -> dict[str, object]:
    sequence = 128
    used = 96
    return {
        "x": torch.randn(1, sequence, 8),
        "audio_x": torch.randn(1, sequence, 2),
        "img_position_ids": torch.zeros(1, sequence, 3, dtype=torch.float64),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(sequence, dtype=torch.long),
        "update_mask": torch.ones(64, dtype=torch.bool),
        "update_audio_mask": torch.ones(16, dtype=torch.bool),
        "token_tags": torch.cat(
            (
                torch.ones(16, dtype=torch.long),
                torch.full((16,), 2, dtype=torch.long),
                torch.zeros(64, dtype=torch.long),
                torch.full((sequence - used,), -1, dtype=torch.long),
            )
        ),
        "prompt_embeds": torch.randn(16, 16),
        "img_pos_info": {"position_ids": torch.arange(32, 96)},
        "audio_pos_info": {"position_ids": torch.arange(16, 32)},
        "text_pos_info": {"position_ids": torch.arange(0, 16)},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(32, 96)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, used, sequence], dtype=torch.int32)},
    }


class _MiniMaxH3UlyssesParityStage(BaseStage):
    def __init__(self, degree: int) -> None:
        super().__init__(
            "minimax-h3-ulysses-parity",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.bfloat16,
                parallel_config=ParallelConfig(device_ids=list(range(degree)), sp_ulysses_degree=degree),
            ),
        )
        torch.manual_seed(17)
        source = MiniMaxH3DiT(_small_config()).eval()
        self.dense = deepcopy(source)
        self.parallel = deepcopy(source)
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        self.parallel = self.parallel.to(self.device)
        if torch.distributed.get_rank() == 0:
            self.dense = self.dense.to(self.device)
        mesh = create_device_mesh_from_config(self.model_runtime_config.parallel_config)
        self.parallel.enable_usp(mesh)

    def compare(self, inputs: dict[str, object]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        parallel = tuple(tensor.cpu() for tensor in self.parallel(**inputs))
        dense: tuple[torch.Tensor, ...] = ()
        if torch.distributed.get_rank() == 0:
            dense = tuple(tensor.cpu() for tensor in self.dense(**inputs))
        return dense, parallel


class _MiniMaxH3SageTPUlyssesParityStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "minimax-h3-sage-tp-ulysses-parity",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.bfloat16,
                parallel_config=ParallelConfig(device_ids=[0, 1, 2, 3], sp_ulysses_degree=2, tp_degree=2),
            ),
        )
        torch.manual_seed(31)
        source = MiniMaxH3DiT(_sage_config()).eval()
        self.dense = deepcopy(source)
        self.parallel = deepcopy(source)
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        mesh = create_device_mesh_from_config(self.model_runtime_config.parallel_config)
        self.parallel.enable_tp(mesh)
        self.parallel = self.parallel.to(self.device)
        self.parallel.enable_usp(mesh)
        self.parallel.set_attention_config(AttentionConfig.dense_attention(AttnImplType.SAGE_ATTN_2_8_8_SM90))
        if torch.distributed.get_rank() == 0:
            self.dense = self.dense.to(self.device)
            self.dense.set_attention_config(AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))

    def compare(self, inputs: dict[str, object]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        parallel = tuple(tensor.cpu() for tensor in self.parallel(**inputs))
        dense: tuple[torch.Tensor, ...] = ()
        if torch.distributed.get_rank() == 0:
            dense = tuple(tensor.cpu() for tensor in self.dense(**inputs))
        return dense, parallel


class _MiniMaxH3FP8SolTPUlyssesParityStage(BaseStage):
    def __init__(self) -> None:
        super().__init__(
            "minimax-h3-fp8-sol-tp-ulysses-parity",
            ModelRuntimeConfig(
                device_type="cuda",
                torch_dtype=torch.bfloat16,
                parallel_config=ParallelConfig(device_ids=[0, 1, 2, 3], sp_ulysses_degree=2, tp_degree=2),
            ),
        )
        torch.manual_seed(41)
        source = MiniMaxH3DiT(_sage_config()).eval()
        self.dense = deepcopy(source)
        self.parallel = deepcopy(source)
        self.empty_cache_after_call = False

    def parallel_models(self) -> None:
        mesh = create_device_mesh_from_config(self.model_runtime_config.parallel_config)
        self.parallel.enable_tp(mesh)
        self.parallel = self.parallel.to(self.device)
        self.parallel.enable_usp(mesh)
        self.parallel.set_attention_config(
            AttentionConfig.sol_attention(
                dense_timesteps=0,
                dense_layers=0,
                tau=-1000.0,
                threshold_type="exact",
                sol_fp8=True,
            )
        )
        self.parallel.enable_quant(
            QuantConfig(
                enabled=True,
                quant_type=QuantType.FP8,
                kernel_backend=QuantKernelBackend.TF_KERNEL,
            )
        )
        if torch.distributed.get_rank() == 0:
            self.dense = self.dense.to(self.device)
            self.dense.set_attention_config(AttentionConfig.dense_attention(AttnImplType.TORCH_SDPA))

    def compare(self, inputs: dict[str, object]) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        parallel = tuple(tensor.cpu() for tensor in self.parallel(**inputs))
        dense: tuple[torch.Tensor, ...] = ()
        if torch.distributed.get_rank() == 0:
            dense = tuple(tensor.cpu() for tensor in self.dense(**inputs))
        return dense, parallel


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.parametrize("degree", [2, 4])
def test_minimax_h3_ulysses_matches_dense_packed_forward(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} CUDA devices")
    torch.manual_seed(29)
    worker = ParallelWorker(_MiniMaxH3UlyssesParityStage(degree))
    try:
        dense, parallel = worker.compare(_inputs(), sync=True)
    finally:
        worker.close()
    assert len(dense) == len(parallel) == 2
    for actual, expected in zip(parallel, dense, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_minimax_h3_sage_sm90_tp2_ulysses2_matches_dense_packed_forward() -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    torch.manual_seed(37)
    worker = ParallelWorker(_MiniMaxH3SageTPUlyssesParityStage())
    try:
        dense, parallel = worker.compare(_sage_inputs(), sync=True)
    finally:
        worker.close()
    assert len(dense) == len(parallel) == 2
    for actual, expected in zip(parallel, dense, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.12, atol=0.12)
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert cosine > 0.999


@pytest.mark.distributed
@pytest.mark.gpu
@pytest.mark.multi_gpu
def test_minimax_h3_fp8_sol_tp2_ulysses2_matches_dense_packed_forward() -> None:
    if torch.cuda.device_count() < 4:
        pytest.skip("requires four CUDA devices")
    torch.manual_seed(43)
    worker = ParallelWorker(_MiniMaxH3FP8SolTPUlyssesParityStage())
    try:
        dense, parallel = worker.compare(_sage_inputs(), sync=True)
    finally:
        worker.close()
    assert len(dense) == len(parallel) == 2
    for actual, expected in zip(parallel, dense, strict=True):
        assert torch.isfinite(actual).all()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        assert cosine > 0.98
