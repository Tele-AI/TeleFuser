"""Integration test for the VLA session RPC across a real replica process."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import torch

from telefuser.service.core.pipeline_pool import PipelinePool
from telefuser.vla import RobotObservation, RobotState
from telefuser.vla.serialization import robot_action_chunk_from_wire, robot_observation_to_wire

PIPELINE_FILE = Path(__file__).parents[1] / "unit" / "service" / "fixtures" / "fake_vla_pipeline.py"


@pytest.mark.slow
def test_pipeline_pool_dispatches_vla_lifecycle_across_replica_process() -> None:
    pool = PipelinePool(
        num_replicas=1,
        replica_device_ids=[[]],
        security_level_name="NONE",
    )
    assert pool.start_all(
        str(PIPELINE_FILE),
        parallelism_per_replica=1,
        task="vla_action",
        skip_validation=True,
        vla_provider_factory="get_vla_provider",
    )

    async def scenario() -> None:
        try:
            await pool.open_session("session")
            opened = await pool.run_vla_operation(
                "session",
                "OPEN",
                {
                    "session_id": "session",
                    "model_id": "fake",
                    "embodiment_id": "fake-robot",
                    "episode_id": "episode",
                },
            )
            assert opened["model_id"] == "fake"
            observation = RobotObservation(RobotState(torch.tensor([1.0]), ("joint",), 100), {})
            response = await pool.run_vla_operation(
                "session",
                "PREDICT",
                {
                    "session_id": "session",
                    "sequence_id": 1,
                    "observation_timestamp_ns": 100,
                    "instruction": "move",
                    "observation": robot_observation_to_wire(observation),
                },
            )
            chunk = robot_action_chunk_from_wire(response["chunk"])
            assert chunk.actions.flatten().tolist() == pytest.approx([1.25, 1.5])
            await pool.run_vla_operation("session", "RESET", {"session_id": "session"})
            await pool.run_vla_operation("session", "CLOSE", {"session_id": "session"})
            await pool.close_session("session")
        finally:
            await pool.aclose()

    asyncio.run(scenario())
