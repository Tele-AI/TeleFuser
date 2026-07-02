"""Ray-based worker for distributed cluster execution.

Provides Ray actor implementation for scaling across multiple nodes
in a Ray cluster.
"""

from __future__ import annotations

import os
from typing import Any

import torch

try:
    import ray
except ImportError:
    ray = None

from telefuser.core.base_stage import BaseStage
from telefuser.core.config import RayConfig
from telefuser.platforms import current_platform
from telefuser.utils.logging import logger
from telefuser.worker.parallel_worker import ParallelWorker


class RayWorker:
    """Ray actor for distributed stage execution.

    Wraps a pipeline stage and executes tasks in a Ray cluster.
    Can optionally use ParallelWorker for multi-GPU within the actor.
    """

    def __init__(self, stage: BaseStage, enable_parallel: bool = False) -> None:
        self.ray_config: RayConfig = stage.model_runtime_config.ray_config
        self.worker_id: str = stage.name
        self._worker: ParallelWorker | BaseStage | None = None
        self._setup_resources()
        if enable_parallel:
            self._worker = ParallelWorker(stage)
        else:
            self._worker = stage

    def _setup_resources(self) -> None:
        """Setup GPU and memory resources for this worker."""
        gpu_config = self.ray_config.gpu_config

        logger.info(f"has device {current_platform.device_type}")
        if gpu_config.num_gpus > 0:
            gpu_ids = list(range(gpu_config.num_gpus))
            os.environ[current_platform.device_control_env_var] = ",".join(map(str, gpu_ids))
            logger.info(f"RayWorker {self.worker_id} use device: {gpu_ids}")
            if gpu_config.memory_limit > 0 and current_platform.device_type == "cuda":
                torch.cuda.set_per_process_memory_fraction(gpu_config.memory_limit)
                logger.info(f"RayWorker {self.worker_id} GPU memory limit: {gpu_config.memory_limit}")

        if self.ray_config.memory_gb > 0:
            logger.info(f"RayWorker {self.worker_id} memory limit: {self.ray_config.memory_gb}GB")

        logger.info(f"RayWorker {self.worker_id} resource configuration completed")

    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Process a task."""
        try:
            result = self._worker.process(*args, **kwargs, sync=True)
            return result
        except Exception as e:
            logger.error(f"RayWorker {self.worker_id} error processing task: {e}")
            raise

    def get_status(self) -> dict[str, Any]:
        """Get worker status and resource usage."""
        worker_type = type(self._worker).__name__ if self._worker is not None else "Unknown"

        return {
            "worker_id": self.worker_id,
            "worker_type": worker_type,
            "ray_config": self.ray_config,
            "gpu_memory_used": current_platform.max_memory_allocated(),
            "gpu_memory_total": current_platform.get_device_properties(0).total_memory,
        }

    def __del__(self) -> None:
        if hasattr(self, "_worker"):
            del self._worker


def create_ray_worker(stage: BaseStage, enable_parallel: bool = False) -> Any:
    """Create a Ray worker actor for the given stage.

    Args:
        stage: Pipeline stage to wrap
        enable_parallel: Whether to use ParallelWorker internally

    Returns:
        Ray actor handle
    """
    if ray is None:
        raise ImportError("Ray workers require installing the optional 'ray' dependency.")
    ray_config: RayConfig = stage.model_runtime_config.ray_config
    ray.init(ignore_reinit_error=True, address=ray_config.ray_address)
    resources = {
        "num_cpus": ray_config.num_cpus,
        "memory": ray_config.memory_gb * 1024 * 1024 * 1024,
        "num_gpus": ray_config.gpu_config.num_gpus,
    }
    logger.info(f"init ray actor {stage.name} with {resources}")
    return ray.remote(RayWorker).options(**resources).remote(stage, enable_parallel=enable_parallel)
