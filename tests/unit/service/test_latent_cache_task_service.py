from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from telefuser.service.api.schema import TaskRequest
from telefuser.service.core.task_service import MediaGenerationService
from telefuser.service_types import PipelineRunStatus


class _FakeFileService:
    def __init__(self, root: Path) -> None:
        self.output_dir = root / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_output_path(self, output_path: str, media_type, task_id: str | None = None):
        if task_id is not None:
            return self.output_dir / "tasks" / task_id / str(media_type) / output_path
        return self.output_dir / output_path


class _FakeInferenceService:
    def __init__(self) -> None:
        self.received_task_data: dict | None = None

    async def run_task_with_stop_event(self, task_data: dict, stop_event, output_root: str):
        self.received_task_data = dict(task_data)
        return {
            "status": PipelineRunStatus.SUCCESS,
            "output_path": task_data["output_path"],
            "raw": {
                "latent_payload": {
                    "latent_states_dict": {5: "latent"},
                    "saved_steps": [5],
                    "num_frames": 81,
                    "final_step": 39,
                }
            },
        }


class _FakeCacheAdapter:
    def __init__(self) -> None:
        self.events: list[str] = []

    def build_query(self, task_request):
        self.events.append(f"build_query:{task_request.prompt}:{task_request.seed}")
        return {"prompt": task_request.prompt, "seed": task_request.seed}

    def apply_resume(self, lookup_result, engine_ctx):
        self.events.append(f"apply_resume:{lookup_result}")
        return {"hit": False, "skip_step": 0, "cached_latent": None, "saved_steps": [5]}

    def on_response(self, task_request, latent_payload):
        self.events.append(f"on_response:{task_request.prompt}:{sorted(latent_payload)}")
        return {"packed": latent_payload}


class _FakeCacheService:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.saved: list[tuple[dict, dict]] = []

    async def lookup(self, cache_query):
        self.events.append(f"lookup:{cache_query['prompt']}:{cache_query['seed']}")
        return "miss"

    async def save(self, cache_query, outputs):
        self.events.append(f"save:{cache_query['prompt']}:{cache_query['seed']}")
        self.saved.append((cache_query, outputs))


def test_media_generation_service_runs_cacheseek_lifecycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        inference = _FakeInferenceService()
        cache_service = _FakeCacheService()
        cache_adapter = _FakeCacheAdapter()
        service = MediaGenerationService(
            file_service=_FakeFileService(tmp_path),
            inference_service=inference,
            cache_service=cache_service,
            cache_adapter=cache_adapter,
        )

        response = await service.generate_media_with_stop_event(
            TaskRequest(task="t2v", prompt="a cacheable prompt", seed=123, output_path="out.mp4"),
            threading.Event(),
        )

        assert response is not None
        assert response.output_path.endswith("out.mp4")
        assert inference.received_task_data is not None
        assert inference.received_task_data["latent_data"] == {
            "hit": False,
            "skip_step": 0,
            "cached_latent": None,
            "saved_steps": [5],
        }
        assert cache_adapter.events == [
            "build_query:a cacheable prompt:123",
            "apply_resume:miss",
            "on_response:a cacheable prompt:['final_step', 'latent_states_dict', 'num_frames', 'saved_steps']",
        ]
        assert cache_service.events == [
            "lookup:a cacheable prompt:123",
            "save:a cacheable prompt:123",
        ]
        assert cache_service.saved[0][1]["packed"]["saved_steps"] == [5]

    asyncio.run(scenario())
