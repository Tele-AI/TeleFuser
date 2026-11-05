from __future__ import annotations

from telefuser.service.api.openai.adapter import OpenAIResponseAdapter
from telefuser.service.api.schema import TaskRequest
from telefuser.service.core.task_manager import TaskManager


def test_completed_video_response_uses_retained_request_metadata() -> None:
    task_manager = TaskManager()
    request = TaskRequest(
        task="t2v",
        prompt="a robot walking",
        resolution="512x512",
        target_video_length=3,
    )
    task_id = task_manager.create_task(request)
    task_manager.complete_task(task_id, output_path="result.mp4")

    task = task_manager.get_task(task_id)
    status = task_manager.get_task_status(task_id)
    assert task is not None
    assert status is not None
    assert task.message is None

    response = OpenAIResponseAdapter.to_video_response_from_task(task_id, status, task.message)

    assert response.model == "wan-video"
    assert response.size == "512x512"
    assert response.seconds == "3"
