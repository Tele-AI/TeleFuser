"""
Test Client for TeleFuser Server

A simple client for testing the server without requiring
the full TFClient dependencies.
"""

import base64
import time
from pathlib import Path

import requests

from telefuser.utils.logging import logger


class TestClient:
    """Lightweight test client for the server."""

    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False  # Don't use environment proxy settings for localhost

    def health_check(self):
        """Check if server is running."""
        try:
            response = self.session.get(f"{self.base_url}/v1/service/status", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def create_t2v_task(
        self,
        prompt,
        resolution="480p",
        seed=42,
        negative_prompt="",
        aspect_ratio="16:9",
        video_length=2,
    ):
        """Create a text-to-video task."""
        payload = {
            "task": "t2v",
            "prompt": prompt,
            "seed": seed,
            "resolution": resolution,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio,
            "target_video_length": video_length,
        }

        response = self.session.post(f"{self.base_url}/v1/tasks/create", json=payload)
        response.raise_for_status()
        return response.json()

    def get_task_status(self, task_id):
        """Get task status."""
        response = self.session.get(f"{self.base_url}/v1/tasks/{task_id}/status")
        response.raise_for_status()
        return response.json()

    def wait_for_task(self, task_id, timeout=60, poll_interval=0.5):
        """Wait for task to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            task_status = status.get("status")

            if task_status == "completed":
                logger.info(f"Task {task_id} completed!")
                return status
            elif task_status == "failed":
                logger.error(f"Task {task_id} failed: {status.get('error')}")
                return status

            time.sleep(poll_interval)

        logger.warning(f"Task {task_id} timeout after {timeout}s")
        return None

    def cancel_task(self, task_id):
        """Cancel a task."""
        response = self.session.delete(f"{self.base_url}/v1/tasks/{task_id}")
        response.raise_for_status()
        return response.json()

    def get_queue_status(self):
        """Get queue status."""
        response = self.session.get(f"{self.base_url}/v1/tasks/queue/status")
        response.raise_for_status()
        return response.json()

    def get_service_metadata(self):
        """Get service metadata."""
        response = self.session.get(f"{self.base_url}/v1/service/metadata")
        response.raise_for_status()
        return response.json()


def run_basic_tests(base_url="http://localhost:8000"):
    """Run basic integration tests."""
    client = TestClient(base_url)

    print("=" * 60)
    print("TeleFuser Server Basic Tests")
    print("=" * 60)

    # Test 1: Health check
    print("\n[Test 1] Health Check...")
    if client.health_check():
        print("✓ Server is running")
    else:
        print("✗ Server is not responding")
        return False

    # Test 2: Get metadata
    print("\n[Test 2] Get Service Metadata...")
    try:
        metadata = client.get_service_metadata()
        print(f"✓ Pipeline: {metadata.get('pipeline_file')}")
        print(f"✓ Task: {metadata.get('task')}")
        print(f"✓ Parallelism: {metadata.get('parallelism')}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 3: Create task
    print("\n[Test 3] Create T2V Task...")
    try:
        result = client.create_t2v_task(
            prompt="A beautiful sunset over the ocean",
            seed=42,
        )
        task_id = result["task_id"]
        print(f"✓ Task created: {task_id}")
        print(f"✓ Status: {result['task_status']}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Test 4: Get task status
    print("\n[Test 4] Get Task Status...")
    try:
        status = client.get_task_status(task_id)
        print(f"✓ Task status: {status['status']}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 5: Get queue status
    print("\n[Test 5] Get Queue Status...")
    try:
        queue_status = client.get_queue_status()
        print(f"✓ Pending: {queue_status['pending_count']}")
        print(f"✓ Active: {queue_status['active_count']}")
        print(f"✓ Queue size: {queue_status['queue_size']}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 6: Wait for task completion
    print("\n[Test 6] Wait for Task Completion...")
    print("(This may take a few seconds...)")
    try:
        final_status = client.wait_for_task(task_id, timeout=30)
        if final_status:
            print(f"✓ Task completed: {final_status['status']}")
            if final_status.get("output_path"):
                print(f"✓ Output: {final_status['output_path']}")
        else:
            print("✗ Task timeout")
    except Exception as e:
        print(f"✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("Basic Tests Complete!")
    print("=" * 60)

    return True


def run_load_test(base_url="http://localhost:8000", num_tasks=3):
    """Run a simple load test with multiple tasks."""
    import time

    client = TestClient(base_url)

    print("\n" + "=" * 60)
    print(f"Load Test: Creating {num_tasks} tasks")
    print("=" * 60)

    task_ids = []

    # Create multiple tasks with delay to avoid rate limiting
    for i in range(num_tasks):
        try:
            result = client.create_t2v_task(
                prompt=f"Test video {i + 1}: A scenic landscape",
                seed=42 + i,
            )
            task_ids.append(result["task_id"])
            print(f"✓ Task {i + 1} created: {result['task_id']}")

            # Add delay between task creation to avoid rate limiting
            if i < num_tasks - 1:  # Don't delay after the last task
                time.sleep(1.5)

        except Exception as e:
            print(f"✗ Task {i + 1} failed: {e}")

    # Wait for all tasks
    print(f"\nWaiting for {len(task_ids)} tasks to complete...")
    completed = 0

    for task_id in task_ids:
        status = client.wait_for_task(task_id, timeout=60)
        if status and status.get("status") == "completed":
            completed += 1
            print(f"✓ {task_id[:8]}... completed")
        else:
            print(f"✗ {task_id[:8]}... failed or timeout")

    print(f"\nCompleted: {completed}/{len(task_ids)}")
    return completed == len(task_ids)


if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print(f"Connecting to: {base_url}")

    # Run basic tests
    success = run_basic_tests(base_url)

    if success:
        # Run load test
        run_load_test(base_url, num_tasks=2)

    sys.exit(0 if success else 1)
