"""
Fake Text-to-Video Pipeline for Testing Server

This is a mock pipeline that simulates the behavior of a real T2V model
without actually loading any models or requiring GPU resources.
"""

import time
import uuid
from pathlib import Path

import torch
from PIL import Image

from telefuser.utils.logging import logger

# Pipeline configuration
PPL_CONFIG = dict(
    name="fake_t2v_pipeline",
    model_root="/fake/models",
    negative_prompt="Static, blurry, low quality",
    num_inference_steps=10,
    num_frames=16,
    resolution="480p",
    cfg_scale=6.0,
    seed=42,
    target_fps=16,
)

PIPELINE_MANIFEST = {
    "contract_version": "v1",
    "pipeline_name": "fake_t2v_pipeline",
    "supported_tasks": ["t2v", "i2v", "t2i", "i2i"],
    "supported_media_types": ["video", "image"],
    "execution_mode": "serial_single_pipeline",
    "effective_max_concurrent_tasks": 1,
    "entrypoints": {
        "get_pipeline": "get_pipeline",
        "run_with_file": "run_with_file",
    },
    "task_contracts": {
        "t2v": {
            "media_type": "video",
            "required_inputs": [],
            "optional_inputs": [],
        },
        "i2v": {
            "media_type": "video",
            "required_inputs": [],
            "optional_inputs": [],
        },
        "t2i": {
            "media_type": "image",
            "required_inputs": [],
            "optional_inputs": [],
        },
        "i2i": {
            "media_type": "image",
            "required_inputs": [],
            "optional_inputs": [],
        },
    },
}


class FakeVideoPipeline:
    """Mock video generation pipeline for testing."""

    def __init__(self, device="cpu", torch_dtype=torch.float32):
        self.device = device
        self.torch_dtype = torch_dtype
        self.initialized = False
        logger.info(f"FakeVideoPipeline initialized on {device}")

    def __call__(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 10,
        num_frames: int = 16,
        cfg_scale: float = 6.0,
        seed: int = 42,
        height: int = 480,
        width: int = 480,
        **kwargs,
    ):
        """Simulate video generation with delay."""
        logger.info(f"Generating video for prompt: {prompt[:50]}...")
        logger.info(
            f"Parameters: steps={num_inference_steps}, frames={num_frames}, resolution={width}x{height}, seed={seed}"
        )

        # Simulate processing time (2-5 seconds based on complexity)
        processing_time = 2.0 + min(len(prompt) / 100, 3.0)
        time.sleep(processing_time)

        # Generate fake video frames (random noise images)
        frames = []
        for i in range(num_frames):
            # Create a simple colored frame with frame number
            img = Image.new("RGB", (width, height), color=(i * 5 % 255, i * 10 % 255, i * 15 % 255))
            frames.append(img)

        logger.info(f"Video generation complete: {len(frames)} frames")
        return frames


class FakeModuleManager:
    """Mock module manager that doesn't load real models."""

    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}

    def load_models(self, paths, torch_dtype=None):
        """Mock loading models."""
        for path in paths:
            model_id = str(uuid.uuid4())[:8]
            self.models[model_id] = {
                "path": path,
                "dtype": str(torch_dtype),
            }
        logger.info(f"Mock loaded {len(paths)} models")

    def get_model(self, name):
        """Return a mock model."""
        return torch.nn.Module()


def get_pipeline(parallelism=1):
    """
    Create a fake pipeline for testing.

    Args:
        parallelism (int): Number of parallel GPUs (ignored in fake pipeline)

    Returns:
        FakeVideoPipeline: Mock video generation pipeline
    """
    logger.info(f"Creating fake pipeline with parallelism={parallelism}")

    # Mock loading models
    module_manager = FakeModuleManager(device="cpu")
    module_manager.load_models(["/fake/vae.pth"], torch_dtype=torch.bfloat16)
    module_manager.load_models(["/fake/dit.pth"], torch_dtype=torch.bfloat16)
    module_manager.load_models(["/fake/t5.pth"], torch_dtype=torch.bfloat16)

    pipe = FakeVideoPipeline(device="cpu", torch_dtype=torch.bfloat16)
    pipe.initialized = True

    logger.info("Fake pipeline ready")
    return pipe


def get_target_video_size_from_ratio(
    aspect_ratio: str,
    resolution: str = "480p",
    height_division_factor: int = 1,
    width_division_factor: int = 1,
):
    """Calculate target video dimensions from aspect ratio."""
    # Parse aspect ratio (e.g., "16:9" -> 16/9)
    if ":" in aspect_ratio:
        w, h = aspect_ratio.split(":")
        ratio = int(w) / int(h)
    else:
        ratio = 1.0

    # Base resolution
    if resolution == "480p":
        base_height = 480
    elif resolution == "720p":
        base_height = 720
    elif resolution == "1080p":
        base_height = 1080
    else:
        base_height = 480

    height = (base_height // height_division_factor) * height_division_factor
    width = int(height * ratio)
    width = (width // width_division_factor) * width_division_factor

    return width, height


def save_video(frames, output_path, fps=16, quality=6):
    """Save frames as a video file (mock implementation)."""
    import numpy as np

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save first frame as image (for testing)
    if frames:
        frames[0].save(output_path.with_suffix(".jpg"))

    # Create a dummy video file
    with open(output_path, "wb") as f:
        f.write(b"FAKE_VIDEO_DATA")

    logger.info(f"Mock video saved to {output_path}")
    return output_path


def run(
    pipeline,
    prompt,
    negative_prompt="",
    seed=PPL_CONFIG["seed"],
    resolution=PPL_CONFIG["resolution"],
    aspect_ratio="16:9",
):
    """
    Run video generation with the fake pipeline.

    Args:
        pipeline: FakeVideoPipeline instance
        prompt: Text prompt
        negative_prompt: Negative prompt
        seed: Random seed
        resolution: Target resolution
        aspect_ratio: Aspect ratio

    Returns:
        List of PIL.Image frames
    """
    width, height = get_target_video_size_from_ratio(
        aspect_ratio,
        resolution=resolution,
        height_division_factor=2,
        width_division_factor=2,
    )

    video = pipeline(
        prompt=prompt,
        negative_prompt=f"{negative_prompt} {PPL_CONFIG['negative_prompt']}".strip(),
        num_inference_steps=PPL_CONFIG["num_inference_steps"],
        num_frames=PPL_CONFIG["num_frames"],
        cfg_scale=PPL_CONFIG["cfg_scale"],
        seed=seed,
        height=height,
        width=width,
    )
    return video


def run_with_file(
    pipeline,
    prompt,
    negative_prompt,
    seed,
    resolution,
    output_path,
    aspect_ratio: str = "16:9",
    **kwargs,
):
    """
    Run generation and save to file.

    Supports both video and image output based on the file extension.
    """
    video = run(
        pipeline,
        prompt,
        negative_prompt,
        seed,
        resolution,
        aspect_ratio,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
        video[0].save(output_path)
        logger.info(f"Mock image saved to {output_path}")
    else:
        logger.info(f"Saving target video to {output_path}")
        save_video(
            video,
            output_path,
            fps=PPL_CONFIG["target_fps"],
            quality=6,
        )


if __name__ == "__main__":
    # Test the fake pipeline
    pipe = get_pipeline(parallelism=1)

    test_prompt = "A beautiful sunset over the ocean"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "test_output.mp4"

    start = time.time()
    run_with_file(
        pipe,
        prompt=test_prompt,
        negative_prompt="",
        seed=42,
        resolution="480p",
        output_path=str(output_path),
        aspect_ratio="16:9",
    )
    elapsed = time.time() - start

    print(f"Test complete! Time: {elapsed:.2f}s")
    print(f"Output: {output_path}")
