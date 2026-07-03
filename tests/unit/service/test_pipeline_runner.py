from __future__ import annotations

from pathlib import Path

from PIL import Image

from telefuser.service.core.pipeline_runner import _select_kwargs


def test_select_kwargs_injects_image_for_var_kwargs_pipeline(tmp_path: Path) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (16, 16), color="white").save(image_path)

    def run_with_file(pipeline, prompt, output_path, **kwargs):
        return None

    kwargs = _select_kwargs(
        run_with_file,
        task_data={
            "prompt": "test prompt",
            "output_path": "out.mp4",
            "first_image_path": str(image_path),
        },
        module=None,
    )

    assert kwargs["prompt"] == "test prompt"
    assert kwargs["output_path"] == "out.mp4"
    assert kwargs["image_path"] == str(image_path)
    assert isinstance(kwargs["image"], Image.Image)
    assert kwargs["image"].size == (16, 16)
