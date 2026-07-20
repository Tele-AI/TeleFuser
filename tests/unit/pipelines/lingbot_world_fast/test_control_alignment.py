from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import torch

from telefuser.pipelines.lingbot_world_fast.control import (
    LingBotWorldFastControlBuilder,
    LingBotWorldFastControlContext,
    LingBotWorldFastOfflineControlSource,
    build_camera_control_chunk,
    compute_relative_poses,
    get_ks_transformed,
    interpolate_camera_poses,
    load_action_control_inputs,
    load_camera_control_inputs,
    truncate_control_sequence,
)
from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline


def _builder() -> LingBotWorldFastControlBuilder:
    return LingBotWorldFastControlBuilder(
        LingBotWorldFastControlContext(
            control_type="act",
            device="cpu",
            control_dtype=torch.float32,
            orig_height=8,
            orig_width=8,
            height=8,
            width=8,
            latent_h=1,
            latent_w=1,
            latent_frames=3,
            chunk_size=3,
            intrinsics=torch.tensor([8.0, 8.0, 4.0, 4.0]),
        )
    )


def test_action_alignment_samples_external_video_rate_actions() -> None:
    action = torch.arange(9, dtype=torch.float32).unsqueeze(1)

    aligned = _builder()._align_action_frames(action, target_frames=3)

    torch.testing.assert_close(aligned[:, 0], torch.tensor([0.0, 4.0, 8.0]))


def test_action_alignment_rejects_an_incomplete_chunk_action() -> None:
    with pytest.raises(ValueError, match="must be"):
        _builder()._align_action_frames(torch.zeros(2, 4), target_frames=3)


def test_prebuilt_control_is_the_only_pipeline_input() -> None:
    control = torch.ones(1, 12, 3, 1, 1)

    built = _builder().build({"control_tensor": control})

    assert built is control


def test_offline_control_source_materializes_once_when_first_requested() -> None:
    builder = _builder()
    calls = 0

    def build_sequence(*args):
        nonlocal calls
        calls += 1
        return [torch.tensor([1]), torch.tensor([2])]

    builder.build_sequence = build_sequence
    source = LingBotWorldFastOfflineControlSource(builder, poses=object(), intrinsics=object())

    second = source.control_at(1)
    first = source.control_at(0)

    assert calls == 0
    assert torch.equal(first(), torch.tensor([1]))
    assert calls == 1
    assert torch.equal(second(), torch.tensor([2]))
    assert calls == 1


def test_camera_control_loader_uses_explicit_intrinsics_path() -> None:
    poses = np.ones((9, 4, 4), dtype=np.float32)
    intrinsics = np.ones(4, dtype=np.float32)

    with patch(
        "telefuser.pipelines.lingbot_world_fast.control.np.load",
        side_effect=[poses, intrinsics],
    ) as load:
        loaded = load_camera_control_inputs("/controls", "/calibration/camera.npy")

    loaded_poses, loaded_intrinsics = loaded
    np.testing.assert_array_equal(loaded_poses, poses)
    np.testing.assert_array_equal(loaded_intrinsics, intrinsics)
    assert load.call_args_list == [
        call(Path("/controls/poses.npy")),
        call(Path("/calibration/camera.npy")),
    ]


def test_action_control_loader_uses_explicit_intrinsics_path() -> None:
    poses = np.ones((9, 4, 4), dtype=np.float32)
    intrinsics = np.ones(4, dtype=np.float32)
    action = np.ones((9, 4), dtype=np.float32)

    with patch(
        "telefuser.pipelines.lingbot_world_fast.control.np.load",
        side_effect=[poses, intrinsics, action],
    ) as load:
        loaded = load_action_control_inputs("/controls", "/calibration/camera.npy")

    assert all(actual is expected for actual, expected in zip(loaded, (poses, intrinsics, action), strict=True))
    assert load.call_args_list == [
        call(Path("/controls/poses.npy")),
        call(Path("/calibration/camera.npy")),
        call(Path("/controls/action.npy")),
    ]


def test_pipeline_prepares_offline_controls_without_exposing_internal_layers() -> None:
    context = LingBotWorldFastControlContext(
        control_type="cam",
        device="cpu",
        control_dtype=torch.float32,
        orig_height=8,
        orig_width=8,
        height=8,
        width=8,
        latent_h=1,
        latent_w=1,
        latent_frames=6,
        chunk_size=3,
        intrinsics=torch.tensor([8.0, 8.0, 4.0, 4.0]),
    )
    session_config = SimpleNamespace(frame_num=9)
    pipeline = LingBotWorldFastPipeline.__new__(LingBotWorldFastPipeline)
    pipeline.config = SimpleNamespace(orig_height=8, orig_width=8)
    first = MagicMock()
    second = MagicMock()

    with (
        patch.object(pipeline, "control_context", return_value=context),
        patch(
            "telefuser.pipelines.lingbot_world_fast.pipeline.load_camera_control_inputs",
            return_value=("poses", "intrinsics"),
        ) as load_camera,
        patch(
            "telefuser.pipelines.lingbot_world_fast.pipeline.truncate_control_sequence",
            return_value=("trimmed_poses", "trimmed_intrinsics", None),
        ) as truncate,
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastOfflineControlSource") as source_cls,
    ):
        source_cls.return_value.control_at.side_effect = [first, second]
        controls = pipeline.prepare_offline_controls(
            session_config,
            "/controls",
            "/calibration/camera.npy",
        )

    assert controls == [first, second]
    load_camera.assert_called_once_with("/controls", "/calibration/camera.npy")
    truncate.assert_called_once_with("poses", "intrinsics", None, 9)
    assert [item.args[0] for item in source_cls.return_value.control_at.call_args_list] == [0, 1]


def test_pipeline_uses_explicit_calibration_size_for_offline_intrinsics() -> None:
    context = LingBotWorldFastControlContext(
        control_type="cam",
        device="cpu",
        control_dtype=torch.float32,
        orig_height=8,
        orig_width=8,
        height=8,
        width=8,
        latent_h=1,
        latent_w=1,
        latent_frames=3,
        chunk_size=3,
        intrinsics=torch.tensor([8.0, 8.0, 4.0, 4.0]),
    )
    pipeline = LingBotWorldFastPipeline.__new__(LingBotWorldFastPipeline)
    pipeline.config = SimpleNamespace(orig_height=480, orig_width=832)

    with (
        patch.object(pipeline, "control_context", return_value=context),
        patch(
            "telefuser.pipelines.lingbot_world_fast.pipeline.load_camera_control_inputs",
            return_value=("poses", "intrinsics"),
        ),
        patch(
            "telefuser.pipelines.lingbot_world_fast.pipeline.truncate_control_sequence",
            return_value=("trimmed_poses", "trimmed_intrinsics", None),
        ),
        patch("telefuser.pipelines.lingbot_world_fast.pipeline.LingBotWorldFastOfflineControlSource") as source_cls,
    ):
        pipeline.prepare_offline_controls(
            SimpleNamespace(frame_num=9),
            "/controls",
            "/calibration/camera.npy",
            intrinsics_width=1920,
            intrinsics_height=1080,
        )

    builder = source_cls.call_args.args[0]
    assert builder.context.orig_width == 1920
    assert builder.context.orig_height == 1080


def test_external_builder_matches_legacy_offline_camera_control_math() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 9, axis=0)
    poses[:, 2, 3] = np.linspace(0.0, 1.0, len(poses))
    intrinsics = np.repeat(np.array([[8.0, 8.0, 4.0, 4.0]], dtype=np.float32), len(poses), axis=0)
    context = LingBotWorldFastControlContext(
        control_type="cam",
        device="cpu",
        control_dtype=torch.float32,
        orig_height=8,
        orig_width=8,
        height=8,
        width=8,
        latent_h=1,
        latent_w=1,
        latent_frames=3,
        chunk_size=3,
        intrinsics=torch.tensor([8.0, 8.0, 4.0, 4.0]),
    )
    controls = torch.cat(LingBotWorldFastControlBuilder(context).build_sequence(poses, intrinsics), dim=2)

    poses_t = torch.as_tensor(poses, dtype=torch.float32)
    intrinsics_t = get_ks_transformed(
        torch.as_tensor(intrinsics, dtype=torch.float32),
        height_org=context.orig_height,
        width_org=context.orig_width,
        height_resize=context.height,
        width_resize=context.width,
        height_final=context.height,
        width_final=context.width,
    )
    interpolated = interpolate_camera_poses(
        src_indices=np.linspace(0, len(poses_t) - 1, len(poses_t)),
        src_rot_mat=np.asarray(poses_t[:, :3, :3]),
        src_trans_vec=np.asarray(poses_t[:, :3, 3]),
        tgt_indices=np.linspace(0, len(poses_t) - 1, context.latent_frames),
    )
    relative = compute_relative_poses(interpolated, framewise=True)
    legacy = build_camera_control_chunk(
        relative,
        intrinsics_t[0].repeat(len(relative), 1),
        context.latent_h,
        context.latent_w,
        context.height,
        context.width,
    )

    assert torch.equal(controls, legacy)


def test_offline_control_window_keeps_first_intrinsics_and_rejects_short_action() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 9, axis=0)
    intrinsics = np.arange(32, dtype=np.float32).reshape(8, 4)

    with pytest.raises(ValueError, match="Action sequence"):
        truncate_control_sequence(poses, np.ones(4, dtype=np.float32), np.ones((8, 4), dtype=np.float32), frame_num=9)

    _, fixed_intrinsics, _ = truncate_control_sequence(poses, intrinsics, None, frame_num=9)
    np.testing.assert_array_equal(fixed_intrinsics, intrinsics[0])


def test_offline_control_window_rejects_empty_intrinsics() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 9, axis=0)

    with pytest.raises(ValueError, match="at least one row"):
        truncate_control_sequence(poses, np.empty((0, 4), dtype=np.float32), None, frame_num=9)


def test_action_mode_requires_actions_and_camera_mode_fixes_per_frame_intrinsics() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    intrinsics = np.repeat(np.array([[8.0, 8.0, 4.0, 4.0]], dtype=np.float32), 3, axis=0)

    with pytest.raises(ValueError, match="requires an action"):
        _builder().build({"poses": poses, "intrinsics": intrinsics})

    camera_context = LingBotWorldFastControlContext(
        control_type="cam",
        device="cpu",
        control_dtype=torch.float32,
        orig_height=8,
        orig_width=8,
        height=8,
        width=8,
        latent_h=1,
        latent_w=1,
        latent_frames=3,
        chunk_size=3,
        intrinsics=torch.tensor([8.0, 8.0, 4.0, 4.0]),
    )
    control = LingBotWorldFastControlBuilder(camera_context).build({"poses": poses, "intrinsics": intrinsics})
    assert control.shape == (1, 384, 3, 1, 1)

    resampled = LingBotWorldFastControlBuilder._resample_intrinsics(
        torch.tensor([[8.0, 8.0, 4.0, 4.0], [9.0, 9.0, 5.0, 5.0]]),
        target_frames=3,
    )
    torch.testing.assert_close(resampled, torch.tensor([[8.0, 8.0, 4.0, 4.0]]).repeat(3, 1))


def test_online_builder_uses_session_intrinsics_and_preserves_chunk_boundary_delta() -> None:
    context = LingBotWorldFastControlContext(
        control_type="cam",
        device="cpu",
        control_dtype=torch.float32,
        orig_height=8,
        orig_width=8,
        height=8,
        width=8,
        latent_h=1,
        latent_w=1,
        latent_frames=6,
        chunk_size=3,
        intrinsics=torch.tensor([8.0, 8.0, 4.0, 4.0]),
    )
    builder = LingBotWorldFastControlBuilder(context)
    captured_intrinsics: list[torch.Tensor] = []

    def capture(poses: torch.Tensor, intrinsics: torch.Tensor, action: object | None) -> torch.Tensor:
        captured_intrinsics.append(intrinsics)
        return poses

    builder._build_tensor = capture
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    poses[:, 2, 3] = [0.3, 0.4, 0.5]
    previous_pose = np.eye(4, dtype=np.float32)
    previous_pose[2, 3] = 0.2

    relative = builder.build({"poses": poses, "previous_pose": previous_pose})

    assert not torch.equal(relative[0], torch.eye(4))
    torch.testing.assert_close(relative[:, 2, 3], torch.ones(3))
    assert len(captured_intrinsics) == 1
    torch.testing.assert_close(captured_intrinsics[0], context.intrinsics.repeat(3, 1))
