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
)


def _builder() -> LingBotWorldFastControlBuilder:
    return LingBotWorldFastControlBuilder(
        LingBotWorldFastControlContext(
            control_type="act",
            device="cpu",
            torch_dtype=torch.float32,
            orig_height=8,
            orig_width=8,
            height=8,
            width=8,
            latent_h=1,
            latent_w=1,
            latent_frames=3,
            chunk_size=3,
        )
    )


def test_action_alignment_samples_external_video_rate_actions() -> None:
    action = torch.arange(9, dtype=torch.float32).unsqueeze(1)

    aligned = _builder()._align_action_frames(action, target_frames=3)

    torch.testing.assert_close(aligned[:, 0], torch.tensor([0.0, 4.0, 8.0]))


def test_action_alignment_rejects_an_incomplete_chunk_action() -> None:
    with pytest.raises(ValueError, match="shorter"):
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


def test_external_builder_matches_legacy_offline_camera_control_math() -> None:
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], 9, axis=0)
    poses[:, 2, 3] = np.linspace(0.0, 1.0, len(poses))
    intrinsics = np.repeat(np.array([[8.0, 8.0, 4.0, 4.0]], dtype=np.float32), len(poses), axis=0)
    context = _builder().context
    controls = torch.cat(_builder().build_sequence(poses, intrinsics), dim=2)

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
    ).control_tensor

    assert torch.equal(controls, legacy)
