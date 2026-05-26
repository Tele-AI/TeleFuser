import numpy as np

from telefuser.pipelines.lingbot_world_fast.pipeline import LingBotWorldFastPipeline


def test_control_inputs_are_truncated_to_available_4n_plus_1_frames() -> None:
    poses = np.zeros((10, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((10, 4), dtype=np.float32)
    action = np.zeros((10, 6), dtype=np.float32)

    trimmed_poses, trimmed_intrinsics, trimmed_action, frame_num = (
        LingBotWorldFastPipeline._truncate_control_inputs_to_frame_num(
            poses=poses,
            intrinsics=intrinsics,
            action=action,
            frame_num=81,
        )
    )

    assert frame_num == 9
    assert trimmed_poses.shape == (9, 4, 4)
    assert trimmed_intrinsics.shape == (9, 4)
    assert trimmed_action.shape == (9, 6)


def test_scalar_intrinsics_are_preserved_when_truncating_controls() -> None:
    poses = np.zeros((10, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((4,), dtype=np.float32)

    trimmed_poses, trimmed_intrinsics, trimmed_action, frame_num = (
        LingBotWorldFastPipeline._truncate_control_inputs_to_frame_num(
            poses=poses,
            intrinsics=intrinsics,
            action=None,
            frame_num=5,
        )
    )

    assert frame_num == 5
    assert trimmed_poses.shape == (5, 4, 4)
    assert trimmed_intrinsics.shape == (4,)
    assert trimmed_action is None


def test_requested_frame_count_caps_longer_controls() -> None:
    poses = np.zeros((269, 4, 4), dtype=np.float32)
    intrinsics = np.zeros((269, 4), dtype=np.float32)
    action = np.zeros((269, 6), dtype=np.float32)

    trimmed_poses, trimmed_intrinsics, trimmed_action, frame_num = (
        LingBotWorldFastPipeline._truncate_control_inputs_to_frame_num(
            poses=poses,
            intrinsics=intrinsics,
            action=action,
            frame_num=81,
        )
    )

    assert frame_num == 81
    assert trimmed_poses.shape == (81, 4, 4)
    assert trimmed_intrinsics.shape == (81, 4)
    assert trimmed_action.shape == (81, 6)
