from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from einops import rearrange


def _se3_inverse(poses: torch.Tensor) -> torch.Tensor:
    rot = poses[:, :3, :3]
    trans = poses[:, :3, 3:]
    rot_inv = rot.transpose(-1, -2)
    trans_inv = -torch.bmm(rot_inv, trans)
    out = torch.eye(4, device=poses.device, dtype=poses.dtype)[None].repeat(poses.shape[0], 1, 1)
    out[:, :3, :3] = rot_inv
    out[:, :3, 3:] = trans_inv
    return out


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    ref_w2c = _se3_inverse(c2ws_mat[0:1])
    relative = torch.matmul(ref_w2c, c2ws_mat)
    relative[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise and relative.shape[0] > 1:
        relative[1:] = torch.bmm(_se3_inverse(relative[:-1]), relative[1:])
    if normalize_trans:
        trans = relative[:, :3, 3]
        max_norm = torch.norm(trans, dim=-1).max()
        if float(max_norm) > 0:
            relative[:, :3, 3] = trans / max_norm
    return relative


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp

    trans_interp = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    trans = trans_interp(tgt_indices)

    src_rot = Rotation.from_matrix(src_rot_mat)
    quats = src_rot.as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_rot = Rotation.from_quat(quats)
    rot = Slerp(src_indices, src_rot)(tgt_indices).as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4), dtype=np.float32)
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = trans
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses)


def get_ks_transformed(
    ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
) -> torch.Tensor:
    fx, fy, cx, cy = ks.chunk(4, dim=-1)
    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    out = torch.zeros_like(ks)
    out[:, 0:1] = fx_resize
    out[:, 1:2] = fy_resize
    out[:, 2:3] = cx_resize - crop_offset_x
    out[:, 3:4] = cy_resize - crop_offset_y
    return out


def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2) + bias
    return grid_xy[None].repeat(n_frames, 1, 1)


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
) -> torch.Tensor:
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    fx, fy, cx, cy = ks.chunk(4, dim=-1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    z = torch.ones_like(i)
    x = (i - cx) / fx * z
    y = (j - cy) / fy * z

    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    rays_d = dirs @ c2ws_mat[:, :3, :3].transpose(-1, -2)

    if only_rays_d:
        return rays_d.view(n_frames, height, width, 3)

    rays_o = c2ws_mat[:, :3, 3][:, None, :].expand_as(rays_d)
    return torch.cat([rays_o, rays_d], dim=-1).view(n_frames, height, width, 6)


def build_camera_control_chunk(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    latent_h: int,
    latent_w: int,
    height: int,
    width: int,
) -> torch.Tensor:
    plucker = get_plucker_embeddings(poses, intrinsics, height, width, only_rays_d=False)
    control = rearrange(
        plucker,
        "f (h c1) (w c2) c -> 1 (c c1 c2) f h w",
        c1=max(1, height // latent_h),
        c2=max(1, width // latent_w),
        h=latent_h,
        w=latent_w,
    ).contiguous()
    return control


def build_action_control_chunk(
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    action: torch.Tensor,
    latent_h: int,
    latent_w: int,
    height: int,
    width: int,
) -> torch.Tensor:
    plucker = get_plucker_embeddings(poses, intrinsics, height, width, only_rays_d=True)
    action_map = action[:, None, None, :].repeat(1, height, width, 1)
    combined = torch.cat([plucker, action_map], dim=-1)
    control = rearrange(
        combined,
        "f (h c1) (w c2) c -> 1 (c c1 c2) f h w",
        c1=max(1, height // latent_h),
        c2=max(1, width // latent_w),
        h=latent_h,
        w=latent_w,
    ).contiguous()
    return control


def truncate_control_sequence(
    poses: object,
    intrinsics: object,
    action: object | None,
    frame_num: int,
) -> tuple[object, object, object | None]:
    """Select a video-rate pose/action window and keep the first camera intrinsics fixed."""
    if frame_num < 1 or (frame_num - 1) % 4:
        raise ValueError(f"frame_num must be 4n+1, got {frame_num}")
    if len(poses) < frame_num:
        raise ValueError(f"Pose sequence has {len(poses)} frames but frame_num requires {frame_num}")

    intrinsics_arr = np.asarray(intrinsics)
    if intrinsics_arr.ndim not in (1, 2) or intrinsics_arr.shape[-1] != 4:
        raise ValueError("Intrinsics must have shape (4,) or (frames, 4)")
    if intrinsics_arr.ndim == 2 and len(intrinsics_arr) < 1:
        raise ValueError("Intrinsics sequence must contain at least one row")
    if action is not None and len(action) < frame_num:
        raise ValueError(f"Action sequence has {len(action)} frames but frame_num requires {frame_num}")

    trimmed_poses = poses[:frame_num]
    trimmed_intrinsics = intrinsics_arr[0] if intrinsics_arr.ndim == 2 else intrinsics_arr
    trimmed_action = action[:frame_num] if action is not None else None
    return trimmed_poses, trimmed_intrinsics, trimmed_action


@dataclass(frozen=True)
class LingBotWorldFastControlContext:
    """Static geometry required to convert external actions into model controls."""

    control_type: str
    device: str | torch.device
    control_dtype: torch.dtype
    orig_height: int
    orig_width: int
    height: int
    width: int
    latent_h: int
    latent_w: int
    latent_frames: int
    chunk_size: int
    intrinsics: torch.Tensor


class LingBotWorldFastDeferredControl:
    """Lazily materialize a control tensor at the pipeline's legacy execution point."""

    def __init__(self, factory: Callable[[], torch.Tensor]) -> None:
        self._factory = factory

    def __call__(self) -> torch.Tensor:
        return self._factory()


class LingBotWorldFastOfflineControlSource:
    """Keep offline action data outside the generation session and materialize it once."""

    def __init__(
        self,
        builder: LingBotWorldFastControlBuilder,
        poses: object,
        intrinsics: object,
        action: object | None = None,
    ) -> None:
        self._builder = builder
        self._poses = poses
        self._intrinsics = intrinsics
        self._action = action
        self._controls: list[torch.Tensor] | None = None

    def control_at(self, chunk_index: int) -> LingBotWorldFastDeferredControl:
        if chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
        return LingBotWorldFastDeferredControl(lambda: self._materialize()[chunk_index])

    def _materialize(self) -> list[torch.Tensor]:
        if self._controls is None:
            self._controls = self._builder.build_sequence(self._poses, self._intrinsics, self._action)
        return self._controls


class LingBotWorldFastControlBuilder:
    """Convert externally owned action data into per-chunk model control tensors."""

    def __init__(self, context: LingBotWorldFastControlContext) -> None:
        self.context = context

    def defer(self, action: dict[str, object]) -> LingBotWorldFastDeferredControl:
        """Return a control factory for materialization within the pipeline call."""
        return LingBotWorldFastDeferredControl(lambda: self.build(action))

    @staticmethod
    def _align_action_frames(action: torch.Tensor, target_frames: int) -> torch.Tensor:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] == target_frames:
            return action
        video_rate_frames = (target_frames - 1) * 4 + 1
        if action.shape[0] == video_rate_frames:
            return action[::4]
        raise ValueError(
            f"Action length must be {target_frames} latent frames or {video_rate_frames} video-rate frames, "
            f"got {action.shape[0]}"
        )

    @staticmethod
    def _validate_poses(poses: torch.Tensor, target_frames: int) -> None:
        if poses.shape != (target_frames, 4, 4):
            raise ValueError(f"Poses must have shape ({target_frames}, 4, 4), got {tuple(poses.shape)}")

    @staticmethod
    def _resample_intrinsics(intrinsics: torch.Tensor, target_frames: int) -> torch.Tensor:
        if intrinsics.ndim == 1:
            if intrinsics.shape[0] != 4:
                raise ValueError(f"Static intrinsics must have shape (4,), got {tuple(intrinsics.shape)}")
            return intrinsics.unsqueeze(0).repeat(target_frames, 1)
        if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
            raise ValueError(f"Intrinsics must have shape (4,) or (frames, 4), got {tuple(intrinsics.shape)}")
        if intrinsics.shape[0] < 1:
            raise ValueError("Intrinsics sequence must contain at least one row")
        return intrinsics[0:1].repeat(target_frames, 1)

    def build(self, action: dict[str, object]) -> torch.Tensor:
        """Build one model control tensor from one external chunk action."""
        if "control_tensor" in action:
            return torch.as_tensor(
                action["control_tensor"],
                device=self.context.device,
                dtype=self.context.control_dtype,
            )
        poses = action.get("poses")
        intrinsics = action.get("intrinsics", self.context.intrinsics)
        if poses is None:
            raise ValueError("External action requires poses")
        poses_t = torch.as_tensor(poses, dtype=torch.float32, device=self.context.device)
        intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.context.device)
        self._validate_poses(poses_t, self.context.chunk_size)
        intrinsics_t = self._transform_intrinsics(self._resample_intrinsics(intrinsics_t, self.context.chunk_size))
        previous_pose = action.get("previous_pose")
        if previous_pose is None:
            poses_rel = compute_relative_poses(poses_t, framewise=True)
        else:
            previous_pose_t = torch.as_tensor(previous_pose, dtype=torch.float32, device=self.context.device)
            if previous_pose_t.shape != (4, 4):
                raise ValueError(f"Previous pose must have shape (4, 4), got {tuple(previous_pose_t.shape)}")
            poses_with_boundary = torch.cat([previous_pose_t.unsqueeze(0), poses_t], dim=0)
            poses_rel = compute_relative_poses(poses_with_boundary, framewise=True)[1:]
        return self._build_tensor(poses_rel, intrinsics_t, action.get("action")).to(dtype=torch.float32)

    def build_sequence(
        self,
        poses: object,
        intrinsics: object,
        action: object | None = None,
    ) -> list[torch.Tensor]:
        """Build all controls for an offline action sequence without storing it in a session."""
        poses_t = torch.as_tensor(poses, dtype=torch.float32)
        source_frames = len(poses_t)
        if source_frames < 2:
            raise ValueError("Control sequence requires at least two poses")
        if poses_t.shape != (source_frames, 4, 4):
            raise ValueError(f"Poses must have shape (frames, 4, 4), got {tuple(poses_t.shape)}")
        interpolated = interpolate_camera_poses(
            src_indices=np.linspace(0, source_frames - 1, source_frames),
            src_rot_mat=np.asarray(poses_t[:, :3, :3]),
            src_trans_vec=np.asarray(poses_t[:, :3, 3]),
            tgt_indices=np.linspace(0, source_frames - 1, self.context.latent_frames),
        )
        poses_rel = compute_relative_poses(interpolated.to(self.context.device), framewise=True)
        intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=self.context.device)
        intrinsics_t = self._transform_intrinsics(self._resample_intrinsics(intrinsics_t, len(poses_rel)))
        control = self._build_tensor(poses_rel, intrinsics_t, action)
        chunks = list(control.to(dtype=torch.float32).split(self.context.chunk_size, dim=2))
        if any(chunk.shape[2] != self.context.chunk_size for chunk in chunks):
            raise ValueError("Control sequence must contain complete chunks")
        return chunks

    def _transform_intrinsics(self, intrinsics: torch.Tensor) -> torch.Tensor:
        return get_ks_transformed(
            intrinsics,
            height_org=self.context.orig_height,
            width_org=self.context.orig_width,
            height_resize=self.context.height,
            width_resize=self.context.width,
            height_final=self.context.height,
            width_final=self.context.width,
        )

    def _build_tensor(self, poses: torch.Tensor, intrinsics: torch.Tensor, action: object | None) -> torch.Tensor:
        if self.context.control_type == "act":
            if action is None:
                raise ValueError("Action control mode requires an action sequence")
            action_t = torch.as_tensor(action, dtype=torch.float32, device=self.context.device)
            action_t = self._align_action_frames(action_t, len(poses))
            return build_action_control_chunk(
                poses,
                intrinsics,
                action_t,
                self.context.latent_h,
                self.context.latent_w,
                self.context.height,
                self.context.width,
            )
        if action is not None:
            raise ValueError("Camera control mode does not accept an action sequence")
        return build_camera_control_chunk(
            poses, intrinsics, self.context.latent_h, self.context.latent_w, self.context.height, self.context.width
        )


def load_camera_control_inputs(action_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = Path(action_path)
    poses = np.load(root / "poses.npy")
    intrinsics = np.load(root / "intrinsics.npy")
    return poses, intrinsics


def load_action_control_inputs(action_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(action_path)
    poses = np.load(root / "poses.npy")
    intrinsics = np.load(root / "intrinsics.npy")
    action = np.load(root / "action.npy")
    return poses, intrinsics, action
