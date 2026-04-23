import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import GAUSSIAN_STATE_STABLE  # noqa: E402
from scene.foundation_atlas_runtime import sample_gaussian_photometric_residuals  # noqa: E402
from utils.graphics_utils import getProjectionMatrix  # noqa: E402


def _quat_to_matrix(q: torch.Tensor):
    q = F.normalize(q, dim=0)
    w, x, y, z = q.unbind(dim=0)
    R = torch.zeros((3, 3), dtype=q.dtype, device=q.device)
    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)
    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)
    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)
    return R


class DummyRuntimeCamera:
    def __init__(self, image: torch.Tensor):
        self.original_image = image
        self.image_height = int(image.shape[1])
        self.image_width = int(image.shape[2])
        self.znear = 0.01
        self.zfar = 10.0
        self.FoVx = math.radians(60.0)
        self.FoVy = math.radians(60.0)
        self.base_world_view_transform = torch.eye(4, dtype=torch.float32, device=image.device)
        self.pose_delta_q = torch.nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=image.device))
        self.pose_delta_t = torch.nn.Parameter(torch.zeros((3,), dtype=torch.float32, device=image.device))
        self.projection_matrix = getProjectionMatrix(
            znear=self.znear,
            zfar=self.zfar,
            fovX=self.FoVx,
            fovY=self.FoVy,
        ).transpose(0, 1).to(image.device)
        self.refresh_pose_matrices()

    def refresh_pose_matrices(self):
        delta = torch.eye(4, dtype=torch.float32, device=self.original_image.device)
        delta[:3, :3] = _quat_to_matrix(self.pose_delta_q)
        delta[:3, 3] = self.pose_delta_t
        w2c = delta @ self.base_world_view_transform
        self.world_view_transform = w2c.transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        self.camera_center = torch.inverse(w2c)[:3, 3]


class DummyGaussians:
    def __init__(self, xyz: torch.Tensor):
        self._xyz = xyz
        self._atlas_node_ids = torch.arange(xyz.shape[0], dtype=torch.long, device=xyz.device)
        self._atlas_radius = torch.full((xyz.shape[0],), 0.15, dtype=torch.float32, device=xyz.device)
        self._atlas_state = torch.full((xyz.shape[0],), GAUSSIAN_STATE_STABLE, dtype=torch.long, device=xyz.device)
        self._center_sigma_support = torch.full((xyz.shape[0], 1), 0.02, dtype=torch.float32, device=xyz.device)
        self._center_sigma_parallel = torch.full((xyz.shape[0], 1), 0.04, dtype=torch.float32, device=xyz.device)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def has_atlas_bindings(self):
        return True

    @property
    def get_gaussian_atlas_radius(self):
        return self._atlas_radius[self._atlas_node_ids]

    @property
    def get_atlas_state(self):
        return self._atlas_state

    @property
    def get_center_sigma_support(self):
        return self._center_sigma_support

    @property
    def get_center_sigma_parallel(self):
        return self._center_sigma_parallel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = torch.zeros((3, 17, 17), dtype=torch.float32, device=device)
    camera = DummyRuntimeCamera(image)
    gaussians = DummyGaussians(torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32, device=device))
    radii = torch.tensor([3.0], dtype=torch.float32, device=device)

    rendered_ring = image.clone()
    rendered_ring[:, 7:10, 7:10] = 1.0
    rendered_ring[:, 8, 8] = 0.0
    matching_invdepth = torch.full((1, 17, 17), 0.5, dtype=torch.float32, device=device)
    mismatch_invdepth = torch.full((1, 17, 17), 0.05, dtype=torch.float32, device=device)

    match_residual, match_visible = sample_gaussian_photometric_residuals(
        camera,
        gaussians,
        rendered_ring,
        image,
        radii,
        rendered_invdepth=matching_invdepth,
    )
    mismatch_residual, mismatch_visible = sample_gaussian_photometric_residuals(
        camera,
        gaussians,
        rendered_ring,
        image,
        radii,
        rendered_invdepth=mismatch_invdepth,
    )

    assert bool(match_visible[0].item())
    assert float(match_residual[0].item()) > 0.10
    assert float(mismatch_residual[0].item()) < float(match_residual[0].item()) * 0.5
    if bool(mismatch_visible[0].item()):
        assert float(mismatch_residual[0].item()) < 0.10

    print(
        json.dumps(
            {
                "matching_residual": float(match_residual[0].item()),
                "matching_visible": bool(match_visible[0].item()),
                "mismatch_residual": float(mismatch_residual[0].item()),
                "mismatch_visible": bool(mismatch_visible[0].item()),
            },
            indent=2,
        )
    )
    print("[OK] Atlas runtime evidence check passed.")


if __name__ == "__main__":
    main()
