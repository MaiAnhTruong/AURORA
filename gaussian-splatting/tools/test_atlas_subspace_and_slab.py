import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import (  # noqa: E402
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
    load_foundation_atlas,
)
from scene.foundation_atlas_exploration import compute_exploration_slab_loss, compute_point_slab_bounds  # noqa: E402
from scene.foundation_atlas_variational import build_variational_subspace  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_subspace"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)

        device = gm.get_xyz.device
        camera_centers = torch.tensor(
            [[0.0, 0.0, -1.0], [0.75, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            gm._xyz[:] = torch.tensor(
                [
                    [0.0, 0.0, 0.2],
                    [0.5, 0.0, 0.25],
                    [1.0, 0.0, 0.2],
                    [1.4, 0.0, 0.35],
                ],
                dtype=torch.float32,
                device=device,
            )
            gm._atlas_positions[:] = gm._xyz.detach()
            gm._atlas_state[:] = torch.tensor(
                [GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE, GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_ACTIVE],
                dtype=torch.long,
                device=device,
            )
            gm._atlas_ref_camera[:] = torch.tensor([0, 0, 0, 0], dtype=torch.long, device=device)

        subspace = build_variational_subspace(gm, camera_centers, lambda_reg=1e-3, max_cameras=8, point_chunk_size=32)
        directions = subspace["direction"]
        surface_normal = gm.get_gaussian_atlas_basis[0, :, 2]
        edge_axis = gm.get_gaussian_atlas_basis[2, :, 0]
        unstable_ray = torch.nn.functional.normalize(gm.get_xyz[3] - camera_centers[0], dim=0)

        assert abs(float((directions[0] * surface_normal).sum().detach().item())) < 1e-4
        assert abs(float((directions[2] * edge_axis).sum().detach().item())) > 0.99
        assert abs(float((directions[3] * unstable_ray).sum().detach().item())) > 0.99

        slab = compute_point_slab_bounds(
            gm,
            torch.tensor([3], dtype=torch.long, device=device),
            camera_centers=camera_centers,
            slab_radius_mult=2.0,
        )
        center_tau = 0.5 * (slab["tau_min"] + slab["tau_max"])
        with torch.no_grad():
            gm._xyz[3] = slab["ref_centers"][0] + center_tau[0] * slab["ray_dirs"][0]
        zero_loss, zero_metrics = compute_exploration_slab_loss(
            gm,
            camera_centers,
            weight=1.0,
            slab_radius_mult=2.0,
        )

        slab = compute_point_slab_bounds(
            gm,
            torch.tensor([3], dtype=torch.long, device=device),
            camera_centers=camera_centers,
            slab_radius_mult=2.0,
        )
        with torch.no_grad():
            gm._xyz[3] = slab["ref_centers"][0] + (slab["tau_max"][0] + 0.35) * slab["ray_dirs"][0]
        bad_loss, bad_metrics = compute_exploration_slab_loss(
            gm,
            camera_centers,
            weight=1.0,
            slab_radius_mult=2.0,
        )

        assert float(zero_loss.item()) < 1e-8
        assert float(bad_loss.item()) > 1e-4
        assert bad_metrics["atlas_slab_active_count"] == 1.0
        assert bad_metrics["atlas_slab_mean_span"] > 0.0

        with torch.no_grad():
            gm._atlas_ref_camera[3] = -1
        invalid_ref_slab = compute_point_slab_bounds(
            gm,
            torch.tensor([3], dtype=torch.long, device=device),
            camera_centers=camera_centers,
            slab_radius_mult=2.0,
            require_valid_ref_camera=True,
            min_reference_score=0.05,
        )
        assert invalid_ref_slab is not None
        assert int(invalid_ref_slab["resolved_ref_camera"][0].item()) >= 0
        assert bool(invalid_ref_slab["repaired_ref_mask"][0].item())
        assert float(invalid_ref_slab["ref_score"][0].item()) >= 0.05

        print(
            json.dumps(
                {
                    "surface_dir": directions[0].detach().cpu().tolist(),
                    "edge_dir": directions[2].detach().cpu().tolist(),
                    "unstable_dir": directions[3].detach().cpu().tolist(),
                    "zero_metrics": zero_metrics,
                    "bad_metrics": bad_metrics,
                },
                indent=2,
            )
        )
        print("[OK] Atlas subspace and slab check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
