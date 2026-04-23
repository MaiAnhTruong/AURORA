import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import GAUSSIAN_STATE_UNSTABLE_ACTIVE, load_foundation_atlas  # noqa: E402
from scene.foundation_atlas_variational import build_variational_subspace, compute_local_exact_kl  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


def make_test_camera(center, fovx=1.0, fovy=0.9, width=640, height=480):
    center = torch.tensor(center, dtype=torch.float32)
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, 3] = center
    w2c = torch.inverse(c2w)
    return SimpleNamespace(
        FoVx=float(fovx),
        FoVy=float(fovy),
        znear=0.01,
        image_width=int(width),
        image_height=int(height),
        camera_center=center,
        world_view_transform=w2c.transpose(0, 1),
    )


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_variational"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        base_runtime_reliability = gm._atlas_reliability_runtime.detach().clone()
        base_effective_reliability = gm._atlas_reliability_effective.detach().clone()
        base_ref_camera = gm._atlas_ref_camera.detach().clone()

        with torch.no_grad():
            gm._atlas_state[-1] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            gm._atlas_visibility_ema[:] = 1.0
            gm._atlas_ref_camera[:] = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=gm.get_xyz.device)

        cameras = [
            make_test_camera([0.0, 0.0, -1.0]),
            make_test_camera([0.6, 0.0, -1.0], fovx=0.95, fovy=0.85),
        ]
        for camera in cameras:
            camera.camera_center = camera.camera_center.to(device=gm.get_xyz.device)
            camera.world_view_transform = camera.world_view_transform.to(device=gm.get_xyz.device)

        subspace_info = build_variational_subspace(
            gm,
            cameras,
            lambda_reg=1e-3,
            max_cameras=8,
            point_chunk_size=2,
        )
        assert subspace_info is not None
        assert float(subspace_info["rank_u"][0].item()) >= 2.0
        assert float(subspace_info["obs_view_count"][0].item()) >= 1.0
        assert int(round(float(subspace_info["rank_u"][-1].item()))) == 1
        assert int(round(float(subspace_info["rank_s"][-1].item()))) == 0
        assert int(round(float(subspace_info["rank_perp"][-1].item()))) == 2
        assert torch.allclose(
            subspace_info["support_minus_u"][-1],
            torch.zeros_like(subspace_info["support_minus_u"][-1]),
            atol=1e-5,
            rtol=1e-5,
        )
        assert torch.allclose(
            subspace_info["U"][-1],
            subspace_info["effective_support"][-1],
            atol=1e-5,
            rtol=1e-5,
        )
        subspace_metrics = subspace_info["subspace_metrics"]
        assert subspace_metrics["atlas_active_ray_count"] > 0.0
        assert subspace_metrics["atlas_active_ray_fallback_count"] == 0.0
        assert subspace_metrics["atlas_subspace_decomposition_error"] < 1e-4
        base_loss, base_metrics = compute_local_exact_kl(
            gm,
            cameras,
            scene_extent=1.0,
            weight=1.0,
            eps_perp=0.002,
            eps_tangent=0.01,
            lambda_parallel_base=5.0,
            lambda_parallel_gain=15.0,
            lambda_support_base=10.0,
            lambda_support_gain=20.0,
            lambda_perp_base=40.0,
            lambda_perp_gain=60.0,
            subspace_info=subspace_info,
        )
        for metric_name in (
            "atlas_rank1_ratio",
            "atlas_rank2_ratio",
            "atlas_rank3_ratio",
            "atlas_surface_rank1_ratio",
            "atlas_surface_rank2_ratio",
            "atlas_active_ray_count",
            "atlas_active_ray_fallback_count",
            "atlas_prior_precision_parallel_mean",
            "atlas_prior_precision_support_mean",
            "atlas_prior_precision_perp_mean",
            "atlas_subspace_decomposition_error",
            "atlas_sigma_perp_mean",
        ):
            assert metric_name in base_metrics
        assert base_metrics["atlas_active_ray_count"] > 0.0
        assert base_metrics["atlas_active_ray_fallback_count"] == 0.0
        assert 0.0 <= base_metrics["atlas_rank1_ratio"] <= 1.0
        assert 0.0 <= base_metrics["atlas_rank2_ratio"] <= 1.0
        assert 0.0 <= base_metrics["atlas_rank3_ratio"] <= 1.0
        assert base_metrics["atlas_subspace_decomposition_error"] < 1e-4

        with torch.no_grad():
            gm._atlas_reliability_runtime.fill_(0.0)
            gm._atlas_reliability_effective.fill_(0.0)
        low_reliability_subspace = build_variational_subspace(
            gm,
            cameras,
            lambda_reg=1e-3,
            max_cameras=8,
            point_chunk_size=2,
        )
        low_reliability_loss, low_reliability_metrics = compute_local_exact_kl(
            gm,
            cameras,
            scene_extent=1.0,
            weight=1.0,
            eps_perp=0.002,
            eps_tangent=0.01,
            lambda_parallel_base=5.0,
            lambda_parallel_gain=15.0,
            lambda_support_base=10.0,
            lambda_support_gain=20.0,
            lambda_perp_base=40.0,
            lambda_perp_gain=60.0,
            subspace_info=low_reliability_subspace,
        )

        with torch.no_grad():
            gm._atlas_reliability_runtime.fill_(1.0)
            gm._atlas_reliability_effective.fill_(1.0)
        high_reliability_subspace = build_variational_subspace(
            gm,
            cameras,
            lambda_reg=1e-3,
            max_cameras=8,
            point_chunk_size=2,
        )
        high_reliability_loss, high_reliability_metrics = compute_local_exact_kl(
            gm,
            cameras,
            scene_extent=1.0,
            weight=1.0,
            eps_perp=0.002,
            eps_tangent=0.01,
            lambda_parallel_base=5.0,
            lambda_parallel_gain=15.0,
            lambda_support_base=10.0,
            lambda_support_gain=20.0,
            lambda_perp_base=40.0,
            lambda_perp_gain=60.0,
            subspace_info=high_reliability_subspace,
        )

        assert torch.allclose(low_reliability_subspace["rank_u"], subspace_info["rank_u"])
        assert torch.allclose(high_reliability_subspace["rank_u"], subspace_info["rank_u"])
        assert torch.allclose(low_reliability_subspace["U"], subspace_info["U"], atol=1e-5, rtol=1e-5)
        assert torch.allclose(high_reliability_subspace["U"], subspace_info["U"], atol=1e-5, rtol=1e-5)
        assert low_reliability_metrics["atlas_prior_precision_parallel_mean"] < base_metrics["atlas_prior_precision_parallel_mean"]
        assert high_reliability_metrics["atlas_prior_precision_parallel_mean"] > base_metrics["atlas_prior_precision_parallel_mean"]
        assert float(low_reliability_loss.item()) != float(high_reliability_loss.item())

        with torch.no_grad():
            gm._atlas_reliability_runtime.copy_(base_runtime_reliability)
            gm._atlas_reliability_effective.copy_(base_effective_reliability)
            gm._atlas_ref_camera.copy_(base_ref_camera)
            gm._atlas_ref_camera[-1] = -1
        fallback_subspace = build_variational_subspace(
            gm,
            cameras,
            lambda_reg=1e-3,
            max_cameras=8,
            point_chunk_size=2,
        )
        assert fallback_subspace["subspace_metrics"]["atlas_active_ray_fallback_count"] > 0.0
        assert int(round(float(fallback_subspace["rank_u"][-1].item()))) == 1

        with torch.no_grad():
            gm._atlas_reliability_runtime.copy_(base_runtime_reliability)
            gm._atlas_reliability_effective.copy_(base_effective_reliability)
            gm._atlas_ref_camera.copy_(torch.tensor([0, 0, 1, 1], dtype=torch.long, device=gm.get_xyz.device))

        with torch.no_grad():
            gm._xyz[0, 0] += 0.5
            gm._center_log_sigma_parallel[1, 0] -= 0.6
            gm._center_log_sigma_support[2, 0] -= 0.4

        perturbed_loss, perturbed_metrics = compute_local_exact_kl(
            gm,
            cameras,
            scene_extent=1.0,
            weight=1.0,
            eps_perp=0.002,
            eps_tangent=0.01,
            lambda_parallel_base=5.0,
            lambda_parallel_gain=15.0,
            lambda_support_base=10.0,
            lambda_support_gain=20.0,
            lambda_perp_base=40.0,
            lambda_perp_gain=60.0,
        )

        assert base_metrics["atlas_kl_active_fraction"] > 0.0
        assert float(perturbed_loss.item()) > float(base_loss.item()) + 1e-4
        assert perturbed_metrics["atlas_sigma_parallel_mean"] > 0.0
        assert perturbed_metrics["atlas_sigma_support_mean"] > 0.0
        assert base_metrics["atlas_obs_view_mean"] > 0.0
        assert base_metrics["atlas_rank_u_mean"] >= 1.0

        print(json.dumps({"base": base_metrics, "perturbed": perturbed_metrics}, indent=2))
        print("[OK] Atlas local exact KL check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
