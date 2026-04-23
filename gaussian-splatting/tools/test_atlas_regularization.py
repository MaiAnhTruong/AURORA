import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import GAUSSIAN_STATE_UNSTABLE_ACTIVE, load_foundation_atlas  # noqa: E402
from scene.foundation_atlas_regularization import compute_atlas_regularization  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_regularization"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        train_cameras = [
            SimpleNamespace(camera_center=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=gm.get_xyz.device)),
            SimpleNamespace(camera_center=torch.tensor([1.5, 0.0, -1.0], dtype=torch.float32, device=gm.get_xyz.device)),
        ]

        base_loss, base_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=1.0,
            aniso_weight=1.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
        )
        assert base_metrics["atlas_mean_loss"] < 1e-4
        assert base_metrics["atlas_ori_loss"] < 1e-8

        with torch.no_grad():
            surface_normal = gm.get_gaussian_atlas_basis[0, :, 2].detach()
            gm._xyz[0] = gm._atlas_positions[0] + 0.25 * surface_normal
        out_of_support_loss, out_of_support_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=0.0,
            aniso_weight=0.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
        )
        assert float(out_of_support_loss.item()) < 1e-8
        assert out_of_support_metrics["atlas_mean_projected_drift_regularized"] < 1e-6
        with torch.no_grad():
            gm._xyz[0] = gm._atlas_positions[0]

        with torch.no_grad():
            original_log_scaling = gm._scaling[1].clone()
            gm._scaling[1] = gm._scaling[1] + 0.35
        uniform_rescale_loss, uniform_rescale_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=1.0,
            aniso_weight=1.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
        )
        assert abs(uniform_rescale_metrics["atlas_ori_loss"] - base_metrics["atlas_ori_loss"]) < 1e-7
        assert abs(uniform_rescale_metrics["atlas_aniso_loss"] - base_metrics["atlas_aniso_loss"]) < 5e-7
        assert abs(uniform_rescale_metrics["atlas_shape_loss"] - base_metrics["atlas_shape_loss"]) < 5e-7
        assert abs(float(uniform_rescale_loss.item()) - float(base_loss.item())) < 5e-7
        with torch.no_grad():
            gm._scaling[1] = original_log_scaling

        with torch.no_grad():
            gm._xyz[0, 0] += 0.2

        translated_loss, translated_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=1.0,
            aniso_weight=1.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
        )
        assert translated_metrics["atlas_mean_projected_drift_regularized"] > base_metrics["atlas_mean_projected_drift_regularized"] + 1e-4
        assert float(translated_loss.item()) > float(base_loss.item()) + 1e-4
        assert translated_metrics["atlas_mean_total_loss"] > base_metrics["atlas_mean_total_loss"] + 1e-4
        assert abs(translated_metrics["atlas_shape_loss"] - base_metrics["atlas_shape_loss"]) < 1e-7

        with torch.no_grad():
            gm._xyz[0] = gm._atlas_positions[0]

        with torch.no_grad():
            gm._atlas_state[3] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            gm._atlas_ref_camera[3] = 1

        active_base_loss, active_base_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=1.0,
            aniso_weight=1.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
            active_state_weight=1.0,
        )
        assert active_base_metrics["atlas_fallback_fraction"] > 0.0

        with torch.no_grad():
            gm._xyz[3] = gm._atlas_positions[3]
        ray_dir = (gm._atlas_positions[3] - train_cameras[1].camera_center).detach()
        ray_dir = ray_dir / torch.linalg.norm(ray_dir).clamp_min(1e-8)
        fallback_axis = torch.tensor([0.0, 1.0, 0.0], dtype=ray_dir.dtype, device=ray_dir.device)
        if torch.abs(torch.dot(ray_dir, fallback_axis)) > 0.95:
            fallback_axis = torch.tensor([1.0, 0.0, 0.0], dtype=ray_dir.dtype, device=ray_dir.device)
        orth_dir = torch.cross(ray_dir, fallback_axis, dim=0)
        orth_dir = orth_dir / torch.linalg.norm(orth_dir).clamp_min(1e-8)

        with torch.no_grad():
            gm._xyz[3] = gm._atlas_positions[3] + 0.15 * ray_dir
        active_along_loss, active_along_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=0.0,
            aniso_weight=0.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
            mean_passive_state_weight=1.0,
            mean_active_state_weight=1.0,
            active_state_weight=0.0,
        )
        with torch.no_grad():
            gm._xyz[3] = gm._atlas_positions[3] + 0.15 * orth_dir
        active_orth_loss, active_orth_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=0.0,
            aniso_weight=0.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
            mean_passive_state_weight=1.0,
            mean_active_state_weight=1.0,
            active_state_weight=0.0,
        )
        assert float(active_along_loss.item()) > float(active_orth_loss.item()) + 1e-4
        assert active_along_metrics["atlas_mean_projected_drift_regularized"] > active_orth_metrics["atlas_mean_projected_drift_regularized"] + 1e-4
        with torch.no_grad():
            gm._xyz[3] = gm._atlas_positions[3]

        with torch.no_grad():
            gm._xyz[3, 2] += 0.2
            gm._rotation[3] = torch.tensor([0.70710677, 0.70710677, 0.0, 0.0], device=gm._rotation.device)
            gm._scaling[3, 0] += 0.5
            gm._scaling[3, 1] -= 0.2
            gm._atlas_reliability_runtime[:3] = 0.0
            gm._atlas_reliability_effective[:3] = 0.0
            gm._atlas_reliability_base[:3] = 0.0

        unresolved_active_loss, unresolved_active_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=1.0,
            aniso_weight=1.0,
            huber_delta=0.02,
            train_cameras=None,
            active_state_weight=1.0,
        )
        assert unresolved_active_metrics["atlas_active_ray_fraction"] == 0.0
        assert unresolved_active_metrics["atlas_unresolved_active_fraction"] > 0.0
        assert unresolved_active_metrics["atlas_regularized_fraction"] < 1.0
        assert float(unresolved_active_loss.item()) < 1e-8
        assert unresolved_active_metrics["atlas_ori_loss"] < 1e-8
        assert unresolved_active_metrics["atlas_aniso_loss"] < 1e-8
        assert unresolved_active_metrics["atlas_mean_projected_drift_regularized"] < 1e-8
        assert unresolved_active_metrics["atlas_mean_total_loss"] < 1e-8
        assert unresolved_active_metrics["atlas_mean_projected_drift"] >= unresolved_active_metrics["atlas_mean_projected_drift_regularized"]
        assert unresolved_active_metrics["atlas_shape_loss"] < 1e-8
        assert unresolved_active_metrics["atlas_regularization_total_loss"] < 1e-8

        perturbed_loss, perturbed_metrics = compute_atlas_regularization(
            gm,
            scene_extent=1.0,
            mean_weight=1.0,
            ori_weight=1.0,
            aniso_weight=1.0,
            huber_delta=0.02,
            train_cameras=train_cameras,
            active_state_weight=1.0,
        )
        assert float(perturbed_loss.item()) > float(unresolved_active_loss.item()) + 1e-3
        assert perturbed_metrics["atlas_active_ray_fraction"] > 0.0
        assert perturbed_metrics["atlas_ori_loss"] > unresolved_active_metrics["atlas_ori_loss"] + 1e-3
        assert perturbed_metrics["atlas_aniso_loss"] > unresolved_active_metrics["atlas_aniso_loss"] + 1e-3
        assert perturbed_metrics["atlas_shape_loss"] > unresolved_active_metrics["atlas_shape_loss"] + 1e-3
        assert perturbed_metrics["atlas_regularization_total_loss"] >= perturbed_metrics["atlas_shape_loss"]

        print(
            json.dumps(
                {
                    "base": base_metrics,
                    "out_of_support": out_of_support_metrics,
                    "active_base": active_base_metrics,
                    "active_along": active_along_metrics,
                    "active_orth": active_orth_metrics,
                    "unresolved_active": unresolved_active_metrics,
                    "perturbed": perturbed_metrics,
                },
                indent=2,
            )
        )
        print("[OK] Atlas regularization check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
