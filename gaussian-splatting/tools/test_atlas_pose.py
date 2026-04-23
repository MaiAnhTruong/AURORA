import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_ACTIVE, load_foundation_atlas  # noqa: E402
from scene.foundation_atlas_pose import (  # noqa: E402
    _select_budgeted_correspondences,
    clamp_camera_pose_delta,
    compute_dynamic_pose_trust_region,
    compute_pose_geometric_correspondence_loss,
    compute_pose_quality_score,
    compute_pose_refinement_losses,
    compute_pose_trust_region_loss,
    summarize_pose_correspondence_budget,
)
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402
from utils.graphics_utils import geom_transform_points, getProjectionMatrix  # noqa: E402
from utils.sh_utils import RGB2SH  # noqa: E402


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


class DummyPoseCamera:
    def __init__(self, image: torch.Tensor, invdepth: torch.Tensor):
        self.uid = 0
        self.original_image = image
        self.invdepthmap = invdepth
        self.depth_confidence = torch.ones_like(invdepth)
        self.depth_reliable = True
        self.pose_correspondences_xy = None
        self.pose_correspondences_xyz = None
        self.pose_correspondence_error = None
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


def _sample_rgb(image: torch.Tensor, coords_ndc: torch.Tensor):
    grid = coords_ndc.view(1, -1, 1, 2)
    sampled = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_pose"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)

        device = gm.get_xyz.device
        width = height = 64
        x = torch.linspace(0.0, 1.0, width, dtype=torch.float32, device=device)[None, :].repeat(height, 1)
        y = torch.linspace(0.0, 1.0, height, dtype=torch.float32, device=device)[:, None].repeat(1, width)
        texture = 0.15 * torch.sin(6.0 * math.pi * x) * torch.cos(4.0 * math.pi * y)
        image = torch.stack(
            (
                (x + texture).clamp(0.0, 1.0),
                (y + 0.5 * texture).clamp(0.0, 1.0),
                (0.35 + 0.3 * x + 0.25 * texture).clamp(0.0, 1.0),
            ),
            dim=0,
        )
        invdepth = (0.35 + 0.2 * x)[None]
        camera = DummyPoseCamera(image, invdepth)

        with torch.no_grad():
            gm._xyz[:] = torch.tensor(
                [
                    [-0.30, 0.00, 2.20],
                    [-0.05, 0.12, 1.90],
                    [0.18, -0.08, 2.00],
                    [0.33, 0.10, 2.30],
                ],
                dtype=torch.float32,
                device=device,
            )
            gm._atlas_positions[:] = gm._xyz.detach()
            gm._atlas_state[:] = torch.tensor(
                [GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_STABLE],
                dtype=torch.long,
                device=device,
            )
            gm._atlas_visibility_ema[:] = 1.0
            gm._atlas_photo_ema[:] = 0.0
            gm._atlas_reliability_base[:] = 0.22
            gm._atlas_reliability_runtime[:] = 0.22

            projected = geom_transform_points(gm.get_xyz.detach(), camera.full_proj_transform)
            colors = _sample_rgb(camera.original_image, projected[:, :2]).clamp(0.0, 1.0)
            gm._features_dc[:, 0, :] = RGB2SH(colors)

        base_loss, base_metrics = compute_pose_refinement_losses(
            camera,
            gm,
            sample_count=64,
            geo_weight=0.0,
            photo_weight=1.0,
            patch_radius=1,
        )

        with torch.no_grad():
            corr_ndc = geom_transform_points(gm.get_xyz.detach(), camera.full_proj_transform)
            width_scale = max(camera.image_width - 1, 1)
            height_scale = max(camera.image_height - 1, 1)
            corr_xy = torch.stack(
                (
                    (corr_ndc[:, 0] + 1.0) * 0.5 * float(width_scale),
                    (1.0 - corr_ndc[:, 1]) * 0.5 * float(height_scale),
                ),
                dim=1,
            )
            camera.pose_correspondences_xyz = gm.get_xyz.detach().clone()
            camera.pose_correspondences_xy = corr_xy.detach().clone()
            camera.pose_correspondence_error = 0.5 * torch.ones((corr_xy.shape[0],), dtype=torch.float32, device=device)

        geo_base_loss, geo_base_metrics = compute_pose_geometric_correspondence_loss(
            camera,
            gm,
            sample_count=64,
            geo_weight=1.0,
        )
        assert geo_base_metrics["pose_geo_loaded_corr"] == 4.0
        assert geo_base_metrics["pose_geo_projected_corr"] == 4.0
        assert geo_base_metrics["pose_geo_in_frame_corr"] == 4.0
        assert geo_base_metrics["pose_geo_rejected_large_error"] == 0.0
        assert geo_base_metrics["pose_geo_median_px_error"] >= 0.0
        corr_budget = summarize_pose_correspondence_budget(
            camera,
            min_correspondences=4,
            bootstrap_min_correspondences=2,
        )
        assert corr_budget["pose_corr_ready"] == 1.0
        assert corr_budget["pose_corr_bootstrap_ready"] == 1.0
        assert corr_budget["pose_corr_trustworthy"] >= 4.0
        assert corr_budget["pose_corr_reason"] == "ready"
        b1_quality = compute_pose_quality_score(
            "b1",
            geo_base_metrics,
            target_count=4,
            success_streak=0,
        )
        assert b1_quality > 0.0
        b1_translation_limit, b1_rotation_limit, b1_trust_budget = compute_dynamic_pose_trust_region(
            "b1",
            base_translation_norm=0.1,
            base_rotation_degrees=5.0,
            quality_score=b1_quality,
            bootstrap_active=True,
        )
        assert b1_translation_limit > 0.05
        assert b1_rotation_limit > 2.5
        assert b1_trust_budget["pose_trust_bootstrap_active"] == 1.0

        with torch.no_grad():
            camera.pose_delta_t.copy_(torch.tensor([0.08, 0.0, 0.0], dtype=torch.float32, device=device))
        camera.refresh_pose_matrices()
        geo_perturbed_loss, geo_perturbed_metrics = compute_pose_geometric_correspondence_loss(
            camera,
            gm,
            sample_count=64,
            geo_weight=1.0,
        )
        with torch.no_grad():
            camera.pose_delta_t.copy_(torch.tensor([1.35, 0.0, 0.0], dtype=torch.float32, device=device))
        camera.refresh_pose_matrices()
        geo_far_loss, geo_far_metrics = compute_pose_geometric_correspondence_loss(
            camera,
            gm,
            sample_count=64,
            geo_weight=1.0,
        )
        trust_loss, trust_metrics = compute_pose_trust_region_loss(
            camera,
            translation_weight=1.0,
            rotation_weight=1.0,
            max_translation_norm=0.1,
            max_rotation_degrees=5.0,
        )
        assert trust_metrics["pose_translation_norm"] > 1.0
        assert trust_metrics["pose_trust_loss"] > 0.0
        with torch.no_grad():
            camera.pose_delta_q.copy_(torch.tensor([0.9659258, 0.2588190, 0.0, 0.0], dtype=torch.float32, device=device))
        camera.refresh_pose_matrices()
        clamp_metrics = clamp_camera_pose_delta(
            camera,
            max_translation_norm=0.1,
            max_rotation_degrees=5.0,
        )
        assert clamp_metrics["pose_translation_clamped"] == 1.0
        assert clamp_metrics["pose_rotation_clamped"] == 1.0
        assert clamp_metrics["pose_translation_norm_after"] <= 0.100001
        assert clamp_metrics["pose_rotation_degrees_after"] <= 5.001
        perturbed_loss, perturbed_metrics = compute_pose_refinement_losses(
            camera,
            gm,
            sample_count=64,
            geo_weight=0.0,
            photo_weight=1.0,
            patch_radius=1,
        )
        with torch.no_grad():
            out_of_frame_xyz = torch.tensor([[3.5, 0.0, 2.0]], dtype=torch.float32, device=device)
            corrupted_xy = corr_xy[:1] + torch.tensor([[24.0, -18.0]], dtype=torch.float32, device=device)
            camera.pose_correspondences_xyz = torch.cat((gm.get_xyz.detach().clone(), gm.get_xyz.detach()[:1], out_of_frame_xyz), dim=0)
            camera.pose_correspondences_xy = torch.cat((corr_xy.detach().clone(), corrupted_xy, corr_xy[:1]), dim=0)
            camera.pose_correspondence_error = 0.5 * torch.ones((camera.pose_correspondences_xy.shape[0],), dtype=torch.float32, device=device)
            camera.pose_delta_t.copy_(torch.zeros((3,), dtype=torch.float32, device=device))
        camera.refresh_pose_matrices()
        geo_filtered_loss, geo_filtered_metrics = compute_pose_geometric_correspondence_loss(
            camera,
            gm,
            sample_count=64,
            geo_weight=1.0,
        )
        assert float(geo_filtered_loss.item()) > 0.0
        assert geo_filtered_metrics["pose_geo_loaded_corr"] == 6.0
        assert geo_filtered_metrics["pose_geo_projected_corr"] == 6.0
        assert geo_filtered_metrics["pose_geo_in_frame_corr"] == 5.0
        assert geo_filtered_metrics["pose_geo_rejected_large_error"] >= 1.0
        assert geo_filtered_metrics["pose_geo_prefilter_median_px_error"] >= geo_filtered_metrics["pose_geo_selected_median_px_error"]
        assert float(geo_perturbed_loss.item()) > float(geo_base_loss.item()) + 1e-5
        assert float(geo_far_loss.item()) > 0.0
        assert float(perturbed_loss.item()) > float(base_loss.item()) + 1e-4

        budget_xy = torch.tensor(
            [
                [4.0, 4.0],
                [5.0, 5.0],
                [36.0, 4.0],
                [37.0, 5.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        budget_reproj = torch.tensor([0.2, 1.4, 0.3, 1.2], dtype=torch.float32, device=device)
        budget_err = torch.tensor([0.6, 0.6, 0.6, 0.6], dtype=torch.float32, device=device)
        budget_keep = _select_budgeted_correspondences(
            budget_xy,
            reproj_px=budget_reproj,
            corr_err=budget_err,
            sample_count=2,
            image_width=64,
            image_height=64,
        )
        assert torch.equal(budget_keep.cpu(), torch.tensor([0, 2], dtype=torch.long))

        camera.pose_delta_t.grad = None
        template_image = camera.original_image.roll(shifts=2, dims=2)
        with torch.no_grad():
            gm._atlas_state[3] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            gm._atlas_ref_camera[3] = 0
            gm._atlas_ref_score[3] = 0.9
            gm._atlas_photo_ema[3] = 0.05
        loss_with_template, template_metrics = compute_pose_refinement_losses(
            camera,
            gm,
            sample_count=64,
            geo_weight=0.0,
            photo_weight=1.0,
            template_image=template_image,
            photo_alpha=0.5,
            gradient_weight=0.1,
            patch_feature_weight=0.1,
            patch_radius=1,
        )
        loss_with_template.backward()
        assert camera.pose_delta_t.grad is not None
        assert float(camera.pose_delta_t.grad.norm().item()) > 0.0
        assert template_metrics["pose_gradient_loss"] >= 0.0
        assert template_metrics["pose_ssim_loss"] >= 0.0
        assert template_metrics["pose_mask_mean"] > 0.0
        assert template_metrics["pose_view_support_mean"] > 0.0
        assert template_metrics["pose_stable_sample_fraction"] > 0.0
        assert template_metrics["pose_active_sample_fraction"] > 0.0
        b2_quality = compute_pose_quality_score(
            "b2",
            template_metrics,
            target_count=64,
            success_streak=2,
            quality_regressed=False,
        )
        assert b2_quality > 0.0
        b2_translation_limit, b2_rotation_limit, b2_trust_budget = compute_dynamic_pose_trust_region(
            "b2",
            base_translation_norm=0.1,
            base_rotation_degrees=5.0,
            quality_score=b2_quality,
            bootstrap_active=False,
        )
        assert b2_translation_limit > 0.0
        assert b2_rotation_limit > 0.0
        assert b2_trust_budget["pose_trust_stage"] == "b2"

        # Zero-variance patches must remain numerically stable during backward.
        camera.pose_delta_t.grad = None
        flat_template = torch.full_like(camera.original_image, 0.5)
        flat_loss, flat_metrics = compute_pose_refinement_losses(
            camera,
            gm,
            sample_count=64,
            geo_weight=0.0,
            photo_weight=1.0,
            template_image=flat_template,
            photo_alpha=0.5,
            gradient_weight=0.1,
            patch_feature_weight=0.1,
            patch_radius=1,
        )
        flat_loss.backward()
        assert camera.pose_delta_t.grad is not None
        assert torch.isfinite(camera.pose_delta_t.grad).all()
        assert flat_metrics["pose_patchfeat_loss"] >= 0.0

        print(
            json.dumps(
                {
                    "base": base_metrics,
                    "corr_budget": corr_budget,
                    "geo_base": geo_base_metrics,
                    "geo_perturbed": geo_perturbed_metrics,
                    "geo_far": geo_far_metrics,
                    "b1_trust_budget": b1_trust_budget,
                    "trust": trust_metrics,
                    "clamp": clamp_metrics,
                    "perturbed": perturbed_metrics,
                    "template": template_metrics,
                    "b2_trust_budget": b2_trust_budget,
                    "flat_template": flat_metrics,
                },
                indent=2,
            )
        )
        print("[OK] Atlas pose refinement check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
