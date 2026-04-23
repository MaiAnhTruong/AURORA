import json
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "diff_gaussian_rasterization" not in sys.modules:
    stub = types.ModuleType("diff_gaussian_rasterization")

    class _DummyRasterizationSettings:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyRasterizer:
        def __init__(self, *args, **kwargs):
            pass

    stub.GaussianRasterizationSettings = _DummyRasterizationSettings
    stub.GaussianRasterizer = _DummyRasterizer
    sys.modules["diff_gaussian_rasterization"] = stub

from scene.foundation_atlas import (  # noqa: E402
    GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    load_foundation_atlas,
)
from scene.foundation_atlas_exploration import compute_point_slab_bounds  # noqa: E402
from scene.foundation_atlas_pose import (  # noqa: E402
    should_enable_pose_photometric_refinement,
    should_enable_pose_refinement,
    should_freeze_pose_refinement,
)
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402
from train import _compute_prune_controls  # noqa: E402


def build_training_args():
    return SimpleNamespace(
        percent_dense=1.0,
        position_lr_init=1e-4,
        position_lr_final=1e-6,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=1000,
        feature_lr=1e-3,
        opacity_lr=1e-2,
        scaling_lr=1e-3,
        rotation_lr=1e-3,
        center_uncertainty_lr=1e-3,
        exposure_lr_init=1e-2,
        exposure_lr_final=1e-3,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        iterations=4000,
    )


def build_prune_opt():
    return SimpleNamespace(
        iterations=4000,
        densify_from_iter=500,
        prune_from_iter=500,
        min_points_before_prune=0,
        prune_min_capacity_ratio=1.25,
        prune_min_capacity_extra=2,
        prune_hard_floor_ratio=0.4,
        pose_enable_stable_ratio=0.45,
        pose_enable_max_drift_ratio=0.01,
        pose_enable_min_capacity_ratio=1.25,
    )


def log_anisotropy_ratios(scales):
    safe_scales = torch.sort(scales.clamp_min(1e-8), dim=1, descending=True).values
    return torch.stack(
        (
            torch.log(safe_scales[:, 0] / safe_scales[:, 1]),
            torch.log(safe_scales[:, 1] / safe_scales[:, 2]),
        ),
        dim=1,
    )


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_capacity_controls"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm.training_setup(build_training_args())

        device = gm.get_xyz.device
        init_points = gm.get_init_point_count()
        assert init_points == int(gm.get_xyz.shape[0])

        gm_split = GaussianModel(sh_degree=0)
        gm_split.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm_split.training_setup(build_training_args())
        split_device = gm_split.get_xyz.device
        with torch.no_grad():
            gm_split._atlas_state[:] = GAUSSIAN_STATE_STABLE
            gm_split.tmp_radii = torch.ones((gm_split.get_xyz.shape[0],), dtype=torch.float32, device=split_device)
            parent_scale = gm_split.get_scaling[0].detach().clone()
            parent_aniso = log_anisotropy_ratios(parent_scale.reshape(1, 3)).detach()
            parent_node_id = gm_split.get_atlas_node_ids[0].detach().clone()
            parent_state = gm_split.get_atlas_state[0].detach().clone()
            parent_rotation = gm_split._rotation[0].detach().clone()
        split_candidate = torch.zeros((gm_split.get_xyz.shape[0],), dtype=torch.bool, device=split_device)
        split_candidate[0] = True
        split_metrics = gm_split.densify_and_split(
            grads=None,
            grad_threshold=0.0,
            scene_extent=1.0,
            N=2,
            candidate_mask=split_candidate,
        )
        assert split_metrics["split_count"] == 2.0
        assert split_metrics["split_child_scale_ratio_max"] < 0.75
        assert split_metrics["split_child_log_anisotropy_delta_mean"] < 1e-6
        child_scales = gm_split.get_scaling[-2:].detach()
        child_aniso = log_anisotropy_ratios(child_scales)
        assert torch.all(child_scales.max(dim=1).values < parent_scale.max() + 1e-6)
        assert torch.allclose(child_aniso, parent_aniso.expand_as(child_aniso), atol=1e-6, rtol=1e-6)
        assert torch.all(gm_split.get_atlas_node_ids[-2:] == parent_node_id)
        assert torch.all(gm_split.get_atlas_state[-2:] == parent_state)
        assert torch.allclose(gm_split._rotation[-2:].detach(), parent_rotation.expand(2, -1), atol=1e-6, rtol=1e-6)

        prune_opt = build_prune_opt()
        early_prune_controls = _compute_prune_controls(600, prune_opt, gm)
        assert early_prune_controls["min_points_before_prune"] == 6
        assert not early_prune_controls["prune_enabled"]

        with torch.no_grad():
            gm._atlas_state[:] = GAUSSIAN_STATE_STABLE
            gm.xyz_gradient_accum.fill_(1.0)
            gm.denom.fill_(1.0)
            gm._atlas_visibility_ema.zero_()
            gm._atlas_gc_fail_count.zero_()
            gm._opacity.fill_(0.0)

        first_radii = torch.ones((gm.get_xyz.shape[0],), dtype=torch.float32, device=device)
        densify_metrics = gm.densify_and_prune_with_atlas(
            max_grad=0.01,
            min_opacity=0.005,
            extent=1.0,
            max_screen_size=None,
            radii=first_radii,
            camera_center=torch.zeros((3,), dtype=torch.float32, device=device),
            explore_grad_scale=0.75,
            explore_slab_radius_mult=2.0,
            explore_jitter_scale=0.45,
            all_camera_centers=torch.zeros((1, 3), dtype=torch.float32, device=device),
            prune_enabled=False,
            min_points_to_keep=early_prune_controls["min_points_to_keep"],
            visibility_threshold=0.02,
            max_reattach_failures=2,
            enable_soft_prune=False,
        )
        assert densify_metrics["clone_count"] > 0.0
        assert densify_metrics["stable_clone_count"] == densify_metrics["clone_count"]
        assert densify_metrics["stable_split_count"] == densify_metrics["split_count"]
        assert densify_metrics["active_explore_clone_count"] == 0.0
        assert densify_metrics["pruned_count"] == 0.0
        grown_points = int(gm.get_xyz.shape[0])
        assert grown_points > init_points

        gm_active = GaussianModel(sh_degree=0)
        gm_active.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm_active.training_setup(build_training_args())
        active_device = gm_active.get_xyz.device
        with torch.no_grad():
            gm_active._atlas_state[:] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            gm_active.xyz_gradient_accum.fill_(1.0)
            gm_active.denom.fill_(1.0)
            gm_active._atlas_ref_camera[:] = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=active_device)
            gm_active._atlas_ref_score.fill_(1.0)
            gm_active._atlas_visibility_ema.fill_(1.0)
            gm_active._atlas_gc_fail_count.zero_()
            gm_active._opacity.fill_(0.0)
        active_radii = torch.ones((gm_active.get_xyz.shape[0],), dtype=torch.float32, device=active_device)
        active_densify_metrics = gm_active.densify_and_prune_with_atlas(
            max_grad=0.01,
            min_opacity=0.005,
            extent=1.0,
            max_screen_size=None,
            radii=active_radii,
            camera_center=torch.zeros((3,), dtype=torch.float32, device=active_device),
            explore_grad_scale=0.75,
            explore_slab_radius_mult=2.0,
            explore_jitter_scale=0.45,
            all_camera_centers=torch.tensor(
                [[0.0, 0.0, -1.0], [0.5, 0.0, -1.0]],
                dtype=torch.float32,
                device=active_device,
            ),
            prune_enabled=False,
            min_points_to_keep=init_points,
            visibility_threshold=0.02,
            max_reattach_failures=2,
            enable_soft_prune=False,
        )
        assert active_densify_metrics["explore_clone_count"] > 0.0
        assert active_densify_metrics["active_explore_clone_count"] == active_densify_metrics["explore_clone_count"]
        assert active_densify_metrics["stable_clone_count"] == 0.0
        assert active_densify_metrics["stable_split_count"] == 0.0

        gm_repair = GaussianModel(sh_degree=0)
        gm_repair.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm_repair.training_setup(build_training_args())
        repair_device = gm_repair.get_xyz.device
        repair_camera_centers = torch.tensor(
            [[0.0, 0.0, -1.0], [0.5, 0.0, -1.0]],
            dtype=torch.float32,
            device=repair_device,
        )
        with torch.no_grad():
            gm_repair._atlas_state[:] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            gm_repair._atlas_ref_camera.fill_(-1)
            gm_repair._atlas_ref_score.zero_()
            gm_repair._atlas_visibility_ema.zero_()
            gm_repair._atlas_high_residual_count.fill_(3)
            gm_repair._atlas_photo_ema.fill_(0.10)
            gm_repair._opacity.fill_(0.0)
            gm_repair.xyz_gradient_accum.zero_()
            gm_repair.denom.fill_(1.0)
            gm_repair.tmp_radii = torch.ones((gm_repair.get_xyz.shape[0],), dtype=torch.float32, device=repair_device)
            if gm_repair._atlas_view_weights.ndim != 2 or gm_repair._atlas_view_weights.shape[1] != 2:
                gm_repair._atlas_view_weights = torch.zeros(
                    (gm_repair._atlas_positions.shape[0], 2),
                    dtype=torch.float32,
                    device=repair_device,
                )
                gm_repair._atlas_view_counts = torch.zeros(
                    (gm_repair._atlas_positions.shape[0], 2),
                    dtype=torch.int32,
                    device=repair_device,
                )
            gm_repair._atlas_view_weights.zero_()
            gm_repair._atlas_view_weights[:, 1] = 1.0
            gm_repair._atlas_view_counts.zero_()
            gm_repair._atlas_view_counts[:, 1] = 3
        repair_start_count = int(gm_repair.get_xyz.shape[0])
        empty_candidate_mask = torch.zeros((repair_start_count,), dtype=torch.bool, device=repair_device)
        repair_metrics = gm_repair.explore_and_clone(
            grads=torch.zeros_like(gm_repair.xyz_gradient_accum),
            grad_threshold=0.01,
            camera_center=repair_camera_centers[0],
            slab_radius_mult=2.0,
            jitter_scale=3.0,
            allowed_mask=gm_repair.get_atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
            all_camera_centers=repair_camera_centers,
            active_min_lifetime_iters=3,
            min_opacity=0.005,
            stable_residual_threshold=0.03,
            candidate_mask=empty_candidate_mask,
            candidate_metrics={"explore_candidate_count": 0.0},
        )
        assert repair_metrics["explore_clone_count"] > 0.0
        assert repair_metrics["explore_slab_admission_added_count"] > 0.0
        assert repair_metrics["explore_slab_admission_ref_repair_count"] > 0.0
        assert repair_metrics["explore_ref_repair_count"] > 0.0
        assert repair_metrics["explore_slab_soft_clamp_count"] >= 0.0
        repair_new_idx = torch.arange(repair_start_count, int(gm_repair.get_xyz.shape[0]), device=repair_device)
        assert torch.all(gm_repair.get_atlas_ref_camera[repair_new_idx] >= 0)
        repair_slab = compute_point_slab_bounds(
            gm_repair,
            repair_new_idx,
            camera_centers=repair_camera_centers,
            slab_radius_mult=2.0,
            detach_points=True,
            require_valid_ref_camera=True,
            min_reference_score=0.05,
        )
        assert repair_slab is not None
        assert torch.all(repair_slab["tau"] >= repair_slab["tau_min"] - 1e-6)
        assert torch.all(repair_slab["tau"] <= repair_slab["tau_max"] + 1e-6)
        gm_repair.tmp_radii = None

        prune_controls = _compute_prune_controls(700, prune_opt, gm)
        assert prune_controls["prune_enabled"]
        assert prune_controls["min_points_to_keep"] == 6

        with torch.no_grad():
            gm.xyz_gradient_accum.zero_()
            gm.denom.fill_(1.0)
            gm._atlas_visibility_ema.zero_()
            gm._atlas_gc_fail_count.fill_(3)
            gm._atlas_state[:2] = GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
            gm._opacity.fill_(-12.0)

        second_radii = torch.ones((gm.get_xyz.shape[0],), dtype=torch.float32, device=device)
        prune_metrics = gm.densify_and_prune_with_atlas(
            max_grad=0.01,
            min_opacity=0.005,
            extent=1.0,
            max_screen_size=None,
            radii=second_radii,
            camera_center=torch.zeros((3,), dtype=torch.float32, device=device),
            explore_grad_scale=0.75,
            explore_slab_radius_mult=2.0,
            explore_jitter_scale=0.45,
            all_camera_centers=torch.zeros((1, 3), dtype=torch.float32, device=device),
            prune_enabled=True,
            min_points_to_keep=prune_controls["min_points_to_keep"],
            visibility_threshold=0.02,
            max_reattach_failures=2,
            enable_soft_prune=False,
        )
        assert prune_metrics["pruned_count"] > 0.0
        assert prune_metrics["prune_after_gc"] > 0.0
        assert int(gm.get_xyz.shape[0]) >= prune_controls["min_points_to_keep"]
        assert int(gm.get_xyz.shape[0]) >= init_points

        disabled_pose_gate, disabled_pose_metrics = should_enable_pose_refinement(
            disable_pose_refine=True,
            stable_ratio=0.8,
            drift_ratio=0.0,
            active_ratio=0.0,
            total_points=grown_points,
            init_point_count=init_points,
        )
        enabled_pose_gate, enabled_pose_metrics = should_enable_pose_refinement(
            disable_pose_refine=False,
            stable_ratio=0.5,
            drift_ratio=0.0,
            active_ratio=0.05,
            total_points=grown_points,
            init_point_count=init_points,
            refresh_done=True,
        )
        drift_blocked_gate, drift_blocked_metrics = should_enable_pose_refinement(
            disable_pose_refine=False,
            stable_ratio=0.5,
            drift_ratio=0.05,
            active_ratio=0.05,
            total_points=grown_points,
            init_point_count=init_points,
            refresh_done=True,
        )
        refresh_blocked_gate, refresh_blocked_metrics = should_enable_pose_refinement(
            disable_pose_refine=False,
            stable_ratio=0.8,
            drift_ratio=0.0,
            active_ratio=0.05,
            total_points=grown_points,
            init_point_count=init_points,
            refresh_done=False,
        )
        active_blocked_gate, active_blocked_metrics = should_enable_pose_refinement(
            disable_pose_refine=False,
            stable_ratio=0.8,
            drift_ratio=0.0,
            active_ratio=0.2,
            total_points=grown_points,
            init_point_count=init_points,
            refresh_done=True,
            active_ratio_threshold=0.12,
        )
        b2_blocked_gate, b2_blocked_metrics = should_enable_pose_photometric_refinement(
            disable_pose_refine=False,
            b1_enabled=True,
            drift_ratio=0.0,
            b1_success_streak=2,
            quality_regressed=False,
            min_b1_success_streak=3,
            drift_ratio_threshold=0.01,
        )
        b2_enabled_gate, b2_enabled_metrics = should_enable_pose_photometric_refinement(
            disable_pose_refine=False,
            b1_enabled=True,
            drift_ratio=0.0,
            b1_success_streak=4,
            quality_regressed=False,
            min_b1_success_streak=3,
            drift_ratio_threshold=0.01,
        )
        freeze_gate, freeze_metrics = should_freeze_pose_refinement(
            pose_active=True,
            drift_ratio=0.03,
            active_ratio=0.05,
            quality_bad_streak=0,
            freeze_cooldown=0,
            drift_ratio_threshold=0.02,
            active_ratio_threshold=0.15,
            max_quality_bad_streak=3,
        )
        quality_freeze_gate, quality_freeze_metrics = should_freeze_pose_refinement(
            pose_active=True,
            drift_ratio=0.0,
            active_ratio=0.05,
            quality_bad_streak=3,
            freeze_cooldown=0,
            drift_ratio_threshold=0.02,
            active_ratio_threshold=0.15,
            max_quality_bad_streak=3,
        )
        assert not disabled_pose_gate
        assert enabled_pose_gate
        assert not drift_blocked_gate
        assert not refresh_blocked_gate
        assert not active_blocked_gate
        assert active_blocked_metrics["pose_gate_active_ratio"] == 0.2
        assert active_blocked_metrics["pose_gate_active_threshold"] == 0.12
        assert not b2_blocked_gate
        assert b2_enabled_gate
        assert freeze_gate
        assert quality_freeze_gate
        assert disabled_pose_metrics["pose_gate_disabled"] == 1.0
        assert enabled_pose_metrics["pose_gate_enabled"] == 1.0
        assert drift_blocked_metrics["pose_gate_enabled"] == 0.0
        assert refresh_blocked_metrics["pose_gate_refresh_done"] == 0.0
        assert b2_blocked_metrics["pose_b2_gate_enabled"] == 0.0
        assert b2_enabled_metrics["pose_b2_gate_enabled"] == 1.0
        assert freeze_metrics["pose_freeze_drift_spike"] == 1.0
        assert quality_freeze_metrics["pose_freeze_quality_regression"] == 1.0

        print(json.dumps({
            "early_prune_controls": early_prune_controls,
            "densify_metrics": densify_metrics,
            "prune_controls": prune_controls,
            "prune_metrics": prune_metrics,
            "disabled_pose_metrics": disabled_pose_metrics,
            "enabled_pose_metrics": enabled_pose_metrics,
            "drift_blocked_metrics": drift_blocked_metrics,
            "refresh_blocked_metrics": refresh_blocked_metrics,
            "b2_blocked_metrics": b2_blocked_metrics,
            "b2_enabled_metrics": b2_enabled_metrics,
            "freeze_metrics": freeze_metrics,
            "quality_freeze_metrics": quality_freeze_metrics,
        }, indent=2))
        print("[OK] Atlas capacity controls check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
