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
    GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
    load_foundation_atlas,
)
from scene.foundation_atlas_exploration import compute_point_slab_bounds  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


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
        iterations=1000,
    )


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_runtime"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm.training_setup(build_training_args())

        camera_centers = torch.tensor(
            [[0.0, 0.0, -1.0], [1.5, 0.0, -1.0]],
            dtype=torch.float32,
            device=gm.get_xyz.device,
        )
        with torch.no_grad():
            gm._atlas_state[:] = torch.tensor(
                [GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE, GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_ACTIVE],
                dtype=torch.long,
                device=gm.get_xyz.device,
            )
            gm._atlas_visibility_ema[:] = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32, device=gm.get_xyz.device)
            gm._center_log_sigma_parallel[:] = torch.log(torch.full_like(gm._center_log_sigma_parallel, 1.0))
            gm._center_log_sigma_support[:] = torch.log(torch.full_like(gm._center_log_sigma_support, 0.5))
        guardrail_metrics = gm.apply_uncertainty_guardrails(
            camera_centers=camera_centers,
            fallback_camera_center=camera_centers[0],
            slab_radius_mult=2.0,
            ray_cap_fraction=1.0,
            parallel_min_ratio=0.03,
            parallel_max_ratio=0.45,
            support_min_ratio=0.01,
            support_max_ratio=0.20,
            decay=0.9,
            low_visibility_threshold=0.05,
        )
        assert guardrail_metrics["sigma_parallel_clamp_hits"] > 0.0
        assert guardrail_metrics["sigma_support_clamp_hits"] > 0.0
        assert guardrail_metrics["sigma_ray_clamp_hits"] > 0.0
        assert guardrail_metrics["sigma_active_ray_valid_count"] > 0.0
        state = gm.get_atlas_state.detach()
        sigma_parallel = gm.get_center_sigma_parallel.detach().squeeze(-1)
        sigma_support = gm.get_center_sigma_support.detach().squeeze(-1)
        radius = gm.get_gaussian_atlas_radius.detach()
        stable_mask = state == GAUSSIAN_STATE_STABLE
        passive_mask = state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
        active_mask = state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        assert float(sigma_parallel[stable_mask].max().item()) <= float((radius[stable_mask] * 0.45).max().item()) + 1e-5
        assert float(sigma_parallel[passive_mask].max().item()) <= float((radius[passive_mask] * 0.45 * 1.35).max().item()) + 1e-5
        assert float(sigma_parallel[active_mask].max().item()) <= float((radius[active_mask] * 0.45 * 2.25).max().item()) + 1e-5
        assert float(sigma_parallel[active_mask].min().item()) >= float((radius[active_mask] * 0.03 * 1.50).min().item()) - 1e-5
        assert float(sigma_support[stable_mask].max().item()) <= float((radius[stable_mask] * 0.20).max().item()) + 1e-5
        assert float(sigma_support[passive_mask].max().item()) <= float((radius[passive_mask] * 0.20 * 1.25).max().item()) + 1e-5
        assert float(sigma_support[active_mask].max().item()) <= float((radius[active_mask] * 0.20 * 1.50).max().item()) + 1e-5
        with torch.no_grad():
            gm._atlas_state[:] = torch.tensor(
                [GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE, GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE],
                dtype=torch.long,
                device=gm.get_xyz.device,
            )

        residuals = torch.tensor([0.01, 0.5, 0.02, 0.35], dtype=torch.float32, device=gm.get_xyz.device)
        visible = torch.tensor([1, 1, 1, 1], dtype=torch.bool, device=gm.get_xyz.device)
        runtime_metrics = gm.update_atlas_runtime_stats(
            residuals,
            visible,
            ema_decay=0.0,
            drift_radius_mult=1.75,
            camera_index=1,
            high_residual_threshold=0.03,
            warmup_only=True,
        )
        assert runtime_metrics["observed_node_ratio"] > 0.0
        assert runtime_metrics["observed_node_count"] > 0.0
        assert runtime_metrics["mean_node_photo_ema"] >= 0.0
        assert runtime_metrics["mean_node_visibility_ema"] >= 0.0
        assert runtime_metrics["mean_node_observed_count"] > 0.0
        assert runtime_metrics["mean_node_support_consistency"] >= 0.0
        assert runtime_metrics["ref_camera_ratio"] > 0.0
        pre_refresh = gm.update_atlas_states(
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
        )
        assert gm.get_atlas_state[0].item() == GAUSSIAN_STATE_STABLE
        assert gm.get_atlas_state[2].item() == GAUSSIAN_STATE_STABLE

        reliability_base_before_refresh = gm._atlas_reliability_base.detach().clone()
        reliability_runtime_before_refresh = gm._atlas_reliability_runtime.detach().clone()
        with torch.no_grad():
            gm._atlas_node_photo_ema.copy_(torch.tensor([0.01, 0.40, 0.20, 0.75], dtype=torch.float32, device=gm.get_xyz.device))
            gm._atlas_node_visibility_ema.copy_(torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=gm.get_xyz.device))
            gm._atlas_node_observed_count.copy_(torch.tensor([3.0, 2.0, 0.0, 0.0], dtype=torch.float32, device=gm.get_xyz.device))
            gm._atlas_node_support_consistent_count.copy_(torch.tensor([3.0, 1.0, 0.0, 0.0], dtype=torch.float32, device=gm.get_xyz.device))
            gm._atlas_view_weights.zero_()
            gm._atlas_view_counts.zero_()
            gm._atlas_view_weights[:, :2] = torch.tensor(
                [[2.7, 0.2], [0.8, 0.7], [0.0, 0.0], [0.0, 0.0]],
                dtype=torch.float32,
                device=gm.get_xyz.device,
            )
            gm._atlas_view_counts[:, :2] = torch.tensor(
                [[3, 1], [1, 1], [0, 0], [0, 0]],
                dtype=torch.int32,
                device=gm.get_xyz.device,
            )
            gm._atlas_ref_camera[1] = -1
            gm._atlas_ref_score[1] = 0.0
        refresh = gm.refresh_atlas_after_warmup(alpha=1.0, gamma=4.0, min_reliability=0.05, min_visibility=0.5)
        assert refresh["observed_node_ratio"] > 0.0
        assert refresh["atlas_refresh_done"] == 1.0
        assert refresh["refresh_applied"] == 1.0
        assert "atlas_reliability_base_mean" in refresh
        assert "atlas_reliability_runtime_mean" in refresh
        assert "atlas_reliability_runtime_min" in refresh
        assert "atlas_reliability_runtime_max" in refresh
        assert refresh["atlas_refresh_snapshot_ready"] == 1.0
        assert gm._atlas_refresh_done
        assert torch.allclose(gm._atlas_reliability_base, reliability_base_before_refresh)
        assert not torch.allclose(gm._atlas_reliability_runtime, reliability_runtime_before_refresh)
        expected_candidate = torch.clamp(
            gm._atlas_raw_score.clamp_min(1e-6) * torch.exp(-4.0 * gm._atlas_node_photo_ema.clamp_min(0.0)),
            min=0.05,
            max=1.0,
        )
        observed_count = gm._atlas_node_observed_count
        support_ratio = gm._atlas_node_support_consistency_ema.clamp(0.0, 1.0)
        coverage_ratio = (gm._atlas_view_counts > 0).float().mean(dim=1)
        view_weights = gm._atlas_view_weights.clamp_min(0.0)
        top_values, top_indices = torch.topk(view_weights, k=2, dim=1)
        top1 = top_values[:, 0]
        top2 = top_values[:, 1]
        ambiguity = torch.where(
            top1 > 1e-6,
            (top2 / top1.clamp_min(1e-6)).clamp(0.0, 1.0),
            torch.zeros_like(top1),
        )
        expected_obs_quality = gm._atlas_node_obs_quality_ema.clamp(0.0, 1.0)
        count_strength = (observed_count / (observed_count + 0.5)).clamp(0.0, 1.0)
        observed_score = gm._atlas_node_observed_score_ema.clamp(0.0, 1.0)
        updated_recently = gm._atlas_node_updated_recently.clamp(0.0, 1.0)
        observed_mask = (
            ((observed_score >= 0.08) | (count_strength >= 0.22))
            & ((gm._atlas_node_visibility_ema >= 0.5 * 0.25) | (updated_recently >= 0.05))
            & ((expected_obs_quality >= 0.05) | (count_strength >= 0.22))
        )
        base_clamped = reliability_base_before_refresh.clamp(min=0.05, max=1.0)
        override_mask = gm._atlas_refresh_runtime_override_mask
        assert torch.allclose(gm._atlas_reliability_runtime, gm._atlas_reliability_effective)
        assert torch.isfinite(gm._atlas_reliability_runtime_raw).all()
        assert torch.isfinite(gm._atlas_reliability_runtime_mapped).all()
        assert torch.isfinite(gm._atlas_reliability_effective).all()
        assert torch.all(gm._atlas_reliability_runtime_mapped >= 0.05)
        assert torch.all(gm._atlas_reliability_runtime_mapped <= 1.0)
        assert torch.allclose(gm._atlas_reliability_effective[~override_mask], base_clamped[~override_mask])
        assert torch.all(gm._atlas_refresh_override_weight[override_mask] >= 0.60)
        assert torch.all(gm._atlas_refresh_override_weight[~override_mask] == 0.0)
        assert torch.all(override_mask <= observed_mask)
        assert torch.equal(gm._atlas_refresh_observed_mask, observed_mask)
        assert torch.allclose(gm._atlas_refresh_node_photo_ema, gm._atlas_node_photo_ema)
        assert torch.allclose(gm._atlas_refresh_node_visibility_ema, gm._atlas_node_visibility_ema)
        assert torch.allclose(gm._atlas_refresh_obs_quality, expected_obs_quality)
        assert torch.allclose(gm._atlas_refresh_node_observed_count, gm._atlas_node_observed_count)
        assert torch.allclose(gm._atlas_refresh_node_support_consistent_ratio, support_ratio)
        assert torch.allclose(gm._atlas_refresh_node_coverage_ratio, coverage_ratio)
        assert torch.allclose(gm._atlas_refresh_node_ambiguity, ambiguity)
        assert refresh["atlas_refresh_snapshot_observed_count"] == float(observed_mask.sum().item())
        assert refresh["atlas_refresh_snapshot_observed_ratio"] == float(observed_mask.float().mean().item())
        assert refresh["atlas_refresh_snapshot_runtime_override_count"] >= 1.0
        assert refresh["atlas_refresh_snapshot_keep_base_count"] >= 1.0
        assert refresh["atlas_refresh_snapshot_node_observed_count_mean"] > 0.0
        assert refresh["atlas_refresh_snapshot_coverage_hist_high"] > 0.0
        assert refresh["atlas_reliability_runtime_mean"] != refresh["atlas_reliability_base_mean"]
        assert gm._atlas_ref_camera[1].item() == int(top_indices[1, 0].item())
        assert gm._atlas_ref_score[1].item() > 0.0
        refresh_again = gm.refresh_atlas_after_warmup(alpha=1.0, gamma=4.0, min_reliability=0.05, min_visibility=0.5)
        assert refresh_again["refresh_applied"] == 0.0
        assert refresh_again["atlas_refresh_snapshot_ready"] == 1.0
        assert torch.allclose(gm._atlas_reliability_runtime, gm._atlas_reliability_effective)

        state_metrics = gm.update_atlas_states(
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
        )
        assert gm.get_atlas_state[0].item() == GAUSSIAN_STATE_STABLE
        assert gm.get_atlas_state[2].item() == GAUSSIAN_STATE_STABLE
        assert gm.get_atlas_state[3].item() == GAUSSIAN_STATE_UNSTABLE_PASSIVE
        reliability_runtime_after_refresh = gm._atlas_reliability_runtime.detach().clone()
        gm.update_atlas_runtime_stats(
            residuals * 1.5,
            visible,
            ema_decay=0.0,
            drift_radius_mult=1.75,
            high_residual_threshold=0.03,
        )
        assert torch.allclose(gm._atlas_reliability_runtime, reliability_runtime_after_refresh)
        gm.update_atlas_runtime_stats(
            residuals,
            visible,
            ema_decay=0.0,
            drift_radius_mult=1.75,
            high_residual_threshold=0.03,
        )
        with torch.no_grad():
            radius = gm.get_gaussian_atlas_radius[3].clamp_min(1e-6)
            gm._xyz[3] = gm.get_gaussian_atlas_positions[3] + torch.tensor(
                [0.0, 0.0, 0.6],
                dtype=torch.float32,
                device=gm.get_xyz.device,
            ) * radius
            gm._atlas_photo_ema[3] = 0.35
            gm._atlas_visibility_ema[3] = 0.8
            gm._atlas_high_residual_count[3] = 3
            gm._atlas_ref_camera[3] = 0
            gm._atlas_ref_score[3] = 0.9
        state_metrics = gm.update_atlas_states(
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
            active_min_lifetime_iters=3,
            active_quota_min=1,
            active_quota_max=1,
        )
        active_after_update = torch.nonzero(gm.get_atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE, as_tuple=False).squeeze(-1)
        assert active_after_update.numel() >= 1
        promoted_idx = int(active_after_update[0].item())
        assert state_metrics["explore_score_max"] > 0.0
        assert state_metrics["explore_candidate_ratio"] > 0.0
        assert state_metrics["active_candidate_pool_count"] >= 1
        assert state_metrics["active_quota_target"] >= 1
        assert state_metrics["sustained_high_residual_ratio"] > 0.0
        assert state_metrics["transition_passive_to_active_count"] >= 1
        assert gm.get_atlas_state_cooldown[promoted_idx].item() >= 3
        assert state_metrics["mean_active_lifetime"] >= 1.0
        state_summary = gm.summarize_atlas_state_metrics()
        assert state_summary["state_stable_count"] >= 1
        assert state_summary["state_passive_count"] >= 0
        assert state_summary["state_active_count"] >= 1
        assert state_summary["state_out_pending_count"] >= 0
        assert state_summary["cooldown_ratio"] >= 0.0
        assert state_summary["mean_gc_fail_count"] >= 0.0
        assert state_summary["mean_active_lifetime"] >= 1.0
        assert state_summary["max_active_lifetime"] >= state_summary["mean_active_lifetime"]

        with torch.no_grad():
            gm._xyz[0] = gm._atlas_positions[1] + torch.tensor([0.0, 0.0, 0.0], device=gm.get_xyz.device)
            gm._atlas_photo_ema[0] = 0.6
            gm._atlas_visibility_ema[0] = 0.8
            gm._atlas_high_residual_count[0] = 4
            gm._atlas_low_residual_count[0] = 2
            gm._atlas_ref_camera[0] = 0
            gm._atlas_ref_score[0] = 0.9
        gm.update_atlas_runtime_stats(
            residuals,
            visible,
            ema_decay=0.0,
            drift_radius_mult=1.0,
            high_residual_threshold=0.03,
        )
        gm.update_atlas_states(
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
        )
        assert gm.get_atlas_state[0].item() != GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING

        gc_summary = gm.run_atlas_gc(
            reattach_radius_mult=2.5,
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
            max_reattach_failures=2,
            forced_prune_opacity=0.01,
        )
        assert gc_summary["mode"] == "voxel_hash"
        assert gc_summary["gc_candidates"] >= 1
        assert gc_summary["gc_drift_candidates"] >= 1
        assert gc_summary["gc_pending_candidates"] == 0
        assert gc_summary["gc_retry_pending_enabled"] == 1.0
        assert gc_summary["hash_bucket_count"] > 0
        assert gc_summary["hash_source"] == "atlas_hash.json"
        assert gc_summary["bucket_queries"] >= 1
        assert gc_summary["fallback_full_search"] == 0
        assert gc_summary["gc_ran"] == 1.0
        assert gc_summary["reattach_success"] >= 1
        assert gc_summary["reattach_fail"] >= 0
        assert gc_summary["reattach_success_ratio"] > 0.0
        assert gc_summary["reattach_tier1_attempt_count"] >= gc_summary["reattach_tier1_success"]
        assert gc_summary["reattach_tier1_raw_accept_count"] >= gc_summary["reattach_tier1_success"]
        assert gc_summary["reattach_candidate_starvation_count"] >= 0
        assert gc_summary["reattach_candidate_starvation_ratio"] >= 0.0
        assert gc_summary["pending_reattach_success"] == 0
        assert gc_summary["pending_reattach_fail"] == 0
        assert gc_summary["forced_pending"] == 0
        assert gc_summary["out_of_anchor_pending_count"] >= 0
        assert gc_summary["mean_gc_fail_count_after"] >= 0.0
        assert gm.get_atlas_node_ids[0].item() == 1
        assert not bool(gm.get_atlas_drift_flag[0].item())
        assert gm.get_atlas_high_residual_count[0].item() == 0
        assert gm.get_atlas_low_residual_count[0].item() == 0
        assert abs(float(gm.get_atlas_photo_ema[0].item())) < 1e-6
        assert abs(float(gm.get_atlas_visibility_ema[0].item())) < 1e-6
        assert gm.get_atlas_ref_camera[0].item() >= -1
        assert float(gm.get_atlas_ref_score[0].item()) >= 0.0
        assert gm.get_atlas_state[0].item() in (
            GAUSSIAN_STATE_UNSTABLE_PASSIVE,
            GAUSSIAN_STATE_STABLE,
            GAUSSIAN_STATE_UNSTABLE_ACTIVE,
        )

        with torch.no_grad():
            gm._xyz[0] = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float32, device=gm.get_xyz.device)
        gm.update_atlas_runtime_stats(
            residuals,
            visible,
            ema_decay=0.0,
            drift_radius_mult=1.0,
            high_residual_threshold=0.03,
        )
        gm.update_atlas_states(
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
        )
        forced_prune_summary = None
        for _ in range(3):
            forced_prune_summary = gm.run_atlas_gc(
                reattach_radius_mult=0.5,
                surface_stable_min=0.12,
                edge_stable_min=0.08,
                min_visibility_ema=0.1,
                stable_residual_threshold=0.03,
                activate_threshold=0.02,
                deactivate_threshold=0.01,
                activate_min_high_residual_iters=2,
                max_reattach_failures=2,
                forced_prune_opacity=0.01,
            )
        assert forced_prune_summary is not None
        assert forced_prune_summary["forced_pending"] >= 1
        assert gm.get_atlas_gc_fail_count[0].item() >= 3
        assert float(gm.get_opacity[0].item()) >= 0.0
        assert gm.get_atlas_state[0].item() == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING

        with torch.no_grad():
            gm._atlas_drift_flag[0] = False
        gc_without_pending_retry = gm.run_atlas_gc(
            reattach_radius_mult=0.5,
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
            max_reattach_failures=2,
            forced_prune_opacity=0.01,
            retry_pending=False,
        )
        assert gc_without_pending_retry["gc_retry_pending_enabled"] == 0.0
        assert gc_without_pending_retry["gc_candidates"] == 0
        assert gc_without_pending_retry["gc_pending_candidates"] == 0
        assert gc_without_pending_retry["pending_reattach_success"] == 0
        assert gc_without_pending_retry["pending_reattach_fail"] == 0

        pending_retry_summary = gm.run_atlas_gc(
            reattach_radius_mult=0.5,
            surface_stable_min=0.12,
            edge_stable_min=0.08,
            min_visibility_ema=0.1,
            stable_residual_threshold=0.03,
            activate_threshold=0.02,
            deactivate_threshold=0.01,
            activate_min_high_residual_iters=2,
            max_reattach_failures=2,
            forced_prune_opacity=0.01,
            retry_pending=True,
        )
        assert pending_retry_summary["gc_pending_candidates"] >= 1
        assert pending_retry_summary["pending_reattach_fail"] >= 1
        assert pending_retry_summary["pending_reattach_success"] == 0
        assert pending_retry_summary["pending_reattach_fail_ratio"] > 0.0
        assert pending_retry_summary["ray_guided_priority_queries"] >= 1
        assert pending_retry_summary["ray_guided_pending_queries"] >= 1
        assert pending_retry_summary["reattach_tier3_attempt_count"] >= pending_retry_summary["ray_guided_priority_queries"]

        with torch.no_grad():
            gm._atlas_state[:] = torch.tensor(
                [GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE, GAUSSIAN_STATE_UNSTABLE_PASSIVE, GAUSSIAN_STATE_UNSTABLE_ACTIVE],
                dtype=torch.long,
                device=gm.get_xyz.device,
            )
            gm._atlas_ref_camera[:] = 0
            gm._atlas_ref_score[:] = 0.9
            gm.xyz_gradient_accum = torch.tensor([[0.2], [0.0], [0.0], [0.25]], dtype=torch.float32, device=gm.get_xyz.device)
            gm.denom = torch.ones_like(gm.xyz_gradient_accum)

        start_count = int(gm.get_xyz.shape[0])
        densify_camera_centers = torch.tensor(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=gm.get_xyz.device,
        )
        densify_metrics = gm.densify_and_prune_with_atlas(
            max_grad=0.1,
            min_opacity=0.0,
            extent=1.0,
            max_screen_size=None,
            radii=torch.ones((start_count,), dtype=torch.float32, device=gm.get_xyz.device),
            camera_center=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=gm.get_xyz.device),
            explore_grad_scale=1.0,
            explore_slab_radius_mult=2.0,
            explore_jitter_scale=0.25,
            active_min_lifetime_iters=3,
            all_camera_centers=densify_camera_centers,
        )
        end_count = int(gm.get_xyz.shape[0])
        assert end_count >= start_count + 1
        assert densify_metrics["explore_clone_count"] >= 1.0
        assert densify_metrics["explore_valid_ref_count"] >= densify_metrics["explore_clone_count"]
        new_states = gm.get_atlas_state[start_count:].detach().cpu().tolist()
        assert GAUSSIAN_STATE_UNSTABLE_ACTIVE in new_states
        if densify_metrics.get("stable_clone_count", 0.0) + densify_metrics.get("stable_split_count", 0.0) > 0.0:
            assert GAUSSIAN_STATE_STABLE in new_states
        new_active_local = torch.nonzero(gm.get_atlas_state[start_count:] == GAUSSIAN_STATE_UNSTABLE_ACTIVE, as_tuple=False).squeeze(-1)
        assert new_active_local.numel() > 0
        new_active_idx = new_active_local + start_count
        assert torch.all(gm.get_atlas_state_cooldown[new_active_idx] >= 3)
        assert torch.all(gm.get_atlas_active_lifetime[new_active_idx] >= 1)
        new_active_slab = compute_point_slab_bounds(
            gm,
            new_active_idx,
            camera_centers=densify_camera_centers,
            slab_radius_mult=2.0,
            detach_points=True,
            require_valid_ref_camera=True,
            min_reference_score=0.05,
        )
        assert new_active_slab is not None
        assert torch.all(new_active_slab["tau"] >= new_active_slab["tau_min"] - 1e-6)
        assert torch.all(new_active_slab["tau"] <= new_active_slab["tau_max"] + 1e-6)

        print(json.dumps({"guardrail_metrics": guardrail_metrics, "pre_refresh": pre_refresh, "state_metrics": state_metrics, "gc_summary": gc_summary, "forced_prune_summary": forced_prune_summary, "start_count": start_count, "end_count": end_count}, indent=2))
        print("[OK] Atlas runtime state check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
