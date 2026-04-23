import json
import shutil
import sys
import types
from argparse import ArgumentParser
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

from scene.foundation_atlas import load_foundation_atlas  # noqa: E402
from scene.foundation_atlas_pose import measure_pose_delta, reset_camera_pose_delta  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402
import train as train_module  # noqa: E402
from arguments import OptimizationParams  # noqa: E402
from train import _compute_atlas_loss_schedule, _compute_atlas_phase_controls  # noqa: E402


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


class DummyPoseCamera:
    def __init__(self, device: torch.device):
        self.pose_delta_q = torch.nn.Parameter(torch.tensor([0.9659258, 0.2588190, 0.0, 0.0], dtype=torch.float32, device=device))
        self.pose_delta_t = torch.nn.Parameter(torch.tensor([0.15, -0.05, 0.02], dtype=torch.float32, device=device))
        self.refresh_count = 0

    def refresh_pose_matrices(self):
        self.refresh_count += 1


def main():
    parser = ArgumentParser()
    opt_group = OptimizationParams(parser)
    exposed_opt = opt_group.extract(parser.parse_args([]))
    for attr_name in (
        "atlas_mean_warmup_scale",
        "atlas_kl_warmup_scale",
        "atlas_shape_main_phase_ramp_iters",
        "atlas_reg_ramp_capacity_ratio",
        "atlas_shape_post_refresh_floor",
        "atlas_shape_floor_min_reliability",
        "atlas_shape_floor_full_reliability",
        "atlas_ori_ramp_stable_ratio",
        "atlas_aniso_ramp_stable_ratio",
        "atlas_reg_passive_state_weight",
        "atlas_reg_active_state_weight",
        "atlas_kl_passive_state_weight",
        "atlas_kl_active_state_weight",
        "atlas_state_out_of_anchor_gc_failures",
        "atlas_gc_retry_pending",
        "pose_b2_min_b1_updates",
        "pose_b2_max_quality_regression",
        "pose_b2_max_drift_ratio",
        "pose_quality_ema_decay",
        "pose_freeze_max_drift_ratio",
        "pose_freeze_max_active_ratio",
        "pose_freeze_bad_loss_iters",
        "pose_freeze_quality_regression",
        "pose_freeze_cooldown_iters",
    ):
        assert hasattr(exposed_opt, attr_name)

    controls_opt = SimpleNamespace(
        atlas_reg_warmup_steps=10,
        atlas_shape_main_phase_ramp_iters=10,
        pose_refine_after_warmup=True,
        atlas_mc_pairs=2,
        atlas_gc_interval=5,
        atlas_slab_weight=0.01,
        atlas_obs_lambda=0.001,
        atlas_obs_max_cameras=16,
        atlas_obs_point_chunk=128,
        pose_enable_stable_ratio=0.45,
        pose_enable_max_drift_ratio=0.01,
        pose_enable_min_capacity_ratio=1.25,
        pose_b2_min_b1_updates=3,
        pose_b2_max_quality_regression=0.01,
        pose_b2_max_drift_ratio=0.01,
        pose_quality_ema_decay=0.0,
        pose_freeze_max_drift_ratio=0.02,
        pose_freeze_max_active_ratio=0.15,
        pose_freeze_bad_loss_iters=3,
        pose_freeze_quality_regression=0.03,
        pose_freeze_cooldown_iters=50,
    )
    warmup_controls = _compute_atlas_phase_controls(5, controls_opt, True)
    assert warmup_controls["in_warmup"]
    assert warmup_controls["warmup_only"]
    assert not warmup_controls["main_phase"]
    assert not warmup_controls["enable_pose_b1"]
    assert not warmup_controls["enable_pose_b2"]
    assert not warmup_controls["enable_densify"]
    assert not warmup_controls["enable_prune"]
    assert not warmup_controls["enable_gc"]
    assert not warmup_controls["enable_state_update"]
    assert not warmup_controls["enable_mc"]
    assert not warmup_controls["enable_explore"]

    main_controls = _compute_atlas_phase_controls(11, controls_opt, True)
    assert not main_controls["in_warmup"]
    assert not main_controls["warmup_only"]
    assert main_controls["main_phase"]
    assert main_controls["enable_pose_b1"]
    assert main_controls["enable_pose_b2"]
    assert main_controls["enable_densify"]
    assert main_controls["enable_prune"]
    assert main_controls["enable_gc"]
    assert main_controls["enable_state_update"]
    assert main_controls["enable_mc"]
    assert main_controls["enable_explore"]

    no_atlas_controls = _compute_atlas_phase_controls(5, controls_opt, False)
    assert not no_atlas_controls["in_warmup"]
    assert no_atlas_controls["enable_densify"]
    assert no_atlas_controls["enable_prune"]

    pose_runtime_state = train_module._init_pose_runtime_state()
    pose_runtime_state["b1_camera_attempt_count"] = 2
    pose_runtime_state["b1_camera_execute_count"] = 1
    pose_runtime_state["b2_camera_attempt_count"] = 1
    pose_runtime_state["b2_camera_execute_count"] = 1
    pose_runtime_state["b1_skip_hist"] = {"trust_region_block": 3}
    pose_runtime_state["b2_skip_hist"] = {"mask_empty": 2}
    pose_runtime_state["delta_t_sum"] = 0.6
    pose_runtime_state["delta_angle_sum"] = 12.0
    pose_runtime_state["delta_samples"] = 3
    initial_quality_metrics = train_module._update_pose_quality_state(pose_runtime_state, 0.8, controls_opt)
    assert initial_quality_metrics["pose_gate_quality_ema"] == 0.8
    assert initial_quality_metrics["pose_gate_quality_bad_streak"] == 0.0
    pose_runtime_log_fields = train_module._build_pose_runtime_log_fields(pose_runtime_state)
    assert pose_runtime_log_fields["pose_b1_camera_attempt_count"] == 2
    assert pose_runtime_log_fields["pose_b2_camera_execute_count"] == 1
    assert abs(pose_runtime_log_fields["pose_delta_t_mean"] - 0.2) < 1e-8
    assert abs(pose_runtime_log_fields["pose_delta_angle_mean"] - 4.0) < 1e-8
    assert pose_runtime_log_fields["pose_b1_skip_hist_trust_region_block"] == 3
    assert pose_runtime_log_fields["pose_b2_skip_hist_mask_empty"] == 2

    blocked_b1, blocked_b2, blocked_pose_metrics, blocked_freeze = train_module._compute_pose_refine_controls(
        refresh_done=False,
        state_metrics={"stable_ratio": 0.8, "drift_ratio": 0.0, "active_ratio": 0.0},
        total_points=13,
        init_points=10,
        opt=controls_opt,
        disable_pose_refine=False,
        pose_runtime_state=pose_runtime_state,
    )
    assert not blocked_b1
    assert not blocked_b2
    assert not blocked_freeze
    assert blocked_pose_metrics["pose_gate_refresh_done"] == 0.0

    pose_runtime_state["b1_success_streak"] = 4
    pose_runtime_state["b1_update_count"] = 4
    ready_b1, ready_b2, ready_pose_metrics, ready_freeze = train_module._compute_pose_refine_controls(
        refresh_done=True,
        state_metrics={"stable_ratio": 0.6, "drift_ratio": 0.0, "active_ratio": 0.0},
        total_points=13,
        init_points=10,
        opt=controls_opt,
        disable_pose_refine=False,
        pose_runtime_state=pose_runtime_state,
    )
    assert ready_b1
    assert ready_b2
    assert not ready_freeze
    assert ready_pose_metrics["pose_gate_b1_enabled"] == 1.0
    assert ready_pose_metrics["pose_gate_b2_enabled"] == 1.0

    for _ in range(3):
        train_module._update_pose_quality_state(pose_runtime_state, 1.0, controls_opt)
    frozen_b1, frozen_b2, frozen_pose_metrics, frozen_triggered = train_module._compute_pose_refine_controls(
        refresh_done=True,
        state_metrics={"stable_ratio": 0.6, "drift_ratio": 0.0, "active_ratio": 0.0},
        total_points=13,
        init_points=10,
        opt=controls_opt,
        disable_pose_refine=False,
        pose_runtime_state=pose_runtime_state,
    )
    assert not frozen_b1
    assert not frozen_b2
    assert frozen_triggered
    assert frozen_pose_metrics["pose_freeze_emergency_stop"] == 1.0
    assert frozen_pose_metrics["pose_freeze_quality_regression"] == 1.0
    assert frozen_pose_metrics["pose_gate_freeze_events"] == 1.0
    assert pose_runtime_state["freeze_cooldown"] == controls_opt.pose_freeze_cooldown_iters
    assert pose_runtime_state["b1_success_streak"] == 0

    required_log_fields = train_module._build_required_training_log_fields(
        has_atlas_bindings=True,
        atlas_phase={
            "enable_pose_b1": True,
            "enable_pose_b2": False,
            "warmup_only": True,
            "main_phase": False,
            "refresh_pending": True,
            "main_phase_ready": False,
            "pose_refine_disabled_or_blocked_by_phase": True,
        },
        atlas_reliability_summary={
            "atlas_refresh_done": 1.0,
            "atlas_reliability_base_mean": 0.4,
            "atlas_reliability_runtime_mean": 0.5,
            "atlas_reliability_runtime_min": 0.2,
            "atlas_reliability_runtime_max": 0.8,
            "atlas_reliability_base_p10": 0.11,
            "atlas_reliability_base_p50": 0.44,
            "atlas_reliability_base_p90": 0.77,
            "atlas_reliability_base_hist_low": 0.2,
            "atlas_reliability_base_hist_mid": 0.5,
            "atlas_reliability_base_hist_high": 0.3,
            "atlas_reliability_runtime_mapped_p10": 0.12,
            "atlas_reliability_runtime_mapped_p50": 0.52,
            "atlas_reliability_runtime_mapped_p90": 0.82,
            "atlas_reliability_runtime_mapped_hist_low": 0.1,
            "atlas_reliability_runtime_mapped_hist_mid": 0.4,
            "atlas_reliability_runtime_mapped_hist_high": 0.5,
            "atlas_reliability_effective_p10": 0.13,
            "atlas_reliability_effective_p50": 0.55,
            "atlas_reliability_effective_p90": 0.85,
            "atlas_reliability_effective_hist_low": 0.1,
            "atlas_reliability_effective_hist_mid": 0.3,
            "atlas_reliability_effective_hist_high": 0.6,
            "refresh_evidence_observed_gate_ratio": 0.7,
            "refresh_evidence_count_gate_ratio": 0.65,
            "refresh_evidence_visibility_gate_ratio": 0.6,
            "refresh_evidence_ref_gate_ratio": 0.55,
            "refresh_evidence_finite_gate_ratio": 0.8,
            "refresh_evidence_support_gate_ratio": 0.75,
            "refresh_evidence_override_gate_ratio": 0.45,
            "refresh_evidence_gate_mean": 0.5,
            "refresh_override_weight_positive_ratio": 0.45,
            "refresh_override_base_bucket_low_override_ratio": 0.2,
            "refresh_override_base_bucket_mid_override_ratio": 0.4,
            "refresh_override_base_bucket_high_override_ratio": 0.6,
            "atlas_refresh_snapshot_ready": 1.0,
            "atlas_refresh_snapshot_observed_ratio": 0.5,
            "atlas_refresh_snapshot_observed_count": 2.0,
            "atlas_refresh_snapshot_photo_ema_mean": 0.25,
            "atlas_refresh_snapshot_visibility_ema_mean": 0.75,
            "atlas_refresh_snapshot_obs_quality_mean": 0.35,
            "atlas_refresh_snapshot_obs_quality_max": 0.8,
        },
        atlas_state_metrics={
            "stable_ratio": 0.6,
            "passive_ratio": 0.15,
            "active_ratio": 0.1,
            "out_of_anchor_ratio": 0.05,
            "pending_ratio": 0.05,
            "state_stable_count": 12,
            "state_passive_count": 3,
            "state_active_count": 2,
            "state_out_pending_count": 1,
            "out_of_anchor_pending_count": 1,
            "cooldown_ratio": 0.2,
            "mean_gc_fail_count": 0.4,
            "mean_active_lifetime": 3.5,
            "max_active_lifetime": 6.0,
            "transition_stable_to_passive_count": 1,
            "transition_passive_to_stable_count": 2,
            "transition_passive_to_active_count": 3,
            "transition_active_to_passive_count": 4,
            "transition_active_to_stable_count": 5,
            "transition_any_to_pending_count": 6,
        },
        atlas_runtime_metrics={
            "gc_ran": 0.0,
            "gc_due": 1.0,
            "gc_interval": 5.0,
            "observed_node_ratio": 0.75,
            "observed_node_count": 4.0,
            "mean_node_photo_ema": 0.2,
            "mean_node_visibility_ema": 0.7,
            "pose_gate_enabled": 1.0,
            "pose_gate_b1_reason": "stable_ratio_low,capacity_ratio_low",
            "pose_gate_b2_reason": "enabled",
            "pose_gate_b2_enabled": 1.0,
            "pose_gate_b2_b1_history_fresh": 1.0,
            "pose_b2_gate_history_ready": 1.0,
            "pose_gate_b2_bootstrap_open": 0.0,
        },
        densify_metrics={
            "split_count": 1.0,
            "clone_count": 2.0,
            "explore_clone_count": 3.0,
            "stable_split_count": 1.0,
            "stable_split_candidate_count": 4.0,
            "stable_split_block_drift_count": 1.0,
            "stable_clone_count": 2.0,
            "stable_clone_candidate_count": 5.0,
            "stable_clone_block_projector_count": 2.0,
            "active_explore_clone_count": 3.0,
            "densify_stale_render_tensors": 1.0,
            "densify_skipped_stale_render_tensors": 0.0,
            "densify_used_cached_stats": 1.0,
            "densify_stale_reason": "gc_pruned_render_tensors",
            "densify_skip_reason": "none",
            "pruned_count": 4.0,
            "prune_after_gc": 5.0,
        },
        atlas_gc_metrics={
            "gc_candidates": 6.0,
            "gc_ran": 1.0,
            "reattach_success": 7.0,
            "reattach_fail": 8.0,
            "reattach_success_ratio": 7.0 / 6.0,
            "reattach_fail_ratio": 8.0 / 6.0,
            "pending_reattach_success": 2.0,
            "pending_reattach_fail": 1.0,
            "pending_reattach_success_ratio": 2.0 / 3.0,
            "pending_reattach_fail_ratio": 1.0 / 3.0,
            "reattach_tier1_attempt_count": 6.0,
            "reattach_tier1_raw_accept_count": 5.0,
            "reattach_tier1_success": 4.0,
            "reattach_tier2_attempt_count": 2.0,
            "reattach_tier2_raw_accept_count": 1.0,
            "reattach_tier2_success": 1.0,
            "reattach_tier3_attempt_count": 1.0,
            "reattach_tier3_raw_accept_count": 1.0,
            "reattach_tier3_success": 1.0,
            "reattach_tier4_attempt_count": 1.0,
            "reattach_tier4_forced_success": 0.0,
            "reattach_candidate_starvation_count": 1.0,
            "reattach_candidate_starvation_ratio": 1.0 / 6.0,
            "ray_guided_queries": 1.0,
            "ray_guided_priority_queries": 1.0,
            "ray_guided_late_queries": 0.0,
            "ray_guided_active_queries": 1.0,
            "ray_guided_quality_accept_count": 1.0,
        },
        pose_metrics={
            "b1_total_loss": 0.125,
            "b2_total_loss": 0.25,
            "pose_translation_clamped": 1.0,
            "pose_rotation_clamped": 0.0,
        },
        atlas_uncertainty_metrics={
            "sigma_parallel_clamp_hits": 9.0,
            "sigma_support_clamp_hits": 10.0,
            "sigma_ray_clamp_hits": 11.0,
            "sigma_stable_parallel_mean": 0.1,
            "sigma_passive_parallel_mean": 0.2,
            "sigma_active_parallel_mean": 0.3,
            "sigma_active_ray_span_mean": 0.4,
            "sigma_active_ray_parallel_p90": 0.5,
        },
        atlas_slab_metrics={
            "atlas_slab_total_loss": 0.33,
            "atlas_slab_active_count": 2.0,
            "atlas_slab_mean_penalty": 0.44,
            "atlas_slab_violation_count": 3.0,
            "atlas_slab_violation_ratio": 0.6,
        },
        atlas_kl_metrics={
            "atlas_rank_u_mean": 1.5,
            "atlas_active_ray_fallback_count": 4.0,
            "atlas_kl_stable_mean": 0.11,
            "atlas_kl_passive_mean": 0.22,
            "atlas_kl_active_mean": 0.33,
        },
        atlas_refresh_done=True,
    )
    assert required_log_fields["atlas_reliability_base_mean"] == 0.4
    assert required_log_fields["atlas_reliability_runtime_mean"] == 0.5
    assert required_log_fields["atlas_reliability_runtime_min"] == 0.2
    assert required_log_fields["atlas_reliability_runtime_max"] == 0.8
    assert required_log_fields["atlas_reliability_base_p50"] == 0.44
    assert required_log_fields["atlas_reliability_runtime_mapped_hist_high"] == 0.5
    assert required_log_fields["atlas_reliability_effective_p90"] == 0.85
    assert required_log_fields["refresh_evidence_override_gate_ratio"] == 0.45
    assert required_log_fields["refresh_override_base_bucket_high_override_ratio"] == 0.6
    assert required_log_fields["atlas_refresh_done"] == 1
    assert required_log_fields["state_stable_count"] == 12
    assert required_log_fields["state_passive_count"] == 3
    assert required_log_fields["state_active_count"] == 2
    assert required_log_fields["state_out_pending_count"] == 1
    assert required_log_fields["stable_ratio"] == 0.6
    assert required_log_fields["passive_ratio"] == 0.15
    assert required_log_fields["active_ratio"] == 0.1
    assert required_log_fields["out_of_anchor_pending_count"] == 1
    assert required_log_fields["cooldown_ratio"] == 0.2
    assert required_log_fields["mean_gc_fail_count"] == 0.4
    assert required_log_fields["mean_active_lifetime"] == 3.5
    assert required_log_fields["max_active_lifetime"] == 6.0
    assert required_log_fields["transition_passive_to_active_count"] == 3
    assert required_log_fields["observed_node_ratio"] == 0.75
    assert required_log_fields["mean_node_photo_ema"] == 0.2
    assert required_log_fields["split_count"] == 1.0
    assert required_log_fields["clone_count"] == 2.0
    assert required_log_fields["explore_clone_count"] == 3.0
    assert required_log_fields["stable_split_count"] == 1.0
    assert required_log_fields["stable_split_candidate_count"] == 4.0
    assert required_log_fields["stable_split_block_drift_count"] == 1.0
    assert required_log_fields["stable_clone_count"] == 2.0
    assert required_log_fields["stable_clone_candidate_count"] == 5.0
    assert required_log_fields["stable_clone_block_projector_count"] == 2.0
    assert required_log_fields["active_explore_clone_count"] == 3.0
    assert required_log_fields["active_to_explore_clone_handoff_count"] == 3.0
    assert required_log_fields["densify_stale_render_tensors"] == 1.0
    assert required_log_fields["densify_stale_reason"] == "gc_pruned_render_tensors"
    assert required_log_fields["densify_skip_reason"] == "none"
    assert required_log_fields["pruned_count"] == 4.0
    assert required_log_fields["gc_ran"] == 1.0
    assert required_log_fields["gc_due"] == 1.0
    assert required_log_fields["gc_interval"] == 5.0
    assert required_log_fields["gc_candidates"] == 6.0
    assert required_log_fields["reattach_success"] == 7.0
    assert required_log_fields["reattach_fail"] == 8.0
    assert required_log_fields["reattach_success_ratio"] == 7.0 / 6.0
    assert required_log_fields["reattach_fail_ratio"] == 8.0 / 6.0
    assert required_log_fields["pending_reattach_success"] == 2.0
    assert required_log_fields["pending_reattach_fail"] == 1.0
    assert required_log_fields["pending_reattach_success_ratio"] == 2.0 / 3.0
    assert required_log_fields["pending_reattach_fail_ratio"] == 1.0 / 3.0
    assert required_log_fields["reattach_tier1_attempt_count"] == 6.0
    assert required_log_fields["reattach_tier3_success"] == 1.0
    assert required_log_fields["reattach_candidate_starvation_count"] == 1.0
    assert required_log_fields["ray_guided_priority_queries"] == 1.0
    assert required_log_fields["ray_guided_active_queries"] == 1.0
    assert required_log_fields["prune_after_gc"] == 5.0
    assert required_log_fields["pose_b1_enabled"] == 1
    assert required_log_fields["pose_b2_enabled"] == 0
    assert required_log_fields["warmup_only"] == 1
    assert required_log_fields["main_phase"] == 0
    assert required_log_fields["refresh_pending"] == 1
    assert required_log_fields["main_phase_ready"] == 0
    assert required_log_fields["pose_refine_disabled_or_blocked_by_phase"] == 1
    assert required_log_fields["pose_b1_loss"] == 0.125
    assert required_log_fields["pose_b2_loss"] == 0.25
    assert required_log_fields["runtime_pose_gate_enabled"] == 1.0
    assert required_log_fields["pose_b1_gate_block_stable_ratio"] == 1.0
    assert required_log_fields["pose_b1_gate_block_capacity_ratio"] == 1.0
    assert required_log_fields["pose_b2_enabled_by_history"] == 1.0
    assert required_log_fields["pose_trust_clamp_count"] == 1.0
    assert required_log_fields["sigma_parallel_clamp_hits"] == 9.0
    assert required_log_fields["sigma_support_clamp_hits"] == 10.0
    assert required_log_fields["sigma_ray_clamp_hits"] == 11.0
    assert required_log_fields["sigma_stable_parallel_mean"] == 0.1
    assert required_log_fields["sigma_passive_parallel_mean"] == 0.2
    assert required_log_fields["sigma_active_parallel_mean"] == 0.3
    assert required_log_fields["sigma_active_ray_span_mean"] == 0.4
    assert required_log_fields["sigma_active_ray_parallel_p90"] == 0.5
    assert required_log_fields["atlas_rank_u_mean"] == 1.5
    assert required_log_fields["atlas_active_ray_fallback_count"] == 4.0
    assert required_log_fields["atlas_kl_stable_mean"] == 0.11
    assert required_log_fields["atlas_kl_passive_mean"] == 0.22
    assert required_log_fields["atlas_kl_active_mean"] == 0.33
    assert required_log_fields["atlas_slab_total_loss"] == 0.33
    assert required_log_fields["atlas_slab_active_count"] == 2.0
    assert required_log_fields["atlas_slab_mean_penalty"] == 0.44
    assert required_log_fields["atlas_slab_violation_count"] == 3.0
    assert required_log_fields["atlas_slab_violation_ratio"] == 0.6
    assert required_log_fields["atlas_refresh_snapshot_ready"] == 1
    assert required_log_fields["atlas_refresh_snapshot_observed_ratio"] == 0.5
    assert required_log_fields["atlas_refresh_snapshot_observed_count"] == 2.0
    assert required_log_fields["atlas_refresh_snapshot_photo_ema_mean"] == 0.25
    assert required_log_fields["atlas_refresh_snapshot_visibility_ema_mean"] == 0.75
    assert required_log_fields["atlas_refresh_snapshot_obs_quality_mean"] == 0.35
    assert required_log_fields["atlas_refresh_snapshot_obs_quality_max"] == 0.8

    manifest = train_module._build_ablation_manifest(
        SimpleNamespace(atlas_path="atlas.npz"),
        SimpleNamespace(
            disable_pose_refine=False,
            atlas_reg_warmup_steps=100,
            atlas_mean_weight=0.0,
            atlas_gc_interval=50,
            atlas_slab_weight=0.01,
            pose_photo_weight=0.01,
            pose_photo_alpha=0.5,
            pose_gradient_weight=0.02,
            pose_patchfeat_weight=0.03,
        ),
        SimpleNamespace(has_atlas_bindings=True),
    )
    assert manifest["init_variant"] == "foundation_atlas"
    assert manifest["atlas_runtime_calibration_variant"] == "self_calibrated_runtime_refresh"
    assert manifest["reliability_variant"] == "detached_runtime_prior"
    assert manifest["shape_prior_variant"] == "scale_free_orientation_anisotropy"
    assert manifest["gc_variant"] == "async_interval_hash_reattach"
    assert manifest["exploration_variant"] == "ray_constrained_slab"
    assert manifest["pose_b2_variant"] == "l1+ssim+grad+patchfeat_budgeted"

    subspace_calls = {"count": 0}

    def fake_build_variational_subspace(*args, **kwargs):
        subspace_calls["count"] += 1
        return {"fake": True, "count": subspace_calls["count"]}

    original_build = train_module.build_variational_subspace
    train_module.build_variational_subspace = fake_build_variational_subspace
    try:
        cached = train_module._ensure_variational_subspace_info(None, object(), object(), controls_opt)
        reused = train_module._ensure_variational_subspace_info(cached, object(), object(), controls_opt)
        assert subspace_calls["count"] == 1
        assert cached is reused
    finally:
        train_module.build_variational_subspace = original_build

    tmp_root = REPO_ROOT / ".tmp_atlas_warmup_controls"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm.training_setup(build_training_args())

        schedule_warmup = _compute_atlas_loss_schedule(
            5,
            controls_opt,
            gm,
            {"stable_ratio": 0.30},
        )
        assert schedule_warmup["mean_scale"] == 0.0
        assert 0.0 < schedule_warmup["kl_scale"] < 0.25
        assert schedule_warmup["main_phase_progress"] == 0.0
        assert schedule_warmup["ori_scale"] == 0.0
        assert schedule_warmup["aniso_scale"] == 0.0

        gm._atlas_refresh_done = True
        gm._init_point_count = 2
        schedule_low_stable = _compute_atlas_loss_schedule(
            10,
            controls_opt,
            gm,
            {"stable_ratio": 0.05},
        )
        assert schedule_low_stable["refresh_gate"] == 1.0
        assert schedule_low_stable["main_phase_progress"] == 0.0
        assert schedule_low_stable["shape_floor_reliability_gate"] > 0.0
        assert schedule_low_stable["shape_floor_scale"] > 0.0
        assert schedule_low_stable["ori_scale"] > 0.0
        assert schedule_low_stable["aniso_scale"] > 0.0

        schedule_post_refresh = _compute_atlas_loss_schedule(
            20,
            controls_opt,
            gm,
            {"stable_ratio": 0.75},
        )
        assert schedule_post_refresh["refresh_gate"] == 1.0
        assert schedule_post_refresh["capacity_gate"] > 0.0
        assert schedule_post_refresh["mean_scale"] > 0.0
        assert schedule_post_refresh["kl_scale"] > schedule_warmup["kl_scale"]
        assert schedule_post_refresh["main_phase_progress"] > 0.0
        assert schedule_post_refresh["ori_scale"] > 0.0
        assert schedule_post_refresh["aniso_scale"] > 0.0
        gm._atlas_refresh_done = False
        gm._init_point_count = int(gm.get_xyz.shape[0])

        device = gm.get_xyz.device
        residuals = torch.tensor([0.20, 0.40, 0.10, 0.35], dtype=torch.float32, device=device)
        visible = torch.tensor([1, 1, 1, 1], dtype=torch.bool, device=device)

        with torch.no_grad():
            gm._xyz[0] = gm._atlas_positions[0] + torch.tensor([5.0, 0.0, 0.0], dtype=torch.float32, device=device)
            gm._atlas_high_residual_count[:] = torch.tensor([2, 3, 4, 5], dtype=torch.long, device=device)
            gm._atlas_gc_fail_count[:] = torch.tensor([1, 1, 2, 2], dtype=torch.long, device=device)
            gm._atlas_drift_flag[:] = torch.tensor([False, True, False, True], dtype=torch.bool, device=device)
            gm._atlas_ref_score[:] = torch.tensor([0.3, 0.2, 0.1, 0.4], dtype=torch.float32, device=device)

        high_before = gm._atlas_high_residual_count.detach().clone()
        gc_before = gm._atlas_gc_fail_count.detach().clone()
        drift_before = gm._atlas_drift_flag.detach().clone()
        ref_camera_before = gm._atlas_ref_camera.detach().clone()
        ref_score_before = gm._atlas_ref_score.detach().clone()
        state_before = gm.get_atlas_state.detach().clone()

        warmup_metrics = gm.update_atlas_runtime_stats(
            residuals,
            visible,
            ema_decay=0.0,
            drift_radius_mult=0.25,
            camera_index=1,
            high_residual_threshold=0.03,
            warmup_only=True,
        )
        assert warmup_metrics["warmup_only"] == 1.0
        assert warmup_metrics["mean_photo_ema"] > 0.0
        assert warmup_metrics["mean_visibility_ema"] > 0.0
        assert float(gm._atlas_node_photo_ema.mean().item()) > 0.0
        assert float(gm._atlas_node_visibility_ema.mean().item()) > 0.0
        assert torch.equal(gm._atlas_high_residual_count, high_before)
        assert torch.equal(gm._atlas_gc_fail_count, gc_before)
        assert torch.equal(gm._atlas_drift_flag, drift_before)
        assert torch.all((gm._atlas_ref_camera >= -1) & (gm._atlas_ref_camera < gm._atlas_view_weights.shape[1]))
        assert torch.all(gm._atlas_ref_score >= ref_score_before)
        assert torch.equal(gm.get_atlas_state, state_before)

        dummy_camera = DummyPoseCamera(device)
        before_pose = measure_pose_delta(dummy_camera)
        assert before_pose["translation_norm"] > 0.0
        assert before_pose["rotation_degrees"] > 0.0
        reset_metrics = reset_camera_pose_delta(dummy_camera)
        after_pose = measure_pose_delta(dummy_camera)
        assert reset_metrics["pose_reset_applied"] == 1.0
        assert after_pose["translation_norm"] <= 1e-8
        assert after_pose["rotation_degrees"] <= 1e-4
        assert dummy_camera.refresh_count >= 1

        print(json.dumps({
            "warmup_controls": warmup_controls,
            "main_controls": main_controls,
            "schedule_warmup": schedule_warmup,
            "schedule_low_stable": schedule_low_stable,
            "schedule_post_refresh": schedule_post_refresh,
            "warmup_metrics": warmup_metrics,
            "pose_reset": reset_metrics,
        }, indent=2))
        print("[OK] Atlas warmup controls check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
