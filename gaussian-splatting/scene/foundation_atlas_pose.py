from __future__ import annotations

import math
import numpy as np

import torch
import torch.nn.functional as F

from scene.foundation_atlas import (
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
)
from scene.foundation_atlas_exploration import compute_point_slab_bounds
from utils.graphics_utils import geom_transform_points
from utils.sh_utils import SH2RGB


def _normalize_quaternion(quaternion: torch.Tensor):
    return torch.nn.functional.normalize(quaternion, dim=0)


def _canonicalize_quaternion(quaternion: torch.Tensor):
    normalized = _normalize_quaternion(quaternion)
    return torch.where(normalized[0] < 0.0, -normalized, normalized)


def _quaternion_angle_radians(quaternion: torch.Tensor):
    normalized = _canonicalize_quaternion(quaternion)
    sin_half = torch.linalg.norm(normalized[1:]).clamp_min(1e-8)
    cos_half = normalized[0].abs().clamp_min(1e-8)
    return 2.0 * torch.atan2(sin_half, cos_half)


def measure_pose_delta(viewpoint_camera):
    if not hasattr(viewpoint_camera, "pose_delta_q") or not hasattr(viewpoint_camera, "pose_delta_t"):
        return {
            "translation_norm": 0.0,
            "rotation_radians": 0.0,
            "rotation_degrees": 0.0,
        }

    pose_q = viewpoint_camera.pose_delta_q
    pose_t = viewpoint_camera.pose_delta_t
    translation_norm = torch.linalg.norm(pose_t)
    rotation_radians = _quaternion_angle_radians(pose_q)
    return {
        "translation_norm": float(translation_norm.detach().item()),
        "rotation_radians": float(rotation_radians.detach().item()),
        "rotation_degrees": float(torch.rad2deg(rotation_radians).detach().item()),
    }


def should_enable_pose_refinement(
    disable_pose_refine: bool,
    stable_ratio: float,
    drift_ratio: float,
    active_ratio: float,
    total_points: int,
    init_point_count: int,
    refresh_done: bool = True,
    stable_ratio_threshold: float = 0.45,
    drift_ratio_threshold: float = 0.01,
    active_ratio_threshold: float = 0.15,
    capacity_ratio_threshold: float = 1.25,
    atlas_reliability_effective_mean: float | None = None,
    camera_local_ready: bool = False,
    local_corr_count: float = 0.0,
    local_corr_trust_median: float = 0.0,
    local_corr_spatial_coverage: float = 0.0,
    local_corr_valid_ratio: float = 0.0,
    local_corr_min_count: float = 24.0,
    local_corr_min_trust_median: float = 0.35,
    local_corr_min_spatial_coverage: float = 0.12,
    local_corr_min_valid_ratio: float = 0.18,
):
    init_point_count = max(int(init_point_count), 1)
    total_points = max(int(total_points), 0)
    capacity_ratio = float(total_points) / float(init_point_count)
    static_stable_ratio_threshold = float(stable_ratio_threshold)
    static_capacity_ratio_threshold = float(capacity_ratio_threshold)
    reliability_mean = None
    if atlas_reliability_effective_mean is not None and math.isfinite(float(atlas_reliability_effective_mean)):
        reliability_mean = float(max(min(atlas_reliability_effective_mean, 1.0), 0.0))
        stable_ratio_threshold = max(0.22, float(stable_ratio_threshold) - 0.35 * reliability_mean)
        if bool(refresh_done):
            capacity_floor = 1.05 + 0.07 * (1.0 - min(max((reliability_mean - 0.10) / 0.18, 0.0), 1.0))
            capacity_ratio_threshold = min(float(capacity_ratio_threshold), capacity_floor)

    global_safety_ready = bool(
        (not bool(disable_pose_refine))
        and bool(refresh_done)
        and float(drift_ratio) <= float(drift_ratio_threshold)
        and float(active_ratio) <= float(active_ratio_threshold)
    )
    global_ready = bool(
        global_safety_ready
        and float(stable_ratio) >= float(stable_ratio_threshold)
        and capacity_ratio >= float(capacity_ratio_threshold)
    )
    local_corr_count_ready = float(local_corr_count) >= float(local_corr_min_count)
    local_corr_trust_ready = float(local_corr_trust_median) >= float(local_corr_min_trust_median)
    local_corr_coverage_ready = float(local_corr_spatial_coverage) >= float(local_corr_min_spatial_coverage)
    local_corr_valid_ready = float(local_corr_valid_ratio) >= float(local_corr_min_valid_ratio)
    camera_local_ready = bool(
        bool(camera_local_ready)
        or (
            local_corr_count_ready
            and local_corr_trust_ready
            and local_corr_coverage_ready
            and local_corr_valid_ready
        )
    )
    local_ready = bool(global_safety_ready and bool(camera_local_ready))
    enabled = bool(global_ready or local_ready)
    if enabled:
        gate_reason = "enabled" if global_ready else "camera_local_ready"
    else:
        reason_parts = []
        if bool(disable_pose_refine):
            reason_parts.append("disabled")
        if not bool(refresh_done):
            reason_parts.append("refresh_pending")
        if float(stable_ratio) < float(stable_ratio_threshold):
            reason_parts.append("stable_ratio_low")
        if float(drift_ratio) > float(drift_ratio_threshold):
            reason_parts.append("drift_ratio_high")
        if float(active_ratio) > float(active_ratio_threshold):
            reason_parts.append("active_ratio_high")
        if capacity_ratio < float(capacity_ratio_threshold):
            reason_parts.append("capacity_ratio_low")
        gate_reason = ",".join(reason_parts) if reason_parts else "threshold_blocked"
    return enabled, {
        "pose_gate_disabled": 1.0 if bool(disable_pose_refine) else 0.0,
        "pose_gate_refresh_done": 1.0 if bool(refresh_done) else 0.0,
        "pose_gate_stable_ratio": float(stable_ratio),
        "pose_gate_drift_ratio": float(drift_ratio),
        "pose_gate_active_ratio": float(active_ratio),
        "pose_gate_capacity_ratio": float(capacity_ratio),
        "pose_gate_stable_threshold": float(stable_ratio_threshold),
        "pose_gate_stable_threshold_static": float(static_stable_ratio_threshold),
        "pose_gate_drift_threshold": float(drift_ratio_threshold),
        "pose_gate_active_threshold": float(active_ratio_threshold),
        "pose_gate_capacity_threshold": float(capacity_ratio_threshold),
        "pose_gate_capacity_threshold_static": float(static_capacity_ratio_threshold),
        "pose_gate_reliability_effective_mean": float(reliability_mean if reliability_mean is not None else -1.0),
        "pose_gate_global_safety_ready": 1.0 if global_safety_ready else 0.0,
        "pose_gate_global_ready": 1.0 if global_ready else 0.0,
        "pose_gate_camera_local_ready": 1.0 if local_ready else 0.0,
        "pose_gate_local_corr_count": float(local_corr_count),
        "pose_gate_local_corr_trust_median": float(local_corr_trust_median),
        "pose_gate_local_corr_spatial_coverage": float(local_corr_spatial_coverage),
        "pose_gate_local_corr_valid_ratio": float(local_corr_valid_ratio),
        "pose_gate_local_corr_count_ready": 1.0 if local_corr_count_ready else 0.0,
        "pose_gate_local_corr_trust_ready": 1.0 if local_corr_trust_ready else 0.0,
        "pose_gate_local_corr_coverage_ready": 1.0 if local_corr_coverage_ready else 0.0,
        "pose_gate_local_corr_valid_ready": 1.0 if local_corr_valid_ready else 0.0,
        "pose_gate_local_corr_min_count": float(local_corr_min_count),
        "pose_gate_local_corr_min_trust_median": float(local_corr_min_trust_median),
        "pose_gate_local_corr_min_spatial_coverage": float(local_corr_min_spatial_coverage),
        "pose_gate_local_corr_min_valid_ratio": float(local_corr_min_valid_ratio),
        "pose_gate_init_points": float(init_point_count),
        "pose_gate_total_points": float(total_points),
        "pose_gate_b1_enabled": 1.0 if enabled else 0.0,
        "pose_gate_enabled": 1.0 if enabled else 0.0,
        "pose_gate_b1_reason": gate_reason,
        "pose_gate_reason": gate_reason,
    }


def should_enable_pose_photometric_refinement(
    disable_pose_refine: bool,
    b1_enabled: bool,
    drift_ratio: float,
    b1_success_streak: int,
    quality_regressed: bool,
    min_b1_success_streak: int = 3,
    drift_ratio_threshold: float = 0.01,
    b1_history_count: int = 0,
    per_camera_b1_count: int = 0,
    per_camera_b1_quality: float = 0.0,
    per_camera_b1_median_px: float | None = None,
    min_per_camera_b1_quality: float = 0.45,
    max_per_camera_b1_median_px: float = 96.0,
    min_global_b1_history: int | None = None,
    min_per_camera_b1_history: int = 1,
    bootstrap_ready: bool = False,
    low_frequency_due: bool = True,
    b1_history_fresh: bool = True,
    photo_corridor_ready: bool = False,
):
    min_b1_success_streak = int(max(min_b1_success_streak, 0))
    min_global_b1_history = (
        min_b1_success_streak
        if min_global_b1_history is None
        else int(max(min_global_b1_history, 0))
    )
    min_per_camera_b1_history = int(max(min_per_camera_b1_history, 0))
    streak_ready = int(b1_success_streak) >= min_b1_success_streak
    global_history_ready = int(b1_history_count) >= min_global_b1_history
    camera_history_ready = int(per_camera_b1_count) >= min_per_camera_b1_history
    camera_quality = float(max(min(per_camera_b1_quality, 1.0), 0.0))
    camera_median_px = float("inf") if per_camera_b1_median_px is None else float(per_camera_b1_median_px)
    camera_residual_ready = math.isfinite(camera_median_px) and camera_median_px <= float(max_per_camera_b1_median_px)
    camera_quality_ready = bool(
        camera_history_ready
        and (
            camera_quality >= float(min_per_camera_b1_quality)
            or camera_residual_ready
        )
    )
    history_ready_raw = streak_ready or global_history_ready or camera_quality_ready
    history_ready = history_ready_raw and bool(b1_history_fresh)
    photo_corridor_ready = bool(photo_corridor_ready)
    b1_ready = bool(b1_enabled) or bool(bootstrap_ready) or history_ready or photo_corridor_ready
    history_or_bootstrap_ready = history_ready or bool(bootstrap_ready)
    history_or_corridor_ready = history_or_bootstrap_ready or photo_corridor_ready
    quality_gate_ready = (not bool(quality_regressed)) or photo_corridor_ready
    b2_data_ready = bool(
        (not bool(disable_pose_refine))
        and b1_ready
        and history_or_corridor_ready
    )
    b2_quality_ready = bool(
        float(drift_ratio) <= float(drift_ratio_threshold)
        and quality_gate_ready
        and bool(b1_history_fresh or bootstrap_ready or photo_corridor_ready)
    )
    b2_optimization_ready = bool(low_frequency_due)
    enabled_for_compute = bool(b2_data_ready)
    enabled_for_step = bool(
        b2_data_ready
        and b2_quality_ready
        and b2_optimization_ready
    )
    if enabled_for_step:
        gate_reason = "enabled"
    elif enabled_for_compute:
        if not b2_optimization_ready:
            gate_reason = "compute_ready_optimization_wait"
        elif float(drift_ratio) > float(drift_ratio_threshold):
            gate_reason = "compute_ready_drift_guard"
        elif bool(quality_regressed) and not photo_corridor_ready:
            gate_reason = "compute_ready_quality_wait"
        else:
            gate_reason = "compute_ready"
    else:
        reason_parts = []
        if bool(disable_pose_refine):
            reason_parts.append("disabled")
        if not b1_ready:
            reason_parts.append("b1_not_ready")
        if history_ready_raw and not bool(b1_history_fresh) and not bool(bootstrap_ready) and not photo_corridor_ready:
            reason_parts.append("stale_b1_history")
        elif not history_or_corridor_ready:
            reason_parts.append("insufficient_b1_history")
        if not bool(low_frequency_due):
            reason_parts.append("low_frequency_wait")
        gate_reason = ",".join(reason_parts) if reason_parts else "threshold_blocked"
    return enabled_for_compute, {
        "pose_b2_gate_disabled": 1.0 if bool(disable_pose_refine) else 0.0,
        "pose_b2_gate_b1_enabled": 1.0 if bool(b1_enabled) else 0.0,
        "pose_b2_gate_b1_ready": 1.0 if b1_ready else 0.0,
        "pose_b2_gate_data_ready": 1.0 if b2_data_ready else 0.0,
        "pose_b2_gate_quality_ready": 1.0 if b2_quality_ready else 0.0,
        "pose_b2_gate_optimization_ready": 1.0 if b2_optimization_ready else 0.0,
        "pose_b2_gate_enabled_for_compute": 1.0 if enabled_for_compute else 0.0,
        "pose_b2_gate_enabled_for_step": 1.0 if enabled_for_step else 0.0,
        "pose_b2_gate_history_ready": 1.0 if history_ready else 0.0,
        "pose_b2_gate_history_ready_raw": 1.0 if history_ready_raw else 0.0,
        "pose_b2_gate_history_fresh": 1.0 if bool(b1_history_fresh) else 0.0,
        "pose_b2_gate_camera_history_ready": 1.0 if camera_history_ready else 0.0,
        "pose_b2_gate_camera_quality_ready": 1.0 if camera_quality_ready else 0.0,
        "pose_b2_gate_camera_b1_quality": float(camera_quality),
        "pose_b2_gate_camera_b1_median_px": float(camera_median_px) if math.isfinite(camera_median_px) else -1.0,
        "pose_b2_gate_min_camera_b1_quality": float(min_per_camera_b1_quality),
        "pose_b2_gate_max_camera_b1_median_px": float(max_per_camera_b1_median_px),
        "pose_b2_gate_history_or_bootstrap_ready": 1.0 if history_or_bootstrap_ready else 0.0,
        "pose_b2_gate_history_or_corridor_ready": 1.0 if history_or_corridor_ready else 0.0,
        "pose_b2_gate_bootstrap_ready": 1.0 if bool(bootstrap_ready) else 0.0,
        "pose_b2_gate_photo_corridor_ready": 1.0 if photo_corridor_ready else 0.0,
        "pose_b2_gate_low_frequency_due": 1.0 if bool(low_frequency_due) else 0.0,
        "pose_b2_gate_drift_ratio": float(drift_ratio),
        "pose_b2_gate_drift_threshold": float(drift_ratio_threshold),
        "pose_b2_gate_b1_success_streak": float(max(int(b1_success_streak), 0)),
        "pose_b2_gate_min_b1_success_streak": float(max(int(min_b1_success_streak), 0)),
        "pose_b2_gate_b1_history_count": float(max(int(b1_history_count), 0)),
        "pose_b2_gate_min_global_b1_history": float(max(int(min_global_b1_history), 0)),
        "pose_b2_gate_per_camera_b1_count": float(max(int(per_camera_b1_count), 0)),
        "pose_b2_gate_min_per_camera_b1_history": float(max(int(min_per_camera_b1_history), 0)),
        "pose_b2_gate_quality_regressed": 1.0 if bool(quality_regressed) else 0.0,
        "pose_b2_gate_enabled": 1.0 if enabled_for_compute else 0.0,
        "pose_gate_b2_reason": gate_reason,
        "pose_b2_gate_reason": gate_reason,
    }


def should_freeze_pose_refinement(
    pose_active: bool,
    drift_ratio: float,
    active_ratio: float,
    quality_bad_streak: int,
    freeze_cooldown: int = 0,
    drift_ratio_threshold: float = 0.02,
    active_ratio_threshold: float = 0.15,
    max_quality_bad_streak: int = 3,
    cooldown_recover_ready: bool = False,
):
    drift_spike = bool(pose_active) and float(drift_ratio) > float(drift_ratio_threshold)
    active_spike = bool(pose_active) and float(active_ratio) > float(active_ratio_threshold)
    quality_regression = bool(pose_active) and int(quality_bad_streak) >= int(max(max_quality_bad_streak, 1))
    emergency_stop = drift_spike or active_spike or quality_regression
    cooldown_active = int(max(freeze_cooldown, 0)) > 0
    freeze_active = emergency_stop or (cooldown_active and not bool(cooldown_recover_ready))
    if emergency_stop:
        reason_parts = []
        if drift_spike:
            reason_parts.append("drift_spike")
        if active_spike:
            reason_parts.append("active_spike")
        if quality_regression:
            reason_parts.append("quality_regression")
        freeze_reason = ",".join(reason_parts)
    elif cooldown_active and bool(cooldown_recover_ready):
        freeze_reason = "cooldown_recovered"
    elif cooldown_active:
        freeze_reason = "cooldown"
    else:
        freeze_reason = "none"
    return freeze_active, {
        "pose_freeze_pose_active": 1.0 if bool(pose_active) else 0.0,
        "pose_freeze_active": 1.0 if freeze_active else 0.0,
        "pose_freeze_emergency_stop": 1.0 if emergency_stop else 0.0,
        "pose_freeze_cooldown_active": 1.0 if cooldown_active else 0.0,
        "pose_freeze_cooldown_recover_ready": 1.0 if bool(cooldown_recover_ready) else 0.0,
        "pose_freeze_drift_ratio": float(drift_ratio),
        "pose_freeze_drift_threshold": float(drift_ratio_threshold),
        "pose_freeze_active_ratio": float(active_ratio),
        "pose_freeze_active_threshold": float(active_ratio_threshold),
        "pose_freeze_quality_bad_streak": float(max(int(quality_bad_streak), 0)),
        "pose_freeze_quality_bad_threshold": float(max(int(max_quality_bad_streak), 1)),
        "pose_freeze_drift_spike": 1.0 if drift_spike else 0.0,
        "pose_freeze_active_spike": 1.0 if active_spike else 0.0,
        "pose_freeze_quality_regression": 1.0 if quality_regression else 0.0,
        "pose_freeze_cooldown": float(max(int(freeze_cooldown), 0)),
        "pose_freeze_reason": freeze_reason,
    }


def reset_camera_pose_delta(viewpoint_camera):
    metrics = {
        "pose_reset_applied": 0.0,
        "pose_translation_norm_after": 0.0,
        "pose_rotation_degrees_after": 0.0,
    }
    if not hasattr(viewpoint_camera, "pose_delta_q") or not hasattr(viewpoint_camera, "pose_delta_t"):
        return metrics

    before = measure_pose_delta(viewpoint_camera)
    with torch.no_grad():
        identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=viewpoint_camera.pose_delta_q.dtype, device=viewpoint_camera.pose_delta_q.device)
        viewpoint_camera.pose_delta_q.copy_(identity_q)
        viewpoint_camera.pose_delta_t.zero_()
    if hasattr(viewpoint_camera, "refresh_pose_matrices"):
        viewpoint_camera.refresh_pose_matrices()
    after = measure_pose_delta(viewpoint_camera)
    metrics["pose_reset_applied"] = 1.0 if (
        before["translation_norm"] > 1e-10 or before["rotation_degrees"] > 1e-8
    ) else 0.0
    metrics["pose_translation_norm_after"] = after["translation_norm"]
    metrics["pose_rotation_degrees_after"] = after["rotation_degrees"]
    return metrics


def compute_pose_trust_region_loss(
    viewpoint_camera,
    translation_weight: float,
    rotation_weight: float,
    max_translation_norm: float,
    max_rotation_degrees: float,
):
    device = viewpoint_camera.pose_delta_t.device
    dtype = viewpoint_camera.pose_delta_t.dtype
    zero = torch.zeros((), dtype=dtype, device=device)
    metrics = {
        "pose_trust_loss": 0.0,
        "pose_translation_norm": 0.0,
        "pose_rotation_degrees": 0.0,
        "pose_translation_penalty": 0.0,
        "pose_rotation_penalty": 0.0,
    }
    if translation_weight <= 0.0 and rotation_weight <= 0.0:
        return zero, metrics

    translation_norm = torch.linalg.norm(viewpoint_camera.pose_delta_t)
    rotation_radians = _quaternion_angle_radians(viewpoint_camera.pose_delta_q)
    rotation_limit_radians = math.radians(max(float(max_rotation_degrees), 1e-3))
    translation_scale = max(float(max_translation_norm), 1e-6)

    translation_penalty = (translation_norm / translation_scale).square()
    rotation_penalty = (rotation_radians / rotation_limit_radians).square()
    trust_loss = float(max(translation_weight, 0.0)) * translation_penalty + float(max(rotation_weight, 0.0)) * rotation_penalty

    metrics["pose_trust_loss"] = float(trust_loss.detach().item())
    metrics["pose_translation_norm"] = float(translation_norm.detach().item())
    metrics["pose_rotation_degrees"] = float(torch.rad2deg(rotation_radians).detach().item())
    metrics["pose_translation_penalty"] = float(translation_penalty.detach().item())
    metrics["pose_rotation_penalty"] = float(rotation_penalty.detach().item())
    return trust_loss, metrics


def clamp_camera_pose_delta(viewpoint_camera, max_translation_norm: float, max_rotation_degrees: float):
    metrics = {
        "pose_translation_clamped": 0.0,
        "pose_rotation_clamped": 0.0,
        "pose_translation_norm_after": 0.0,
        "pose_rotation_degrees_after": 0.0,
    }
    if not hasattr(viewpoint_camera, "pose_delta_q") or not hasattr(viewpoint_camera, "pose_delta_t"):
        return metrics

    max_translation_norm = max(float(max_translation_norm), 0.0)
    max_rotation_radians = math.radians(max(float(max_rotation_degrees), 0.0))
    with torch.no_grad():
        translation_norm = torch.linalg.norm(viewpoint_camera.pose_delta_t)
        if max_translation_norm > 0.0 and float(translation_norm.item()) > max_translation_norm:
            viewpoint_camera.pose_delta_t.mul_(max_translation_norm / float(translation_norm.item()))
            metrics["pose_translation_clamped"] = 1.0

        canonical_q = _canonicalize_quaternion(viewpoint_camera.pose_delta_q.detach())
        rotation_radians = _quaternion_angle_radians(canonical_q)
        if max_rotation_radians > 0.0 and float(rotation_radians.item()) > max_rotation_radians:
            axis = canonical_q[1:]
            axis_norm = torch.linalg.norm(axis)
            if float(axis_norm.item()) <= 1e-8:
                axis = torch.tensor([1.0, 0.0, 0.0], dtype=canonical_q.dtype, device=canonical_q.device)
            else:
                axis = axis / axis_norm
            half_angle = 0.5 * max_rotation_radians
            clamped_q = torch.cat(
                (
                    torch.tensor([math.cos(half_angle)], dtype=canonical_q.dtype, device=canonical_q.device),
                    axis * math.sin(half_angle),
                ),
                dim=0,
            )
            viewpoint_camera.pose_delta_q.copy_(clamped_q)
            metrics["pose_rotation_clamped"] = 1.0
        else:
            viewpoint_camera.pose_delta_q.copy_(canonical_q)

    if hasattr(viewpoint_camera, "refresh_pose_matrices"):
        viewpoint_camera.refresh_pose_matrices()
    after = measure_pose_delta(viewpoint_camera)
    metrics["pose_translation_norm_after"] = after["translation_norm"]
    metrics["pose_rotation_degrees_after"] = after["rotation_degrees"]
    return metrics


def summarize_pose_correspondence_budget(
    viewpoint_camera,
    min_correspondences: int,
    bootstrap_min_correspondences: int = 0,
):
    metrics = {
        "pose_corr_loaded": 0.0,
        "pose_corr_projected": 0.0,
        "pose_corr_in_frame": 0.0,
        "pose_corr_trustworthy": 0.0,
        "pose_corr_trust_median": 0.0,
        "pose_corr_spatial_coverage": 0.0,
        "pose_corr_valid_ratio": 0.0,
        "pose_corr_ready": 0.0,
        "pose_corr_bootstrap_ready": 0.0,
        "pose_corr_min_target": float(max(int(min_correspondences), 0)),
        "pose_corr_bootstrap_min_target": float(max(int(bootstrap_min_correspondences), 0)),
        "pose_corr_reason": "missing_correspondence",
    }
    corr_xyz = getattr(viewpoint_camera, "pose_correspondences_xyz", None)
    corr_xy = getattr(viewpoint_camera, "pose_correspondences_xy", None)
    if corr_xyz is None or corr_xy is None:
        return metrics
    if corr_xyz.shape[0] == 0 or corr_xy.shape[0] == 0:
        metrics["pose_corr_reason"] = "empty_correspondence"
        return metrics

    corr_err = getattr(viewpoint_camera, "pose_correspondence_error", None)
    if corr_err is None:
        corr_err = torch.ones((corr_xyz.shape[0],), dtype=torch.float32, device=corr_xyz.device)
    else:
        corr_err = corr_err.detach().reshape(-1).clamp_min(1e-4)
    corr_trust = getattr(viewpoint_camera, "pose_correspondence_trust", None)
    if corr_trust is None or corr_trust.shape[0] != corr_xyz.shape[0]:
        corr_trust = torch.ones((corr_xyz.shape[0],), dtype=torch.float32, device=corr_xyz.device)
    else:
        corr_trust = corr_trust.detach().reshape(-1).to(device=corr_xyz.device, dtype=torch.float32).clamp(0.0, 1.0)

    with torch.no_grad():
        coords_ndc_loose, depth_loose, projected = _project_points(viewpoint_camera, corr_xyz.detach(), require_in_frame=False)
        coords_ndc, depth, in_frame = _project_points(viewpoint_camera, corr_xyz.detach(), require_in_frame=True)
        target_ndc = _pixel_to_ndc(viewpoint_camera, corr_xy.detach())
        finite_projected = (
            torch.isfinite(depth_loose)
            & torch.isfinite(corr_err)
            & torch.isfinite(coords_ndc_loose).all(dim=1)
            & torch.isfinite(target_ndc).all(dim=1)
        )
        finite_in_frame = (
            torch.isfinite(depth)
            & torch.isfinite(corr_err)
            & torch.isfinite(coords_ndc).all(dim=1)
            & torch.isfinite(target_ndc).all(dim=1)
        )
        projected = projected & finite_projected
        in_frame = in_frame & finite_in_frame
        trustworthy = in_frame.clone()
        if torch.any(in_frame):
            valid_err = corr_err[in_frame]
            median_err = torch.median(valid_err)
            mad_err = torch.median((valid_err - median_err).abs())
            err_threshold = torch.maximum(median_err + 1e-4, median_err + 2.5 * 1.4826 * mad_err)
            trustworthy_indices = torch.nonzero(in_frame, as_tuple=False).squeeze(-1)
            trustworthy_mask = valid_err <= err_threshold
            trustworthy = torch.zeros_like(in_frame)
            trustworthy[trustworthy_indices[trustworthy_mask]] = True

    loaded_corr = int(min(corr_xyz.shape[0], corr_xy.shape[0]))
    projected_corr = int(projected.sum().item())
    in_frame_corr = int(in_frame.sum().item())
    trustworthy_corr = int(trustworthy.sum().item())
    trust_median = 0.0
    spatial_coverage = 0.0
    valid_ratio = float(trustworthy_corr) / max(float(loaded_corr), 1.0)
    if trustworthy_corr > 0:
        trust_median = float(torch.median(corr_trust[trustworthy]).detach().item())
        trustworthy_ndc = coords_ndc[trustworthy].detach()
        grid_size = 4
        grid_xy = torch.floor(((trustworthy_ndc + 1.0) * 0.5 * grid_size).clamp(0.0, grid_size - 1e-4)).to(torch.int64)
        cell_ids = grid_xy[:, 1] * grid_size + grid_xy[:, 0]
        spatial_coverage = float(torch.unique(cell_ids).numel()) / float(grid_size * grid_size)
    min_correspondences = max(int(min_correspondences), 0)
    bootstrap_min_correspondences = max(int(bootstrap_min_correspondences), 0)
    ready = trustworthy_corr >= min_correspondences if min_correspondences > 0 else trustworthy_corr > 0
    bootstrap_ready = trustworthy_corr >= bootstrap_min_correspondences if bootstrap_min_correspondences > 0 else ready

    if ready:
        reason = "ready"
    elif trustworthy_corr == 0:
        reason = "no_trustworthy_correspondence"
    elif in_frame_corr == 0:
        reason = "all_out_of_frame"
    elif projected_corr == 0:
        reason = "projection_invalid"
    else:
        reason = "insufficient_correspondence"

    metrics.update(
        {
            "pose_corr_loaded": float(loaded_corr),
            "pose_corr_projected": float(projected_corr),
            "pose_corr_in_frame": float(in_frame_corr),
            "pose_corr_trustworthy": float(trustworthy_corr),
            "pose_corr_trust_median": float(trust_median),
            "pose_corr_spatial_coverage": float(spatial_coverage),
            "pose_corr_valid_ratio": float(valid_ratio),
            "pose_corr_ready": 1.0 if ready else 0.0,
            "pose_corr_bootstrap_ready": 1.0 if bootstrap_ready else 0.0,
            "pose_corr_reason": reason,
        }
    )
    return metrics


def compute_pose_quality_score(
    stage: str,
    metrics: dict,
    target_count: int,
    success_streak: int = 0,
    quality_regressed: bool = False,
):
    target_count = max(int(target_count), 1)
    stage = str(stage).lower()
    if stage == "b1":
        corr_count = float(metrics.get("pose_geo_num_corr", 0.0))
        in_frame_corr = float(metrics.get("pose_geo_in_frame_corr", corr_count))
        loaded_corr = max(float(metrics.get("pose_geo_loaded_corr", corr_count)), 1.0)
        median_px = float(
            metrics.get(
                "pose_geo_selected_median_px_error",
                metrics.get("pose_geo_median_px_error", 0.0),
            )
        )
        count_quality = min(corr_count / float(target_count), 1.0)
        coverage_quality = min(in_frame_corr / loaded_corr, 1.0)
        residual_quality = math.exp(-max(median_px, 0.0) / 4.0)
        quality = 0.45 * count_quality + 0.25 * coverage_quality + 0.30 * residual_quality
    else:
        sample_count = float(metrics.get("pose_num_samples", 0.0))
        mask_mean = float(metrics.get("pose_mask_mean", 0.0))
        view_support = float(metrics.get("pose_view_support_mean", 0.0))
        active_safety = float(metrics.get("pose_active_safe_fraction", 1.0))
        sample_quality = min(sample_count / float(target_count), 1.0)
        mask_quality = min(max(mask_mean / 0.10, 0.0), 1.0)
        support_quality = min(max(view_support, 0.0), 1.0)
        safety_quality = min(max(active_safety, 0.0), 1.0)
        streak_bonus = min(max(int(success_streak), 0) / 3.0, 1.0) * 0.10
        regression_penalty = 0.15 if bool(quality_regressed) else 0.0
        quality = 0.35 * sample_quality + 0.30 * mask_quality + 0.20 * support_quality + 0.15 * safety_quality
        quality = quality + streak_bonus - regression_penalty

    return float(min(max(quality, 0.0), 1.0))


def compute_dynamic_pose_trust_region(
    stage: str,
    base_translation_norm: float,
    base_rotation_degrees: float,
    quality_score: float,
    bootstrap_active: bool = False,
    min_scale: float = 0.35,
    max_scale: float = 1.75,
    b2_max_scale: float = 1.15,
):
    stage = str(stage).lower()
    quality_score = float(min(max(quality_score, 0.0), 1.0))
    min_scale = float(max(min_scale, 0.05))
    max_scale = float(max(max_scale, min_scale))
    b2_max_scale = float(max(b2_max_scale, min_scale))

    if stage == "b1":
        scale_floor = max(min_scale, 0.55 if bool(bootstrap_active) else min_scale)
        scale_ceiling = max(max_scale, scale_floor)
    else:
        scale_floor = max(min_scale, 0.30)
        scale_ceiling = min(max_scale, max(b2_max_scale, scale_floor))

    trust_scale = scale_floor + (scale_ceiling - scale_floor) * quality_score
    translation_limit = max(float(base_translation_norm), 1e-6) * trust_scale
    rotation_limit_degrees = max(float(base_rotation_degrees), 1e-3) * trust_scale
    return translation_limit, rotation_limit_degrees, {
        "pose_trust_stage": stage,
        "pose_trust_quality_score": float(quality_score),
        "pose_trust_scale": float(trust_scale),
        "pose_trust_scale_floor": float(scale_floor),
        "pose_trust_scale_ceiling": float(scale_ceiling),
        "pose_trust_bootstrap_active": 1.0 if bool(bootstrap_active) else 0.0,
        "pose_trust_translation_limit": float(translation_limit),
        "pose_trust_rotation_limit_degrees": float(rotation_limit_degrees),
    }


def _project_points(camera, points: torch.Tensor, require_in_frame: bool = True):
    if hasattr(camera, "refresh_pose_matrices"):
        pose_trainable = bool(
            getattr(getattr(camera, "pose_delta_q", None), "requires_grad", False)
            or getattr(getattr(camera, "pose_delta_t", None), "requires_grad", False)
        )
        try:
            camera.refresh_pose_matrices(differentiable=pose_trainable)
        except TypeError:
            camera.refresh_pose_matrices()
    ndc = geom_transform_points(points, camera.full_proj_transform)
    w2c = camera.world_view_transform.transpose(0, 1)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    cam_h = torch.cat((points, ones), dim=1) @ w2c
    depth = cam_h[:, 2] / cam_h[:, 3].clamp_min(1e-6)
    valid = depth > camera.znear
    if require_in_frame:
        valid = valid & (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0)
    return ndc[:, :2], depth, valid


def _sample_map(image: torch.Tensor, coords_ndc: torch.Tensor):
    if coords_ndc.shape[0] == 0:
        return torch.empty((0, image.shape[0]), dtype=image.dtype, device=image.device)
    grid = coords_ndc.view(1, -1, 1, 2)
    sampled = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
    return sampled.squeeze(0).squeeze(-1).transpose(0, 1)


def _sample_patches(image: torch.Tensor, coords_ndc: torch.Tensor, radius: int):
    radius = int(max(radius, 0))
    patch_size = radius * 2 + 1
    if coords_ndc.shape[0] == 0:
        return torch.empty((0, image.shape[0], patch_size, patch_size), dtype=image.dtype, device=image.device)

    if radius == 0:
        center = _sample_map(image, coords_ndc)
        return center[:, :, None, None]

    height = max(int(image.shape[1]) - 1, 1)
    width = max(int(image.shape[2]) - 1, 1)
    offset_x = (2.0 / width) * torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    offset_y = (2.0 / height) * torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    grid_y, grid_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
    offsets = torch.stack((grid_x, grid_y), dim=-1)
    patch_grid = coords_ndc[:, None, None, :] + offsets[None, :, :, :]
    image_batch = image.unsqueeze(0).expand(coords_ndc.shape[0], -1, -1, -1)
    return F.grid_sample(image_batch, patch_grid, mode="bilinear", padding_mode="border", align_corners=True)


def _patch_descriptor(patches: torch.Tensor):
    if patches.numel() == 0:
        return patches.reshape(patches.shape[0], 0)
    centered = patches - patches.mean(dim=(2, 3), keepdim=True)
    variance = centered.square().mean(dim=(2, 3), keepdim=True)
    std = torch.sqrt(variance + 1e-8)
    normalized = torch.nan_to_num(centered / std, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.flatten(start_dim=1)


def _gradient_map(image: torch.Tensor):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    channels = image.shape[0]
    weight_x = kernel_x.expand(channels, 1, -1, -1)
    weight_y = kernel_y.expand(channels, 1, -1, -1)
    image_batch = image.unsqueeze(0)
    grad_x = F.conv2d(image_batch, weight_x, padding=1, groups=channels)
    grad_y = F.conv2d(image_batch, weight_y, padding=1, groups=channels)
    return torch.cat((grad_x.squeeze(0), grad_y.squeeze(0)), dim=0)


def _patch_ssim(pred_patch: torch.Tensor, gt_patch: torch.Tensor):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred_patch.mean(dim=(1, 2, 3), keepdim=True)
    mu_y = gt_patch.mean(dim=(1, 2, 3), keepdim=True)
    sigma_x = ((pred_patch - mu_x) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_y = ((gt_patch - mu_y) ** 2).mean(dim=(1, 2, 3), keepdim=True)
    sigma_xy = ((pred_patch - mu_x) * (gt_patch - mu_y)).mean(dim=(1, 2, 3), keepdim=True)
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim = (numerator / denominator.clamp_min(1e-6)).squeeze(-1).squeeze(-1).squeeze(-1)
    return ssim.clamp(0.0, 1.0)


def _pixel_to_ndc(camera, xy_pixels: torch.Tensor):
    width = max(int(camera.image_width) - 1, 1)
    height = max(int(camera.image_height) - 1, 1)
    x_ndc = 2.0 * xy_pixels[:, 0] / float(width) - 1.0
    y_ndc = 1.0 - 2.0 * xy_pixels[:, 1] / float(height)
    return torch.stack((x_ndc, y_ndc), dim=1)


def _offset_distance_sq(radius: int, device, dtype):
    offsets = torch.arange(-int(radius), int(radius) + 1, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    return grid_x.square() + grid_y.square()


def _weighted_patch_l1(pred_patch: torch.Tensor, gt_patch: torch.Tensor, mask: torch.Tensor):
    error = (pred_patch - gt_patch).abs().mean(dim=1)
    mask_sum = mask.sum(dim=(1, 2)).clamp_min(1e-6)
    return (error * mask).sum(dim=(1, 2)) / mask_sum


def _resolve_current_view_support(gaussians, active_idx: torch.Tensor, viewpoint_camera):
    view_weights = gaussians.get_gaussian_atlas_view_weights.detach()[active_idx]
    view_counts = gaussians.get_gaussian_atlas_view_counts.detach()[active_idx]
    current_view = int(getattr(viewpoint_camera, "uid", -1))
    if (
        view_weights.ndim != 2
        or view_counts.ndim != 2
        or current_view < 0
        or current_view >= view_weights.shape[1]
        or current_view >= view_counts.shape[1]
    ):
        ones = torch.ones((active_idx.shape[0],), dtype=torch.float32, device=gaussians.get_xyz.device)
        zeros = torch.zeros_like(ones)
        return ones, zeros, zeros

    current_weight = view_weights[:, current_view].to(dtype=torch.float32)
    current_count = view_counts[:, current_view].to(dtype=torch.float32)
    total_weight = view_weights.to(dtype=torch.float32).sum(dim=1).clamp_min(1e-6)
    normalized_weight = (current_weight / total_weight).clamp(0.0, 1.0)
    count_support = 1.0 - torch.exp(-current_count / 3.0)
    view_support = (0.20 + 0.80 * torch.sqrt(normalized_weight * count_support.clamp_min(0.0))).clamp(0.20, 1.0)
    return view_support, current_weight, current_count


def _compute_uncertainty_ratio(gaussians, active_idx: torch.Tensor, state: torch.Tensor):
    atlas_radius = gaussians.get_gaussian_atlas_radius.detach()[active_idx].clamp_min(1e-4)
    sigma_support = gaussians.get_center_sigma_support.detach()[active_idx].squeeze(-1)
    sigma_parallel = gaussians.get_center_sigma_parallel.detach()[active_idx].squeeze(-1)
    dominant_sigma = torch.where(
        state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
        torch.maximum(sigma_support, sigma_parallel),
        sigma_support,
    )
    uncertainty_ratio = dominant_sigma / atlas_radius
    return uncertainty_ratio, sigma_support, sigma_parallel, atlas_radius


def _compute_pose_selection_scores(gaussians, active_idx: torch.Tensor, viewpoint_camera):
    reliability = gaussians.get_gaussian_atlas_reliability_effective.detach()[active_idx]
    state = gaussians.get_atlas_state.detach()[active_idx]
    visibility = gaussians.get_atlas_visibility_ema.detach()[active_idx]
    photo_ema = gaussians.get_atlas_photo_ema.detach()[active_idx]
    view_support, _, _ = _resolve_current_view_support(gaussians, active_idx, viewpoint_camera)
    uncertainty_ratio, _, _, _ = _compute_uncertainty_ratio(gaussians, active_idx, state)

    state_weight = torch.where(state == GAUSSIAN_STATE_UNSTABLE_ACTIVE, 0.85, 1.0)
    state_weight = torch.where(state == GAUSSIAN_STATE_UNSTABLE_PASSIVE, torch.full_like(state_weight, 0.45), state_weight)
    visibility_weight = 0.25 + 0.75 * visibility.clamp(0.0, 1.0)
    residual_weight = 0.20 + 0.80 * torch.exp(-1.5 * photo_ema.clamp_min(0.0))
    uncertainty_weight = torch.exp(-1.15 * uncertainty_ratio.clamp_min(0.0))
    return (reliability * state_weight * visibility_weight * residual_weight * view_support * uncertainty_weight).clamp_min(1e-4)


def _compute_active_safety_gate(gaussians, active_idx: torch.Tensor, state: torch.Tensor, viewpoint_camera):
    gate = torch.ones((active_idx.shape[0],), dtype=torch.float32, device=gaussians.get_xyz.device)
    active_mask = state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
    if not torch.any(active_mask):
        return gate

    slab = compute_point_slab_bounds(
        gaussians,
        active_idx[active_mask],
        fallback_camera_center=viewpoint_camera.camera_center.detach(),
        slab_radius_mult=1.0,
        detach_points=False,
    )
    if slab is None:
        gate[active_mask] = 0.15
        return gate

    tau = slab["tau"]
    tau_min = slab["tau_min"]
    tau_max = slab["tau_max"]
    span = (tau_max - tau_min).clamp_min(1e-4)
    interior_margin = torch.minimum(tau - tau_min, tau_max - tau)
    safety = (interior_margin / span).clamp(0.0, 0.5) * 2.0
    gate[active_mask] = (0.05 + 0.95 * safety).clamp(0.05, 1.0)
    return gate


def _compute_pose_sample_weights(gaussians, active_idx: torch.Tensor, viewpoint_camera):
    reliability = gaussians.get_gaussian_atlas_reliability_effective.detach()[active_idx]
    state = gaussians.get_atlas_state.detach()[active_idx]
    visibility = gaussians.get_atlas_visibility_ema.detach()[active_idx]
    photo_ema = gaussians.get_atlas_photo_ema.detach()[active_idx]
    ref_score = gaussians.get_atlas_ref_score.detach()[active_idx].clamp(0.0, 1.0)

    view_support, current_view_weight, current_view_count = _resolve_current_view_support(gaussians, active_idx, viewpoint_camera)
    active_safety = _compute_active_safety_gate(gaussians, active_idx, state, viewpoint_camera)
    uncertainty_ratio, sigma_support, sigma_parallel, atlas_radius = _compute_uncertainty_ratio(gaussians, active_idx, state)

    state_weight = torch.where(state == GAUSSIAN_STATE_UNSTABLE_ACTIVE, 0.35 + 0.65 * active_safety, 1.0)
    state_weight = torch.where(
        state == GAUSSIAN_STATE_UNSTABLE_PASSIVE,
        torch.full_like(state_weight, 0.35),
        state_weight,
    )
    visibility_weight = 0.25 + 0.75 * visibility.clamp(0.0, 1.0)
    residual_weight = 0.20 + 0.80 * torch.exp(-1.5 * photo_ema.clamp_min(0.0))
    uncertainty_weight = torch.exp(-1.25 * uncertainty_ratio.clamp_min(0.0))
    ref_weight = 0.35 + 0.65 * torch.maximum(ref_score, view_support)
    sample_weight = reliability * state_weight * visibility_weight * residual_weight * uncertainty_weight * ref_weight * view_support
    aux = {
        "uncertainty_ratio": uncertainty_ratio,
        "uncertainty_weight": uncertainty_weight,
        "view_support": view_support,
        "current_view_weight": current_view_weight,
        "current_view_count": current_view_count,
        "active_safety": active_safety,
        "reliability": reliability,
        "ref_weight": ref_weight,
        "sigma_support": sigma_support,
        "sigma_parallel": sigma_parallel,
        "atlas_radius": atlas_radius,
    }
    return sample_weight.clamp_min(1e-4), state, aux


def _compute_passive_safe_pose_mask(
    gaussians,
    atlas_state: torch.Tensor,
    reliability_min: float = 0.16,
    support_consistency_min: float = 0.24,
    ref_score_min: float = 0.05,
):
    if not gaussians.has_atlas_bindings or atlas_state.numel() == 0:
        return torch.zeros_like(atlas_state, dtype=torch.bool), {}

    passive_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
    if not torch.any(passive_mask):
        return passive_mask, {
            "passive_safe_reliability_mean": 0.0,
            "passive_safe_support_mean": 0.0,
            "passive_safe_ref_mean": 0.0,
        }

    reliability = gaussians.get_gaussian_atlas_reliability_effective.detach().clamp(0.0, 1.0)
    support_consistency = gaussians._compute_support_consistency_score().detach().clamp(0.0, 1.0)
    ref_score = gaussians.get_atlas_ref_score.detach().clamp(0.0, 1.0)
    drift_ok = ~gaussians.get_atlas_drift_flag.detach()
    cooldown_ok = gaussians.get_atlas_state_cooldown.detach() <= 0
    visibility = gaussians.get_atlas_visibility_ema.detach().clamp(0.0, 1.0)
    photo_ema = gaussians.get_atlas_photo_ema.detach().clamp(0.0, 1.0)

    passive_safe = (
        passive_mask
        & (reliability >= float(reliability_min))
        & (support_consistency >= float(support_consistency_min))
        & (
            (ref_score >= float(ref_score_min))
            | (visibility >= 0.01)
        )
        & (photo_ema <= 0.70)
        & drift_ok
        & cooldown_ok
    )
    passive_values = passive_mask
    return passive_safe, {
        "passive_safe_reliability_mean": float(reliability[passive_values].mean().detach().item()) if torch.any(passive_values) else 0.0,
        "passive_safe_support_mean": float(support_consistency[passive_values].mean().detach().item()) if torch.any(passive_values) else 0.0,
        "passive_safe_ref_mean": float(ref_score[passive_values].mean().detach().item()) if torch.any(passive_values) else 0.0,
    }


def _select_budgeted_correspondences(
    corr_xy: torch.Tensor,
    reproj_px: torch.Tensor,
    corr_err: torch.Tensor,
    sample_count: int,
    image_width: int,
    image_height: int,
    atlas_native: torch.Tensor | None = None,
    atlas_reliability: torch.Tensor | None = None,
    corr_trust: torch.Tensor | None = None,
    atlas_node_ids: torch.Tensor | None = None,
):
    sample_count = int(max(sample_count, 0))
    if sample_count <= 0 or corr_xy.shape[0] <= sample_count:
        return torch.arange(corr_xy.shape[0], device=corr_xy.device, dtype=torch.long)

    reproj_scale = torch.median(reproj_px).clamp_min(1.0)
    err_scale = torch.median(corr_err).clamp_min(1e-3)
    priority = (reproj_px / reproj_scale) + 0.15 * (corr_err / err_scale)
    if atlas_native is not None and atlas_native.shape[0] == priority.shape[0]:
        priority = priority - 0.45 * atlas_native.to(dtype=priority.dtype)
    if atlas_reliability is not None and atlas_reliability.shape[0] == priority.shape[0]:
        priority = priority - 0.25 * atlas_reliability.to(dtype=priority.dtype).clamp(0.0, 1.0)
    if corr_trust is not None and corr_trust.shape[0] == priority.shape[0]:
        priority = priority - 0.20 * corr_trust.to(dtype=priority.dtype).clamp(0.0, 1.0)

    xy_np = corr_xy.detach().cpu().numpy()
    priority_np = priority.detach().cpu().numpy()
    order = np.argsort(priority_np, kind="stable")
    cell_area = (float(max(image_width, 1)) * float(max(image_height, 1))) / float(max(sample_count, 1))
    cell_size = max(1, int(round(math.sqrt(max(cell_area, 1.0)))))

    occupied_cells = set()
    occupied_nodes = set()
    selected: list[int] = []
    for index in order.tolist():
        cell_key = (
            int(xy_np[index, 0] // float(cell_size)),
            int(xy_np[index, 1] // float(cell_size)),
        )
        if cell_key in occupied_cells:
            continue
        if atlas_node_ids is not None and atlas_node_ids.shape[0] == corr_xy.shape[0]:
            node_id = int(atlas_node_ids[index].detach().cpu().item())
            if node_id >= 0 and node_id in occupied_nodes and len(selected) < sample_count // 2:
                continue
            if node_id >= 0:
                occupied_nodes.add(node_id)
        occupied_cells.add(cell_key)
        selected.append(int(index))
        if len(selected) >= sample_count:
            break

    if len(selected) < sample_count:
        used = np.zeros((corr_xy.shape[0],), dtype=bool)
        if selected:
            used[np.asarray(selected, dtype=np.int64)] = True
        remaining = order[~used[order]]
        needed = sample_count - len(selected)
        selected.extend(remaining[:needed].tolist())

    return torch.as_tensor(selected[:sample_count], device=corr_xy.device, dtype=torch.long)


def _estimate_screen_footprint_radius(
    viewpoint_camera,
    gaussians,
    active_idx: torch.Tensor,
    depth: torch.Tensor,
    uncertainty_ratio: torch.Tensor,
    base_patch_radius: int,
):
    scaling = gaussians.get_scaling.detach()[active_idx].max(dim=1).values
    atlas_radius = gaussians.get_gaussian_atlas_radius.detach()[active_idx].clamp_min(1e-4)
    world_radius = torch.maximum(scaling, atlas_radius)
    fx = 0.5 * float(viewpoint_camera.image_width) / max(math.tan(float(viewpoint_camera.FoVx) * 0.5), 1e-4)
    fy = 0.5 * float(viewpoint_camera.image_height) / max(math.tan(float(viewpoint_camera.FoVy) * 0.5), 1e-4)
    focal = 0.5 * (fx + fy)
    footprint = (world_radius * float(focal) / depth.clamp_min(1e-3)) * (1.0 + 0.5 * uncertainty_ratio.clamp(0.0, 4.0))
    minimum = torch.full_like(footprint, float(max(int(base_patch_radius), 1)))
    return torch.maximum(footprint, minimum).clamp(1.0, 6.0)


def _build_uncertainty_aware_ray_mask(
    viewpoint_camera,
    coords: torch.Tensor,
    depth: torch.Tensor,
    effective_radius_px: torch.Tensor,
    uncertainty_ratio: torch.Tensor,
    active_safety: torch.Tensor,
    sample_patch_radius: int,
):
    dtype = coords.dtype
    device = coords.device
    grad_patch = _sample_patches(_gradient_map(viewpoint_camera.original_image.detach()), coords, sample_patch_radius)
    channels = int(grad_patch.shape[1] // 2)
    grad_mag = torch.sqrt(
        grad_patch[:, :channels].square() + grad_patch[:, channels:].square() + 1e-8
    ).mean(dim=1)
    grad_gate = grad_mag / (grad_mag + 0.05)

    offset_sq = _offset_distance_sq(sample_patch_radius, device=device, dtype=dtype).unsqueeze(0)
    sigma_px = (0.65 * effective_radius_px[:, None, None]).clamp_min(0.5)
    footprint = torch.exp(-0.5 * offset_sq / sigma_px.square())
    footprint = footprint * (offset_sq <= (effective_radius_px[:, None, None] + 0.75).square()).to(dtype)

    uncertainty_gate = torch.exp(-0.9 * uncertainty_ratio[:, None, None].clamp_min(0.0))
    mask = footprint * (0.25 + 0.75 * grad_gate) * (0.20 + 0.80 * uncertainty_gate) * (0.10 + 0.90 * active_safety[:, None, None])

    depth_gate_mean = mask.new_tensor(1.0)
    if getattr(viewpoint_camera, "depth_reliable", False) and viewpoint_camera.invdepthmap is not None:
        gt_invdepth_patch = _sample_patches(viewpoint_camera.invdepthmap, coords, sample_patch_radius).squeeze(1)
        if viewpoint_camera.depth_confidence is not None:
            depth_conf_patch = _sample_patches(viewpoint_camera.depth_confidence, coords, sample_patch_radius).squeeze(1).clamp(0.0, 1.0)
        else:
            depth_conf_patch = torch.ones_like(gt_invdepth_patch)
        expected_invdepth = depth.reciprocal().clamp_max(1e4)[:, None, None]
        depth_tolerance = 0.01 + expected_invdepth * (0.05 + 0.20 * uncertainty_ratio[:, None, None].clamp(0.0, 4.0))
        depth_gate = torch.exp(-(gt_invdepth_patch - expected_invdepth).abs() / depth_tolerance.clamp_min(1e-4))
        depth_gate = depth_gate * (gt_invdepth_patch > 0).to(dtype)
        soft_depth_gate = 0.10 + 0.90 * depth_gate
        mask = mask * soft_depth_gate * (0.25 + 0.75 * depth_conf_patch)
        depth_gate_mean = depth_gate.mean()

    raw_mask = mask.clamp_min(0.0)
    normalized_mask = raw_mask / raw_mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    return normalized_mask, {
        "mask_strength_mean": float(raw_mask.mean().detach().item()) if raw_mask.numel() > 0 else 0.0,
        "gradient_gate_mean": float(grad_gate.mean().detach().item()) if grad_gate.numel() > 0 else 0.0,
        "gradient_observable_ratio": float((grad_gate > 0.05).float().mean().detach().item()) if grad_gate.numel() > 0 else 0.0,
        "depth_gate_mean": float(depth_gate_mean.detach().item()),
    }


def evaluate_correspondence_reprojection_px(
    viewpoint_camera,
    corr_xyz: torch.Tensor,
    corr_xy: torch.Tensor,
) -> float:
    """Measure median reprojection error for a FIXED set of correspondences with the current camera pose.

    Use this for stable pre/post B1 comparison — avoids re-selection noise.
    Returns NaN if no correspondences project in-frame.
    """
    with torch.no_grad():
        coords_ndc, depth, in_frame = _project_points(
            viewpoint_camera, corr_xyz.detach(), require_in_frame=True
        )
        target_ndc = _pixel_to_ndc(viewpoint_camera, corr_xy.detach())
        if not torch.any(in_frame):
            return float("nan")
        coords_ndc = coords_ndc[in_frame]
        target_ndc = target_ndc[in_frame]
        width = max(int(viewpoint_camera.image_width) - 1, 1)
        height = max(int(viewpoint_camera.image_height) - 1, 1)
        px_x = (coords_ndc[:, 0] - target_ndc[:, 0]) * 0.5 * float(width)
        px_y = (coords_ndc[:, 1] - target_ndc[:, 1]) * 0.5 * float(height)
        reproj_px = torch.sqrt(px_x.square() + px_y.square())
        return float(torch.median(reproj_px).item())


def compute_pose_geometric_correspondence_loss(
    viewpoint_camera,
    gaussians,
    sample_count: int,
    geo_weight: float,
    residual_mad_scale: float = 3.5,
    residual_percentile: float = 0.9,
    min_correspondences: int = 32,
    return_selected: bool = False,
):
    zero = gaussians.get_xyz.new_zeros(())
    metrics = {
        "pose_geo_loss": 0.0,
        "pose_geo_num_corr": 0.0,
        "pose_geo_mean_px_error": 0.0,
        "pose_geo_median_px_error": 0.0,
        "pose_geo_prefilter_median_px_error": 0.0,
        "pose_geo_selected_median_px_error": 0.0,
        "pose_geo_after_filter_corr": 0.0,
        "pose_geo_selected_count": 0.0,
        "pose_geo_selected_unique_node_count": 0.0,
        "pose_geo_loaded_corr": 0.0,
        "pose_geo_projected_corr": 0.0,
        "pose_geo_in_frame_corr": 0.0,
        "pose_geo_rejected_large_error": 0.0,
        "pose_geo_trustworthy_corr": 0.0,
        "pose_geo_threshold_px": 0.0,
        "pose_geo_atlas_native_ratio": 0.0,
        "pose_geo_fallback_ratio": 0.0,
        "pose_geo_atlas_native_selected_ratio": 0.0,
        "pose_geo_fallback_selected_ratio": 0.0,
        "pose_geo_prefilter_atlas_native_count": 0.0,
        "pose_geo_prefilter_fallback_count": 0.0,
        "pose_geo_selected_atlas_native_count": 0.0,
        "pose_geo_selected_fallback_count": 0.0,
        "pose_geo_count_quality": 0.0,
        "pose_geo_coverage_quality": 0.0,
        "pose_geo_residual_quality": 0.0,
        "pose_geo_min_corr_target": float(max(int(min_correspondences), 0)),
        "pose_geo_skip_reason": "missing_correspondence",
    }
    corr_xyz = getattr(viewpoint_camera, "pose_correspondences_xyz", None)
    corr_xy = getattr(viewpoint_camera, "pose_correspondences_xy", None)
    _no_sel = (None, None)
    if geo_weight <= 0.0 or corr_xyz is None or corr_xy is None:
        if geo_weight <= 0.0:
            metrics["pose_geo_skip_reason"] = "geo_weight_disabled"
        return (zero, metrics, *_no_sel) if return_selected else (zero, metrics)
    if corr_xyz.shape[0] == 0:
        metrics["pose_geo_skip_reason"] = "empty_correspondence"
        return (zero, metrics, *_no_sel) if return_selected else (zero, metrics)

    corr_err = getattr(viewpoint_camera, "pose_correspondence_error", None)
    metrics["pose_geo_loaded_corr"] = float(corr_xyz.shape[0])
    corr_xyz = corr_xyz.detach()
    corr_xy = corr_xy.detach()
    if corr_err is None:
        corr_err = torch.ones((corr_xyz.shape[0],), dtype=torch.float32, device=corr_xyz.device)
    else:
        corr_err = corr_err.detach().reshape(-1).clamp_min(1e-4)
    atlas_native = getattr(viewpoint_camera, "pose_correspondence_is_atlas_native", None)
    atlas_node_ids = getattr(viewpoint_camera, "pose_correspondence_atlas_node_ids", None)
    atlas_reliability_hint = getattr(viewpoint_camera, "pose_correspondence_atlas_reliability", None)
    corr_trust = getattr(viewpoint_camera, "pose_correspondence_trust", None)
    if atlas_native is None or atlas_native.shape[0] != corr_xyz.shape[0]:
        atlas_native = torch.zeros((corr_xyz.shape[0],), dtype=torch.bool, device=corr_xyz.device)
    else:
        atlas_native = atlas_native.detach().reshape(-1).to(device=corr_xyz.device, dtype=torch.bool)
    if atlas_node_ids is None or atlas_node_ids.shape[0] != corr_xyz.shape[0]:
        atlas_node_ids = torch.full((corr_xyz.shape[0],), -1, dtype=torch.long, device=corr_xyz.device)
    else:
        atlas_node_ids = atlas_node_ids.detach().reshape(-1).to(device=corr_xyz.device, dtype=torch.long)
    if atlas_reliability_hint is None or atlas_reliability_hint.shape[0] != corr_xyz.shape[0]:
        atlas_reliability_hint = torch.ones((corr_xyz.shape[0],), dtype=torch.float32, device=corr_xyz.device)
    else:
        atlas_reliability_hint = atlas_reliability_hint.detach().reshape(-1).to(device=corr_xyz.device, dtype=torch.float32).clamp(0.0, 1.0)
    if corr_trust is None or corr_trust.shape[0] != corr_xyz.shape[0]:
        corr_trust = torch.ones((corr_xyz.shape[0],), dtype=torch.float32, device=corr_xyz.device)
    else:
        corr_trust = corr_trust.detach().reshape(-1).to(device=corr_xyz.device, dtype=torch.float32).clamp(0.0, 1.0)
    metrics["pose_geo_atlas_native_ratio"] = float(atlas_native.float().mean().detach().item()) if atlas_native.numel() > 0 else 0.0
    metrics["pose_geo_fallback_ratio"] = 1.0 - metrics["pose_geo_atlas_native_ratio"]

    coords_ndc_loose, depth_loose, projected = _project_points(viewpoint_camera, corr_xyz, require_in_frame=False)
    coords_ndc, depth, in_frame = _project_points(viewpoint_camera, corr_xyz, require_in_frame=True)
    target_ndc = _pixel_to_ndc(viewpoint_camera, corr_xy)
    finite_projected = (
        torch.isfinite(depth_loose)
        & torch.isfinite(corr_err)
        & torch.isfinite(coords_ndc_loose).all(dim=1)
        & torch.isfinite(target_ndc).all(dim=1)
    )
    finite_in_frame = (
        torch.isfinite(depth)
        & torch.isfinite(corr_err)
        & torch.isfinite(coords_ndc).all(dim=1)
        & torch.isfinite(target_ndc).all(dim=1)
    )
    projected = projected & finite_projected
    metrics["pose_geo_projected_corr"] = float(projected.sum().detach().item())
    if not torch.any(projected):
        metrics["pose_geo_skip_reason"] = "projection_invalid"
        return (zero, metrics, *_no_sel) if return_selected else (zero, metrics)

    in_frame = in_frame & finite_in_frame
    metrics["pose_geo_in_frame_corr"] = float(in_frame.sum().detach().item())
    if not torch.any(in_frame):
        metrics["pose_geo_skip_reason"] = "all_out_of_frame"
        return (zero, metrics, *_no_sel) if return_selected else (zero, metrics)

    corr_xyz = corr_xyz[in_frame]
    corr_xy = corr_xy[in_frame]
    corr_err = corr_err[in_frame]
    atlas_native = atlas_native[in_frame]
    atlas_node_ids = atlas_node_ids[in_frame]
    atlas_reliability_hint = atlas_reliability_hint[in_frame]
    corr_trust = corr_trust[in_frame]
    coords_ndc = coords_ndc[in_frame]
    depth = depth[in_frame]
    target_ndc = target_ndc[in_frame]
    metrics["pose_geo_prefilter_atlas_native_count"] = float(atlas_native.sum().detach().item()) if atlas_native.numel() > 0 else 0.0
    metrics["pose_geo_prefilter_fallback_count"] = float((~atlas_native).sum().detach().item()) if atlas_native.numel() > 0 else 0.0

    width = max(int(viewpoint_camera.image_width) - 1, 1)
    height = max(int(viewpoint_camera.image_height) - 1, 1)
    px_x = (coords_ndc[:, 0] - target_ndc[:, 0]) * 0.5 * float(width)
    px_y = (coords_ndc[:, 1] - target_ndc[:, 1]) * 0.5 * float(height)
    reproj_error = torch.stack((px_x, px_y), dim=1)
    reproj_sq = reproj_error.square().sum(dim=1)
    reproj_px = torch.sqrt(reproj_sq.clamp_min(0.0))
    if reproj_px.numel() > 0:
        metrics["pose_geo_prefilter_median_px_error"] = float(torch.median(reproj_px).detach().item())
        metrics["pose_geo_median_px_error"] = metrics["pose_geo_prefilter_median_px_error"]

    reject_mask = torch.zeros_like(reproj_px, dtype=torch.bool)
    if reproj_px.shape[0] > 1:
        median_px = torch.median(reproj_px)
        mad_px = torch.median((reproj_px - median_px).abs())
        robust_sigma = 1.4826 * mad_px
        adaptive_mad_scale = float(max(residual_mad_scale, 0.0))
        adaptive_percentile = min(max(float(residual_percentile), 0.5), 1.0)
        if float(median_px.detach().item()) > 128.0:
            adaptive_mad_scale = min(adaptive_mad_scale, 1.50)
            adaptive_percentile = min(adaptive_percentile, 0.60)
        elif float(median_px.detach().item()) > 64.0:
            adaptive_mad_scale = min(adaptive_mad_scale, 2.00)
            adaptive_percentile = min(adaptive_percentile, 0.70)
        elif float(median_px.detach().item()) > 32.0:
            adaptive_mad_scale = min(adaptive_mad_scale, 2.50)
            adaptive_percentile = min(adaptive_percentile, 0.80)
        mad_threshold = median_px + adaptive_mad_scale * robust_sigma
        if reproj_px.shape[0] > 1:
            quantile_value = adaptive_percentile
            percentile_threshold = torch.quantile(reproj_px, quantile_value)
        else:
            percentile_threshold = median_px
        residual_threshold = torch.maximum(
            median_px + 1e-6,
            torch.minimum(mad_threshold, percentile_threshold),
        )
        residual_threshold = torch.maximum(residual_threshold, median_px + 1e-3)
        metrics["pose_geo_threshold_px"] = float(residual_threshold.detach().item())
        keep_mask = reproj_px <= residual_threshold
        if torch.any(atlas_native):
            atlas_keep = atlas_native & (
                (atlas_reliability_hint >= torch.quantile(atlas_reliability_hint[atlas_native], 0.35))
                | (corr_trust >= torch.quantile(corr_trust[atlas_native], 0.35))
            )
            keep_mask = keep_mask & ((~atlas_native) | atlas_keep)
        min_correspondences = int(max(min_correspondences, 0))
        if keep_mask.sum().item() < min_correspondences <= reproj_px.shape[0]:
            robust_priority = reproj_px / reproj_px.median().clamp_min(1.0)
            robust_priority = robust_priority - 0.35 * atlas_native.to(dtype=robust_priority.dtype)
            robust_priority = robust_priority - 0.25 * atlas_reliability_hint.clamp(0.0, 1.0)
            robust_priority = robust_priority - 0.15 * corr_trust.clamp(0.0, 1.0)
            best_indices = torch.argsort(robust_priority)[:min_correspondences]
            keep_mask = torch.zeros_like(keep_mask, dtype=torch.bool)
            keep_mask[best_indices] = True
        reject_mask = ~keep_mask
        metrics["pose_geo_trustworthy_corr"] = float(keep_mask.sum().detach().item())
        if torch.any(keep_mask):
            corr_xyz = corr_xyz[keep_mask]
            corr_xy = corr_xy[keep_mask]
            corr_err = corr_err[keep_mask]
            atlas_native = atlas_native[keep_mask]
            atlas_node_ids = atlas_node_ids[keep_mask]
            atlas_reliability_hint = atlas_reliability_hint[keep_mask]
            corr_trust = corr_trust[keep_mask]
            coords_ndc = coords_ndc[keep_mask]
            depth = depth[keep_mask]
            target_ndc = target_ndc[keep_mask]
            reproj_sq = reproj_sq[keep_mask]
        else:
            metrics["pose_geo_skip_reason"] = "below_min_correspondence_after_filtering"
            return (zero, metrics, *_no_sel) if return_selected else (zero, metrics)
    metrics["pose_geo_rejected_large_error"] = float(reject_mask.sum().detach().item())
    if metrics["pose_geo_trustworthy_corr"] == 0.0:
        metrics["pose_geo_trustworthy_corr"] = float(corr_xyz.shape[0])
    metrics["pose_geo_after_filter_corr"] = float(corr_xyz.shape[0])

    if corr_xyz.shape[0] > int(sample_count) > 0:
        keep = _select_budgeted_correspondences(
            corr_xy,
            reproj_px=torch.sqrt(reproj_sq.clamp_min(0.0)),
            corr_err=corr_err,
            sample_count=int(sample_count),
            image_width=int(viewpoint_camera.image_width),
            image_height=int(viewpoint_camera.image_height),
            atlas_native=atlas_native,
            atlas_reliability=atlas_reliability_hint,
            corr_trust=corr_trust,
            atlas_node_ids=atlas_node_ids,
        )
        corr_xyz = corr_xyz[keep]
        corr_xy = corr_xy[keep]
        corr_err = corr_err[keep]
        atlas_native = atlas_native[keep]
        atlas_node_ids = atlas_node_ids[keep]
        atlas_reliability_hint = atlas_reliability_hint[keep]
        corr_trust = corr_trust[keep]
        coords_ndc = coords_ndc[keep]
        depth = depth[keep]
        target_ndc = target_ndc[keep]
        reproj_sq = reproj_sq[keep]

    final_reproj_px = torch.sqrt(reproj_sq.clamp_min(0.0))
    metrics["pose_geo_selected_count"] = float(corr_xyz.shape[0])
    if atlas_node_ids.numel() > 0:
        valid_node_ids = atlas_node_ids[atlas_node_ids >= 0]
        metrics["pose_geo_selected_unique_node_count"] = float(torch.unique(valid_node_ids).numel()) if valid_node_ids.numel() > 0 else 0.0
    if final_reproj_px.numel() > 0:
        metrics["pose_geo_selected_median_px_error"] = float(torch.median(final_reproj_px).detach().item())
        metrics["pose_geo_median_px_error"] = metrics["pose_geo_selected_median_px_error"]

    atlas_positions = gaussians._atlas_positions.detach() if gaussians.has_atlas_bindings else corr_xyz
    if gaussians.has_atlas_bindings and atlas_positions.shape[0] > 0:
        valid_node_ids = (atlas_node_ids >= 0) & (atlas_node_ids < atlas_positions.shape[0])
        nearest = torch.empty((corr_xyz.shape[0],), dtype=torch.long, device=corr_xyz.device)
        if torch.any(valid_node_ids):
            nearest[valid_node_ids] = atlas_node_ids[valid_node_ids]
        if torch.any(~valid_node_ids):
            nearest[~valid_node_ids] = torch.cdist(corr_xyz[~valid_node_ids], atlas_positions).argmin(dim=1)
        atlas_radius = gaussians._atlas_radius[nearest]
        atlas_reliability = torch.maximum(
            gaussians.get_atlas_node_reliability_effective.detach()[nearest],
            atlas_reliability_hint.to(device=corr_xyz.device).clamp(0.0, 1.0) * 0.75,
        )
        sigma_geom = atlas_radius * (1.0 - 0.75 * atlas_reliability).clamp_min(0.05)
        sigma_geom = torch.where(atlas_native, sigma_geom * 0.80, sigma_geom)
    else:
        sigma_geom = 0.01 * depth.abs().clamp_min(1e-3)
        atlas_reliability = torch.ones_like(sigma_geom)

    if corr_err.numel() > 1:
        corr_median = torch.median(corr_err)
        corr_mad = torch.median((corr_err - corr_median).abs())
        corr_cap = torch.maximum(corr_median + 2.5 * 1.4826 * corr_mad, torch.quantile(corr_err, 0.85))
    else:
        corr_cap = corr_err.max() if corr_err.numel() else torch.tensor(1.0, dtype=corr_xyz.dtype, device=corr_xyz.device)
    corr_err_loss = torch.minimum(corr_err, corr_cap.clamp_min(0.5))
    pose_sigma = 0.35 + 0.65 * corr_err_loss
    denom = pose_sigma.square() + (sigma_geom / depth.abs().clamp_min(1e-3)).square() * float(max(width, height) ** 2) + 1e-6
    normalized = reproj_sq / denom
    weights = atlas_reliability * (0.35 + 0.65 * corr_trust) / corr_err_loss.clamp_min(1e-4)
    weights = torch.where(atlas_native, weights * 1.35, weights * 0.70)
    geo_loss = (weights * F.smooth_l1_loss(normalized, torch.zeros_like(normalized), reduction="none")).sum() / weights.sum().clamp_min(1e-6)

    metrics["pose_geo_loss"] = float(geo_loss.detach().item())
    metrics["pose_geo_num_corr"] = float(corr_xyz.shape[0])
    metrics["pose_geo_mean_px_error"] = float(final_reproj_px.mean().detach().item()) if final_reproj_px.numel() > 0 else 0.0
    metrics["pose_geo_atlas_native_selected_ratio"] = float(atlas_native.float().mean().detach().item()) if atlas_native.numel() > 0 else 0.0
    metrics["pose_geo_fallback_selected_ratio"] = 1.0 - metrics["pose_geo_atlas_native_selected_ratio"]
    metrics["pose_geo_selected_atlas_native_count"] = float(atlas_native.sum().detach().item()) if atlas_native.numel() > 0 else 0.0
    metrics["pose_geo_selected_fallback_count"] = float((~atlas_native).sum().detach().item()) if atlas_native.numel() > 0 else 0.0
    target_count = max(int(min_correspondences), 1)
    loaded_corr = max(float(metrics.get("pose_geo_loaded_corr", corr_xyz.shape[0])), 1.0)
    metrics["pose_geo_count_quality"] = float(min(float(corr_xyz.shape[0]) / float(target_count), 1.0))
    metrics["pose_geo_coverage_quality"] = float(min(float(metrics.get("pose_geo_in_frame_corr", corr_xyz.shape[0])) / loaded_corr, 1.0))
    metrics["pose_geo_residual_quality"] = float(math.exp(-max(float(metrics["pose_geo_median_px_error"]), 0.0) / 4.0))
    metrics["pose_geo_skip_reason"] = "ok"
    if return_selected:
        return float(geo_weight) * geo_loss, metrics, corr_xyz.detach(), corr_xy.detach()
    return float(geo_weight) * geo_loss, metrics


def _finish_pose_refinement_loss(total_loss, metrics, probe_context, return_probe_context: bool):
    if return_probe_context:
        return total_loss, metrics, probe_context
    return total_loss, metrics


def _downsample_image_like(image: torch.Tensor, downsample_factor: int):
    downsample_factor = int(max(downsample_factor, 1))
    if downsample_factor <= 1:
        return image
    height = max(int(image.shape[-2]) // downsample_factor, 1)
    width = max(int(image.shape[-1]) // downsample_factor, 1)
    return F.interpolate(
        image.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def compute_pose_fullframe_photo_loss(
    viewpoint_camera,
    template_image: torch.Tensor,
    photo_alpha: float,
    gradient_weight: float,
    downsample_factor: int = 1,
    use_gradient_term: bool = True,
):
    target_image = viewpoint_camera.original_image
    zero = target_image.new_zeros(())
    metrics = {
        "pose_fullframe_mode": 1.0,
        "pose_fullframe_num_pixels": 0.0,
        "pose_fullframe_l1": 0.0,
        "pose_fullframe_ssim": 0.0,
        "pose_fullframe_gradient": 0.0,
        "pose_fullframe_total": 0.0,
        "pose_fullframe_downsample_factor": float(max(int(downsample_factor), 1)),
        "pose_photo_loss": 0.0,
        "pose_photo_signal_strength": 0.0,
        "pose_gradient_loss": 0.0,
        "pose_ssim_loss": 0.0,
        "pose_patchfeat_loss": 0.0,
        "pose_mask_mean": 0.0,
        "pose_mask_grad_mean": 0.0,
        "pose_mask_depth_mean": 1.0,
        "pose_mask_nonzero_ratio": 0.0,
        "pose_patch_grad_observable_ratio": 0.0,
        "pose_patch_count_used": 0.0,
        "pose_num_samples": 0.0,
        "pose_view_support_mean": 1.0,
        "pose_active_safe_fraction": 1.0,
        "pose_total_loss_requires_grad": 0.0,
        "pose_photo_skip_reason": "inactive",
        "pose_geo_skip_reason": "geo_weight_disabled",
    }
    if template_image is None or not torch.is_tensor(template_image):
        metrics["pose_photo_skip_reason"] = "missing_template"
        return zero, metrics
    if template_image.ndim != 3 or target_image.ndim != 3:
        metrics["pose_photo_skip_reason"] = "invalid_image_shape"
        return zero, metrics

    downsample_factor = int(max(downsample_factor, 1))
    pred = template_image.to(dtype=target_image.dtype, device=target_image.device)
    target = target_image
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(
            pred.unsqueeze(0),
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    pred = _downsample_image_like(pred, downsample_factor)
    target = _downsample_image_like(target, downsample_factor)

    alpha_mask = getattr(viewpoint_camera, "alpha_mask", None)
    if torch.is_tensor(alpha_mask):
        mask = alpha_mask.to(dtype=target.dtype, device=target.device).clamp(0.0, 1.0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[-2:] != target.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).clamp(0.0, 1.0)
    else:
        mask = torch.ones((1, target.shape[-2], target.shape[-1]), dtype=target.dtype, device=target.device)

    mask_sum = mask.sum().clamp_min(1e-6)
    channel_count = max(int(target.shape[0]), 1)
    l1 = ((pred - target).abs() * mask).sum() / (mask_sum * float(channel_count))

    masked_pred = pred * mask
    masked_target = target * mask
    ssim_value = _patch_ssim(masked_pred.unsqueeze(0), masked_target.unsqueeze(0)).mean()
    ssim_loss = (1.0 - ssim_value).clamp_min(0.0)

    if bool(use_gradient_term) and float(gradient_weight) > 0.0:
        pred_grad = _gradient_map(pred)
        target_grad = _gradient_map(target.detach())
        grad_mask = mask.expand(pred_grad.shape[0], -1, -1)
        gradient_loss = ((pred_grad - target_grad).abs() * grad_mask).sum() / (
            mask_sum * float(max(int(pred_grad.shape[0]), 1))
        )
    else:
        gradient_loss = torch.zeros((), dtype=target.dtype, device=target.device)

    total = float(photo_alpha) * l1 + (1.0 - float(photo_alpha)) * ssim_loss + float(gradient_weight) * gradient_loss
    total = torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)
    num_pixels = int(target.shape[-2] * target.shape[-1])
    mask_nonzero = float((mask > 1e-6).float().mean().detach().item()) if mask.numel() > 0 else 0.0
    metrics.update({
        "pose_fullframe_num_pixels": float(num_pixels),
        "pose_fullframe_l1": float(l1.detach().item()),
        "pose_fullframe_ssim": float(ssim_value.detach().item()),
        "pose_fullframe_gradient": float(gradient_loss.detach().item()),
        "pose_fullframe_total": float(total.detach().item()),
        "pose_fullframe_downsample_factor": float(downsample_factor),
        "pose_photo_loss": float(total.detach().item()),
        "pose_photo_signal_strength": float(l1.detach().item()),
        "pose_gradient_loss": float(gradient_loss.detach().item()),
        "pose_ssim_loss": float(ssim_loss.detach().item()),
        "pose_mask_mean": float(mask.mean().detach().item()) if mask.numel() > 0 else 0.0,
        "pose_mask_grad_mean": 1.0 if bool(use_gradient_term) and float(gradient_weight) > 0.0 else 0.0,
        "pose_mask_nonzero_ratio": mask_nonzero,
        "pose_patch_grad_observable_ratio": mask_nonzero,
        "pose_patch_count_used": float(num_pixels),
        "pose_num_samples": float(num_pixels),
        "pose_total_loss_requires_grad": 1.0 if bool(getattr(total, "requires_grad", False)) else 0.0,
        "pose_photo_skip_reason": "ok" if num_pixels > 0 and mask_nonzero > 0.0 else "mask_empty",
    })
    return total, metrics


def _probe_context_tensor(probe_context: dict, key: str, reference: torch.Tensor, *, dtype=None, default=None):
    value = probe_context.get(key, default)
    if not torch.is_tensor(value):
        return default
    return value.to(device=reference.device, dtype=dtype if dtype is not None else reference.dtype)


def compute_pose_refinement_data_loss_from_context(
    viewpoint_camera,
    gaussians,
    probe_context: dict | None,
    photo_alpha: float | None = None,
    gradient_weight: float | None = None,
    patch_feature_weight: float | None = None,
    patch_radius_override: int | None = None,
    gradient_weight_scale: float = 1.0,
):
    zero = gaussians.get_xyz.new_zeros(())
    metrics = {
        "pose_probe_status": "inactive",
        "pose_probe_patch_count": 0.0,
        "pose_probe_patch_radius": 0.0,
        "pose_probe_mask_mean": 0.0,
        "pose_probe_photo_loss": 0.0,
        "pose_probe_photo_signal_strength": 0.0,
    }
    if not probe_context:
        metrics["pose_probe_status"] = "missing_context"
        return zero, metrics

    xyz = gaussians.get_xyz.detach()
    active_idx = probe_context.get("active_idx", None)
    if not torch.is_tensor(active_idx) or active_idx.numel() == 0 or xyz.numel() == 0:
        metrics["pose_probe_status"] = "empty_context"
        return zero, metrics
    active_idx = active_idx.to(device=xyz.device, dtype=torch.long).reshape(-1)
    active_idx = active_idx[(active_idx >= 0) & (active_idx < xyz.shape[0])]
    if active_idx.numel() == 0:
        metrics["pose_probe_status"] = "invalid_indices"
        return zero, metrics

    coords_all, depth_all, _ = _project_points(viewpoint_camera, xyz, require_in_frame=False)
    coords = coords_all[active_idx]
    depth = depth_all[active_idx].clamp_min(1e-4)
    sample_weight = _probe_context_tensor(probe_context, "sample_weight", coords, default=None)
    if sample_weight is None or sample_weight.numel() != active_idx.numel():
        sample_weight = coords.new_ones((active_idx.numel(),), dtype=coords.dtype)
    else:
        sample_weight = sample_weight.reshape(-1)

    finite = (
        torch.isfinite(coords).all(dim=1)
        & torch.isfinite(depth)
        & torch.isfinite(sample_weight)
        & (depth > float(getattr(viewpoint_camera, "znear", 0.0)))
    )
    if not torch.any(finite):
        metrics["pose_probe_status"] = "no_finite_projection"
        return zero, metrics

    active_idx = active_idx[finite]
    coords = coords[finite]
    depth = depth[finite]
    sample_weight = sample_weight[finite].clamp_min(1e-6)
    base_coords = _probe_context_tensor(probe_context, "coords_ndc", coords, default=None)
    if base_coords is None or base_coords.shape[0] != finite.shape[0]:
        base_coords = coords.detach()
    else:
        base_coords = base_coords[finite].detach()

    sample_patch_radius = int(
        max(
            0,
            patch_radius_override
            if patch_radius_override is not None
            else int(probe_context.get("patch_radius", probe_context.get("base_patch_radius", 1))),
        )
    )
    patch_size = sample_patch_radius * 2 + 1
    ray_mask = _probe_context_tensor(probe_context, "ray_mask", coords, default=None)
    if (
        ray_mask is not None
        and ray_mask.shape[0] == finite.shape[0]
        and ray_mask.shape[-2] == patch_size
        and ray_mask.shape[-1] == patch_size
        and patch_radius_override is None
    ):
        ray_mask = ray_mask[finite].detach()
        mask_metrics = {
            "mask_strength_mean": float(ray_mask.mean().detach().item()) if ray_mask.numel() > 0 else 0.0,
            "gradient_observable_ratio": 0.0,
        }
    else:
        uncertainty_ratio = _probe_context_tensor(probe_context, "uncertainty_ratio", coords, default=None)
        active_safety = _probe_context_tensor(probe_context, "active_safety", coords, default=None)
        effective_radius_px = _probe_context_tensor(probe_context, "effective_radius_px", coords, default=None)
        if uncertainty_ratio is None or uncertainty_ratio.numel() != finite.shape[0]:
            uncertainty_ratio = coords.new_zeros((finite.shape[0],))
        if active_safety is None or active_safety.numel() != finite.shape[0]:
            active_safety = coords.new_ones((finite.shape[0],))
        if effective_radius_px is None or effective_radius_px.numel() != finite.shape[0]:
            effective_radius_px = _estimate_screen_footprint_radius(
                viewpoint_camera,
                gaussians,
                active_idx,
                depth,
                uncertainty_ratio[finite] if uncertainty_ratio.shape[0] == finite.shape[0] else coords.new_zeros((active_idx.numel(),)),
                base_patch_radius=max(sample_patch_radius, 1),
            )
        else:
            effective_radius_px = effective_radius_px[finite]
        ray_mask, mask_metrics = _build_uncertainty_aware_ray_mask(
            viewpoint_camera,
            coords,
            depth,
            effective_radius_px,
            uncertainty_ratio[finite] if uncertainty_ratio.shape[0] == finite.shape[0] else coords.new_zeros((active_idx.numel(),)),
            active_safety[finite] if active_safety.shape[0] == finite.shape[0] else coords.new_ones((active_idx.numel(),)),
            sample_patch_radius=sample_patch_radius,
        )

    photo_alpha = float(probe_context.get("photo_alpha", 0.5) if photo_alpha is None else photo_alpha)
    gradient_weight = float(probe_context.get("gradient_weight", 0.0) if gradient_weight is None else gradient_weight)
    gradient_weight = max(gradient_weight * float(max(gradient_weight_scale, 0.0)), 0.0)
    patch_feature_weight = float(
        probe_context.get("patch_feature_weight", 0.0) if patch_feature_weight is None else patch_feature_weight
    )

    gt_patch = _sample_patches(viewpoint_camera.original_image, coords, sample_patch_radius)
    template_image = probe_context.get("template_image", None)
    if torch.is_tensor(template_image):
        template_image = template_image.to(device=coords.device, dtype=viewpoint_camera.original_image.dtype).detach()
    else:
        template_image = None

    inverse_pred_patch = None
    inverse_gt_patch = None
    if template_image is None:
        pred_rgb = SH2RGB(gaussians.get_features_dc.detach().squeeze(1))[active_idx].clamp(0.0, 1.0)
        pred_patch = pred_rgb[:, :, None, None].expand(-1, -1, gt_patch.shape[2], gt_patch.shape[3])
    else:
        pred_patch = _sample_patches(template_image, base_coords, sample_patch_radius)
        inverse_pred_patch = _sample_patches(template_image, coords, sample_patch_radius)
        inverse_gt_patch = _sample_patches(viewpoint_camera.original_image.detach(), base_coords, sample_patch_radius)

    masked_pred_patch = pred_patch * ray_mask[:, None, :, :]
    masked_gt_patch = gt_patch * ray_mask[:, None, :, :]
    l1_error = _weighted_patch_l1(pred_patch, gt_patch, ray_mask)
    ssim_term = 1.0 - _patch_ssim(masked_pred_patch, masked_gt_patch)
    if inverse_pred_patch is not None and inverse_gt_patch is not None:
        inverse_masked_pred = inverse_pred_patch * ray_mask[:, None, :, :]
        inverse_masked_gt = inverse_gt_patch * ray_mask[:, None, :, :]
        inverse_l1_error = _weighted_patch_l1(inverse_pred_patch, inverse_gt_patch, ray_mask)
        inverse_ssim_term = 1.0 - _patch_ssim(inverse_masked_pred, inverse_masked_gt)
        l1_error = 0.5 * (l1_error + inverse_l1_error)
        ssim_term = 0.5 * (ssim_term + inverse_ssim_term)

    if gradient_weight > 0.0:
        template_grad_source = template_image.detach() if template_image is not None else viewpoint_camera.original_image.detach()
        pred_grad = _sample_patches(_gradient_map(template_grad_source), base_coords if template_image is not None else coords, sample_patch_radius)
        target_grad = _sample_patches(_gradient_map(viewpoint_camera.original_image.detach()), coords, sample_patch_radius)
        grad_term = _weighted_patch_l1(pred_grad, target_grad, ray_mask)
        if template_image is not None:
            inverse_pred_grad = _sample_patches(_gradient_map(template_grad_source), coords, sample_patch_radius)
            inverse_target_grad = _sample_patches(
                _gradient_map(viewpoint_camera.original_image.detach()),
                base_coords,
                sample_patch_radius,
            )
            inverse_grad_term = _weighted_patch_l1(inverse_pred_grad, inverse_target_grad, ray_mask)
            grad_term = 0.5 * (grad_term + inverse_grad_term)
    else:
        grad_term = torch.zeros_like(l1_error)

    if patch_feature_weight > 0.0:
        pred_desc = _patch_descriptor(masked_pred_patch)
        gt_desc = _patch_descriptor(masked_gt_patch)
        patchfeat_term = torch.nan_to_num(
            (1.0 - F.cosine_similarity(pred_desc, gt_desc, dim=1, eps=1e-6)).clamp_min(0.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if inverse_pred_patch is not None and inverse_gt_patch is not None:
            inverse_pred_desc = _patch_descriptor(inverse_pred_patch * ray_mask[:, None, :, :])
            inverse_gt_desc = _patch_descriptor(inverse_gt_patch * ray_mask[:, None, :, :])
            inverse_patchfeat_term = torch.nan_to_num(
                (1.0 - F.cosine_similarity(inverse_pred_desc, inverse_gt_desc, dim=1, eps=1e-6)).clamp_min(0.0),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            patchfeat_term = 0.5 * (patchfeat_term + inverse_patchfeat_term)
    else:
        patchfeat_term = torch.zeros_like(l1_error)

    photo_error = (
        float(photo_alpha) * l1_error
        + (1.0 - float(photo_alpha)) * ssim_term
        + float(gradient_weight) * grad_term
        + float(patch_feature_weight) * patchfeat_term
    )
    finite_photo = torch.isfinite(photo_error) & torch.isfinite(sample_weight)
    if not torch.any(finite_photo):
        metrics["pose_probe_status"] = "gradient_or_mask_empty"
        return zero, metrics

    safe_photo_error = photo_error[finite_photo]
    safe_sample_weight = sample_weight[finite_photo]
    photo_loss = (safe_sample_weight * safe_photo_error).sum() / safe_sample_weight.sum().clamp_min(1e-6)
    metrics["pose_probe_status"] = "ok"
    metrics["pose_probe_patch_count"] = float(int(active_idx.shape[0]))
    metrics["pose_probe_patch_radius"] = float(sample_patch_radius)
    metrics["pose_probe_mask_mean"] = float(mask_metrics.get("mask_strength_mean", 0.0))
    metrics["pose_probe_photo_loss"] = float(photo_loss.detach().item())
    metrics["pose_probe_photo_signal_strength"] = float(
        torch.nan_to_num(safe_photo_error, nan=0.0, posinf=0.0, neginf=0.0).mean().detach().item()
    )
    return photo_loss, metrics


def compute_pose_refinement_losses(
    viewpoint_camera,
    gaussians,
    sample_count: int,
    geo_weight: float,
    photo_weight: float,
    template_image: torch.Tensor | None = None,
    photo_alpha: float = 0.5,
    gradient_weight: float = 0.0,
    patch_feature_weight: float = 0.0,
    patch_radius: int = 1,
    return_probe_context: bool = False,
):
    zero = gaussians.get_xyz.new_zeros(())
    probe_context = None
    metrics = {
        "pose_geo_loss": 0.0,
        "pose_photo_loss": 0.0,
        "pose_photo_signal_strength": 0.0,
        "pose_gradient_loss": 0.0,
        "pose_ssim_loss": 0.0,
        "pose_patchfeat_loss": 0.0,
        "pose_mask_mean": 0.0,
        "pose_mask_grad_mean": 0.0,
        "pose_mask_depth_mean": 0.0,
        "pose_view_support_mean": 0.0,
        "pose_active_safe_fraction": 1.0,
        "pose_uncertainty_ratio_mean": 0.0,
        "pose_effective_patch_radius": 0.0,
        "pose_selection_edge_mean": 0.0,
        "pose_num_samples": 0.0,
        "pose_stable_sample_fraction": 0.0,
        "pose_active_sample_fraction": 0.0,
        "pose_passive_safe_candidate_count": 0.0,
        "pose_passive_safe_sample_count": 0.0,
        "pose_passive_safe_sample_fraction": 0.0,
        "pose_passive_safe_trust_mean": 0.0,
        "pose_passive_safe_weight_scale": 0.35,
        "pose_passive_safe_reliability_mean": 0.0,
        "pose_passive_safe_support_mean": 0.0,
        "pose_passive_safe_ref_mean": 0.0,
        "pose_template_fixed_coords": 0.0,
        "pose_template_symmetric_coords": 0.0,
        "pose_coords_requires_grad": 0.0,
        "pose_gt_patch_requires_grad": 0.0,
        "pose_pred_patch_requires_grad": 0.0,
        "pose_inverse_pred_patch_requires_grad": 0.0,
        "pose_template_coords_requires_grad": 0.0,
        "pose_ray_mask_requires_grad": 0.0,
        "pose_patch_grad_observable_ratio": 0.0,
        "pose_total_loss_requires_grad": 0.0,
        "pose_geo_skip_reason": "inactive",
        "pose_photo_skip_reason": "inactive",
    }
    if (geo_weight <= 0.0 and photo_weight <= 0.0) or not gaussians.has_atlas_bindings:
        if not gaussians.has_atlas_bindings:
            metrics["pose_geo_skip_reason"] = "atlas_disabled"
            metrics["pose_photo_skip_reason"] = "atlas_disabled"
        else:
            if geo_weight <= 0.0:
                metrics["pose_geo_skip_reason"] = "geo_weight_disabled"
            if photo_weight <= 0.0:
                metrics["pose_photo_skip_reason"] = "photo_weight_disabled"
        return _finish_pose_refinement_loss(zero, metrics, probe_context, return_probe_context)

    xyz = gaussians.get_xyz.detach()
    atlas_state = gaussians.get_atlas_state.detach()
    passive_safe_mask, passive_safe_metrics = _compute_passive_safe_pose_mask(gaussians, atlas_state)
    metrics.update({
        "pose_passive_safe_candidate_count": float(passive_safe_mask.sum().detach().item()) if passive_safe_mask.numel() > 0 else 0.0,
        "pose_passive_safe_reliability_mean": passive_safe_metrics.get("passive_safe_reliability_mean", 0.0),
        "pose_passive_safe_support_mean": passive_safe_metrics.get("passive_safe_support_mean", 0.0),
        "pose_passive_safe_ref_mean": passive_safe_metrics.get("passive_safe_ref_mean", 0.0),
    })
    active_mask = (
        (atlas_state == GAUSSIAN_STATE_STABLE)
        | (atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
        | passive_safe_mask
    )
    if not torch.any(active_mask):
        metrics["pose_geo_skip_reason"] = "no_active_samples"
        metrics["pose_photo_skip_reason"] = "no_active_samples"
        return _finish_pose_refinement_loss(zero, metrics, probe_context, return_probe_context)

    coords, depth, valid = _project_points(viewpoint_camera, xyz)
    active_mask = active_mask & valid
    if not torch.any(active_mask):
        metrics["pose_geo_skip_reason"] = "mask_empty"
        metrics["pose_photo_skip_reason"] = "mask_empty"
        return _finish_pose_refinement_loss(zero, metrics, probe_context, return_probe_context)

    active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    selection_edge_mean = 0.0
    if active_idx.shape[0] > int(sample_count) > 0:
        scores = _compute_pose_selection_scores(gaussians, active_idx, viewpoint_camera)
        if photo_weight > 0.0:
            with torch.no_grad():
                edge_samples = _sample_map(_gradient_map(viewpoint_camera.original_image.detach()), coords[active_idx].detach())
                edge_channels = int(edge_samples.shape[1] // 2)
                edge_mag = torch.sqrt(
                    edge_samples[:, :edge_channels].square()
                    + edge_samples[:, edge_channels:].square()
                    + 1e-8
                ).mean(dim=1)
                if edge_mag.numel() > 0:
                    edge_scale = torch.quantile(edge_mag, 0.75).clamp_min(1e-4)
                    edge_score = (edge_mag / edge_scale).clamp(0.0, 2.0) * 0.5
                    scores = scores * (0.70 + 0.30 * edge_score.to(dtype=scores.dtype))
                    selection_edge_mean = float(edge_mag.mean().detach().item())
        candidate_state = gaussians.get_atlas_state.detach()[active_idx]
        keep_count = int(sample_count)
        selected_chunks: list[torch.Tensor] = []
        used_mask = torch.zeros((active_idx.shape[0],), dtype=torch.bool, device=active_idx.device)

        active_candidate_mask = candidate_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        stable_candidate_mask = candidate_state == GAUSSIAN_STATE_STABLE
        passive_candidate_mask = candidate_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
        active_quota = min(int(active_candidate_mask.sum().item()), max(int(round(0.30 * keep_count)), 0))
        if active_quota > 0:
            active_candidates = torch.nonzero(active_candidate_mask, as_tuple=False).squeeze(-1)
            active_keep = active_candidates[torch.topk(scores[active_candidates], k=active_quota, largest=True).indices]
            selected_chunks.append(active_keep)
            used_mask[active_keep] = True

        remaining = max(keep_count - int(sum(chunk.shape[0] for chunk in selected_chunks)), 0)
        passive_quota = min(int(passive_candidate_mask.sum().item()), max(int(round(0.15 * keep_count)), 0), remaining)
        if passive_quota > 0:
            passive_candidates = torch.nonzero(passive_candidate_mask & (~used_mask), as_tuple=False).squeeze(-1)
            if passive_candidates.numel() > 0:
                passive_keep = passive_candidates[
                    torch.topk(scores[passive_candidates], k=min(passive_quota, int(passive_candidates.numel())), largest=True).indices
                ]
                selected_chunks.append(passive_keep)
                used_mask[passive_keep] = True

        remaining = max(keep_count - int(sum(chunk.shape[0] for chunk in selected_chunks)), 0)
        stable_quota = min(int(stable_candidate_mask.sum().item()), remaining)
        if stable_quota > 0:
            stable_candidates = torch.nonzero(stable_candidate_mask & (~used_mask), as_tuple=False).squeeze(-1)
            if stable_candidates.numel() > 0:
                stable_keep = stable_candidates[
                    torch.topk(scores[stable_candidates], k=min(stable_quota, int(stable_candidates.numel())), largest=True).indices
                ]
                selected_chunks.append(stable_keep)
                used_mask[stable_keep] = True

        remaining = max(keep_count - int(sum(chunk.shape[0] for chunk in selected_chunks)), 0)
        if remaining > 0:
            fallback_candidates = torch.nonzero(~used_mask, as_tuple=False).squeeze(-1)
            if fallback_candidates.numel() > 0:
                fallback_keep = fallback_candidates[
                    torch.topk(scores[fallback_candidates], k=min(remaining, int(fallback_candidates.numel())), largest=True).indices
                ]
                selected_chunks.append(fallback_keep)

        keep = torch.cat(selected_chunks, dim=0) if selected_chunks else torch.topk(scores, k=keep_count, largest=True).indices
        active_idx = active_idx[keep]

    coords = coords[active_idx]
    depth = depth[active_idx].clamp_min(1e-4)
    sample_weight, selected_state, weight_aux = _compute_pose_sample_weights(gaussians, active_idx, viewpoint_camera)
    metrics["pose_selection_edge_mean"] = float(selection_edge_mean)
    metrics["pose_stable_sample_fraction"] = float((selected_state == GAUSSIAN_STATE_STABLE).float().mean().detach().item()) if selected_state.numel() > 0 else 0.0
    metrics["pose_active_sample_fraction"] = float((selected_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE).float().mean().detach().item()) if selected_state.numel() > 0 else 0.0
    passive_selected = selected_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
    metrics["pose_passive_safe_sample_count"] = float(passive_selected.sum().detach().item()) if passive_selected.numel() > 0 else 0.0
    metrics["pose_passive_safe_sample_fraction"] = float(passive_selected.float().mean().detach().item()) if selected_state.numel() > 0 else 0.0
    metrics["pose_passive_safe_trust_mean"] = float(sample_weight[passive_selected].mean().detach().item()) if torch.any(passive_selected) else 0.0
    metrics["pose_passive_safe_reliability_mean"] = float(weight_aux["reliability"][passive_selected].mean().detach().item()) if torch.any(passive_selected) else metrics["pose_passive_safe_reliability_mean"]

    metrics["pose_view_support_mean"] = float(weight_aux["view_support"].mean().detach().item()) if weight_aux["view_support"].numel() > 0 else 0.0
    metrics["pose_uncertainty_ratio_mean"] = float(weight_aux["uncertainty_ratio"].mean().detach().item()) if weight_aux["uncertainty_ratio"].numel() > 0 else 0.0
    active_state_mask = selected_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
    if torch.any(active_state_mask):
        metrics["pose_active_safe_fraction"] = float(weight_aux["active_safety"][active_state_mask].mean().detach().item())

    total_loss = zero

    if geo_weight > 0.0 and getattr(viewpoint_camera, "depth_reliable", False):
        gt_invdepth = _sample_map(viewpoint_camera.invdepthmap, coords).squeeze(-1)
        if viewpoint_camera.depth_confidence is not None:
            conf = _sample_map(viewpoint_camera.depth_confidence, coords).squeeze(-1).clamp(0.0, 1.0)
        else:
            conf = torch.ones_like(gt_invdepth)
        valid_depth = gt_invdepth > 0
        if torch.any(valid_depth):
            pred_invdepth = depth[valid_depth].reciprocal()
            sigma_support = gaussians.get_center_sigma_support.detach()[active_idx[valid_depth]].squeeze(-1)
            sigma_parallel = gaussians.get_center_sigma_parallel.detach()[active_idx[valid_depth]].squeeze(-1)
            geom_sigma = torch.where(selected_state[valid_depth] == GAUSSIAN_STATE_UNSTABLE_ACTIVE, sigma_parallel, sigma_support)
            pose_sigma = 0.01 + 0.05 * (1.0 - conf[valid_depth])
            denom = geom_sigma.square() + pose_sigma.square() + 1e-6
            normalized_residual = (pred_invdepth - gt_invdepth[valid_depth]) / torch.sqrt(denom)
            geo_error = F.smooth_l1_loss(normalized_residual, torch.zeros_like(normalized_residual), reduction="none")
            geo_weight_tensor = sample_weight[valid_depth] * conf[valid_depth]
            finite_geo = torch.isfinite(geo_error) & torch.isfinite(geo_weight_tensor)
            if torch.any(finite_geo):
                geo_error = geo_error[finite_geo]
                geo_weight_tensor = geo_weight_tensor[finite_geo]
                geo_loss = (geo_weight_tensor * geo_error).sum() / geo_weight_tensor.sum().clamp_min(1e-6)
                total_loss = total_loss + float(geo_weight) * geo_loss
                metrics["pose_geo_loss"] = float(geo_loss.detach().item())
                metrics["pose_geo_skip_reason"] = "ok"
        elif metrics["pose_geo_skip_reason"] == "inactive":
            metrics["pose_geo_skip_reason"] = "depth_mask_empty"
    elif metrics["pose_geo_skip_reason"] == "inactive":
        metrics["pose_geo_skip_reason"] = "geo_weight_disabled"

    if photo_weight > 0.0:
        # When B2 template is provided, boost selection for high-error regions so that
        # patch sampling focuses on areas where the current pose causes misregistration.
        if template_image is not None and active_idx.shape[0] > 0:
            with torch.no_grad():
                error_boost = coords.new_empty((0,))
                active_ndc = coords.detach()
                t_px = _sample_map(template_image, active_ndc)
                g_px = _sample_map(viewpoint_camera.original_image.detach(), active_ndc)
                pixel_error = (t_px - g_px).abs().mean(dim=1).clamp_min(0.0)
                if pixel_error.numel() > 0:
                    err_scale = torch.quantile(pixel_error, 0.75).clamp_min(1e-5)
                    error_boost = (pixel_error / err_scale).clamp(0.0, 3.0)
                    # Re-weight scores from _compute_pose_selection_scores
                    # (scores may not be defined yet if no subsampling was needed)
                    if active_idx.shape[0] > int(sample_count) > 0:
                        _raw_scores = _compute_pose_selection_scores(gaussians, active_idx, viewpoint_camera)
                        _boosted = (_raw_scores * (0.50 + 0.50 * error_boost.to(dtype=_raw_scores.dtype))).clamp_min(1e-4)
                        keep_count = int(sample_count)
                        keep_local = torch.topk(_boosted, k=min(keep_count, int(active_idx.shape[0])), largest=True).indices
                        active_idx = active_idx[keep_local]
                        coords = coords[keep_local]
                        depth = depth[keep_local].clamp_min(1e-4)
                        sample_weight, selected_state, weight_aux = _compute_pose_sample_weights(gaussians, active_idx, viewpoint_camera)
                        metrics["pose_stable_sample_fraction"] = float((selected_state == GAUSSIAN_STATE_STABLE).float().mean().detach().item()) if selected_state.numel() > 0 else 0.0
                        metrics["pose_active_sample_fraction"] = float((selected_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE).float().mean().detach().item()) if selected_state.numel() > 0 else 0.0
                        passive_selected = selected_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
                        metrics["pose_passive_safe_sample_count"] = float(passive_selected.sum().detach().item()) if passive_selected.numel() > 0 else 0.0
                        metrics["pose_passive_safe_sample_fraction"] = float(passive_selected.float().mean().detach().item()) if selected_state.numel() > 0 else 0.0
                        metrics["pose_passive_safe_trust_mean"] = float(sample_weight[passive_selected].mean().detach().item()) if torch.any(passive_selected) else 0.0
                        metrics["pose_passive_safe_reliability_mean"] = float(weight_aux["reliability"][passive_selected].mean().detach().item()) if torch.any(passive_selected) else metrics["pose_passive_safe_reliability_mean"]
                        metrics["pose_view_support_mean"] = float(weight_aux["view_support"].mean().detach().item()) if weight_aux["view_support"].numel() > 0 else 0.0
                        metrics["pose_uncertainty_ratio_mean"] = float(weight_aux["uncertainty_ratio"].mean().detach().item()) if weight_aux["uncertainty_ratio"].numel() > 0 else 0.0
                        active_state_mask = selected_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
                        if torch.any(active_state_mask):
                            metrics["pose_active_safe_fraction"] = float(weight_aux["active_safety"][active_state_mask].mean().detach().item())
                metrics["pose_b2_error_boost_mean"] = float(error_boost.mean().detach().item()) if error_boost.numel() > 0 else 0.0

        effective_radius_px = _estimate_screen_footprint_radius(
            viewpoint_camera,
            gaussians,
            active_idx,
            depth,
            weight_aux["uncertainty_ratio"],
            base_patch_radius=int(patch_radius),
        )
        sample_patch_radius = int(max(min(int(torch.ceil(effective_radius_px.max()).item()), 6), int(max(patch_radius, 1))))
        metrics["pose_effective_patch_radius"] = float(effective_radius_px.mean().detach().item()) if effective_radius_px.numel() > 0 else 0.0

        gt_patch = _sample_patches(viewpoint_camera.original_image, coords, sample_patch_radius)
        metrics["pose_coords_requires_grad"] = 1.0 if bool(getattr(coords, "requires_grad", False)) else 0.0
        metrics["pose_gt_patch_requires_grad"] = 1.0 if bool(getattr(gt_patch, "requires_grad", False)) else 0.0
        template_coords = coords.detach() if template_image is not None else coords
        ray_mask, mask_metrics = _build_uncertainty_aware_ray_mask(
            viewpoint_camera,
            coords,
            depth,
            effective_radius_px,
            weight_aux["uncertainty_ratio"],
            weight_aux["active_safety"],
            sample_patch_radius=sample_patch_radius,
        )
        metrics["pose_mask_mean"] = mask_metrics["mask_strength_mean"]
        metrics["pose_mask_grad_mean"] = mask_metrics["gradient_gate_mean"]
        metrics["pose_mask_depth_mean"] = mask_metrics["depth_gate_mean"]
        metrics["pose_patch_grad_observable_ratio"] = float(mask_metrics.get("gradient_observable_ratio", 0.0))
        metrics["pose_mask_nonzero_ratio"] = float((ray_mask > 1e-6).float().mean().detach().item()) if ray_mask.numel() > 0 else 0.0
        metrics["pose_patch_count_used"] = float(int(active_idx.shape[0]))
        metrics["pose_ray_mask_requires_grad"] = 1.0 if bool(getattr(ray_mask, "requires_grad", False)) else 0.0
        if return_probe_context:
            probe_context = {
                "active_idx": active_idx.detach().clone(),
                "coords_ndc": coords.detach().clone(),
                "depth": depth.detach().clone(),
                "sample_weight": sample_weight.detach().clone(),
                "selected_state": selected_state.detach().clone(),
                "uncertainty_ratio": weight_aux["uncertainty_ratio"].detach().clone(),
                "active_safety": weight_aux["active_safety"].detach().clone(),
                "effective_radius_px": effective_radius_px.detach().clone(),
                "ray_mask": ray_mask.detach().clone(),
                "patch_radius": int(sample_patch_radius),
                "base_patch_radius": int(max(patch_radius, 1)),
                "photo_alpha": float(photo_alpha),
                "gradient_weight": float(gradient_weight),
                "patch_feature_weight": float(patch_feature_weight),
                "photo_components_enabled": {
                    "l1": 1.0,
                    "ssim": 1.0,
                    "gradient": 1.0 if float(gradient_weight) > 0.0 else 0.0,
                    "patch_feature": 1.0 if float(patch_feature_weight) > 0.0 else 0.0,
                    "template": 1.0 if template_image is not None else 0.0,
                },
                "template_image": template_image.detach() if template_image is not None else None,
            }

        inverse_pred_patch = None
        inverse_gt_patch = None
        if template_image is None:
            pred_rgb = SH2RGB(gaussians.get_features_dc.detach().squeeze(1))[active_idx].clamp(0.0, 1.0)
            pred_patch = pred_rgb[:, :, None, None].expand(-1, -1, gt_patch.shape[2], gt_patch.shape[3])
        else:
            template_image = template_image.detach()
            pred_patch = _sample_patches(template_image, template_coords, sample_patch_radius)
            inverse_pred_patch = _sample_patches(template_image, coords, sample_patch_radius)
            inverse_gt_patch = _sample_patches(viewpoint_camera.original_image.detach(), coords.detach(), sample_patch_radius)
            metrics["pose_template_fixed_coords"] = 1.0
            metrics["pose_template_symmetric_coords"] = 1.0
            metrics["pose_template_coords_requires_grad"] = 1.0 if bool(getattr(coords, "requires_grad", False)) else 0.0
            metrics["pose_inverse_pred_patch_requires_grad"] = 1.0 if bool(getattr(inverse_pred_patch, "requires_grad", False)) else 0.0
        metrics["pose_pred_patch_requires_grad"] = 1.0 if bool(getattr(pred_patch, "requires_grad", False)) else 0.0

        masked_pred_patch = pred_patch * ray_mask[:, None, :, :]
        masked_gt_patch = gt_patch * ray_mask[:, None, :, :]
        l1_error = _weighted_patch_l1(pred_patch, gt_patch, ray_mask)
        ssim_term = 1.0 - _patch_ssim(masked_pred_patch, masked_gt_patch)
        if inverse_pred_patch is not None and inverse_gt_patch is not None:
            inverse_masked_pred = inverse_pred_patch * ray_mask[:, None, :, :]
            inverse_masked_gt = inverse_gt_patch * ray_mask[:, None, :, :]
            inverse_l1_error = _weighted_patch_l1(inverse_pred_patch, inverse_gt_patch, ray_mask)
            inverse_ssim_term = 1.0 - _patch_ssim(inverse_masked_pred, inverse_masked_gt)
            # Keep the same patch objective, but use both forward and inverse
            # compositional sampling so either image can provide pose gradient.
            l1_error = 0.5 * (l1_error + inverse_l1_error)
            ssim_term = 0.5 * (ssim_term + inverse_ssim_term)

        if gradient_weight > 0.0:
            template_grad_source = template_image.detach() if template_image is not None else viewpoint_camera.original_image.detach()
            pred_grad = _sample_patches(_gradient_map(template_grad_source), template_coords, sample_patch_radius)
            target_grad = _sample_patches(_gradient_map(viewpoint_camera.original_image.detach()), coords, sample_patch_radius)
            grad_term = _weighted_patch_l1(pred_grad, target_grad, ray_mask)
            if template_image is not None:
                inverse_pred_grad = _sample_patches(_gradient_map(template_grad_source), coords, sample_patch_radius)
                inverse_target_grad = _sample_patches(
                    _gradient_map(viewpoint_camera.original_image.detach()),
                    coords.detach(),
                    sample_patch_radius,
                )
                inverse_grad_term = _weighted_patch_l1(inverse_pred_grad, inverse_target_grad, ray_mask)
                grad_term = 0.5 * (grad_term + inverse_grad_term)
        else:
            grad_term = torch.zeros_like(l1_error)

        if patch_feature_weight > 0.0:
            pred_desc = _patch_descriptor(masked_pred_patch)
            gt_desc = _patch_descriptor(masked_gt_patch)
            patchfeat_term = torch.nan_to_num(
                (1.0 - F.cosine_similarity(pred_desc, gt_desc, dim=1, eps=1e-6)).clamp_min(0.0),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if inverse_pred_patch is not None and inverse_gt_patch is not None:
                inverse_pred_desc = _patch_descriptor(inverse_pred_patch * ray_mask[:, None, :, :])
                inverse_gt_desc = _patch_descriptor(inverse_gt_patch * ray_mask[:, None, :, :])
                inverse_patchfeat_term = torch.nan_to_num(
                    (1.0 - F.cosine_similarity(inverse_pred_desc, inverse_gt_desc, dim=1, eps=1e-6)).clamp_min(0.0),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                patchfeat_term = 0.5 * (patchfeat_term + inverse_patchfeat_term)
        else:
            patchfeat_term = torch.zeros_like(l1_error)

        photo_error = (
            float(photo_alpha) * l1_error
            + (1.0 - float(photo_alpha)) * ssim_term
            + float(gradient_weight) * grad_term
            + float(patch_feature_weight) * patchfeat_term
        )
        finite_photo = torch.isfinite(photo_error) & torch.isfinite(sample_weight)
        if torch.any(finite_photo):
            safe_photo_error = photo_error[finite_photo]
            safe_sample_weight = sample_weight[finite_photo]
            photo_loss = (safe_sample_weight * safe_photo_error).sum() / safe_sample_weight.sum().clamp_min(1e-6)
            total_loss = total_loss + float(photo_weight) * photo_loss
            metrics["pose_photo_loss"] = float(photo_loss.detach().item())
            metrics["pose_photo_signal_strength"] = float(torch.nan_to_num(safe_photo_error, nan=0.0, posinf=0.0, neginf=0.0).mean().detach().item())
            metrics["pose_gradient_loss"] = float(torch.nan_to_num(grad_term[finite_photo], nan=0.0, posinf=0.0, neginf=0.0).mean().detach().item()) if grad_term.numel() > 0 else 0.0
            metrics["pose_ssim_loss"] = float(torch.nan_to_num(ssim_term[finite_photo], nan=0.0, posinf=0.0, neginf=0.0).mean().detach().item()) if ssim_term.numel() > 0 else 0.0
            metrics["pose_patchfeat_loss"] = float(torch.nan_to_num(patchfeat_term[finite_photo], nan=0.0, posinf=0.0, neginf=0.0).mean().detach().item()) if patchfeat_term.numel() > 0 else 0.0
            metrics["pose_photo_skip_reason"] = "ok"
        elif metrics["pose_photo_skip_reason"] == "inactive":
            metrics["pose_photo_skip_reason"] = "gradient_or_mask_empty"
    elif metrics["pose_photo_skip_reason"] == "inactive":
        metrics["pose_photo_skip_reason"] = "photo_weight_disabled"

    metrics["pose_num_samples"] = float(active_idx.shape[0])
    metrics["pose_total_loss_requires_grad"] = 1.0 if bool(getattr(total_loss, "requires_grad", False)) else 0.0
    return _finish_pose_refinement_loss(total_loss, metrics, probe_context, return_probe_context)
