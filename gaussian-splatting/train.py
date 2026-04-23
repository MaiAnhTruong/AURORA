#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from __future__ import annotations

import os
import json
import math
import time
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.foundation_atlas_exploration import compute_exploration_slab_loss
from scene.foundation_atlas_regularization import compute_atlas_regularization
from scene.foundation_atlas_runtime import (
    compute_render_validation_proxies,
    sample_gaussian_photometric_residuals,
)
from scene.foundation_atlas_variational import (
    build_variational_subspace,
    compute_local_exact_kl,
    sample_antithetic_center_offsets,
)
from scene.foundation_atlas_pose import (
    clamp_camera_pose_delta,
    compute_dynamic_pose_trust_region,
    compute_pose_geometric_correspondence_loss,
    compute_pose_fullframe_photo_loss,
    compute_pose_refinement_data_loss_from_context,
    compute_pose_quality_score,
    compute_pose_refinement_losses,
    compute_pose_trust_region_loss,
    evaluate_correspondence_reprojection_px,
    measure_pose_delta,
    reset_camera_pose_delta,
    summarize_pose_correspondence_budget,
    should_enable_pose_photometric_refinement,
    should_enable_pose_refinement,
    should_freeze_pose_refinement,
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

try:
    from lpipsPyTorch.modules.lpips import LPIPS
    LPIPS_AVAILABLE = True
    LPIPS_IMPORT_ERROR = ""
except Exception as _lpips_import_err:
    LPIPS_AVAILABLE = False
    LPIPS_IMPORT_ERROR = str(_lpips_import_err)
    print(f"[WARNING] lpipsPyTorch unavailable, LPIPS will be marked unavailable: {_lpips_import_err}")


_LPIPS_MODELS = {}
_LPIPS_WARNED = [False]


def append_training_log(model_path, record):
    log_path = os.path.join(model_path, "training_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(_serialize_json_payload(record), sort_keys=True, default=str) + "\n")


def append_quality_metrics_log(model_path, iteration: int, validation_summary: dict | None, extra_metrics: dict | None = None):
    if not validation_summary and not extra_metrics:
        return
    record = {"iteration": int(iteration)}
    if validation_summary:
        for split_name in ("train", "test"):
            split_metrics = validation_summary.get(split_name)
            if not split_metrics:
                continue
            for metric_name, metric_value in split_metrics.items():
                record[f"{split_name}_{metric_name}"] = serialize_metric_value(metric_value)
    if extra_metrics:
        for metric_name, metric_value in extra_metrics.items():
            record[str(metric_name)] = serialize_metric_value(metric_value)
    if len(record) == 1:
        return
    log_path = os.path.join(model_path, "quality_metrics.jsonl")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, sort_keys=True) + "\n")


def append_pose_refinement_log(model_path, record):
    log_path = os.path.join(model_path, "pose_refinement_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(_serialize_json_payload(record), sort_keys=True, default=str) + "\n")


def serialize_metric_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {str(key): serialize_metric_value(metric_value) for key, metric_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_metric_value(metric_value) for metric_value in value]
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            value = value.detach().item()
        else:
            return serialize_metric_value(value.detach().cpu().tolist())
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(numeric):
        return 0.0
    return numeric


def _safe_log_scalar(value, default: float = 0.0):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default), True
        detached = value.detach()
        had_nonfinite = not bool(torch.isfinite(detached).all().item())
        safe_value = torch.nan_to_num(
            detached,
            nan=float(default),
            posinf=float(default),
            neginf=float(default),
        )
        if safe_value.numel() == 1:
            return float(safe_value.item()), had_nonfinite
        return float(safe_value.mean().item()), had_nonfinite
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default), False
    if not math.isfinite(numeric):
        return float(default), True
    return numeric, False


def _record_controller_ms(metrics: dict | None, name: str, start_time: float):
    if metrics is None:
        return
    metrics[f"controller_{name}_wall_ms"] = float(max(time.perf_counter() - start_time, 0.0) * 1000.0)


def _run_runtime_observation_update(
    viewpoint_cam,
    gaussians,
    rendered_image,
    gt_image,
    radii,
    rendered_invdepth,
    opt,
    camera_index: int,
    warmup_only: bool,
    refresh_pending: bool,
):
    per_gaussian_residual, visible_mask = sample_gaussian_photometric_residuals(
        viewpoint_cam,
        gaussians,
        rendered_image,
        gt_image,
        radii,
        rendered_invdepth=rendered_invdepth,
    )
    return gaussians.update_atlas_runtime_stats(
        per_gaussian_residual,
        visible_mask,
        ema_decay=opt.atlas_state_ema_decay,
        drift_radius_mult=opt.atlas_drift_radius_mult,
        camera_index=camera_index,
        high_residual_threshold=opt.atlas_stable_residual_threshold,
        warmup_only=bool(warmup_only or refresh_pending),
    )


def _run_atlas_state_controller(
    *,
    scene,
    viewpoint_cam,
    rendered_image,
    gt_image,
    radii,
    render_pkg,
    opt,
    dataset,
    iteration: int,
    camera_index: int,
    atlas_phase: dict,
    prune_controls: dict,
    pose_gate_metrics: dict,
    atlas_state_metrics: dict,
    atlas_uncertainty_metrics: dict | None,
    train_camera_centers,
    in_warmup: bool,
    warmup_only: bool,
    refresh_pending: bool,
    main_phase: bool,
    main_phase_ready: bool,
    current_pose_delta: dict,
    promote_to_active_threshold: float,
    demote_to_passive_threshold: float,
    active_min_lifetime_iters: int,
    active_quota_ratio: float,
    active_quota_min: int,
    active_quota_max: int,
    active_max_lifetime_iters: int,
    active_nonimprove_iters: int,
    controller_timing_metrics: dict,
):
    controller_start = time.perf_counter()
    atlas_runtime_metrics = None
    atlas_refresh_metrics = None
    atlas_loss_schedule = None

    if scene.gaussians.has_atlas_bindings:
        atlas_runtime_metrics = _run_runtime_observation_update(
            viewpoint_cam,
            scene.gaussians,
            rendered_image,
            gt_image,
            radii,
            rendered_invdepth=render_pkg.get("depth"),
            opt=opt,
            camera_index=camera_index,
            warmup_only=warmup_only,
            refresh_pending=refresh_pending,
        )
        if refresh_pending:
            atlas_refresh_metrics = scene.gaussians.refresh_atlas_after_warmup(
                alpha=opt.atlas_refresh_alpha,
                gamma=opt.atlas_refresh_gamma,
                min_reliability=opt.atlas_refresh_min_reliability,
                min_visibility=opt.atlas_refresh_min_visibility,
                refresh_low_band_power=float(getattr(opt, "atlas_refresh_low_band_power", 0.65)),
                refresh_high_band_power=float(getattr(opt, "atlas_refresh_mid_band_power", 1.05)),
                refresh_support_consistency_weight=float(getattr(opt, "atlas_refresh_support_consistency_weight", 0.12)),
                refresh_visibility_weight=float(getattr(opt, "atlas_refresh_visibility_weight", 0.14)),
                refresh_override_min_evidence=float(getattr(opt, "atlas_refresh_override_min_evidence", 0.18)),
            )
        if atlas_phase["enable_state_update"]:
            atlas_state_metrics = scene.gaussians.update_atlas_states(
                surface_stable_min=dataset.atlas_surface_stable_reliability,
                edge_stable_min=dataset.atlas_edge_stable_reliability,
                min_visibility_ema=opt.atlas_state_min_visibility,
                stable_residual_threshold=opt.atlas_stable_residual_threshold,
                activate_threshold=opt.atlas_activate_threshold,
                deactivate_threshold=opt.atlas_deactivate_threshold,
                activate_min_high_residual_iters=opt.atlas_activate_min_high_residual_iters,
                recover_low_residual_iters=opt.atlas_state_low_residual_iters,
                drift_activate_iters=opt.atlas_state_drift_iters,
                out_of_anchor_drift_iters=opt.atlas_state_out_of_anchor_iters,
                out_of_anchor_gc_failures=opt.atlas_state_out_of_anchor_gc_failures,
                state_cooldown_iters=opt.atlas_state_cooldown_iters,
                active_min_lifetime_iters=active_min_lifetime_iters,
                active_quota_ratio=active_quota_ratio,
                active_quota_min=active_quota_min,
                active_quota_max=active_quota_max,
                min_active_opacity=opt.atlas_state_min_active_opacity,
                promote_to_active_threshold=promote_to_active_threshold,
                demote_to_passive_threshold=demote_to_passive_threshold,
                active_max_lifetime_iters=active_max_lifetime_iters,
                active_nonimprove_iters=active_nonimprove_iters,
                passive_to_stable_reliability_min=float(getattr(opt, "atlas_passive_to_stable_reliability_min", 0.18)),
                passive_to_stable_support_consistency_min=float(getattr(opt, "atlas_passive_to_stable_support_consistency_min", 0.28)),
                passive_to_stable_drift_max=float(getattr(opt, "atlas_passive_to_stable_drift_max", 1.15)),
                passive_to_stable_photo_ema_max=float(getattr(opt, "atlas_passive_to_stable_photo_ema_max", 0.045)),
            )
            atlas_state_metrics["state_frozen"] = 0.0
        else:
            atlas_state_metrics = scene.gaussians.summarize_atlas_state_metrics()
            atlas_state_metrics["state_frozen"] = 1.0

        atlas_loss_schedule = _compute_atlas_loss_schedule(
            iteration,
            opt,
            scene.gaussians,
            atlas_state_metrics,
        )
        atlas_runtime_metrics = atlas_runtime_metrics or {}
        atlas_runtime_metrics.update(scene.gaussians.summarize_atlas_reliability_state())
        atlas_runtime_metrics.update(scene.gaussians.summarize_atlas_refresh_snapshot())
        atlas_runtime_metrics.update({
            "phase_in_warmup": 1.0 if in_warmup else 0.0,
            "phase_warmup_only": 1.0 if warmup_only else 0.0,
            "phase_main_phase": 1.0 if main_phase else 0.0,
            "phase_refresh_pending": 1.0 if refresh_pending else 0.0,
            "phase_main_phase_ready": 1.0 if main_phase_ready else 0.0,
            "phase_enable_pose_b1": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
            "phase_enable_pose_b2": 1.0 if atlas_phase["enable_pose_b2"] else 0.0,
            "phase_enable_densify": 1.0 if atlas_phase["enable_densify"] else 0.0,
            "phase_enable_prune": 1.0 if atlas_phase["enable_prune"] else 0.0,
            "phase_enable_soft_prune": 1.0 if atlas_phase["enable_soft_prune"] else 0.0,
            "phase_enable_gc": 1.0 if atlas_phase["enable_gc"] else 0.0,
            "phase_enable_state_update": 1.0 if atlas_phase["enable_state_update"] else 0.0,
            "phase_enable_mc": 1.0 if atlas_phase["enable_mc"] else 0.0,
            "phase_enable_explore": 1.0 if atlas_phase["enable_explore"] else 0.0,
            "phase_pose_translation_norm": float(current_pose_delta["translation_norm"]),
            "phase_pose_rotation_degrees": float(current_pose_delta["rotation_degrees"]),
        })
        if atlas_uncertainty_metrics is not None:
            atlas_runtime_metrics.update(atlas_uncertainty_metrics)
        if atlas_refresh_metrics is not None:
            atlas_runtime_metrics.update(
                {
                    metric_name: metric_value
                    for metric_name, metric_value in atlas_refresh_metrics.items()
                    if metric_name
                    in {
                        "refresh_override_count",
                        "refresh_override_ratio",
                        "refresh_std_before",
                        "refresh_std_after",
                        "refresh_std_before_after",
                        "runtime_override_count",
                        "runtime_override_ratio",
                        "runtime_reliability_std",
                    }
                }
            )
        atlas_runtime_metrics.update({f"prune_{k}": v for k, v in prune_controls.items()})
        atlas_runtime_metrics.update(pose_gate_metrics)
        atlas_runtime_metrics.update({f"state_{k}": v for k, v in atlas_state_metrics.items()})
        if atlas_loss_schedule is not None:
            atlas_runtime_metrics.update({f"schedule_{k}": v for k, v in atlas_loss_schedule.items()})

    _record_controller_ms(controller_timing_metrics, "runtime_state", controller_start)
    return atlas_runtime_metrics, atlas_refresh_metrics, atlas_state_metrics, atlas_loss_schedule


def _run_densify_prune_gc_controller(
    *,
    scene,
    gaussians,
    viewpoint_cam,
    iteration: int,
    opt,
    dataset,
    atlas_phase: dict,
    atlas_runtime_metrics: dict | None,
    atlas_state_metrics: dict,
    densify_metrics: dict,
    atlas_gc_metrics: dict | None,
    prune_controls: dict,
    pose_runtime_state: dict,
    densify_runtime_state: dict,
    densify_radii,
    densify_visibility_filter,
    densify_viewspace_point_tensor,
    train_camera_centers,
    train_camera_centers_cache,
    active_min_lifetime_iters: int,
    active_quota_ratio: float,
    active_quota_min: int,
    active_quota_max: int,
    promote_to_active_threshold: float,
    demote_to_passive_threshold: float,
    active_max_lifetime_iters: int,
    active_nonimprove_iters: int,
    controller_timing_metrics: dict,
    run_gc: bool = True,
    run_densify: bool = True,
    gc_pruned_this_iter: bool = False,
    timing_name: str = "densify_prune_gc",
):
    controller_start = time.perf_counter()
    gc_interval = max(int(getattr(opt, "atlas_gc_interval", 0)), 0)
    gc_due = bool(
        run_gc
        and scene.gaussians.has_atlas_bindings
        and iteration < opt.densify_until_iter
        and atlas_phase["enable_gc"]
        and gc_interval > 0
        and iteration % gc_interval == 0
    )
    if scene.gaussians.has_atlas_bindings:
        if atlas_runtime_metrics is None:
            atlas_runtime_metrics = {}
        atlas_runtime_metrics["gc_interval"] = float(gc_interval)
        atlas_runtime_metrics["gc_due"] = 1.0 if gc_due else 0.0
        atlas_runtime_metrics["gc_ran"] = 0.0

    if gc_due:
        atlas_gc_metrics = scene.gaussians.run_atlas_gc(
            reattach_radius_mult=opt.atlas_reattach_radius_mult,
            surface_stable_min=dataset.atlas_surface_stable_reliability,
            edge_stable_min=dataset.atlas_edge_stable_reliability,
            min_visibility_ema=opt.atlas_state_min_visibility,
            stable_residual_threshold=opt.atlas_stable_residual_threshold,
            activate_threshold=opt.atlas_activate_threshold,
            deactivate_threshold=opt.atlas_deactivate_threshold,
            activate_min_high_residual_iters=opt.atlas_activate_min_high_residual_iters,
            recover_low_residual_iters=opt.atlas_state_low_residual_iters,
            drift_activate_iters=opt.atlas_state_drift_iters,
            out_of_anchor_drift_iters=opt.atlas_state_out_of_anchor_iters,
            out_of_anchor_gc_failures=opt.atlas_state_out_of_anchor_gc_failures,
            state_cooldown_iters=opt.atlas_state_cooldown_iters,
            active_min_lifetime_iters=active_min_lifetime_iters,
            active_quota_ratio=active_quota_ratio,
            active_quota_min=active_quota_min,
            active_quota_max=active_quota_max,
            min_active_opacity=opt.atlas_state_min_active_opacity,
            max_reattach_failures=opt.atlas_gc_max_reattach_failures,
            forced_prune_opacity=opt.atlas_out_of_anchor_prune_opacity,
            retry_pending=bool(getattr(opt, "atlas_gc_retry_pending", True)),
            promote_to_active_threshold=promote_to_active_threshold,
            demote_to_passive_threshold=demote_to_passive_threshold,
            active_max_lifetime_iters=active_max_lifetime_iters,
            active_nonimprove_iters=active_nonimprove_iters,
        )
        gc_pruned_this_iter = (
            float(atlas_gc_metrics.get("pending_prune_count", 0.0)) > 0.0
            or float(atlas_gc_metrics.get("prune_after_gc", 0.0)) > 0.0
        )
        if atlas_runtime_metrics is not None:
            atlas_runtime_metrics["gc_ran"] = 1.0
            atlas_runtime_metrics["gc_pruned_this_iter"] = 1.0 if gc_pruned_this_iter else 0.0
        atlas_state_metrics.update(scene.gaussians.summarize_atlas_state_metrics())

    if run_densify and iteration < opt.densify_until_iter and atlas_phase["enable_densify"]:
        atlas_densify_start = opt.densify_from_iter
        if scene.gaussians.has_atlas_bindings:
            atlas_densify_start = max(atlas_densify_start, int(opt.atlas_reg_warmup_steps))
        current_gaussian_count = int(gaussians.get_xyz.shape[0])
        has_render_densify_tensors = (
            densify_radii is not None
            and densify_visibility_filter is not None
            and densify_viewspace_point_tensor is not None
        )
        densify_tensors_valid = (
            not gc_pruned_this_iter
            and has_render_densify_tensors
            and int(densify_radii.shape[0]) == current_gaussian_count
            and int(densify_visibility_filter.shape[0]) == current_gaussian_count
            and int(densify_viewspace_point_tensor.shape[0]) == current_gaussian_count
        )
        cached_densify_stats_ready = (
            int(gaussians.xyz_gradient_accum.shape[0]) == current_gaussian_count
            and int(gaussians.denom.shape[0]) == current_gaussian_count
            and int(gaussians.max_radii2D.shape[0]) == current_gaussian_count
        )
        densify_radii_for_prune = densify_radii
        densify_can_run = densify_tensors_valid
        if not densify_tensors_valid:
            render_tensor_count = float(int(densify_radii.shape[0])) if densify_radii is not None else 0.0
            visibility_tensor_count = float(int(densify_visibility_filter.shape[0])) if densify_visibility_filter is not None else 0.0
            viewspace_tensor_count = float(int(densify_viewspace_point_tensor.shape[0])) if densify_viewspace_point_tensor is not None else 0.0
            stale_reasons = []
            if gc_pruned_this_iter:
                stale_reasons.append("gc_pruned_render_tensors")
            if not has_render_densify_tensors:
                stale_reasons.append("missing_render_tensors")
            if densify_radii is not None and int(densify_radii.shape[0]) != current_gaussian_count:
                stale_reasons.append("radii_shape_mismatch")
            if densify_visibility_filter is not None and int(densify_visibility_filter.shape[0]) != current_gaussian_count:
                stale_reasons.append("visibility_shape_mismatch")
            if densify_viewspace_point_tensor is not None and int(densify_viewspace_point_tensor.shape[0]) != current_gaussian_count:
                stale_reasons.append("viewspace_shape_mismatch")
            stale_reason = ",".join(stale_reasons) if stale_reasons else "unknown_stale_render_tensors"
            densify_metrics.update(
                {
                    "densify_stale_render_tensors": 1.0,
                    "densify_stale_reason": stale_reason,
                    "densify_skip_reason": "none",
                    "densify_render_tensor_count": render_tensor_count,
                    "densify_visibility_tensor_count": visibility_tensor_count,
                    "densify_viewspace_tensor_count": viewspace_tensor_count,
                    "densify_current_gaussian_count": float(current_gaussian_count),
                    "densify_cached_stats_ready": 1.0 if cached_densify_stats_ready else 0.0,
                    "densify_gc_pruned_render_tensor_invalidated": 1.0 if gc_pruned_this_iter else 0.0,
                }
            )
            if cached_densify_stats_ready:
                densify_radii_for_prune = gaussians.max_radii2D.detach().clone()
                densify_can_run = True
                densify_metrics["densify_used_cached_stats"] = 1.0
                densify_metrics["densify_skip_reason"] = "none"
                if atlas_runtime_metrics is not None:
                    atlas_runtime_metrics["densify_stale_render_tensors"] = 1.0
                    atlas_runtime_metrics["densify_used_cached_stats"] = 1.0
                    atlas_runtime_metrics["densify_stale_reason"] = stale_reason
                    atlas_runtime_metrics["densify_skip_reason"] = "none"
            else:
                densify_metrics["densify_skipped_stale_render_tensors"] = 1.0
                densify_metrics["densify_skip_reason"] = stale_reason
                if atlas_runtime_metrics is not None:
                    atlas_runtime_metrics["densify_skipped_stale_render_tensors"] = 1.0
                    atlas_runtime_metrics["densify_stale_reason"] = stale_reason
                    atlas_runtime_metrics["densify_skip_reason"] = stale_reason
        else:
            gaussians.max_radii2D[densify_visibility_filter] = torch.max(
                gaussians.max_radii2D[densify_visibility_filter],
                densify_radii[densify_visibility_filter],
            )
            gaussians.add_densification_stats(densify_viewspace_point_tensor, densify_visibility_filter)

        if densify_can_run and iteration > atlas_densify_start and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            densify_preserved_metrics = {
                key: densify_metrics[key]
                for key in (
                    "densify_stale_render_tensors",
                    "densify_render_tensor_count",
                    "densify_current_gaussian_count",
                    "densify_cached_stats_ready",
                    "densify_used_cached_stats",
                    "densify_skipped_stale_render_tensors",
                )
                if key in densify_metrics
            }
            if scene.gaussians.has_atlas_bindings:
                densify_runtime_controls = _compute_densify_runtime_controls(
                    iteration,
                    opt,
                    scene.gaussians,
                    pose_runtime_state,
                    densify_runtime_state,
                    atlas_runtime_metrics=atlas_runtime_metrics,
                    atlas_state_metrics=atlas_state_metrics,
                )
                densify_metrics = gaussians.densify_and_prune_with_atlas(
                    opt.densify_grad_threshold,
                    0.005,
                    scene.cameras_extent,
                    size_threshold,
                    densify_radii_for_prune,
                    camera_center=viewpoint_cam.camera_center,
                    explore_grad_scale=opt.atlas_explore_grad_scale,
                    explore_slab_radius_mult=opt.atlas_explore_slab_radius_mult,
                    explore_jitter_scale=opt.atlas_explore_jitter_scale,
                    active_min_lifetime_iters=active_min_lifetime_iters,
                    stable_residual_threshold=opt.atlas_stable_residual_threshold,
                    all_camera_centers=train_camera_centers if train_camera_centers is not None else train_camera_centers_cache,
                    prune_enabled=bool(atlas_phase["enable_prune"]),
                    min_points_to_keep=int(prune_controls["min_points_to_keep"]),
                    visibility_threshold=float(getattr(opt, "prune_visibility_threshold", 0.02)),
                    max_reattach_failures=int(opt.atlas_gc_max_reattach_failures),
                    enable_soft_prune=bool(atlas_phase["enable_soft_prune"]),
                    densify_runtime_controls=densify_runtime_controls,
                )
            else:
                densify_metrics = gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    0.005,
                    scene.cameras_extent,
                    size_threshold,
                    densify_radii_for_prune,
                    prune_enabled=bool(atlas_phase["enable_prune"]),
                    min_points_to_keep=int(prune_controls["min_points_to_keep"]),
                    visibility_threshold=float(getattr(opt, "prune_visibility_threshold", 0.02)),
                    enable_soft_prune=bool(atlas_phase["enable_soft_prune"]),
                )
            densify_metrics.update(densify_preserved_metrics)
            if scene.gaussians.has_atlas_bindings:
                atlas_state_metrics.update(scene.gaussians.summarize_atlas_state_metrics())

        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

    _record_controller_ms(controller_timing_metrics, timing_name, controller_start)
    return atlas_runtime_metrics, atlas_state_metrics, densify_metrics, atlas_gc_metrics, gc_pruned_this_iter


def _guard_aux_loss(loss_raw, metrics: dict | None, safe_key: str, had_nonfinite_key: str, counter_key: str | None):
    if torch.is_tensor(loss_raw):
        safe_tensor = loss_raw
        had_nonfinite = not bool(torch.isfinite(loss_raw.detach()).all().item())
        if had_nonfinite:
            safe_tensor = torch.zeros((), dtype=loss_raw.dtype, device=loss_raw.device)
        safe_value, _ = _safe_log_scalar(loss_raw)
    else:
        had_nonfinite = not math.isfinite(float(loss_raw))
        safe_tensor = 0.0 if had_nonfinite else float(loss_raw)
        safe_value = 0.0 if had_nonfinite else float(loss_raw)
    if metrics is not None:
        metrics[safe_key] = float(safe_value)
        metrics[had_nonfinite_key] = 1.0 if had_nonfinite else 0.0
        if counter_key is not None:
            metrics[counter_key] = float(metrics.get(counter_key, 0.0)) + (1.0 if had_nonfinite else 0.0)
    return safe_tensor, float(safe_value), had_nonfinite


def _sanitize_metric_key(value):
    text = str(value if value not in (None, "") else "none").strip().lower()
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "none"


def _increment_histogram(histogram: dict, key, amount: int = 1):
    safe_key = _sanitize_metric_key(key)
    histogram[safe_key] = int(histogram.get(safe_key, 0)) + int(amount)


def _flatten_histogram(prefix: str, histogram: dict | None):
    histogram = histogram or {}
    return {
        f"{prefix}_{_sanitize_metric_key(metric_name)}": int(metric_value)
        for metric_name, metric_value in histogram.items()
    }


def _reason_breakdown(reason: str | None):
    text = str(reason if reason not in (None, "") else "none")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        parts = ["none"]
    breakdown = {}
    for part in parts:
        key = _sanitize_metric_key(part)
        breakdown[key] = int(breakdown.get(key, 0)) + 1
    return breakdown


def _build_quality_patch_metrics(
    atlas_runtime_metrics: dict | None,
    atlas_refresh_metrics: dict | None,
    atlas_state_metrics: dict | None,
    densify_metrics: dict | None,
    atlas_gc_metrics: dict | None,
    pose_metrics: dict | None,
):
    atlas_runtime_metrics = atlas_runtime_metrics or {}
    atlas_refresh_metrics = atlas_refresh_metrics or {}
    atlas_state_metrics = atlas_state_metrics or {}
    densify_metrics = densify_metrics or {}
    atlas_gc_metrics = atlas_gc_metrics or {}
    pose_metrics = pose_metrics or {}
    corr_atlas = pose_metrics.get("b1_pose_geo_atlas_native_selected_ratio", pose_metrics.get("b1_pose_geo_atlas_native_ratio", 0.0))
    corr_fallback = pose_metrics.get("b1_pose_geo_fallback_selected_ratio", pose_metrics.get("b1_pose_geo_fallback_ratio", 0.0))
    return {
        "corr_source_atlas_ratio": corr_atlas,
        "corr_source_fallback_ratio": corr_fallback,
        "refresh_override_count": atlas_refresh_metrics.get("refresh_override_count", atlas_refresh_metrics.get("runtime_override_count", 0.0)),
        "refresh_override_ratio": atlas_refresh_metrics.get("refresh_override_ratio", atlas_refresh_metrics.get("runtime_override_ratio", 0.0)),
        "refresh_std_before_after": atlas_refresh_metrics.get(
            "refresh_std_before_after",
            {
                "before": atlas_refresh_metrics.get("refresh_std_before", 0.0),
                "after": atlas_refresh_metrics.get("refresh_std_after", atlas_refresh_metrics.get("runtime_reliability_std", 0.0)),
            },
        ),
        "state_candidate_reason_breakdown": atlas_state_metrics.get("state_candidate_reason_breakdown", {}),
        "promotion_block_reason_breakdown": atlas_state_metrics.get("promotion_block_reason_breakdown", {}),
        "active_state_pipeline": {
            "formed": atlas_state_metrics.get("active_formed_count", atlas_state_metrics.get("candidate_formation_count", 0)),
            "formed_standard": atlas_state_metrics.get("active_standard_formation_count", 0),
            "formed_rescue_fallback": atlas_state_metrics.get("active_rescue_fallback_formation_count", 0),
            "admitted": atlas_state_metrics.get("active_admitted_count", atlas_state_metrics.get("active_admission_pool_count", 0)),
            "admitted_standard": atlas_state_metrics.get("active_standard_admission_pool_count", 0),
            "admitted_rescue": atlas_state_metrics.get("active_rescue_admission_pool_count", 0),
            "promoted_standard": atlas_state_metrics.get("active_standard_promote_count", atlas_state_metrics.get("active_promoted_count", 0)),
            "promoted_rescue": atlas_state_metrics.get("active_forced_rescue_promote_count", atlas_state_metrics.get("active_rescue_promote_count", 0)),
            "new_active": atlas_state_metrics.get("active_new_active_count", atlas_state_metrics.get("active_promote_count", 0)),
        },
        "active_provenance_breakdown": {
            "transition_passive_to_active": atlas_state_metrics.get("active_provenance_from_transition_passive_to_active_count", 0),
            "state_rebuild_after_gc": atlas_state_metrics.get("active_provenance_from_state_rebuild_after_gc_count", 0),
            "forced_rescue_bootstrap": atlas_state_metrics.get("active_provenance_from_forced_rescue_bootstrap_count", 0),
            "quota_carryover": atlas_state_metrics.get("active_provenance_from_quota_carryover_count", 0),
            "restore_checkpoint": atlas_state_metrics.get("active_provenance_from_restore_checkpoint_count", 0),
            "active_explore_clone": atlas_state_metrics.get("active_provenance_from_active_explore_clone_count", 0),
        },
        "pose_b1_quality_breakdown": pose_metrics.get("pose_b1_quality_breakdown", {}),
        "pose_b2_skip_reason_breakdown": pose_metrics.get(
            "pose_b2_skip_reason_breakdown",
            _reason_breakdown(pose_metrics.get("b2_skip_reason", "none")),
        ),
        "explore_clone_success_count": densify_metrics.get("explore_clone_success_count", densify_metrics.get("explore_clone_count", 0.0)),
        "active_to_explore_clone_handoff_count": densify_metrics.get("active_explore_clone_count", 0.0),
        "stable_split_candidate_count": densify_metrics.get("stable_split_candidate_count", 0.0),
        "stable_clone_candidate_count": densify_metrics.get("stable_clone_candidate_count", 0.0),
        "active_explore_candidate_count": densify_metrics.get("explore_candidate_count", 0.0),
        "densify_used_cached_stats": densify_metrics.get("densify_used_cached_stats", 0.0),
        "densify_skip_reason": densify_metrics.get("densify_skip_reason", "none"),
        "densify_budget_scale": densify_metrics.get("densify_budget_scale", 1.0),
        "densify_global_quota": densify_metrics.get("densify_global_quota", 0.0),
        "densify_split_quota": densify_metrics.get("densify_split_quota", 0.0),
        "densify_clone_quota": densify_metrics.get("densify_clone_quota", 0.0),
        "densify_explore_quota": densify_metrics.get("densify_explore_quota", 0.0),
        "densify_budget_phase_ramp": densify_metrics.get("densify_budget_phase_ramp", 1.0),
        "densify_budget_b2_health": densify_metrics.get("densify_budget_b2_health", 1.0),
        "densify_budget_floater_guard": densify_metrics.get("densify_budget_floater_guard", 1.0),
        "densify_budget_quality_guard": densify_metrics.get("densify_budget_quality_guard", 1.0),
        "fidelity_handoff_gate": densify_metrics.get("fidelity_handoff_gate", 0.0),
        "fidelity_handoff_completion_gate": densify_metrics.get("fidelity_handoff_completion_gate", 0.0),
        "fidelity_mode_gate": densify_metrics.get("fidelity_mode_gate", 0.0),
        "fidelity_mode_enabled": densify_metrics.get("fidelity_mode_enabled", densify_metrics.get("fidelity_mode_gate", 0.0)),
        "fidelity_mode_dark_gate": densify_metrics.get("fidelity_mode_dark_gate", 0.0),
        "fidelity_mode_l1_gate": densify_metrics.get("fidelity_mode_l1_gate", 0.0),
        "fidelity_mode_floater_gate": densify_metrics.get("fidelity_mode_floater_gate", 0.0),
        "fidelity_mode_reliability_gate": densify_metrics.get("fidelity_mode_reliability_gate", 0.0),
        "fidelity_handoff_explore_scale": densify_metrics.get("fidelity_handoff_explore_scale", 1.0),
        "fidelity_handoff_active_noisy_prune_count": densify_metrics.get("fidelity_handoff_active_noisy_prune_count", 0.0),
        "fidelity_handoff_unsupported_explore_prune_count": densify_metrics.get("fidelity_handoff_unsupported_explore_prune_count", 0.0),
        "fidelity_handoff_unsupported_rescue_prune_count": densify_metrics.get("fidelity_handoff_unsupported_rescue_prune_count", 0.0),
        "active_noisy_pruned_count": densify_metrics.get("active_noisy_pruned_count", 0.0),
        "unsupported_explore_pruned_count": densify_metrics.get("unsupported_explore_pruned_count", 0.0),
        "unsupported_rescue_pruned_count": densify_metrics.get("unsupported_rescue_pruned_count", 0.0),
        "background_fidelity_protected_count": densify_metrics.get("background_fidelity_protected_count", 0.0),
        "stable_split_count": densify_metrics.get("stable_split_count", 0.0),
        "stable_clone_count": densify_metrics.get("stable_clone_count", 0.0),
        "stable_split_budget": densify_metrics.get("stable_split_budget", 0.0),
        "stable_clone_budget": densify_metrics.get("stable_clone_budget", 0.0),
        "active_explore_budget": densify_metrics.get("explore_budget", 0.0),
        "stable_clone_suppressed_by_split_count": densify_metrics.get("stable_clone_suppressed_by_split_count", 0.0),
        "stable_split_support_ready_count": densify_metrics.get("stable_split_support_ready_count", 0.0),
        "stable_split_coverage_not_thin_count": densify_metrics.get("stable_split_coverage_not_thin_count", 0.0),
        "stable_clone_projected_drift_small_count": densify_metrics.get("stable_clone_projected_drift_small_count", 0.0),
        "stable_clone_recent_transition_block_count": densify_metrics.get("stable_clone_recent_transition_block_count", 0.0),
        "active_explore_ref_or_visibility_ready_count": densify_metrics.get("explore_ref_or_visibility_ready_count", 0.0),
        "active_explore_slab_valid_count": densify_metrics.get("explore_slab_valid_count", 0.0),
        "active_explore_slab_discard_count": densify_metrics.get("explore_slab_discard_count", 0.0),
        "active_explore_background_like_block_count": densify_metrics.get("explore_background_like_block_count", 0.0),
        "active_explore_background_like_slab_count": densify_metrics.get("explore_background_like_slab_count", 0.0),
        "active_explore_adaptive_slab_mult_mean": densify_metrics.get("explore_adaptive_slab_mult_mean", 0.0),
        "active_explore_depth_delta_ratio_mean": densify_metrics.get("explore_depth_delta_ratio_mean", 0.0),
        "active_explore_neighbor_stable_slab_count": densify_metrics.get("explore_neighbor_stable_slab_count", 0.0),
        "active_explore_support_only_conflict_count": densify_metrics.get("explore_support_only_conflict_count", 0.0),
        "pose_b2_template_fixed_coords": pose_metrics.get("b2_pose_template_fixed_coords", 0.0),
        "pose_b2_grad_norm_total": pose_metrics.get("b2_pose_grad_norm_total", 0.0),
        "pose_b2_grad_nonzero": pose_metrics.get("b2_grad_nonzero", 0.0),
        "pose_b2_data_grad_nonzero": pose_metrics.get("b2_data_grad_nonzero", 0.0),
        "pose_b2_pose_q_grad_norm": pose_metrics.get("b2_pose_q_grad_norm", 0.0),
        "pose_b2_pose_t_grad_norm": pose_metrics.get("b2_pose_t_grad_norm", 0.0),
        "pose_b2_loss_depends_on_pose_path": pose_metrics.get("b2_loss_depends_on_pose_path", 0.0),
        "pose_b2_step_by_total_grad": pose_metrics.get("b2_step_by_total_grad", 0.0),
        "pose_b2_step_by_data_grad": pose_metrics.get("b2_step_by_data_grad", 0.0),
        "pose_b2_pre_trust_grad_norm": pose_metrics.get("b2_pre_trust_grad_norm", 0.0),
        "pose_b2_post_trust_grad_norm": pose_metrics.get("b2_post_trust_grad_norm", 0.0),
        "pose_b2_step_allowed": pose_metrics.get("b2_step_allowed", 0.0),
        "pose_b2_skip_reason_detailed": pose_metrics.get("b2_skip_reason_detailed", pose_metrics.get("b2_skip_reason", "none")),
        "pose_b2_mode": pose_metrics.get("b2_mode", "none"),
        "pose_b2_fullframe_stress_enabled": pose_metrics.get("b2_fullframe_stress_enabled", 0.0),
        "pose_fullframe_l1": pose_metrics.get("b2_pose_fullframe_l1", 0.0),
        "pose_fullframe_ssim": pose_metrics.get("b2_pose_fullframe_ssim", 0.0),
        "pose_fullframe_gradient": pose_metrics.get("b2_pose_fullframe_gradient", 0.0),
        "pose_fullframe_total": pose_metrics.get("b2_pose_fullframe_total", 0.0),
        "pose_fullframe_num_pixels": pose_metrics.get("b2_pose_fullframe_num_pixels", 0.0),
        "pose_fullframe_downsample_factor": pose_metrics.get("b2_pose_fullframe_downsample_factor", 0.0),
        "pose_b2_pose_delta_q_requires_grad": pose_metrics.get("b2_pose_delta_q_requires_grad", 0.0),
        "pose_b2_pose_delta_t_requires_grad": pose_metrics.get("b2_pose_delta_t_requires_grad", 0.0),
        "pose_b2_world_view_transform_requires_grad": pose_metrics.get("b2_world_view_transform_requires_grad", 0.0),
        "pose_b2_full_proj_transform_requires_grad": pose_metrics.get("b2_full_proj_transform_requires_grad", 0.0),
        "pose_b2_camera_center_requires_grad": pose_metrics.get("b2_camera_center_requires_grad", 0.0),
        "pose_b2_fd_probe_enabled": pose_metrics.get("b2_fd_probe_enabled", 0.0),
        "pose_b2_fd_probe_status": pose_metrics.get("b2_fd_probe_status", "disabled"),
        "pose_b2_fd_probe_count": pose_metrics.get("b2_fd_probe_count", 0.0),
        "pose_b2_fd_trans_sensitivity_max": pose_metrics.get("b2_fd_trans_sensitivity_max", 0.0),
        "pose_b2_fd_rot_sensitivity_max": pose_metrics.get("b2_fd_rot_sensitivity_max", 0.0),
        "pose_b2_fd_any_positive": pose_metrics.get("b2_fd_any_positive", 0.0),
        "pose_b2_fd_nonflat_autograd_tiny": pose_metrics.get("b2_fd_nonflat_autograd_tiny", 0.0),
        "pose_b2_fd_nonflat_trust_choked": pose_metrics.get("b2_fd_nonflat_trust_choked", 0.0),
        "pose_b2_data_only_grad_total": pose_metrics.get("b2_data_only_grad_total", 0.0),
        "pose_b2_data_only_vs_total_grad_ratio": pose_metrics.get("b2_data_only_vs_total_grad_ratio", 0.0),
        "pose_b2_pre_to_post_grad_shrink_ratio": pose_metrics.get("b2_pre_to_post_grad_shrink_ratio", 0.0),
        "pose_b2_microstep_mode": pose_metrics.get("b2_microstep_mode", 0.0),
        "pose_b2_microstep_translation_applied": pose_metrics.get("b2_microstep_translation_applied", 0.0),
        "pose_b2_microstep_rotation_applied_deg": pose_metrics.get("b2_microstep_rotation_applied_deg", 0.0),
        "pose_b2_microstep_reason": pose_metrics.get("b2_microstep_reason", "inactive"),
        "pose_b1_geo_pre_median_px": pose_metrics.get("b1_pose_geo_pre_median_px", 0.0),
        "pose_b1_geo_post_median_px": pose_metrics.get("b1_pose_geo_post_median_px", 0.0),
        "pose_b1_geo_median_px_reduction": pose_metrics.get("b1_pose_geo_median_px_reduction", 0.0),
        "pose_b1_success_residual_reduced": pose_metrics.get("b1_success_residual_reduced", 0.0),
        "pose_b1_camera_success_criterion": pose_metrics.get("b1_camera_success_criterion", 0.0),
        "pose_b2_photo_corridor_open": pose_metrics.get("b2_photo_corridor_open", 0.0),
        "pose_b2_photo_corridor_scene_signal": pose_metrics.get("b2_photo_corridor_scene_signal", 0.0),
        "pose_b2_photo_corridor_support_ready": pose_metrics.get("b2_photo_corridor_support_ready", 0.0),
        "pose_b2_corridor_step_ok": pose_metrics.get("b2_corridor_step_ok", 0.0),
        "reattach_success_ratio": atlas_gc_metrics.get("reattach_success_ratio", 0.0),
        "reattach_candidate_starvation_ratio": atlas_gc_metrics.get("reattach_candidate_starvation_ratio", 0.0),
        "ray_guided_priority_queries": atlas_gc_metrics.get("ray_guided_priority_queries", 0.0),
        "floater_proxy_by_state": atlas_runtime_metrics.get("floater_proxy_by_state", {}),
        "dark_region_completeness_by_state": atlas_runtime_metrics.get("dark_region_completeness_by_state", {}),
    }


def _pose_camera_key(camera):
    if camera is None:
        return "unknown"
    image_name = getattr(camera, "image_name", None)
    if image_name is not None:
        return str(image_name)
    uid = getattr(camera, "uid", None)
    if uid is not None:
        return f"uid:{uid}"
    colmap_id = getattr(camera, "colmap_id", None)
    if colmap_id is not None:
        return f"colmap:{colmap_id}"
    return "unknown"


def _serialize_json_payload(payload):
    if isinstance(payload, dict):
        return {str(key): _serialize_json_payload(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_serialize_json_payload(value) for value in payload]
    return serialize_metric_value(payload)


def _write_training_summary(model_path: str, payload: dict):
    summary_path = os.path.join(model_path, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(_serialize_json_payload(payload), summary_file, indent=2, sort_keys=True, default=str)


def _is_scalar_metric_value(value):
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    if torch.is_tensor(value):
        return value.numel() == 1
    return False


def _tb_add_metric_group(tb_writer, prefix: str, metrics: dict | None, iteration: int):
    if tb_writer is None or metrics is None:
        return
    for metric_name, metric_value in metrics.items():
        if not _is_scalar_metric_value(metric_value):
            continue
        safe_value, _ = _safe_log_scalar(metric_value)
        tb_writer.add_scalar(f"{prefix}/{metric_name}", safe_value, iteration)


def _get_lpips_model(net_type: str = "vgg", device: torch.device | None = None):
    if not LPIPS_AVAILABLE:
        return None
    device = torch.device("cuda") if device is None else torch.device(device)
    cache_key = (str(device), str(net_type))
    model = _LPIPS_MODELS.get(cache_key)
    if model is None:
        try:
            model = LPIPS(net_type=net_type, version="0.1").to(device)
            model.eval()
            _LPIPS_MODELS[cache_key] = model
        except Exception:
            return None
    return model


def _compute_lpips_metric(image: torch.Tensor, gt_image: torch.Tensor, net_type: str = "vgg"):
    if not LPIPS_AVAILABLE:
        if not _LPIPS_WARNED[0]:
            _LPIPS_WARNED[0] = True
            print("[WARNING] LPIPS unavailable (import failed at startup); LPIPS metrics will be null")
        return None
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if gt_image.ndim == 3:
        gt_image = gt_image.unsqueeze(0)
    model = _get_lpips_model(net_type=net_type, device=image.device)
    if model is None:
        if not _LPIPS_WARNED[0]:
            _LPIPS_WARNED[0] = True
            print("[WARNING] LPIPS model failed to load; LPIPS metrics will be null")
        return None
    try:
        with torch.no_grad():
            lpips_value = model(image * 2.0 - 1.0, gt_image * 2.0 - 1.0)
        return float(lpips_value.mean().detach().item())
    except Exception as e:
        if not _LPIPS_WARNED[0]:
            _LPIPS_WARNED[0] = True
            print(f"[WARNING] LPIPS computation raised exception: {e}")
        return None


def _cuda_peak_vram_mb():
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _summarize_camera_pose_deltas(cameras):
    metrics = {
        "pose_translation_proxy_mean": 0.0,
        "pose_rotation_proxy_mean": 0.0,
    }
    if not cameras:
        return metrics
    translation_total = 0.0
    rotation_total = 0.0
    valid = 0
    for camera in cameras:
        pose_delta = measure_pose_delta(camera)
        translation_total += float(pose_delta.get("translation_norm", 0.0))
        rotation_total += float(pose_delta.get("rotation_degrees", 0.0))
        valid += 1
    if valid > 0:
        metrics["pose_translation_proxy_mean"] = translation_total / float(valid)
        metrics["pose_rotation_proxy_mean"] = rotation_total / float(valid)
    return metrics


def _camera_pose_grad_metrics(viewpoint_camera):
    q_grad = getattr(getattr(viewpoint_camera, "pose_delta_q", None), "grad", None)
    t_grad = getattr(getattr(viewpoint_camera, "pose_delta_t", None), "grad", None)
    q_norm = float(torch.linalg.norm(q_grad).detach().item()) if q_grad is not None else 0.0
    t_norm = float(torch.linalg.norm(t_grad).detach().item()) if t_grad is not None else 0.0
    return {
        "pose_grad_norm_rotation": q_norm,
        "pose_grad_norm_translation": t_norm,
        "pose_grad_norm_total": math.sqrt(max(q_norm, 0.0) ** 2 + max(t_norm, 0.0) ** 2),
    }


def _clone_camera_pose_grads(viewpoint_camera):
    grads = {}
    for name in ("pose_delta_q", "pose_delta_t"):
        param = getattr(viewpoint_camera, name, None)
        grad = getattr(param, "grad", None)
        grads[name] = grad.detach().clone() if grad is not None else None
    return grads


def _restore_camera_pose_grads(viewpoint_camera, grads):
    for name, grad in (grads or {}).items():
        param = getattr(viewpoint_camera, name, None)
        if param is None:
            continue
        if grad is None:
            param.grad = None
        else:
            param.grad = grad.detach().clone()


def _audit_camera_pose_loss_autograd(loss, viewpoint_camera):
    metrics = {
        "b2_pose_q_grad_is_none": 1.0,
        "b2_pose_t_grad_is_none": 1.0,
        "b2_pose_q_grad_norm": 0.0,
        "b2_pose_t_grad_norm": 0.0,
        "b2_pose_grad_norm_total": 0.0,
        "b2_pose_graph_connected": 0.0,
        "b2_loss_depends_on_pose_path": 0.0,
        "b2_pose_autograd_error": "none",
    }
    cloned_grads = {"pose_delta_q": None, "pose_delta_t": None}
    if loss is None or not bool(getattr(loss, "requires_grad", False)):
        metrics["b2_pose_autograd_error"] = "loss_requires_grad_false"
        return metrics, cloned_grads

    params = []
    names = []
    for name in ("pose_delta_q", "pose_delta_t"):
        param = getattr(viewpoint_camera, name, None)
        if param is not None and bool(getattr(param, "requires_grad", False)):
            params.append(param)
            names.append(name)

    if not params:
        metrics["b2_pose_autograd_error"] = "pose_params_not_trainable"
        return metrics, cloned_grads

    try:
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
    except RuntimeError as exc:
        metrics["b2_pose_autograd_error"] = str(exc).split("\n", 1)[0][:160]
        return metrics, cloned_grads

    for name, grad in zip(names, grads):
        key = "q" if name == "pose_delta_q" else "t"
        metrics[f"b2_pose_{key}_grad_is_none"] = 1.0 if grad is None else 0.0
        if grad is not None:
            safe_grad = torch.nan_to_num(grad.detach(), nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = float(torch.linalg.norm(safe_grad).item())
            metrics[f"b2_pose_{key}_grad_norm"] = grad_norm
            cloned_grads[name] = safe_grad.clone()

    total_norm = math.sqrt(
        max(metrics["b2_pose_q_grad_norm"], 0.0) ** 2
        + max(metrics["b2_pose_t_grad_norm"], 0.0) ** 2
    )
    metrics["b2_pose_grad_norm_total"] = float(total_norm)
    graph_connected = bool(
        metrics["b2_pose_q_grad_is_none"] < 0.5
        or metrics["b2_pose_t_grad_is_none"] < 0.5
    )
    metrics["b2_pose_graph_connected"] = 1.0 if graph_connected else 0.0
    metrics["b2_loss_depends_on_pose_path"] = 1.0 if total_norm > 0.0 else 0.0
    return metrics, cloned_grads


def _camera_pose_debug_snapshot(viewpoint_camera):
    defaults = {
        "b2_pose_delta_q_requires_grad": 0.0,
        "b2_pose_delta_t_requires_grad": 0.0,
        "b2_world_view_transform_requires_grad": 0.0,
        "b2_full_proj_transform_requires_grad": 0.0,
        "b2_camera_center_requires_grad": 0.0,
        "b2_pose_delta_t_norm": 0.0,
        "b2_pose_delta_rotation_degrees": 0.0,
        "b2_pose_trainable": 0.0,
    }
    if viewpoint_camera is None:
        return defaults
    try:
        if hasattr(viewpoint_camera, "get_pose_debug_snapshot"):
            snapshot = viewpoint_camera.get_pose_debug_snapshot()
            for key, value in snapshot.items():
                defaults[f"b2_{key}"] = float(value) if isinstance(value, (int, float, bool)) else value
        else:
            pose_q = getattr(viewpoint_camera, "pose_delta_q", None)
            pose_t = getattr(viewpoint_camera, "pose_delta_t", None)
            world_view = getattr(viewpoint_camera, "world_view_transform", None)
            full_proj = getattr(viewpoint_camera, "full_proj_transform", None)
            camera_center = getattr(viewpoint_camera, "camera_center", None)
            defaults["b2_pose_delta_q_requires_grad"] = 1.0 if pose_q is not None and bool(getattr(pose_q, "requires_grad", False)) else 0.0
            defaults["b2_pose_delta_t_requires_grad"] = 1.0 if pose_t is not None and bool(getattr(pose_t, "requires_grad", False)) else 0.0
            defaults["b2_world_view_transform_requires_grad"] = 1.0 if world_view is not None and bool(getattr(world_view, "requires_grad", False)) else 0.0
            defaults["b2_full_proj_transform_requires_grad"] = 1.0 if full_proj is not None and bool(getattr(full_proj, "requires_grad", False)) else 0.0
            defaults["b2_camera_center_requires_grad"] = 1.0 if camera_center is not None and bool(getattr(camera_center, "requires_grad", False)) else 0.0
            defaults["b2_pose_trainable"] = 1.0 if (
                defaults["b2_pose_delta_q_requires_grad"] > 0.5
                or defaults["b2_pose_delta_t_requires_grad"] > 0.5
            ) else 0.0
    except Exception as exc:
        defaults["b2_pose_debug_snapshot_error"] = str(exc).split("\n", 1)[0][:160]
    return defaults


def _iteration_window_enabled(enabled: bool, iteration: int, start_iter: int, end_iter: int):
    if not bool(enabled):
        return False
    iteration = int(iteration)
    start_iter = int(max(start_iter, 0))
    end_iter = int(end_iter)
    return bool(iteration >= start_iter and (end_iter < 0 or iteration <= end_iter))


def _empty_b2_fd_probe_metrics(status="disabled"):
    return {
        "b2_fd_probe_enabled": 0.0,
        "b2_fd_probe_status": status,
        "b2_fd_probe_count": 0.0,
        "b2_fd_trans_sensitivity_x": 0.0,
        "b2_fd_trans_sensitivity_y": 0.0,
        "b2_fd_trans_sensitivity_z": 0.0,
        "b2_fd_rot_sensitivity_x": 0.0,
        "b2_fd_rot_sensitivity_y": 0.0,
        "b2_fd_rot_sensitivity_z": 0.0,
        "b2_fd_trans_sensitivity_max": 0.0,
        "b2_fd_rot_sensitivity_max": 0.0,
        "b2_fd_any_positive": 0.0,
        "b2_fd_flat": 0.0,
        "b2_fd_nonflat_autograd_tiny": 0.0,
        "b2_fd_nonflat_trust_choked": 0.0,
    }


def _capture_camera_pose_delta(viewpoint_camera):
    pose_q = getattr(viewpoint_camera, "pose_delta_q", None)
    pose_t = getattr(viewpoint_camera, "pose_delta_t", None)
    return (
        pose_q.detach().clone() if pose_q is not None else None,
        pose_t.detach().clone() if pose_t is not None else None,
    )


def _restore_camera_pose_delta_snapshot(viewpoint_camera, snapshot):
    pose_q, pose_t = snapshot
    with torch.no_grad():
        if pose_q is not None and hasattr(viewpoint_camera, "pose_delta_q"):
            viewpoint_camera.pose_delta_q.copy_(pose_q)
        if pose_t is not None and hasattr(viewpoint_camera, "pose_delta_t"):
            viewpoint_camera.pose_delta_t.copy_(pose_t)
    if hasattr(viewpoint_camera, "refresh_pose_matrices"):
        pose_trainable = bool(
            getattr(getattr(viewpoint_camera, "pose_delta_q", None), "requires_grad", False)
            or getattr(getattr(viewpoint_camera, "pose_delta_t", None), "requires_grad", False)
        )
        try:
            viewpoint_camera.refresh_pose_matrices(differentiable=pose_trainable)
        except TypeError:
            viewpoint_camera.refresh_pose_matrices()


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=0)
    w2, x2, y2, z2 = q2.unbind(dim=0)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _apply_camera_pose_perturbation(viewpoint_camera, translation_delta=None, rotation_axis=None, rotation_degrees=0.0):
    with torch.no_grad():
        if translation_delta is not None and hasattr(viewpoint_camera, "pose_delta_t"):
            viewpoint_camera.pose_delta_t.add_(translation_delta.to(device=viewpoint_camera.pose_delta_t.device, dtype=viewpoint_camera.pose_delta_t.dtype))
        if rotation_axis is not None and abs(float(rotation_degrees)) > 0.0 and hasattr(viewpoint_camera, "pose_delta_q"):
            axis = rotation_axis.to(device=viewpoint_camera.pose_delta_q.device, dtype=viewpoint_camera.pose_delta_q.dtype)
            axis = axis / torch.linalg.norm(axis).clamp_min(1e-12)
            half_angle = math.radians(float(rotation_degrees)) * 0.5
            delta_q = torch.cat(
                (
                    viewpoint_camera.pose_delta_q.new_tensor([math.cos(half_angle)]),
                    axis * math.sin(half_angle),
                ),
                dim=0,
            )
            pose_q = torch.nn.functional.normalize(viewpoint_camera.pose_delta_q.detach(), dim=0)
            viewpoint_camera.pose_delta_q.copy_(torch.nn.functional.normalize(_quat_multiply(delta_q, pose_q), dim=0))
    if hasattr(viewpoint_camera, "refresh_pose_matrices"):
        pose_trainable = bool(
            getattr(getattr(viewpoint_camera, "pose_delta_q", None), "requires_grad", False)
            or getattr(getattr(viewpoint_camera, "pose_delta_t", None), "requires_grad", False)
        )
        try:
            viewpoint_camera.refresh_pose_matrices(differentiable=pose_trainable)
        except TypeError:
            viewpoint_camera.refresh_pose_matrices()


def _probe_pose_loss_sensitivity(
    viewpoint_camera,
    gaussians,
    probe_context,
    *,
    photo_alpha: float,
    gradient_weight: float,
    patch_feature_weight: float,
    sensitivity_floor: float = 1e-7,
    probe_count: int = 0,
):
    metrics = _empty_b2_fd_probe_metrics(status="disabled")
    metrics["b2_fd_probe_enabled"] = 1.0
    metrics["b2_fd_probe_count"] = float(max(int(probe_count), 0))
    if not probe_context:
        metrics["b2_fd_probe_status"] = "missing_context"
        return metrics
    if viewpoint_camera is None or not hasattr(viewpoint_camera, "pose_delta_q") or not hasattr(viewpoint_camera, "pose_delta_t"):
        metrics["b2_fd_probe_status"] = "missing_pose_params"
        return metrics

    base_snapshot = _capture_camera_pose_delta(viewpoint_camera)
    device = viewpoint_camera.pose_delta_t.device
    dtype = viewpoint_camera.pose_delta_t.dtype
    trans_eps = (1e-5, 3e-5, 1e-4)
    rot_eps_deg = (0.001, 0.003, 0.01)
    axes = (
        ("x", torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)),
        ("y", torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)),
        ("z", torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)),
    )

    def _loss_value():
        loss, probe_metrics = compute_pose_refinement_data_loss_from_context(
            viewpoint_camera,
            gaussians,
            probe_context,
            photo_alpha=photo_alpha,
            gradient_weight=gradient_weight,
            patch_feature_weight=patch_feature_weight,
        )
        status = str(probe_metrics.get("pose_probe_status", "unknown"))
        finite_loss = bool(torch.isfinite(loss.detach()).item()) if loss is not None and loss.numel() == 1 else False
        if status != "ok" or loss is None or not finite_loss:
            return None, status
        return float(loss.detach().item()), status

    try:
        with torch.no_grad():
            for axis_name, axis in axes:
                best_slope = 0.0
                for eps in trans_eps:
                    _restore_camera_pose_delta_snapshot(viewpoint_camera, base_snapshot)
                    _apply_camera_pose_perturbation(viewpoint_camera, translation_delta=axis * float(eps))
                    loss_plus, status_plus = _loss_value()
                    _restore_camera_pose_delta_snapshot(viewpoint_camera, base_snapshot)
                    _apply_camera_pose_perturbation(viewpoint_camera, translation_delta=axis * float(-eps))
                    loss_minus, status_minus = _loss_value()
                    if loss_plus is None or loss_minus is None:
                        metrics["b2_fd_probe_status"] = f"loss_error_{status_plus}_{status_minus}"[:120]
                        continue
                    slope = abs((loss_plus - loss_minus) / max(2.0 * float(eps), 1e-20))
                    if math.isfinite(slope):
                        best_slope = max(best_slope, float(slope))
                metrics[f"b2_fd_trans_sensitivity_{axis_name}"] = float(best_slope)

            for axis_name, axis in axes:
                best_slope = 0.0
                for eps_deg in rot_eps_deg:
                    _restore_camera_pose_delta_snapshot(viewpoint_camera, base_snapshot)
                    _apply_camera_pose_perturbation(viewpoint_camera, rotation_axis=axis, rotation_degrees=float(eps_deg))
                    loss_plus, status_plus = _loss_value()
                    _restore_camera_pose_delta_snapshot(viewpoint_camera, base_snapshot)
                    _apply_camera_pose_perturbation(viewpoint_camera, rotation_axis=axis, rotation_degrees=float(-eps_deg))
                    loss_minus, status_minus = _loss_value()
                    if loss_plus is None or loss_minus is None:
                        metrics["b2_fd_probe_status"] = f"loss_error_{status_plus}_{status_minus}"[:120]
                        continue
                    slope = abs((loss_plus - loss_minus) / max(2.0 * float(eps_deg), 1e-20))
                    if math.isfinite(slope):
                        best_slope = max(best_slope, float(slope))
                metrics[f"b2_fd_rot_sensitivity_{axis_name}"] = float(best_slope)
    except Exception as exc:
        metrics["b2_fd_probe_status"] = ("error_" + str(exc).split("\n", 1)[0])[:160]
    finally:
        _restore_camera_pose_delta_snapshot(viewpoint_camera, base_snapshot)

    trans_max = max(
        float(metrics["b2_fd_trans_sensitivity_x"]),
        float(metrics["b2_fd_trans_sensitivity_y"]),
        float(metrics["b2_fd_trans_sensitivity_z"]),
    )
    rot_max = max(
        float(metrics["b2_fd_rot_sensitivity_x"]),
        float(metrics["b2_fd_rot_sensitivity_y"]),
        float(metrics["b2_fd_rot_sensitivity_z"]),
    )
    metrics["b2_fd_trans_sensitivity_max"] = float(trans_max)
    metrics["b2_fd_rot_sensitivity_max"] = float(rot_max)
    any_positive = bool(max(trans_max, rot_max) > float(max(sensitivity_floor, 0.0)))
    metrics["b2_fd_any_positive"] = 1.0 if any_positive else 0.0
    if str(metrics.get("b2_fd_probe_status", "disabled")) in ("disabled", ""):
        metrics["b2_fd_probe_status"] = "ok"
    elif str(metrics.get("b2_fd_probe_status", "")).startswith("loss_error") and any_positive:
        metrics["b2_fd_probe_status"] = "partial_ok"
    return metrics


def _apply_b2_microstep_by_pose_audit_grad(viewpoint_camera, audit_grads, opt):
    metrics = {
        "b2_microstep_mode": 0.0,
        "b2_microstep_translation_applied": 0.0,
        "b2_microstep_rotation_applied_deg": 0.0,
        "b2_microstep_reason": "inactive",
    }
    if viewpoint_camera is None or not audit_grads:
        metrics["b2_microstep_reason"] = "missing_audit_grads"
        return False, metrics
    pose_t = getattr(viewpoint_camera, "pose_delta_t", None)
    pose_q = getattr(viewpoint_camera, "pose_delta_q", None)
    t_grad = audit_grads.get("pose_delta_t", None)
    q_grad = audit_grads.get("pose_delta_q", None)
    translation_cap = float(max(getattr(opt, "pose_b2_microstep_translation", 1e-4), 0.0))
    rotation_cap_deg = float(max(getattr(opt, "pose_b2_microstep_rotation_deg", 0.003), 0.0))
    applied = False
    with torch.no_grad():
        if pose_t is not None and torch.is_tensor(t_grad):
            safe_t_grad = torch.nan_to_num(t_grad.to(device=pose_t.device, dtype=pose_t.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            t_norm = torch.linalg.norm(safe_t_grad)
            if float(t_norm.item()) > 0.0 and translation_cap > 0.0:
                step_t = -safe_t_grad / t_norm.clamp_min(1e-20) * translation_cap
                pose_t.add_(step_t)
                metrics["b2_microstep_translation_applied"] = float(torch.linalg.norm(step_t).item())
                applied = True
        if pose_q is not None and torch.is_tensor(q_grad) and q_grad.numel() >= 4:
            safe_q_grad = torch.nan_to_num(q_grad.to(device=pose_q.device, dtype=pose_q.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            vec_grad = safe_q_grad[1:4]
            vec_norm = torch.linalg.norm(vec_grad)
            if float(vec_norm.item()) > 0.0 and rotation_cap_deg > 0.0:
                half_angle = math.radians(rotation_cap_deg) * 0.5
                vec_step = -vec_grad / vec_norm.clamp_min(1e-20) * math.sin(half_angle)
                new_q = torch.nn.functional.normalize(pose_q.detach(), dim=0).clone()
                new_q[1:4] = new_q[1:4] + vec_step
                new_q = torch.nn.functional.normalize(new_q, dim=0)
                if float(new_q[0].detach().item()) < 0.0:
                    new_q = -new_q
                pose_q.copy_(new_q)
                metrics["b2_microstep_rotation_applied_deg"] = float(rotation_cap_deg)
                applied = True
    if applied:
        if hasattr(viewpoint_camera, "refresh_pose_matrices"):
            viewpoint_camera.refresh_pose_matrices()
        metrics["b2_microstep_mode"] = 1.0
        metrics["b2_microstep_reason"] = "audit_direction_nonflat_probe"
        return True, metrics
    metrics["b2_microstep_reason"] = "zero_audit_direction"
    return False, metrics


def _should_step_exposure(
    has_atlas_bindings: bool,
    disable_pose_refine: bool,
    main_phase_ready: bool,
    pose_runtime_state: dict,
    render_loss_value: float,
    current_pose_delta: dict,
    opt,
):
    metrics = {
        "exposure_step_enabled": 1.0,
        "exposure_stable_mismatch": 0.0,
        "exposure_pose_history_ready": 0.0,
        "exposure_pose_delta_nonzero": 0.0,
        "exposure_step_reason": "default_allow",
    }
    if (not has_atlas_bindings) or bool(disable_pose_refine) or (not bool(main_phase_ready)):
        return True, metrics

    quality_ema = pose_runtime_state.get("quality_ema", None)
    stable_mismatch_threshold = float(getattr(opt, "pose_exposure_mismatch_threshold", getattr(opt, "atlas_stable_residual_threshold", 0.03)))
    stable_mismatch = (
        float(render_loss_value) >= stable_mismatch_threshold
        and quality_ema is not None
        and float(quality_ema) >= stable_mismatch_threshold
    )
    pose_history_ready = int(pose_runtime_state.get("b1_update_count", 0)) > 0 or int(pose_runtime_state.get("b2_update_count", 0)) > 0
    pose_delta_nonzero = (
        float(current_pose_delta.get("translation_norm", 0.0)) > 1e-6
        or float(current_pose_delta.get("rotation_degrees", 0.0)) > 1e-4
    )
    enabled = bool(stable_mismatch and (pose_history_ready or pose_delta_nonzero))
    metrics["exposure_step_enabled"] = 1.0 if enabled else 0.0
    metrics["exposure_stable_mismatch"] = 1.0 if stable_mismatch else 0.0
    metrics["exposure_pose_history_ready"] = 1.0 if pose_history_ready else 0.0
    metrics["exposure_pose_delta_nonzero"] = 1.0 if pose_delta_nonzero else 0.0
    if enabled:
        metrics["exposure_step_reason"] = "stable_mismatch_after_pose"
    elif not stable_mismatch:
        metrics["exposure_step_reason"] = "no_stable_mismatch"
    else:
        metrics["exposure_step_reason"] = "pose_not_ready"
    return enabled, metrics


def _infer_pose_b2_variant(opt):
    if bool(getattr(opt, "disable_pose_refine", False)) or float(getattr(opt, "pose_photo_weight", 0.0)) <= 0.0:
        return "disabled"
    terms = ["l1"]
    if float(getattr(opt, "pose_photo_alpha", 0.0)) > 0.0:
        terms.append("ssim")
    if float(getattr(opt, "pose_gradient_weight", 0.0)) > 0.0:
        terms.append("grad")
    if float(getattr(opt, "pose_b2_patchfeat_weight", getattr(opt, "pose_patchfeat_weight", 0.0))) > 0.0:
        terms.append("patchfeat_budgeted")
    return "+".join(terms)


def _build_ablation_manifest(dataset, opt, gaussians):
    has_atlas = bool(getattr(gaussians, "has_atlas_bindings", False))
    return {
        "init_variant": "foundation_atlas" if has_atlas else "sfm_colmap",
        "atlas_runtime_calibration_variant": (
            "self_calibrated_runtime_refresh"
            if has_atlas and int(getattr(opt, "atlas_reg_warmup_steps", 0)) > 0
            else ("raw_atlas_runtime" if has_atlas else "sfm_runtime_only")
        ),
        "reliability_variant": "detached_runtime_prior",
        "shape_prior_variant": "scale_free_orientation_anisotropy",
        "mean_anchor_variant": (
            "support_projected_center_anchor"
            if float(getattr(opt, "atlas_mean_weight", 0.0)) > 0.0
            else "disabled"
        ),
        "gc_variant": (
            "async_interval_hash_reattach"
            if has_atlas and int(getattr(opt, "atlas_gc_interval", 0)) > 0
            else "disabled"
        ),
        "exploration_variant": (
            "ray_constrained_slab"
            if has_atlas and float(getattr(opt, "atlas_slab_weight", 0.0)) > 0.0
            else "disabled"
        ),
        "pose_refine_variant": "two_stage_gated" if not bool(getattr(opt, "disable_pose_refine", False)) else "disabled",
        "pose_b2_variant": _infer_pose_b2_variant(opt),
        "atlas_path_present": bool(getattr(dataset, "atlas_path", "")),
        "atlas_hash_prior": "static_atlas_hash" if has_atlas else "none",
        "uncertainty_variant": "local_exact_center_kl",
    }


def _write_ablation_manifest(model_path: str, manifest: dict):
    manifest_path = os.path.join(model_path, "ablation_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True, default=str)


def _build_required_training_log_fields(
    has_atlas_bindings: bool,
    atlas_phase: dict | None,
    atlas_reliability_summary: dict | None,
    atlas_state_metrics: dict | None,
    atlas_runtime_metrics: dict | None,
    densify_metrics: dict | None,
    atlas_gc_metrics: dict | None,
    pose_metrics: dict | None,
    atlas_metrics: dict | None = None,
    atlas_uncertainty_metrics: dict | None = None,
    atlas_slab_metrics: dict | None = None,
    atlas_kl_metrics: dict | None = None,
    atlas_refresh_done: bool = False,
):
    required_fields = {
        "atlas_reliability_base_mean": 0.0,
        "atlas_reliability_runtime_mean": 0.0,
        "atlas_reliability_runtime_min": 0.0,
        "atlas_reliability_runtime_max": 0.0,
        "atlas_reliability_base_p10": 0.0,
        "atlas_reliability_base_p50": 0.0,
        "atlas_reliability_base_p90": 0.0,
        "atlas_reliability_base_hist_low": 0.0,
        "atlas_reliability_base_hist_mid": 0.0,
        "atlas_reliability_base_hist_high": 0.0,
        "atlas_reliability_runtime_mapped_p10": 0.0,
        "atlas_reliability_runtime_mapped_p50": 0.0,
        "atlas_reliability_runtime_mapped_p90": 0.0,
        "atlas_reliability_runtime_mapped_hist_low": 0.0,
        "atlas_reliability_runtime_mapped_hist_mid": 0.0,
        "atlas_reliability_runtime_mapped_hist_high": 0.0,
        "atlas_reliability_effective_p10": 0.0,
        "atlas_reliability_effective_p50": 0.0,
        "atlas_reliability_effective_p90": 0.0,
        "atlas_reliability_effective_hist_low": 0.0,
        "atlas_reliability_effective_hist_mid": 0.0,
        "atlas_reliability_effective_hist_high": 0.0,
        "refresh_evidence_observed_gate_ratio": 0.0,
        "refresh_evidence_count_gate_ratio": 0.0,
        "refresh_evidence_visibility_gate_ratio": 0.0,
        "refresh_evidence_ref_gate_ratio": 0.0,
        "refresh_evidence_finite_gate_ratio": 0.0,
        "refresh_evidence_support_gate_ratio": 0.0,
        "refresh_base_runtime_override_gate_ratio": 0.0,
        "refresh_strong_runtime_evidence_ratio": 0.0,
        "refresh_evidence_override_gate_ratio": 0.0,
        "refresh_evidence_gate_mean": 0.0,
        "refresh_override_weight_positive_ratio": 0.0,
        "refresh_override_base_bucket_low_override_ratio": 0.0,
        "refresh_override_base_bucket_mid_override_ratio": 0.0,
        "refresh_override_base_bucket_high_override_ratio": 0.0,
        "atlas_refresh_done": 1 if bool(atlas_refresh_done) else 0,
        "state_stable_count": 0,
        "state_passive_count": 0,
        "state_active_count": 0,
        "state_out_pending_count": 0,
        "stable_ratio": 0.0,
        "passive_ratio": 0.0,
        "active_ratio": 0.0,
        "out_of_anchor_ratio": 0.0,
        "pending_ratio": 0.0,
        "out_of_anchor_pending_count": 0,
        "cooldown_ratio": 0.0,
        "mean_gc_fail_count": 0.0,
        "mean_active_lifetime": 0.0,
        "max_active_lifetime": 0.0,
        "active_candidate_pool_count": 0,
        "active_candidate_pool_ratio": 0.0,
        "candidate_formation_count": 0,
        "active_formed_count": 0,
        "active_standard_formation_count": 0,
        "active_rescue_fallback_formation_count": 0,
        "active_standard_candidate_pool_count": 0,
        "active_rescue_candidate_pool_count": 0,
        "active_admitted_count": 0,
        "active_promoted_count": 0,
        "active_admission_pool_count": 0,
        "active_standard_admission_pool_count": 0,
        "active_rescue_admission_pool_count": 0,
        "active_promote_count": 0,
        "active_standard_promote_count": 0,
        "active_forced_rescue_promote_count": 0,
        "active_new_active_count": 0,
        "active_rescue_candidate_count": 0,
        "active_rescue_promote_count": 0,
        "active_quota_target": 0,
        "active_quota_soft_target": 0,
        "active_quota_hard_target": 0,
        "active_quota_effective_live_active_count": 0,
        "active_quota_over_target_count": 0,
        "active_quota_available": 0,
        "active_quota_current_count": 0,
        "active_quota_before_release_count": 0,
        "active_quota_release_count": 0,
        "active_quota_projected_exit_count": 0,
        "active_quota_projected_after_exit": 0,
        "active_quota_after_release_count": 0,
        "active_quota_transition_overflow": 0,
        "explore_candidate_ratio": 0.0,
        "active_hard_exit_count": 0,
        "active_soft_exit_count": 0,
        "active_lifetime_cap_exit_count": 0,
        "active_nonimproving_exit_count": 0,
        "active_stale_fallback_exit_count": 0,
        "active_fallback_handoff_exit_count": 0,
        "active_lifetime_release_count": 0,
        "active_demoted_count": 0,
        "active_exited_count": 0,
        "active_provenance_preserved_count": 0,
        "passive_stable_ready_count": 0,
        "passive_stable_cooldown_bypass_count": 0,
        "state_reliability_stable_ready_count": 0,
        "runtime_stable_support_count": 0,
        "runtime_stable_support_ratio": 0.0,
        "runtime_recovery_support_count": 0,
        "runtime_recovery_support_ratio": 0.0,
        "effective_recovery_stable_ready_count": 0,
        "passive_runtime_recovery_ready_count": 0,
        "passive_explicit_stable_ready_count": 0,
        "active_carryover_count": 0,
        "active_state_rebuild_count": 0,
        "active_provenance_from_transition_passive_to_active_count": 0,
        "active_provenance_from_transition_passive_to_active_ratio": 0.0,
        "active_provenance_from_restore_checkpoint_count": 0,
        "active_provenance_from_restore_checkpoint_ratio": 0.0,
        "active_provenance_from_state_rebuild_after_gc_count": 0,
        "active_provenance_from_state_rebuild_after_gc_ratio": 0.0,
        "active_provenance_from_quota_carryover_count": 0,
        "active_provenance_from_quota_carryover_ratio": 0.0,
        "active_provenance_from_forced_rescue_bootstrap_count": 0,
        "active_provenance_from_forced_rescue_bootstrap_ratio": 0.0,
        "active_provenance_from_active_explore_clone_count": 0,
        "active_provenance_from_active_explore_clone_ratio": 0.0,
        "active_provenance_tracked_count": 0,
        "active_provenance_tracked_ratio": 0.0,
        "active_provenance_untracked_count": 0,
        "active_provenance_untracked_ratio": 0.0,
        "active_max_lifetime_iters": 0,
        "active_nonimprove_iters": 0,
        "active_to_stable_exit_count": 0,
        "active_to_passive_exit_count": 0,
        "active_to_explore_clone_handoff_count": 0.0,
        "transition_stable_to_passive_count": 0,
        "transition_passive_to_stable_count": 0,
        "transition_passive_to_active_count": 0,
        "transition_passive_to_active_standard_count": 0,
        "transition_passive_to_active_rescue_count": 0,
        "transition_passive_to_active_unclassified_count": 0,
        "transition_active_to_passive_count": 0,
        "transition_active_to_stable_count": 0,
        "transition_any_to_pending_count": 0,
        "observed_node_ratio": 0.0,
        "observed_node_count": 0.0,
        "updated_node_ratio": 0.0,
        "updated_node_count": 0.0,
        "coverage_node_update_count": 0.0,
        "mean_node_update_strength": 0.0,
        "mean_node_photo_ema": 0.0,
        "mean_node_visibility_ema": 0.0,
        "mean_node_obs_quality_ema": 0.0,
        "mean_node_observed_score_current": 0.0,
        "mean_node_observed_score_ema": 0.0,
        "mean_node_updated_recently": 0.0,
        "mean_node_support_consistency_current": 0.0,
        "mean_node_finite_projection_ema": 0.0,
        "mean_node_ref_consistency_ema": 0.0,
        "patch_quality_score": 0.0,
        "mask_nonzero_ratio": 0.0,
        "bg_like_ratio": 0.0,
        "background_like_ratio": 0.0,
        "thin_support_like_ratio": 0.0,
        "photo_signal_strength": 0.0,
        "patch_quality_candidate_mean": 0.0,
        "photo_signal_candidate_mean": 0.0,
        "split_count": 0.0,
        "clone_count": 0.0,
        "explore_clone_count": 0.0,
        "explore_candidate_count": 0.0,
        "explore_valid_ref_count": 0.0,
        "explore_live_active_count": 0.0,
        "explore_slab_discard_count": 0.0,
        "explore_slab_valid_count": 0.0,
        "explore_ref_repair_count": 0.0,
        "explore_slab_fallback_count": 0.0,
        "explore_slab_soft_clamp_count": 0.0,
        "explore_tau_soft_penalty_mean": 0.0,
        "explore_tau_jitter_abs_mean": 0.0,
        "explore_tau_span_mean": 0.0,
        "explore_adaptive_slab_mult_mean": 0.0,
        "explore_depth_delta_ratio_mean": 0.0,
        "explore_background_like_block_count": 0.0,
        "explore_background_like_selected_count": 0.0,
        "explore_background_like_slab_count": 0.0,
        "explore_active_rescue_slab_count": 0.0,
        "explore_neighbor_stable_slab_count": 0.0,
        "explore_support_only_conflict_count": 0.0,
        "explore_slab_admission_candidate_count": 0.0,
        "explore_slab_admission_valid_count": 0.0,
        "explore_slab_admission_added_count": 0.0,
        "explore_slab_admission_ref_repair_count": 0.0,
        "explore_slab_admission_background_like_block_count": 0.0,
        "explore_view_evidence_ready_count": 0.0,
        "stable_split_count": 0.0,
        "stable_split_candidate_count": 0.0,
        "stable_split_candidate_prebudget_count": 0.0,
        "stable_split_fidelity_refine_signal_count": 0.0,
        "stable_split_block_drift_count": 0.0,
        "stable_split_block_projector_count": 0.0,
        "stable_clone_count": 0.0,
        "stable_clone_candidate_count": 0.0,
        "stable_clone_candidate_prebudget_count": 0.0,
        "stable_clone_thin_background_support_count": 0.0,
        "stable_clone_block_pose_ref_count": 0.0,
        "stable_clone_block_projector_count": 0.0,
        "stable_clone_block_drift_count": 0.0,
        "active_explore_clone_count": 0.0,
        "split_child_scale_ratio_mean": 0.0,
        "split_child_scale_ratio_max": 0.0,
        "split_child_log_anisotropy_delta_mean": 0.0,
        "pruned_count": 0.0,
        "densify_stale_render_tensors": 0.0,
        "densify_skipped_stale_render_tensors": 0.0,
        "densify_used_cached_stats": 0.0,
        "densify_skip_reason": "none",
        "densify_stale_reason": "none",
        "densify_gc_pruned_render_tensor_invalidated": 0.0,
        "fidelity_handoff_gate": 0.0,
        "fidelity_handoff_completion_gate": 0.0,
        "fidelity_handoff_observed_gate": 0.0,
        "fidelity_handoff_dark_gate": 0.0,
        "fidelity_handoff_stable_gate": 0.0,
        "fidelity_handoff_floater_gate": 0.0,
        "fidelity_handoff_quality_gate": 0.0,
        "fidelity_handoff_late_phase_boost": 0.0,
        "fidelity_mode_gate": 0.0,
        "fidelity_mode_enabled": 0.0,
        "fidelity_mode_dark_gate": 0.0,
        "fidelity_mode_l1_gate": 0.0,
        "fidelity_mode_floater_gate": 0.0,
        "fidelity_mode_reliability_gate": 0.0,
        "fidelity_mode_pose_gate": 0.0,
        "fidelity_mode_recovery_gate": 0.0,
        "fidelity_prune_gate": 0.0,
        "maintenance_mode_enabled": 1.0,
        "densify_b2_unhealthy_gate": 0.0,
        "densify_b2_recovery_unhealthy_gate": 0.0,
        "densify_atlas_recovery_event_count": 0.0,
        "densify_atlas_recovery_seen": 0.0,
        "densify_b2_zero_grad_skip_delta": 0.0,
        "densify_effective_split_fraction": 0.0,
        "densify_effective_clone_fraction": 0.0,
        "densify_effective_explore_fraction": 0.0,
        "background_dead_prune_protected_count": 0.0,
        "fidelity_handoff_budget_scale": 1.0,
        "fidelity_handoff_split_scale": 1.0,
        "fidelity_handoff_clone_scale": 1.0,
        "fidelity_handoff_explore_scale": 1.0,
        "fidelity_handoff_active_noisy_prune_count": 0.0,
        "fidelity_handoff_unsupported_explore_prune_count": 0.0,
        "fidelity_handoff_unsupported_rescue_prune_count": 0.0,
        "active_noisy_pruned_count": 0.0,
        "unsupported_explore_pruned_count": 0.0,
        "unsupported_rescue_pruned_count": 0.0,
        "fidelity_handoff_dark_region_completeness_ema": 0.0,
        "gc_ran": 0.0,
        "gc_due": 0.0,
        "gc_interval": 0.0,
        "gc_candidates": 0.0,
        "reattach_success": 0.0,
        "reattach_fail": 0.0,
        "reattach_success_ratio": 0.0,
        "reattach_fail_ratio": 0.0,
        "pending_reattach_success": 0.0,
        "pending_reattach_fail": 0.0,
        "pending_reattach_success_ratio": 0.0,
        "pending_reattach_fail_ratio": 0.0,
        "reattach_tier1_attempt_count": 0.0,
        "reattach_tier1_raw_accept_count": 0.0,
        "reattach_tier1_success": 0.0,
        "reattach_tier2_attempt_count": 0.0,
        "reattach_tier2_raw_accept_count": 0.0,
        "reattach_tier2_success": 0.0,
        "reattach_tier3_attempt_count": 0.0,
        "reattach_tier3_raw_accept_count": 0.0,
        "reattach_tier3_success": 0.0,
        "reattach_tier4_attempt_count": 0.0,
        "reattach_tier4_forced_success": 0.0,
        "reattach_candidate_starvation_count": 0.0,
        "reattach_candidate_starvation_ratio": 0.0,
        "ray_guided_queries": 0.0,
        "ray_guided_priority_queries": 0.0,
        "ray_guided_late_queries": 0.0,
        "ray_guided_active_queries": 0.0,
        "ray_guided_pending_queries": 0.0,
        "ray_guided_quality_accept_count": 0.0,
        "prune_after_gc": 0.0,
        "nonfinite_kl_count": 0.0,
        "nonfinite_projected_energy_count": 0.0,
        "nonfinite_pose_count": 0.0,
        "nonfinite_clone_discard_count": 0.0,
        "invalid_gaussian_prune_count": 0.0,
        "pose_b1_enabled": 0,
        "pose_b2_enabled": 0,
        "pose_b1_gate_open": 0.0,
        "pose_b2_gate_open": 0.0,
        "pose_b1_loss": 0.0,
        "pose_b2_loss": 0.0,
        "pose_b1_attempted": 0.0,
        "pose_b1_executed": 0.0,
        "pose_b2_attempted": 0.0,
        "pose_b2_executed": 0.0,
        "pose_b1_attempt_count": 0.0,
        "pose_b1_execute_count": 0.0,
        "pose_b1_optimizer_step_count": 0.0,
        "pose_b2_attempt_count": 0.0,
        "pose_b2_execute_count": 0.0,
        "pose_b2_optimizer_step_count": 0.0,
        "pose_b1_skip_reason": "none",
        "pose_b2_skip_reason": "none",
        "pose_b1_gate_reason": "none",
        "pose_b2_gate_reason": "none",
        "pose_b1_gate_block_stable_ratio": 0.0,
        "pose_b1_gate_block_capacity_ratio": 0.0,
        "pose_b1_gate_block_trustworthy_corr": 0.0,
        "pose_b1_gate_block_drift": 0.0,
        "pose_b1_gate_block_freeze": 0.0,
        "pose_b2_gate_history_ready": 0.0,
        "pose_b2_gate_history_fresh": 0.0,
        "pose_b2_gate_history_ready_raw": 0.0,
        "pose_b2_gate_data_ready": 0.0,
        "pose_b2_gate_quality_ready": 0.0,
        "pose_b2_gate_optimization_ready": 0.0,
        "pose_b2_gate_enabled_for_compute": 0.0,
        "pose_b2_gate_enabled_for_step": 0.0,
        "pose_b2_data_ready": 0.0,
        "pose_b2_quality_ready": 0.0,
        "pose_b2_optimization_ready": 0.0,
        "pose_b2_enabled_for_compute": 0.0,
        "pose_b2_enabled_for_step": 0.0,
        "pose_b2_gate_bootstrap_open": 0.0,
        "pose_b2_gate_low_frequency_due": 0.0,
        "pose_b1_history_fresh": 0.0,
        "pose_b2_enabled_by_history": 0.0,
        "pose_b2_enabled_by_bootstrap": 0.0,
        "pose_b2_enabled_by_photo_corridor": 0.0,
        "pose_b1_geometry_ready": 0.0,
        "pose_b1_geometry_override": 0.0,
        "pose_b1_corridor_open": 0.0,
        "pose_b1_corr_quality": 0.0,
        "pose_b1_effective_update_interval": 0.0,
        "pose_b1_lr_scale": 1.0,
        "pose_b1_geo_pre_median_px": 0.0,
        "pose_b1_geo_post_median_px": 0.0,
        "pose_b1_geo_median_px_reduction": 0.0,
        "pose_b1_geo_reduction_ratio": 0.0,
        "pose_b1_geo_after_filter_corr": 0.0,
        "pose_b1_geo_selected_count": 0.0,
        "pose_b1_geo_selected_unique_node_count": 0.0,
        "pose_b1_success_residual_reduced": 0.0,
        "pose_b1_camera_success_criterion": 0.0,
        "pose_b1_history_healthy": 0.0,
        "pose_b1_no_improve_streak": 0.0,
        "pose_b2_photo_corridor_open": 0.0,
        "pose_b2_photo_corridor_scene_signal": 0.0,
        "pose_b2_photo_corridor_support_ready": 0.0,
        "pose_b2_corridor_step_ok": 0.0,
        "pose_b2_photo_loss_raw": 0.0,
        "pose_b2_photo_loss_weighted": 0.0,
        "pose_b2_trust_loss": 0.0,
        "pose_b2_combined_loss": 0.0,
        "pose_b2_mode": "none",
        "pose_b2_fullframe_stress_enabled": 0.0,
        "pose_fullframe_l1": 0.0,
        "pose_fullframe_ssim": 0.0,
        "pose_fullframe_gradient": 0.0,
        "pose_fullframe_total": 0.0,
        "pose_fullframe_num_pixels": 0.0,
        "pose_fullframe_downsample_factor": 0.0,
        "pose_b2_trust_max_scale_effective": 0.0,
        "pose_b2_patch_count_used": 0.0,
        "pose_b2_mask_mean": 0.0,
        "pose_b2_mask_nonzero_ratio": 0.0,
        "pose_b2_photo_signal_strength": 0.0,
        "pose_b2_patch_grad_observable_ratio": 0.0,
        "pose_b2_passive_safe_sample_count": 0.0,
        "pose_b2_passive_safe_sample_fraction": 0.0,
        "pose_b2_passive_safe_candidate_count": 0.0,
        "pose_b2_passive_safe_trust_mean": 0.0,
        "pose_b2_passive_safe_reliability_mean": 0.0,
        "pose_b2_observable_patch_ok": 0.0,
        "pose_b2_grad_nonzero": 0.0,
        "pose_b2_any_grad_nonzero": 0.0,
        "pose_b2_data_grad_nonzero": 0.0,
        "pose_b2_post_trust_grad_nonzero": 0.0,
        "pose_b2_grad_total_ok": 0.0,
        "pose_b2_grad_axis_ok": 0.0,
        "pose_b2_pose_delta_q_requires_grad": 0.0,
        "pose_b2_pose_delta_t_requires_grad": 0.0,
        "pose_b2_pose_q_grad_is_none": 1.0,
        "pose_b2_pose_t_grad_is_none": 1.0,
        "pose_b2_pose_q_grad_norm": 0.0,
        "pose_b2_pose_t_grad_norm": 0.0,
        "pose_b2_pose_grad_norm_total": 0.0,
        "pose_b2_pose_graph_connected": 0.0,
        "pose_b2_loss_depends_on_pose_path": 0.0,
        "pose_b2_pose_graph_connected_but_tiny": 0.0,
        "pose_b2_trust_choked_but_pose_grad_exists": 0.0,
        "pose_b2_data_grad_from_autograd_audit": 0.0,
        "pose_b2_pose_autograd_error": "none",
        "pose_b2_world_view_transform_requires_grad": 0.0,
        "pose_b2_full_proj_transform_requires_grad": 0.0,
        "pose_b2_camera_center_requires_grad": 0.0,
        "pose_b2_pre_trust_grad_norm": 0.0,
        "pose_b2_post_trust_grad_norm": 0.0,
        "pose_b2_pre_to_post_grad_shrink_ratio": 0.0,
        "pose_b2_pre_to_post_trans_grad_shrink_ratio": 0.0,
        "pose_b2_pre_to_post_rot_grad_shrink_ratio": 0.0,
        "pose_b2_data_only_q_grad_norm": 0.0,
        "pose_b2_data_only_t_grad_norm": 0.0,
        "pose_b2_data_only_grad_total": 0.0,
        "pose_b2_data_only_vs_total_grad_ratio": 0.0,
        "pose_b2_fd_probe_enabled": 0.0,
        "pose_b2_fd_probe_status": "disabled",
        "pose_b2_fd_probe_count": 0.0,
        "pose_b2_fd_trans_sensitivity_x": 0.0,
        "pose_b2_fd_trans_sensitivity_y": 0.0,
        "pose_b2_fd_trans_sensitivity_z": 0.0,
        "pose_b2_fd_rot_sensitivity_x": 0.0,
        "pose_b2_fd_rot_sensitivity_y": 0.0,
        "pose_b2_fd_rot_sensitivity_z": 0.0,
        "pose_b2_fd_trans_sensitivity_max": 0.0,
        "pose_b2_fd_rot_sensitivity_max": 0.0,
        "pose_b2_fd_any_positive": 0.0,
        "pose_b2_fd_flat": 0.0,
        "pose_b2_fd_nonflat_autograd_tiny": 0.0,
        "pose_b2_fd_nonflat_trust_choked": 0.0,
        "pose_b2_photo_grad_norm_rot": 0.0,
        "pose_b2_photo_grad_norm_trans": 0.0,
        "pose_b2_step_by_total_grad": 0.0,
        "pose_b2_step_by_data_grad": 0.0,
        "pose_b2_step_allowed": 0.0,
        "pose_b2_skip_reason_detailed": "none",
        "pose_b2_small_step_mode": 0.0,
        "pose_b2_small_valid_step_ok": 0.0,
        "pose_b2_small_step_lr_scale": 0.0,
        "pose_b2_use_data_grad_for_step": 0.0,
        "pose_b2_microstep_allowed": 0.0,
        "pose_b2_microstep_mode": 0.0,
        "pose_b2_microstep_translation_applied": 0.0,
        "pose_b2_microstep_rotation_applied_deg": 0.0,
        "pose_b2_microstep_reason": "inactive",
        "pose_freeze_recovery_good_streak": 0.0,
        "runtime_pose_gate_enabled": 0.0,
        "runtime_pose_gate_open": 0.0,
        "pose_trust_clamp_count": 0.0,
        "sigma_parallel_clamp_hits": 0.0,
        "sigma_support_clamp_hits": 0.0,
        "sigma_ray_clamp_hits": 0.0,
        "sigma_ray_floor_hits": 0.0,
        "sigma_active_ray_valid_count": 0.0,
        "sigma_active_ray_unresolved_count": 0.0,
        "sigma_parallel_clamp_hits_stable": 0.0,
        "sigma_parallel_clamp_hits_passive": 0.0,
        "sigma_parallel_clamp_hits_active": 0.0,
        "sigma_support_clamp_hits_stable": 0.0,
        "sigma_support_clamp_hits_passive": 0.0,
        "sigma_support_clamp_hits_active": 0.0,
        "sigma_stable_parallel_mean": 0.0,
        "sigma_stable_support_mean": 0.0,
        "sigma_passive_parallel_mean": 0.0,
        "sigma_passive_support_mean": 0.0,
        "sigma_active_parallel_mean": 0.0,
        "sigma_active_support_mean": 0.0,
        "sigma_active_parallel_p50": 0.0,
        "sigma_active_parallel_p90": 0.0,
        "sigma_active_support_p50": 0.0,
        "sigma_active_support_p90": 0.0,
        "sigma_active_ray_span_mean": 0.0,
        "sigma_active_ray_floor_mean": 0.0,
        "sigma_active_ray_cap_mean": 0.0,
        "sigma_active_ray_parallel_mean": 0.0,
        "sigma_active_ray_parallel_p90": 0.0,
        "atlas_rank_u_mean": 0.0,
        "atlas_active_ray_count": 0.0,
        "atlas_active_ray_fraction": 0.0,
        "atlas_active_ray_valid_fraction": 0.0,
        "atlas_active_ray_fallback_count": 0.0,
        "atlas_active_ray_fallback_fraction": 0.0,
        "atlas_kl_stable_mean": 0.0,
        "atlas_kl_passive_mean": 0.0,
        "atlas_kl_active_mean": 0.0,
        "atlas_slab_total_loss": 0.0,
        "atlas_slab_active_fraction": 0.0,
        "atlas_slab_active_count": 0.0,
        "atlas_slab_valid_count": 0.0,
        "atlas_slab_mean_penalty": 0.0,
        "atlas_slab_violation_count": 0.0,
        "atlas_slab_violation_ratio": 0.0,
        "atlas_slab_fallback_count": 0.0,
        "atlas_slab_ref_repair_count": 0.0,
        "warmup_only": 0,
        "main_phase": 0,
        "refresh_pending": 0,
        "main_phase_ready": 0,
        "pose_refine_disabled_or_blocked_by_phase": 0,
        "atlas_refresh_snapshot_ready": 0,
        "atlas_refresh_snapshot_observed_ratio": 0.0,
        "atlas_refresh_snapshot_observed_count": 0.0,
        "atlas_refresh_snapshot_photo_ema_mean": 0.0,
        "atlas_refresh_snapshot_visibility_ema_mean": 0.0,
        "atlas_refresh_snapshot_obs_quality_mean": 0.0,
        "atlas_refresh_snapshot_obs_quality_max": 0.0,
        "corr_source_atlas_ratio": 0.0,
        "corr_source_fallback_ratio": 0.0,
        "atlas_mean_weight_effective": 0.0,
        "atlas_mean_weight_stress_scale": 1.0,
        "atlas_mean_weight_stress_enabled": 0.0,
        "refresh_override_count": 0.0,
        "refresh_override_ratio": 0.0,
        "refresh_std_before_after": {"before": 0.0, "after": 0.0},
        "state_candidate_reason_breakdown": {},
        "promotion_block_reason_breakdown": {},
        "pose_b1_quality_breakdown": {},
        "pose_b2_skip_reason_breakdown": {},
        "explore_clone_success_count": 0.0,
        "floater_proxy_by_state": {},
        "dark_region_completeness_by_state": {},
    }

    densify_metrics = densify_metrics or {}
    atlas_gc_metrics = atlas_gc_metrics or {}
    pose_metrics = pose_metrics or {}
    atlas_metrics = atlas_metrics or {}
    atlas_uncertainty_metrics = atlas_uncertainty_metrics or {}
    atlas_runtime_metrics = atlas_runtime_metrics or {}
    atlas_slab_metrics = atlas_slab_metrics or {}
    atlas_kl_metrics = atlas_kl_metrics or {}
    atlas_phase = atlas_phase or {}

    required_fields["atlas_mean_weight_effective"] = serialize_metric_value(
        atlas_metrics.get("atlas_mean_weight_effective", 0.0)
    )
    required_fields["atlas_mean_weight_stress_scale"] = serialize_metric_value(
        atlas_metrics.get("atlas_mean_weight_stress_scale", 1.0)
    )
    required_fields["atlas_mean_weight_stress_enabled"] = serialize_metric_value(
        atlas_metrics.get("atlas_mean_weight_stress_enabled", 0.0)
    )
    required_fields["split_count"] = serialize_metric_value(densify_metrics.get("split_count", 0.0))
    required_fields["clone_count"] = serialize_metric_value(densify_metrics.get("clone_count", 0.0))
    required_fields["explore_clone_count"] = serialize_metric_value(densify_metrics.get("explore_clone_count", 0.0))
    required_fields["explore_clone_success_count"] = serialize_metric_value(
        densify_metrics.get("explore_clone_success_count", densify_metrics.get("explore_clone_count", 0.0))
    )
    required_fields["explore_candidate_count"] = serialize_metric_value(densify_metrics.get("explore_candidate_count", 0.0))
    required_fields["explore_candidate_prebudget_count"] = serialize_metric_value(densify_metrics.get("explore_candidate_prebudget_count", 0.0))
    required_fields["explore_valid_ref_count"] = serialize_metric_value(densify_metrics.get("explore_valid_ref_count", 0.0))
    required_fields["explore_live_active_count"] = serialize_metric_value(densify_metrics.get("explore_live_active_count", 0.0))
    required_fields["explore_slab_discard_count"] = serialize_metric_value(densify_metrics.get("explore_slab_discard_count", 0.0))
    required_fields["explore_slab_valid_count"] = serialize_metric_value(densify_metrics.get("explore_slab_valid_count", 0.0))
    required_fields["explore_ref_repair_count"] = serialize_metric_value(densify_metrics.get("explore_ref_repair_count", 0.0))
    required_fields["explore_slab_fallback_count"] = serialize_metric_value(densify_metrics.get("explore_slab_fallback_count", 0.0))
    required_fields["explore_slab_soft_clamp_count"] = serialize_metric_value(densify_metrics.get("explore_slab_soft_clamp_count", 0.0))
    required_fields["explore_tau_soft_penalty_mean"] = serialize_metric_value(densify_metrics.get("explore_tau_soft_penalty_mean", 0.0))
    required_fields["explore_tau_jitter_abs_mean"] = serialize_metric_value(densify_metrics.get("explore_tau_jitter_abs_mean", 0.0))
    required_fields["explore_tau_span_mean"] = serialize_metric_value(densify_metrics.get("explore_tau_span_mean", 0.0))
    for metric_name in (
        "explore_adaptive_slab_mult_mean",
        "explore_depth_delta_ratio_mean",
        "explore_background_like_block_count",
        "explore_background_like_selected_count",
        "explore_background_like_slab_count",
        "explore_active_rescue_slab_count",
        "explore_neighbor_stable_slab_count",
        "explore_support_only_conflict_count",
    ):
        required_fields[metric_name] = serialize_metric_value(densify_metrics.get(metric_name, 0.0))
    required_fields["explore_slab_admission_candidate_count"] = serialize_metric_value(
        densify_metrics.get("explore_slab_admission_candidate_count", 0.0)
    )
    required_fields["explore_slab_admission_valid_count"] = serialize_metric_value(
        densify_metrics.get("explore_slab_admission_valid_count", 0.0)
    )
    required_fields["explore_slab_admission_added_count"] = serialize_metric_value(
        densify_metrics.get("explore_slab_admission_added_count", 0.0)
    )
    required_fields["explore_slab_admission_ref_repair_count"] = serialize_metric_value(
        densify_metrics.get("explore_slab_admission_ref_repair_count", 0.0)
    )
    required_fields["explore_slab_admission_background_like_block_count"] = serialize_metric_value(
        densify_metrics.get("explore_slab_admission_background_like_block_count", 0.0)
    )
    required_fields["explore_view_evidence_ready_count"] = serialize_metric_value(
        densify_metrics.get("explore_view_evidence_ready_count", 0.0)
    )
    required_fields["stable_split_count"] = serialize_metric_value(densify_metrics.get("stable_split_count", 0.0))
    required_fields["stable_split_candidate_count"] = serialize_metric_value(densify_metrics.get("stable_split_candidate_count", 0.0))
    required_fields["stable_split_candidate_prebudget_count"] = serialize_metric_value(densify_metrics.get("stable_split_candidate_prebudget_count", 0.0))
    required_fields["stable_split_fidelity_refine_signal_count"] = serialize_metric_value(densify_metrics.get("stable_split_fidelity_refine_signal_count", 0.0))
    required_fields["stable_split_block_drift_count"] = serialize_metric_value(densify_metrics.get("stable_split_block_drift_count", 0.0))
    required_fields["stable_split_block_projector_count"] = serialize_metric_value(densify_metrics.get("stable_split_block_projector_count", 0.0))
    required_fields["stable_split_support_ready_count"] = serialize_metric_value(densify_metrics.get("stable_split_support_ready_count", 0.0))
    required_fields["stable_split_coverage_not_thin_count"] = serialize_metric_value(densify_metrics.get("stable_split_coverage_not_thin_count", 0.0))
    required_fields["stable_clone_count"] = serialize_metric_value(densify_metrics.get("stable_clone_count", 0.0))
    required_fields["stable_clone_candidate_count"] = serialize_metric_value(densify_metrics.get("stable_clone_candidate_count", 0.0))
    required_fields["stable_clone_candidate_prebudget_count"] = serialize_metric_value(densify_metrics.get("stable_clone_candidate_prebudget_count", 0.0))
    required_fields["stable_clone_thin_background_support_count"] = serialize_metric_value(densify_metrics.get("stable_clone_thin_background_support_count", 0.0))
    required_fields["stable_clone_block_pose_ref_count"] = serialize_metric_value(densify_metrics.get("stable_clone_block_pose_ref_count", 0.0))
    required_fields["stable_clone_block_projector_count"] = serialize_metric_value(densify_metrics.get("stable_clone_block_projector_count", 0.0))
    required_fields["stable_clone_block_drift_count"] = serialize_metric_value(densify_metrics.get("stable_clone_block_drift_count", 0.0))
    required_fields["stable_clone_projected_drift_small_count"] = serialize_metric_value(densify_metrics.get("stable_clone_projected_drift_small_count", 0.0))
    required_fields["stable_clone_recent_transition_block_count"] = serialize_metric_value(densify_metrics.get("stable_clone_recent_transition_block_count", 0.0))
    required_fields["stable_clone_suppressed_by_split_count"] = serialize_metric_value(densify_metrics.get("stable_clone_suppressed_by_split_count", 0.0))
    required_fields["active_explore_clone_count"] = serialize_metric_value(densify_metrics.get("active_explore_clone_count", 0.0))
    required_fields["active_to_explore_clone_handoff_count"] = serialize_metric_value(densify_metrics.get("active_explore_clone_count", 0.0))
    required_fields["background_fidelity_protected_count"] = serialize_metric_value(densify_metrics.get("background_fidelity_protected_count", 0.0))
    required_fields["split_child_scale_ratio_mean"] = serialize_metric_value(
        densify_metrics.get("split_child_scale_ratio_mean", 0.0)
    )
    required_fields["split_child_scale_ratio_max"] = serialize_metric_value(
        densify_metrics.get("split_child_scale_ratio_max", 0.0)
    )
    required_fields["split_child_log_anisotropy_delta_mean"] = serialize_metric_value(
        densify_metrics.get("split_child_log_anisotropy_delta_mean", 0.0)
    )
    required_fields["densify_stale_render_tensors"] = serialize_metric_value(densify_metrics.get("densify_stale_render_tensors", 0.0))
    required_fields["densify_used_cached_stats"] = serialize_metric_value(densify_metrics.get("densify_used_cached_stats", 0.0))
    required_fields["densify_skipped_stale_render_tensors"] = serialize_metric_value(densify_metrics.get("densify_skipped_stale_render_tensors", 0.0))
    required_fields["densify_skip_reason"] = serialize_metric_value(densify_metrics.get("densify_skip_reason", "none"))
    required_fields["densify_budget_scale"] = serialize_metric_value(densify_metrics.get("densify_budget_scale", 1.0))
    required_fields["densify_global_quota"] = serialize_metric_value(densify_metrics.get("densify_global_quota", 0.0))
    required_fields["densify_split_quota"] = serialize_metric_value(densify_metrics.get("densify_split_quota", 0.0))
    required_fields["densify_clone_quota"] = serialize_metric_value(densify_metrics.get("densify_clone_quota", 0.0))
    required_fields["densify_explore_quota"] = serialize_metric_value(densify_metrics.get("densify_explore_quota", 0.0))
    required_fields["densify_budget_phase_ramp"] = serialize_metric_value(densify_metrics.get("densify_budget_phase_ramp", 1.0))
    required_fields["densify_budget_b2_health"] = serialize_metric_value(densify_metrics.get("densify_budget_b2_health", 1.0))
    required_fields["densify_budget_floater_guard"] = serialize_metric_value(densify_metrics.get("densify_budget_floater_guard", 1.0))
    required_fields["densify_budget_quality_guard"] = serialize_metric_value(densify_metrics.get("densify_budget_quality_guard", 1.0))
    for metric_name in (
        "densify_b2_unhealthy_gate",
        "densify_b2_recovery_unhealthy_gate",
        "densify_atlas_recovery_event_count",
        "densify_atlas_recovery_seen",
        "densify_b2_zero_grad_skip_delta",
        "densify_effective_split_fraction",
        "densify_effective_clone_fraction",
        "densify_effective_explore_fraction",
        "maintenance_mode_enabled",
        "background_dead_prune_protected_count",
        "background_dead_prune_guard_candidate_count",
        "fidelity_prune_gate",
    ):
        required_fields[metric_name] = serialize_metric_value(densify_metrics.get(metric_name, 0.0))
    for metric_name, default_value in (
        ("fidelity_handoff_gate", 0.0),
        ("fidelity_handoff_completion_gate", 0.0),
        ("fidelity_handoff_observed_gate", 0.0),
        ("fidelity_handoff_dark_gate", 0.0),
        ("fidelity_handoff_stable_gate", 0.0),
        ("fidelity_handoff_floater_gate", 0.0),
        ("fidelity_handoff_quality_gate", 0.0),
        ("fidelity_handoff_late_phase_boost", 0.0),
        ("fidelity_mode_gate", 0.0),
        ("fidelity_mode_enabled", densify_metrics.get("fidelity_mode_gate", 0.0)),
        ("fidelity_mode_dark_gate", 0.0),
        ("fidelity_mode_l1_gate", 0.0),
        ("fidelity_mode_floater_gate", 0.0),
        ("fidelity_mode_reliability_gate", 0.0),
        ("fidelity_mode_pose_gate", 0.0),
        ("fidelity_mode_recovery_gate", 0.0),
        ("fidelity_handoff_budget_scale", 1.0),
        ("fidelity_handoff_split_scale", 1.0),
        ("fidelity_handoff_clone_scale", 1.0),
        ("fidelity_handoff_explore_scale", 1.0),
        ("fidelity_handoff_active_noisy_prune_count", 0.0),
        ("fidelity_handoff_unsupported_explore_prune_count", 0.0),
        ("fidelity_handoff_unsupported_rescue_prune_count", 0.0),
        ("active_noisy_pruned_count", 0.0),
        ("unsupported_explore_pruned_count", 0.0),
        ("unsupported_rescue_pruned_count", 0.0),
        ("fidelity_handoff_dark_region_completeness_ema", 0.0),
    ):
        required_fields[metric_name] = serialize_metric_value(densify_metrics.get(metric_name, default_value))
    required_fields["densify_stale_reason"] = serialize_metric_value(densify_metrics.get("densify_stale_reason", "none"))
    required_fields["densify_gc_pruned_render_tensor_invalidated"] = serialize_metric_value(
        densify_metrics.get("densify_gc_pruned_render_tensor_invalidated", 0.0)
    )
    required_fields["pruned_count"] = serialize_metric_value(densify_metrics.get("pruned_count", 0.0))
    required_fields["prune_after_gc"] = serialize_metric_value(
        densify_metrics.get("prune_after_gc", atlas_gc_metrics.get("prune_after_gc", 0.0))
    )
    required_fields["gc_ran"] = serialize_metric_value(
        atlas_gc_metrics.get("gc_ran", atlas_runtime_metrics.get("gc_ran", 0.0))
    )
    required_fields["gc_due"] = serialize_metric_value(atlas_runtime_metrics.get("gc_due", 0.0))
    required_fields["gc_interval"] = serialize_metric_value(atlas_runtime_metrics.get("gc_interval", 0.0))
    required_fields["gc_candidates"] = serialize_metric_value(atlas_gc_metrics.get("gc_candidates", 0.0))
    required_fields["reattach_success"] = serialize_metric_value(atlas_gc_metrics.get("reattach_success", 0.0))
    required_fields["reattach_fail"] = serialize_metric_value(atlas_gc_metrics.get("reattach_fail", 0.0))
    required_fields["reattach_success_ratio"] = serialize_metric_value(atlas_gc_metrics.get("reattach_success_ratio", 0.0))
    required_fields["reattach_fail_ratio"] = serialize_metric_value(atlas_gc_metrics.get("reattach_fail_ratio", 0.0))
    required_fields["pending_reattach_success"] = serialize_metric_value(atlas_gc_metrics.get("pending_reattach_success", 0.0))
    required_fields["pending_reattach_fail"] = serialize_metric_value(atlas_gc_metrics.get("pending_reattach_fail", 0.0))
    required_fields["pending_reattach_success_ratio"] = serialize_metric_value(atlas_gc_metrics.get("pending_reattach_success_ratio", 0.0))
    required_fields["pending_reattach_fail_ratio"] = serialize_metric_value(atlas_gc_metrics.get("pending_reattach_fail_ratio", 0.0))
    for metric_name in (
        "reattach_tier1_attempt_count",
        "reattach_tier1_raw_accept_count",
        "reattach_tier1_success",
        "reattach_tier2_attempt_count",
        "reattach_tier2_raw_accept_count",
        "reattach_tier2_success",
        "reattach_tier3_attempt_count",
        "reattach_tier3_raw_accept_count",
        "reattach_tier3_success",
        "reattach_tier4_attempt_count",
        "reattach_tier4_forced_success",
        "reattach_candidate_starvation_count",
        "reattach_candidate_starvation_ratio",
        "ray_guided_queries",
        "ray_guided_priority_queries",
        "ray_guided_late_queries",
        "ray_guided_active_queries",
        "ray_guided_pending_queries",
        "ray_guided_quality_accept_count",
    ):
        required_fields[metric_name] = serialize_metric_value(atlas_gc_metrics.get(metric_name, 0.0))
    required_fields["nonfinite_kl_count"] = serialize_metric_value(atlas_kl_metrics.get("nonfinite_kl_count", 0.0))
    required_fields["nonfinite_projected_energy_count"] = serialize_metric_value(atlas_metrics.get("nonfinite_projected_energy_count", 0.0))
    required_fields["nonfinite_pose_count"] = serialize_metric_value(pose_metrics.get("nonfinite_pose_count", 0.0))
    required_fields["nonfinite_clone_discard_count"] = serialize_metric_value(densify_metrics.get("nonfinite_clone_discard_count", 0.0))
    required_fields["invalid_gaussian_prune_count"] = serialize_metric_value(densify_metrics.get("invalid_gaussian_prune_count", 0.0))
    required_fields["pose_b1_enabled"] = int(bool(atlas_phase.get("enable_pose_b1", False)))
    required_fields["pose_b2_enabled"] = int(bool(atlas_phase.get("enable_pose_b2", False)))
    required_fields["pose_b1_gate_open"] = serialize_metric_value(
        pose_metrics.get("b1_gate_open", pose_metrics.get("b1_gate_enabled", 1.0 if atlas_phase.get("enable_pose_b1", False) else 0.0))
    )
    required_fields["pose_b2_gate_open"] = serialize_metric_value(
        pose_metrics.get("b2_gate_open", pose_metrics.get("b2_gate_enabled", 1.0 if atlas_phase.get("enable_pose_b2", False) else 0.0))
    )
    required_fields["warmup_only"] = int(bool(atlas_phase.get("warmup_only", False)))
    required_fields["main_phase"] = int(bool(atlas_phase.get("main_phase", False)))
    required_fields["refresh_pending"] = int(bool(atlas_phase.get("refresh_pending", False)))
    required_fields["main_phase_ready"] = int(bool(atlas_phase.get("main_phase_ready", False)))
    required_fields["pose_refine_disabled_or_blocked_by_phase"] = int(
        bool(atlas_phase.get("pose_refine_disabled_or_blocked_by_phase", False))
    )
    required_fields["pose_b1_loss"] = serialize_metric_value(pose_metrics.get("b1_total_loss", 0.0))
    required_fields["pose_b2_loss"] = serialize_metric_value(pose_metrics.get("b2_total_loss", 0.0))
    required_fields["pose_b1_attempted"] = serialize_metric_value(pose_metrics.get("b1_attempted", 0.0))
    required_fields["pose_b1_executed"] = serialize_metric_value(pose_metrics.get("b1_executed", 0.0))
    required_fields["pose_b2_attempted"] = serialize_metric_value(pose_metrics.get("b2_attempted", 0.0))
    required_fields["pose_b2_executed"] = serialize_metric_value(pose_metrics.get("b2_executed", 0.0))
    required_fields["pose_b1_attempt_count"] = serialize_metric_value(
        pose_metrics.get("b1_attempt_count", pose_metrics.get("b1_attempted", 0.0))
    )
    required_fields["pose_b1_execute_count"] = serialize_metric_value(
        pose_metrics.get("b1_execute_count", pose_metrics.get("b1_executed", 0.0))
    )
    required_fields["pose_b1_optimizer_step_count"] = serialize_metric_value(
        pose_metrics.get("b1_optimizer_step_count", pose_metrics.get("b1_executed", 0.0))
    )
    required_fields["pose_b2_attempt_count"] = serialize_metric_value(
        pose_metrics.get("b2_attempt_count", pose_metrics.get("b2_attempted", 0.0))
    )
    required_fields["pose_b2_execute_count"] = serialize_metric_value(
        pose_metrics.get("b2_execute_count", pose_metrics.get("b2_executed", 0.0))
    )
    required_fields["pose_b2_optimizer_step_count"] = serialize_metric_value(
        pose_metrics.get("b2_optimizer_step_count", pose_metrics.get("b2_executed", 0.0))
    )
    required_fields["pose_b2_template_fixed_coords"] = serialize_metric_value(
        pose_metrics.get("b2_pose_template_fixed_coords", 0.0)
    )
    required_fields["pose_b2_grad_norm_total"] = serialize_metric_value(
        pose_metrics.get("b2_pose_grad_norm_total", 0.0)
    )
    required_fields["pose_b1_skip_reason"] = serialize_metric_value(pose_metrics.get("b1_skip_reason", "none"))
    required_fields["pose_b2_skip_reason"] = serialize_metric_value(pose_metrics.get("b2_skip_reason", "none"))
    pose_b1_gate_reason = str(
        atlas_runtime_metrics.get(
            "pose_gate_b1_reason",
            pose_metrics.get("pose_gate_b1_reason", pose_metrics.get("b1_skip_reason", "none")),
        )
    )
    pose_b2_gate_reason = str(
        atlas_runtime_metrics.get(
            "pose_gate_b2_reason",
            pose_metrics.get("pose_gate_b2_reason", pose_metrics.get("b2_skip_reason", "none")),
        )
    )
    required_fields["pose_b1_gate_reason"] = serialize_metric_value(pose_b1_gate_reason)
    required_fields["pose_b2_gate_reason"] = serialize_metric_value(pose_b2_gate_reason)
    required_fields["pose_b1_gate_block_stable_ratio"] = 1.0 if "stable_ratio_low" in pose_b1_gate_reason else 0.0
    required_fields["pose_b1_gate_block_capacity_ratio"] = 1.0 if "capacity_ratio_low" in pose_b1_gate_reason else 0.0
    required_fields["pose_b1_gate_block_trustworthy_corr"] = (
        1.0
        if (
            "insufficient_correspondence" in pose_b1_gate_reason
            or "trustworthy_corr" in pose_b1_gate_reason
            or "corr" in pose_b1_gate_reason and "ready" not in pose_b1_gate_reason
        )
        else 0.0
    )
    required_fields["pose_b1_gate_block_drift"] = 1.0 if "drift_ratio_high" in pose_b1_gate_reason else 0.0
    required_fields["pose_b1_gate_block_freeze"] = (
        1.0
        if float(atlas_runtime_metrics.get("pose_freeze_active", pose_metrics.get("pose_freeze_active", 0.0))) > 0.5
        else 0.0
    )
    required_fields["corr_source_atlas_ratio"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_atlas_native_selected_ratio", pose_metrics.get("b1_pose_geo_atlas_native_ratio", 0.0))
    )
    required_fields["corr_source_fallback_ratio"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_fallback_selected_ratio", pose_metrics.get("b1_pose_geo_fallback_ratio", 0.0))
    )
    required_fields["pose_b1_quality_breakdown"] = serialize_metric_value(pose_metrics.get("pose_b1_quality_breakdown", {}))
    required_fields["pose_b2_skip_reason_breakdown"] = serialize_metric_value(
        pose_metrics.get("pose_b2_skip_reason_breakdown", _reason_breakdown(pose_metrics.get("b2_skip_reason", "none")))
    )
    required_fields["pose_b2_gate_history_ready"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_b2_gate_history_ready", pose_metrics.get("b2_history_ready", 0.0))
    )
    required_fields["pose_b2_gate_history_ready_raw"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_b2_gate_history_ready_raw", 0.0)
    )
    required_fields["pose_b2_gate_history_fresh"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_b2_gate_history_fresh", atlas_runtime_metrics.get("pose_gate_b2_b1_history_fresh", 0.0))
    )
    for metric_name in (
        "pose_b2_gate_data_ready",
        "pose_b2_gate_quality_ready",
        "pose_b2_gate_optimization_ready",
        "pose_b2_gate_enabled_for_compute",
        "pose_b2_gate_enabled_for_step",
    ):
        required_fields[metric_name] = serialize_metric_value(
            atlas_runtime_metrics.get(metric_name, pose_metrics.get(metric_name.replace("pose_b2_gate_", "b2_gate_"), 0.0))
        )
    for alias_name, gate_name in (
        ("pose_b2_data_ready", "pose_b2_gate_data_ready"),
        ("pose_b2_quality_ready", "pose_b2_gate_quality_ready"),
        ("pose_b2_optimization_ready", "pose_b2_gate_optimization_ready"),
        ("pose_b2_enabled_for_compute", "pose_b2_gate_enabled_for_compute"),
        ("pose_b2_enabled_for_step", "pose_b2_gate_enabled_for_step"),
    ):
        required_fields[alias_name] = serialize_metric_value(required_fields.get(gate_name, 0.0))
    required_fields["pose_b2_gate_bootstrap_open"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b2_bootstrap_open", pose_metrics.get("b2_bootstrap_open", 0.0))
    )
    required_fields["pose_b2_gate_low_frequency_due"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b2_low_frequency_due", pose_metrics.get("b2_low_frequency_due", 0.0))
    )
    b2_history_ready_value = float(required_fields.get("pose_b2_gate_history_ready", 0.0))
    b2_bootstrap_open_value = float(required_fields.get("pose_b2_gate_bootstrap_open", 0.0))
    b2_gate_enabled_value = float(
        atlas_runtime_metrics.get("pose_gate_b2_enabled", pose_metrics.get("b2_gate_enabled", required_fields.get("pose_b2_enabled", 0.0)))
    )
    required_fields["pose_b1_history_fresh"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b2_b1_history_fresh", required_fields.get("pose_b2_gate_history_fresh", 0.0))
    )
    required_fields["pose_b2_enabled_by_history"] = 1.0 if b2_gate_enabled_value > 0.5 and b2_history_ready_value > 0.5 else 0.0
    required_fields["pose_b2_enabled_by_bootstrap"] = 1.0 if b2_gate_enabled_value > 0.5 and b2_bootstrap_open_value > 0.5 else 0.0
    required_fields["pose_b2_enabled_by_photo_corridor"] = serialize_metric_value(
        pose_metrics.get("b2_photo_corridor_open", atlas_runtime_metrics.get("pose_gate_b2_photo_corridor_open", 0.0))
    )
    required_fields["pose_b1_geometry_ready"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b1_geometry_ready", 0.0)
    )
    required_fields["pose_b1_geometry_override"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b1_geometry_override", 0.0)
    )
    required_fields["pose_b1_corridor_open"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b1_corridor_open", 0.0)
    )
    required_fields["pose_b1_corr_quality"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b1_corr_quality", 0.0)
    )
    required_fields["pose_b1_effective_update_interval"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b1_effective_update_interval", pose_metrics.get("b1_effective_update_interval", 0.0))
    )
    required_fields["pose_b1_lr_scale"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_b1_lr_scale", pose_metrics.get("b1_lr_scale", 1.0))
    )
    required_fields["pose_b1_geo_pre_median_px"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_pre_median_px", pose_metrics.get("b1_pose_geo_selected_median_px_error", 0.0))
    )
    required_fields["pose_b1_geo_post_median_px"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_post_median_px", 0.0)
    )
    required_fields["pose_b1_geo_median_px_reduction"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_median_px_reduction", 0.0)
    )
    required_fields["pose_b1_geo_reduction_ratio"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_reduction_ratio", 0.0)
    )
    required_fields["pose_b1_geo_after_filter_corr"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_after_filter_corr", 0.0)
    )
    required_fields["pose_b1_geo_selected_count"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_selected_count", 0.0)
    )
    required_fields["pose_b1_geo_selected_unique_node_count"] = serialize_metric_value(
        pose_metrics.get("b1_pose_geo_selected_unique_node_count", 0.0)
    )
    required_fields["pose_b1_success_residual_reduced"] = serialize_metric_value(
        pose_metrics.get("b1_success_residual_reduced", 0.0)
    )
    required_fields["pose_b1_camera_success_criterion"] = serialize_metric_value(
        pose_metrics.get("b1_camera_success_criterion", 0.0)
    )
    required_fields["pose_b1_history_healthy"] = serialize_metric_value(
        pose_metrics.get("b1_history_healthy", 0.0)
    )
    required_fields["pose_b1_no_improve_streak"] = serialize_metric_value(
        pose_metrics.get("b1_no_improve_streak", atlas_runtime_metrics.get("pose_gate_b1_no_improve_streak", 0.0))
    )
    required_fields["pose_b2_photo_corridor_open"] = serialize_metric_value(
        pose_metrics.get("b2_photo_corridor_open", atlas_runtime_metrics.get("pose_gate_b2_photo_corridor_open", 0.0))
    )
    required_fields["pose_b2_photo_corridor_scene_signal"] = serialize_metric_value(
        pose_metrics.get(
            "b2_photo_corridor_scene_signal",
            atlas_runtime_metrics.get("pose_gate_b2_photo_corridor_scene_signal", 0.0),
        )
    )
    required_fields["pose_b2_photo_corridor_support_ready"] = serialize_metric_value(
        pose_metrics.get(
            "b2_photo_corridor_support_ready",
            atlas_runtime_metrics.get("pose_gate_b2_photo_corridor_support_ready", 0.0),
        )
    )
    required_fields["pose_b2_corridor_step_ok"] = serialize_metric_value(
        pose_metrics.get("b2_corridor_step_ok", 0.0)
    )
    required_fields["pose_b2_trust_max_scale_effective"] = serialize_metric_value(
        pose_metrics.get("b2_trust_max_scale_effective", 0.0)
    )
    for metric_name, default_value in (
        ("pose_b2_photo_loss_raw", pose_metrics.get("b2_photo_loss_raw", pose_metrics.get("b2_pose_photo_loss", 0.0))),
        ("pose_b2_photo_loss_weighted", pose_metrics.get("b2_photo_loss_weighted", pose_metrics.get("b2_data_loss", 0.0))),
        ("pose_b2_trust_loss", pose_metrics.get("b2_trust_loss", 0.0)),
        ("pose_b2_combined_loss", pose_metrics.get("b2_combined_loss", pose_metrics.get("b2_total_loss", 0.0))),
        ("pose_b2_mode", pose_metrics.get("b2_mode", "none")),
        ("pose_b2_fullframe_stress_enabled", pose_metrics.get("b2_fullframe_stress_enabled", 0.0)),
        ("pose_fullframe_l1", pose_metrics.get("b2_pose_fullframe_l1", 0.0)),
        ("pose_fullframe_ssim", pose_metrics.get("b2_pose_fullframe_ssim", 0.0)),
        ("pose_fullframe_gradient", pose_metrics.get("b2_pose_fullframe_gradient", 0.0)),
        ("pose_fullframe_total", pose_metrics.get("b2_pose_fullframe_total", 0.0)),
        ("pose_fullframe_num_pixels", pose_metrics.get("b2_pose_fullframe_num_pixels", 0.0)),
        ("pose_fullframe_downsample_factor", pose_metrics.get("b2_pose_fullframe_downsample_factor", 0.0)),
        ("pose_b2_patch_count_used", pose_metrics.get("b2_patch_count_used", pose_metrics.get("b2_pose_patch_count_used", 0.0))),
        ("pose_b2_mask_mean", pose_metrics.get("b2_mask_mean", pose_metrics.get("b2_pose_mask_mean", 0.0))),
        ("pose_b2_mask_nonzero_ratio", pose_metrics.get("b2_pose_mask_nonzero_ratio", pose_metrics.get("b2_mask_nonzero_ratio", 0.0))),
        ("pose_b2_photo_signal_strength", pose_metrics.get("b2_photo_signal_strength", pose_metrics.get("b2_pose_photo_signal_strength", 0.0))),
        ("pose_b2_patch_grad_observable_ratio", pose_metrics.get("b2_pose_patch_grad_observable_ratio", pose_metrics.get("b2_patch_grad_observable_ratio", 0.0))),
        ("pose_b2_passive_safe_sample_count", pose_metrics.get("b2_pose_passive_safe_sample_count", 0.0)),
        ("pose_b2_passive_safe_sample_fraction", pose_metrics.get("b2_pose_passive_safe_sample_fraction", 0.0)),
        ("pose_b2_passive_safe_candidate_count", pose_metrics.get("b2_pose_passive_safe_candidate_count", 0.0)),
        ("pose_b2_passive_safe_trust_mean", pose_metrics.get("b2_pose_passive_safe_trust_mean", 0.0)),
        ("pose_b2_passive_safe_reliability_mean", pose_metrics.get("b2_pose_passive_safe_reliability_mean", 0.0)),
        ("pose_b2_observable_patch_ok", pose_metrics.get("b2_observable_patch_ok", 0.0)),
        ("pose_b2_grad_nonzero", pose_metrics.get("b2_grad_nonzero", 0.0)),
        ("pose_b2_any_grad_nonzero", pose_metrics.get("b2_any_grad_nonzero", 0.0)),
        ("pose_b2_data_grad_nonzero", pose_metrics.get("b2_data_grad_nonzero", 0.0)),
        ("pose_b2_post_trust_grad_nonzero", pose_metrics.get("b2_post_trust_grad_nonzero", 0.0)),
        ("pose_b2_grad_total_ok", pose_metrics.get("b2_grad_total_ok", 0.0)),
        ("pose_b2_grad_axis_ok", pose_metrics.get("b2_grad_axis_ok", 0.0)),
        ("pose_b2_pose_q_grad_is_none", pose_metrics.get("b2_pose_q_grad_is_none", 1.0)),
        ("pose_b2_pose_t_grad_is_none", pose_metrics.get("b2_pose_t_grad_is_none", 1.0)),
        ("pose_b2_pose_q_grad_norm", pose_metrics.get("b2_pose_q_grad_norm", 0.0)),
        ("pose_b2_pose_t_grad_norm", pose_metrics.get("b2_pose_t_grad_norm", 0.0)),
        ("pose_b2_pose_grad_norm_total", pose_metrics.get("b2_pose_grad_norm_total", 0.0)),
        ("pose_b2_pose_graph_connected", pose_metrics.get("b2_pose_graph_connected", 0.0)),
        ("pose_b2_loss_depends_on_pose_path", pose_metrics.get("b2_loss_depends_on_pose_path", 0.0)),
        ("pose_b2_pose_graph_connected_but_tiny", pose_metrics.get("b2_pose_graph_connected_but_tiny", 0.0)),
        ("pose_b2_trust_choked_but_pose_grad_exists", pose_metrics.get("b2_trust_choked_but_pose_grad_exists", 0.0)),
        ("pose_b2_data_grad_from_autograd_audit", pose_metrics.get("b2_data_grad_from_autograd_audit", 0.0)),
        ("pose_b2_pose_autograd_error", pose_metrics.get("b2_pose_autograd_error", "none")),
        ("pose_b2_pose_delta_q_requires_grad", pose_metrics.get("b2_pose_delta_q_requires_grad", 0.0)),
        ("pose_b2_pose_delta_t_requires_grad", pose_metrics.get("b2_pose_delta_t_requires_grad", 0.0)),
        ("pose_b2_world_view_transform_requires_grad", pose_metrics.get("b2_world_view_transform_requires_grad", 0.0)),
        ("pose_b2_full_proj_transform_requires_grad", pose_metrics.get("b2_full_proj_transform_requires_grad", 0.0)),
        ("pose_b2_camera_center_requires_grad", pose_metrics.get("b2_camera_center_requires_grad", 0.0)),
        ("pose_b2_pre_trust_grad_norm", pose_metrics.get("b2_pre_trust_grad_norm", 0.0)),
        ("pose_b2_post_trust_grad_norm", pose_metrics.get("b2_post_trust_grad_norm", 0.0)),
        ("pose_b2_pre_to_post_grad_shrink_ratio", pose_metrics.get("b2_pre_to_post_grad_shrink_ratio", 0.0)),
        ("pose_b2_pre_to_post_trans_grad_shrink_ratio", pose_metrics.get("b2_pre_to_post_trans_grad_shrink_ratio", 0.0)),
        ("pose_b2_pre_to_post_rot_grad_shrink_ratio", pose_metrics.get("b2_pre_to_post_rot_grad_shrink_ratio", 0.0)),
        ("pose_b2_data_only_q_grad_norm", pose_metrics.get("b2_data_only_q_grad_norm", 0.0)),
        ("pose_b2_data_only_t_grad_norm", pose_metrics.get("b2_data_only_t_grad_norm", 0.0)),
        ("pose_b2_data_only_grad_total", pose_metrics.get("b2_data_only_grad_total", 0.0)),
        ("pose_b2_data_only_vs_total_grad_ratio", pose_metrics.get("b2_data_only_vs_total_grad_ratio", 0.0)),
        ("pose_b2_fd_probe_enabled", pose_metrics.get("b2_fd_probe_enabled", 0.0)),
        ("pose_b2_fd_probe_status", pose_metrics.get("b2_fd_probe_status", "disabled")),
        ("pose_b2_fd_probe_count", pose_metrics.get("b2_fd_probe_count", 0.0)),
        ("pose_b2_fd_trans_sensitivity_x", pose_metrics.get("b2_fd_trans_sensitivity_x", 0.0)),
        ("pose_b2_fd_trans_sensitivity_y", pose_metrics.get("b2_fd_trans_sensitivity_y", 0.0)),
        ("pose_b2_fd_trans_sensitivity_z", pose_metrics.get("b2_fd_trans_sensitivity_z", 0.0)),
        ("pose_b2_fd_rot_sensitivity_x", pose_metrics.get("b2_fd_rot_sensitivity_x", 0.0)),
        ("pose_b2_fd_rot_sensitivity_y", pose_metrics.get("b2_fd_rot_sensitivity_y", 0.0)),
        ("pose_b2_fd_rot_sensitivity_z", pose_metrics.get("b2_fd_rot_sensitivity_z", 0.0)),
        ("pose_b2_fd_trans_sensitivity_max", pose_metrics.get("b2_fd_trans_sensitivity_max", 0.0)),
        ("pose_b2_fd_rot_sensitivity_max", pose_metrics.get("b2_fd_rot_sensitivity_max", 0.0)),
        ("pose_b2_fd_any_positive", pose_metrics.get("b2_fd_any_positive", 0.0)),
        ("pose_b2_fd_flat", pose_metrics.get("b2_fd_flat", 0.0)),
        ("pose_b2_fd_nonflat_autograd_tiny", pose_metrics.get("b2_fd_nonflat_autograd_tiny", 0.0)),
        ("pose_b2_fd_nonflat_trust_choked", pose_metrics.get("b2_fd_nonflat_trust_choked", 0.0)),
        ("pose_b2_photo_grad_norm_rot", pose_metrics.get("b2_photo_grad_norm_rot", 0.0)),
        ("pose_b2_photo_grad_norm_trans", pose_metrics.get("b2_photo_grad_norm_trans", 0.0)),
        ("pose_b2_step_by_total_grad", pose_metrics.get("b2_step_by_total_grad", 0.0)),
        ("pose_b2_step_by_data_grad", pose_metrics.get("b2_step_by_data_grad", 0.0)),
        ("pose_b2_step_allowed", pose_metrics.get("b2_step_allowed", 0.0)),
        ("pose_b2_skip_reason_detailed", pose_metrics.get("b2_skip_reason_detailed", pose_metrics.get("b2_skip_reason", "none"))),
        ("pose_b2_small_step_mode", pose_metrics.get("b2_small_step_mode", 0.0)),
        ("pose_b2_small_valid_step_ok", pose_metrics.get("b2_small_valid_step_ok", 0.0)),
        ("pose_b2_small_step_lr_scale", pose_metrics.get("b2_step_lr_scale", 0.0)),
        ("pose_b2_use_data_grad_for_step", pose_metrics.get("b2_use_data_grad_for_step", 0.0)),
        ("pose_b2_microstep_allowed", pose_metrics.get("b2_microstep_allowed", 0.0)),
        ("pose_b2_microstep_mode", pose_metrics.get("b2_microstep_mode", 0.0)),
        ("pose_b2_microstep_translation_applied", pose_metrics.get("b2_microstep_translation_applied", 0.0)),
        ("pose_b2_microstep_rotation_applied_deg", pose_metrics.get("b2_microstep_rotation_applied_deg", 0.0)),
        ("pose_b2_microstep_reason", pose_metrics.get("b2_microstep_reason", "inactive")),
    ):
        required_fields[metric_name] = serialize_metric_value(default_value)
    required_fields["pose_freeze_recovery_good_streak"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_freeze_recovery_good_streak", 0.0)
    )
    required_fields["runtime_pose_gate_enabled"] = serialize_metric_value(
        atlas_runtime_metrics.get("pose_gate_enabled", pose_metrics.get("pose_gate_enabled", 0.0))
    )
    required_fields["runtime_pose_gate_open"] = serialize_metric_value(
        atlas_runtime_metrics.get(
            "pose_gate_b1_enabled",
            atlas_runtime_metrics.get("pose_gate_enabled", pose_metrics.get("b1_gate_open", 0.0)),
        )
    )
    required_fields["pose_trust_clamp_count"] = serialize_metric_value(
        float(pose_metrics.get("pose_translation_clamped", 0.0))
        + float(pose_metrics.get("pose_rotation_clamped", 0.0))
        + float(pose_metrics.get("b1_pose_translation_clamped", 0.0))
        + float(pose_metrics.get("b1_pose_rotation_clamped", 0.0))
        + float(pose_metrics.get("b2_pose_translation_clamped", 0.0))
        + float(pose_metrics.get("b2_pose_rotation_clamped", 0.0))
    )
    required_fields["sigma_parallel_clamp_hits"] = serialize_metric_value(atlas_uncertainty_metrics.get("sigma_parallel_clamp_hits", 0.0))
    required_fields["sigma_support_clamp_hits"] = serialize_metric_value(atlas_uncertainty_metrics.get("sigma_support_clamp_hits", 0.0))
    required_fields["sigma_ray_clamp_hits"] = serialize_metric_value(atlas_uncertainty_metrics.get("sigma_ray_clamp_hits", 0.0))
    required_fields["sigma_ray_floor_hits"] = serialize_metric_value(atlas_uncertainty_metrics.get("sigma_ray_floor_hits", 0.0))
    required_fields["sigma_active_ray_valid_count"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_active_ray_valid_count", 0.0)
    )
    required_fields["sigma_active_ray_unresolved_count"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_active_ray_unresolved_count", 0.0)
    )
    required_fields["sigma_parallel_clamp_hits_stable"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_parallel_clamp_hits_stable", 0.0)
    )
    required_fields["sigma_parallel_clamp_hits_passive"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_parallel_clamp_hits_passive", 0.0)
    )
    required_fields["sigma_parallel_clamp_hits_active"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_parallel_clamp_hits_active", 0.0)
    )
    required_fields["sigma_support_clamp_hits_stable"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_support_clamp_hits_stable", 0.0)
    )
    required_fields["sigma_support_clamp_hits_passive"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_support_clamp_hits_passive", 0.0)
    )
    required_fields["sigma_support_clamp_hits_active"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_support_clamp_hits_active", 0.0)
    )
    required_fields["sigma_stable_parallel_mean"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_stable_parallel_mean", 0.0)
    )
    required_fields["sigma_stable_support_mean"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_stable_support_mean", 0.0)
    )
    required_fields["sigma_passive_parallel_mean"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_passive_parallel_mean", 0.0)
    )
    required_fields["sigma_passive_support_mean"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_passive_support_mean", 0.0)
    )
    required_fields["sigma_active_parallel_mean"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_active_parallel_mean", 0.0)
    )
    required_fields["sigma_active_support_mean"] = serialize_metric_value(
        atlas_uncertainty_metrics.get("sigma_active_support_mean", 0.0)
    )
    for metric_name in (
        "sigma_active_parallel_p50",
        "sigma_active_parallel_p90",
        "sigma_active_support_p50",
        "sigma_active_support_p90",
        "sigma_active_ray_span_mean",
        "sigma_active_ray_floor_mean",
        "sigma_active_ray_cap_mean",
        "sigma_active_ray_parallel_mean",
        "sigma_active_ray_parallel_p90",
    ):
        required_fields[metric_name] = serialize_metric_value(atlas_uncertainty_metrics.get(metric_name, 0.0))
    required_fields["atlas_rank_u_mean"] = serialize_metric_value(atlas_kl_metrics.get("atlas_rank_u_mean", 0.0))
    required_fields["atlas_active_ray_count"] = serialize_metric_value(
        atlas_kl_metrics.get("atlas_active_ray_count", 0.0)
    )
    required_fields["atlas_active_ray_fraction"] = serialize_metric_value(
        atlas_kl_metrics.get("atlas_active_ray_fraction", 0.0)
    )
    required_fields["atlas_active_ray_valid_fraction"] = serialize_metric_value(
        atlas_kl_metrics.get("atlas_active_ray_valid_fraction", 0.0)
    )
    required_fields["atlas_active_ray_fallback_count"] = serialize_metric_value(
        atlas_kl_metrics.get("atlas_active_ray_fallback_count", 0.0)
    )
    required_fields["atlas_active_ray_fallback_fraction"] = serialize_metric_value(
        atlas_kl_metrics.get("atlas_active_ray_fallback_fraction", 0.0)
    )
    required_fields["atlas_kl_stable_mean"] = serialize_metric_value(atlas_kl_metrics.get("atlas_kl_stable_mean", 0.0))
    required_fields["atlas_kl_passive_mean"] = serialize_metric_value(atlas_kl_metrics.get("atlas_kl_passive_mean", 0.0))
    required_fields["atlas_kl_active_mean"] = serialize_metric_value(atlas_kl_metrics.get("atlas_kl_active_mean", 0.0))
    required_fields["atlas_slab_total_loss"] = serialize_metric_value(atlas_slab_metrics.get("atlas_slab_total_loss", 0.0))
    required_fields["atlas_slab_active_fraction"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_active_fraction", 0.0)
    )
    required_fields["atlas_slab_active_count"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_active_count", 0.0)
    )
    required_fields["atlas_slab_valid_count"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_valid_count", 0.0)
    )
    required_fields["atlas_slab_mean_penalty"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_mean_penalty", 0.0)
    )
    required_fields["atlas_slab_violation_count"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_violation_count", 0.0)
    )
    required_fields["atlas_slab_violation_ratio"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_violation_ratio", 0.0)
    )
    required_fields["atlas_slab_fallback_count"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_fallback_count", 0.0)
    )
    required_fields["atlas_slab_ref_repair_count"] = serialize_metric_value(
        atlas_slab_metrics.get("atlas_slab_ref_repair_count", 0.0)
    )

    if has_atlas_bindings:
        atlas_reliability_summary = atlas_reliability_summary or {}
        atlas_state_metrics = atlas_state_metrics or {}
        required_fields["atlas_reliability_base_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_reliability_base_mean", 0.0)
        )
        required_fields["atlas_reliability_runtime_raw_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_reliability_runtime_raw_mean", 0.0)
        )
        required_fields["atlas_reliability_runtime_mapped_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_reliability_runtime_mapped_mean", 0.0)
        )
        required_fields["atlas_reliability_effective_mean"] = serialize_metric_value(
            atlas_reliability_summary.get(
                "atlas_reliability_effective_mean",
                atlas_reliability_summary.get("atlas_reliability_runtime_mean", 0.0),
            )
        )
        required_fields["atlas_reliability_runtime_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_reliability_runtime_mean", 0.0)
        )
        required_fields["atlas_reliability_runtime_min"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_reliability_runtime_min", 0.0)
        )
        required_fields["atlas_reliability_runtime_max"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_reliability_runtime_max", 0.0)
        )
        for metric_name in (
            "atlas_reliability_base_p10",
            "atlas_reliability_base_p50",
            "atlas_reliability_base_p90",
            "atlas_reliability_base_hist_low",
            "atlas_reliability_base_hist_mid",
            "atlas_reliability_base_hist_high",
            "atlas_reliability_runtime_mapped_p10",
            "atlas_reliability_runtime_mapped_p50",
            "atlas_reliability_runtime_mapped_p90",
            "atlas_reliability_runtime_mapped_hist_low",
            "atlas_reliability_runtime_mapped_hist_mid",
            "atlas_reliability_runtime_mapped_hist_high",
            "atlas_reliability_effective_p10",
            "atlas_reliability_effective_p50",
            "atlas_reliability_effective_p90",
            "atlas_reliability_effective_hist_low",
            "atlas_reliability_effective_hist_mid",
            "atlas_reliability_effective_hist_high",
            "refresh_evidence_observed_gate_ratio",
            "refresh_evidence_count_gate_ratio",
            "refresh_evidence_visibility_gate_ratio",
            "refresh_evidence_ref_gate_ratio",
            "refresh_evidence_finite_gate_ratio",
            "refresh_evidence_support_gate_ratio",
            "refresh_base_runtime_override_gate_ratio",
            "refresh_strong_runtime_evidence_ratio",
            "refresh_evidence_override_gate_ratio",
            "refresh_evidence_gate_mean",
            "refresh_override_weight_positive_ratio",
            "refresh_override_base_bucket_low_override_ratio",
            "refresh_override_base_bucket_mid_override_ratio",
            "refresh_override_base_bucket_high_override_ratio",
        ):
            required_fields[metric_name] = serialize_metric_value(atlas_reliability_summary.get(metric_name, 0.0))
        required_fields["atlas_refresh_snapshot_ready"] = int(
            bool(atlas_reliability_summary.get("atlas_refresh_snapshot_ready", 0.0))
        )
        required_fields["atlas_refresh_snapshot_observed_ratio"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_refresh_snapshot_observed_ratio", 0.0)
        )
        required_fields["atlas_refresh_snapshot_observed_count"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_refresh_snapshot_observed_count", 0.0)
        )
        required_fields["atlas_refresh_snapshot_photo_ema_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_refresh_snapshot_photo_ema_mean", 0.0)
        )
        required_fields["atlas_refresh_snapshot_visibility_ema_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_refresh_snapshot_visibility_ema_mean", 0.0)
        )
        required_fields["atlas_refresh_snapshot_obs_quality_mean"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_refresh_snapshot_obs_quality_mean", 0.0)
        )
        required_fields["atlas_refresh_snapshot_obs_quality_max"] = serialize_metric_value(
            atlas_reliability_summary.get("atlas_refresh_snapshot_obs_quality_max", 0.0)
        )
        required_fields["atlas_refresh_done"] = int(
            bool(atlas_reliability_summary.get("atlas_refresh_done", atlas_refresh_done))
        )
        required_fields["state_stable_count"] = int(atlas_state_metrics.get("state_stable_count", 0))
        required_fields["state_passive_count"] = int(atlas_state_metrics.get("state_passive_count", 0))
        required_fields["state_active_count"] = int(atlas_state_metrics.get("state_active_count", 0))
        required_fields["state_out_pending_count"] = int(atlas_state_metrics.get("state_out_pending_count", 0))
        required_fields["stable_ratio"] = serialize_metric_value(atlas_state_metrics.get("stable_ratio", 0.0))
        required_fields["passive_ratio"] = serialize_metric_value(atlas_state_metrics.get("passive_ratio", 0.0))
        required_fields["active_ratio"] = serialize_metric_value(atlas_state_metrics.get("active_ratio", 0.0))
        required_fields["out_of_anchor_ratio"] = serialize_metric_value(atlas_state_metrics.get("out_of_anchor_ratio", 0.0))
        required_fields["pending_ratio"] = serialize_metric_value(
            atlas_state_metrics.get("pending_ratio", atlas_state_metrics.get("out_of_anchor_ratio", 0.0))
        )
        required_fields["out_of_anchor_pending_count"] = int(
            atlas_state_metrics.get("out_of_anchor_pending_count", atlas_state_metrics.get("state_out_pending_count", 0))
        )
        required_fields["cooldown_ratio"] = serialize_metric_value(
            atlas_state_metrics.get("cooldown_ratio", atlas_state_metrics.get("state_cooldown_ratio", 0.0))
        )
        required_fields["mean_gc_fail_count"] = serialize_metric_value(
            atlas_state_metrics.get("mean_gc_fail_count", atlas_runtime_metrics.get("mean_gc_fail_count", 0.0))
        )
        required_fields["mean_active_lifetime"] = serialize_metric_value(atlas_state_metrics.get("mean_active_lifetime", 0.0))
        required_fields["max_active_lifetime"] = serialize_metric_value(atlas_state_metrics.get("max_active_lifetime", 0.0))
        required_fields["active_candidate_pool_count"] = int(atlas_state_metrics.get("active_candidate_pool_count", 0))
        required_fields["active_candidate_pool_ratio"] = serialize_metric_value(
            atlas_state_metrics.get("active_candidate_pool_ratio", 0.0)
        )
        required_fields["candidate_formation_count"] = int(atlas_state_metrics.get("candidate_formation_count", 0))
        required_fields["active_formed_count"] = int(
            atlas_state_metrics.get("active_formed_count", atlas_state_metrics.get("candidate_formation_count", 0))
        )
        for metric_name in (
            "active_standard_formation_count",
            "active_rescue_fallback_formation_count",
            "active_standard_candidate_pool_count",
            "active_rescue_candidate_pool_count",
        ):
            required_fields[metric_name] = int(atlas_state_metrics.get(metric_name, 0))
        required_fields["active_admitted_count"] = int(
            atlas_state_metrics.get("active_admitted_count", atlas_state_metrics.get("active_admission_pool_count", 0))
        )
        required_fields["active_promoted_count"] = int(
            atlas_state_metrics.get("active_promoted_count", atlas_state_metrics.get("active_promote_count", 0))
        )
        required_fields["active_admission_pool_count"] = int(atlas_state_metrics.get("active_admission_pool_count", 0))
        required_fields["active_standard_admission_pool_count"] = int(
            atlas_state_metrics.get("active_standard_admission_pool_count", 0)
        )
        required_fields["active_rescue_admission_pool_count"] = int(
            atlas_state_metrics.get("active_rescue_admission_pool_count", 0)
        )
        required_fields["active_promote_count"] = int(atlas_state_metrics.get("active_promote_count", 0))
        required_fields["active_standard_promote_count"] = int(atlas_state_metrics.get("active_standard_promote_count", 0))
        required_fields["active_forced_rescue_promote_count"] = int(
            atlas_state_metrics.get("active_forced_rescue_promote_count", atlas_state_metrics.get("active_rescue_promote_count", 0))
        )
        required_fields["active_new_active_count"] = int(atlas_state_metrics.get("active_new_active_count", 0))
        required_fields["active_rescue_candidate_count"] = int(atlas_state_metrics.get("active_rescue_candidate_count", 0))
        required_fields["active_rescue_promote_count"] = int(atlas_state_metrics.get("active_rescue_promote_count", 0))
        required_fields["active_quota_target"] = int(atlas_state_metrics.get("active_quota_target", 0))
        required_fields["active_quota_soft_target"] = int(atlas_state_metrics.get("active_quota_soft_target", 0))
        required_fields["active_quota_hard_target"] = int(atlas_state_metrics.get("active_quota_hard_target", 0))
        required_fields["active_quota_effective_live_active_count"] = int(
            atlas_state_metrics.get("active_quota_effective_live_active_count", atlas_state_metrics.get("state_active_count", 0))
        )
        required_fields["active_quota_over_target_count"] = int(
            atlas_state_metrics.get(
                "active_quota_over_target_count",
                max(
                    int(required_fields["active_quota_effective_live_active_count"])
                    - int(required_fields["active_quota_target"]),
                    0,
                ),
            )
        )
        required_fields["active_quota_available"] = int(atlas_state_metrics.get("active_quota_available", 0))
        required_fields["active_quota_current_count"] = int(atlas_state_metrics.get("active_quota_current_count", 0))
        required_fields["active_quota_before_release_count"] = int(
            atlas_state_metrics.get("active_quota_before_release_count", atlas_state_metrics.get("active_quota_current_count", 0))
        )
        required_fields["active_quota_release_count"] = int(
            atlas_state_metrics.get("active_quota_release_count", atlas_state_metrics.get("active_quota_projected_exit_count", 0))
        )
        required_fields["active_quota_projected_exit_count"] = int(atlas_state_metrics.get("active_quota_projected_exit_count", 0))
        required_fields["active_quota_projected_after_exit"] = int(atlas_state_metrics.get("active_quota_projected_after_exit", 0))
        required_fields["active_quota_after_release_count"] = int(
            atlas_state_metrics.get("active_quota_after_release_count", atlas_state_metrics.get("active_quota_projected_after_exit", 0))
        )
        required_fields["active_quota_transition_overflow"] = int(
            atlas_state_metrics.get("active_quota_transition_overflow", 0)
        )
        required_fields["explore_candidate_ratio"] = serialize_metric_value(
            atlas_state_metrics.get("explore_candidate_ratio", 0.0)
        )
        for metric_name in (
            "active_hard_exit_count",
            "active_soft_exit_count",
            "active_lifetime_cap_exit_count",
            "active_nonimproving_exit_count",
            "active_stale_fallback_exit_count",
            "active_fallback_handoff_exit_count",
            "active_lifetime_release_count",
            "active_demoted_count",
            "active_exited_count",
            "active_provenance_preserved_count",
            "passive_to_stable_candidate_count",
            "active_to_stable_candidate_count",
            "passive_stable_ready_count",
            "active_stable_ready_count",
            "passive_stable_cooldown_bypass_count",
            "state_reliability_stable_ready_count",
            "runtime_stable_support_count",
            "runtime_recovery_support_count",
            "recovery_score_candidate_count",
            "recovery_candidate_count",
            "recovery_ready_soft_count",
            "recovery_promote_hard_count",
            "recovery_required_streak",
            "recovery_streak_max",
            "recovery_streak_ready_count",
            "recovery_block_low_reliability",
            "recovery_block_low_support_consistency",
            "recovery_block_low_ref_consistency",
            "recovery_block_high_photo_ema",
            "recovery_block_high_drift",
            "recovery_block_cooldown",
            "recovery_block_persistent_out",
            "effective_recovery_stable_ready_count",
            "passive_runtime_recovery_ready_count",
            "passive_explicit_stable_ready_count",
            "active_carryover_count",
            "active_state_rebuild_count",
            "active_provenance_from_transition_passive_to_active_count",
            "active_provenance_from_restore_checkpoint_count",
            "active_provenance_from_state_rebuild_after_gc_count",
            "active_provenance_from_quota_carryover_count",
            "active_provenance_from_forced_rescue_bootstrap_count",
            "active_provenance_from_active_explore_clone_count",
            "active_provenance_tracked_count",
            "active_provenance_untracked_count",
            "active_to_stable_exit_count",
            "active_to_passive_exit_count",
        ):
            required_fields[metric_name] = int(atlas_state_metrics.get(metric_name, 0))
        for metric_name in (
            "runtime_stable_support_ratio",
            "runtime_recovery_support_ratio",
            "recovery_score_mean",
            "recovery_score_max",
            "recovery_score_threshold",
            "recovery_photo_reference",
            "recovery_photo_soft_max",
            "recovery_photo_hard_max",
            "recovery_photo_median",
            "recovery_photo_q75",
            "active_provenance_from_transition_passive_to_active_ratio",
            "active_provenance_from_restore_checkpoint_ratio",
            "active_provenance_from_state_rebuild_after_gc_ratio",
            "active_provenance_from_quota_carryover_ratio",
            "active_provenance_from_forced_rescue_bootstrap_ratio",
            "active_provenance_from_active_explore_clone_ratio",
            "active_provenance_tracked_ratio",
            "active_provenance_untracked_ratio",
        ):
            required_fields[metric_name] = serialize_metric_value(atlas_state_metrics.get(metric_name, 0.0))
        required_fields["active_max_lifetime_iters"] = int(atlas_state_metrics.get("active_max_lifetime_iters", 0))
        required_fields["active_nonimprove_iters"] = int(atlas_state_metrics.get("active_nonimprove_iters", 0))
        required_fields["transition_stable_to_passive_count"] = int(
            atlas_state_metrics.get("transition_stable_to_passive_count", 0)
        )
        required_fields["transition_passive_to_stable_count"] = int(
            atlas_state_metrics.get("transition_passive_to_stable_count", 0)
        )
        required_fields["transition_passive_to_active_count"] = int(
            atlas_state_metrics.get("transition_passive_to_active_count", 0)
        )
        required_fields["transition_passive_to_active_standard_count"] = int(
            atlas_state_metrics.get("transition_passive_to_active_standard_count", 0)
        )
        required_fields["transition_passive_to_active_rescue_count"] = int(
            atlas_state_metrics.get("transition_passive_to_active_rescue_count", 0)
        )
        required_fields["transition_passive_to_active_unclassified_count"] = int(
            atlas_state_metrics.get("transition_passive_to_active_unclassified_count", 0)
        )
        required_fields["transition_active_to_passive_count"] = int(
            atlas_state_metrics.get("transition_active_to_passive_count", 0)
        )
        required_fields["transition_active_to_stable_count"] = int(
            atlas_state_metrics.get("transition_active_to_stable_count", 0)
        )
        required_fields["transition_any_to_pending_count"] = int(
            atlas_state_metrics.get("transition_any_to_pending_count", 0)
        )
        required_fields["observed_node_ratio"] = serialize_metric_value(atlas_runtime_metrics.get("observed_node_ratio", 0.0))
        required_fields["observed_node_count"] = serialize_metric_value(atlas_runtime_metrics.get("observed_node_count", 0.0))
        required_fields["updated_node_ratio"] = serialize_metric_value(atlas_runtime_metrics.get("updated_node_ratio", 0.0))
        required_fields["updated_node_count"] = serialize_metric_value(atlas_runtime_metrics.get("updated_node_count", 0.0))
        required_fields["coverage_node_update_count"] = serialize_metric_value(atlas_runtime_metrics.get("coverage_node_update_count", 0.0))
        required_fields["mean_node_update_strength"] = serialize_metric_value(atlas_runtime_metrics.get("mean_node_update_strength", 0.0))
        required_fields["mean_node_photo_ema"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_photo_ema", 0.0)
        )
        required_fields["mean_node_visibility_ema"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_visibility_ema", 0.0)
        )
        required_fields["mean_node_obs_quality_ema"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_obs_quality_ema", atlas_runtime_metrics.get("mean_node_obs_quality", 0.0))
        )
        required_fields["mean_node_observed_score_current"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_observed_score_current", 0.0)
        )
        required_fields["mean_node_observed_score_ema"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_observed_score_ema", 0.0)
        )
        required_fields["mean_node_updated_recently"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_updated_recently", 0.0)
        )
        required_fields["mean_node_support_consistency_current"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_support_consistency_current", 0.0)
        )
        required_fields["mean_node_finite_projection_ema"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_finite_projection_ema", 0.0)
        )
        required_fields["mean_node_ref_consistency_ema"] = serialize_metric_value(
            atlas_runtime_metrics.get("mean_node_ref_consistency_ema", 0.0)
        )
        for metric_name in (
            "patch_quality_score",
            "mask_nonzero_ratio",
            "bg_like_ratio",
            "background_like_ratio",
            "thin_support_like_ratio",
            "photo_signal_strength",
            "patch_quality_candidate_mean",
            "photo_signal_candidate_mean",
        ):
            required_fields[metric_name] = serialize_metric_value(atlas_runtime_metrics.get(metric_name, 0.0))
        required_fields["refresh_override_count"] = serialize_metric_value(
            atlas_runtime_metrics.get(
                "refresh_override_count",
                atlas_runtime_metrics.get(
                    "runtime_override_count",
                    atlas_reliability_summary.get("atlas_refresh_snapshot_runtime_override_count", 0.0),
                ),
            )
        )
        required_fields["refresh_override_ratio"] = serialize_metric_value(
            atlas_runtime_metrics.get(
                "refresh_override_ratio",
                atlas_runtime_metrics.get(
                    "runtime_override_ratio",
                    atlas_reliability_summary.get("atlas_refresh_snapshot_runtime_override_ratio", 0.0),
                ),
            )
        )
        required_fields["refresh_std_before_after"] = serialize_metric_value(
            atlas_runtime_metrics.get(
                "refresh_std_before_after",
                {
                    "before": atlas_runtime_metrics.get("refresh_std_before", 0.0),
                    "after": atlas_runtime_metrics.get("refresh_std_after", atlas_runtime_metrics.get("runtime_reliability_std", 0.0)),
                },
            )
        )
        required_fields["state_candidate_reason_breakdown"] = serialize_metric_value(
            atlas_state_metrics.get("state_candidate_reason_breakdown", {})
        )
        required_fields["promotion_block_reason_breakdown"] = serialize_metric_value(
            atlas_state_metrics.get("promotion_block_reason_breakdown", {})
        )
        required_fields["floater_proxy_by_state"] = serialize_metric_value(
            atlas_runtime_metrics.get("floater_proxy_by_state", {})
        )
        required_fields["dark_region_completeness_by_state"] = serialize_metric_value(
            atlas_runtime_metrics.get("dark_region_completeness_by_state", {})
        )

    return required_fields


def compute_reconstruction_loss(rendered_image, gt_image, lambda_dssim):
    Ll1 = l1_loss(rendered_image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(rendered_image, gt_image)
    loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
    return loss, Ll1, ssim_value


def estimate_antithetic_render_loss(
    viewpoint_camera,
    gaussians,
    pipe,
    bg,
    gt_image,
    alpha_mask,
    lambda_dssim,
    use_trained_exp,
    separate_sh,
    subspace_info,
    mc_pairs,
    mc_scale,
):
    zero = gt_image.new_zeros(())
    metrics = {
        "atlas_mc_loss": 0.0,
        "atlas_mc_pairs": 0.0,
        "atlas_mc_active_fraction": 0.0,
        "atlas_mc_mean_offset": 0.0,
    }
    if subspace_info is None or int(mc_pairs) <= 0:
        return zero, metrics, None

    total_loss = zero
    grad_render_pkg = None
    active_fraction_total = 0.0
    offset_mean_total = 0.0
    valid_pairs = 0

    for _ in range(int(mc_pairs)):
        offsets, offset_metrics = sample_antithetic_center_offsets(
            gaussians,
            subspace_info,
            sample_scale=float(mc_scale),
        )
        if offset_metrics["atlas_mc_active_fraction"] <= 0.0:
            continue

        pair_loss = zero
        for sign in (1.0, -1.0):
            sample_render_pkg = render(
                viewpoint_camera,
                gaussians,
                pipe,
                bg,
                use_trained_exp=use_trained_exp,
                separate_sh=separate_sh,
                override_means3D=gaussians.get_xyz + (float(sign) * offsets),
            )
            sample_image = sample_render_pkg["render"]
            if alpha_mask is not None:
                sample_image = sample_image * alpha_mask
            sample_loss, _, _ = compute_reconstruction_loss(sample_image, gt_image, lambda_dssim)
            pair_loss = pair_loss + sample_loss
            if grad_render_pkg is None:
                grad_render_pkg = sample_render_pkg
        total_loss = total_loss + 0.5 * pair_loss
        active_fraction_total += offset_metrics["atlas_mc_active_fraction"]
        offset_mean_total += offset_metrics["atlas_mc_mean_offset"]
        valid_pairs += 1

    if valid_pairs == 0:
        return zero, metrics, None

    total_loss = total_loss / float(valid_pairs)
    metrics["atlas_mc_loss"] = float(total_loss.detach().item())
    metrics["atlas_mc_pairs"] = float(valid_pairs)
    metrics["atlas_mc_active_fraction"] = active_fraction_total / float(valid_pairs)
    metrics["atlas_mc_mean_offset"] = offset_mean_total / float(valid_pairs)
    return total_loss, metrics, grad_render_pkg


def _is_finite_scalar(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value.detach()).all().item())
    return math.isfinite(float(value))


def _reset_camera_pose_if_nonfinite(camera):
    if not hasattr(camera, "pose_delta_q") or not hasattr(camera, "pose_delta_t"):
        return False

    pose_q = camera.pose_delta_q.detach()
    pose_t = camera.pose_delta_t.detach()
    if bool(torch.isfinite(pose_q).all().item()) and bool(torch.isfinite(pose_t).all().item()):
        return False

    with torch.no_grad():
        if not bool(torch.isfinite(pose_q).all().item()):
            camera.pose_delta_q.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=camera.pose_delta_q.dtype, device=camera.pose_delta_q.device))
        if not bool(torch.isfinite(pose_t).all().item()):
            camera.pose_delta_t.zero_()
    if hasattr(camera, "refresh_pose_matrices"):
        camera.refresh_pose_matrices()
    return True


def _compute_atlas_phase_controls(iteration: int, opt, has_atlas_bindings: bool):
    warmup_steps = max(int(getattr(opt, "atlas_reg_warmup_steps", 0)), 0)
    in_warmup = bool(has_atlas_bindings and iteration <= warmup_steps)
    main_phase = bool((not has_atlas_bindings) or (iteration > warmup_steps))
    pose_enabled = bool(
        has_atlas_bindings
        and main_phase
        and getattr(opt, "pose_refine_after_warmup", False)
    )
    return {
        "warmup_steps": warmup_steps,
        "in_warmup": in_warmup,
        "warmup_only": in_warmup,
        "main_phase": main_phase,
        "enable_pose_b1": pose_enabled,
        "enable_pose_b2": pose_enabled,
        "enable_densify": bool((not has_atlas_bindings) or main_phase),
        "enable_prune": bool((not has_atlas_bindings) or main_phase),
        "enable_gc": bool(has_atlas_bindings and main_phase and int(getattr(opt, "atlas_gc_interval", 0)) > 0),
        "enable_state_update": bool(has_atlas_bindings and main_phase),
        "enable_mc": bool(has_atlas_bindings and main_phase and int(getattr(opt, "atlas_mc_pairs", 0)) > 0),
        "enable_explore": bool(has_atlas_bindings and main_phase and float(getattr(opt, "atlas_slab_weight", 0.0)) > 0.0),
    }


def _linear_gate(value: float, start: float, end: float):
    value = float(value)
    start = float(start)
    end = float(end)
    if end <= start + 1e-8:
        return 1.0 if value >= start else 0.0
    return max(0.0, min((value - start) / (end - start), 1.0))


def _init_densify_runtime_state():
    return {
        "last_psnr": None,
        "last_l1": None,
        "last_floater_proxy": None,
        "last_dark_region_completeness_proxy": None,
        "psnr_delta": 0.0,
        "l1_delta": 0.0,
        "floater_delta": 0.0,
        "dark_region_completeness_delta": 0.0,
        "psnr_ema": None,
        "l1_ema": None,
        "floater_ema": None,
        "dark_region_completeness_ema": None,
        "validation_count": 0,
    }


def _update_densify_runtime_state(state: dict, validation_summary: dict | None):
    if not validation_summary:
        return state
    train_metrics = validation_summary.get("train") if isinstance(validation_summary, dict) else None
    if not train_metrics:
        return state

    def _finite_or_none(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    psnr_value = _finite_or_none(train_metrics.get("psnr"))
    l1_value = _finite_or_none(train_metrics.get("l1"))
    floater_value = _finite_or_none(train_metrics.get("floater_proxy"))
    dark_completeness_value = _finite_or_none(train_metrics.get("dark_region_completeness_proxy"))
    alpha = 0.35
    for name, value in (
        ("psnr", psnr_value),
        ("l1", l1_value),
        ("floater_proxy", floater_value),
        ("dark_region_completeness_proxy", dark_completeness_value),
    ):
        if value is None:
            continue
        last_key = f"last_{name}"
        if name == "floater_proxy":
            ema_key = "floater_ema"
            delta_key = "floater_delta"
        elif name == "dark_region_completeness_proxy":
            ema_key = "dark_region_completeness_ema"
            delta_key = "dark_region_completeness_delta"
        else:
            ema_key = f"{name}_ema"
            delta_key = f"{name}_delta"
        previous = state.get(last_key)
        state[delta_key] = 0.0 if previous is None else float(value - float(previous))
        ema_previous = state.get(ema_key)
        state[ema_key] = float(value) if ema_previous is None else float((1.0 - alpha) * float(ema_previous) + alpha * value)
        state[last_key] = float(value)
    state["validation_count"] = int(state.get("validation_count", 0)) + 1
    return state


def _compute_densify_runtime_controls(
    iteration: int,
    opt,
    gaussians,
    pose_runtime_state: dict,
    densify_runtime_state: dict,
    atlas_runtime_metrics: dict | None = None,
    atlas_state_metrics: dict | None = None,
):
    atlas_runtime_metrics = atlas_runtime_metrics or {}
    atlas_state_metrics = atlas_state_metrics or {}
    warmup_steps = max(int(getattr(opt, "atlas_reg_warmup_steps", 0)), 0)
    ramp_iters = max(int(getattr(opt, "atlas_densify_ramp_iters", 2500)), 1)
    post_warmup = max(int(iteration) - warmup_steps, 0)
    phase_ramp = 0.35 + 0.65 * _linear_gate(post_warmup, 0.0, float(ramp_iters))

    b2_attempts = int(max(pose_runtime_state.get("b2_camera_attempt_count", 0), 0))
    b2_steps = int(max(pose_runtime_state.get("b2_camera_optimizer_step_count", 0), 0))
    b2_skip_hist = pose_runtime_state.get("b2_skip_hist", {}) or {}
    b2_zero_grad_skip_total = int(
        sum(int(v) for k, v in b2_skip_hist.items() if str(k).startswith("zero_gradient"))
    )
    b2_zero_grad_skip_prev = densify_runtime_state.get("b2_zero_grad_skip_total_last")
    b2_zero_grad_skip_delta = 0 if b2_zero_grad_skip_prev is None else max(
        b2_zero_grad_skip_total - int(b2_zero_grad_skip_prev),
        0,
    )
    densify_runtime_state["b2_zero_grad_skip_total_last"] = int(b2_zero_grad_skip_total)
    if b2_attempts < 5:
        b2_step_health = 1.0
        b2_scale = 1.0
    else:
        b2_step_health = float(b2_steps) / float(max(b2_attempts, 1))
        health_gate = _linear_gate(b2_step_health, 0.05, 0.35)
        unhealthy_scale = float(max(min(getattr(opt, "atlas_densify_b2_unhealthy_scale", 0.55), 1.0), 0.05))
        b2_scale = unhealthy_scale + (1.0 - unhealthy_scale) * health_gate
    b2_unhealthy = bool(
        b2_attempts >= 5
        and (
            b2_steps <= 0
            or b2_step_health < 0.05
            or b2_zero_grad_skip_delta >= int(max(getattr(opt, "atlas_densify_b2_zero_grad_delta_unhealthy", 3), 1))
        )
    )
    recovery_event_count = int(
        max(atlas_state_metrics.get("transition_passive_to_stable_count", 0), 0)
        + max(atlas_state_metrics.get("transition_active_to_stable_count", 0), 0)
        + max(atlas_state_metrics.get("passive_stable_ready_count", 0), 0)
        + max(atlas_state_metrics.get("active_stable_ready_count", 0), 0)
        + max(atlas_state_metrics.get("recovery_promote_hard_count", 0), 0)
    )
    if recovery_event_count > 0:
        densify_runtime_state["atlas_recovery_seen"] = 1
    recovery_seen = bool(int(densify_runtime_state.get("atlas_recovery_seen", 0)) > 0)
    b2_recovery_unhealthy = bool(b2_unhealthy and not recovery_seen)

    floater_delta = float(densify_runtime_state.get("floater_delta", 0.0) or 0.0)
    floater_ema = densify_runtime_state.get("floater_ema")
    floater_ema = 0.0 if floater_ema is None else float(floater_ema)
    floater_guard = 1.0
    if floater_delta > 0.004 or floater_ema > 0.070:
        floater_guard = float(max(min(getattr(opt, "atlas_densify_floater_guard_scale", 0.55), 1.0), 0.05))
    elif floater_delta > 0.0015 or floater_ema > 0.050:
        floater_guard = 0.5 + 0.5 * float(max(min(getattr(opt, "atlas_densify_floater_guard_scale", 0.55), 1.0), 0.05))

    psnr_delta = float(densify_runtime_state.get("psnr_delta", 0.0) or 0.0)
    l1_delta = float(densify_runtime_state.get("l1_delta", 0.0) or 0.0)
    l1_ema_value = densify_runtime_state.get("l1_ema")
    l1_ema = 0.0 if l1_ema_value is None else float(l1_ema_value)
    quality_guard = 1.0
    if psnr_delta < -0.05 or l1_delta > 0.002:
        quality_guard = float(max(min(getattr(opt, "atlas_densify_quality_guard_scale", 0.70), 1.0), 0.05))

    init_points = max(int(gaussians.get_init_point_count()), 1)
    total_points = int(gaussians.get_xyz.shape[0])
    capacity_ratio = float(total_points) / float(init_points)
    observed_node_ratio = float(atlas_runtime_metrics.get("observed_node_ratio", 0.0) or 0.0)
    stable_ratio = float(atlas_state_metrics.get("stable_ratio", 0.0) or 0.0)
    reliability_effective_mean = float(
        atlas_runtime_metrics.get(
            "atlas_reliability_effective_mean",
            atlas_runtime_metrics.get("runtime_atlas_reliability_effective_mean", 0.0),
        )
        or 0.0
    )
    dark_completeness_ema = densify_runtime_state.get("dark_region_completeness_ema", None)
    dark_completeness = float(dark_completeness_ema) if dark_completeness_ema is not None else float(
        atlas_runtime_metrics.get("dark_region_completeness_proxy", 0.0) or 0.0
    )
    handoff_enabled = bool(getattr(opt, "atlas_fidelity_handoff_enabled", True))
    observed_gate = _linear_gate(
        observed_node_ratio,
        float(getattr(opt, "atlas_fidelity_handoff_min_observed_ratio", 0.82)) - 0.08,
        float(getattr(opt, "atlas_fidelity_handoff_min_observed_ratio", 0.82)),
    )
    dark_gate = _linear_gate(
        dark_completeness,
        float(getattr(opt, "atlas_fidelity_handoff_min_dark_completeness", 0.94)) - 0.06,
        float(getattr(opt, "atlas_fidelity_handoff_min_dark_completeness", 0.94)),
    )
    stable_gate = _linear_gate(
        stable_ratio,
        float(getattr(opt, "atlas_fidelity_handoff_min_stable_ratio", 0.30)) - 0.08,
        float(getattr(opt, "atlas_fidelity_handoff_min_stable_ratio", 0.30)),
    )
    completion_gate_raw = min(observed_gate, dark_gate, stable_gate) if handoff_enabled else 0.0
    # If dark_completeness and observed_node_ratio are both high but stable_ratio is blocked
    # (e.g. state machine promotion lag), allow a partial handoff so exploration tapers off.
    dark_observed_override = bool(getattr(opt, "atlas_fidelity_handoff_dark_observed_override", True))
    if dark_observed_override and handoff_enabled and stable_gate < 0.10 and completion_gate_raw < 0.10:
        dark_observed_gate = min(dark_gate, observed_gate)
        completion_gate = max(completion_gate_raw, 0.45 * dark_observed_gate)
    else:
        completion_gate = completion_gate_raw
    floater_handoff_gate = _linear_gate(floater_ema, 0.035, 0.075)
    quality_handoff_gate = 1.0 if (psnr_delta < -0.03 or l1_delta > 0.0015 or l1_ema > 0.055) else 0.0
    fidelity_handoff_gate = float(completion_gate * max(floater_handoff_gate, quality_handoff_gate, 0.35 * completion_gate))
    fidelity_mode_metrics = _compute_fidelity_mode(
        dark_completeness=dark_completeness,
        l1_ema=l1_ema,
        floater_ema=floater_ema,
        reliability_effective_mean=reliability_effective_mean,
        completion_gate=completion_gate,
        dark_gate=dark_gate,
        observed_gate=observed_gate,
        stable_gate=stable_gate,
        opt=opt,
        handoff_enabled=handoff_enabled,
    )
    if b2_attempts >= 5:
        pose_fidelity_gate = _linear_gate(b2_step_health, 0.03, 0.18)
    else:
        pose_fidelity_gate = 0.35
    if b2_steps <= 0 and b2_attempts >= 5:
        pose_fidelity_gate = 0.0
    recovery_fidelity_gate = 1.0 if recovery_seen else 0.0
    fidelity_mode_metrics["fidelity_mode_pose_gate"] = float(pose_fidelity_gate)
    fidelity_mode_metrics["fidelity_mode_recovery_gate"] = float(recovery_fidelity_gate)
    fidelity_mode_metrics["fidelity_mode_gate"] = (
        float(fidelity_mode_metrics["fidelity_mode_gate"])
        * float(pose_fidelity_gate)
        * float(recovery_fidelity_gate)
    )
    fidelity_mode_gate = float(fidelity_mode_metrics["fidelity_mode_gate"])
    fidelity_handoff_gate = max(fidelity_handoff_gate, 0.55 * fidelity_mode_gate)
    late_phase_fidelity_boost = bool(getattr(opt, "atlas_late_phase_fidelity_boost", True))
    if late_phase_fidelity_boost and handoff_enabled and completion_gate >= 0.45:
        absolute_quality_gap = max(_linear_gate(l1_ema, 0.045, 0.075), _linear_gate(floater_ema, 0.040, 0.070))
        fidelity_handoff_gate = max(
            fidelity_handoff_gate,
            float(completion_gate * max(0.35, absolute_quality_gap)),
        )
    budget_handoff_scale = 1.0 - fidelity_handoff_gate * (
        1.0 - float(max(min(getattr(opt, "atlas_fidelity_handoff_budget_scale", 0.55), 1.0), 0.05))
    )
    split_handoff_scale = 1.0 - fidelity_handoff_gate * (
        1.0 - float(max(min(getattr(opt, "atlas_fidelity_handoff_split_scale", 0.85), 1.0), 0.05))
    )
    clone_handoff_scale = 1.0 - fidelity_handoff_gate * (
        1.0 - float(max(min(getattr(opt, "atlas_fidelity_handoff_clone_scale", 0.65), 1.0), 0.05))
    )
    explore_handoff_scale = 1.0 - fidelity_handoff_gate * (
        1.0 - float(max(min(getattr(opt, "atlas_fidelity_handoff_explore_scale", 0.20), 1.0), 0.0))
    )

    budget_scale = max(0.05, min(phase_ramp * b2_scale * floater_guard * quality_guard * budget_handoff_scale, 1.0))
    if b2_recovery_unhealthy:
        budget_scale *= float(max(min(getattr(opt, "atlas_densify_b2_recovery_dead_budget_scale", 0.45), 1.0), 0.05))
    split_fraction = float(getattr(opt, "atlas_densify_split_quota_fraction", 0.55)) * split_handoff_scale
    clone_fraction = float(getattr(opt, "atlas_densify_clone_quota_fraction", 0.30)) * clone_handoff_scale
    explore_fraction = float(getattr(opt, "atlas_densify_explore_quota_fraction", 0.15)) * explore_handoff_scale
    b2_unhealthy_split_scale = float(max(min(getattr(opt, "atlas_densify_b2_unhealthy_split_scale", 0.35), 1.0), 0.0))
    b2_unhealthy_explore_scale = float(max(min(getattr(opt, "atlas_densify_b2_unhealthy_explore_scale", 0.10), 1.0), 0.0))
    b2_unhealthy_clone_scale = float(max(min(getattr(opt, "atlas_densify_b2_unhealthy_clone_scale", 1.15), 2.0), 0.0))
    if b2_recovery_unhealthy:
        b2_unhealthy_split_scale = min(b2_unhealthy_split_scale, 0.20)
        b2_unhealthy_explore_scale = min(b2_unhealthy_explore_scale, 0.05)
        b2_unhealthy_clone_scale = max(b2_unhealthy_clone_scale, 1.10)
    return {
        "budget_scale": float(budget_scale),
        "phase_ramp": float(phase_ramp),
        "b2_step_health": float(b2_step_health),
        "b2_attempts": float(b2_attempts),
        "b2_steps": float(b2_steps),
        "b2_zero_grad_skip_total": float(b2_zero_grad_skip_total),
        "b2_zero_grad_skip_delta": float(b2_zero_grad_skip_delta),
        "b2_unhealthy_gate": 1.0 if b2_unhealthy else 0.0,
        "b2_recovery_unhealthy_gate": 1.0 if b2_recovery_unhealthy else 0.0,
        "atlas_recovery_event_count": float(recovery_event_count),
        "atlas_recovery_seen": 1.0 if recovery_seen else 0.0,
        "b2_unhealthy_split_scale": float(b2_unhealthy_split_scale),
        "b2_unhealthy_explore_scale": float(b2_unhealthy_explore_scale),
        "b2_unhealthy_clone_scale": float(b2_unhealthy_clone_scale),
        "floater_guard": float(floater_guard),
        "quality_guard": float(quality_guard),
        "floater_delta": float(floater_delta),
        "floater_ema": float(floater_ema),
        "psnr_delta": float(psnr_delta),
        "l1_delta": float(l1_delta),
        "l1_ema": float(l1_ema),
        "dark_region_completeness_ema": float(dark_completeness),
        "dark_region_completeness_delta": float(densify_runtime_state.get("dark_region_completeness_delta", 0.0) or 0.0),
        "fidelity_handoff_gate": float(fidelity_handoff_gate),
        "fidelity_handoff_completion_gate": float(completion_gate),
        "fidelity_handoff_completion_gate_raw": float(completion_gate_raw),
        "fidelity_handoff_observed_gate": float(observed_gate),
        "fidelity_handoff_dark_gate": float(dark_gate),
        "fidelity_handoff_stable_gate": float(stable_gate),
        "fidelity_handoff_floater_gate": float(floater_handoff_gate),
        "fidelity_handoff_quality_gate": float(quality_handoff_gate),
        "fidelity_handoff_late_phase_boost": 1.0 if late_phase_fidelity_boost else 0.0,
        **fidelity_mode_metrics,
        "fidelity_handoff_budget_scale": float(budget_handoff_scale),
        "fidelity_handoff_split_scale": float(split_handoff_scale),
        "fidelity_handoff_clone_scale": float(clone_handoff_scale),
        "fidelity_handoff_explore_scale": float(explore_handoff_scale),
        "observed_node_ratio": float(observed_node_ratio),
        "stable_ratio": float(stable_ratio),
        "reliability_effective_mean": float(reliability_effective_mean),
        "capacity_ratio": float(capacity_ratio),
        "max_new_ratio": float(getattr(opt, "atlas_densify_max_new_ratio", 0.012)),
        "max_new_points": int(getattr(opt, "atlas_densify_max_new_points", 2048)),
        "min_new_points": int(getattr(opt, "atlas_densify_min_new_points", 64)),
        "split_quota_fraction": float(split_fraction),
        "clone_quota_fraction": float(clone_fraction),
        "explore_quota_fraction": float(explore_fraction),
        "fidelity_mode_split_scale": float(getattr(opt, "atlas_fidelity_mode_split_scale", 0.45)),
        "fidelity_mode_clone_boost": float(getattr(opt, "atlas_fidelity_mode_clone_boost", 0.20)),
        "fidelity_mode_explore_scale": float(getattr(opt, "atlas_fidelity_mode_explore_scale", 0.08)),
        "fidelity_mode_background_guard_strength": float(getattr(opt, "atlas_fidelity_mode_background_guard_strength", 1.0)),
        "fidelity_mode_active_prune_boost": float(getattr(opt, "atlas_fidelity_mode_active_prune_boost", 1.0)),
        "active_prune_min_gate": float(getattr(opt, "atlas_fidelity_handoff_active_prune_min_gate", 0.65)),
        "background_ref_score_min": float(getattr(opt, "atlas_background_ref_score_min", 0.06)),
        "background_visibility_min": float(getattr(opt, "atlas_background_visibility_min", 0.003)),
        "background_dead_prune_guard": bool(getattr(opt, "atlas_background_dead_prune_guard", True)),
        "background_soft_prune_guard": bool(getattr(opt, "atlas_background_soft_prune_guard", True)),
    }


def _compute_fidelity_mode(
    dark_completeness: float,
    l1_ema: float,
    floater_ema: float,
    reliability_effective_mean: float,
    completion_gate: float,
    dark_gate: float,
    observed_gate: float,
    stable_gate: float,
    opt,
    handoff_enabled: bool = True,
):
    fidelity_mode_min_dark = float(getattr(
        opt,
        "atlas_fidelity_mode_min_dark_completeness",
        getattr(opt, "atlas_fidelity_handoff_min_dark_completeness", 0.94),
    ))
    fidelity_mode_max_l1 = float(getattr(opt, "atlas_fidelity_mode_max_l1", 0.075))
    fidelity_mode_max_floater = float(getattr(opt, "atlas_fidelity_mode_max_floater", 0.070))
    fidelity_dark_gate = _linear_gate(dark_completeness, fidelity_mode_min_dark - 0.06, fidelity_mode_min_dark)
    fidelity_l1_gate = _linear_gate(
        l1_ema,
        max(fidelity_mode_max_l1 * 0.70, 0.0),
        fidelity_mode_max_l1,
    )
    fidelity_floater_gate = _linear_gate(
        floater_ema,
        max(fidelity_mode_max_floater * 0.70, 0.0),
        fidelity_mode_max_floater,
    )
    fidelity_reliability_gate = _linear_gate(reliability_effective_mean, 0.10, 0.24)
    fidelity_mode_gate = float(
        min(
            max(completion_gate, 0.45 * min(dark_gate, observed_gate)),
            fidelity_dark_gate,
            max(fidelity_reliability_gate, stable_gate),
            fidelity_l1_gate,
            fidelity_floater_gate,
        )
    ) if handoff_enabled else 0.0
    return {
        "fidelity_mode_gate": float(fidelity_mode_gate),
        "fidelity_mode_dark_gate": float(fidelity_dark_gate),
        "fidelity_mode_l1_gate": float(fidelity_l1_gate),
        "fidelity_mode_floater_gate": float(fidelity_floater_gate),
        "fidelity_mode_reliability_gate": float(fidelity_reliability_gate),
        "fidelity_mode_min_dark_completeness": float(fidelity_mode_min_dark),
        "fidelity_mode_max_l1": float(fidelity_mode_max_l1),
        "fidelity_mode_max_floater": float(fidelity_mode_max_floater),
    }


def _compute_mc_blend_weight(atlas_mc_metrics: dict, opt):
    if atlas_mc_metrics is None:
        return 0.0
    valid_pairs = float(atlas_mc_metrics.get("atlas_mc_pairs", 0.0))
    active_fraction = float(atlas_mc_metrics.get("atlas_mc_active_fraction", 0.0))
    if valid_pairs <= 0.0 or active_fraction <= 0.0:
        return 0.0
    pair_gate = min(valid_pairs / float(max(int(getattr(opt, "atlas_mc_pairs", 1)), 1)), 1.0)
    coverage_gate = _linear_gate(
        active_fraction,
        float(getattr(opt, "atlas_mc_active_fraction_start", 0.002)),
        float(getattr(opt, "atlas_mc_active_fraction_full", 0.08)),
    )
    max_weight = float(max(min(getattr(opt, "atlas_mc_max_blend_weight", 0.35), 1.0), 0.0))
    return float(max_weight * pair_gate * coverage_gate)


def _compute_atlas_loss_schedule(iteration: int, opt, gaussians, state_metrics):
    warmup_steps = max(int(getattr(opt, "atlas_reg_warmup_steps", 0)), 0)
    if warmup_steps <= 0:
        warmup_progress = 1.0
    else:
        warmup_progress = min(max(float(iteration) / float(warmup_steps), 0.0), 1.0)
    shape_ramp_iters = max(int(getattr(opt, "atlas_shape_main_phase_ramp_iters", 500)), 0)

    init_points = max(int(gaussians.get_init_point_count()), 1)
    total_points = int(gaussians.get_xyz.shape[0])
    capacity_ratio = float(total_points) / float(init_points)
    stable_ratio = float(state_metrics.get("stable_ratio", 0.0))
    refresh_done = bool(getattr(gaussians, "atlas_refresh_done", False))
    reliability_summary = gaussians.summarize_atlas_reliability_state() if refresh_done else {}
    refresh_snapshot = gaussians.summarize_atlas_refresh_snapshot() if refresh_done else {}
    runtime_mean = float(
        reliability_summary.get(
            "atlas_reliability_effective_mean",
            reliability_summary.get("atlas_reliability_runtime_mean", 0.0),
        )
    )
    runtime_std = float(
        reliability_summary.get(
            "atlas_reliability_effective_std",
            reliability_summary.get("atlas_reliability_runtime_std", 0.0),
        )
    )
    runtime_raw_mean = float(reliability_summary.get("atlas_reliability_runtime_raw_mean", 0.0))
    runtime_mapped_mean = float(reliability_summary.get("atlas_reliability_runtime_mapped_mean", 0.0))
    runtime_override_ratio = float(refresh_snapshot.get("atlas_refresh_snapshot_runtime_override_ratio", 0.0))
    runtime_high_ratio = float(refresh_snapshot.get("atlas_refresh_snapshot_effective_reliability_hist_high", 0.0))
    runtime_mid_ratio = float(refresh_snapshot.get("atlas_refresh_snapshot_effective_reliability_hist_mid", 0.0))
    runtime_mapped_high_ratio = float(refresh_snapshot.get("atlas_refresh_snapshot_runtime_mapped_hist_high", 0.0))
    runtime_mapped_mid_ratio = float(refresh_snapshot.get("atlas_refresh_snapshot_runtime_mapped_hist_mid", 0.0))
    confidence_gate = 0.0
    reliability_stability_signal = 0.0
    if refresh_done:
        mean_gate_for_conf = _linear_gate(runtime_mean, 0.08, 0.32)
        mapped_gate_for_conf = _linear_gate(runtime_mapped_mean, 0.10, 0.48)
        std_gate_for_conf = _linear_gate(runtime_std, 0.02, 0.16)
        high_gate_for_conf = _linear_gate(max(runtime_high_ratio, runtime_mapped_high_ratio), 0.002, 0.08)
        override_gate_for_conf = _linear_gate(runtime_override_ratio, 0.005, 0.12)
        confidence_gate = max(
            0.0,
            min(
                0.28 * mean_gate_for_conf
                + 0.24 * mapped_gate_for_conf
                + 0.18 * std_gate_for_conf
                + 0.18 * high_gate_for_conf
                + 0.12 * override_gate_for_conf,
                1.0,
            ),
        )
        effective_mean_gate = _linear_gate(runtime_mean, 0.10, 0.24)
        effective_separation_gate = _linear_gate(runtime_mid_ratio + 2.0 * runtime_high_ratio, 0.04, 0.34)
        mapped_separation_gate = _linear_gate(runtime_mapped_mid_ratio + 2.0 * runtime_mapped_high_ratio, 0.08, 0.46)
        override_gate_for_state = _linear_gate(runtime_override_ratio, 0.03, 0.24)
        reliability_stability_signal = max(
            0.0,
            min(
                0.34 * effective_mean_gate
                + 0.28 * effective_separation_gate
                + 0.24 * mapped_separation_gate
                + 0.14 * override_gate_for_state,
                1.0,
            ),
        )

    kl_warmup_scale = float(getattr(opt, "atlas_kl_warmup_scale", 0.25))
    kl_scale = kl_warmup_scale * warmup_progress
    if refresh_done:
        kl_scale = kl_warmup_scale + (1.0 - kl_warmup_scale) * warmup_progress
    capacity_gate = _linear_gate(
        capacity_ratio,
        1.0,
        float(getattr(opt, "atlas_reg_ramp_capacity_ratio", 1.15)),
    )
    schedule_stable_signal = max(stable_ratio, reliability_stability_signal)
    mean_gate = _linear_gate(
        schedule_stable_signal,
        float(getattr(opt, "atlas_mean_ramp_stable_ratio", 0.30)),
        1.0,
    )
    ori_gate = _linear_gate(
        schedule_stable_signal,
        float(getattr(opt, "atlas_ori_ramp_stable_ratio", 0.45)),
        1.0,
    )
    aniso_gate = _linear_gate(
        schedule_stable_signal,
        float(getattr(opt, "atlas_aniso_ramp_stable_ratio", 0.60)),
        1.0,
    )
    if shape_ramp_iters <= 0:
        main_phase_progress = 1.0 if refresh_done else 0.0
    else:
        main_phase_progress = _linear_gate(
            max(int(iteration) - warmup_steps, 0),
            0.0,
            float(shape_ramp_iters),
        )
    shape_gate = (1.0 if refresh_done else 0.0) * main_phase_progress
    mean_start_scale = float(max(0.0, min(getattr(opt, "atlas_mean_warmup_scale", 0.35), 1.0)))
    mean_phase_scale = 0.0
    if refresh_done:
        mean_phase_scale = mean_start_scale + (1.0 - mean_start_scale) * main_phase_progress
    refresh_gate = 1.0 if refresh_done else 0.0
    reliability_factor = refresh_gate * (0.15 + 0.85 * confidence_gate)
    capacity_factor = 0.45 + 0.55 * capacity_gate
    mean_stable_factor = 0.35 + 0.65 * mean_gate
    ori_stable_factor = 0.30 + 0.70 * ori_gate
    aniso_stable_factor = 0.25 + 0.75 * aniso_gate
    mean_conf_boost = 1.0 + float(getattr(opt, "atlas_after_refresh_mean_boost", 0.20)) * confidence_gate
    shape_conf_boost = 1.0 + float(getattr(opt, "atlas_after_refresh_shape_boost", 0.12)) * confidence_gate
    mean_scale = refresh_gate * mean_phase_scale * reliability_factor * capacity_factor * mean_stable_factor * mean_conf_boost
    ori_scale = shape_gate * reliability_factor * capacity_factor * ori_stable_factor * shape_conf_boost
    aniso_scale = shape_gate * reliability_factor * capacity_factor * aniso_stable_factor * shape_conf_boost
    shape_floor_reliability_gate = 0.0
    shape_floor_scale = 0.0
    if refresh_done:
        floor_min_reliability = float(getattr(opt, "atlas_shape_floor_min_reliability", 0.06))
        floor_full_reliability = float(getattr(opt, "atlas_shape_floor_full_reliability", 0.16))
        if floor_full_reliability <= floor_min_reliability:
            floor_full_reliability = floor_min_reliability + 1e-6
        effective_reliability_gate = _linear_gate(
            runtime_mean,
            floor_min_reliability,
            floor_full_reliability,
        )
        mapped_reliability_gate = _linear_gate(
            runtime_mapped_mean,
            max(floor_min_reliability, 0.08),
            max(floor_full_reliability, 0.32),
        )
        separated_reliability_gate = _linear_gate(
            runtime_mid_ratio + runtime_high_ratio + runtime_mapped_high_ratio,
            0.001,
            0.06,
        )
        shape_floor_reliability_gate = max(
            confidence_gate,
            effective_reliability_gate,
            mapped_reliability_gate,
            separated_reliability_gate,
        )
        shape_floor_scale = (
            refresh_gate
            * float(max(getattr(opt, "atlas_shape_post_refresh_floor", 0.08), 0.0))
            * capacity_factor
            * shape_conf_boost
            * shape_floor_reliability_gate
        )
        ori_scale = max(ori_scale, shape_floor_scale)
        aniso_scale = max(aniso_scale, shape_floor_scale)
    mean_scale_max = float(max(getattr(opt, "atlas_after_refresh_mean_max_scale", 1.20), 1.0))
    shape_scale_max = float(max(getattr(opt, "atlas_after_refresh_shape_max_scale", 1.12), 1.0))

    return {
        "warmup_progress": float(warmup_progress),
        "main_phase_progress": float(max(0.0, min(main_phase_progress, 1.0))),
        "refresh_gate": float(refresh_gate),
        "after_refresh_confidence_gate": float(confidence_gate),
        "runtime_reliability_mean": float(runtime_mean),
        "runtime_reliability_std": float(runtime_std),
        "runtime_reliability_raw_mean": float(runtime_raw_mean),
        "runtime_reliability_mapped_mean": float(runtime_mapped_mean),
        "runtime_override_ratio": float(runtime_override_ratio),
        "runtime_high_ratio": float(runtime_high_ratio),
        "runtime_mid_ratio": float(runtime_mid_ratio),
        "runtime_mapped_high_ratio": float(runtime_mapped_high_ratio),
        "runtime_mapped_mid_ratio": float(runtime_mapped_mid_ratio),
        "reliability_stability_signal": float(reliability_stability_signal),
        "schedule_stable_signal": float(schedule_stable_signal),
        "capacity_ratio": float(capacity_ratio),
        "capacity_gate": float(capacity_gate),
        "capacity_factor": float(capacity_factor),
        "stable_ratio": float(stable_ratio),
        "mean_gate": float(max(0.0, min(mean_gate, 1.0))),
        "mean_stable_factor": float(mean_stable_factor),
        "ori_stable_factor": float(ori_stable_factor),
        "aniso_stable_factor": float(aniso_stable_factor),
        "reliability_factor": float(reliability_factor),
        "shape_floor_reliability_gate": float(shape_floor_reliability_gate),
        "shape_floor_scale": float(shape_floor_scale),
        "mean_phase_scale": float(max(0.0, min(mean_phase_scale, 1.0))),
        "mean_scale": float(max(0.0, min(mean_scale, mean_scale_max))),
        "kl_scale": float(max(0.0, min(kl_scale, 1.0))),
        "ori_scale": float(max(0.0, min(ori_scale, shape_scale_max))),
        "aniso_scale": float(max(0.0, min(aniso_scale, shape_scale_max))),
    }


def _compute_prune_controls(iteration: int, opt, gaussians):
    init_points = gaussians.get_init_point_count()
    total_points = int(gaussians.get_xyz.shape[0])
    prune_from_iter = max(int(getattr(opt, "prune_from_iter", getattr(opt, "densify_from_iter", 0))), 0)
    min_points_before_prune = gaussians.resolve_min_points_before_prune(
        base_min_points=int(getattr(opt, "min_points_before_prune", 0)),
        growth_ratio=float(getattr(opt, "prune_min_capacity_ratio", 1.25)),
        growth_extra=int(getattr(opt, "prune_min_capacity_extra", 1024)),
    )
    hard_floor_ratio = float(getattr(opt, "prune_hard_floor_ratio", 0.4))
    hard_floor_until_iter = max(int(round(float(getattr(opt, "iterations", 0)) * max(hard_floor_ratio, 0.0))), 0)
    if iteration <= hard_floor_until_iter:
        min_points_to_keep = max(init_points, min_points_before_prune)
    else:
        min_points_to_keep = max(init_points, 0)
    prune_enabled = bool(iteration >= prune_from_iter and total_points >= min_points_before_prune)
    soft_prune_enabled = bool(prune_enabled and total_points > min_points_to_keep)
    return {
        "init_points": int(init_points),
        "total_points": int(total_points),
        "prune_from_iter": int(prune_from_iter),
        "min_points_before_prune": int(min_points_before_prune),
        "min_points_to_keep": int(min_points_to_keep),
        "hard_floor_until_iter": int(hard_floor_until_iter),
        "prune_enabled": prune_enabled,
        "soft_prune_enabled": soft_prune_enabled,
    }


def _init_pose_runtime_state():
    return {
        "quality_ema": None,
        "best_quality_ema": None,
        "quality_bad_streak": 0,
        "quality_regressed_b2": False,
        "b1_success_streak": 0,
        "b1_update_count": 0,
        "b2_update_count": 0,
        "post_refresh_main_iters": 0,
        "freeze_cooldown": 0,
        "freeze_event_count": 0,
        "last_freeze_reason": "none",
        "last_b1_skip_reason": "none",
        "last_b2_skip_reason": "none",
        "last_b1_success_iter": -1,
        "last_b1_residual_reduction_px": 0.0,
        "last_b1_residual_reduction_ratio": 0.0,
        "b1_no_improve_streak": 0,
        "last_exposure_reason": "default_allow",
        "b1_camera_attempt_count": 0,
        "b1_camera_execute_count": 0,
        "b1_camera_optimizer_step_count": 0,
        "b2_camera_attempt_count": 0,
        "b2_camera_execute_count": 0,
        "b2_camera_optimizer_step_count": 0,
        "b1_skip_hist": {},
        "b2_skip_hist": {},
        "b1_camera_success": {},
        "b2_camera_success": {},
        "b1_camera_quality": {},
        "b1_camera_median_px": {},
        "b1_camera_no_improve": {},
        "b1_camera_last_success_iter": {},
        "b2_camera_last_attempt_iter": {},
        "last_b2_attempt_iter": -1,
        "last_b2_success_iter": -1,
        "freeze_recovery_good_streak": 0,
        "delta_t_sum": 0.0,
        "delta_angle_sum": 0.0,
        "delta_samples": 0,
    }


def _update_pose_runtime_aggregates(pose_runtime_state: dict, pose_metrics: dict | None, current_pose_delta: dict | None):
    if pose_metrics is None:
        return

    if float(pose_metrics.get("b1_attempt_count", pose_metrics.get("b1_attempted", 0.0))) > 0.5:
        pose_runtime_state["b1_camera_attempt_count"] = int(pose_runtime_state.get("b1_camera_attempt_count", 0)) + 1
    if float(pose_metrics.get("b1_execute_count", pose_metrics.get("b1_executed", 0.0))) > 0.5:
        pose_runtime_state["b1_camera_execute_count"] = int(pose_runtime_state.get("b1_camera_execute_count", 0)) + 1
    if float(pose_metrics.get("b1_optimizer_step_count", pose_metrics.get("b1_executed", 0.0))) > 0.5:
        pose_runtime_state["b1_camera_optimizer_step_count"] = int(pose_runtime_state.get("b1_camera_optimizer_step_count", 0)) + 1
    if float(pose_metrics.get("b2_attempt_count", pose_metrics.get("b2_attempted", 0.0))) > 0.5:
        pose_runtime_state["b2_camera_attempt_count"] = int(pose_runtime_state.get("b2_camera_attempt_count", 0)) + 1
    if float(pose_metrics.get("b2_execute_count", pose_metrics.get("b2_executed", 0.0))) > 0.5:
        pose_runtime_state["b2_camera_execute_count"] = int(pose_runtime_state.get("b2_camera_execute_count", 0)) + 1
    if float(pose_metrics.get("b2_optimizer_step_count", pose_metrics.get("b2_executed", 0.0))) > 0.5:
        pose_runtime_state["b2_camera_optimizer_step_count"] = int(pose_runtime_state.get("b2_camera_optimizer_step_count", 0)) + 1

    b1_skip_reason = str(pose_metrics.get("b1_skip_reason", "none"))
    b2_skip_reason = str(pose_metrics.get("b2_skip_reason", "none"))
    if b1_skip_reason not in ("none", "ok"):
        _increment_histogram(pose_runtime_state.setdefault("b1_skip_hist", {}), b1_skip_reason)
    if b2_skip_reason not in ("none", "ok"):
        _increment_histogram(pose_runtime_state.setdefault("b2_skip_hist", {}), b2_skip_reason)

    if current_pose_delta is not None:
        pose_runtime_state["delta_t_sum"] = float(pose_runtime_state.get("delta_t_sum", 0.0)) + float(current_pose_delta.get("translation_norm", 0.0))
        pose_runtime_state["delta_angle_sum"] = float(pose_runtime_state.get("delta_angle_sum", 0.0)) + float(current_pose_delta.get("rotation_degrees", 0.0))
        pose_runtime_state["delta_samples"] = int(pose_runtime_state.get("delta_samples", 0)) + 1


def _build_pose_runtime_log_fields(pose_runtime_state: dict | None):
    pose_runtime_state = pose_runtime_state or {}
    delta_samples = max(int(pose_runtime_state.get("delta_samples", 0)), 0)
    b1_camera_success = pose_runtime_state.get("b1_camera_success", {}) or {}
    b2_camera_success = pose_runtime_state.get("b2_camera_success", {}) or {}
    fields = {
        "pose_b1_camera_attempt_count": int(pose_runtime_state.get("b1_camera_attempt_count", 0)),
        "pose_b1_camera_execute_count": int(pose_runtime_state.get("b1_camera_execute_count", 0)),
        "pose_b1_camera_optimizer_step_count": int(pose_runtime_state.get("b1_camera_optimizer_step_count", 0)),
        "pose_b2_camera_attempt_count": int(pose_runtime_state.get("b2_camera_attempt_count", 0)),
        "pose_b2_camera_execute_count": int(pose_runtime_state.get("b2_camera_execute_count", 0)),
        "pose_b2_camera_optimizer_step_count": int(pose_runtime_state.get("b2_camera_optimizer_step_count", 0)),
        "pose_b1_success_camera_count": int(len(b1_camera_success)),
        "pose_b2_success_camera_count": int(len(b2_camera_success)),
        "pose_b1_success_camera_max": int(max(b1_camera_success.values(), default=0)),
        "pose_b2_success_camera_max": int(max(b2_camera_success.values(), default=0)),
        "pose_freeze_recovery_good_streak": int(pose_runtime_state.get("freeze_recovery_good_streak", 0)),
        "pose_last_b1_success_iter": int(pose_runtime_state.get("last_b1_success_iter", -1)),
        "pose_b1_no_improve_streak": int(pose_runtime_state.get("b1_no_improve_streak", 0)),
        "pose_b1_last_residual_reduction_px": float(pose_runtime_state.get("last_b1_residual_reduction_px", 0.0)),
        "pose_b1_last_residual_reduction_ratio": float(pose_runtime_state.get("last_b1_residual_reduction_ratio", 0.0)),
        "pose_last_b2_attempt_iter": int(pose_runtime_state.get("last_b2_attempt_iter", -1)),
        "pose_last_b2_success_iter": int(pose_runtime_state.get("last_b2_success_iter", -1)),
        "pose_delta_t_mean": float(pose_runtime_state.get("delta_t_sum", 0.0) / max(delta_samples, 1)),
        "pose_delta_angle_mean": float(pose_runtime_state.get("delta_angle_sum", 0.0) / max(delta_samples, 1)),
        "pose_runtime_samples": int(delta_samples),
    }
    fields.update(_flatten_histogram("pose_b1_skip_hist", pose_runtime_state.get("b1_skip_hist")))
    fields.update(_flatten_histogram("pose_b2_skip_hist", pose_runtime_state.get("b2_skip_hist")))
    return fields


def _update_pose_quality_state(pose_runtime_state: dict, render_loss_value: float, opt):
    decay = float(max(min(getattr(opt, "pose_quality_ema_decay", 0.9), 0.9999), 0.0))
    render_loss_value = float(render_loss_value)
    quality_ema = pose_runtime_state.get("quality_ema", None)
    if quality_ema is None:
        quality_ema = render_loss_value
    else:
        quality_ema = decay * float(quality_ema) + (1.0 - decay) * render_loss_value

    best_quality = pose_runtime_state.get("best_quality_ema", None)
    pose_history_active = bool(int(pose_runtime_state.get("b1_update_count", 0)) > 0 or int(pose_runtime_state.get("b2_update_count", 0)) > 0)
    quality_regressed_b2 = False
    if pose_history_active and best_quality is not None:
        quality_regressed_b2 = quality_ema > float(best_quality) * (1.0 + float(getattr(opt, "pose_b2_max_quality_regression", 0.01)))
        freeze_quality_regressed = quality_ema > float(best_quality) * (1.0 + float(getattr(opt, "pose_freeze_quality_regression", 0.03)))
        if freeze_quality_regressed:
            pose_runtime_state["quality_bad_streak"] = int(pose_runtime_state.get("quality_bad_streak", 0)) + 1
        else:
            pose_runtime_state["quality_bad_streak"] = 0
    else:
        pose_runtime_state["quality_bad_streak"] = 0

    if best_quality is None or quality_ema < float(best_quality):
        best_quality = quality_ema

    pose_runtime_state["quality_ema"] = float(quality_ema)
    pose_runtime_state["best_quality_ema"] = float(best_quality)
    pose_runtime_state["quality_regressed_b2"] = bool(quality_regressed_b2)
    return {
        "pose_gate_quality_ema": float(quality_ema),
        "pose_gate_quality_best_ema": float(best_quality),
        "pose_gate_quality_bad_streak": float(max(int(pose_runtime_state.get("quality_bad_streak", 0)), 0)),
        "pose_gate_quality_regressed_b2": 1.0 if quality_regressed_b2 else 0.0,
    }


def _compute_pose_refine_controls(
    refresh_done: bool,
    state_metrics: dict,
    total_points: int,
    init_points: int,
    opt,
    disable_pose_refine: bool,
    pose_runtime_state: dict,
    pose_corr_metrics: dict | None = None,
    runtime_metrics: dict | None = None,
    bootstrap_active: bool = False,
    bootstrap_corr_ready: bool = False,
    iteration: int = 0,
    camera_key: str = "unknown",
):
    pose_corr_metrics = pose_corr_metrics or {}
    runtime_metrics = runtime_metrics or {}
    pose_runtime_state["freeze_cooldown"] = max(int(pose_runtime_state.get("freeze_cooldown", 0)) - 1, 0)
    stable_ratio_threshold_static = float(getattr(opt, "pose_enable_stable_ratio", 0.45))
    stable_ratio_threshold = stable_ratio_threshold_static
    drift_ratio_threshold = float(getattr(opt, "pose_enable_max_drift_ratio", 0.01))
    active_ratio_threshold = float(getattr(opt, "pose_enable_max_active_ratio", getattr(opt, "pose_freeze_max_active_ratio", 0.15)))
    capacity_ratio_threshold_static = float(getattr(opt, "pose_enable_min_capacity_ratio", 1.25))
    capacity_ratio_threshold = capacity_ratio_threshold_static
    stable_ratio_observed = float(state_metrics.get("stable_ratio", 0.0))
    stable_ratio = stable_ratio_observed
    runtime_effective_mean = float(
        runtime_metrics.get(
            "atlas_reliability_effective_mean",
            runtime_metrics.get("runtime_atlas_reliability_effective_mean", 0.0),
        )
        or 0.0
    )
    runtime_effective_mid = float(
        runtime_metrics.get(
            "atlas_reliability_effective_hist_mid",
            runtime_metrics.get("runtime_atlas_reliability_effective_hist_mid", 0.0),
        )
        or 0.0
    )
    runtime_effective_high = float(
        runtime_metrics.get(
            "atlas_reliability_effective_hist_high",
            runtime_metrics.get("runtime_atlas_reliability_effective_hist_high", 0.0),
        )
        or 0.0
    )
    runtime_override_ratio = float(
        runtime_metrics.get(
            "atlas_refresh_snapshot_runtime_override_ratio",
            runtime_metrics.get("refresh_override_ratio", 0.0),
        )
        or 0.0
    )
    reliability_pose_signal = 0.0
    if bool(refresh_done):
        stable_ratio_threshold = max(0.22, stable_ratio_threshold_static - 0.35 * max(min(runtime_effective_mean, 1.0), 0.0))
        capacity_rel_gate = _linear_gate(runtime_effective_mean, 0.10, 0.28)
        capacity_dynamic_ceiling = 1.05 + 0.07 * (1.0 - capacity_rel_gate)
        capacity_ratio_threshold = min(capacity_ratio_threshold_static, capacity_dynamic_ceiling)
        reliability_pose_signal = max(
            0.0,
            min(
                0.40 * _linear_gate(runtime_effective_mean, 0.10, 0.24)
                + 0.38 * _linear_gate(runtime_effective_mid + 2.0 * runtime_effective_high, 0.05, 0.34)
                + 0.22 * _linear_gate(runtime_override_ratio, 0.03, 0.24),
                1.0,
            ),
        )
        stable_ratio = max(stable_ratio_observed, min(reliability_pose_signal, stable_ratio_observed + 0.25))
    drift_ratio = float(state_metrics.get("drift_ratio", 0.0))
    active_ratio = float(state_metrics.get("active_ratio", 0.0))
    quality_bad_streak = int(pose_runtime_state.get("quality_bad_streak", 0))
    freeze_drift_threshold = float(getattr(opt, "pose_freeze_max_drift_ratio", 0.02))
    freeze_active_threshold = float(getattr(opt, "pose_freeze_max_active_ratio", 0.15))
    cooldown_recovery_good = (
        stable_ratio >= stable_ratio_threshold
        and drift_ratio <= 0.5 * freeze_drift_threshold
        and active_ratio <= 0.75 * freeze_active_threshold
        and quality_bad_streak <= 0
    )
    if int(pose_runtime_state.get("freeze_cooldown", 0)) > 0 and cooldown_recovery_good:
        pose_runtime_state["freeze_recovery_good_streak"] = int(pose_runtime_state.get("freeze_recovery_good_streak", 0)) + 1
    elif int(pose_runtime_state.get("freeze_cooldown", 0)) > 0:
        pose_runtime_state["freeze_recovery_good_streak"] = 0
    else:
        pose_runtime_state["freeze_recovery_good_streak"] = 0
    cooldown_recover_ready = bool(
        int(pose_runtime_state.get("freeze_cooldown", 0)) > 0
        and int(pose_runtime_state.get("freeze_recovery_good_streak", 0)) >= int(max(getattr(opt, "pose_freeze_recovery_good_iters", 5), 1))
    )
    if cooldown_recover_ready:
        pose_runtime_state["freeze_cooldown"] = 0
    if bool(bootstrap_active):
        stable_ratio_threshold = min(stable_ratio_threshold, float(getattr(opt, "pose_b1_bootstrap_stable_ratio", stable_ratio_threshold)))
        active_ratio_threshold = max(active_ratio_threshold, float(getattr(opt, "pose_b1_bootstrap_max_active_ratio", active_ratio_threshold)))
        capacity_ratio_threshold = min(capacity_ratio_threshold, float(getattr(opt, "pose_b1_bootstrap_min_capacity_ratio", capacity_ratio_threshold)))
    b1_enabled, pose_gate_metrics = should_enable_pose_refinement(
        disable_pose_refine=disable_pose_refine,
        stable_ratio=stable_ratio,
        drift_ratio=drift_ratio,
        active_ratio=active_ratio,
        total_points=total_points,
        init_point_count=init_points,
        refresh_done=bool(refresh_done),
        stable_ratio_threshold=stable_ratio_threshold,
        drift_ratio_threshold=drift_ratio_threshold,
        active_ratio_threshold=active_ratio_threshold,
        capacity_ratio_threshold=capacity_ratio_threshold,
        atlas_reliability_effective_mean=None,
        local_corr_count=float(pose_corr_metrics.get("pose_corr_trustworthy", 0.0)),
        local_corr_trust_median=float(pose_corr_metrics.get("pose_corr_trust_median", 0.0)),
        local_corr_spatial_coverage=float(pose_corr_metrics.get("pose_corr_spatial_coverage", 0.0)),
        local_corr_valid_ratio=float(pose_corr_metrics.get("pose_corr_valid_ratio", 0.0)),
        local_corr_min_count=float(getattr(opt, "pose_b1_local_min_corr", getattr(opt, "pose_geo_min_corr", 32))),
        local_corr_min_trust_median=float(getattr(opt, "pose_b1_local_min_trust_median", 0.35)),
        local_corr_min_spatial_coverage=float(getattr(opt, "pose_b1_local_min_spatial_coverage", 0.12)),
        local_corr_min_valid_ratio=float(getattr(opt, "pose_b1_local_min_valid_ratio", 0.18)),
    )
    corr_loaded = float(pose_corr_metrics.get("pose_corr_loaded", 0.0))
    corr_projected = float(pose_corr_metrics.get("pose_corr_projected", 0.0))
    corr_in_frame = float(pose_corr_metrics.get("pose_corr_in_frame", 0.0))
    corr_trustworthy = float(pose_corr_metrics.get("pose_corr_trustworthy", 0.0))
    corr_metrics_available = "pose_corr_loaded" in pose_corr_metrics
    corr_min_target = max(float(pose_corr_metrics.get("pose_corr_min_target", getattr(opt, "pose_geo_min_corr", 32))), 1.0)
    geometry_min_corr = max(float(getattr(opt, "pose_b1_geometry_min_corr", corr_min_target)), corr_min_target)
    corridor_min_corr = max(float(getattr(opt, "pose_b1_corridor_min_corr", max(corr_min_target * 0.75, 1.0))), 1.0)
    corr_quality = min(corr_trustworthy / max(geometry_min_corr, 1.0), 4.0)
    corridor_corr_quality = min(corr_trustworthy / max(corridor_min_corr, 1.0), 4.0)
    corr_projected_ratio = corr_projected / max(corr_loaded, 1.0)
    corr_in_frame_ratio = corr_in_frame / max(corr_loaded, 1.0)
    corr_ready = bool(pose_corr_metrics.get("pose_corr_ready", 0.0) > 0.5 or corr_trustworthy >= geometry_min_corr)
    corridor_corr_ready = bool(corr_trustworthy >= corridor_min_corr or pose_corr_metrics.get("pose_corr_bootstrap_ready", 0.0) > 0.5)
    corr_quality_ready = corr_quality >= float(getattr(opt, "pose_b1_geometry_min_corr_quality", 0.75))
    corridor_corr_quality_ready = corridor_corr_quality >= float(getattr(opt, "pose_b1_corridor_min_corr_quality", 0.35))
    ref_quality_ready = bool(
        corr_projected_ratio >= float(getattr(opt, "pose_b1_geometry_min_projected_ratio", 0.35))
        and corr_in_frame_ratio >= float(getattr(opt, "pose_b1_geometry_min_in_frame_ratio", 0.20))
    )
    corridor_ref_quality_ready = bool(
        corr_projected_ratio >= float(getattr(opt, "pose_b1_corridor_min_projected_ratio", 0.15))
        and corr_in_frame_ratio >= float(getattr(opt, "pose_b1_corridor_min_in_frame_ratio", 0.10))
    )
    geometry_safety_ready = bool(
        bool(refresh_done)
        and (not bool(disable_pose_refine))
        and drift_ratio <= float(getattr(opt, "pose_b1_geometry_max_drift_ratio", getattr(opt, "pose_freeze_max_drift_ratio", 0.02)))
        and active_ratio <= float(getattr(opt, "pose_b1_geometry_max_active_ratio", getattr(opt, "pose_freeze_max_active_ratio", 0.15)))
    )
    geometry_ready = bool(corr_ready and corr_quality_ready and ref_quality_ready)
    b1_corridor_ready = bool(corridor_corr_ready and corridor_corr_quality_ready and corridor_ref_quality_ready)
    b1_corridor_safety_ready = bool(
        bool(refresh_done)
        and (not bool(disable_pose_refine))
        and drift_ratio <= float(getattr(opt, "pose_b1_geometry_max_drift_ratio", getattr(opt, "pose_freeze_max_drift_ratio", 0.02)))
        and active_ratio <= float(getattr(opt, "pose_b1_corridor_max_active_ratio", getattr(opt, "pose_b1_geometry_max_active_ratio", 0.18)))
    )
    geometry_override_b1 = bool(geometry_ready and geometry_safety_ready)
    global_b1_corr_ready = bool((not corr_metrics_available) or corr_ready)
    b1_enabled_by_global = bool(b1_enabled and global_b1_corr_ready)
    bootstrap_geometry_ready = bool(
        bool(bootstrap_active)
        and bool(bootstrap_corr_ready)
        and bool(refresh_done)
        and (not bool(disable_pose_refine))
        and ref_quality_ready
        and drift_ratio <= float(getattr(opt, "pose_b1_geometry_max_drift_ratio", getattr(opt, "pose_freeze_max_drift_ratio", 0.02)))
    )
    freeze_active, freeze_metrics = should_freeze_pose_refinement(
        pose_active=bool(
            int(pose_runtime_state.get("b1_update_count", 0)) > 0
            or int(pose_runtime_state.get("b2_update_count", 0)) > 0
            or b1_enabled_by_global
            or geometry_override_b1
            or bootstrap_geometry_ready
            or (b1_corridor_ready and b1_corridor_safety_ready)
        ),
        drift_ratio=drift_ratio,
        active_ratio=active_ratio,
        quality_bad_streak=quality_bad_streak,
        freeze_cooldown=int(pose_runtime_state.get("freeze_cooldown", 0)),
        drift_ratio_threshold=freeze_drift_threshold,
        active_ratio_threshold=freeze_active_threshold,
        max_quality_bad_streak=int(getattr(opt, "pose_freeze_bad_loss_iters", 3)),
        cooldown_recover_ready=cooldown_recover_ready,
    )
    freeze_triggered = bool(freeze_metrics.get("pose_freeze_emergency_stop", 0.0) > 0.5)
    if freeze_triggered:
        pose_runtime_state["freeze_cooldown"] = max(int(getattr(opt, "pose_freeze_cooldown_iters", 50)), 0)
        pose_runtime_state["quality_bad_streak"] = 0
        pose_runtime_state["b1_success_streak"] = 0
        pose_runtime_state["freeze_recovery_good_streak"] = 0
        if pose_runtime_state.get("quality_ema", None) is not None:
            pose_runtime_state["best_quality_ema"] = float(pose_runtime_state["quality_ema"])
        pose_runtime_state["freeze_event_count"] = int(pose_runtime_state.get("freeze_event_count", 0)) + 1
        pose_runtime_state["last_freeze_reason"] = freeze_metrics.get("pose_freeze_reason", "none")
        freeze_active = True
        freeze_metrics["pose_freeze_cooldown"] = float(max(int(pose_runtime_state["freeze_cooldown"]), 0))
        freeze_metrics["pose_freeze_cooldown_active"] = 1.0 if pose_runtime_state["freeze_cooldown"] > 0 else 0.0
        freeze_metrics["pose_freeze_active"] = 1.0
    elif freeze_active:
        pose_runtime_state["last_freeze_reason"] = freeze_metrics.get("pose_freeze_reason", "cooldown")
    else:
        pose_runtime_state["last_freeze_reason"] = "none"

    bootstrap_forced_b1 = bool(
        bool(bootstrap_geometry_ready)
        and (not bool(freeze_active))
    )
    geometry_forced_b1 = bool(geometry_override_b1 and (not bool(freeze_active)))
    corridor_forced_b1 = bool(b1_corridor_ready and b1_corridor_safety_ready and (not bool(freeze_active)))
    final_b1_enabled = bool((b1_enabled_by_global or bootstrap_forced_b1 or geometry_forced_b1 or corridor_forced_b1) and not freeze_active)
    b1_camera_success = pose_runtime_state.setdefault("b1_camera_success", {})
    camera_b1_count = int(b1_camera_success.get(str(camera_key), 0))
    b1_camera_quality = pose_runtime_state.setdefault("b1_camera_quality", {})
    b1_camera_median_px = pose_runtime_state.setdefault("b1_camera_median_px", {})
    b1_camera_no_improve = pose_runtime_state.setdefault("b1_camera_no_improve", {})
    camera_b1_quality = float(b1_camera_quality.get(str(camera_key), 0.0))
    camera_b1_median_px = float(b1_camera_median_px.get(str(camera_key), float("inf")))
    camera_b1_no_improve = int(b1_camera_no_improve.get(str(camera_key), pose_runtime_state.get("b1_no_improve_streak", 0)))
    global_b1_history = int(pose_runtime_state.get("b1_update_count", 0))
    b2_update_interval = int(max(getattr(opt, "pose_b2_update_interval", max(int(getattr(opt, "pose_update_interval", 5)) * 5, 25)), 1))
    b2_fast_interval = int(max(getattr(opt, "pose_b2_quality_update_interval", max(int(round(0.45 * b2_update_interval)), int(getattr(opt, "pose_update_interval", 5)))), 1))
    camera_quality_ready_for_cadence = bool(
        camera_b1_count >= int(max(getattr(opt, "pose_b2_min_camera_b1_updates", 1), 1))
        and (
            camera_b1_quality >= float(getattr(opt, "pose_b2_min_camera_b1_quality", 0.45))
            or (
                math.isfinite(camera_b1_median_px)
                and camera_b1_median_px <= float(getattr(opt, "pose_b2_max_camera_b1_median_px", 96.0))
            )
        )
    )
    b2_effective_interval = b2_fast_interval if camera_quality_ready_for_cadence else b2_update_interval
    b2_last_attempt_by_camera = pose_runtime_state.setdefault("b2_camera_last_attempt_iter", {})
    last_b2_attempt_iter = int(b2_last_attempt_by_camera.get(str(camera_key), -1))
    b2_low_frequency_due = bool(last_b2_attempt_iter < 0 or int(iteration) - last_b2_attempt_iter >= b2_effective_interval)
    b2_photo_corridor_interval = int(max(getattr(opt, "pose_b2_photo_corridor_interval", max(2 * b2_update_interval, 1)), 1))
    b2_photo_corridor_due = bool(
        last_b2_attempt_iter < 0 or int(iteration) - last_b2_attempt_iter >= b2_photo_corridor_interval
    )
    quality_ema_value = pose_runtime_state.get("quality_ema", None)
    quality_best_value = pose_runtime_state.get("best_quality_ema", None)
    if quality_ema_value is not None and quality_best_value is not None:
        quality_gap = max(float(quality_ema_value) - float(quality_best_value), 0.0)
        quality_gap_ratio = quality_gap / max(abs(float(quality_best_value)), 1e-6)
    else:
        quality_gap = 0.0
        quality_gap_ratio = 0.0
    photo_corridor_min_gap = float(max(getattr(opt, "pose_b2_photo_corridor_min_quality_gap", 0.002), 0.0))
    photo_corridor_dark_l1 = float(runtime_metrics.get("dark_region_l1", 0.0))
    photo_corridor_min_dark_l1 = float(max(getattr(opt, "pose_b2_photo_corridor_min_dark_l1", 0.03), 0.0))
    photo_corridor_dark_completeness = float(
        runtime_metrics.get(
            "dark_region_completeness_ema",
            runtime_metrics.get("dark_region_completeness", 0.0),
        )
    )
    photo_corridor_min_dark_completeness = float(max(getattr(opt, "pose_b2_photo_corridor_min_dark_completeness", 0.90), 0.0))
    photo_signal_strength = float(runtime_metrics.get("photo_signal_strength", runtime_metrics.get("photo_signal_candidate_mean", 0.0)) or 0.0)
    thin_support_like_ratio = float(runtime_metrics.get("thin_support_like_ratio", 0.0) or 0.0)
    bg_like_ratio = float(runtime_metrics.get("bg_like_ratio", 0.0) or 0.0)
    photo_corridor_runtime_signal = bool(
        photo_signal_strength >= float(getattr(opt, "pose_b2_photo_signal_strength_min", 0.012))
        or thin_support_like_ratio >= float(getattr(opt, "pose_b2_thin_support_like_min", 0.18))
        or bg_like_ratio >= float(getattr(opt, "pose_b2_bg_like_min", 0.30))
    )
    photo_corridor_background_signal = bool(
        photo_corridor_dark_l1 >= photo_corridor_min_dark_l1
        or photo_corridor_dark_completeness >= photo_corridor_min_dark_completeness
        or photo_corridor_runtime_signal
    )
    photo_corridor_quality_signal = bool(
        bool(pose_runtime_state.get("quality_regressed_b2", False))
        or quality_gap >= photo_corridor_min_gap
        or quality_gap_ratio >= photo_corridor_min_gap
        or photo_corridor_background_signal
    )
    photo_corridor_after_iters = int(max(getattr(opt, "pose_b2_photo_corridor_after_iters", 80), 0))
    photo_corridor_late_ready = bool(int(pose_runtime_state.get("post_refresh_main_iters", 0)) >= photo_corridor_after_iters)
    photo_corridor_min_stable = float(getattr(opt, "pose_b2_photo_corridor_min_stable_ratio", 0.55))
    photo_corridor_max_active = float(getattr(opt, "pose_b2_photo_corridor_max_active_ratio", 0.18))
    photo_corridor_min_corr_ratio = float(getattr(opt, "pose_b2_photo_corridor_min_corr_ratio", 0.15))
    photo_corridor_ref_ready = bool(
        ref_quality_ready
        or (
            corr_projected_ratio >= photo_corridor_min_corr_ratio
            and corr_in_frame_ratio >= photo_corridor_min_corr_ratio
        )
        or not corr_metrics_available
    )
    photo_corridor_scene_signal = bool(
        photo_corridor_quality_signal
        or (
            photo_corridor_late_ready
            and photo_corridor_background_signal
            and stable_ratio >= max(photo_corridor_min_stable * 0.75, 0.20)
        )
        or (
            photo_corridor_late_ready
            and ref_quality_ready
            and stable_ratio >= photo_corridor_min_stable
        )
    )
    photo_corridor_support_ready = bool(
        stable_ratio >= photo_corridor_min_stable
        or (
            b1_corridor_ready
            and corridor_ref_quality_ready
            and stable_ratio >= max(photo_corridor_min_stable * 0.65, 0.20)
        )
        or (
            reliability_pose_signal >= max(photo_corridor_min_stable * 0.75, 0.20)
            and stable_ratio >= max(photo_corridor_min_stable * 0.55, 0.18)
        )
    )
    b2_photo_corridor_open = bool(
        bool(getattr(opt, "pose_b2_photo_corridor_enabled", True))
        and bool(refresh_done)
        and (not bool(disable_pose_refine))
        and (not bool(freeze_active))
        and b2_photo_corridor_due
        and photo_corridor_late_ready
        and photo_corridor_scene_signal
        and photo_corridor_support_ready
        and active_ratio <= photo_corridor_max_active
        and drift_ratio <= float(getattr(opt, "pose_b2_photo_corridor_max_drift_ratio", getattr(opt, "pose_b2_max_drift_ratio", 0.01)))
        and photo_corridor_ref_ready
    )
    b2_due = bool(b2_low_frequency_due or b2_photo_corridor_open)
    b1_history_fresh_window = int(max(getattr(opt, "pose_b2_b1_history_fresh_iters", max(6 * b2_update_interval, 1)), 1))
    last_global_b1_success_iter = int(pose_runtime_state.get("last_b1_success_iter", -1))
    b1_camera_last_success_iter = pose_runtime_state.setdefault("b1_camera_last_success_iter", {})
    last_camera_b1_success_iter = int(b1_camera_last_success_iter.get(str(camera_key), -1))
    global_b1_history_age = int(iteration) - last_global_b1_success_iter if last_global_b1_success_iter >= 0 else -1
    camera_b1_history_age = int(iteration) - last_camera_b1_success_iter if last_camera_b1_success_iter >= 0 else -1
    global_b1_history_fresh = bool(
        global_b1_history > 0
        and (last_global_b1_success_iter < 0 or global_b1_history_age <= b1_history_fresh_window)
    )
    camera_b1_history_fresh = bool(
        camera_b1_count > 0
        and (last_camera_b1_success_iter < 0 or camera_b1_history_age <= b1_history_fresh_window)
    )
    b1_history_fresh = bool(final_b1_enabled or global_b1_history_fresh or camera_b1_history_fresh)
    b2_bootstrap_open = bool(
        bool(refresh_done)
        and bool(bootstrap_corr_ready)
        and stable_ratio >= float(getattr(opt, "pose_b2_bootstrap_stable_ratio", max(stable_ratio_threshold, 0.75)))
        and active_ratio <= float(getattr(opt, "pose_b2_bootstrap_max_active_ratio", active_ratio_threshold))
        and drift_ratio <= float(getattr(opt, "pose_b2_max_drift_ratio", 0.01))
        and int(pose_runtime_state.get("post_refresh_main_iters", 0)) >= int(max(getattr(opt, "pose_b2_bootstrap_after_iters", 50), 0))
    )
    b2_enabled, pose_b2_metrics = should_enable_pose_photometric_refinement(
        disable_pose_refine=disable_pose_refine,
        b1_enabled=final_b1_enabled,
        drift_ratio=drift_ratio,
        b1_success_streak=int(pose_runtime_state.get("b1_success_streak", 0)),
        quality_regressed=bool(pose_runtime_state.get("quality_regressed_b2", False)),
        min_b1_success_streak=int(getattr(opt, "pose_b2_min_b1_updates", 3)),
        drift_ratio_threshold=float(getattr(opt, "pose_b2_max_drift_ratio", 0.01)),
        b1_history_count=global_b1_history,
        per_camera_b1_count=camera_b1_count,
        per_camera_b1_quality=camera_b1_quality,
        per_camera_b1_median_px=camera_b1_median_px,
        min_per_camera_b1_quality=float(getattr(opt, "pose_b2_min_camera_b1_quality", 0.45)),
        max_per_camera_b1_median_px=float(getattr(opt, "pose_b2_max_camera_b1_median_px", 96.0)),
        min_global_b1_history=int(getattr(opt, "pose_b2_min_global_b1_updates", getattr(opt, "pose_b2_min_b1_updates", 3))),
        min_per_camera_b1_history=int(getattr(opt, "pose_b2_min_camera_b1_updates", 1)),
        bootstrap_ready=b2_bootstrap_open,
        low_frequency_due=b2_due,
        b1_history_fresh=b1_history_fresh,
        photo_corridor_ready=b2_photo_corridor_open,
    )
    final_b2_enabled = bool(b2_enabled and not freeze_active)

    metrics = {}
    metrics.update(pose_gate_metrics)
    metrics.update(pose_b2_metrics)
    metrics.update(freeze_metrics)
    metrics["pose_gate_bootstrap_active"] = 1.0 if bool(bootstrap_active) else 0.0
    metrics["pose_gate_bootstrap_corr_ready"] = 1.0 if bool(bootstrap_corr_ready) else 0.0
    metrics["pose_gate_bootstrap_forced_b1"] = 1.0 if bool(bootstrap_forced_b1 and not b1_enabled_by_global) else 0.0
    metrics["pose_gate_b1_geometry_ready"] = 1.0 if geometry_ready else 0.0
    metrics["pose_gate_b1_geometry_override"] = 1.0 if bool(geometry_forced_b1 and not b1_enabled_by_global and not bootstrap_forced_b1) else 0.0
    metrics["pose_gate_b1_corridor_ready"] = 1.0 if b1_corridor_ready else 0.0
    metrics["pose_gate_b1_corridor_safety_ready"] = 1.0 if b1_corridor_safety_ready else 0.0
    metrics["pose_gate_b1_corridor_open"] = 1.0 if corridor_forced_b1 else 0.0
    metrics["pose_gate_b1_corr_quality"] = float(corr_quality)
    metrics["pose_gate_b1_corridor_corr_quality"] = float(corridor_corr_quality)
    metrics["pose_gate_b1_corr_projected_ratio"] = float(corr_projected_ratio)
    metrics["pose_gate_b1_corr_in_frame_ratio"] = float(corr_in_frame_ratio)
    metrics["pose_gate_b1_ref_quality_ready"] = 1.0 if ref_quality_ready else 0.0
    metrics["pose_gate_b1_corridor_ref_quality_ready"] = 1.0 if corridor_ref_quality_ready else 0.0
    metrics["pose_gate_b1_geometry_safety_ready"] = 1.0 if geometry_safety_ready else 0.0
    metrics["pose_gate_b1_enabled"] = 1.0 if final_b1_enabled else 0.0
    metrics["pose_gate_b2_enabled"] = 1.0 if final_b2_enabled else 0.0
    metrics["pose_gate_stable_ratio_observed"] = float(stable_ratio_observed)
    metrics["pose_gate_stable_ratio_effective"] = float(stable_ratio)
    metrics["pose_gate_reliability_stability_signal"] = float(reliability_pose_signal)
    metrics["pose_gate_runtime_effective_mean"] = float(runtime_effective_mean)
    metrics["pose_gate_runtime_effective_mid_ratio"] = float(runtime_effective_mid)
    metrics["pose_gate_runtime_effective_high_ratio"] = float(runtime_effective_high)
    metrics["pose_gate_stable_threshold_static"] = float(stable_ratio_threshold_static)
    metrics["pose_gate_capacity_threshold_static"] = float(capacity_ratio_threshold_static)
    metrics["pose_gate_stable_threshold_dynamic"] = float(stable_ratio_threshold)
    metrics["pose_gate_capacity_threshold_dynamic"] = float(capacity_ratio_threshold)
    metrics["pose_gate_b1_success_streak"] = float(max(int(pose_runtime_state.get("b1_success_streak", 0)), 0))
    metrics["pose_gate_b1_history_count"] = float(max(global_b1_history, 0))
    metrics["pose_gate_camera_b1_history_count"] = float(max(camera_b1_count, 0))
    metrics["pose_gate_b1_no_improve_streak"] = float(max(camera_b1_no_improve, 0))
    metrics["pose_gate_b2_low_frequency_due"] = 1.0 if b2_low_frequency_due else 0.0
    metrics["pose_gate_b2_due"] = 1.0 if b2_due else 0.0
    metrics["pose_gate_b2_update_interval"] = float(b2_update_interval)
    metrics["pose_gate_b2_effective_update_interval"] = float(b2_effective_interval)
    metrics["pose_gate_b2_quality_update_interval"] = float(b2_fast_interval)
    metrics["pose_gate_b2_camera_quality_cadence"] = 1.0 if camera_quality_ready_for_cadence else 0.0
    metrics["pose_gate_b2_camera_last_attempt_iter"] = float(last_b2_attempt_iter)
    metrics["pose_gate_b2_bootstrap_open"] = 1.0 if b2_bootstrap_open else 0.0
    metrics["pose_gate_b2_photo_corridor_open"] = 1.0 if b2_photo_corridor_open else 0.0
    metrics["pose_gate_b2_photo_corridor_enabled"] = 1.0 if bool(getattr(opt, "pose_b2_photo_corridor_enabled", True)) else 0.0
    metrics["pose_gate_b2_photo_corridor_due"] = 1.0 if b2_photo_corridor_due else 0.0
    metrics["pose_gate_b2_photo_corridor_interval"] = float(b2_photo_corridor_interval)
    metrics["pose_gate_b2_photo_corridor_quality_signal"] = 1.0 if photo_corridor_quality_signal else 0.0
    metrics["pose_gate_b2_photo_corridor_scene_signal"] = 1.0 if photo_corridor_scene_signal else 0.0
    metrics["pose_gate_b2_photo_corridor_support_ready"] = 1.0 if photo_corridor_support_ready else 0.0
    metrics["pose_gate_b2_photo_corridor_late_ready"] = 1.0 if photo_corridor_late_ready else 0.0
    metrics["pose_gate_b2_photo_corridor_after_iters"] = float(photo_corridor_after_iters)
    metrics["pose_gate_b2_photo_corridor_quality_gap"] = float(quality_gap)
    metrics["pose_gate_b2_photo_corridor_quality_gap_ratio"] = float(quality_gap_ratio)
    metrics["pose_gate_b2_photo_corridor_min_quality_gap"] = float(photo_corridor_min_gap)
    metrics["pose_gate_b2_photo_corridor_background_signal"] = 1.0 if photo_corridor_background_signal else 0.0
    metrics["pose_gate_b2_photo_corridor_dark_l1"] = float(photo_corridor_dark_l1)
    metrics["pose_gate_b2_photo_corridor_min_dark_l1"] = float(photo_corridor_min_dark_l1)
    metrics["pose_gate_b2_photo_corridor_dark_completeness"] = float(photo_corridor_dark_completeness)
    metrics["pose_gate_b2_photo_corridor_min_dark_completeness"] = float(photo_corridor_min_dark_completeness)
    metrics["pose_gate_b2_photo_signal_strength"] = float(photo_signal_strength)
    metrics["pose_gate_b2_thin_support_like_ratio"] = float(thin_support_like_ratio)
    metrics["pose_gate_b2_bg_like_ratio"] = float(bg_like_ratio)
    metrics["pose_gate_b2_runtime_signal"] = 1.0 if photo_corridor_runtime_signal else 0.0
    metrics["pose_gate_b2_photo_corridor_ref_ready"] = 1.0 if photo_corridor_ref_ready else 0.0
    metrics["pose_gate_b2_photo_corridor_min_stable_ratio"] = float(photo_corridor_min_stable)
    metrics["pose_gate_b2_photo_corridor_max_active_ratio"] = float(photo_corridor_max_active)
    metrics["pose_gate_b2_photo_corridor_min_corr_ratio"] = float(photo_corridor_min_corr_ratio)
    metrics["pose_gate_b2_b1_history_fresh"] = 1.0 if b1_history_fresh else 0.0
    metrics["pose_gate_b2_b1_history_fresh_window"] = float(b1_history_fresh_window)
    metrics["pose_gate_b2_global_b1_history_age"] = float(global_b1_history_age)
    metrics["pose_gate_b2_camera_b1_history_age"] = float(camera_b1_history_age)
    metrics["pose_gate_freeze_recovery_good"] = 1.0 if cooldown_recovery_good else 0.0
    metrics["pose_gate_freeze_recovery_good_streak"] = float(max(int(pose_runtime_state.get("freeze_recovery_good_streak", 0)), 0))
    metrics["pose_gate_freeze_events"] = float(max(int(pose_runtime_state.get("freeze_event_count", 0)), 0))
    b1_effective_update_interval = int(max(getattr(opt, "pose_update_interval", 5), 1))
    b1_lr_scale = 1.0
    if bool(bootstrap_forced_b1 and not b1_enabled_by_global):
        metrics["pose_gate_b1_reason"] = "bootstrap_geometry_ready"
        b1_effective_update_interval = int(max(getattr(opt, "pose_b1_bootstrap_update_interval", b1_effective_update_interval), 1))
        b1_lr_scale = float(getattr(opt, "pose_b1_bootstrap_lr_scale", 0.35))
    elif bool(geometry_forced_b1 and not b1_enabled_by_global):
        metrics["pose_gate_b1_reason"] = "geometry_ready"
        b1_effective_update_interval = int(max(getattr(opt, "pose_b1_geometry_update_interval", b1_effective_update_interval), 1))
        b1_lr_scale = float(getattr(opt, "pose_b1_geometry_lr_scale", 0.50))
    elif bool(corridor_forced_b1 and not b1_enabled_by_global):
        metrics["pose_gate_b1_reason"] = "geometry_corridor_ready"
        b1_effective_update_interval = int(max(getattr(opt, "pose_b1_corridor_update_interval", b1_effective_update_interval), 1))
        b1_lr_scale = float(getattr(opt, "pose_b1_corridor_lr_scale", 0.30))
    elif bool(bootstrap_active and final_b1_enabled):
        b1_effective_update_interval = int(max(getattr(opt, "pose_b1_bootstrap_update_interval", b1_effective_update_interval), 1))
        b1_lr_scale = float(getattr(opt, "pose_b1_bootstrap_lr_scale", 0.35))
    elif bool(b1_enabled and not global_b1_corr_ready and not final_b1_enabled):
        metrics["pose_gate_b1_reason"] = str(pose_corr_metrics.get("pose_corr_reason", "insufficient_correspondence"))
    slowdown_streak = int(max(getattr(opt, "pose_b1_no_improve_streak_for_slowdown", 3), 0))
    if (
        slowdown_streak > 0
        and camera_b1_no_improve >= slowdown_streak
        and final_b1_enabled
        and not bool(bootstrap_forced_b1)
    ):
        interval_mult = float(max(getattr(opt, "pose_b1_no_improve_interval_mult", 2.0), 1.0))
        b1_effective_update_interval = int(max(round(float(b1_effective_update_interval) * interval_mult), 1))
        metrics["pose_gate_b1_no_improve_slowdown"] = 1.0
    else:
        metrics["pose_gate_b1_no_improve_slowdown"] = 0.0
    metrics["pose_gate_b1_effective_update_interval"] = float(b1_effective_update_interval)
    metrics["pose_gate_b1_lr_scale"] = float(max(min(b1_lr_scale, 1.0), 0.0))
    return final_b1_enabled, final_b2_enabled, metrics, freeze_triggered


def _ensure_variational_subspace_info(subspace_info, gaussians, camera_evidence, opt):
    if subspace_info is not None:
        return subspace_info
    return build_variational_subspace(
        gaussians,
        camera_evidence,
        lambda_reg=float(opt.atlas_obs_lambda),
        max_cameras=int(opt.atlas_obs_max_cameras),
        point_chunk_size=int(opt.atlas_obs_point_chunk),
    )


def _reset_scene_pose_deltas(scene, scale: float = 1.0):
    metrics = {
        "pose_reset_count": 0.0,
        "pose_translation_norm_max": 0.0,
        "pose_rotation_degrees_max": 0.0,
    }
    cameras = scene.getTrainCameras(scale)
    if cameras is None:
        return metrics

    reset_count = 0
    translation_max = 0.0
    rotation_max = 0.0
    for camera in cameras:
        reset_metrics = reset_camera_pose_delta(camera)
        reset_count += int(reset_metrics["pose_reset_applied"] > 0.5)
        translation_max = max(translation_max, float(reset_metrics["pose_translation_norm_after"]))
        rotation_max = max(rotation_max, float(reset_metrics["pose_rotation_degrees_after"]))
    scene.set_pose_trainable(False, scale=scale)
    metrics["pose_reset_count"] = float(reset_count)
    metrics["pose_translation_norm_max"] = float(translation_max)
    metrics["pose_rotation_degrees_max"] = float(rotation_max)
    return metrics

def save_training_checkpoint(scene, gaussians, pose_optimizer, iteration, checkpoint_path, run_args=None):
    checkpoint_payload = {
        "format_version": 2,
        "iteration": int(iteration),
        "gaussians": gaussians.capture(),
        "camera_pose_state": scene.export_pose_state(),
        "pose_camera_order": scene.get_pose_camera_order(),
        "pose_optimizer_state": pose_optimizer.state_dict() if pose_optimizer is not None else None,
    }
    if run_args is not None:
        checkpoint_payload["args"] = dict(vars(run_args))
    torch.save(checkpoint_payload, checkpoint_path)


def load_training_checkpoint(checkpoint_path, scene, gaussians, opt, pose_optimizer=None):
    checkpoint_payload = torch.load(checkpoint_path, map_location=gaussians._device())
    if isinstance(checkpoint_payload, dict) and "gaussians" in checkpoint_payload:
        gaussians.restore(checkpoint_payload["gaussians"], opt)
        scene.apply_pose_state(checkpoint_payload.get("camera_pose_state"))
        pose_optimizer_state = checkpoint_payload.get("pose_optimizer_state")
        saved_pose_order = checkpoint_payload.get("pose_camera_order")
        current_pose_order = scene.get_pose_camera_order()
        if pose_optimizer is not None and pose_optimizer_state is not None:
            if saved_pose_order is None or saved_pose_order == current_pose_order:
                pose_optimizer.load_state_dict(pose_optimizer_state)
            else:
                print("Warning: skipping pose optimizer state restore because camera order changed.")
        return int(checkpoint_payload.get("iteration", 0))

    if isinstance(checkpoint_payload, tuple) and len(checkpoint_payload) == 2:
        model_params, first_iter = checkpoint_payload
        gaussians.restore(model_params, opt)
        return int(first_iter)

    raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, run_args=None):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    log_args = run_args if run_args is not None else dataset
    tb_writer = prepare_output_and_logger(log_args)
    if run_args is not None:
        dataset.model_path = run_args.model_path
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    ablation_manifest = _build_ablation_manifest(dataset, opt, gaussians)
    _write_ablation_manifest(scene.model_path, ablation_manifest)
    gaussians.training_setup(opt)
    scene.set_pose_trainable(False)
    pose_optimizer = None
    pose_params = scene.get_pose_parameters()
    if pose_params and opt.pose_lr > 0 and (not bool(getattr(opt, "disable_pose_refine", True))):
        pose_optimizer = torch.optim.Adam(pose_params, lr=opt.pose_lr)
    if checkpoint:
        first_iter = load_training_checkpoint(checkpoint, scene, gaussians, opt, pose_optimizer)
    scene.set_pose_trainable(False)
    if bool(getattr(opt, "disable_pose_refine", True)):
        _reset_scene_pose_deltas(scene)
        if pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)
            pose_optimizer.state.clear()
    atlas_warmup_steps = max(int(getattr(opt, "atlas_reg_warmup_steps", 0)), 0)
    if scene.gaussians.has_atlas_bindings and atlas_warmup_steps > 0 and first_iter <= atlas_warmup_steps:
        _reset_scene_pose_deltas(scene)
        if pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)
            pose_optimizer.state.clear()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    train_cameras_cache = scene.getTrainCameras()
    train_camera_centers_cache = None
    train_camera_centers_dirty = True
    viewpoint_stack = train_cameras_cache.copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    pose_runtime_state = _init_pose_runtime_state()
    densify_runtime_state = _init_densify_runtime_state()
    latest_validation_summary = {}
    progress_update_interval = max(1, int(opt.log_interval))

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    start_iteration = first_iter
    total_training_steps = opt.iterations - start_iteration + 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras_cache.copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)
        atlas_phase = _compute_atlas_phase_controls(iteration, opt, scene.gaussians.has_atlas_bindings)
        warmup_only = bool(atlas_phase["warmup_only"])
        main_phase = bool(atlas_phase["main_phase"])
        refresh_pending = bool(scene.gaussians.has_atlas_bindings and main_phase and (not scene.gaussians.atlas_refresh_done))
        main_phase_ready = bool(main_phase and (not refresh_pending))
        if scene.gaussians.has_atlas_bindings and (warmup_only or refresh_pending):
            atlas_phase["enable_pose_b1"] = False
            atlas_phase["enable_pose_b2"] = False
            atlas_phase["enable_densify"] = False
            atlas_phase["enable_prune"] = False
            atlas_phase["enable_gc"] = False
            atlas_phase["enable_state_update"] = False
            atlas_phase["enable_mc"] = False
            atlas_phase["enable_explore"] = False
        atlas_phase["warmup_only"] = warmup_only
        atlas_phase["main_phase"] = main_phase
        atlas_phase["refresh_pending"] = refresh_pending
        atlas_phase["main_phase_ready"] = main_phase_ready
        disable_pose_refine = bool(getattr(opt, "disable_pose_refine", True))
        pose_phase_eligible = bool(main_phase_ready and pose_optimizer is not None and (not disable_pose_refine))
        atlas_phase["enable_pose_b1"] = False
        atlas_phase["enable_pose_b2"] = False
        prune_controls = _compute_prune_controls(iteration, opt, scene.gaussians)
        atlas_phase["enable_prune"] = bool(atlas_phase["enable_prune"] and prune_controls["prune_enabled"])
        atlas_phase["enable_soft_prune"] = bool(atlas_phase["enable_prune"] and prune_controls["soft_prune_enabled"])
        atlas_phase["pose_refine_disabled_or_blocked_by_phase"] = bool(
            disable_pose_refine or (not main_phase_ready)
        )
        if scene.gaussians.has_atlas_bindings and main_phase_ready and bool(scene.gaussians.atlas_refresh_done):
            pose_runtime_state["post_refresh_main_iters"] = int(pose_runtime_state.get("post_refresh_main_iters", 0)) + 1
        else:
            pose_runtime_state["post_refresh_main_iters"] = 0
        pose_bootstrap_active = bool(
            scene.gaussians.has_atlas_bindings
            and main_phase_ready
            and bool(scene.gaussians.atlas_refresh_done)
            and int(pose_runtime_state.get("post_refresh_main_iters", 0)) <= int(max(getattr(opt, "pose_b1_bootstrap_iters", 0), 0))
        )
        pose_corr_metrics = summarize_pose_correspondence_budget(
            viewpoint_cam,
            min_correspondences=int(getattr(opt, "pose_geo_min_corr", 32)),
            bootstrap_min_correspondences=int(getattr(opt, "pose_b1_bootstrap_min_corr", max(int(getattr(opt, "pose_geo_min_corr", 32)) // 2, 1))),
        ) if main_phase_ready else {
            "pose_corr_loaded": 0.0,
            "pose_corr_projected": 0.0,
            "pose_corr_in_frame": 0.0,
            "pose_corr_trustworthy": 0.0,
            "pose_corr_ready": 0.0,
            "pose_corr_bootstrap_ready": 0.0,
            "pose_corr_min_target": float(max(int(getattr(opt, "pose_geo_min_corr", 32)), 0)),
            "pose_corr_bootstrap_min_target": float(max(int(getattr(opt, "pose_b1_bootstrap_min_corr", max(int(getattr(opt, "pose_geo_min_corr", 32)) // 2, 1))), 0)),
            "pose_corr_reason": "phase_blocked",
        }
        pose_gate_metrics = {
            "pose_gate_disabled": 1.0 if disable_pose_refine else 0.0,
            "pose_gate_enabled": 0.0,
            "pose_gate_b1_enabled": 0.0,
            "pose_gate_b2_enabled": 0.0,
            "pose_gate_blocked_by_phase": 1.0 if (not main_phase_ready) else 0.0,
            "pose_gate_stable_ratio": 0.0,
            "pose_gate_drift_ratio": 0.0,
            "pose_gate_capacity_ratio": 0.0,
            "pose_gate_init_points": float(prune_controls["init_points"]),
            "pose_gate_total_points": float(prune_controls["total_points"]),
            "pose_gate_quality_ema": float(pose_runtime_state["quality_ema"]) if pose_runtime_state["quality_ema"] is not None else 0.0,
            "pose_gate_quality_best_ema": float(pose_runtime_state["best_quality_ema"]) if pose_runtime_state["best_quality_ema"] is not None else 0.0,
            "pose_gate_quality_bad_streak": float(max(int(pose_runtime_state.get("quality_bad_streak", 0)), 0)),
            "pose_gate_b1_success_streak": float(max(int(pose_runtime_state.get("b1_success_streak", 0)), 0)),
            "pose_gate_freeze_events": float(max(int(pose_runtime_state.get("freeze_event_count", 0)), 0)),
            "pose_gate_post_refresh_main_iters": float(max(int(pose_runtime_state.get("post_refresh_main_iters", 0)), 0)),
            "pose_gate_bootstrap_active": 1.0 if pose_bootstrap_active else 0.0,
        }
        pose_gate_metrics.update(pose_corr_metrics)
        in_warmup = bool(atlas_phase["in_warmup"])
        if not main_phase_ready:
            scene.set_pose_trainable(False)
        current_pose_delta = measure_pose_delta(viewpoint_cam)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        densify_viewspace_point_tensor = viewspace_point_tensor
        densify_visibility_filter = visibility_filter
        densify_radii = radii

        alpha_mask = None

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        controller_timing_metrics = {}
        atlas_runtime_metrics = None
        atlas_refresh_metrics = None
        atlas_gc_metrics = None
        pose_metrics = None
        atlas_mc_metrics = None
        atlas_slab_metrics = None
        densify_metrics = {
            "clone_count": 0.0,
            "split_count": 0.0,
            "explore_clone_count": 0.0,
            "stable_clone_count": 0.0,
            "stable_split_count": 0.0,
            "active_explore_clone_count": 0.0,
            "pruned_count": 0.0,
            "prune_after_gc": 0.0,
        }
        atlas_uncertainty_metrics = None
        atlas_loss_schedule = None
        train_cameras = None
        train_camera_centers = None
        subspace_info = None
        promote_to_active_threshold = float(
            getattr(opt, "atlas_promote_to_active_threshold", getattr(opt, "atlas_activate_threshold", 0.0))
        )
        demote_to_passive_threshold = float(
            getattr(opt, "atlas_demote_to_passive_threshold", getattr(opt, "atlas_deactivate_threshold", 0.0))
        )
        active_min_lifetime_iters = int(
            getattr(opt, "atlas_state_active_min_iters", getattr(opt, "atlas_state_cooldown_iters", 0))
        )
        active_max_lifetime_iters = int(
            getattr(opt, "atlas_state_active_max_iters", max(active_min_lifetime_iters * 24, active_min_lifetime_iters + 1))
        )
        active_nonimprove_iters = int(
            getattr(opt, "atlas_state_active_nonimprove_iters", max(active_min_lifetime_iters * 6, active_min_lifetime_iters + 1))
        )
        active_quota_ratio = float(getattr(opt, "atlas_state_active_quota_ratio", 0.0))
        active_quota_min = int(getattr(opt, "atlas_state_active_quota_min", 0))
        active_quota_max = int(getattr(opt, "atlas_state_active_quota_max", 0))
        atlas_state_metrics = scene.gaussians.summarize_atlas_state_metrics() if scene.gaussians.has_atlas_bindings else {
            "stable_ratio": 0.0,
            "passive_ratio": 0.0,
            "active_ratio": 0.0,
            "out_of_anchor_ratio": 0.0,
            "drift_ratio": 0.0,
        }
        if scene.gaussians.has_atlas_bindings:
            train_cameras = train_cameras_cache
            if train_camera_centers_dirty or train_camera_centers_cache is None:
                train_camera_centers_cache = torch.stack([cam.camera_center for cam in train_cameras_cache], dim=0)
                train_camera_centers_dirty = False
            train_camera_centers = train_camera_centers_cache
            if main_phase_ready:
                atlas_uncertainty_metrics = scene.gaussians.apply_uncertainty_guardrails(
                    camera_centers=train_camera_centers,
                    fallback_camera_center=viewpoint_cam.camera_center,
                    slab_radius_mult=opt.atlas_explore_slab_radius_mult,
                    ray_cap_fraction=opt.atlas_sigma_active_ray_max_fraction,
                    parallel_min_ratio=opt.atlas_sigma_parallel_min_ratio,
                    parallel_max_ratio=opt.atlas_sigma_parallel_max_ratio,
                    support_min_ratio=opt.atlas_sigma_support_min_ratio,
                    support_max_ratio=opt.atlas_sigma_support_max_ratio,
                    passive_parallel_max_mult=opt.atlas_sigma_passive_parallel_max_mult,
                    passive_support_max_mult=opt.atlas_sigma_passive_support_max_mult,
                    active_parallel_min_mult=opt.atlas_sigma_active_parallel_min_mult,
                    active_parallel_max_mult=opt.atlas_sigma_active_parallel_max_mult,
                    active_support_min_mult=opt.atlas_sigma_active_support_min_mult,
                    active_support_max_mult=opt.atlas_sigma_active_support_max_mult,
                    active_ray_min_fraction=opt.atlas_sigma_active_ray_min_fraction,
                    active_low_visibility_decay=opt.atlas_sigma_active_low_visibility_decay,
                    decay=opt.atlas_sigma_decay,
                    low_visibility_threshold=opt.atlas_sigma_low_visibility_threshold,
                )
        atlas_runtime_metrics, atlas_refresh_metrics, atlas_state_metrics, atlas_loss_schedule = _run_atlas_state_controller(
            scene=scene,
            viewpoint_cam=viewpoint_cam,
            rendered_image=image,
            gt_image=gt_image,
            radii=radii,
            render_pkg=render_pkg,
            opt=opt,
            dataset=dataset,
            iteration=int(iteration),
            camera_index=vind,
            atlas_phase=atlas_phase,
            prune_controls=prune_controls,
            pose_gate_metrics=pose_gate_metrics,
            atlas_state_metrics=atlas_state_metrics,
            atlas_uncertainty_metrics=atlas_uncertainty_metrics,
            train_camera_centers=train_camera_centers,
            in_warmup=in_warmup,
            warmup_only=warmup_only,
            refresh_pending=refresh_pending,
            main_phase=main_phase,
            main_phase_ready=main_phase_ready,
            current_pose_delta=current_pose_delta,
            promote_to_active_threshold=promote_to_active_threshold,
            demote_to_passive_threshold=demote_to_passive_threshold,
            active_min_lifetime_iters=active_min_lifetime_iters,
            active_quota_ratio=active_quota_ratio,
            active_quota_min=active_quota_min,
            active_quota_max=active_quota_max,
            active_max_lifetime_iters=active_max_lifetime_iters,
            active_nonimprove_iters=active_nonimprove_iters,
            controller_timing_metrics=controller_timing_metrics,
        )

        base_render_loss, Ll1, _ = compute_reconstruction_loss(image, gt_image, opt.lambda_dssim)
        base_render_loss_safe_for_log, base_render_loss_had_nonfinite = _safe_log_scalar(base_render_loss)
        loss = base_render_loss
        if scene.gaussians.has_atlas_bindings and atlas_phase["enable_mc"] and int(opt.atlas_mc_pairs) > 0:
            subspace_info = _ensure_variational_subspace_info(
                subspace_info,
                scene.gaussians,
                train_cameras,
                opt,
            )
            mc_loss, atlas_mc_metrics, grad_render_pkg = estimate_antithetic_render_loss(
                viewpoint_cam,
                scene.gaussians,
                pipe,
                bg,
                gt_image,
                alpha_mask,
                opt.lambda_dssim,
                dataset.train_test_exp,
                SPARSE_ADAM_AVAILABLE,
                subspace_info,
                opt.atlas_mc_pairs,
                opt.atlas_mc_scale,
            )
            atlas_mc_metrics = atlas_mc_metrics or {}
            mc_loss, _, mc_loss_had_nonfinite = _guard_aux_loss(
                mc_loss,
                atlas_mc_metrics,
                "atlas_mc_loss_safe_for_log",
                "atlas_mc_loss_had_nonfinite",
                "nonfinite_mc_count",
            )
            if grad_render_pkg is not None and (not mc_loss_had_nonfinite):
                mc_blend_weight = _compute_mc_blend_weight(atlas_mc_metrics, opt)
                if mc_blend_weight > 0.0:
                    loss = (1.0 - mc_blend_weight) * base_render_loss + mc_blend_weight * mc_loss
                atlas_mc_metrics["atlas_mc_blend_weight"] = float(mc_blend_weight)
                atlas_mc_metrics["atlas_mc_correction_weight"] = float(mc_blend_weight)
                atlas_mc_metrics["atlas_mc_base_render_loss"] = float(base_render_loss_safe_for_log)
                atlas_mc_metrics["atlas_mc_render_supervision_weight"] = float(1.0 - mc_blend_weight)
                atlas_mc_metrics["atlas_mc_densify_source"] = "base_render"
            elif atlas_mc_metrics is not None:
                atlas_mc_metrics["atlas_mc_blend_weight"] = 0.0
                atlas_mc_metrics["atlas_mc_correction_weight"] = 0.0
                atlas_mc_metrics["atlas_mc_base_render_loss"] = float(base_render_loss_safe_for_log)
                atlas_mc_metrics["atlas_mc_render_supervision_weight"] = 1.0
                atlas_mc_metrics["atlas_mc_densify_source"] = "base_render"

        # Depth regularization
        Ll1depth_pure = 0.0
        depth_confidence_mean = None
        depth_weight_mean = None
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            if viewpoint_cam.depth_confidence is not None:
                depth_confidence = viewpoint_cam.depth_confidence.cuda().clamp(0.0, 1.0)
                depth_weight = opt.depth_confidence_min + (1.0 - opt.depth_confidence_min) * depth_confidence.pow(opt.depth_confidence_exponent)
                depth_weight = depth_weight * depth_mask
                weight_sum = depth_weight.sum()
                depth_confidence_mean = depth_confidence.mean().item()
                depth_weight_mean = depth_weight.mean().item()
                if weight_sum.item() > 0:
                    Ll1depth_pure = (torch.abs(invDepth - mono_invdepth) * depth_weight).sum() / weight_sum
                else:
                    Ll1depth_pure = torch.zeros((), device=loss.device, dtype=loss.dtype)
            else:
                Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()

            Ll1depth_tensor = depth_l1_weight(iteration) * Ll1depth_pure
            Ll1depth_tensor, Ll1depth, _ = _guard_aux_loss(
                Ll1depth_tensor,
                None,
                "depth_loss_safe_for_log",
                "depth_loss_had_nonfinite",
                "nonfinite_depth_count",
            )
            loss += Ll1depth_tensor
        else:
            Ll1depth = 0

        atlas_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        atlas_metrics = None
        atlas_kl_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        atlas_kl_metrics = None
        atlas_slab_loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
        atlas_warmup = 1.0
        atlas_loss_controller_start = time.perf_counter()
        if scene.gaussians.has_atlas_bindings:
            atlas_loss_schedule = atlas_loss_schedule or _compute_atlas_loss_schedule(
                iteration,
                opt,
                scene.gaussians,
                scene.gaussians.summarize_atlas_state_metrics(),
            )
            atlas_warmup = float(atlas_loss_schedule["warmup_progress"])
            if float(opt.atlas_kl_weight) * float(atlas_loss_schedule["kl_scale"]) > 0.0:
                subspace_info = _ensure_variational_subspace_info(
                    subspace_info,
                    scene.gaussians,
                    train_cameras,
                    opt,
                )
            atlas_kl_loss, atlas_kl_metrics = compute_local_exact_kl(
                scene.gaussians,
                train_cameras,
                scene.cameras_extent,
                weight=opt.atlas_kl_weight * float(atlas_loss_schedule["kl_scale"]),
                eps_perp=opt.atlas_kl_eps_perp,
                eps_tangent=opt.atlas_kl_eps_tangent,
                lambda_parallel_base=opt.atlas_kl_lambda_parallel_base,
                lambda_parallel_gain=opt.atlas_kl_lambda_parallel_gain,
                lambda_support_base=opt.atlas_kl_lambda_support_base,
                lambda_support_gain=opt.atlas_kl_lambda_support_gain,
                lambda_perp_base=opt.atlas_kl_lambda_perp_base,
                lambda_perp_gain=opt.atlas_kl_lambda_perp_gain,
                subspace_info=subspace_info,
                lambda_reg=float(opt.atlas_obs_lambda),
                max_cameras=int(opt.atlas_obs_max_cameras),
                point_chunk_size=int(opt.atlas_obs_point_chunk),
                passive_state_weight=float(opt.atlas_kl_passive_state_weight),
                active_state_weight=float(opt.atlas_kl_active_state_weight),
            )
            atlas_kl_metrics = atlas_kl_metrics or {}
            atlas_kl_loss, _, _ = _guard_aux_loss(
                atlas_kl_loss,
                atlas_kl_metrics,
                "atlas_kl_total_loss_safe_for_log",
                "atlas_kl_total_loss_had_nonfinite",
                None,
            )
            loss += atlas_kl_loss
            if (
                float(atlas_loss_schedule["mean_scale"]) > 0.0
                or float(atlas_loss_schedule["ori_scale"]) > 0.0
                or float(atlas_loss_schedule["aniso_scale"]) > 0.0
            ):
                atlas_mean_weight_stress_enabled = _iteration_window_enabled(
                    bool(getattr(opt, "atlas_mean_weight_stress_enable", False)),
                    int(iteration),
                    int(getattr(opt, "atlas_mean_weight_stress_start_iter", 0)),
                    int(getattr(opt, "atlas_mean_weight_stress_end_iter", -1)),
                )
                atlas_mean_weight_stress_scale = (
                    float(max(getattr(opt, "atlas_mean_weight_stress_scale", 1.0), 0.0))
                    if atlas_mean_weight_stress_enabled
                    else 1.0
                )
                atlas_mean_weight_effective = (
                    float(opt.atlas_mean_weight)
                    * float(atlas_loss_schedule["mean_scale"])
                    * float(atlas_mean_weight_stress_scale)
                )
                atlas_loss, atlas_metrics = compute_atlas_regularization(
                    scene.gaussians,
                    scene.cameras_extent,
                    mean_weight=atlas_mean_weight_effective,
                    ori_weight=opt.atlas_ori_weight * float(atlas_loss_schedule["ori_scale"]),
                    aniso_weight=opt.atlas_aniso_weight * float(atlas_loss_schedule["aniso_scale"]),
                    huber_delta=opt.atlas_huber_delta,
                    train_cameras=train_cameras,
                    mean_passive_state_weight=float(getattr(opt, "atlas_mean_passive_state_weight", opt.atlas_reg_passive_state_weight)),
                    mean_active_state_weight=float(getattr(opt, "atlas_mean_active_state_weight", 0.0)),
                    passive_state_weight=float(opt.atlas_reg_passive_state_weight),
                    active_state_weight=float(opt.atlas_reg_active_state_weight),
                )
                atlas_metrics = atlas_metrics or {}
                atlas_metrics["atlas_mean_weight_effective"] = float(atlas_mean_weight_effective)
                atlas_metrics["atlas_mean_weight_stress_scale"] = float(atlas_mean_weight_stress_scale)
                atlas_metrics["atlas_mean_weight_stress_enabled"] = 1.0 if atlas_mean_weight_stress_enabled else 0.0
                atlas_metrics["atlas_ori_weight_effective"] = float(opt.atlas_ori_weight * float(atlas_loss_schedule["ori_scale"]))
                atlas_metrics["atlas_aniso_weight_effective"] = float(opt.atlas_aniso_weight * float(atlas_loss_schedule["aniso_scale"]))
                atlas_loss, _, _ = _guard_aux_loss(
                    atlas_loss,
                    atlas_metrics,
                    "atlas_regularization_total_loss_safe_for_log",
                    "atlas_regularization_total_loss_had_nonfinite",
                    "nonfinite_regularization_loss_count",
                )
                loss += atlas_loss
            if atlas_phase["enable_explore"] and atlas_warmup > 0.0 and opt.atlas_slab_weight > 0.0:
                atlas_slab_loss, atlas_slab_metrics = compute_exploration_slab_loss(
                    scene.gaussians,
                    train_camera_centers,
                    weight=opt.atlas_slab_weight * atlas_warmup,
                    slab_radius_mult=opt.atlas_explore_slab_radius_mult,
                    enabled=bool(atlas_phase["enable_explore"]),
                )
                atlas_slab_metrics = atlas_slab_metrics or {}
                atlas_slab_loss, _, _ = _guard_aux_loss(
                    atlas_slab_loss,
                    atlas_slab_metrics,
                    "atlas_slab_total_loss_safe_for_log",
                    "atlas_slab_total_loss_had_nonfinite",
                    "nonfinite_slab_count",
                )
                loss += atlas_slab_loss
        _record_controller_ms(controller_timing_metrics, "atlas_losses", atlas_loss_controller_start)

        pose_controller_start = None
        pose_freeze_reset_metrics = None
        pose_freeze_reset_pending = False
        if scene.gaussians.has_atlas_bindings and main_phase_ready:
            pose_gate_metrics.update(
                _update_pose_quality_state(
                    pose_runtime_state,
                    render_loss_value=float(base_render_loss_safe_for_log),
                    opt=opt,
                )
            )
            pose_b1_enabled, pose_b2_enabled, refined_pose_gate_metrics, pose_freeze_triggered = _compute_pose_refine_controls(
                refresh_done=bool(scene.gaussians.atlas_refresh_done),
                state_metrics=atlas_state_metrics,
                total_points=int(scene.gaussians.get_xyz.shape[0]),
                init_points=int(scene.gaussians.get_init_point_count()),
                opt=opt,
                disable_pose_refine=disable_pose_refine,
                pose_runtime_state=pose_runtime_state,
                pose_corr_metrics=pose_corr_metrics,
                runtime_metrics=atlas_runtime_metrics,
                bootstrap_active=pose_bootstrap_active,
                bootstrap_corr_ready=bool(pose_corr_metrics.get("pose_corr_bootstrap_ready", 0.0) > 0.5),
                iteration=int(iteration),
                camera_key=_pose_camera_key(viewpoint_cam),
            )
            pose_gate_metrics.update(refined_pose_gate_metrics)
            atlas_phase["enable_pose_b1"] = bool(pose_phase_eligible and pose_b1_enabled)
            atlas_phase["enable_pose_b2"] = bool(pose_phase_eligible and pose_b2_enabled)
            if pose_freeze_triggered:
                pose_freeze_reset_pending = True
            if atlas_runtime_metrics is None:
                atlas_runtime_metrics = {}
            atlas_runtime_metrics.update({
                "phase_enable_pose_b1": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "phase_enable_pose_b2": 1.0 if atlas_phase["enable_pose_b2"] else 0.0,
            })
            atlas_runtime_metrics.update(pose_gate_metrics)
        elif scene.gaussians.has_atlas_bindings and atlas_runtime_metrics is not None:
            atlas_runtime_metrics.update(pose_gate_metrics)

        loss.backward()

        if pose_freeze_reset_pending:
            pose_freeze_reset_metrics = _reset_scene_pose_deltas(scene)
            if pose_optimizer is not None:
                pose_optimizer.zero_grad(set_to_none=True)
                pose_optimizer.state.clear()
            if atlas_runtime_metrics is not None:
                atlas_runtime_metrics.update({f"pose_reset_{k}": v for k, v in pose_freeze_reset_metrics.items()})

        if iteration < opt.iterations:
            if use_sparse_adam:
                visible = radii > 0
                gaussians.optimizer.step(visible, radii.shape[0])
                gaussians.optimizer.zero_grad(set_to_none = True)
            else:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        pose_controller_start = time.perf_counter()
        base_pose_translation_limit = max(float(opt.pose_max_translation_ratio) * float(scene.cameras_extent), 1e-6)
        base_pose_rotation_limit_degrees = float(opt.pose_max_rotation_degrees)
        current_pose_camera_key = _pose_camera_key(viewpoint_cam)
        pose_grad_norm_floor = float(getattr(opt, "pose_grad_norm_floor", 1e-8))
        pose_b2_grad_norm_floor = float(getattr(opt, "pose_b2_grad_norm_floor", pose_grad_norm_floor))
        pose_b2_translation_grad_floor = float(getattr(opt, "pose_b2_translation_grad_floor", pose_b2_grad_norm_floor))
        pose_b2_rotation_grad_floor = float(getattr(opt, "pose_b2_rotation_grad_floor", pose_b2_grad_norm_floor))
        pose_b2_early_grad_floor_scale = float(max(min(getattr(opt, "pose_b2_early_grad_floor_scale", 0.10), 1.0), 0.0))
        pose_b2_small_step_warmup_steps = int(max(getattr(opt, "pose_b2_small_step_warmup_steps", 8), 0))
        pose_trust_block_ratio = float(getattr(opt, "pose_trust_block_ratio", 2.0))
        bootstrap_min_corr = int(
            getattr(
                opt,
                "pose_b1_bootstrap_min_corr",
                max(int(getattr(opt, "pose_geo_min_corr", 32)) // 2, 1),
            )
        )
        b1_min_correspondences = int(bootstrap_min_corr if pose_bootstrap_active else opt.pose_geo_min_corr)
        b1_bootstrap_forced = bool(
            pose_bootstrap_active
            and bool(pose_corr_metrics.get("pose_corr_bootstrap_ready", 0.0) > 0.5)
            and bool(pose_gate_metrics.get("pose_gate_bootstrap_forced_b1", 0.0) > 0.5)
        )
        b1_effective_update_interval = int(
            max(
                int(round(float(pose_gate_metrics.get("pose_gate_b1_effective_update_interval", opt.pose_update_interval)))),
                1,
            )
        )
        b1_lr_scale = float(max(min(pose_gate_metrics.get("pose_gate_b1_lr_scale", 1.0), 1.0), 0.0))
        pose_update_due = bool(
            pose_optimizer is not None
            and atlas_phase["enable_pose_b1"]
            and iteration % b1_effective_update_interval == 0
        )
        if not pose_update_due:
            pose_metrics = {
                "b1_gate_open": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "b2_gate_open": 1.0 if atlas_phase["enable_pose_b2"] else 0.0,
                "b1_gate_enabled": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "b2_gate_enabled": 1.0 if atlas_phase["enable_pose_b2"] else 0.0,
                "pose_gate_enabled": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "b1_attempt_count": 0.0,
                "b1_execute_count": 0.0,
                "b1_optimizer_step_count": 0.0,
                "b2_attempt_count": 0.0,
                "b2_execute_count": 0.0,
                "b2_optimizer_step_count": 0.0,
                "b1_attempted": 0.0,
                "b1_executed": 0.0,
                "b2_attempted": 0.0,
                "b2_executed": 0.0,
                "nonfinite_pose_count": 0.0,
                "b1_bootstrap_active": 1.0 if pose_bootstrap_active else 0.0,
                "b1_bootstrap_forced": 1.0 if b1_bootstrap_forced else 0.0,
                "b1_effective_update_interval": float(b1_effective_update_interval),
                "b1_lr_scale": float(b1_lr_scale),
                "b1_corridor_open": float(pose_gate_metrics.get("pose_gate_b1_corridor_open", 0.0)),
                "b1_no_improve_streak": float(pose_gate_metrics.get("pose_gate_b1_no_improve_streak", 0.0)),
                "b1_min_corr_target": float(max(b1_min_correspondences, 0)),
                "corr_ready": float(pose_corr_metrics.get("pose_corr_ready", 0.0)),
                "corr_bootstrap_ready": float(pose_corr_metrics.get("pose_corr_bootstrap_ready", 0.0)),
                "corr_trustworthy": float(pose_corr_metrics.get("pose_corr_trustworthy", 0.0)),
                "corr_reason": str(pose_corr_metrics.get("pose_corr_reason", "missing_correspondence")),
                "b1_skip_reason": "update_interval" if atlas_phase["enable_pose_b1"] else pose_gate_metrics.get("pose_gate_b1_reason", "gate_blocked"),
                "b2_skip_reason": (
                    "b2_update_interval"
                    if atlas_phase["enable_pose_b2"]
                    else pose_gate_metrics.get("pose_gate_b2_reason", "gate_blocked")
                ),
                "b1_data_loss": 0.0,
                "b1_total_loss": 0.0,
                "b2_data_loss": 0.0,
                "b2_total_loss": 0.0,
                "trust_translation_limit": base_pose_translation_limit,
                "trust_rotation_limit_degrees": base_pose_rotation_limit_degrees,
                "b2_photo_corridor_open": float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_open", 0.0)),
                "b2_photo_corridor_scene_signal": float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_scene_signal", 0.0)),
                "b2_photo_corridor_support_ready": float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_support_ready", 0.0)),
            }
        if pose_update_due:
            pose_metrics = {
                "b1_gate_open": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "b2_gate_open": 1.0 if atlas_phase["enable_pose_b2"] else 0.0,
                "b1_gate_enabled": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "b2_gate_enabled": 1.0 if atlas_phase["enable_pose_b2"] else 0.0,
                "pose_gate_enabled": 1.0 if atlas_phase["enable_pose_b1"] else 0.0,
                "b1_attempt_count": 1.0,
                "b1_execute_count": 0.0,
                "b1_optimizer_step_count": 0.0,
                "b2_attempt_count": 0.0,
                "b2_execute_count": 0.0,
                "b2_optimizer_step_count": 0.0,
                "b1_attempted": 1.0,
                "b1_executed": 0.0,
                "b2_attempted": 0.0,
                "b2_executed": 0.0,
                "nonfinite_pose_count": 0.0,
                "b1_bootstrap_active": 1.0 if pose_bootstrap_active else 0.0,
                "b1_bootstrap_forced": 1.0 if b1_bootstrap_forced else 0.0,
                "b1_effective_update_interval": float(b1_effective_update_interval),
                "b1_lr_scale": float(b1_lr_scale),
                "b1_corridor_open": float(pose_gate_metrics.get("pose_gate_b1_corridor_open", 0.0)),
                "b1_no_improve_streak": float(pose_gate_metrics.get("pose_gate_b1_no_improve_streak", 0.0)),
                "b1_min_corr_target": float(max(b1_min_correspondences, 0)),
                "corr_ready": float(pose_corr_metrics.get("pose_corr_ready", 0.0)),
                "corr_bootstrap_ready": float(pose_corr_metrics.get("pose_corr_bootstrap_ready", 0.0)),
                "corr_trustworthy": float(pose_corr_metrics.get("pose_corr_trustworthy", 0.0)),
                "corr_reason": str(pose_corr_metrics.get("pose_corr_reason", "missing_correspondence")),
                "b1_skip_reason": "none",
                "b2_skip_reason": "none",
                "trust_translation_limit": base_pose_translation_limit,
                "trust_rotation_limit_degrees": base_pose_rotation_limit_degrees,
            }
            scene.set_pose_trainable(True)
            pose_metrics["reset_nonfinite_before"] = float(_reset_camera_pose_if_nonfinite(viewpoint_cam))

            pose_optimizer.zero_grad(set_to_none=True)
            b1_success = False
            b1_pose_delta_before = measure_pose_delta(viewpoint_cam)
            pose_geo_data_loss, pose_geo_metrics, b1_selected_xyz, b1_selected_xy = compute_pose_geometric_correspondence_loss(
                viewpoint_cam,
                scene.gaussians,
                sample_count=int(opt.pose_sample_count),
                geo_weight=float(opt.pose_geo_weight),
                residual_mad_scale=float(opt.pose_geo_mad_scale),
                residual_percentile=float(opt.pose_geo_percentile),
                min_correspondences=int(b1_min_correspondences),
                return_selected=True,
            )
            b1_used_fallback = False
            if float(pose_geo_data_loss.detach().item()) <= 0.0 and getattr(viewpoint_cam, "depth_reliable", False):
                fallback_geo_loss, fallback_geo_metrics = compute_pose_refinement_losses(
                    viewpoint_cam,
                    scene.gaussians,
                    sample_count=int(opt.pose_sample_count),
                    geo_weight=float(opt.pose_geo_weight),
                    photo_weight=0.0,
                    patch_radius=int(opt.pose_patch_radius),
                )
                if float(fallback_geo_loss.detach().item()) > 0.0:
                    b1_used_fallback = True
                    pose_geo_data_loss = fallback_geo_loss
                    pose_geo_metrics = {f"fallback_{k}": v for k, v in fallback_geo_metrics.items()}

            b1_quality_score = (
                0.25
                if b1_used_fallback
                else compute_pose_quality_score(
                    "b1",
                    pose_geo_metrics,
                    target_count=int(b1_min_correspondences),
                    success_streak=int(pose_runtime_state.get("b1_success_streak", 0)),
                )
            )
            pose_translation_limit, pose_rotation_limit_degrees, pose_b1_trust_window = compute_dynamic_pose_trust_region(
                "b1",
                base_translation_norm=base_pose_translation_limit,
                base_rotation_degrees=base_pose_rotation_limit_degrees,
                quality_score=b1_quality_score,
                bootstrap_active=bool(b1_bootstrap_forced or pose_bootstrap_active),
                min_scale=float(getattr(opt, "pose_trust_min_scale", 0.35)),
                max_scale=float(getattr(opt, "pose_trust_max_scale", 1.75)),
                b2_max_scale=float(getattr(opt, "pose_b2_trust_max_scale", 1.15)),
            )
            pose_metrics["trust_translation_limit"] = pose_translation_limit
            pose_metrics["trust_rotation_limit_degrees"] = pose_rotation_limit_degrees
            b1_trust_weight_scale = float(
                max(0.25, 1.0 - 0.25 * min(int(pose_runtime_state.get("b1_no_improve_streak", 0)), 3))
            )
            pose_metrics["b1_trust_weight_scale"] = float(b1_trust_weight_scale)
            pose_geo_trust_loss, pose_geo_trust_metrics = compute_pose_trust_region_loss(
                viewpoint_cam,
                translation_weight=float(opt.pose_translation_l2_weight) * b1_trust_weight_scale,
                rotation_weight=float(opt.pose_rotation_l2_weight) * b1_trust_weight_scale,
                max_translation_norm=pose_translation_limit,
                max_rotation_degrees=pose_rotation_limit_degrees,
            )
            pose_geo_loss = pose_geo_data_loss + pose_geo_trust_loss
            pose_geo_is_finite = _is_finite_scalar(pose_geo_loss)
            pose_geo_data_loss_safe_for_log, pose_geo_data_had_nonfinite = _safe_log_scalar(pose_geo_data_loss)
            pose_geo_loss_safe_for_log, pose_geo_loss_had_nonfinite = _safe_log_scalar(pose_geo_loss)
            pose_b1_clamp_metrics = {}
            pose_b1_grad_metrics = {
                "pose_grad_norm_translation": 0.0,
                "pose_grad_norm_rotation": 0.0,
                "pose_grad_norm_total": 0.0,
            }
            pose_b1_trust_ratio = float(
                pose_geo_trust_loss.detach().item() / max(float(pose_geo_data_loss.detach().item()), 1e-8)
            ) if float(pose_geo_data_loss.detach().item()) > 0.0 else 0.0
            if pose_geo_loss.requires_grad and pose_geo_is_finite and float(pose_geo_loss.detach().item()) > 0.0:
                pose_geo_loss.backward()
                pose_metrics["b1_execute_count"] = 1.0
                pose_b1_grad_metrics = _camera_pose_grad_metrics(viewpoint_cam)
                trust_region_block = (
                    pose_b1_trust_ratio > pose_trust_block_ratio
                    and pose_b1_grad_metrics["pose_grad_norm_total"] <= pose_grad_norm_floor
                )
                if pose_b1_grad_metrics["pose_grad_norm_total"] > pose_grad_norm_floor and (not trust_region_block):
                    if b1_lr_scale < 1.0:
                        if getattr(viewpoint_cam, "pose_delta_t", None) is not None and viewpoint_cam.pose_delta_t.grad is not None:
                            viewpoint_cam.pose_delta_t.grad.mul_(b1_lr_scale)
                        if getattr(viewpoint_cam, "pose_delta_q", None) is not None and viewpoint_cam.pose_delta_q.grad is not None:
                            viewpoint_cam.pose_delta_q.grad.mul_(b1_lr_scale)
                    pose_optimizer.step()
                    b1_success = True
                    pose_metrics["b1_optimizer_step_count"] = 1.0
                    pose_b1_clamp_metrics = clamp_camera_pose_delta(
                        viewpoint_cam,
                        max_translation_norm=pose_translation_limit,
                        max_rotation_degrees=pose_rotation_limit_degrees,
                    )
                    pose_metrics["b1_skip_reason"] = "ok"
                else:
                    pose_metrics["b1_skip_reason"] = "trust_region_block" if trust_region_block else "gradient_too_small"
            else:
                if not pose_geo_is_finite:
                    pose_metrics["b1_skip_reason"] = "nonfinite"
                elif float(pose_geo_data_loss.detach().item()) <= 0.0:
                    pose_metrics["b1_skip_reason"] = str(
                        pose_geo_metrics.get(
                            "pose_geo_skip_reason",
                            pose_geo_metrics.get("fallback_pose_geo_skip_reason", "quality_below_threshold"),
                        )
                    )
                else:
                    pose_metrics["b1_skip_reason"] = "quality_below_threshold"
            pose_metrics["b1_skipped_nonfinite"] = 0.0 if pose_geo_is_finite else 1.0
            pose_metrics["nonfinite_pose_count"] = float(pose_metrics.get("nonfinite_pose_count", 0.0)) + (
                1.0 if (pose_geo_loss_had_nonfinite or pose_geo_data_had_nonfinite) else 0.0
            )
            pose_metrics["b1_executed"] = 1.0 if b1_success else 0.0
            pose_optimizer.zero_grad(set_to_none=True)
            pose_metrics["reset_nonfinite_after_b1"] = float(_reset_camera_pose_if_nonfinite(viewpoint_cam))
            if hasattr(viewpoint_cam, "refresh_pose_matrices"):
                viewpoint_cam.refresh_pose_matrices()
            b1_pose_delta_after = measure_pose_delta(viewpoint_cam)
            b1_post_geo_metrics = {}
            b1_pre_median_px = float(
                pose_geo_metrics.get(
                    "pose_geo_selected_median_px_error",
                    pose_geo_metrics.get("pose_geo_median_px_error", 0.0),
                )
            )
            if not math.isfinite(b1_pre_median_px):
                b1_pre_median_px = 0.0
            b1_post_median_px = b1_pre_median_px
            if not b1_used_fallback and float(pose_geo_metrics.get("pose_geo_num_corr", 0.0)) > 0.0:
                # Use the same selected correspondence set for post-eval — avoids re-selection
                # noise that could make a valid improvement look like regression.
                if b1_selected_xyz is not None and b1_selected_xy is not None:
                    _post_px = evaluate_correspondence_reprojection_px(viewpoint_cam, b1_selected_xyz, b1_selected_xy)
                    if math.isfinite(_post_px):
                        b1_post_median_px = _post_px
                else:
                    with torch.no_grad():
                        _, b1_post_geo_metrics = compute_pose_geometric_correspondence_loss(
                            viewpoint_cam,
                            scene.gaussians,
                            sample_count=int(opt.pose_sample_count),
                            geo_weight=float(opt.pose_geo_weight),
                            residual_mad_scale=float(opt.pose_geo_mad_scale),
                            residual_percentile=float(opt.pose_geo_percentile),
                            min_correspondences=int(b1_min_correspondences),
                        )
                    b1_post_median_px = float(
                        b1_post_geo_metrics.get(
                            "pose_geo_selected_median_px_error",
                            b1_post_geo_metrics.get("pose_geo_median_px_error", b1_pre_median_px),
                        )
                    )
                    if not math.isfinite(b1_post_median_px):
                        b1_post_median_px = b1_pre_median_px
            b1_reduction_px = float(b1_pre_median_px - b1_post_median_px)
            b1_reduction_ratio = float(b1_reduction_px / max(abs(b1_pre_median_px), 1e-6))
            b1_min_reduction_px = float(max(getattr(opt, "pose_b1_success_min_px_reduction", 0.05), 0.0))
            b1_residual_reduced = bool(b1_success and b1_reduction_px >= b1_min_reduction_px)
            b1_camera_success_criterion = bool(b1_residual_reduced)
            b1_history_healthy = bool(b1_residual_reduced)
            pose_metrics["b1_pose_geo_pre_median_px"] = float(b1_pre_median_px)
            pose_metrics["b1_pose_geo_post_median_px"] = float(b1_post_median_px)
            pose_metrics["b1_pose_geo_median_px_reduction"] = float(b1_reduction_px)
            pose_metrics["b1_pose_geo_reduction_ratio"] = float(b1_reduction_ratio)
            pose_metrics["b1_success_residual_reduced"] = 1.0 if b1_residual_reduced else 0.0
            pose_metrics["b1_camera_success_criterion"] = 1.0 if b1_camera_success_criterion else 0.0
            pose_metrics["b1_history_healthy"] = 1.0 if b1_history_healthy else 0.0
            pose_metrics["b1_camera_key"] = str(current_pose_camera_key)
            pose_metrics["b1_camera_residual_reduced"] = 1.0 if b1_residual_reduced else 0.0
            pose_metrics["b1_pose_delta_translation_before"] = float(b1_pose_delta_before.get("translation_norm", 0.0))
            pose_metrics["b1_pose_delta_rotation_before"] = float(b1_pose_delta_before.get("rotation_degrees", 0.0))
            pose_metrics["b1_pose_delta_translation_after"] = float(b1_pose_delta_after.get("translation_norm", 0.0))
            pose_metrics["b1_pose_delta_rotation_after"] = float(b1_pose_delta_after.get("rotation_degrees", 0.0))
            pose_metrics["b1_pose_delta_translation_step"] = float(
                pose_metrics["b1_pose_delta_translation_after"] - pose_metrics["b1_pose_delta_translation_before"]
            )
            pose_metrics["b1_pose_delta_rotation_step"] = float(
                pose_metrics["b1_pose_delta_rotation_after"] - pose_metrics["b1_pose_delta_rotation_before"]
            )
            for metric_name, metric_value in b1_post_geo_metrics.items():
                pose_metrics[f"b1_post_{metric_name}"] = metric_value
            pose_metrics["b1_data_loss"] = float(pose_geo_data_loss_safe_for_log)
            pose_metrics["b1_total_loss"] = float(pose_geo_loss_safe_for_log)
            pose_metrics["b1_data_loss_had_nonfinite"] = 1.0 if pose_geo_data_had_nonfinite else 0.0
            pose_metrics["b1_total_loss_had_nonfinite"] = 1.0 if pose_geo_loss_had_nonfinite else 0.0
            pose_metrics["b1_used_fallback"] = 1.0 if b1_used_fallback else 0.0
            pose_metrics["b1_quality_score"] = float(b1_quality_score)
            pose_metrics["b1_trust_to_data_ratio"] = float(pose_b1_trust_ratio)
            for metric_name, metric_value in pose_geo_metrics.items():
                pose_metrics[f"b1_{metric_name}"] = metric_value
            for metric_name, metric_value in pose_b1_trust_window.items():
                pose_metrics[f"b1_{metric_name}"] = metric_value
            for metric_name, metric_value in pose_geo_trust_metrics.items():
                pose_metrics[f"b1_{metric_name}"] = metric_value
            for metric_name, metric_value in pose_b1_grad_metrics.items():
                pose_metrics[f"b1_{metric_name}"] = metric_value
            for metric_name, metric_value in pose_b1_clamp_metrics.items():
                pose_metrics[f"b1_{metric_name}"] = metric_value

            if b1_history_healthy:
                pose_runtime_state["b1_success_streak"] = int(pose_runtime_state.get("b1_success_streak", 0)) + 1
                pose_runtime_state["b1_update_count"] = int(pose_runtime_state.get("b1_update_count", 0)) + 1
                pose_runtime_state["last_b1_success_iter"] = int(iteration)
                pose_runtime_state["last_b1_residual_reduction_px"] = float(b1_reduction_px)
                pose_runtime_state["last_b1_residual_reduction_ratio"] = float(b1_reduction_ratio)
                pose_runtime_state["b1_no_improve_streak"] = 0
                b1_camera_success = pose_runtime_state.setdefault("b1_camera_success", {})
                b1_camera_success[current_pose_camera_key] = int(b1_camera_success.get(current_pose_camera_key, 0)) + 1
                b1_camera_quality = pose_runtime_state.setdefault("b1_camera_quality", {})
                b1_camera_median_px = pose_runtime_state.setdefault("b1_camera_median_px", {})
                b1_camera_no_improve = pose_runtime_state.setdefault("b1_camera_no_improve", {})
                b1_camera_last_success_iter = pose_runtime_state.setdefault("b1_camera_last_success_iter", {})
                previous_quality = float(b1_camera_quality.get(current_pose_camera_key, b1_quality_score))
                b1_camera_quality[current_pose_camera_key] = float(max(previous_quality, b1_quality_score))
                b1_camera_median_px[current_pose_camera_key] = float(b1_post_median_px)
                b1_camera_no_improve[current_pose_camera_key] = 0
                b1_camera_last_success_iter[current_pose_camera_key] = int(iteration)
            else:
                pose_runtime_state["b1_success_streak"] = 0
                if b1_success:
                    pose_runtime_state["b1_no_improve_streak"] = int(pose_runtime_state.get("b1_no_improve_streak", 0)) + 1
                    b1_camera_no_improve = pose_runtime_state.setdefault("b1_camera_no_improve", {})
                    b1_camera_no_improve[current_pose_camera_key] = int(b1_camera_no_improve.get(current_pose_camera_key, 0)) + 1
                else:
                    pose_runtime_state["b1_no_improve_streak"] = 0
            pose_metrics["b1_no_improve_streak"] = float(pose_runtime_state.get("b1_no_improve_streak", 0))
            pose_runtime_state["last_b1_skip_reason"] = str(pose_metrics.get("b1_skip_reason", "none"))

            run_b2_now = bool(atlas_phase["enable_pose_b2"])
            pose_metrics["b2_history_ready"] = float(pose_gate_metrics.get("pose_b2_gate_history_ready", 0.0))
            pose_metrics["b2_gate_data_ready"] = float(pose_gate_metrics.get("pose_b2_gate_data_ready", 0.0))
            pose_metrics["b2_gate_quality_ready"] = float(pose_gate_metrics.get("pose_b2_gate_quality_ready", 0.0))
            pose_metrics["b2_gate_optimization_ready"] = float(pose_gate_metrics.get("pose_b2_gate_optimization_ready", 0.0))
            pose_metrics["b2_gate_enabled_for_compute"] = float(pose_gate_metrics.get("pose_b2_gate_enabled_for_compute", 0.0))
            pose_metrics["b2_gate_enabled_for_step"] = float(pose_gate_metrics.get("pose_b2_gate_enabled_for_step", 0.0))
            pose_metrics["b2_bootstrap_open"] = float(pose_gate_metrics.get("pose_gate_b2_bootstrap_open", 0.0))
            pose_metrics["b2_low_frequency_due"] = float(pose_gate_metrics.get("pose_gate_b2_low_frequency_due", 0.0))
            pose_metrics["b2_due"] = float(pose_gate_metrics.get("pose_gate_b2_due", pose_metrics["b2_low_frequency_due"]))
            pose_metrics["b2_photo_corridor_open"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_open", 0.0))
            pose_metrics["b2_photo_corridor_quality_signal"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_quality_signal", 0.0))
            pose_metrics["b2_photo_corridor_scene_signal"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_scene_signal", 0.0))
            pose_metrics["b2_photo_corridor_support_ready"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_support_ready", 0.0))
            pose_metrics["b2_photo_corridor_late_ready"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_late_ready", 0.0))
            pose_metrics["b2_photo_corridor_background_signal"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_background_signal", 0.0))
            pose_metrics["b2_photo_corridor_ref_ready"] = float(pose_gate_metrics.get("pose_gate_b2_photo_corridor_ref_ready", 0.0))
            if run_b2_now:
                pose_runtime_state["last_b2_attempt_iter"] = int(iteration)
                pose_runtime_state.setdefault("b2_camera_last_attempt_iter", {})[current_pose_camera_key] = int(iteration)
                pose_metrics["b2_attempt_count"] = 1.0
                pose_metrics["b2_attempted"] = 1.0
                if hasattr(viewpoint_cam, "refresh_pose_matrices"):
                    viewpoint_cam.refresh_pose_matrices()
                pose_delta_t_param = getattr(viewpoint_cam, "pose_delta_t", None)
                pose_delta_q_param = getattr(viewpoint_cam, "pose_delta_q", None)
                pose_metrics["b2_pose_delta_t_requires_grad"] = (
                    1.0 if pose_delta_t_param is not None and bool(getattr(pose_delta_t_param, "requires_grad", False)) else 0.0
                )
                pose_metrics["b2_pose_delta_q_requires_grad"] = (
                    1.0 if pose_delta_q_param is not None and bool(getattr(pose_delta_q_param, "requires_grad", False)) else 0.0
                )
                pose_metrics["b2_pose_trainable"] = (
                    1.0
                    if pose_metrics["b2_pose_delta_t_requires_grad"] > 0.5
                    or pose_metrics["b2_pose_delta_q_requires_grad"] > 0.5
                    else 0.0
                )
                b2_fullframe_stress_enabled = _iteration_window_enabled(
                    bool(getattr(opt, "pose_b2_fullframe_stress_enable", False)),
                    int(iteration),
                    int(getattr(opt, "pose_b2_fullframe_stress_start_iter", 0)),
                    int(getattr(opt, "pose_b2_fullframe_stress_end_iter", -1)),
                )
                pose_metrics["b2_mode"] = "fullframe" if b2_fullframe_stress_enabled else "patch_budgeted"
                pose_metrics["b2_fullframe_stress_enabled"] = 1.0 if b2_fullframe_stress_enabled else 0.0
                pose_metrics["b2_fullframe_downsample_factor"] = float(
                    max(int(getattr(opt, "pose_b2_fullframe_downsample", 1)), 1)
                )
                if b2_fullframe_stress_enabled:
                    pose_template_image = render(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        bg,
                        use_trained_exp=dataset.train_test_exp,
                        separate_sh=SPARSE_ADAM_AVAILABLE,
                    )["render"]
                else:
                    with torch.no_grad():
                        pose_template_image = render(
                            viewpoint_cam,
                            gaussians,
                            pipe,
                            bg,
                            use_trained_exp=dataset.train_test_exp,
                            separate_sh=SPARSE_ADAM_AVAILABLE,
                        )["render"].detach()

                b2_patchfeat_weight = float(getattr(opt, "pose_b2_patchfeat_weight", getattr(opt, "pose_patchfeat_weight", 0.0)))
                if b2_fullframe_stress_enabled and bool(getattr(opt, "pose_b2_fullframe_disable_patchfeat", True)):
                    b2_patchfeat_weight = 0.0
                    pose_metrics["b2_patchfeat_disabled_by_fullframe_stress"] = 1.0
                else:
                    pose_metrics["b2_patchfeat_disabled_by_fullframe_stress"] = 0.0
                pose_metrics["b2_patchfeat_weight_effective"] = float(b2_patchfeat_weight)
                pose_b2_probe_context = None
                if b2_fullframe_stress_enabled:
                    pose_photo_data_loss_raw, pose_photo_metrics = compute_pose_fullframe_photo_loss(
                        viewpoint_cam,
                        pose_template_image,
                        photo_alpha=float(opt.pose_photo_alpha),
                        gradient_weight=float(opt.pose_gradient_weight),
                        downsample_factor=int(getattr(opt, "pose_b2_fullframe_downsample", 1)),
                        use_gradient_term=True,
                    )
                    pose_photo_data_loss = float(opt.pose_photo_weight) * pose_photo_data_loss_raw
                    pose_photo_metrics["pose_photo_loss_weighted"] = (
                        float(pose_photo_data_loss.detach().item()) if pose_photo_data_loss.numel() == 1 else 0.0
                    )
                else:
                    pose_photo_data_loss, pose_photo_metrics, pose_b2_probe_context = compute_pose_refinement_losses(
                        viewpoint_cam,
                        scene.gaussians,
                        sample_count=int(opt.pose_sample_count),
                        geo_weight=0.0,
                        photo_weight=float(opt.pose_photo_weight),
                        template_image=pose_template_image,
                        photo_alpha=float(opt.pose_photo_alpha),
                        gradient_weight=float(opt.pose_gradient_weight),
                        patch_feature_weight=b2_patchfeat_weight,
                        patch_radius=int(opt.pose_patch_radius),
                        return_probe_context=True,
                    )
                pose_metrics["b2_photo_data_loss_requires_grad"] = 1.0 if pose_photo_data_loss.requires_grad else 0.0
                pose_metrics.update(_camera_pose_debug_snapshot(viewpoint_cam))
                b2_quality_score = compute_pose_quality_score(
                    "b2",
                    pose_photo_metrics,
                    target_count=int(opt.pose_sample_count),
                    success_streak=int(pose_runtime_state.get("b1_success_streak", 0)),
                    quality_regressed=bool(pose_runtime_state.get("quality_regressed_b2", False)),
                )
                b2_corridor_open = bool(float(pose_metrics.get("b2_photo_corridor_open", 0.0)) > 0.5)
                b2_trust_max_scale = float(
                    getattr(
                        opt,
                        "pose_b2_corridor_trust_max_scale" if b2_corridor_open else "pose_b2_trust_max_scale",
                        getattr(opt, "pose_b2_trust_max_scale", 1.15),
                    )
                )
                pose_metrics["b2_trust_max_scale_effective"] = float(b2_trust_max_scale)
                pose_photo_translation_limit, pose_photo_rotation_limit_degrees, pose_b2_trust_window = compute_dynamic_pose_trust_region(
                    "b2",
                    base_translation_norm=base_pose_translation_limit,
                    base_rotation_degrees=base_pose_rotation_limit_degrees,
                    quality_score=b2_quality_score,
                    bootstrap_active=False,
                    min_scale=float(getattr(opt, "pose_trust_min_scale", 0.35)),
                    max_scale=float(getattr(opt, "pose_trust_max_scale", 1.75)),
                    b2_max_scale=b2_trust_max_scale,
                )
                pose_metrics["trust_translation_limit"] = pose_photo_translation_limit
                pose_metrics["trust_rotation_limit_degrees"] = pose_photo_rotation_limit_degrees
                pose_photo_trust_loss, pose_photo_trust_metrics = compute_pose_trust_region_loss(
                    viewpoint_cam,
                    translation_weight=float(opt.pose_translation_l2_weight),
                    rotation_weight=float(opt.pose_rotation_l2_weight),
                    max_translation_norm=pose_photo_translation_limit,
                    max_rotation_degrees=pose_photo_rotation_limit_degrees,
                )
                pose_photo_loss = pose_photo_data_loss + pose_photo_trust_loss
                pose_metrics["b2_photo_loss_raw"] = float(pose_photo_metrics.get("pose_photo_loss", 0.0))
                pose_metrics["b2_photo_loss_weighted"] = float(pose_photo_data_loss.detach().item()) if pose_photo_data_loss.numel() == 1 else 0.0
                pose_metrics["b2_trust_loss"] = float(pose_photo_trust_loss.detach().item()) if pose_photo_trust_loss.numel() == 1 else 0.0
                pose_metrics["b2_combined_loss"] = float(pose_photo_loss.detach().item()) if pose_photo_loss.numel() == 1 else 0.0
                pose_metrics["b2_total_loss_requires_grad"] = 1.0 if pose_photo_loss.requires_grad else 0.0
                pose_photo_is_finite = _is_finite_scalar(pose_photo_loss)
                pose_photo_data_loss_safe_for_log, pose_photo_data_had_nonfinite = _safe_log_scalar(pose_photo_data_loss)
                pose_photo_loss_safe_for_log, pose_photo_loss_had_nonfinite = _safe_log_scalar(pose_photo_loss)
                pose_b2_clamp_metrics = {}
                pose_b2_grad_metrics = {
                    "pose_grad_norm_translation": 0.0,
                    "pose_grad_norm_rotation": 0.0,
                    "pose_grad_norm_total": 0.0,
                }
                pose_b2_data_grad_metrics = dict(pose_b2_grad_metrics)
                pose_b2_post_trust_grad_metrics = dict(pose_b2_grad_metrics)
                pose_b2_data_grads = {}
                b2_success = False
                pose_b2_trust_ratio = float(
                    pose_photo_trust_loss.detach().item() / max(float(pose_photo_data_loss.detach().item()), 1e-8)
                ) if float(pose_photo_data_loss.detach().item()) > 0.0 else 0.0
                pose_b2_autograd_metrics, pose_b2_autograd_grads = _audit_camera_pose_loss_autograd(
                    pose_photo_data_loss,
                    viewpoint_cam,
                )
                pose_metrics.update(pose_b2_autograd_metrics)
                pose_metrics["b2_data_grad_from_autograd_audit"] = 0.0
                if pose_photo_loss.requires_grad and pose_photo_is_finite and float(pose_photo_loss.detach().item()) > 0.0:
                    pose_optimizer.zero_grad(set_to_none=True)
                    if pose_photo_data_loss.requires_grad and _is_finite_scalar(pose_photo_data_loss) and float(pose_photo_data_loss.detach().item()) > 0.0:
                        pose_photo_data_loss.backward(retain_graph=True)
                        pose_b2_data_grad_metrics = _camera_pose_grad_metrics(viewpoint_cam)
                        pose_b2_data_grads = _clone_camera_pose_grads(viewpoint_cam)
                        if (
                            float(pose_b2_data_grad_metrics.get("pose_grad_norm_total", 0.0)) <= 0.0
                            and float(pose_b2_autograd_metrics.get("b2_pose_grad_norm_total", 0.0)) > 0.0
                        ):
                            pose_b2_data_grads = pose_b2_autograd_grads
                            pose_b2_data_grad_metrics = {
                                "pose_grad_norm_rotation": float(pose_b2_autograd_metrics.get("b2_pose_q_grad_norm", 0.0)),
                                "pose_grad_norm_translation": float(pose_b2_autograd_metrics.get("b2_pose_t_grad_norm", 0.0)),
                                "pose_grad_norm_total": float(pose_b2_autograd_metrics.get("b2_pose_grad_norm_total", 0.0)),
                            }
                            pose_metrics["b2_data_grad_from_autograd_audit"] = 1.0
                        else:
                            pose_metrics["b2_data_grad_from_autograd_audit"] = 0.0
                    pose_optimizer.zero_grad(set_to_none=True)
                    pose_photo_loss.backward()
                    pose_metrics["b2_execute_count"] = 1.0
                    pose_b2_post_trust_grad_metrics = _camera_pose_grad_metrics(viewpoint_cam)
                    b2_steps_so_far = int(max(pose_runtime_state.get("b2_update_count", 0), 0))
                    b2_early_window = bool(b2_steps_so_far < pose_b2_small_step_warmup_steps)
                    b2_floor_scale = pose_b2_early_grad_floor_scale if b2_early_window else 1.0
                    if b2_corridor_open:
                        b2_floor_scale = min(
                            b2_floor_scale,
                            float(max(min(getattr(opt, "pose_b2_corridor_step_floor_scale", 0.25), 1.0), 0.0)),
                        )
                    b2_total_floor = max(pose_b2_grad_norm_floor * b2_floor_scale, 0.0)
                    b2_translation_floor = max(pose_b2_translation_grad_floor * b2_floor_scale, 0.0)
                    b2_rotation_floor = max(pose_b2_rotation_grad_floor * b2_floor_scale, 0.0)
                    b2_data_grad_total = float(pose_b2_data_grad_metrics["pose_grad_norm_total"])
                    b2_data_grad_translation = float(pose_b2_data_grad_metrics["pose_grad_norm_translation"])
                    b2_data_grad_rotation = float(pose_b2_data_grad_metrics["pose_grad_norm_rotation"])
                    b2_post_grad_total = float(pose_b2_post_trust_grad_metrics["pose_grad_norm_total"])
                    b2_post_grad_translation = float(pose_b2_post_trust_grad_metrics["pose_grad_norm_translation"])
                    b2_post_grad_rotation = float(pose_b2_post_trust_grad_metrics["pose_grad_norm_rotation"])
                    b2_pre_to_post_grad_shrink_ratio = (
                        b2_post_grad_total / max(b2_data_grad_total, 1e-20)
                        if b2_data_grad_total > 0.0
                        else 0.0
                    )
                    b2_pre_to_post_trans_grad_shrink_ratio = (
                        b2_post_grad_translation / max(b2_data_grad_translation, 1e-20)
                        if b2_data_grad_translation > 0.0
                        else 0.0
                    )
                    b2_pre_to_post_rot_grad_shrink_ratio = (
                        b2_post_grad_rotation / max(b2_data_grad_rotation, 1e-20)
                        if b2_data_grad_rotation > 0.0
                        else 0.0
                    )
                    b2_data_only_vs_total_grad_ratio = (
                        b2_data_grad_total / max(b2_post_grad_total, 1e-20)
                        if b2_data_grad_total > 0.0
                        else 0.0
                    )
                    b2_data_grad_nonzero = bool(math.isfinite(b2_data_grad_total) and b2_data_grad_total > 0.0)
                    b2_post_grad_nonzero = bool(math.isfinite(b2_post_grad_total) and b2_post_grad_total > 0.0)
                    b2_grad_nonzero = bool(b2_post_grad_nonzero)
                    b2_any_grad_nonzero = bool(b2_grad_nonzero or b2_data_grad_nonzero)
                    b2_grad_total_ok = bool(b2_post_grad_total > b2_total_floor)
                    b2_grad_axis_ok = bool(
                        b2_post_grad_translation > b2_translation_floor
                        or b2_post_grad_rotation > b2_rotation_floor
                    )
                    b2_zero_rot_floor = float(max(getattr(opt, "pose_b2_zero_grad_skip_threshold_rot", 1e-12), 0.0))
                    b2_zero_trans_floor = float(max(getattr(opt, "pose_b2_zero_grad_skip_threshold_trans", 1e-12), 0.0))
                    b2_pose_audit_grad_total = float(pose_metrics.get("b2_pose_grad_norm_total", 0.0))
                    b2_pose_graph_connected = bool(float(pose_metrics.get("b2_pose_graph_connected", 0.0)) > 0.5)
                    b2_pose_graph_has_grad = bool(
                        b2_pose_graph_connected
                        and math.isfinite(b2_pose_audit_grad_total)
                        and b2_pose_audit_grad_total > max(b2_zero_trans_floor, b2_zero_rot_floor)
                    )
                    b2_pose_graph_connected_but_tiny = bool(
                        b2_pose_graph_connected
                        and not b2_pose_graph_has_grad
                    )
                    b2_mask_mean = float(pose_photo_metrics.get("pose_mask_mean", 0.0))
                    b2_mask_nonzero_ratio = float(pose_photo_metrics.get("pose_mask_nonzero_ratio", 0.0))
                    b2_photo_signal_strength = float(pose_photo_metrics.get("pose_photo_signal_strength", 0.0))
                    b2_patch_count_used = int(pose_photo_metrics.get("pose_patch_count_used", 0))
                    b2_patch_observable_ratio = float(pose_photo_metrics.get("pose_patch_grad_observable_ratio", 0.0))
                    b2_min_mask_nonzero_ratio = float(max(getattr(opt, "pose_b2_min_mask_nonzero_ratio", 0.05), 0.0))
                    b2_force_step_l1 = float(max(getattr(opt, "pose_b2_force_step_if_photo_high_l1", 0.015), 0.0))
                    b2_min_patch_count = int(max(8, min(int(opt.pose_sample_count), int(round(float(opt.pose_sample_count) * 0.05)))))
                    b2_patch_quality_ok = bool(b2_patch_count_used >= b2_min_patch_count)
                    b2_observable_patch_ok = bool(b2_patch_observable_ratio >= 0.03 or float(pose_photo_metrics.get("pose_mask_grad_mean", 0.0)) >= 0.015)
                    b2_photo_signal_ok = bool(
                        str(pose_photo_metrics.get("pose_photo_skip_reason", "inactive")) == "ok"
                        and float(pose_photo_data_loss.detach().item()) > 0.0
                        and b2_mask_mean > 0.0
                    )
                    b2_photo_signal_strong = bool(
                        b2_photo_signal_ok
                        and b2_mask_nonzero_ratio >= b2_min_mask_nonzero_ratio
                        and float(pose_photo_data_loss.detach().item()) >= b2_force_step_l1
                        and b2_patch_quality_ok
                    )
                    b2_force_small_step_enabled = bool(getattr(opt, "pose_b2_force_small_step_if_photo_high", True))
                    b2_data_grad_usable = bool(
                        b2_data_grad_nonzero
                        and (
                            b2_data_grad_translation > b2_zero_trans_floor
                            or b2_data_grad_rotation > b2_zero_rot_floor
                            or b2_data_grad_total > max(b2_zero_trans_floor, b2_zero_rot_floor)
                        )
                    )
                    b2_small_valid_step_ok = bool(b2_early_window and b2_photo_signal_ok and (b2_grad_nonzero or b2_data_grad_usable))
                    b2_corridor_step_ok = bool(b2_corridor_open and b2_photo_signal_ok and (b2_grad_nonzero or b2_data_grad_usable))
                    b2_tiny_grad_usable = bool(
                        b2_any_grad_nonzero
                        and b2_photo_signal_strong
                        and b2_patch_quality_ok
                        and b2_observable_patch_ok
                    )
                    b2_trust_choked_photo_grad = bool(
                        b2_data_grad_usable
                        and (
                            (not b2_post_grad_nonzero)
                            or b2_post_grad_total < max(b2_data_grad_total * 0.25, max(b2_total_floor, 1e-20))
                        )
                    )
                    b2_trust_choked_pose_grad_exists = bool(
                        b2_pose_graph_has_grad
                        and (
                            (not b2_post_grad_nonzero)
                            or b2_post_grad_total < max(b2_pose_audit_grad_total * 0.25, max(b2_total_floor, 1e-20))
                        )
                    )
                    b2_force_small_step_ok = bool(
                        b2_force_small_step_enabled
                        and b2_photo_signal_strong
                        and (b2_data_grad_usable or b2_tiny_grad_usable)
                        and (
                            b2_trust_choked_photo_grad
                            or b2_corridor_open
                            or b2_early_window
                            or (not b2_grad_total_ok)
                        )
                    )
                    b2_gate_step_ready = bool(float(pose_metrics.get("b2_gate_enabled_for_step", 0.0)) > 0.5)
                    trust_region_block = (
                        pose_b2_trust_ratio > pose_trust_block_ratio
                        and (not b2_photo_signal_ok)
                        and (not b2_photo_signal_strong)
                        and b2_post_grad_total <= b2_total_floor
                        and b2_data_grad_total <= b2_total_floor
                    )
                    b2_low_info_zero_grad = bool(
                        b2_mask_nonzero_ratio < b2_min_mask_nonzero_ratio
                        and (not b2_patch_quality_ok)
                        and (not b2_observable_patch_ok)
                        and b2_data_grad_total <= max(b2_zero_trans_floor, b2_zero_rot_floor)
                        and b2_post_grad_total <= max(b2_zero_trans_floor, b2_zero_rot_floor)
                    )
                    b2_fd_probe_count_state = int(max(pose_runtime_state.get("b2_fd_probe_count", 0), 0))
                    b2_fd_probe_warmup_attempts = int(max(getattr(opt, "pose_b2_fd_probe_warmup_attempts", 8), 0))
                    b2_fd_probe_every = int(max(getattr(opt, "pose_b2_fd_probe_every", 20), 0))
                    b2_fd_probe_max_count = int(max(getattr(opt, "pose_b2_fd_probe_max_count", 64), 0))
                    b2_attempts_so_far = int(max(pose_runtime_state.get("b2_camera_attempt_count", 0), 0))
                    b2_fd_probe_due = bool(
                        b2_fd_probe_max_count > 0
                        and b2_fd_probe_count_state < b2_fd_probe_max_count
                        and (not b2_fullframe_stress_enabled)
                        and b2_photo_signal_ok
                        and b2_patch_quality_ok
                        and (
                            b2_pose_graph_connected_but_tiny
                            or b2_attempts_so_far < b2_fd_probe_warmup_attempts
                            or (b2_fd_probe_every > 0 and (b2_attempts_so_far % b2_fd_probe_every) == 0)
                        )
                    )
                    b2_fd_metrics = _empty_b2_fd_probe_metrics(status="not_due")
                    b2_fd_metrics["b2_fd_probe_count"] = float(b2_fd_probe_count_state)
                    if b2_fd_probe_due:
                        b2_fd_probe_count_state += 1
                        pose_runtime_state["b2_fd_probe_count"] = b2_fd_probe_count_state
                        b2_fd_metrics = _probe_pose_loss_sensitivity(
                            viewpoint_cam,
                            scene.gaussians,
                            pose_b2_probe_context,
                            photo_alpha=float(opt.pose_photo_alpha),
                            gradient_weight=float(opt.pose_gradient_weight),
                            patch_feature_weight=b2_patchfeat_weight,
                            sensitivity_floor=float(max(getattr(opt, "pose_b2_fd_sensitivity_floor", 1e-7), 0.0)),
                            probe_count=b2_fd_probe_count_state,
                        )
                    b2_fd_any_positive = bool(float(b2_fd_metrics.get("b2_fd_any_positive", 0.0)) > 0.5)
                    b2_fd_probe_enabled = bool(float(b2_fd_metrics.get("b2_fd_probe_enabled", 0.0)) > 0.5)
                    b2_fd_status = str(b2_fd_metrics.get("b2_fd_probe_status", "not_due"))
                    b2_fd_flat = bool(
                        b2_fd_probe_enabled
                        and b2_fd_status in ("ok", "partial_ok")
                        and not b2_fd_any_positive
                    )
                    b2_fd_nonflat_autograd_tiny = bool(b2_fd_any_positive and b2_pose_graph_connected_but_tiny)
                    b2_fd_nonflat_trust_choked = bool(
                        b2_fd_any_positive
                        and (
                            b2_trust_choked_pose_grad_exists
                            or (
                                b2_data_grad_total > 0.0
                                and b2_post_grad_total < max(b2_data_grad_total * 0.25, max(b2_total_floor, 1e-20))
                            )
                        )
                    )
                    b2_fd_metrics["b2_fd_flat"] = 1.0 if b2_fd_flat else 0.0
                    b2_fd_metrics["b2_fd_nonflat_autograd_tiny"] = 1.0 if b2_fd_nonflat_autograd_tiny else 0.0
                    b2_fd_metrics["b2_fd_nonflat_trust_choked"] = 1.0 if b2_fd_nonflat_trust_choked else 0.0
                    pose_metrics.update(b2_fd_metrics)
                    step_by_total_grad = bool(
                        b2_grad_nonzero
                        and (b2_grad_total_ok or b2_grad_axis_ok)
                    )
                    step_by_data_grad = bool(
                        b2_data_grad_usable
                        and (
                            b2_force_small_step_ok
                            or b2_trust_choked_photo_grad
                            or b2_corridor_step_ok
                            or b2_photo_signal_strong
                            or b2_small_valid_step_ok
                        )
                    )
                    b2_use_data_grad_for_step = bool(
                        step_by_data_grad
                        and (
                            b2_force_small_step_ok
                            or b2_trust_choked_photo_grad
                            or (not step_by_total_grad)
                            or (b2_photo_signal_strong and b2_data_grad_total > b2_post_grad_total)
                        )
                    )
                    b2_step_allowed = bool(
                        (step_by_total_grad or step_by_data_grad)
                        and (b2_gate_step_ready or b2_photo_signal_strong or b2_force_small_step_ok)
                        and (not trust_region_block)
                    )
                    b2_microstep_enabled = bool(getattr(opt, "pose_b2_microstep_enabled", True))
                    b2_microstep_allowed = bool(
                        b2_microstep_enabled
                        and (not b2_step_allowed)
                        and b2_pose_graph_connected
                        and b2_fd_any_positive
                        and (not b2_fd_flat)
                        and (b2_fd_nonflat_autograd_tiny or b2_pose_graph_connected_but_tiny)
                        and b2_patch_quality_ok
                        and b2_observable_patch_ok
                        and (not trust_region_block)
                        and (b2_pose_audit_grad_total > 0.0 or b2_data_grad_total > 0.0)
                    )
                    pose_metrics["b2_grad_norm_floor_effective"] = float(b2_total_floor)
                    pose_metrics["b2_translation_grad_floor_effective"] = float(b2_translation_floor)
                    pose_metrics["b2_rotation_grad_floor_effective"] = float(b2_rotation_floor)
                    pose_metrics["b2_grad_floor_scale"] = float(b2_floor_scale)
                    pose_metrics["b2_early_step_window"] = 1.0 if b2_early_window else 0.0
                    pose_metrics["b2_grad_nonzero"] = 1.0 if b2_grad_nonzero else 0.0
                    pose_metrics["b2_any_grad_nonzero"] = 1.0 if b2_any_grad_nonzero else 0.0
                    pose_metrics["b2_data_grad_nonzero"] = 1.0 if b2_data_grad_nonzero else 0.0
                    pose_metrics["b2_post_trust_grad_nonzero"] = 1.0 if b2_post_grad_nonzero else 0.0
                    pose_metrics["b2_grad_total_ok"] = 1.0 if b2_grad_total_ok else 0.0
                    pose_metrics["b2_grad_axis_ok"] = 1.0 if b2_grad_axis_ok else 0.0
                    pose_metrics["b2_photo_signal_ok"] = 1.0 if b2_photo_signal_ok else 0.0
                    pose_metrics["b2_photo_signal_strong"] = 1.0 if b2_photo_signal_strong else 0.0
                    pose_metrics["b2_mask_mean"] = float(b2_mask_mean)
                    pose_metrics["b2_mask_nonzero_ratio"] = float(b2_mask_nonzero_ratio)
                    pose_metrics["b2_photo_signal_strength"] = float(b2_photo_signal_strength)
                    pose_metrics["b2_patch_count_used"] = float(b2_patch_count_used)
                    pose_metrics["b2_patch_grad_observable_ratio"] = float(b2_patch_observable_ratio)
                    pose_metrics["b2_observable_patch_ok"] = 1.0 if b2_observable_patch_ok else 0.0
                    pose_metrics["b2_patch_quality_ok"] = 1.0 if b2_patch_quality_ok else 0.0
                    pose_metrics["b2_min_patch_count"] = float(b2_min_patch_count)
                    pose_metrics["b2_photo_grad_norm_rot"] = float(b2_data_grad_rotation)
                    pose_metrics["b2_photo_grad_norm_trans"] = float(b2_data_grad_translation)
                    pose_metrics["b2_pre_trust_grad_norm"] = float(b2_data_grad_total)
                    pose_metrics["b2_post_trust_grad_norm"] = float(b2_post_grad_total)
                    pose_metrics["b2_tiny_grad_usable"] = 1.0 if b2_tiny_grad_usable else 0.0
                    pose_metrics["b2_trust_choked_photo_grad"] = 1.0 if b2_trust_choked_photo_grad else 0.0
                    pose_metrics["b2_trust_choked_but_pose_grad_exists"] = 1.0 if b2_trust_choked_pose_grad_exists else 0.0
                    pose_metrics["b2_pose_graph_connected_but_tiny"] = 1.0 if b2_pose_graph_connected_but_tiny else 0.0
                    pose_metrics["b2_force_small_step_ok"] = 1.0 if b2_force_small_step_ok else 0.0
                    pose_metrics["b2_optimization_ready"] = float(pose_metrics.get("b2_gate_optimization_ready", 0.0))
                    pose_metrics["b2_step_by_total_grad"] = 1.0 if step_by_total_grad else 0.0
                    pose_metrics["b2_step_by_data_grad"] = 1.0 if step_by_data_grad else 0.0
                    pose_metrics["b2_use_data_grad_for_step"] = 1.0 if b2_use_data_grad_for_step else 0.0
                    pose_metrics["b2_low_info_zero_grad"] = 1.0 if b2_low_info_zero_grad else 0.0
                    pose_metrics["b2_small_valid_step_ok"] = 1.0 if b2_small_valid_step_ok else 0.0
                    pose_metrics["b2_corridor_step_ok"] = 1.0 if b2_corridor_step_ok else 0.0
                    pose_metrics["b2_trust_region_block"] = 1.0 if trust_region_block else 0.0
                    pose_metrics["b2_step_allowed"] = 1.0 if b2_step_allowed else 0.0
                    pose_metrics["b2_small_step_mode"] = 0.0
                    pose_metrics["b2_data_only_q_grad_norm"] = float(b2_data_grad_rotation)
                    pose_metrics["b2_data_only_t_grad_norm"] = float(b2_data_grad_translation)
                    pose_metrics["b2_data_only_grad_total"] = float(b2_data_grad_total)
                    pose_metrics["b2_data_only_vs_total_grad_ratio"] = float(b2_data_only_vs_total_grad_ratio)
                    pose_metrics["b2_pre_to_post_grad_shrink_ratio"] = float(b2_pre_to_post_grad_shrink_ratio)
                    pose_metrics["b2_pre_to_post_trans_grad_shrink_ratio"] = float(b2_pre_to_post_trans_grad_shrink_ratio)
                    pose_metrics["b2_pre_to_post_rot_grad_shrink_ratio"] = float(b2_pre_to_post_rot_grad_shrink_ratio)
                    pose_metrics["b2_microstep_allowed"] = 1.0 if b2_microstep_allowed else 0.0
                    pose_metrics["b2_microstep_mode"] = 0.0
                    pose_metrics["b2_microstep_translation_applied"] = 0.0
                    pose_metrics["b2_microstep_rotation_applied_deg"] = 0.0
                    pose_metrics["b2_microstep_reason"] = "inactive"
                    if b2_step_allowed:
                        if b2_use_data_grad_for_step:
                            _restore_camera_pose_grads(viewpoint_cam, pose_b2_data_grads)
                        b2_step_lr_scale = 1.0
                        if b2_force_small_step_ok or b2_corridor_open or (step_by_data_grad and not step_by_total_grad):
                            b2_step_lr_scale = float(max(min(getattr(opt, "pose_b2_small_step_lr_scale", 0.25), 1.0), 0.01))
                        elif b2_photo_signal_strong and not b2_grad_total_ok:
                            b2_step_lr_scale = float(max(min(getattr(opt, "pose_b2_small_step_lr_scale", 0.25), 1.0), 0.01))
                        pose_metrics["b2_step_lr_scale"] = float(b2_step_lr_scale)
                        pose_metrics["b2_small_step_mode"] = 1.0 if b2_step_lr_scale < 0.999 else 0.0
                        pose_b2_grad_metrics = _camera_pose_grad_metrics(viewpoint_cam)
                        original_pose_lrs = None
                        if b2_step_lr_scale < 0.999:
                            original_pose_lrs = [float(group.get("lr", opt.pose_lr)) for group in pose_optimizer.param_groups]
                            for group, original_lr in zip(pose_optimizer.param_groups, original_pose_lrs):
                                group["lr"] = original_lr * b2_step_lr_scale
                        try:
                            pose_optimizer.step()
                        finally:
                            if original_pose_lrs is not None:
                                for group, original_lr in zip(pose_optimizer.param_groups, original_pose_lrs):
                                    group["lr"] = original_lr
                        b2_success = True
                        pose_metrics["b2_optimizer_step_count"] = 1.0
                        pose_b2_clamp_metrics = clamp_camera_pose_delta(
                            viewpoint_cam,
                            max_translation_norm=pose_photo_translation_limit,
                            max_rotation_degrees=pose_photo_rotation_limit_degrees,
                        )
                        pose_metrics["b2_skip_reason"] = "ok"
                        pose_metrics["b2_skip_reason_detailed"] = "ok_data_grad_step" if b2_use_data_grad_for_step else "ok_total_grad_step"
                    elif b2_microstep_allowed:
                        b2_microstep_success, b2_microstep_metrics = _apply_b2_microstep_by_pose_audit_grad(
                            viewpoint_cam,
                            pose_b2_autograd_grads,
                            opt,
                        )
                        pose_metrics.update(b2_microstep_metrics)
                        pose_metrics["b2_step_lr_scale"] = 0.0
                        pose_b2_grad_metrics = pose_b2_data_grad_metrics if b2_data_grad_nonzero else pose_b2_post_trust_grad_metrics
                        if b2_microstep_success:
                            b2_success = True
                            pose_metrics["b2_optimizer_step_count"] = 1.0
                            pose_b2_clamp_metrics = clamp_camera_pose_delta(
                                viewpoint_cam,
                                max_translation_norm=pose_photo_translation_limit,
                                max_rotation_degrees=pose_photo_rotation_limit_degrees,
                            )
                            pose_metrics["b2_skip_reason"] = "ok"
                            pose_metrics["b2_skip_reason_detailed"] = "ok_microstep_audit_grad"
                        else:
                            pose_metrics["b2_skip_reason"] = "microstep_failed"
                            pose_metrics["b2_skip_reason_detailed"] = str(
                                pose_metrics.get("b2_microstep_reason", "microstep_failed")
                            )
                    else:
                        pose_metrics["b2_step_lr_scale"] = 0.0
                        pose_b2_grad_metrics = pose_b2_post_trust_grad_metrics
                        if b2_trust_choked_pose_grad_exists and (trust_region_block or not b2_post_grad_nonzero):
                            pose_metrics["b2_skip_reason"] = "zero_gradient_trust_choked_but_pose_grad_exists"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_trust_choked_but_pose_grad_exists"
                        elif (
                            bool(getattr(pose_photo_data_loss, "requires_grad", False))
                            and not b2_pose_graph_connected
                            and not b2_any_grad_nonzero
                        ):
                            pose_metrics["b2_skip_reason"] = "zero_gradient_pose_graph_disconnected"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_pose_graph_disconnected"
                        elif b2_pose_graph_connected_but_tiny and not b2_any_grad_nonzero:
                            pose_metrics["b2_skip_reason"] = "zero_gradient_pose_graph_connected_but_tiny"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_pose_graph_connected_but_tiny"
                        elif trust_region_block:
                            pose_metrics["b2_skip_reason"] = "trust_region_block"
                            pose_metrics["b2_skip_reason_detailed"] = "trust_region_block_low_photo_signal"
                        elif (not (step_by_total_grad or step_by_data_grad)) and b2_trust_choked_photo_grad and b2_data_grad_nonzero:
                            pose_metrics["b2_skip_reason"] = "zero_gradient_trust_choked_but_data_grad_exists"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_trust_choked_but_data_grad_exists"
                        elif b2_data_grad_nonzero and not b2_data_grad_usable:
                            pose_metrics["b2_skip_reason"] = "zero_gradient_data_grad_below_floor"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_data_grad_below_floor"
                        elif (step_by_total_grad or step_by_data_grad) and not (b2_gate_step_ready or b2_photo_signal_strong or b2_force_small_step_ok):
                            pose_metrics["b2_skip_reason"] = "zero_gradient_total_grad_blocked_by_gate"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_total_grad_blocked_by_gate"
                        elif (not b2_any_grad_nonzero) and (not b2_photo_signal_ok):
                            pose_metrics["b2_skip_reason"] = "zero_gradient_true_no_signal"
                            pose_metrics["b2_skip_reason_detailed"] = "zero_gradient_true_no_signal"
                        elif not b2_any_grad_nonzero:
                            zero_detail = (
                                "zero_gradient_low_mask_low_patch"
                                if b2_low_info_zero_grad
                                else (
                                    "zero_gradient_unobservable_patch_signal"
                                    if not b2_observable_patch_ok
                                    else "zero_gradient_no_pose_path_despite_photo_signal"
                                )
                            )
                            pose_metrics["b2_skip_reason"] = zero_detail
                            pose_metrics["b2_skip_reason_detailed"] = zero_detail
                        elif b2_photo_signal_ok and not b2_photo_signal_strong:
                            pose_metrics["b2_skip_reason"] = "signal_ok_but_weak_mask"
                            pose_metrics["b2_skip_reason_detailed"] = "signal_ok_but_mask_patch_or_photo_below_threshold"
                        elif not b2_gate_step_ready:
                            if float(pose_metrics.get("b2_gate_optimization_ready", 0.0)) <= 0.5:
                                pose_metrics["b2_skip_reason"] = "optimization_not_ready"
                                pose_metrics["b2_skip_reason_detailed"] = "data_ready_but_optimization_gate_wait"
                            else:
                                pose_metrics["b2_skip_reason"] = "quality_not_ready"
                                pose_metrics["b2_skip_reason_detailed"] = "data_ready_but_quality_or_drift_gate_wait"
                        else:
                            pose_metrics["b2_skip_reason"] = "gradient_too_small"
                            pose_metrics["b2_skip_reason_detailed"] = "gradient_below_floor_without_force_small_step"
                else:
                    if not pose_photo_is_finite:
                        pose_metrics["b2_skip_reason"] = "nonfinite"
                    elif float(pose_photo_data_loss.detach().item()) <= 0.0:
                        pose_metrics["b2_skip_reason"] = str(pose_photo_metrics.get("pose_photo_skip_reason", "mask_empty"))
                    else:
                        pose_metrics["b2_skip_reason"] = "quality_below_threshold"
                pose_metrics["b2_skipped_nonfinite"] = 0.0 if pose_photo_is_finite else 1.0
                pose_metrics["nonfinite_pose_count"] = float(pose_metrics.get("nonfinite_pose_count", 0.0)) + (
                    1.0 if (pose_photo_loss_had_nonfinite or pose_photo_data_had_nonfinite) else 0.0
                )
                pose_metrics["b2_executed"] = 1.0 if b2_success else 0.0
                if b2_success:
                    pose_runtime_state["b2_update_count"] = int(pose_runtime_state.get("b2_update_count", 0)) + 1
                    pose_runtime_state["last_b2_success_iter"] = int(iteration)
                    b2_camera_success = pose_runtime_state.setdefault("b2_camera_success", {})
                    b2_camera_success[current_pose_camera_key] = int(b2_camera_success.get(current_pose_camera_key, 0)) + 1
                pose_optimizer.zero_grad(set_to_none=True)
                if bool(pose_metrics.get("b2_fullframe_stress_enabled", 0.0) > 0.5):
                    scene.gaussians.optimizer.zero_grad(set_to_none=True)
                pose_metrics["reset_nonfinite_after_b2"] = float(_reset_camera_pose_if_nonfinite(viewpoint_cam))
                if hasattr(viewpoint_cam, "refresh_pose_matrices"):
                    viewpoint_cam.refresh_pose_matrices()
                pose_metrics["b2_data_loss"] = float(pose_photo_data_loss_safe_for_log)
                pose_metrics["b2_total_loss"] = float(pose_photo_loss_safe_for_log)
                pose_metrics["b2_data_loss_had_nonfinite"] = 1.0 if pose_photo_data_had_nonfinite else 0.0
                pose_metrics["b2_total_loss_had_nonfinite"] = 1.0 if pose_photo_loss_had_nonfinite else 0.0
                pose_metrics["b2_quality_score"] = float(b2_quality_score)
                pose_metrics["b2_trust_to_data_ratio"] = float(pose_b2_trust_ratio)
                for metric_name, metric_value in pose_photo_metrics.items():
                    pose_metrics[f"b2_{metric_name}"] = metric_value
                for metric_name, metric_value in pose_b2_trust_window.items():
                    pose_metrics[f"b2_{metric_name}"] = metric_value
                for metric_name, metric_value in pose_photo_trust_metrics.items():
                    pose_metrics[f"b2_{metric_name}"] = metric_value
                for metric_name, metric_value in pose_b2_grad_metrics.items():
                    pose_metrics[f"b2_{metric_name}"] = metric_value
                for metric_name, metric_value in pose_b2_clamp_metrics.items():
                    pose_metrics[f"b2_{metric_name}"] = metric_value
            else:
                pose_metrics["b2_skipped_nonfinite"] = 0.0
                pose_metrics["b2_gate_blocked"] = 1.0
                pose_metrics["reset_nonfinite_after_b2"] = 0.0
                pose_metrics["b2_data_loss"] = 0.0
                pose_metrics["b2_total_loss"] = 0.0
                pose_metrics["b2_data_loss_had_nonfinite"] = 0.0
                pose_metrics["b2_total_loss_had_nonfinite"] = 0.0
                pose_metrics["b2_skip_reason"] = (
                    pose_gate_metrics.get("pose_gate_b2_reason", "gate_blocked")
                    if not atlas_phase["enable_pose_b2"]
                    else "b2_not_due"
                )

            scene.set_pose_trainable(False)
        else:
            scene.set_pose_trainable(False)

        current_pose_delta = measure_pose_delta(viewpoint_cam)
        if pose_metrics is not None:
            pose_runtime_state["last_b1_skip_reason"] = str(pose_metrics.get("b1_skip_reason", "none"))
            pose_runtime_state["last_b2_skip_reason"] = str(pose_metrics.get("b2_skip_reason", "none"))
            if (
                float(pose_metrics.get("b1_optimizer_step_count", 0.0)) > 0.0
                or float(pose_metrics.get("b2_optimizer_step_count", 0.0)) > 0.0
            ):
                train_camera_centers_dirty = True
            pose_metrics["pose_b1_quality_breakdown"] = {
                "quality_score": float(pose_metrics.get("b1_quality_score", 0.0)),
                "count_quality": float(pose_metrics.get("b1_pose_geo_count_quality", 0.0)),
                "coverage_quality": float(pose_metrics.get("b1_pose_geo_coverage_quality", 0.0)),
                "residual_quality": float(pose_metrics.get("b1_pose_geo_residual_quality", 0.0)),
                "pre_median_px": float(pose_metrics.get("b1_pose_geo_pre_median_px", 0.0)),
                "post_median_px": float(pose_metrics.get("b1_pose_geo_post_median_px", 0.0)),
                "median_px_reduction": float(pose_metrics.get("b1_pose_geo_median_px_reduction", 0.0)),
                "residual_reduced": float(pose_metrics.get("b1_success_residual_reduced", 0.0)),
                "camera_success_criterion": float(pose_metrics.get("b1_camera_success_criterion", 0.0)),
                "no_improve_streak": float(pose_metrics.get("b1_no_improve_streak", 0.0)),
                "atlas_native_ratio": float(pose_metrics.get("b1_pose_geo_atlas_native_selected_ratio", pose_metrics.get("b1_pose_geo_atlas_native_ratio", 0.0))),
                "fallback_ratio": float(pose_metrics.get("b1_pose_geo_fallback_selected_ratio", pose_metrics.get("b1_pose_geo_fallback_ratio", 0.0))),
                "skip_reason": str(pose_metrics.get("b1_skip_reason", "none")),
            }
            pose_metrics["pose_b2_skip_reason_breakdown"] = _reason_breakdown(pose_metrics.get("b2_skip_reason", "none"))
        if hasattr(viewpoint_cam, "record_pose_runtime"):
            viewpoint_cam.record_pose_runtime(
                b1_skip_reason=str(pose_metrics.get("b1_skip_reason", "none")) if pose_metrics is not None else "none",
                b2_skip_reason=str(pose_metrics.get("b2_skip_reason", "none")) if pose_metrics is not None else "none",
                trust_translation_limit=float(pose_metrics.get("trust_translation_limit", base_pose_translation_limit)) if pose_metrics is not None else base_pose_translation_limit,
                trust_rotation_limit_degrees=float(pose_metrics.get("trust_rotation_limit_degrees", base_pose_rotation_limit_degrees)) if pose_metrics is not None else base_pose_rotation_limit_degrees,
                last_stage="b2" if pose_metrics is not None and float(pose_metrics.get("b2_executed", 0.0)) > 0.5 else ("b1" if pose_metrics is not None and float(pose_metrics.get("b1_executed", 0.0)) > 0.5 else "skip"),
                last_iteration=int(iteration),
            )
        exposure_step_enabled, exposure_step_metrics = _should_step_exposure(
            has_atlas_bindings=scene.gaussians.has_atlas_bindings,
            disable_pose_refine=disable_pose_refine,
            main_phase_ready=main_phase_ready,
            pose_runtime_state=pose_runtime_state,
            render_loss_value=float(base_render_loss_safe_for_log),
            current_pose_delta=current_pose_delta,
            opt=opt,
        )
        pose_runtime_state["last_exposure_reason"] = str(exposure_step_metrics.get("exposure_step_reason", "default_allow"))
        if pose_metrics is not None:
            pose_metrics.update(exposure_step_metrics)
        if main_phase_ready and pose_optimizer is not None:
            _update_pose_runtime_aggregates(pose_runtime_state, pose_metrics, current_pose_delta)
        if pose_controller_start is not None:
            _record_controller_ms(controller_timing_metrics, "pose_refine", pose_controller_start)
        if main_phase_ready and pose_optimizer is not None:
            append_pose_refinement_log(
                scene.model_path,
                {
                    "iteration": int(iteration),
                    "camera_uid": int(getattr(viewpoint_cam, "uid", -1)),
                    "camera_name": str(getattr(viewpoint_cam, "image_name", getattr(viewpoint_cam, "uid", "unknown"))),
                    "pose_update_due": 1.0 if pose_update_due else 0.0,
                    "render_loss": float(base_render_loss_safe_for_log),
                    "pose_translation_norm": float(current_pose_delta.get("translation_norm", 0.0)),
                    "pose_rotation_degrees": float(current_pose_delta.get("rotation_degrees", 0.0)),
                    "freeze_reason": str(pose_gate_metrics.get("pose_freeze_reason", "none")),
                    "gate_b1_reason": str(pose_gate_metrics.get("pose_gate_b1_reason", "none")),
                    "gate_b2_reason": str(pose_gate_metrics.get("pose_gate_b2_reason", "none")),
                    "corr_reason": str(pose_corr_metrics.get("pose_corr_reason", "missing_correspondence")),
                    "corr_trustworthy": float(pose_corr_metrics.get("pose_corr_trustworthy", 0.0)),
                    "corr_ready": float(pose_corr_metrics.get("pose_corr_ready", 0.0)),
                    "corr_bootstrap_ready": float(pose_corr_metrics.get("pose_corr_bootstrap_ready", 0.0)),
                    "b1_gate_open": float(pose_metrics.get("b1_gate_open", pose_metrics.get("b1_gate_enabled", 0.0))) if pose_metrics is not None else 0.0,
                    "b2_gate_open": float(pose_metrics.get("b2_gate_open", pose_metrics.get("b2_gate_enabled", 0.0))) if pose_metrics is not None else 0.0,
                    "b1_gate_enabled": float(pose_metrics.get("b1_gate_enabled", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_gate_enabled": float(pose_metrics.get("b2_gate_enabled", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_attempt_count": float(pose_metrics.get("b1_attempt_count", pose_metrics.get("b1_attempted", 0.0))) if pose_metrics is not None else 0.0,
                    "b1_execute_count": float(pose_metrics.get("b1_execute_count", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_optimizer_step_count": float(pose_metrics.get("b1_optimizer_step_count", pose_metrics.get("b1_executed", 0.0))) if pose_metrics is not None else 0.0,
                    "b2_attempt_count": float(pose_metrics.get("b2_attempt_count", pose_metrics.get("b2_attempted", 0.0))) if pose_metrics is not None else 0.0,
                    "b2_execute_count": float(pose_metrics.get("b2_execute_count", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_optimizer_step_count": float(pose_metrics.get("b2_optimizer_step_count", pose_metrics.get("b2_executed", 0.0))) if pose_metrics is not None else 0.0,
                    "b2_mode": str(pose_metrics.get("b2_mode", "none")) if pose_metrics is not None else "none",
                    "pose_fullframe_l1": float(pose_metrics.get("b2_pose_fullframe_l1", 0.0)) if pose_metrics is not None else 0.0,
                    "pose_fullframe_ssim": float(pose_metrics.get("b2_pose_fullframe_ssim", 0.0)) if pose_metrics is not None else 0.0,
                    "pose_fullframe_gradient": float(pose_metrics.get("b2_pose_fullframe_gradient", 0.0)) if pose_metrics is not None else 0.0,
                    "pose_fullframe_total": float(pose_metrics.get("b2_pose_fullframe_total", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_history_ready": float(pose_metrics.get("b2_history_ready", pose_gate_metrics.get("pose_b2_gate_history_ready", 0.0))) if pose_metrics is not None else 0.0,
                    "b2_bootstrap_open": float(pose_metrics.get("b2_bootstrap_open", pose_gate_metrics.get("pose_gate_b2_bootstrap_open", 0.0))) if pose_metrics is not None else 0.0,
                    "b2_low_frequency_due": float(pose_metrics.get("b2_low_frequency_due", pose_gate_metrics.get("pose_gate_b2_low_frequency_due", 0.0))) if pose_metrics is not None else 0.0,
                    "b1_attempted": float(pose_metrics.get("b1_attempted", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_executed": float(pose_metrics.get("b1_executed", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_attempted": float(pose_metrics.get("b2_attempted", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_executed": float(pose_metrics.get("b2_executed", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_skip_reason": str(pose_metrics.get("b1_skip_reason", "none")) if pose_metrics is not None else "none",
                    "b2_skip_reason": str(pose_metrics.get("b2_skip_reason", "none")) if pose_metrics is not None else "none",
                    "b1_quality_score": float(pose_metrics.get("b1_quality_score", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_quality_score": float(pose_metrics.get("b2_quality_score", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_geo_pre_median_px": float(pose_metrics.get("b1_pose_geo_pre_median_px", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_geo_post_median_px": float(pose_metrics.get("b1_pose_geo_post_median_px", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_geo_median_px_reduction": float(pose_metrics.get("b1_pose_geo_median_px_reduction", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_success_residual_reduced": float(pose_metrics.get("b1_success_residual_reduced", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_camera_success_criterion": float(pose_metrics.get("b1_camera_success_criterion", 0.0)) if pose_metrics is not None else 0.0,
                    "b1_no_improve_streak": float(pose_metrics.get("b1_no_improve_streak", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_photo_corridor_open": float(pose_metrics.get("b2_photo_corridor_open", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_photo_corridor_scene_signal": float(pose_metrics.get("b2_photo_corridor_scene_signal", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_photo_corridor_support_ready": float(pose_metrics.get("b2_photo_corridor_support_ready", 0.0)) if pose_metrics is not None else 0.0,
                    "b2_corridor_step_ok": float(pose_metrics.get("b2_corridor_step_ok", 0.0)) if pose_metrics is not None else 0.0,
                    "trust_translation_limit": float(pose_metrics.get("trust_translation_limit", base_pose_translation_limit)) if pose_metrics is not None else base_pose_translation_limit,
                    "trust_rotation_limit_degrees": float(pose_metrics.get("trust_rotation_limit_degrees", base_pose_rotation_limit_degrees)) if pose_metrics is not None else base_pose_rotation_limit_degrees,
                    "exposure_step_enabled": float(exposure_step_metrics.get("exposure_step_enabled", 0.0)),
                    "exposure_step_reason": str(exposure_step_metrics.get("exposure_step_reason", "default_allow")),
                },
            )
        if iteration < opt.iterations:
            if exposure_step_enabled:
                gaussians.exposure_optimizer.step()
            gaussians.exposure_optimizer.zero_grad(set_to_none = True)
        iter_end.record()

        with torch.no_grad():
            if atlas_runtime_metrics is not None:
                atlas_runtime_metrics["phase_pose_translation_norm"] = float(current_pose_delta["translation_norm"])
                atlas_runtime_metrics["phase_pose_rotation_degrees"] = float(current_pose_delta["rotation_degrees"])
            # Progress bar
            if _is_finite_scalar(loss):
                ema_loss_for_log = 0.4 * float(loss.detach().item()) + 0.6 * ema_loss_for_log
            if _is_finite_scalar(Ll1depth):
                ema_Ll1depth_for_log = 0.4 * float(Ll1depth) + 0.6 * ema_Ll1depth_for_log

            reattach_success_count = 0.0
            reattach_fail_count = 0.0
            atlas_runtime_metrics, atlas_state_metrics, densify_metrics, atlas_gc_metrics, gc_pruned_this_iter = _run_densify_prune_gc_controller(
                scene=scene,
                gaussians=gaussians,
                viewpoint_cam=viewpoint_cam,
                iteration=int(iteration),
                opt=opt,
                dataset=dataset,
                atlas_phase=atlas_phase,
                atlas_runtime_metrics=atlas_runtime_metrics,
                atlas_state_metrics=atlas_state_metrics,
                densify_metrics=densify_metrics,
                atlas_gc_metrics=atlas_gc_metrics,
                prune_controls=prune_controls,
                pose_runtime_state=pose_runtime_state,
                densify_runtime_state=densify_runtime_state,
                densify_radii=densify_radii,
                densify_visibility_filter=densify_visibility_filter,
                densify_viewspace_point_tensor=densify_viewspace_point_tensor,
                train_camera_centers=train_camera_centers,
                train_camera_centers_cache=train_camera_centers_cache,
                active_min_lifetime_iters=active_min_lifetime_iters,
                active_quota_ratio=active_quota_ratio,
                active_quota_min=active_quota_min,
                active_quota_max=active_quota_max,
                promote_to_active_threshold=promote_to_active_threshold,
                demote_to_passive_threshold=demote_to_passive_threshold,
                active_max_lifetime_iters=active_max_lifetime_iters,
                active_nonimprove_iters=active_nonimprove_iters,
                controller_timing_metrics=controller_timing_metrics,
                run_gc=True,
                run_densify=False,
                gc_pruned_this_iter=False,
                timing_name="gc",
            )
            if atlas_gc_metrics is not None:
                reattach_success_count = float(atlas_gc_metrics.get("reattach_success", 0.0))
                reattach_fail_count = float(atlas_gc_metrics.get("reattach_fail", 0.0))

            if tb_writer and atlas_metrics is not None:
                tb_writer.add_scalar(
                    "atlas_loss/total",
                    _safe_log_scalar(
                        atlas_metrics.get("atlas_regularization_total_loss_safe_for_log", atlas_loss)
                    )[0],
                    iteration,
                )
                tb_writer.add_scalar("atlas_loss/warmup", atlas_warmup, iteration)
                _tb_add_metric_group(tb_writer, "atlas_metrics", atlas_metrics, iteration)
            if tb_writer and atlas_kl_metrics is not None:
                tb_writer.add_scalar(
                    "atlas_kl/total",
                    _safe_log_scalar(atlas_kl_metrics.get("atlas_kl_total_loss_safe_for_log", atlas_kl_loss))[0],
                    iteration,
                )
                _tb_add_metric_group(tb_writer, "atlas_kl_metrics", atlas_kl_metrics, iteration)
            if tb_writer and atlas_mc_metrics is not None:
                _tb_add_metric_group(tb_writer, "atlas_mc", atlas_mc_metrics, iteration)
            if tb_writer and atlas_runtime_metrics is not None:
                _tb_add_metric_group(tb_writer, "atlas_runtime", atlas_runtime_metrics, iteration)
            if tb_writer and atlas_refresh_metrics is not None:
                _tb_add_metric_group(tb_writer, "atlas_refresh", atlas_refresh_metrics, iteration)
            if tb_writer and atlas_slab_metrics is not None:
                tb_writer.add_scalar(
                    "atlas_slab/total",
                    _safe_log_scalar(atlas_slab_metrics.get("atlas_slab_total_loss_safe_for_log", atlas_slab_loss))[0],
                    iteration,
                )
                _tb_add_metric_group(tb_writer, "atlas_slab", atlas_slab_metrics, iteration)
            if tb_writer and atlas_gc_metrics is not None:
                _tb_add_metric_group(tb_writer, "atlas_gc", atlas_gc_metrics, iteration)
            if tb_writer and pose_metrics is not None:
                _tb_add_metric_group(tb_writer, "pose_metrics", pose_metrics, iteration)

            if iteration % progress_update_interval == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(progress_update_interval)
            if iteration == opt.iterations:
                remaining = total_training_steps % progress_update_interval
                if remaining != 0:
                    progress_bar.update(remaining)
                progress_bar.close()

            # Log and save
            elapsed_ms = iter_start.elapsed_time(iter_end)
            validation_summary = training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                elapsed_ms,
                testing_iterations,
                scene,
                render,
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                dataset.train_test_exp,
                opt=opt,
            )
            if validation_summary:
                latest_validation_summary = validation_summary
                densify_runtime_state = _update_densify_runtime_state(densify_runtime_state, validation_summary)
            append_quality_metrics_log(
                scene.model_path,
                iteration,
                validation_summary,
                extra_metrics=(
                    _build_quality_patch_metrics(
                        atlas_runtime_metrics=atlas_runtime_metrics,
                        atlas_refresh_metrics=atlas_refresh_metrics,
                        atlas_state_metrics=atlas_state_metrics,
                        densify_metrics=densify_metrics,
                        atlas_gc_metrics=atlas_gc_metrics,
                        pose_metrics=pose_metrics,
                    )
                    if validation_summary
                    else None
                ),
            )
            should_log_iteration = (
                iteration == start_iteration
                or iteration % progress_update_interval == 0
                or iteration in testing_iterations
                or iteration == opt.iterations
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification / pruning / GC
            atlas_runtime_metrics, atlas_state_metrics, densify_metrics, atlas_gc_metrics, gc_pruned_this_iter = _run_densify_prune_gc_controller(
                scene=scene,
                gaussians=gaussians,
                viewpoint_cam=viewpoint_cam,
                iteration=int(iteration),
                opt=opt,
                dataset=dataset,
                atlas_phase=atlas_phase,
                atlas_runtime_metrics=atlas_runtime_metrics,
                atlas_state_metrics=atlas_state_metrics,
                densify_metrics=densify_metrics,
                atlas_gc_metrics=atlas_gc_metrics,
                prune_controls=prune_controls,
                pose_runtime_state=pose_runtime_state,
                densify_runtime_state=densify_runtime_state,
                densify_radii=densify_radii,
                densify_visibility_filter=densify_visibility_filter,
                densify_viewspace_point_tensor=densify_viewspace_point_tensor,
                train_camera_centers=train_camera_centers,
                train_camera_centers_cache=train_camera_centers_cache,
                active_min_lifetime_iters=active_min_lifetime_iters,
                active_quota_ratio=active_quota_ratio,
                active_quota_min=active_quota_min,
                active_quota_max=active_quota_max,
                promote_to_active_threshold=promote_to_active_threshold,
                demote_to_passive_threshold=demote_to_passive_threshold,
                active_max_lifetime_iters=active_max_lifetime_iters,
                active_nonimprove_iters=active_nonimprove_iters,
                controller_timing_metrics=controller_timing_metrics,
                run_gc=False,
                run_densify=True,
                gc_pruned_this_iter=gc_pruned_this_iter,
                timing_name="densify_prune_gc",
            )
            if atlas_gc_metrics is not None:
                reattach_success_count = float(atlas_gc_metrics.get("reattach_success", 0.0))
                reattach_fail_count = float(atlas_gc_metrics.get("reattach_fail", 0.0))

            if atlas_gc_metrics is not None:
                densify_prune_after_gc = float(densify_metrics.get("prune_after_gc", 0.0))
                atlas_gc_metrics["densify_prune_after_gc"] = densify_prune_after_gc
                atlas_gc_metrics["total_prune_after_gc"] = float(atlas_gc_metrics.get("prune_after_gc", 0.0)) + densify_prune_after_gc
                if tb_writer:
                    tb_writer.add_scalar("atlas_gc/prune_after_gc", atlas_gc_metrics["total_prune_after_gc"], iteration)

            if controller_timing_metrics:
                if atlas_runtime_metrics is None:
                    atlas_runtime_metrics = {}
                atlas_runtime_metrics.update(controller_timing_metrics)

            if should_log_iteration:
                atlas_reliability_summary = scene.gaussians.summarize_atlas_reliability_state() if scene.gaussians.has_atlas_bindings else None
                if atlas_reliability_summary is not None:
                    atlas_reliability_summary.update(scene.gaussians.summarize_atlas_refresh_snapshot())
                atlas_state_summary = scene.gaussians.summarize_atlas_state_metrics() if scene.gaussians.has_atlas_bindings else None
                if atlas_state_summary is not None and atlas_state_metrics is not None:
                    atlas_state_summary.update(atlas_state_metrics)
                elif atlas_state_metrics is not None:
                    atlas_state_summary = dict(atlas_state_metrics)
                pose_runtime_log_fields = _build_pose_runtime_log_fields(pose_runtime_state)
                total_loss_safe_for_log, total_loss_had_nonfinite = _safe_log_scalar(loss)
                l1_loss_safe_for_log, l1_loss_had_nonfinite = _safe_log_scalar(Ll1)
                log_record = {
                    "iteration": int(iteration),
                    "elapsed_ms": float(elapsed_ms),
                    "l1_loss": float(l1_loss_safe_for_log),
                    "l1_loss_had_nonfinite": 1 if l1_loss_had_nonfinite else 0,
                    "total_loss": float(total_loss_safe_for_log),
                    "total_loss_had_nonfinite": 1 if total_loss_had_nonfinite else 0,
                    "depth_loss": float(Ll1depth),
                    "total_points": int(scene.gaussians.get_xyz.shape[0]),
                    "init_point_count": int(prune_controls["init_points"]),
                    "point_capacity_ratio": float(int(scene.gaussians.get_xyz.shape[0]) / max(prune_controls["init_points"], 1)),
                    "active_sh_degree": int(scene.gaussians.active_sh_degree),
                    "split_count": serialize_metric_value(densify_metrics.get("split_count", 0.0)),
                    "clone_count": serialize_metric_value(densify_metrics.get("clone_count", 0.0)),
                    "explore_clone_count": serialize_metric_value(densify_metrics.get("explore_clone_count", 0.0)),
                    "stable_split_count": serialize_metric_value(densify_metrics.get("stable_split_count", 0.0)),
                    "stable_clone_count": serialize_metric_value(densify_metrics.get("stable_clone_count", 0.0)),
                    "active_explore_clone_count": serialize_metric_value(densify_metrics.get("active_explore_clone_count", 0.0)),
                    "pruned_count": serialize_metric_value(densify_metrics.get("pruned_count", 0.0)),
                    "prune_after_gc": serialize_metric_value(densify_metrics.get("prune_after_gc", 0.0)),
                    "reattach_success_count": serialize_metric_value(reattach_success_count),
                    "reattach_fail_count": serialize_metric_value(reattach_fail_count),
                    "cuda_peak_vram_mb": float(_cuda_peak_vram_mb()),
                }
                log_record.update(
                    _build_required_training_log_fields(
                        has_atlas_bindings=bool(scene.gaussians.has_atlas_bindings),
                        atlas_phase=atlas_phase,
                        atlas_reliability_summary=atlas_reliability_summary,
                        atlas_state_metrics=atlas_state_summary,
                        atlas_runtime_metrics=atlas_runtime_metrics,
                        densify_metrics=densify_metrics,
                        atlas_gc_metrics=atlas_gc_metrics,
                        pose_metrics=pose_metrics,
                        atlas_metrics=atlas_metrics,
                        atlas_uncertainty_metrics=atlas_uncertainty_metrics,
                        atlas_slab_metrics=(
                            {
                                **(atlas_slab_metrics or {}),
                                "atlas_slab_total_loss": (
                                    float(_safe_log_scalar(atlas_slab_metrics.get("atlas_slab_total_loss_safe_for_log", atlas_slab_loss))[0])
                                    if atlas_slab_metrics is not None
                                    else 0.0
                                ),
                            }
                        ),
                        atlas_kl_metrics=atlas_kl_metrics,
                        atlas_refresh_done=bool(scene.gaussians.atlas_refresh_done) if scene.gaussians.has_atlas_bindings else False,
                    )
                )
                if depth_confidence_mean is not None:
                    log_record["depth_confidence_mean"] = float(depth_confidence_mean)
                if depth_weight_mean is not None:
                    log_record["depth_weight_mean"] = float(depth_weight_mean)
                if scene.gaussians.has_atlas_bindings:
                    atlas_init_summary = scene.gaussians.summarize_atlas_init_metrics()
                    log_record["atlas_refresh_done"] = int(scene.gaussians.atlas_refresh_done)
                    log_record["atlas_in_warmup"] = int(in_warmup)
                    log_record["warmup_only"] = int(warmup_only)
                    log_record["main_phase"] = int(main_phase)
                    log_record["refresh_pending"] = int(refresh_pending)
                    log_record["main_phase_ready"] = int(main_phase_ready)
                    log_record["atlas_enable_pose_b1"] = int(atlas_phase["enable_pose_b1"])
                    log_record["atlas_enable_pose_b2"] = int(atlas_phase["enable_pose_b2"])
                    log_record["atlas_enable_densify"] = int(atlas_phase["enable_densify"])
                    log_record["atlas_enable_prune"] = int(atlas_phase["enable_prune"])
                    log_record["atlas_enable_soft_prune"] = int(atlas_phase["enable_soft_prune"])
                    log_record["atlas_enable_gc"] = int(atlas_phase["enable_gc"])
                    log_record["atlas_enable_state_update"] = int(atlas_phase["enable_state_update"])
                    log_record["atlas_enable_mc"] = int(atlas_phase["enable_mc"])
                    log_record["atlas_enable_explore"] = int(atlas_phase["enable_explore"])
                    log_record["pose_refine_disabled"] = int(disable_pose_refine)
                    log_record["pose_refine_disabled_or_blocked_by_phase"] = int(
                        atlas_phase["pose_refine_disabled_or_blocked_by_phase"]
                    )
                    log_record["pose_translation_norm"] = float(current_pose_delta["translation_norm"])
                    log_record["pose_rotation_degrees"] = float(current_pose_delta["rotation_degrees"])
                    log_record["atlas_reliability_base_mean"] = serialize_metric_value(atlas_reliability_summary["atlas_reliability_base_mean"])
                    log_record["atlas_reliability_runtime_raw_mean"] = serialize_metric_value(
                        atlas_reliability_summary.get("atlas_reliability_runtime_raw_mean", 0.0)
                    )
                    log_record["atlas_reliability_runtime_mapped_mean"] = serialize_metric_value(
                        atlas_reliability_summary.get("atlas_reliability_runtime_mapped_mean", 0.0)
                    )
                    log_record["atlas_reliability_effective_mean"] = serialize_metric_value(
                        atlas_reliability_summary.get(
                            "atlas_reliability_effective_mean",
                            atlas_reliability_summary["atlas_reliability_runtime_mean"],
                        )
                    )
                    log_record["atlas_reliability_runtime_mean"] = serialize_metric_value(atlas_reliability_summary["atlas_reliability_runtime_mean"])
                    log_record["atlas_reliability_runtime_min"] = serialize_metric_value(atlas_reliability_summary["atlas_reliability_runtime_min"])
                    log_record["atlas_reliability_runtime_max"] = serialize_metric_value(atlas_reliability_summary["atlas_reliability_runtime_max"])
                    log_record["atlas_refresh_snapshot_ready"] = int(
                        bool(atlas_reliability_summary["atlas_refresh_snapshot_ready"])
                    )
                    log_record["atlas_refresh_snapshot_observed_ratio"] = serialize_metric_value(
                        atlas_reliability_summary["atlas_refresh_snapshot_observed_ratio"]
                    )
                    log_record["atlas_refresh_snapshot_observed_count"] = serialize_metric_value(
                        atlas_reliability_summary["atlas_refresh_snapshot_observed_count"]
                    )
                    log_record["atlas_refresh_snapshot_photo_ema_mean"] = serialize_metric_value(
                        atlas_reliability_summary["atlas_refresh_snapshot_photo_ema_mean"]
                    )
                    log_record["atlas_refresh_snapshot_visibility_ema_mean"] = serialize_metric_value(
                        atlas_reliability_summary["atlas_refresh_snapshot_visibility_ema_mean"]
                    )
                    log_record["atlas_refresh_snapshot_obs_quality_mean"] = serialize_metric_value(
                        atlas_reliability_summary["atlas_refresh_snapshot_obs_quality_mean"]
                    )
                    log_record["atlas_refresh_snapshot_obs_quality_max"] = serialize_metric_value(
                        atlas_reliability_summary["atlas_refresh_snapshot_obs_quality_max"]
                    )
                    for metric_name, metric_value in atlas_init_summary.items():
                        log_record[metric_name] = serialize_metric_value(metric_value)
                for metric_name, metric_value in ablation_manifest.items():
                    log_record[f"ablation_{metric_name}"] = serialize_metric_value(metric_value)
                if atlas_loss_schedule is not None:
                    for metric_name, metric_value in atlas_loss_schedule.items():
                        log_record[f"atlas_schedule_{metric_name}"] = serialize_metric_value(metric_value)
                if atlas_uncertainty_metrics is not None:
                    for metric_name, metric_value in atlas_uncertainty_metrics.items():
                        log_record[f"atlas_uncertainty_{metric_name}"] = serialize_metric_value(metric_value)
                if atlas_metrics is not None:
                    log_record["atlas_total_loss"] = float(
                        _safe_log_scalar(atlas_metrics.get("atlas_regularization_total_loss_safe_for_log", atlas_loss))[0]
                    )
                    log_record["atlas_warmup"] = float(atlas_warmup)
                    for metric_name, metric_value in atlas_metrics.items():
                        log_record[metric_name] = serialize_metric_value(metric_value)
                if atlas_kl_metrics is not None:
                    log_record["atlas_kl_total_loss"] = float(
                        _safe_log_scalar(atlas_kl_metrics.get("atlas_kl_total_loss_safe_for_log", atlas_kl_loss))[0]
                    )
                    for metric_name, metric_value in atlas_kl_metrics.items():
                        log_record[metric_name] = serialize_metric_value(metric_value)
                if atlas_mc_metrics is not None:
                    for metric_name, metric_value in atlas_mc_metrics.items():
                        log_record[metric_name] = serialize_metric_value(metric_value)
                if atlas_state_summary is not None:
                    for metric_name, metric_value in atlas_state_summary.items():
                        log_record[f"atlas_state_{metric_name}"] = serialize_metric_value(metric_value)
                if atlas_runtime_metrics is not None:
                    for metric_name, metric_value in atlas_runtime_metrics.items():
                        log_record[f"runtime_{metric_name}"] = serialize_metric_value(metric_value)
                if atlas_refresh_metrics is not None:
                    for metric_name, metric_value in atlas_refresh_metrics.items():
                        log_record[f"refresh_{metric_name}"] = serialize_metric_value(metric_value)
                if atlas_slab_metrics is not None:
                    log_record["atlas_slab_total_loss"] = float(
                        _safe_log_scalar(atlas_slab_metrics.get("atlas_slab_total_loss_safe_for_log", atlas_slab_loss))[0]
                    )
                    for metric_name, metric_value in atlas_slab_metrics.items():
                        log_record[metric_name] = serialize_metric_value(metric_value)
                if atlas_gc_metrics is not None:
                    for metric_name, metric_value in atlas_gc_metrics.items():
                        log_record[metric_name] = serialize_metric_value(metric_value)
                if pose_metrics is not None:
                    for metric_name, metric_value in pose_metrics.items():
                        log_record[f"pose_{metric_name}"] = serialize_metric_value(metric_value)
                for metric_name, metric_value in pose_runtime_log_fields.items():
                    log_record[metric_name] = serialize_metric_value(metric_value)
                if "train" in validation_summary:
                    for metric_name, metric_value in validation_summary["train"].items():
                        log_record[f"train_eval_{metric_name}"] = serialize_metric_value(metric_value)
                if "test" in validation_summary:
                    for metric_name, metric_value in validation_summary["test"].items():
                        log_record[f"test_eval_{metric_name}"] = serialize_metric_value(metric_value)
                append_training_log(scene.model_path, log_record)
                _write_training_summary(
                    scene.model_path,
                    {
                        "iteration": int(iteration),
                        "ablation_manifest": ablation_manifest,
                        "latest_log_record": log_record,
                        "latest_validation_summary": latest_validation_summary,
                        "atlas_phase": atlas_phase,
                        "atlas_reliability_summary": atlas_reliability_summary or {},
                        "atlas_state_summary": atlas_state_summary or {},
                        "atlas_runtime_metrics": atlas_runtime_metrics or {},
                        "atlas_regularization_metrics": atlas_metrics or {},
                        "atlas_slab_metrics": atlas_slab_metrics or {},
                        "atlas_gc_metrics": atlas_gc_metrics or {},
                        "pose_runtime_summary": pose_runtime_log_fields,
                    },
                )

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                save_training_checkpoint(
                    scene,
                    gaussians,
                    pose_optimizer,
                    iteration,
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                    run_args=run_args,
                )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    args_payload = dict(vars(args))
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**args_payload)))
    with open(os.path.join(args.model_path, "cfg_args.json"), "w", encoding="utf-8") as cfg_log_json_f:
        json.dump(args_payload, cfg_log_json_f, indent=2, sort_keys=True, default=str)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp, opt=None):
    validation_summary = {}
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', _safe_log_scalar(Ll1)[0], iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', _safe_log_scalar(loss)[0], iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_values = []
                dark_region_completeness = 0.0
                floater_proxy = 0.0
                pose_translation_proxy = 0.0
                pose_rotation_proxy = 0.0
                with torch.no_grad():
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        depth_image = render_pkg.get("depth")
                        if train_test_exp:
                            image = image[..., image.shape[-1] // 2:]
                            gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                            if depth_image is not None and depth_image.numel() > 0:
                                depth_image = depth_image[..., depth_image.shape[-1] // 2:]
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                        if FUSED_SSIM_AVAILABLE:
                            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                        else:
                            ssim_value = ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                        ssim_test += ssim_value.mean().double()
                        lpips_value = _compute_lpips_metric(image, gt_image)
                        if lpips_value is not None and math.isfinite(float(lpips_value)):
                            lpips_values.append(float(lpips_value))
                        proxy_metrics = compute_render_validation_proxies(
                            image,
                            gt_image,
                            rendered_invdepth=depth_image,
                        )
                        dark_region_completeness += float(proxy_metrics.get("dark_region_completeness_proxy", 0.0))
                        floater_proxy += float(proxy_metrics.get("floater_proxy", 0.0))
                        pose_delta = measure_pose_delta(viewpoint)
                        pose_translation_proxy += float(pose_delta.get("translation_norm", 0.0))
                        pose_rotation_proxy += float(pose_delta.get("rotation_degrees", 0.0))
                camera_count = float(len(config['cameras']))
                psnr_test /= camera_count
                l1_test /= camera_count
                ssim_test /= camera_count
                lpips_test = (sum(lpips_values) / float(len(lpips_values))) if lpips_values else None
                lpips_status = "ok" if lpips_test is not None else "unavailable"
                dark_region_completeness /= camera_count
                floater_proxy /= camera_count
                pose_translation_proxy /= camera_count
                pose_rotation_proxy /= camera_count
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                        iteration,
                        config['name'],
                        l1_test,
                        psnr_test,
                        ssim_test,
                        lpips_test if lpips_test is not None else lpips_status,
                    )
                )
                validation_summary[config['name']] = {
                    "l1": float(l1_test),
                    "psnr": float(psnr_test),
                    "ssim": float(ssim_test),
                    "lpips": float(lpips_test) if lpips_test is not None else None,
                    "lpips_status": lpips_status,
                    "lpips_available": 1.0 if lpips_test is not None else 0.0,
                    "floater_proxy": float(floater_proxy),
                    "dark_region_completeness_proxy": float(dark_region_completeness),
                    "pose_translation_proxy_mean": float(pose_translation_proxy),
                    "pose_rotation_proxy_mean": float(pose_rotation_proxy),
                    "gaussian_count": float(scene.gaussians.get_xyz.shape[0]),
                    "cuda_peak_vram_mb": float(_cuda_peak_vram_mb()),
                }
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    if lpips_test is not None:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips_available', 1.0 if lpips_test is not None else 0.0, iteration)
                    tb_writer.add_scalar(config['name'] + '/proxy - floater', floater_proxy, iteration)
                    tb_writer.add_scalar(config['name'] + '/proxy - dark_region_completeness', dark_region_completeness, iteration)
                    tb_writer.add_scalar(config['name'] + '/pose_proxy - translation', pose_translation_proxy, iteration)
                    tb_writer.add_scalar(config['name'] + '/pose_proxy - rotation_degrees', pose_rotation_proxy, iteration)
                    tb_writer.add_scalar(config['name'] + '/system - gaussian_count', scene.gaussians.get_xyz.shape[0], iteration)
                    tb_writer.add_scalar(config['name'] + '/system - cuda_peak_vram_mb', _cuda_peak_vram_mb(), iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('system/cuda_peak_vram_mb', _cuda_peak_vram_mb(), iteration)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return validation_summary

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--disable_pose_refine", dest="disable_pose_refine", action="store_true", default=False)
    parser.add_argument("--enable_pose_refine", dest="disable_pose_refine", action="store_false")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    opt_params = op.extract(args)
    opt_params.disable_pose_refine = bool(args.disable_pose_refine)
    training(
        lp.extract(args),
        opt_params,
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        run_args=args,
    )

    # All done
    print("\nTraining complete.")
