from __future__ import annotations

import torch

from scene.foundation_atlas import GAUSSIAN_STATE_UNSTABLE_ACTIVE


def _safe_normalize(vectors: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-6):
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    safe_fallback = fallback / torch.linalg.norm(fallback, dim=-1, keepdim=True).clamp_min(eps)
    normalized = vectors / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, safe_fallback)


def build_line_basis(directions: torch.Tensor):
    safe_dir = _safe_normalize(directions, fallback=directions.new_tensor([0.0, 0.0, 1.0]))
    helper = torch.zeros_like(safe_dir)
    helper[:, 2] = 1.0
    near_parallel = torch.abs((safe_dir * helper).sum(dim=1)) > 0.95
    if torch.any(near_parallel):
        helper[near_parallel] = torch.tensor([0.0, 1.0, 0.0], dtype=safe_dir.dtype, device=safe_dir.device)
    tangent_1 = _safe_normalize(torch.cross(helper, safe_dir, dim=1), fallback=helper)
    tangent_2 = _safe_normalize(torch.cross(safe_dir, tangent_1, dim=1), fallback=helper)
    return torch.stack((safe_dir, tangent_1, tangent_2), dim=2)


def _point_indices(point_selector: torch.Tensor):
    if point_selector.dtype == torch.bool:
        return torch.nonzero(point_selector, as_tuple=False).squeeze(-1)
    return point_selector.to(dtype=torch.long)


def _finite_rows(values: torch.Tensor):
    if values.numel() == 0:
        return torch.zeros((values.shape[0],), dtype=torch.bool, device=values.device)
    return torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)


def _resolve_reference_centers(
    gaussians,
    point_indices: torch.Tensor,
    camera_centers: torch.Tensor | None = None,
    fallback_camera_center: torch.Tensor | None = None,
):
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    xyz = gaussians.get_xyz.detach()[point_indices]
    atlas_pos = gaussians.get_gaussian_atlas_positions.detach()[point_indices]
    ref_centers = atlas_pos.clone()
    resolved_ref_camera = torch.full((point_indices.shape[0],), -1, dtype=torch.long, device=device)
    repaired_ref = torch.zeros((point_indices.shape[0],), dtype=torch.bool, device=device)
    valid_center = _finite_rows(ref_centers)

    if fallback_camera_center is not None:
        fallback_camera_center = fallback_camera_center.to(device=device, dtype=dtype).reshape(1, 3)
        ref_centers[:] = fallback_camera_center
        valid_center = torch.isfinite(fallback_camera_center).all().expand(point_indices.shape[0]).clone()

    if camera_centers is None or camera_centers.numel() == 0:
        return ref_centers, resolved_ref_camera, valid_center, repaired_ref

    camera_centers = camera_centers.to(device=device, dtype=dtype)
    camera_finite = _finite_rows(camera_centers)
    if not torch.any(camera_finite):
        return ref_centers, resolved_ref_camera, valid_center, repaired_ref

    ref_camera = gaussians.get_atlas_ref_camera.detach()[point_indices]
    valid_ref = (ref_camera >= 0) & (ref_camera < camera_centers.shape[0])
    valid_ref = valid_ref & camera_finite[ref_camera.clamp(0, max(camera_centers.shape[0] - 1, 0))]
    if torch.any(valid_ref):
        ref_centers[valid_ref] = camera_centers[ref_camera[valid_ref]]
        resolved_ref_camera[valid_ref] = ref_camera[valid_ref]

    invalid_ref = ~valid_ref
    if torch.any(invalid_ref):
        invalid_indices = torch.nonzero(invalid_ref, as_tuple=False).squeeze(-1)
        repaired_camera = torch.full((invalid_indices.shape[0],), -1, dtype=torch.long, device=device)

        view_weights = gaussians.get_gaussian_atlas_view_weights.detach()
        if (
            view_weights.ndim == 2
            and view_weights.shape[0] == gaussians.get_xyz.shape[0]
            and view_weights.shape[1] == camera_centers.shape[0]
        ):
            local_weights = view_weights[point_indices[invalid_indices]].to(device=device, dtype=dtype).clamp_min(0.0)
            local_weights[:, ~camera_finite] = 0.0
            best_weight, best_camera = torch.max(local_weights, dim=1)
            has_evidence = best_weight > 0.0
            if torch.any(has_evidence):
                repaired_camera[has_evidence] = best_camera[has_evidence]

        still_missing = repaired_camera < 0
        if torch.any(still_missing):
            finite_ids = torch.nonzero(camera_finite, as_tuple=False).squeeze(-1)
            finite_centers = camera_centers[finite_ids]
            nearest = torch.cdist(xyz[invalid_indices[still_missing]], finite_centers).argmin(dim=1)
            repaired_camera[still_missing] = finite_ids[nearest]

        repaired_valid = repaired_camera >= 0
        if torch.any(repaired_valid):
            target_indices = invalid_indices[repaired_valid]
            ref_centers[target_indices] = camera_centers[repaired_camera[repaired_valid]]
            resolved_ref_camera[target_indices] = repaired_camera[repaired_valid]
            repaired_ref[target_indices] = True

    valid_center = _finite_rows(ref_centers)
    return ref_centers, resolved_ref_camera, valid_center, repaired_ref


def compute_point_slab_bounds(
    gaussians,
    point_selector: torch.Tensor,
    camera_centers: torch.Tensor | None = None,
    fallback_camera_center: torch.Tensor | None = None,
    slab_radius_mult: float = 0.0,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
    detach_points: bool = False,
    require_valid_ref_camera: bool = False,
    min_reference_score: float = 0.0,
    repair_ref_camera: bool = False,
):
    if not gaussians.has_atlas_bindings:
        return None

    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    point_indices = _point_indices(point_selector)
    if point_indices.numel() == 0:
        return None

    xyz = gaussians.get_xyz.detach()[point_indices] if detach_points else gaussians.get_xyz[point_indices]
    anchor_ids = gaussians.get_atlas_node_ids.detach()[point_indices]
    anchor_radius = gaussians._atlas_radius[anchor_ids].detach().clamp_min(1e-4)
    anchor_points = gaussians.get_gaussian_atlas_positions.detach()[point_indices]
    ref_centers, resolved_ref_camera, valid_ref_center, repaired_ref = _resolve_reference_centers(
        gaussians,
        point_indices,
        camera_centers=camera_centers,
        fallback_camera_center=fallback_camera_center,
    )
    ref_score = gaussians.get_atlas_ref_score.detach()[point_indices].clamp(0.0, 1.0)
    repaired_score = torch.where(repaired_ref, torch.full_like(ref_score, 0.5), ref_score)
    ref_score = torch.maximum(ref_score, repaired_score)
    reliability = gaussians.get_gaussian_atlas_reliability.detach()[point_indices].clamp(0.0, 1.0)
    support_score = gaussians._compute_support_consistency_score().detach()[point_indices].clamp(0.0, 1.0)
    photo_ema = gaussians._atlas_photo_ema.detach()[point_indices].clamp_min(0.0)
    high_residual_count = gaussians._atlas_high_residual_count.detach()[point_indices]
    atlas_state = gaussians.get_atlas_state.detach()[point_indices]
    valid_ref = valid_ref_center
    if require_valid_ref_camera:
        # A valid ray reference is a geometry requirement; ref_score is evidence,
        # not a hard posterior/uncertainty gate. Repaired cameras therefore stay usable.
        valid_ref = valid_ref & (resolved_ref_camera >= 0)
        if min_reference_score > 0.0:
            valid_ref = valid_ref & (ref_score >= float(min_reference_score))
        if not torch.any(valid_ref):
            return None
        point_indices = point_indices[valid_ref]
        xyz = xyz[valid_ref]
        anchor_ids = anchor_ids[valid_ref]
        anchor_radius = anchor_radius[valid_ref]
        anchor_points = anchor_points[valid_ref]
        ref_centers = ref_centers[valid_ref]
        resolved_ref_camera = resolved_ref_camera[valid_ref]
        repaired_ref = repaired_ref[valid_ref]
        ref_score = ref_score[valid_ref]
        reliability = reliability[valid_ref]
        support_score = support_score[valid_ref]
        photo_ema = photo_ema[valid_ref]
        high_residual_count = high_residual_count[valid_ref]
        atlas_state = atlas_state[valid_ref]

    anchor_dirs = anchor_points - ref_centers
    basis_dirs = gaussians.get_gaussian_atlas_basis.detach()[point_indices, :, 0]
    anchor_dir_norm = torch.linalg.norm(anchor_dirs, dim=1, keepdim=True)
    fallback_dirs = torch.where(anchor_dir_norm > 1e-6, anchor_dirs, basis_dirs)
    ray_dirs = _safe_normalize(xyz - ref_centers, fallback=fallback_dirs)

    tau = ((xyz - ref_centers) * ray_dirs).sum(dim=1)
    anchor_tau = ((anchor_points - ref_centers) * ray_dirs).sum(dim=1)
    base_mult = max(float(slab_radius_mult), 0.25)
    adaptive_mult = torch.full_like(anchor_radius, base_mult)
    low_reliability_support = (reliability < 0.35) & (support_score < 0.35)
    weak_ref = ref_score < max(float(min_reference_score), 0.08)
    active_rescue = (
        (atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
        & (ref_score >= max(float(min_reference_score), 0.05))
        & ((high_residual_count >= 2) | (photo_ema >= 0.05))
    )
    adaptive_mult = torch.where(low_reliability_support, adaptive_mult * 0.60, adaptive_mult)
    adaptive_mult = torch.where(weak_ref, adaptive_mult * 0.75, adaptive_mult)
    depth_delta_ratio = (tau - anchor_tau).abs() / anchor_radius.clamp_min(1e-6)
    far_depth = depth_delta_ratio > torch.maximum(adaptive_mult * 1.25, torch.full_like(adaptive_mult, 1.25))
    background_like_depth = far_depth & (weak_ref | low_reliability_support) & (~active_rescue)
    adaptive_mult = torch.where(background_like_depth, adaptive_mult * 0.55, adaptive_mult)
    adaptive_mult = adaptive_mult.clamp(min=0.25, max=max(base_mult * 1.25, 0.25))
    min_half_width = anchor_radius * adaptive_mult
    fallback_low = anchor_tau - min_half_width
    fallback_high = anchor_tau + min_half_width

    tau_min = fallback_low.clone()
    tau_max = fallback_high.clone()
    fallback_slab = torch.ones((point_indices.shape[0],), dtype=torch.bool, device=device)
    neighbor_count = torch.zeros((point_indices.shape[0],), dtype=torch.long, device=device)
    neighbor_stable = torch.zeros((point_indices.shape[0],), dtype=torch.bool, device=device)
    if (
        getattr(gaussians, "_atlas_neighbor_indices", None) is not None
        and gaussians._atlas_neighbor_indices.numel() > 0
        and gaussians._atlas_neighbor_indices.ndim == 2
        and gaussians._atlas_neighbor_indices.shape[0] == gaussians._atlas_positions.shape[0]
    ):
        neighbor_ids = gaussians._atlas_neighbor_indices[anchor_ids].detach().clamp(0, max(gaussians._atlas_positions.shape[0] - 1, 0))
        neighbor_points = gaussians._atlas_positions[neighbor_ids].detach()
        slab_depth = ((neighbor_points - ref_centers[:, None, :]) * ray_dirs[:, None, :]).sum(dim=2)
        finite_depth = torch.isfinite(slab_depth)
        neighbor_count = finite_depth.sum(dim=1)
        if torch.any(neighbor_count >= 2):
            safe_depth = torch.where(finite_depth, slab_depth, anchor_tau[:, None])
            q_low = torch.quantile(safe_depth, float(low_quantile), dim=1)
            q_high = torch.quantile(safe_depth, float(high_quantile), dim=1)
            q_span = (q_high - q_low).clamp_min(0.0)
            min_quantile_span = torch.maximum(
                torch.minimum(2.0 * min_half_width, anchor_radius * 0.50),
                anchor_radius * 0.10,
            )
            neighbor_stable = (
                torch.isfinite(q_span)
                & (q_span <= anchor_radius * max(base_mult * 3.0, 0.75))
            )
            rescue_widen = active_rescue & neighbor_stable & (neighbor_count >= 3) & (ref_score >= 0.20)
            half_cap = min_half_width * torch.where(rescue_widen, torch.full_like(min_half_width, 1.20), torch.ones_like(min_half_width))
            half_cap = torch.where(background_like_depth, torch.minimum(half_cap, anchor_radius * max(base_mult * 0.45, 0.25)), half_cap)
            clamped_low = torch.maximum(q_low, anchor_tau - half_cap)
            clamped_high = torch.minimum(q_high, anchor_tau + half_cap)
            quantile_valid = (
                (neighbor_count >= 2)
                & torch.isfinite(clamped_low)
                & torch.isfinite(clamped_high)
                & (q_span >= min_quantile_span)
                & ((clamped_high - clamped_low) > 1e-6)
            )
            tau_min = torch.where(quantile_valid, clamped_low, tau_min)
            tau_max = torch.where(quantile_valid, clamped_high, tau_max)
            fallback_slab = ~quantile_valid

    if torch.any(active_rescue):
        tau_min = torch.where(active_rescue, torch.minimum(tau_min, tau), tau_min)
        tau_max = torch.where(active_rescue, torch.maximum(tau_max, tau), tau_max)

    finite_slab = (
        torch.isfinite(xyz).all(dim=1)
        & torch.isfinite(ref_centers).all(dim=1)
        & torch.isfinite(ray_dirs).all(dim=1)
        & torch.isfinite(tau)
        & torch.isfinite(tau_min)
        & torch.isfinite(tau_max)
        & ((tau_max - tau_min) > 1e-6)
    )
    if not torch.any(finite_slab):
        return None
    if not torch.all(finite_slab):
        point_indices = point_indices[finite_slab]
        ref_centers = ref_centers[finite_slab]
        ray_dirs = ray_dirs[finite_slab]
        tau = tau[finite_slab]
        tau_min = tau_min[finite_slab]
        tau_max = tau_max[finite_slab]
        anchor_ids = anchor_ids[finite_slab]
        anchor_radius = anchor_radius[finite_slab]
        ref_score = ref_score[finite_slab]
        resolved_ref_camera = resolved_ref_camera[finite_slab]
        repaired_ref = repaired_ref[finite_slab]
        fallback_slab = fallback_slab[finite_slab]
        neighbor_count = neighbor_count[finite_slab]
        adaptive_mult = adaptive_mult[finite_slab]
        depth_delta_ratio = depth_delta_ratio[finite_slab]
        background_like_depth = background_like_depth[finite_slab]
        active_rescue = active_rescue[finite_slab]
        neighbor_stable = neighbor_stable[finite_slab]

    if repair_ref_camera and resolved_ref_camera.numel() > 0:
        repairable = resolved_ref_camera >= 0
        if torch.any(repairable):
            with torch.no_grad():
                target_indices = point_indices[repairable]
                gaussians._atlas_ref_camera[target_indices] = resolved_ref_camera[repairable].to(
                    device=gaussians._atlas_ref_camera.device,
                    dtype=torch.long,
                )
                gaussians._atlas_ref_score[target_indices] = torch.maximum(
                    gaussians._atlas_ref_score[target_indices],
                    ref_score[repairable].to(device=gaussians._atlas_ref_score.device, dtype=gaussians._atlas_ref_score.dtype),
                )

    return {
        "point_indices": point_indices,
        "ref_centers": ref_centers,
        "ray_dirs": ray_dirs,
        "tau": tau,
        "tau_min": tau_min,
        "tau_max": tau_max,
        "anchor_ids": anchor_ids,
        "anchor_radius": anchor_radius,
        "ref_score": ref_score,
        "resolved_ref_camera": resolved_ref_camera,
        "repaired_ref_mask": repaired_ref,
        "fallback_slab_mask": fallback_slab,
        "finite_neighbor_count": neighbor_count,
        "adaptive_slab_mult": adaptive_mult,
        "depth_delta_ratio": depth_delta_ratio,
        "background_like_depth_mask": background_like_depth,
        "active_rescue_slab_mask": active_rescue,
        "neighbor_stable_mask": neighbor_stable,
    }


def compute_exploration_slab_loss(
    gaussians,
    camera_centers: torch.Tensor,
    weight: float,
    slab_radius_mult: float,
    low_quantile: float = 0.2,
    high_quantile: float = 0.8,
    enabled: bool = True,
):
    zero = gaussians.get_xyz.new_zeros(())
    metrics = {
        "atlas_slab_active_fraction": 0.0,
        "atlas_slab_active_count": 0.0,
        "atlas_slab_valid_count": 0.0,
        "atlas_slab_mean_penalty": 0.0,
        "atlas_slab_mean_overrun": 0.0,
        "atlas_slab_mean_span": 0.0,
        "atlas_slab_violation_count": 0.0,
        "atlas_slab_violation_ratio": 0.0,
        "atlas_slab_fallback_count": 0.0,
        "atlas_slab_ref_repair_count": 0.0,
    }
    if (not enabled) or weight <= 0.0 or not gaussians.has_atlas_bindings:
        return zero, metrics

    active_mask = gaussians.get_atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
    if not torch.any(active_mask):
        return zero, metrics

    slab = compute_point_slab_bounds(
        gaussians,
        active_mask,
        camera_centers=camera_centers,
        slab_radius_mult=slab_radius_mult,
        low_quantile=low_quantile,
        high_quantile=high_quantile,
        detach_points=False,
        require_valid_ref_camera=True,
        min_reference_score=0.05,
        repair_ref_camera=True,
    )
    if slab is None:
        return zero, metrics

    tau = slab["tau"]
    tau_min = slab["tau_min"]
    tau_max = slab["tau_max"]
    upper = torch.relu(tau - tau_max)
    lower = torch.relu(tau_min - tau)
    penalty = upper.square() + lower.square()
    violation_mask = (upper > 0.0) | (lower > 0.0)
    loss = float(weight) * penalty.mean()

    valid_count = int(tau.shape[0])
    metrics["atlas_slab_active_fraction"] = float(active_mask.float().mean().item())
    metrics["atlas_slab_active_count"] = float(int(active_mask.sum().item()))
    metrics["atlas_slab_valid_count"] = float(valid_count)
    metrics["atlas_slab_mean_penalty"] = float(penalty.mean().detach().item())
    metrics["atlas_slab_mean_overrun"] = float((upper + lower).mean().detach().item())
    metrics["atlas_slab_mean_span"] = float((tau_max - tau_min).mean().detach().item())
    metrics["atlas_slab_violation_count"] = float(int(violation_mask.sum().item()))
    metrics["atlas_slab_violation_ratio"] = float(violation_mask.float().mean().detach().item())
    metrics["atlas_slab_fallback_count"] = float(int(slab["fallback_slab_mask"].sum().item()))
    metrics["atlas_slab_ref_repair_count"] = float(int(slab["repaired_ref_mask"].sum().item()))
    return loss, metrics
