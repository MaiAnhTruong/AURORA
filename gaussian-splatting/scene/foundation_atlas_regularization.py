from __future__ import annotations

import math

import torch

from scene.foundation_atlas import (
    ATLAS_CLASS_EDGE,
    ATLAS_CLASS_SURFACE,
    GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
)
from scene.foundation_atlas_exploration import build_line_basis


def _quaternion_to_matrix(quaternions: torch.Tensor):
    quaternions = torch.nn.functional.normalize(quaternions, dim=1)
    w, x, y, z = quaternions.unbind(dim=1)

    rotation = torch.zeros((quaternions.shape[0], 3, 3), dtype=quaternions.dtype, device=quaternions.device)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - w * z)
    rotation[:, 0, 2] = 2 * (x * z + w * y)
    rotation[:, 1, 0] = 2 * (x * y + w * z)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - w * x)
    rotation[:, 2, 0] = 2 * (x * z - w * y)
    rotation[:, 2, 1] = 2 * (y * z + w * x)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotation


def _huber(values: torch.Tensor, delta: float):
    abs_values = values.abs()
    delta_t = torch.tensor(float(delta), dtype=values.dtype, device=values.device)
    quadratic = torch.minimum(abs_values, delta_t)
    linear = abs_values - quadratic
    return 0.5 * quadratic.square() + delta_t * linear


def _safe_metric_scalar(value, default: float = 0.0):
    if torch.is_tensor(value):
        if value.numel() == 0:
            return float(default), 0.0
        detached = value.detach()
        had_nonfinite = 0.0 if bool(torch.isfinite(detached).all().item()) else 1.0
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
        return float(default), 0.0
    if not math.isfinite(numeric):
        return float(default), 1.0
    return numeric, 0.0


def _nan_to_finite(tensor: torch.Tensor, value: float = 0.0):
    return torch.nan_to_num(
        tensor,
        nan=float(value),
        posinf=float(value),
        neginf=float(value),
    )


def _symmetrize(matrix: torch.Tensor):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _sanitize_psd_projector(projector: torch.Tensor):
    if projector.numel() == 0:
        return projector
    safe = _symmetrize(_nan_to_finite(projector, 0.0))
    eigvals, eigvecs = torch.linalg.eigh(safe)
    eigvals = eigvals.clamp(0.0, 1.0)
    sanitized = torch.bmm(eigvecs, torch.bmm(torch.diag_embed(eigvals), eigvecs.transpose(1, 2)))
    return _symmetrize(_nan_to_finite(sanitized, 0.0))


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, zero: torch.Tensor):
    if values.numel() == 0:
        return zero
    safe_weights = weights.to(device=values.device, dtype=values.dtype).clamp_min(0.0)
    positive = safe_weights > 1e-8
    if not torch.any(positive):
        return zero
    return (values[positive] * safe_weights[positive]).sum() / safe_weights[positive].sum().clamp_min(1e-8)


def _atlas_state_weight(atlas_state: torch.Tensor, passive_state_weight: float, active_state_weight: float):
    state_weight = torch.ones_like(atlas_state, dtype=torch.float32)
    state_weight[atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE] = float(passive_state_weight)
    state_weight[atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE] = float(active_state_weight)
    state_weight[atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING] = 0.0
    state_weight[atlas_state == GAUSSIAN_STATE_STABLE] = 1.0
    return state_weight.clamp_min(0.0)


def _sorted_gaussian_axes(gaussians):
    scales = gaussians.get_scaling
    order = torch.argsort(scales, dim=1, descending=True)
    rotation = _quaternion_to_matrix(gaussians.get_rotation)
    gather_index = order.unsqueeze(1).expand(-1, 3, -1)
    sorted_axes = torch.gather(rotation, 2, gather_index)
    sorted_scales = torch.gather(scales, 1, order)
    return sorted_axes, sorted_scales


def _compute_log_anisotropy_ratios(sorted_scales: torch.Tensor):
    safe_scales = sorted_scales.clamp_min(1e-8)
    return torch.stack(
        (
            torch.log(safe_scales[:, 0] / safe_scales[:, 1]),
            torch.log(safe_scales[:, 1] / safe_scales[:, 2]),
        ),
        dim=1,
    )


def _masked_metric_mean(values: torch.Tensor, mask: torch.Tensor):
    if values.numel() == 0 or mask.numel() == 0 or not torch.any(mask):
        return 0.0
    return float(values[mask].mean().detach().item())


def _resolve_fallback_support(gaussians, active_mask: torch.Tensor, fallback_mask: torch.Tensor, train_cameras):
    fallback_count = int(torch.count_nonzero(fallback_mask).item())
    if train_cameras is None or len(train_cameras) == 0 or not torch.any(fallback_mask):
        empty_support = gaussians.get_xyz.new_zeros((fallback_count, 3, 3))
        empty_basis = gaussians.get_xyz.new_zeros((fallback_count, 3, 3))
        empty_valid = torch.zeros((fallback_count,), dtype=torch.bool, device=gaussians.get_xyz.device)
        return empty_support, empty_basis, empty_valid

    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    camera_centers = []
    for camera in train_cameras:
        camera_center = getattr(camera, "camera_center", None)
        if camera_center is None:
            return (
                gaussians.get_xyz.new_zeros((fallback_count, 3, 3)),
                gaussians.get_xyz.new_zeros((fallback_count, 3, 3)),
                torch.zeros((fallback_count,), dtype=torch.bool, device=device),
            )
        camera_centers.append(camera_center.detach().to(device=device, dtype=dtype))
    camera_centers = torch.stack(camera_centers, dim=0)

    active_idx = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)
    fallback_idx = torch.nonzero(fallback_mask, as_tuple=False).squeeze(-1)
    ref_camera = gaussians.get_atlas_ref_camera.detach()[active_idx][fallback_idx].clone()
    valid_ref = (ref_camera >= 0) & (ref_camera < camera_centers.shape[0])
    view_weights = gaussians.get_gaussian_atlas_view_weights.detach()[active_idx][fallback_idx]
    if view_weights.ndim == 2 and view_weights.shape[1] == camera_centers.shape[0]:
        missing_ref = ~valid_ref
        if torch.any(missing_ref):
            ref_camera[missing_ref] = torch.argmax(view_weights[missing_ref], dim=1)
            valid_ref = (ref_camera >= 0) & (ref_camera < camera_centers.shape[0])

    fallback_support = gaussians.get_xyz.new_zeros((fallback_idx.shape[0], 3, 3))
    fallback_basis = gaussians.get_xyz.new_zeros((fallback_idx.shape[0], 3, 3))
    if not torch.any(valid_ref):
        return fallback_support, fallback_basis, valid_ref

    xyz = gaussians.get_xyz[active_idx][fallback_idx]
    ray_dir = xyz[valid_ref] - camera_centers[ref_camera[valid_ref]]
    ray_norm = torch.linalg.norm(ray_dir, dim=1)
    valid_dir = ray_norm > 1e-6
    if not torch.any(valid_dir):
        return fallback_support, fallback_basis, torch.zeros_like(valid_ref)

    safe_ray_dir = ray_dir[valid_dir] / ray_norm[valid_dir, None]
    support = torch.einsum("ni,nj->nij", safe_ray_dir, safe_ray_dir)
    basis = build_line_basis(safe_ray_dir)
    resolved_idx = torch.nonzero(valid_ref, as_tuple=False).squeeze(-1)[valid_dir]
    fallback_support[resolved_idx] = support
    fallback_basis[resolved_idx] = basis

    final_valid = torch.zeros_like(valid_ref)
    final_valid[resolved_idx] = True
    return fallback_support, fallback_basis, final_valid


def compute_atlas_regularization(
    gaussians,
    scene_extent: float,
    mean_weight: float,
    ori_weight: float,
    aniso_weight: float,
    huber_delta: float,
    train_cameras=None,
    mean_passive_state_weight: float | None = None,
    mean_active_state_weight: float | None = None,
    passive_state_weight: float = 0.35,
    active_state_weight: float = 0.0,
):
    zero = gaussians.get_xyz.new_zeros(())
    metrics = {
        "atlas_active_fraction": 0.0,
        "atlas_regularized_fraction": 0.0,
        "atlas_mean_loss": 0.0,
        "atlas_ori_loss": 0.0,
        "atlas_aniso_loss": 0.0,
        "atlas_mean_projected_drift": 0.0,
        "atlas_mean_projected_drift_regularized": 0.0,
        "atlas_mean_projected_drift_unresolved_active": 0.0,
        "atlas_mean_projected_energy": 0.0,
        "atlas_mean_total_loss": 0.0,
        "atlas_shape_loss": 0.0,
        "atlas_regularization_total_loss": 0.0,
        "atlas_fallback_fraction": 0.0,
        "atlas_active_ray_fraction": 0.0,
        "atlas_unresolved_active_fraction": 0.0,
        "atlas_ori_surface_loss": 0.0,
        "atlas_ori_edge_loss": 0.0,
        "atlas_ori_active_ray_loss": 0.0,
        "atlas_aniso_stable_loss": 0.0,
        "atlas_aniso_passive_loss": 0.0,
        "atlas_aniso_active_loss": 0.0,
        "atlas_mean_projected_drift_stable": 0.0,
        "atlas_mean_projected_drift_passive": 0.0,
        "atlas_mean_projected_drift_active": 0.0,
        "atlas_drift_hist_stable_low": 0.0,
        "atlas_drift_hist_stable_mid": 0.0,
        "atlas_drift_hist_stable_high": 0.0,
        "atlas_drift_hist_passive_low": 0.0,
        "atlas_drift_hist_passive_mid": 0.0,
        "atlas_drift_hist_passive_high": 0.0,
        "atlas_drift_hist_active_low": 0.0,
        "atlas_drift_hist_active_mid": 0.0,
        "atlas_drift_hist_active_high": 0.0,
        "atlas_regularization_total_loss_safe_for_log": 0.0,
        "atlas_regularization_total_loss_had_nonfinite": 0.0,
        "atlas_mean_projected_energy_safe_for_log": 0.0,
        "atlas_mean_projected_energy_had_nonfinite": 0.0,
        "nonfinite_projected_energy_count": 0.0,
    }

    if not gaussians.has_atlas_bindings:
        return zero, metrics

    atlas_state = gaussians.get_atlas_state
    atlas_class = gaussians.get_gaussian_atlas_class
    fallback_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
    surface_mask = atlas_class == ATLAS_CLASS_SURFACE
    edge_mask = atlas_class == ATLAS_CLASS_EDGE
    active_mask = atlas_state != GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING

    active_count = int(active_mask.sum().item())
    if active_count == 0:
        return zero, metrics

    xyz = gaussians.get_xyz[active_mask]
    atlas_positions = gaussians.get_gaussian_atlas_positions[active_mask]
    atlas_support = gaussians.get_gaussian_atlas_support[active_mask]
    atlas_basis = gaussians.get_gaussian_atlas_basis[active_mask]
    atlas_reliability = gaussians.get_gaussian_atlas_reliability.detach()[active_mask]
    atlas_anisotropy_ref = gaussians.get_gaussian_atlas_anisotropy_ref[active_mask]
    atlas_state_active = atlas_state[active_mask]
    active_fallback_mask = fallback_mask[active_mask]
    shape_state_weight = _atlas_state_weight(
        atlas_state_active,
        passive_state_weight=float(passive_state_weight),
        active_state_weight=float(active_state_weight),
    ).to(device=xyz.device, dtype=xyz.dtype)
    if mean_passive_state_weight is None:
        mean_passive_state_weight = passive_state_weight
    if mean_active_state_weight is None:
        mean_active_state_weight = active_state_weight
    mean_state_weight = _atlas_state_weight(
        atlas_state_active,
        passive_state_weight=float(mean_passive_state_weight),
        active_state_weight=float(mean_active_state_weight),
    ).to(device=xyz.device, dtype=xyz.dtype)
    mean_combined_weight = atlas_reliability * mean_state_weight
    clear_shape_class = surface_mask[active_mask] | edge_mask[active_mask] | (atlas_state_active == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
    shape_class_gate = torch.where(
        clear_shape_class,
        torch.ones_like(shape_state_weight),
        torch.full_like(shape_state_weight, 0.25),
    )
    shape_combined_weight = atlas_reliability * shape_state_weight * shape_class_gate

    effective_support = atlas_support.clone()
    effective_basis = atlas_basis.clone()
    fallback_support, fallback_basis, fallback_valid = _resolve_fallback_support(
        gaussians,
        active_mask=active_mask,
        fallback_mask=active_fallback_mask,
        train_cameras=train_cameras,
    )
    resolved_fallback_mask = torch.zeros_like(active_fallback_mask)
    resolved_fallback_mask[active_fallback_mask] = fallback_valid
    if torch.any(fallback_valid):
        effective_support[active_fallback_mask] = torch.where(
            fallback_valid[:, None, None],
            fallback_support,
            effective_support[active_fallback_mask],
        )
        effective_basis[active_fallback_mask] = torch.where(
            fallback_valid[:, None, None],
            fallback_basis,
            effective_basis[active_fallback_mask],
        )

    unresolved_active_mask = (atlas_state_active == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & (~resolved_fallback_mask)
    effective_support = _sanitize_psd_projector(effective_support)

    delta = xyz - atlas_positions
    projected_valid_mask = (
        torch.isfinite(delta).all(dim=1)
        & torch.isfinite(effective_support).reshape(effective_support.shape[0], -1).all(dim=1)
    )
    regularized_mask = (~unresolved_active_mask) & projected_valid_mask
    mean_regularized_weight = torch.where(regularized_mask, mean_combined_weight, torch.zeros_like(mean_combined_weight))
    shape_regularized_weight = torch.where(regularized_mask, shape_combined_weight, torch.zeros_like(shape_combined_weight))

    safe_delta = _nan_to_finite(delta, 0.0)
    projected_delta = torch.bmm(effective_support, safe_delta.unsqueeze(-1)).squeeze(-1)
    projected_energy = _nan_to_finite(torch.sum(projected_delta * safe_delta, dim=1), 0.0).clamp_min(0.0)
    projected_energy = torch.where(projected_valid_mask, projected_energy, torch.zeros_like(projected_energy))
    projected_drift = torch.where(
        projected_energy > 1e-12,
        torch.sqrt(projected_energy),
        torch.zeros_like(projected_energy),
    )
    mean_error = _huber(projected_drift, max(float(huber_delta) * float(scene_extent), 1e-5))
    mean_loss = _weighted_mean(mean_error, mean_regularized_weight, zero)

    sorted_axes, sorted_scales = _sorted_gaussian_axes(gaussians)
    sorted_axes = sorted_axes[active_mask]
    sorted_scales = sorted_scales[active_mask]

    ori_values = []
    ori_weights = []
    non_active_unstable = atlas_state_active != GAUSSIAN_STATE_UNSTABLE_ACTIVE
    active_surface = surface_mask[active_mask] & non_active_unstable
    active_edge = edge_mask[active_mask] & non_active_unstable
    surface_ori = zero
    edge_ori = zero
    active_ray_ori = zero
    if torch.any(active_surface):
        gaussian_normal = sorted_axes[active_surface, :, 2]
        atlas_normal = effective_basis[active_surface, :, 2]
        normal_misalignment = 1.0 - torch.abs((gaussian_normal * atlas_normal).sum(dim=1))
        gaussian_plane = torch.eye(3, dtype=gaussian_normal.dtype, device=gaussian_normal.device).unsqueeze(0) - torch.einsum(
            "ni,nj->nij", gaussian_normal, gaussian_normal
        )
        plane_misalignment = (gaussian_plane - effective_support[active_surface]).square().mean(dim=(1, 2))
        surface_ori_values = normal_misalignment + plane_misalignment
        ori_values.append(surface_ori_values)
        ori_weights.append(shape_regularized_weight[active_surface])
        surface_ori = _weighted_mean(surface_ori_values, shape_regularized_weight[active_surface], zero)

    if torch.any(active_edge):
        gaussian_edge = sorted_axes[active_edge, :, 0]
        atlas_edge = effective_basis[active_edge, :, 0]
        edge_misalignment = 1.0 - torch.abs((gaussian_edge * atlas_edge).sum(dim=1))
        edge_projector = torch.einsum("ni,nj->nij", gaussian_edge, gaussian_edge)
        projector_misalignment = (edge_projector - effective_support[active_edge]).square().mean(dim=(1, 2))
        edge_ori_values = edge_misalignment + projector_misalignment
        ori_values.append(edge_ori_values)
        ori_weights.append(shape_regularized_weight[active_edge])
        edge_ori = _weighted_mean(edge_ori_values, shape_regularized_weight[active_edge], zero)

    if torch.any(resolved_fallback_mask):
        gaussian_ray = sorted_axes[resolved_fallback_mask, :, 0]
        support_ray = effective_basis[resolved_fallback_mask, :, 0]
        ray_misalignment = 1.0 - torch.abs((gaussian_ray * support_ray).sum(dim=1))
        ray_projector = torch.einsum("ni,nj->nij", gaussian_ray, gaussian_ray)
        projector_misalignment = (ray_projector - effective_support[resolved_fallback_mask]).square().mean(dim=(1, 2))
        active_ray_ori_values = ray_misalignment + projector_misalignment
        ori_values.append(active_ray_ori_values)
        ori_weights.append(shape_regularized_weight[resolved_fallback_mask])
        active_ray_ori = _weighted_mean(active_ray_ori_values, shape_regularized_weight[resolved_fallback_mask], zero)

    ori_loss = _weighted_mean(torch.cat(ori_values, dim=0), torch.cat(ori_weights, dim=0), zero) if ori_values else zero

    anisotropy = _compute_log_anisotropy_ratios(sorted_scales)
    sigma_parallel = _nan_to_finite(gaussians.get_center_sigma_parallel.detach()[active_mask].squeeze(-1), 1e-6).clamp(1e-6, 1e8)
    sigma_support = _nan_to_finite(gaussians.get_center_sigma_support.detach()[active_mask].squeeze(-1), 1e-6).clamp(1e-6, 1e8)
    aniso_target = atlas_anisotropy_ref.clone()
    if torch.any(resolved_fallback_mask):
        parallel_ratio = torch.log(
            (
                torch.maximum(sigma_parallel[resolved_fallback_mask], sigma_support[resolved_fallback_mask])
                / sigma_support[resolved_fallback_mask].clamp_min(1e-8)
            ).clamp(1e-8, 1e8)
        )
        aniso_target[resolved_fallback_mask, 0] = parallel_ratio
        aniso_target[resolved_fallback_mask, 1] = 0.0
    aniso_error = (anisotropy - aniso_target).square().sum(dim=1)
    aniso_regularized_weight = shape_regularized_weight
    aniso_loss = _weighted_mean(aniso_error, aniso_regularized_weight, zero)

    # The mean anchor stays separate from the scale-free shape prior: it only pulls
    # centers through the atlas support projector and uses detached reliability.
    mean_total_loss = float(mean_weight) * mean_loss
    shape_loss = float(ori_weight) * ori_loss + float(aniso_weight) * aniso_loss
    total_loss = mean_total_loss + shape_loss
    metrics["atlas_active_fraction"], _ = _safe_metric_scalar(active_count / max(int(gaussians.get_xyz.shape[0]), 1))
    metrics["atlas_regularized_fraction"], _ = _safe_metric_scalar(
        regularized_mask.float().mean() if regularized_mask.numel() > 0 else 0.0
    )
    metrics["atlas_mean_loss"], _ = _safe_metric_scalar(mean_loss)
    metrics["atlas_ori_loss"], _ = _safe_metric_scalar(ori_loss)
    metrics["atlas_aniso_loss"], _ = _safe_metric_scalar(aniso_loss)
    metrics["atlas_mean_total_loss"], _ = _safe_metric_scalar(mean_total_loss)
    metrics["atlas_shape_loss"], _ = _safe_metric_scalar(shape_loss)
    metrics["atlas_regularization_total_loss"], _ = _safe_metric_scalar(total_loss)
    metrics["atlas_regularization_total_loss_safe_for_log"], metrics["atlas_regularization_total_loss_had_nonfinite"] = _safe_metric_scalar(total_loss)
    metrics["atlas_mean_projected_drift"], _ = _safe_metric_scalar(projected_drift.mean())
    metrics["atlas_mean_projected_drift_regularized"] = (
        _safe_metric_scalar(projected_drift[regularized_mask].mean())[0] if torch.any(regularized_mask) else 0.0
    )
    metrics["atlas_mean_projected_drift_unresolved_active"] = (
        _safe_metric_scalar(projected_drift[unresolved_active_mask].mean())[0] if torch.any(unresolved_active_mask) else 0.0
    )
    metrics["atlas_mean_projected_energy"], metrics["atlas_mean_projected_energy_had_nonfinite"] = _safe_metric_scalar(
        projected_energy.mean() if projected_energy.numel() > 0 else 0.0
    )
    metrics["atlas_mean_projected_energy_safe_for_log"] = metrics["atlas_mean_projected_energy"]
    metrics["nonfinite_projected_energy_count"] = float(metrics["atlas_mean_projected_energy_had_nonfinite"])
    metrics["atlas_fallback_fraction"], _ = _safe_metric_scalar(fallback_valid.float().sum() / max(active_count, 1))
    active_unstable_count = int(torch.count_nonzero(atlas_state_active == GAUSSIAN_STATE_UNSTABLE_ACTIVE).item())
    metrics["atlas_active_ray_fraction"], _ = _safe_metric_scalar(fallback_valid.float().sum() / max(active_unstable_count, 1))
    metrics["atlas_unresolved_active_fraction"], _ = _safe_metric_scalar(
        unresolved_active_mask.float().sum() / max(active_unstable_count, 1)
    )
    metrics["atlas_ori_surface_loss"], _ = _safe_metric_scalar(surface_ori)
    metrics["atlas_ori_edge_loss"], _ = _safe_metric_scalar(edge_ori)
    metrics["atlas_ori_active_ray_loss"], _ = _safe_metric_scalar(active_ray_ori)
    state_masks = {
        "stable": atlas_state_active == GAUSSIAN_STATE_STABLE,
        "passive": atlas_state_active == GAUSSIAN_STATE_UNSTABLE_PASSIVE,
        "active": atlas_state_active == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    }
    atlas_radius = gaussians.get_gaussian_atlas_radius.detach()[active_mask].clamp_min(1e-4)
    normalized_drift = projected_drift / atlas_radius
    for state_name, state_mask in state_masks.items():
        metrics[f"atlas_mean_projected_drift_{state_name}"], _ = _safe_metric_scalar(
            _masked_metric_mean(projected_drift, state_mask)
        )
        metrics[f"atlas_aniso_{state_name}_loss"], _ = _safe_metric_scalar(
            _masked_metric_mean(aniso_error, state_mask)
        )
        if torch.any(state_mask):
            state_norm_drift = normalized_drift[state_mask]
            metrics[f"atlas_drift_hist_{state_name}_low"], _ = _safe_metric_scalar((state_norm_drift < 0.25).float().mean())
            metrics[f"atlas_drift_hist_{state_name}_mid"], _ = _safe_metric_scalar(
                ((state_norm_drift >= 0.25) & (state_norm_drift < 1.0)).float().mean()
            )
            metrics[f"atlas_drift_hist_{state_name}_high"], _ = _safe_metric_scalar((state_norm_drift >= 1.0).float().mean())
    metrics["atlas_state_weight_mean"], _ = _safe_metric_scalar(
        shape_state_weight.mean() if shape_state_weight.numel() > 0 else 0.0
    )
    metrics["atlas_mean_state_weight_mean"], _ = _safe_metric_scalar(
        mean_state_weight.mean() if mean_state_weight.numel() > 0 else 0.0
    )
    metrics["atlas_shape_state_weight_mean"], _ = _safe_metric_scalar(
        shape_state_weight.mean() if shape_state_weight.numel() > 0 else 0.0
    )
    metrics["atlas_runtime_reliability_mean"], _ = _safe_metric_scalar(
        atlas_reliability.mean() if atlas_reliability.numel() > 0 else 0.0
    )
    return total_loss, metrics
