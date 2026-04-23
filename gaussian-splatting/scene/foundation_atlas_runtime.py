from __future__ import annotations

import torch
import torch.nn.functional as F

from scene.foundation_atlas import GAUSSIAN_STATE_UNSTABLE_ACTIVE
from utils.graphics_utils import geom_transform_points


def _cache_runtime_observation(gaussians, payload: dict):
    try:
        setattr(gaussians, "_atlas_runtime_last_observation", payload)
    except Exception:
        pass


def _project_points(camera, points: torch.Tensor):
    if hasattr(camera, "refresh_pose_matrices"):
        camera.refresh_pose_matrices()
    ndc = geom_transform_points(points, camera.full_proj_transform)
    w2c = camera.world_view_transform.transpose(0, 1)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    cam_h = torch.cat((points, ones), dim=1) @ w2c
    depth = cam_h[:, 2] / cam_h[:, 3].clamp_min(1e-6)
    valid = depth > camera.znear
    valid = valid & torch.isfinite(depth) & torch.isfinite(ndc[:, :2]).all(dim=1)
    valid = valid & (ndc[:, 0].abs() <= 1.0) & (ndc[:, 1].abs() <= 1.0)
    return ndc[:, :2], depth, valid


def _sample_patches(image: torch.Tensor, coords_ndc: torch.Tensor, radius: int, batch_size: int):
    radius = int(max(radius, 0))
    patch_size = radius * 2 + 1
    if coords_ndc.shape[0] == 0:
        return torch.empty((0, image.shape[0], patch_size, patch_size), dtype=image.dtype, device=image.device)

    if radius == 0:
        grid = coords_ndc.view(1, -1, 1, 2)
        sampled = F.grid_sample(image.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True)
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1)[:, :, None, None]

    height = max(int(image.shape[1]) - 1, 1)
    width = max(int(image.shape[2]) - 1, 1)
    offset_x = (2.0 / width) * torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    offset_y = (2.0 / height) * torch.arange(-radius, radius + 1, dtype=image.dtype, device=image.device)
    grid_y, grid_x = torch.meshgrid(offset_y, offset_x, indexing="ij")
    offsets = torch.stack((grid_x, grid_y), dim=-1)

    outputs = []
    image_batch = image.unsqueeze(0)
    step = max(int(batch_size), 1)
    for start in range(0, coords_ndc.shape[0], step):
        end = min(start + step, coords_ndc.shape[0])
        chunk_coords = coords_ndc[start:end]
        patch_grid = chunk_coords[:, None, None, :] + offsets[None, :, :, :]
        outputs.append(
            F.grid_sample(
                image_batch.expand(chunk_coords.shape[0], -1, -1, -1),
                patch_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
        )
    return torch.cat(outputs, dim=0)


def _gradient_energy(image: torch.Tensor):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=image.dtype,
        device=image.device,
    ).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    channels = image.shape[0]
    image_batch = image.unsqueeze(0)
    grad_x = F.conv2d(image_batch, kernel_x.expand(channels, 1, -1, -1), padding=1, groups=channels)
    grad_y = F.conv2d(image_batch, kernel_y.expand(channels, 1, -1, -1), padding=1, groups=channels)
    grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-8).mean(dim=1, keepdim=True)
    return grad_mag.squeeze(0)


def _offset_distance_sq(radius: int, device, dtype):
    offsets = torch.arange(-int(radius), int(radius) + 1, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    return grid_x.square() + grid_y.square()


def _estimate_effective_patch_radius(gaussians, visible_idx: torch.Tensor, radii: torch.Tensor, max_patch_radius: int):
    base_radius = radii[visible_idx].detach().to(dtype=torch.float32).clamp_min(0.75)
    if not gaussians.has_atlas_bindings or visible_idx.numel() == 0:
        return base_radius.clamp_max(float(max_patch_radius)), torch.zeros_like(base_radius)

    atlas_radius = gaussians.get_gaussian_atlas_radius.detach()[visible_idx].clamp_min(1e-4)
    sigma_support = gaussians.get_center_sigma_support.detach()[visible_idx].squeeze(-1)
    sigma_parallel = gaussians.get_center_sigma_parallel.detach()[visible_idx].squeeze(-1)
    state = gaussians.get_atlas_state.detach()[visible_idx]
    dominant_sigma = torch.where(
        state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
        torch.maximum(sigma_support, sigma_parallel),
        sigma_support,
    )
    uncertainty_ratio = dominant_sigma / atlas_radius
    effective_radius = base_radius * (1.0 + 0.5 * uncertainty_ratio.clamp(0.0, 4.0))
    return effective_radius.clamp(0.75, float(max_patch_radius)), uncertainty_ratio


def sample_gaussian_photometric_residuals(
    viewpoint_camera,
    gaussians,
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    radii: torch.Tensor,
    rendered_invdepth: torch.Tensor | None = None,
    max_patch_radius: int = 6,
    patch_batch_size: int = 2048,
    dark_luma_threshold: float = 0.15,
    low_texture_threshold: float = 0.05,
    floater_error_threshold: float = 0.20,
    enabled: bool = True,
):
    device = gaussians.get_xyz.device
    num_points = int(gaussians.get_xyz.shape[0])
    residuals = torch.zeros((num_points,), dtype=torch.float32, device=device)
    candidate_mask = (radii > 0).to(device=device)
    runtime_cache = {
        "candidate_mask": candidate_mask.detach(),
        "projected_mask": torch.zeros_like(candidate_mask),
        "observed_mask": torch.zeros_like(candidate_mask),
        "visibility_contribution": torch.zeros((num_points,), dtype=torch.float32, device=device),
        "dark_region_mask": torch.zeros_like(candidate_mask),
        "smooth_region_mask": torch.zeros_like(candidate_mask),
        "floater_region_mask": torch.zeros_like(candidate_mask),
        "patch_quality_score": torch.zeros((num_points,), dtype=torch.float32, device=device),
        "mask_nonzero_ratio": torch.zeros((num_points,), dtype=torch.float32, device=device),
        "bg_like_ratio": torch.zeros((num_points,), dtype=torch.float32, device=device),
        "background_like_ratio": torch.zeros((num_points,), dtype=torch.float32, device=device),
        "thin_support_like_ratio": torch.zeros((num_points,), dtype=torch.float32, device=device),
        "photo_signal_strength": torch.zeros((num_points,), dtype=torch.float32, device=device),
    }
    if not enabled:
        _cache_runtime_observation(gaussians, runtime_cache)
        return residuals, torch.zeros_like(candidate_mask)
    if num_points == 0 or not torch.any(candidate_mask):
        _cache_runtime_observation(gaussians, runtime_cache)
        return residuals, candidate_mask

    coords_ndc, depth, valid_projection = _project_points(viewpoint_camera, gaussians.get_xyz.detach())
    projected_mask = candidate_mask & valid_projection
    runtime_cache["projected_mask"] = projected_mask.detach()
    visible_mask = projected_mask
    if not torch.any(visible_mask):
        _cache_runtime_observation(gaussians, runtime_cache)
        return residuals, visible_mask

    visible_idx = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
    effective_radius, uncertainty_ratio = _estimate_effective_patch_radius(
        gaussians,
        visible_idx,
        radii,
        max_patch_radius=int(max_patch_radius),
    )
    patch_radius = int(max(min(int(torch.ceil(effective_radius.max()).item()), int(max_patch_radius)), 1))

    residual_map = (rendered_image.detach() - gt_image.detach()).abs().mean(dim=0, keepdim=True)
    residual_patch = _sample_patches(residual_map, coords_ndc[visible_idx], patch_radius, batch_size=patch_batch_size).squeeze(1)
    gradient_map = _gradient_energy(gt_image.detach())
    gradient_patch = _sample_patches(gradient_map, coords_ndc[visible_idx], patch_radius, batch_size=patch_batch_size).squeeze(1)
    gt = gt_image.detach()
    luma_map = (0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]).unsqueeze(0)
    luma_patch = _sample_patches(luma_map, coords_ndc[visible_idx], patch_radius, batch_size=patch_batch_size).squeeze(1)

    offset_sq = _offset_distance_sq(patch_radius, device=residual_patch.device, dtype=residual_patch.dtype).unsqueeze(0)
    sigma_px = (0.65 * effective_radius[:, None, None]).clamp_min(0.5)
    footprint = torch.exp(-0.5 * offset_sq / sigma_px.square())
    footprint = footprint * (offset_sq <= (effective_radius[:, None, None] + 0.75).square()).to(dtype=residual_patch.dtype)
    footprint_mass = footprint.sum(dim=(1, 2)).clamp_min(1e-6)

    texture_gate = gradient_patch / (gradient_patch + 0.05)
    weights = footprint * (0.30 + 0.70 * texture_gate)

    visibility_consistency = torch.ones((visible_idx.shape[0],), dtype=torch.float32, device=device)
    if rendered_invdepth is not None:
        invdepth_map = rendered_invdepth.detach()
        if invdepth_map.ndim == 2:
            invdepth_map = invdepth_map.unsqueeze(0)
        depth_patch = _sample_patches(invdepth_map, coords_ndc[visible_idx], patch_radius, batch_size=patch_batch_size).squeeze(1)
        expected_invdepth = depth[visible_idx].reciprocal().clamp_max(1e4)[:, None, None]
        depth_tolerance = 0.015 + expected_invdepth * (0.05 + 0.20 * uncertainty_ratio[:, None, None].clamp(0.0, 4.0))
        depth_gate = torch.exp(-(depth_patch - expected_invdepth).abs() / depth_tolerance.clamp_min(1e-4))
        depth_gate = depth_gate * (depth_patch > 0).to(dtype=depth_gate.dtype)
        weights = weights * depth_gate
        visibility_consistency = ((footprint * depth_gate).sum(dim=(1, 2)) / footprint_mass).clamp(0.0, 1.0)

    weight_sum = weights.sum(dim=(1, 2))
    support_mass_ratio = (weight_sum / footprint.sum(dim=(1, 2)).clamp_min(1e-6)).clamp(0.0, 1.0)
    visibility_contribution = torch.zeros((num_points,), dtype=torch.float32, device=device)
    visibility_contribution[visible_idx] = (support_mass_ratio * visibility_consistency).clamp(0.0, 1.0)
    observed = weight_sum > 1e-4
    observed = observed & (visibility_consistency > 0.05)
    if not torch.any(observed):
        runtime_cache["visibility_contribution"] = visibility_contribution.detach()
        _cache_runtime_observation(gaussians, runtime_cache)
        return residuals, torch.zeros_like(visible_mask)

    reduced = torch.zeros((visible_idx.shape[0],), dtype=torch.float32, device=device)
    safe_weight_sum = weight_sum[observed].clamp_min(1e-6)
    reduced[observed] = (
        (residual_patch[observed] * weights[observed]).sum(dim=(1, 2)) / safe_weight_sum
    ) * visibility_consistency[observed]
    final_visible_mask = torch.zeros_like(visible_mask)
    final_visible_mask[visible_idx[observed]] = True
    residuals[visible_idx[observed]] = reduced[observed].clamp(0.0, 1.0)
    dark_strength = (
        footprint * (luma_patch <= float(max(dark_luma_threshold, 0.0))).to(dtype=footprint.dtype)
    ).sum(dim=(1, 2)) / footprint_mass
    smooth_strength = (
        footprint * (gradient_patch <= float(max(low_texture_threshold, 0.0))).to(dtype=footprint.dtype)
    ).sum(dim=(1, 2)) / footprint_mass
    dark_region_mask = torch.zeros_like(visible_mask)
    smooth_region_mask = torch.zeros_like(visible_mask)
    floater_region_mask = torch.zeros_like(visible_mask)
    patch_quality_score = torch.zeros((num_points,), dtype=torch.float32, device=device)
    mask_nonzero_ratio = torch.zeros((num_points,), dtype=torch.float32, device=device)
    bg_like_ratio = torch.zeros((num_points,), dtype=torch.float32, device=device)
    thin_support_like_ratio = torch.zeros((num_points,), dtype=torch.float32, device=device)
    photo_signal_strength = torch.zeros((num_points,), dtype=torch.float32, device=device)
    observed_indices = visible_idx[observed]
    dark_region_mask[observed_indices] = dark_strength[observed] >= 0.35
    smooth_region_mask[observed_indices] = smooth_strength[observed] >= 0.35
    floater_region_mask[observed_indices] = (
        smooth_region_mask[observed_indices]
        & (residuals[observed_indices] >= float(max(floater_error_threshold, 0.0)))
    )
    grad_strength = (
        (footprint * texture_gate).sum(dim=(1, 2)) / footprint_mass
    ).clamp(0.0, 1.0)
    patch_quality_visible = (
        support_mass_ratio
        * visibility_consistency
        * (0.35 + 0.65 * grad_strength)
    ).clamp(0.0, 1.0)
    nonzero_visible = (weights > 1e-6).to(dtype=torch.float32).mean(dim=(1, 2)).clamp(0.0, 1.0)
    bg_like_visible = (0.55 * dark_strength + 0.45 * smooth_strength).clamp(0.0, 1.0)
    edge_like_visible = (grad_strength * (1.0 - smooth_strength).clamp(0.0, 1.0)).clamp(0.0, 1.0)
    thin_like_visible = (
        edge_like_visible
        * (0.40 + 0.60 * support_mass_ratio)
        * (0.35 + 0.65 * visibility_consistency)
    ).clamp(0.0, 1.0)
    photo_signal_visible = (reduced * patch_quality_visible).clamp(0.0, 1.0)
    patch_quality_score[observed_indices] = patch_quality_visible[observed]
    mask_nonzero_ratio[observed_indices] = nonzero_visible[observed]
    bg_like_ratio[observed_indices] = bg_like_visible[observed]
    thin_support_like_ratio[observed_indices] = thin_like_visible[observed]
    photo_signal_strength[observed_indices] = photo_signal_visible[observed]
    runtime_cache["observed_mask"] = final_visible_mask.detach()
    runtime_cache["visibility_contribution"] = visibility_contribution.detach()
    runtime_cache["dark_region_mask"] = dark_region_mask.detach()
    runtime_cache["smooth_region_mask"] = smooth_region_mask.detach()
    runtime_cache["floater_region_mask"] = floater_region_mask.detach()
    runtime_cache["patch_quality_score"] = patch_quality_score.detach()
    runtime_cache["mask_nonzero_ratio"] = mask_nonzero_ratio.detach()
    runtime_cache["bg_like_ratio"] = bg_like_ratio.detach()
    runtime_cache["background_like_ratio"] = bg_like_ratio.detach()
    runtime_cache["thin_support_like_ratio"] = thin_support_like_ratio.detach()
    runtime_cache["photo_signal_strength"] = photo_signal_strength.detach()
    _cache_runtime_observation(gaussians, runtime_cache)
    return residuals, final_visible_mask


def compute_render_validation_proxies(
    rendered_image: torch.Tensor,
    gt_image: torch.Tensor,
    rendered_invdepth: torch.Tensor | None = None,
    dark_luma_threshold: float = 0.15,
    low_texture_threshold: float = 0.05,
    floater_error_threshold: float = 0.20,
):
    metrics = {
        "dark_region_completeness_proxy": 0.0,
        "dark_region_pixel_ratio": 0.0,
        "dark_region_l1": 0.0,
        "floater_proxy": 0.0,
        "smooth_region_pixel_ratio": 0.0,
        "smooth_region_l1": 0.0,
        "valid_depth_ratio": 0.0,
    }
    if rendered_image.numel() == 0 or gt_image.numel() == 0:
        return metrics

    render = rendered_image.detach()
    gt = gt_image.detach()
    if render.ndim == 4:
        render = render.squeeze(0)
    if gt.ndim == 4:
        gt = gt.squeeze(0)
    if render.ndim != 3 or gt.ndim != 3:
        return metrics

    error_map = (render - gt).abs().mean(dim=0)
    gt_luma = 0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]
    gradient_map = _gradient_energy(gt).squeeze(0)

    valid_depth = torch.ones_like(error_map, dtype=torch.bool)
    if rendered_invdepth is not None and rendered_invdepth.numel() > 0:
        depth_map = rendered_invdepth.detach()
        if depth_map.ndim == 3 and depth_map.shape[0] == 1:
            depth_map = depth_map.squeeze(0)
        if depth_map.ndim == 2 and depth_map.shape == error_map.shape:
            valid_depth = torch.isfinite(depth_map) & (depth_map > 0)

    metrics["valid_depth_ratio"] = float(valid_depth.float().mean().item()) if valid_depth.numel() > 0 else 0.0

    dark_mask = gt_luma <= float(max(dark_luma_threshold, 0.0))
    if torch.any(dark_mask):
        metrics["dark_region_pixel_ratio"] = float(dark_mask.float().mean().item())
        metrics["dark_region_completeness_proxy"] = float(valid_depth[dark_mask].float().mean().item())
        metrics["dark_region_l1"] = float(error_map[dark_mask].mean().item())

    smooth_mask = gradient_map <= float(max(low_texture_threshold, 0.0))
    if torch.any(smooth_mask):
        floater_mask = smooth_mask & valid_depth & (error_map >= float(max(floater_error_threshold, 0.0)))
        metrics["smooth_region_pixel_ratio"] = float(smooth_mask.float().mean().item())
        metrics["smooth_region_l1"] = float(error_map[smooth_mask].mean().item())
        metrics["floater_proxy"] = float(floater_mask.float().mean().item())

    return metrics
