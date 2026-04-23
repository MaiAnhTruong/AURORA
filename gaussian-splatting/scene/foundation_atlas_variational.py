from __future__ import annotations

import math

import torch

from scene.foundation_atlas import (
    ATLAS_CLASS_SURFACE,
    GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
)


ACTIVE_SUBSPACE_ENERGY_KEEP = 0.78
ACTIVE_SUBSPACE_RELATIVE_KEEP = 0.45
OBSERVATION_FRUSTUM_MARGIN = 0.05
OBSERVATION_WEIGHT_FLOOR = 0.05
SUPPORT_EIGENVALUE_KEEP = 0.5


def _symmetrize(matrix: torch.Tensor):
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def _nan_to_finite(tensor: torch.Tensor, value: float = 0.0):
    return torch.nan_to_num(
        tensor,
        nan=float(value),
        posinf=float(value),
        neginf=float(value),
    )


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


def _safe_normalize(vectors: torch.Tensor, fallback: torch.Tensor, eps: float = 1e-6):
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    safe_fallback = fallback / torch.linalg.norm(fallback, dim=-1, keepdim=True).clamp_min(eps)
    normalized = vectors / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, safe_fallback)


def _projector_from_basis(basis: torch.Tensor, rank: torch.Tensor):
    rank = rank.to(device=basis.device).round().long().clamp(0, basis.shape[-1])
    keep = (
        torch.arange(basis.shape[-1], device=basis.device)
        .view(1, 1, -1)
        < rank.view(-1, 1, 1)
    ).to(dtype=basis.dtype)
    selected = basis * keep
    return _symmetrize(torch.bmm(selected, selected.transpose(1, 2)))


def _sanitize_psd_projector(projector: torch.Tensor, fallback: torch.Tensor | None = None):
    if projector.numel() == 0:
        return projector
    safe = _symmetrize(_nan_to_finite(projector, 0.0))
    eigvals, eigvecs = torch.linalg.eigh(safe)
    eigvals = eigvals.clamp(0.0, 1.0)
    sanitized = torch.bmm(eigvecs, torch.bmm(torch.diag_embed(eigvals), eigvecs.transpose(1, 2)))
    sanitized = _symmetrize(_nan_to_finite(sanitized, 0.0))
    finite = torch.isfinite(sanitized).reshape(sanitized.shape[0], -1).all(dim=1)
    if fallback is not None and torch.any(~finite):
        fallback = fallback.to(device=projector.device, dtype=projector.dtype)
        sanitized[~finite] = fallback[~finite]
    return sanitized


def _safe_projected_sq(projector: torch.Tensor, vectors: torch.Tensor):
    if vectors.numel() == 0:
        return vectors.new_zeros((0,))
    safe_projector = _sanitize_psd_projector(projector)
    safe_vectors = _nan_to_finite(vectors, 0.0)
    projected = torch.bmm(safe_projector, safe_vectors.unsqueeze(-1)).squeeze(-1)
    return _nan_to_finite(torch.sum(projected * safe_vectors, dim=1), 0.0).clamp_min(0.0)


def _safe_positive(tensor: torch.Tensor, minimum: float, maximum: float):
    return _nan_to_finite(tensor, float(minimum)).clamp(float(minimum), float(maximum))


def _scalar_center_kl(rank: torch.Tensor, sigma: torch.Tensor, precision: torch.Tensor, delta_sq: torch.Tensor):
    precision = _safe_positive(precision, 1e-8, 1e8)
    sigma = _safe_positive(sigma, 1e-8, 1e8)
    variance = sigma.square().clamp(1e-16, 1e16)
    scaled_variance = (precision * variance).clamp(1e-12, 1e12)
    rank = _nan_to_finite(rank.to(dtype=sigma.dtype), 0.0).clamp(0.0, 3.0)
    delta_sq = _nan_to_finite(delta_sq, 0.0).clamp_min(0.0)
    return 0.5 * rank * (scaled_variance - torch.log(scaled_variance) - 1.0) + 0.5 * precision * delta_sq


def _projector_from_matrix(matrix: torch.Tensor, target_rank: torch.Tensor):
    eigvals, eigvecs = torch.linalg.eigh(_symmetrize(matrix))
    eigvals = torch.flip(eigvals, dims=[1]).clamp_min(0.0)
    eigvecs = torch.flip(eigvecs, dims=[2])
    target_rank = target_rank.round().long().clamp(0, matrix.shape[-1])
    target_rank = torch.where(eigvals[:, 0] > 1e-8, target_rank, torch.zeros_like(target_rank))
    return _projector_from_basis(eigvecs, target_rank)


def _stable_psd_inverse(matrix: torch.Tensor, floor: float):
    eigvals, eigvecs = torch.linalg.eigh(_symmetrize(matrix))
    floor = max(float(floor), 1e-8)
    inv_eigvals = eigvals.clamp_min(floor).reciprocal()
    inverse = torch.bmm(eigvecs, torch.bmm(torch.diag_embed(inv_eigvals), eigvecs.transpose(1, 2)))
    return _symmetrize(inverse)


def _round_projector_rank(projector: torch.Tensor):
    trace = _nan_to_finite(projector, 0.0).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    return trace.round().clamp(0.0, 3.0)


def _rank_ratio(rank: torch.Tensor, value: int):
    if rank.numel() == 0:
        return 0.0
    return float((rank.round().long() == int(value)).float().mean().detach().item())


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


def _decompose_support_projector(projector: torch.Tensor):
    projector = _sanitize_psd_projector(projector)
    eigvals, eigvecs = torch.linalg.eigh(_symmetrize(projector))
    eigvals = torch.flip(eigvals, dims=[1]).clamp(0.0, 1.0)
    eigvecs = torch.flip(eigvecs, dims=[2])
    rank_from_trace = _round_projector_rank(projector).long()
    rank_from_spectrum = (eigvals >= float(SUPPORT_EIGENVALUE_KEEP)).sum(dim=1)
    rank = torch.minimum(rank_from_trace, rank_from_spectrum).clamp(0, projector.shape[-1])
    keep = (
        torch.arange(projector.shape[-1], device=projector.device)
        .view(1, -1)
        < rank.view(-1, 1)
    ).to(dtype=projector.dtype)
    support_local = torch.diag_embed(keep)
    return eigvecs, rank, support_local


def _complete_basis_from_primary(primary: torch.Tensor, fallback_basis: torch.Tensor):
    primary = _safe_normalize(primary, fallback=fallback_basis[:, :, 0])
    second = fallback_basis[:, :, 1] - torch.sum(fallback_basis[:, :, 1] * primary, dim=1, keepdim=True) * primary
    second = _safe_normalize(second, fallback=fallback_basis[:, :, 1])
    third = torch.cross(primary, second, dim=1)
    third = _safe_normalize(third, fallback=fallback_basis[:, :, 2])
    second = torch.cross(third, primary, dim=1)
    second = _safe_normalize(second, fallback=fallback_basis[:, :, 1])
    return torch.stack((primary, second, third), dim=2)


def _select_reference_aware_camera_ids(
    total_cameras: int,
    max_cameras: int,
    ref_camera: torch.Tensor,
    ref_score: torch.Tensor,
    global_view_weight: torch.Tensor | None = None,
    global_view_count: torch.Tensor | None = None,
):
    if total_cameras <= 0:
        return torch.empty((0,), dtype=torch.long, device=ref_camera.device)
    if max_cameras <= 0 or total_cameras <= int(max_cameras):
        return torch.arange(total_cameras, dtype=torch.long, device=ref_camera.device)

    max_cameras = int(max_cameras)
    selected = []
    selected_set = set()

    valid_ref = (ref_camera >= 0) & (ref_camera < total_cameras)
    global_score = None
    if global_view_weight is not None and global_view_weight.numel() == total_cameras:
        global_score = global_view_weight.to(dtype=ref_score.dtype)
        if global_view_count is not None and global_view_count.numel() == total_cameras:
            global_score = global_score + 0.1 * global_view_count.to(dtype=ref_score.dtype)
    if torch.any(valid_ref):
        ref_ids = ref_camera[valid_ref].long()
        ref_weights = ref_score[valid_ref].clamp_min(0.0) + 1.0
        score_per_camera = torch.zeros((total_cameras,), dtype=ref_weights.dtype, device=ref_weights.device)
        count_per_camera = torch.zeros((total_cameras,), dtype=ref_weights.dtype, device=ref_weights.device)
        score_per_camera.index_add_(0, ref_ids, ref_weights)
        count_per_camera.index_add_(0, ref_ids, torch.ones_like(ref_weights))
        if global_score is not None:
            score_per_camera = score_per_camera + global_score
        order = torch.argsort(score_per_camera + 0.1 * count_per_camera, descending=True)
        for camera_id in order.tolist():
            if score_per_camera[camera_id] <= 0.0:
                continue
            selected.append(int(camera_id))
            selected_set.add(int(camera_id))
            if len(selected) >= min(max_cameras, max(1, max_cameras // 2)):
                break
    elif global_score is not None and torch.any(global_score > 0.0):
        order = torch.argsort(global_score, descending=True)
        for camera_id in order.tolist():
            if global_score[camera_id] <= 0.0:
                continue
            selected.append(int(camera_id))
            selected_set.add(int(camera_id))
            if len(selected) >= min(max_cameras, max(1, max_cameras // 2)):
                break

    remaining = [camera_id for camera_id in range(total_cameras) if camera_id not in selected_set]
    needed = max_cameras - len(selected)
    if needed > 0:
        if len(remaining) <= needed:
            selected.extend(remaining)
        else:
            indices = torch.linspace(
                0,
                len(remaining) - 1,
                steps=needed,
                dtype=torch.float32,
                device=ref_camera.device,
            ).round().long()
            selected.extend(remaining[int(idx)] for idx in torch.unique_consecutive(indices).tolist())
        selected_set.update(selected)

    if len(selected) > max_cameras:
        selected = selected[:max_cameras]
    if len(selected) < max_cameras:
        for camera_id in range(total_cameras):
            if camera_id in selected_set:
                continue
            selected.append(camera_id)
            if len(selected) >= max_cameras:
                break

    return torch.tensor(selected, dtype=torch.long, device=ref_camera.device)


def _build_camera_observations(camera_evidence, gaussians, max_cameras: int):
    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    ref_camera = gaussians.get_atlas_ref_camera.detach()
    ref_score = gaussians.get_atlas_ref_score.detach()
    atlas_view_weights = gaussians.get_gaussian_atlas_view_weights.detach()
    atlas_view_counts = gaussians.get_gaussian_atlas_view_counts.detach()
    global_view_weight = atlas_view_weights.sum(dim=0) if atlas_view_weights.ndim == 2 and atlas_view_weights.shape[1] > 0 else None
    global_view_count = atlas_view_counts.sum(dim=0) if atlas_view_counts.ndim == 2 and atlas_view_counts.shape[1] > 0 else None

    if torch.is_tensor(camera_evidence):
        all_centers = camera_evidence.to(device=device, dtype=dtype).reshape(-1, 3)
        selected_ids = _select_reference_aware_camera_ids(
            all_centers.shape[0],
            int(max_cameras),
            ref_camera,
            ref_score,
            global_view_weight=global_view_weight,
            global_view_count=global_view_count,
        )
        return {
            "mode": "centers",
            "camera_ids": selected_ids,
            "all_centers": all_centers,
            "centers": all_centers.index_select(0, selected_ids) if selected_ids.numel() > 0 else all_centers,
        }

    cameras = [] if camera_evidence is None else list(camera_evidence)
    centers = []
    rotations = []
    translations = []
    fx = []
    fy = []
    tan_half_x = []
    tan_half_y = []
    znear = []

    for camera in cameras:
        if camera is None:
            continue
        if hasattr(camera, "refresh_pose_matrices"):
            camera.refresh_pose_matrices()
        if not hasattr(camera, "camera_center") or not hasattr(camera, "world_view_transform"):
            continue
        if not hasattr(camera, "FoVx") or not hasattr(camera, "FoVy"):
            continue
        center = camera.camera_center.detach().to(device=device, dtype=dtype).reshape(3)
        w2c = camera.world_view_transform.transpose(0, 1).detach().to(device=device, dtype=dtype)
        centers.append(center)
        rotations.append(w2c[:3, :3])
        translations.append(w2c[:3, 3])
        width = float(max(int(getattr(camera, "image_width", 1)), 1))
        height = float(max(int(getattr(camera, "image_height", 1)), 1))
        tan_x = max(math.tan(float(camera.FoVx) * 0.5), 1e-6)
        tan_y = max(math.tan(float(camera.FoVy) * 0.5), 1e-6)
        fx.append(0.5 * width / tan_x)
        fy.append(0.5 * height / tan_y)
        tan_half_x.append(tan_x)
        tan_half_y.append(tan_y)
        znear.append(float(getattr(camera, "znear", 0.01)))

    if not centers:
        empty = torch.empty((0, 3), dtype=dtype, device=device)
        return {
            "mode": "centers",
            "camera_ids": torch.empty((0,), dtype=torch.long, device=device),
            "all_centers": empty,
            "centers": empty,
        }

    centers = torch.stack(centers, dim=0)
    rotations = torch.stack(rotations, dim=0)
    translations = torch.stack(translations, dim=0)
    fx = torch.tensor(fx, dtype=dtype, device=device)
    fy = torch.tensor(fy, dtype=dtype, device=device)
    tan_half_x = torch.tensor(tan_half_x, dtype=dtype, device=device)
    tan_half_y = torch.tensor(tan_half_y, dtype=dtype, device=device)
    znear = torch.tensor(znear, dtype=dtype, device=device)
    selected_ids = _select_reference_aware_camera_ids(
        centers.shape[0],
        int(max_cameras),
        ref_camera,
        ref_score,
        global_view_weight=global_view_weight,
        global_view_count=global_view_count,
    )

    return {
        "mode": "projective",
        "camera_ids": selected_ids,
        "all_centers": centers,
        "centers": centers.index_select(0, selected_ids),
        "rotations": rotations.index_select(0, selected_ids),
        "translations": translations.index_select(0, selected_ids),
        "fx": fx.index_select(0, selected_ids),
        "fy": fy.index_select(0, selected_ids),
        "tan_half_x": tan_half_x.index_select(0, selected_ids),
        "tan_half_y": tan_half_y.index_select(0, selected_ids),
        "znear": znear.index_select(0, selected_ids),
    }


def _compute_reference_centers(xyz: torch.Tensor, atlas_pos: torch.Tensor, atlas_basis: torch.Tensor, camera_obs, ref_camera: torch.Tensor):
    all_centers = camera_obs["all_centers"]
    if all_centers.numel() == 0:
        return atlas_pos

    ref_centers = all_centers.mean(dim=0, keepdim=True).expand(xyz.shape[0], -1).clone()
    valid_ref = (ref_camera >= 0) & (ref_camera < all_centers.shape[0])
    if torch.any(valid_ref):
        ref_centers[valid_ref] = all_centers[ref_camera[valid_ref]]

    invalid_ref = ~valid_ref
    if torch.any(invalid_ref):
        d2 = torch.cdist(xyz[invalid_ref], all_centers)
        nearest = d2.argmin(dim=1)
        ref_centers[invalid_ref] = all_centers[nearest]

    fallback_dirs = atlas_basis[:, :, 0]
    fallback_centers = xyz - fallback_dirs
    return torch.where(torch.isfinite(ref_centers).all(dim=1, keepdim=True), ref_centers, fallback_centers)


def _compute_center_based_observation_hessian(xyz: torch.Tensor, centers: torch.Tensor, point_chunk_size: int):
    device = xyz.device
    dtype = xyz.dtype
    hessian = torch.zeros((xyz.shape[0], 3, 3), dtype=dtype, device=device)
    view_count = torch.zeros((xyz.shape[0],), dtype=dtype, device=device)
    weight_sum = torch.zeros((xyz.shape[0],), dtype=dtype, device=device)
    if centers.numel() == 0:
        return hessian, view_count, weight_sum

    eye = torch.eye(3, dtype=dtype, device=device).view(1, 1, 3, 3)
    chunk_size = max(int(point_chunk_size), 1)
    for start in range(0, xyz.shape[0], chunk_size):
        end = min(start + chunk_size, xyz.shape[0])
        pts = xyz[start:end]
        disp = pts[:, None, :] - centers[None, :, :]
        depth = torch.linalg.norm(disp, dim=-1).clamp_min(1e-4)
        rays = disp / depth.unsqueeze(-1)
        jt_j = eye - torch.einsum("nci,ncj->ncij", rays, rays)
        weights = depth.reciprocal().square()
        hessian[start:end] = (jt_j * weights[:, :, None, None]).sum(dim=1)
        view_count[start:end] = float(centers.shape[0])
        weight_sum[start:end] = weights.sum(dim=1)
    return hessian, view_count, weight_sum


def _compute_projective_observation_hessian(gaussians, xyz: torch.Tensor, camera_obs, point_chunk_size: int):
    device = xyz.device
    dtype = xyz.dtype
    hessian = torch.zeros((xyz.shape[0], 3, 3), dtype=dtype, device=device)
    view_count = torch.zeros((xyz.shape[0],), dtype=dtype, device=device)
    weight_sum = torch.zeros((xyz.shape[0],), dtype=dtype, device=device)
    if camera_obs["centers"].numel() == 0:
        return hessian, view_count, weight_sum

    rotations = camera_obs["rotations"]
    translations = camera_obs["translations"]
    fx = camera_obs["fx"]
    fy = camera_obs["fy"]
    tan_half_x = camera_obs["tan_half_x"]
    tan_half_y = camera_obs["tan_half_y"]
    znear = camera_obs["znear"]
    camera_ids = camera_obs["camera_ids"]

    visibility_ema = gaussians.get_atlas_visibility_ema.detach().to(device=device, dtype=dtype).clamp(0.0, 1.0)
    photo_ema = gaussians.get_atlas_photo_ema.detach().to(device=device, dtype=dtype).clamp_min(0.0)
    ref_camera = gaussians.get_atlas_ref_camera.detach().to(device=device, dtype=torch.long)
    ref_score = gaussians.get_atlas_ref_score.detach().to(device=device, dtype=dtype).clamp_min(0.0)
    atlas_view_weights = gaussians.get_gaussian_atlas_view_weights.detach().to(device=device, dtype=dtype)
    atlas_view_counts = gaussians.get_gaussian_atlas_view_counts.detach().to(device=device, dtype=dtype)
    has_view_evidence = (
        atlas_view_weights.ndim == 2
        and atlas_view_counts.ndim == 2
        and atlas_view_weights.shape == atlas_view_counts.shape
        and atlas_view_weights.shape[0] == xyz.shape[0]
        and atlas_view_weights.shape[1] > 0
    )

    point_weight_base = (0.15 + 0.85 * visibility_ema) * torch.exp(-2.0 * photo_ema)
    point_weight_base = point_weight_base.clamp_min(OBSERVATION_WEIGHT_FLOOR)

    chunk_size = max(int(point_chunk_size), 1)
    margin = 1.0 + float(OBSERVATION_FRUSTUM_MARGIN)
    for start in range(0, xyz.shape[0], chunk_size):
        end = min(start + chunk_size, xyz.shape[0])
        pts = xyz[start:end]
        chunk_hessian = torch.zeros((pts.shape[0], 3, 3), dtype=dtype, device=device)
        chunk_view_count = torch.zeros((pts.shape[0],), dtype=dtype, device=device)
        chunk_weight_sum = torch.zeros((pts.shape[0],), dtype=dtype, device=device)
        chunk_base_weight = point_weight_base[start:end]
        chunk_ref_camera = ref_camera[start:end]
        chunk_ref_score = ref_score[start:end]
        if has_view_evidence:
            chunk_view_weights = atlas_view_weights[start:end]
            chunk_view_counts = atlas_view_counts[start:end]
        else:
            chunk_view_weights = None
            chunk_view_counts = None

        for camera_slot in range(camera_ids.shape[0]):
            camera_id = int(camera_ids[camera_slot].item())
            R = rotations[camera_slot]
            t = translations[camera_slot]
            cam_xyz = pts @ R.transpose(0, 1) + t
            X = cam_xyz[:, 0]
            Y = cam_xyz[:, 1]
            Z = cam_xyz[:, 2]
            inv_Z = Z.clamp_min(1e-4).reciprocal()
            x_ndc = X * inv_Z / tan_half_x[camera_slot]
            y_ndc = Y * inv_Z / tan_half_y[camera_slot]
            valid = (
                torch.isfinite(cam_xyz).all(dim=1)
                & (Z > znear[camera_slot])
                & (x_ndc.abs() <= margin)
                & (y_ndc.abs() <= margin)
            )
            if not torch.any(valid):
                continue

            jac_cam = torch.zeros((pts.shape[0], 2, 3), dtype=dtype, device=device)
            jac_cam[:, 0, 0] = fx[camera_slot] * inv_Z
            jac_cam[:, 0, 2] = -fx[camera_slot] * X * inv_Z.square()
            jac_cam[:, 1, 1] = fy[camera_slot] * inv_Z
            jac_cam[:, 1, 2] = -fy[camera_slot] * Y * inv_Z.square()
            jac_world = torch.matmul(jac_cam, R)
            jt_j = torch.einsum("npi,npj->nij", jac_world, jac_world)

            ref_boost = torch.ones_like(chunk_base_weight)
            match_ref = chunk_ref_camera == camera_id
            if torch.any(match_ref):
                ref_boost[match_ref] = ref_boost[match_ref] + chunk_ref_score[match_ref]

            if has_view_evidence and camera_id < chunk_view_weights.shape[1]:
                view_weight = chunk_view_weights[:, camera_id].clamp_min(0.0)
                view_evidence_count = chunk_view_counts[:, camera_id].clamp_min(0.0)
                count_gain = 1.0 - torch.exp(-view_evidence_count / 4.0)
                evidence_weight = view_weight * (0.35 + 0.65 * count_gain)
                support_mask = view_evidence_count > 0.0
                effective_valid = valid & support_mask
                weights = chunk_base_weight * ref_boost * evidence_weight * effective_valid.to(dtype=dtype)
                chunk_view_count = chunk_view_count + effective_valid.to(dtype=dtype)
            else:
                weights = chunk_base_weight * ref_boost * valid.to(dtype=dtype)
                chunk_view_count = chunk_view_count + valid.to(dtype=dtype)
            chunk_hessian = chunk_hessian + jt_j * weights[:, None, None]
            chunk_weight_sum = chunk_weight_sum + weights

        hessian[start:end] = chunk_hessian
        view_count[start:end] = chunk_view_count
        weight_sum[start:end] = chunk_weight_sum

    return hessian, view_count, weight_sum


def _compute_observation_hessian(gaussians, xyz: torch.Tensor, camera_obs, point_chunk_size: int):
    if camera_obs["mode"] == "projective":
        return _compute_projective_observation_hessian(gaussians, xyz, camera_obs, point_chunk_size)
    return _compute_center_based_observation_hessian(xyz, camera_obs["centers"], point_chunk_size)


def _select_active_rank(eigvals: torch.Tensor, support_rank: torch.Tensor):
    support_rank = support_rank.round().long().clamp(0, eigvals.shape[1])
    selected = torch.zeros_like(support_rank)
    if not torch.any(support_rank > 0):
        return selected

    selected[support_rank > 0] = 1
    for rank_value in range(2, eigvals.shape[1] + 1):
        mask = support_rank >= rank_value
        if not torch.any(mask):
            continue
        current = eigvals[mask, :rank_value].clamp_min(0.0)
        total = current.sum(dim=1).clamp_min(1e-8)
        cumulative = current.cumsum(dim=1) / total.unsqueeze(1)
        relative = current[:, rank_value - 1] / current[:, 0].clamp_min(1e-8)
        promote = (cumulative[:, rank_value - 2] < float(ACTIVE_SUBSPACE_ENERGY_KEEP)) & (relative >= float(ACTIVE_SUBSPACE_RELATIVE_KEEP))
        if torch.any(promote):
            target_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)[promote]
            selected[target_idx] = rank_value
    return selected


def _projective_quadratic_form(direction: torch.Tensor, inverse_precision: torch.Tensor):
    return torch.bmm(direction.unsqueeze(1), torch.bmm(inverse_precision, direction.unsqueeze(-1))).squeeze(-1).squeeze(-1)


def build_variational_subspace(
    gaussians,
    camera_centers,
    lambda_reg: float = 1e-3,
    max_cameras: int = 64,
    point_chunk_size: int = 2048,
):
    if not gaussians.has_atlas_bindings:
        return None

    with torch.no_grad():
        device = gaussians.get_xyz.device
        dtype = gaussians.get_xyz.dtype
        xyz = gaussians.get_xyz.detach()
        atlas_pos = gaussians.get_gaussian_atlas_positions.detach()
        atlas_support = _symmetrize(gaussians.get_gaussian_atlas_support.detach())
        atlas_basis = gaussians.get_gaussian_atlas_basis.detach()
        atlas_class = gaussians.get_gaussian_atlas_class.detach()
        atlas_state = gaussians.get_atlas_state.detach()
        ref_camera = gaussians.get_atlas_ref_camera.detach()

        camera_obs = _build_camera_observations(camera_centers, gaussians, max_cameras=int(max_cameras))
        ref_centers = _compute_reference_centers(xyz, atlas_pos, atlas_basis, camera_obs, ref_camera)
        fallback_dirs = atlas_basis[:, :, 0]
        ref_rays = _safe_normalize(xyz - ref_centers, fallback=fallback_dirs)

        obs_hessian, obs_view_count, obs_weight_sum = _compute_observation_hessian(
            gaussians,
            xyz,
            camera_obs,
            point_chunk_size=int(point_chunk_size),
        )

        eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(xyz.shape[0], -1, -1)
        reg_hessian = _symmetrize(obs_hessian) + float(lambda_reg) * eye
        reg_inverse = _stable_psd_inverse(reg_hessian, floor=float(lambda_reg))

        active_mask = atlas_state != GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
        effective_support = atlas_support.clone()
        support_basis_all, _, _ = _decompose_support_projector(effective_support)
        basis_u = support_basis_all.clone()
        active_u = torch.zeros_like(atlas_support)
        support_minus_u = effective_support.clone()
        orth_proj = _symmetrize(eye - effective_support)
        ambiguity_score = torch.zeros((xyz.shape[0],), dtype=dtype, device=device)
        direction = basis_u[:, :, 0]

        support_guided_mask = (atlas_state == GAUSSIAN_STATE_STABLE) | (atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
        if torch.any(support_guided_mask):
            support_guided_indices = torch.nonzero(support_guided_mask, as_tuple=False).squeeze(-1)
            support_basis = support_basis_all[support_guided_mask]
            _, support_rank, support_local = _decompose_support_projector(effective_support[support_guided_mask])
            local_ambiguity = torch.bmm(
                support_basis.transpose(1, 2),
                torch.bmm(reg_inverse[support_guided_mask], support_basis),
            )
            local_ambiguity = _symmetrize(torch.bmm(support_local, torch.bmm(local_ambiguity, support_local)))
            eigvals, eigvecs = torch.linalg.eigh(local_ambiguity)
            eigvals = torch.flip(eigvals, dims=[1]).clamp_min(0.0)
            eigvecs = torch.flip(eigvecs, dims=[2])
            selected_rank = _select_active_rank(eigvals, support_rank)
            selected_rank = torch.where(
                (support_rank > 0) & (eigvals[:, 0] > 1e-10),
                selected_rank.clamp_min(1),
                torch.zeros_like(selected_rank),
            )
            local_basis = torch.bmm(support_basis, eigvecs)
            U_local = _projector_from_basis(eigvecs, selected_rank)
            support_local_minus_u = _symmetrize(support_local - U_local)
            orth_local = _symmetrize(eye[support_guided_mask] - support_local)
            active_u[support_guided_mask] = _symmetrize(torch.bmm(support_basis, torch.bmm(U_local, support_basis.transpose(1, 2))))
            support_minus_u[support_guided_mask] = _symmetrize(
                torch.bmm(support_basis, torch.bmm(support_local_minus_u, support_basis.transpose(1, 2)))
            )
            orth_proj[support_guided_mask] = _symmetrize(
                torch.bmm(support_basis, torch.bmm(orth_local, support_basis.transpose(1, 2)))
            )
            basis_u[support_guided_mask] = local_basis
            direction[support_guided_mask] = local_basis[:, :, 0]
            active_energy = []
            for rank_value in range(1, 4):
                rank_mask = selected_rank == rank_value
                if not torch.any(rank_mask):
                    continue
                active_energy.append((
                    rank_mask,
                    eigvals[rank_mask, :rank_value].mean(dim=1),
                ))
            for rank_mask, score in active_energy:
                ambiguity_score[support_guided_indices[rank_mask]] = score

        unstable_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        active_ray_fallback_count = 0.0
        active_ray_valid_count = 0.0
        if torch.any(unstable_mask):
            all_centers = camera_obs["all_centers"]
            valid_ref = (
                (ref_camera[unstable_mask] >= 0)
                & (ref_camera[unstable_mask] < all_centers.shape[0])
            ) if all_centers.ndim == 2 else torch.zeros_like(ref_camera[unstable_mask], dtype=torch.bool)
            active_ray_fallback_count = float((~valid_ref).sum().item())
            ray_dirs = ref_rays[unstable_mask]
            ray_basis = _complete_basis_from_primary(ray_dirs, atlas_basis[unstable_mask])
            ray_projector = torch.einsum("ni,nj->nij", ray_basis[:, :, 0], ray_basis[:, :, 0])
            active_u[unstable_mask] = ray_projector
            effective_support[unstable_mask] = ray_projector
            basis_u[unstable_mask] = ray_basis
            support_minus_u[unstable_mask] = torch.zeros_like(ray_projector)
            orth_proj[unstable_mask] = _symmetrize(eye[unstable_mask] - ray_projector)
            direction[unstable_mask] = ray_basis[:, :, 0]
            ambiguity_score[unstable_mask] = _projective_quadratic_form(ray_basis[:, :, 0], reg_inverse[unstable_mask]).clamp_min(0.0)
            active_ray_valid_count = float(
                (
                    torch.isfinite(ray_basis).reshape(ray_basis.shape[0], -1).all(dim=1)
                    & torch.isfinite(ray_projector).reshape(ray_projector.shape[0], -1).all(dim=1)
                ).sum().item()
            )

        zero_projector = torch.zeros_like(eye)
        active_u = _sanitize_psd_projector(active_u, fallback=zero_projector)
        effective_support = _sanitize_psd_projector(effective_support, fallback=active_u)
        support_minus_u = _sanitize_psd_projector(support_minus_u, fallback=zero_projector)
        orth_proj = _sanitize_psd_projector(orth_proj, fallback=_symmetrize(eye - active_u))

        atlas_support_rank = _round_projector_rank(atlas_support)
        support_rank = _round_projector_rank(effective_support)
        rank_u = _round_projector_rank(active_u)
        rank_s = _round_projector_rank(support_minus_u)
        rank_perp = _round_projector_rank(orth_proj)
        decomposition_error = torch.linalg.matrix_norm(
            _symmetrize(active_u + support_minus_u + orth_proj - eye),
            dim=(1, 2),
        )
        surface_mask = atlas_class == ATLAS_CLASS_SURFACE
        surface_rank1_ratio = 0.0
        surface_rank2_ratio = 0.0
        surface_rank3_ratio = 0.0
        if torch.any(surface_mask):
            surface_rank = rank_u[surface_mask]
            surface_rank1_ratio = _rank_ratio(surface_rank, 1)
            surface_rank2_ratio = _rank_ratio(surface_rank, 2)
            surface_rank3_ratio = _rank_ratio(surface_rank, 3)
        active_ray_count = float(unstable_mask.sum().item())
        point_count = float(max(int(xyz.shape[0]), 1))

        return {
            "active_mask": active_mask,
            "U": active_u,
            "basis_u": basis_u,
            "support_minus_u": support_minus_u,
            "orth_proj": orth_proj,
            "effective_support": effective_support,
            "rank_u": rank_u,
            "rank_s": rank_s,
            "rank_perp": rank_perp,
            "atlas_support_rank": atlas_support_rank,
            "direction": direction,
            "ambiguity_score": ambiguity_score,
            "obs_hessian": obs_hessian,
            "obs_view_count": obs_view_count,
            "obs_weight_sum": obs_weight_sum,
            "obs_mode": camera_obs["mode"],
            "subspace_metrics": {
                "atlas_support_rank_mean": float(atlas_support_rank.mean().detach().item()) if atlas_support_rank.numel() > 0 else 0.0,
                "atlas_effective_support_rank_mean": float(support_rank.mean().detach().item()) if support_rank.numel() > 0 else 0.0,
                "atlas_rank_u_mean": float(rank_u.mean().detach().item()) if rank_u.numel() > 0 else 0.0,
                "atlas_rank1_ratio": _rank_ratio(rank_u, 1),
                "atlas_rank2_ratio": _rank_ratio(rank_u, 2),
                "atlas_rank3_ratio": _rank_ratio(rank_u, 3),
                "atlas_surface_rank1_ratio": surface_rank1_ratio,
                "atlas_surface_rank2_ratio": surface_rank2_ratio,
                "atlas_surface_rank3_ratio": surface_rank3_ratio,
                "atlas_active_ray_count": active_ray_count,
                "atlas_active_ray_fraction": float(active_ray_count / point_count),
                "atlas_active_ray_valid_count": active_ray_valid_count,
                "atlas_active_ray_valid_fraction": float(active_ray_valid_count / max(active_ray_count, 1.0)),
                "atlas_active_ray_fallback_count": active_ray_fallback_count,
                "atlas_active_ray_fallback_fraction": float(active_ray_fallback_count / max(active_ray_count, 1.0)),
                "atlas_subspace_decomposition_error": float(decomposition_error.mean().detach().item()) if decomposition_error.numel() > 0 else 0.0,
            },
        }


def sample_antithetic_center_offsets(gaussians, subspace_info, sample_scale: float = 1.0, generator=None):
    offsets = torch.zeros_like(gaussians.get_xyz)
    metrics = {
        "atlas_mc_active_fraction": 0.0,
        "atlas_mc_mean_offset": 0.0,
        "atlas_mc_rank_u_mean": 0.0,
        "atlas_mc_rank1_ratio": 0.0,
        "atlas_mc_rank2_ratio": 0.0,
        "atlas_mc_rank3_ratio": 0.0,
    }
    if subspace_info is None:
        return offsets, metrics

    active_mask = subspace_info["active_mask"].to(device=gaussians.get_xyz.device, dtype=torch.bool)
    if not torch.any(active_mask):
        return offsets, metrics

    basis_u = subspace_info["basis_u"].to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    rank_u = subspace_info["rank_u"].to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    sigma_parallel = gaussians.get_center_sigma_parallel.squeeze(-1)
    samples = torch.randn(
        (int(active_mask.sum().item()), 3),
        dtype=gaussians.get_xyz.dtype,
        device=gaussians.get_xyz.device,
        generator=generator,
    )
    active_rank = rank_u[active_mask].round().long().clamp(0, 3)
    keep = (
        torch.arange(3, device=gaussians.get_xyz.device)
        .view(1, 3)
        < active_rank.view(-1, 1)
    ).to(dtype=gaussians.get_xyz.dtype)
    sample_coords = samples * keep
    offsets[active_mask] = torch.bmm(basis_u[active_mask], sample_coords.unsqueeze(-1)).squeeze(-1)
    offsets[active_mask] = offsets[active_mask] * sigma_parallel[active_mask].unsqueeze(-1) * float(sample_scale)
    metrics["atlas_mc_active_fraction"] = float(active_mask.float().mean().item())
    metrics["atlas_mc_mean_offset"] = float(offsets[active_mask].norm(dim=1).mean().detach().item())
    metrics["atlas_mc_rank_u_mean"] = float(rank_u[active_mask].mean().detach().item())
    metrics["atlas_mc_rank1_ratio"] = _rank_ratio(rank_u[active_mask], 1)
    metrics["atlas_mc_rank2_ratio"] = _rank_ratio(rank_u[active_mask], 2)
    metrics["atlas_mc_rank3_ratio"] = _rank_ratio(rank_u[active_mask], 3)
    return offsets, metrics


def compute_local_exact_kl(
    gaussians,
    camera_centers,
    scene_extent: float,
    weight: float,
    eps_perp: float,
    eps_tangent: float,
    lambda_parallel_base: float,
    lambda_parallel_gain: float,
    lambda_support_base: float,
    lambda_support_gain: float,
    lambda_perp_base: float,
    lambda_perp_gain: float,
    subspace_info=None,
    lambda_reg: float = 1e-3,
    max_cameras: int = 64,
    point_chunk_size: int = 2048,
    passive_state_weight: float = 1.0,
    active_state_weight: float = 1.0,
):
    zero = gaussians.get_xyz.new_zeros(())
    metrics = {
        "atlas_kl_active_fraction": 0.0,
        "atlas_kl_mean": 0.0,
        "atlas_kl_stable_mean": 0.0,
        "atlas_kl_passive_mean": 0.0,
        "atlas_kl_active_mean": 0.0,
        "atlas_sigma_parallel_mean": 0.0,
        "atlas_sigma_support_mean": 0.0,
        "atlas_sigma_perp_mean": 0.0,
        "atlas_ambiguity_mean": 0.0,
        "atlas_obs_view_mean": 0.0,
        "atlas_obs_weight_mean": 0.0,
        "atlas_rank_u_mean": 0.0,
        "atlas_rank1_ratio": 0.0,
        "atlas_rank2_ratio": 0.0,
        "atlas_rank3_ratio": 0.0,
        "atlas_support_rank_mean": 0.0,
        "atlas_effective_support_rank_mean": 0.0,
        "atlas_surface_rank1_ratio": 0.0,
        "atlas_surface_rank2_ratio": 0.0,
        "atlas_surface_rank3_ratio": 0.0,
        "atlas_active_ray_count": 0.0,
        "atlas_active_ray_fraction": 0.0,
        "atlas_active_ray_valid_count": 0.0,
        "atlas_active_ray_valid_fraction": 0.0,
        "atlas_active_ray_fallback_count": 0.0,
        "atlas_active_ray_fallback_fraction": 0.0,
        "atlas_subspace_decomposition_error": 0.0,
        "atlas_prior_precision_parallel_mean": 0.0,
        "atlas_prior_precision_support_mean": 0.0,
        "atlas_prior_precision_perp_mean": 0.0,
        "atlas_kl_state_weight_mean": 0.0,
        "atlas_kl_total_loss_safe_for_log": 0.0,
        "atlas_kl_total_loss_had_nonfinite": 0.0,
        "nonfinite_kl_count": 0.0,
        "nonfinite_subspace_metric_count": 0.0,
    }
    if not gaussians.has_atlas_bindings:
        return zero, metrics
    if weight <= 0.0:
        atlas_state = gaussians.get_atlas_state.detach()
        active_state_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        active_ray_count = float(active_state_mask.sum().item()) if active_state_mask.numel() > 0 else 0.0
        point_count = float(max(int(active_state_mask.numel()), 1))
        metrics["atlas_active_ray_count"] = active_ray_count
        metrics["atlas_active_ray_fraction"] = float(active_ray_count / point_count)
        metrics["atlas_active_ray_valid_count"] = active_ray_count
        metrics["atlas_active_ray_valid_fraction"] = float(active_ray_count / max(active_ray_count, 1.0))
        return zero, metrics

    if subspace_info is None:
        subspace_info = build_variational_subspace(
            gaussians,
            camera_centers,
            lambda_reg=lambda_reg,
            max_cameras=max_cameras,
            point_chunk_size=point_chunk_size,
        )
    if subspace_info is None:
        return zero, metrics

    active_mask = subspace_info["active_mask"].to(device=gaussians.get_xyz.device, dtype=torch.bool)
    if not torch.any(active_mask):
        return zero, metrics

    device = gaussians.get_xyz.device
    dtype = gaussians.get_xyz.dtype
    xyz = gaussians.get_xyz[active_mask]
    atlas_pos = gaussians.get_gaussian_atlas_positions[active_mask]
    reliability = _safe_positive(gaussians.get_gaussian_atlas_reliability.detach()[active_mask], 0.0, 1.0)
    atlas_state = gaussians.get_atlas_state.detach()[active_mask]
    delta = _nan_to_finite(xyz - atlas_pos, 0.0)

    eye = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(delta.shape[0], -1, -1)
    U = _sanitize_psd_projector(subspace_info["U"][active_mask].to(device=device, dtype=dtype), fallback=torch.zeros_like(eye))
    S_minus_U = _sanitize_psd_projector(
        subspace_info["support_minus_u"][active_mask].to(device=device, dtype=dtype),
        fallback=torch.zeros_like(eye),
    )
    I_minus_S = _sanitize_psd_projector(
        subspace_info["orth_proj"][active_mask].to(device=device, dtype=dtype),
        fallback=eye,
    )
    rank_u = _round_projector_rank(U).to(device=device, dtype=dtype)
    rank_s = _round_projector_rank(S_minus_U).to(device=device, dtype=dtype)
    rank_perp = _round_projector_rank(I_minus_S).to(device=device, dtype=dtype)
    ambiguity_score = _nan_to_finite(subspace_info["ambiguity_score"][active_mask].to(device=device, dtype=dtype), 0.0).clamp_min(0.0)
    obs_view_count = _nan_to_finite(subspace_info["obs_view_count"][active_mask].to(device=device, dtype=dtype), 0.0).clamp_min(0.0)
    obs_weight_sum = _nan_to_finite(subspace_info["obs_weight_sum"][active_mask].to(device=device, dtype=dtype), 0.0).clamp_min(0.0)
    state_weight = _atlas_state_weight(
        atlas_state,
        passive_state_weight=float(passive_state_weight),
        active_state_weight=float(active_state_weight),
    ).to(device=device, dtype=dtype)

    scene_extent_value = float(scene_extent)
    if (not math.isfinite(scene_extent_value)) or scene_extent_value <= 0.0:
        scene_extent_value = 1.0
    eps_perp_val = max(float(eps_perp) * scene_extent_value, 1e-6)
    eps_tangent_val = max(float(eps_tangent) * scene_extent_value, eps_perp_val)
    sigma_max = max(scene_extent_value * 10.0, eps_tangent_val * 10.0, 1e-4)
    sigma_parallel = _safe_positive(
        gaussians.get_center_sigma_parallel[active_mask].squeeze(-1),
        1e-6,
        sigma_max,
    )
    sigma_support = _safe_positive(
        gaussians.get_center_sigma_support[active_mask].squeeze(-1),
        1e-6,
        sigma_max,
    )
    sigma_perp = torch.full_like(sigma_parallel, eps_perp_val).clamp(1e-6, sigma_max)

    # Reliability stays a detached atlas prior: it only stiffens local prior
    # precision and never rewrites posterior covariance.
    prior_precision_parallel = _safe_positive(
        float(lambda_parallel_base) + float(lambda_parallel_gain) * reliability,
        1e-8,
        1e8,
    )
    prior_precision_support = _safe_positive(
        float(lambda_support_base) + float(lambda_support_gain) * reliability,
        1e-8,
        1e8,
    )
    prior_precision_perp = _safe_positive(
        float(lambda_perp_base) + float(lambda_perp_gain) * reliability,
        1e-8,
        1e8,
    )

    point_kl = torch.zeros_like(sigma_parallel)
    stable_mask = atlas_state == GAUSSIAN_STATE_STABLE
    passive_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
    active_state_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE

    if torch.any(stable_mask):
        stable_u_sq = _safe_projected_sq(U[stable_mask], delta[stable_mask])
        stable_s_sq = _safe_projected_sq(S_minus_U[stable_mask], delta[stable_mask])
        stable_perp_sq = _safe_projected_sq(I_minus_S[stable_mask], delta[stable_mask])
        point_kl[stable_mask] = (
            _scalar_center_kl(rank_u[stable_mask], sigma_parallel[stable_mask], prior_precision_parallel[stable_mask], stable_u_sq)
            + _scalar_center_kl(rank_s[stable_mask], sigma_support[stable_mask], prior_precision_support[stable_mask], stable_s_sq)
            + _scalar_center_kl(rank_perp[stable_mask], sigma_perp[stable_mask], prior_precision_perp[stable_mask], stable_perp_sq)
        )

    if torch.any(passive_mask):
        passive_sigma = torch.maximum(
            0.5 * (sigma_parallel[passive_mask] + sigma_support[passive_mask]),
            torch.full_like(sigma_parallel[passive_mask], eps_tangent_val),
        ).clamp(1e-6, sigma_max)
        passive_precision = _safe_positive(
            0.5 * (prior_precision_support[passive_mask] + prior_precision_perp[passive_mask]),
            1e-8,
            1e8,
        )
        passive_delta_sq = _nan_to_finite(delta[passive_mask].square().sum(dim=1), 0.0).clamp_min(0.0)
        point_kl[passive_mask] = _scalar_center_kl(
            torch.full_like(passive_sigma, 3.0),
            passive_sigma,
            passive_precision,
            passive_delta_sq,
        )
        rank_u[passive_mask] = 3.0
        rank_s[passive_mask] = 0.0
        rank_perp[passive_mask] = 0.0

    if torch.any(active_state_mask):
        direction = subspace_info.get("direction", None)
        if direction is not None:
            active_direction = direction[active_mask].to(device=device, dtype=dtype)
        else:
            active_direction = torch.zeros_like(delta)
            active_direction[:, 2] = 1.0
        active_direction = _safe_normalize(
            active_direction,
            fallback=delta.new_tensor([0.0, 0.0, 1.0]).expand(delta.shape[0], -1),
        )
        ray_projector = torch.einsum("ni,nj->nij", active_direction[active_state_mask], active_direction[active_state_mask])
        ray_projector = _sanitize_psd_projector(ray_projector, fallback=U[active_state_mask])
        cross_projector = _sanitize_psd_projector(eye[active_state_mask] - ray_projector, fallback=eye[active_state_mask])
        active_ray_sq = _safe_projected_sq(ray_projector, delta[active_state_mask])
        active_cross_sq = _safe_projected_sq(cross_projector, delta[active_state_mask])
        active_cross_sigma = torch.full_like(sigma_parallel[active_state_mask], eps_perp_val).clamp(1e-6, sigma_max)
        point_kl[active_state_mask] = (
            _scalar_center_kl(
                torch.ones_like(sigma_parallel[active_state_mask]),
                sigma_parallel[active_state_mask],
                prior_precision_parallel[active_state_mask],
                active_ray_sq,
            )
            + _scalar_center_kl(
                torch.full_like(sigma_parallel[active_state_mask], 2.0),
                active_cross_sigma,
                prior_precision_perp[active_state_mask],
                active_cross_sq,
            )
        )
        rank_u[active_state_mask] = 1.0
        rank_s[active_state_mask] = 0.0
        rank_perp[active_state_mask] = 2.0

    nonfinite_point_kl_count = float((~torch.isfinite(point_kl.detach())).sum().item()) if point_kl.numel() > 0 else 0.0
    point_kl = _nan_to_finite(point_kl, 0.0).clamp_min(0.0)
    kl = _weighted_mean(point_kl, state_weight, zero)
    total = float(weight) * kl
    metrics["atlas_kl_active_fraction"], _ = _safe_metric_scalar(active_mask.float().mean())
    metrics["atlas_kl_mean"], _ = _safe_metric_scalar(kl)
    metrics["atlas_kl_total_loss_safe_for_log"], metrics["atlas_kl_total_loss_had_nonfinite"] = _safe_metric_scalar(total)
    metrics["nonfinite_kl_count"] = float(metrics["atlas_kl_total_loss_had_nonfinite"]) + nonfinite_point_kl_count
    if torch.any(stable_mask):
        metrics["atlas_kl_stable_mean"], _ = _safe_metric_scalar(point_kl[stable_mask].mean())
    if torch.any(passive_mask):
        metrics["atlas_kl_passive_mean"], _ = _safe_metric_scalar(point_kl[passive_mask].mean())
    if torch.any(active_state_mask):
        metrics["atlas_kl_active_mean"], _ = _safe_metric_scalar(point_kl[active_state_mask].mean())
    metrics["atlas_sigma_parallel_mean"], _ = _safe_metric_scalar(sigma_parallel.mean())
    metrics["atlas_sigma_support_mean"], _ = _safe_metric_scalar(sigma_support.mean())
    metrics["atlas_sigma_perp_mean"], _ = _safe_metric_scalar(sigma_perp.mean())
    metrics["atlas_ambiguity_mean"], _ = _safe_metric_scalar(ambiguity_score.mean())
    metrics["atlas_obs_view_mean"], _ = _safe_metric_scalar(obs_view_count.mean())
    metrics["atlas_obs_weight_mean"], _ = _safe_metric_scalar(obs_weight_sum.mean())
    metrics["atlas_rank_u_mean"], _ = _safe_metric_scalar(rank_u.mean())
    metrics["atlas_rank1_ratio"], _ = _safe_metric_scalar(_rank_ratio(rank_u, 1))
    metrics["atlas_rank2_ratio"], _ = _safe_metric_scalar(_rank_ratio(rank_u, 2))
    metrics["atlas_rank3_ratio"], _ = _safe_metric_scalar(_rank_ratio(rank_u, 3))
    metrics["atlas_prior_precision_parallel_mean"], _ = _safe_metric_scalar(prior_precision_parallel.mean())
    metrics["atlas_prior_precision_support_mean"], _ = _safe_metric_scalar(prior_precision_support.mean())
    metrics["atlas_prior_precision_perp_mean"], _ = _safe_metric_scalar(prior_precision_perp.mean())
    metrics["atlas_kl_state_weight_mean"], _ = _safe_metric_scalar(
        state_weight.mean() if state_weight.numel() > 0 else 0.0
    )
    subspace_metrics = subspace_info.get("subspace_metrics", {})
    subspace_nonfinite_count = 0.0
    for metric_name in (
        "atlas_support_rank_mean",
        "atlas_effective_support_rank_mean",
        "atlas_surface_rank1_ratio",
        "atlas_surface_rank2_ratio",
        "atlas_surface_rank3_ratio",
        "atlas_active_ray_count",
        "atlas_active_ray_fraction",
        "atlas_active_ray_valid_count",
        "atlas_active_ray_valid_fraction",
        "atlas_active_ray_fallback_count",
        "atlas_active_ray_fallback_fraction",
        "atlas_subspace_decomposition_error",
    ):
        metrics[metric_name], had_nonfinite = _safe_metric_scalar(subspace_metrics.get(metric_name, 0.0))
        subspace_nonfinite_count += float(had_nonfinite)
    metrics["nonfinite_subspace_metric_count"] = float(subspace_nonfinite_count)
    return total, metrics
