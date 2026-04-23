from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from plyfile import PlyData, PlyElement

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


ATLAS_CLASS_SURFACE = 0
ATLAS_CLASS_EDGE = 1
ATLAS_CLASS_UNSTABLE = 2


@dataclass
class FoundationAtlas:
    positions: np.ndarray
    support: np.ndarray
    basis: np.ndarray
    normal: np.ndarray
    radius: np.ndarray
    raw_score: np.ndarray
    reliability: np.ndarray
    atlas_class: np.ndarray
    anisotropy_ref: np.ndarray
    neighbor_indices: np.ndarray
    calibration_residual: np.ndarray
    node_confidence: np.ndarray
    point_support: np.ndarray
    view_support: np.ndarray
    view_coverage: np.ndarray
    support_score: np.ndarray
    linearness: np.ndarray
    planarness: np.ndarray
    scattering: np.ndarray
    structure_score: np.ndarray | None = None
    scale_consistency: np.ndarray | None = None
    class_consistency: np.ndarray | None = None
    support_consistency: np.ndarray | None = None
    view_balance: np.ndarray | None = None
    view_outlier_score: np.ndarray | None = None
    robust_inlier_ratio: np.ndarray | None = None
    unstable_reason_code: np.ndarray | None = None


UNSTABLE_REASON_LOW_STRUCTURE = 0
UNSTABLE_REASON_LOW_COVERAGE = 1
UNSTABLE_REASON_SUPPORT_INCONSISTENCY = 2
UNSTABLE_REASON_HIGH_CALIBRATION_RESIDUAL = 3

UNSTABLE_REASON_NAMES = {
    UNSTABLE_REASON_LOW_STRUCTURE: "low_structure",
    UNSTABLE_REASON_LOW_COVERAGE: "low_coverage",
    UNSTABLE_REASON_SUPPORT_INCONSISTENCY: "support_inconsistency",
    UNSTABLE_REASON_HIGH_CALIBRATION_RESIDUAL: "high_calibration_residual",
}


def _require_matplotlib():
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting foundation atlas outputs.")


def _normalize_scores(values: np.ndarray, low_q=0.05, high_q=0.95) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values

    low = float(np.quantile(values, low_q))
    high = float(np.quantile(values, high_q))
    if not np.isfinite(low):
        low = 0.0
    if not np.isfinite(high):
        high = 1.0

    if high <= low:
        return np.ones_like(values, dtype=np.float32)

    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _normalize_confidence(
    confidences: np.ndarray,
    image_ids: np.ndarray | None = None,
    parallax_scores: np.ndarray | None = None,
    depth_consistency: np.ndarray | None = None,
) -> np.ndarray:
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if confidences.size == 0:
        return confidences

    log_confidences = np.log1p(np.clip(confidences, 0.0, None))
    global_score = _normalize_scores(log_confidences)
    if image_ids is None:
        confidence_score = global_score
    else:
        image_ids = np.asarray(image_ids, dtype=np.int32).reshape(-1)
        if image_ids.shape[0] != confidences.shape[0]:
            raise ValueError("image_ids must match confidences for per-view normalization.")

        per_view_score = np.zeros_like(global_score, dtype=np.float32)
        for image_id in np.unique(image_ids):
            mask = image_ids == image_id
            if not np.any(mask):
                continue
            per_view_score[mask] = _normalize_scores(log_confidences[mask], low_q=0.10, high_q=0.90)

        confidence_score = np.clip(0.65 * global_score + 0.35 * per_view_score, 0.0, 1.0).astype(np.float32)

    if parallax_scores is None or depth_consistency is None:
        return confidence_score.astype(np.float32)

    parallax_scores = np.asarray(parallax_scores, dtype=np.float32).reshape(-1)
    depth_consistency = np.asarray(depth_consistency, dtype=np.float32).reshape(-1)
    if parallax_scores.shape[0] != confidences.shape[0] or depth_consistency.shape[0] != confidences.shape[0]:
        return confidence_score.astype(np.float32)

    return np.clip(
        0.55 * confidence_score
        + 0.25 * np.clip(parallax_scores, 0.0, 1.0)
        + 0.20 * np.clip(depth_consistency, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)


def _atlas_metric_or_default(atlas: FoundationAtlas, field_name: str, default_value: float) -> np.ndarray:
    value = getattr(atlas, field_name, None)
    if value is None:
        return np.full((atlas.positions.shape[0],), float(default_value), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] != atlas.positions.shape[0]:
        return np.full((atlas.positions.shape[0],), float(default_value), dtype=np.float32)
    return array.astype(np.float32)


def _compute_unstable_reason_payload(
    atlas: FoundationAtlas,
    *,
    structure_score: np.ndarray | None = None,
    support_consistency: np.ndarray | None = None,
    scale_consistency: np.ndarray | None = None,
    class_consistency: np.ndarray | None = None,
):
    structure_score = (
        np.asarray(structure_score, dtype=np.float32).reshape(-1)
        if structure_score is not None
        else _atlas_metric_or_default(atlas, "structure_score", 0.0)
    )
    support_consistency = (
        np.asarray(support_consistency, dtype=np.float32).reshape(-1)
        if support_consistency is not None
        else _atlas_metric_or_default(atlas, "support_consistency", 1.0)
    )
    scale_consistency = (
        np.asarray(scale_consistency, dtype=np.float32).reshape(-1)
        if scale_consistency is not None
        else _atlas_metric_or_default(atlas, "scale_consistency", 1.0)
    )
    class_consistency = (
        np.asarray(class_consistency, dtype=np.float32).reshape(-1)
        if class_consistency is not None
        else _atlas_metric_or_default(atlas, "class_consistency", 1.0)
    )

    view_coverage = np.asarray(atlas.view_coverage, dtype=np.float32).reshape(-1)
    support_score = np.asarray(atlas.support_score, dtype=np.float32).reshape(-1)
    calibration_residual = np.asarray(atlas.calibration_residual, dtype=np.float32).reshape(-1)
    atlas_class = np.asarray(atlas.atlas_class, dtype=np.int64).reshape(-1)

    reason_scores = np.stack(
        [
            np.clip(1.0 - structure_score, 0.0, 1.0),
            np.clip(1.0 - np.maximum(view_coverage, support_score), 0.0, 1.0),
            np.clip(1.0 - support_consistency, 0.0, 1.0),
            np.clip(calibration_residual, 0.0, 1.0),
        ],
        axis=1,
    ).astype(np.float32)

    reason_codes = np.argmax(reason_scores, axis=1).astype(np.int32)
    reason_codes[atlas_class != ATLAS_CLASS_UNSTABLE] = -1
    return reason_codes, reason_scores, {
        "structure_score": structure_score,
        "support_consistency": support_consistency,
        "scale_consistency": scale_consistency,
        "class_consistency": class_consistency,
        "view_coverage": view_coverage,
        "support_score": support_score,
    }


def build_unstable_node_audit(atlas: FoundationAtlas):
    reason_codes, reason_scores, metrics = _compute_unstable_reason_payload(atlas)
    unstable_indices = np.flatnonzero(np.asarray(atlas.atlas_class, dtype=np.int64) == ATLAS_CLASS_UNSTABLE)
    records = []
    counts = {name: 0 for name in UNSTABLE_REASON_NAMES.values()}

    for node_index in unstable_indices.tolist():
        reason_code = int(reason_codes[node_index])
        reason_name = UNSTABLE_REASON_NAMES.get(reason_code, "unknown")
        counts[reason_name] = counts.get(reason_name, 0) + 1
        records.append(
            {
                "node_index": int(node_index),
                "reason": reason_name,
                "reason_score": float(reason_scores[node_index, max(reason_code, 0)]),
                "reliability": float(atlas.reliability[node_index]),
                "raw_score": float(atlas.raw_score[node_index]),
                "calibration_residual": float(atlas.calibration_residual[node_index]),
                "structure_score": float(metrics["structure_score"][node_index]),
                "view_coverage": float(metrics["view_coverage"][node_index]),
                "support_score": float(metrics["support_score"][node_index]),
                "support_consistency": float(metrics["support_consistency"][node_index]),
                "scale_consistency": float(metrics["scale_consistency"][node_index]),
                "class_consistency": float(metrics["class_consistency"][node_index]),
            }
        )

    records.sort(key=lambda item: (item["reason"], item["reliability"], -item["calibration_residual"]))
    return {
        "unstable_count": int(len(records)),
        "reason_counts": counts,
        "records": records,
    }


def _select_voxel_representatives(
    points: np.ndarray,
    confidences: np.ndarray,
    inverse: np.ndarray,
    num_voxels: int,
    centroid_positions: np.ndarray,
    image_ids: np.ndarray | None = None,
    voxel_size: float = 0.0,
    view_outlier_score: np.ndarray | None = None,
    robust_inlier_ratio: np.ndarray | None = None,
    return_secondary: bool = False,
):
    if num_voxels <= 0:
        empty = centroid_positions.astype(np.float32)
        ones = np.ones((empty.shape[0],), dtype=np.float32)
        if return_secondary:
            return empty, ones, ones, ones, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        return empty, ones, ones, ones

    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
    groups = np.split(order, boundaries)

    representatives = np.empty_like(centroid_positions, dtype=np.float32)
    voxel_support_consistency = np.ones((num_voxels,), dtype=np.float32)
    view_balance = np.ones((num_voxels,), dtype=np.float32)
    confidence_peak = np.ones((num_voxels,), dtype=np.float32)
    secondary_indices = []
    secondary_scores = []
    voxel_scale = max(float(voxel_size), 1e-6)
    for voxel_id, point_indices in enumerate(groups):
        if point_indices.size == 0:
            representatives[voxel_id] = centroid_positions[voxel_id]
            continue
        if point_indices.size == 1:
            representatives[voxel_id] = points[point_indices[0]]
            confidence_peak[voxel_id] = float(np.clip(confidences[point_indices[0]], 0.0, 1.0))
            continue

        group_points = points[point_indices]
        group_conf = np.clip(confidences[point_indices], 0.0, 1.0)
        confidence_peak[voxel_id] = float(group_conf.max())
        center = centroid_positions[voxel_id]
        offsets = group_points - center[None, :]
        d2 = np.sum(offsets ** 2, axis=1)
        conf_weights = np.clip(group_conf, 1e-3, None)
        spread = float(np.sqrt(np.average(d2, weights=conf_weights)))
        voxel_support_consistency[voxel_id] = float(
            np.clip(math.exp(-spread / max(0.85 * voxel_scale, 1e-6)), 0.0, 1.0)
        )
        distance = np.sqrt(np.clip(d2, 0.0, None)).astype(np.float32)
        max_distance = max(float(distance.max()) if distance.size else 0.0, 1e-6)
        detail_offset = np.clip(distance / max_distance, 0.0, 1.0).astype(np.float32)

        if image_ids is not None:
            local_view_ids = image_ids[point_indices]
            unique_views, inverse_views, local_view_counts = np.unique(
                local_view_ids,
                return_inverse=True,
                return_counts=True,
            )
            view_balance[voxel_id] = float(
                np.clip(unique_views.size / max(float(point_indices.size), 1.0), 0.0, 1.0)
            )
            view_rarity = 1.0 / np.maximum(local_view_counts[inverse_views].astype(np.float32), 1.0)
            view_rarity = view_rarity / max(float(view_rarity.max()), 1e-6)
        else:
            view_rarity = np.ones_like(group_conf, dtype=np.float32)

        centrality = np.exp(-d2 / max(2.0 * (0.65 * voxel_scale) ** 2, 1e-6)).astype(np.float32)
        selection_score = (
            0.55 * centrality
            + 0.25 * group_conf
            + 0.20 * view_rarity.astype(np.float32)
        ) * (0.65 + 0.35 * voxel_support_consistency[voxel_id])
        primary_local = int(np.argmax(selection_score))
        representatives[voxel_id] = group_points[primary_local]

        if not return_secondary or point_indices.size < 3:
            continue

        centered = group_points - group_points.mean(axis=0, keepdims=True)
        try:
            covariance = np.cov(centered.T, aweights=conf_weights)
            eigenvalues = np.sort(np.linalg.eigvalsh(covariance).astype(np.float32))[::-1]
            edge_likeness = float((eigenvalues[0] - eigenvalues[1]) / max(float(eigenvalues[0]), 1e-6))
        except (ValueError, np.linalg.LinAlgError):
            edge_likeness = 0.0
        edge_likeness = float(np.clip(edge_likeness, 0.0, 1.0))
        span_ratio = max_distance / voxel_scale
        outlier_ok = True
        if view_outlier_score is not None:
            outlier_ok = outlier_ok and float(view_outlier_score[voxel_id]) >= 0.25
        if robust_inlier_ratio is not None:
            outlier_ok = outlier_ok or float(robust_inlier_ratio[voxel_id]) >= 0.35
        multi_structure = outlier_ok and (
            (span_ratio > 0.55 and voxel_support_consistency[voxel_id] < 0.88)
            or edge_likeness > 0.35
            or (view_balance[voxel_id] > 0.30 and span_ratio > 0.42)
        )
        if not multi_structure:
            continue

        secondary_score = (
            (0.45 + 0.55 * group_conf)
            * (0.55 + 0.45 * view_rarity.astype(np.float32))
            * (0.50 + 0.50 * edge_likeness)
            * (0.45 + 0.55 * detail_offset)
        ).astype(np.float32)
        primary_distance = np.linalg.norm(group_points - group_points[primary_local][None, :], axis=1)
        secondary_score[primary_local] = -np.inf
        secondary_score[primary_distance <= 0.18 * voxel_scale] = -np.inf
        secondary_local = int(np.argmax(secondary_score))
        if np.isfinite(secondary_score[secondary_local]):
            secondary_indices.append(int(point_indices[secondary_local]))
            secondary_scores.append(float(np.clip(secondary_score[secondary_local], 0.0, 1.0)))

    result = (
        representatives.astype(np.float32),
        voxel_support_consistency.astype(np.float32),
        view_balance.astype(np.float32),
        confidence_peak.astype(np.float32),
    )
    if return_secondary:
        return result + (
            np.asarray(secondary_indices, dtype=np.int64),
            np.asarray(secondary_scores, dtype=np.float32),
        )
    return result


def _aggregate_group(values: np.ndarray, inverse: np.ndarray, count: int, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    aggregated = np.zeros((count, values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        aggregated[:, dim] = np.bincount(inverse, weights=values[:, dim] * weights, minlength=count)
    return aggregated


def _groups_from_inverse(inverse: np.ndarray, count: int):
    if int(count) <= 0:
        return []
    order = np.argsort(inverse, kind="stable")
    sorted_inverse = inverse[order]
    boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
    groups = np.split(order, boundaries)
    if len(groups) < int(count):
        by_id = [np.zeros((0,), dtype=np.int64) for _ in range(int(count))]
        for group in groups:
            if group.size > 0:
                by_id[int(inverse[group[0]])] = group
        return by_id
    return groups


def _view_balanced_observation_weights(
    confidences: np.ndarray,
    inverse: np.ndarray,
    count: int,
    image_ids: np.ndarray | None,
) -> np.ndarray:
    weights = np.power(np.clip(confidences, 1e-4, None), 0.45).astype(np.float32)
    if image_ids is None or weights.size == 0:
        return weights

    groups = _groups_from_inverse(inverse, count)
    for group in groups:
        if group.size <= 1:
            continue
        local_views = image_ids[group]
        unique_views, inverse_views, local_counts = np.unique(
            local_views,
            return_inverse=True,
            return_counts=True,
        )
        if unique_views.size <= 0:
            continue
        # Use per-view rarity plus a hard view-sum cap so one dense view cannot
        # dominate a voxel that has weaker but geometrically consistent support
        # from other views.
        rarity = 1.0 / np.maximum(local_counts[inverse_views].astype(np.float32), 1.0)
        rarity = rarity / max(float(np.max(rarity)), 1e-6)
        weights[group] *= rarity.astype(np.float32)

        view_sums = np.zeros((unique_views.size,), dtype=np.float32)
        np.add.at(view_sums, inverse_views, weights[group])
        positive = view_sums > 0.0
        if np.count_nonzero(positive) <= 1:
            continue
        cap = float(np.median(view_sums[positive]) * 1.25)
        if cap <= 0.0:
            continue
        view_scale = np.ones_like(view_sums, dtype=np.float32)
        view_scale[positive] = np.minimum(1.0, cap / np.clip(view_sums[positive], 1e-6, None))
        weights[group] *= view_scale[inverse_views]

    return np.clip(weights, 1e-6, None).astype(np.float32)


def _apply_voxel_outlier_penalty(
    points: np.ndarray,
    inverse: np.ndarray,
    count: int,
    base_weights: np.ndarray,
    centers: np.ndarray,
    image_ids: np.ndarray | None,
    voxel_size: float,
):
    weights = np.asarray(base_weights, dtype=np.float32).copy()
    view_outlier_score = np.ones((count,), dtype=np.float32)
    robust_inlier_ratio = np.ones((count,), dtype=np.float32)
    if points.shape[0] == 0 or int(count) <= 0:
        return weights, view_outlier_score, robust_inlier_ratio

    groups = _groups_from_inverse(inverse, count)
    voxel_scale = max(float(voxel_size), 1e-6)
    for voxel_id, group in enumerate(groups):
        if group.size <= 2:
            continue
        group_points = points[group]
        center = centers[voxel_id]
        dist = np.linalg.norm(group_points - center[None, :], axis=1).astype(np.float32)
        finite = np.isfinite(dist)
        if not np.any(finite):
            weights[group] *= 0.1
            view_outlier_score[voxel_id] = 0.0
            robust_inlier_ratio[voxel_id] = 0.0
            continue

        finite_dist = dist[finite]
        median_dist = float(np.median(finite_dist))
        mad = float(np.median(np.abs(finite_dist - median_dist)))
        robust_scale = max(1.4826 * mad, 0.50 * median_dist, 0.35 * voxel_scale, 1e-6)
        inlier_weight = 1.0 / (1.0 + np.power(dist / (2.5 * robust_scale), 4.0))
        inlier_weight = np.where(finite, inlier_weight, 0.0).astype(np.float32)

        if image_ids is not None:
            local_views = image_ids[group]
            unique_views, inverse_views = np.unique(local_views, return_inverse=True)
            view_median = np.zeros((unique_views.size,), dtype=np.float32)
            view_count = np.bincount(inverse_views, minlength=unique_views.size).astype(np.float32)
            for local_view in range(unique_views.size):
                view_mask = inverse_views == local_view
                view_median[local_view] = float(np.median(dist[view_mask])) if np.any(view_mask) else median_dist
            excess = np.maximum(view_median - median_dist, 0.0)
            view_score = np.exp(-np.square(excess / max(1.5 * robust_scale, 0.35 * voxel_scale, 1e-6))).astype(np.float32)
            absolute_score = np.exp(-view_median / max(3.0 * robust_scale, 0.75 * voxel_scale, 1e-6)).astype(np.float32)
            view_score = np.clip(0.70 * view_score + 0.30 * absolute_score, 0.0, 1.0)
            inlier_weight *= np.clip(view_score[inverse_views], 0.02, 1.0)
            dense_view_cap = np.minimum(view_count, max(float(np.median(view_count[view_count > 0.0])) if np.any(view_count > 0.0) else 1.0, 1.0))
            view_outlier_score[voxel_id] = float(np.clip(np.average(view_score, weights=np.maximum(dense_view_cap, 1.0)), 0.0, 1.0))
        else:
            view_outlier_score[voxel_id] = float(np.clip(np.mean(inlier_weight), 0.0, 1.0))

        robust_inlier_ratio[voxel_id] = float(np.mean(dist <= (2.5 * robust_scale)))
        weights[group] *= np.clip(inlier_weight, 0.05, 1.0)

    return np.clip(weights, 1e-6, None).astype(np.float32), view_outlier_score, robust_inlier_ratio


def _aggregate_voxel_candidates(
    points,
    colors,
    confidences,
    image_ids,
    max_nodes,
    candidate_oversample,
    target_voxel_size=0.0,
    parallax_scores=None,
    depth_consistency=None,
):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    input_colors = colors
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    image_ids = None if image_ids is None else np.asarray(image_ids, dtype=np.int32).reshape(-1)
    if parallax_scores is not None:
        parallax_scores = np.asarray(parallax_scores, dtype=np.float32).reshape(-1)
        if parallax_scores.shape[0] != confidences.shape[0]:
            parallax_scores = None
    if depth_consistency is not None:
        depth_consistency = np.asarray(depth_consistency, dtype=np.float32).reshape(-1)
        if depth_consistency.shape[0] != confidences.shape[0]:
            depth_consistency = None
    calibrated_confidences = _normalize_confidence(
        confidences,
        image_ids=image_ids,
        parallax_scores=parallax_scores,
        depth_consistency=depth_consistency,
    )
    parallax_values = (
        np.clip(parallax_scores, 0.0, 1.0).astype(np.float32)
        if parallax_scores is not None
        else np.ones_like(calibrated_confidences, dtype=np.float32)
    )
    depth_values = (
        np.clip(depth_consistency, 0.0, 1.0).astype(np.float32)
        if depth_consistency is not None
        else np.ones_like(calibrated_confidences, dtype=np.float32)
    )

    if points.shape[0] <= max_nodes:
        point_support = np.ones((points.shape[0],), dtype=np.int32)
        if image_ids is None:
            view_support = np.ones((points.shape[0],), dtype=np.int32)
        else:
            view_support = np.ones((points.shape[0],), dtype=np.int32)
        return {
            "positions": points,
            "colors": colors,
            "confidences": calibrated_confidences,
            "point_support": point_support,
            "view_support": view_support,
            "view_balance": np.ones((points.shape[0],), dtype=np.float32),
            "voxel_support_consistency": np.ones((points.shape[0],), dtype=np.float32),
            "view_outlier_score": np.ones((points.shape[0],), dtype=np.float32),
            "robust_inlier_ratio": np.ones((points.shape[0],), dtype=np.float32),
            "confidence_peak": calibrated_confidences.astype(np.float32),
            "parallax_score": parallax_values.astype(np.float32),
            "depth_consistency": depth_values.astype(np.float32),
            "detail_score": np.zeros((points.shape[0],), dtype=np.float32),
            "voxel_size": 0.0,
        }

    target_candidates = max(int(max_nodes), int(round(max_nodes * candidate_oversample)))
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    span = np.clip(bbox_max - bbox_min, 1e-6, None)
    diag = float(np.linalg.norm(span))
    if float(target_voxel_size) > 0.0:
        voxel_size = max(float(target_voxel_size), diag / 1024.0, 1e-5)
    else:
        voxel_size = max(float(np.cbrt(np.prod(span) / max(target_candidates, 1))), diag / 1024.0, 1e-5)

    best = None
    best_gap = None
    num_views = int(image_ids.max()) + 1 if image_ids is not None and image_ids.size else 1

    for _ in range(1 if float(target_voxel_size) > 0.0 else 14):
        coords = np.floor((points - bbox_min) / voxel_size).astype(np.int32)
        _, inverse, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
        num_voxels = int(counts.shape[0])
        gap = abs(num_voxels - target_candidates)
        if best is None or gap < best_gap:
            best = (inverse.copy(), counts.copy(), voxel_size)
            best_gap = gap

        if target_candidates * 0.75 <= num_voxels <= target_candidates * 1.30:
            break
        if num_voxels > target_candidates:
            voxel_size *= 1.25
        else:
            voxel_size *= 0.8

    inverse, counts, voxel_size = best
    num_voxels = int(counts.shape[0])

    weights = _view_balanced_observation_weights(calibrated_confidences, inverse, num_voxels, image_ids)
    weight_sums = np.bincount(inverse, weights=weights, minlength=num_voxels).astype(np.float32)
    weight_sums = np.clip(weight_sums, 1e-6, None)

    centroid_positions = _aggregate_group(points, inverse, num_voxels, weights) / weight_sums[:, None]
    weights, view_outlier_score, robust_inlier_ratio = _apply_voxel_outlier_penalty(
        points,
        inverse,
        num_voxels,
        weights,
        centroid_positions,
        image_ids,
        voxel_size,
    )
    weight_sums = np.bincount(inverse, weights=weights, minlength=num_voxels).astype(np.float32)
    weight_sums = np.clip(weight_sums, 1e-6, None)
    centroid_positions = _aggregate_group(points, inverse, num_voxels, weights) / weight_sums[:, None]
    (
        positions,
        voxel_support_consistency,
        view_balance,
        confidence_peak,
        secondary_indices,
        secondary_scores,
    ) = _select_voxel_representatives(
        points,
        calibrated_confidences,
        inverse,
        num_voxels,
        centroid_positions,
        image_ids=image_ids,
        voxel_size=voxel_size,
        view_outlier_score=view_outlier_score,
        robust_inlier_ratio=robust_inlier_ratio,
        return_secondary=True,
    )
    colors = _aggregate_group(colors, inverse, num_voxels, weights) / weight_sums[:, None]
    confidence_sum = np.bincount(inverse, weights=calibrated_confidences * weights, minlength=num_voxels).astype(np.float32)
    confidences = confidence_sum / weight_sums
    parallax_sum = np.bincount(inverse, weights=parallax_values * weights, minlength=num_voxels).astype(np.float32)
    depth_sum = np.bincount(inverse, weights=depth_values * weights, minlength=num_voxels).astype(np.float32)
    candidate_parallax = parallax_sum / weight_sums
    candidate_depth = depth_sum / weight_sums
    point_support = counts.astype(np.int32)

    if image_ids is None:
        view_support = np.ones((num_voxels,), dtype=np.int32)
    else:
        encoded = inverse.astype(np.int64) * max(num_views, 1) + image_ids.astype(np.int64)
        unique_pairs = np.unique(encoded)
        voxel_ids = unique_pairs // max(num_views, 1)
        view_support = np.bincount(voxel_ids, minlength=num_voxels).astype(np.int32)

    detail_score = np.clip(
        0.20 * confidence_peak
        + 0.25 * view_balance
        + 0.25 * robust_inlier_ratio
        + 0.30 * (1.0 - voxel_support_consistency),
        0.0,
        1.0,
    ).astype(np.float32) * 0.45

    max_secondary = max(0, min(int(secondary_indices.shape[0]), int(round(max_nodes * 0.75))))
    if max_secondary > 0:
        if secondary_indices.shape[0] > max_secondary:
            keep_secondary = np.argsort(secondary_scores)[::-1][:max_secondary]
            secondary_indices = secondary_indices[keep_secondary]
            secondary_scores = secondary_scores[keep_secondary]
        secondary_voxels = inverse[secondary_indices].astype(np.int64)
        positions = np.concatenate([positions, points[secondary_indices].astype(np.float32)], axis=0)
        colors = np.concatenate([colors, input_colors[secondary_indices].astype(np.float32)], axis=0)
        confidences = np.concatenate([confidences, calibrated_confidences[secondary_indices].astype(np.float32)], axis=0)
        point_support = np.concatenate([point_support, counts[secondary_voxels].astype(np.int32)], axis=0)
        view_support = np.concatenate([view_support, view_support[secondary_voxels].astype(np.int32)], axis=0)
        view_balance = np.concatenate([view_balance, view_balance[secondary_voxels].astype(np.float32)], axis=0)
        voxel_support_consistency = np.concatenate(
            [
                voxel_support_consistency,
                np.clip(
                    0.50 * voxel_support_consistency[secondary_voxels]
                    + 0.50 * secondary_scores.astype(np.float32),
                    0.0,
                    1.0,
                ),
            ],
            axis=0,
        )
        view_outlier_score = np.concatenate([view_outlier_score, view_outlier_score[secondary_voxels].astype(np.float32)], axis=0)
        robust_inlier_ratio = np.concatenate([robust_inlier_ratio, robust_inlier_ratio[secondary_voxels].astype(np.float32)], axis=0)
        confidence_peak = np.concatenate(
            [
                confidence_peak,
                np.maximum(confidence_peak[secondary_voxels], calibrated_confidences[secondary_indices]).astype(np.float32),
            ],
            axis=0,
        )
        candidate_parallax = np.concatenate([candidate_parallax, parallax_values[secondary_indices].astype(np.float32)], axis=0)
        candidate_depth = np.concatenate([candidate_depth, depth_values[secondary_indices].astype(np.float32)], axis=0)
        detail_score = np.concatenate([detail_score, secondary_scores.astype(np.float32)], axis=0)

    return {
        "positions": positions.astype(np.float32),
        "colors": colors.astype(np.float32),
        "confidences": confidences.astype(np.float32),
        "point_support": point_support,
        "view_support": view_support,
        "view_balance": view_balance.astype(np.float32),
        "voxel_support_consistency": voxel_support_consistency.astype(np.float32),
        "view_outlier_score": view_outlier_score.astype(np.float32),
        "robust_inlier_ratio": robust_inlier_ratio.astype(np.float32),
        "confidence_peak": confidence_peak.astype(np.float32),
        "parallax_score": candidate_parallax.astype(np.float32),
        "depth_consistency": candidate_depth.astype(np.float32),
        "detail_score": detail_score.astype(np.float32),
        "voxel_size": float(voxel_size),
    }


def _chunked_self_knn(points: torch.Tensor, k_neighbors: int, chunk_size: int):
    num_points = int(points.shape[0])
    if num_points <= 1:
        zeros_idx = torch.zeros((num_points, 1), dtype=torch.long, device=points.device)
        zeros_dist = torch.zeros((num_points, 1), dtype=points.dtype, device=points.device)
        return zeros_idx, zeros_dist

    effective_k = max(1, min(int(k_neighbors), num_points - 1))
    all_indices = []
    all_distances = []
    inf = torch.tensor(float("inf"), dtype=points.dtype, device=points.device)

    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        queries = points[start:end]
        distances = torch.cdist(queries, points)
        local_indices = torch.arange(end - start, device=points.device)
        global_indices = torch.arange(start, end, device=points.device)
        distances[local_indices, global_indices] = inf
        knn_distances, knn_indices = torch.topk(distances, k=effective_k, largest=False, dim=1)
        all_indices.append(knn_indices)
        all_distances.append(knn_distances)

    return torch.cat(all_indices, dim=0), torch.cat(all_distances, dim=0)


def _make_single_point_geometry(positions: torch.Tensor):
    basis = torch.eye(3, dtype=positions.dtype, device=positions.device).unsqueeze(0)
    zeros = torch.zeros((1,), dtype=positions.dtype, device=positions.device)
    return {
        "positions": positions,
        "knn_indices": torch.zeros((1, 1), dtype=torch.long, device=positions.device),
        "knn_distances": torch.zeros((1, 1), dtype=positions.dtype, device=positions.device),
        "eigenvalues": torch.ones((1, 3), dtype=positions.dtype, device=positions.device) * 1e-6,
        "eigenvectors": basis,
        "linearness": zeros,
        "planarness": zeros,
        "scattering": torch.ones((1,), dtype=positions.dtype, device=positions.device),
        "surface_metric": torch.ones((1,), dtype=positions.dtype, device=positions.device),
        "edge_metric": torch.ones((1,), dtype=positions.dtype, device=positions.device),
        "radius": torch.ones((1,), dtype=positions.dtype, device=positions.device),
    }


def _compute_geometry_from_weights(
    positions: torch.Tensor,
    knn_indices: torch.Tensor,
    knn_distances: torch.Tensor,
    neighbor_weights: torch.Tensor,
):
    neighbor_positions = positions[knn_indices]
    weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    neighbor_mean = (weights.unsqueeze(-1) * neighbor_positions).sum(dim=1)
    centered = neighbor_positions - neighbor_mean.unsqueeze(1)
    covariance = torch.einsum("nk,nki,nkj->nij", weights, centered, centered)
    covariance = covariance + torch.eye(3, device=positions.device, dtype=positions.dtype).unsqueeze(0) * 1e-6

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = torch.flip(eigenvalues, dims=[1])
    eigenvectors = torch.flip(eigenvectors, dims=[2])

    lambda1 = eigenvalues[:, 0].clamp_min(1e-6)
    lambda2 = eigenvalues[:, 1].clamp_min(1e-6)
    lambda3 = eigenvalues[:, 2].clamp_min(1e-6)
    linearness = ((lambda1 - lambda2) / lambda1).clamp(0.0, 1.0)
    planarness = ((lambda2 - lambda3) / lambda1).clamp(0.0, 1.0)
    scattering = (lambda3 / lambda1).clamp(0.0, 1.0)
    surface_metric = (lambda3 / lambda2).clamp_min(0.0)
    edge_metric = (lambda2 / lambda1).clamp_min(0.0)
    radius = knn_distances[:, -1].clamp_min(1e-6)

    return {
        "positions": positions,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "linearness": linearness,
        "planarness": planarness,
        "scattering": scattering,
        "surface_metric": surface_metric,
        "edge_metric": edge_metric,
        "radius": radius,
    }


def _compute_geometry_from_neighbors(positions: torch.Tensor, knn_indices: torch.Tensor, knn_distances: torch.Tensor):
    distance_scale = knn_distances.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weights = torch.exp(-(knn_distances ** 2) / (distance_scale ** 2))
    return _compute_geometry_from_weights(positions, knn_indices, knn_distances, weights)


def _classify_geometry(linearness, planarness, scattering, surface_metric, edge_metric, surface_ratio, edge_ratio):
    surface_mask = (surface_metric < surface_ratio) & (planarness > 0.22)
    edge_threshold = max(edge_ratio * 2.0, 0.32)
    edge_mask = (~surface_mask) & (edge_metric < edge_threshold) & (linearness > 0.45) & (planarness < 0.30)
    atlas_class = torch.full_like(surface_mask, ATLAS_CLASS_UNSTABLE, dtype=torch.long)
    atlas_class[surface_mask] = ATLAS_CLASS_SURFACE
    atlas_class[edge_mask] = ATLAS_CLASS_EDGE
    return atlas_class, surface_mask, edge_mask


def _compute_geometry_features(points, k_neighbors, chunk_size, device, surface_ratio, edge_ratio):
    positions = torch.tensor(points, dtype=torch.float32, device=device)
    if positions.shape[0] == 1:
        return _make_single_point_geometry(positions)

    knn_indices, knn_distances = _chunked_self_knn(positions, k_neighbors, chunk_size)
    return _compute_geometry_from_neighbors(positions, knn_indices, knn_distances)


def _build_multiscale_k_values(k_neighbors: int, num_points: int):
    max_neighbors = max(1, int(num_points) - 1)
    base_k = max(1, int(k_neighbors))
    min_local_k = 6 if max_neighbors >= 6 else 1
    raw_values = [
        min_local_k,
        int(round(base_k * 0.5)),
        base_k,
        int(round(base_k * 1.5)),
    ]

    unique_values = []
    for value in raw_values:
        clamped = max(min_local_k, min(int(value), max_neighbors))
        if clamped not in unique_values:
            unique_values.append(clamped)
    unique_values.sort()
    return unique_values


def _edge_structure_score(linearness, planarness, scattering):
    return (linearness * torch.clamp(1.0 - planarness + 0.25 * (1.0 - scattering), 0.0, 1.0)).clamp(0.0, 1.0)


def _structure_scores(linearness, planarness, scattering):
    surface_score = (planarness * (1.0 - scattering)).clamp(0.0, 1.0)
    edge_score = _edge_structure_score(linearness, planarness, scattering)
    return surface_score, edge_score, torch.maximum(surface_score, edge_score)


def _geometric_residual(surface_mask, edge_mask, surface_metric, edge_metric, surface_ratio, edge_ratio, edge_score=None):
    surface_residual = torch.clamp(surface_metric / max(surface_ratio, 1e-6), 0.0, 1.0)
    edge_residual = torch.clamp(edge_metric / max(max(edge_ratio * 2.0, 0.32), 1e-6), 0.0, 1.0)
    if edge_score is not None:
        edge_residual = (0.45 * edge_residual + 0.55 * (1.0 - edge_score)).clamp(0.0, 1.0)
    return torch.where(
        surface_mask,
        surface_residual,
        torch.where(edge_mask, edge_residual, torch.ones_like(surface_residual)),
    )


def _refine_classes_with_neighbor_consensus(geometry, support_quality_mask, surface_ratio, edge_ratio):
    atlas_class = geometry["atlas_class"].clone()
    surface_mask = geometry["surface_mask"].clone()
    edge_mask = geometry["edge_mask"].clone()
    exploratory_support_mask = geometry.get("exploratory_support_mask")

    if support_quality_mask is None or geometry["knn_indices"].shape[1] == 0:
        geometry["class_consistency"] = torch.ones_like(geometry["structure_score"])
        return atlas_class, surface_mask, edge_mask

    support_quality_mask = torch.as_tensor(support_quality_mask, dtype=torch.bool, device=atlas_class.device)
    if exploratory_support_mask is None:
        edge_support_gate = support_quality_mask
    else:
        exploratory_support_mask = torch.as_tensor(exploratory_support_mask, dtype=torch.bool, device=atlas_class.device)
        edge_support_gate = support_quality_mask | exploratory_support_mask
    neighbor_k = min(max(8, int(torch.median(geometry["selected_scale"].float()).item())), geometry["knn_indices"].shape[1])
    neighbor_indices = geometry["knn_indices"][:, :neighbor_k]
    neighbor_distances = geometry["knn_distances"][:, :neighbor_k]
    distance_scale = neighbor_distances.mean(dim=1, keepdim=True).clamp_min(1e-6)
    neighbor_weights = torch.exp(-(neighbor_distances ** 2) / (distance_scale ** 2))
    neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    neighbor_class = atlas_class[neighbor_indices]
    surface_vote = (neighbor_weights * (neighbor_class == ATLAS_CLASS_SURFACE).float()).sum(dim=1)
    edge_vote = (neighbor_weights * (neighbor_class == ATLAS_CLASS_EDGE).float()).sum(dim=1)
    unstable_vote = (neighbor_weights * (neighbor_class == ATLAS_CLASS_UNSTABLE).float()).sum(dim=1)

    normals = geometry["eigenvectors"][:, :, 2]
    edge_dirs = geometry["eigenvectors"][:, :, 0]
    neighbor_normals = normals[neighbor_indices]
    neighbor_edge_dirs = edge_dirs[neighbor_indices]
    normal_alignment = torch.abs((neighbor_normals * normals[:, None, :]).sum(dim=2))
    edge_alignment = torch.abs((neighbor_edge_dirs * edge_dirs[:, None, :]).sum(dim=2))

    surface_alignment = (
        neighbor_weights * normal_alignment * (neighbor_class == ATLAS_CLASS_SURFACE).float()
    ).sum(dim=1) / surface_vote.clamp_min(1e-6)
    edge_alignment_score = (
        neighbor_weights * edge_alignment * (neighbor_class == ATLAS_CLASS_EDGE).float()
    ).sum(dim=1) / edge_vote.clamp_min(1e-6)

    surface_consistency = (surface_vote * surface_alignment).clamp(0.0, 1.0)
    edge_consistency = (edge_vote * edge_alignment_score).clamp(0.0, 1.0)

    edge_threshold = max(edge_ratio * 2.0, 0.32)
    promote_surface = (
        (atlas_class == ATLAS_CLASS_UNSTABLE)
        & support_quality_mask
        & (geometry["planarness"] > 0.42)
        & (geometry["surface_metric"] < surface_ratio * 1.65)
        & (surface_consistency > 0.56)
    )
    promote_edge = (
        (atlas_class == ATLAS_CLASS_UNSTABLE)
        & edge_support_gate
        & (geometry["linearness"] > 0.54)
        & (geometry["edge_metric"] < edge_threshold * 1.30)
        & (edge_consistency > 0.40)
        & (edge_consistency > surface_consistency)
    )

    surface_mask[promote_surface] = True
    edge_mask[promote_surface] = False
    atlas_class[promote_surface] = ATLAS_CLASS_SURFACE

    edge_mask[promote_edge] = True
    surface_mask[promote_edge] = False
    atlas_class[promote_edge] = ATLAS_CLASS_EDGE

    class_consistency = torch.where(
        atlas_class == ATLAS_CLASS_SURFACE,
        surface_consistency,
        torch.where(atlas_class == ATLAS_CLASS_EDGE, edge_consistency, unstable_vote),
    ).clamp(0.0, 1.0)
    geometry["class_consistency"] = class_consistency
    return atlas_class, surface_mask, edge_mask


def _compute_support_consistency(geometry, surface_mask, edge_mask):
    positions = geometry["positions"]
    knn_indices = geometry["knn_indices"]
    knn_distances = geometry["knn_distances"]
    if knn_indices.shape[1] == 0:
        return torch.ones((positions.shape[0],), dtype=positions.dtype, device=positions.device)

    if "selected_scale" in geometry:
        preferred_k = int(torch.median(geometry["selected_scale"].float()).item())
    else:
        preferred_k = knn_indices.shape[1]
    neighbor_k = min(max(6, preferred_k), knn_indices.shape[1])
    neighbor_indices = knn_indices[:, :neighbor_k]
    neighbor_distances = knn_distances[:, :neighbor_k]

    distance_scale = neighbor_distances.mean(dim=1, keepdim=True).clamp_min(1e-6)
    neighbor_weights = torch.exp(-(neighbor_distances ** 2) / (distance_scale ** 2))
    neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    neighbor_offsets = positions[neighbor_indices] - positions[:, None, :]
    total_energy = (neighbor_weights * neighbor_offsets.square().sum(dim=2)).sum(dim=1).clamp_min(1e-6)

    normals = geometry["eigenvectors"][:, :, 2]
    edge_dirs = geometry["eigenvectors"][:, :, 0]
    plane_residual = (neighbor_offsets * normals[:, None, :]).sum(dim=2).square()
    edge_parallel = (neighbor_offsets * edge_dirs[:, None, :]).sum(dim=2).square()
    edge_residual = (neighbor_offsets.square().sum(dim=2) - edge_parallel).clamp_min(0.0)

    surface_consistency = (1.0 - (neighbor_weights * plane_residual).sum(dim=1) / total_energy).clamp(0.0, 1.0)
    edge_consistency = (1.0 - (neighbor_weights * edge_residual).sum(dim=1) / total_energy).clamp(0.0, 1.0)
    fallback_consistency = torch.maximum(surface_consistency, edge_consistency)
    structure_score = geometry.get("structure_score")
    if structure_score is not None:
        fallback_consistency = (fallback_consistency * structure_score).clamp(0.0, 1.0)

    return torch.where(
        surface_mask,
        surface_consistency,
        torch.where(edge_mask, edge_consistency, fallback_consistency),
    )


def _self_calibrate_geometry_support(
    points,
    k_neighbors,
    chunk_size,
    device,
    surface_ratio,
    edge_ratio,
    support_quality_mask=None,
    exploratory_support_mask=None,
    detail_preservation_score=None,
    enable_self_calibration: bool = True,
):
    positions = torch.tensor(points, dtype=torch.float32, device=device)
    if positions.shape[0] == 1:
        geometry = _make_single_point_geometry(positions)
        geometry["atlas_class"] = torch.full((1,), ATLAS_CLASS_UNSTABLE, dtype=torch.long, device=device)
        geometry["surface_mask"] = torch.zeros((1,), dtype=torch.bool, device=device)
        geometry["edge_mask"] = torch.zeros((1,), dtype=torch.bool, device=device)
        geometry["surface_score"] = torch.zeros((1,), dtype=positions.dtype, device=device)
        geometry["edge_score"] = torch.zeros((1,), dtype=positions.dtype, device=device)
        geometry["structure_score"] = torch.zeros((1,), dtype=positions.dtype, device=device)
        geometry["geometric_residual"] = torch.ones((1,), dtype=positions.dtype, device=device)
        geometry["scale_consistency"] = torch.ones((1,), dtype=positions.dtype, device=device)
        geometry["class_consistency"] = torch.ones((1,), dtype=positions.dtype, device=device)
        geometry["support_consistency"] = torch.zeros((1,), dtype=positions.dtype, device=device)
        geometry["selected_scale"] = torch.ones((1,), dtype=torch.int64, device=device)
        return geometry

    if support_quality_mask is not None:
        support_quality_mask = torch.as_tensor(support_quality_mask, dtype=torch.bool, device=device)
    if exploratory_support_mask is not None:
        exploratory_support_mask = torch.as_tensor(exploratory_support_mask, dtype=torch.bool, device=device)
    if detail_preservation_score is not None:
        detail_preservation_score = torch.as_tensor(detail_preservation_score, dtype=torch.float32, device=device).reshape(-1)
        if detail_preservation_score.shape[0] != positions.shape[0]:
            detail_preservation_score = None

    if enable_self_calibration:
        scale_values = _build_multiscale_k_values(k_neighbors, positions.shape[0])
    else:
        scale_values = [max(1, min(int(k_neighbors), int(positions.shape[0]) - 1))]
    knn_indices, knn_distances = _chunked_self_knn(positions, max(scale_values), chunk_size)

    geometry_per_scale = []
    class_per_scale = []
    surface_per_scale = []
    edge_per_scale = []
    structure_per_scale = []
    residual_per_scale = []
    radius_per_scale = []

    for scale_k in scale_values:
        scale_geometry = _compute_geometry_from_neighbors(positions, knn_indices[:, :scale_k], knn_distances[:, :scale_k])
        atlas_class, surface_mask, edge_mask = _classify_geometry(
            scale_geometry["linearness"],
            scale_geometry["planarness"],
            scale_geometry["scattering"],
            scale_geometry["surface_metric"],
            scale_geometry["edge_metric"],
            surface_ratio,
            edge_ratio,
        )
        if support_quality_mask is not None:
            strong_support = support_quality_mask
            exploratory_support = exploratory_support_mask if exploratory_support_mask is not None else support_quality_mask
            edge_threshold = max(edge_ratio * 2.0, 0.32)
            exploratory_edge = (
                exploratory_support
                & (scale_geometry["linearness"] > 0.52)
                & (scale_geometry["edge_metric"] < edge_threshold * 1.20)
            )
            surface_mask = surface_mask & strong_support
            edge_mask = edge_mask & (strong_support | exploratory_edge)
            atlas_class = torch.full_like(atlas_class, ATLAS_CLASS_UNSTABLE)
            atlas_class[surface_mask] = ATLAS_CLASS_SURFACE
            atlas_class[edge_mask] = ATLAS_CLASS_EDGE

        surface_score, edge_score, structure_score = _structure_scores(
            scale_geometry["linearness"],
            scale_geometry["planarness"],
            scale_geometry["scattering"],
        )
        geometric_residual = _geometric_residual(
            surface_mask,
            edge_mask,
            scale_geometry["surface_metric"],
            scale_geometry["edge_metric"],
            surface_ratio,
            edge_ratio,
            edge_score=edge_score,
        )

        scale_geometry["atlas_class"] = atlas_class
        scale_geometry["surface_mask"] = surface_mask
        scale_geometry["edge_mask"] = edge_mask
        scale_geometry["surface_score"] = surface_score
        scale_geometry["edge_score"] = edge_score
        scale_geometry["structure_score"] = structure_score
        scale_geometry["geometric_residual"] = geometric_residual

        geometry_per_scale.append(scale_geometry)
        class_per_scale.append(atlas_class)
        surface_per_scale.append(surface_mask)
        edge_per_scale.append(edge_mask)
        structure_per_scale.append(structure_score)
        residual_per_scale.append(geometric_residual)
        radius_per_scale.append(scale_geometry["radius"])

    class_stack = torch.stack(class_per_scale, dim=0)
    surface_stack = torch.stack(surface_per_scale, dim=0)
    edge_stack = torch.stack(edge_per_scale, dim=0)
    structure_stack = torch.stack(structure_per_scale, dim=0)
    residual_stack = torch.stack(residual_per_scale, dim=0)
    radius_stack = torch.stack(radius_per_scale, dim=0)

    if class_stack.shape[0] == 1:
        vote_support = torch.ones_like(residual_stack)
    else:
        vote_support = torch.stack(
            [
                (class_stack == class_stack[scale_idx : scale_idx + 1]).float().mean(dim=0)
                for scale_idx in range(class_stack.shape[0])
            ],
            dim=0,
        )

    unstable_penalty = 0.05 * (class_stack == ATLAS_CLASS_UNSTABLE).float()
    consistency_penalty = 0.10 * (1.0 - vote_support)
    structure_bonus = 0.10 * structure_stack
    edge_stack_score = torch.stack([item["edge_score"] for item in geometry_per_scale], dim=0)
    if detail_preservation_score is None:
        detail_prior = torch.zeros((positions.shape[0],), dtype=positions.dtype, device=device)
    else:
        detail_prior = detail_preservation_score.to(dtype=positions.dtype, device=device).clamp(0.0, 1.0)
    detail_bonus = edge_stack_score * (0.55 + 0.45 * detail_prior.unsqueeze(0)) * (1.0 - 0.35 * vote_support)
    selection_objective = (
        residual_stack
        + unstable_penalty
        + consistency_penalty
        - structure_bonus
        - 0.06 * detail_bonus
    )

    best_scale_idx = torch.argmin(selection_objective, dim=0)
    node_idx = torch.arange(positions.shape[0], device=device)

    chosen = {
        "positions": positions,
        "knn_indices": knn_indices,
        "knn_distances": knn_distances,
        "eigenvalues": torch.stack([item["eigenvalues"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "eigenvectors": torch.stack([item["eigenvectors"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "linearness": torch.stack([item["linearness"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "planarness": torch.stack([item["planarness"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "scattering": torch.stack([item["scattering"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "surface_metric": torch.stack([item["surface_metric"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "edge_metric": torch.stack([item["edge_metric"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "radius": radius_stack[best_scale_idx, node_idx],
        "atlas_class": class_stack[best_scale_idx, node_idx],
        "surface_mask": surface_stack[best_scale_idx, node_idx],
        "edge_mask": edge_stack[best_scale_idx, node_idx],
        "surface_score": torch.stack([item["surface_score"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "edge_score": torch.stack([item["edge_score"] for item in geometry_per_scale], dim=0)[best_scale_idx, node_idx],
        "structure_score": structure_stack[best_scale_idx, node_idx],
        "geometric_residual": residual_stack[best_scale_idx, node_idx],
        "scale_consistency": vote_support[best_scale_idx, node_idx].clamp(0.0, 1.0),
        "selected_scale": torch.tensor(scale_values, dtype=torch.int64, device=device)[best_scale_idx],
    }
    if exploratory_support_mask is not None:
        chosen["exploratory_support_mask"] = exploratory_support_mask
    atlas_class, surface_mask, edge_mask = _refine_classes_with_neighbor_consensus(
        chosen,
        support_quality_mask,
        surface_ratio,
        edge_ratio,
    )
    chosen["atlas_class"] = atlas_class
    chosen["surface_mask"] = surface_mask
    chosen["edge_mask"] = edge_mask
    chosen["geometric_residual"] = _geometric_residual(
        surface_mask,
        edge_mask,
        chosen["surface_metric"],
        chosen["edge_metric"],
        surface_ratio,
        edge_ratio,
        edge_score=chosen["edge_score"],
    )
    chosen["support_consistency"] = _compute_support_consistency(chosen, surface_mask, edge_mask)
    return chosen


def _select_candidate_indices(
    priority_score,
    edge_priority,
    unstable_priority,
    edge_mask,
    unstable_mask,
    max_nodes,
    edge_quota_fraction,
    unstable_quota_fraction,
    detail_priority=None,
    detail_mask=None,
    detail_quota_fraction=0.15,
):
    num_candidates = priority_score.shape[0]
    if num_candidates <= max_nodes:
        return np.arange(num_candidates, dtype=np.int64)

    selected = np.zeros((num_candidates,), dtype=bool)
    chosen = []

    def take(order, count):
        for idx in order:
            idx = int(idx)
            if selected[idx]:
                continue
            selected[idx] = True
            chosen.append(idx)
            if len(chosen) >= count:
                break

    edge_candidates = np.flatnonzero(edge_mask)
    unstable_candidates = np.flatnonzero(unstable_mask)
    edge_reserve_fraction = max(float(edge_quota_fraction), 0.10)
    edge_quota = min(int(round(max_nodes * edge_reserve_fraction)), edge_candidates.size)
    unstable_quota = min(int(round(max_nodes * unstable_quota_fraction)), unstable_candidates.size)
    if detail_priority is not None and detail_mask is not None:
        detail_priority = np.asarray(detail_priority, dtype=np.float32).reshape(-1)
        detail_candidates = np.flatnonzero(np.asarray(detail_mask, dtype=bool).reshape(-1))
        detail_quota = min(int(round(max_nodes * float(detail_quota_fraction))), detail_candidates.size)
    else:
        detail_candidates = np.zeros((0,), dtype=np.int64)
        detail_quota = 0

    if edge_quota > 0:
        take(edge_candidates[np.argsort(edge_priority[edge_candidates])[::-1]], edge_quota)
    if detail_quota > 0:
        take(detail_candidates[np.argsort(detail_priority[detail_candidates])[::-1]], edge_quota + detail_quota)
    if unstable_quota > 0:
        take(unstable_candidates[np.argsort(unstable_priority[unstable_candidates])[::-1]], edge_quota + detail_quota + unstable_quota)

    remaining = max_nodes - len(chosen)
    if remaining > 0:
        take(np.argsort(priority_score)[::-1], max_nodes)

    indices = np.asarray(chosen[:max_nodes], dtype=np.int64)
    indices.sort()
    return indices


def _build_support_from_geometry(geometry, surface_mask, edge_mask):
    positions = geometry["positions"]
    eigenvalues = geometry["eigenvalues"]
    eigenvectors = geometry["eigenvectors"]
    knn_indices = geometry["knn_indices"]
    knn_distances = geometry["knn_distances"]

    support = torch.zeros((positions.shape[0], 3, 3), dtype=positions.dtype, device=positions.device)
    identity = torch.eye(3, dtype=positions.dtype, device=positions.device).unsqueeze(0)
    normals = eigenvectors[:, :, 2]
    surface_support = identity - torch.einsum("ni,nj->nij", normals, normals)
    edge_dir = eigenvectors[:, :, 0]
    edge_support = torch.einsum("ni,nj->nij", edge_dir, edge_dir)
    support[surface_mask] = surface_support[surface_mask]
    support[edge_mask] = edge_support[edge_mask]

    sigmas = torch.sqrt(eigenvalues.clamp_min(1e-8))
    anisotropy_ref = torch.stack(
        (
            torch.log(sigmas[:, 0] / sigmas[:, 1].clamp_min(1e-8)),
            torch.log(sigmas[:, 1] / sigmas[:, 2].clamp_min(1e-8)),
        ),
        dim=1,
    )
    radius = geometry.get("radius", knn_distances[:, -1]).clamp_min(1e-6)

    return support, normals, anisotropy_ref, radius, knn_indices


def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if colors.dtype == np.uint8:
        return colors
    if colors.size == 0:
        return colors.astype(np.uint8)
    if colors.max() <= 1.0:
        return np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    return np.clip(colors, 0, 255).astype(np.uint8)


def flatten_dense_geometry(pts3d, confs, rgb_images, min_conf_thr=1.5):
    dense_points = []
    dense_colors = []
    dense_confidences = []
    point_image_ids = []
    point_pixels = []

    for image_id, (points_view, conf_view, image_rgb) in enumerate(zip(pts3d, confs, rgb_images)):
        conf_array = np.asarray(conf_view, dtype=np.float32)
        height, width = conf_array.shape
        points_array = np.asarray(points_view, dtype=np.float32).reshape(height, width, 3)
        color_array = np.asarray(image_rgb, dtype=np.float32)

        valid_mask = (conf_array >= float(min_conf_thr)) & np.isfinite(points_array).all(axis=2)
        if not np.any(valid_mask):
            continue

        yy, xx = np.nonzero(valid_mask)
        dense_points.append(points_array[valid_mask])
        dense_colors.append(color_array[valid_mask])
        dense_confidences.append(conf_array[valid_mask])
        point_image_ids.append(np.full(yy.shape[0], image_id, dtype=np.int32))
        point_pixels.append(np.stack([xx, yy], axis=1).astype(np.int32))

    if not dense_points:
        raise ValueError("No dense geometry survived the confidence threshold. Lower --min_conf_thr or inspect the cache.")

    return {
        "points": np.concatenate(dense_points, axis=0),
        "colors": np.concatenate(dense_colors, axis=0),
        "confidences": np.concatenate(dense_confidences, axis=0),
        "image_ids": np.concatenate(point_image_ids, axis=0),
        "pixels_xy": np.concatenate(point_pixels, axis=0),
    }


def build_foundation_geometry_atlas(
    points,
    colors,
    confidences,
    image_ids=None,
    parallax_scores=None,
    depth_consistency=None,
    num_views=None,
    max_nodes=8192,
    k_neighbors=16,
    surface_ratio=0.12,
    edge_ratio=0.18,
    reliability_alpha=1.0,
    reliability_gamma=2.0,
    reliability_min=0.05,
    candidate_oversample=2.0,
    target_voxel_size=0.0,
    edge_quota_fraction=0.05,
    unstable_quota_fraction=0.05,
    min_point_support=4,
    min_view_support=2,
    voxel_support_consistency_min=0.18,
    enable_self_calibration=True,
    chunk_size=512,
    device="cpu",
    seed=42,
):
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    image_ids = None if image_ids is None else np.asarray(image_ids, dtype=np.int32).reshape(-1)
    parallax_scores = None if parallax_scores is None else np.asarray(parallax_scores, dtype=np.float32).reshape(-1)
    depth_consistency = None if depth_consistency is None else np.asarray(depth_consistency, dtype=np.float32).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3].")
    if points.shape[0] == 0:
        raise ValueError("Cannot build a foundation atlas from an empty point set.")
    if colors.shape[0] != points.shape[0]:
        raise ValueError("colors must match points in the first dimension.")
    if confidences.shape[0] != points.shape[0]:
        raise ValueError("confidences must match points in the first dimension.")
    if image_ids is not None and image_ids.shape[0] != points.shape[0]:
        raise ValueError("image_ids must match points in the first dimension.")
    if parallax_scores is not None and parallax_scores.shape[0] != points.shape[0]:
        raise ValueError("parallax_scores must match points in the first dimension.")
    if depth_consistency is not None and depth_consistency.shape[0] != points.shape[0]:
        raise ValueError("depth_consistency must match points in the first dimension.")

    if num_views is None:
        num_views = int(image_ids.max()) + 1 if image_ids is not None and image_ids.size else 1

    point_support_min = 1 if points.shape[0] <= max_nodes else max(1, int(min_point_support))
    view_support_min = 1 if image_ids is None else max(1, int(min_view_support))

    candidates = _aggregate_voxel_candidates(
        points,
        colors,
        confidences,
        image_ids,
        int(max_nodes),
        candidate_oversample,
        target_voxel_size=target_voxel_size,
        parallax_scores=parallax_scores,
        depth_consistency=depth_consistency,
    )
    candidate_view_balance = np.clip(candidates.get("view_balance", np.ones_like(candidates["confidences"])).astype(np.float32), 0.0, 1.0)
    candidate_view_outlier_score = np.clip(candidates.get("view_outlier_score", np.ones_like(candidates["confidences"])).astype(np.float32), 0.0, 1.0)
    candidate_robust_inlier_ratio = np.clip(candidates.get("robust_inlier_ratio", np.ones_like(candidates["confidences"])).astype(np.float32), 0.0, 1.0)
    candidate_parallax_score = np.clip(candidates.get("parallax_score", np.ones_like(candidates["confidences"])).astype(np.float32), 0.0, 1.0)
    candidate_depth_consistency = np.clip(candidates.get("depth_consistency", np.ones_like(candidates["confidences"])).astype(np.float32), 0.0, 1.0)
    candidate_detail_prior = np.clip(candidates.get("detail_score", np.zeros_like(candidates["confidences"])).astype(np.float32), 0.0, 1.0)
    candidate_point_ok = candidates["point_support"].astype(np.float32) >= float(point_support_min)
    candidate_view_ok = (
        (candidates["view_support"].astype(np.float32) >= float(view_support_min))
        | ((candidate_view_balance >= 0.35) & (candidate_robust_inlier_ratio >= 0.50))
    )
    candidate_strong_support = (
        candidate_point_ok
        & candidate_view_ok
        & (
            (candidates["voxel_support_consistency"].astype(np.float32) >= float(voxel_support_consistency_min) * 0.85)
            | ((candidate_view_outlier_score >= 0.50) & (candidate_robust_inlier_ratio >= 0.65))
        )
        & (candidate_view_outlier_score >= 0.25)
        & (candidate_robust_inlier_ratio >= 0.50)
    )
    candidate_exploratory_support = (
        (candidates["point_support"].astype(np.float32) >= float(max(2, point_support_min - 1)))
        & (
            (candidates["view_support"].astype(np.float32) >= float(max(1, view_support_min - 1)))
            | (candidate_view_balance >= 0.25)
        )
        & (candidate_robust_inlier_ratio >= 0.30)
        & (candidates["voxel_support_consistency"].astype(np.float32) >= float(voxel_support_consistency_min) * 0.70)
    )
    geometry_candidates = _self_calibrate_geometry_support(
        candidates["positions"],
        k_neighbors,
        chunk_size,
        device,
        surface_ratio,
        edge_ratio,
        support_quality_mask=candidate_strong_support,
        exploratory_support_mask=candidate_exploratory_support,
        detail_preservation_score=candidate_detail_prior,
        enable_self_calibration=bool(enable_self_calibration),
    )
    cand_class = geometry_candidates["atlas_class"]
    cand_edge = geometry_candidates["edge_mask"]

    mean_confidence_score = _normalize_confidence(
        candidates["confidences"],
        parallax_scores=candidate_parallax_score,
        depth_consistency=candidate_depth_consistency,
    )
    peak_confidence_score = _normalize_scores(np.log1p(candidates["confidence_peak"]))
    confidence_score = np.clip(0.72 * mean_confidence_score + 0.28 * peak_confidence_score, 0.0, 1.0)
    point_support_score = _normalize_scores(np.log1p(candidates["point_support"]))
    target_view_support = max(1, min(int(num_views), 4))
    if target_view_support == 1:
        view_support_score = np.ones_like(confidence_score, dtype=np.float32)
    else:
        view_support_score = np.clip(
            (candidates["view_support"].astype(np.float32) - 1.0) / float(target_view_support - 1),
            0.0,
            1.0,
        )

    structure_score = geometry_candidates["structure_score"].detach().cpu().numpy()
    class_consistency = geometry_candidates["class_consistency"].detach().cpu().numpy()
    support_consistency = geometry_candidates["support_consistency"].detach().cpu().numpy()
    edge_structure = geometry_candidates["edge_score"].detach().cpu().numpy()
    scale_consistency = geometry_candidates["scale_consistency"].detach().cpu().numpy()
    view_balance_score = candidate_view_balance
    voxel_support_consistency = np.clip(candidates["voxel_support_consistency"].astype(np.float32), 0.0, 1.0)
    outlier_consistency = candidate_view_outlier_score
    robust_inlier_score = candidate_robust_inlier_ratio
    support_score = np.sqrt(np.clip(point_support_score * view_support_score, 0.0, 1.0))
    consistency_score = np.sqrt(
        np.clip(
            scale_consistency
            * np.power(class_consistency, 1.05)
            * np.power(np.maximum(support_consistency, voxel_support_consistency), 1.10)
            * outlier_consistency
            * (0.45 + 0.55 * robust_inlier_score),
            0.0,
            1.0,
        )
    )
    coverage_score = np.maximum(view_support_score, view_balance_score) * (0.45 + 0.55 * outlier_consistency)
    evidence_priority_score = np.sqrt(
        np.clip(
            support_score
            * np.maximum(view_support_score, view_balance_score)
            * (0.55 + 0.45 * candidate_parallax_score),
            0.0,
            1.0,
        )
    )
    priority_score = (
        confidence_score
        * (0.35 + 0.65 * evidence_priority_score)
        * (0.22 + 0.78 * structure_score)
        * (0.50 + 0.50 * consistency_score)
        * (0.45 + 0.55 * coverage_score)
    )
    edge_priority = priority_score * np.sqrt(
        np.clip(edge_structure * np.maximum(class_consistency, support_consistency), 0.0, 1.0)
    )
    unstable_score = (
        np.clip(1.0 - structure_score, 0.0, 1.0)
        * (0.35 + 0.65 * np.maximum(support_score, voxel_support_consistency))
        * (0.35 + 0.65 * coverage_score)
        * (0.45 + 0.55 * consistency_score)
        * (0.45 + 0.55 * confidence_score)
        * (0.60 + 0.40 * robust_inlier_score)
    )
    structure_certainty = np.clip(structure_score * np.maximum(class_consistency, support_consistency), 0.0, 1.0)
    detail_priority = (
        np.clip(1.0 - structure_certainty, 0.0, 1.0)
        * (0.40 + 0.60 * confidence_score)
        * (0.40 + 0.60 * view_balance_score)
        * (0.40 + 0.60 * robust_inlier_score)
        * (0.35 + 0.65 * np.maximum(support_consistency, voxel_support_consistency))
        * (0.60 + 0.40 * np.maximum(candidate_detail_prior, candidate_parallax_score))
    ).astype(np.float32)
    detail_mask = candidate_exploratory_support & (detail_priority > 0.0)

    selected_indices = _select_candidate_indices(
        priority_score,
        edge_priority,
        unstable_score,
        cand_edge.detach().cpu().numpy(),
        (cand_class == ATLAS_CLASS_UNSTABLE).detach().cpu().numpy(),
        int(max_nodes),
        edge_quota_fraction,
        unstable_quota_fraction,
        detail_priority=detail_priority,
        detail_mask=detail_mask,
        detail_quota_fraction=0.15,
    )

    sampled_points = candidates["positions"][selected_indices]
    sampled_colors = candidates["colors"][selected_indices]
    sampled_confidences = candidates["confidences"][selected_indices]
    sampled_point_support = candidates["point_support"][selected_indices]
    sampled_view_support = candidates["view_support"][selected_indices]
    sampled_view_balance = candidates["view_balance"][selected_indices]
    sampled_voxel_support_consistency = candidates["voxel_support_consistency"][selected_indices]
    sampled_view_outlier_score = candidates["view_outlier_score"][selected_indices]
    sampled_robust_inlier_ratio = candidates["robust_inlier_ratio"][selected_indices]
    sampled_confidence_peak = candidates["confidence_peak"][selected_indices]
    sampled_parallax_score = candidates["parallax_score"][selected_indices]
    sampled_depth_consistency = candidates["depth_consistency"][selected_indices]
    sampled_detail_prior = candidates["detail_score"][selected_indices]

    point_support = torch.tensor(sampled_point_support.astype(np.float32), dtype=torch.float32, device=device)
    view_support = torch.tensor(sampled_view_support.astype(np.float32), dtype=torch.float32, device=device)
    sampled_view_balance_t = torch.tensor(np.clip(sampled_view_balance, 0.0, 1.0), dtype=torch.float32, device=device)
    sampled_view_outlier_t = torch.tensor(np.clip(sampled_view_outlier_score, 0.0, 1.0), dtype=torch.float32, device=device)
    sampled_robust_inlier_t = torch.tensor(np.clip(sampled_robust_inlier_ratio, 0.0, 1.0), dtype=torch.float32, device=device)
    sampled_voxel_support_t = torch.tensor(np.clip(sampled_voxel_support_consistency, 0.0, 1.0), dtype=torch.float32, device=device)
    strong_support_mask = (
        (point_support >= point_support_min)
        & ((view_support >= view_support_min) | ((sampled_view_balance_t >= 0.35) & (sampled_robust_inlier_t >= 0.50)))
        & (
            (sampled_voxel_support_t >= float(voxel_support_consistency_min) * 0.85)
            | ((sampled_view_outlier_t >= 0.50) & (sampled_robust_inlier_t >= 0.65))
        )
        & (sampled_view_outlier_t >= 0.25)
        & (sampled_robust_inlier_t >= 0.50)
    )
    exploratory_support_mask = (
        (point_support >= max(2, point_support_min - 1))
        & ((view_support >= max(1, view_support_min - 1)) | (sampled_view_balance_t >= 0.25))
        & (sampled_robust_inlier_t >= 0.30)
        & (sampled_voxel_support_t >= float(voxel_support_consistency_min) * 0.70)
    )
    geometry = _self_calibrate_geometry_support(
        sampled_points,
        k_neighbors,
        chunk_size,
        device,
        surface_ratio,
        edge_ratio,
        support_quality_mask=strong_support_mask,
        exploratory_support_mask=exploratory_support_mask,
        detail_preservation_score=sampled_detail_prior,
        enable_self_calibration=bool(enable_self_calibration),
    )
    positions = geometry["positions"]

    if positions.shape[0] == 1:
        zero_support = torch.zeros((1, 3, 3), dtype=positions.dtype, device=device)
        basis = torch.eye(3, dtype=positions.dtype, device=device).unsqueeze(0)
        atlas = FoundationAtlas(
            positions=sampled_points,
            support=zero_support.cpu().numpy(),
            basis=basis.cpu().numpy(),
            normal=basis[:, :, 2].cpu().numpy(),
            radius=np.ones((1,), dtype=np.float32),
            raw_score=np.full((1,), reliability_min, dtype=np.float32),
            reliability=np.full((1,), reliability_min, dtype=np.float32),
            atlas_class=np.full((1,), ATLAS_CLASS_UNSTABLE, dtype=np.int64),
            anisotropy_ref=np.zeros((1, 2), dtype=np.float32),
            neighbor_indices=np.zeros((1, 1), dtype=np.int64),
            calibration_residual=np.ones((1,), dtype=np.float32),
            node_confidence=sampled_confidences.astype(np.float32),
            point_support=np.ones((1,), dtype=np.int32),
            view_support=np.ones((1,), dtype=np.int32),
            view_coverage=np.ones((1,), dtype=np.float32),
            support_score=np.ones((1,), dtype=np.float32),
            linearness=np.zeros((1,), dtype=np.float32),
            planarness=np.zeros((1,), dtype=np.float32),
            scattering=np.ones((1,), dtype=np.float32),
            structure_score=np.zeros((1,), dtype=np.float32),
            scale_consistency=np.ones((1,), dtype=np.float32),
            class_consistency=np.ones((1,), dtype=np.float32),
            support_consistency=np.zeros((1,), dtype=np.float32),
            view_balance=np.ones((1,), dtype=np.float32),
            view_outlier_score=np.ones((1,), dtype=np.float32),
            robust_inlier_ratio=np.ones((1,), dtype=np.float32),
            unstable_reason_code=np.array([UNSTABLE_REASON_HIGH_CALIBRATION_RESIDUAL], dtype=np.int32),
        )
        return atlas

    linearness = geometry["linearness"]
    planarness = geometry["planarness"]
    scattering = geometry["scattering"]
    atlas_class = geometry["atlas_class"]
    surface_mask = geometry["surface_mask"]
    edge_mask = geometry["edge_mask"]

    support, normals, anisotropy_ref, radius, knn_indices = _build_support_from_geometry(geometry, surface_mask, edge_mask)
    radius_scale = radius.median().clamp_min(1e-6)
    density_score = torch.exp(-(radius / radius_scale))
    mean_confidence_score = _normalize_confidence(
        sampled_confidences,
        parallax_scores=sampled_parallax_score,
        depth_consistency=sampled_depth_consistency,
    )
    peak_confidence_score = _normalize_scores(np.log1p(sampled_confidence_peak))
    confidence_score = torch.tensor(
        np.clip(0.72 * mean_confidence_score + 0.28 * peak_confidence_score, 0.0, 1.0),
        dtype=torch.float32,
        device=device,
    )
    point_support_score = torch.tensor(_normalize_scores(np.log1p(sampled_point_support)), dtype=torch.float32, device=device)
    if target_view_support == 1:
        view_support_score = torch.ones_like(confidence_score)
    else:
        view_support_score = torch.clamp((view_support - 1.0) / float(target_view_support - 1), 0.0, 1.0)
    edge_target_view_support = max(1, min(int(num_views), 3))
    if edge_target_view_support == 1:
        edge_view_support_score = torch.ones_like(confidence_score)
    else:
        edge_view_support_score = torch.clamp((view_support - 1.0) / float(edge_target_view_support - 1), 0.0, 1.0)
    surface_score = geometry["surface_score"]
    edge_score = geometry["edge_score"]
    structure_score = geometry["structure_score"]
    scale_consistency = geometry["scale_consistency"]
    class_consistency = geometry["class_consistency"]
    geom_support_consistency = geometry["support_consistency"]
    view_balance_score = sampled_view_balance_t
    view_outlier_score = sampled_view_outlier_t
    robust_inlier_score = sampled_robust_inlier_t
    voxel_support_consistency = torch.tensor(
        np.clip(sampled_voxel_support_consistency, 0.0, 1.0),
        dtype=torch.float32,
        device=device,
    )
    support_consistency = (
        0.55 * geom_support_consistency
        + 0.25 * voxel_support_consistency
        + 0.20 * view_outlier_score
    ).clamp(0.0, 1.0)
    sampled_parallax_t = torch.tensor(np.clip(sampled_parallax_score, 0.0, 1.0), dtype=torch.float32, device=device)
    sampled_depth_t = torch.tensor(np.clip(sampled_depth_consistency, 0.0, 1.0), dtype=torch.float32, device=device)
    sampled_detail_t = torch.tensor(np.clip(sampled_detail_prior, 0.0, 1.0), dtype=torch.float32, device=device)
    effective_view_support_score = torch.where(edge_mask, torch.maximum(view_support_score, edge_view_support_score), view_support_score)
    support_score = torch.sqrt((point_support_score * effective_view_support_score).clamp(0.0, 1.0))
    density_term = 0.30 + 0.70 * density_score
    edge_density_term = 0.55 + 0.45 * torch.sqrt(density_score.clamp_min(0.0))
    density_term = torch.where(edge_mask, edge_density_term, density_term)
    evidence_score = torch.sqrt(
        (
            support_score
            * torch.maximum(effective_view_support_score, view_balance_score)
            * (0.55 + 0.45 * sampled_parallax_t)
        ).clamp(0.0, 1.0)
    )
    consistency_score = torch.sqrt(
        (
            scale_consistency
            * class_consistency.pow(1.05)
            * torch.maximum(support_consistency, voxel_support_consistency).pow(1.10)
            * view_outlier_score
            * (0.45 + 0.55 * robust_inlier_score)
        ).clamp(0.0, 1.0)
    )

    raw_score = (
        confidence_score
        * (0.32 + 0.68 * evidence_score)
        * (0.22 + 0.78 * structure_score)
        * density_term
        * (0.44 + 0.56 * consistency_score)
        * (0.52 + 0.48 * torch.maximum(view_balance_score, view_outlier_score))
    ).clamp(reliability_min, 1.0)

    geometric_residual = geometry["geometric_residual"]
    support_penalty = 1.0 - torch.sqrt((point_support_score * density_score * support_consistency).clamp(0.0, 1.0))
    coverage_penalty = 1.0 - torch.maximum(effective_view_support_score, view_balance_score)
    outlier_penalty = 1.0 - view_outlier_score
    detail_node_mask = edge_mask | (sampled_detail_t > 0.45)
    coverage_weight = torch.where(surface_mask, torch.full_like(coverage_penalty, 0.16), torch.full_like(coverage_penalty, 0.10))
    coverage_weight = torch.where(detail_node_mask, torch.full_like(coverage_penalty, 0.08), coverage_weight)
    support_weight = torch.where(surface_mask, torch.full_like(support_penalty, 0.12), torch.full_like(support_penalty, 0.10))
    support_weight = torch.where(detail_node_mask, torch.full_like(support_penalty, 0.08), support_weight)
    geometric_weight = torch.where(detail_node_mask, torch.full_like(geometric_residual, 0.34), torch.full_like(geometric_residual, 0.30))
    calibration_residual = (
        geometric_weight * geometric_residual
        + coverage_weight * coverage_penalty
        + support_weight * support_penalty
        + 0.10 * (1.0 - scale_consistency)
        + 0.12 * (1.0 - class_consistency)
        + 0.14 * (1.0 - support_consistency)
        + 0.10 * outlier_penalty
    ).clamp(0.0, 1.0)
    good_calibration_evidence = torch.sqrt((structure_score * evidence_score * consistency_score * (0.75 + 0.25 * sampled_depth_t)).clamp(0.0, 1.0))
    calibration_residual = (calibration_residual * (1.0 - 0.24 * good_calibration_evidence)).clamp(0.0, 1.0)
    unstable_mask = atlas_class == ATLAS_CLASS_UNSTABLE
    detail_evidence = torch.sqrt(
        (
            torch.maximum(effective_view_support_score, view_balance_score)
            * robust_inlier_score
            * support_consistency
            * (0.50 + 0.50 * sampled_detail_t)
        ).clamp(0.0, 1.0)
    )
    unstable_floor = (
        0.50
        + 0.16 * (1.0 - structure_score)
        + 0.10 * (1.0 - scale_consistency)
        + 0.08 * (1.0 - class_consistency)
        + 0.10 * (1.0 - support_consistency)
        + 0.06 * (1.0 - torch.maximum(effective_view_support_score, view_balance_score))
        + 0.06 * outlier_penalty
        - 0.08 * detail_evidence
    ).clamp(0.40, 0.88)
    calibration_residual[unstable_mask] = torch.maximum(calibration_residual[unstable_mask], unstable_floor[unstable_mask])

    reliability = torch.clamp(
        raw_score.pow(reliability_alpha) * torch.exp(-reliability_gamma * calibration_residual),
        reliability_min,
        1.0,
    )
    view_coverage = torch.clamp(view_support / max(float(num_views), 1.0), 0.0, 1.0)
    atlas = FoundationAtlas(
        positions=positions.cpu().numpy(),
        support=support.cpu().numpy(),
        basis=geometry["eigenvectors"].cpu().numpy(),
        normal=normals.cpu().numpy(),
        radius=radius.cpu().numpy(),
        raw_score=raw_score.detach().cpu().numpy(),
        reliability=reliability.detach().cpu().numpy(),
        atlas_class=atlas_class.cpu().numpy(),
        anisotropy_ref=anisotropy_ref.detach().cpu().numpy(),
        neighbor_indices=knn_indices.cpu().numpy(),
        calibration_residual=calibration_residual.detach().cpu().numpy(),
        node_confidence=sampled_confidences.astype(np.float32),
        point_support=sampled_point_support.astype(np.int32),
        view_support=sampled_view_support.astype(np.int32),
        view_coverage=view_coverage.detach().cpu().numpy(),
        support_score=support_score.detach().cpu().numpy(),
        linearness=linearness.detach().cpu().numpy(),
        planarness=planarness.detach().cpu().numpy(),
        scattering=scattering.detach().cpu().numpy(),
        structure_score=structure_score.detach().cpu().numpy(),
        scale_consistency=scale_consistency.detach().cpu().numpy(),
        class_consistency=class_consistency.detach().cpu().numpy(),
        support_consistency=support_consistency.detach().cpu().numpy(),
        view_balance=np.asarray(sampled_view_balance, dtype=np.float32),
        view_outlier_score=np.asarray(sampled_view_outlier_score, dtype=np.float32),
        robust_inlier_ratio=np.asarray(sampled_robust_inlier_ratio, dtype=np.float32),
    )
    unstable_reason_code, _, _ = _compute_unstable_reason_payload(
        atlas,
        structure_score=atlas.structure_score,
        support_consistency=atlas.support_consistency,
        scale_consistency=atlas.scale_consistency,
        class_consistency=atlas.class_consistency,
    )
    atlas.unstable_reason_code = unstable_reason_code.astype(np.int32)
    return atlas


def save_atlas_npz(atlas: FoundationAtlas, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=atlas.positions,
        support=atlas.support,
        basis=atlas.basis,
        normal=atlas.normal,
        radius=atlas.radius,
        raw_score=atlas.raw_score,
        reliability=atlas.reliability,
        atlas_class=atlas.atlas_class,
        anisotropy_ref=atlas.anisotropy_ref,
        neighbor_indices=atlas.neighbor_indices,
        calibration_residual=atlas.calibration_residual,
        node_confidence=atlas.node_confidence,
        point_support=atlas.point_support,
        view_support=atlas.view_support,
        view_coverage=atlas.view_coverage,
        support_score=atlas.support_score,
        linearness=atlas.linearness,
        planarness=atlas.planarness,
        scattering=atlas.scattering,
        structure_score=_atlas_metric_or_default(atlas, "structure_score", 0.0),
        scale_consistency=_atlas_metric_or_default(atlas, "scale_consistency", 1.0),
        class_consistency=_atlas_metric_or_default(atlas, "class_consistency", 1.0),
        support_consistency=_atlas_metric_or_default(atlas, "support_consistency", 1.0),
        view_balance=_atlas_metric_or_default(atlas, "view_balance", 1.0),
        view_outlier_score=_atlas_metric_or_default(atlas, "view_outlier_score", 1.0),
        robust_inlier_ratio=_atlas_metric_or_default(atlas, "robust_inlier_ratio", 1.0),
        unstable_reason_code=np.asarray(
            atlas.unstable_reason_code
            if atlas.unstable_reason_code is not None
            else _compute_unstable_reason_payload(atlas)[0],
            dtype=np.int32,
        ),
    )


def load_atlas_npz(path) -> FoundationAtlas:
    archive = np.load(path)
    num_nodes = int(archive["positions"].shape[0])
    def _load_or_default(key, default):
        if key in archive.files:
            return archive[key]
        return np.asarray(default)

    return FoundationAtlas(
        positions=archive["positions"],
        support=archive["support"],
        basis=archive["basis"],
        normal=archive["normal"],
        radius=archive["radius"],
        raw_score=archive["raw_score"],
        reliability=archive["reliability"],
        atlas_class=archive["atlas_class"],
        anisotropy_ref=archive["anisotropy_ref"],
        neighbor_indices=archive["neighbor_indices"],
        calibration_residual=archive["calibration_residual"],
        node_confidence=archive["node_confidence"],
        point_support=_load_or_default("point_support", np.ones((num_nodes,), dtype=np.int32)),
        view_support=_load_or_default("view_support", np.ones((num_nodes,), dtype=np.int32)),
        view_coverage=_load_or_default("view_coverage", np.ones((num_nodes,), dtype=np.float32)),
        support_score=_load_or_default("support_score", np.ones((num_nodes,), dtype=np.float32)),
        linearness=_load_or_default("linearness", np.zeros((num_nodes,), dtype=np.float32)),
        planarness=_load_or_default("planarness", np.zeros((num_nodes,), dtype=np.float32)),
        scattering=_load_or_default("scattering", np.ones((num_nodes,), dtype=np.float32)),
        structure_score=_load_or_default("structure_score", np.zeros((num_nodes,), dtype=np.float32)),
        scale_consistency=_load_or_default("scale_consistency", np.ones((num_nodes,), dtype=np.float32)),
        class_consistency=_load_or_default("class_consistency", np.ones((num_nodes,), dtype=np.float32)),
        support_consistency=_load_or_default("support_consistency", np.ones((num_nodes,), dtype=np.float32)),
        view_balance=_load_or_default("view_balance", np.ones((num_nodes,), dtype=np.float32)),
        view_outlier_score=_load_or_default("view_outlier_score", np.ones((num_nodes,), dtype=np.float32)),
        robust_inlier_ratio=_load_or_default("robust_inlier_ratio", np.ones((num_nodes,), dtype=np.float32)),
        unstable_reason_code=_load_or_default("unstable_reason_code", np.full((num_nodes,), -1, dtype=np.int32)).astype(np.int32),
    )


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


_COLMAP_CAMERA_MODEL_PARAM_COUNTS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    return struct.unpack(endian_character + format_char_sequence, fid.read(num_bytes))


def _qvec2rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64).reshape(4)
    return np.array(
        [
            [
                1.0 - 2.0 * qvec[2] ** 2 - 2.0 * qvec[3] ** 2,
                2.0 * qvec[1] * qvec[2] - 2.0 * qvec[0] * qvec[3],
                2.0 * qvec[3] * qvec[1] + 2.0 * qvec[0] * qvec[2],
            ],
            [
                2.0 * qvec[1] * qvec[2] + 2.0 * qvec[0] * qvec[3],
                1.0 - 2.0 * qvec[1] ** 2 - 2.0 * qvec[3] ** 2,
                2.0 * qvec[2] * qvec[3] - 2.0 * qvec[0] * qvec[1],
            ],
            [
                2.0 * qvec[3] * qvec[1] - 2.0 * qvec[0] * qvec[2],
                2.0 * qvec[2] * qvec[3] + 2.0 * qvec[0] * qvec[1],
                1.0 - 2.0 * qvec[1] ** 2 - 2.0 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def _rotation_matrix_to_qvec(rotation: np.ndarray):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3).T
    K = np.array(
        [
            [
                rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
                0.0,
                0.0,
                0.0,
            ],
            [
                rotation[1, 0] + rotation[0, 1],
                rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
                0.0,
                0.0,
            ],
            [
                rotation[2, 0] + rotation[0, 2],
                rotation[2, 1] + rotation[1, 2],
                rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
                0.0,
            ],
            [
                rotation[1, 2] - rotation[2, 1],
                rotation[2, 0] - rotation[0, 2],
                rotation[0, 1] - rotation[1, 0],
                rotation[0, 0] + rotation[1, 1] + rotation[2, 2],
            ],
        ],
        dtype=np.float64,
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0.0:
        qvec *= -1.0
    return qvec.astype(np.float64)


def _camera_params_to_matrix(camera):
    model = str(camera["model"]).upper()
    params = np.asarray(camera["params"], dtype=np.float64).reshape(-1)
    if model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif model == "PINHOLE":
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise ValueError(f"Unsupported COLMAP camera model for alignment: {model}")

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    return intrinsic


def _read_colmap_intrinsics_text(path):
    cameras = {}
    with open(path, "r", encoding="utf-8") as fid:
        for raw_line in fid:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            cameras[camera_id] = {
                "id": camera_id,
                "model": elems[1],
                "width": int(elems[2]),
                "height": int(elems[3]),
                "params": np.asarray(tuple(map(float, elems[4:])), dtype=np.float64),
            }
    return cameras


def _read_colmap_intrinsics_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = _COLMAP_CAMERA_MODEL_PARAM_COUNTS[int(model_id)]
            params = _read_next_bytes(fid, 8 * int(num_params), "d" * int(num_params))
            cameras[int(camera_id)] = {
                "id": int(camera_id),
                "model": model_name,
                "width": int(width),
                "height": int(height),
                "params": np.asarray(params, dtype=np.float64),
            }
    return cameras


def _read_colmap_extrinsics_text(path):
    images = {}
    with open(path, "r", encoding="utf-8") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            image_id = int(elems[0])
            images[image_id] = {
                "id": image_id,
                "qvec": np.asarray(tuple(map(float, elems[1:5])), dtype=np.float64),
                "tvec": np.asarray(tuple(map(float, elems[5:8])), dtype=np.float64),
                "camera_id": int(elems[8]),
                "name": elems[9],
            }
            fid.readline()
    return images


def _read_colmap_extrinsics_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = _read_next_bytes(fid, 64, "idddddddi")
            image_id = int(binary_image_properties[0])
            qvec = np.asarray(binary_image_properties[1:5], dtype=np.float64)
            tvec = np.asarray(binary_image_properties[5:8], dtype=np.float64)
            camera_id = int(binary_image_properties[8])
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * int(num_points2d))
            images[image_id] = {
                "id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": image_name,
            }
    return images


def resolve_colmap_model_dir(path):
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "0", path / "sparse", path / "sparse" / "0"]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() and (candidate / "images.bin").exists():
            return candidate
        if (candidate / "cameras.txt").exists() and (candidate / "images.txt").exists():
            return candidate
    return None


def load_colmap_camera_bundle(path):
    model_dir = resolve_colmap_model_dir(path)
    if model_dir is None:
        raise FileNotFoundError(f"Unable to locate a COLMAP model under: {path}")

    if (model_dir / "cameras.bin").exists() and (model_dir / "images.bin").exists():
        cameras = _read_colmap_intrinsics_binary(model_dir / "cameras.bin")
        images = _read_colmap_extrinsics_binary(model_dir / "images.bin")
    else:
        cameras = _read_colmap_intrinsics_text(model_dir / "cameras.txt")
        images = _read_colmap_extrinsics_text(model_dir / "images.txt")

    entries = []
    for image_payload in images.values():
        camera_payload = cameras.get(int(image_payload["camera_id"]))
        if camera_payload is None:
            continue
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = _qvec2rotmat(image_payload["qvec"])
        world_to_camera[:3, 3] = np.asarray(image_payload["tvec"], dtype=np.float64)
        camera_to_world = np.linalg.inv(world_to_camera)
        entries.append(
            (
                str(image_payload["name"]),
                camera_to_world.astype(np.float32),
                _camera_params_to_matrix(camera_payload).astype(np.float32),
            )
        )

    entries.sort(key=lambda item: item[0].lower())
    if not entries:
        raise ValueError(f"No registered COLMAP images found in {model_dir}")

    return {
        "source_path": str(model_dir),
        "image_names": [entry[0] for entry in entries],
        "cams2world": np.stack([entry[1] for entry in entries], axis=0).astype(np.float32),
        "intrinsics": np.stack([entry[2] for entry in entries], axis=0).astype(np.float32),
    }


def fit_similarity_transform(source_points, target_points):
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("source_points and target_points must have shape [N, 3].")
    if source_points.shape[0] < 3:
        raise ValueError("At least three correspondences are required to fit a similarity transform.")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean[None, :]
    target_centered = target_points - target_mean[None, :]
    covariance = (target_centered.T @ source_centered) / float(source_points.shape[0])
    U, singular_values, Vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = U @ correction @ Vt
    source_variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    if source_variance <= 1e-12:
        raise ValueError("Degenerate source_points for similarity fit.")
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def apply_similarity_to_points(points, scale, rotation, translation):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        return points.copy()
    if points.shape[-1] != 3:
        raise ValueError("points must end with dimension 3.")

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    flat_points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transformed = (float(scale) * (rotation @ flat_points.T)).T + translation[None, :]
    return transformed.reshape(points.shape).astype(np.float32)


def apply_similarity_to_dense_views(points_views, scale, rotation, translation):
    return apply_similarity_to_points(points_views, scale, rotation, translation)


def scale_dense_depth_views(depth_views, scale):
    depth_views = np.asarray(depth_views, dtype=np.float32)
    if depth_views.size == 0:
        return depth_views.copy()
    return (depth_views * float(scale)).astype(np.float32)


def apply_similarity_to_cams2world(cams2world, scale, rotation, translation):
    cams2world = np.asarray(cams2world, dtype=np.float32)
    if cams2world.ndim != 3 or cams2world.shape[-2:] != (4, 4):
        raise ValueError("cams2world must have shape [N, 4, 4].")
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    translated_centers = apply_similarity_to_points(cams2world[:, :3, 3], scale, rotation, translation)
    transformed = cams2world.copy()
    transformed[:, :3, :3] = np.einsum("ij,njk->nik", rotation, transformed[:, :3, :3]).astype(np.float32)
    transformed[:, :3, 3] = translated_centers.astype(np.float32)
    transformed[:, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return transformed


def apply_similarity_to_foundation_atlas(atlas: FoundationAtlas, scale, rotation, translation):
    rotation = np.asarray(rotation, dtype=np.float32).reshape(3, 3)
    rotated_support = np.einsum("ij,njk,lk->nil", rotation, atlas.support, rotation).astype(np.float32)
    rotated_basis = np.einsum("ij,njk->nik", rotation, atlas.basis).astype(np.float32)
    rotated_normal = apply_similarity_to_points(atlas.normal, 1.0, rotation, np.zeros((3,), dtype=np.float32))
    return FoundationAtlas(
        positions=apply_similarity_to_points(atlas.positions, scale, rotation, translation),
        support=rotated_support,
        basis=rotated_basis,
        normal=rotated_normal.astype(np.float32),
        radius=(np.asarray(atlas.radius, dtype=np.float32) * float(scale)).astype(np.float32),
        raw_score=np.asarray(atlas.raw_score, dtype=np.float32).copy(),
        reliability=np.asarray(atlas.reliability, dtype=np.float32).copy(),
        atlas_class=np.asarray(atlas.atlas_class, dtype=np.int64).copy(),
        anisotropy_ref=np.asarray(atlas.anisotropy_ref, dtype=np.float32).copy(),
        neighbor_indices=np.asarray(atlas.neighbor_indices, dtype=np.int64).copy(),
        calibration_residual=np.asarray(atlas.calibration_residual, dtype=np.float32).copy(),
        node_confidence=np.asarray(atlas.node_confidence, dtype=np.float32).copy(),
        point_support=np.asarray(atlas.point_support, dtype=np.int32).copy(),
        view_support=np.asarray(atlas.view_support, dtype=np.int32).copy(),
        view_coverage=np.asarray(atlas.view_coverage, dtype=np.float32).copy(),
        support_score=np.asarray(atlas.support_score, dtype=np.float32).copy(),
        linearness=np.asarray(atlas.linearness, dtype=np.float32).copy(),
        planarness=np.asarray(atlas.planarness, dtype=np.float32).copy(),
        scattering=np.asarray(atlas.scattering, dtype=np.float32).copy(),
        structure_score=_atlas_metric_or_default(atlas, "structure_score", 0.0),
        scale_consistency=_atlas_metric_or_default(atlas, "scale_consistency", 1.0),
        class_consistency=_atlas_metric_or_default(atlas, "class_consistency", 1.0),
        support_consistency=_atlas_metric_or_default(atlas, "support_consistency", 1.0),
        view_balance=_atlas_metric_or_default(atlas, "view_balance", 1.0),
        view_outlier_score=_atlas_metric_or_default(atlas, "view_outlier_score", 1.0),
        robust_inlier_ratio=_atlas_metric_or_default(atlas, "robust_inlier_ratio", 1.0),
        unstable_reason_code=np.asarray(
            atlas.unstable_reason_code
            if atlas.unstable_reason_code is not None
            else _compute_unstable_reason_payload(atlas)[0],
            dtype=np.int32,
        ),
    )


def _build_unique_name_indices(image_names):
    by_name = {}
    by_stem = {}
    ambiguous_stems = set()
    for index, image_name in enumerate(image_names):
        normalized_name = str(image_name).lower()
        if normalized_name not in by_name:
            by_name[normalized_name] = index
        stem = Path(str(image_name)).stem.lower()
        if stem in by_stem:
            ambiguous_stems.add(stem)
        else:
            by_stem[stem] = index
    for stem in ambiguous_stems:
        by_stem.pop(stem, None)
    return by_name, by_stem


def fit_scene_alignment_from_camera_bundles(
    source_image_names,
    source_cams2world,
    target_image_names,
    target_cams2world,
    source_frame: str = "mast3r_sparse_global_alignment",
    target_frame: str = "colmap",
    target_source_path: str | None = None,
):
    source_image_names = [str(name) for name in source_image_names]
    target_image_names = [str(name) for name in target_image_names]
    source_cams2world = np.asarray(source_cams2world, dtype=np.float32)
    target_cams2world = np.asarray(target_cams2world, dtype=np.float32)
    if source_cams2world.ndim != 3 or source_cams2world.shape[-2:] != (4, 4):
        raise ValueError("source_cams2world must have shape [N, 4, 4].")
    if target_cams2world.ndim != 3 or target_cams2world.shape[-2:] != (4, 4):
        raise ValueError("target_cams2world must have shape [N, 4, 4].")

    target_by_name, target_by_stem = _build_unique_name_indices(target_image_names)
    matches = []
    used_target_indices = set()
    for source_index, source_name in enumerate(source_image_names):
        target_index = target_by_name.get(source_name.lower())
        match_kind = "exact"
        if target_index is None:
            target_index = target_by_stem.get(Path(source_name).stem.lower())
            match_kind = "stem"
        if target_index is None or target_index in used_target_indices:
            continue
        used_target_indices.add(target_index)
        matches.append(
            {
                "source_index": int(source_index),
                "target_index": int(target_index),
                "source_image_name": source_name,
                "target_image_name": target_image_names[target_index],
                "match_kind": match_kind,
            }
        )

    if len(matches) < 3:
        raise ValueError(
            f"Need at least 3 shared views to fit scene alignment, found {len(matches)} "
            f"between {len(source_image_names)} source views and {len(target_image_names)} target views."
        )

    source_centers = np.stack([source_cams2world[match["source_index"], :3, 3] for match in matches], axis=0)
    target_centers = np.stack([target_cams2world[match["target_index"], :3, 3] for match in matches], axis=0)
    scale, rotation, translation = fit_similarity_transform(source_centers, target_centers)
    aligned_centers = apply_similarity_to_points(source_centers, scale, rotation, translation)
    center_errors = np.linalg.norm(aligned_centers - target_centers, axis=1)

    rotation_errors_deg = []
    for match in matches:
        predicted_rotation = rotation @ source_cams2world[match["source_index"], :3, :3]
        target_rotation = target_cams2world[match["target_index"], :3, :3]
        rotation_delta = predicted_rotation @ target_rotation.T
        cos_angle = float(np.clip((np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0))
        rotation_errors_deg.append(math.degrees(math.acos(cos_angle)))

    similarity_matrix = np.eye(4, dtype=np.float32)
    similarity_matrix[:3, :3] = float(scale) * np.asarray(rotation, dtype=np.float32)
    similarity_matrix[:3, 3] = np.asarray(translation, dtype=np.float32)

    return {
        "schema_version": 1,
        "applied": True,
        "source_frame": str(source_frame),
        "target_frame": str(target_frame),
        "target_source_path": "" if target_source_path is None else str(Path(target_source_path).expanduser().resolve()),
        "matched_view_count": int(len(matches)),
        "matched_views": matches,
        "scale": float(scale),
        "rotation": np.asarray(rotation, dtype=np.float32).tolist(),
        "translation": np.asarray(translation, dtype=np.float32).tolist(),
        "similarity_matrix": similarity_matrix.tolist(),
        "camera_center_rmse": float(np.sqrt(np.mean(center_errors ** 2))),
        "camera_center_mean": float(np.mean(center_errors)),
        "camera_center_max": float(np.max(center_errors)),
        "rotation_error_deg_mean": float(np.mean(rotation_errors_deg)),
        "rotation_error_deg_median": float(np.median(rotation_errors_deg)),
        "rotation_error_deg_max": float(np.max(rotation_errors_deg)),
    }


def _to_relative_path(path: Path, root: Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _build_image_size_entries(image_names, dense_stats):
    entries = []
    dense_stats = dense_stats or {}
    for image_name in image_names:
        stats = dense_stats.get(str(image_name), {})
        entries.append(
            {
                "width": int(stats.get("width", 0)),
                "height": int(stats.get("height", 0)),
            }
        )
    return entries


def _compute_mast3r_preprocess_metadata(
    image_path,
    image_size: int,
    output_width: int,
    output_height: int,
    patch_size: int = 16,
    square_ok: bool = False,
):
    if image_path is None or int(image_size) <= 0:
        return None

    image_path = Path(image_path)
    if not image_path.exists():
        return None

    source_width, source_height = _load_exif_transposed_size(image_path)

    long_edge = max(int(source_width), int(source_height))
    if long_edge <= 0:
        return None

    scale = float(image_size) / float(long_edge)
    resized_width = int(round(float(source_width) * scale))
    resized_height = int(round(float(source_height) * scale))
    cx = resized_width // 2
    cy = resized_height // 2
    if int(image_size) == 224:
        half = min(cx, cy)
        crop_left = cx - half
        crop_top = cy - half
        crop_right = cx + half
        crop_bottom = cy + half
    else:
        halfw = ((2 * cx) // max(int(patch_size), 1)) * max(int(patch_size), 1) / 2.0
        halfh = ((2 * cy) // max(int(patch_size), 1)) * max(int(patch_size), 1) / 2.0
        if not square_ok and resized_width == resized_height:
            halfh = 3 * halfw / 4.0
        crop_left = cx - halfw
        crop_top = cy - halfh
        crop_right = cx + halfw
        crop_bottom = cy + halfh

    final_width = int(round(crop_right - crop_left))
    final_height = int(round(crop_bottom - crop_top))
    if final_width != int(output_width) or final_height != int(output_height):
        return None

    scale_x = float(resized_width) / float(max(int(source_width), 1))
    scale_y = float(resized_height) / float(max(int(source_height), 1))
    return {
        "coordinate_space": "mast3r_preprocessed_image",
        "source_width": int(source_width),
        "source_height": int(source_height),
        "resized_width": int(resized_width),
        "resized_height": int(resized_height),
        "crop_left": float(crop_left),
        "crop_top": float(crop_top),
        "crop_right": float(crop_right),
        "crop_bottom": float(crop_bottom),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "patch_size": int(patch_size),
        "image_size": int(image_size),
    }


def _load_source_image_size(image_path):
    if image_path is None:
        return None
    image_path = Path(image_path)
    if not image_path.exists():
        return None
    width, height = _load_exif_transposed_size(image_path)
    return {"width": int(width), "height": int(height)}


def _load_exif_transposed_size(image_path):
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        width, height = ImageOps.exif_transpose(img).size
    return int(width), int(height)


def _map_preprocessed_pixels_to_source_xy(
    corr_xy,
    output_width: int,
    output_height: int,
    image_path=None,
    preprocess_image_size=None,
    patch_size: int = 16,
    square_ok: bool = False,
    preprocess_metadata=None,
):
    corr_xy = np.asarray(corr_xy, dtype=np.float32)
    metadata = preprocess_metadata
    if metadata is None and image_path is not None and preprocess_image_size is not None:
        metadata = _compute_mast3r_preprocess_metadata(
            image_path,
            int(preprocess_image_size),
            int(output_width),
            int(output_height),
            patch_size=patch_size,
            square_ok=square_ok,
        )

    if metadata is None:
        return corr_xy.astype(np.float32), int(output_width), int(output_height), "dense_pointmap_pixels"

    mapped = corr_xy.astype(np.float32).copy()
    mapped[:, 0] += float(metadata["crop_left"])
    mapped[:, 1] += float(metadata["crop_top"])
    mapped[:, 0] *= float(metadata["source_width"]) / float(max(int(metadata["resized_width"]), 1))
    mapped[:, 1] *= float(metadata["source_height"]) / float(max(int(metadata["resized_height"]), 1))
    return (
        mapped.astype(np.float32),
        int(metadata["source_width"]),
        int(metadata["source_height"]),
        str(metadata["coordinate_space"]),
    )


def _map_intrinsics_to_correspondence_pixel_space(intrinsics, preprocess_metadata=None):
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    if preprocess_metadata is None:
        return intrinsics.astype(np.float32).copy()

    mapped = intrinsics.astype(np.float32).copy()
    scale_x = float(preprocess_metadata["source_width"]) / float(max(int(preprocess_metadata["resized_width"]), 1))
    scale_y = float(preprocess_metadata["source_height"]) / float(max(int(preprocess_metadata["resized_height"]), 1))
    mapped[0, 0] *= scale_x
    mapped[1, 1] *= scale_y
    mapped[0, 2] = (float(intrinsics[0, 2]) + float(preprocess_metadata["crop_left"])) * scale_x
    mapped[1, 2] = (float(intrinsics[1, 2]) + float(preprocess_metadata["crop_top"])) * scale_y
    return mapped.astype(np.float32)


def _project_world_points_to_pixels(points_world, camera_to_world, intrinsics):
    points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    camera_to_world = np.asarray(camera_to_world, dtype=np.float32).reshape(4, 4)
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(3, 3)
    if points_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
    points_h = np.concatenate(
        (points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)),
        axis=1,
    )
    points_camera_h = (world_to_camera @ points_h.T).T
    points_camera = points_camera_h[:, :3]
    depth = points_camera[:, 2].astype(np.float32)
    projected = (intrinsics @ points_camera.T).T
    denom = projected[:, 2:3]
    pixels = np.divide(
        projected[:, :2],
        denom,
        out=np.full((points_world.shape[0], 2), np.nan, dtype=np.float32),
        where=np.abs(denom) > 1e-8,
    )
    return pixels.astype(np.float32), depth.astype(np.float32)


def audit_dense_correspondence_alignment(
    image_names,
    points_views,
    confidence_views,
    target_image_names,
    target_cams2world,
    target_intrinsics,
    min_conf_thr=0.0,
    image_paths=None,
    preprocess_image_size=None,
    patch_size: int = 16,
    square_ok: bool = False,
    max_samples_per_view: int = 2048,
    seed: int = 42,
):
    image_names = [str(name) for name in image_names]
    target_image_names = [str(name) for name in target_image_names]
    target_cams2world = np.asarray(target_cams2world, dtype=np.float32)
    target_intrinsics = np.asarray(target_intrinsics, dtype=np.float32)
    image_paths = [] if image_paths is None else list(image_paths)
    rng = np.random.default_rng(int(seed))
    target_by_name, target_by_stem = _build_unique_name_indices(target_image_names)

    matched_views = []
    per_view = {}
    all_errors = []
    total_sampled_corr = 0
    total_projected_corr = 0
    total_in_frame_corr = 0

    for source_index, image_name in enumerate(image_names):
        target_index = target_by_name.get(image_name.lower())
        match_kind = "exact"
        if target_index is None:
            target_index = target_by_stem.get(Path(image_name).stem.lower())
            match_kind = "stem"

        if target_index is None:
            continue

        points_view = np.asarray(points_views[source_index], dtype=np.float32)
        confidence_view = np.asarray(confidence_views[source_index], dtype=np.float32)
        if (
            points_view.ndim == 2
            and points_view.shape[1] == 3
            and confidence_view.ndim == 2
            and points_view.shape[0] == int(confidence_view.shape[0] * confidence_view.shape[1])
        ):
            points_view = points_view.reshape(confidence_view.shape[0], confidence_view.shape[1], 3)
        if points_view.ndim != 3 or points_view.shape[2] != 3:
            continue
        if confidence_view.ndim != 2 or confidence_view.shape != points_view.shape[:2]:
            continue

        valid = np.isfinite(points_view).all(axis=2) & np.isfinite(confidence_view) & (confidence_view >= float(min_conf_thr))
        yy, xx = np.nonzero(valid)
        if yy.size == 0:
            per_view[image_name] = {
                "match_kind": match_kind,
                "target_image_name": target_image_names[target_index],
                "sampled_corr": 0,
                "projected_corr": 0,
                "in_frame_corr": 0,
                "mean_px_error": None,
                "median_px_error": None,
                "p90_px_error": None,
            }
            matched_views.append({"source_image_name": image_name, "target_image_name": target_image_names[target_index], "match_kind": match_kind})
            continue

        corr_xy = np.stack((xx, yy), axis=1).astype(np.float32)
        corr_xyz = points_view[valid].astype(np.float32)
        if int(max_samples_per_view) > 0 and corr_xy.shape[0] > int(max_samples_per_view):
            sample_indices = rng.choice(corr_xy.shape[0], size=int(max_samples_per_view), replace=False)
            sample_indices.sort()
            corr_xy = corr_xy[sample_indices]
            corr_xyz = corr_xyz[sample_indices]

        image_path = image_paths[source_index] if source_index < len(image_paths) else None
        preprocess_metadata = None
        if image_path is not None and preprocess_image_size is not None:
            preprocess_metadata = _compute_mast3r_preprocess_metadata(
                image_path,
                int(preprocess_image_size),
                int(points_view.shape[1]),
                int(points_view.shape[0]),
                patch_size=patch_size,
                square_ok=square_ok,
            )
        corr_xy_source, source_width, source_height, coord_space = _map_preprocessed_pixels_to_source_xy(
            corr_xy,
            int(points_view.shape[1]),
            int(points_view.shape[0]),
            image_path=image_path,
            preprocess_image_size=preprocess_image_size,
            patch_size=patch_size,
            square_ok=square_ok,
            preprocess_metadata=preprocess_metadata,
        )
        projection_intrinsics = _map_intrinsics_to_correspondence_pixel_space(
            target_intrinsics[target_index],
            preprocess_metadata=preprocess_metadata,
        )
        projected_xy, depth = _project_world_points_to_pixels(
            corr_xyz,
            target_cams2world[target_index],
            projection_intrinsics,
        )
        projected = np.isfinite(projected_xy).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
        in_frame = projected & (
            (projected_xy[:, 0] >= 0.0)
            & (projected_xy[:, 0] <= float(max(source_width - 1, 0)))
            & (projected_xy[:, 1] >= 0.0)
            & (projected_xy[:, 1] <= float(max(source_height - 1, 0)))
        )

        errors = np.linalg.norm(projected_xy[in_frame] - corr_xy_source[in_frame], axis=1).astype(np.float32)
        if errors.size > 0:
            mean_px_error = float(np.mean(errors))
            median_px_error = float(np.median(errors))
            p90_px_error = float(np.quantile(errors, 0.90))
            all_errors.append(errors)
        else:
            mean_px_error = None
            median_px_error = None
            p90_px_error = None

        total_sampled_corr += int(corr_xy.shape[0])
        total_projected_corr += int(projected.sum())
        total_in_frame_corr += int(in_frame.sum())
        per_view[image_name] = {
            "match_kind": match_kind,
            "target_image_name": target_image_names[target_index],
            "coordinate_space": coord_space,
            "source_width": int(source_width),
            "source_height": int(source_height),
            "sampled_corr": int(corr_xy.shape[0]),
            "projected_corr": int(projected.sum()),
            "in_frame_corr": int(in_frame.sum()),
            "mean_px_error": mean_px_error,
            "median_px_error": median_px_error,
            "p90_px_error": p90_px_error,
        }
        matched_views.append(
            {
                "source_image_name": image_name,
                "target_image_name": target_image_names[target_index],
                "match_kind": match_kind,
            }
        )

    concatenated_errors = np.concatenate(all_errors, axis=0) if all_errors else np.zeros((0,), dtype=np.float32)
    mean_px_error = None if concatenated_errors.size == 0 else float(np.mean(concatenated_errors))
    median_px_error = None if concatenated_errors.size == 0 else float(np.median(concatenated_errors))
    p90_px_error = None if concatenated_errors.size == 0 else float(np.quantile(concatenated_errors, 0.90))
    return {
        "schema_version": 1,
        "metric_space": "correspondence_pixel_space",
        "min_conf_thr": float(min_conf_thr),
        "max_samples_per_view": int(max_samples_per_view),
        "matched_view_count": int(len(matched_views)),
        "matched_views": matched_views,
        "total_sampled_corr": int(total_sampled_corr),
        "total_projected_corr": int(total_projected_corr),
        "total_in_frame_corr": int(total_in_frame_corr),
        "mean_px_error": mean_px_error,
        "median_px_error": median_px_error,
        "p90_px_error": p90_px_error,
        "views": per_view,
    }


def validate_scene_alignment_contract(
    scene_alignment,
    dense_correspondence_audit,
    max_camera_center_rmse: float = 0.05,
    max_rotation_error_deg: float = 1.0,
    max_median_px_error: float = 12.0,
    min_in_frame_corr: int = 256,
):
    failures = []
    scene_alignment = {} if scene_alignment is None else dict(scene_alignment)
    dense_correspondence_audit = {} if dense_correspondence_audit is None else dict(dense_correspondence_audit)

    matched_view_count = int(scene_alignment.get("matched_view_count", 0))
    if matched_view_count < 3:
        failures.append(f"scene_alignment.matched_view_count={matched_view_count} < 3")

    center_rmse = float(scene_alignment.get("camera_center_rmse", float("inf")))
    if not np.isfinite(center_rmse) or center_rmse > float(max_camera_center_rmse):
        failures.append(
            f"scene_alignment.camera_center_rmse={center_rmse:.6f} > {float(max_camera_center_rmse):.6f}"
        )

    rotation_error = float(scene_alignment.get("rotation_error_deg_mean", float("inf")))
    if not np.isfinite(rotation_error) or rotation_error > float(max_rotation_error_deg):
        failures.append(
            f"scene_alignment.rotation_error_deg_mean={rotation_error:.6f} > {float(max_rotation_error_deg):.6f}"
        )

    in_frame_corr = int(dense_correspondence_audit.get("total_in_frame_corr", 0))
    if in_frame_corr < int(min_in_frame_corr):
        failures.append(
            f"dense_correspondence_audit.total_in_frame_corr={in_frame_corr} < {int(min_in_frame_corr)}"
        )

    median_px_error = dense_correspondence_audit.get("median_px_error")
    if median_px_error is None or not np.isfinite(float(median_px_error)):
        failures.append("dense_correspondence_audit.median_px_error is missing or non-finite")
    elif float(median_px_error) > float(max_median_px_error):
        failures.append(
            f"dense_correspondence_audit.median_px_error={float(median_px_error):.6f} > {float(max_median_px_error):.6f}"
        )

    return {
        "schema_version": 1,
        "passed": len(failures) == 0,
        "thresholds": {
            "max_camera_center_rmse": float(max_camera_center_rmse),
            "max_rotation_error_deg": float(max_rotation_error_deg),
            "max_median_px_error": float(max_median_px_error),
            "min_in_frame_corr": int(min_in_frame_corr),
        },
        "failures": failures,
    }


def save_camera_bundle(path, image_names, cams2world, intrinsics, dense_stats=None, image_paths=None):
    path = Path(path)
    image_names = [str(name) for name in image_names]
    cams2world = np.asarray(cams2world, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    if cams2world.ndim != 3 or cams2world.shape[-2:] != (4, 4):
        raise ValueError("cams2world must have shape [N, 4, 4].")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [N, 3, 3].")
    if len(image_names) != cams2world.shape[0] or cams2world.shape[0] != intrinsics.shape[0]:
        raise ValueError("image_names, cams2world, and intrinsics must agree on camera count.")

    payload = {
        "schema_version": 1,
        "num_cameras": int(len(image_names)),
        "image_names": image_names,
        "cams2world": cams2world.tolist(),
        "world2cams": np.linalg.inv(cams2world).tolist(),
        "intrinsics": intrinsics.tolist(),
        "camera_centers": cams2world[:, :3, 3].tolist(),
        "image_sizes": _build_image_size_entries(image_names, dense_stats),
    }
    if image_paths is not None:
        payload["source_image_paths"] = [str(Path(path_item)) for path_item in image_paths]
        payload["source_image_sizes"] = [_load_source_image_size(path_item) for path_item in image_paths]

    save_json(payload, path)
    return payload


def save_correspondence_manifest(
    path,
    image_names,
    dense_output_dir,
    dense_stats,
    min_conf_thr=0.0,
    image_paths=None,
    preprocess_image_size=None,
    patch_size: int = 16,
    square_ok: bool = False,
    sparse_correspondence_dir=None,
):
    path = Path(path)
    run_root = path.parent
    dense_output_dir = Path(dense_output_dir)
    points_root = dense_output_dir / "points3d"
    confidence_root = dense_output_dir / "confidence"
    depth_root = dense_output_dir / "depth"
    sparse_root = None if sparse_correspondence_dir is None else Path(sparse_correspondence_dir)
    dense_stats = dense_stats or {}

    payload = {
        "schema_version": 1,
        "min_conf_thr": float(min_conf_thr),
        "correspondence_schema": "dense_or_sparse_atlas_node",
        "dense_geometry_root": _to_relative_path(dense_output_dir, run_root),
        "points_root": _to_relative_path(points_root, run_root),
        "confidence_root": _to_relative_path(confidence_root, run_root),
        "depth_root": _to_relative_path(depth_root, run_root),
        "sparse_root": None if sparse_root is None else _to_relative_path(sparse_root, run_root),
        "preprocess_image_size": None if preprocess_image_size is None else int(preprocess_image_size),
        "patch_size": int(patch_size),
        "square_ok": bool(square_ok),
        "view_order": [str(name) for name in image_names],
        "views": {},
    }

    image_paths = [] if image_paths is None else list(image_paths)
    for index, image_name in enumerate(image_names):
        image_name = str(image_name)
        stem = Path(image_name).stem
        stats = dense_stats.get(image_name, {})
        points_path = points_root / f"{stem}.npy"
        confidence_path = confidence_root / f"{stem}.npy"
        depth_path = depth_root / f"{stem}.npy"
        sparse_path = None if sparse_root is None else sparse_root / f"{stem}.npz"
        view_payload = {
            "image_name": image_name,
            "image_index": int(index),
            "stem": stem,
            "width": int(stats.get("width", 0)),
            "height": int(stats.get("height", 0)),
            "mean_confidence": float(stats.get("mean_confidence", 0.0)),
            "median_confidence": float(stats.get("median_confidence", 0.0)),
            "finite_points": int(stats.get("finite_points", 0)),
            "points_above_conf_thr": int(stats.get("points_above_conf_thr", 0)),
            "points_path": _to_relative_path(points_path, run_root) if points_path.exists() else None,
            "confidence_path": _to_relative_path(confidence_path, run_root) if confidence_path.exists() else None,
            "depth_path": _to_relative_path(depth_path, run_root) if depth_path.exists() else None,
            "sparse_path": _to_relative_path(sparse_path, run_root) if sparse_path is not None and sparse_path.exists() else None,
            "sparse_correspondence_count": int(stats.get("sparse_correspondence_count", 0)),
            "correspondence_mode": str(stats.get("sparse_correspondence_mode", "dense_pointmap")),
        }
        if index < len(image_paths):
            view_payload["source_image_path"] = str(Path(image_paths[index]))
            preprocess_metadata = _compute_mast3r_preprocess_metadata(
                image_paths[index],
                preprocess_image_size if preprocess_image_size is not None else 0,
                int(stats.get("width", 0)),
                int(stats.get("height", 0)),
                patch_size=patch_size,
                square_ok=square_ok,
            )
            if preprocess_metadata is not None:
                view_payload.update(preprocess_metadata)
        payload["views"][image_name] = view_payload

    save_json(payload, path)
    return payload


def build_atlas_hash_metadata(atlas: FoundationAtlas, cell_size=None):
    positions = np.asarray(atlas.positions, dtype=np.float32)
    radius = np.asarray(atlas.radius, dtype=np.float32).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("atlas.positions must have shape [N, 3].")

    if positions.shape[0] == 0:
        return {
            "kind": "voxel_hash",
            "schema_version": 1,
            "cell_size": 0.0,
            "node_count": 0,
            "bucket_count": 0,
            "neighbor_k": 0,
            "buckets": [],
        }

    if cell_size is None:
        finite_radius = radius[np.isfinite(radius) & (radius > 0.0)]
        if finite_radius.size > 0:
            cell_size = float(np.median(finite_radius))
        else:
            span = np.linalg.norm(np.ptp(positions, axis=0))
            cell_size = float(span / max(np.cbrt(max(positions.shape[0], 1)), 1.0))
    cell_size = max(float(cell_size), 1e-6)

    bbox_min = positions.min(axis=0)
    bbox_max = positions.max(axis=0)
    coords = np.floor((positions - bbox_min[None, :]) / cell_size).astype(np.int32)
    unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)

    buckets = []
    bucket_sizes = []
    for bucket_id, coord in enumerate(unique_coords):
        node_ids = np.flatnonzero(inverse == bucket_id).astype(np.int64)
        bucket_sizes.append(int(node_ids.shape[0]))
        buckets.append(
            {
                "coord": coord.astype(int).tolist(),
                "node_ids": node_ids.tolist(),
            }
        )

    bucket_sizes_array = np.asarray(bucket_sizes, dtype=np.float32) if bucket_sizes else np.zeros((0,), dtype=np.float32)
    return {
        "kind": "voxel_hash",
        "schema_version": 1,
        "cell_size": float(cell_size),
        "bbox_min": bbox_min.astype(np.float32).tolist(),
        "bbox_max": bbox_max.astype(np.float32).tolist(),
        "node_count": int(positions.shape[0]),
        "bucket_count": int(len(buckets)),
        "neighbor_k": int(atlas.neighbor_indices.shape[1]) if atlas.neighbor_indices.ndim == 2 else 0,
        "mean_bucket_size": float(bucket_sizes_array.mean()) if bucket_sizes_array.size else 0.0,
        "max_bucket_size": int(bucket_sizes_array.max()) if bucket_sizes_array.size else 0,
        "buckets": buckets,
    }


def _chunked_nearest_indices(query_points: np.ndarray, ref_points: np.ndarray, chunk_size: int = 1024):
    query_points = np.asarray(query_points, dtype=np.float32)
    ref_points = np.asarray(ref_points, dtype=np.float32)
    nearest_ids = np.empty((query_points.shape[0],), dtype=np.int64)
    nearest_dist = np.empty((query_points.shape[0],), dtype=np.float32)

    for start in range(0, query_points.shape[0], max(int(chunk_size), 1)):
        end = min(start + max(int(chunk_size), 1), query_points.shape[0])
        chunk = query_points[start:end]
        diff = chunk[:, None, :] - ref_points[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        local_ids = np.argmin(d2, axis=1)
        local_dist = np.sqrt(d2[np.arange(local_ids.shape[0]), local_ids].clip(min=1e-12))
        nearest_ids[start:end] = local_ids.astype(np.int64)
        nearest_dist[start:end] = local_dist.astype(np.float32)

    return nearest_ids, nearest_dist


def build_reference_camera_evidence(
    atlas: FoundationAtlas,
    dense_geometry,
    image_names,
    cams2world=None,
    chunk_size: int = 1024,
    assignment_radius_mult: float = 2.5,
):
    positions = np.asarray(atlas.positions, dtype=np.float32)
    radius = np.asarray(atlas.radius, dtype=np.float32).reshape(-1)
    image_names = [str(name) for name in image_names]
    num_nodes = int(positions.shape[0])
    num_views = int(len(image_names))

    reference_camera_ids = np.full((num_nodes,), -1, dtype=np.int64)
    reference_camera_scores = np.zeros((num_nodes,), dtype=np.float32)
    view_weights = np.zeros((num_nodes, num_views), dtype=np.float32)
    view_counts = np.zeros((num_nodes, num_views), dtype=np.int32)

    if num_nodes == 0 or num_views == 0:
        reference_quality = {
            "reference_camera_counts": np.zeros((num_nodes,), dtype=np.int32),
            "reference_camera_weights": np.zeros((num_nodes,), dtype=np.float32),
            "reference_camera_weighted_score": np.zeros((num_nodes,), dtype=np.float32),
            "reference_camera_visibility": np.zeros((num_nodes,), dtype=np.float32),
            "reference_camera_reprojection_score": np.zeros((num_nodes,), dtype=np.float32),
            "reference_camera_view_support": np.zeros((num_nodes,), dtype=np.int32),
            "reference_camera_total_weight": np.zeros((num_nodes,), dtype=np.float32),
            "reference_camera_dominance": np.zeros((num_nodes,), dtype=np.float32),
        }
        return reference_camera_ids, reference_camera_scores, view_weights, view_counts, reference_quality

    dense_points = np.asarray(dense_geometry.get("points", []), dtype=np.float32)
    dense_confidences = np.asarray(dense_geometry.get("confidences", []), dtype=np.float32).reshape(-1)
    dense_image_ids = np.asarray(dense_geometry.get("image_ids", []), dtype=np.int32).reshape(-1)
    dense_errors = np.asarray(dense_geometry.get("point_errors", []), dtype=np.float32).reshape(-1)
    dense_depths = np.asarray(dense_geometry.get("depths", []), dtype=np.float32).reshape(-1)
    view_error_weights = np.zeros((num_nodes, num_views), dtype=np.float32)
    view_depth_counts = np.zeros((num_nodes, num_views), dtype=np.int32)
    if dense_points.ndim == 2 and dense_points.shape[1] == 3 and dense_points.shape[0] > 0:
        if dense_confidences.shape[0] != dense_points.shape[0] or dense_image_ids.shape[0] != dense_points.shape[0]:
            raise ValueError("dense_geometry points, confidences, and image_ids must have matching lengths.")
        if np.any(dense_image_ids < 0) or np.any(dense_image_ids >= num_views):
            raise ValueError("dense_geometry image_ids must be in [0, len(image_names)).")

        confidence_scores = _normalize_confidence(dense_confidences, image_ids=dense_image_ids)
        nearest_ids, nearest_dist = _chunked_nearest_indices(dense_points, positions, chunk_size=chunk_size)
        base_radius = max(float(np.median(radius[np.isfinite(radius) & (radius > 0.0)])) if np.any(np.isfinite(radius) & (radius > 0.0)) else 1e-3, 1e-6)
        local_radius = np.maximum(radius[nearest_ids], base_radius * 0.35)
        accept_radius = np.maximum(local_radius * float(assignment_radius_mult), base_radius)
        accepted = np.isfinite(nearest_dist) & (nearest_dist <= accept_radius)

        if np.any(accepted):
            accepted_ids = nearest_ids[accepted]
            accepted_views = dense_image_ids[accepted]
            accepted_dist = nearest_dist[accepted]
            accepted_radius = local_radius[accepted]
            accepted_conf = confidence_scores[accepted]
            distance_score = np.exp(-accepted_dist / np.clip(accepted_radius * 1.5, 1e-6, None))
            if dense_errors.shape[0] == dense_points.shape[0]:
                accepted_error = dense_errors[accepted]
                finite_error = np.isfinite(accepted_error) & (accepted_error >= 0.0)
                reproj_score = np.ones_like(accepted_conf, dtype=np.float32)
                reproj_score[finite_error] = 1.0 / (1.0 + accepted_error[finite_error].astype(np.float32))
            else:
                reproj_score = np.ones_like(accepted_conf, dtype=np.float32)
            if dense_depths.shape[0] == dense_points.shape[0]:
                accepted_depth = dense_depths[accepted]
                depth_valid = np.isfinite(accepted_depth) & (accepted_depth > 0.0)
            else:
                depth_valid = np.ones_like(accepted_conf, dtype=bool)
            accepted_weight = accepted_conf * distance_score * (0.35 + 0.65 * reproj_score)
            if accepted_weight.size > 0:
                # Cap repeated observations from the same node/view after scoring.
                # Counts still contribute through count_strength below, but raw
                # weight is prevented from making a single dense view look like
                # multi-view support.
                pair_key = accepted_ids.astype(np.int64) * max(num_views, 1) + accepted_views.astype(np.int64)
                unique_pairs, pair_inverse = np.unique(pair_key, return_inverse=True)
                pair_sum = np.bincount(pair_inverse, weights=accepted_weight, minlength=unique_pairs.shape[0]).astype(np.float32)
                positive_pair_sum = pair_sum[pair_sum > 0.0]
                pair_cap = max(float(np.median(positive_pair_sum)) if positive_pair_sum.size else 1.0, 1e-6) * 1.35
                pair_scale = np.minimum(1.0, pair_cap / np.clip(pair_sum, 1e-6, None))
                accepted_weight = accepted_weight * pair_scale[pair_inverse]
            np.add.at(view_weights, (accepted_ids, accepted_views), accepted_weight.astype(np.float32))
            np.add.at(view_counts, (accepted_ids, accepted_views), 1)
            np.add.at(view_error_weights, (accepted_ids, accepted_views), reproj_score.astype(np.float32))
            np.add.at(view_depth_counts, (accepted_ids[depth_valid], accepted_views[depth_valid]), 1)

    total_weight = view_weights.sum(axis=1)
    supported = total_weight > 0.0
    reference_quality = {
        "reference_camera_counts": np.zeros((num_nodes,), dtype=np.int32),
        "reference_camera_weights": np.zeros((num_nodes,), dtype=np.float32),
        "reference_camera_weighted_score": np.zeros((num_nodes,), dtype=np.float32),
        "reference_camera_visibility": np.zeros((num_nodes,), dtype=np.float32),
        "reference_camera_reprojection_score": np.zeros((num_nodes,), dtype=np.float32),
        "reference_camera_view_support": np.zeros((num_nodes,), dtype=np.int32),
        "reference_camera_total_weight": np.zeros((num_nodes,), dtype=np.float32),
        "reference_camera_dominance": np.zeros((num_nodes,), dtype=np.float32),
    }
    if np.any(supported):
        count_strength_matrix = 1.0 - np.exp(-view_counts.astype(np.float32) / 4.0)
        depth_visibility_matrix = view_depth_counts.astype(np.float32) / np.maximum(view_counts.astype(np.float32), 1.0)
        reproj_matrix = view_error_weights / np.maximum(view_counts.astype(np.float32), 1.0)
        per_node_positive_views = np.maximum((view_counts > 0).sum(axis=1).astype(np.float32), 1.0)
        view_support_matrix = np.where(view_counts > 0, 1.0 / np.sqrt(per_node_positive_views[:, None]), 0.0).astype(np.float32)
        balanced_view_score = (
            np.sqrt(np.clip(view_weights, 0.0, None))
            * (0.35 + 0.65 * count_strength_matrix)
            * (0.45 + 0.55 * depth_visibility_matrix)
            * (0.45 + 0.55 * reproj_matrix)
            * (0.75 + 0.25 * view_support_matrix)
        )
        best_view = np.argmax(balanced_view_score, axis=1)
        best_weight = view_weights[np.arange(num_nodes), best_view]
        best_count = view_counts[np.arange(num_nodes), best_view].astype(np.float32)
        best_balanced_score = balanced_view_score[np.arange(num_nodes), best_view]
        best_visibility = depth_visibility_matrix[np.arange(num_nodes), best_view]
        best_reproj = reproj_matrix[np.arange(num_nodes), best_view]
        support_scale = max(float(np.median(total_weight[supported])), 1e-6)
        support_strength = 1.0 - np.exp(-total_weight[supported] / support_scale)
        dominance = best_weight[supported] / np.clip(total_weight[supported], 1e-6, None)
        view_support_strength = np.clip((view_counts[supported] > 0).sum(axis=1).astype(np.float32) / max(min(num_views, 4), 1), 0.0, 1.0)
        count_strength = 1.0 - np.exp(-best_count[supported] / 4.0)
        reference_camera_ids[supported] = best_view[supported].astype(np.int64)
        reference_camera_scores[supported] = np.clip(
            np.sqrt(dominance * support_strength)
            * (0.30 + 0.70 * count_strength)
            * (0.70 + 0.30 * view_support_strength)
            * (0.50 + 0.50 * best_visibility[supported])
            * (0.45 + 0.55 * best_reproj[supported]),
            0.0,
            1.0,
        ).astype(np.float32)
        reference_quality["reference_camera_counts"][supported] = best_count[supported].astype(np.int32)
        reference_quality["reference_camera_weights"][supported] = best_weight[supported].astype(np.float32)
        reference_quality["reference_camera_weighted_score"][supported] = best_balanced_score[supported].astype(np.float32)
        reference_quality["reference_camera_visibility"][supported] = best_visibility[supported].astype(np.float32)
        reference_quality["reference_camera_reprojection_score"][supported] = best_reproj[supported].astype(np.float32)
        reference_quality["reference_camera_view_support"][supported] = (view_counts[supported] > 0).sum(axis=1).astype(np.int32)
        reference_quality["reference_camera_total_weight"][supported] = total_weight[supported].astype(np.float32)
        reference_quality["reference_camera_dominance"][supported] = dominance.astype(np.float32)

    missing = reference_camera_ids < 0
    if np.any(missing) and cams2world is not None:
        cams2world = np.asarray(cams2world, dtype=np.float32)
        if cams2world.ndim == 3 and cams2world.shape[-2:] == (4, 4) and cams2world.shape[0] == num_views:
            camera_centers = cams2world[:, :3, 3]
            fallback_ids, fallback_dist = _chunked_nearest_indices(positions[missing], camera_centers, chunk_size=max(128, num_views))
            scale = max(float(np.median(fallback_dist)) if fallback_dist.size > 0 else 1.0, 1e-6)
            reference_camera_ids[missing] = fallback_ids.astype(np.int64)
            reference_camera_scores[missing] = (0.15 * np.exp(-fallback_dist / scale)).astype(np.float32)
            reference_quality["reference_camera_weights"][missing] = reference_camera_scores[missing]
            reference_quality["reference_camera_weighted_score"][missing] = reference_camera_scores[missing]
            reference_quality["reference_camera_visibility"][missing] = 0.0
            reference_quality["reference_camera_reprojection_score"][missing] = 0.0
            reference_quality["reference_camera_view_support"][missing] = 0
            reference_quality["reference_camera_total_weight"][missing] = reference_camera_scores[missing]
            reference_quality["reference_camera_dominance"][missing] = 0.0

    return reference_camera_ids, reference_camera_scores, view_weights, view_counts, reference_quality


def save_reference_camera_evidence(
    path,
    reference_camera_ids,
    reference_camera_scores,
    image_names,
    view_weights=None,
    view_counts=None,
    reference_quality=None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reference_camera_ids": np.asarray(reference_camera_ids, dtype=np.int64),
        "reference_camera_scores": np.asarray(reference_camera_scores, dtype=np.float32),
        "image_names": np.asarray(image_names, dtype=np.str_),
    }
    if view_weights is not None:
        payload["view_weights"] = np.asarray(view_weights, dtype=np.float32)
    if view_counts is not None:
        payload["view_counts"] = np.asarray(view_counts, dtype=np.int32)
    if isinstance(reference_quality, dict):
        for key, value in reference_quality.items():
            payload[str(key)] = np.asarray(value)
    np.savez_compressed(path, **payload)


def save_foundation_atlas_sidecars(
    output_dir,
    atlas: FoundationAtlas,
    image_names,
    cams2world,
    intrinsics,
    dense_geometry,
    dense_stats=None,
    image_paths=None,
    min_conf_thr=0.0,
    hash_cell_size=None,
    scene_alignment=None,
    preprocess_image_size=None,
    sparse_correspondence_dir=None,
):
    output_dir = Path(output_dir)
    dense_output_dir = output_dir / "dense_geometry"
    if dense_stats is None:
        dense_stats_path = dense_output_dir / "dense_views_stats.json"
        dense_stats = {}
        if dense_stats_path.exists():
            with open(dense_stats_path, "r", encoding="utf-8") as handle:
                dense_stats = json.load(handle)

    camera_bundle = save_camera_bundle(
        output_dir / "camera_bundle.json",
        image_names,
        cams2world,
        intrinsics,
        dense_stats=dense_stats,
        image_paths=image_paths,
    )
    correspondence_manifest = save_correspondence_manifest(
        output_dir / "correspondence_manifest.json",
        image_names,
        dense_output_dir,
        dense_stats,
        min_conf_thr=min_conf_thr,
        image_paths=image_paths,
        preprocess_image_size=preprocess_image_size,
        sparse_correspondence_dir=sparse_correspondence_dir,
    )
    atlas_hash = build_atlas_hash_metadata(atlas, cell_size=hash_cell_size)
    save_json(atlas_hash, output_dir / "atlas_hash.json")
    reference_camera_ids, reference_camera_scores, view_weights, view_counts, reference_quality = build_reference_camera_evidence(
        atlas,
        dense_geometry,
        image_names,
        cams2world=cams2world,
    )
    save_reference_camera_evidence(
        output_dir / "reference_camera_evidence.npz",
        reference_camera_ids,
        reference_camera_scores,
        image_names=image_names,
        view_weights=view_weights,
        view_counts=view_counts,
        reference_quality=reference_quality,
    )
    unstable_audit = build_unstable_node_audit(atlas)
    unstable_audit_path = output_dir / "atlas_unstable_audit.json"
    save_json(unstable_audit, unstable_audit_path)
    summary = {
        "camera_bundle_path": str(output_dir / "camera_bundle.json"),
        "correspondence_manifest_path": str(output_dir / "correspondence_manifest.json"),
        "atlas_hash_path": str(output_dir / "atlas_hash.json"),
        "reference_camera_evidence_path": str(output_dir / "reference_camera_evidence.npz"),
        "unstable_audit_path": str(unstable_audit_path),
        "camera_count": int(len(camera_bundle["image_names"])),
        "correspondence_view_count": int(len(correspondence_manifest["views"])),
        "hash_bucket_count": int(atlas_hash.get("bucket_count", 0)),
        "reference_camera_coverage": float(np.mean(reference_camera_ids >= 0)) if reference_camera_ids.size else 0.0,
        "reference_camera_score_mean": float(np.mean(reference_camera_scores)) if reference_camera_scores.size else 0.0,
        "reference_camera_weighted_score_mean": float(np.mean(reference_quality.get("reference_camera_weighted_score", np.zeros_like(reference_camera_scores)))) if reference_camera_scores.size else 0.0,
        "reference_camera_visibility_mean": float(np.mean(reference_quality.get("reference_camera_visibility", np.zeros_like(reference_camera_scores)))) if reference_camera_scores.size else 0.0,
        "reference_camera_reprojection_score_mean": float(np.mean(reference_quality.get("reference_camera_reprojection_score", np.zeros_like(reference_camera_scores)))) if reference_camera_scores.size else 0.0,
        "reference_camera_view_support_mean": float(np.mean(reference_quality.get("reference_camera_view_support", np.zeros_like(reference_camera_scores)))) if reference_camera_scores.size else 0.0,
        "unstable_audit_count": int(unstable_audit.get("unstable_count", 0)),
        "unstable_reason_counts": {
            str(name): int(count) for name, count in unstable_audit.get("reason_counts", {}).items()
        },
    }
    if scene_alignment is not None:
        save_json(scene_alignment, output_dir / "scene_alignment.json")
        summary["scene_alignment_path"] = str(output_dir / "scene_alignment.json")
        summary["scene_alignment_applied"] = bool(scene_alignment.get("applied", False))
        summary["scene_alignment_matched_views"] = int(scene_alignment.get("matched_view_count", 0))
    return summary


def _load_optional_json_file(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_camera_bundle_payload(path):
    path = Path(path)
    payload = _load_optional_json_file(path)
    if payload is None:
        raise FileNotFoundError(f"Camera bundle not found: {path}")

    image_names = [str(name) for name in payload.get("image_names", [])]
    intrinsics = np.asarray(payload.get("intrinsics", []), dtype=np.float32)

    if "cams2world" in payload:
        cams2world = np.asarray(payload.get("cams2world", []), dtype=np.float32)
    elif "world2cams" in payload:
        world2cams = np.asarray(payload.get("world2cams", []), dtype=np.float32)
        if world2cams.ndim != 3 or world2cams.shape[-2:] != (4, 4):
            raise ValueError(f"world2cams in {path} must have shape [N, 4, 4].")
        cams2world = np.linalg.inv(world2cams)
    else:
        raise ValueError(f"Camera bundle {path} must contain cams2world or world2cams.")

    if cams2world.ndim != 3 or cams2world.shape[-2:] != (4, 4):
        raise ValueError(f"cams2world in {path} must have shape [N, 4, 4].")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsics in {path} must have shape [N, 3, 3].")
    if len(image_names) != cams2world.shape[0] or cams2world.shape[0] != intrinsics.shape[0]:
        raise ValueError(f"image_names, cams2world, and intrinsics must agree in {path}.")

    return {
        "payload": payload,
        "image_names": image_names,
        "cams2world": cams2world.astype(np.float32),
        "intrinsics": intrinsics.astype(np.float32),
    }


def _infer_image_names_from_dense_geometry(dense_output_dir: Path):
    points_dir = dense_output_dir / "points3d"
    confidence_dir = dense_output_dir / "confidence"
    stems = set()
    if points_dir.exists():
        stems.update(path.stem for path in points_dir.glob("*.npy"))
    if confidence_dir.exists():
        stems.update(path.stem for path in confidence_dir.glob("*.npy"))
    return [f"{stem}.png" for stem in sorted(stems)]


def resolve_image_paths(image_dir, image_names):
    if not image_dir:
        return None

    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.exists():
        return None

    by_name = {}
    by_stem = {}
    for candidate in image_dir.iterdir():
        if not candidate.is_file():
            continue
        key_name = candidate.name.lower()
        if key_name not in by_name:
            by_name[key_name] = candidate.resolve()
        key_stem = candidate.stem.lower()
        if key_stem not in by_stem:
            by_stem[key_stem] = candidate.resolve()

    resolved = []
    for image_name in image_names:
        candidate = by_name.get(str(image_name).lower())
        if candidate is None:
            candidate = by_stem.get(Path(str(image_name)).stem.lower())
        resolved.append(None if candidate is None else str(candidate))
    return resolved


def load_dense_geometry_exports_for_backfill(dense_output_dir, image_names=None, min_conf_thr=0.0):
    dense_output_dir = Path(dense_output_dir)
    points_dir = dense_output_dir / "points3d"
    confidence_dir = dense_output_dir / "confidence"
    depth_dir = dense_output_dir / "depth"

    if image_names is None:
        image_names = _infer_image_names_from_dense_geometry(dense_output_dir)
    image_names = [str(name) for name in image_names]
    if not image_names:
        raise ValueError(f"No dense geometry views found in {dense_output_dir}.")

    dense_stats = {}
    dense_points = []
    dense_confidences = []
    dense_image_ids = []

    for image_id, image_name in enumerate(image_names):
        stem = Path(image_name).stem
        points_path = points_dir / f"{stem}.npy"
        confidence_path = confidence_dir / f"{stem}.npy"
        depth_path = depth_dir / f"{stem}.npy"
        if not points_path.exists():
            raise FileNotFoundError(f"Missing dense points for {image_name}: {points_path}")
        if not confidence_path.exists():
            raise FileNotFoundError(f"Missing dense confidence for {image_name}: {confidence_path}")

        points_view = np.asarray(np.load(points_path), dtype=np.float32)
        conf_view = np.asarray(np.load(confidence_path), dtype=np.float32)
        if points_view.ndim != 3 or points_view.shape[2] != 3:
            raise ValueError(f"Dense points for {image_name} must have shape [H, W, 3], got {points_view.shape}.")
        if conf_view.ndim != 2 or conf_view.shape != points_view.shape[:2]:
            raise ValueError(f"Dense confidence for {image_name} must match points shape, got {conf_view.shape}.")

        valid = np.isfinite(points_view).all(axis=2) & np.isfinite(conf_view)
        keep = valid & (conf_view >= float(min_conf_thr))
        if np.any(keep):
            dense_points.append(points_view[keep])
            dense_confidences.append(conf_view[keep].astype(np.float32))
            dense_image_ids.append(np.full((int(keep.sum()),), image_id, dtype=np.int32))

        dense_stats[image_name] = {
            "height": int(conf_view.shape[0]),
            "width": int(conf_view.shape[1]),
            "mean_confidence": float(np.mean(conf_view)),
            "median_confidence": float(np.median(conf_view)),
            "finite_points": int(valid.sum()),
            "points_above_conf_thr": int(keep.sum()),
            "has_depth": bool(depth_path.exists()),
        }

    if dense_points:
        dense_geometry = {
            "points": np.concatenate(dense_points, axis=0).astype(np.float32),
            "confidences": np.concatenate(dense_confidences, axis=0).astype(np.float32),
            "image_ids": np.concatenate(dense_image_ids, axis=0).astype(np.int32),
        }
    else:
        dense_geometry = {
            "points": np.zeros((0, 3), dtype=np.float32),
            "confidences": np.zeros((0,), dtype=np.float32),
            "image_ids": np.zeros((0,), dtype=np.int32),
        }

    return {
        "image_names": image_names,
        "dense_geometry": dense_geometry,
        "dense_stats": dense_stats,
    }


def backfill_foundation_atlas_sidecars(
    atlas_dir,
    camera_bundle_path=None,
    image_dir=None,
    min_conf_thr=None,
    hash_cell_size=None,
    update_summary=True,
):
    atlas_dir = Path(atlas_dir).expanduser().resolve()
    if not atlas_dir.exists():
        raise FileNotFoundError(f"Atlas directory not found: {atlas_dir}")

    atlas_path = atlas_dir / "atlas_nodes.npz"
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas archive not found: {atlas_path}")

    atlas = load_atlas_npz(atlas_path)

    build_config = _load_optional_json_file(atlas_dir / "build_config.json") or {}
    if camera_bundle_path is None:
        camera_bundle_path = atlas_dir / "camera_bundle.json"
    bundle = load_camera_bundle_payload(camera_bundle_path)
    image_names = bundle["image_names"]
    cams2world = bundle["cams2world"]
    intrinsics = bundle["intrinsics"]

    if min_conf_thr is None:
        min_conf_thr = float(build_config.get("min_conf_thr", 0.0))
    else:
        min_conf_thr = float(min_conf_thr)

    if image_dir is None:
        image_dir = build_config.get("image_dir", "")
    image_paths = resolve_image_paths(image_dir, image_names)

    dense_output_dir = atlas_dir / "dense_geometry"
    dense_exports = load_dense_geometry_exports_for_backfill(
        dense_output_dir,
        image_names=image_names,
        min_conf_thr=min_conf_thr,
    )
    dense_stats = dense_exports["dense_stats"]
    dense_geometry = dense_exports["dense_geometry"]
    save_json(dense_stats, dense_output_dir / "dense_views_stats.json")

    sidecar_summary = save_foundation_atlas_sidecars(
        atlas_dir,
        atlas,
        image_names,
        cams2world,
        intrinsics,
        dense_geometry,
        dense_stats=dense_stats,
        image_paths=image_paths,
        min_conf_thr=min_conf_thr,
        hash_cell_size=hash_cell_size,
        preprocess_image_size=build_config.get("image_size", None),
    )

    backfill_summary = {
        "schema_version": 1,
        "atlas_dir": str(atlas_dir),
        "camera_bundle_source": str(Path(camera_bundle_path).expanduser().resolve()),
        "image_dir": "" if not image_dir else str(Path(image_dir).expanduser().resolve()),
        "min_conf_thr": float(min_conf_thr),
        "num_views": int(len(image_names)),
        "num_atlas_nodes": int(atlas.positions.shape[0]),
        "num_dense_points_above_conf_thr": int(dense_geometry["points"].shape[0]),
        "outputs": sidecar_summary,
    }
    save_json(backfill_summary, atlas_dir / "atlas_backfill_summary.json")

    if update_summary:
        atlas_summary_path = atlas_dir / "atlas_summary.json"
        atlas_summary = _load_optional_json_file(atlas_summary_path) or {}
        atlas_summary["export_sidecars"] = sidecar_summary
        atlas_summary["backfill_sidecars"] = {
            "schema_version": 1,
            "min_conf_thr": float(min_conf_thr),
            "num_views": int(len(image_names)),
            "num_dense_points_above_conf_thr": int(dense_geometry["points"].shape[0]),
        }
        save_json(atlas_summary, atlas_summary_path)

    return backfill_summary


def save_ply(path, points, colors):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(points, dtype=np.float32)
    rgb = _to_uint8_colors(colors)
    normals = np.zeros_like(xyz, dtype=np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    vertex_element = PlyElement.describe(elements, "vertex")
    PlyData([vertex_element]).write(str(path))


def atlas_debug_colors(atlas: FoundationAtlas) -> np.ndarray:
    base_colors = np.zeros((atlas.positions.shape[0], 3), dtype=np.float32)
    base_colors[atlas.atlas_class == ATLAS_CLASS_SURFACE] = np.array([60, 180, 75], dtype=np.float32)
    base_colors[atlas.atlas_class == ATLAS_CLASS_EDGE] = np.array([245, 130, 48], dtype=np.float32)
    base_colors[atlas.atlas_class == ATLAS_CLASS_UNSTABLE] = np.array([230, 25, 75], dtype=np.float32)

    brightness = 0.35 + 0.65 * atlas.reliability[:, None]
    return np.clip(base_colors * brightness, 0, 255).astype(np.uint8)


def summarize_foundation_atlas(atlas: FoundationAtlas, num_input_points: int, min_conf_thr: float):
    structure_score = _atlas_metric_or_default(atlas, "structure_score", 0.0)
    scale_consistency = _atlas_metric_or_default(atlas, "scale_consistency", 1.0)
    class_consistency = _atlas_metric_or_default(atlas, "class_consistency", 1.0)
    support_consistency = _atlas_metric_or_default(atlas, "support_consistency", 1.0)
    view_balance = _atlas_metric_or_default(atlas, "view_balance", 1.0)
    view_outlier_score = _atlas_metric_or_default(atlas, "view_outlier_score", 1.0)
    robust_inlier_ratio = _atlas_metric_or_default(atlas, "robust_inlier_ratio", 1.0)
    unstable_reason_code, _, _ = _compute_unstable_reason_payload(
        atlas,
        structure_score=structure_score,
        support_consistency=support_consistency,
        scale_consistency=scale_consistency,
        class_consistency=class_consistency,
    )
    class_counts = {
        "surface": int(np.sum(atlas.atlas_class == ATLAS_CLASS_SURFACE)),
        "edge": int(np.sum(atlas.atlas_class == ATLAS_CLASS_EDGE)),
        "unstable": int(np.sum(atlas.atlas_class == ATLAS_CLASS_UNSTABLE)),
    }
    reliability_floor = float(np.min(atlas.reliability)) if atlas.reliability.size else 0.0
    unstable_reason_counts = {name: 0 for name in UNSTABLE_REASON_NAMES.values()}
    unstable_mask = atlas.atlas_class == ATLAS_CLASS_UNSTABLE
    for reason_code, reason_name in UNSTABLE_REASON_NAMES.items():
        unstable_reason_counts[reason_name] = int(np.sum(unstable_mask & (unstable_reason_code == reason_code)))
    class_stats = {}
    for class_id, class_name in (
        (ATLAS_CLASS_SURFACE, "surface"),
        (ATLAS_CLASS_EDGE, "edge"),
        (ATLAS_CLASS_UNSTABLE, "unstable"),
    ):
        mask = atlas.atlas_class == class_id
        if not np.any(mask):
            continue
        class_stats[class_name] = {
            "count": int(mask.sum()),
            "mean_reliability": float(np.mean(atlas.reliability[mask])),
            "mean_residual": float(np.mean(atlas.calibration_residual[mask])),
            "mean_point_support": float(np.mean(atlas.point_support[mask])),
            "mean_view_support": float(np.mean(atlas.view_support[mask])),
            "mean_structure_score": float(np.mean(structure_score[mask])),
            "mean_scale_consistency": float(np.mean(scale_consistency[mask])),
            "mean_class_consistency": float(np.mean(class_consistency[mask])),
            "mean_support_consistency": float(np.mean(support_consistency[mask])),
            "mean_view_balance": float(np.mean(view_balance[mask])),
            "mean_view_outlier_score": float(np.mean(view_outlier_score[mask])),
            "mean_robust_inlier_ratio": float(np.mean(robust_inlier_ratio[mask])),
        }
    return {
        "num_input_points": int(num_input_points),
        "min_conf_thr": float(min_conf_thr),
        "num_atlas_nodes": int(atlas.positions.shape[0]),
        "mean_reliability": float(np.mean(atlas.reliability)),
        "median_reliability": float(np.median(atlas.reliability)),
        "mean_raw_score": float(np.mean(atlas.raw_score)),
        "mean_radius": float(np.mean(atlas.radius)),
        "median_radius": float(np.median(atlas.radius)),
        "mean_calibration_residual": float(np.mean(atlas.calibration_residual)),
        "median_calibration_residual": float(np.median(atlas.calibration_residual)),
        "mean_point_support": float(np.mean(atlas.point_support)),
        "median_point_support": float(np.median(atlas.point_support)),
        "mean_view_support": float(np.mean(atlas.view_support)),
        "median_view_support": float(np.median(atlas.view_support)),
        "mean_view_coverage": float(np.mean(atlas.view_coverage)),
        "mean_support_score": float(np.mean(atlas.support_score)),
        "median_support_score": float(np.median(atlas.support_score)),
        "mean_linearness": float(np.mean(atlas.linearness)),
        "mean_planarness": float(np.mean(atlas.planarness)),
        "mean_scattering": float(np.mean(atlas.scattering)),
        "mean_structure_score": float(np.mean(structure_score)),
        "mean_scale_consistency": float(np.mean(scale_consistency)),
        "mean_class_consistency": float(np.mean(class_consistency)),
        "mean_support_consistency": float(np.mean(support_consistency)),
        "mean_view_balance": float(np.mean(view_balance)),
        "mean_view_outlier_score": float(np.mean(view_outlier_score)),
        "mean_robust_inlier_ratio": float(np.mean(robust_inlier_ratio)),
        "reliability_floor": reliability_floor,
        "reliability_floor_ratio": float(np.mean(np.isclose(atlas.reliability, reliability_floor, atol=1e-6))),
        "class_counts": class_counts,
        "class_stats": class_stats,
        "unstable_reason_counts": unstable_reason_counts,
    }


def save_dense_geometry_exports(output_dir, image_names, pts3d, depthmaps, confs, rgb_images, preview_max_points=200000, seed=42, min_conf_thr=None):
    output_dir = Path(output_dir)
    points_dir = output_dir / "points3d"
    depth_dir = output_dir / "depth"
    confidence_dir = output_dir / "confidence"
    preview_dir = output_dir / "preview"
    points_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    preview_points = []
    preview_colors = []
    dense_stats = {}

    for image_name, points_view, depth_view, conf_view, rgb_view in zip(image_names, pts3d, depthmaps, confs, rgb_images):
        stem = Path(image_name).stem
        conf_array = np.asarray(conf_view, dtype=np.float32)
        height, width = conf_array.shape
        depth_array = np.asarray(depth_view, dtype=np.float32).reshape(height, width)
        points_array = np.asarray(points_view, dtype=np.float32).reshape(height, width, 3)
        rgb_array = np.asarray(rgb_view, dtype=np.float32)

        np.save(points_dir / f"{stem}.npy", points_array)
        np.save(depth_dir / f"{stem}.npy", depth_array)
        np.save(confidence_dir / f"{stem}.npy", conf_array)
        _save_preview_png(depth_array, preview_dir / f"{stem}_depth.png", invert=True)
        _save_preview_png(conf_array, preview_dir / f"{stem}_confidence.png", invert=False)

        valid = np.isfinite(points_array).all(axis=2) & np.isfinite(conf_array)
        if np.any(valid):
            flattened_points = points_array[valid]
            flattened_colors = rgb_array[valid]
            if flattened_points.shape[0] > preview_max_points // max(len(image_names), 1):
                sample_size = preview_max_points // max(len(image_names), 1)
                if sample_size > 0:
                    indices = rng.choice(flattened_points.shape[0], size=sample_size, replace=False)
                    flattened_points = flattened_points[indices]
                    flattened_colors = flattened_colors[indices]
            preview_points.append(flattened_points)
            preview_colors.append(flattened_colors)

        dense_stats[image_name] = {
            "height": int(height),
            "width": int(width),
            "mean_confidence": float(np.mean(conf_array)),
            "median_confidence": float(np.median(conf_array)),
            "finite_points": int(valid.sum()),
            "points_above_conf_thr": int(((conf_array >= float(min_conf_thr)) & valid).sum()) if min_conf_thr is not None else int(valid.sum()),
        }

    if preview_points:
        save_ply(
            output_dir / "dense_points_preview.ply",
            np.concatenate(preview_points, axis=0),
            np.concatenate(preview_colors, axis=0),
        )

    save_json(dense_stats, output_dir / "dense_views_stats.json")
    return dense_stats


def _save_preview_png(array, path, invert=False):
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array)
    preview = np.zeros_like(array, dtype=np.float32)
    if np.any(valid):
        low = float(np.quantile(array[valid], 0.01))
        high = float(np.quantile(array[valid], 0.99))
        if high > low:
            preview = (array - low) / (high - low)
        preview = np.clip(preview, 0.0, 1.0)
    if invert:
        preview = 1.0 - preview
    image = Image.fromarray(np.clip(preview * 255.0, 0, 255).astype(np.uint8))
    image.save(path)


def plot_foundation_atlas_report(atlas: FoundationAtlas, output_dir, summary=None, title="Foundation Geometry Atlas"):
    _require_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    axes[0].hist(atlas.reliability, bins=30, color="#2e86de")
    axes[0].set_title("Reliability")
    axes[1].hist(atlas.radius, bins=30, color="#20bf6b")
    axes[1].set_title("Radius")
    axes[2].hist(atlas.calibration_residual, bins=30, color="#eb3b5a")
    axes[2].set_title("Calibration Residual")
    axes[3].hist(atlas.view_support, bins=min(20, max(5, int(np.max(atlas.view_support)) + 1)), color="#6c5ce7")
    axes[3].set_title("View Support")
    axes[4].hist(np.log10(1.0 + atlas.point_support), bins=30, color="#ff9f43")
    axes[4].set_title("log10(1 + Point Support)")

    counts = [
        int(np.sum(atlas.atlas_class == ATLAS_CLASS_SURFACE)),
        int(np.sum(atlas.atlas_class == ATLAS_CLASS_EDGE)),
        int(np.sum(atlas.atlas_class == ATLAS_CLASS_UNSTABLE)),
    ]
    axes[5].bar(["surface", "edge", "unstable"], counts, color=["#3cb371", "#f4a261", "#d1495b"])
    axes[5].set_title("Class Counts")

    if summary is not None:
        fig.suptitle(
            f"{title}\nNodes={summary['num_atlas_nodes']} | Mean reliability={summary['mean_reliability']:.3f} | Floor ratio={summary.get('reliability_floor_ratio', 0.0):.3f}",
            y=0.98,
        )
    else:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()
    fig.savefig(output_dir / "atlas_metrics.png", dpi=180)
    plt.close(fig)

    if atlas.positions.shape[0] < 3:
        return

    centered = atlas.positions - atlas.positions.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vh[:2].T

    class_colors = np.array(["#3cb371", "#f4a261", "#d1495b"], dtype=object)
    fig, axis = plt.subplots(figsize=(8, 7))
    for class_id, label in (
        (ATLAS_CLASS_SURFACE, "surface"),
        (ATLAS_CLASS_EDGE, "edge"),
        (ATLAS_CLASS_UNSTABLE, "unstable"),
    ):
        mask = atlas.atlas_class == class_id
        if not np.any(mask):
            continue
        axis.scatter(projection[mask, 0], projection[mask, 1], s=10, alpha=0.75, label=label, color=class_colors[class_id])
    axis.set_title("Atlas PCA Projection by Class")
    axis.legend()
    axis.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "atlas_pca_class.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 7))
    scatter = axis.scatter(
        projection[:, 0],
        projection[:, 1],
        c=atlas.reliability,
        s=10,
        alpha=0.8,
        cmap="viridis",
    )
    axis.set_title("Atlas PCA Projection by Reliability")
    axis.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=axis, label="Reliability")
    fig.tight_layout()
    fig.savefig(output_dir / "atlas_pca_reliability.png", dpi=180)
    plt.close(fig)
