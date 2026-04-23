from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from plyfile import PlyData

from utils.graphics_utils import BasicPointCloud


ATLAS_CLASS_SURFACE = 0
ATLAS_CLASS_EDGE = 1
ATLAS_CLASS_UNSTABLE = 2

GAUSSIAN_STATE_STABLE = 0
GAUSSIAN_STATE_UNSTABLE_PASSIVE = 1
GAUSSIAN_STATE_UNSTABLE_ACTIVE = 2
GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING = 3

FOUNDATION_POSE_MAX_CORRESPONDENCES = 8192

FOUNDATION_PREPROCESSED_IMAGE_COORDINATE_SPACES = {
    "mast3r_preprocessed_image",
    "vggt_preprocessed_image",
    "foundation_preprocessed_image",
}


ATLAS_CLASS_NAMES = {
    ATLAS_CLASS_SURFACE: "surface",
    ATLAS_CLASS_EDGE: "edge",
    ATLAS_CLASS_UNSTABLE: "unstable",
}

GAUSSIAN_STATE_NAMES = {
    GAUSSIAN_STATE_STABLE: "stable",
    GAUSSIAN_STATE_UNSTABLE_PASSIVE: "unstable_passive",
    GAUSSIAN_STATE_UNSTABLE_ACTIVE: "unstable_active",
    GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING: "out_of_anchor_pending",
}

ATLAS_CLASS_COLORS = {
    ATLAS_CLASS_SURFACE: np.array([0.38, 0.72, 0.41], dtype=np.float32),
    ATLAS_CLASS_EDGE: np.array([0.96, 0.56, 0.19], dtype=np.float32),
    ATLAS_CLASS_UNSTABLE: np.array([0.82, 0.24, 0.35], dtype=np.float32),
}


@dataclass
class FoundationCameraBundle:
    image_names: list[str]
    cams2world: np.ndarray
    intrinsics: np.ndarray


@dataclass
class FoundationDenseViewEvidence:
    image_name: str
    width: int
    height: int
    mean_confidence: float
    median_confidence: float
    finite_points: int
    points_above_conf_thr: int
    points_path: str | None
    confidence_path: str | None
    depth_path: str | None
    coordinate_space: str | None = None
    source_width: int = 0
    source_height: int = 0
    resized_width: int = 0
    resized_height: int = 0
    crop_left: float = 0.0
    crop_top: float = 0.0
    crop_right: float = 0.0
    crop_bottom: float = 0.0
    scale_x: float = 0.0
    scale_y: float = 0.0
    audit_sampled_corr: int = 0
    audit_projected_corr: int = 0
    audit_in_frame_corr: int = 0
    audit_mean_px_error: float = 0.0
    audit_median_px_error: float = 0.0
    audit_p90_px_error: float = 0.0
    sparse_path: str | None = None
    sparse_correspondence_count: int = 0
    correspondence_mode: str | None = None


@dataclass
class FoundationCorrespondenceManifest:
    min_conf_thr: float
    points_root: str
    confidence_root: str
    depth_root: str
    sparse_root: str = ""
    views: dict[str, FoundationDenseViewEvidence] = field(default_factory=dict)


@dataclass
class FoundationAtlasInit:
    source_path: str
    positions: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    support: np.ndarray
    basis: np.ndarray
    raw_score: np.ndarray
    radius: np.ndarray
    reliability: np.ndarray
    calibration_residual: np.ndarray
    atlas_class: np.ndarray
    anisotropy_ref: np.ndarray
    neighbor_indices: np.ndarray
    node_confidence: np.ndarray
    point_support: np.ndarray
    view_support: np.ndarray
    view_coverage: np.ndarray
    support_score: np.ndarray
    linearness: np.ndarray
    planarness: np.ndarray
    scattering: np.ndarray
    atlas_ids: np.ndarray
    gaussian_state: np.ndarray
    init_scales: np.ndarray
    init_rotations: np.ndarray
    reference_camera_ids: np.ndarray
    reference_camera_scores: np.ndarray
    reference_view_names: list[str]
    reference_view_weights: np.ndarray
    reference_view_counts: np.ndarray
    reference_camera_source: str
    hash_info: dict
    build_config: dict | None
    camera_bundle: FoundationCameraBundle | None
    correspondence_manifest: FoundationCorrespondenceManifest | None


def _load_ply_points(path: Path):
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    xyz = np.stack(
        [
            np.asarray(vertices["x"], dtype=np.float32),
            np.asarray(vertices["y"], dtype=np.float32),
            np.asarray(vertices["z"], dtype=np.float32),
        ],
        axis=1,
    )
    has_rgb = all(channel in vertices.data.dtype.names for channel in ("red", "green", "blue"))
    if has_rgb:
        rgb = np.stack(
            [
                np.asarray(vertices["red"], dtype=np.float32),
                np.asarray(vertices["green"], dtype=np.float32),
                np.asarray(vertices["blue"], dtype=np.float32),
            ],
            axis=1,
        )
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = None
    return xyz.astype(np.float32), None if rgb is None else rgb.astype(np.float32)


def _load_preview_cloud(run_root: Path, color_sample_limit: int, seed: int):
    candidates = [
        run_root / "dense_geometry" / "dense_points_preview.ply",
        run_root / "atlas_nodes_debug.ply",
    ]
    for path in candidates:
        if not path.exists():
            continue
        points, colors = _load_ply_points(path)
        if colors is None or points.shape[0] == 0:
            continue
        if color_sample_limit > 0 and points.shape[0] > color_sample_limit:
            rng = np.random.default_rng(seed)
            keep = rng.choice(points.shape[0], size=color_sample_limit, replace=False)
            keep.sort()
            points = points[keep]
            colors = colors[keep]
        return points, colors
    return None, None


def _class_colors(atlas_class: np.ndarray):
    colors = np.zeros((atlas_class.shape[0], 3), dtype=np.float32)
    for class_id, class_color in ATLAS_CLASS_COLORS.items():
        colors[atlas_class == class_id] = class_color
    return colors


def _nearest_colors(query_points: np.ndarray, ref_points: np.ndarray, ref_colors: np.ndarray, chunk_size: int = 128):
    colors = np.zeros((query_points.shape[0], 3), dtype=np.float32)
    for start in range(0, query_points.shape[0], chunk_size):
        end = min(start + chunk_size, query_points.shape[0])
        chunk = query_points[start:end]
        d2 = np.sum((chunk[:, None, :] - ref_points[None, :, :]) ** 2, axis=2)
        indices = np.argmin(d2, axis=1)
        colors[start:end] = ref_colors[indices]
    return colors


def _estimate_atlas_colors(
    run_root: Path,
    atlas_positions: np.ndarray,
    atlas_class: np.ndarray,
    fallback_point_cloud: BasicPointCloud | None,
    color_sample_limit: int,
    seed: int,
):
    ref_points, ref_colors = _load_preview_cloud(run_root, color_sample_limit, seed)
    if ref_points is None and fallback_point_cloud is not None:
        ref_points = np.asarray(fallback_point_cloud.points, dtype=np.float32)
        ref_colors = np.asarray(fallback_point_cloud.colors, dtype=np.float32)

    if ref_points is None or ref_colors is None or ref_points.shape[0] == 0:
        return _class_colors(atlas_class)

    return np.clip(_nearest_colors(atlas_positions, ref_points, ref_colors), 0.0, 1.0).astype(np.float32)


def _make_right_handed(basis: np.ndarray):
    basis = np.asarray(basis, dtype=np.float32).copy()
    det = np.linalg.det(basis)
    flip = det < 0.0
    if np.any(flip):
        basis[flip, :, 2] *= -1.0
    return basis


def _rotation_matrices_to_quaternions(rotation_mats: np.ndarray):
    rotation_mats = _make_right_handed(rotation_mats)
    quaternions = np.zeros((rotation_mats.shape[0], 4), dtype=np.float32)

    trace = rotation_mats[:, 0, 0] + rotation_mats[:, 1, 1] + rotation_mats[:, 2, 2]
    positive = trace > 0.0

    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        quaternions[positive, 0] = 0.25 * s
        quaternions[positive, 1] = (rotation_mats[positive, 2, 1] - rotation_mats[positive, 1, 2]) / s
        quaternions[positive, 2] = (rotation_mats[positive, 0, 2] - rotation_mats[positive, 2, 0]) / s
        quaternions[positive, 3] = (rotation_mats[positive, 1, 0] - rotation_mats[positive, 0, 1]) / s

    remain = ~positive
    if np.any(remain):
        local = rotation_mats[remain]
        diag = np.stack([local[:, 0, 0], local[:, 1, 1], local[:, 2, 2]], axis=1)
        argmax = np.argmax(diag, axis=1)

        x_mask = argmax == 0
        if np.any(x_mask):
            mats = local[x_mask]
            s = np.sqrt(np.maximum(1.0 + mats[:, 0, 0] - mats[:, 1, 1] - mats[:, 2, 2], 1e-8)) * 2.0
            target = np.flatnonzero(remain)[x_mask]
            quaternions[target, 0] = (mats[:, 2, 1] - mats[:, 1, 2]) / s
            quaternions[target, 1] = 0.25 * s
            quaternions[target, 2] = (mats[:, 0, 1] + mats[:, 1, 0]) / s
            quaternions[target, 3] = (mats[:, 0, 2] + mats[:, 2, 0]) / s

        y_mask = argmax == 1
        if np.any(y_mask):
            mats = local[y_mask]
            s = np.sqrt(np.maximum(1.0 + mats[:, 1, 1] - mats[:, 0, 0] - mats[:, 2, 2], 1e-8)) * 2.0
            target = np.flatnonzero(remain)[y_mask]
            quaternions[target, 0] = (mats[:, 0, 2] - mats[:, 2, 0]) / s
            quaternions[target, 1] = (mats[:, 0, 1] + mats[:, 1, 0]) / s
            quaternions[target, 2] = 0.25 * s
            quaternions[target, 3] = (mats[:, 1, 2] + mats[:, 2, 1]) / s

        z_mask = argmax == 2
        if np.any(z_mask):
            mats = local[z_mask]
            s = np.sqrt(np.maximum(1.0 + mats[:, 2, 2] - mats[:, 0, 0] - mats[:, 1, 1], 1e-8)) * 2.0
            target = np.flatnonzero(remain)[z_mask]
            quaternions[target, 0] = (mats[:, 1, 0] - mats[:, 0, 1]) / s
            quaternions[target, 1] = (mats[:, 0, 2] + mats[:, 2, 0]) / s
            quaternions[target, 2] = (mats[:, 1, 2] + mats[:, 2, 1]) / s
            quaternions[target, 3] = 0.25 * s

    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return (quaternions / np.clip(norms, 1e-8, None)).astype(np.float32)


def _initial_gaussian_states(atlas_class: np.ndarray, reliability: np.ndarray, surface_stable_min: float, edge_stable_min: float):
    stable_surface = (atlas_class == ATLAS_CLASS_SURFACE) & (reliability >= surface_stable_min)
    stable_edge = (atlas_class == ATLAS_CLASS_EDGE) & (reliability >= edge_stable_min)
    stable = stable_surface | stable_edge

    states = np.full((atlas_class.shape[0],), GAUSSIAN_STATE_UNSTABLE_PASSIVE, dtype=np.int64)
    states[stable] = GAUSSIAN_STATE_STABLE
    return states


def _initial_scales(
    radius: np.ndarray,
    atlas_class: np.ndarray,
    scale_multiplier: float,
    surface_thickness_ratio: float,
    edge_thickness_ratio: float,
    unstable_scale_ratio: float,
):
    radius = np.asarray(radius, dtype=np.float32).reshape(-1, 1)
    scales = np.repeat(radius, 3, axis=1)

    surface = atlas_class == ATLAS_CLASS_SURFACE
    edge = atlas_class == ATLAS_CLASS_EDGE
    unstable = atlas_class == ATLAS_CLASS_UNSTABLE

    scales[surface, 0] = radius[surface, 0]
    scales[surface, 1] = radius[surface, 0]
    scales[surface, 2] = radius[surface, 0] * surface_thickness_ratio

    scales[edge, 0] = radius[edge, 0]
    scales[edge, 1] = radius[edge, 0] * edge_thickness_ratio
    scales[edge, 2] = radius[edge, 0] * edge_thickness_ratio * 0.55

    scales[unstable, :] = radius[unstable, 0:1] * unstable_scale_ratio

    scales = np.clip(scales * scale_multiplier, 1e-4, None)
    return scales.astype(np.float32)


def _load_optional_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_foundation_atlas_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "atlas_nodes.npz").exists()
        and (path / "camera_bundle.json").exists()
    )


def resolve_foundation_atlas_root(atlas_path: str | Path, strict: bool = True) -> Path:
    candidate = Path(atlas_path).expanduser().resolve()
    if _is_foundation_atlas_root(candidate):
        return candidate

    search_parent = candidate.parent if candidate.parent.exists() else None
    if search_parent is not None:
        if _is_foundation_atlas_root(search_parent):
            return search_parent
        sibling_roots = [
            child.resolve()
            for child in search_parent.iterdir()
            if _is_foundation_atlas_root(child)
        ]
        if len(sibling_roots) == 1:
            return sibling_roots[0]
        if len(sibling_roots) > 1 and not strict:
            return candidate

    if not strict:
        return candidate

    message = (
        f"Expected atlas_path to point to a Foundation Atlas artifact directory containing "
        f"'atlas_nodes.npz' and 'camera_bundle.json', but got: {candidate}"
    )
    if search_parent is not None:
        sibling_roots = [
            str(child.resolve())
            for child in search_parent.iterdir()
            if _is_foundation_atlas_root(child)
        ]
        if len(sibling_roots) == 1:
            message += f". Nearby atlas root found at: {sibling_roots[0]}"
        elif len(sibling_roots) > 1:
            message += f". Nearby atlas roots found: {', '.join(sibling_roots)}"
    raise ValueError(message)


def _load_scene_alignment_contract(run_root: Path):
    payload = _load_optional_json(run_root / "scene_alignment.json")
    if not payload:
        return None

    contract = payload.get("contract_validation")
    if contract is None:
        return payload

    if bool(contract.get("passed", False)):
        return payload

    failures = [str(item) for item in contract.get("failures", []) if str(item)]
    failure_text = "; ".join(failures) if failures else "unknown contract failure"
    raise ValueError(
        f"Foundation atlas contract validation failed for {run_root}: {failure_text}"
    )


def _load_scene_alignment_audit_views(run_root: Path):
    payload = _load_optional_json(run_root / "scene_alignment.json")
    if not payload:
        return {}
    dense_audit = payload.get("dense_correspondence_audit")
    if not isinstance(dense_audit, dict):
        return {}
    view_payload = dense_audit.get("views")
    return view_payload if isinstance(view_payload, dict) else {}


def _resolve_optional_path(run_root: Path, value):
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = run_root / path
    return path.expanduser().resolve()


def _normalize_scores(values: np.ndarray):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    low = float(values.min())
    high = float(values.max())
    if high <= low + 1e-6:
        return np.ones_like(values, dtype=np.float32)
    return np.clip((values - low) / float(high - low), 0.0, 1.0).astype(np.float32)


def _load_archive_vector(archive, key: str, length: int, dtype, default_value):
    if key in archive:
        return np.asarray(archive[key], dtype=dtype).reshape(-1)
    if np.issubdtype(np.dtype(dtype), np.integer):
        return np.full((length,), int(default_value), dtype=dtype)
    return np.full((length,), float(default_value), dtype=dtype)


def _load_camera_bundle(run_root: Path):
    payload = _load_optional_json(run_root / "camera_bundle.json")
    if not payload:
        return None
    image_names = [str(name) for name in payload.get("image_names", [])]
    cams2world = np.asarray(payload.get("cams2world", []), dtype=np.float32)
    intrinsics = np.asarray(payload.get("intrinsics", []), dtype=np.float32)
    if cams2world.ndim != 3 or cams2world.shape[-2:] != (4, 4):
        return None
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        return None
    if len(image_names) != cams2world.shape[0] or cams2world.shape[0] != intrinsics.shape[0]:
        return None
    return FoundationCameraBundle(
        image_names=image_names,
        cams2world=cams2world,
        intrinsics=intrinsics,
    )


def _load_correspondence_manifest(run_root: Path, camera_bundle: FoundationCameraBundle | None):
    dense_root = run_root / "dense_geometry"
    if not dense_root.exists():
        return None
    audit_views = _load_scene_alignment_audit_views(run_root)

    explicit_manifest = _load_optional_json(run_root / "correspondence_manifest.json")
    if explicit_manifest is None:
        explicit_manifest = _load_optional_json(dense_root / "correspondence_manifest.json")
    if explicit_manifest:
        min_conf_thr = float(explicit_manifest.get("min_conf_thr", 0.0))
        points_root = _resolve_optional_path(run_root, explicit_manifest.get("points_root"))
        confidence_root = _resolve_optional_path(run_root, explicit_manifest.get("confidence_root"))
        depth_root = _resolve_optional_path(run_root, explicit_manifest.get("depth_root"))
        sparse_root = _resolve_optional_path(run_root, explicit_manifest.get("sparse_root"))
        views = {}
        for image_name, view_payload in (explicit_manifest.get("views", {}) or {}).items():
            image_name = str(image_name)
            audit_payload = audit_views.get(image_name, {}) or {}
            points_path = _resolve_optional_path(run_root, view_payload.get("points_path"))
            confidence_path = _resolve_optional_path(run_root, view_payload.get("confidence_path"))
            depth_path = _resolve_optional_path(run_root, view_payload.get("depth_path"))
            sparse_path = _resolve_optional_path(run_root, view_payload.get("sparse_path"))
            views[image_name] = FoundationDenseViewEvidence(
                image_name=image_name,
                width=int(view_payload.get("width", 0)),
                height=int(view_payload.get("height", 0)),
                mean_confidence=float(view_payload.get("mean_confidence", 0.0)),
                median_confidence=float(view_payload.get("median_confidence", 0.0)),
                finite_points=int(view_payload.get("finite_points", 0)),
                points_above_conf_thr=int(view_payload.get("points_above_conf_thr", 0)),
                points_path=None if points_path is None or not points_path.exists() else str(points_path),
                confidence_path=None if confidence_path is None or not confidence_path.exists() else str(confidence_path),
                depth_path=None if depth_path is None or not depth_path.exists() else str(depth_path),
                coordinate_space=view_payload.get("coordinate_space"),
                source_width=int(view_payload.get("source_width", 0)),
                source_height=int(view_payload.get("source_height", 0)),
                resized_width=int(view_payload.get("resized_width", 0)),
                resized_height=int(view_payload.get("resized_height", 0)),
                crop_left=float(view_payload.get("crop_left", 0.0)),
                crop_top=float(view_payload.get("crop_top", 0.0)),
                crop_right=float(view_payload.get("crop_right", 0.0)),
                crop_bottom=float(view_payload.get("crop_bottom", 0.0)),
                scale_x=float(view_payload.get("scale_x", 0.0)),
                scale_y=float(view_payload.get("scale_y", 0.0)),
                audit_sampled_corr=int(audit_payload.get("sampled_corr", 0)),
                audit_projected_corr=int(audit_payload.get("projected_corr", 0)),
                audit_in_frame_corr=int(audit_payload.get("in_frame_corr", 0)),
                audit_mean_px_error=float(audit_payload.get("mean_px_error", 0.0) or 0.0),
                audit_median_px_error=float(audit_payload.get("median_px_error", 0.0) or 0.0),
                audit_p90_px_error=float(audit_payload.get("p90_px_error", 0.0) or 0.0),
                sparse_path=None if sparse_path is None or not sparse_path.exists() else str(sparse_path),
                sparse_correspondence_count=int(view_payload.get("sparse_correspondence_count", 0)),
                correspondence_mode=view_payload.get("correspondence_mode"),
            )
        if views:
            return FoundationCorrespondenceManifest(
                min_conf_thr=min_conf_thr,
                points_root="" if points_root is None else str(points_root),
                confidence_root="" if confidence_root is None else str(confidence_root),
                depth_root="" if depth_root is None else str(depth_root),
                sparse_root="" if sparse_root is None else str(sparse_root),
                views=views,
            )

    points_root = dense_root / "points3d"
    confidence_root = dense_root / "confidence"
    depth_root = dense_root / "depth"
    dense_stats = _load_optional_json(dense_root / "dense_views_stats.json") or {}
    build_config = _load_optional_json(run_root / "build_config.json") or {}
    min_conf_thr = float(build_config.get("min_conf_thr", 0.0))

    view_names = []
    if camera_bundle is not None:
        view_names.extend(camera_bundle.image_names)
    for image_name in dense_stats.keys():
        if image_name not in view_names:
            view_names.append(str(image_name))
    if not view_names:
        for points_path in sorted(points_root.glob("*.npy")):
            view_names.append(f"{points_path.stem}.png")

    views = {}
    for image_name in view_names:
        stem = Path(image_name).stem
        audit_payload = audit_views.get(str(image_name), {}) or {}
        points_path = points_root / f"{stem}.npy"
        confidence_path = confidence_root / f"{stem}.npy"
        depth_path = depth_root / f"{stem}.npy"
        stats = dense_stats.get(image_name, {})
        if not stats and not points_path.exists() and not confidence_path.exists() and not depth_path.exists():
            continue
        views[image_name] = FoundationDenseViewEvidence(
            image_name=image_name,
            width=int(stats.get("width", 0)),
            height=int(stats.get("height", 0)),
            mean_confidence=float(stats.get("mean_confidence", 0.0)),
            median_confidence=float(stats.get("median_confidence", 0.0)),
            finite_points=int(stats.get("finite_points", 0)),
            points_above_conf_thr=int(stats.get("points_above_conf_thr", 0)),
            points_path=str(points_path) if points_path.exists() else None,
            confidence_path=str(confidence_path) if confidence_path.exists() else None,
            depth_path=str(depth_path) if depth_path.exists() else None,
            audit_sampled_corr=int(audit_payload.get("sampled_corr", 0)),
            audit_projected_corr=int(audit_payload.get("projected_corr", 0)),
            audit_in_frame_corr=int(audit_payload.get("in_frame_corr", 0)),
            audit_mean_px_error=float(audit_payload.get("mean_px_error", 0.0) or 0.0),
            audit_median_px_error=float(audit_payload.get("median_px_error", 0.0) or 0.0),
            audit_p90_px_error=float(audit_payload.get("p90_px_error", 0.0) or 0.0),
        )

    if not views:
        return None

    return FoundationCorrespondenceManifest(
        min_conf_thr=min_conf_thr,
        points_root=str(points_root),
        confidence_root=str(confidence_root),
        depth_root=str(depth_root),
        sparse_root="",
        views=views,
    )


def _load_hash_info(run_root: Path, archive):
    sidecar = _load_optional_json(run_root / "atlas_hash.json")
    if sidecar:
        hash_info = dict(sidecar)
        hash_info["source"] = "atlas_hash.json"
        return hash_info

    archive_keys = list(archive.keys()) if isinstance(archive, dict) else list(archive.files)
    hash_keys = [key for key in archive_keys if "hash" in key.lower()]
    if hash_keys:
        return {
            "source": "archive",
            "keys": hash_keys,
            "shapes": {key: list(np.asarray(archive[key]).shape) for key in hash_keys},
        }

    return {
        "source": "neighbor_indices_only",
        "available": False,
        "keys": ["neighbor_indices"],
    }


def _load_reference_camera_evidence(run_root: Path, archive, camera_bundle: FoundationCameraBundle | None, positions: np.ndarray):
    node_count = int(positions.shape[0])
    default_view_names = list(camera_bundle.image_names) if camera_bundle is not None else []

    def _empty_view_evidence():
        return (
            list(default_view_names),
            np.zeros((node_count, 0), dtype=np.float32),
            np.zeros((node_count, 0), dtype=np.int32),
        )

    def _sanitize_view_evidence(view_names, view_weights, view_counts):
        if view_names is None:
            names = list(default_view_names)
        else:
            names = [str(name) for name in list(view_names)]
        weights = np.asarray(view_weights, dtype=np.float32) if view_weights is not None else np.zeros((node_count, 0), dtype=np.float32)
        counts = np.asarray(view_counts, dtype=np.int32) if view_counts is not None else np.zeros((node_count, 0), dtype=np.int32)
        if weights.ndim == 1:
            weights = weights.reshape(node_count, -1)
        if counts.ndim == 1:
            counts = counts.reshape(node_count, -1)
        if weights.ndim != 2 or counts.ndim != 2:
            return _empty_view_evidence()
        if weights.shape[0] != node_count or counts.shape[0] != node_count:
            return _empty_view_evidence()
        if weights.shape[1] != counts.shape[1]:
            return _empty_view_evidence()
        if names and len(names) != weights.shape[1]:
            return _empty_view_evidence()
        if not names and weights.shape[1] == len(default_view_names):
            names = list(default_view_names)
        return names, weights.astype(np.float32), counts.astype(np.int32)

    if "reference_camera_ids" in archive or "reference_camera_scores" in archive:
        ids = _load_archive_vector(archive, "reference_camera_ids", node_count, np.int64, -1)
        scores = _load_archive_vector(archive, "reference_camera_scores", node_count, np.float32, 0.0)
        view_names, view_weights, view_counts = _sanitize_view_evidence(
            archive["image_names"] if "image_names" in archive else default_view_names,
            archive["view_weights"] if "view_weights" in archive else None,
            archive["view_counts"] if "view_counts" in archive else None,
        )
        return ids.astype(np.int64), scores.astype(np.float32), "archive", view_names, view_weights, view_counts

    npz_path = run_root / "reference_camera_evidence.npz"
    if npz_path.exists():
        with np.load(npz_path) as payload:
            ids = np.asarray(payload["reference_camera_ids"], dtype=np.int64).reshape(-1)
            scores = np.asarray(payload["reference_camera_scores"], dtype=np.float32).reshape(-1)
            view_names, view_weights, view_counts = _sanitize_view_evidence(
                payload["image_names"] if "image_names" in payload else default_view_names,
                payload["view_weights"] if "view_weights" in payload else None,
                payload["view_counts"] if "view_counts" in payload else None,
            )
        return ids, scores, "reference_camera_evidence.npz", view_names, view_weights, view_counts

    json_path = run_root / "reference_camera_evidence.json"
    json_payload = _load_optional_json(json_path)
    if json_payload:
        ids = np.asarray(json_payload.get("reference_camera_ids", []), dtype=np.int64).reshape(-1)
        scores = np.asarray(json_payload.get("reference_camera_scores", []), dtype=np.float32).reshape(-1)
        view_names, view_weights, view_counts = _sanitize_view_evidence(
            json_payload.get("image_names", default_view_names),
            json_payload.get("view_weights"),
            json_payload.get("view_counts"),
        )
        return ids, scores, "reference_camera_evidence.json", view_names, view_weights, view_counts

    if camera_bundle is not None and camera_bundle.cams2world.shape[0] > 0:
        camera_centers = np.asarray(camera_bundle.cams2world[:, :3, 3], dtype=np.float32)
        d2 = np.sum((positions[:, None, :] - camera_centers[None, :, :]) ** 2, axis=2)
        ids = np.argmin(d2, axis=1).astype(np.int64)
        dist = np.sqrt(np.min(d2, axis=1).clip(min=1e-8))
        scale = max(float(np.median(dist)), 1e-6)
        scores = np.exp(-dist / scale).astype(np.float32)
        view_names, view_weights, view_counts = _empty_view_evidence()
        return ids, scores, "derived_nearest_camera_center", view_names, view_weights, view_counts

    view_names, view_weights, view_counts = _empty_view_evidence()
    return (
        np.full((node_count,), -1, dtype=np.int64),
        np.zeros((node_count,), dtype=np.float32),
        "missing",
        view_names,
        view_weights,
        view_counts,
    )


def _confidence_to_pose_error(confidence: np.ndarray, min_conf_thr: float):
    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if confidence.size == 0:
        return confidence

    log_conf = np.log1p(np.clip(confidence, 0.0, None))
    low = np.log1p(max(float(min_conf_thr), 0.0))
    high = float(np.quantile(log_conf, 0.90))
    if high <= low + 1e-6:
        normalized = np.ones_like(log_conf, dtype=np.float32)
    else:
        normalized = np.clip((log_conf - low) / float(high - low), 0.0, 1.0).astype(np.float32)
    return (0.35 + 2.65 * (1.0 - normalized)).astype(np.float32)


def _calibrate_pose_error_with_view_audit(corr_error: np.ndarray, view: FoundationDenseViewEvidence):
    corr_error = np.asarray(corr_error, dtype=np.float32).reshape(-1)
    if corr_error.size == 0:
        return corr_error

    audit_median = max(float(getattr(view, "audit_median_px_error", 0.0) or 0.0), 0.0)
    audit_p90 = max(float(getattr(view, "audit_p90_px_error", 0.0) or 0.0), audit_median)
    audit_sampled = max(int(getattr(view, "audit_sampled_corr", 0) or 0), 0)
    audit_projected = max(int(getattr(view, "audit_projected_corr", 0) or 0), 0)
    audit_in_frame = max(int(getattr(view, "audit_in_frame_corr", 0) or 0), 0)
    if audit_median <= 0.0 and audit_p90 <= 0.0 and audit_projected <= 0:
        return corr_error

    tail_px = max(audit_p90 - audit_median, 0.0)
    inlier_ratio = 1.0 if audit_projected <= 0 else float(np.clip(audit_in_frame / max(audit_projected, 1), 0.0, 1.0))
    effective_samples = max(audit_sampled, audit_in_frame, audit_projected)
    sample_blend = 1.0 if effective_samples <= 0 else float(np.clip(effective_samples / 256.0, 0.25, 1.0))
    audit_sigma = sample_blend * (audit_median + 0.35 * tail_px + 2.0 * max(0.0, 0.95 - inlier_ratio))
    audit_sigma = float(np.clip(audit_sigma, 0.0, 48.0))
    return np.sqrt(np.square(corr_error) + audit_sigma * audit_sigma).astype(np.float32)


def _subsample_pose_correspondences(
    corr_xy: np.ndarray,
    corr_xyz: np.ndarray,
    corr_error: np.ndarray,
    corr_score: np.ndarray,
    source_width: int,
    source_height: int,
    max_correspondences: int,
    return_indices: bool = False,
):
    if corr_xy.shape[0] <= max_correspondences or max_correspondences <= 0:
        indices = np.arange(corr_xy.shape[0], dtype=np.int64)
        if return_indices:
            return corr_xy, corr_xyz, corr_error, indices
        return corr_xy, corr_xyz, corr_error

    order = np.argsort(-corr_score.astype(np.float32), kind="stable")
    cell_area = (float(source_width) * float(source_height)) / float(max_correspondences)
    cell_size = max(1, int(round(np.sqrt(max(cell_area, 1.0)))))
    occupied_cells = set()
    selected = []

    for index in order:
        cell_key = (
            int(corr_xy[index, 0] // float(cell_size)),
            int(corr_xy[index, 1] // float(cell_size)),
        )
        if cell_key in occupied_cells:
            continue
        occupied_cells.add(cell_key)
        selected.append(int(index))
        if len(selected) >= int(max_correspondences):
            break

    if len(selected) < int(max_correspondences):
        used = np.zeros((corr_xy.shape[0],), dtype=bool)
        if selected:
            used[np.asarray(selected, dtype=np.int64)] = True
        remaining = order[~used[order]]
        needed = int(max_correspondences) - len(selected)
        selected.extend(remaining[:needed].tolist())

    selected = np.asarray(selected[: int(max_correspondences)], dtype=np.int64)
    selected.sort()
    if return_indices:
        return corr_xy[selected], corr_xyz[selected], corr_error[selected], selected
    return corr_xy[selected], corr_xyz[selected], corr_error[selected]


def _map_preprocessed_pixels_to_source(corr_xy: np.ndarray, view: FoundationDenseViewEvidence):
    coord_space = (view.coordinate_space or "").lower()
    if coord_space not in FOUNDATION_PREPROCESSED_IMAGE_COORDINATE_SPACES:
        return corr_xy, int(view.width), int(view.height)
    if (
        view.source_width <= 0
        or view.source_height <= 0
        or view.resized_width <= 0
        or view.resized_height <= 0
    ):
        return corr_xy, int(view.width), int(view.height)

    mapped = np.asarray(corr_xy, dtype=np.float32).copy()
    mapped[:, 0] += float(view.crop_left)
    mapped[:, 1] += float(view.crop_top)
    scale_x = float(view.source_width) / float(max(view.resized_width, 1))
    scale_y = float(view.source_height) / float(max(view.resized_height, 1))
    mapped[:, 0] *= scale_x
    mapped[:, 1] *= scale_y
    return mapped.astype(np.float32), int(view.source_width), int(view.source_height)


def load_foundation_pose_correspondences(
    atlas_path: str | Path,
    image_name: str,
    max_correspondences: int = FOUNDATION_POSE_MAX_CORRESPONDENCES,
):
    run_root = resolve_foundation_atlas_root(atlas_path)
    camera_bundle = _load_camera_bundle(run_root)
    correspondence_manifest = _load_correspondence_manifest(run_root, camera_bundle)
    if correspondence_manifest is None:
        return None

    view = correspondence_manifest.views.get(image_name)
    if view is None:
        return None
    if view.sparse_path is not None:
        try:
            with np.load(view.sparse_path) as payload:
                corr_xy = np.asarray(payload["xy"], dtype=np.float32).reshape(-1, 2)
                corr_xyz = np.asarray(payload["xyz"], dtype=np.float32).reshape(-1, 3)
                if "pose_error" in payload:
                    corr_error = np.asarray(payload["pose_error"], dtype=np.float32).reshape(-1)
                elif "confidence" in payload:
                    corr_error = _confidence_to_pose_error(np.asarray(payload["confidence"], dtype=np.float32), correspondence_manifest.min_conf_thr)
                else:
                    corr_error = np.ones((corr_xy.shape[0],), dtype=np.float32)
                corr_score = (
                    np.asarray(payload["trust"], dtype=np.float32).reshape(-1)
                    if "trust" in payload
                    else np.reciprocal(corr_error.clip(min=1e-4))
                )
                atlas_node_ids = (
                    np.asarray(payload["atlas_node_id"], dtype=np.int64).reshape(-1)
                    if "atlas_node_id" in payload
                    else np.full((corr_xy.shape[0],), -1, dtype=np.int64)
                )
                atlas_reliability = (
                    np.asarray(payload["atlas_reliability"], dtype=np.float32).reshape(-1)
                    if "atlas_reliability" in payload
                    else np.ones((corr_xy.shape[0],), dtype=np.float32)
                )
        except Exception:
            return None

        if corr_xy.shape[0] == 0 or corr_xyz.shape[0] != corr_xy.shape[0]:
            return None
        count = corr_xy.shape[0]
        if corr_error.shape[0] != count:
            corr_error = np.ones((count,), dtype=np.float32)
        if corr_score.shape[0] != count:
            corr_score = np.reciprocal(corr_error.clip(min=1e-4))
        if atlas_node_ids.shape[0] != count:
            atlas_node_ids = np.full((count,), -1, dtype=np.int64)
        if atlas_reliability.shape[0] != count:
            atlas_reliability = np.ones((count,), dtype=np.float32)

        valid = (
            np.isfinite(corr_xy).all(axis=1)
            & np.isfinite(corr_xyz).all(axis=1)
            & np.isfinite(corr_error)
            & np.isfinite(corr_score)
        )
        if not np.any(valid):
            return None
        corr_xy = corr_xy[valid]
        corr_xyz = corr_xyz[valid]
        corr_error = corr_error[valid]
        corr_score = corr_score[valid]
        atlas_node_ids = atlas_node_ids[valid]
        atlas_reliability = atlas_reliability[valid]

        source_width = int(view.source_width if view.source_width > 0 else view.width)
        source_height = int(view.source_height if view.source_height > 0 else view.height)
        corr_xy, corr_xyz, corr_error, selected_indices = _subsample_pose_correspondences(
            corr_xy,
            corr_xyz,
            corr_error,
            corr_score,
            source_width=int(source_width),
            source_height=int(source_height),
            max_correspondences=int(max_correspondences),
            return_indices=True,
        )
        atlas_node_ids = atlas_node_ids[selected_indices]
        atlas_reliability = atlas_reliability[selected_indices]
        corr_score = corr_score[selected_indices]
        return {
            "xy": corr_xy,
            "xyz": corr_xyz,
            "error": corr_error,
            "source_width": int(source_width),
            "source_height": int(source_height),
            "atlas_node_ids": atlas_node_ids.astype(np.int64),
            "atlas_reliability": atlas_reliability.astype(np.float32),
            "trust": corr_score.astype(np.float32),
            "is_atlas_native": np.ones((corr_xy.shape[0],), dtype=np.bool_),
            "source": "atlas_sparse",
        }

    if view.points_path is None or view.confidence_path is None:
        return None

    points_xyz = np.asarray(np.load(view.points_path), dtype=np.float32)
    confidence = np.asarray(np.load(view.confidence_path), dtype=np.float32)
    if points_xyz.ndim != 3 or points_xyz.shape[2] != 3:
        return None
    if confidence.ndim != 2 or confidence.shape != points_xyz.shape[:2]:
        return None

    min_conf_thr = float(correspondence_manifest.min_conf_thr)
    valid = np.isfinite(points_xyz).all(axis=2) & np.isfinite(confidence) & (confidence >= min_conf_thr)
    if not np.any(valid):
        return None

    yy, xx = np.nonzero(valid)
    corr_xy = np.stack((xx, yy), axis=1).astype(np.float32)
    corr_xyz = points_xyz[valid].astype(np.float32)
    corr_score = confidence[valid].astype(np.float32)
    corr_error = _confidence_to_pose_error(corr_score, min_conf_thr)
    corr_error = _calibrate_pose_error_with_view_audit(corr_error, view)
    corr_xy, source_width, source_height = _map_preprocessed_pixels_to_source(corr_xy, view)
    corr_xy, corr_xyz, corr_error = _subsample_pose_correspondences(
        corr_xy,
        corr_xyz,
        corr_error,
        corr_score,
        source_width=int(source_width),
        source_height=int(source_height),
        max_correspondences=int(max_correspondences),
    )
    return (
        corr_xy,
        corr_xyz,
        corr_error,
        int(source_width),
        int(source_height),
    )


def load_foundation_atlas(
    atlas_path: str | Path,
    fallback_point_cloud: BasicPointCloud | None = None,
    color_sample_limit: int = 50000,
    seed: int = 42,
    surface_stable_min: float = 0.12,
    edge_stable_min: float = 0.08,
    scale_multiplier: float = 1.0,
    surface_thickness_ratio: float = 0.15,
    edge_thickness_ratio: float = 0.18,
    unstable_scale_ratio: float = 0.35,
):
    run_root = resolve_foundation_atlas_root(atlas_path)
    _load_scene_alignment_contract(run_root)
    archive_path = run_root / "atlas_nodes.npz"
    if not archive_path.exists():
        raise FileNotFoundError(f"Atlas archive not found: {archive_path}")

    with np.load(archive_path) as archive_file:
        archive = {key: np.asarray(archive_file[key]) for key in archive_file.files}
    positions = np.asarray(archive["positions"], dtype=np.float32)
    support = np.asarray(archive["support"], dtype=np.float32)
    basis = np.asarray(archive["basis"], dtype=np.float32)
    normals = np.asarray(archive["normal"], dtype=np.float32)
    raw_score = np.asarray(archive["raw_score"], dtype=np.float32).reshape(-1) if "raw_score" in archive else None
    radius = np.asarray(archive["radius"], dtype=np.float32).reshape(-1)
    reliability = np.asarray(archive["reliability"], dtype=np.float32).reshape(-1)
    atlas_class = np.asarray(archive["atlas_class"], dtype=np.int64).reshape(-1)
    anisotropy_ref = np.asarray(archive["anisotropy_ref"], dtype=np.float32)
    if "neighbor_indices" in archive:
        neighbor_indices = np.asarray(archive["neighbor_indices"], dtype=np.int64)
    else:
        neighbor_indices = np.arange(positions.shape[0], dtype=np.int64)[:, None]
    if raw_score is None:
        raw_score = reliability.copy()

    node_count = int(positions.shape[0])
    camera_bundle = _load_camera_bundle(run_root)
    correspondence_manifest = _load_correspondence_manifest(run_root, camera_bundle)
    hash_info = _load_hash_info(run_root, archive)
    build_config = _load_optional_json(run_root / "build_config.json") or {}

    calibration_residual = _load_archive_vector(
        archive,
        "calibration_residual",
        node_count,
        np.float32,
        0.0,
    )
    if "calibration_residual" not in archive:
        calibration_residual = np.clip(1.0 - reliability, 0.0, 1.0).astype(np.float32)

    node_confidence = _load_archive_vector(
        archive,
        "node_confidence",
        node_count,
        np.float32,
        0.0,
    )
    if "node_confidence" not in archive:
        node_confidence = raw_score.astype(np.float32).copy()

    point_support = _load_archive_vector(
        archive,
        "point_support",
        node_count,
        np.int32,
        max(int(neighbor_indices.shape[1]), 1),
    )
    view_support = _load_archive_vector(
        archive,
        "view_support",
        node_count,
        np.int32,
        1,
    )
    if "view_coverage" in archive:
        view_coverage = np.asarray(archive["view_coverage"], dtype=np.float32).reshape(-1)
    else:
        camera_count = len(camera_bundle.image_names) if camera_bundle is not None else 1
        view_coverage = np.clip(view_support.astype(np.float32) / max(float(camera_count), 1.0), 0.0, 1.0)

    if "support_score" in archive:
        support_score = np.asarray(archive["support_score"], dtype=np.float32).reshape(-1)
    else:
        point_support_score = _normalize_scores(np.log1p(point_support.astype(np.float32)))
        support_score = np.sqrt(np.clip(point_support_score * view_coverage, 0.0, 1.0)).astype(np.float32)

    linearness = _load_archive_vector(archive, "linearness", node_count, np.float32, 0.0)
    planarness = _load_archive_vector(archive, "planarness", node_count, np.float32, 0.0)
    if "scattering" in archive:
        scattering = np.asarray(archive["scattering"], dtype=np.float32).reshape(-1)
    else:
        scattering = np.clip(1.0 - linearness - planarness, 0.0, 1.0).astype(np.float32)

    (
        reference_camera_ids,
        reference_camera_scores,
        reference_camera_source,
        reference_view_names,
        reference_view_weights,
        reference_view_counts,
    ) = _load_reference_camera_evidence(
        run_root,
        archive,
        camera_bundle,
        positions,
    )

    colors = _estimate_atlas_colors(
        run_root,
        positions,
        atlas_class,
        fallback_point_cloud,
        int(color_sample_limit),
        int(seed),
    )
    atlas_ids = np.arange(positions.shape[0], dtype=np.int64)
    gaussian_state = _initial_gaussian_states(atlas_class, reliability, surface_stable_min, edge_stable_min)
    init_scales = _initial_scales(
        radius,
        atlas_class,
        scale_multiplier,
        surface_thickness_ratio,
        edge_thickness_ratio,
        unstable_scale_ratio,
    )
    init_rotations = _rotation_matrices_to_quaternions(basis)

    return FoundationAtlasInit(
        source_path=str(run_root),
        positions=positions,
        colors=colors,
        normals=normals,
        support=support,
        basis=basis,
        raw_score=raw_score.astype(np.float32),
        radius=radius.astype(np.float32),
        reliability=reliability.astype(np.float32),
        calibration_residual=calibration_residual.astype(np.float32),
        atlas_class=atlas_class,
        anisotropy_ref=anisotropy_ref.astype(np.float32),
        neighbor_indices=neighbor_indices.astype(np.int64),
        node_confidence=node_confidence.astype(np.float32),
        point_support=point_support.astype(np.int32),
        view_support=view_support.astype(np.int32),
        view_coverage=view_coverage.astype(np.float32),
        support_score=support_score.astype(np.float32),
        linearness=linearness.astype(np.float32),
        planarness=planarness.astype(np.float32),
        scattering=scattering.astype(np.float32),
        atlas_ids=atlas_ids,
        gaussian_state=gaussian_state,
        init_scales=init_scales,
        init_rotations=init_rotations,
        reference_camera_ids=reference_camera_ids.astype(np.int64),
        reference_camera_scores=reference_camera_scores.astype(np.float32),
        reference_view_names=list(reference_view_names),
        reference_view_weights=np.asarray(reference_view_weights, dtype=np.float32),
        reference_view_counts=np.asarray(reference_view_counts, dtype=np.int32),
        reference_camera_source=reference_camera_source,
        hash_info=hash_info,
        build_config=build_config,
        camera_bundle=camera_bundle,
        correspondence_manifest=correspondence_manifest,
    )


def summarize_atlas_initialization(atlas: FoundationAtlasInit):
    class_counts = {
        ATLAS_CLASS_NAMES[class_id]: int(np.sum(atlas.atlas_class == class_id))
        for class_id in ATLAS_CLASS_NAMES
    }
    state_counts = {
        GAUSSIAN_STATE_NAMES[state_id]: int(np.sum(atlas.gaussian_state == state_id))
        for state_id in GAUSSIAN_STATE_NAMES
    }
    return {
        "source_path": atlas.source_path,
        "num_nodes": int(atlas.positions.shape[0]),
        "mean_raw_score": float(np.mean(atlas.raw_score)),
        "mean_reliability": float(np.mean(atlas.reliability)),
        "median_reliability": float(np.median(atlas.reliability)),
        "mean_calibration_residual": float(np.mean(atlas.calibration_residual)),
        "median_calibration_residual": float(np.median(atlas.calibration_residual)),
        "mean_radius": float(np.mean(atlas.radius)),
        "median_radius": float(np.median(atlas.radius)),
        "mean_point_support": float(np.mean(atlas.point_support)),
        "mean_view_support": float(np.mean(atlas.view_support)),
        "mean_view_coverage": float(np.mean(atlas.view_coverage)),
        "mean_support_score": float(np.mean(atlas.support_score)),
        "mean_node_confidence": float(np.mean(atlas.node_confidence)),
        "reference_camera_source": atlas.reference_camera_source,
        "reference_camera_coverage": float(np.mean(atlas.reference_camera_ids >= 0)),
        "has_camera_bundle": atlas.camera_bundle is not None,
        "camera_count": 0 if atlas.camera_bundle is None else int(len(atlas.camera_bundle.image_names)),
        "has_correspondence_manifest": atlas.correspondence_manifest is not None,
        "correspondence_view_count": 0 if atlas.correspondence_manifest is None else int(len(atlas.correspondence_manifest.views)),
        "hash_info": atlas.hash_info,
        "class_counts": class_counts,
        "state_counts": state_counts,
    }


def save_json(data, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
