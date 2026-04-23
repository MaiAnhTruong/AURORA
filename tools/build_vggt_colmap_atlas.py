from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
GS_ROOT = REPO_ROOT / "gaussian-splatting"
MAST3R_ROOT = REPO_ROOT / "mast3r"

if str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))
if str(MAST3R_ROOT) not in sys.path:
    sys.path.insert(0, str(MAST3R_ROOT))

from utils.read_write_model import qvec2rotmat, read_model  # noqa: E402
from mast3r import foundation_atlas as foundation_atlas_module  # noqa: E402
from mast3r.foundation_atlas import (  # noqa: E402
    atlas_debug_colors,
    build_foundation_geometry_atlas,
    plot_foundation_atlas_report,
    save_atlas_npz,
    save_foundation_atlas_sidecars,
    save_json,
    save_ply,
    summarize_foundation_atlas,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata_for_path(path: Path) -> dict:
    path = Path(path).expanduser().resolve()

    def _run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path.parent), *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return result.stdout.strip()
        except Exception:
            return None

    root = _run_git(["rev-parse", "--show-toplevel"])
    if not root:
        return {"available": False}

    commit = _run_git(["rev-parse", "HEAD"])
    rel_path = None
    try:
        rel_path = str(path.relative_to(Path(root).resolve())).replace("\\", "/")
    except Exception:
        rel_path = None

    status_args = ["status", "--short", "--untracked-files=no"]
    if rel_path:
        status_args.extend(["--", rel_path])
    status = _run_git(status_args)

    return {
        "available": True,
        "root": str(Path(root).resolve()),
        "commit": commit,
        "relative_path": rel_path,
        "dirty": bool(status),
        "status_short": status or "",
    }


def build_source_stamp() -> dict:
    builder_path = Path(__file__).expanduser().resolve()
    foundation_path = Path(inspect.getsourcefile(foundation_atlas_module) or "").expanduser().resolve()
    return {
        "schema_version": 1,
        "stamp_status": "recorded_at_build_time",
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "builder": {
            "absolute_path": str(builder_path),
            "sha256": _sha256_file(builder_path),
            "git": _git_metadata_for_path(builder_path),
        },
        "foundation_atlas_module": {
            "import_name": foundation_atlas_module.__name__,
            "absolute_path": str(foundation_path),
            "sha256": _sha256_file(foundation_path),
            "git": _git_metadata_for_path(foundation_path),
        },
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _extract_expected_hash(expected: dict, section: str) -> str | None:
    section_payload = expected.get(section)
    if isinstance(section_payload, dict) and section_payload.get("sha256"):
        return str(section_payload["sha256"]).lower()

    if section == "builder":
        candidates = (
            expected.get("builder_sha256"),
            expected.get("current_builder_sha256"),
            expected.get("current_builder_after_source_stamp_patch", {}).get("sha256")
            if isinstance(expected.get("current_builder_after_source_stamp_patch"), dict)
            else None,
        )
    else:
        candidates = (
            expected.get("foundation_atlas_sha256"),
            expected.get("foundation_atlas_module_sha256"),
            expected.get("foundation_atlas_module_candidate", {}).get("sha256")
            if isinstance(expected.get("foundation_atlas_module_candidate"), dict)
            else None,
        )
    for value in candidates:
        if value:
            return str(value).lower()
    source_stamp = expected.get("source_stamp")
    if isinstance(source_stamp, dict):
        return _extract_expected_hash(source_stamp, section)
    return None


def validate_source_stamp(
    stamp: dict,
    expected_source_stamp: str = "",
    enforce: bool = False,
    allow_external_foundation_atlas: bool = False,
) -> list[str]:
    failures: list[str] = []
    warnings: list[str] = []

    foundation_path = Path(stamp["foundation_atlas_module"]["absolute_path"]).resolve()
    expected_foundation_root = (MAST3R_ROOT / "mast3r").resolve()
    if not allow_external_foundation_atlas and not _is_relative_to(foundation_path, expected_foundation_root):
        failures.append(
            f"foundation_atlas_module resolves outside expected repo module root: "
            f"{foundation_path} not under {expected_foundation_root}"
        )

    if expected_source_stamp:
        expected_path = Path(expected_source_stamp).expanduser().resolve()
        with open(expected_path, "r", encoding="utf-8") as file:
            expected_payload = json.load(file)
        expected_builder_hash = _extract_expected_hash(expected_payload, "builder")
        expected_foundation_hash = _extract_expected_hash(expected_payload, "foundation_atlas_module")
        if expected_builder_hash and stamp["builder"]["sha256"].lower() != expected_builder_hash:
            failures.append(
                "builder sha256 mismatch: "
                f"actual={stamp['builder']['sha256']} expected={expected_builder_hash}"
            )
        if expected_foundation_hash and stamp["foundation_atlas_module"]["sha256"].lower() != expected_foundation_hash:
            failures.append(
                "foundation_atlas_module sha256 mismatch: "
                f"actual={stamp['foundation_atlas_module']['sha256']} expected={expected_foundation_hash}"
            )
        if not expected_builder_hash and not expected_foundation_hash:
            warnings.append(f"expected source stamp did not contain recognized sha256 fields: {expected_path}")

    for message in warnings:
        print(f"[SOURCE WARNING] {message}")
    for message in failures:
        print(f"[SOURCE ERROR] {message}")
    if failures and enforce:
        raise RuntimeError("Source stamp validation failed under --enforce_source_stamp.")
    return failures + warnings


def print_source_stamp(stamp: dict, validation_messages: list[str] | None = None):
    builder = stamp["builder"]
    foundation = stamp["foundation_atlas_module"]
    foundation_git = foundation.get("git", {}) or {}
    builder_git = builder.get("git", {}) or {}
    print("[SOURCE] atlas_input_mode=vggt_colmap_sparse_tracks dense_replay_used=false")
    print(f"[SOURCE] builder path={builder['absolute_path']}")
    print(f"[SOURCE] builder sha256={builder['sha256']}")
    print(f"[SOURCE] builder git_commit={builder_git.get('commit', 'unavailable')} dirty={builder_git.get('dirty', 'unavailable')}")
    print(f"[SOURCE] foundation_atlas_module path={foundation['absolute_path']}")
    print(f"[SOURCE] foundation_atlas_module sha256={foundation['sha256']}")
    print(
        "[SOURCE] foundation_atlas_module "
        f"git_commit={foundation_git.get('commit', 'unavailable')} dirty={foundation_git.get('dirty', 'unavailable')}"
    )
    if validation_messages:
        print(f"[SOURCE] validation_messages={len(validation_messages)}")


def find_sparse_dir(dataset_root: Path) -> Path:
    candidates = [dataset_root / "sparse" / "0", dataset_root / "sparse"]
    for candidate in candidates:
        if (candidate / "cameras.bin").exists() and (candidate / "images.bin").exists():
            return candidate
        if (candidate / "cameras.txt").exists() and (candidate / "images.txt").exists():
            return candidate
    raise FileNotFoundError(f"Could not find COLMAP model under {dataset_root / 'sparse'}")


def ensure_sparse_zero(dataset_root: Path, sparse_dir: Path) -> Path:
    target = dataset_root / "sparse" / "0"
    if sparse_dir.resolve() == target.resolve():
        return target
    target.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"):
        src = sparse_dir / name
        if src.exists():
            dst = target / name
            if not dst.exists():
                shutil.copy2(src, dst)
    return target


def camera_intrinsics_matrix(camera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float32).reshape(-1)
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx = float(params[1])
        cy = float(params[2])
    elif camera.model == "PINHOLE":
        fx = float(params[0])
        fy = float(params[1])
        cx = float(params[2])
        cy = float(params[3])
    else:
        raise ValueError(f"Unsupported camera model for Gaussian loader: {camera.model}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def image_cam2world(image) -> np.ndarray:
    rotation = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
    translation = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[:3, :3] = rotation.T.astype(np.float32)
    cam2world[:3, 3] = (-rotation.T @ translation).astype(np.float32)
    return cam2world


def _point_error_score(point) -> float:
    error = float(point.error)
    if error < 0.0 or not math.isfinite(error):
        return 1.0
    return float(math.exp(-max(error, 0.0) / 2.0))


def _track_geometry_scores(
    xyz: np.ndarray,
    camera_centers: list[np.ndarray],
    depths: list[float],
    *,
    theta_target_deg: float = 12.0,
    depth_cv_tau: float = 0.35,
):
    centers = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    xyz64 = np.asarray(xyz, dtype=np.float64).reshape(3)
    if centers.shape[0] >= 2:
        rays = xyz64[None, :] - centers
        ray_norm = np.linalg.norm(rays, axis=1, keepdims=True)
        valid_rays = np.isfinite(rays).all(axis=1) & (ray_norm[:, 0] > 1e-8)
        rays = rays[valid_rays] / np.clip(ray_norm[valid_rays], 1e-8, None)
    else:
        rays = np.zeros((0, 3), dtype=np.float64)

    if rays.shape[0] >= 2:
        dots = np.clip(rays @ rays.T, -1.0, 1.0)
        tri = np.triu_indices(rays.shape[0], k=1)
        angles = np.arccos(dots[tri])
        median_angle = float(np.median(angles)) if angles.size else 0.0
        theta_target = max(math.radians(float(theta_target_deg)), 1e-6)
        parallax_score = float(np.clip(median_angle / theta_target, 0.0, 1.0))
        mean_ray = rays.mean(axis=0)
        mean_norm = float(np.linalg.norm(mean_ray))
        if mean_norm > 1e-8:
            view_incidence_cos = np.clip(rays @ (mean_ray / mean_norm), -1.0, 1.0).astype(np.float32)
        else:
            view_incidence_cos = np.ones((rays.shape[0],), dtype=np.float32)
    else:
        median_angle = 0.0
        parallax_score = 0.0
        view_incidence_cos = np.ones((max(len(camera_centers), 1),), dtype=np.float32)

    depth_values = np.asarray(depths, dtype=np.float64).reshape(-1)
    depth_values = depth_values[np.isfinite(depth_values) & (depth_values > 0.0)]
    if depth_values.size >= 2:
        depth_cv = float(np.std(depth_values) / (np.mean(depth_values) + 1e-6))
        depth_consistency = float(np.exp(-depth_cv / max(float(depth_cv_tau), 1e-6)))
    elif depth_values.size == 1:
        depth_consistency = 1.0
    else:
        depth_consistency = 0.5

    distances = np.linalg.norm(xyz64[None, :] - centers, axis=1) if centers.shape[0] else np.ones((1,), dtype=np.float64)
    finite_distances = distances[np.isfinite(distances) & (distances > 0.0)]
    distance_scale = float(np.median(finite_distances)) if finite_distances.size else 1.0
    distance_norm = np.clip(distances / max(distance_scale, 1e-6), 0.0, 8.0).astype(np.float32)

    return {
        "parallax_score": parallax_score,
        "depth_consistency": float(np.clip(depth_consistency, 0.0, 1.0)),
        "view_angle_spread": median_angle,
        "view_incidence_cos": view_incidence_cos,
        "camera_center_distance_norm": distance_norm,
    }


def load_colmap_sparse(sparse_dir: Path):
    ext = ".bin" if (sparse_dir / "cameras.bin").exists() else ".txt"
    model = read_model(str(sparse_dir), ext=ext)
    if model is None:
        raise RuntimeError(f"Failed to read COLMAP model from {sparse_dir}")
    cameras, images, points3d = model
    if not images:
        raise RuntimeError(f"No registered images in {sparse_dir}")
    if not points3d:
        raise RuntimeError(f"No 3D points in {sparse_dir}")
    return cameras, images, points3d


def build_camera_payload(cameras, images, image_dir: Path):
    ordered_images = sorted(images.values(), key=lambda image: image.name)
    image_names = [image.name for image in ordered_images]
    image_paths = [image_dir / image.name for image in ordered_images]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing image files referenced by COLMAP: " + ", ".join(missing[:8]))

    cams2world = []
    intrinsics = []
    image_id_to_view = {}
    dense_stats = {}
    for view_id, image in enumerate(ordered_images):
        camera = cameras[image.camera_id]
        cams2world.append(image_cam2world(image))
        intrinsics.append(camera_intrinsics_matrix(camera))
        image_id_to_view[int(image.id)] = int(view_id)
        dense_stats[image.name] = {
            "height": int(camera.height),
            "width": int(camera.width),
            "mean_confidence": 0.0,
            "median_confidence": 0.0,
            "finite_points": 0,
            "points_above_conf_thr": 0,
        }
    return (
        ordered_images,
        image_names,
        [str(path) for path in image_paths],
        np.asarray(cams2world, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float32),
        image_id_to_view,
        dense_stats,
    )


def confidence_from_track(
    point,
    num_views: int,
    parallax_score: float | None = None,
    depth_consistency: float | None = None,
) -> float:
    track_len = max(int(len(point.image_ids)), 1)
    track_saturation = 1.0 - math.exp(-track_len / 3.0)
    track_coverage = min(track_len / max(float(num_views), 1.0), 1.0)
    error_score = _point_error_score(point)
    track_score = 0.55 * track_saturation + 0.20 * track_coverage + 0.25 * error_score
    if parallax_score is None:
        parallax_score = track_coverage
    if depth_consistency is None:
        depth_consistency = error_score
    confidence_base = (
        0.45 * track_score
        + 0.30 * float(np.clip(parallax_score, 0.0, 1.0))
        + 0.25 * float(np.clip(depth_consistency, 0.0, 1.0))
    )
    return float(1.0 + 9.0 * np.clip(confidence_base, 0.0, 1.0))


def make_observation_cloud(points3d, images, cameras, image_id_to_view: dict[int, int], num_views: int):
    points = []
    colors = []
    confidences = []
    image_ids = []
    pixels_xy = []
    point3d_ids = []
    track_lengths = []
    point_errors = []
    depths = []
    parallax_scores = []
    depth_consistencies = []
    view_angle_spreads = []
    view_incidence_cosines = []
    camera_center_distance_norms = []
    view_counts = []

    image_rot_trans = {}
    image_centers = {}
    for image_id, image in images.items():
        if int(image_id) not in image_id_to_view:
            continue
        rotation = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
        translation = np.asarray(image.tvec, dtype=np.float64).reshape(3)
        image_rot_trans[int(image_id)] = (rotation, translation)
        image_centers[int(image_id)] = (-rotation.T @ translation).astype(np.float64)

    for point_id, point in points3d.items():
        xyz = np.asarray(point.xyz, dtype=np.float32).reshape(3)
        if not np.isfinite(xyz).all():
            continue
        rgb = np.asarray(point.rgb, dtype=np.float32).reshape(3)
        if rgb.max(initial=0.0) > 1.0:
            rgb = rgb / 255.0
        observation_records = []
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            image_id = int(image_id)
            if image_id not in image_id_to_view or image_id not in images:
                continue
            point2d_idx = int(point2d_idx)
            image = images[image_id]
            if point2d_idx < 0 or point2d_idx >= len(image.xys):
                continue
            xy = np.asarray(image.xys[point2d_idx], dtype=np.float32).reshape(2)
            if not np.isfinite(xy).all():
                continue
            rotation, translation = image_rot_trans.get(image_id, (None, None))
            if rotation is None:
                continue
            depth = float((rotation @ xyz.astype(np.float64) + translation)[2])
            if not math.isfinite(depth) or depth <= 0.0:
                continue
            observation_records.append((image_id_to_view[image_id], xy, depth, image_centers[image_id]))
        if not observation_records:
            continue
        observation_depths = [record[2] for record in observation_records]
        observation_centers = [record[3] for record in observation_records]
        geometry_scores = _track_geometry_scores(xyz, observation_centers, observation_depths)
        conf = confidence_from_track(
            point,
            num_views,
            parallax_score=geometry_scores["parallax_score"],
            depth_consistency=geometry_scores["depth_consistency"],
        )
        incidence_cos = np.asarray(geometry_scores["view_incidence_cos"], dtype=np.float32).reshape(-1)
        distance_norm = np.asarray(geometry_scores["camera_center_distance_norm"], dtype=np.float32).reshape(-1)
        if incidence_cos.shape[0] != len(observation_records):
            incidence_cos = np.ones((len(observation_records),), dtype=np.float32)
        if distance_norm.shape[0] != len(observation_records):
            distance_norm = np.ones((len(observation_records),), dtype=np.float32)
        track_len = max(int(len(point.image_ids)), 1)
        valid_view_count = int(len({record[0] for record in observation_records}))
        point_error = float(point.error) if math.isfinite(float(point.error)) else -1.0
        for record_index, (view_id, xy, depth, _) in enumerate(observation_records):
            points.append(xyz)
            colors.append(rgb)
            confidences.append(conf)
            image_ids.append(view_id)
            pixels_xy.append(xy)
            point3d_ids.append(int(point_id))
            track_lengths.append(track_len)
            point_errors.append(point_error)
            depths.append(depth)
            parallax_scores.append(geometry_scores["parallax_score"])
            depth_consistencies.append(geometry_scores["depth_consistency"])
            view_angle_spreads.append(geometry_scores["view_angle_spread"])
            view_incidence_cosines.append(float(incidence_cos[record_index]))
            camera_center_distance_norms.append(float(distance_norm[record_index]))
            view_counts.append(valid_view_count)

    if not points:
        raise RuntimeError("No valid VGGT/COLMAP observations could be converted to atlas input.")

    return {
        "points": np.asarray(points, dtype=np.float32),
        "colors": np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0),
        "confidences": np.asarray(confidences, dtype=np.float32),
        "image_ids": np.asarray(image_ids, dtype=np.int32),
        "pixels_xy": np.asarray(pixels_xy, dtype=np.float32),
        "point3d_ids": np.asarray(point3d_ids, dtype=np.int64),
        "track_lengths": np.asarray(track_lengths, dtype=np.int32),
        "point_errors": np.asarray(point_errors, dtype=np.float32),
        "depths": np.asarray(depths, dtype=np.float32),
        "parallax_score": np.asarray(parallax_scores, dtype=np.float32),
        "depth_consistency": np.asarray(depth_consistencies, dtype=np.float32),
        "view_angle_spread": np.asarray(view_angle_spreads, dtype=np.float32),
        "view_incidence_cos": np.asarray(view_incidence_cosines, dtype=np.float32),
        "camera_center_distance_norm": np.asarray(camera_center_distance_norms, dtype=np.float32),
        "view_counts": np.asarray(view_counts, dtype=np.int32),
        "view_count": np.asarray(view_counts, dtype=np.int32),
        "source_types": np.zeros((len(points),), dtype=np.int32),
    }


def make_dense_geometry_maps(
    output_dir: Path,
    ordered_images,
    cameras,
    points3d,
    image_names: list[str],
    dense_stats: dict,
    mode: str,
    min_conf_thr: float,
):
    dense_root = output_dir / "dense_geometry"
    points_root = dense_root / "points3d"
    confidence_root = dense_root / "confidence"
    depth_root = dense_root / "depth"
    points_root.mkdir(parents=True, exist_ok=True)
    confidence_root.mkdir(parents=True, exist_ok=True)
    depth_root.mkdir(parents=True, exist_ok=True)

    if mode == "sparse":
        # Sparse-first VGGT/COLMAP exports write real per-view correspondence
        # archives after atlas construction. Dense maps are intentionally absent.
        for stale_root in (points_root, confidence_root, depth_root):
            for stale_path in stale_root.glob("*.npy"):
                stale_path.unlink()
        save_json(dense_stats, dense_root / "dense_views_stats.json")
        return

    if mode != "full":
        raise ValueError(f"Unknown correspondence map mode: {mode}")

    num_views = len(ordered_images)
    for image in ordered_images:
        camera = cameras[image.camera_id]
        height = int(camera.height)
        width = int(camera.width)
        stem = Path(image.name).stem

        points_path = points_root / f"{stem}.npy"
        confidence_path = confidence_root / f"{stem}.npy"
        depth_path = depth_root / f"{stem}.npy"

        points_map = np.lib.format.open_memmap(points_path, mode="w+", dtype=np.float32, shape=(height, width, 3))
        confidence_map = np.lib.format.open_memmap(confidence_path, mode="w+", dtype=np.float32, shape=(height, width))
        depth_map = np.lib.format.open_memmap(depth_path, mode="w+", dtype=np.float32, shape=(height, width))
        points_map[:] = np.nan
        confidence_map[:] = np.nan
        depth_map[:] = np.nan

        rotation = qvec2rotmat(np.asarray(image.qvec, dtype=np.float64))
        translation = np.asarray(image.tvec, dtype=np.float64).reshape(3)
        for xy, point_id in zip(image.xys, image.point3D_ids):
            point_id = int(point_id)
            if point_id < 0 or point_id not in points3d:
                continue
            x = int(round(float(xy[0])))
            y = int(round(float(xy[1])))
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            point = points3d[point_id]
            conf = confidence_from_track(point, num_views)
            if np.isfinite(confidence_map[y, x]) and confidence_map[y, x] >= conf:
                continue
            xyz = np.asarray(point.xyz, dtype=np.float32).reshape(3)
            depth = float((rotation @ xyz.astype(np.float64) + translation)[2])
            points_map[y, x] = xyz
            confidence_map[y, x] = conf
            depth_map[y, x] = depth

        finite = np.isfinite(confidence_map)
        conf_values = np.asarray(confidence_map[finite], dtype=np.float32)
        dense_stats[image.name].update(
            {
                "mean_confidence": float(conf_values.mean()) if conf_values.size else 0.0,
                "median_confidence": float(np.median(conf_values)) if conf_values.size else 0.0,
                "finite_points": int(conf_values.size),
                "points_above_conf_thr": int(np.sum(conf_values >= float(min_conf_thr))) if conf_values.size else 0,
            }
        )

        del points_map
        del confidence_map
        del depth_map

    save_json(dense_stats, dense_root / "dense_views_stats.json")


def _chunked_nearest_indices(query_points: np.ndarray, ref_points: np.ndarray, chunk_size: int = 2048):
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 3)
    ref_points = np.asarray(ref_points, dtype=np.float32).reshape(-1, 3)
    nearest_ids = np.full((query_points.shape[0],), -1, dtype=np.int64)
    nearest_dist = np.full((query_points.shape[0],), np.inf, dtype=np.float32)
    if query_points.shape[0] == 0 or ref_points.shape[0] == 0:
        return nearest_ids, nearest_dist

    chunk_size = max(int(chunk_size), 1)
    for start in range(0, query_points.shape[0], chunk_size):
        end = min(start + chunk_size, query_points.shape[0])
        chunk = query_points[start:end]
        d2 = np.sum((chunk[:, None, :] - ref_points[None, :, :]) ** 2, axis=2)
        local_ids = np.argmin(d2, axis=1)
        nearest_ids[start:end] = local_ids.astype(np.int64)
        nearest_dist[start:end] = np.sqrt(d2[np.arange(local_ids.shape[0]), local_ids].clip(min=1e-12)).astype(np.float32)
    return nearest_ids, nearest_dist


def _chunked_topk_indices(query_points: np.ndarray, ref_points: np.ndarray, k: int = 4, chunk_size: int = 1024):
    query_points = np.asarray(query_points, dtype=np.float32).reshape(-1, 3)
    ref_points = np.asarray(ref_points, dtype=np.float32).reshape(-1, 3)
    k = max(1, min(int(k), ref_points.shape[0]))
    top_ids = np.full((query_points.shape[0], k), -1, dtype=np.int64)
    top_dist = np.full((query_points.shape[0], k), np.inf, dtype=np.float32)
    if query_points.shape[0] == 0 or ref_points.shape[0] == 0:
        return top_ids, top_dist

    chunk_size = max(int(chunk_size), 1)
    for start in range(0, query_points.shape[0], chunk_size):
        end = min(start + chunk_size, query_points.shape[0])
        chunk = query_points[start:end]
        d2 = np.sum((chunk[:, None, :] - ref_points[None, :, :]) ** 2, axis=2)
        if ref_points.shape[0] == k:
            local_ids = np.argsort(d2, axis=1)[:, :k]
        else:
            local_ids = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
            local_order = np.argsort(np.take_along_axis(d2, local_ids, axis=1), axis=1)
            local_ids = np.take_along_axis(local_ids, local_order, axis=1)
        local_d2 = np.take_along_axis(d2, local_ids, axis=1)
        top_ids[start:end] = local_ids.astype(np.int64)
        top_dist[start:end] = np.sqrt(local_d2.clip(min=1e-12)).astype(np.float32)
    return top_ids, top_dist


def _normalize_confidence_for_sparse(
    confidences: np.ndarray,
    image_ids: np.ndarray,
    parallax_scores: np.ndarray | None = None,
    depth_consistency: np.ndarray | None = None,
):
    confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    image_ids = np.asarray(image_ids, dtype=np.int32).reshape(-1)
    if confidences.size == 0:
        return confidences

    log_conf = np.log1p(np.clip(confidences, 0.0, None))

    def norm(values, low_q, high_q):
        low = float(np.quantile(values, low_q))
        high = float(np.quantile(values, high_q))
        if not np.isfinite(low):
            low = 0.0
        if not np.isfinite(high) or high <= low:
            return np.ones_like(values, dtype=np.float32)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    global_score = norm(log_conf, 0.05, 0.95)
    per_view = np.zeros_like(global_score, dtype=np.float32)
    for view_id in np.unique(image_ids):
        mask = image_ids == view_id
        per_view[mask] = norm(log_conf[mask], 0.10, 0.90)
    score = np.clip(0.65 * global_score + 0.35 * per_view, 0.0, 1.0).astype(np.float32)
    if parallax_scores is not None and depth_consistency is not None:
        parallax_scores = np.asarray(parallax_scores, dtype=np.float32).reshape(-1)
        depth_consistency = np.asarray(depth_consistency, dtype=np.float32).reshape(-1)
        if parallax_scores.shape[0] == confidences.shape[0] and depth_consistency.shape[0] == confidences.shape[0]:
            score = np.clip(
                0.55 * score
                + 0.25 * np.clip(parallax_scores, 0.0, 1.0)
                + 0.20 * np.clip(depth_consistency, 0.0, 1.0),
                0.0,
                1.0,
            ).astype(np.float32)
    return score


def make_sparse_atlas_correspondences(
    output_dir: Path,
    atlas,
    dense_geometry: dict,
    image_names: list[str],
    dense_stats: dict,
    assignment_radius_mult: float = 2.75,
):
    sparse_root = output_dir / "sparse_correspondences"
    sparse_root.mkdir(parents=True, exist_ok=True)

    points = np.asarray(dense_geometry.get("points", []), dtype=np.float32).reshape(-1, 3)
    pixels_xy = np.asarray(dense_geometry.get("pixels_xy", []), dtype=np.float32).reshape(-1, 2)
    image_ids = np.asarray(dense_geometry.get("image_ids", []), dtype=np.int32).reshape(-1)
    confidences = np.asarray(dense_geometry.get("confidences", []), dtype=np.float32).reshape(-1)
    depths = np.asarray(dense_geometry.get("depths", []), dtype=np.float32).reshape(-1)
    point_errors = np.asarray(dense_geometry.get("point_errors", []), dtype=np.float32).reshape(-1)
    track_lengths = np.asarray(dense_geometry.get("track_lengths", []), dtype=np.int32).reshape(-1)
    point3d_ids = np.asarray(dense_geometry.get("point3d_ids", []), dtype=np.int64).reshape(-1)
    parallax_scores = np.asarray(dense_geometry.get("parallax_score", np.ones_like(confidences)), dtype=np.float32).reshape(-1)
    depth_consistency = np.asarray(dense_geometry.get("depth_consistency", np.ones_like(confidences)), dtype=np.float32).reshape(-1)

    count = points.shape[0]
    required_lengths = (pixels_xy.shape[0], image_ids.shape[0], confidences.shape[0], depths.shape[0], point_errors.shape[0], track_lengths.shape[0])
    if count == 0 or any(length != count for length in required_lengths):
        raise RuntimeError("Sparse correspondence export requires per-observation points, pixels, confidences, depths, errors, and track lengths.")
    if parallax_scores.shape[0] != count:
        parallax_scores = np.ones((count,), dtype=np.float32)
    if depth_consistency.shape[0] != count:
        depth_consistency = np.ones((count,), dtype=np.float32)

    atlas_positions = np.asarray(atlas.positions, dtype=np.float32).reshape(-1, 3)
    atlas_radius = np.asarray(atlas.radius, dtype=np.float32).reshape(-1)
    atlas_reliability = np.asarray(atlas.reliability, dtype=np.float32).reshape(-1)
    atlas_class = np.asarray(atlas.atlas_class, dtype=np.int32).reshape(-1)
    atlas_support = np.asarray(atlas.support, dtype=np.float32).reshape(-1, 3, 3)
    atlas_point_support = np.asarray(getattr(atlas, "point_support", np.ones((atlas_positions.shape[0],), dtype=np.int32)), dtype=np.int32).reshape(-1)
    atlas_view_support = np.asarray(getattr(atlas, "view_support", np.ones((atlas_positions.shape[0],), dtype=np.int32)), dtype=np.int32).reshape(-1)
    atlas_support_score = np.asarray(getattr(atlas, "support_score", np.ones((atlas_positions.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if atlas_positions.shape[0] == 0:
        raise RuntimeError("Cannot export atlas correspondences for an empty atlas.")

    finite_radius = atlas_radius[np.isfinite(atlas_radius) & (atlas_radius > 0.0)]
    base_radius = max(float(np.median(finite_radius)) if finite_radius.size else 1e-3, 1e-6)
    top_ids, top_dist = _chunked_topk_indices(points, atlas_positions, k=4)
    safe_top_ids = np.clip(top_ids, 0, atlas_positions.shape[0] - 1)
    candidate_radius = np.maximum(atlas_radius[safe_top_ids], base_radius * 0.35)
    accept_radius = np.maximum(candidate_radius * float(assignment_radius_mult), base_radius)

    confidence_score = _normalize_confidence_for_sparse(
        confidences,
        image_ids,
        parallax_scores=parallax_scores,
        depth_consistency=depth_consistency,
    )
    distance_score = np.exp(-top_dist / np.clip(candidate_radius * 1.5, 1e-6, None)).astype(np.float32)
    reliability_score = atlas_reliability[safe_top_ids]
    error_score = np.where(
        np.isfinite(point_errors) & (point_errors >= 0.0),
        1.0 / (1.0 + np.maximum(point_errors, 0.0)),
        1.0,
    ).astype(np.float32)
    track_score = np.clip(track_lengths.astype(np.float32) / max(float(len(image_names)), 1.0), 0.0, 1.0)
    base_trust = np.clip(
        confidence_score[:, None]
        * (0.35 + 0.65 * distance_score)
        * (0.45 + 0.55 * reliability_score)
        * (0.55 + 0.45 * error_score[:, None])
        * (0.45 + 0.55 * track_score[:, None]),
        0.0,
        1.0,
    ).astype(np.float32)
    candidate_offsets = points[:, None, :] - atlas_positions[safe_top_ids]
    support_projected = np.einsum("nkd,nkde->nke", candidate_offsets, atlas_support[safe_top_ids])
    support_residual = np.linalg.norm(candidate_offsets - support_projected, axis=2).astype(np.float32)
    support_agreement_candidates = np.exp(-support_residual / np.clip(candidate_radius, 1e-6, None)).astype(np.float32)
    class_bonus_candidates = np.where(atlas_class[safe_top_ids] == 2, 0.75, 1.0).astype(np.float32)
    finite_source = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(pixels_xy).all(axis=1)
        & np.isfinite(confidences)
        & (image_ids >= 0)
        & (image_ids < len(image_names))
    )
    valid_candidates = (
        (top_ids >= 0)
        & np.isfinite(top_dist)
        & (top_dist <= accept_radius)
        & finite_source[:, None]
    )
    match_score_candidates = (
        base_trust
        * (0.65 + 0.35 * support_agreement_candidates)
        * class_bonus_candidates
    ).astype(np.float32)
    match_score_candidates = np.where(valid_candidates, match_score_candidates, -np.inf).astype(np.float32)
    best_cols = np.argmax(match_score_candidates, axis=1)
    rows = np.arange(count)
    best_match_score = match_score_candidates[rows, best_cols]
    nearest_ids = top_ids[rows, best_cols].astype(np.int64)
    nearest_dist = top_dist[rows, best_cols].astype(np.float32)
    local_radius = candidate_radius[rows, best_cols].astype(np.float32)
    support_agreement = support_agreement_candidates[rows, best_cols].astype(np.float32)
    trust = np.clip(best_match_score, 0.0, 1.0).astype(np.float32)
    accepted = (
        np.isfinite(best_match_score)
        & (best_match_score > -np.inf)
        & (nearest_ids >= 0)
        & np.isfinite(nearest_dist)
        & finite_source
    )
    node_view_weight = np.zeros((atlas_positions.shape[0], len(image_names)), dtype=np.float32)
    accepted_node_ids = nearest_ids[accepted]
    accepted_view_ids = image_ids[accepted]
    if accepted_node_ids.size > 0:
        np.add.at(node_view_weight, (accepted_node_ids, accepted_view_ids), trust[accepted].astype(np.float32))
    node_weight_total = node_view_weight.sum(axis=1).clip(min=1e-6)
    ref_view_consistency = np.zeros((count,), dtype=np.float32)
    valid_ref_rows = accepted & (nearest_ids >= 0) & (image_ids >= 0) & (image_ids < len(image_names))
    ref_view_consistency[valid_ref_rows] = (
        node_view_weight[nearest_ids[valid_ref_rows], image_ids[valid_ref_rows]]
        / node_weight_total[nearest_ids[valid_ref_rows]]
    ).astype(np.float32)
    pose_error = (
        0.35
        + 1.75 * (1.0 - trust)
        + np.clip(point_errors, 0.0, 16.0) * 0.12
        + np.clip(nearest_dist / np.clip(local_radius, 1e-6, None), 0.0, 4.0) * 0.18
    ).astype(np.float32)

    view_counts = {}
    total_written = 0
    for view_id, image_name in enumerate(image_names):
        image_name = str(image_name)
        stem = Path(image_name).stem
        mask = accepted & (image_ids == int(view_id))
        view_count = int(mask.sum())
        view_counts[image_name] = view_count
        stats = dense_stats.setdefault(image_name, {})
        stats["sparse_correspondence_count"] = view_count
        stats["sparse_correspondence_mode"] = "atlas_node_sparse"
        if view_count == 0:
            continue

        node_ids = nearest_ids[mask].astype(np.int64)
        source_xyz = points[mask].astype(np.float32)
        node_xyz = atlas_positions[node_ids].astype(np.float32)
        np.savez_compressed(
            sparse_root / f"{stem}.npz",
            schema_version=np.asarray([1], dtype=np.int32),
            source_type=np.asarray(["vggt_colmap_sparse_atlas_node"], dtype=np.str_),
            view_id=np.full((view_count,), int(view_id), dtype=np.int32),
            image_name=np.asarray([image_name], dtype=np.str_),
            atlas_node_id=node_ids,
            xy=pixels_xy[mask].astype(np.float32),
            xyz=node_xyz,
            source_xyz=source_xyz,
            confidence=confidences[mask].astype(np.float32),
            trust=trust[mask].astype(np.float32),
            atlas_match_score=best_match_score[mask].astype(np.float32),
            support_agreement=support_agreement[mask].astype(np.float32),
            pose_error=pose_error[mask].astype(np.float32),
            depth=depths[mask].astype(np.float32),
            parallax_score=parallax_scores[mask].astype(np.float32),
            depth_consistency=depth_consistency[mask].astype(np.float32),
            track_length=track_lengths[mask].astype(np.int32),
            reprojection_error=point_errors[mask].astype(np.float32),
            point3d_id=point3d_ids[mask].astype(np.int64),
            atlas_reliability=atlas_reliability[node_ids].astype(np.float32),
            atlas_radius=atlas_radius[node_ids].astype(np.float32),
            atlas_class=atlas_class[node_ids].astype(np.int32),
            atlas_point_support=atlas_point_support[node_ids].astype(np.int32),
            atlas_view_support=atlas_view_support[node_ids].astype(np.int32),
            atlas_support_score=atlas_support_score[node_ids].astype(np.float32),
            reference_view_consistency=ref_view_consistency[mask].astype(np.float32),
            node_distance=nearest_dist[mask].astype(np.float32),
        )
        total_written += view_count

    save_json(dense_stats, output_dir / "dense_geometry" / "dense_views_stats.json")
    return {
        "sparse_root": str(sparse_root),
        "total_sparse_correspondences": int(total_written),
        "view_counts": view_counts,
    }


def save_build_config(
    args,
    output_dir: Path,
    sparse_dir: Path,
    observation_count: int,
    point_count: int,
    source_stamp: dict,
    source_stamp_validation: list[str] | None = None,
):
    payload = vars(args).copy()
    payload.update(
        {
            "schema_version": 1,
            "builder": "build_vggt_colmap_atlas.py",
            "atlas_input_mode": "vggt_colmap_sparse_tracks",
            "dense_replay_used": False,
            "source_sparse_dir": str(sparse_dir),
            "observation_count": int(observation_count),
            "point3d_count": int(point_count),
            "source_stamp": source_stamp,
            "source_stamp_validation": source_stamp_validation or [],
            "note": "Atlas built from VGGT COLMAP tracks, not MASt3R dense replay.",
        }
    )
    save_json(payload, output_dir / "build_config.json")
    save_json(source_stamp, output_dir / "builder_source_stamp.json")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Build Foundation Atlas artifacts from a VGGT COLMAP export.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--atlas_max_nodes", type=int, default=8192)
    parser.add_argument("--atlas_k", type=int, default=16)
    parser.add_argument("--atlas_surface_ratio", type=float, default=0.12)
    parser.add_argument("--atlas_edge_ratio", type=float, default=0.18)
    parser.add_argument("--atlas_reliability_alpha", type=float, default=1.0)
    parser.add_argument("--atlas_reliability_gamma", type=float, default=2.0)
    parser.add_argument("--atlas_reliability_min", type=float, default=0.05)
    parser.add_argument("--atlas_candidate_oversample", type=float, default=3.0)
    parser.add_argument("--atlas_target_voxel_size", type=float, default=0.0)
    parser.add_argument("--atlas_edge_quota_fraction", type=float, default=0.05)
    parser.add_argument("--atlas_unstable_quota_fraction", type=float, default=0.05)
    parser.add_argument("--atlas_min_point_support", type=int, default=4)
    parser.add_argument("--atlas_min_view_support", type=int, default=2)
    parser.add_argument("--atlas_voxel_support_consistency_min", type=float, default=0.18)
    parser.add_argument(
        "--disable_self_calibration",
        action="store_true",
        help="Ablation only: build atlas from single-scale geometry without builder-side self-calibration.",
    )
    parser.add_argument("--min_conf_thr", type=float, default=0.0)
    parser.add_argument(
        "--correspondence_maps",
        choices=("sparse", "full"),
        default="sparse",
        help=(
            "sparse writes real atlas-node sparse correspondences from VGGT/COLMAP tracks; "
            "full also writes original-resolution sparse VGGT/COLMAP dense-style maps."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument(
        "--expected_source_stamp",
        type=str,
        default="",
        help="Optional JSON manifest/stamp with expected builder and foundation_atlas_module SHA256 values.",
    )
    parser.add_argument(
        "--enforce_source_stamp",
        action="store_true",
        help="Fail the build if source stamp validation reports a mismatch.",
    )
    parser.add_argument(
        "--allow_external_foundation_atlas",
        action="store_true",
        help="Allow importing mast3r.foundation_atlas from outside the expected workspace mast3r/mast3r module root.",
    )
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve() if args.image_dir else dataset_root / "images"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else dataset_root / "foundation_atlas"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_stamp = build_source_stamp()
    source_stamp_validation = validate_source_stamp(
        source_stamp,
        expected_source_stamp=args.expected_source_stamp,
        enforce=bool(args.enforce_source_stamp),
        allow_external_foundation_atlas=bool(args.allow_external_foundation_atlas),
    )
    print_source_stamp(source_stamp, source_stamp_validation)

    sparse_dir = find_sparse_dir(dataset_root)
    sparse_zero = ensure_sparse_zero(dataset_root, sparse_dir)
    cameras, images, points3d = load_colmap_sparse(sparse_zero)
    (
        ordered_images,
        image_names,
        image_paths,
        cams2world,
        intrinsics,
        image_id_to_view,
        dense_stats,
    ) = build_camera_payload(cameras, images, image_dir)

    dense_geometry = make_observation_cloud(points3d, images, cameras, image_id_to_view, num_views=len(ordered_images))
    make_dense_geometry_maps(
        output_dir,
        ordered_images,
        cameras,
        points3d,
        image_names,
        dense_stats,
        mode=args.correspondence_maps,
        min_conf_thr=args.min_conf_thr,
    )

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Sparse model: {sparse_zero}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Cameras: {len(image_names)}")
    print(f"[INFO] 3D points: {len(points3d)}")
    print(f"[INFO] Track observations for atlas: {dense_geometry['points'].shape[0]}")

    atlas = build_foundation_geometry_atlas(
        dense_geometry["points"],
        dense_geometry["colors"],
        dense_geometry["confidences"],
        image_ids=dense_geometry["image_ids"],
        parallax_scores=dense_geometry.get("parallax_score"),
        depth_consistency=dense_geometry.get("depth_consistency"),
        num_views=len(image_names),
        max_nodes=args.atlas_max_nodes,
        k_neighbors=args.atlas_k,
        surface_ratio=args.atlas_surface_ratio,
        edge_ratio=args.atlas_edge_ratio,
        reliability_alpha=args.atlas_reliability_alpha,
        reliability_gamma=args.atlas_reliability_gamma,
        reliability_min=args.atlas_reliability_min,
        candidate_oversample=args.atlas_candidate_oversample,
        target_voxel_size=args.atlas_target_voxel_size,
        edge_quota_fraction=args.atlas_edge_quota_fraction,
        unstable_quota_fraction=args.atlas_unstable_quota_fraction,
        min_point_support=args.atlas_min_point_support,
        min_view_support=args.atlas_min_view_support,
        voxel_support_consistency_min=args.atlas_voxel_support_consistency_min,
        enable_self_calibration=not bool(args.disable_self_calibration),
        device=args.device,
        seed=args.seed,
    )
    sparse_correspondence = make_sparse_atlas_correspondences(
        output_dir,
        atlas,
        dense_geometry,
        image_names,
        dense_stats,
    )

    sidecars = save_foundation_atlas_sidecars(
        output_dir,
        atlas,
        image_names,
        cams2world,
        intrinsics,
        dense_geometry,
        dense_stats=dense_stats,
        image_paths=image_paths,
        min_conf_thr=args.min_conf_thr,
        sparse_correspondence_dir=Path(sparse_correspondence["sparse_root"]),
        scene_alignment={
            "schema_version": 1,
            "applied": False,
            "reason": "vggt_colmap_already_in_gaussian_scene_space",
            "contract_validation": {"schema_version": 1, "passed": True, "failures": []},
        },
        preprocess_image_size=None,
    )

    summary = summarize_foundation_atlas(
        atlas,
        num_input_points=dense_geometry["points"].shape[0],
        min_conf_thr=args.min_conf_thr,
    )
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "image_dir": str(image_dir),
            "sparse_dir": str(sparse_zero),
            "camera_count": int(len(image_names)),
            "point3d_count": int(len(points3d)),
            "observation_count": int(dense_geometry["points"].shape[0]),
            "sparse_correspondence_count": int(sparse_correspondence["total_sparse_correspondences"]),
            "source_foundation_model": "VGGT",
            "atlas_input_mode": "vggt_colmap_sparse_tracks",
            "dense_replay_used": False,
            "self_calibration_enabled": not bool(args.disable_self_calibration),
            "export_sidecars": sidecars,
        }
    )

    save_atlas_npz(atlas, output_dir / "atlas_nodes.npz")
    save_ply(output_dir / "atlas_nodes_debug.ply", atlas.positions, atlas_debug_colors(atlas))
    save_json(summary, output_dir / "atlas_summary.json")
    save_build_config(
        args,
        output_dir,
        sparse_zero,
        dense_geometry["points"].shape[0],
        len(points3d),
        source_stamp=source_stamp,
        source_stamp_validation=source_stamp_validation,
    )

    if not args.skip_plots:
        plot_foundation_atlas_report(atlas, output_dir, summary=summary, title=dataset_root.name)

    print("[DONE] Foundation atlas exported.")
    print(f"[DONE] Atlas nodes   -> {output_dir / 'atlas_nodes.npz'}")
    print(f"[DONE] Atlas summary -> {output_dir / 'atlas_summary.json'}")
    print(f"[DONE] Use in train  -> --atlas_path \"{output_dir}\"")


if __name__ == "__main__":
    main()
