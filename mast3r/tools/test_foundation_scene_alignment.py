import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AURORA.mast3r.mast3r.foundation_atlas import (  # noqa: E402
    FoundationAtlas,
    _rotation_matrix_to_qvec,
    apply_similarity_to_cams2world,
    apply_similarity_to_dense_views,
    apply_similarity_to_foundation_atlas,
    audit_dense_correspondence_alignment,
    fit_scene_alignment_from_camera_bundles,
    load_colmap_camera_bundle,
    validate_scene_alignment_contract,
)


def _rotation_z(degrees: float):
    theta = math.radians(float(degrees))
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _build_source_cams():
    image_names = ["cam_0.png", "cam_1.png", "cam_2.png"]
    cams2world = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], len(image_names), axis=0)
    cams2world[0, :3, :3] = _rotation_z(0.0)
    cams2world[1, :3, :3] = _rotation_z(12.0)
    cams2world[2, :3, :3] = _rotation_z(-8.0)
    cams2world[0, :3, 3] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cams2world[1, :3, 3] = np.array([0.8, -0.1, 0.2], dtype=np.float32)
    cams2world[2, :3, 3] = np.array([-0.3, 0.7, 0.4], dtype=np.float32)
    return image_names, cams2world


def _write_colmap_text_model(model_dir: Path, image_names, cams2world):
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "cameras.txt", "w", encoding="utf-8") as handle:
        handle.write("# Camera list\n")
        handle.write("1 PINHOLE 32 24 18 18 16 12\n")
    with open(model_dir / "images.txt", "w", encoding="utf-8") as handle:
        handle.write("# Image list\n")
        for image_id, (image_name, camera_to_world) in enumerate(zip(image_names, cams2world), start=1):
            rotation_c2w = camera_to_world[:3, :3].astype(np.float64)
            center = camera_to_world[:3, 3].astype(np.float64)
            rotation_w2c = rotation_c2w.T
            translation = -rotation_w2c @ center
            qvec = _rotation_matrix_to_qvec(rotation_w2c)
            handle.write(
                f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} "
                f"{translation[0]} {translation[1]} {translation[2]} 1 {image_name}\n"
            )
            handle.write("\n")


def main():
    tmp_root = REPO_ROOT / ".tmp_foundation_scene_alignment"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    try:
        image_names, source_cams2world = _build_source_cams()
        known_scale = 1.35
        known_rotation = _rotation_z(18.0)
        known_translation = np.array([0.25, -0.15, 0.4], dtype=np.float32)
        target_cams2world = apply_similarity_to_cams2world(
            source_cams2world,
            known_scale,
            known_rotation,
            known_translation,
        )

        sparse_root = tmp_root / "scene" / "sparse" / "0"
        _write_colmap_text_model(sparse_root, image_names, target_cams2world)
        loaded_colmap = load_colmap_camera_bundle(tmp_root / "scene")
        assert loaded_colmap["image_names"] == image_names
        assert np.allclose(loaded_colmap["cams2world"], target_cams2world, atol=1e-5)

        alignment = fit_scene_alignment_from_camera_bundles(
            image_names,
            source_cams2world,
            loaded_colmap["image_names"],
            loaded_colmap["cams2world"],
            target_source_path=loaded_colmap["source_path"],
        )
        recovered_rotation = np.asarray(alignment["rotation"], dtype=np.float32)
        recovered_translation = np.asarray(alignment["translation"], dtype=np.float32)
        recovered_scale = float(alignment["scale"])
        assert alignment["matched_view_count"] == 3
        assert abs(recovered_scale - known_scale) < 1e-5
        assert np.allclose(recovered_rotation, known_rotation, atol=1e-5)
        assert np.allclose(recovered_translation, known_translation, atol=1e-5)
        assert alignment["camera_center_rmse"] < 1e-5
        assert alignment["rotation_error_deg_mean"] < 1e-4

        dense_views = np.array(
            [
                [
                    [[0.1, 0.0, 2.0], [0.2, 0.1, 2.1]],
                    [[0.0, -0.1, 1.9], [0.3, 0.2, 2.2]],
                ],
                [
                    [[0.6, -0.2, 2.0], [0.7, -0.1, 2.1]],
                    [[0.5, -0.3, 1.8], [0.8, 0.0, 2.2]],
                ],
                [
                    [[-0.2, 0.5, 2.3], [-0.1, 0.4, 2.0]],
                    [[-0.3, 0.6, 2.4], [0.0, 0.3, 1.9]],
                ],
            ],
            dtype=np.float32,
        )
        aligned_dense_views = apply_similarity_to_dense_views(
            dense_views,
            recovered_scale,
            recovered_rotation,
            recovered_translation,
        )
        expected_dense_views = apply_similarity_to_dense_views(
            dense_views,
            known_scale,
            known_rotation,
            known_translation,
        )
        assert np.allclose(aligned_dense_views, expected_dense_views, atol=1e-5)

        atlas = FoundationAtlas(
            positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.3]], dtype=np.float32),
            support=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
            basis=np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 2, axis=0),
            normal=np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            radius=np.array([0.2, 0.3], dtype=np.float32),
            raw_score=np.array([0.8, 0.6], dtype=np.float32),
            reliability=np.array([0.7, 0.5], dtype=np.float32),
            atlas_class=np.array([0, 1], dtype=np.int64),
            anisotropy_ref=np.array([[1.2, 1.0], [1.4, 1.1]], dtype=np.float32),
            neighbor_indices=np.array([[0, 1], [1, 0]], dtype=np.int64),
            calibration_residual=np.array([0.1, 0.2], dtype=np.float32),
            node_confidence=np.array([0.9, 0.8], dtype=np.float32),
            point_support=np.array([8, 12], dtype=np.int32),
            view_support=np.array([3, 3], dtype=np.int32),
            view_coverage=np.array([0.75, 0.75], dtype=np.float32),
            support_score=np.array([0.6, 0.7], dtype=np.float32),
            linearness=np.array([0.2, 0.4], dtype=np.float32),
            planarness=np.array([0.7, 0.3], dtype=np.float32),
            scattering=np.array([0.1, 0.3], dtype=np.float32),
        )
        aligned_atlas = apply_similarity_to_foundation_atlas(
            atlas,
            recovered_scale,
            recovered_rotation,
            recovered_translation,
        )
        expected_positions = apply_similarity_to_dense_views(
            atlas.positions,
            known_scale,
            known_rotation,
            known_translation,
        )
        assert np.allclose(aligned_atlas.positions, expected_positions, atol=1e-5)
        assert np.allclose(aligned_atlas.radius, atlas.radius * known_scale, atol=1e-6)
        assert np.allclose(aligned_atlas.basis, np.einsum("ij,njk->nik", known_rotation, atlas.basis), atol=1e-5)

        dense_height = 24
        dense_width = 32
        dense_views = []
        confidence_views = []
        u = np.arange(dense_width, dtype=np.float32)[None, :].repeat(dense_height, axis=0)
        v = np.arange(dense_height, dtype=np.float32)[:, None].repeat(dense_width, axis=1)
        for camera_to_world, intrinsics in zip(target_cams2world, loaded_colmap["intrinsics"]):
            z = np.full((dense_height, dense_width), 2.0, dtype=np.float32)
            x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
            y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
            camera_points = np.stack((x, y, z), axis=2).reshape(-1, 3)
            rotation = camera_to_world[:3, :3]
            center = camera_to_world[:3, 3]
            world_points = (rotation @ camera_points.T).T + center[None, :]
            dense_views.append(world_points.reshape(dense_height, dense_width, 3).astype(np.float32))
            confidence_views.append(np.full((dense_height, dense_width), 2.0, dtype=np.float32))

        flattened_dense_views = [view.reshape(-1, 3) for view in dense_views]
        audit = audit_dense_correspondence_alignment(
            image_names,
            flattened_dense_views,
            confidence_views,
            loaded_colmap["image_names"],
            loaded_colmap["cams2world"],
            loaded_colmap["intrinsics"],
            min_conf_thr=1.0,
            max_samples_per_view=0,
            seed=11,
        )
        assert audit["matched_view_count"] == 3
        assert audit["total_sampled_corr"] == 3 * dense_height * dense_width
        assert audit["total_projected_corr"] == 3 * dense_height * dense_width
        assert audit["total_in_frame_corr"] >= int(0.96 * 3 * dense_height * dense_width)
        assert audit["median_px_error"] is not None and audit["median_px_error"] < 1e-4
        assert audit["metric_space"] == "correspondence_pixel_space"

        image_root = tmp_root / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        image_paths = []
        for image_name in image_names:
            image_path = image_root / image_name
            Image.new("RGB", (64, 48), color=(127, 127, 127)).save(image_path)
            image_paths.append(str(image_path))

        source_audit = audit_dense_correspondence_alignment(
            image_names,
            flattened_dense_views,
            confidence_views,
            loaded_colmap["image_names"],
            loaded_colmap["cams2world"],
            loaded_colmap["intrinsics"],
            min_conf_thr=1.0,
            image_paths=image_paths,
            preprocess_image_size=32,
            max_samples_per_view=0,
            seed=11,
        )
        assert source_audit["matched_view_count"] == 3
        assert source_audit["total_projected_corr"] == 3 * dense_height * dense_width
        assert source_audit["total_in_frame_corr"] >= int(0.96 * 3 * dense_height * dense_width)
        assert source_audit["median_px_error"] is not None and source_audit["median_px_error"] < 1e-4

        contract = validate_scene_alignment_contract(
            alignment,
            source_audit,
            max_camera_center_rmse=1e-4,
            max_rotation_error_deg=1e-3,
            max_median_px_error=1e-3,
            min_in_frame_corr=int(0.96 * 3 * dense_height * dense_width),
        )
        assert contract["passed"], contract["failures"]

        print(json.dumps(alignment, indent=2))
        print("[OK] Foundation scene alignment check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
