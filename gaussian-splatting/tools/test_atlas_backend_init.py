import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import (  # noqa: E402
    ATLAS_CLASS_EDGE,
    ATLAS_CLASS_SURFACE,
    ATLAS_CLASS_UNSTABLE,
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
    load_foundation_atlas,
    summarize_atlas_initialization,
)
from scene.gaussian_model import GaussianModel  # noqa: E402


def write_preview_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    normals = np.zeros_like(xyz, dtype=np.float32)
    rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb.astype(np.float32)), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def build_synthetic_atlas_run(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    (root / "dense_geometry").mkdir(parents=True, exist_ok=True)
    (root / "dense_geometry" / "points3d").mkdir(parents=True, exist_ok=True)
    (root / "dense_geometry" / "confidence").mkdir(parents=True, exist_ok=True)
    (root / "dense_geometry" / "depth").mkdir(parents=True, exist_ok=True)

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    basis = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (4, 1, 1))
    support = np.tile(np.eye(3, dtype=np.float32)[None, :, :], (4, 1, 1))
    support[0, 2, 2] = 0.0
    support[1, 2, 2] = 0.0
    support[2, 1, 1] = 0.0
    support[2, 2, 2] = 0.0
    support[3] *= 0.0
    normals = basis[:, :, 2]
    radius = np.array([0.2, 0.25, 0.18, 0.22], dtype=np.float32)
    reliability = np.array([0.18, 0.07, 0.11, 0.05], dtype=np.float32)
    atlas_class = np.array(
        [ATLAS_CLASS_SURFACE, ATLAS_CLASS_SURFACE, ATLAS_CLASS_EDGE, ATLAS_CLASS_UNSTABLE],
        dtype=np.int64,
    )
    anisotropy_ref = np.array(
        [
            [1.2, 1.1],
            [1.0, 1.0],
            [1.8, 1.4],
            [0.4, 0.4],
        ],
        dtype=np.float32,
    )
    neighbor_indices = np.tile(np.arange(4, dtype=np.int64)[None, :], (4, 1))
    raw_score = np.array([0.3, 0.15, 0.14, 0.08], dtype=np.float32)
    residual = np.array([0.1, 0.28, 0.34, 0.72], dtype=np.float32)
    point_support = np.array([120, 90, 70, 12], dtype=np.int32)
    view_support = np.array([5, 4, 4, 2], dtype=np.int32)
    view_coverage = view_support.astype(np.float32) / 6.0
    support_score = np.array([0.85, 0.72, 0.69, 0.15], dtype=np.float32)
    linearness = np.array([0.12, 0.18, 0.82, 0.38], dtype=np.float32)
    planarness = np.array([0.88, 0.81, 0.10, 0.44], dtype=np.float32)
    scattering = np.array([0.02, 0.03, 0.08, 0.18], dtype=np.float32)
    node_confidence = np.array([0.9, 0.7, 0.6, 0.2], dtype=np.float32)

    np.savez_compressed(
        root / "atlas_nodes.npz",
        positions=positions,
        support=support,
        basis=basis,
        normal=normals,
        radius=radius,
        raw_score=raw_score,
        reliability=reliability,
        atlas_class=atlas_class,
        anisotropy_ref=anisotropy_ref,
        neighbor_indices=neighbor_indices,
        calibration_residual=residual,
        node_confidence=node_confidence,
        point_support=point_support,
        view_support=view_support,
        view_coverage=view_coverage,
        support_score=support_score,
        linearness=linearness,
        planarness=planarness,
        scattering=scattering,
    )

    preview_xyz = positions.copy()
    preview_rgb = np.array(
        [
            [0.8, 0.2, 0.2],
            [0.2, 0.8, 0.2],
            [0.2, 0.2, 0.8],
            [0.8, 0.8, 0.2],
        ],
        dtype=np.float32,
    )
    write_preview_ply(root / "dense_geometry" / "dense_points_preview.ply", preview_xyz, preview_rgb)
    dense_height = 6
    dense_width = 8
    dense_x = np.linspace(-0.35, 0.35, dense_width, dtype=np.float32)[None, :].repeat(dense_height, axis=0)
    dense_y = np.linspace(-0.20, 0.20, dense_height, dtype=np.float32)[:, None].repeat(dense_width, axis=1)
    base_conf = np.linspace(0.1, 4.0, dense_height * dense_width, dtype=np.float32).reshape(dense_height, dense_width)
    dense_points = np.stack((dense_x, dense_y, 2.0 + 0.15 * dense_x), axis=2).astype(np.float32)
    np.save(root / "dense_geometry" / "points3d" / "cam_0.npy", dense_points)
    np.save(root / "dense_geometry" / "points3d" / "cam_1.npy", dense_points + np.array([0.1, 0.0, 0.0], dtype=np.float32))
    np.save(root / "dense_geometry" / "confidence" / "cam_0.npy", base_conf)
    np.save(root / "dense_geometry" / "confidence" / "cam_1.npy", np.flip(base_conf, axis=1).copy())
    np.save(root / "dense_geometry" / "depth" / "cam_0.npy", np.full((dense_height, dense_width), 2.0, dtype=np.float32))
    np.save(root / "dense_geometry" / "depth" / "cam_1.npy", np.full((dense_height, dense_width), 2.1, dtype=np.float32))
    with open(root / "dense_geometry" / "dense_views_stats.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cam_0.png": {
                    "height": dense_height,
                    "width": dense_width,
                    "mean_confidence": float(base_conf.mean()),
                    "median_confidence": float(np.median(base_conf)),
                    "finite_points": int(dense_height * dense_width),
                    "points_above_conf_thr": int((base_conf >= 0.5).sum()),
                },
                "cam_1.png": {
                    "height": dense_height,
                    "width": dense_width,
                    "mean_confidence": float(base_conf.mean()),
                    "median_confidence": float(np.median(base_conf)),
                    "finite_points": int(dense_height * dense_width),
                    "points_above_conf_thr": int((base_conf >= 0.5).sum()),
                },
            },
            handle,
            indent=2,
        )
    with open(root / "camera_bundle.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "num_cameras": 2,
                "image_names": ["cam_0.png", "cam_1.png"],
                "cams2world": [
                    np.eye(4, dtype=np.float32).tolist(),
                    np.array(
                        [
                            [1.0, 0.0, 0.0, 0.5],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    ).tolist(),
                ],
                "intrinsics": [
                    np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32).tolist(),
                    np.array([[8.0, 0.0, 4.0], [0.0, 8.0, 3.0], [0.0, 0.0, 1.0]], dtype=np.float32).tolist(),
                ],
                "image_sizes": [
                    {"width": dense_width, "height": dense_height},
                    {"width": dense_width, "height": dense_height},
                ],
            },
            handle,
            indent=2,
        )
    with open(root / "correspondence_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "min_conf_thr": 0.5,
                "dense_geometry_root": "dense_geometry",
                "points_root": "dense_geometry/points3d",
                "confidence_root": "dense_geometry/confidence",
                "depth_root": "dense_geometry/depth",
                "view_order": ["cam_0.png", "cam_1.png"],
                "views": {
                    "cam_0.png": {
                        "image_name": "cam_0.png",
                        "image_index": 0,
                        "stem": "cam_0",
                        "width": dense_width,
                        "height": dense_height,
                        "mean_confidence": float(base_conf.mean()),
                        "median_confidence": float(np.median(base_conf)),
                        "finite_points": int(dense_height * dense_width),
                        "points_above_conf_thr": int((base_conf >= 0.5).sum()),
                        "points_path": "dense_geometry/points3d/cam_0.npy",
                        "confidence_path": "dense_geometry/confidence/cam_0.npy",
                        "depth_path": "dense_geometry/depth/cam_0.npy",
                    },
                    "cam_1.png": {
                        "image_name": "cam_1.png",
                        "image_index": 1,
                        "stem": "cam_1",
                        "width": dense_width,
                        "height": dense_height,
                        "mean_confidence": float(base_conf.mean()),
                        "median_confidence": float(np.median(base_conf)),
                        "finite_points": int(dense_height * dense_width),
                        "points_above_conf_thr": int((base_conf >= 0.5).sum()),
                        "points_path": "dense_geometry/points3d/cam_1.npy",
                        "confidence_path": "dense_geometry/confidence/cam_1.npy",
                        "depth_path": "dense_geometry/depth/cam_1.npy",
                    },
                },
            },
            handle,
            indent=2,
        )
    with open(root / "build_config.json", "w", encoding="utf-8") as handle:
        json.dump({"min_conf_thr": 0.5}, handle, indent=2)
    with open(root / "atlas_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_nodes": int(positions.shape[0]),
                "camera_count": 2,
                "class_counts": {
                    "surface": 2,
                    "edge": 1,
                    "unstable": 1,
                },
            },
            handle,
            indent=2,
        )
    with open(root / "atlas_hash.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "kind": "voxel_hash",
                "schema_version": 1,
                "cell_size": 0.25,
                "bbox_min": [0.0, 0.0, 0.0],
                "bbox_max": [1.5, 0.0, 0.0],
                "node_count": 4,
                "bucket_count": 4,
                "neighbor_k": 4,
                "mean_bucket_size": 1.0,
                "max_bucket_size": 1,
                "buckets": [
                    {"coord": [0, 0, 0], "node_ids": [0]},
                    {"coord": [2, 0, 0], "node_ids": [1]},
                    {"coord": [4, 0, 0], "node_ids": [2]},
                    {"coord": [6, 0, 0], "node_ids": [3]},
                ],
            },
            handle,
            indent=2,
        )
    np.savez_compressed(
        root / "reference_camera_evidence.npz",
        reference_camera_ids=np.array([0, 0, 1, 1], dtype=np.int64),
        reference_camera_scores=np.array([0.95, 0.85, 0.75, 0.65], dtype=np.float32),
        image_names=np.array(["cam_0.png", "cam_1.png"]),
        view_weights=np.array(
            [
                [4.2, 1.1],
                [3.6, 1.8],
                [0.9, 4.4],
                [0.2, 0.6],
            ],
            dtype=np.float32,
        ),
        view_counts=np.array(
            [
                [8, 2],
                [7, 3],
                [2, 8],
                [1, 2],
            ],
            dtype=np.int32,
        ),
    )
    return root


def synthetic_check():
    tmp_root = REPO_ROOT / ".tmp_atlas_backend_init"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")
    try:
        atlas_init = load_foundation_atlas(atlas_root)
        summary = summarize_atlas_initialization(atlas_init)

        assert summary["num_nodes"] == 4
        assert summary["state_counts"]["stable"] == 2
        assert summary["state_counts"]["unstable_passive"] == 2
        assert summary["has_camera_bundle"]
        assert summary["has_correspondence_manifest"]
        assert summary["correspondence_view_count"] == 2
        assert summary["reference_camera_source"] == "reference_camera_evidence.npz"
        assert summary["hash_info"]["kind"] == "voxel_hash"

        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)

        assert gm.has_atlas_bindings
        assert gm.get_xyz.shape[0] == 4
        assert torch.equal(gm.get_atlas_node_ids.cpu(), torch.arange(4, dtype=torch.long))
        expected_states = torch.tensor(
            [GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE, GAUSSIAN_STATE_STABLE, GAUSSIAN_STATE_UNSTABLE_PASSIVE],
            dtype=torch.long,
        )
        assert torch.equal(gm.get_atlas_state.cpu(), expected_states)
        assert np.allclose(gm.get_xyz.detach().cpu().numpy(), atlas_init.positions, atol=1e-6)
        assert atlas_init.calibration_residual.shape[0] == 4
        assert atlas_init.point_support.shape[0] == 4
        assert atlas_init.view_support.shape[0] == 4
        assert atlas_init.support_score.shape[0] == 4
        assert atlas_init.camera_bundle is not None
        assert atlas_init.correspondence_manifest is not None
        assert sorted(atlas_init.correspondence_manifest.views.keys()) == ["cam_0.png", "cam_1.png"]
        assert np.array_equal(atlas_init.reference_camera_ids, np.array([0, 0, 1, 1], dtype=np.int64))
        assert atlas_init.reference_view_weights.shape == (4, 2)
        assert atlas_init.reference_view_counts.shape == (4, 2)

        build_config_path = atlas_root / "build_config.json"
        with open(build_config_path, "r", encoding="utf-8") as handle:
            build_config = json.load(handle)
        build_config.update(
            {
                "spawn_extra_surface_gaussians": True,
                "atlas_extra_surface_children": 2,
                "atlas_extra_surface_rel_thr": 0.0,
                "atlas_extra_surface_radius_thr": 0.0,
                "atlas_extra_surface_support_thr": 0.0,
                "atlas_extra_surface_view_thr": 0.0,
            }
        )
        with open(build_config_path, "w", encoding="utf-8") as handle:
            json.dump(build_config, handle, indent=2)

        spawn_init = load_foundation_atlas(atlas_root)
        gm_spawn = GaussianModel(sh_degree=0)
        gm_spawn.create_from_atlas(spawn_init, cams, spatial_lr_scale=1.0)
        assert gm_spawn.get_xyz.shape[0] > spawn_init.positions.shape[0]
        assert gm_spawn.get_init_point_count() == int(gm_spawn.get_xyz.shape[0])
        spawn_summary = gm_spawn.summarize_atlas_init_metrics()
        assert spawn_summary["atlas_init_num_nodes"] == 4
        assert spawn_summary["atlas_extra_surface_spawn_count"] == 2
        assert spawn_summary["atlas_init_gaussian_count_post_spawn"] == int(gm_spawn.get_xyz.shape[0])

        state_path = atlas_root / "atlas_state_roundtrip.npz"
        gm.save_atlas_state(state_path)
        gm_roundtrip = GaussianModel(sh_degree=0)
        gm_roundtrip._xyz = torch.zeros_like(gm.get_xyz.detach())
        gm_roundtrip.load_atlas_state(state_path)
        assert torch.equal(gm_roundtrip.get_atlas_node_ids.cpu(), gm.get_atlas_node_ids.cpu())
        assert torch.equal(gm_roundtrip.get_atlas_state.cpu(), gm.get_atlas_state.cpu())
        assert torch.allclose(gm_roundtrip.get_atlas_view_weights.cpu(), gm.get_atlas_view_weights.cpu())
        assert torch.equal(gm_roundtrip.get_atlas_view_counts.cpu(), gm.get_atlas_view_counts.cpu())
        assert gm_roundtrip.summarize_atlas_init_metrics()["atlas_init_num_nodes"] == 4

        print(json.dumps(gm.summarize_atlas_bindings(), indent=2))
        print("[OK] Synthetic atlas backend init check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


def real_atlas_check(atlas_path: Path):
    atlas_init = load_foundation_atlas(atlas_path)
    print(json.dumps(summarize_atlas_initialization(atlas_init), indent=2))

    gm = GaussianModel(sh_degree=0)
    cams = [SimpleNamespace(image_name=f"cam_{idx}") for idx in range(2)]
    gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
    summary = gm.summarize_atlas_bindings()
    assert summary["num_gaussians"] >= summary["num_atlas_nodes"]
    assert gm.get_atlas_node_ids.shape[0] == gm.get_xyz.shape[0]
    print(json.dumps(summary, indent=2))
    print("[OK] Real atlas backend init check passed.")


def main():
    parser = argparse.ArgumentParser(description="Check atlas-backed Gaussian initialization.")
    parser.add_argument("--atlas_path", type=str, default="")
    args = parser.parse_args()

    if args.atlas_path:
        real_atlas_check(Path(args.atlas_path).resolve())
    else:
        synthetic_check()


if __name__ == "__main__":
    main()
