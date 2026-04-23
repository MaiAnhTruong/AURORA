import json
import shutil
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AURORA.mast3r.mast3r.foundation_atlas import (  # noqa: E402
    backfill_foundation_atlas_sidecars,
    build_foundation_geometry_atlas,
    save_atlas_npz,
    save_dense_geometry_exports,
)


def build_synthetic_cloud(seed=42):
    rng = np.random.default_rng(seed)

    plane_x = rng.uniform(-1.0, 1.0, size=480)
    plane_y = rng.uniform(-1.0, 1.0, size=480)
    plane_z = rng.normal(0.0, 0.01, size=480)
    plane_points = np.stack([plane_x, plane_y, plane_z], axis=1).astype(np.float32)
    plane_colors = np.tile(np.array([[0.2, 0.8, 0.4]], dtype=np.float32), (plane_points.shape[0], 1))
    plane_conf = rng.uniform(2.0, 3.5, size=plane_points.shape[0]).astype(np.float32)

    line_x = rng.uniform(-1.0, 1.0, size=160)
    line_y = rng.normal(0.0, 0.015, size=160)
    line_z = rng.normal(0.6, 0.015, size=160)
    line_points = np.stack([line_x, line_y, line_z], axis=1).astype(np.float32)
    line_colors = np.tile(np.array([[0.9, 0.6, 0.2]], dtype=np.float32), (line_points.shape[0], 1))
    line_conf = rng.uniform(1.8, 3.0, size=line_points.shape[0]).astype(np.float32)

    noise_points = rng.uniform(-1.5, 1.5, size=(60, 3)).astype(np.float32)
    noise_colors = np.tile(np.array([[0.9, 0.2, 0.3]], dtype=np.float32), (noise_points.shape[0], 1))
    noise_conf = rng.uniform(0.2, 0.8, size=noise_points.shape[0]).astype(np.float32)

    points = np.concatenate([plane_points, line_points, noise_points], axis=0)
    colors = np.concatenate([plane_colors, line_colors, noise_colors], axis=0)
    confidences = np.concatenate([plane_conf, line_conf, noise_conf], axis=0)
    image_ids = np.concatenate(
        [
            rng.integers(0, 4, size=plane_points.shape[0], endpoint=False),
            rng.integers(0, 4, size=line_points.shape[0], endpoint=False),
            rng.integers(0, 2, size=noise_points.shape[0], endpoint=False),
        ],
        axis=0,
    ).astype(np.int32)
    return points, colors, confidences, image_ids


def build_synthetic_dense_views(num_views=4):
    pts3d = []
    depthmaps = []
    confs = []
    rgb_images = []
    image_names = [f"cam_{index}.png" for index in range(num_views)]

    height = 8
    width = 10
    xx = np.linspace(-0.6, 0.6, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    yy = np.linspace(-0.4, 0.4, height, dtype=np.float32)[:, None].repeat(width, axis=1)

    for view_id in range(num_views):
        z = 2.0 + 0.15 * view_id + 0.05 * xx
        points = np.stack([xx + 0.12 * view_id, yy, z], axis=2).astype(np.float32)
        depth = z.astype(np.float32)
        conf = (1.8 + 0.25 * view_id + 0.2 * np.cos(xx * np.pi)).astype(np.float32)
        rgb = np.stack(
            [
                np.clip(0.25 + 0.18 * view_id + 0.15 * (xx + 0.6), 0.0, 1.0),
                np.clip(0.30 + 0.12 * (yy + 0.4), 0.0, 1.0),
                np.full((height, width), np.clip(0.45 + 0.10 * view_id, 0.0, 1.0), dtype=np.float32),
            ],
            axis=2,
        ).astype(np.float32)
        pts3d.append(points)
        depthmaps.append(depth)
        confs.append(conf)
        rgb_images.append(rgb)

    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], num_views, axis=0)
    intrinsics[:, 0, 0] = 12.0
    intrinsics[:, 1, 1] = 12.0
    intrinsics[:, 0, 2] = width * 0.5
    intrinsics[:, 1, 2] = height * 0.5

    cams2world = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], num_views, axis=0)
    cams2world[:, 0, 3] = np.linspace(0.0, 0.9, num_views, dtype=np.float32)

    return image_names, pts3d, depthmaps, confs, rgb_images, intrinsics, cams2world


def main():
    tmp_root = REPO_ROOT / ".tmp_foundation_atlas_backfill"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_dir = tmp_root / "legacy_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    try:
        points, colors, confidences, image_ids = build_synthetic_cloud()
        atlas = build_foundation_geometry_atlas(
            points,
            colors,
            confidences,
            image_ids=image_ids,
            num_views=4,
            max_nodes=96,
            k_neighbors=12,
            device="cpu",
            seed=7,
        )
        save_atlas_npz(atlas, atlas_dir / "atlas_nodes.npz")

        image_names, pts3d, depthmaps, confs, rgb_images, intrinsics, cams2world = build_synthetic_dense_views()
        save_dense_geometry_exports(
            atlas_dir / "dense_geometry",
            image_names,
            pts3d,
            depthmaps,
            confs,
            rgb_images,
            min_conf_thr=1.0,
        )
        (atlas_dir / "dense_geometry" / "dense_views_stats.json").unlink()

        with open(atlas_dir / "camera_bundle.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "image_names": image_names,
                    "cams2world": cams2world.tolist(),
                    "intrinsics": intrinsics.tolist(),
                },
                handle,
                indent=2,
            )
        with open(atlas_dir / "build_config.json", "w", encoding="utf-8") as handle:
            json.dump({"min_conf_thr": 1.0}, handle, indent=2)

        summary = backfill_foundation_atlas_sidecars(atlas_dir, update_summary=True)

        camera_bundle = json.loads((atlas_dir / "camera_bundle.json").read_text(encoding="utf-8"))
        manifest = json.loads((atlas_dir / "correspondence_manifest.json").read_text(encoding="utf-8"))
        atlas_hash = json.loads((atlas_dir / "atlas_hash.json").read_text(encoding="utf-8"))
        atlas_summary = json.loads((atlas_dir / "atlas_summary.json").read_text(encoding="utf-8"))
        with np.load(atlas_dir / "reference_camera_evidence.npz") as ref_payload:
            assert camera_bundle["schema_version"] == 1
            assert len(camera_bundle["world2cams"]) == 4
            assert len(camera_bundle["image_sizes"]) == 4
            assert manifest["schema_version"] == 1
            assert manifest["view_order"] == image_names
            assert len(manifest["views"]) == 4
            assert atlas_hash["kind"] == "voxel_hash"
            assert int(ref_payload["reference_camera_ids"].shape[0]) == int(atlas.positions.shape[0])
            assert int(ref_payload["view_weights"].shape[1]) == 4
            assert summary["outputs"]["correspondence_view_count"] == 4
            assert "export_sidecars" in atlas_summary
            assert "backfill_sidecars" in atlas_summary

        print(json.dumps(summary, indent=2))
        print("[OK] Foundation atlas backfill check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
