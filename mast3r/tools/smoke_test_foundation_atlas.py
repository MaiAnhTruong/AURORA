import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AURORA.mast3r.mast3r.foundation_atlas import (
    ATLAS_CLASS_EDGE,
    ATLAS_CLASS_SURFACE,
    FoundationAtlas,
    atlas_debug_colors,
    build_foundation_geometry_atlas,
    flatten_dense_geometry,
    plot_foundation_atlas_report,
    save_atlas_npz,
    save_dense_geometry_exports,
    save_foundation_atlas_sidecars,
    save_ply,
    summarize_foundation_atlas,
)


def build_synthetic_cloud(seed=42):
    rng = np.random.default_rng(seed)

    plane_x = rng.uniform(-1.0, 1.0, size=800)
    plane_y = rng.uniform(-1.0, 1.0, size=800)
    plane_z = rng.normal(0.0, 0.01, size=800)
    plane_points = np.stack([plane_x, plane_y, plane_z], axis=1).astype(np.float32)
    plane_colors = np.tile(np.array([[0.2, 0.8, 0.4]], dtype=np.float32), (plane_points.shape[0], 1))
    plane_conf = rng.uniform(2.0, 3.5, size=plane_points.shape[0]).astype(np.float32)

    line_x = rng.uniform(-1.0, 1.0, size=250)
    line_y = rng.normal(0.0, 0.015, size=250)
    line_z = rng.normal(0.6, 0.015, size=250)
    line_points = np.stack([line_x, line_y, line_z], axis=1).astype(np.float32)
    line_colors = np.tile(np.array([[0.9, 0.6, 0.2]], dtype=np.float32), (line_points.shape[0], 1))
    line_conf = rng.uniform(1.8, 3.0, size=line_points.shape[0]).astype(np.float32)

    noise_points = rng.uniform(-1.5, 1.5, size=(80, 3)).astype(np.float32)
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


def validate_atlas(atlas: FoundationAtlas):
    assert atlas.positions.ndim == 2 and atlas.positions.shape[1] == 3
    assert atlas.support.shape == (atlas.positions.shape[0], 3, 3)
    assert atlas.basis.shape == (atlas.positions.shape[0], 3, 3)
    assert atlas.normal.shape == (atlas.positions.shape[0], 3)
    assert atlas.radius.shape[0] == atlas.positions.shape[0]
    assert np.isfinite(atlas.positions).all()
    assert np.isfinite(atlas.reliability).all()
    assert (atlas.reliability >= 0.0).all() and (atlas.reliability <= 1.0).all()
    assert atlas.point_support.shape[0] == atlas.positions.shape[0]
    assert atlas.view_support.shape[0] == atlas.positions.shape[0]
    assert atlas.support_score.shape[0] == atlas.positions.shape[0]
    assert atlas.linearness.shape[0] == atlas.positions.shape[0]
    assert atlas.planarness.shape[0] == atlas.positions.shape[0]
    assert atlas.scattering.shape[0] == atlas.positions.shape[0]
    assert atlas.structure_score is not None and atlas.structure_score.shape[0] == atlas.positions.shape[0]
    assert atlas.scale_consistency is not None and atlas.scale_consistency.shape[0] == atlas.positions.shape[0]
    assert atlas.class_consistency is not None and atlas.class_consistency.shape[0] == atlas.positions.shape[0]
    assert atlas.support_consistency is not None and atlas.support_consistency.shape[0] == atlas.positions.shape[0]
    assert atlas.unstable_reason_code is not None and atlas.unstable_reason_code.shape[0] == atlas.positions.shape[0]
    assert np.sum(atlas.atlas_class == ATLAS_CLASS_SURFACE) >= 40
    assert np.sum(atlas.atlas_class == ATLAS_CLASS_EDGE) >= 5
    assert np.max(atlas.view_support) >= 2


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
    points, colors, confidences, image_ids = build_synthetic_cloud()
    atlas = build_foundation_geometry_atlas(
        points,
        colors,
        confidences,
        image_ids=image_ids,
        num_views=4,
        max_nodes=128,
        k_neighbors=12,
        device="cpu",
        seed=7,
    )
    validate_atlas(atlas)

    out_dir = REPO_ROOT / ".tmp_foundation_atlas_smoke" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_names, pts3d, depthmaps, confs, rgb_images, intrinsics, cams2world = build_synthetic_dense_views()
    dense_stats = save_dense_geometry_exports(
        out_dir / "dense_geometry",
        image_names,
        pts3d,
        depthmaps,
        confs,
        rgb_images,
        min_conf_thr=1.0,
    )
    dense_geometry = flatten_dense_geometry(pts3d, confs, rgb_images, min_conf_thr=1.0)
    sidecar_summary = save_foundation_atlas_sidecars(
        out_dir,
        atlas,
        image_names,
        cams2world,
        intrinsics,
        dense_geometry,
        dense_stats=dense_stats,
        min_conf_thr=1.0,
    )

    save_atlas_npz(atlas, out_dir / "atlas_nodes.npz")
    save_ply(out_dir / "atlas_nodes_debug.ply", atlas.positions, atlas_debug_colors(atlas))
    summary = summarize_foundation_atlas(atlas, len(points), 0.0)
    plot_foundation_atlas_report(atlas, out_dir, summary=summary, title="Synthetic Smoke Test")

    assert (out_dir / "camera_bundle.json").exists()
    assert (out_dir / "correspondence_manifest.json").exists()
    assert (out_dir / "atlas_hash.json").exists()
    assert (out_dir / "reference_camera_evidence.npz").exists()
    assert (out_dir / "atlas_unstable_audit.json").exists()
    assert sidecar_summary["correspondence_view_count"] == 4
    assert sidecar_summary["unstable_audit_count"] == summary["class_counts"]["unstable"]
    assert "unstable_reason_counts" in summary
    print(f"[OK] Foundation atlas smoke test passed. Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
