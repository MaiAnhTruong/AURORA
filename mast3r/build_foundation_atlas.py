import argparse
from pathlib import Path

import numpy as np
import torch

import AURORA.mast3r.mast3r.utils.path_to_dust3r  # noqa: F401
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images
from AURORA.mast3r.mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from AURORA.mast3r.mast3r.foundation_atlas import (
    apply_similarity_to_cams2world,
    apply_similarity_to_dense_views,
    audit_dense_correspondence_alignment,
    atlas_debug_colors,
    build_foundation_geometry_atlas,
    fit_scene_alignment_from_camera_bundles,
    load_colmap_camera_bundle,
    save_foundation_atlas_sidecars,
    flatten_dense_geometry,
    plot_foundation_atlas_report,
    save_atlas_npz,
    save_dense_geometry_exports,
    save_json,
    save_ply,
    scale_dense_depth_views,
    summarize_foundation_atlas,
    validate_scene_alignment_contract,
)
from AURORA.mast3r.mast3r.image_pairs import make_pairs


def collect_unique_images(image_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png"}
    filelist = []
    seen = set()

    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_exts:
            continue
        resolved = str(path.resolve())
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        filelist.append(resolved)

    return filelist


def maybe_to_numpy(value):
    if torch.is_tensor(value):
        return to_numpy(value.cpu())
    if isinstance(value, (list, tuple)):
        return np.asarray([maybe_to_numpy(item) for item in value])
    return np.asarray(value)


def main():
    parser = argparse.ArgumentParser(
        description="Build a Foundation Geometry Atlas from existing MASt3R cache outputs."
    )
    parser.add_argument("--dataset_root", type=str, default=r"D:\All for one\data\kitchen_12 copy")
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--scene_graph", type=str, default="complete")
    parser.add_argument("--subsample", type=int, default=8)
    parser.add_argument("--matching_conf_thr", type=float, default=0.0)
    parser.add_argument("--shared_intrinsics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean_depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--niter1", type=int, default=300)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--niter2", type=int, default=300)
    parser.add_argument("--min_conf_thr", type=float, default=1.0)
    parser.add_argument("--atlas_max_nodes", type=int, default=16384)
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
    parser.add_argument("--spawn_extra_surface_gaussians", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--atlas_extra_surface_children", type=int, default=2)
    parser.add_argument("--atlas_extra_surface_rel_thr", type=float, default=0.16)
    parser.add_argument("--atlas_extra_surface_radius_thr", type=float, default=0.0)
    parser.add_argument("--atlas_extra_surface_support_thr", type=float, default=0.50)
    parser.add_argument("--atlas_extra_surface_view_thr", type=float, default=0.35)
    parser.add_argument("--atlas_extra_surface_offset_scale", type=float, default=0.45)
    parser.add_argument("--preview_max_points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--align_to_colmap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--colmap_sparse_dir", type=str, default="")
    parser.add_argument("--enforce_contract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contract_max_camera_center_rmse", type=float, default=0.05)
    parser.add_argument("--contract_max_rotation_error_deg", type=float, default=1.0)
    parser.add_argument("--contract_max_median_px_error", type=float, default=12.0)
    parser.add_argument("--contract_min_in_frame_corr", type=int, default=256)
    parser.add_argument("--contract_max_samples_per_view", type=int, default=2048)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else dataset_root / "images"
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else dataset_root / "mast3r_cache"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else dataset_root / "foundation_atlas"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not cache_dir.exists():
        raise FileNotFoundError(f"MASt3R cache directory not found: {cache_dir}")
    if not (cache_dir / "forward").exists():
        raise FileNotFoundError(f"Expected cached pairwise predictions under: {cache_dir / 'forward'}")

    output_dir.mkdir(parents=True, exist_ok=True)

    filelist = collect_unique_images(image_dir)
    if len(filelist) < 2:
        raise RuntimeError(f"Need at least 2 images in {image_dir}, found {len(filelist)}")

    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Image dir: {image_dir}")
    print(f"[INFO] Cache dir: {cache_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Found {len(filelist)} images")

    print("[INFO] Loading images for cache replay...")
    imgs = load_images(filelist, size=args.image_size, verbose=True)

    print("[INFO] Building scene graph...")
    pairs = make_pairs(
        imgs,
        scene_graph=args.scene_graph,
        prefilter=None,
        symmetrize=True,
    )

    print("[INFO] Replaying sparse global alignment from cache...")
    scene = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_dir),
        model=None,
        subsample=args.subsample,
        device=args.device,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        opt_depth=True,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
    )

    print("[INFO] Exporting dense per-view geometry package...")
    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=args.clean_depth, subsample=args.subsample)
    pts3d_np = to_numpy(pts3d)
    depthmaps_np = to_numpy(depthmaps)
    confs_np = to_numpy(confs)
    rgb_images_np = to_numpy(scene.imgs)
    image_names = [Path(path).name for path in filelist]
    intrinsics = maybe_to_numpy(scene.intrinsics)
    cams2world = maybe_to_numpy(scene.get_im_poses())

    scene_alignment = {
        "schema_version": 1,
        "applied": False,
        "reason": "align_to_colmap_disabled",
    }
    colmap_bundle = None
    if args.align_to_colmap:
        colmap_bundle = load_colmap_camera_bundle(args.colmap_sparse_dir if args.colmap_sparse_dir else dataset_root)
        scene_alignment = fit_scene_alignment_from_camera_bundles(
            image_names,
            cams2world,
            colmap_bundle["image_names"],
            colmap_bundle["cams2world"],
            target_source_path=colmap_bundle["source_path"],
        )
        alignment_scale = float(scene_alignment["scale"])
        alignment_rotation = np.asarray(scene_alignment["rotation"], dtype=np.float32)
        alignment_translation = np.asarray(scene_alignment["translation"], dtype=np.float32)
        pts3d_np = apply_similarity_to_dense_views(
            pts3d_np,
            alignment_scale,
            alignment_rotation,
            alignment_translation,
        )
        depthmaps_np = scale_dense_depth_views(depthmaps_np, alignment_scale)
        cams2world = apply_similarity_to_cams2world(
            cams2world,
            alignment_scale,
            alignment_rotation,
            alignment_translation,
        )
        print(
            "[INFO] Applied MASt3R -> COLMAP scene alignment "
            f"(views={scene_alignment['matched_view_count']}, "
            f"center_rmse={scene_alignment['camera_center_rmse']:.6f}, "
            f"rot_mean_deg={scene_alignment['rotation_error_deg_mean']:.4f})."
        )
        dense_correspondence_audit = audit_dense_correspondence_alignment(
            image_names,
            pts3d_np,
            confs_np,
            colmap_bundle["image_names"],
            colmap_bundle["cams2world"],
            colmap_bundle["intrinsics"],
            min_conf_thr=args.min_conf_thr,
            image_paths=filelist,
            preprocess_image_size=args.image_size,
            max_samples_per_view=args.contract_max_samples_per_view,
            seed=args.seed,
        )
        scene_alignment["dense_correspondence_audit"] = dense_correspondence_audit
        contract_validation = validate_scene_alignment_contract(
            scene_alignment,
            dense_correspondence_audit,
            max_camera_center_rmse=args.contract_max_camera_center_rmse,
            max_rotation_error_deg=args.contract_max_rotation_error_deg,
            max_median_px_error=args.contract_max_median_px_error,
            min_in_frame_corr=args.contract_min_in_frame_corr,
        )
        scene_alignment["contract_validation"] = contract_validation
        if args.enforce_contract and not contract_validation["passed"]:
            failure_text = "; ".join(contract_validation["failures"])
            raise RuntimeError(
                "Foundation atlas contract validation failed after MASt3R -> COLMAP alignment: "
                f"{failure_text}"
            )

    dense_output_dir = output_dir / "dense_geometry"
    dense_stats = save_dense_geometry_exports(
        dense_output_dir,
        image_names,
        pts3d_np,
        depthmaps_np,
        confs_np,
        rgb_images_np,
        preview_max_points=args.preview_max_points,
        seed=args.seed,
        min_conf_thr=args.min_conf_thr,
    )

    dense_geometry = flatten_dense_geometry(
        pts3d_np,
        confs_np,
        rgb_images_np,
        min_conf_thr=args.min_conf_thr,
    )

    print("[INFO] Building Foundation Geometry Atlas...")
    atlas = build_foundation_geometry_atlas(
        dense_geometry["points"],
        dense_geometry["colors"],
        dense_geometry["confidences"],
        image_ids=dense_geometry["image_ids"],
        num_views=len(filelist),
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
        device=args.device,
        seed=args.seed,
    )

    summary = summarize_foundation_atlas(
        atlas,
        num_input_points=dense_geometry["points"].shape[0],
        min_conf_thr=args.min_conf_thr,
    )
    summary["dataset_root"] = str(dataset_root)
    summary["cache_dir"] = str(cache_dir)
    summary["image_dir"] = str(image_dir)
    summary["camera_count"] = len(filelist)
    summary["scene_alignment"] = scene_alignment
    print(
        "[INFO] Atlas summary "
        f"(mean_rel={summary['mean_reliability']:.4f}, "
        f"median_rel={summary['median_reliability']:.4f}, "
        f"mean_support_consistency={summary.get('mean_support_consistency', 0.0):.4f}, "
        f"unstable={summary['class_counts']['unstable']})."
    )
    if "unstable_reason_counts" in summary:
        print(f"[INFO] Unstable audit: {summary['unstable_reason_counts']}")

    sidecar_summary = save_foundation_atlas_sidecars(
        output_dir,
        atlas,
        image_names,
        cams2world,
        intrinsics,
        dense_geometry,
        dense_stats=dense_stats,
        image_paths=filelist,
        min_conf_thr=args.min_conf_thr,
        scene_alignment=scene_alignment,
        preprocess_image_size=args.image_size,
    )
    summary["export_sidecars"] = sidecar_summary

    save_atlas_npz(atlas, output_dir / "atlas_nodes.npz")
    save_ply(output_dir / "atlas_nodes_debug.ply", atlas.positions, atlas_debug_colors(atlas))
    save_json(summary, output_dir / "atlas_summary.json")
    save_json(vars(args), output_dir / "build_config.json")

    if not args.skip_plots:
        print("[INFO] Plotting atlas report...")
        plot_foundation_atlas_report(atlas, output_dir, summary=summary, title=dataset_root.name)

    print("[DONE] Foundation atlas exported.")
    print(f"[DONE] Atlas nodes   -> {output_dir / 'atlas_nodes.npz'}")
    print(f"[DONE] Atlas summary -> {output_dir / 'atlas_summary.json'}")
    print(f"[DONE] Atlas preview -> {output_dir / 'atlas_nodes_debug.ply'}")
    print(f"[DONE] Dense package -> {dense_output_dir}")


if __name__ == "__main__":
    main()
