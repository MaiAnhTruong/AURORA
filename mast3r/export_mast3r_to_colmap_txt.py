import os
import tempfile
import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

import AURORA.mast3r.mast3r.utils.path_to_dust3r  # noqa: F401
from AURORA.mast3r.mast3r.model import AsymmetricMASt3R
from AURORA.mast3r.mast3r.image_pairs import make_pairs
from AURORA.mast3r.mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to COLMAP quaternion (Hamilton convention)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0.0, 0.0, 0.0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0.0, 0.0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0.0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def collect_unique_images(image_dir: Path):
    valid_exts = {".jpg", ".jpeg", ".png"}
    filelist = []
    seen = set()

    for p in sorted(image_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in valid_exts:
            continue

        rp = str(p.resolve())
        key = rp.lower()
        if key in seen:
            continue
        seen.add(key)
        filelist.append(rp)

    return filelist


def to_uint8_hwc(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return arr


def save_resized_images(scene_imgs, original_filelist, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    np_imgs = to_numpy(scene_imgs)
    saved_names = []
    sizes = []

    for img_np, orig_path in zip(np_imgs, original_filelist):
        name = Path(orig_path).name
        out_path = out_dir / name
        img_u8 = to_uint8_hwc(img_np)
        Image.fromarray(img_u8).save(out_path)
        saved_names.append(name)
        h, w = img_u8.shape[:2]
        sizes.append((w, h))

    return saved_names, sizes


def try_get_intrinsics(scene, img_sizes):
    """
    Prefer scene.get_intrinsics() if available.
    Fall back to focal + image center if not.
    """
    if hasattr(scene, "get_intrinsics"):
        try:
            K = to_numpy(scene.get_intrinsics().cpu())
            return K
        except Exception:
            pass

    focals = to_numpy(scene.get_focals().cpu())
    Ks = []
    for f, (w, h) in zip(focals, img_sizes):
        fx = float(f)
        fy = float(f)
        cx = w / 2.0
        cy = h / 2.0
        Ks.append(np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64))
    return np.stack(Ks, axis=0)


def export_cameras_txt(path: Path, intrinsics: np.ndarray, img_sizes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(img_sizes)}\n")
        for cam_id, (K, (w, h)) in enumerate(zip(intrinsics, img_sizes), start=1):
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            f.write(f"{cam_id} PINHOLE {w} {h} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")


def export_images_txt(path: Path, cams2world: np.ndarray, image_names):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_names)}, mean observations per image: 0\n")

        for image_id, (c2w, name) in enumerate(zip(cams2world, image_names), start=1):
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            q = rotmat2qvec(R)
            f.write(
                f"{image_id} "
                f"{q[0]:.12f} {q[1]:.12f} {q[2]:.12f} {q[3]:.12f} "
                f"{t[0]:.12f} {t[1]:.12f} {t[2]:.12f} "
                f"{image_id} {name}\n"
            )
            f.write("\n")


def export_points3D_txt(path: Path, pts: np.ndarray, rgb: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(pts)}, mean track length: 0\n")
        for pid, (p, c) in enumerate(zip(pts, rgb), start=1):
            r, g, b = int(c[0]), int(c[1]), int(c[2])
            f.write(
                f"{pid} "
                f"{float(p[0]):.10f} {float(p[1]):.10f} {float(p[2]):.10f} "
                f"{r} {g} {b} 0.0\n"
            )


def build_point_cloud(scene, min_conf_thr: float, clean_depth: bool, max_points: int, seed: int):
    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    rgbimg = to_numpy(scene.imgs)
    masks = to_numpy([c > min_conf_thr for c in confs])

    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, masks)], axis=0).reshape(-1, 3)
    col = np.concatenate([im[m] for im, m in zip(rgbimg, masks)], axis=0).reshape(-1, 3)

    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    col = col[valid]

    if col.dtype != np.uint8:
        if col.max() <= 1.0:
            col = (col * 255.0).clip(0, 255).astype(np.uint8)
        else:
            col = col.clip(0, 255).astype(np.uint8)

    if max_points > 0 and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        col = col[idx]

    return pts, col


from plyfile import PlyData, PlyElement

def export_ply(path: Path, pts: np.ndarray, rgb: np.ndarray):
    xyz = pts.astype(np.float32)
    rgb = rgb.astype(np.uint8)
    normals = np.zeros_like(xyz, dtype=np.float32)

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Run MASt3R-SfM and export COLMAP text files + resized images for gaussian-splatting."
    )
    parser.add_argument("--image_dir", type=str,
                        default=r"D:\All for one\data\kitchen_12 copy\images")
    parser.add_argument("--dataset_root", type=str,
                        default=r"D:\All for one\data\kitchen_12 copy")
    parser.add_argument("--output_sparse", type=str, default="",
                        help="Default: <dataset_root>/sparse/0")
    parser.add_argument("--output_images", type=str, default="",
                        help="Default: <dataset_root>/images_mast3r")
    parser.add_argument("--model_name", type=str,
                        default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--min_conf_thr", type=float, default=1.5)
    parser.add_argument("--max_points", type=int, default=200000,
                        help="Cap exported points for 3DGS init. Use <=0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--niter1", type=int, default=300)
    parser.add_argument("--lr2", type=float, default=0.01)
    parser.add_argument("--niter2", type=int, default=300)
    parser.add_argument("--matching_conf_thr", type=float, default=0.0)
    parser.add_argument("--shared_intrinsics", action="store_true", default=True)
    parser.add_argument("--clean_depth", action="store_true", default=True)
    parser.add_argument("--opt_depth", action="store_true", default=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    image_dir = Path(args.image_dir)
    output_sparse = Path(args.output_sparse) if args.output_sparse else dataset_root / "sparse" / "0"
    output_images = Path(args.output_images) if args.output_images else dataset_root / "images_mast3r"

    output_sparse.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    filelist = collect_unique_images(image_dir)
    if len(filelist) < 2:
        raise RuntimeError(f"Need at least 2 images, found {len(filelist)}")

    print(f"[INFO] Found {len(filelist)} unique images")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Dataset root: {dataset_root}")
    print(f"[INFO] Output sparse: {output_sparse}")
    print(f"[INFO] Output images: {output_images}")

    cache_dir = tempfile.mkdtemp(prefix="mast3r_cache_", dir=str(dataset_root))
    print(f"[INFO] Fresh cache dir: {cache_dir}")

    print(f"[INFO] Loading model: {args.model_name}")
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    print("[INFO] Loading images...")
    imgs = load_images(filelist, size=args.image_size, verbose=True)

    print("[INFO] Building pairs...")
    pairs = make_pairs(
        imgs,
        scene_graph="complete",
        prefilter=None,
        symmetrize=True,
    )

    print("[INFO] Running sparse global alignment...")
    scene = sparse_global_alignment(
        filelist,
        pairs,
        cache_dir,
        model,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        device=args.device,
        opt_depth=args.opt_depth,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr,
    )

    print("[INFO] Saving resized MASt3R images...")
    image_names, img_sizes = save_resized_images(scene.imgs, filelist, output_images)

    print("[INFO] Extracting camera poses/intrinsics...")
    cams2world = to_numpy(scene.get_im_poses().cpu())
    intrinsics = try_get_intrinsics(scene, img_sizes)

    print("[INFO] Building export point cloud...")
    pts, rgb = build_point_cloud(
        scene=scene,
        min_conf_thr=args.min_conf_thr,
        clean_depth=args.clean_depth,
        max_points=args.max_points,
        seed=args.seed,
    )

    print("[INFO] Writing COLMAP text files...")
    export_cameras_txt(output_sparse / "cameras.txt", intrinsics, img_sizes)
    export_images_txt(output_sparse / "images.txt", cams2world, image_names)
    export_points3D_txt(output_sparse / "points3D.txt", pts, rgb)
    export_ply(output_sparse / "points3D.ply", pts, rgb)

    print("[DONE] Export complete.")
    print(f"[DONE] cameras.txt  -> {output_sparse / 'cameras.txt'}")
    print(f"[DONE] images.txt   -> {output_sparse / 'images.txt'}")
    print(f"[DONE] points3D.txt -> {output_sparse / 'points3D.txt'}")
    print(f"[DONE] points3D.ply -> {output_sparse / 'points3D.ply'}")
    print(f"[DONE] images       -> {output_images}")
    print()
    print("[NEXT] Train 3DGS with the resized images that match MASt3R intrinsics:")
    print(f'python train.py -s "{dataset_root}" -i "{output_images.name}" -m "D:\\All for one\\output\\kitchen_12_mast3rgs" --eval --iterations 7000')


if __name__ == "__main__":
    main()
