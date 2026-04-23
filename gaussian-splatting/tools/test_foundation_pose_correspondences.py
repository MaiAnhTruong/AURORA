import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.cameras import Camera  # noqa: E402
from scene.dataset_readers import readColmapCameras, readColmapSceneInfo  # noqa: E402
from scene.foundation_atlas import load_foundation_pose_correspondences  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


def main():
    tmp_root = REPO_ROOT / ".tmp_foundation_pose_corr"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        manifest_path = atlas_root / "correspondence_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["views"]["cam_0.png"].update(
            {
                "coordinate_space": "mast3r_preprocessed_image",
                "source_width": 16,
                "source_height": 12,
                "resized_width": 8,
                "resized_height": 6,
                "crop_left": 0.0,
                "crop_top": 0.0,
                "crop_right": 8.0,
                "crop_bottom": 6.0,
                "scale_x": 0.5,
                "scale_y": 0.5,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        baseline_loaded = load_foundation_pose_correspondences(atlas_root, "cam_0.png", max_correspondences=0)
        assert baseline_loaded is not None
        _, _, baseline_corr_error, _, _ = baseline_loaded

        scene_alignment = {
            "schema_version": 1,
            "applied": True,
            "dense_correspondence_audit": {
                "views": {
                    "cam_0.png": {
                        "sampled_corr": 384,
                        "projected_corr": 360,
                        "in_frame_corr": 332,
                        "mean_px_error": 5.4,
                        "median_px_error": 4.8,
                        "p90_px_error": 7.2,
                    },
                },
            },
        }
        (atlas_root / "scene_alignment.json").write_text(json.dumps(scene_alignment, indent=2), encoding="utf-8")

        loaded = load_foundation_pose_correspondences(atlas_root, "cam_0.png", max_correspondences=0)
        assert loaded is not None
        corr_xy, corr_xyz, corr_error, src_width, src_height = loaded
        assert corr_xy.shape[0] > 0
        assert corr_xyz.shape[0] == corr_xy.shape[0]
        assert corr_error.shape[0] == corr_xy.shape[0]
        assert src_width == 16
        assert src_height == 12
        assert np.isfinite(corr_xyz).all()
        assert np.isfinite(corr_error).all()
        assert np.all(corr_error > baseline_corr_error)
        conf = np.load(atlas_root / "dense_geometry" / "confidence" / "cam_0.npy")
        yy, xx = np.nonzero(np.isfinite(conf) & (conf >= 0.5))
        expected_corr_xy = np.stack((xx, yy), axis=1).astype(np.float32)
        expected_corr_xy[:, 0] *= 2.0
        expected_corr_xy[:, 1] *= 2.0
        assert np.allclose(corr_xy, expected_corr_xy, atol=1e-6)

        images_dir = tmp_root / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / "cam_0.png"
        Image.new("RGB", (16, 12), color=(128, 128, 128)).save(image_path)

        extr = SimpleNamespace(
            camera_id=1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            tvec=np.zeros((3,), dtype=np.float64),
            name="cam_0.png",
            xys=np.zeros((0, 2), dtype=np.float32),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
        intr = SimpleNamespace(
            id=1,
            model="PINHOLE",
            params=np.array([16.0, 16.0, 8.0, 6.0], dtype=np.float32),
            width=16,
            height=12,
        )
        cam_infos = readColmapCameras(
            cam_extrinsics={1: extr},
            cam_intrinsics={1: intr},
            depths_params=None,
            images_folder=str(images_dir),
            depths_folder="",
            confidence_folder="",
            test_cam_names_list=[],
            points3D_lookup=None,
            atlas_path=str(atlas_root),
        )
        assert len(cam_infos) == 1
        cam_info = cam_infos[0]
        assert cam_info.pose_correspondences_xy is not None
        assert cam_info.pose_correspondences_xyz is not None
        assert cam_info.pose_correspondence_source_width == 16
        assert cam_info.pose_correspondence_source_height == 12

        camera = Camera(
            resolution=(4, 3),
            colmap_id=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            depth_params=None,
            image=Image.open(image_path),
            invdepthmap=None,
            depth_confidence=None,
            image_name=cam_info.image_name,
            uid=0,
            pose_correspondences_xy=cam_info.pose_correspondences_xy,
            pose_correspondences_xyz=cam_info.pose_correspondences_xyz,
            pose_correspondence_error=cam_info.pose_correspondence_error,
            pose_correspondence_source_width=cam_info.pose_correspondence_source_width,
            pose_correspondence_source_height=cam_info.pose_correspondence_source_height,
            data_device="cpu",
        )
        expected_xy = cam_info.pose_correspondences_xy.copy()
        expected_xy[:, 0] *= 0.25
        expected_xy[:, 1] *= 0.25
        expected_error = cam_info.pose_correspondence_error.copy() * 0.25
        assert camera.pose_correspondences_xy is not None
        assert np.allclose(camera.pose_correspondences_xy.cpu().numpy(), expected_xy, atol=1e-6)
        assert camera.pose_correspondence_error is not None
        assert np.allclose(camera.pose_correspondence_error.cpu().numpy(), expected_error, atol=1e-6)

        scene_root = tmp_root / "synthetic_scene"
        sparse_root = scene_root / "sparse" / "0"
        sparse_root.mkdir(parents=True, exist_ok=True)
        scene_images_dir = scene_root / "images"
        scene_images_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), color=(64, 64, 64)).save(scene_images_dir / "cam_0.png")
        with open(sparse_root / "cameras.txt", "w", encoding="utf-8") as handle:
            handle.write("# Camera list\n")
            handle.write("1 PINHOLE 16 12 16 16 8 6\n")
        with open(sparse_root / "images.txt", "w", encoding="utf-8") as handle:
            handle.write("# Image list\n")
            handle.write("1 1 0 0 0 0 0 0 1 cam_0.png\n")
            handle.write("\n")
        scene_info = readColmapSceneInfo(
            str(scene_root),
            "images",
            "",
            "",
            False,
            False,
            atlas_path=str(atlas_root),
        )
        assert len(scene_info.train_cameras) == 1
        assert scene_info.train_cameras[0].pose_correspondences_xyz is not None
        print("[OK] Foundation pose correspondence loading check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
