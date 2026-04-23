#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import importlib.util
import os
import random
import json
from pathlib import Path
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.foundation_atlas import (
    load_foundation_atlas,
    resolve_foundation_atlas_root,
    summarize_atlas_initialization,
    save_json,
)
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


_ATLAS_VALIDATOR = None


def _load_atlas_validator():
    global _ATLAS_VALIDATOR
    if _ATLAS_VALIDATOR is not None:
        return _ATLAS_VALIDATOR

    scene_path = Path(__file__).resolve()
    candidate_paths = [
        scene_path.parents[1] / "tools" / "validate_atlas_artifact.py",
        scene_path.parents[2] / "tools" / "validate_atlas_artifact.py",
    ]
    validator_path = next((path for path in candidate_paths if path.exists()), None)
    if validator_path is None:
        searched = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(f"Atlas validator is missing. Searched: {searched}")

    spec = importlib.util.spec_from_file_location("aurora_validate_atlas_artifact", validator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load atlas validator module from {validator_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validator = getattr(module, "validate_atlas_artifact", None)
    if validator is None:
        raise AttributeError(f"Atlas validator module does not expose validate_atlas_artifact(): {validator_path}")

    _ATLAS_VALIDATOR = validator
    return _ATLAS_VALIDATOR


def _run_atlas_preflight(atlas_path: str, model_path: str):
    validator = _load_atlas_validator()
    report = validator(atlas_path, strict=True)
    report["atlas_path"] = str(Path(atlas_path).expanduser().resolve())
    if model_path:
        save_json(report, Path(model_path) / "atlas_preflight.json")
    if not bool(report.get("valid", False)):
        errors = report.get("errors", []) or []
        failure = errors[0] if errors else "unknown atlas validation error"
        raise ValueError(f"Atlas artifact preflight failed for {report['atlas_path']}: {failure}")
    return report


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if args.atlas_path:
            args.atlas_path = str(resolve_foundation_atlas_root(args.atlas_path))
            _run_atlas_preflight(args.atlas_path, self.model_path)

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.depths,
                args.depth_confidences,
                args.eval,
                args.train_test_exp,
                args.atlas_path,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.depths,
                args.depth_confidences,
                args.eval,
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
            atlas_state_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "atlas_state.npz",
            )
            if os.path.exists(atlas_state_path):
                self.gaussians.load_atlas_state(atlas_state_path)
            elif args.atlas_path:
                print(f"Warning: atlas bindings were requested but no saved atlas state was found at {atlas_state_path}.")
            pose_state_path = os.path.join(
                self.model_path,
                "point_cloud",
                "iteration_" + str(self.loaded_iter),
                "camera_pose_deltas.json",
            )
            self.load_pose_state(pose_state_path)
        else:
            if args.atlas_path:
                atlas_init = load_foundation_atlas(
                    args.atlas_path,
                    fallback_point_cloud=scene_info.point_cloud,
                    color_sample_limit=args.atlas_color_sample_limit,
                    seed=args.atlas_seed,
                    surface_stable_min=args.atlas_surface_stable_reliability,
                    edge_stable_min=args.atlas_edge_stable_reliability,
                    scale_multiplier=args.atlas_scale_multiplier,
                    surface_thickness_ratio=args.atlas_surface_thickness,
                    edge_thickness_ratio=args.atlas_edge_thickness,
                    unstable_scale_ratio=args.atlas_unstable_scale,
                )
                self.gaussians.create_from_atlas(atlas_init, scene_info.train_cameras, self.cameras_extent)
                save_json(
                    summarize_atlas_initialization(atlas_init),
                    os.path.join(self.model_path, "atlas_init_summary.json"),
                )
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_atlas_state(os.path.join(point_cloud_path, "atlas_state.npz"))
        save_json(self.gaussians.summarize_atlas_bindings(), os.path.join(point_cloud_path, "atlas_state_summary.json"))
        self.save_pose_state(os.path.join(point_cloud_path, "camera_pose_deltas.json"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(point_cloud_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def set_pose_trainable(self, enabled: bool, scale=1.0):
        for camera in self.train_cameras[scale]:
            if hasattr(camera, "set_pose_trainable"):
                camera.set_pose_trainable(enabled)

    def get_pose_parameters(self, scale=1.0):
        params = []
        for camera in self.train_cameras[scale]:
            if hasattr(camera, "get_pose_parameters"):
                params.extend(camera.get_pose_parameters())
        return params

    def get_pose_camera_order(self, scale=1.0):
        return [
            camera.image_name
            for camera in self.train_cameras[scale]
            if hasattr(camera, "export_pose_delta")
        ]

    def export_pose_state(self, scale=1.0):
        payload = {}
        for camera in self.train_cameras[scale]:
            if hasattr(camera, "export_pose_delta"):
                payload[camera.image_name] = camera.export_pose_delta()
        return payload

    def apply_pose_state(self, payload, scale=1.0):
        if not payload:
            return
        for camera in self.train_cameras[scale]:
            if camera.image_name in payload and hasattr(camera, "load_pose_delta"):
                entry = payload[camera.image_name]
                camera.load_pose_delta(
                    entry.get("pose_delta_q", [1.0, 0.0, 0.0, 0.0]),
                    entry.get("pose_delta_t", [0.0, 0.0, 0.0]),
                )

    def save_pose_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.export_pose_state()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_pose_state(self, path, scale=1.0):
        path = Path(path)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.apply_pose_state(payload, scale=scale)
