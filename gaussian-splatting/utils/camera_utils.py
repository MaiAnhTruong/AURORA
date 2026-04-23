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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2
from pathlib import Path

WARNED = False


def _load_confidence_map(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        confidence = np.load(path)
    elif suffix == ".npz":
        archive = np.load(path)
        confidence = archive[archive.files[0]]
    else:
        confidence = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if confidence is None:
        raise FileNotFoundError(f"Unable to load confidence map from '{path}'.")

    confidence = np.asarray(confidence)
    if confidence.ndim == 3:
        confidence = confidence[..., 0]

    original_dtype = confidence.dtype
    confidence = confidence.astype(np.float32)
    confidence[~np.isfinite(confidence)] = 0.0

    if np.issubdtype(original_dtype, np.integer):
        confidence /= max(float(np.iinfo(original_dtype).max), 1.0)
    else:
        min_value = float(confidence.min()) if confidence.size > 0 else 0.0
        max_value = float(confidence.max()) if confidence.size > 0 else 0.0
        if min_value < 0.0 or max_value > 1.0:
            if max_value > min_value:
                confidence = (confidence - min_value) / (max_value - min_value)
            else:
                confidence = np.zeros_like(confidence)

    return np.clip(confidence, 0.0, 1.0)

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)

    if cam_info.depth_path != "":
        try:
            if is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None

    if cam_info.confidence_path != "":
        try:
            depth_confidence = _load_confidence_map(cam_info.confidence_path)
        except FileNotFoundError:
            print(f"Error: The confidence file at path '{cam_info.confidence_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the confidence file '{cam_info.confidence_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read confidence at {cam_info.confidence_path}: {e}")
            raise
    else:
        depth_confidence = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap, depth_confidence=depth_confidence,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  pose_correspondences_xy=cam_info.pose_correspondences_xy,
                  pose_correspondences_xyz=cam_info.pose_correspondences_xyz,
                  pose_correspondence_error=cam_info.pose_correspondence_error,
                  pose_correspondence_source_width=cam_info.pose_correspondence_source_width,
                  pose_correspondence_source_height=cam_info.pose_correspondence_source_height,
                  pose_correspondence_atlas_node_ids=cam_info.pose_correspondence_atlas_node_ids,
                  pose_correspondence_atlas_reliability=cam_info.pose_correspondence_atlas_reliability,
                  pose_correspondence_trust=cam_info.pose_correspondence_trust,
                  pose_correspondence_is_atlas_native=cam_info.pose_correspondence_is_atlas_native,
                  pose_correspondence_source=cam_info.pose_correspondence_source,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
