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

import torch
from torch import nn
import math
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2


def _quaternion_to_matrix(q: torch.Tensor):
    q = torch.nn.functional.normalize(q, dim=0)
    w, x, y, z = q.unbind(dim=0)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y))),
            torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x))),
            torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))),
        ),
        dim=0,
    )

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap, depth_confidence,
                 image_name, uid,
                 pose_correspondences_xy=None, pose_correspondences_xyz=None, pose_correspondence_error=None,
                 pose_correspondence_source_width=None, pose_correspondence_source_height=None,
                 pose_correspondence_atlas_node_ids=None, pose_correspondence_atlas_reliability=None,
                 pose_correspondence_trust=None, pose_correspondence_is_atlas_native=None,
                 pose_correspondence_source=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.pose_correspondences_xy = None
        self.pose_correspondences_xyz = None
        self.pose_correspondence_error = None
        self.pose_correspondence_atlas_node_ids = None
        self.pose_correspondence_atlas_reliability = None
        self.pose_correspondence_trust = None
        self.pose_correspondence_is_atlas_native = None
        self.pose_correspondence_source = str(pose_correspondence_source or "missing")
        if pose_correspondences_xy is not None and pose_correspondences_xyz is not None:
            pose_correspondences_xy = np.asarray(pose_correspondences_xy, dtype=np.float32)
            pose_correspondences_xyz = np.asarray(pose_correspondences_xyz, dtype=np.float32)
            error_scale = 1.0
            source_width = int(self.image_width if pose_correspondence_source_width is None else pose_correspondence_source_width)
            source_height = int(self.image_height if pose_correspondence_source_height is None else pose_correspondence_source_height)
            if source_width > 0 and source_height > 0:
                scale_x = float(self.image_width) / float(source_width)
                scale_y = float(self.image_height) / float(source_height)
                error_scale = float(np.sqrt(0.5 * (scale_x * scale_x + scale_y * scale_y)))
                pose_correspondences_xy = pose_correspondences_xy.copy()
                pose_correspondences_xy[:, 0] *= scale_x
                pose_correspondences_xy[:, 1] *= scale_y
            pose_correspondence_error = np.asarray(
                np.ones((pose_correspondences_xy.shape[0],), dtype=np.float32) if pose_correspondence_error is None else pose_correspondence_error,
                dtype=np.float32,
            ).reshape(-1)
            pose_correspondence_error = (pose_correspondence_error * error_scale).astype(np.float32, copy=False)
            if pose_correspondences_xy.shape[0] > 0 and pose_correspondences_xyz.shape[0] == pose_correspondences_xy.shape[0]:
                corr_count = int(pose_correspondences_xy.shape[0])
                self.pose_correspondences_xy = torch.from_numpy(pose_correspondences_xy).to(self.data_device)
                self.pose_correspondences_xyz = torch.from_numpy(pose_correspondences_xyz).to(self.data_device)
                self.pose_correspondence_error = torch.from_numpy(pose_correspondence_error.reshape(-1)).to(self.data_device)
                atlas_node_ids = np.full((corr_count,), -1, dtype=np.int64) if pose_correspondence_atlas_node_ids is None else np.asarray(pose_correspondence_atlas_node_ids, dtype=np.int64).reshape(-1)
                atlas_reliability = np.ones((corr_count,), dtype=np.float32) if pose_correspondence_atlas_reliability is None else np.asarray(pose_correspondence_atlas_reliability, dtype=np.float32).reshape(-1)
                corr_trust = np.ones((corr_count,), dtype=np.float32) if pose_correspondence_trust is None else np.asarray(pose_correspondence_trust, dtype=np.float32).reshape(-1)
                is_atlas_native = np.zeros((corr_count,), dtype=np.bool_) if pose_correspondence_is_atlas_native is None else np.asarray(pose_correspondence_is_atlas_native, dtype=np.bool_).reshape(-1)
                if atlas_node_ids.shape[0] != corr_count:
                    atlas_node_ids = np.full((corr_count,), -1, dtype=np.int64)
                if atlas_reliability.shape[0] != corr_count:
                    atlas_reliability = np.ones((corr_count,), dtype=np.float32)
                if corr_trust.shape[0] != corr_count:
                    corr_trust = np.ones((corr_count,), dtype=np.float32)
                if is_atlas_native.shape[0] != corr_count:
                    is_atlas_native = np.zeros((corr_count,), dtype=np.bool_)
                self.pose_correspondence_atlas_node_ids = torch.from_numpy(atlas_node_ids).to(self.data_device)
                self.pose_correspondence_atlas_reliability = torch.from_numpy(np.clip(atlas_reliability, 0.0, 1.0).astype(np.float32)).to(self.data_device)
                self.pose_correspondence_trust = torch.from_numpy(np.clip(corr_trust, 0.0, 1.0).astype(np.float32)).to(self.data_device)
                self.pose_correspondence_is_atlas_native = torch.from_numpy(is_atlas_native).to(self.data_device)

        self.invdepthmap = None
        self.depth_confidence = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]

            if depth_confidence is not None:
                resized_confidence = cv2.resize(depth_confidence, resolution)
                if resized_confidence.ndim != 2:
                    resized_confidence = resized_confidence[..., 0]
                resized_confidence = np.clip(resized_confidence, 0.0, 1.0).astype(np.float32)
                resized_confidence[self.invdepthmap <= 0] = 0.0
                self.depth_confidence = torch.from_numpy(resized_confidence[None]).to(self.data_device)

            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.base_world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32, device=self.data_device)
        self.pose_delta_q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.data_device), requires_grad=False)
        self.pose_delta_t = nn.Parameter(torch.zeros((3,), dtype=torch.float32, device=self.data_device), requires_grad=False)
        self.pose_runtime_metadata = {
            "b1_skip_reason": "none",
            "b2_skip_reason": "none",
            "trust_translation_limit": 0.0,
            "trust_rotation_limit_degrees": 0.0,
            "last_stage": "none",
            "last_iteration": -1,
        }
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).to(self.data_device)
        self.world_view_transform = None
        self.full_proj_transform = None
        self.camera_center = None
        self.refresh_pose_matrices()

    def set_pose_trainable(self, enabled: bool):
        self.pose_delta_q.requires_grad_(enabled)
        self.pose_delta_t.requires_grad_(enabled)
        self.refresh_pose_matrices()

    def build_pose_matrices(self, differentiable=None):
        pose_trainable = (
            bool(self.pose_delta_q.requires_grad or self.pose_delta_t.requires_grad)
            if differentiable is None
            else bool(differentiable)
        )

        def _build_pose_matrices():
            delta_rot = _quaternion_to_matrix(self.pose_delta_q)
            # Use cat/stack to build the 4x4 delta matrix without in-place ops.
            # In-place assignment (delta[:3,:3] = delta_rot) into a leaf tensor
            # does NOT create autograd edges, severing the gradient path to
            # pose_delta_q and pose_delta_t. cat preserves the gradient graph.
            col_t = self.pose_delta_t.unsqueeze(1)                         # (3,1)
            top_block = torch.cat([delta_rot, col_t], dim=1)               # (3,4)
            bottom_row = self.pose_delta_q.new_tensor([[0.0, 0.0, 0.0, 1.0]])  # (1,4)
            delta = torch.cat([top_block, bottom_row], dim=0)              # (4,4)
            w2c = delta @ self.base_world_view_transform
            world_view = w2c.transpose(0, 1)
            full_proj = (world_view.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
            c2w = torch.inverse(w2c)
            camera_center = c2w[:3, 3]
            return world_view, full_proj, camera_center

        if pose_trainable:
            return _build_pose_matrices()

        with torch.no_grad():
            world_view, full_proj, camera_center = _build_pose_matrices()
        return world_view.detach(), full_proj.detach(), camera_center.detach()

    def refresh_pose_matrices(self, differentiable=None):
        self.world_view_transform, self.full_proj_transform, self.camera_center = self.build_pose_matrices(
            differentiable=differentiable
        )

    def refresh_pose_matrices_differentiable(self):
        self.refresh_pose_matrices(differentiable=True)

    def refresh_pose_matrices_static(self):
        self.refresh_pose_matrices(differentiable=False)

    def get_pose_debug_snapshot(self):
        pose_trainable = bool(self.pose_delta_q.requires_grad or self.pose_delta_t.requires_grad)
        if self.world_view_transform is None or self.full_proj_transform is None or self.camera_center is None:
            self.refresh_pose_matrices(differentiable=pose_trainable)
        with torch.no_grad():
            pose_t_norm = float(torch.linalg.norm(self.pose_delta_t.detach()).item())
            pose_q = torch.nn.functional.normalize(self.pose_delta_q.detach(), dim=0)
            sin_half = torch.linalg.norm(pose_q[1:]).clamp_min(1e-8)
            cos_half = pose_q[0].abs().clamp_min(1e-8)
            rotation_degrees = float(math.degrees(float((2.0 * torch.atan2(sin_half, cos_half)).item())))
        return {
            "pose_delta_q_requires_grad": 1.0 if bool(getattr(self.pose_delta_q, "requires_grad", False)) else 0.0,
            "pose_delta_t_requires_grad": 1.0 if bool(getattr(self.pose_delta_t, "requires_grad", False)) else 0.0,
            "world_view_transform_requires_grad": 1.0 if bool(getattr(self.world_view_transform, "requires_grad", False)) else 0.0,
            "full_proj_transform_requires_grad": 1.0 if bool(getattr(self.full_proj_transform, "requires_grad", False)) else 0.0,
            "camera_center_requires_grad": 1.0 if bool(getattr(self.camera_center, "requires_grad", False)) else 0.0,
            "pose_delta_t_norm": pose_t_norm,
            "pose_delta_rotation_degrees": rotation_degrees,
            "pose_trainable": 1.0 if pose_trainable else 0.0,
        }

    def get_pose_parameters(self):
        return [self.pose_delta_q, self.pose_delta_t]

    def record_pose_runtime(self, **kwargs):
        self.pose_runtime_metadata.update(kwargs)

    def get_pose_runtime_metadata(self):
        return dict(self.pose_runtime_metadata)

    def export_pose_delta(self):
        return {
            "pose_delta_q": self.pose_delta_q.detach().cpu().tolist(),
            "pose_delta_t": self.pose_delta_t.detach().cpu().tolist(),
        }

    def load_pose_delta(self, pose_delta_q, pose_delta_t):
        with torch.no_grad():
            self.pose_delta_q.copy_(torch.tensor(pose_delta_q, dtype=torch.float32, device=self.data_device))
            self.pose_delta_t.copy_(torch.tensor(pose_delta_t, dtype=torch.float32, device=self.data_device))
        self.refresh_pose_matrices()
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
