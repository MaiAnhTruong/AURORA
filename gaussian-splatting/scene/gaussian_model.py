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

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.foundation_atlas import (
    FoundationAtlasInit,
    ATLAS_CLASS_EDGE,
    ATLAS_CLASS_SURFACE,
    ATLAS_CLASS_UNSTABLE,
    ATLAS_CLASS_NAMES,
    GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
    GAUSSIAN_STATE_STABLE,
    GAUSSIAN_STATE_NAMES,
    GAUSSIAN_STATE_UNSTABLE_ACTIVE,
    GAUSSIAN_STATE_UNSTABLE_PASSIVE,
)
from scene.foundation_atlas_exploration import compute_point_slab_bounds

ACTIVE_PROVENANCE_NONE = 0
ACTIVE_PROVENANCE_FROM_TRANSITION_PASSIVE_TO_ACTIVE = 1
ACTIVE_PROVENANCE_FROM_RESTORE_CHECKPOINT = 2
ACTIVE_PROVENANCE_FROM_STATE_REBUILD_AFTER_GC = 3
ACTIVE_PROVENANCE_FROM_QUOTA_CARRYOVER = 4
ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP = 5
ACTIVE_PROVENANCE_FROM_ACTIVE_EXPLORE_CLONE = 6
ACTIVE_PROVENANCE_NAMES = {
    ACTIVE_PROVENANCE_FROM_TRANSITION_PASSIVE_TO_ACTIVE: "from_transition_passive_to_active",
    ACTIVE_PROVENANCE_FROM_RESTORE_CHECKPOINT: "from_restore_checkpoint",
    ACTIVE_PROVENANCE_FROM_STATE_REBUILD_AFTER_GC: "from_state_rebuild_after_gc",
    ACTIVE_PROVENANCE_FROM_QUOTA_CARRYOVER: "from_quota_carryover",
    ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP: "from_forced_rescue_bootstrap",
    ACTIVE_PROVENANCE_FROM_ACTIVE_EXPLORE_CLONE: "from_active_explore_clone",
}

try:
    from simple_knn._C import distCUDA2
except ImportError:
    distCUDA2 = None

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def _device(self):
        if self._xyz.numel() > 0:
            return self._xyz.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _estimate_point_dist2(self, points: torch.Tensor):
        if distCUDA2 is not None and points.is_cuda:
            return torch.clamp_min(distCUDA2(points), 1e-7)

        with torch.no_grad():
            distances = torch.cdist(points, points)
            inf = torch.tensor(float("inf"), dtype=distances.dtype, device=distances.device)
            diag = torch.arange(points.shape[0], device=distances.device)
            distances[diag, diag] = inf
            return torch.clamp_min(torch.min(distances, dim=1).values.square(), 1e-7)

    def _rowwise_finite_mask(self, tensor: torch.Tensor | None):
        if tensor is None:
            return None
        if tensor.ndim == 0:
            if tensor.is_floating_point() or tensor.is_complex():
                is_finite = bool(torch.isfinite(tensor).item())
                return torch.tensor([is_finite], dtype=torch.bool, device=tensor.device)
            return torch.ones((1,), dtype=torch.bool, device=tensor.device)
        row_count = int(tensor.shape[0])
        if row_count == 0:
            return torch.zeros((0,), dtype=torch.bool, device=tensor.device)
        if not (tensor.is_floating_point() or tensor.is_complex()):
            return torch.ones((row_count,), dtype=torch.bool, device=tensor.device)
        finite = torch.isfinite(tensor)
        if tensor.ndim == 1:
            return finite
        return finite.reshape(row_count, -1).all(dim=1)

    def _filter_nonfinite_clone_payload(self, payload: dict):
        row_count = None
        valid_mask = None
        for value in payload.values():
            if value is None or (not torch.is_tensor(value)) or value.ndim == 0:
                continue
            if row_count is None:
                row_count = int(value.shape[0])
            elif int(value.shape[0]) != row_count:
                continue
            row_mask = self._rowwise_finite_mask(value)
            if row_mask is None:
                continue
            valid_mask = row_mask if valid_mask is None else torch.logical_and(valid_mask, row_mask)

        if row_count is None or valid_mask is None:
            return payload, 0

        discard_count = int((~valid_mask).sum().item())
        if discard_count <= 0:
            return payload, 0

        filtered = {}
        for key, value in payload.items():
            if value is None or (not torch.is_tensor(value)) or value.ndim == 0:
                filtered[key] = value
            elif int(value.shape[0]) == row_count:
                filtered[key] = value[valid_mask]
            else:
                filtered[key] = value
        return filtered, discard_count

    def _build_invalid_gaussian_mask(self):
        point_count = int(self.get_xyz.shape[0])
        device = self._device()
        if point_count == 0:
            return torch.zeros((0,), dtype=torch.bool, device=device)

        with torch.no_grad():
            masks = [
                self._rowwise_finite_mask(self._xyz.detach()),
                self._rowwise_finite_mask(self._features_dc.detach()),
                self._rowwise_finite_mask(self._features_rest.detach()),
                self._rowwise_finite_mask(self._opacity.detach()),
                self._rowwise_finite_mask(self._scaling.detach()),
                self._rowwise_finite_mask(self._rotation.detach()),
                self._rowwise_finite_mask(self._center_log_sigma_parallel.detach()),
                self._rowwise_finite_mask(self._center_log_sigma_support.detach()),
                self._rowwise_finite_mask(self.get_xyz.detach()),
                self._rowwise_finite_mask(self.get_scaling.detach()),
                self._rowwise_finite_mask(self.get_rotation.detach()),
                self._rowwise_finite_mask(self.get_opacity.detach()),
                self._rowwise_finite_mask(self.get_center_sigma_parallel.detach()),
                self._rowwise_finite_mask(self.get_center_sigma_support.detach()),
            ]
            valid_mask = torch.ones((point_count,), dtype=torch.bool, device=device)
            for row_mask in masks:
                if row_mask is None or row_mask.numel() == 0:
                    continue
                valid_mask = torch.logical_and(valid_mask, row_mask.to(device=device, dtype=torch.bool))
            return ~valid_mask

    def _build_feature_tensors(self, points_np, colors_np):
        fused_point_cloud = torch.tensor(np.asarray(points_np), dtype=torch.float32, device=self._device())
        fused_color = RGB2SH(torch.tensor(np.asarray(colors_np), dtype=torch.float32, device=self._device()))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2), dtype=torch.float32, device=self._device())
        features[:, :3, 0] = fused_color
        return fused_point_cloud, features

    def _initialize_gaussian_parameters(self, fused_point_cloud, features, scales, rots, opacities):
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._device())

    def _initialize_center_uncertainty(self, sigma_parallel, sigma_support):
        sigma_parallel = torch.as_tensor(sigma_parallel, dtype=torch.float32, device=self._device()).reshape(-1, 1)
        sigma_support = torch.as_tensor(sigma_support, dtype=torch.float32, device=self._device()).reshape(-1, 1)
        self._center_log_sigma_parallel = nn.Parameter(torch.log(sigma_parallel.clamp_min(1e-6)).requires_grad_(True))
        self._center_log_sigma_support = nn.Parameter(torch.log(sigma_support.clamp_min(1e-6)).requires_grad_(True))

    def _ordered_exposure_names(self):
        if not self.exposure_mapping:
            return []
        ordered_names = [None] * (max(self.exposure_mapping.values()) + 1)
        for image_name, index in self.exposure_mapping.items():
            if 0 <= index < len(ordered_names):
                ordered_names[index] = image_name
        return ordered_names

    def _capture_exposure_payload(self):
        if self._exposure.numel() == 0 and not self.exposure_mapping:
            return None
        return {
            "names": self._ordered_exposure_names(),
            "values": self._exposure.detach().clone(),
        }

    def _restore_exposure_payload(self, exposure_payload):
        if not exposure_payload:
            return False

        saved_names = list(exposure_payload.get("names", []))
        saved_values = exposure_payload.get("values", None)
        if saved_values is None:
            return False

        device = self._device()
        saved_values = torch.as_tensor(saved_values, dtype=torch.float32, device=device)
        current_names = self._ordered_exposure_names()
        if not current_names and saved_names:
            self.exposure_mapping = {
                image_name: idx
                for idx, image_name in enumerate(saved_names)
                if image_name is not None
            }
            current_names = list(saved_names)

        can_restore_optimizer = current_names == saved_names
        if current_names:
            if can_restore_optimizer:
                exposure_tensor = saved_values
            else:
                saved_index = {
                    image_name: idx
                    for idx, image_name in enumerate(saved_names)
                    if image_name is not None
                }
                default_exposure = torch.eye(3, 4, dtype=torch.float32, device=device)
                rows = []
                for image_name in current_names:
                    if image_name in saved_index and saved_index[image_name] < saved_values.shape[0]:
                        rows.append(saved_values[saved_index[image_name]])
                    else:
                        rows.append(default_exposure)
                exposure_tensor = torch.stack(rows, dim=0)
        else:
            exposure_tensor = torch.empty((0, 3, 4), dtype=torch.float32, device=device)

        self.pretrained_exposures = None
        self._exposure = nn.Parameter(exposure_tensor.requires_grad_(True))
        return can_restore_optimizer

    def _clear_atlas_bindings(self):
        device = self._device()
        self._atlas_node_ids = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_state = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_positions = torch.empty((0, 3), dtype=torch.float32, device=device)
        self._atlas_support = torch.empty((0, 3, 3), dtype=torch.float32, device=device)
        self._atlas_basis = torch.empty((0, 3, 3), dtype=torch.float32, device=device)
        self._atlas_raw_score = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_reliability_base = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_radius = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_reliability_runtime_raw = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_reliability_runtime_mapped = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_reliability_effective = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_reliability_runtime = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_class = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_anisotropy_ref = torch.empty((0, 2), dtype=torch.float32, device=device)
        self._atlas_neighbor_indices = torch.empty((0, 1), dtype=torch.long, device=device)
        self._atlas_node_photo_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_visibility_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_obs_quality_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_support_consistency_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_finite_projection_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_ref_consistency_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_observed_score_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_updated_recently = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_observed_count = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_node_support_consistent_count = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_observed_mask = torch.empty((0,), dtype=torch.bool, device=device)
        self._atlas_refresh_node_photo_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_node_visibility_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_obs_quality = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_node_observed_count = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_node_support_consistent_ratio = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_node_coverage_ratio = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_node_ambiguity = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_override_weight = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_refresh_runtime_override_mask = torch.empty((0,), dtype=torch.bool, device=device)
        self._atlas_photo_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_visibility_ema = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_high_residual_count = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_low_residual_count = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_promotion_streak = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_demotion_streak = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_recovery_streak = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_last_transition_iter = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_gc_fail_count = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_drift_flag = torch.empty((0,), dtype=torch.bool, device=device)
        self._atlas_drift_count = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_state_cooldown = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_active_lifetime = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_ref_camera = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_ref_score = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_last_good_node_ids = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_pending_ref_camera = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_pending_ref_tau = torch.empty((0,), dtype=torch.float32, device=device)
        self._atlas_pending_retry_count = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_last_pending_iter = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_active_provenance = torch.empty((0,), dtype=torch.long, device=device)
        self._atlas_view_weights = torch.empty((0, 0), dtype=torch.float32, device=device)
        self._atlas_view_counts = torch.empty((0, 0), dtype=torch.int32, device=device)
        self._atlas_refresh_done = False
        self._atlas_source_path = ""
        self._atlas_init_num_nodes = 0
        self._atlas_init_gaussian_count_pre_spawn = 0
        self._atlas_extra_surface_spawn_count = 0
        self._atlas_init_gaussian_count_post_spawn = 0
        self._atlas_state_update_iter = 0
        self._atlas_runtime_last_observation = None
        self._invalidate_atlas_spatial_hash(clear_metadata=True)

    def _set_atlas_store(self, atlas_init: FoundationAtlasInit):
        device = self._device()
        self._atlas_positions = torch.tensor(atlas_init.positions, dtype=torch.float32, device=device)
        self._atlas_support = torch.tensor(atlas_init.support, dtype=torch.float32, device=device)
        self._atlas_basis = torch.tensor(atlas_init.basis, dtype=torch.float32, device=device)
        self._atlas_raw_score = torch.tensor(atlas_init.raw_score, dtype=torch.float32, device=device)
        self._atlas_reliability_base = torch.tensor(atlas_init.reliability, dtype=torch.float32, device=device)
        self._atlas_radius = torch.tensor(atlas_init.radius, dtype=torch.float32, device=device)
        self._atlas_reliability_runtime_raw = self._atlas_reliability_base.detach().clone()
        self._atlas_reliability_runtime_mapped = self._atlas_reliability_base.detach().clone()
        self._atlas_reliability_effective = self._atlas_reliability_base.detach().clone()
        self._atlas_reliability_runtime = self._atlas_reliability_effective.detach().clone()
        self._atlas_class = torch.tensor(atlas_init.atlas_class, dtype=torch.long, device=device)
        self._atlas_anisotropy_ref = torch.tensor(atlas_init.anisotropy_ref, dtype=torch.float32, device=device)
        self._atlas_neighbor_indices = torch.tensor(atlas_init.neighbor_indices, dtype=torch.long, device=device)
        self._atlas_view_weights = torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.float32, device=device)
        self._atlas_view_counts = torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.int32, device=device)
        self._atlas_node_photo_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_visibility_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_obs_quality_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_support_consistency_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_finite_projection_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_ref_consistency_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_observed_score_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_updated_recently = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_observed_count = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_node_support_consistent_count = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_observed_mask = torch.zeros_like(self._atlas_reliability_base, dtype=torch.bool)
        self._atlas_refresh_node_photo_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_node_visibility_ema = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_obs_quality = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_node_observed_count = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_node_support_consistent_ratio = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_node_coverage_ratio = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_node_ambiguity = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_override_weight = torch.zeros_like(self._atlas_reliability_base)
        self._atlas_refresh_runtime_override_mask = torch.zeros_like(self._atlas_reliability_base, dtype=torch.bool)
        self._atlas_refresh_done = False
        self._atlas_source_path = atlas_init.source_path
        self._atlas_hash_metadata = dict(atlas_init.hash_info) if getattr(atlas_init, "hash_info", None) else None
        self._invalidate_atlas_spatial_hash()

    def _append_atlas_bindings(
        self,
        atlas_node_ids: torch.Tensor,
        atlas_state: torch.Tensor,
        photo_ema: torch.Tensor | None = None,
        visibility_ema: torch.Tensor | None = None,
        high_residual_count: torch.Tensor | None = None,
        low_residual_count: torch.Tensor | None = None,
        promotion_streak: torch.Tensor | None = None,
        demotion_streak: torch.Tensor | None = None,
        recovery_streak: torch.Tensor | None = None,
        last_transition_iter: torch.Tensor | None = None,
        gc_fail_count: torch.Tensor | None = None,
        drift_flag: torch.Tensor | None = None,
        drift_count: torch.Tensor | None = None,
        state_cooldown: torch.Tensor | None = None,
        active_lifetime: torch.Tensor | None = None,
        ref_camera: torch.Tensor | None = None,
        ref_score: torch.Tensor | None = None,
        last_good_node_ids: torch.Tensor | None = None,
        pending_ref_camera: torch.Tensor | None = None,
        pending_ref_tau: torch.Tensor | None = None,
        pending_retry_count: torch.Tensor | None = None,
        last_pending_iter: torch.Tensor | None = None,
        active_provenance: torch.Tensor | None = None,
    ):
        if self._atlas_positions.numel() == 0:
            return
        self._atlas_node_ids = torch.cat((self._atlas_node_ids, atlas_node_ids.to(self._atlas_node_ids.device, dtype=torch.long)), dim=0)
        self._atlas_state = torch.cat((self._atlas_state, atlas_state.to(self._atlas_state.device, dtype=torch.long)), dim=0)
        appended = atlas_node_ids.shape[0]
        if photo_ema is None:
            photo_ema = torch.zeros((appended,), dtype=torch.float32, device=self._atlas_photo_ema.device)
        if visibility_ema is None:
            visibility_ema = torch.zeros((appended,), dtype=torch.float32, device=self._atlas_visibility_ema.device)
        if high_residual_count is None:
            high_residual_count = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if low_residual_count is None:
            low_residual_count = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if promotion_streak is None:
            promotion_streak = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if demotion_streak is None:
            demotion_streak = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if recovery_streak is None:
            recovery_streak = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if last_transition_iter is None:
            last_transition_iter = torch.full((appended,), -1, dtype=torch.long, device=self._atlas_node_ids.device)
        if gc_fail_count is None:
            gc_fail_count = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if drift_flag is None:
            drift_flag = torch.zeros((appended,), dtype=torch.bool, device=self._atlas_drift_flag.device)
        if drift_count is None:
            drift_count = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if state_cooldown is None:
            state_cooldown = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if active_lifetime is None:
            active_lifetime = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if ref_camera is None:
            ref_camera = torch.full((appended,), -1, dtype=torch.long, device=self._atlas_node_ids.device)
        if ref_score is None:
            ref_score = torch.zeros((appended,), dtype=torch.float32, device=self._atlas_photo_ema.device)
        if last_good_node_ids is None:
            last_good_node_ids = atlas_node_ids.detach().clone()
        if pending_ref_camera is None:
            pending_ref_camera = ref_camera.detach().clone()
        if pending_ref_tau is None:
            pending_ref_tau = torch.full((appended,), float("nan"), dtype=torch.float32, device=self._atlas_photo_ema.device)
        if pending_retry_count is None:
            pending_retry_count = torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device)
        if last_pending_iter is None:
            last_pending_iter = torch.full((appended,), -1, dtype=torch.long, device=self._atlas_node_ids.device)
        if active_provenance is None:
            active_provenance = torch.where(
                atlas_state.to(device=self._atlas_node_ids.device, dtype=torch.long) == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                torch.full((appended,), ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP, dtype=torch.long, device=self._atlas_node_ids.device),
                torch.zeros((appended,), dtype=torch.long, device=self._atlas_node_ids.device),
            )
        self._atlas_photo_ema = torch.cat((self._atlas_photo_ema, photo_ema.to(self._atlas_photo_ema.device, dtype=torch.float32)), dim=0)
        self._atlas_visibility_ema = torch.cat((self._atlas_visibility_ema, visibility_ema.to(self._atlas_visibility_ema.device, dtype=torch.float32)), dim=0)
        self._atlas_high_residual_count = torch.cat((self._atlas_high_residual_count, high_residual_count.to(self._atlas_high_residual_count.device, dtype=torch.long)), dim=0)
        self._atlas_low_residual_count = torch.cat((self._atlas_low_residual_count, low_residual_count.to(self._atlas_low_residual_count.device, dtype=torch.long)), dim=0)
        self._atlas_promotion_streak = torch.cat((self._atlas_promotion_streak, promotion_streak.to(self._atlas_promotion_streak.device, dtype=torch.long)), dim=0)
        self._atlas_demotion_streak = torch.cat((self._atlas_demotion_streak, demotion_streak.to(self._atlas_demotion_streak.device, dtype=torch.long)), dim=0)
        self._atlas_recovery_streak = torch.cat((self._atlas_recovery_streak, recovery_streak.to(self._atlas_recovery_streak.device, dtype=torch.long)), dim=0)
        self._atlas_last_transition_iter = torch.cat((self._atlas_last_transition_iter, last_transition_iter.to(self._atlas_last_transition_iter.device, dtype=torch.long)), dim=0)
        self._atlas_gc_fail_count = torch.cat((self._atlas_gc_fail_count, gc_fail_count.to(self._atlas_gc_fail_count.device, dtype=torch.long)), dim=0)
        self._atlas_drift_flag = torch.cat((self._atlas_drift_flag, drift_flag.to(self._atlas_drift_flag.device, dtype=torch.bool)), dim=0)
        self._atlas_drift_count = torch.cat((self._atlas_drift_count, drift_count.to(self._atlas_drift_count.device, dtype=torch.long)), dim=0)
        self._atlas_state_cooldown = torch.cat((self._atlas_state_cooldown, state_cooldown.to(self._atlas_state_cooldown.device, dtype=torch.long)), dim=0)
        self._atlas_active_lifetime = torch.cat((self._atlas_active_lifetime, active_lifetime.to(self._atlas_active_lifetime.device, dtype=torch.long)), dim=0)
        self._atlas_ref_camera = torch.cat((self._atlas_ref_camera, ref_camera.to(self._atlas_ref_camera.device, dtype=torch.long)), dim=0)
        self._atlas_ref_score = torch.cat((self._atlas_ref_score, ref_score.to(self._atlas_ref_score.device, dtype=torch.float32)), dim=0)
        self._atlas_last_good_node_ids = torch.cat((self._atlas_last_good_node_ids, last_good_node_ids.to(self._atlas_last_good_node_ids.device, dtype=torch.long)), dim=0)
        self._atlas_pending_ref_camera = torch.cat((self._atlas_pending_ref_camera, pending_ref_camera.to(self._atlas_pending_ref_camera.device, dtype=torch.long)), dim=0)
        self._atlas_pending_ref_tau = torch.cat((self._atlas_pending_ref_tau, pending_ref_tau.to(self._atlas_pending_ref_tau.device, dtype=torch.float32)), dim=0)
        self._atlas_pending_retry_count = torch.cat((self._atlas_pending_retry_count, pending_retry_count.to(self._atlas_pending_retry_count.device, dtype=torch.long)), dim=0)
        self._atlas_last_pending_iter = torch.cat((self._atlas_last_pending_iter, last_pending_iter.to(self._atlas_last_pending_iter.device, dtype=torch.long)), dim=0)
        self._atlas_active_provenance = torch.cat(
            (
                self._atlas_active_provenance,
                active_provenance.to(self._atlas_active_provenance.device, dtype=torch.long),
            ),
            dim=0,
        )

    def _prune_atlas_bindings(self, valid_points_mask: torch.Tensor):
        if self._atlas_positions.numel() == 0:
            return
        self._atlas_node_ids = self._atlas_node_ids[valid_points_mask]
        self._atlas_state = self._atlas_state[valid_points_mask]
        self._atlas_photo_ema = self._atlas_photo_ema[valid_points_mask]
        self._atlas_visibility_ema = self._atlas_visibility_ema[valid_points_mask]
        self._atlas_high_residual_count = self._atlas_high_residual_count[valid_points_mask]
        self._atlas_low_residual_count = self._atlas_low_residual_count[valid_points_mask]
        self._atlas_promotion_streak = self._atlas_promotion_streak[valid_points_mask]
        self._atlas_demotion_streak = self._atlas_demotion_streak[valid_points_mask]
        self._atlas_recovery_streak = self._atlas_recovery_streak[valid_points_mask]
        self._atlas_last_transition_iter = self._atlas_last_transition_iter[valid_points_mask]
        self._atlas_gc_fail_count = self._atlas_gc_fail_count[valid_points_mask]
        self._atlas_drift_flag = self._atlas_drift_flag[valid_points_mask]
        self._atlas_drift_count = self._atlas_drift_count[valid_points_mask]
        self._atlas_state_cooldown = self._atlas_state_cooldown[valid_points_mask]
        self._atlas_active_lifetime = self._atlas_active_lifetime[valid_points_mask]
        self._atlas_ref_camera = self._atlas_ref_camera[valid_points_mask]
        self._atlas_ref_score = self._atlas_ref_score[valid_points_mask]
        self._atlas_last_good_node_ids = self._atlas_last_good_node_ids[valid_points_mask]
        self._atlas_pending_ref_camera = self._atlas_pending_ref_camera[valid_points_mask]
        self._atlas_pending_ref_tau = self._atlas_pending_ref_tau[valid_points_mask]
        self._atlas_pending_retry_count = self._atlas_pending_retry_count[valid_points_mask]
        self._atlas_last_pending_iter = self._atlas_last_pending_iter[valid_points_mask]
        self._atlas_active_provenance = self._atlas_active_provenance[valid_points_mask]

    def _pad_selection_mask(self, allowed_mask: torch.Tensor | None, target_size: int):
        if allowed_mask is None:
            return None
        allowed_mask = allowed_mask.to(device=self._device(), dtype=torch.bool)
        if allowed_mask.shape[0] == target_size:
            return allowed_mask
        padded = torch.zeros((target_size,), dtype=torch.bool, device=allowed_mask.device)
        copy_count = min(int(allowed_mask.shape[0]), int(target_size))
        if copy_count > 0:
            padded[:copy_count] = allowed_mask[:copy_count]
        return padded

    def _uniform_rescale_log_scaling(self, log_scaling: torch.Tensor, factor: float):
        # Uniform rescaling keeps the orientation/anisotropy ratios unchanged, so
        # densification stays compatible with the scale-free atlas shape prior.
        return self.scaling_inverse_activation(self.scaling_activation(log_scaling).clamp_min(1e-4) * float(factor))

    def _scale_free_log_anisotropy(self, scales: torch.Tensor):
        safe_scales = torch.sort(scales.clamp_min(1e-8), dim=1, descending=True).values
        return torch.stack(
            (
                torch.log(safe_scales[:, 0] / safe_scales[:, 1]),
                torch.log(safe_scales[:, 1] / safe_scales[:, 2]),
            ),
            dim=1,
        )

    def _select_topk_mask(self, candidate_mask: torch.Tensor, scores: torch.Tensor, k: int):
        selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
        if k <= 0 or candidate_mask.numel() == 0 or not torch.any(candidate_mask):
            return selected
        candidate_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
        if candidate_idx.numel() <= int(k):
            selected[candidate_idx] = True
            return selected
        topk = torch.topk(scores[candidate_idx], k=int(k), largest=True).indices
        selected[candidate_idx[topk]] = True
        return selected

    def _padded_grad_norm(self, grads: torch.Tensor | None):
        point_count = int(self.get_xyz.shape[0])
        grad_norm = torch.zeros((point_count,), dtype=torch.float32, device=self._device())
        if grads is None or point_count == 0:
            return grad_norm
        if grads.ndim > 1:
            source = torch.norm(grads.detach(), dim=-1)
        else:
            source = grads.detach().reshape(-1)
        copy_count = min(int(source.shape[0]), point_count)
        if copy_count > 0:
            source = source[:copy_count].to(device=grad_norm.device, dtype=torch.float32)
            source = torch.where(torch.isfinite(source), source, torch.zeros_like(source))
            grad_norm[:copy_count] = source
        return grad_norm

    def _ensure_atlas_runtime_state(self):
        if self._atlas_positions.numel() == 0:
            self._clear_atlas_bindings()
            return
        device = self._atlas_positions.device
        node_count = int(self._atlas_positions.shape[0])
        gaussian_count = int(self._atlas_node_ids.shape[0]) if self._atlas_node_ids.numel() > 0 else 0

        def ensure_tensor(name, shape, dtype, fill_value=0):
            value = getattr(self, name, None)
            if value is None or value.numel() == 0 or tuple(value.shape) != tuple(shape):
                if dtype == torch.bool:
                    tensor = torch.full(shape, bool(fill_value), dtype=dtype, device=device)
                else:
                    tensor = torch.full(shape, float(fill_value), dtype=dtype, device=device)
                setattr(self, name, tensor)

        ensure_tensor("_atlas_raw_score", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_reliability_base", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_reliability_runtime_raw", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_reliability_runtime_mapped", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_reliability_effective", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_reliability_runtime", (node_count,), torch.float32, 0.0)
        legacy_runtime_valid = (
            self._atlas_reliability_runtime.numel() == node_count
            and float(self._atlas_reliability_runtime.abs().sum().item()) > 0.0
        )
        if self._atlas_reliability_base.numel() == 0 or float(self._atlas_reliability_base.abs().sum().item()) == 0.0:
            if legacy_runtime_valid:
                self._atlas_reliability_base = self._atlas_reliability_runtime.detach().clone()
            else:
                self._atlas_reliability_base = torch.ones((node_count,), dtype=torch.float32, device=device)
        if self._atlas_raw_score.numel() == 0 or float(self._atlas_raw_score.abs().sum().item()) == 0.0:
            self._atlas_raw_score = self._atlas_reliability_base.detach().clone()
        if self._atlas_reliability_runtime_raw.numel() == 0 or float(self._atlas_reliability_runtime_raw.abs().sum().item()) == 0.0:
            self._atlas_reliability_runtime_raw = self._atlas_reliability_base.detach().clone()
        if self._atlas_reliability_runtime_mapped.numel() == 0 or float(self._atlas_reliability_runtime_mapped.abs().sum().item()) == 0.0:
            self._atlas_reliability_runtime_mapped = (
                self._atlas_reliability_runtime.detach().clone()
                if legacy_runtime_valid
                else self._atlas_reliability_base.detach().clone()
            )
        if self._atlas_reliability_effective.numel() == 0 or float(self._atlas_reliability_effective.abs().sum().item()) == 0.0:
            self._atlas_reliability_effective = (
                self._atlas_reliability_runtime.detach().clone()
                if legacy_runtime_valid
                else self._atlas_reliability_runtime_mapped.detach().clone()
            )
        self._atlas_reliability_runtime = self._atlas_reliability_effective.detach().clone()
        if getattr(self, "_atlas_neighbor_indices", None) is None or self._atlas_neighbor_indices.numel() == 0:
            self._atlas_neighbor_indices = torch.arange(node_count, dtype=torch.long, device=device)[:, None]
        ensure_tensor("_atlas_node_photo_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_visibility_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_obs_quality_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_support_consistency_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_finite_projection_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_ref_consistency_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_observed_score_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_updated_recently", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_observed_count", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_node_support_consistent_count", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_observed_mask", (node_count,), torch.bool, False)
        ensure_tensor("_atlas_refresh_node_photo_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_node_visibility_ema", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_obs_quality", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_node_observed_count", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_node_support_consistent_ratio", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_node_coverage_ratio", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_node_ambiguity", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_override_weight", (node_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_refresh_runtime_override_mask", (node_count,), torch.bool, False)
        ensure_tensor("_atlas_photo_ema", (gaussian_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_visibility_ema", (gaussian_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_high_residual_count", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_low_residual_count", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_promotion_streak", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_demotion_streak", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_recovery_streak", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_last_transition_iter", (gaussian_count,), torch.long, -1)
        ensure_tensor("_atlas_gc_fail_count", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_drift_flag", (gaussian_count,), torch.bool, False)
        ensure_tensor("_atlas_drift_count", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_state_cooldown", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_active_lifetime", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_ref_camera", (gaussian_count,), torch.long, -1)
        ensure_tensor("_atlas_ref_score", (gaussian_count,), torch.float32, 0.0)
        ensure_tensor("_atlas_last_good_node_ids", (gaussian_count,), torch.long, -1)
        ensure_tensor("_atlas_pending_ref_camera", (gaussian_count,), torch.long, -1)
        ensure_tensor("_atlas_pending_ref_tau", (gaussian_count,), torch.float32, float("nan"))
        ensure_tensor("_atlas_pending_retry_count", (gaussian_count,), torch.long, 0)
        ensure_tensor("_atlas_last_pending_iter", (gaussian_count,), torch.long, -1)
        ensure_tensor("_atlas_active_provenance", (gaussian_count,), torch.long, ACTIVE_PROVENANCE_NONE)
        if gaussian_count > 0 and self._atlas_node_ids.numel() == gaussian_count:
            missing_active_origin = (
                (self._atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (self._atlas_active_provenance <= ACTIVE_PROVENANCE_NONE)
            )
            if torch.any(missing_active_origin):
                self._atlas_active_provenance[missing_active_origin] = ACTIVE_PROVENANCE_FROM_RESTORE_CHECKPOINT
            inactive_origin = self._atlas_state != GAUSSIAN_STATE_UNSTABLE_ACTIVE
            if torch.any(inactive_origin):
                self._atlas_active_provenance[inactive_origin] = ACTIVE_PROVENANCE_NONE
            valid_last_good = (
                (self._atlas_last_good_node_ids >= 0)
                & (self._atlas_last_good_node_ids < node_count)
            )
            valid_current = (
                (self._atlas_node_ids >= 0)
                & (self._atlas_node_ids < node_count)
            )
            repair_last_good = (~valid_last_good) & valid_current
            if torch.any(repair_last_good):
                self._atlas_last_good_node_ids[repair_last_good] = self._atlas_node_ids[repair_last_good]
            repair_pending_ref = (self._atlas_pending_ref_camera < 0) & (self._atlas_ref_camera >= 0)
            if torch.any(repair_pending_ref):
                self._atlas_pending_ref_camera[repair_pending_ref] = self._atlas_ref_camera[repair_pending_ref]
        view_cols = 0
        if getattr(self, "_atlas_view_weights", None) is not None and self._atlas_view_weights.ndim == 2:
            view_cols = int(self._atlas_view_weights.shape[1])
        if getattr(self, "_atlas_view_counts", None) is not None and self._atlas_view_counts.ndim == 2:
            view_cols = max(view_cols, int(self._atlas_view_counts.shape[1]))
        ensure_tensor("_atlas_view_weights", (node_count, view_cols), torch.float32, 0.0)
        ensure_tensor("_atlas_view_counts", (node_count, view_cols), torch.int32, 0)
        if not hasattr(self, "_atlas_refresh_done"):
            self._atlas_refresh_done = False
        if not hasattr(self, "_atlas_state_update_iter"):
            self._atlas_state_update_iter = 0

    def _ensure_atlas_view_capacity(self, min_columns: int):
        self._ensure_atlas_runtime_state()
        min_columns = int(max(min_columns, 0))
        node_count = int(self._atlas_positions.shape[0])
        current_cols = int(self._atlas_view_weights.shape[1]) if self._atlas_view_weights.ndim == 2 else 0
        if min_columns <= current_cols:
            return
        device = self._atlas_positions.device
        new_weights = torch.zeros((node_count, min_columns), dtype=torch.float32, device=device)
        new_counts = torch.zeros((node_count, min_columns), dtype=torch.int32, device=device)
        if current_cols > 0:
            new_weights[:, :current_cols] = self._atlas_view_weights
            new_counts[:, :current_cols] = self._atlas_view_counts
        self._atlas_view_weights = new_weights
        self._atlas_view_counts = new_counts

    def _compute_support_consistency_score(self):
        if not self.has_atlas_bindings or self._xyz.numel() == 0:
            return torch.zeros((self._xyz.shape[0],), dtype=torch.float32, device=self._device())
        atlas_positions = self.get_gaussian_atlas_positions.detach()
        atlas_support = self.get_gaussian_atlas_support.detach()
        atlas_radius = self.get_gaussian_atlas_radius.detach().clamp_min(1e-6)
        delta = self._xyz.detach() - atlas_positions
        projected_delta = torch.bmm(atlas_support, delta.unsqueeze(-1)).squeeze(-1)
        orthogonal_delta = delta - projected_delta
        projected_norm = torch.linalg.norm(projected_delta, dim=1)
        orthogonal_norm = torch.linalg.norm(orthogonal_delta, dim=1)
        orthogonal_score = torch.exp(-0.5 * (orthogonal_norm / (0.55 * atlas_radius).clamp_min(1e-6)).square())
        projected_score = torch.exp(-0.5 * (projected_norm / (2.5 * atlas_radius).clamp_min(1e-6)).square())
        return (orthogonal_score * projected_score).clamp(0.0, 1.0)

    def _compute_visible_support_consistency(self, visible_mask: torch.Tensor):
        if not self.has_atlas_bindings or visible_mask.numel() == 0:
            return torch.zeros_like(visible_mask, dtype=torch.bool)
        support_score = self._compute_support_consistency_score()
        return visible_mask & (support_score >= 0.35)

    def _evaluate_reattach_accept_policy(
        self,
        point_indices: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_dist: torch.Tensor,
        radius_mult: float,
        preaccept_mask: torch.Tensor | None = None,
        pending_candidate_mask: torch.Tensor | None = None,
        force_relaxed: bool = False,
    ):
        device = self._device()
        count = int(candidate_ids.shape[0])
        accept = torch.zeros((count,), dtype=torch.bool, device=device)
        stats = {
            "reattach_quality_valid": 0,
            "reattach_quality_reject_reliability": 0,
            "reattach_quality_reject_ref": 0,
            "reattach_quality_reject_class": 0,
            "reattach_quality_reliability_mean": 0.0,
            "reattach_quality_ref_mean": 0.0,
        }
        if count == 0 or self._atlas_positions.numel() == 0:
            return accept, stats

        point_indices = point_indices.to(device=device, dtype=torch.long)
        candidate_ids = candidate_ids.to(device=device, dtype=torch.long)
        candidate_dist = candidate_dist.to(device=device, dtype=torch.float32)
        valid_ids = (candidate_ids >= 0) & (candidate_ids < self._atlas_positions.shape[0])
        safe_ids = candidate_ids.clamp(0, max(int(self._atlas_positions.shape[0]) - 1, 0))
        candidate_radius = self._atlas_radius.detach().clamp_min(1e-6).index_select(0, safe_ids)
        distance_accept = valid_ids & torch.isfinite(candidate_dist) & (
            candidate_dist <= float(radius_mult) * candidate_radius
        )
        if preaccept_mask is not None:
            distance_accept = distance_accept & preaccept_mask.to(device=device, dtype=torch.bool)

        runtime_reliability = self._atlas_reliability_effective.detach()
        if runtime_reliability.numel() != self._atlas_positions.shape[0]:
            runtime_reliability = self._atlas_reliability_base.detach()
        candidate_reliability = runtime_reliability.index_select(0, safe_ids).clamp(0.0, 1.0)
        finite_runtime = runtime_reliability[torch.isfinite(runtime_reliability)]
        if finite_runtime.numel() > 0:
            reliability_floor = max(0.05, min(float(torch.median(finite_runtime).item()) * 0.35, 0.25))
        else:
            reliability_floor = 0.05

        ref_camera = self._resolve_retry_ref_camera(point_indices).to(device=device, dtype=torch.long)
        node_ref_consistency = self._atlas_node_ref_consistency_ema.detach().index_select(0, safe_ids).clamp(0.0, 1.0)
        ref_view_consistency = torch.zeros_like(node_ref_consistency)
        ref_known = torch.zeros((count,), dtype=torch.bool, device=device)
        if self._atlas_view_weights.ndim == 2 and self._atlas_view_weights.shape[1] > 0:
            valid_ref = (ref_camera >= 0) & (ref_camera < self._atlas_view_weights.shape[1])
            if torch.any(valid_ref):
                ref_view_consistency[valid_ref] = self._atlas_view_weights.detach()[
                    safe_ids[valid_ref],
                    ref_camera[valid_ref],
                ].clamp(0.0, 1.0)
                ref_known[valid_ref] = True
        ref_consistency = torch.maximum(node_ref_consistency, ref_view_consistency)
        ref_consistency = torch.where(ref_known, ref_consistency, torch.full_like(ref_consistency, 0.5))

        current_ids = self._atlas_node_ids[point_indices].to(device=device, dtype=torch.long)
        last_good = self._atlas_last_good_node_ids[point_indices].to(device=device, dtype=torch.long)
        use_last_good = ~((current_ids >= 0) & (current_ids < self._atlas_positions.shape[0]))
        current_ids = torch.where(use_last_good, last_good, current_ids)
        current_valid = (current_ids >= 0) & (current_ids < self._atlas_positions.shape[0])
        safe_current_ids = current_ids.clamp(0, max(int(self._atlas_positions.shape[0]) - 1, 0))
        current_class = self._atlas_class.detach().index_select(0, safe_current_ids)
        candidate_class = self._atlas_class.detach().index_select(0, safe_ids)
        same_class = current_class == candidate_class
        unstable_bridge = (current_class == ATLAS_CLASS_UNSTABLE) | (candidate_class == ATLAS_CLASS_UNSTABLE)
        class_compatible = (~current_valid) | same_class | unstable_bridge

        retry_count = torch.maximum(
            self._atlas_gc_fail_count.detach()[point_indices],
            self._atlas_pending_retry_count.detach()[point_indices],
        )
        pending_candidate_mask = (
            torch.zeros((count,), dtype=torch.bool, device=device)
            if pending_candidate_mask is None
            else pending_candidate_mask.to(device=device, dtype=torch.bool)
        )
        candidate_state = self._atlas_state.detach()[point_indices]
        relaxed = (
            torch.full((count,), bool(force_relaxed), dtype=torch.bool, device=device)
            | pending_candidate_mask
            | (retry_count > 0)
            | (candidate_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING)
        )
        active_unstable = candidate_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        reliability_ok = (
            candidate_reliability >= reliability_floor
        ) | (
            relaxed & (candidate_reliability >= 0.5 * reliability_floor)
        ) | (
            (ref_consistency >= 0.35) & (candidate_reliability >= 0.025)
        )
        ref_ok = (~ref_known) | (ref_consistency >= 0.10) | (relaxed & (ref_consistency >= 0.03))
        class_ok = class_compatible | (relaxed & (candidate_reliability >= reliability_floor)) | (
            active_unstable & (ref_consistency >= 0.20)
        )
        accept = distance_accept & reliability_ok & ref_ok & class_ok

        considered = distance_accept & valid_ids
        stats["reattach_quality_valid"] = int(accept.sum().item())
        stats["reattach_quality_reject_reliability"] = int((considered & (~reliability_ok)).sum().item())
        stats["reattach_quality_reject_ref"] = int((considered & reliability_ok & (~ref_ok)).sum().item())
        stats["reattach_quality_reject_class"] = int((considered & reliability_ok & ref_ok & (~class_ok)).sum().item())
        stats["reattach_quality_reliability_mean"] = float(candidate_reliability[valid_ids].mean().item()) if torch.any(valid_ids) else 0.0
        stats["reattach_quality_ref_mean"] = float(ref_consistency[valid_ids].mean().item()) if torch.any(valid_ids) else 0.0
        return accept, stats

    def _combine_node_obs_quality(
        self,
        photo_residual: torch.Tensor,
        support_consistency: torch.Tensor,
        finite_projection: torch.Tensor,
        ref_consistency: torch.Tensor,
    ):
        photo_quality = torch.exp(-photo_residual.clamp(0.0, 4.0))
        return (
            0.45 * photo_quality
            + 0.20 * support_consistency.clamp(0.0, 1.0)
            + 0.20 * finite_projection.clamp(0.0, 1.0)
            + 0.15 * ref_consistency.clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)

    def _combine_node_observed_score(
        self,
        visibility_ema: torch.Tensor,
        obs_quality_ema: torch.Tensor,
        updated_recently: torch.Tensor,
    ):
        recent_weight = (0.25 + 0.75 * updated_recently.clamp(0.0, 1.0)).clamp(0.25, 1.0)
        return (
            0.40 * visibility_ema.clamp(0.0, 1.0)
            + 0.60 * (obs_quality_ema.clamp(0.0, 1.0) * recent_weight)
        ).clamp(0.0, 1.0)

    def _update_runtime_ema_tensor(
        self,
        tensor: torch.Tensor,
        current: torch.Tensor,
        updated_mask: torch.Tensor,
        ema_decay: float,
        idle_alpha: float = 0.0,
        idle_target: float | torch.Tensor = 0.0,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
    ):
        if tensor.numel() == 0:
            return
        decay = float(np.clip(ema_decay, 0.0, 0.9999))
        alpha = 1.0 - decay
        if torch.any(updated_mask):
            tensor[updated_mask] = decay * tensor[updated_mask] + alpha * current[updated_mask]
        if idle_alpha > 0.0:
            idle_mask = ~updated_mask
            if torch.any(idle_mask):
                if torch.is_tensor(idle_target):
                    target = idle_target.to(device=tensor.device, dtype=tensor.dtype)
                else:
                    target = torch.full_like(tensor, float(idle_target))
                tensor[idle_mask] = (1.0 - idle_alpha) * tensor[idle_mask] + idle_alpha * target[idle_mask]
        if clamp_min is not None or clamp_max is not None:
            min_value = clamp_min if clamp_min is not None else None
            max_value = clamp_max if clamp_max is not None else None
            tensor.clamp_(min=min_value, max=max_value)

    def _consume_runtime_observation_cache(self, fallback_visible_mask: torch.Tensor):
        default_visibility = fallback_visible_mask.to(dtype=torch.float32)
        payload = {
            "candidate_mask": fallback_visible_mask.detach().clone(),
            "projected_mask": fallback_visible_mask.detach().clone(),
            "observed_mask": fallback_visible_mask.detach().clone(),
            "visibility_contribution": default_visibility.detach().clone(),
            "dark_region_mask": torch.zeros_like(fallback_visible_mask, dtype=torch.bool),
            "smooth_region_mask": torch.zeros_like(fallback_visible_mask, dtype=torch.bool),
            "floater_region_mask": torch.zeros_like(fallback_visible_mask, dtype=torch.bool),
            "patch_quality_score": torch.zeros_like(default_visibility),
            "mask_nonzero_ratio": torch.zeros_like(default_visibility),
            "bg_like_ratio": torch.zeros_like(default_visibility),
            "background_like_ratio": torch.zeros_like(default_visibility),
            "thin_support_like_ratio": torch.zeros_like(default_visibility),
            "photo_signal_strength": torch.zeros_like(default_visibility),
        }
        cache = getattr(self, "_atlas_runtime_last_observation", None)
        self._atlas_runtime_last_observation = None
        if not isinstance(cache, dict):
            return payload

        def _read_mask(name: str, fallback: torch.Tensor):
            value = cache.get(name)
            if torch.is_tensor(value) and value.shape == fallback.shape:
                return value.detach().to(device=fallback.device, dtype=torch.bool)
            return fallback.detach().clone()

        def _read_value(name: str, fallback: torch.Tensor):
            value = cache.get(name)
            if torch.is_tensor(value) and value.shape == fallback.shape:
                return value.detach().to(device=fallback.device, dtype=torch.float32)
            return fallback.detach().clone()

        payload["candidate_mask"] = _read_mask("candidate_mask", payload["candidate_mask"])
        payload["projected_mask"] = _read_mask("projected_mask", payload["projected_mask"])
        payload["observed_mask"] = _read_mask("observed_mask", payload["observed_mask"])
        payload["dark_region_mask"] = _read_mask("dark_region_mask", payload["dark_region_mask"])
        payload["smooth_region_mask"] = _read_mask("smooth_region_mask", payload["smooth_region_mask"])
        payload["floater_region_mask"] = _read_mask("floater_region_mask", payload["floater_region_mask"])
        payload["visibility_contribution"] = _read_value("visibility_contribution", payload["visibility_contribution"]).clamp(0.0, 1.0)
        payload["patch_quality_score"] = _read_value("patch_quality_score", payload["patch_quality_score"]).clamp(0.0, 1.0)
        payload["mask_nonzero_ratio"] = _read_value("mask_nonzero_ratio", payload["mask_nonzero_ratio"]).clamp(0.0, 1.0)
        payload["bg_like_ratio"] = _read_value("bg_like_ratio", payload["bg_like_ratio"]).clamp(0.0, 1.0)
        payload["background_like_ratio"] = _read_value("background_like_ratio", payload["bg_like_ratio"]).clamp(0.0, 1.0)
        payload["thin_support_like_ratio"] = _read_value("thin_support_like_ratio", payload["thin_support_like_ratio"]).clamp(0.0, 1.0)
        payload["photo_signal_strength"] = _read_value("photo_signal_strength", payload["photo_signal_strength"]).clamp(0.0, 1.0)
        payload["projected_mask"] = payload["projected_mask"] & payload["candidate_mask"]
        payload["observed_mask"] = payload["observed_mask"] & payload["projected_mask"]
        payload["dark_region_mask"] = payload["dark_region_mask"] & payload["observed_mask"]
        payload["smooth_region_mask"] = payload["smooth_region_mask"] & payload["observed_mask"]
        payload["floater_region_mask"] = payload["floater_region_mask"] & payload["smooth_region_mask"]
        payload["visibility_contribution"] = torch.where(
            payload["projected_mask"],
            payload["visibility_contribution"],
            torch.zeros_like(payload["visibility_contribution"]),
        )
        for value_name in (
            "patch_quality_score",
            "mask_nonzero_ratio",
            "bg_like_ratio",
            "background_like_ratio",
            "thin_support_like_ratio",
            "photo_signal_strength",
        ):
            payload[value_name] = torch.where(
                payload["observed_mask"],
                payload[value_name],
                torch.zeros_like(payload[value_name]),
            )
        return payload

    def _snapshot_histogram_ratios(self, values: torch.Tensor, prefix: str):
        if values.numel() == 0:
            return {
                f"{prefix}_low": 0.0,
                f"{prefix}_mid": 0.0,
                f"{prefix}_high": 0.0,
            }
        values = values.detach().clamp(0.0, 1.0)
        return {
            f"{prefix}_low": float((values < 0.25).float().mean().item()),
            f"{prefix}_mid": float(((values >= 0.25) & (values < 0.65)).float().mean().item()),
            f"{prefix}_high": float((values >= 0.65).float().mean().item()),
        }

    def _snapshot_distribution_stats(self, values: torch.Tensor, prefix: str):
        empty = {
            f"{prefix}_min": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
        }
        if values.numel() == 0:
            return empty
        finite = values.detach().to(dtype=torch.float32)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return empty
        if finite.numel() == 1:
            value = float(finite[0].item())
            return {
                f"{prefix}_min": value,
                f"{prefix}_p10": value,
                f"{prefix}_p25": value,
                f"{prefix}_p50": value,
                f"{prefix}_p75": value,
                f"{prefix}_p90": value,
                f"{prefix}_max": value,
                f"{prefix}_mean": value,
                f"{prefix}_std": 0.0,
            }
        quantiles = torch.quantile(
            finite,
            torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], dtype=finite.dtype, device=finite.device),
        )
        return {
            f"{prefix}_min": float(finite.min().item()),
            f"{prefix}_p10": float(quantiles[0].item()),
            f"{prefix}_p25": float(quantiles[1].item()),
            f"{prefix}_p50": float(quantiles[2].item()),
            f"{prefix}_p75": float(quantiles[3].item()),
            f"{prefix}_p90": float(quantiles[4].item()),
            f"{prefix}_max": float(finite.max().item()),
            f"{prefix}_mean": float(finite.mean().item()),
            f"{prefix}_std": float(finite.std(unbiased=False).item()),
        }

    def _override_bucket_stats(self, values: torch.Tensor, override_mask: torch.Tensor, prefix: str):
        stats = {}
        if values.numel() == 0 or override_mask.numel() != values.numel():
            for bucket_name in ("low", "mid", "high"):
                stats[f"{prefix}_{bucket_name}_count"] = 0.0
                stats[f"{prefix}_{bucket_name}_override_count"] = 0.0
                stats[f"{prefix}_{bucket_name}_override_ratio"] = 0.0
            return stats
        values = values.detach().clamp(0.0, 1.0)
        override_mask = override_mask.detach().to(device=values.device, dtype=torch.bool)
        buckets = {
            "low": values < 0.25,
            "mid": (values >= 0.25) & (values < 0.65),
            "high": values >= 0.65,
        }
        for bucket_name, bucket_mask in buckets.items():
            bucket_count = float(bucket_mask.sum().item())
            override_count = float((bucket_mask & override_mask).sum().item())
            stats[f"{prefix}_{bucket_name}_count"] = bucket_count
            stats[f"{prefix}_{bucket_name}_override_count"] = override_count
            stats[f"{prefix}_{bucket_name}_override_ratio"] = override_count / max(bucket_count, 1.0)
        return stats

    def _robust_percentile_score(
        self,
        values: torch.Tensor,
        sample_mask: torch.Tensor,
        q_low: float = 0.10,
        q_high: float = 0.90,
        invert: bool = False,
        default: float = 0.0,
    ):
        if values.numel() == 0:
            return values.detach().to(dtype=torch.float32)
        values = values.detach().to(dtype=torch.float32)
        finite_mask = torch.isfinite(values)
        if sample_mask.numel() != values.numel():
            sample_mask = finite_mask
        else:
            sample_mask = sample_mask.detach().to(device=values.device, dtype=torch.bool) & finite_mask
        output = torch.full_like(values, float(default))
        if not torch.any(finite_mask):
            return output
        sample = values[sample_mask]
        if sample.numel() < 2:
            sample = values[finite_mask]
        if sample.numel() < 2:
            score = torch.zeros_like(values)
            score[finite_mask] = 1.0 if not invert else 0.0
            return score
        if sample.numel() < 4:
            low = sample.min()
            high = sample.max()
        else:
            q0 = float(max(0.0, min(q_low, 1.0)))
            q1 = float(max(q0 + 1e-3, min(q_high, 1.0)))
            low = torch.quantile(sample, q0)
            high = torch.quantile(sample, q1)
        scale = (high - low).abs().clamp_min(1e-6)
        score = ((values - low) / scale).clamp(0.0, 1.0)
        if invert:
            score = 1.0 - score
        output[finite_mask] = score[finite_mask]
        return output.clamp(0.0, 1.0)

    def _linear_tensor_gate(self, values: torch.Tensor, start: float, full: float):
        values = values.detach().to(dtype=torch.float32)
        start = float(start)
        full = float(full)
        if full <= start:
            return (values >= full).to(dtype=torch.float32)
        return ((values - start) / max(full - start, 1e-6)).clamp(0.0, 1.0)

    def _invalidate_atlas_spatial_hash(self, clear_metadata: bool = False):
        if clear_metadata:
            self._atlas_hash_metadata = None
        self._atlas_hash_lookup = {}
        self._atlas_hash_cell_size = 0.0
        self._atlas_hash_bbox_min = None
        self._atlas_hash_bucket_count = 0
        self._atlas_hash_neighbor_k = 0
        self._atlas_hash_source = "missing"
        self._atlas_hash_ready = False

    def _build_runtime_atlas_hash(self):
        if self._atlas_positions.numel() == 0:
            self._invalidate_atlas_spatial_hash()
            return

        positions = self._atlas_positions.detach().cpu().numpy().astype(np.float32, copy=False)
        radius = self._atlas_radius.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        hash_info = self._atlas_hash_metadata if isinstance(self._atlas_hash_metadata, dict) else None

        cell_size = None
        bbox_min = None
        buckets = None
        source = "recomputed_runtime"
        neighbor_k = int(self._atlas_neighbor_indices.shape[1]) if self._atlas_neighbor_indices.ndim == 2 else 0
        if hash_info and str(hash_info.get("kind", "")) == "voxel_hash":
            raw_cell_size = hash_info.get("cell_size", 0.0)
            raw_bbox_min = hash_info.get("bbox_min")
            raw_buckets = hash_info.get("buckets")
            if raw_cell_size is not None and float(raw_cell_size) > 0.0 and raw_bbox_min is not None and raw_buckets is not None:
                cell_size = float(raw_cell_size)
                bbox_min = np.asarray(raw_bbox_min, dtype=np.float32).reshape(3)
                buckets = raw_buckets
                source = str(hash_info.get("source", "atlas_hash.json"))
                neighbor_k = int(hash_info.get("neighbor_k", neighbor_k))

        if cell_size is None or bbox_min is None or buckets is None:
            finite_radius = radius[np.isfinite(radius) & (radius > 0.0)]
            if finite_radius.size > 0:
                cell_size = max(float(np.median(finite_radius)), 1e-6)
            else:
                span = float(np.linalg.norm(np.ptp(positions, axis=0))) if positions.shape[0] > 0 else 1.0
                cell_size = max(span / max(np.cbrt(max(positions.shape[0], 1)), 1.0), 1e-6)
            bbox_min = positions.min(axis=0).astype(np.float32)
            coords = np.floor((positions - bbox_min[None, :]) / cell_size).astype(np.int32)
            unique_coords, inverse = np.unique(coords, axis=0, return_inverse=True)
            buckets = []
            for bucket_id, coord in enumerate(unique_coords):
                node_ids = np.flatnonzero(inverse == bucket_id).astype(np.int64)
                buckets.append({"coord": coord.astype(int).tolist(), "node_ids": node_ids.tolist()})

        lookup = {}
        for bucket in buckets:
            coord = tuple(int(value) for value in bucket.get("coord", [0, 0, 0]))
            node_ids = np.asarray(bucket.get("node_ids", []), dtype=np.int64).reshape(-1)
            if node_ids.size == 0:
                continue
            lookup[coord] = node_ids

        self._atlas_hash_lookup = lookup
        self._atlas_hash_cell_size = float(max(cell_size, 1e-6))
        self._atlas_hash_bbox_min = torch.tensor(bbox_min, dtype=torch.float32, device=self._atlas_positions.device)
        self._atlas_hash_bucket_count = int(len(lookup))
        self._atlas_hash_neighbor_k = int(max(neighbor_k, 0))
        self._atlas_hash_source = source
        self._atlas_hash_ready = True

    def _ensure_atlas_spatial_hash(self):
        if not self.has_atlas_bindings:
            self._invalidate_atlas_spatial_hash()
            return
        if self._atlas_hash_ready and self._atlas_hash_bucket_count > 0 and self._atlas_hash_bbox_min is not None:
            return
        self._build_runtime_atlas_hash()

    def _prepare_atlas_gc_batch(self, include_pending_retries: bool = True):
        drifted = self._atlas_drift_flag
        pending = self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
        selected = drifted | (pending if bool(include_pending_retries) else torch.zeros_like(pending))
        if not torch.any(selected):
            return None
        drifted_idx = torch.nonzero(selected, as_tuple=False).squeeze(-1)
        return {
            "mask": selected,
            "indices": drifted_idx,
            "xyz": self._xyz.detach()[drifted_idx],
            "anchor_ids": self._atlas_node_ids[drifted_idx].detach(),
            "drift_candidate_count": int(torch.count_nonzero(drifted).item()),
            "pending_candidate_count": int(torch.count_nonzero(pending).item()),
            "selected_drift_candidate_count": int(torch.count_nonzero(drifted[drifted_idx]).item()),
            "selected_pending_candidate_count": int(torch.count_nonzero(pending[drifted_idx]).item()),
            "retry_pending_enabled": bool(include_pending_retries),
        }

    def _set_opacity_subset(self, point_indices: torch.Tensor, target_opacity: float):
        if point_indices is None or point_indices.numel() == 0:
            return 0
        point_indices = point_indices.to(device=self._opacity.device, dtype=torch.long)
        with torch.no_grad():
            clamped_opacity = float(np.clip(target_opacity, 1e-6, 1.0 - 1e-6))
            self._opacity[point_indices] = self.inverse_opacity_activation(
                torch.full(
                    (point_indices.shape[0], 1),
                    clamped_opacity,
                    dtype=self._opacity.dtype,
                    device=self._opacity.device,
                )
            )
        return int(point_indices.shape[0])

    def _voxel_hash_neighbor_offsets(self, probe_radius: int):
        offsets = []
        for dx in range(-probe_radius, probe_radius + 1):
            for dy in range(-probe_radius, probe_radius + 1):
                for dz in range(-probe_radius, probe_radius + 1):
                    offsets.append((dx, dy, dz))
        return offsets

    def _find_nearest_atlas_nodes_hashed(self, query_xyz: torch.Tensor, anchor_node_ids: torch.Tensor, probe_radius: int, chunk_size: int = 512):
        if query_xyz.shape[0] == 0 or self._atlas_positions.shape[0] == 0:
            empty_ids = torch.empty((0,), dtype=torch.long, device=self._device())
            empty_dist = torch.empty((0,), dtype=torch.float32, device=self._device())
            return empty_ids, empty_dist, {
                "mode": "empty",
                "bucket_queries": 0,
                "fallback_full_search": 0,
                "candidate_starvation_count": 0,
                "candidate_starvation_ratio": 0.0,
                "mean_candidate_count": 0.0,
                "max_candidate_count": 0,
            }

        self._ensure_atlas_spatial_hash()
        if not self._atlas_hash_ready or self._atlas_hash_bucket_count <= 0 or self._atlas_hash_bbox_min is None:
            nearest_ids, nearest_dist = self._chunked_nearest_atlas_nodes(query_xyz, chunk_size=chunk_size)
            return nearest_ids, nearest_dist, {
                "mode": "full_scan_fallback",
                "bucket_queries": 0,
                "fallback_full_search": int(query_xyz.shape[0]),
                "candidate_starvation_count": 0,
                "candidate_starvation_ratio": 0.0,
                "mean_candidate_count": float(self._atlas_positions.shape[0]),
                "max_candidate_count": int(self._atlas_positions.shape[0]),
            }

        device = query_xyz.device
        bbox_min = self._atlas_hash_bbox_min.to(device=device, dtype=query_xyz.dtype)
        coords = torch.floor((query_xyz - bbox_min.unsqueeze(0)) / float(self._atlas_hash_cell_size)).to(dtype=torch.int32)
        offsets = self._voxel_hash_neighbor_offsets(int(max(probe_radius, 1)))
        atlas_positions = self._atlas_positions
        neighbor_indices = self._atlas_neighbor_indices
        nearest_ids = torch.empty((query_xyz.shape[0],), dtype=torch.long, device=device)
        nearest_dist = torch.empty((query_xyz.shape[0],), dtype=torch.float32, device=device)
        candidate_counts = []
        fallback_full_search = 0
        bucket_queries = 0
        candidate_starvation_count = 0

        unique_coords, inverse = torch.unique(coords, dim=0, return_inverse=True)
        for group_idx in range(unique_coords.shape[0]):
            coord = unique_coords[group_idx]
            group_mask = inverse == group_idx
            group_indices = torch.nonzero(group_mask, as_tuple=False).squeeze(-1)
            coord_tuple = tuple(int(value) for value in coord.detach().cpu().tolist())
            bucket_candidate_ids = []
            for dx, dy, dz in offsets:
                neighbor_coord = (coord_tuple[0] + dx, coord_tuple[1] + dy, coord_tuple[2] + dz)
                node_ids = self._atlas_hash_lookup.get(neighbor_coord)
                if node_ids is None or len(node_ids) == 0:
                    continue
                bucket_candidate_ids.extend(int(node_id) for node_id in node_ids)
            bucket_queries += int(group_indices.shape[0])

            base_bucket_ids = np.unique(np.asarray(bucket_candidate_ids, dtype=np.int64)) if bucket_candidate_ids else np.zeros((0,), dtype=np.int64)
            for query_local_idx in group_indices.tolist():
                anchor_id = int(anchor_node_ids[query_local_idx].item()) if query_local_idx < anchor_node_ids.shape[0] else -1
                query_candidates = base_bucket_ids.tolist() if base_bucket_ids.size > 0 else []
                if 0 <= anchor_id < neighbor_indices.shape[0]:
                    query_candidates.append(anchor_id)
                    query_candidates.extend(int(node_id) for node_id in neighbor_indices[anchor_id].detach().cpu().tolist())
                if query_candidates:
                    candidate_ids = np.unique(np.asarray(query_candidates, dtype=np.int64))
                else:
                    candidate_ids = np.zeros((0,), dtype=np.int64)
                if candidate_ids.size == 0:
                    candidate_starvation_count += 1
                    fallback_full_search += 1
                    candidate_tensor = torch.arange(atlas_positions.shape[0], dtype=torch.long, device=device)
                else:
                    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
                candidate_counts.append(int(candidate_tensor.shape[0]))
                query_point = query_xyz[query_local_idx : query_local_idx + 1]
                candidate_pos = atlas_positions.index_select(0, candidate_tensor)
                d2 = torch.sum((candidate_pos - query_point) ** 2, dim=1)
                local_idx = torch.argmin(d2)
                nearest_ids[query_local_idx] = candidate_tensor[local_idx]
                nearest_dist[query_local_idx] = torch.sqrt(d2[local_idx].clamp_min(1e-8))

        candidate_counts_np = np.asarray(candidate_counts, dtype=np.float32) if candidate_counts else np.zeros((0,), dtype=np.float32)
        return nearest_ids, nearest_dist, {
            "mode": "voxel_hash",
            "bucket_queries": int(bucket_queries),
            "fallback_full_search": int(fallback_full_search),
            "candidate_starvation_count": int(candidate_starvation_count),
            "candidate_starvation_ratio": float(candidate_starvation_count / max(int(query_xyz.shape[0]), 1)),
            "mean_candidate_count": float(candidate_counts_np.mean()) if candidate_counts_np.size else 0.0,
            "max_candidate_count": int(candidate_counts_np.max()) if candidate_counts_np.size else 0,
        }

    def _chunked_nearest_atlas_nodes(self, query_xyz: torch.Tensor, chunk_size: int = 1024):
        if query_xyz.shape[0] == 0 or self._atlas_positions.shape[0] == 0:
            empty_ids = torch.empty((0,), dtype=torch.long, device=self._device())
            empty_dist = torch.empty((0,), dtype=torch.float32, device=self._device())
            return empty_ids, empty_dist

        nearest_ids = []
        nearest_dist = []
        atlas_positions = self._atlas_positions
        for start in range(0, query_xyz.shape[0], chunk_size):
            end = min(start + chunk_size, query_xyz.shape[0])
            d2 = torch.cdist(query_xyz[start:end], atlas_positions).square()
            values, indices = torch.min(d2, dim=1)
            nearest_ids.append(indices)
            nearest_dist.append(torch.sqrt(values.clamp_min(1e-8)))
        return torch.cat(nearest_ids, dim=0), torch.cat(nearest_dist, dim=0)

    def _estimate_pending_anchor_tau(self, point_indices: torch.Tensor, node_ids: torch.Tensor | None = None):
        device = self._device()
        if point_indices is None or point_indices.numel() == 0 or self._atlas_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=device)

        point_indices = point_indices.to(device=device, dtype=torch.long)
        if node_ids is None:
            node_ids = self._atlas_node_ids[point_indices]
        node_ids = node_ids.to(device=device, dtype=torch.long)
        tau = torch.full((point_indices.shape[0],), float("nan"), dtype=torch.float32, device=device)
        valid = (node_ids >= 0) & (node_ids < self._atlas_positions.shape[0])
        if not torch.any(valid):
            return tau

        valid_point_indices = point_indices[valid]
        valid_node_ids = node_ids[valid]
        anchor = self._atlas_positions[valid_node_ids].detach()
        points = self._xyz.detach()[valid_point_indices]
        delta = points - anchor
        basis_dir = self._atlas_basis[valid_node_ids].detach()[:, :, 0]
        basis_norm = torch.linalg.norm(basis_dir, dim=1, keepdim=True)
        delta_norm = torch.linalg.norm(delta, dim=1, keepdim=True)
        z_axis = torch.zeros_like(basis_dir)
        z_axis[:, 2] = 1.0
        valid_basis = torch.isfinite(basis_dir).all(dim=1, keepdim=True) & (basis_norm > 1e-6)
        valid_delta = torch.isfinite(delta).all(dim=1, keepdim=True) & (delta_norm > 1e-6)
        ray_dir = torch.where(
            valid_basis,
            basis_dir / basis_norm.clamp_min(1e-6),
            torch.where(valid_delta, delta / delta_norm.clamp_min(1e-6), z_axis),
        )
        tau[valid] = torch.sum(delta * ray_dir, dim=1)
        return tau

    def _record_pending_retry_metadata(self, point_indices: torch.Tensor, prefer_current_anchor: bool = True):
        if point_indices is None or point_indices.numel() == 0 or self._atlas_positions.numel() == 0:
            return
        device = self._device()
        point_indices = point_indices.to(device=device, dtype=torch.long)
        current_node_ids = self._atlas_node_ids[point_indices].detach()
        valid_current = (current_node_ids >= 0) & (current_node_ids < self._atlas_positions.shape[0])
        current_last_good = self._atlas_last_good_node_ids[point_indices]
        valid_last_good = (current_last_good >= 0) & (current_last_good < self._atlas_positions.shape[0])
        update_last_good = valid_current if bool(prefer_current_anchor) else (valid_current & (~valid_last_good))
        if torch.any(update_last_good):
            self._atlas_last_good_node_ids[point_indices[update_last_good]] = current_node_ids[update_last_good]

        ref_camera = self._atlas_ref_camera[point_indices]
        valid_ref = ref_camera >= 0
        if torch.any(valid_ref):
            self._atlas_pending_ref_camera[point_indices[valid_ref]] = ref_camera[valid_ref]

        last_good_node_ids = self._atlas_last_good_node_ids[point_indices]
        tau = self._estimate_pending_anchor_tau(point_indices, node_ids=last_good_node_ids)
        finite_tau = torch.isfinite(tau)
        if torch.any(finite_tau):
            self._atlas_pending_ref_tau[point_indices[finite_tau]] = tau[finite_tau]

        current_iter = int(max(getattr(self, "_atlas_state_update_iter", 0), 0))
        self._atlas_last_pending_iter[point_indices] = current_iter

    def _clear_pending_retry_metadata(self, point_indices: torch.Tensor, new_node_ids: torch.Tensor | None = None):
        if point_indices is None or point_indices.numel() == 0:
            return
        device = self._device()
        point_indices = point_indices.to(device=device, dtype=torch.long)
        if new_node_ids is None:
            new_node_ids = self._atlas_node_ids[point_indices].detach()
        new_node_ids = new_node_ids.to(device=device, dtype=torch.long)
        self._atlas_last_good_node_ids[point_indices] = new_node_ids
        self._atlas_pending_retry_count[point_indices] = 0
        self._atlas_pending_ref_camera[point_indices] = -1
        self._atlas_pending_ref_tau[point_indices] = float("nan")
        self._atlas_last_pending_iter[point_indices] = -1

    def _resolve_retry_ref_camera(self, point_indices: torch.Tensor):
        if point_indices is None or point_indices.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self._device())
        point_indices = point_indices.to(device=self._device(), dtype=torch.long)
        ref_camera = self._atlas_ref_camera[point_indices].detach().clone()
        pending_ref = self._atlas_pending_ref_camera[point_indices].detach()
        missing = ref_camera < 0
        if torch.any(missing):
            ref_camera[missing] = pending_ref[missing]
        return ref_camera

    def _ray_guided_pending_reattach(
        self,
        query_xyz: torch.Tensor,
        point_indices: torch.Tensor,
        anchor_node_ids: torch.Tensor,
        radius_mult: float,
    ):
        device = query_xyz.device
        count = int(query_xyz.shape[0])
        fallback_ids = torch.full((count,), -1, dtype=torch.long, device=device)
        fallback_dist = torch.full((count,), float("inf"), dtype=torch.float32, device=device)
        accept = torch.zeros((count,), dtype=torch.bool, device=device)
        if count == 0 or self._atlas_positions.numel() == 0:
            return fallback_ids, fallback_dist, accept, {
                "ray_guided_queries": 0,
                "ray_guided_ref_valid": 0,
                "ray_guided_active_queries": 0,
                "ray_guided_pending_queries": 0,
                "ray_guided_preaccept_count": 0,
                "ray_guided_empty_seed_count": 0,
                "ray_guided_mean_candidate_count": 0.0,
                "ray_guided_max_candidate_count": 0,
            }

        point_indices = point_indices.to(device=device, dtype=torch.long)
        anchor_node_ids = anchor_node_ids.to(device=device, dtype=torch.long)
        last_good_ids = self._atlas_last_good_node_ids[point_indices].to(device=device, dtype=torch.long)
        ref_camera = self._resolve_retry_ref_camera(point_indices).to(device=device, dtype=torch.long)
        stored_tau = self._atlas_pending_ref_tau[point_indices].to(device=device, dtype=torch.float32)
        atlas_positions = self._atlas_positions.detach()
        atlas_radius = self._atlas_radius.detach().clamp_min(1e-6)
        neighbor_indices = self._atlas_neighbor_indices.detach()
        candidate_counts = []
        ref_valid_count = int((ref_camera >= 0).sum().item())
        candidate_state = self._atlas_state.detach()[point_indices]
        active_query_count = int((candidate_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE).sum().item())
        pending_query_count = int((candidate_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item())
        empty_seed_count = 0

        for local_idx in range(count):
            query_point = query_xyz[local_idx]
            current_anchor = int(anchor_node_ids[local_idx].item())
            last_good = int(last_good_ids[local_idx].item())
            seed_ids = []
            for seed in (last_good, current_anchor):
                if 0 <= seed < atlas_positions.shape[0]:
                    seed_ids.append(seed)
                    if neighbor_indices.ndim == 2 and seed < neighbor_indices.shape[0]:
                        seed_ids.extend(int(value) for value in neighbor_indices[seed].detach().cpu().tolist())
            if seed_ids:
                candidate_ids_np = np.unique(np.asarray(seed_ids, dtype=np.int64))
                candidate_ids_np = candidate_ids_np[
                    (candidate_ids_np >= 0) & (candidate_ids_np < int(atlas_positions.shape[0]))
                ]
            else:
                candidate_ids_np = np.zeros((0,), dtype=np.int64)

            if candidate_ids_np.size == 0:
                empty_seed_count += 1
                nearest_id, nearest_dist = self._chunked_nearest_atlas_nodes(query_point[None, :], chunk_size=1024)
                fallback_ids[local_idx] = nearest_id[0]
                fallback_dist[local_idx] = nearest_dist[0]
                candidate_counts.append(1)
                continue

            candidate_tensor = torch.tensor(candidate_ids_np, dtype=torch.long, device=device)
            candidate_counts.append(int(candidate_tensor.shape[0]))
            base_anchor_id = last_good if 0 <= last_good < atlas_positions.shape[0] else current_anchor
            if not (0 <= base_anchor_id < atlas_positions.shape[0]):
                base_anchor_id = int(candidate_tensor[0].item())
            base_anchor = atlas_positions[base_anchor_id]
            query_delta = query_point - base_anchor
            basis_dir = self._atlas_basis[base_anchor_id].detach()[:, 0]
            basis_norm = torch.linalg.norm(basis_dir)
            delta_norm = torch.linalg.norm(query_delta)
            if bool(torch.isfinite(basis_norm).item()) and float(basis_norm.item()) > 1e-6:
                ray_dir = basis_dir / basis_norm.clamp_min(1e-6)
            elif bool(torch.isfinite(delta_norm).item()) and float(delta_norm.item()) > 1e-6:
                ray_dir = query_delta / delta_norm.clamp_min(1e-6)
            else:
                ray_dir = torch.tensor([0.0, 0.0, 1.0], dtype=query_xyz.dtype, device=device)

            q_tau = torch.sum(query_delta * ray_dir)
            if torch.isfinite(stored_tau[local_idx]):
                q_tau = 0.5 * q_tau + 0.5 * stored_tau[local_idx].to(dtype=q_tau.dtype)

            candidate_pos = atlas_positions.index_select(0, candidate_tensor)
            candidate_delta = candidate_pos - base_anchor[None, :]
            candidate_tau = torch.sum(candidate_delta * ray_dir[None, :], dim=1)
            lateral_vec = candidate_delta - candidate_tau[:, None] * ray_dir[None, :]
            lateral = torch.linalg.norm(lateral_vec, dim=1)
            tau_gap = torch.abs(candidate_tau - q_tau)
            euclidean = torch.linalg.norm(candidate_pos - query_point[None, :], dim=1)
            candidate_radius = atlas_radius.index_select(0, candidate_tensor)

            ref_bonus = torch.zeros_like(euclidean)
            ref_idx = int(ref_camera[local_idx].item())
            if (
                ref_idx >= 0
                and getattr(self, "_atlas_view_weights", None) is not None
                and self._atlas_view_weights.ndim == 2
                and ref_idx < self._atlas_view_weights.shape[1]
            ):
                ref_bonus = self._atlas_view_weights.index_select(0, candidate_tensor)[:, ref_idx].detach().clamp(0.0, 1.0)

            lateral_limit = (float(radius_mult) * candidate_radius).clamp_min(1e-6)
            tau_limit = torch.maximum(2.0 * lateral_limit, 3.0 * candidate_radius)
            ref_relax = 1.0 + 0.35 * ref_bonus
            ray_valid = (
                torch.isfinite(euclidean)
                & torch.isfinite(lateral)
                & torch.isfinite(tau_gap)
                & (lateral <= lateral_limit * ref_relax)
                & (tau_gap <= tau_limit * ref_relax)
                & (candidate_tau >= -candidate_radius)
            )
            nearest_valid = euclidean <= (float(radius_mult) * candidate_radius * ref_relax)
            valid = ray_valid | nearest_valid
            score = euclidean + 0.35 * lateral + 0.15 * tau_gap - 0.10 * ref_bonus * candidate_radius
            if torch.any(valid):
                valid_ids = torch.nonzero(valid, as_tuple=False).squeeze(-1)
                best_local = valid_ids[torch.argmin(score[valid_ids])]
                accept[local_idx] = True
            else:
                best_local = torch.argmin(score)
            fallback_ids[local_idx] = candidate_tensor[best_local]
            fallback_dist[local_idx] = euclidean[best_local]

        counts_np = np.asarray(candidate_counts, dtype=np.float32) if candidate_counts else np.zeros((0,), dtype=np.float32)
        return fallback_ids, fallback_dist, accept, {
            "ray_guided_queries": int(count),
            "ray_guided_ref_valid": int(ref_valid_count),
            "ray_guided_active_queries": int(active_query_count),
            "ray_guided_pending_queries": int(pending_query_count),
            "ray_guided_preaccept_count": int(accept.sum().item()),
            "ray_guided_empty_seed_count": int(empty_seed_count),
            "ray_guided_mean_candidate_count": float(counts_np.mean()) if counts_np.size else 0.0,
            "ray_guided_max_candidate_count": int(counts_np.max()) if counts_np.size else 0,
        }

    def _apply_atlas_reattach(
        self,
        point_indices: torch.Tensor,
        node_ids: torch.Tensor,
        state_cooldown_iters: int,
        low_confidence: bool = False,
        ref_camera: torch.Tensor | None = None,
    ):
        if point_indices is None or point_indices.numel() == 0:
            return
        device = self._device()
        point_indices = point_indices.to(device=device, dtype=torch.long)
        node_ids = node_ids.to(device=device, dtype=torch.long)
        self._atlas_node_ids[point_indices] = node_ids
        self._atlas_state[point_indices] = GAUSSIAN_STATE_UNSTABLE_PASSIVE
        self._atlas_drift_flag[point_indices] = False
        self._atlas_drift_count[point_indices] = 0
        self._atlas_gc_fail_count[point_indices] = 0
        self._atlas_high_residual_count[point_indices] = 0
        self._atlas_low_residual_count[point_indices] = 0
        self._atlas_promotion_streak[point_indices] = 0
        self._atlas_demotion_streak[point_indices] = 0
        self._atlas_recovery_streak[point_indices] = 0
        self._atlas_last_transition_iter[point_indices] = int(max(getattr(self, "_atlas_state_update_iter", 0), 0))
        self._atlas_photo_ema[point_indices] = 0.0
        self._atlas_visibility_ema[point_indices] = 0.0
        self._atlas_active_lifetime[point_indices] = 0
        self._atlas_active_provenance[point_indices] = ACTIVE_PROVENANCE_NONE

        resolved_ref = self._resolve_retry_ref_camera(point_indices)
        if ref_camera is not None:
            ref_camera = ref_camera.to(device=device, dtype=torch.long)
            missing = resolved_ref < 0
            if torch.any(missing):
                resolved_ref[missing] = ref_camera[missing]
        self._atlas_ref_camera[point_indices] = resolved_ref
        if bool(low_confidence):
            self._atlas_ref_score[point_indices] = 0.0
        else:
            valid_ref = resolved_ref >= 0
            if torch.any(valid_ref):
                current_score = self._atlas_ref_score[point_indices[valid_ref]]
                self._atlas_ref_score[point_indices[valid_ref]] = torch.maximum(
                    current_score,
                    torch.full_like(current_score, 0.05),
                )
            self._atlas_ref_score[point_indices[~valid_ref]] = 0.0

        if int(max(state_cooldown_iters, 0)) > 0:
            self._atlas_state_cooldown[point_indices] = torch.maximum(
                self._atlas_state_cooldown[point_indices],
                torch.full_like(self._atlas_state_cooldown[point_indices], int(max(state_cooldown_iters, 0))),
            )
        self._clear_pending_retry_metadata(point_indices, new_node_ids=node_ids)


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._center_log_sigma_parallel = torch.empty(0)
        self._center_log_sigma_support = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.tmp_radii = None
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self._init_point_count = 0
        self.optimizer = None
        self.exposure_optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._atlas_node_ids = torch.empty(0, dtype=torch.long)
        self._atlas_state = torch.empty(0, dtype=torch.long)
        self._atlas_positions = torch.empty(0)
        self._atlas_support = torch.empty(0)
        self._atlas_basis = torch.empty(0)
        self._atlas_raw_score = torch.empty(0)
        self._atlas_reliability_base = torch.empty(0)
        self._atlas_radius = torch.empty(0)
        self._atlas_reliability_runtime_raw = torch.empty(0)
        self._atlas_reliability_runtime_mapped = torch.empty(0)
        self._atlas_reliability_effective = torch.empty(0)
        self._atlas_reliability_runtime = torch.empty(0)
        self._atlas_class = torch.empty(0, dtype=torch.long)
        self._atlas_anisotropy_ref = torch.empty(0)
        self._atlas_neighbor_indices = torch.empty(0, dtype=torch.long)
        self._atlas_node_photo_ema = torch.empty(0)
        self._atlas_node_visibility_ema = torch.empty(0)
        self._atlas_node_obs_quality_ema = torch.empty(0)
        self._atlas_node_support_consistency_ema = torch.empty(0)
        self._atlas_node_finite_projection_ema = torch.empty(0)
        self._atlas_node_ref_consistency_ema = torch.empty(0)
        self._atlas_node_observed_score_ema = torch.empty(0)
        self._atlas_node_updated_recently = torch.empty(0)
        self._atlas_node_observed_count = torch.empty(0)
        self._atlas_node_support_consistent_count = torch.empty(0)
        self._atlas_refresh_observed_mask = torch.empty(0, dtype=torch.bool)
        self._atlas_refresh_node_photo_ema = torch.empty(0)
        self._atlas_refresh_node_visibility_ema = torch.empty(0)
        self._atlas_refresh_obs_quality = torch.empty(0)
        self._atlas_refresh_node_observed_count = torch.empty(0)
        self._atlas_refresh_node_support_consistent_ratio = torch.empty(0)
        self._atlas_refresh_node_coverage_ratio = torch.empty(0)
        self._atlas_refresh_node_ambiguity = torch.empty(0)
        self._atlas_refresh_override_weight = torch.empty(0)
        self._atlas_refresh_runtime_override_mask = torch.empty(0, dtype=torch.bool)
        self._atlas_photo_ema = torch.empty(0)
        self._atlas_visibility_ema = torch.empty(0)
        self._atlas_high_residual_count = torch.empty(0, dtype=torch.long)
        self._atlas_low_residual_count = torch.empty(0, dtype=torch.long)
        self._atlas_promotion_streak = torch.empty(0, dtype=torch.long)
        self._atlas_demotion_streak = torch.empty(0, dtype=torch.long)
        self._atlas_recovery_streak = torch.empty(0, dtype=torch.long)
        self._atlas_last_transition_iter = torch.empty(0, dtype=torch.long)
        self._atlas_gc_fail_count = torch.empty(0, dtype=torch.long)
        self._atlas_drift_flag = torch.empty(0, dtype=torch.bool)
        self._atlas_drift_count = torch.empty(0, dtype=torch.long)
        self._atlas_state_cooldown = torch.empty(0, dtype=torch.long)
        self._atlas_active_lifetime = torch.empty(0, dtype=torch.long)
        self._atlas_ref_camera = torch.empty(0, dtype=torch.long)
        self._atlas_ref_score = torch.empty(0)
        self._atlas_last_good_node_ids = torch.empty(0, dtype=torch.long)
        self._atlas_pending_ref_camera = torch.empty(0, dtype=torch.long)
        self._atlas_pending_ref_tau = torch.empty(0)
        self._atlas_pending_retry_count = torch.empty(0, dtype=torch.long)
        self._atlas_last_pending_iter = torch.empty(0, dtype=torch.long)
        self._atlas_active_provenance = torch.empty(0, dtype=torch.long)
        self._atlas_view_weights = torch.empty((0, 0))
        self._atlas_view_counts = torch.empty((0, 0), dtype=torch.int32)
        self._atlas_refresh_done = False
        self._atlas_source_path = ""
        self._atlas_init_num_nodes = 0
        self._atlas_init_gaussian_count_pre_spawn = 0
        self._atlas_extra_surface_spawn_count = 0
        self._atlas_init_gaussian_count_post_spawn = 0
        self._atlas_hash_metadata = None
        self._atlas_hash_lookup = {}
        self._atlas_hash_cell_size = 0.0
        self._atlas_hash_bbox_min = None
        self._atlas_hash_bucket_count = 0
        self._atlas_hash_neighbor_k = 0
        self._atlas_hash_source = "missing"
        self._atlas_hash_ready = False
        self._atlas_state_update_iter = 0
        self._atlas_runtime_last_observation = None
        self.exposure_mapping = {}
        self.pretrained_exposures = None
        self._exposure = nn.Parameter(torch.empty((0, 3, 4), dtype=torch.float32, device=self._device()), requires_grad=True)
        self.setup_functions()

    def capture(self):
        return {
            "version": 10,
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "center_log_sigma_parallel": self._center_log_sigma_parallel,
            "center_log_sigma_support": self._center_log_sigma_support,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "init_point_count": int(self.get_init_point_count()),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None else None,
            "exposure_optimizer_state": self.exposure_optimizer.state_dict() if self.exposure_optimizer is not None else None,
            "spatial_lr_scale": self.spatial_lr_scale,
            "atlas_node_ids": self._atlas_node_ids,
            "atlas_state": self._atlas_state,
            "atlas_positions": self._atlas_positions,
            "atlas_support": self._atlas_support,
            "atlas_basis": self._atlas_basis,
            "atlas_raw_score": self._atlas_raw_score,
            "atlas_reliability_base": self._atlas_reliability_base,
            "atlas_radius": self._atlas_radius,
            "atlas_reliability_runtime_raw": self._atlas_reliability_runtime_raw,
            "atlas_reliability_runtime_mapped": self._atlas_reliability_runtime_mapped,
            "atlas_reliability_effective": self._atlas_reliability_effective,
            "atlas_reliability_runtime": self._atlas_reliability_effective,
            "atlas_reliability": self._atlas_reliability_effective,
            "atlas_class": self._atlas_class,
            "atlas_anisotropy_ref": self._atlas_anisotropy_ref,
            "atlas_neighbor_indices": self._atlas_neighbor_indices,
            "atlas_node_photo_ema": self._atlas_node_photo_ema,
            "atlas_node_visibility_ema": self._atlas_node_visibility_ema,
            "atlas_node_obs_quality_ema": self._atlas_node_obs_quality_ema,
            "atlas_node_support_consistency_ema": self._atlas_node_support_consistency_ema,
            "atlas_node_finite_projection_ema": self._atlas_node_finite_projection_ema,
            "atlas_node_ref_consistency_ema": self._atlas_node_ref_consistency_ema,
            "atlas_node_observed_score_ema": self._atlas_node_observed_score_ema,
            "atlas_node_updated_recently": self._atlas_node_updated_recently,
            "atlas_node_observed_count": self._atlas_node_observed_count,
            "atlas_node_support_consistent_count": self._atlas_node_support_consistent_count,
            "atlas_refresh_observed_mask": self._atlas_refresh_observed_mask,
            "atlas_refresh_node_photo_ema": self._atlas_refresh_node_photo_ema,
            "atlas_refresh_node_visibility_ema": self._atlas_refresh_node_visibility_ema,
            "atlas_refresh_obs_quality": self._atlas_refresh_obs_quality,
            "atlas_refresh_node_observed_count": self._atlas_refresh_node_observed_count,
            "atlas_refresh_node_support_consistent_ratio": self._atlas_refresh_node_support_consistent_ratio,
            "atlas_refresh_node_coverage_ratio": self._atlas_refresh_node_coverage_ratio,
            "atlas_refresh_node_ambiguity": self._atlas_refresh_node_ambiguity,
            "atlas_refresh_override_weight": self._atlas_refresh_override_weight,
            "atlas_refresh_runtime_override_mask": self._atlas_refresh_runtime_override_mask,
            "atlas_photo_ema": self._atlas_photo_ema,
            "atlas_visibility_ema": self._atlas_visibility_ema,
            "atlas_high_residual_count": self._atlas_high_residual_count,
            "atlas_low_residual_count": self._atlas_low_residual_count,
            "atlas_promotion_streak": self._atlas_promotion_streak,
            "atlas_demotion_streak": self._atlas_demotion_streak,
            "atlas_recovery_streak": self._atlas_recovery_streak,
            "atlas_last_transition_iter": self._atlas_last_transition_iter,
            "atlas_gc_fail_count": self._atlas_gc_fail_count,
            "atlas_drift_flag": self._atlas_drift_flag,
            "atlas_drift_count": self._atlas_drift_count,
            "atlas_state_cooldown": self._atlas_state_cooldown,
            "atlas_active_lifetime": self._atlas_active_lifetime,
            "atlas_ref_camera": self._atlas_ref_camera,
            "atlas_ref_score": self._atlas_ref_score,
            "atlas_last_good_node_ids": self._atlas_last_good_node_ids,
            "atlas_pending_ref_camera": self._atlas_pending_ref_camera,
            "atlas_pending_ref_tau": self._atlas_pending_ref_tau,
            "atlas_pending_retry_count": self._atlas_pending_retry_count,
            "atlas_last_pending_iter": self._atlas_last_pending_iter,
            "atlas_active_provenance": self._atlas_active_provenance,
            "atlas_view_weights": self._atlas_view_weights,
            "atlas_view_counts": self._atlas_view_counts,
            "atlas_refresh_done": self._atlas_refresh_done,
            "atlas_state_update_iter": int(self._atlas_state_update_iter),
            "atlas_source_path": self._atlas_source_path,
            "atlas_init_num_nodes": int(self._atlas_init_num_nodes),
            "atlas_init_gaussian_count_pre_spawn": int(self._atlas_init_gaussian_count_pre_spawn),
            "atlas_extra_surface_spawn_count": int(self._atlas_extra_surface_spawn_count),
            "atlas_init_gaussian_count_post_spawn": int(self._atlas_init_gaussian_count_post_spawn),
            "exposure": self._capture_exposure_payload(),
        }
    
    def restore(self, model_args, training_args):
        exposure_optimizer_state = None
        restore_exposure_optimizer = False
        if isinstance(model_args, dict):
            self.active_sh_degree = model_args["active_sh_degree"]
            self._xyz = model_args["xyz"]
            self._features_dc = model_args["features_dc"]
            self._features_rest = model_args["features_rest"]
            self._scaling = model_args["scaling"]
            self._rotation = model_args["rotation"]
            self._opacity = model_args["opacity"]
            self._center_log_sigma_parallel = model_args["center_log_sigma_parallel"]
            self._center_log_sigma_support = model_args["center_log_sigma_support"]
            self.max_radii2D = model_args["max_radii2D"]
            self._init_point_count = int(model_args.get("init_point_count", model_args["xyz"].shape[0]))
            xyz_gradient_accum = model_args["xyz_gradient_accum"]
            denom = model_args["denom"]
            opt_dict = model_args.get("optimizer_state", None)
            exposure_optimizer_state = model_args.get("exposure_optimizer_state", None)
            self.spatial_lr_scale = model_args["spatial_lr_scale"]
            self._atlas_node_ids = model_args["atlas_node_ids"]
            self._atlas_state = model_args["atlas_state"]
            self._atlas_positions = model_args["atlas_positions"]
            self._atlas_support = model_args["atlas_support"]
            self._atlas_basis = model_args["atlas_basis"]
            self._atlas_raw_score = model_args["atlas_raw_score"]
            self._atlas_reliability_base = model_args.get(
                "atlas_reliability_base",
                model_args.get("atlas_base_reliability"),
            )
            self._atlas_radius = model_args["atlas_radius"]
            legacy_reliability_runtime = model_args.get(
                "atlas_reliability_runtime",
                model_args.get("atlas_reliability"),
            )
            self._atlas_reliability_runtime_raw = model_args.get(
                "atlas_reliability_runtime_raw",
                legacy_reliability_runtime,
            )
            self._atlas_reliability_runtime_mapped = model_args.get(
                "atlas_reliability_runtime_mapped",
                legacy_reliability_runtime,
            )
            self._atlas_reliability_effective = model_args.get(
                "atlas_reliability_effective",
                legacy_reliability_runtime,
            )
            self._atlas_reliability_runtime = self._atlas_reliability_effective
            self._atlas_class = model_args["atlas_class"]
            self._atlas_anisotropy_ref = model_args["atlas_anisotropy_ref"]
            self._atlas_neighbor_indices = model_args["atlas_neighbor_indices"]
            self._atlas_node_photo_ema = model_args["atlas_node_photo_ema"]
            self._atlas_node_visibility_ema = model_args["atlas_node_visibility_ema"]
            self._atlas_node_obs_quality_ema = model_args.get(
                "atlas_node_obs_quality_ema",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_support_consistency_ema = model_args.get(
                "atlas_node_support_consistency_ema",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_finite_projection_ema = model_args.get(
                "atlas_node_finite_projection_ema",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_ref_consistency_ema = model_args.get(
                "atlas_node_ref_consistency_ema",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_observed_score_ema = model_args.get(
                "atlas_node_observed_score_ema",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_updated_recently = model_args.get(
                "atlas_node_updated_recently",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_observed_count = model_args.get(
                "atlas_node_observed_count",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_node_support_consistent_count = model_args.get(
                "atlas_node_support_consistent_count",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_observed_mask = model_args.get(
                "atlas_refresh_observed_mask",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.bool, device=self._atlas_positions.device),
            )
            self._atlas_refresh_node_photo_ema = model_args.get(
                "atlas_refresh_node_photo_ema",
                self._atlas_node_photo_ema.detach().clone(),
            )
            self._atlas_refresh_node_visibility_ema = model_args.get(
                "atlas_refresh_node_visibility_ema",
                self._atlas_node_visibility_ema.detach().clone(),
            )
            self._atlas_refresh_obs_quality = model_args.get(
                "atlas_refresh_obs_quality",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_node_observed_count = model_args.get(
                "atlas_refresh_node_observed_count",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_node_support_consistent_ratio = model_args.get(
                "atlas_refresh_node_support_consistent_ratio",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_node_coverage_ratio = model_args.get(
                "atlas_refresh_node_coverage_ratio",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_node_ambiguity = model_args.get(
                "atlas_refresh_node_ambiguity",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_override_weight = model_args.get(
                "atlas_refresh_override_weight",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_refresh_runtime_override_mask = model_args.get(
                "atlas_refresh_runtime_override_mask",
                torch.zeros((self._atlas_positions.shape[0],), dtype=torch.bool, device=self._atlas_positions.device),
            )
            self._atlas_photo_ema = model_args["atlas_photo_ema"]
            self._atlas_visibility_ema = model_args["atlas_visibility_ema"]
            self._atlas_high_residual_count = model_args.get(
                "atlas_high_residual_count",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_low_residual_count = model_args.get(
                "atlas_low_residual_count",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_promotion_streak = model_args.get(
                "atlas_promotion_streak",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_demotion_streak = model_args.get(
                "atlas_demotion_streak",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_recovery_streak = model_args.get(
                "atlas_recovery_streak",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_last_transition_iter = model_args.get(
                "atlas_last_transition_iter",
                torch.full((self._atlas_node_ids.shape[0],), -1, dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_gc_fail_count = model_args.get(
                "atlas_gc_fail_count",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_drift_flag = model_args["atlas_drift_flag"]
            self._atlas_drift_count = model_args.get(
                "atlas_drift_count",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_state_cooldown = model_args.get(
                "atlas_state_cooldown",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_active_lifetime = model_args.get(
                "atlas_active_lifetime",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_ref_camera = model_args["atlas_ref_camera"]
            self._atlas_ref_score = model_args["atlas_ref_score"]
            self._atlas_last_good_node_ids = model_args.get(
                "atlas_last_good_node_ids",
                self._atlas_node_ids.detach().clone(),
            )
            self._atlas_pending_ref_camera = model_args.get(
                "atlas_pending_ref_camera",
                self._atlas_ref_camera.detach().clone(),
            )
            self._atlas_pending_ref_tau = model_args.get(
                "atlas_pending_ref_tau",
                torch.full((self._atlas_node_ids.shape[0],), float("nan"), dtype=torch.float32, device=self._atlas_positions.device),
            )
            self._atlas_pending_retry_count = model_args.get(
                "atlas_pending_retry_count",
                torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_last_pending_iter = model_args.get(
                "atlas_last_pending_iter",
                torch.full((self._atlas_node_ids.shape[0],), -1, dtype=torch.long, device=self._atlas_positions.device),
            )
            self._atlas_active_provenance = model_args.get(
                "atlas_active_provenance",
                torch.where(
                    self._atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                    torch.full((self._atlas_node_ids.shape[0],), ACTIVE_PROVENANCE_FROM_RESTORE_CHECKPOINT, dtype=torch.long, device=self._atlas_positions.device),
                    torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device),
                ),
            )
            self._atlas_view_weights = model_args.get("atlas_view_weights", torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.float32, device=self._atlas_positions.device))
            self._atlas_view_counts = model_args.get("atlas_view_counts", torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.int32, device=self._atlas_positions.device))
            self._atlas_refresh_done = model_args["atlas_refresh_done"]
            self._atlas_state_update_iter = int(model_args.get("atlas_state_update_iter", 0))
            self._atlas_source_path = model_args["atlas_source_path"]
            self._atlas_init_num_nodes = int(model_args.get("atlas_init_num_nodes", self._atlas_positions.shape[0]))
            self._atlas_init_gaussian_count_pre_spawn = int(model_args.get("atlas_init_gaussian_count_pre_spawn", self._atlas_init_num_nodes))
            self._atlas_extra_surface_spawn_count = int(model_args.get("atlas_extra_surface_spawn_count", 0))
            self._atlas_init_gaussian_count_post_spawn = int(
                model_args.get(
                    "atlas_init_gaussian_count_post_spawn",
                    self._atlas_init_gaussian_count_pre_spawn + self._atlas_extra_surface_spawn_count,
                )
            )
            self._invalidate_atlas_spatial_hash(clear_metadata=True)
            self._ensure_atlas_runtime_state()
            restore_exposure_optimizer = self._restore_exposure_payload(model_args.get("exposure"))
        elif len(model_args) == 12:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
            ) = model_args
            self._init_point_count = int(self._xyz.shape[0])
            sigma_init = torch.full((self._xyz.shape[0], 1), 1e-3, dtype=torch.float32, device=self._xyz.device)
            self._center_log_sigma_parallel = nn.Parameter(torch.log(sigma_init).requires_grad_(True))
            self._center_log_sigma_support = nn.Parameter(torch.log(sigma_init).requires_grad_(True))
            self._clear_atlas_bindings()
        elif len(model_args) == 24:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._center_log_sigma_parallel,
                self._center_log_sigma_support,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._atlas_node_ids,
                self._atlas_state,
                self._atlas_positions,
                self._atlas_support,
                self._atlas_basis,
                self._atlas_radius,
                self._atlas_reliability_runtime,
                self._atlas_class,
                self._atlas_anisotropy_ref,
                self._atlas_source_path,
            ) = model_args
            self._init_point_count = int(self._xyz.shape[0])
            self._atlas_reliability_base = self._atlas_reliability_runtime.detach().clone()
            self._invalidate_atlas_spatial_hash(clear_metadata=True)
            self._ensure_atlas_runtime_state()
        elif len(model_args) == 35:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._center_log_sigma_parallel,
                self._center_log_sigma_support,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._atlas_node_ids,
                self._atlas_state,
                self._atlas_positions,
                self._atlas_support,
                self._atlas_basis,
                self._atlas_raw_score,
                self._atlas_reliability_base,
                self._atlas_radius,
                self._atlas_reliability_runtime,
                self._atlas_class,
                self._atlas_anisotropy_ref,
                self._atlas_neighbor_indices,
                self._atlas_node_photo_ema,
                self._atlas_node_visibility_ema,
                self._atlas_photo_ema,
                self._atlas_visibility_ema,
                self._atlas_drift_flag,
                self._atlas_ref_camera,
                self._atlas_ref_score,
                self._atlas_refresh_done,
                self._atlas_source_path,
            ) = model_args
            self._init_point_count = int(self._xyz.shape[0])
            self._atlas_high_residual_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device)
            self._atlas_gc_fail_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._atlas_positions.device)
            self._atlas_view_weights = torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.float32, device=self._atlas_positions.device)
            self._atlas_view_counts = torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.int32, device=self._atlas_positions.device)
            self._invalidate_atlas_spatial_hash(clear_metadata=True)
            self._ensure_atlas_runtime_state()
        elif len(model_args) == 37:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._center_log_sigma_parallel,
                self._center_log_sigma_support,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._atlas_node_ids,
                self._atlas_state,
                self._atlas_positions,
                self._atlas_support,
                self._atlas_basis,
                self._atlas_raw_score,
                self._atlas_reliability_base,
                self._atlas_radius,
                self._atlas_reliability_runtime,
                self._atlas_class,
                self._atlas_anisotropy_ref,
                self._atlas_neighbor_indices,
                self._atlas_node_photo_ema,
                self._atlas_node_visibility_ema,
                self._atlas_photo_ema,
                self._atlas_visibility_ema,
                self._atlas_high_residual_count,
                self._atlas_gc_fail_count,
                self._atlas_drift_flag,
                self._atlas_ref_camera,
                self._atlas_ref_score,
                self._atlas_refresh_done,
                self._atlas_source_path,
            ) = model_args
            self._init_point_count = int(self._xyz.shape[0])
            self._atlas_view_weights = torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.float32, device=self._atlas_positions.device)
            self._atlas_view_counts = torch.empty((self._atlas_positions.shape[0], 0), dtype=torch.int32, device=self._atlas_positions.device)
            self._invalidate_atlas_spatial_hash(clear_metadata=True)
            self._ensure_atlas_runtime_state()
        else:
            (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._center_log_sigma_parallel,
                self._center_log_sigma_support,
                self.max_radii2D,
                xyz_gradient_accum,
                denom,
                opt_dict,
                self.spatial_lr_scale,
                self._atlas_node_ids,
                self._atlas_state,
                self._atlas_positions,
                self._atlas_support,
                self._atlas_basis,
                self._atlas_raw_score,
                self._atlas_reliability_base,
                self._atlas_radius,
                self._atlas_reliability_runtime,
                self._atlas_class,
                self._atlas_anisotropy_ref,
                self._atlas_neighbor_indices,
                self._atlas_node_photo_ema,
                self._atlas_node_visibility_ema,
                self._atlas_photo_ema,
                self._atlas_visibility_ema,
                self._atlas_high_residual_count,
                self._atlas_gc_fail_count,
                self._atlas_drift_flag,
                self._atlas_ref_camera,
                self._atlas_ref_score,
                self._atlas_view_weights,
                self._atlas_view_counts,
                self._atlas_refresh_done,
                self._atlas_source_path,
            ) = model_args
            self._init_point_count = int(self._xyz.shape[0])
            self._invalidate_atlas_spatial_hash(clear_metadata=True)
            self._ensure_atlas_runtime_state()
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        if opt_dict is not None:
            self.optimizer.load_state_dict(opt_dict)
        if exposure_optimizer_state is not None and restore_exposure_optimizer:
            self.exposure_optimizer.load_state_dict(exposure_optimizer_state)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    @property
    def get_center_sigma_parallel(self):
        return torch.exp(self._center_log_sigma_parallel)

    @property
    def get_center_sigma_support(self):
        return torch.exp(self._center_log_sigma_support)

    @property
    def has_atlas_bindings(self):
        return self._atlas_positions.numel() > 0 and self._atlas_node_ids.numel() == self._xyz.shape[0]

    @property
    def get_atlas_node_ids(self):
        return self._atlas_node_ids

    @property
    def get_atlas_state(self):
        return self._atlas_state

    @property
    def get_gaussian_atlas_positions(self):
        if not self.has_atlas_bindings:
            return torch.empty((0, 3), dtype=torch.float32, device=self._device())
        return self._atlas_positions[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_support(self):
        if not self.has_atlas_bindings:
            return torch.empty((0, 3, 3), dtype=torch.float32, device=self._device())
        return self._atlas_support[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_basis(self):
        if not self.has_atlas_bindings:
            return torch.empty((0, 3, 3), dtype=torch.float32, device=self._device())
        return self._atlas_basis[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_reliability(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_effective[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_reliability_base(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_base[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_reliability_runtime_raw(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_runtime_raw[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_reliability_runtime_mapped(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_runtime_mapped[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_reliability_effective(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_effective[self._atlas_node_ids]

    @property
    def get_atlas_node_reliability_runtime(self):
        if self._atlas_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_effective

    @property
    def get_atlas_node_reliability_base(self):
        if self._atlas_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_base

    @property
    def get_atlas_node_reliability_runtime_raw(self):
        if self._atlas_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_runtime_raw

    @property
    def get_atlas_node_reliability_runtime_mapped(self):
        if self._atlas_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_runtime_mapped

    @property
    def get_atlas_node_reliability_effective(self):
        if self._atlas_positions.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_reliability_effective

    @property
    def get_gaussian_atlas_radius(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.float32, device=self._device())
        return self._atlas_radius[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_class(self):
        if not self.has_atlas_bindings:
            return torch.empty((0,), dtype=torch.long, device=self._device())
        return self._atlas_class[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_anisotropy_ref(self):
        if not self.has_atlas_bindings:
            return torch.empty((0, 2), dtype=torch.float32, device=self._device())
        return self._atlas_anisotropy_ref[self._atlas_node_ids]

    @property
    def get_atlas_photo_ema(self):
        return self._atlas_photo_ema

    @property
    def get_atlas_visibility_ema(self):
        return self._atlas_visibility_ema

    @property
    def get_atlas_high_residual_count(self):
        return self._atlas_high_residual_count

    @property
    def get_atlas_low_residual_count(self):
        return self._atlas_low_residual_count

    @property
    def get_atlas_gc_fail_count(self):
        return self._atlas_gc_fail_count

    @property
    def get_atlas_drift_flag(self):
        return self._atlas_drift_flag

    @property
    def get_atlas_drift_count(self):
        return self._atlas_drift_count

    @property
    def get_atlas_state_cooldown(self):
        return self._atlas_state_cooldown

    @property
    def get_atlas_active_lifetime(self):
        return self._atlas_active_lifetime

    @property
    def get_atlas_ref_camera(self):
        return self._atlas_ref_camera

    @property
    def get_atlas_ref_score(self):
        return self._atlas_ref_score

    def summarize_atlas_reliability_state(self):
        if self._atlas_positions.numel() == 0:
            empty_tensor = torch.empty((0,), dtype=torch.float32, device=self._device())
            metrics = {
                "atlas_refresh_done": 0.0,
                "atlas_reliability_base_mean": 0.0,
                "atlas_reliability_base_std": 0.0,
                "atlas_reliability_runtime_raw_mean": 0.0,
                "atlas_reliability_runtime_raw_std": 0.0,
                "atlas_reliability_runtime_mapped_mean": 0.0,
                "atlas_reliability_runtime_mapped_std": 0.0,
                "atlas_reliability_effective_mean": 0.0,
                "atlas_reliability_effective_std": 0.0,
                "atlas_reliability_effective_min": 0.0,
                "atlas_reliability_effective_max": 0.0,
                "atlas_reliability_effective_delta_mean": 0.0,
                "atlas_reliability_runtime_mean": 0.0,
                "atlas_reliability_runtime_std": 0.0,
                "atlas_reliability_runtime_min": 0.0,
                "atlas_reliability_runtime_max": 0.0,
                "atlas_reliability_runtime_delta_mean": 0.0,
                "atlas_reliability_mapped_base_mean": 0.0,
                "atlas_reliability_mapped_runtime_mean": 0.0,
                "atlas_reliability_mapped_delta_mean": 0.0,
            }
            for name in ("base", "runtime_raw", "runtime_mapped", "effective"):
                metrics.update(self._snapshot_distribution_stats(empty_tensor, f"atlas_reliability_{name}"))
                metrics.update(self._snapshot_histogram_ratios(empty_tensor, f"atlas_reliability_{name}_hist"))
            return metrics
        self._ensure_atlas_runtime_state()
        raw = self._atlas_reliability_runtime_raw
        mapped = self._atlas_reliability_runtime_mapped
        effective = self._atlas_reliability_effective
        delta = effective - self._atlas_reliability_base
        mapped_base = self.get_gaussian_atlas_reliability_base
        mapped_runtime = self.get_gaussian_atlas_reliability
        mapped_delta = mapped_runtime - mapped_base
        metrics = {
            "atlas_refresh_done": 1.0 if self._atlas_refresh_done else 0.0,
            "atlas_reliability_base_mean": float(self._atlas_reliability_base.mean().item()) if self._atlas_reliability_base.numel() > 0 else 0.0,
            "atlas_reliability_base_std": float(self._atlas_reliability_base.std(unbiased=False).item()) if self._atlas_reliability_base.numel() > 0 else 0.0,
            "atlas_reliability_runtime_raw_mean": float(raw.mean().item()) if raw.numel() > 0 else 0.0,
            "atlas_reliability_runtime_raw_std": float(raw.std(unbiased=False).item()) if raw.numel() > 0 else 0.0,
            "atlas_reliability_runtime_mapped_mean": float(mapped.mean().item()) if mapped.numel() > 0 else 0.0,
            "atlas_reliability_runtime_mapped_std": float(mapped.std(unbiased=False).item()) if mapped.numel() > 0 else 0.0,
            "atlas_reliability_effective_mean": float(effective.mean().item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_effective_std": float(effective.std(unbiased=False).item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_effective_min": float(effective.min().item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_effective_max": float(effective.max().item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_effective_delta_mean": float(delta.mean().item()) if delta.numel() > 0 else 0.0,
            "atlas_reliability_runtime_mean": float(effective.mean().item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_runtime_std": float(effective.std(unbiased=False).item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_runtime_min": float(effective.min().item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_runtime_max": float(effective.max().item()) if effective.numel() > 0 else 0.0,
            "atlas_reliability_runtime_delta_mean": float(delta.mean().item()) if delta.numel() > 0 else 0.0,
            "atlas_reliability_mapped_base_mean": float(mapped_base.mean().item()) if mapped_base.numel() > 0 else 0.0,
            "atlas_reliability_mapped_runtime_mean": float(mapped_runtime.mean().item()) if mapped_runtime.numel() > 0 else 0.0,
            "atlas_reliability_mapped_delta_mean": float(mapped_delta.mean().item()) if mapped_delta.numel() > 0 else 0.0,
        }
        for name, values in (
            ("base", self._atlas_reliability_base),
            ("runtime_raw", raw),
            ("runtime_mapped", mapped),
            ("effective", effective),
        ):
            metrics.update(self._snapshot_distribution_stats(values, f"atlas_reliability_{name}"))
            metrics.update(self._snapshot_histogram_ratios(values, f"atlas_reliability_{name}_hist"))
        return metrics

    def summarize_atlas_refresh_snapshot(self):
        if self._atlas_positions.numel() == 0:
            return {
                "atlas_refresh_snapshot_ready": 0.0,
                "atlas_refresh_snapshot_observed_ratio": 0.0,
                "atlas_refresh_snapshot_observed_count": 0.0,
                "atlas_refresh_snapshot_photo_ema_mean": 0.0,
                "atlas_refresh_snapshot_visibility_ema_mean": 0.0,
                "atlas_refresh_snapshot_obs_quality_mean": 0.0,
                "atlas_refresh_snapshot_obs_quality_max": 0.0,
                "atlas_refresh_snapshot_node_observed_count_mean": 0.0,
                "atlas_refresh_snapshot_support_consistency_mean": 0.0,
                "atlas_refresh_snapshot_coverage_ratio_mean": 0.0,
                "atlas_refresh_snapshot_ambiguity_mean": 0.0,
                "atlas_refresh_snapshot_override_weight_mean": 0.0,
                "atlas_refresh_snapshot_runtime_override_count": 0.0,
                "atlas_refresh_snapshot_runtime_override_ratio": 0.0,
                "atlas_refresh_snapshot_keep_base_count": 0.0,
                "atlas_refresh_snapshot_runtime_raw_mean": 0.0,
                "atlas_refresh_snapshot_runtime_mapped_mean": 0.0,
                "atlas_refresh_snapshot_effective_reliability_mean": 0.0,
                "atlas_refresh_snapshot_quality_hist_low": 0.0,
                "atlas_refresh_snapshot_quality_hist_mid": 0.0,
                "atlas_refresh_snapshot_quality_hist_high": 0.0,
                "atlas_refresh_snapshot_coverage_hist_low": 0.0,
                "atlas_refresh_snapshot_coverage_hist_mid": 0.0,
                "atlas_refresh_snapshot_coverage_hist_high": 0.0,
            }
        self._ensure_atlas_runtime_state()
        observed_mask = self._atlas_refresh_observed_mask
        photo_ema = self._atlas_refresh_node_photo_ema
        visibility_ema = self._atlas_refresh_node_visibility_ema
        obs_quality = self._atlas_refresh_obs_quality
        observed_count = self._atlas_refresh_node_observed_count
        support_consistency = self._atlas_refresh_node_support_consistent_ratio
        coverage_ratio = self._atlas_refresh_node_coverage_ratio
        ambiguity = self._atlas_refresh_node_ambiguity
        override_weight = self._atlas_refresh_override_weight
        runtime_override_mask = self._atlas_refresh_runtime_override_mask
        snapshot_ready = bool(self._atlas_refresh_done and observed_mask.numel() == self._atlas_positions.shape[0])
        metrics = {
            "atlas_refresh_snapshot_ready": 1.0 if snapshot_ready else 0.0,
            "atlas_refresh_snapshot_observed_ratio": float(observed_mask.float().mean().item()) if observed_mask.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_observed_count": float(observed_mask.sum().item()) if observed_mask.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_photo_ema_mean": float(photo_ema.mean().item()) if photo_ema.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_visibility_ema_mean": float(visibility_ema.mean().item()) if visibility_ema.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_obs_quality_mean": float(obs_quality.mean().item()) if obs_quality.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_obs_quality_max": float(obs_quality.max().item()) if obs_quality.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_node_observed_count_mean": float(observed_count.mean().item()) if observed_count.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_support_consistency_mean": float(support_consistency.mean().item()) if support_consistency.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_coverage_ratio_mean": float(coverage_ratio.mean().item()) if coverage_ratio.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_ambiguity_mean": float(ambiguity.mean().item()) if ambiguity.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_override_weight_mean": float(override_weight.mean().item()) if override_weight.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_runtime_override_count": float(runtime_override_mask.sum().item()) if runtime_override_mask.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_runtime_override_ratio": float(runtime_override_mask.float().mean().item()) if runtime_override_mask.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_keep_base_count": float((~runtime_override_mask).sum().item()) if runtime_override_mask.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_runtime_raw_mean": float(self._atlas_reliability_runtime_raw.mean().item()) if self._atlas_reliability_runtime_raw.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_runtime_mapped_mean": float(self._atlas_reliability_runtime_mapped.mean().item()) if self._atlas_reliability_runtime_mapped.numel() > 0 else 0.0,
            "atlas_refresh_snapshot_effective_reliability_mean": float(self._atlas_reliability_effective.mean().item()) if self._atlas_reliability_effective.numel() > 0 else 0.0,
        }
        metrics.update(self._snapshot_histogram_ratios(obs_quality, "atlas_refresh_snapshot_quality_hist"))
        metrics.update(self._snapshot_histogram_ratios(coverage_ratio, "atlas_refresh_snapshot_coverage_hist"))
        metrics.update(self._snapshot_histogram_ratios(self._atlas_reliability_runtime_raw, "atlas_refresh_snapshot_runtime_raw_hist"))
        metrics.update(self._snapshot_histogram_ratios(self._atlas_reliability_runtime_mapped, "atlas_refresh_snapshot_runtime_mapped_hist"))
        metrics.update(self._snapshot_histogram_ratios(self._atlas_reliability_effective, "atlas_refresh_snapshot_effective_reliability_hist"))
        metrics.update(self._snapshot_histogram_ratios(self._atlas_reliability_effective, "atlas_refresh_snapshot_runtime_reliability_hist"))
        metrics.update(self._snapshot_distribution_stats(self._atlas_reliability_base, "atlas_refresh_snapshot_base"))
        metrics.update(self._snapshot_distribution_stats(self._atlas_reliability_runtime_mapped, "atlas_refresh_snapshot_runtime_mapped"))
        metrics.update(self._snapshot_distribution_stats(self._atlas_reliability_effective, "atlas_refresh_snapshot_effective"))
        snapshot_override_bucket = self._override_bucket_stats(
            self._atlas_reliability_base,
            runtime_override_mask,
            "atlas_refresh_snapshot_override_base_bucket",
        )
        metrics.update(snapshot_override_bucket)
        metrics.update(
            {
                key.replace("atlas_refresh_snapshot_override_base_bucket", "refresh_override_base_bucket"): value
                for key, value in snapshot_override_bucket.items()
            }
        )
        metrics.update({
            "refresh_evidence_observed_gate_ratio": float(observed_mask.float().mean().item()) if observed_mask.numel() > 0 else 0.0,
            "refresh_evidence_count_gate_ratio": float((observed_count >= 0.24).float().mean().item()) if observed_count.numel() > 0 else 0.0,
            "refresh_evidence_visibility_gate_ratio": float((visibility_ema >= 0.01).float().mean().item()) if visibility_ema.numel() > 0 else 0.0,
            "refresh_evidence_ref_gate_ratio": float((override_weight > 0.0).float().mean().item()) if override_weight.numel() > 0 else 0.0,
            "refresh_evidence_finite_gate_ratio": float(observed_mask.float().mean().item()) if observed_mask.numel() > 0 else 0.0,
            "refresh_evidence_support_gate_ratio": float((support_consistency >= 0.12).float().mean().item()) if support_consistency.numel() > 0 else 0.0,
            "refresh_evidence_override_gate_ratio": float(runtime_override_mask.float().mean().item()) if runtime_override_mask.numel() > 0 else 0.0,
            "refresh_evidence_gate_mean": float(override_weight.mean().item()) if override_weight.numel() > 0 else 0.0,
            "refresh_override_weight_positive_ratio": float((override_weight > 0.0).float().mean().item()) if override_weight.numel() > 0 else 0.0,
        })
        return metrics

    def summarize_atlas_state_metrics(self):
        if not self.has_atlas_bindings:
            metrics = {
                "stable_ratio": 0.0,
                "passive_ratio": 0.0,
                "active_ratio": 0.0,
                "out_of_anchor_ratio": 0.0,
                "pending_ratio": 0.0,
                "drift_ratio": 0.0,
                "cooldown_ratio": 0.0,
                "mean_gc_fail_count": 0.0,
                "mean_active_lifetime": 0.0,
                "max_active_lifetime": 0.0,
                "state_stable_count": 0,
                "state_passive_count": 0,
                "state_active_count": 0,
                "state_out_pending_count": 0,
                "out_of_anchor_pending_count": 0,
                "active_quota_effective_live_active_count": 0,
            }
            metrics.update(self._summarize_active_provenance_metrics(empty=True))
            return metrics

        atlas_state = self._atlas_state
        pending_mask = atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
        active_mask = atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        active_lifetime = self._atlas_active_lifetime.float() if self._atlas_active_lifetime.numel() > 0 else torch.empty((0,), dtype=torch.float32, device=atlas_state.device)
        metrics = {
            "stable_ratio": float((atlas_state == GAUSSIAN_STATE_STABLE).float().mean().item()),
            "passive_ratio": float((atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).float().mean().item()),
            "active_ratio": float(active_mask.float().mean().item()),
            "out_of_anchor_ratio": float(pending_mask.float().mean().item()),
            "pending_ratio": float(pending_mask.float().mean().item()),
            "drift_ratio": float(self._atlas_drift_flag.float().mean().item()) if self._atlas_drift_flag.numel() > 0 else 0.0,
            "cooldown_ratio": float((self._atlas_state_cooldown > 0).float().mean().item()) if self._atlas_state_cooldown.numel() > 0 else 0.0,
            "mean_gc_fail_count": float(self._atlas_gc_fail_count.float().mean().item()) if self._atlas_gc_fail_count.numel() > 0 else 0.0,
            "mean_active_lifetime": float(active_lifetime[active_mask].mean().item()) if torch.any(active_mask) else 0.0,
            "max_active_lifetime": float(active_lifetime[active_mask].max().item()) if torch.any(active_mask) else 0.0,
            "state_stable_count": int((atlas_state == GAUSSIAN_STATE_STABLE).sum().item()),
            "state_passive_count": int((atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).sum().item()),
            "state_active_count": int(active_mask.sum().item()),
            "state_out_pending_count": int(pending_mask.sum().item()),
            "out_of_anchor_pending_count": int(pending_mask.sum().item()),
            "active_quota_effective_live_active_count": int(active_mask.sum().item()),
        }
        metrics.update(self._summarize_active_provenance_metrics(active_mask=active_mask))
        return metrics

    def _summarize_active_provenance_metrics(self, active_mask: torch.Tensor | None = None, empty: bool = False):
        metrics = {
            f"active_provenance_{name}_count": 0
            for name in ACTIVE_PROVENANCE_NAMES.values()
        }
        metrics.update({
            f"active_provenance_{name}_ratio": 0.0
            for name in ACTIVE_PROVENANCE_NAMES.values()
        })
        metrics.update({
            "active_provenance_tracked_count": 0,
            "active_provenance_tracked_ratio": 0.0,
            "active_provenance_untracked_count": 0,
            "active_provenance_untracked_ratio": 0.0,
            "active_carryover_count": 0,
            "active_state_rebuild_count": 0,
        })
        if empty or (not self.has_atlas_bindings):
            return metrics
        if active_mask is None:
            active_mask = self._atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        if (
            getattr(self, "_atlas_active_provenance", None) is None
            or self._atlas_active_provenance.numel() != active_mask.numel()
            or active_mask.numel() == 0
        ):
            metrics["active_provenance_untracked_count"] = int(active_mask.sum().item()) if active_mask.numel() > 0 else 0
            return metrics

        active_count = int(active_mask.sum().item())
        tracked_count = 0
        provenance = self._atlas_active_provenance.detach()
        for provenance_id, name in ACTIVE_PROVENANCE_NAMES.items():
            count = int((active_mask & (provenance == int(provenance_id))).sum().item())
            metrics[f"active_provenance_{name}_count"] = count
            metrics[f"active_provenance_{name}_ratio"] = float(count / max(active_count, 1))
            tracked_count += count
        metrics["active_provenance_tracked_count"] = int(tracked_count)
        metrics["active_provenance_untracked_count"] = int(max(active_count - tracked_count, 0))
        metrics["active_provenance_tracked_ratio"] = float(tracked_count / max(active_count, 1))
        metrics["active_provenance_untracked_ratio"] = float(metrics["active_provenance_untracked_count"] / max(active_count, 1))
        metrics["active_carryover_count"] = int(metrics["active_provenance_from_quota_carryover_count"])
        metrics["active_state_rebuild_count"] = int(metrics["active_provenance_from_state_rebuild_after_gc_count"])
        return metrics

    @property
    def get_atlas_view_weights(self):
        return self._atlas_view_weights

    @property
    def get_atlas_view_counts(self):
        return self._atlas_view_counts

    @property
    def get_gaussian_atlas_view_weights(self):
        if not self.has_atlas_bindings or self._atlas_view_weights.numel() == 0:
            return torch.empty((self._xyz.shape[0], 0), dtype=torch.float32, device=self._device())
        return self._atlas_view_weights[self._atlas_node_ids]

    @property
    def get_gaussian_atlas_view_counts(self):
        if not self.has_atlas_bindings or self._atlas_view_counts.numel() == 0:
            return torch.empty((self._xyz.shape[0], 0), dtype=torch.int32, device=self._device())
        return self._atlas_view_counts[self._atlas_node_ids]

    @property
    def atlas_refresh_done(self):
        return bool(self._atlas_refresh_done)

    def summarize_atlas_init_metrics(self):
        return {
            "atlas_init_num_nodes": int(self._atlas_init_num_nodes),
            "atlas_init_gaussian_count_pre_spawn": int(self._atlas_init_gaussian_count_pre_spawn),
            "atlas_extra_surface_spawn_count": int(self._atlas_extra_surface_spawn_count),
            "atlas_init_gaussian_count_post_spawn": int(self._atlas_init_gaussian_count_post_spawn),
        }

    def get_init_point_count(self):
        if int(self._init_point_count) > 0:
            return int(self._init_point_count)
        return int(self._xyz.shape[0])

    def resolve_min_points_before_prune(self, base_min_points: int = 0, growth_ratio: float = 1.25, growth_extra: int = 1024):
        init_points = self.get_init_point_count()
        if init_points <= 0:
            return max(int(base_min_points), 0)
        dynamic_floor = max(
            int(np.ceil(float(init_points) * float(max(growth_ratio, 0.0)))),
            int(init_points + max(int(growth_extra), 0)),
        )
        return max(int(base_min_points), dynamic_floor)

    def summarize_capacity_state(self):
        init_points = self.get_init_point_count()
        total_points = int(self.get_xyz.shape[0])
        return {
            "init_point_count": float(init_points),
            "total_points": float(total_points),
            "capacity_ratio": float(total_points) / float(max(init_points, 1)),
        }

    def _concat_gaussian_parameters_without_optimizer(
        self,
        xyz: torch.Tensor,
        features_dc: torch.Tensor,
        features_rest: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: torch.Tensor,
        center_log_sigma_parallel: torch.Tensor,
        center_log_sigma_support: torch.Tensor,
    ):
        self._xyz = nn.Parameter(torch.cat((self._xyz.detach(), xyz.detach()), dim=0).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.cat((self._features_dc.detach(), features_dc.detach()), dim=0).requires_grad_(True))
        self._features_rest = nn.Parameter(torch.cat((self._features_rest.detach(), features_rest.detach()), dim=0).requires_grad_(True))
        self._scaling = nn.Parameter(torch.cat((self._scaling.detach(), scaling.detach()), dim=0).requires_grad_(True))
        self._rotation = nn.Parameter(torch.cat((self._rotation.detach(), rotation.detach()), dim=0).requires_grad_(True))
        self._opacity = nn.Parameter(torch.cat((self._opacity.detach(), opacity.detach()), dim=0).requires_grad_(True))
        self._center_log_sigma_parallel = nn.Parameter(
            torch.cat((self._center_log_sigma_parallel.detach(), center_log_sigma_parallel.detach()), dim=0).requires_grad_(True)
        )
        self._center_log_sigma_support = nn.Parameter(
            torch.cat((self._center_log_sigma_support.detach(), center_log_sigma_support.detach()), dim=0).requires_grad_(True)
        )
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), dtype=torch.float32, device=self._device())

    def _spawn_extra_surface_gaussians_from_atlas(self, atlas_init: FoundationAtlasInit):
        build_config = dict(getattr(atlas_init, "build_config", {}) or {})
        if not bool(build_config.get("spawn_extra_surface_gaussians", False)):
            return 0

        child_count = max(int(build_config.get("atlas_extra_surface_children", 2)), 0)
        if child_count <= 0:
            return 0

        surface_mask = np.asarray(atlas_init.atlas_class, dtype=np.int64) == ATLAS_CLASS_SURFACE
        reliability = np.asarray(atlas_init.reliability, dtype=np.float32).reshape(-1)
        radius = np.asarray(atlas_init.radius, dtype=np.float32).reshape(-1)
        support_score = np.asarray(getattr(atlas_init, "support_score", np.ones_like(reliability)), dtype=np.float32).reshape(-1)
        view_coverage = np.asarray(getattr(atlas_init, "view_coverage", np.ones_like(reliability)), dtype=np.float32).reshape(-1)
        radius_thr = float(build_config.get("atlas_extra_surface_radius_thr", 0.0))
        if radius_thr <= 0.0 and np.any(surface_mask):
            radius_thr = float(np.median(radius[surface_mask]))

        eligible = (
            surface_mask
            & (reliability >= float(build_config.get("atlas_extra_surface_rel_thr", 0.16)))
            & (radius >= max(radius_thr, 0.0))
            & (support_score >= float(build_config.get("atlas_extra_surface_support_thr", 0.50)))
            & (view_coverage >= float(build_config.get("atlas_extra_surface_view_thr", 0.35)))
        )
        eligible_ids = np.flatnonzero(eligible)
        if eligible_ids.size == 0:
            return 0

        offset_scale = float(build_config.get("atlas_extra_surface_offset_scale", 0.45))
        offset_patterns = np.asarray(
            [
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.0, 1.0],
                [0.0, -1.0],
            ],
            dtype=np.float32,
        )[:child_count]
        if offset_patterns.shape[0] == 0:
            return 0

        extra_positions = []
        extra_colors = []
        extra_scales = []
        extra_rotations = []
        extra_atlas_ids = []
        extra_states = []
        extra_ref_camera = []
        extra_ref_score = []
        extra_sigma_parallel = []
        extra_sigma_support = []

        ref_camera_ids = np.asarray(getattr(atlas_init, "reference_camera_ids", np.full((reliability.shape[0],), -1, dtype=np.int64)), dtype=np.int64).reshape(-1)
        ref_camera_scores = np.asarray(getattr(atlas_init, "reference_camera_scores", np.zeros((reliability.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)
        atlas_positions = np.asarray(atlas_init.positions, dtype=np.float32)
        atlas_basis = np.asarray(atlas_init.basis, dtype=np.float32)
        atlas_colors = np.asarray(atlas_init.colors, dtype=np.float32)
        init_scales = np.asarray(atlas_init.init_scales, dtype=np.float32)
        init_rotations = np.asarray(atlas_init.init_rotations, dtype=np.float32)
        gaussian_state = np.asarray(atlas_init.gaussian_state, dtype=np.int64)

        for atlas_id in eligible_ids.tolist():
            tangent_basis = atlas_basis[atlas_id, :, :2]
            node_radius = radius[atlas_id]
            for pattern in offset_patterns:
                offset = tangent_basis @ (pattern * (offset_scale * node_radius))
                extra_positions.append(atlas_positions[atlas_id] + offset.astype(np.float32))
                extra_colors.append(atlas_colors[atlas_id])
                extra_scale = init_scales[atlas_id].copy()
                extra_scale[0:2] *= 0.72
                extra_scale[2] *= 0.95
                extra_scales.append(extra_scale)
                extra_rotations.append(init_rotations[atlas_id])
                extra_atlas_ids.append(atlas_id)
                extra_states.append(gaussian_state[atlas_id])
                extra_ref_camera.append(ref_camera_ids[atlas_id] if atlas_id < ref_camera_ids.shape[0] else -1)
                extra_ref_score.append(ref_camera_scores[atlas_id] if atlas_id < ref_camera_scores.shape[0] else 0.0)
                extra_sigma_parallel.append(max(node_radius * 0.18, 1e-6))
                extra_sigma_support.append(max(node_radius * 0.08, 1e-6))

        if not extra_positions:
            return 0

        extra_point_cloud, extra_features = self._build_feature_tensors(
            np.asarray(extra_positions, dtype=np.float32),
            np.asarray(extra_colors, dtype=np.float32),
        )
        device = self._device()
        extra_scaling = torch.log(torch.tensor(np.asarray(extra_scales, dtype=np.float32), dtype=torch.float32, device=device).clamp_min(1e-6))
        extra_rotation = torch.tensor(np.asarray(extra_rotations, dtype=np.float32), dtype=torch.float32, device=device)
        extra_opacity = self.inverse_opacity_activation(
            0.1 * torch.ones((extra_point_cloud.shape[0], 1), dtype=torch.float32, device=device)
        )
        extra_center_sigma_parallel = torch.log(
            torch.tensor(np.asarray(extra_sigma_parallel, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(-1).clamp_min(1e-6)
        )
        extra_center_sigma_support = torch.log(
            torch.tensor(np.asarray(extra_sigma_support, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(-1).clamp_min(1e-6)
        )

        self._concat_gaussian_parameters_without_optimizer(
            extra_point_cloud,
            extra_features[:, :, 0:1].transpose(1, 2).contiguous(),
            extra_features[:, :, 1:].transpose(1, 2).contiguous(),
            extra_scaling,
            extra_rotation,
            extra_opacity,
            extra_center_sigma_parallel,
            extra_center_sigma_support,
        )
        self._append_atlas_bindings(
            torch.tensor(np.asarray(extra_atlas_ids, dtype=np.int64), dtype=torch.long, device=device),
            torch.tensor(np.asarray(extra_states, dtype=np.int64), dtype=torch.long, device=device),
            ref_camera=torch.tensor(np.asarray(extra_ref_camera, dtype=np.int64), dtype=torch.long, device=device),
            ref_score=torch.tensor(np.asarray(extra_ref_score, dtype=np.float32), dtype=torch.float32, device=device),
        )
        return int(len(extra_positions))

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud, features = self._build_feature_tensors(np.asarray(pcd.points), np.asarray(pcd.colors))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = self._estimate_point_dist2(fused_point_cloud)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=self._device())
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self._device()))

        self._initialize_gaussian_parameters(fused_point_cloud, features, scales, rots, opacities)
        init_sigma = torch.sqrt(dist2).unsqueeze(-1) * 0.1
        self._initialize_center_uncertainty(init_sigma, init_sigma * 0.5)
        self._init_point_count = int(self.get_xyz.shape[0])
        self._clear_atlas_bindings()
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device=self._device())[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def create_from_atlas(self, atlas_init: FoundationAtlasInit, cam_infos: int, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud, features = self._build_feature_tensors(atlas_init.positions, atlas_init.colors)
        print("Number of atlas nodes at initialisation : ", fused_point_cloud.shape[0])
        self._atlas_init_num_nodes = int(fused_point_cloud.shape[0])
        train_name_to_idx = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        train_stem_to_idx = {
            Path(cam_info.image_name).stem: idx
            for idx, cam_info in enumerate(cam_infos)
        }

        scales = torch.log(torch.tensor(atlas_init.init_scales, dtype=torch.float32, device=self._device()))
        rots = torch.tensor(atlas_init.init_rotations, dtype=torch.float32, device=self._device())
        opacities = self.inverse_opacity_activation(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float32, device=self._device())
        )

        self._initialize_gaussian_parameters(fused_point_cloud, features, scales, rots, opacities)
        atlas_radius = torch.tensor(atlas_init.radius, dtype=torch.float32, device=self._device()).unsqueeze(-1)
        sigma_parallel = atlas_radius * 0.2
        sigma_support = atlas_radius * 0.08
        self._initialize_center_uncertainty(sigma_parallel, sigma_support)
        self._init_point_count = int(self.get_xyz.shape[0])
        self._set_atlas_store(atlas_init)
        self._atlas_node_ids = torch.tensor(atlas_init.atlas_ids, dtype=torch.long, device=self._device())
        self._atlas_state = torch.tensor(atlas_init.gaussian_state, dtype=torch.long, device=self._device())
        self._atlas_photo_ema = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.float32, device=self._device())
        self._atlas_visibility_ema = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.float32, device=self._device())
        self._atlas_high_residual_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_low_residual_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_promotion_streak = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_demotion_streak = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_recovery_streak = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_last_transition_iter = torch.full((self._atlas_node_ids.shape[0],), -1, dtype=torch.long, device=self._device())
        self._atlas_gc_fail_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_drift_flag = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.bool, device=self._device())
        self._atlas_drift_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_state_cooldown = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_active_lifetime = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_last_good_node_ids = self._atlas_node_ids.detach().clone()
        self._atlas_pending_ref_camera = torch.full((self._atlas_node_ids.shape[0],), -1, dtype=torch.long, device=self._device())
        self._atlas_pending_ref_tau = torch.full((self._atlas_node_ids.shape[0],), float("nan"), dtype=torch.float32, device=self._device())
        self._atlas_pending_retry_count = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        self._atlas_last_pending_iter = torch.full((self._atlas_node_ids.shape[0],), -1, dtype=torch.long, device=self._device())
        self._atlas_active_provenance = torch.zeros((self._atlas_node_ids.shape[0],), dtype=torch.long, device=self._device())
        remapped_ref_camera = np.full((self._atlas_node_ids.shape[0],), -1, dtype=np.int64)
        if (
            atlas_init.camera_bundle is not None
            and getattr(atlas_init, "reference_camera_ids", None) is not None
            and len(atlas_init.reference_camera_ids) == int(self._atlas_node_ids.shape[0])
        ):
            atlas_image_names = list(atlas_init.camera_bundle.image_names)
            for gaussian_idx, atlas_camera_idx in enumerate(np.asarray(atlas_init.reference_camera_ids, dtype=np.int64).reshape(-1)):
                if atlas_camera_idx < 0 or atlas_camera_idx >= len(atlas_image_names):
                    continue
                image_name = atlas_image_names[int(atlas_camera_idx)]
                if image_name in train_name_to_idx:
                    remapped_ref_camera[gaussian_idx] = int(train_name_to_idx[image_name])
                    continue
                image_stem = Path(image_name).stem
                if image_stem in train_stem_to_idx:
                    remapped_ref_camera[gaussian_idx] = int(train_stem_to_idx[image_stem])
        elif getattr(atlas_init, "reference_camera_ids", None) is not None:
            raw_ref_camera = np.asarray(atlas_init.reference_camera_ids, dtype=np.int64).reshape(-1)
            valid = (raw_ref_camera >= 0) & (raw_ref_camera < len(cam_infos))
            remapped_ref_camera[valid] = raw_ref_camera[valid]
        self._atlas_ref_camera = torch.tensor(remapped_ref_camera, dtype=torch.long, device=self._device())
        self._atlas_ref_score = torch.tensor(
            np.asarray(getattr(atlas_init, "reference_camera_scores", np.zeros((self._atlas_node_ids.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self._device(),
        )
        self._atlas_pending_ref_camera = self._atlas_ref_camera.detach().clone()
        remapped_view_weights = np.zeros((self._atlas_positions.shape[0], 0), dtype=np.float32)
        remapped_view_counts = np.zeros((self._atlas_positions.shape[0], 0), dtype=np.int32)
        source_view_weights = np.asarray(getattr(atlas_init, "reference_view_weights", np.zeros((self._atlas_positions.shape[0], 0), dtype=np.float32)), dtype=np.float32)
        source_view_counts = np.asarray(getattr(atlas_init, "reference_view_counts", np.zeros((self._atlas_positions.shape[0], 0), dtype=np.int32)), dtype=np.int32)
        source_view_names = list(getattr(atlas_init, "reference_view_names", []))
        if (
            source_view_weights.ndim == 2
            and source_view_counts.ndim == 2
            and source_view_weights.shape == source_view_counts.shape
            and source_view_weights.shape[0] == self._atlas_positions.shape[0]
            and source_view_weights.shape[1] > 0
            and len(source_view_names) == source_view_weights.shape[1]
            and len(cam_infos) > 0
        ):
            remapped_view_weights = np.zeros((self._atlas_positions.shape[0], len(cam_infos)), dtype=np.float32)
            remapped_view_counts = np.zeros((self._atlas_positions.shape[0], len(cam_infos)), dtype=np.int32)
            for source_idx, image_name in enumerate(source_view_names):
                target_idx = train_name_to_idx.get(image_name)
                if target_idx is None:
                    target_idx = train_stem_to_idx.get(Path(image_name).stem)
                if target_idx is None:
                    continue
                remapped_view_weights[:, target_idx] += source_view_weights[:, source_idx]
                remapped_view_counts[:, target_idx] += source_view_counts[:, source_idx]
        self._atlas_view_weights = torch.tensor(remapped_view_weights, dtype=torch.float32, device=self._device())
        self._atlas_view_counts = torch.tensor(remapped_view_counts, dtype=torch.int32, device=self._device())
        self._atlas_init_gaussian_count_pre_spawn = int(self.get_xyz.shape[0])
        extra_spawn_count = self._spawn_extra_surface_gaussians_from_atlas(atlas_init)
        self._atlas_extra_surface_spawn_count = int(extra_spawn_count)
        self._atlas_init_gaussian_count_post_spawn = int(self.get_xyz.shape[0])
        if extra_spawn_count > 0:
            print(f"Spawned {extra_spawn_count} extra surface Gaussians from atlas prior.")
        self._init_point_count = int(self.get_xyz.shape[0])
        print(
            "Atlas init summary: "
            f"num_atlas_nodes={self._atlas_init_num_nodes}, "
            f"num_gaussians_pre_spawn={self._atlas_init_gaussian_count_pre_spawn}, "
            f"extra_surface_spawn_count={self._atlas_extra_surface_spawn_count}, "
            f"num_gaussians_after_spawn={self._atlas_init_gaussian_count_post_spawn}"
        )
        self.exposure_mapping = train_name_to_idx
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device=self._device())[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._center_log_sigma_parallel], 'lr': training_args.center_uncertainty_lr, "name": "center_sigma_parallel"},
            {'params': [self._center_log_sigma_support], 'lr': training_args.center_uncertainty_lr, "name": "center_sigma_support"},
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), "exposure.json")
            if not os.path.exists(exposure_file):
                exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        scale_tensor = torch.tensor(scales, dtype=torch.float, device="cuda")
        sigma_parallel = torch.exp(scale_tensor[:, :1]) * 0.1
        sigma_support = torch.exp(scale_tensor[:, :1]) * 0.05
        self._center_log_sigma_parallel = nn.Parameter(torch.log(sigma_parallel.clamp_min(1e-6)).requires_grad_(True))
        self._center_log_sigma_support = nn.Parameter(torch.log(sigma_support.clamp_min(1e-6)).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def save_atlas_state(self, path):
        if not self.has_atlas_bindings:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            atlas_node_ids=self._atlas_node_ids.detach().cpu().numpy(),
            atlas_state=self._atlas_state.detach().cpu().numpy(),
            atlas_positions=self._atlas_positions.detach().cpu().numpy(),
            atlas_support=self._atlas_support.detach().cpu().numpy(),
            atlas_basis=self._atlas_basis.detach().cpu().numpy(),
            atlas_raw_score=self._atlas_raw_score.detach().cpu().numpy(),
            atlas_reliability_base=self._atlas_reliability_base.detach().cpu().numpy(),
            atlas_radius=self._atlas_radius.detach().cpu().numpy(),
            atlas_reliability_runtime_raw=self._atlas_reliability_runtime_raw.detach().cpu().numpy(),
            atlas_reliability_runtime_mapped=self._atlas_reliability_runtime_mapped.detach().cpu().numpy(),
            atlas_reliability_effective=self._atlas_reliability_effective.detach().cpu().numpy(),
            atlas_reliability_runtime=self._atlas_reliability_effective.detach().cpu().numpy(),
            atlas_reliability=self._atlas_reliability_effective.detach().cpu().numpy(),
            atlas_class=self._atlas_class.detach().cpu().numpy(),
            atlas_anisotropy_ref=self._atlas_anisotropy_ref.detach().cpu().numpy(),
            atlas_neighbor_indices=self._atlas_neighbor_indices.detach().cpu().numpy(),
            atlas_node_photo_ema=self._atlas_node_photo_ema.detach().cpu().numpy(),
            atlas_node_visibility_ema=self._atlas_node_visibility_ema.detach().cpu().numpy(),
            atlas_node_obs_quality_ema=self._atlas_node_obs_quality_ema.detach().cpu().numpy(),
            atlas_node_support_consistency_ema=self._atlas_node_support_consistency_ema.detach().cpu().numpy(),
            atlas_node_finite_projection_ema=self._atlas_node_finite_projection_ema.detach().cpu().numpy(),
            atlas_node_ref_consistency_ema=self._atlas_node_ref_consistency_ema.detach().cpu().numpy(),
            atlas_node_observed_score_ema=self._atlas_node_observed_score_ema.detach().cpu().numpy(),
            atlas_node_updated_recently=self._atlas_node_updated_recently.detach().cpu().numpy(),
            atlas_node_observed_count=self._atlas_node_observed_count.detach().cpu().numpy(),
            atlas_node_support_consistent_count=self._atlas_node_support_consistent_count.detach().cpu().numpy(),
            atlas_refresh_observed_mask=self._atlas_refresh_observed_mask.detach().cpu().numpy(),
            atlas_refresh_node_photo_ema=self._atlas_refresh_node_photo_ema.detach().cpu().numpy(),
            atlas_refresh_node_visibility_ema=self._atlas_refresh_node_visibility_ema.detach().cpu().numpy(),
            atlas_refresh_obs_quality=self._atlas_refresh_obs_quality.detach().cpu().numpy(),
            atlas_refresh_node_observed_count=self._atlas_refresh_node_observed_count.detach().cpu().numpy(),
            atlas_refresh_node_support_consistent_ratio=self._atlas_refresh_node_support_consistent_ratio.detach().cpu().numpy(),
            atlas_refresh_node_coverage_ratio=self._atlas_refresh_node_coverage_ratio.detach().cpu().numpy(),
            atlas_refresh_node_ambiguity=self._atlas_refresh_node_ambiguity.detach().cpu().numpy(),
            atlas_refresh_override_weight=self._atlas_refresh_override_weight.detach().cpu().numpy(),
            atlas_refresh_runtime_override_mask=self._atlas_refresh_runtime_override_mask.detach().cpu().numpy(),
            atlas_photo_ema=self._atlas_photo_ema.detach().cpu().numpy(),
            atlas_visibility_ema=self._atlas_visibility_ema.detach().cpu().numpy(),
            atlas_high_residual_count=self._atlas_high_residual_count.detach().cpu().numpy(),
            atlas_low_residual_count=self._atlas_low_residual_count.detach().cpu().numpy(),
            atlas_promotion_streak=self._atlas_promotion_streak.detach().cpu().numpy(),
            atlas_demotion_streak=self._atlas_demotion_streak.detach().cpu().numpy(),
            atlas_recovery_streak=self._atlas_recovery_streak.detach().cpu().numpy(),
            atlas_last_transition_iter=self._atlas_last_transition_iter.detach().cpu().numpy(),
            atlas_gc_fail_count=self._atlas_gc_fail_count.detach().cpu().numpy(),
            atlas_drift_flag=self._atlas_drift_flag.detach().cpu().numpy(),
            atlas_drift_count=self._atlas_drift_count.detach().cpu().numpy(),
            atlas_state_cooldown=self._atlas_state_cooldown.detach().cpu().numpy(),
            atlas_active_lifetime=self._atlas_active_lifetime.detach().cpu().numpy(),
            atlas_ref_camera=self._atlas_ref_camera.detach().cpu().numpy(),
            atlas_ref_score=self._atlas_ref_score.detach().cpu().numpy(),
            atlas_last_good_node_ids=self._atlas_last_good_node_ids.detach().cpu().numpy(),
            atlas_pending_ref_camera=self._atlas_pending_ref_camera.detach().cpu().numpy(),
            atlas_pending_ref_tau=self._atlas_pending_ref_tau.detach().cpu().numpy(),
            atlas_pending_retry_count=self._atlas_pending_retry_count.detach().cpu().numpy(),
            atlas_last_pending_iter=self._atlas_last_pending_iter.detach().cpu().numpy(),
            atlas_active_provenance=self._atlas_active_provenance.detach().cpu().numpy(),
            atlas_view_weights=self._atlas_view_weights.detach().cpu().numpy(),
            atlas_view_counts=self._atlas_view_counts.detach().cpu().numpy(),
            center_log_sigma_parallel=self._center_log_sigma_parallel.detach().cpu().numpy(),
            center_log_sigma_support=self._center_log_sigma_support.detach().cpu().numpy(),
            init_point_count=np.asarray(self.get_init_point_count(), dtype=np.int64),
            atlas_refresh_done=np.asarray(self._atlas_refresh_done),
            atlas_state_update_iter=np.asarray(self._atlas_state_update_iter, dtype=np.int64),
            atlas_source_path=np.asarray(self._atlas_source_path),
            atlas_init_num_nodes=np.asarray(self._atlas_init_num_nodes, dtype=np.int64),
            atlas_init_gaussian_count_pre_spawn=np.asarray(self._atlas_init_gaussian_count_pre_spawn, dtype=np.int64),
            atlas_extra_surface_spawn_count=np.asarray(self._atlas_extra_surface_spawn_count, dtype=np.int64),
            atlas_init_gaussian_count_post_spawn=np.asarray(self._atlas_init_gaussian_count_post_spawn, dtype=np.int64),
        )

    def load_atlas_state(self, path):
        with np.load(path, allow_pickle=True) as archive_file:
            archive = {key: np.asarray(archive_file[key]) for key in archive_file.files}
        device = self._device()
        self._atlas_node_ids = torch.tensor(archive["atlas_node_ids"], dtype=torch.long, device=device)
        self._atlas_state = torch.tensor(archive["atlas_state"], dtype=torch.long, device=device)
        self._atlas_positions = torch.tensor(archive["atlas_positions"], dtype=torch.float32, device=device)
        self._atlas_support = torch.tensor(archive["atlas_support"], dtype=torch.float32, device=device)
        self._atlas_basis = torch.tensor(archive["atlas_basis"], dtype=torch.float32, device=device)
        self._atlas_raw_score = torch.tensor(
            archive["atlas_raw_score"] if "atlas_raw_score" in archive else archive["atlas_reliability"],
            dtype=torch.float32,
            device=device,
        )
        self._atlas_reliability_base = torch.tensor(
            archive["atlas_reliability_base"] if "atlas_reliability_base" in archive else (
                archive["atlas_base_reliability"] if "atlas_base_reliability" in archive else archive["atlas_reliability"]
            ),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_radius = torch.tensor(archive["atlas_radius"], dtype=torch.float32, device=device)
        legacy_reliability_runtime = archive["atlas_reliability_runtime"] if "atlas_reliability_runtime" in archive else archive["atlas_reliability"]
        self._atlas_reliability_runtime_raw = torch.tensor(
            archive["atlas_reliability_runtime_raw"] if "atlas_reliability_runtime_raw" in archive else legacy_reliability_runtime,
            dtype=torch.float32,
            device=device,
        )
        self._atlas_reliability_runtime_mapped = torch.tensor(
            archive["atlas_reliability_runtime_mapped"] if "atlas_reliability_runtime_mapped" in archive else legacy_reliability_runtime,
            dtype=torch.float32,
            device=device,
        )
        self._atlas_reliability_effective = torch.tensor(
            archive["atlas_reliability_effective"] if "atlas_reliability_effective" in archive else legacy_reliability_runtime,
            dtype=torch.float32,
            device=device,
        )
        self._atlas_reliability_runtime = self._atlas_reliability_effective.detach().clone()
        self._atlas_class = torch.tensor(archive["atlas_class"], dtype=torch.long, device=device)
        self._atlas_anisotropy_ref = torch.tensor(archive["atlas_anisotropy_ref"], dtype=torch.float32, device=device)
        self._atlas_neighbor_indices = torch.tensor(
            archive["atlas_neighbor_indices"] if "atlas_neighbor_indices" in archive else np.arange(self._atlas_positions.shape[0])[:, None],
            dtype=torch.long,
            device=device,
        )
        self._atlas_node_photo_ema = torch.tensor(
            archive["atlas_node_photo_ema"] if "atlas_node_photo_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_visibility_ema = torch.tensor(
            archive["atlas_node_visibility_ema"] if "atlas_node_visibility_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_obs_quality_ema = torch.tensor(
            archive["atlas_node_obs_quality_ema"] if "atlas_node_obs_quality_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_support_consistency_ema = torch.tensor(
            archive["atlas_node_support_consistency_ema"] if "atlas_node_support_consistency_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_finite_projection_ema = torch.tensor(
            archive["atlas_node_finite_projection_ema"] if "atlas_node_finite_projection_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_ref_consistency_ema = torch.tensor(
            archive["atlas_node_ref_consistency_ema"] if "atlas_node_ref_consistency_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_observed_score_ema = torch.tensor(
            archive["atlas_node_observed_score_ema"] if "atlas_node_observed_score_ema" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_updated_recently = torch.tensor(
            archive["atlas_node_updated_recently"] if "atlas_node_updated_recently" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_observed_count = torch.tensor(
            archive["atlas_node_observed_count"] if "atlas_node_observed_count" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_node_support_consistent_count = torch.tensor(
            archive["atlas_node_support_consistent_count"] if "atlas_node_support_consistent_count" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_observed_mask = torch.tensor(
            archive["atlas_refresh_observed_mask"] if "atlas_refresh_observed_mask" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.bool_),
            dtype=torch.bool,
            device=device,
        )
        self._atlas_refresh_node_photo_ema = torch.tensor(
            archive["atlas_refresh_node_photo_ema"] if "atlas_refresh_node_photo_ema" in archive else self._atlas_node_photo_ema.detach().cpu().numpy(),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_node_visibility_ema = torch.tensor(
            archive["atlas_refresh_node_visibility_ema"] if "atlas_refresh_node_visibility_ema" in archive else self._atlas_node_visibility_ema.detach().cpu().numpy(),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_obs_quality = torch.tensor(
            archive["atlas_refresh_obs_quality"] if "atlas_refresh_obs_quality" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_node_observed_count = torch.tensor(
            archive["atlas_refresh_node_observed_count"] if "atlas_refresh_node_observed_count" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_node_support_consistent_ratio = torch.tensor(
            archive["atlas_refresh_node_support_consistent_ratio"] if "atlas_refresh_node_support_consistent_ratio" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_node_coverage_ratio = torch.tensor(
            archive["atlas_refresh_node_coverage_ratio"] if "atlas_refresh_node_coverage_ratio" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_node_ambiguity = torch.tensor(
            archive["atlas_refresh_node_ambiguity"] if "atlas_refresh_node_ambiguity" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_override_weight = torch.tensor(
            archive["atlas_refresh_override_weight"] if "atlas_refresh_override_weight" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_refresh_runtime_override_mask = torch.tensor(
            archive["atlas_refresh_runtime_override_mask"] if "atlas_refresh_runtime_override_mask" in archive else np.zeros(self._atlas_positions.shape[0], dtype=np.bool_),
            dtype=torch.bool,
            device=device,
        )
        self._atlas_photo_ema = torch.tensor(
            archive["atlas_photo_ema"] if "atlas_photo_ema" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_visibility_ema = torch.tensor(
            archive["atlas_visibility_ema"] if "atlas_visibility_ema" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_high_residual_count = torch.tensor(
            archive["atlas_high_residual_count"] if "atlas_high_residual_count" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_low_residual_count = torch.tensor(
            archive["atlas_low_residual_count"] if "atlas_low_residual_count" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_promotion_streak = torch.tensor(
            archive["atlas_promotion_streak"] if "atlas_promotion_streak" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_demotion_streak = torch.tensor(
            archive["atlas_demotion_streak"] if "atlas_demotion_streak" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_recovery_streak = torch.tensor(
            archive["atlas_recovery_streak"] if "atlas_recovery_streak" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_last_transition_iter = torch.tensor(
            archive["atlas_last_transition_iter"] if "atlas_last_transition_iter" in archive else np.full(self._atlas_node_ids.shape[0], -1, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_gc_fail_count = torch.tensor(
            archive["atlas_gc_fail_count"] if "atlas_gc_fail_count" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_drift_flag = torch.tensor(
            archive["atlas_drift_flag"] if "atlas_drift_flag" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.bool_),
            dtype=torch.bool,
            device=device,
        )
        self._atlas_drift_count = torch.tensor(
            archive["atlas_drift_count"] if "atlas_drift_count" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_state_cooldown = torch.tensor(
            archive["atlas_state_cooldown"] if "atlas_state_cooldown" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_active_lifetime = torch.tensor(
            archive["atlas_active_lifetime"] if "atlas_active_lifetime" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_ref_camera = torch.tensor(
            archive["atlas_ref_camera"] if "atlas_ref_camera" in archive else -np.ones(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_ref_score = torch.tensor(
            archive["atlas_ref_score"] if "atlas_ref_score" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_last_good_node_ids = torch.tensor(
            archive["atlas_last_good_node_ids"] if "atlas_last_good_node_ids" in archive else self._atlas_node_ids.detach().cpu().numpy(),
            dtype=torch.long,
            device=device,
        )
        self._atlas_pending_ref_camera = torch.tensor(
            archive["atlas_pending_ref_camera"] if "atlas_pending_ref_camera" in archive else self._atlas_ref_camera.detach().cpu().numpy(),
            dtype=torch.long,
            device=device,
        )
        self._atlas_pending_ref_tau = torch.tensor(
            archive["atlas_pending_ref_tau"] if "atlas_pending_ref_tau" in archive else np.full(self._atlas_node_ids.shape[0], np.nan, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_pending_retry_count = torch.tensor(
            archive["atlas_pending_retry_count"] if "atlas_pending_retry_count" in archive else np.zeros(self._atlas_node_ids.shape[0], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_last_pending_iter = torch.tensor(
            archive["atlas_last_pending_iter"] if "atlas_last_pending_iter" in archive else np.full(self._atlas_node_ids.shape[0], -1, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_active_provenance = torch.tensor(
            archive["atlas_active_provenance"] if "atlas_active_provenance" in archive else np.where(
                archive["atlas_state"] == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                ACTIVE_PROVENANCE_FROM_RESTORE_CHECKPOINT,
                ACTIVE_PROVENANCE_NONE,
            ).astype(np.int64),
            dtype=torch.long,
            device=device,
        )
        self._atlas_view_weights = torch.tensor(
            archive["atlas_view_weights"] if "atlas_view_weights" in archive else np.zeros((self._atlas_positions.shape[0], 0), dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        self._atlas_view_counts = torch.tensor(
            archive["atlas_view_counts"] if "atlas_view_counts" in archive else np.zeros((self._atlas_positions.shape[0], 0), dtype=np.int32),
            dtype=torch.int32,
            device=device,
        )
        if "center_log_sigma_parallel" in archive:
            self._center_log_sigma_parallel = nn.Parameter(
                torch.tensor(archive["center_log_sigma_parallel"], dtype=torch.float32, device=device).requires_grad_(True)
            )
        if "center_log_sigma_support" in archive:
            self._center_log_sigma_support = nn.Parameter(
                torch.tensor(archive["center_log_sigma_support"], dtype=torch.float32, device=device).requires_grad_(True)
            )
        if "init_point_count" in archive:
            init_count = archive["init_point_count"]
            self._init_point_count = int(init_count.item() if np.asarray(init_count).shape == () else init_count[0])
        else:
            self._init_point_count = int(self._xyz.shape[0])
        if "atlas_refresh_done" in archive:
            refresh_done = archive["atlas_refresh_done"]
            self._atlas_refresh_done = bool(refresh_done.item() if np.asarray(refresh_done).shape == () else refresh_done[0])
        else:
            self._atlas_refresh_done = False
        if "atlas_state_update_iter" in archive:
            state_update_iter = archive["atlas_state_update_iter"]
            self._atlas_state_update_iter = int(state_update_iter.item() if np.asarray(state_update_iter).shape == () else state_update_iter[0])
        else:
            self._atlas_state_update_iter = 0
        self._atlas_init_num_nodes = int(
            archive["atlas_init_num_nodes"].item() if "atlas_init_num_nodes" in archive else self._atlas_positions.shape[0]
        )
        self._atlas_init_gaussian_count_pre_spawn = int(
            archive["atlas_init_gaussian_count_pre_spawn"].item()
            if "atlas_init_gaussian_count_pre_spawn" in archive
            else self._atlas_init_num_nodes
        )
        self._atlas_extra_surface_spawn_count = int(
            archive["atlas_extra_surface_spawn_count"].item()
            if "atlas_extra_surface_spawn_count" in archive
            else 0
        )
        self._atlas_init_gaussian_count_post_spawn = int(
            archive["atlas_init_gaussian_count_post_spawn"].item()
            if "atlas_init_gaussian_count_post_spawn" in archive
            else (self._atlas_init_gaussian_count_pre_spawn + self._atlas_extra_surface_spawn_count)
        )
        source_path = archive["atlas_source_path"]
        self._atlas_source_path = str(source_path.item() if np.asarray(source_path).shape == () else source_path)
        self._invalidate_atlas_spatial_hash(clear_metadata=True)
        self._ensure_atlas_runtime_state()

    def summarize_atlas_bindings(self):
        if not self.has_atlas_bindings:
            return {"has_atlas_bindings": False}

        atlas_class = self._atlas_class[self._atlas_node_ids].detach().cpu().numpy()
        atlas_state = self._atlas_state.detach().cpu().numpy()
        reliability_runtime = self._atlas_reliability_effective[self._atlas_node_ids].detach().cpu().numpy()
        reliability_base = self._atlas_reliability_base[self._atlas_node_ids].detach().cpu().numpy()
        photo_ema = self._atlas_photo_ema.detach().cpu().numpy()
        visibility_ema = self._atlas_visibility_ema.detach().cpu().numpy()
        high_residual_count = self._atlas_high_residual_count.detach().cpu().numpy()
        gc_fail_count = self._atlas_gc_fail_count.detach().cpu().numpy()
        pending_retry_count = self._atlas_pending_retry_count.detach().cpu().numpy()
        drift_flag = self._atlas_drift_flag.detach().cpu().numpy()
        ref_camera = self._atlas_ref_camera.detach().cpu().numpy()
        pending_ref_camera = self._atlas_pending_ref_camera.detach().cpu().numpy()

        class_counts = {
            ATLAS_CLASS_NAMES[class_id]: int(np.sum(atlas_class == class_id))
            for class_id in ATLAS_CLASS_NAMES
        }
        state_counts = {
            GAUSSIAN_STATE_NAMES[state_id]: int(np.sum(atlas_state == state_id))
            for state_id in GAUSSIAN_STATE_NAMES
        }
        summary = {
            "has_atlas_bindings": True,
            "atlas_source_path": self._atlas_source_path,
            "num_gaussians": int(self._xyz.shape[0]),
            "num_atlas_nodes": int(self._atlas_positions.shape[0]),
            "mean_atlas_reliability": float(np.mean(reliability_runtime)),
            "median_atlas_reliability": float(np.median(reliability_runtime)),
            "mean_atlas_reliability_base": float(np.mean(reliability_base)) if reliability_base.size else 0.0,
            "mean_atlas_reliability_runtime": float(np.mean(reliability_runtime)) if reliability_runtime.size else 0.0,
            "mean_atlas_node_reliability_base": float(self._atlas_reliability_base.mean().item()) if self._atlas_reliability_base.numel() > 0 else 0.0,
            "mean_atlas_node_reliability_runtime": float(self._atlas_reliability_effective.mean().item()) if self._atlas_reliability_effective.numel() > 0 else 0.0,
            "mean_atlas_node_reliability_runtime_raw": float(self._atlas_reliability_runtime_raw.mean().item()) if self._atlas_reliability_runtime_raw.numel() > 0 else 0.0,
            "mean_atlas_node_reliability_runtime_mapped": float(self._atlas_reliability_runtime_mapped.mean().item()) if self._atlas_reliability_runtime_mapped.numel() > 0 else 0.0,
            "mean_atlas_node_reliability_effective": float(self._atlas_reliability_effective.mean().item()) if self._atlas_reliability_effective.numel() > 0 else 0.0,
            "mean_photo_ema": float(np.mean(photo_ema)) if photo_ema.size else 0.0,
            "mean_visibility_ema": float(np.mean(visibility_ema)) if visibility_ema.size else 0.0,
            "mean_high_residual_count": float(np.mean(high_residual_count)) if high_residual_count.size else 0.0,
            "max_high_residual_count": int(np.max(high_residual_count)) if high_residual_count.size else 0,
            "mean_gc_fail_count": float(np.mean(gc_fail_count)) if gc_fail_count.size else 0.0,
            "max_gc_fail_count": int(np.max(gc_fail_count)) if gc_fail_count.size else 0,
            "mean_pending_retry_count": float(np.mean(pending_retry_count)) if pending_retry_count.size else 0.0,
            "max_pending_retry_count": int(np.max(pending_retry_count)) if pending_retry_count.size else 0,
            "drift_ratio": float(np.mean(drift_flag.astype(np.float32))) if drift_flag.size else 0.0,
            "ref_camera_ratio": float(np.mean((ref_camera >= 0).astype(np.float32))) if ref_camera.size else 0.0,
            "pending_ref_camera_ratio": float(np.mean((pending_ref_camera >= 0).astype(np.float32))) if pending_ref_camera.size else 0.0,
            "refresh_done": bool(self._atlas_refresh_done),
            "atlas_init_num_nodes": int(self._atlas_init_num_nodes),
            "atlas_init_gaussian_count_pre_spawn": int(self._atlas_init_gaussian_count_pre_spawn),
            "atlas_extra_surface_spawn_count": int(self._atlas_extra_surface_spawn_count),
            "atlas_init_gaussian_count_post_spawn": int(self._atlas_init_gaussian_count_post_spawn),
            "class_counts": class_counts,
            "state_counts": state_counts,
        }
        summary.update(self.summarize_atlas_refresh_snapshot())
        return summary

    def update_atlas_runtime_stats(
        self,
        photo_residuals,
        visible_mask,
        ema_decay: float,
        drift_radius_mult: float,
        camera_index: int | None = None,
        high_residual_threshold: float | None = None,
        warmup_only: bool = False,
    ):
        if not self.has_atlas_bindings:
            return {}
        self._ensure_atlas_runtime_state()

        with torch.no_grad():
            self._ensure_atlas_runtime_state()
            device = self._device()
            decay = float(np.clip(ema_decay, 0.0, 0.9999))
            ema_alpha = 1.0 - decay
            idle_alpha = min(max(ema_alpha * 0.02, 0.0), 1e-4)
            residuals = photo_residuals.detach().to(device=device, dtype=torch.float32)
            visible_mask = visible_mask.detach().to(device=device, dtype=torch.bool)
            runtime_observation = self._consume_runtime_observation_cache(visible_mask)
            candidate_mask = runtime_observation["candidate_mask"]
            projected_mask = runtime_observation["projected_mask"]
            observed_mask = runtime_observation["observed_mask"]
            visible_mask = visible_mask & observed_mask
            visibility_contribution = runtime_observation["visibility_contribution"].to(device=device, dtype=torch.float32)
            dark_region_mask = runtime_observation["dark_region_mask"].to(device=device, dtype=torch.bool)
            smooth_region_mask = runtime_observation["smooth_region_mask"].to(device=device, dtype=torch.bool)
            floater_region_mask = runtime_observation["floater_region_mask"].to(device=device, dtype=torch.bool)
            patch_quality_score = runtime_observation["patch_quality_score"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            mask_nonzero_ratio = runtime_observation["mask_nonzero_ratio"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            bg_like_ratio = runtime_observation["bg_like_ratio"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            background_like_ratio = runtime_observation["background_like_ratio"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            thin_support_like_ratio = runtime_observation["thin_support_like_ratio"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            photo_signal_strength = runtime_observation["photo_signal_strength"].to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            visibility_contribution = torch.where(
                candidate_mask,
                visibility_contribution.clamp(0.0, 1.0),
                torch.zeros_like(visibility_contribution),
            )

            support_score = self._compute_support_consistency_score()
            support_consistent_visible = visible_mask & (support_score >= 0.35)

            gaussian_photo_current = torch.ones_like(self._atlas_photo_ema, dtype=torch.float32, device=device)
            if torch.any(candidate_mask):
                gaussian_photo_current[candidate_mask] = (1.0 - visibility_contribution[candidate_mask]).clamp(0.0, 1.0)
            if torch.any(visible_mask):
                gaussian_photo_current[visible_mask] = residuals[visible_mask].clamp(0.0, 1.0)

            self._update_runtime_ema_tensor(
                self._atlas_photo_ema,
                gaussian_photo_current,
                candidate_mask,
                decay,
                idle_alpha=0.0,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_visibility_ema,
                visibility_contribution,
                candidate_mask,
                decay,
                idle_alpha=idle_alpha,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            if (not warmup_only) and high_residual_threshold is not None and self._atlas_high_residual_count.numel() == residuals.shape[0]:
                high_threshold = float(max(high_residual_threshold, 0.0))
                high_visible = visible_mask & torch.isfinite(residuals) & (residuals > high_threshold)
                low_threshold = high_threshold * 0.75
                low_visible = visible_mask & torch.isfinite(residuals) & (residuals <= low_threshold)
                mid_visible = visible_mask & (~high_visible) & (~low_visible)
                if torch.any(high_visible):
                    self._atlas_high_residual_count[high_visible] = self._atlas_high_residual_count[high_visible] + 1
                    self._atlas_low_residual_count[high_visible] = 0
                if torch.any(low_visible):
                    self._atlas_low_residual_count[low_visible] = self._atlas_low_residual_count[low_visible] + 1
                    self._atlas_high_residual_count[low_visible] = 0
                if torch.any(mid_visible):
                    self._atlas_high_residual_count[mid_visible] = 0
                    self._atlas_low_residual_count[mid_visible] = 0
                if self._atlas_state_cooldown.numel() == residuals.shape[0]:
                    self._atlas_state_cooldown.sub_(1)
                    self._atlas_state_cooldown.clamp_(min=0)
            if camera_index is not None and torch.any(visible_mask):
                current_ref_score = (1.0 - residuals[visible_mask]).clamp(0.0, 1.0)
                support_multiplier = (0.35 + 0.65 * support_score[visible_mask].clamp(0.0, 1.0)).clamp(0.35, 1.0)
                current_ref_score = current_ref_score * support_multiplier
                visible_indices = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
                better_ref = current_ref_score > self._atlas_ref_score[visible_indices]
                if torch.any(better_ref):
                    better_indices = visible_indices[better_ref]
                    self._atlas_ref_camera[better_indices] = int(camera_index)
                    self._atlas_ref_score[better_indices] = current_ref_score[better_ref]

            if not warmup_only:
                atlas_positions = self.get_gaussian_atlas_positions.detach()
                atlas_radius = self.get_gaussian_atlas_radius.detach()
                drift_distance = torch.linalg.norm(self._xyz.detach() - atlas_positions, dim=1)
                self._atlas_drift_flag = drift_distance > (drift_radius_mult * atlas_radius.clamp_min(1e-6))
                if self._atlas_drift_count.numel() == self._atlas_drift_flag.shape[0]:
                    drifted = self._atlas_drift_flag
                    if torch.any(drifted):
                        self._atlas_drift_count[drifted] = self._atlas_drift_count[drifted] + 1
                    self._atlas_drift_count[~drifted] = 0
                not_pending = self._atlas_state != GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
                self._atlas_gc_fail_count[(~self._atlas_drift_flag) & not_pending] = 0

            node_count = int(self._atlas_positions.shape[0])
            node_population = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_candidate = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_projected = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_visibility = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_residual = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_support_consistent = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_ref_consistent = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_coverage_candidate = torch.zeros((node_count,), dtype=torch.float32, device=device)

            ones_all = torch.ones((self._atlas_node_ids.shape[0],), dtype=torch.float32, device=device)
            node_population.index_add_(0, self._atlas_node_ids, ones_all)
            if torch.any(candidate_mask):
                candidate_ids = self._atlas_node_ids[candidate_mask]
                candidate_state = self._atlas_state[candidate_mask]
                candidate_weight = torch.ones((candidate_ids.shape[0],), dtype=torch.float32, device=device)
                candidate_weight = torch.where(
                    candidate_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                    torch.full_like(candidate_weight, 0.25),
                    candidate_weight,
                )
                candidate_weight = torch.where(
                    candidate_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE,
                    torch.full_like(candidate_weight, 0.55),
                    candidate_weight,
                )
                candidate_weight = torch.where(
                    candidate_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
                    torch.zeros_like(candidate_weight),
                    candidate_weight,
                )
                coverage_weight = torch.where(
                    candidate_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                    torch.zeros_like(candidate_weight),
                    candidate_weight,
                )
                node_candidate.index_add_(0, candidate_ids, candidate_weight)
                node_coverage_candidate.index_add_(0, candidate_ids, coverage_weight)
                node_visibility.index_add_(0, candidate_ids, visibility_contribution[candidate_mask] * candidate_weight)
                node_residual.index_add_(0, candidate_ids, gaussian_photo_current[candidate_mask] * candidate_weight)
                node_support_consistent.index_add_(
                    0,
                    candidate_ids,
                    (support_score[candidate_mask] * projected_mask[candidate_mask].float() * candidate_weight).clamp(0.0, 1.0),
                )

                ref_consistency = torch.full_like(visibility_contribution, 0.5)
                ref_consistency[~projected_mask] = 0.0
                if camera_index is not None:
                    known_ref = self._atlas_ref_camera >= 0
                    same_ref = projected_mask & known_ref & (self._atlas_ref_camera == int(camera_index))
                    other_ref = projected_mask & known_ref & (~same_ref)
                    ref_consistency[same_ref] = 1.0
                    ref_consistency[other_ref] = (0.20 + 0.50 * self._atlas_ref_score[other_ref].clamp(0.0, 1.0)).clamp(0.20, 0.70)
                node_ref_consistent.index_add_(0, candidate_ids, ref_consistency[candidate_mask].clamp(0.0, 1.0) * candidate_weight)
            if torch.any(projected_mask):
                projected_ids = self._atlas_node_ids[projected_mask]
                projected_state = self._atlas_state[projected_mask]
                projected_weight = torch.ones((projected_ids.shape[0],), dtype=torch.float32, device=device)
                projected_weight = torch.where(
                    projected_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                    torch.full_like(projected_weight, 0.25),
                    projected_weight,
                )
                projected_weight = torch.where(
                    projected_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE,
                    torch.full_like(projected_weight, 0.55),
                    projected_weight,
                )
                projected_weight = torch.where(
                    projected_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING,
                    torch.zeros_like(projected_weight),
                    projected_weight,
                )
                node_projected.index_add_(0, projected_ids, projected_weight)

            node_updated = node_candidate > 0
            node_candidate_denom = node_candidate.clamp_min(1.0)
            node_update_strength = node_candidate.clamp(0.0, 1.0)
            coverage_update = node_coverage_candidate >= 0.50
            node_visibility_current = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_photo_current = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_support_current = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_finite_projection_current = torch.zeros((node_count,), dtype=torch.float32, device=device)
            node_ref_current = torch.zeros((node_count,), dtype=torch.float32, device=device)
            if torch.any(node_updated):
                node_visibility_current[node_updated] = (node_visibility[node_updated] / node_candidate_denom[node_updated]).clamp(0.0, 1.0)
                node_photo_current[node_updated] = (node_residual[node_updated] / node_candidate_denom[node_updated]).clamp(0.0, 1.0)
                node_support_current[node_updated] = (node_support_consistent[node_updated] / node_candidate_denom[node_updated]).clamp(0.0, 1.0)
                node_finite_projection_current[node_updated] = (node_projected[node_updated] / node_candidate_denom[node_updated]).clamp(0.0, 1.0)
                node_ref_current[node_updated] = (node_ref_consistent[node_updated] / node_candidate_denom[node_updated]).clamp(0.0, 1.0)
            node_obs_quality_current = self._combine_node_obs_quality(
                node_photo_current,
                node_support_current,
                node_finite_projection_current,
                node_ref_current,
            )

            self._update_runtime_ema_tensor(
                self._atlas_node_photo_ema,
                node_photo_current,
                node_updated,
                decay,
                idle_alpha=0.0,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_node_visibility_ema,
                node_visibility_current,
                node_updated,
                decay,
                idle_alpha=idle_alpha,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_node_obs_quality_ema,
                node_obs_quality_current,
                node_updated,
                decay,
                idle_alpha=idle_alpha,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_node_support_consistency_ema,
                node_support_current,
                node_updated,
                decay,
                idle_alpha=idle_alpha,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_node_finite_projection_ema,
                node_finite_projection_current,
                node_updated,
                decay,
                idle_alpha=idle_alpha,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_node_ref_consistency_ema,
                node_ref_current,
                node_updated,
                decay,
                idle_alpha=idle_alpha,
                clamp_min=0.0,
                clamp_max=1.0,
            )
            self._update_runtime_ema_tensor(
                self._atlas_node_updated_recently,
                node_update_strength,
                node_updated,
                decay,
                idle_alpha=idle_alpha,
                idle_target=0.0,
                clamp_min=0.0,
                clamp_max=1.0,
            )

            node_observed_score_current = self._combine_node_observed_score(
                node_visibility_current,
                node_obs_quality_current,
                node_update_strength,
            )
            node_observed_score_ema = self._combine_node_observed_score(
                self._atlas_node_visibility_ema,
                self._atlas_node_obs_quality_ema,
                self._atlas_node_updated_recently,
            )
            self._atlas_node_observed_score_ema.copy_(node_observed_score_ema.detach())
            self._atlas_node_observed_count.add_(node_observed_score_current.detach())
            self._atlas_node_support_consistent_count.add_(node_support_current.detach())
            observed_count_strength = (
                self._atlas_node_observed_count / (self._atlas_node_observed_count + 0.5)
            ).clamp(0.0, 1.0)

            if camera_index is not None:
                self._ensure_atlas_view_capacity(int(camera_index) + 1)
                self._atlas_view_weights[:, int(camera_index)] = (
                    self._atlas_view_weights[:, int(camera_index)] + node_observed_score_current
                )
                self._atlas_view_counts[:, int(camera_index)] = (
                    self._atlas_view_counts[:, int(camera_index)] + coverage_update.to(dtype=torch.int32)
                )

            observed_nodes = (
                ((node_observed_score_ema >= 0.08) | (observed_count_strength >= 0.22))
                & (
                    (self._atlas_node_visibility_ema >= 0.02)
                    | (self._atlas_node_updated_recently >= 0.05)
                )
                & (
                    (self._atlas_node_obs_quality_ema >= 0.05)
                    | (observed_count_strength >= 0.22)
                )
            )
            node_obs_quality = self._atlas_node_obs_quality_ema.clamp(0.0, 1.0)
            node_support_consistency_ratio = self._atlas_node_support_consistency_ema.clamp(0.0, 1.0)
            if self._atlas_view_counts.ndim == 2 and self._atlas_view_counts.shape[1] > 0:
                node_coverage_ratio = (self._atlas_view_counts > 0).float().mean(dim=1)
            else:
                node_coverage_ratio = torch.zeros((node_count,), dtype=torch.float32, device=device)
            if self._atlas_view_weights.ndim == 2 and self._atlas_view_weights.shape[1] > 0:
                nonnegative_view_weight = self._atlas_view_weights.clamp_min(0.0)
                topk = min(2, int(nonnegative_view_weight.shape[1]))
                top_values, _ = torch.topk(nonnegative_view_weight, k=topk, dim=1)
                top1 = top_values[:, 0]
                top2 = top_values[:, 1] if topk > 1 else torch.zeros_like(top1)
                node_ambiguity = torch.where(
                    top1 > 1e-6,
                    (top2 / top1.clamp_min(1e-6)).clamp(0.0, 1.0),
                    torch.zeros_like(top1),
                )
            else:
                node_ambiguity = torch.zeros((node_count,), dtype=torch.float32, device=device)

            floater_proxy_by_state = {}
            dark_region_completeness_by_state = {}
            for state_id, state_name in GAUSSIAN_STATE_NAMES.items():
                state_mask = self._atlas_state == int(state_id)
                state_projected = projected_mask & state_mask
                state_smooth = smooth_region_mask & state_mask
                state_floater = floater_region_mask & state_mask
                state_dark = dark_region_mask & state_mask
                floater_proxy_by_state[str(state_name)] = (
                    float(state_floater.float().sum().item() / max(float(state_smooth.float().sum().item()), 1.0))
                    if state_mask.numel() > 0
                    else 0.0
                )
                dark_region_completeness_by_state[str(state_name)] = (
                    float(state_dark.float().sum().item() / max(float(state_projected.float().sum().item()), 1.0))
                    if state_mask.numel() > 0
                    else 0.0
                )

            metrics = {
                "mean_photo_ema": float(self._atlas_photo_ema.mean().item()) if self._atlas_photo_ema.numel() > 0 else 0.0,
                "mean_visibility_ema": float(self._atlas_visibility_ema.mean().item()) if self._atlas_visibility_ema.numel() > 0 else 0.0,
                "mean_high_residual_iters": float(self._atlas_high_residual_count.float().mean().item()) if self._atlas_high_residual_count.numel() > 0 else 0.0,
                "max_high_residual_iters": int(self._atlas_high_residual_count.max().item()) if self._atlas_high_residual_count.numel() > 0 else 0,
                "mean_low_residual_iters": float(self._atlas_low_residual_count.float().mean().item()) if self._atlas_low_residual_count.numel() > 0 else 0.0,
                "max_low_residual_iters": int(self._atlas_low_residual_count.max().item()) if self._atlas_low_residual_count.numel() > 0 else 0,
                "mean_gc_fail_count": float(self._atlas_gc_fail_count.float().mean().item()) if self._atlas_gc_fail_count.numel() > 0 else 0.0,
                "max_gc_fail_count": int(self._atlas_gc_fail_count.max().item()) if self._atlas_gc_fail_count.numel() > 0 else 0,
                "drift_ratio": float(self._atlas_drift_flag.float().mean().item()) if self._atlas_drift_flag.numel() > 0 else 0.0,
                "persistent_drift_ratio": float((self._atlas_drift_count > 0).float().mean().item()) if self._atlas_drift_count.numel() > 0 else 0.0,
                "cooldown_ratio": float((self._atlas_state_cooldown > 0).float().mean().item()) if self._atlas_state_cooldown.numel() > 0 else 0.0,
                "ref_camera_ratio": float((self._atlas_ref_camera >= 0).float().mean().item()) if self._atlas_ref_camera.numel() > 0 else 0.0,
                "patch_quality_score": float(patch_quality_score[observed_mask].mean().item()) if torch.any(observed_mask) else 0.0,
                "mask_nonzero_ratio": float(mask_nonzero_ratio[observed_mask].mean().item()) if torch.any(observed_mask) else 0.0,
                "bg_like_ratio": float(bg_like_ratio[observed_mask].mean().item()) if torch.any(observed_mask) else 0.0,
                "background_like_ratio": float(background_like_ratio[observed_mask].mean().item()) if torch.any(observed_mask) else 0.0,
                "thin_support_like_ratio": float(thin_support_like_ratio[observed_mask].mean().item()) if torch.any(observed_mask) else 0.0,
                "photo_signal_strength": float(photo_signal_strength[observed_mask].mean().item()) if torch.any(observed_mask) else 0.0,
                "patch_quality_candidate_mean": float(patch_quality_score[candidate_mask].mean().item()) if torch.any(candidate_mask) else 0.0,
                "photo_signal_candidate_mean": float(photo_signal_strength[candidate_mask].mean().item()) if torch.any(candidate_mask) else 0.0,
                "updated_node_ratio": float(node_updated.float().mean().item()) if node_updated.numel() > 0 else 0.0,
                "updated_node_count": float(node_updated.sum().item()) if node_updated.numel() > 0 else 0.0,
                "coverage_node_update_count": float(coverage_update.sum().item()) if coverage_update.numel() > 0 else 0.0,
                "mean_node_update_strength": float(node_update_strength.mean().item()) if node_update_strength.numel() > 0 else 0.0,
                "observed_node_ratio": float(observed_nodes.float().mean().item()) if observed_nodes.numel() > 0 else 0.0,
                "observed_node_count": float(observed_nodes.sum().item()) if observed_nodes.numel() > 0 else 0.0,
                "mean_node_photo_ema": float(self._atlas_node_photo_ema.mean().item()) if self._atlas_node_photo_ema.numel() > 0 else 0.0,
                "mean_node_visibility_ema": float(self._atlas_node_visibility_ema.mean().item()) if self._atlas_node_visibility_ema.numel() > 0 else 0.0,
                "mean_node_obs_quality_ema": float(self._atlas_node_obs_quality_ema.mean().item()) if self._atlas_node_obs_quality_ema.numel() > 0 else 0.0,
                "mean_node_observed_score_current": float(node_observed_score_current.mean().item()) if node_observed_score_current.numel() > 0 else 0.0,
                "mean_node_observed_score_ema": float(node_observed_score_ema.mean().item()) if node_observed_score_ema.numel() > 0 else 0.0,
                "mean_node_updated_recently": float(self._atlas_node_updated_recently.mean().item()) if self._atlas_node_updated_recently.numel() > 0 else 0.0,
                "mean_node_observed_count": float(self._atlas_node_observed_count.mean().item()) if self._atlas_node_observed_count.numel() > 0 else 0.0,
                "mean_node_support_consistency": float(node_support_consistency_ratio.mean().item()) if node_support_consistency_ratio.numel() > 0 else 0.0,
                "mean_node_support_consistency_current": float(node_support_current.mean().item()) if node_support_current.numel() > 0 else 0.0,
                "mean_node_finite_projection_ema": float(self._atlas_node_finite_projection_ema.mean().item()) if self._atlas_node_finite_projection_ema.numel() > 0 else 0.0,
                "mean_node_ref_consistency_ema": float(self._atlas_node_ref_consistency_ema.mean().item()) if self._atlas_node_ref_consistency_ema.numel() > 0 else 0.0,
                "mean_node_coverage_ratio": float(node_coverage_ratio.mean().item()) if node_coverage_ratio.numel() > 0 else 0.0,
                "mean_node_ambiguity": float(node_ambiguity.mean().item()) if node_ambiguity.numel() > 0 else 0.0,
                "mean_node_obs_quality": float(node_obs_quality.mean().item()) if node_obs_quality.numel() > 0 else 0.0,
                "max_node_obs_quality": float(node_obs_quality.max().item()) if node_obs_quality.numel() > 0 else 0.0,
                "floater_proxy_by_state": floater_proxy_by_state,
                "dark_region_completeness_by_state": dark_region_completeness_by_state,
                "warmup_only": 1.0 if warmup_only else 0.0,
            }
            metrics.update(self.summarize_atlas_reliability_state())
            return metrics

    def refresh_atlas_after_warmup(
        self,
        alpha: float,
        gamma: float,
        min_reliability: float,
        min_visibility: float,
        refresh_low_band_power: float = 0.65,
        refresh_high_band_power: float = 1.05,
        refresh_support_consistency_weight: float = 0.12,
        refresh_visibility_weight: float = 0.14,
        refresh_override_min_evidence: float = 0.18,
    ):
        if not self.has_atlas_bindings:
            return {}

        with torch.no_grad():
            self._ensure_atlas_runtime_state()
            observed_count = self._atlas_node_observed_count.clamp_min(0.0)
            support_consistency_ratio = self._atlas_node_support_consistency_ema.clamp(0.0, 1.0)
            finite_projection_ema = self._atlas_node_finite_projection_ema.clamp(0.0, 1.0)
            ref_consistency_ema = self._atlas_node_ref_consistency_ema.clamp(0.0, 1.0)
            observed_score_ema = self._atlas_node_observed_score_ema.clamp(0.0, 1.0)
            obs_quality_ema = self._atlas_node_obs_quality_ema.clamp(0.0, 1.0)
            updated_recently = self._atlas_node_updated_recently.clamp(0.0, 1.0)
            if self._atlas_view_counts.ndim == 2 and self._atlas_view_counts.shape[1] > 0:
                node_coverage_ratio = (self._atlas_view_counts > 0).float().mean(dim=1)
            else:
                node_coverage_ratio = torch.zeros_like(observed_count)
            if self._atlas_view_weights.ndim == 2 and self._atlas_view_weights.shape[1] > 0:
                nonnegative_view_weight = self._atlas_view_weights.clamp_min(0.0)
                topk = min(2, int(nonnegative_view_weight.shape[1]))
                top_values, top_indices = torch.topk(nonnegative_view_weight, k=topk, dim=1)
                top1 = top_values[:, 0]
                top1_index = top_indices[:, 0].long()
                top2 = top_values[:, 1] if topk > 1 else torch.zeros_like(top1)
                best_ref_score = torch.where(
                    nonnegative_view_weight.sum(dim=1) > 1e-6,
                    top1 / nonnegative_view_weight.sum(dim=1).clamp_min(1e-6),
                    torch.zeros_like(top1),
                )
                ambiguity = torch.where(
                    top1 > 1e-6,
                    (top2 / top1.clamp_min(1e-6)).clamp(0.0, 1.0),
                    torch.zeros_like(top1),
                )
            else:
                top1_index = torch.full_like(self._atlas_raw_score, -1, dtype=torch.long)
                best_ref_score = torch.zeros_like(observed_count)
                ambiguity = torch.zeros_like(observed_count)
            visibility_gate = (
                self._atlas_node_visibility_ema / max(float(min_visibility), 1e-4)
            ).clamp(0.0, 1.0)
            observed_count_strength = (observed_count / (observed_count + 0.5)).clamp(0.0, 1.0)
            observed_nodes = (
                ((observed_score_ema >= 0.08) | (observed_count_strength >= 0.22))
                & (
                    (self._atlas_node_visibility_ema >= max(float(min_visibility), 0.0) * 0.25)
                    | (updated_recently >= 0.05)
                )
                & ((obs_quality_ema >= 0.05) | (observed_count_strength >= 0.22))
            )
            snapshot_ref_camera_valid = (top1_index >= 0) & (best_ref_score > 1e-6)
            snapshot_count_gate = observed_count_strength >= 0.24
            snapshot_visibility_gate = self._atlas_node_visibility_ema >= max(float(min_visibility) * 0.35, 0.01)
            snapshot_finite_gate = finite_projection_ema >= 0.08
            snapshot_support_gate = support_consistency_ratio >= 0.12
            if self._atlas_refresh_done:
                runtime_std = float(self._atlas_reliability_effective.std(unbiased=False).item()) if self._atlas_reliability_effective.numel() > 0 else 0.0
                metrics = self.summarize_atlas_reliability_state()
                metrics.update(self.summarize_atlas_refresh_snapshot())
                metrics.update({
                    "observed_node_ratio": float(observed_nodes.float().mean().item()) if observed_nodes.numel() > 0 else 0.0,
                    "mean_node_photo_ema": float(self._atlas_node_photo_ema.mean().item()) if self._atlas_node_photo_ema.numel() > 0 else 0.0,
                    "mean_node_visibility_ema": float(self._atlas_node_visibility_ema.mean().item()) if self._atlas_node_visibility_ema.numel() > 0 else 0.0,
                    "mean_node_obs_quality_ema": float(obs_quality_ema.mean().item()) if obs_quality_ema.numel() > 0 else 0.0,
                    "mean_node_observed_score_ema": float(observed_score_ema.mean().item()) if observed_score_ema.numel() > 0 else 0.0,
                    "mean_node_updated_recently": float(updated_recently.mean().item()) if updated_recently.numel() > 0 else 0.0,
                    "mean_node_observed_count": float(observed_count.mean().item()) if observed_count.numel() > 0 else 0.0,
                    "mean_node_support_consistency": float(support_consistency_ratio.mean().item()) if support_consistency_ratio.numel() > 0 else 0.0,
                    "mean_node_finite_projection_ema": float(finite_projection_ema.mean().item()) if finite_projection_ema.numel() > 0 else 0.0,
                    "mean_node_ref_consistency_ema": float(ref_consistency_ema.mean().item()) if ref_consistency_ema.numel() > 0 else 0.0,
                    "mean_node_coverage_ratio": float(node_coverage_ratio.mean().item()) if node_coverage_ratio.numel() > 0 else 0.0,
                    "mean_node_ambiguity": float(ambiguity.mean().item()) if ambiguity.numel() > 0 else 0.0,
                    "refresh_override_count": float(self._atlas_refresh_runtime_override_mask.float().sum().item()) if self._atlas_refresh_runtime_override_mask.numel() > 0 else 0.0,
                    "refresh_override_ratio": float(self._atlas_refresh_runtime_override_mask.float().mean().item()) if self._atlas_refresh_runtime_override_mask.numel() > 0 else 0.0,
                    "refresh_std_before": runtime_std,
                    "refresh_std_after": runtime_std,
                    "refresh_std_before_after": {"before": runtime_std, "after": runtime_std},
                    "refresh_evidence_observed_gate_ratio": float(observed_nodes.float().mean().item()) if observed_nodes.numel() > 0 else 0.0,
                    "refresh_evidence_count_gate_ratio": float(snapshot_count_gate.float().mean().item()) if snapshot_count_gate.numel() > 0 else 0.0,
                    "refresh_evidence_visibility_gate_ratio": float(snapshot_visibility_gate.float().mean().item()) if snapshot_visibility_gate.numel() > 0 else 0.0,
                    "refresh_evidence_ref_gate_ratio": float(snapshot_ref_camera_valid.float().mean().item()) if snapshot_ref_camera_valid.numel() > 0 else 0.0,
                    "refresh_evidence_finite_gate_ratio": float(snapshot_finite_gate.float().mean().item()) if snapshot_finite_gate.numel() > 0 else 0.0,
                    "refresh_evidence_support_gate_ratio": float(snapshot_support_gate.float().mean().item()) if snapshot_support_gate.numel() > 0 else 0.0,
                    "refresh_evidence_override_gate_ratio": float(self._atlas_refresh_runtime_override_mask.float().mean().item()) if self._atlas_refresh_runtime_override_mask.numel() > 0 else 0.0,
                    "refresh_override_weight_positive_ratio": float((self._atlas_refresh_override_weight > 0.0).float().mean().item()) if self._atlas_refresh_override_weight.numel() > 0 else 0.0,
                    "refresh_applied": 0.0,
                })
                return metrics

            base_reliability = self._atlas_reliability_base.detach()
            refresh_std_before = float(base_reliability.std(unbiased=False).item()) if base_reliability.numel() > 0 else 0.0
            runtime_min = float(max(min_reliability, 1e-4))
            base_reliability = base_reliability.clamp(min=runtime_min, max=1.0)
            refresh_obs_quality = obs_quality_ema.detach()
            count_strength = (observed_count / (observed_count + 1.0)).clamp(0.0, 1.0)
            norm_mask = observed_nodes & torch.isfinite(refresh_obs_quality) & torch.isfinite(self._atlas_node_photo_ema)
            obs_quality_score = self._robust_percentile_score(refresh_obs_quality, norm_mask, 0.10, 0.90)
            photo_quality_score = self._robust_percentile_score(
                self._atlas_node_photo_ema.clamp_min(0.0),
                norm_mask,
                0.10,
                0.85,
                invert=True,
            )
            support_score = self._robust_percentile_score(support_consistency_ratio, norm_mask, 0.10, 0.90)
            ref_consistency_score = self._robust_percentile_score(ref_consistency_ema, norm_mask, 0.10, 0.90)
            ref_anchor_score = self._robust_percentile_score(
                best_ref_score,
                norm_mask & (top1_index >= 0),
                0.05,
                0.90,
            )
            ref_score = torch.maximum(
                ref_consistency_score,
                (0.70 * ref_anchor_score + 0.30 * ref_consistency_score).clamp(0.0, 1.0),
            )
            visibility_score = self._robust_percentile_score(self._atlas_node_visibility_ema.clamp_min(0.0), norm_mask, 0.10, 0.90)
            finite_score = self._robust_percentile_score(finite_projection_ema, norm_mask, 0.10, 0.90)
            count_score = torch.maximum(
                self._robust_percentile_score(observed_count, norm_mask, 0.10, 0.90),
                count_strength,
            ).clamp(0.0, 1.0)
            observed_score = self._robust_percentile_score(observed_score_ema, norm_mask, 0.10, 0.90)
            coverage_score = self._robust_percentile_score(node_coverage_ratio, norm_mask, 0.10, 0.90)

            freshness_gate = (0.55 + 0.45 * updated_recently).clamp(0.55, 1.0)
            ambiguity_gate = (1.0 - 0.45 * ambiguity.clamp(0.0, 1.0)).clamp(0.40, 1.0)
            evidence_support_gate = (
                0.35
                + 0.30 * support_score
                + 0.20 * finite_score
                + 0.15 * visibility_gate
            ).clamp(0.35, 1.0)
            support_weight = float(max(min(refresh_support_consistency_weight, 0.35), 0.02))
            visibility_weight = float(max(min(refresh_visibility_weight, 0.35), 0.02))
            quality_weights = {
                "observed": 0.22,
                "obs_quality": 0.16,
                "photo": 0.12,
                "visibility": visibility_weight,
                "support": support_weight,
                "count": 0.09,
                "coverage": 0.06,
                "finite": 0.05,
                "ref": 0.04,
            }
            quality_weight_sum = max(float(sum(quality_weights.values())), 1e-6)
            runtime_raw_quality = (
                quality_weights["observed"] * observed_score
                + quality_weights["obs_quality"] * obs_quality_score
                + quality_weights["photo"] * photo_quality_score
                + quality_weights["visibility"] * visibility_score
                + quality_weights["support"] * support_score
                + quality_weights["count"] * count_score
                + quality_weights["coverage"] * coverage_score
                + quality_weights["finite"] * finite_score
                + quality_weights["ref"] * ref_score
            ).div(quality_weight_sum).clamp(0.0, 1.0)
            runtime_raw_quality = torch.where(
                observed_nodes,
                (runtime_raw_quality * freshness_gate * ambiguity_gate * evidence_support_gate).clamp(0.0, 1.0),
                torch.zeros_like(runtime_raw_quality),
            )
            contrast_mask = norm_mask & observed_nodes & torch.isfinite(runtime_raw_quality)
            raw_quality_before_contrast = runtime_raw_quality.detach().clone()
            contrast_q20 = runtime_raw_quality.new_tensor(0.20)
            contrast_q50 = runtime_raw_quality.new_tensor(0.50)
            contrast_q82 = runtime_raw_quality.new_tensor(0.82)
            contrast_q94 = runtime_raw_quality.new_tensor(0.94)
            if int(contrast_mask.sum().item()) >= 16:
                contrast_sample = runtime_raw_quality[contrast_mask]
                quantiles = torch.quantile(
                    contrast_sample,
                    torch.tensor([0.20, 0.50, 0.82, 0.94], dtype=contrast_sample.dtype, device=contrast_sample.device),
                )
                contrast_q20, contrast_q50, contrast_q82, contrast_q94 = quantiles.unbind(dim=0)
                low_band = (
                    ((runtime_raw_quality - contrast_q20) / (contrast_q50 - contrast_q20).abs().clamp_min(1e-6))
                    .clamp(0.0, 1.0)
                    .pow(float(max(refresh_low_band_power, 0.05)))
                    * 0.42
                )
                mid_band = 0.42 + (
                    ((runtime_raw_quality - contrast_q50) / (contrast_q82 - contrast_q50).abs().clamp_min(1e-6))
                    .clamp(0.0, 1.0)
                    .pow(float(max(refresh_high_band_power, 0.05)))
                    * 0.38
                )
                high_band = 0.80 + (
                    ((runtime_raw_quality - contrast_q82) / (contrast_q94 - contrast_q82).abs().clamp_min(1e-6))
                    .clamp(0.0, 1.0)
                    .pow(0.58)
                    * 0.20
                )
                contrasted_quality = torch.where(
                    runtime_raw_quality < contrast_q50,
                    low_band,
                    torch.where(runtime_raw_quality < contrast_q82, mid_band, high_band),
                ).clamp(0.0, 1.0)
                runtime_raw_quality = torch.where(
                    contrast_mask,
                    (0.25 * runtime_raw_quality + 0.75 * contrasted_quality).clamp(0.0, 1.0),
                    (runtime_raw_quality * 0.65).clamp(0.0, 1.0),
                )
            # 2-band mapping: low band uses p<1 (convex) to spread low-mid upward,
            # high band uses p~1 to preserve separation above midpoint.
            q_split = 0.42
            low_unit = (
                (runtime_raw_quality / q_split)
                .clamp(0.0, 1.0)
                .pow(float(refresh_low_band_power))
                * 0.50
            )
            high_unit = 0.50 + (
                ((runtime_raw_quality - q_split) / (1.0 - q_split + 1e-6))
                .clamp(0.0, 1.0)
                .pow(float(refresh_high_band_power))
                * 0.50
            )
            mapped_unit = torch.where(
                runtime_raw_quality < q_split,
                low_unit,
                high_unit,
            ).clamp(0.0, 1.0)
            mid_tail_lift = 0.04 * self._linear_tensor_gate(
                runtime_raw_quality,
                float(contrast_q50.item()),
                float(contrast_q82.item()),
            )
            high_tail_lift = 0.07 * self._linear_tensor_gate(
                runtime_raw_quality,
                float(contrast_q82.item()),
                float(contrast_q94.item()),
            )
            mapped_unit = torch.where(
                observed_nodes,
                (mapped_unit + mid_tail_lift + high_tail_lift).clamp(0.0, 1.0),
                mapped_unit,
            )
            weak_evidence_mask = (
                (support_consistency_ratio < 0.08)
                | ((visibility_score < 0.10) & (count_score < 0.25))
                | (finite_projection_ema < 0.04)
            )
            mapped_unit = torch.where(
                weak_evidence_mask,
                torch.minimum(mapped_unit, torch.full_like(mapped_unit, 0.32)),
                mapped_unit,
            )
            runtime_mapped = (runtime_min + (1.0 - runtime_min) * mapped_unit).clamp(min=runtime_min, max=1.0)
            runtime_mapped = torch.where(
                observed_nodes,
                runtime_mapped,
                torch.full_like(runtime_mapped, runtime_min),
            )

            ref_camera_valid = (top1_index >= 0) & (best_ref_score > 1e-6)
            support_gate_ok = (support_consistency_ratio >= 0.12) | (
                (obs_quality_score >= 0.55) & (finite_projection_ema >= 0.16)
            )
            base_runtime_override_gate = (
                observed_nodes
                & (count_strength >= 0.24)
                & (self._atlas_node_visibility_ema >= max(float(min_visibility) * 0.35, 0.01))
                & ref_camera_valid
                & (finite_projection_ema >= 0.08)
                & support_gate_ok
            )
            strong_runtime_evidence_gate = (
                observed_nodes
                & (count_strength >= 0.20)
                & ref_camera_valid
                & (finite_projection_ema >= 0.08)
                & (support_score >= 0.35)
                & ((visibility_score >= 0.18) | (count_score >= 0.55))
            )
            evidence_quality = (
                0.17 * obs_quality_score
                + 0.17 * photo_quality_score
                + 0.18 * support_score
                + 0.16 * ref_score
                + 0.13 * visibility_score
                + 0.10 * count_score
                + 0.06 * finite_score
                + 0.03 * coverage_score
            ).clamp(0.0, 1.0)
            evidence_gate = self._linear_tensor_gate(
                evidence_quality,
                float(refresh_override_min_evidence),
                float(refresh_override_min_evidence) + 0.40,
            )
            runtime_override_gate = base_runtime_override_gate | (
                strong_runtime_evidence_gate
                & (evidence_quality >= float(refresh_override_min_evidence) + 0.18)
            )
            high_runtime_weight = 0.15 * self._linear_tensor_gate(runtime_mapped, 0.45, 0.75)
            override_value = (0.25 + 0.60 * evidence_gate + high_runtime_weight).clamp(0.0, 1.0)
            # Only gate-passing nodes are allowed to override base reliability.
            # Non-gate nodes keep the strict foundation prior.
            override_weight = torch.where(
                runtime_override_gate,
                override_value,
                torch.zeros_like(evidence_gate),
            )
            refreshed = torch.lerp(base_reliability, runtime_mapped, override_weight).clamp(min=runtime_min, max=1.0)
            runtime_override_mask = runtime_override_gate
            self._atlas_reliability_runtime_raw.copy_(runtime_raw_quality.detach())
            self._atlas_reliability_runtime_mapped.copy_(runtime_mapped.detach())
            self._atlas_reliability_effective.copy_(refreshed.detach())
            self._atlas_reliability_runtime = self._atlas_reliability_effective.detach().clone()
            refresh_std_after = float(self._atlas_reliability_effective.std(unbiased=False).item()) if self._atlas_reliability_effective.numel() > 0 else 0.0
            self._atlas_refresh_observed_mask.copy_(observed_nodes.detach())
            self._atlas_refresh_node_photo_ema.copy_(self._atlas_node_photo_ema.detach())
            self._atlas_refresh_node_visibility_ema.copy_(self._atlas_node_visibility_ema.detach())
            self._atlas_refresh_obs_quality.copy_(refresh_obs_quality.detach())
            self._atlas_refresh_node_observed_count.copy_(observed_count.detach())
            self._atlas_refresh_node_support_consistent_ratio.copy_(support_consistency_ratio.detach())
            self._atlas_refresh_node_coverage_ratio.copy_(node_coverage_ratio.detach())
            self._atlas_refresh_node_ambiguity.copy_(ambiguity.detach())
            self._atlas_refresh_override_weight.copy_(override_weight.detach())
            self._atlas_refresh_runtime_override_mask.copy_(runtime_override_mask.detach())
            if top1_index.numel() == self._atlas_positions.shape[0] and self._atlas_ref_camera.numel() == self._atlas_node_ids.shape[0]:
                mapped_node_camera = top1_index[self._atlas_node_ids]
                mapped_node_score = (best_ref_score * override_weight).clamp(0.0, 1.0)[self._atlas_node_ids]
                valid_mapped = mapped_node_camera >= 0
                missing_ref = self._atlas_ref_camera < 0
                same_camera_weaker = (
                    (self._atlas_ref_camera == mapped_node_camera)
                    & (mapped_node_score > self._atlas_ref_score)
                )
                adopt_mask = valid_mapped & (missing_ref | same_camera_weaker)
                if torch.any(adopt_mask):
                    self._atlas_ref_camera[adopt_mask] = mapped_node_camera[adopt_mask]
                    self._atlas_ref_score[adopt_mask] = mapped_node_score[adopt_mask]
            self._atlas_refresh_done = True
            metrics = self.summarize_atlas_reliability_state()
            metrics.update(self.summarize_atlas_refresh_snapshot())
            metrics.update({
                "observed_node_ratio": float(observed_nodes.float().mean().item()) if observed_nodes.numel() > 0 else 0.0,
                "mean_node_photo_ema": float(self._atlas_node_photo_ema.mean().item()) if self._atlas_node_photo_ema.numel() > 0 else 0.0,
                "mean_node_visibility_ema": float(self._atlas_node_visibility_ema.mean().item()) if self._atlas_node_visibility_ema.numel() > 0 else 0.0,
                "mean_node_obs_quality_ema": float(obs_quality_ema.mean().item()) if obs_quality_ema.numel() > 0 else 0.0,
                "mean_node_observed_score_ema": float(observed_score_ema.mean().item()) if observed_score_ema.numel() > 0 else 0.0,
                "mean_node_updated_recently": float(updated_recently.mean().item()) if updated_recently.numel() > 0 else 0.0,
                "mean_node_observed_count": float(observed_count.mean().item()) if observed_count.numel() > 0 else 0.0,
                "mean_node_support_consistency": float(support_consistency_ratio.mean().item()) if support_consistency_ratio.numel() > 0 else 0.0,
                "mean_node_finite_projection_ema": float(finite_projection_ema.mean().item()) if finite_projection_ema.numel() > 0 else 0.0,
                "mean_node_ref_consistency_ema": float(ref_consistency_ema.mean().item()) if ref_consistency_ema.numel() > 0 else 0.0,
                "mean_node_coverage_ratio": float(node_coverage_ratio.mean().item()) if node_coverage_ratio.numel() > 0 else 0.0,
                "mean_node_ambiguity": float(ambiguity.mean().item()) if ambiguity.numel() > 0 else 0.0,
                "runtime_override_count": float(runtime_override_mask.sum().item()) if runtime_override_mask.numel() > 0 else 0.0,
                "runtime_override_ratio": float(runtime_override_mask.float().mean().item()) if runtime_override_mask.numel() > 0 else 0.0,
                "refresh_override_count": float(runtime_override_mask.sum().item()) if runtime_override_mask.numel() > 0 else 0.0,
                "refresh_override_ratio": float(runtime_override_mask.float().mean().item()) if runtime_override_mask.numel() > 0 else 0.0,
                "keep_base_count": float((~runtime_override_mask).sum().item()) if runtime_override_mask.numel() > 0 else 0.0,
                "runtime_reliability_std": float(self._atlas_reliability_effective.std(unbiased=False).item()) if self._atlas_reliability_effective.numel() > 0 else 0.0,
                "refresh_std_before": refresh_std_before,
                "refresh_std_after": refresh_std_after,
                "refresh_std_before_after": {"before": refresh_std_before, "after": refresh_std_after},
                "refresh_quality_threshold": 0.42,
                "refresh_mapping_mid_threshold": 0.72,
                "refresh_raw_pre_contrast_mean": float(raw_quality_before_contrast.mean().item()) if raw_quality_before_contrast.numel() > 0 else 0.0,
                "refresh_contrast_q20": float(contrast_q20.item()),
                "refresh_contrast_q50": float(contrast_q50.item()),
                "refresh_contrast_q82": float(contrast_q82.item()),
                "refresh_contrast_q94": float(contrast_q94.item()),
                "refresh_runtime_raw_quality_mean": float(runtime_raw_quality.mean().item()) if runtime_raw_quality.numel() > 0 else 0.0,
                "refresh_runtime_mapped_mean": float(runtime_mapped.mean().item()) if runtime_mapped.numel() > 0 else 0.0,
                "refresh_runtime_effective_mean": float(refreshed.mean().item()) if refreshed.numel() > 0 else 0.0,
                "refresh_support_consistency_weight": float(support_weight),
                "refresh_visibility_weight": float(visibility_weight),
                "refresh_quality_evidence_mean": float(evidence_quality.mean().item()) if evidence_quality.numel() > 0 else 0.0,
                "refresh_evidence_gate_mean": float(evidence_gate.mean().item()) if evidence_gate.numel() > 0 else 0.0,
                "refresh_evidence_observed_gate_ratio": float(observed_nodes.float().mean().item()) if observed_nodes.numel() > 0 else 0.0,
                "refresh_evidence_count_gate_ratio": float((count_strength >= 0.24).float().mean().item()) if count_strength.numel() > 0 else 0.0,
                "refresh_evidence_visibility_gate_ratio": float((self._atlas_node_visibility_ema >= max(float(min_visibility) * 0.35, 0.01)).float().mean().item()) if self._atlas_node_visibility_ema.numel() > 0 else 0.0,
                "refresh_evidence_ref_gate_ratio": float(ref_camera_valid.float().mean().item()) if ref_camera_valid.numel() > 0 else 0.0,
                "refresh_evidence_finite_gate_ratio": float((finite_projection_ema >= 0.08).float().mean().item()) if finite_projection_ema.numel() > 0 else 0.0,
                "refresh_evidence_support_gate_ratio": float(support_gate_ok.float().mean().item()) if support_gate_ok.numel() > 0 else 0.0,
                "refresh_base_runtime_override_gate_ratio": float(base_runtime_override_gate.float().mean().item()) if base_runtime_override_gate.numel() > 0 else 0.0,
                "refresh_strong_runtime_evidence_ratio": float(strong_runtime_evidence_gate.float().mean().item()) if strong_runtime_evidence_gate.numel() > 0 else 0.0,
                "refresh_evidence_override_gate_ratio": float(runtime_override_gate.float().mean().item()) if runtime_override_gate.numel() > 0 else 0.0,
                "refresh_override_weight_mean": float(override_weight.mean().item()) if override_weight.numel() > 0 else 0.0,
                "refresh_override_weight_positive_ratio": float((override_weight > 0.0).float().mean().item()) if override_weight.numel() > 0 else 0.0,
                "refresh_ref_valid_ratio": float(ref_camera_valid.float().mean().item()) if ref_camera_valid.numel() > 0 else 0.0,
                "refresh_support_gate_ratio": float(support_gate_ok.float().mean().item()) if support_gate_ok.numel() > 0 else 0.0,
                "refresh_finite_gate_ratio": float((finite_projection_ema >= 0.08).float().mean().item()) if finite_projection_ema.numel() > 0 else 0.0,
                "mean_refresh_source_raw_score": float(self._atlas_raw_score.mean().item()) if self._atlas_raw_score.numel() > 0 else 0.0,
                "refresh_applied": 1.0,
            })
            metrics.update(self._override_bucket_stats(base_reliability, runtime_override_mask, "refresh_override_base_bucket"))
            metrics.update(self._override_bucket_stats(evidence_quality, runtime_override_mask, "refresh_override_evidence_bucket"))
            return metrics

    def apply_uncertainty_guardrails(
        self,
        camera_centers: torch.Tensor | None = None,
        fallback_camera_center: torch.Tensor | None = None,
        slab_radius_mult: float = 2.0,
        ray_cap_fraction: float = 1.0,
        parallel_min_ratio: float = 0.03,
        parallel_max_ratio: float = 0.45,
        support_min_ratio: float = 0.01,
        support_max_ratio: float = 0.20,
        passive_parallel_max_mult: float = 1.35,
        passive_support_max_mult: float = 1.25,
        active_parallel_min_mult: float = 1.50,
        active_parallel_max_mult: float = 2.25,
        active_support_min_mult: float = 0.75,
        active_support_max_mult: float = 1.50,
        active_ray_min_fraction: float = 0.10,
        active_low_visibility_decay: float = 0.995,
        decay: float = 0.98,
        low_visibility_threshold: float = 0.05,
    ):
        metrics = {
            "sigma_parallel_clamp_hits": 0.0,
            "sigma_support_clamp_hits": 0.0,
            "sigma_ray_clamp_hits": 0.0,
            "sigma_ray_floor_hits": 0.0,
            "sigma_active_ray_valid_count": 0.0,
            "sigma_active_ray_unresolved_count": 0.0,
            "sigma_parallel_clamp_hits_stable": 0.0,
            "sigma_parallel_clamp_hits_passive": 0.0,
            "sigma_parallel_clamp_hits_active": 0.0,
            "sigma_support_clamp_hits_stable": 0.0,
            "sigma_support_clamp_hits_passive": 0.0,
            "sigma_support_clamp_hits_active": 0.0,
            "sigma_parallel_mean": 0.0,
            "sigma_support_mean": 0.0,
            "sigma_stable_parallel_mean": 0.0,
            "sigma_stable_support_mean": 0.0,
            "sigma_passive_parallel_mean": 0.0,
            "sigma_passive_support_mean": 0.0,
            "sigma_active_parallel_mean": 0.0,
            "sigma_active_support_mean": 0.0,
            "sigma_active_parallel_p50": 0.0,
            "sigma_active_parallel_p90": 0.0,
            "sigma_active_support_p50": 0.0,
            "sigma_active_support_p90": 0.0,
            "sigma_active_ray_span_mean": 0.0,
            "sigma_active_ray_floor_mean": 0.0,
            "sigma_active_ray_cap_mean": 0.0,
            "sigma_active_ray_parallel_mean": 0.0,
            "sigma_active_ray_parallel_p90": 0.0,
        }
        if not self.has_atlas_bindings or self._center_log_sigma_parallel.numel() == 0:
            return metrics

        with torch.no_grad():
            atlas_radius = self.get_gaussian_atlas_radius.detach().clamp_min(1e-4)
            state = self.get_atlas_state.detach()
            visibility_ema = self._atlas_visibility_ema.detach().clamp(0.0, 1.0)
            sigma_parallel = self.get_center_sigma_parallel.detach().squeeze(-1)
            sigma_support = self.get_center_sigma_support.detach().squeeze(-1)

            parallel_floor = (float(max(parallel_min_ratio, 0.0)) * atlas_radius).clamp_min(1e-6)
            parallel_cap = torch.maximum(float(max(parallel_max_ratio, 0.0)) * atlas_radius, parallel_floor)
            support_floor = (float(max(support_min_ratio, 0.0)) * atlas_radius).clamp_min(1e-6)
            support_cap = torch.maximum(float(max(support_max_ratio, 0.0)) * atlas_radius, support_floor)

            stable_mask = state == GAUSSIAN_STATE_STABLE
            passive_mask = state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
            active_mask = state == GAUSSIAN_STATE_UNSTABLE_ACTIVE

            state_parallel_floor = parallel_floor.clone()
            state_parallel_cap = parallel_cap.clone()
            state_support_floor = support_floor.clone()
            state_support_cap = support_cap.clone()

            if torch.any(passive_mask):
                state_parallel_cap[passive_mask] = torch.maximum(
                    state_parallel_cap[passive_mask] * float(max(passive_parallel_max_mult, 1.0)),
                    state_parallel_floor[passive_mask],
                )
                state_support_cap[passive_mask] = torch.maximum(
                    state_support_cap[passive_mask] * float(max(passive_support_max_mult, 1.0)),
                    state_support_floor[passive_mask],
                )
            if torch.any(active_mask):
                state_parallel_floor[active_mask] = state_parallel_floor[active_mask] * float(max(active_parallel_min_mult, 0.0))
                state_parallel_cap[active_mask] = torch.maximum(
                    state_parallel_cap[active_mask] * float(max(active_parallel_max_mult, 1.0)),
                    state_parallel_floor[active_mask],
                )
                state_support_floor[active_mask] = state_support_floor[active_mask] * float(max(active_support_min_mult, 0.0))
                state_support_cap[active_mask] = torch.maximum(
                    state_support_cap[active_mask] * float(max(active_support_max_mult, 1.0)),
                    state_support_floor[active_mask],
                )

            low_visibility = visibility_ema <= float(max(low_visibility_threshold, 0.0))
            inflated_parallel = sigma_parallel > (state_parallel_floor * 1.25)
            inflated_support = sigma_support > (state_support_floor * 1.25)
            non_active_decay_mask = (low_visibility & (~active_mask)) | ((~active_mask) & (inflated_parallel | inflated_support))
            if torch.any(non_active_decay_mask):
                sigma_parallel[non_active_decay_mask] = torch.maximum(
                    state_parallel_floor[non_active_decay_mask],
                    sigma_parallel[non_active_decay_mask] * float(decay),
                )
                sigma_support[non_active_decay_mask] = torch.maximum(
                    state_support_floor[non_active_decay_mask],
                    sigma_support[non_active_decay_mask] * float(decay),
                )

            active_low_visibility_mask = low_visibility & active_mask
            if torch.any(active_low_visibility_mask):
                active_decay = float(min(max(active_low_visibility_decay, decay), 1.0))
                sigma_parallel[active_low_visibility_mask] = torch.maximum(
                    state_parallel_floor[active_low_visibility_mask],
                    sigma_parallel[active_low_visibility_mask] * active_decay,
                )
                sigma_support[active_low_visibility_mask] = torch.maximum(
                    state_support_floor[active_low_visibility_mask],
                    sigma_support[active_low_visibility_mask] * active_decay,
                )

            ray_clamp_hits = 0
            ray_floor_hits = 0
            active_ray_valid_count = 0
            active_ray_span_stats = torch.empty((0,), dtype=sigma_parallel.dtype, device=sigma_parallel.device)
            active_ray_floor_stats = torch.empty((0,), dtype=sigma_parallel.dtype, device=sigma_parallel.device)
            active_ray_cap_stats = torch.empty((0,), dtype=sigma_parallel.dtype, device=sigma_parallel.device)
            active_ray_parallel_stats = torch.empty((0,), dtype=sigma_parallel.dtype, device=sigma_parallel.device)
            if torch.any(active_mask):
                slab = compute_point_slab_bounds(
                    self,
                    active_mask,
                    camera_centers=camera_centers,
                    fallback_camera_center=fallback_camera_center,
                    slab_radius_mult=float(slab_radius_mult),
                    detach_points=True,
                    require_valid_ref_camera=True,
                    min_reference_score=0.05,
                    repair_ref_camera=True,
                )
                if slab is not None:
                    selected_indices = slab["point_indices"].to(device=sigma_parallel.device, dtype=torch.long)
                    ray_span = (slab["tau_max"] - slab["tau_min"]).clamp_min(1e-4)
                    ray_floor = (
                        float(max(active_ray_min_fraction, 0.0)) * ray_span
                    ).clamp_min(state_parallel_floor[selected_indices])
                    ray_cap = torch.maximum(
                        0.5 * float(max(ray_cap_fraction, 0.0)) * ray_span,
                        ray_floor,
                    )
                    selected_sigma = sigma_parallel[selected_indices]
                    ray_floor_hits = int((selected_sigma < ray_floor).sum().item())
                    ray_clamp_hits = int((selected_sigma > ray_cap).sum().item())
                    sigma_parallel[selected_indices] = torch.minimum(torch.maximum(selected_sigma, ray_floor), ray_cap)
                    active_ray_valid_count = int(selected_indices.numel())
                    active_ray_span_stats = ray_span.detach()
                    active_ray_floor_stats = ray_floor.detach()
                    active_ray_cap_stats = ray_cap.detach()
                    active_ray_parallel_stats = sigma_parallel[selected_indices].detach()

            active_count = int(active_mask.sum().item())
            active_ray_unresolved_count = max(active_count - active_ray_valid_count, 0)
            parallel_clamp_mask = (sigma_parallel < state_parallel_floor) | (sigma_parallel > state_parallel_cap)
            support_clamp_mask = (sigma_support < state_support_floor) | (sigma_support > state_support_cap)
            parallel_hits = int(parallel_clamp_mask.sum().item())
            support_hits = int(support_clamp_mask.sum().item())
            sigma_parallel = torch.minimum(torch.maximum(sigma_parallel, state_parallel_floor), state_parallel_cap)
            sigma_support = torch.minimum(torch.maximum(sigma_support, state_support_floor), state_support_cap)

            self._center_log_sigma_parallel.copy_(torch.log(sigma_parallel.unsqueeze(-1).clamp_min(1e-6)))
            self._center_log_sigma_support.copy_(torch.log(sigma_support.unsqueeze(-1).clamp_min(1e-6)))

            metrics["sigma_parallel_clamp_hits"] = float(parallel_hits)
            metrics["sigma_support_clamp_hits"] = float(support_hits)
            metrics["sigma_ray_clamp_hits"] = float(ray_clamp_hits)
            metrics["sigma_ray_floor_hits"] = float(ray_floor_hits)
            metrics["sigma_active_ray_valid_count"] = float(active_ray_valid_count)
            metrics["sigma_active_ray_unresolved_count"] = float(active_ray_unresolved_count)
            metrics["sigma_parallel_clamp_hits_stable"] = float((parallel_clamp_mask & stable_mask).sum().item())
            metrics["sigma_parallel_clamp_hits_passive"] = float((parallel_clamp_mask & passive_mask).sum().item())
            metrics["sigma_parallel_clamp_hits_active"] = float((parallel_clamp_mask & active_mask).sum().item())
            metrics["sigma_support_clamp_hits_stable"] = float((support_clamp_mask & stable_mask).sum().item())
            metrics["sigma_support_clamp_hits_passive"] = float((support_clamp_mask & passive_mask).sum().item())
            metrics["sigma_support_clamp_hits_active"] = float((support_clamp_mask & active_mask).sum().item())
            metrics["sigma_parallel_mean"] = float(sigma_parallel.mean().item()) if sigma_parallel.numel() > 0 else 0.0
            metrics["sigma_support_mean"] = float(sigma_support.mean().item()) if sigma_support.numel() > 0 else 0.0
            if torch.any(stable_mask):
                metrics["sigma_stable_parallel_mean"] = float(sigma_parallel[stable_mask].mean().item())
                metrics["sigma_stable_support_mean"] = float(sigma_support[stable_mask].mean().item())
            if torch.any(passive_mask):
                metrics["sigma_passive_parallel_mean"] = float(sigma_parallel[passive_mask].mean().item())
                metrics["sigma_passive_support_mean"] = float(sigma_support[passive_mask].mean().item())
            if torch.any(active_mask):
                active_parallel = sigma_parallel[active_mask]
                active_support = sigma_support[active_mask]
                metrics["sigma_active_parallel_mean"] = float(active_parallel.mean().item())
                metrics["sigma_active_support_mean"] = float(active_support.mean().item())
                if active_parallel.numel() > 0:
                    metrics["sigma_active_parallel_p50"] = float(torch.quantile(active_parallel, 0.50).item())
                    metrics["sigma_active_parallel_p90"] = float(torch.quantile(active_parallel, 0.90).item())
                if active_support.numel() > 0:
                    metrics["sigma_active_support_p50"] = float(torch.quantile(active_support, 0.50).item())
                    metrics["sigma_active_support_p90"] = float(torch.quantile(active_support, 0.90).item())
            if active_ray_span_stats.numel() > 0:
                metrics["sigma_active_ray_span_mean"] = float(active_ray_span_stats.mean().item())
                metrics["sigma_active_ray_floor_mean"] = float(active_ray_floor_stats.mean().item())
                metrics["sigma_active_ray_cap_mean"] = float(active_ray_cap_stats.mean().item())
                metrics["sigma_active_ray_parallel_mean"] = float(active_ray_parallel_stats.mean().item())
                metrics["sigma_active_ray_parallel_p90"] = float(torch.quantile(active_ray_parallel_stats, 0.90).item())
            return metrics

    def update_atlas_states(
        self,
        surface_stable_min: float,
        edge_stable_min: float,
        min_visibility_ema: float,
        stable_residual_threshold: float,
        activate_threshold: float | None = None,
        deactivate_threshold: float | None = None,
        activate_min_high_residual_iters: int = 1,
        recover_low_residual_iters: int = 3,
        drift_activate_iters: int = 2,
        out_of_anchor_drift_iters: int = 3,
        out_of_anchor_gc_failures: int = 2,
        state_cooldown_iters: int = 5,
        active_min_lifetime_iters: int = 5,
        active_quota_ratio: float = 0.03,
        active_quota_min: int = 1,
        active_quota_max: int = 128,
        min_active_opacity: float = 0.02,
        promote_to_active_threshold: float | None = None,
        demote_to_passive_threshold: float | None = None,
        active_max_lifetime_iters: int = 600,
        active_nonimprove_iters: int = 180,
        passive_to_stable_reliability_min: float | None = None,
        passive_to_stable_support_consistency_min: float | None = None,
        passive_to_stable_drift_max: float | None = None,
        passive_to_stable_photo_ema_max: float | None = None,
        state_rebuild_after_gc: bool = False,
    ):
        if not self.has_atlas_bindings:
            return {}

        with torch.no_grad():
            promote_to_active_threshold = float(
                promote_to_active_threshold
                if promote_to_active_threshold is not None
                else (activate_threshold if activate_threshold is not None else 0.0)
            )
            demote_to_passive_threshold = float(
                demote_to_passive_threshold
                if demote_to_passive_threshold is not None
                else (
                    deactivate_threshold
                    if deactivate_threshold is not None
                    else max(promote_to_active_threshold * 0.5, 0.0)
                )
            )
            active_min_lifetime_iters = int(max(active_min_lifetime_iters, 0))
            state_cooldown_iters = int(max(state_cooldown_iters, 0))
            active_quota_ratio = float(max(active_quota_ratio, 0.0))
            active_quota_min = int(max(active_quota_min, 0))
            active_quota_max = int(max(active_quota_max, 0))
            active_max_lifetime_iters = int(max(active_max_lifetime_iters, active_min_lifetime_iters + 1))
            active_nonimprove_iters = int(max(active_nonimprove_iters, active_min_lifetime_iters + 1))
            passive_to_stable_reliability_min = 0.18 if passive_to_stable_reliability_min is None else float(passive_to_stable_reliability_min)
            passive_to_stable_support_consistency_min = (
                0.28
                if passive_to_stable_support_consistency_min is None
                else float(passive_to_stable_support_consistency_min)
            )
            passive_to_stable_drift_max = 1.15 if passive_to_stable_drift_max is None else float(passive_to_stable_drift_max)
            passive_to_stable_photo_ema_max = (
                float(stable_residual_threshold) * 1.50
                if passive_to_stable_photo_ema_max is None
                else float(passive_to_stable_photo_ema_max)
            )
            reliability = self.get_gaussian_atlas_reliability.detach()
            atlas_class = self.get_gaussian_atlas_class.detach()
            opacity = self.get_opacity.detach().squeeze(-1)
            photo_ema = self._atlas_photo_ema
            visibility_ema = self._atlas_visibility_ema
            high_residual_count = self._atlas_high_residual_count
            low_residual_count = self._atlas_low_residual_count
            drift_flag = self._atlas_drift_flag
            drift_count = self._atlas_drift_count
            gc_fail_count = self._atlas_gc_fail_count
            state_cooldown = self._atlas_state_cooldown
            active_lifetime = self._atlas_active_lifetime
            active_provenance = self._atlas_active_provenance
            promotion_streak = self._atlas_promotion_streak
            demotion_streak = self._atlas_demotion_streak
            recovery_streak = self._atlas_recovery_streak
            last_transition_iter = self._atlas_last_transition_iter
            current_state = self._atlas_state
            self._atlas_state_update_iter = int(max(getattr(self, "_atlas_state_update_iter", 0), 0)) + 1
            state_update_iter = int(self._atlas_state_update_iter)

            if not self._atlas_refresh_done:
                next_state = current_state.clone()
                persistent_out = (
                    (drift_count >= int(max(out_of_anchor_drift_iters, 1)))
                    & (gc_fail_count >= int(max(out_of_anchor_gc_failures, 1)))
                )
                next_state[persistent_out] = GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
                transition_any_to_pending = (current_state != GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING) & (
                    next_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
                )
                if torch.any(transition_any_to_pending):
                    promotion_streak[transition_any_to_pending] = 0
                    demotion_streak[transition_any_to_pending] = 0
                    recovery_streak[transition_any_to_pending] = 0
                    last_transition_iter[transition_any_to_pending] = state_update_iter
                next_active_lifetime = torch.zeros_like(active_lifetime)
                next_active_provenance = torch.zeros_like(active_provenance)
                stay_active = (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & (next_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                new_active = (current_state != GAUSSIAN_STATE_UNSTABLE_ACTIVE) & (next_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                next_active_lifetime[stay_active] = active_lifetime[stay_active] + 1
                next_active_lifetime[new_active] = 1
                if torch.any(stay_active):
                    carried_provenance = active_provenance[stay_active].clamp_min(ACTIVE_PROVENANCE_NONE)
                    missing_provenance = carried_provenance <= ACTIVE_PROVENANCE_NONE
                    fallback_provenance = (
                        ACTIVE_PROVENANCE_FROM_STATE_REBUILD_AFTER_GC
                        if bool(state_rebuild_after_gc)
                        else ACTIVE_PROVENANCE_FROM_QUOTA_CARRYOVER
                    )
                    carried_provenance = torch.where(
                        missing_provenance,
                        torch.full_like(carried_provenance, fallback_provenance),
                        carried_provenance,
                    )
                    next_active_provenance[stay_active] = carried_provenance
                next_active_provenance[new_active] = ACTIVE_PROVENANCE_FROM_TRANSITION_PASSIVE_TO_ACTIVE
                self._atlas_active_lifetime = next_active_lifetime
                self._atlas_active_provenance = next_active_provenance
                self._atlas_state = next_state
                active_mask = self._atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
                metrics = {
                    "stable_ratio": float((self._atlas_state == GAUSSIAN_STATE_STABLE).float().mean().item()),
                    "passive_ratio": float((self._atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).float().mean().item()),
                    "active_ratio": float(active_mask.float().mean().item()),
                    "out_of_anchor_ratio": float((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).float().mean().item()),
                    "pending_ratio": float((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).float().mean().item()),
                    "state_stable_count": int((self._atlas_state == GAUSSIAN_STATE_STABLE).sum().item()),
                    "state_passive_count": int((self._atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).sum().item()),
                    "state_active_count": int(active_mask.sum().item()),
                    "state_out_pending_count": int((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item()),
                    "out_of_anchor_pending_count": int((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item()),
                    "state_cooldown_ratio": float((state_cooldown > 0).float().mean().item()) if state_cooldown.numel() > 0 else 0.0,
                    "mean_active_lifetime": float(self._atlas_active_lifetime[active_mask].float().mean().item()) if torch.any(active_mask) else 0.0,
                    "max_active_lifetime": float(self._atlas_active_lifetime[active_mask].max().item()) if torch.any(active_mask) else 0.0,
                    "transition_stable_to_passive_count": 0,
                    "transition_passive_to_stable_count": 0,
                    "transition_passive_to_active_count": 0,
                    "transition_active_to_passive_count": 0,
                    "transition_active_to_stable_count": 0,
                    "transition_any_to_pending_count": int(transition_any_to_pending.sum().item()),
                    "promote_to_active_threshold": promote_to_active_threshold,
                    "demote_to_passive_threshold": demote_to_passive_threshold,
                    "active_quota_target": 0,
                    "active_quota_soft_target": 0,
                    "active_quota_hard_target": int(active_quota_max),
                    "active_quota_effective_live_active_count": int(active_mask.sum().item()),
                    "active_quota_over_target_count": int(active_mask.sum().item()),
                    "active_quota_available": 0,
                    "active_candidate_pool_count": 0,
                    "active_candidate_pool_ratio": 0.0,
                    "passive_to_stable_candidate_count": 0,
                    "active_to_stable_candidate_count": 0,
                    "active_formed_count": 0,
                    "active_admitted_count": 0,
                    "active_promoted_count": 0,
                    "active_demoted_count": 0,
                    "active_exited_count": int(transition_any_to_pending.sum().item()),
                    "passive_stable_ready_count": 0,
                    "active_stable_ready_count": 0,
                    "active_lifetime_lock_ratio": 0.0,
                    "hard_active_exit_ratio": 0.0,
                    "soft_active_exit_ratio": 0.0,
                    "state_candidate_reason_breakdown": {},
                    "promotion_block_reason_breakdown": {},
                    "transition_passive_to_stable_block_reason_breakdown": {},
                    "transition_active_to_stable_block_reason_breakdown": {},
                }
                metrics.update(self._summarize_active_provenance_metrics(active_mask=active_mask))
                return metrics

            required_reliability = torch.full_like(reliability, float(surface_stable_min))
            required_reliability[atlas_class == ATLAS_CLASS_EDGE] = float(edge_stable_min)
            runtime_mapped_reliability = self.get_gaussian_atlas_reliability_runtime_mapped.detach().clamp(0.0, 1.0)
            runtime_raw_reliability = self.get_gaussian_atlas_reliability_runtime_raw.detach().clamp(0.0, 1.0)
            support_consistency = self._compute_support_consistency_score()
            support_inconsistency = (1.0 - support_consistency).clamp(0.0, 1.0)
            geometric_stable_class = (atlas_class == ATLAS_CLASS_SURFACE) | (atlas_class == ATLAS_CLASS_EDGE)
            reliability_stable_support = reliability >= required_reliability
            runtime_stable_support = (
                geometric_stable_class
                & (runtime_mapped_reliability >= (required_reliability * 1.20).clamp(max=0.65))
                & (runtime_raw_reliability >= 0.42)
                & (support_consistency >= 0.32)
            )
            runtime_recovery_support = (
                geometric_stable_class
                & (
                    (runtime_mapped_reliability >= (required_reliability * 0.90).clamp(max=0.60))
                    | (reliability >= (required_reliability * 0.82).clamp(max=0.58))
                )
                & (runtime_raw_reliability >= 0.30)
                & (support_consistency >= 0.26)
            )
            stable_surface = (
                (atlas_class == ATLAS_CLASS_SURFACE)
                & (reliability_stable_support | runtime_stable_support | runtime_recovery_support)
            )
            stable_edge = (
                (atlas_class == ATLAS_CLASS_EDGE)
                & (reliability_stable_support | runtime_stable_support | runtime_recovery_support)
            )
            stable_support = stable_surface | stable_edge
            min_active_opacity_value = float(max(min_active_opacity, 0.0))
            opacity_soft_gate = (
                (opacity - min_active_opacity_value)
                / max(0.10 - min_active_opacity_value, 1e-4)
            ).clamp(0.0, 1.0)
            visibility_soft_gate = (visibility_ema / max(float(min_visibility_ema), 1e-4)).clamp(0.0, 1.0)
            formation_visibility_soft_gate = (visibility_ema / max(float(min_visibility_ema) * 0.35, 1e-4)).clamp(0.0, 1.0)
            photo_score = (photo_ema / max(float(stable_residual_threshold), 1e-4)).clamp(0.0, 4.0)
            required_gap = (
                (required_reliability - reliability)
                / required_reliability.clamp_min(1e-4)
            ).clamp(0.0, 1.0)
            unreliability = torch.where(
                atlas_class == ATLAS_CLASS_UNSTABLE,
                (1.0 - reliability).clamp(0.0, 1.0),
                required_gap,
            )
            streak_gate = (
                high_residual_count.float() / float(max(int(activate_min_high_residual_iters), 1))
            ).clamp(0.0, 1.0)
            visibility_ready = visibility_ema >= float(min_visibility_ema)
            sustained_high_residual = high_residual_count >= int(max(activate_min_high_residual_iters, 1))
            sustained_low_residual = low_residual_count >= int(max(recover_low_residual_iters, 1))
            sustained_drift = drift_count >= int(max(drift_activate_iters, 1))
            persistent_out = (
                (drift_count >= int(max(out_of_anchor_drift_iters, 1)))
                & (gc_fail_count >= int(max(out_of_anchor_gc_failures, 1)))
            )
            cooldown_active = state_cooldown > 0
            opacity_active = opacity >= min_active_opacity_value
            opacity_alive = opacity >= max(min_active_opacity_value * 0.20, 1e-4)
            visibility_rescue_ready = (
                (visibility_ema >= float(min_visibility_ema) * 0.05)
                | (self._atlas_ref_score >= 0.05)
                | (self._atlas_ref_camera >= 0)
            )
            formation_visibility_ready = (
                (visibility_ema >= float(min_visibility_ema) * 0.25)
                | ((visibility_ema >= float(min_visibility_ema) * 0.05) & (self._atlas_ref_score >= 0.05))
                | ((self._atlas_ref_camera >= 0) & (high_residual_count > 0))
            )
            formation_opacity_ready = opacity >= max(min_active_opacity_value * 0.25, 1e-4)
            ref_camera_valid = (self._atlas_ref_camera >= 0) | (self._atlas_ref_score >= 0.05)
            ref_camera_ready = (
                ref_camera_valid
                | visibility_ready
            )
            ref_readiness_score = torch.maximum(
                self._atlas_ref_score.detach().clamp(0.0, 1.0),
                ref_camera_ready.to(dtype=torch.float32) * 0.35,
            )
            active_evidence_core = (
                (photo_score / 4.0).clamp(0.0, 1.0)
                * (0.55 + 0.45 * formation_visibility_soft_gate)
                * (0.55 + 0.45 * opacity_soft_gate)
                * (0.30 + 0.70 * unreliability.clamp(0.0, 1.0))
            )
            persistence_score = torch.maximum(
                streak_gate,
                (high_residual_count.float() / float(max(int(activate_min_high_residual_iters), 1) * 2)).clamp(0.0, 1.0),
            )
            non_drift_safety = (~drift_flag & ~persistent_out).to(dtype=torch.float32)
            support_residual_score = (
                support_inconsistency
                * (photo_score / 4.0).clamp(0.0, 1.0)
                * (0.35 + 0.65 * persistence_score)
            ).clamp(0.0, 1.0)
            promotion_score = (
                0.48 * active_evidence_core
                + 0.18 * persistence_score
                + 0.16 * support_residual_score
                + 0.10 * ref_readiness_score
                + 0.05 * formation_visibility_soft_gate
                + 0.03 * opacity_soft_gate
            ) * (0.60 + 0.40 * non_drift_safety)
            promotion_score = promotion_score.clamp_min(0.0)
            explore_score = promotion_score
            passive_pool = current_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE
            active_pool = current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
            reattach_recently_failed = gc_fail_count > 0
            gc_heavy_block = gc_fail_count >= int(max(out_of_anchor_gc_failures, 1))
            residual_half_iters = max(1, int(max(activate_min_high_residual_iters, 1)) // 2)
            prolonged_residual_iters = max(2, int(max(activate_min_high_residual_iters, 1)) * 2)
            residual_candidate_signal = (
                sustained_high_residual
                | (
                    (high_residual_count >= residual_half_iters)
                    & (photo_ema >= float(stable_residual_threshold) * 0.75)
                )
            )
            prolonged_residual_signal = (
                (high_residual_count >= prolonged_residual_iters)
                | (
                    sustained_high_residual
                    & (photo_ema >= float(stable_residual_threshold) * 1.20)
                )
            )
            support_inconsistent_residual = (
                (support_inconsistency >= 0.35)
                & residual_candidate_signal
            )
            low_reliability_support_ok = (
                (unreliability > 0.01)
                & (support_inconsistency < 0.35)
                & (~residual_candidate_signal)
            )
            slab_ready_proxy = (
                ref_camera_valid
                & torch.isfinite(self.get_gaussian_atlas_radius.detach())
                & torch.isfinite(self.get_xyz.detach()).all(dim=1)
            )
            rescue_promotion_signal = (
                passive_pool
                & prolonged_residual_signal
                & (support_inconsistency >= 0.25)
                & ref_camera_valid
                & slab_ready_proxy
                & opacity_alive
                & (~persistent_out)
                & (~gc_heavy_block)
            )
            # Pure photometric path: high residual + low reliability, no support_inconsistency
            # requirement. Needed when Gaussians stay near their atlas anchors (support_consistency
            # high) but photometric error is still large — atlas-aware init case.
            photo_residual_unreliable = (
                sustained_high_residual
                & (unreliability > 0.08)
                & ref_camera_valid
                & (photo_ema >= float(stable_residual_threshold) * 1.5)
            )
            standard_formation_evidence = (
                (
                    support_inconsistent_residual
                    | (
                        residual_candidate_signal
                        & (unreliability > 0.08)
                        & (support_inconsistency >= 0.20)
                    )
                    | photo_residual_unreliable
                )
                & (~sustained_drift)
            )
            rescue_fallback_evidence = (
                rescue_promotion_signal
                & (~standard_formation_evidence)
                & (~sustained_drift)
            )
            formation_evidence = standard_formation_evidence | rescue_fallback_evidence
            reliability_recovery_ready = (
                stable_support
                & (runtime_mapped_reliability >= (required_reliability * 1.35).clamp(max=0.72))
                & (support_consistency >= 0.38)
                & (photo_ema <= float(stable_residual_threshold) * 1.15)
                & (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
            )
            state_reliability_stable_ready = (
                stable_support
                & (reliability >= (required_reliability * 0.92).clamp(max=0.70))
                & (runtime_mapped_reliability >= (required_reliability * 1.10).clamp(max=0.66))
                & (support_consistency >= 0.38)
                & (photo_ema <= float(stable_residual_threshold) * 1.10)
                & (
                    (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
                    | (runtime_raw_reliability >= 0.45)
                )
                & (~drift_flag)
                & (~persistent_out)
            )
            effective_recovery_stable_ready = (
                stable_support
                & (reliability >= (required_reliability * 0.85).clamp(max=0.62))
                & (runtime_mapped_reliability >= (required_reliability * 0.95).clamp(max=0.62))
                & (runtime_raw_reliability >= 0.32)
                & (support_consistency >= 0.28)
                & (photo_ema <= float(stable_residual_threshold) * 1.25)
                & (
                    (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
                    | (runtime_raw_reliability >= 0.42)
                )
                & (visibility_ready | ref_camera_valid | (visibility_ema >= float(min_visibility_ema) * 0.08))
                & opacity_alive
                & (~drift_flag)
                & (~persistent_out)
            )
            passive_runtime_recovery_ready = (
                stable_support
                & runtime_recovery_support
                & (photo_ema <= float(stable_residual_threshold) * 1.35)
                & (
                    (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
                    | (runtime_raw_reliability >= 0.38)
                )
                & (visibility_ready | ref_camera_valid | (visibility_ema >= float(min_visibility_ema) * 0.06))
                & opacity_alive
                & (~drift_flag)
                & (~persistent_out)
            )
            node_ref_consistency = self._atlas_node_metric_for_gaussians(
                self._atlas_node_ref_consistency_ema,
                default_value=0.0,
            ).clamp(0.0, 1.0)
            recovery_ref_consistency = torch.maximum(node_ref_consistency, self._atlas_ref_score.detach().clamp(0.0, 1.0))
            anchor_drift_ratio = torch.zeros_like(reliability, dtype=torch.float32)
            if self.has_atlas_bindings and self.get_xyz.numel() > 0:
                atlas_positions_for_state = self.get_gaussian_atlas_positions.detach()
                atlas_radius_for_state = self.get_gaussian_atlas_radius.detach().clamp_min(1e-6)
                anchor_drift_ratio = (
                    torch.linalg.norm(self.get_xyz.detach() - atlas_positions_for_state, dim=1)
                    / atlas_radius_for_state
                ).clamp_min(0.0)
            passive_explicit_stable_ready = (
                passive_pool
                & stable_support
                & (reliability >= float(passive_to_stable_reliability_min))
                & (support_consistency >= float(passive_to_stable_support_consistency_min))
                & (recovery_ref_consistency >= 0.20)
                & (anchor_drift_ratio <= float(passive_to_stable_drift_max))
                & (photo_ema <= float(passive_to_stable_photo_ema_max))
                & (
                    (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
                    | (runtime_raw_reliability >= 0.34)
                    | (visibility_ema >= float(min_visibility_ema) * 0.12)
                )
                & (visibility_ready | ref_camera_valid | (visibility_ema >= float(min_visibility_ema) * 0.06))
                & opacity_alive
                & (~drift_flag)
                & (~persistent_out)
            )
            recovery_effective_reliability = torch.maximum(reliability, runtime_mapped_reliability).clamp(0.0, 1.0)
            recovery_visibility_score = torch.maximum(
                visibility_soft_gate,
                torch.maximum(
                    ref_camera_valid.to(dtype=torch.float32) * 0.45,
                    (visibility_ema / max(float(min_visibility_ema) * 0.25, 1e-4)).clamp(0.0, 1.0) * 0.60,
                ),
            ).clamp(0.0, 1.0)
            recovery_pool_for_photo = (passive_pool | active_pool | (current_state == GAUSSIAN_STATE_STABLE)) & torch.isfinite(photo_ema)
            if torch.any(recovery_pool_for_photo):
                pooled_photo = photo_ema[recovery_pool_for_photo].detach().clamp_min(0.0)
                photo_median = float(torch.quantile(pooled_photo, 0.50).item())
                photo_q75 = float(torch.quantile(pooled_photo, 0.75).item())
            else:
                photo_median = float(stable_residual_threshold)
                photo_q75 = float(stable_residual_threshold)
            recovery_photo_reference = float(
                max(
                    float(stable_residual_threshold) * 4.0,
                    photo_median,
                    float(passive_to_stable_photo_ema_max) * 2.5,
                    1e-4,
                )
            )
            recovery_photo_soft_max = float(max(recovery_photo_reference * 1.35, photo_q75, recovery_photo_reference))
            recovery_photo_hard_max = float(max(recovery_photo_reference * 2.25, photo_q75 * 1.35, recovery_photo_soft_max))
            recovery_photo_norm = (photo_ema / recovery_photo_reference).clamp(0.0, 2.5)
            recovery_photo_quality = torch.exp(-recovery_photo_norm).clamp(0.0, 1.0)
            recovery_drift_norm = (
                anchor_drift_ratio / max(float(passive_to_stable_drift_max), 1e-4)
            ).clamp(0.0, 1.75)
            recovery_score = (
                0.40 * recovery_effective_reliability
                + 0.24 * support_consistency.clamp(0.0, 1.0)
                + 0.16 * recovery_ref_consistency.clamp(0.0, 1.0)
                + 0.10 * recovery_visibility_score
                + 0.07 * recovery_photo_quality
                - 0.08 * recovery_photo_norm
                - 0.07 * recovery_drift_norm
            ).clamp_min(0.0)
            recovery_score_threshold = float(
                max(0.12, min(0.20, float(passive_to_stable_reliability_min) * 0.75))
            )
            recovery_relaxed_reliability_min = float(max(0.14, min(0.16, float(passive_to_stable_reliability_min) * 0.85)))
            recovery_relaxed_support_min = float(max(0.18, min(0.22, float(passive_to_stable_support_consistency_min) * 0.80)))
            recovery_relaxed_ref_min = 0.12
            recovery_drift_max = float(max(float(passive_to_stable_drift_max) * 1.20, 1.15))
            recovery_node_evidence = (
                (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
                | (runtime_raw_reliability >= 0.32)
                | (visibility_ema >= float(min_visibility_ema) * 0.08)
                | (recovery_ref_consistency >= 0.16)
                | runtime_recovery_support
            )
            recovery_photo_soft_ok = (
                (photo_ema <= recovery_photo_soft_max)
                | (recovery_photo_quality >= 0.20)
                | runtime_recovery_support
                | (low_residual_count >= max(1, int(max(recover_low_residual_iters, 1)) // 2))
            )
            recovery_ready_soft = (
                stable_support
                & (recovery_effective_reliability >= recovery_relaxed_reliability_min)
                & (support_consistency >= recovery_relaxed_support_min)
                & ((recovery_ref_consistency >= recovery_relaxed_ref_min) | ref_camera_valid | runtime_recovery_support)
                & (anchor_drift_ratio <= recovery_drift_max)
                & recovery_photo_soft_ok
                & (visibility_ready | ref_camera_valid | (recovery_visibility_score >= 0.30) | runtime_recovery_support)
                & recovery_node_evidence
                & opacity_alive
                & (~drift_flag)
                & (~persistent_out)
            )
            legacy_recovery_candidate = (
                reliability_recovery_ready
                | state_reliability_stable_ready
                | effective_recovery_stable_ready
                | passive_runtime_recovery_ready
                | passive_explicit_stable_ready
                | recovery_ready_soft
            )
            recovery_required_streak = int(max(3, min(5, int(max(recover_low_residual_iters, 3)))))
            recovery_score_ready = recovery_score >= recovery_score_threshold
            recovery_high_score_override = (
                (recovery_score >= recovery_score_threshold + 0.10)
                & recovery_node_evidence
                & (photo_ema <= recovery_photo_hard_max)
            )
            recovery_streak_signal = recovery_ready_soft & (recovery_score_ready | legacy_recovery_candidate)
            recovery_candidate_pool = (passive_pool | active_pool) & recovery_streak_signal
            if torch.any(recovery_candidate_pool):
                recovery_streak[recovery_candidate_pool] = recovery_streak[recovery_candidate_pool] + 1
            recovery_streak[(~(passive_pool | active_pool)) | (~recovery_streak_signal)] = 0
            recovery_promote_hard = recovery_ready_soft & (
                (recovery_streak >= recovery_required_streak)
                | recovery_high_score_override
                | legacy_recovery_candidate
            )
            recovery_score_candidate = recovery_ready_soft
            recovery_candidate = recovery_streak_signal
            passive_recovery_candidate = passive_pool & recovery_ready_soft
            active_recovery_candidate = active_pool & recovery_ready_soft
            passive_recovery_streak_ready = passive_pool & recovery_promote_hard
            active_recovery_streak_ready = active_pool & recovery_promote_hard
            stable_candidate = (
                stable_support
                & (
                    visibility_ready
                    | ((self._atlas_ref_score >= 0.08) & (visibility_ema >= float(min_visibility_ema) * 0.10))
                    | effective_recovery_stable_ready
                    | passive_runtime_recovery_ready
                    | passive_explicit_stable_ready
                    | recovery_score_candidate
                )
                & (
                    (
                        (photo_ema <= float(stable_residual_threshold))
                        & sustained_low_residual
                    )
                    | reliability_recovery_ready
                    | state_reliability_stable_ready
                    | effective_recovery_stable_ready
                    | passive_runtime_recovery_ready
                    | passive_explicit_stable_ready
                    | recovery_score_candidate
                )
                & (~drift_flag)
            )

            passive_promote = (
                (current_state == GAUSSIAN_STATE_STABLE)
                & (~persistent_out)
                & ((sustained_high_residual & (photo_ema >= float(stable_residual_threshold))) | sustained_drift)
            )
            candidate_formation_signal = (
                passive_pool
                & (~persistent_out)
                & (~gc_heavy_block)
                & (~stable_candidate)
                & (~passive_recovery_candidate)
                & formation_evidence
                & (promotion_score >= max(promote_to_active_threshold * 0.45, demote_to_passive_threshold))
            )
            standard_candidate_formation_signal = candidate_formation_signal & standard_formation_evidence
            rescue_candidate_formation_signal = candidate_formation_signal & rescue_fallback_evidence
            if torch.any(candidate_formation_signal):
                promotion_streak[candidate_formation_signal] = promotion_streak[candidate_formation_signal] + 1
            reset_promotion_streak = (
                (current_state != GAUSSIAN_STATE_UNSTABLE_PASSIVE)
                | (~candidate_formation_signal)
            )
            promotion_streak[reset_promotion_streak] = 0
            promotion_required_streak = int(max(1, min(2, int(max(activate_min_high_residual_iters, 1)))))
            promotion_ema_override = (
                standard_candidate_formation_signal
                & (promotion_score >= promote_to_active_threshold * 1.05)
                & (photo_ema >= float(stable_residual_threshold) * 1.25)
            )
            standard_streak_ready = standard_candidate_formation_signal & (
                (promotion_streak >= promotion_required_streak)
                | promotion_ema_override
            )
            rescue_ready_override = rescue_candidate_formation_signal & rescue_promotion_signal
            rescue_streak_ready = rescue_ready_override & (
                (promotion_streak >= promotion_required_streak)
                | (
                    (promotion_score >= max(promote_to_active_threshold * 0.80, demote_to_passive_threshold))
                    & prolonged_residual_signal
                )
            )
            active_candidate_pool = standard_streak_ready | rescue_streak_ready
            active_lifetime_lock = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & ((active_lifetime < int(max(active_min_lifetime_iters, 0))) | cooldown_active)
            )
            fallback_origin_active = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (
                    (active_provenance == ACTIVE_PROVENANCE_FROM_STATE_REBUILD_AFTER_GC)
                    | (active_provenance == ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP)
                    | (active_provenance == ACTIVE_PROVENANCE_FROM_RESTORE_CHECKPOINT)
                    | (active_provenance == ACTIVE_PROVENANCE_FROM_ACTIVE_EXPLORE_CLONE)
                )
            )
            hard_active_exit = (
                (~opacity_alive)
                | persistent_out
                | gc_heavy_block
            )
            active_lifetime_cap_exit = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (active_lifetime >= active_max_lifetime_iters)
            )
            active_nonimproving_exit = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (active_lifetime >= active_nonimprove_iters)
                & (high_residual_count >= int(max(activate_min_high_residual_iters, 1)))
                & (photo_ema >= float(stable_residual_threshold) * 0.90)
                & (low_residual_count <= 0)
            )
            active_stale_fallback_exit = (
                fallback_origin_active
                & (active_lifetime >= active_nonimprove_iters)
                & (~stable_candidate)
                & (
                    (promotion_score < max(promote_to_active_threshold * 0.80, demote_to_passive_threshold))
                    | (support_consistency < 0.30)
                    | (
                        (visibility_ema < float(min_visibility_ema) * 0.10)
                        & (~ref_camera_valid)
                    )
                )
            )
            fallback_transition_handoff_exit = torch.zeros_like(hard_active_exit, dtype=torch.bool)
            if bool(torch.any(standard_streak_ready).item()) and active_quota_max > 0:
                active_over_hard_quota = int((current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE).sum().item()) >= active_quota_max
                if active_over_hard_quota:
                    fallback_transition_handoff_exit = (
                        fallback_origin_active
                        & (active_lifetime >= max(active_min_lifetime_iters * 2, active_min_lifetime_iters + 1))
                        & (~stable_candidate)
                        & (
                            (promotion_score < max(promote_to_active_threshold, demote_to_passive_threshold))
                            | (support_consistency < 0.34)
                            | (
                                (photo_ema <= float(stable_residual_threshold) * 0.90)
                                & (low_residual_count > 0)
                            )
                        )
                    )
            hard_active_exit = (
                hard_active_exit
                | active_lifetime_cap_exit
                | active_nonimproving_exit
                | active_stale_fallback_exit
                | fallback_transition_handoff_exit
            )
            soft_active_exit_signal = (
                (promotion_score < demote_to_passive_threshold)
                | (
                    (visibility_ema < float(min_visibility_ema) * 0.10)
                    & (~ref_camera_valid)
                )
                | (
                    (active_lifetime >= max(active_min_lifetime_iters * 2, active_nonimprove_iters // 2))
                    & (photo_ema >= float(stable_residual_threshold) * 0.75)
                    & (high_residual_count >= max(1, int(max(activate_min_high_residual_iters, 1)) // 2))
                    & (low_residual_count <= 0)
                )
            )
            active_soft_exit_signal = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & soft_active_exit_signal
            )
            if torch.any(active_soft_exit_signal):
                demotion_streak[active_soft_exit_signal] = demotion_streak[active_soft_exit_signal] + 1
            demotion_streak[(current_state != GAUSSIAN_STATE_UNSTABLE_ACTIVE) | (~soft_active_exit_signal)] = 0
            demotion_required_streak = int(max(2, min(3, int(max(recover_low_residual_iters, 2)))))
            soft_active_exit = soft_active_exit_signal & (demotion_streak >= demotion_required_streak)
            active_demote = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (
                    hard_active_exit
                    | ((~active_lifetime_lock) & soft_active_exit & (~stable_candidate))
                )
            )
            active_recover = (
                active_recovery_streak_ready
                & (~hard_active_exit)
                & opacity_alive
                & (~persistent_out)
            )
            current_active_count = int((current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE).sum().item())
            passive_count = int((current_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).sum().item())
            all_projected_exits_mask = (active_demote | active_recover | (persistent_out & (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)))
            projected_active_exits = int(all_projected_exits_mask.sum().item())
            projected_active_after_exit = max(current_active_count - projected_active_exits, 0)
            # Fallback-origin active (GC rebuild / bootstrap / checkpoint / explore clone)
            # must not block passive->active promotions. Count only real transitions toward quota.
            real_active_count = max(current_active_count - int(fallback_origin_active.sum().item()), 0)
            real_projected_exits = int((all_projected_exits_mask & (~fallback_origin_active)).sum().item())
            projected_real_active_after_exit = max(real_active_count - real_projected_exits, 0)
            quota_from_ratio = int(np.ceil(active_quota_ratio * passive_count)) if passive_count > 0 else 0
            active_quota_target = max(active_quota_min, quota_from_ratio) if passive_count > 0 else 0
            if active_quota_max > 0:
                active_quota_target = min(active_quota_target, active_quota_max)
            active_quota_available = max(active_quota_target - projected_real_active_after_exit, 0)
            standard_ready_count = int(standard_streak_ready.sum().item())
            transition_overflow_quota = 0
            if active_quota_target > 0 and standard_ready_count > active_quota_available:
                transition_overflow_quota = min(
                    max(active_quota_min, int(np.ceil(active_quota_target * 0.20))),
                    32,
                    standard_ready_count - active_quota_available,
                )
                active_quota_available += transition_overflow_quota
            admission_score_ready = promotion_score >= promote_to_active_threshold
            rescue_score_ready = rescue_promotion_signal & (
                promotion_score >= max(promote_to_active_threshold * 0.65, demote_to_passive_threshold)
            )
            cooldown_block = cooldown_active & (~rescue_ready_override)
            admission_visibility_ready = visibility_rescue_ready | rescue_ready_override
            admission_safety_gate = (
                admission_visibility_ready
                & opacity_alive
                & ref_camera_valid
                & slab_ready_proxy
                & (~persistent_out)
                & (~cooldown_block)
                & (~gc_heavy_block)
            )
            standard_active_admission_pool = standard_streak_ready & admission_safety_gate & admission_score_ready
            standard_active_promote = self._select_topk_mask(
                standard_active_admission_pool,
                promotion_score,
                active_quota_available,
            )
            remaining_active_quota = max(active_quota_available - int(standard_active_promote.sum().item()), 0)
            rescue_active_admission_pool = rescue_streak_ready & admission_safety_gate & rescue_score_ready
            rescue_active_promote = self._select_topk_mask(
                rescue_active_admission_pool & (~standard_active_promote),
                promotion_score,
                remaining_active_quota,
            )
            active_admission_pool = standard_active_admission_pool | rescue_active_admission_pool
            active_promote = standard_active_promote | rescue_active_promote
            promotion_base_pool = passive_pool & formation_evidence
            below_score_threshold = promotion_base_pool & (promotion_score < max(promote_to_active_threshold * 0.45, demote_to_passive_threshold))
            promotion_block_reason_breakdown = {
                "formation_low_visibility_soft": int((promotion_base_pool & (~formation_visibility_ready)).sum().item()),
                "formation_low_opacity_soft": int((promotion_base_pool & (~formation_opacity_ready)).sum().item()),
                "admission_low_visibility_hard": int((active_candidate_pool & (~admission_visibility_ready)).sum().item()),
                "admission_dead_opacity_hard": int((active_candidate_pool & (~opacity_alive)).sum().item()),
                "missing_ref": int((active_candidate_pool & (~ref_camera_valid)).sum().item()),
                "missing_slab_proxy": int((active_candidate_pool & ref_camera_valid & (~slab_ready_proxy)).sum().item()),
                "persistent_out": int((active_candidate_pool & persistent_out).sum().item()),
                "cooldown": int((active_candidate_pool & cooldown_block).sum().item()),
                "gc_heavy": int((active_candidate_pool & gc_heavy_block).sum().item()),
                "score_below_threshold": int(below_score_threshold.sum().item()),
                "quota_full": int((active_admission_pool & (~active_promote)).sum().item()) if active_quota_available <= int(active_admission_pool.sum().item()) else 0,
            }
            passive_stable_ready = (
                passive_recovery_streak_ready
                & opacity_alive
                & (~persistent_out)
            )
            passive_stable_base = passive_pool & stable_support
            passive_stable_recovery_gate = (
                reliability_recovery_ready
                | state_reliability_stable_ready
                | effective_recovery_stable_ready
                | passive_runtime_recovery_ready
                | passive_explicit_stable_ready
                | recovery_ready_soft
            )
            passive_stable_visibility_gate = (
                visibility_ready
                | ref_camera_valid
                | (visibility_ema >= float(min_visibility_ema) * 0.06)
                | effective_recovery_stable_ready
                | passive_runtime_recovery_ready
                | passive_explicit_stable_ready
            )
            passive_stable_residual_gate = (
                ((photo_ema <= float(stable_residual_threshold)) & sustained_low_residual)
                | passive_stable_recovery_gate
            )
            passive_stable_cooldown_gate = (~cooldown_active) | passive_stable_recovery_gate
            passive_to_stable_block_reason_breakdown = {
                "recovery_block_low_reliability": int((passive_pool & stable_support & (recovery_effective_reliability < recovery_relaxed_reliability_min)).sum().item()),
                "recovery_block_low_support_consistency": int((passive_pool & stable_support & (support_consistency < recovery_relaxed_support_min)).sum().item()),
                "recovery_block_low_ref_consistency": int((passive_pool & stable_support & (recovery_ref_consistency < recovery_relaxed_ref_min) & (~ref_camera_valid) & (~runtime_recovery_support)).sum().item()),
                "recovery_block_high_photo_ema": int((passive_pool & stable_support & (photo_ema > recovery_photo_soft_max) & (recovery_photo_quality < 0.20) & (~runtime_recovery_support)).sum().item()),
                "recovery_block_high_drift": int((passive_pool & stable_support & (anchor_drift_ratio > recovery_drift_max)).sum().item()),
                "recovery_block_cooldown": int((passive_recovery_candidate & cooldown_active).sum().item()),
                "recovery_block_persistent_out": int((passive_pool & persistent_out).sum().item()),
                "low_effective_reliability": int((passive_pool & (reliability < float(passive_to_stable_reliability_min))).sum().item()),
                "low_support_consistency": int((passive_pool & (support_consistency < float(passive_to_stable_support_consistency_min))).sum().item()),
                "low_ref_consistency": int((passive_pool & (recovery_ref_consistency < 0.20)).sum().item()),
                "photo_ema_high": int((passive_pool & (photo_ema > float(passive_to_stable_photo_ema_max))).sum().item()),
                "projected_drift_high": int((passive_pool & (anchor_drift_ratio > float(passive_to_stable_drift_max))).sum().item()),
                "drift_flag": int((passive_pool & drift_flag).sum().item()),
                "persistent_out": int((passive_pool & persistent_out).sum().item()),
                "pending_or_gc_heavy": int((passive_pool & gc_heavy_block).sum().item()),
                "low_visibility": int((passive_stable_base & (~passive_stable_visibility_gate)).sum().item()),
                "residual_not_recovered": int((passive_stable_base & (~passive_stable_residual_gate)).sum().item()),
                "cooldown_ignored_for_recovery": int((passive_recovery_candidate & cooldown_active).sum().item()),
                "recovery_score_below_threshold": int((passive_pool & (recovery_score < recovery_score_threshold)).sum().item()),
                "recovery_streak_not_ready": int((passive_recovery_candidate & (recovery_streak < recovery_required_streak)).sum().item()),
                "missing_recovery_node_evidence": int((passive_pool & stable_support & (~recovery_node_evidence)).sum().item()),
                "not_recovery_candidate": int((passive_pool & (~recovery_candidate)).sum().item()),
                "not_stable_support": int((passive_pool & (~stable_support)).sum().item()),
            }
            active_stable_base = active_pool & stable_support
            active_to_stable_block_reason_breakdown = {
                "recovery_block_low_reliability": int((active_pool & stable_support & (recovery_effective_reliability < recovery_relaxed_reliability_min)).sum().item()),
                "recovery_block_low_support_consistency": int((active_pool & stable_support & (support_consistency < recovery_relaxed_support_min)).sum().item()),
                "recovery_block_low_ref_consistency": int((active_pool & stable_support & (recovery_ref_consistency < recovery_relaxed_ref_min) & (~ref_camera_valid) & (~runtime_recovery_support)).sum().item()),
                "recovery_block_high_photo_ema": int((active_pool & stable_support & (photo_ema > recovery_photo_soft_max) & (recovery_photo_quality < 0.20) & (~runtime_recovery_support)).sum().item()),
                "recovery_block_high_drift": int((active_pool & stable_support & (anchor_drift_ratio > recovery_drift_max)).sum().item()),
                "recovery_block_cooldown": int((active_recovery_candidate & cooldown_active).sum().item()),
                "recovery_block_persistent_out": int((active_pool & persistent_out).sum().item()),
                "lifetime_lock_ignored_for_recovery": int((active_recovery_candidate & active_lifetime_lock).sum().item()),
                "hard_exit": int((active_stable_base & hard_active_exit).sum().item()),
                "not_stable_candidate": int((active_pool & (~stable_candidate)).sum().item()),
                "low_effective_reliability": int((active_pool & (reliability < float(passive_to_stable_reliability_min))).sum().item()),
                "low_support_consistency": int((active_pool & (support_consistency < float(passive_to_stable_support_consistency_min))).sum().item()),
                "photo_ema_high": int((active_pool & (photo_ema > float(passive_to_stable_photo_ema_max))).sum().item()),
                "drift_flag": int((active_pool & drift_flag).sum().item()),
                "persistent_out": int((active_pool & persistent_out).sum().item()),
                "recovery_score_below_threshold": int((active_pool & (recovery_score < recovery_score_threshold)).sum().item()),
                "recovery_streak_not_ready": int((active_recovery_candidate & (recovery_streak < recovery_required_streak)).sum().item()),
                "missing_recovery_node_evidence": int((active_pool & stable_support & (~recovery_node_evidence)).sum().item()),
                "not_recovery_candidate": int((active_pool & (~recovery_candidate)).sum().item()),
            }
            state_candidate_reason_breakdown = {
                "high_residual": int((passive_pool & sustained_high_residual).sum().item()),
                "photo_residual": int((passive_pool & (photo_ema >= float(stable_residual_threshold) * 0.75)).sum().item()),
                "persistent_drift_pending": int((passive_pool & sustained_drift & (gc_fail_count > 0)).sum().item()),
                "drift_monitoring": int((passive_pool & sustained_drift & (gc_fail_count <= 0)).sum().item()),
                "low_reliability_support_ok": int((passive_pool & low_reliability_support_ok).sum().item()),
                "low_reliability_with_residual": int((passive_pool & (unreliability > 0.01) & residual_candidate_signal).sum().item()),
                "support_inconsistency_residual": int((passive_pool & support_inconsistent_residual).sum().item()),
                "support_inconsistency_only": int((passive_pool & (support_inconsistency > 0.35) & (~residual_candidate_signal)).sum().item()),
                "rescue_ready": int(rescue_promotion_signal.sum().item()),
                "formed": int(candidate_formation_signal.sum().item()),
                "standard_formed": int(standard_candidate_formation_signal.sum().item()),
                "rescue_fallback_formed": int(rescue_candidate_formation_signal.sum().item()),
                "admitted": int(active_admission_pool.sum().item()),
                "promoted": int(active_promote.sum().item()),
                "standard_promoted": int(standard_active_promote.sum().item()),
                "forced_rescue_promoted": int(rescue_active_promote.sum().item()),
                "passive_to_stable_candidate": int(passive_recovery_candidate.sum().item()),
                "active_to_stable_candidate": int(active_recovery_candidate.sum().item()),
                "passive_stable_ready": int(passive_stable_ready.sum().item()),
                "active_stable_ready": int(active_recover.sum().item()),
                "recovery_ready_soft": int(recovery_ready_soft.sum().item()),
                "recovery_promote_hard": int(recovery_promote_hard.sum().item()),
                "recovery_score_candidate": int(recovery_score_candidate.sum().item()),
                "recovery_streak_ready": int(((passive_recovery_streak_ready | active_recovery_streak_ready)).sum().item()),
                "effective_recovery_stable_ready": int(effective_recovery_stable_ready.sum().item()),
                "passive_runtime_recovery_ready": int(passive_runtime_recovery_ready.sum().item()),
                "passive_explicit_stable_ready": int(passive_explicit_stable_ready.sum().item()),
            }

            next_state = current_state.clone()
            next_state[passive_promote] = GAUSSIAN_STATE_UNSTABLE_PASSIVE
            next_state[active_promote] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            next_state[active_demote] = GAUSSIAN_STATE_UNSTABLE_PASSIVE
            next_state[active_recover] = GAUSSIAN_STATE_STABLE
            next_state[passive_stable_ready] = GAUSSIAN_STATE_STABLE
            next_state[persistent_out] = GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING

            state_changed = next_state != current_state
            transition_stable_to_passive = (
                (current_state == GAUSSIAN_STATE_STABLE)
                & (next_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
            )
            transition_passive_to_stable = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
                & (next_state == GAUSSIAN_STATE_STABLE)
            )
            transition_passive_to_active = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
                & (next_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
            )
            transition_active_to_passive = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (next_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
            )
            transition_active_to_stable = (
                (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
                & (next_state == GAUSSIAN_STATE_STABLE)
            )
            transition_any_to_pending = (
                (current_state != GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING)
                & (next_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING)
            )
            stable_recover = passive_stable_ready | active_recover
            if torch.any(state_changed):
                last_transition_iter[state_changed] = state_update_iter
                promotion_streak[state_changed] = 0
                demotion_streak[state_changed] = 0
                recovery_streak[state_changed] = 0
            promote_cooldown_reset = active_promote
            transition_cooldown_reset = active_demote | active_recover | persistent_out | stable_recover
            if torch.any(promote_cooldown_reset):
                self._atlas_state_cooldown[promote_cooldown_reset] = torch.maximum(
                    self._atlas_state_cooldown[promote_cooldown_reset],
                    torch.full_like(
                        self._atlas_state_cooldown[promote_cooldown_reset],
                        active_min_lifetime_iters,
                    ),
                )
            if torch.any(transition_cooldown_reset):
                self._atlas_state_cooldown[transition_cooldown_reset] = torch.maximum(
                    self._atlas_state_cooldown[transition_cooldown_reset],
                    torch.full_like(
                        self._atlas_state_cooldown[transition_cooldown_reset],
                        state_cooldown_iters,
                    ),
                )
            self._atlas_high_residual_count[stable_candidate | persistent_out] = 0
            self._atlas_low_residual_count[passive_promote | active_promote | persistent_out] = 0
            next_active_lifetime = torch.zeros_like(active_lifetime)
            next_active_provenance = torch.zeros_like(active_provenance)
            stay_active = (current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & (next_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
            new_active = (current_state != GAUSSIAN_STATE_UNSTABLE_ACTIVE) & (next_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
            next_active_lifetime[stay_active] = active_lifetime[stay_active] + 1
            next_active_lifetime[new_active] = 1
            if torch.any(stay_active):
                carried_provenance = active_provenance[stay_active].clamp_min(ACTIVE_PROVENANCE_NONE)
                missing_provenance = carried_provenance <= ACTIVE_PROVENANCE_NONE
                fallback_provenance = (
                    ACTIVE_PROVENANCE_FROM_STATE_REBUILD_AFTER_GC
                    if bool(state_rebuild_after_gc)
                    else ACTIVE_PROVENANCE_FROM_QUOTA_CARRYOVER
                )
                carried_provenance = torch.where(
                    missing_provenance,
                    torch.full_like(carried_provenance, fallback_provenance),
                    carried_provenance,
                )
                next_active_provenance[stay_active] = carried_provenance
            transition_active = new_active & standard_active_promote
            forced_rescue_active = new_active & rescue_active_promote
            unclassified_new_active = new_active & (~transition_active) & (~forced_rescue_active)
            next_active_provenance[transition_active] = ACTIVE_PROVENANCE_FROM_TRANSITION_PASSIVE_TO_ACTIVE
            next_active_provenance[forced_rescue_active] = ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP
            next_active_provenance[unclassified_new_active] = (
                ACTIVE_PROVENANCE_FROM_STATE_REBUILD_AFTER_GC
                if bool(state_rebuild_after_gc)
                else ACTIVE_PROVENANCE_FROM_TRANSITION_PASSIVE_TO_ACTIVE
            )
            self._atlas_active_lifetime = next_active_lifetime
            self._atlas_active_provenance = next_active_provenance
            self._atlas_state = next_state
            active_mask = self._atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE

            metrics = {
                "stable_ratio": float((self._atlas_state == GAUSSIAN_STATE_STABLE).float().mean().item()),
                "passive_ratio": float((self._atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).float().mean().item()),
                "active_ratio": float(active_mask.float().mean().item()),
                "out_of_anchor_ratio": float((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).float().mean().item()),
                "pending_ratio": float((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).float().mean().item()),
                "state_stable_count": int((self._atlas_state == GAUSSIAN_STATE_STABLE).sum().item()),
                "state_passive_count": int((self._atlas_state == GAUSSIAN_STATE_UNSTABLE_PASSIVE).sum().item()),
                "state_active_count": int(active_mask.sum().item()),
                "state_out_pending_count": int((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item()),
                "out_of_anchor_pending_count": int((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item()),
                "explore_score_mean": float(explore_score.mean().item()) if explore_score.numel() > 0 else 0.0,
                "explore_score_max": float(explore_score.max().item()) if explore_score.numel() > 0 else 0.0,
                "promotion_score_mean": float(explore_score.mean().item()) if explore_score.numel() > 0 else 0.0,
                "promotion_score_max": float(explore_score.max().item()) if explore_score.numel() > 0 else 0.0,
                "explore_candidate_ratio": float(active_candidate_pool.float().mean().item()) if active_candidate_pool.numel() > 0 else 0.0,
                "active_candidate_pool_ratio": float(active_candidate_pool.float().mean().item()) if active_candidate_pool.numel() > 0 else 0.0,
                "active_admission_pool_ratio": float(active_admission_pool.float().mean().item()) if active_admission_pool.numel() > 0 else 0.0,
                "active_safety_gate_ratio": float(admission_safety_gate.float().mean().item()) if admission_safety_gate.numel() > 0 else 0.0,
                "active_evidence_ratio": float(formation_evidence.float().mean().item()) if formation_evidence.numel() > 0 else 0.0,
                "candidate_formation_ratio": float(candidate_formation_signal.float().mean().item()) if candidate_formation_signal.numel() > 0 else 0.0,
                "runtime_stable_support_count": int(runtime_stable_support.sum().item()),
                "runtime_stable_support_ratio": float(runtime_stable_support.float().mean().item()) if runtime_stable_support.numel() > 0 else 0.0,
                "runtime_recovery_support_count": int(runtime_recovery_support.sum().item()),
                "runtime_recovery_support_ratio": float(runtime_recovery_support.float().mean().item()) if runtime_recovery_support.numel() > 0 else 0.0,
                "reliability_stable_support_count": int((geometric_stable_class & reliability_stable_support).sum().item()),
                "reliability_recovery_ready_count": int(reliability_recovery_ready.sum().item()),
                "effective_recovery_stable_ready_count": int(effective_recovery_stable_ready.sum().item()),
                "passive_runtime_recovery_ready_count": int(passive_runtime_recovery_ready.sum().item()),
                "passive_explicit_stable_ready_count": int(passive_explicit_stable_ready.sum().item()),
                "recovery_score_mean": float(recovery_score.mean().item()) if recovery_score.numel() > 0 else 0.0,
                "recovery_score_max": float(recovery_score.max().item()) if recovery_score.numel() > 0 else 0.0,
                "recovery_score_threshold": float(recovery_score_threshold),
                "recovery_photo_reference": float(recovery_photo_reference),
                "recovery_photo_soft_max": float(recovery_photo_soft_max),
                "recovery_photo_hard_max": float(recovery_photo_hard_max),
                "recovery_photo_median": float(photo_median),
                "recovery_photo_q75": float(photo_q75),
                "recovery_ready_soft_count": int(recovery_ready_soft.sum().item()),
                "recovery_promote_hard_count": int(recovery_promote_hard.sum().item()),
                "recovery_score_candidate_count": int(recovery_score_candidate.sum().item()),
                "recovery_candidate_count": int(recovery_candidate_pool.sum().item()),
                "recovery_required_streak": int(recovery_required_streak),
                "recovery_streak_max": int(recovery_streak.max().item()) if recovery_streak.numel() > 0 else 0,
                "recovery_streak_ready_count": int(((passive_recovery_streak_ready | active_recovery_streak_ready)).sum().item()),
                "recovery_block_low_reliability": int(((passive_pool | active_pool) & stable_support & (recovery_effective_reliability < recovery_relaxed_reliability_min)).sum().item()),
                "recovery_block_low_support_consistency": int(((passive_pool | active_pool) & stable_support & (support_consistency < recovery_relaxed_support_min)).sum().item()),
                "recovery_block_low_ref_consistency": int(((passive_pool | active_pool) & stable_support & (recovery_ref_consistency < recovery_relaxed_ref_min) & (~ref_camera_valid) & (~runtime_recovery_support)).sum().item()),
                "recovery_block_high_photo_ema": int(((passive_pool | active_pool) & stable_support & (photo_ema > recovery_photo_soft_max) & (recovery_photo_quality < 0.20) & (~runtime_recovery_support)).sum().item()),
                "recovery_block_high_drift": int(((passive_pool | active_pool) & stable_support & (anchor_drift_ratio > recovery_drift_max)).sum().item()),
                "recovery_block_cooldown": int(((passive_recovery_candidate | active_recovery_candidate) & cooldown_active).sum().item()),
                "recovery_block_persistent_out": int(((passive_pool | active_pool) & persistent_out).sum().item()),
                "passive_to_stable_candidate_count": int(passive_recovery_candidate.sum().item()),
                "active_to_stable_candidate_count": int(active_recovery_candidate.sum().item()),
                "passive_to_stable_reliability_min": float(passive_to_stable_reliability_min),
                "passive_to_stable_support_consistency_min": float(passive_to_stable_support_consistency_min),
                "passive_to_stable_drift_max": float(passive_to_stable_drift_max),
                "passive_to_stable_photo_ema_max": float(passive_to_stable_photo_ema_max),
                "sustained_high_residual_ratio": float(sustained_high_residual.float().mean().item()) if sustained_high_residual.numel() > 0 else 0.0,
                "sustained_low_residual_ratio": float(sustained_low_residual.float().mean().item()) if sustained_low_residual.numel() > 0 else 0.0,
                "persistent_drift_ratio": float(sustained_drift.float().mean().item()) if sustained_drift.numel() > 0 else 0.0,
                "gc_block_ratio": float(reattach_recently_failed.float().mean().item()) if reattach_recently_failed.numel() > 0 else 0.0,
                "gc_heavy_block_ratio": float(gc_heavy_block.float().mean().item()) if gc_heavy_block.numel() > 0 else 0.0,
                "formation_visibility_ready_ratio": float(formation_visibility_ready.float().mean().item()) if formation_visibility_ready.numel() > 0 else 0.0,
                "formation_opacity_ready_ratio": float(formation_opacity_ready.float().mean().item()) if formation_opacity_ready.numel() > 0 else 0.0,
                "visibility_rescue_ready_ratio": float(visibility_rescue_ready.float().mean().item()) if visibility_rescue_ready.numel() > 0 else 0.0,
                "opacity_alive_ratio": float(opacity_alive.float().mean().item()) if opacity_alive.numel() > 0 else 0.0,
                "ref_camera_ready_ratio": float(ref_camera_ready.float().mean().item()) if ref_camera_ready.numel() > 0 else 0.0,
                "ref_camera_valid_ratio": float(ref_camera_valid.float().mean().item()) if ref_camera_valid.numel() > 0 else 0.0,
                "slab_ready_proxy_ratio": float(slab_ready_proxy.float().mean().item()) if slab_ready_proxy.numel() > 0 else 0.0,
                "state_cooldown_ratio": float(cooldown_active.float().mean().item()) if cooldown_active.numel() > 0 else 0.0,
                "cooldown_block_ratio": float(cooldown_block.float().mean().item()) if cooldown_block.numel() > 0 else 0.0,
                "active_lifetime_lock_ratio": float(active_lifetime_lock.float().mean().item()) if active_lifetime_lock.numel() > 0 else 0.0,
                "promotion_streak_max": int(promotion_streak.max().item()) if promotion_streak.numel() > 0 else 0,
                "promotion_streak_ready_count": int((promotion_streak >= promotion_required_streak).sum().item()) if promotion_streak.numel() > 0 else 0,
                "demotion_streak_max": int(demotion_streak.max().item()) if demotion_streak.numel() > 0 else 0,
                "mean_active_lifetime": float(self._atlas_active_lifetime[active_mask].float().mean().item()) if torch.any(active_mask) else 0.0,
                "max_active_lifetime": float(self._atlas_active_lifetime[active_mask].max().item()) if torch.any(active_mask) else 0.0,
                "hard_active_exit_ratio": float(hard_active_exit.float().mean().item()) if hard_active_exit.numel() > 0 else 0.0,
                "soft_active_exit_ratio": float(soft_active_exit.float().mean().item()) if soft_active_exit.numel() > 0 else 0.0,
                "state_changed_ratio": float(state_changed.float().mean().item()) if state_changed.numel() > 0 else 0.0,
                "promote_to_active_threshold": promote_to_active_threshold,
                "demote_to_passive_threshold": demote_to_passive_threshold,
                "active_max_lifetime_iters": int(active_max_lifetime_iters),
                "active_nonimprove_iters": int(active_nonimprove_iters),
                "active_quota_target": int(active_quota_target),
                "active_quota_soft_target": int(quota_from_ratio),
                "active_quota_hard_target": int(active_quota_max),
                "active_quota_effective_live_active_count": int(active_mask.sum().item()),
                "active_quota_over_target_count": int(max(int(active_mask.sum().item()) - int(active_quota_target), 0)),
                "active_quota_available": int(active_quota_available),
                "active_quota_current_count": int(current_active_count),
                "active_quota_before_release_count": int(current_active_count),
                "active_quota_release_count": int(projected_active_exits),
                "active_quota_projected_exit_count": int(projected_active_exits),
                "active_quota_projected_after_exit": int(projected_active_after_exit),
                "active_quota_after_release_count": int(projected_active_after_exit),
                "active_quota_transition_overflow": int(transition_overflow_quota),
                "active_candidate_pool_count": int(active_candidate_pool.sum().item()),
                "candidate_formation_count": int(candidate_formation_signal.sum().item()),
                "active_formed_count": int(candidate_formation_signal.sum().item()),
                "active_standard_formation_count": int(standard_candidate_formation_signal.sum().item()),
                "active_rescue_fallback_formation_count": int(rescue_candidate_formation_signal.sum().item()),
                "active_standard_candidate_pool_count": int(standard_streak_ready.sum().item()),
                "active_rescue_candidate_pool_count": int(rescue_streak_ready.sum().item()),
                "active_admitted_count": int(active_admission_pool.sum().item()),
                "active_promoted_count": int(active_promote.sum().item()),
                "active_admission_pool_count": int(active_admission_pool.sum().item()),
                "active_standard_admission_pool_count": int(standard_active_admission_pool.sum().item()),
                "active_rescue_admission_pool_count": int(rescue_active_admission_pool.sum().item()),
                "active_promote_count": int(active_promote.sum().item()),
                "active_standard_promote_count": int(standard_active_promote.sum().item()),
                "active_forced_rescue_promote_count": int(rescue_active_promote.sum().item()),
                "active_new_active_count": int(new_active.sum().item()),
                "active_rescue_candidate_count": int(rescue_ready_override.sum().item()),
                "active_rescue_promote_count": int(rescue_active_promote.sum().item()),
                "active_admission_score_ready_count": int((standard_streak_ready & admission_score_ready).sum().item()),
                "active_rescue_score_ready_count": int((rescue_streak_ready & rescue_score_ready).sum().item()),
                "active_hard_exit_count": int(((current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & hard_active_exit).sum().item()),
                "active_soft_exit_count": int(((current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & soft_active_exit).sum().item()),
                "active_lifetime_cap_exit_count": int(((current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & active_lifetime_cap_exit).sum().item()),
                "active_nonimproving_exit_count": int(((current_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE) & active_nonimproving_exit).sum().item()),
                "active_stale_fallback_exit_count": int(active_stale_fallback_exit.sum().item()),
                "active_fallback_handoff_exit_count": int(fallback_transition_handoff_exit.sum().item()),
                "active_lifetime_release_count": int((active_mask & (~active_lifetime_lock)).sum().item()),
                "active_demoted_count": int(transition_active_to_passive.sum().item()),
                "active_exited_count": int((transition_active_to_passive | transition_active_to_stable | transition_any_to_pending).sum().item()),
                "active_to_stable_exit_count": int(transition_active_to_stable.sum().item()),
                "active_to_passive_exit_count": int(transition_active_to_passive.sum().item()),
                "active_provenance_preserved_count": int(stay_active.sum().item()),
                "passive_to_stable_candidate_count": int(passive_recovery_candidate.sum().item()),
                "active_to_stable_candidate_count": int(active_recovery_candidate.sum().item()),
                "passive_stable_ready_count": int(passive_stable_ready.sum().item()),
                "active_stable_ready_count": int(active_recover.sum().item()),
                "passive_stable_cooldown_bypass_count": int((passive_stable_ready & cooldown_active).sum().item()),
                "state_reliability_stable_ready_count": int(state_reliability_stable_ready.sum().item()),
                "transition_stable_to_passive_count": int(transition_stable_to_passive.sum().item()),
                "transition_passive_to_stable_count": int(transition_passive_to_stable.sum().item()),
                "transition_passive_to_active_count": int(transition_passive_to_active.sum().item()),
                "transition_passive_to_active_standard_count": int((transition_passive_to_active & transition_active).sum().item()),
                "transition_passive_to_active_rescue_count": int((transition_passive_to_active & forced_rescue_active).sum().item()),
                "transition_passive_to_active_unclassified_count": int((transition_passive_to_active & unclassified_new_active).sum().item()),
                "transition_active_to_passive_count": int(transition_active_to_passive.sum().item()),
                "transition_active_to_stable_count": int(transition_active_to_stable.sum().item()),
                "transition_any_to_pending_count": int(transition_any_to_pending.sum().item()),
                "max_high_residual_iters": int(high_residual_count.max().item()) if high_residual_count.numel() > 0 else 0,
                "max_low_residual_iters": int(low_residual_count.max().item()) if low_residual_count.numel() > 0 else 0,
                "state_candidate_reason_breakdown": state_candidate_reason_breakdown,
                "promotion_block_reason_breakdown": promotion_block_reason_breakdown,
                "transition_passive_to_stable_block_reason_breakdown": passive_to_stable_block_reason_breakdown,
                "transition_active_to_stable_block_reason_breakdown": active_to_stable_block_reason_breakdown,
            }
            metrics.update(self._summarize_active_provenance_metrics(active_mask=active_mask))
            return metrics

    def run_atlas_gc(
        self,
        reattach_radius_mult: float,
        surface_stable_min: float,
        edge_stable_min: float,
        min_visibility_ema: float,
        stable_residual_threshold: float,
        activate_threshold: float | None = None,
        deactivate_threshold: float | None = None,
        activate_min_high_residual_iters: int = 1,
        recover_low_residual_iters: int = 3,
        drift_activate_iters: int = 2,
        out_of_anchor_drift_iters: int = 3,
        out_of_anchor_gc_failures: int = 2,
        state_cooldown_iters: int = 5,
        active_min_lifetime_iters: int = 5,
        active_quota_ratio: float = 0.03,
        active_quota_min: int = 1,
        active_quota_max: int = 128,
        min_active_opacity: float = 0.02,
        max_reattach_failures: int = 2,
        forced_prune_opacity: float = 0.01,
        retry_pending: bool = True,
        promote_to_active_threshold: float | None = None,
        demote_to_passive_threshold: float | None = None,
        active_max_lifetime_iters: int = 600,
        active_nonimprove_iters: int = 180,
    ):
        if not self.has_atlas_bindings:
            return {}

        with torch.no_grad():
            gc_batch = self._prepare_atlas_gc_batch(include_pending_retries=bool(retry_pending))
            if gc_batch is None:
                pending_count = int((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item())
                mean_gc_fail = float(self._atlas_gc_fail_count.float().mean().item()) if self._atlas_gc_fail_count.numel() > 0 else 0.0
                return {
                    "gc_ran": 1.0,
                    "gc_candidates": 0,
                    "gc_drift_candidates": 0,
                    "gc_pending_candidates": 0,
                    "gc_retry_pending_enabled": 1.0 if retry_pending else 0.0,
                    "gc_compacted": 0,
                    "reattach_success": 0,
                    "reattach_fail": 0,
                    "reattach_success_ratio": 0.0,
                    "reattach_fail_ratio": 0.0,
                    "pending_reattach_success": 0,
                    "pending_reattach_fail": 0,
                    "pending_reattach_success_ratio": 0.0,
                    "pending_reattach_fail_ratio": 0.0,
                    "forced_pending": 0,
                    "pending_forced_attach_count": 0,
                    "forced_attach_count": 0,
                    "pending_prune_count": 0,
                    "out_of_anchor_pending_count": pending_count,
                    "mean_gc_fail_count": mean_gc_fail,
                    "mean_gc_fail_count_after": mean_gc_fail,
                    "prune_after_gc": 0,
                    "reattach_tier1_attempt_count": 0,
                    "reattach_tier1_raw_accept_count": 0,
                    "reattach_tier1_success": 0,
                    "reattach_tier2_attempt_count": 0,
                    "reattach_tier2_raw_accept_count": 0,
                    "reattach_tier2_success": 0,
                    "reattach_tier3_attempt_count": 0,
                    "reattach_tier3_raw_accept_count": 0,
                    "reattach_tier3_success": 0,
                    "reattach_tier4_attempt_count": 0,
                    "reattach_tier4_forced_success": 0,
                    "reattach_candidate_starvation_count": 0,
                    "reattach_candidate_starvation_ratio": 0.0,
                    "ray_guided_queries": 0,
                    "ray_guided_priority_queries": 0,
                    "ray_guided_late_queries": 0,
                    "mode": "idle",
                }

            drifted_idx = gc_batch["indices"]
            drifted_xyz = gc_batch["xyz"]
            anchor_node_ids = gc_batch["anchor_ids"]
            candidate_state = self._atlas_state[drifted_idx].clone()
            pending_candidate_mask = candidate_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
            finite_radius = self._atlas_radius.detach()
            finite_radius = finite_radius[torch.isfinite(finite_radius) & (finite_radius > 0.0)]
            median_radius = float(torch.median(finite_radius).item()) if finite_radius.numel() > 0 else 0.0
            cell_size = float(self._atlas_hash_cell_size) if self._atlas_hash_ready and self._atlas_hash_cell_size > 0.0 else max(median_radius, 1e-6)
            probe_radius = max(int(np.ceil((float(reattach_radius_mult) * max(median_radius, cell_size)) / max(cell_size, 1e-6))), 1)
            nearest_ids, nearest_dist, hash_stats = self._find_nearest_atlas_nodes_hashed(
                drifted_xyz,
                anchor_node_ids,
                probe_radius=probe_radius,
            )
            raw_accept = nearest_dist <= (float(reattach_radius_mult) * self._atlas_radius[nearest_ids].clamp_min(1e-6))
            accept, reattach_quality_stats = self._evaluate_reattach_accept_policy(
                drifted_idx,
                nearest_ids,
                nearest_dist,
                radius_mult=float(reattach_radius_mult),
                preaccept_mask=raw_accept,
                pending_candidate_mask=pending_candidate_mask,
            )
            final_nearest_ids = nearest_ids.clone()
            final_nearest_dist = nearest_dist.clone()
            accept_tier = torch.zeros_like(anchor_node_ids, dtype=torch.long)
            accept_tier[accept] = 1
            def _merge_reattach_quality_stats(new_stats):
                for key, value in (new_stats or {}).items():
                    if key in ("reattach_quality_reliability_mean", "reattach_quality_ref_mean"):
                        reattach_quality_stats[key] = max(float(reattach_quality_stats.get(key, 0.0)), float(value))
                    else:
                        reattach_quality_stats[key] = int(reattach_quality_stats.get(key, 0)) + int(value)

            expanded_hash_stats = {
                "expanded_mode": "skipped",
                "expanded_bucket_queries": 0,
                "expanded_fallback_full_search": 0,
                "expanded_mean_candidate_count": 0.0,
                "expanded_max_candidate_count": 0,
            }
            ray_stats = {
                "ray_guided_queries": 0,
                "ray_guided_ref_valid": 0,
                "ray_guided_active_queries": 0,
                "ray_guided_pending_queries": 0,
                "ray_guided_preaccept_count": 0,
                "ray_guided_quality_accept_count": 0,
                "ray_guided_empty_seed_count": 0,
                "ray_guided_mean_candidate_count": 0.0,
                "ray_guided_max_candidate_count": 0,
                "ray_guided_priority_queries": 0,
                "ray_guided_late_queries": 0,
            }
            def _merge_ray_guided_stats(new_stats, quality_accept_count: int = 0, prefix: str | None = None):
                nonlocal ray_stats
                new_stats = new_stats or {}
                old_queries = int(ray_stats.get("ray_guided_queries", 0))
                new_queries = int(new_stats.get("ray_guided_queries", 0))
                total_queries = old_queries + new_queries
                old_mean = float(ray_stats.get("ray_guided_mean_candidate_count", 0.0))
                new_mean = float(new_stats.get("ray_guided_mean_candidate_count", 0.0))
                ray_stats["ray_guided_queries"] = total_queries
                ray_stats["ray_guided_ref_valid"] = int(ray_stats.get("ray_guided_ref_valid", 0)) + int(new_stats.get("ray_guided_ref_valid", 0))
                ray_stats["ray_guided_active_queries"] = int(ray_stats.get("ray_guided_active_queries", 0)) + int(new_stats.get("ray_guided_active_queries", 0))
                ray_stats["ray_guided_pending_queries"] = int(ray_stats.get("ray_guided_pending_queries", 0)) + int(new_stats.get("ray_guided_pending_queries", 0))
                ray_stats["ray_guided_preaccept_count"] = int(ray_stats.get("ray_guided_preaccept_count", 0)) + int(new_stats.get("ray_guided_preaccept_count", 0))
                ray_stats["ray_guided_quality_accept_count"] = int(ray_stats.get("ray_guided_quality_accept_count", 0)) + int(quality_accept_count)
                ray_stats["ray_guided_empty_seed_count"] = int(ray_stats.get("ray_guided_empty_seed_count", 0)) + int(new_stats.get("ray_guided_empty_seed_count", 0))
                ray_stats["ray_guided_mean_candidate_count"] = (
                    ((old_mean * old_queries) + (new_mean * new_queries)) / max(total_queries, 1)
                )
                ray_stats["ray_guided_max_candidate_count"] = max(
                    int(ray_stats.get("ray_guided_max_candidate_count", 0)),
                    int(new_stats.get("ray_guided_max_candidate_count", 0)),
                )
                if prefix is not None:
                    ray_stats[f"ray_guided_{prefix}_queries"] = new_queries
                    ray_stats[f"ray_guided_{prefix}_preaccept_count"] = int(new_stats.get("ray_guided_preaccept_count", 0))
                    ray_stats[f"ray_guided_{prefix}_quality_accept_count"] = int(quality_accept_count)

            expanded_probe_radius = int(probe_radius)
            expanded_radius_mult = max(float(reattach_radius_mult) * 2.0, float(reattach_radius_mult) + 1.0)
            tier1_attempt_count = int(drifted_idx.shape[0])
            tier1_raw_accept_count = int(raw_accept.sum().item()) if raw_accept.numel() > 0 else 0
            tier2_attempt_count = 0
            tier2_raw_accept_count = 0
            tier4_attempt_count = 0

            active_or_pending_remaining = (~accept) & (
                pending_candidate_mask | (candidate_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE)
            )
            if torch.any(active_or_pending_remaining):
                priority_positions = torch.nonzero(active_or_pending_remaining, as_tuple=False).squeeze(-1)
                ray_ids, ray_dist, ray_preaccept, priority_ray_stats = self._ray_guided_pending_reattach(
                    drifted_xyz[priority_positions],
                    drifted_idx[priority_positions],
                    anchor_node_ids[priority_positions],
                    radius_mult=expanded_radius_mult,
                )
                ray_accept, ray_quality_stats = self._evaluate_reattach_accept_policy(
                    drifted_idx[priority_positions],
                    ray_ids,
                    ray_dist,
                    radius_mult=expanded_radius_mult,
                    preaccept_mask=ray_preaccept,
                    pending_candidate_mask=pending_candidate_mask[priority_positions],
                )
                _merge_reattach_quality_stats(ray_quality_stats)
                _merge_ray_guided_stats(priority_ray_stats, int(ray_accept.sum().item()), prefix="priority")
                if torch.any(ray_accept):
                    accepted_positions = priority_positions[ray_accept]
                    final_nearest_ids[accepted_positions] = ray_ids[ray_accept]
                    final_nearest_dist[accepted_positions] = ray_dist[ray_accept]
                    accept[accepted_positions] = True
                    accept_tier[accepted_positions] = 3

            if torch.any(~accept):
                remaining_positions = torch.nonzero(~accept, as_tuple=False).squeeze(-1)
                remaining_idx = drifted_idx[remaining_positions]
                remaining_anchor_ids = anchor_node_ids[remaining_positions].clone()
                last_good_ids = self._atlas_last_good_node_ids[remaining_idx]
                valid_last_good = (last_good_ids >= 0) & (last_good_ids < self._atlas_positions.shape[0])
                if torch.any(valid_last_good):
                    remaining_anchor_ids[valid_last_good] = last_good_ids[valid_last_good]
                expanded_probe_radius = max(int(probe_radius) + 1, int(probe_radius) * 2)
                expanded_ids, expanded_dist, expanded_hash_stats_raw = self._find_nearest_atlas_nodes_hashed(
                    drifted_xyz[remaining_positions],
                    remaining_anchor_ids,
                    probe_radius=expanded_probe_radius,
                )
                expanded_accept_raw = expanded_dist <= (expanded_radius_mult * self._atlas_radius[expanded_ids].clamp_min(1e-6))
                tier2_attempt_count = int(remaining_positions.shape[0])
                tier2_raw_accept_count = int(expanded_accept_raw.sum().item()) if expanded_accept_raw.numel() > 0 else 0
                expanded_accept, expanded_quality_stats = self._evaluate_reattach_accept_policy(
                    remaining_idx,
                    expanded_ids,
                    expanded_dist,
                    radius_mult=expanded_radius_mult,
                    preaccept_mask=expanded_accept_raw,
                    pending_candidate_mask=pending_candidate_mask[remaining_positions],
                )
                _merge_reattach_quality_stats(expanded_quality_stats)
                if torch.any(expanded_accept):
                    accepted_positions = remaining_positions[expanded_accept]
                    final_nearest_ids[accepted_positions] = expanded_ids[expanded_accept]
                    final_nearest_dist[accepted_positions] = expanded_dist[expanded_accept]
                    accept[accepted_positions] = True
                    accept_tier[accepted_positions] = 2
                expanded_hash_stats = {
                    f"expanded_{key}": value
                    for key, value in expanded_hash_stats_raw.items()
                }

            if torch.any(~accept):
                remaining_positions = torch.nonzero(~accept, as_tuple=False).squeeze(-1)
                ray_ids, ray_dist, ray_preaccept, late_ray_stats = self._ray_guided_pending_reattach(
                    drifted_xyz[remaining_positions],
                    drifted_idx[remaining_positions],
                    anchor_node_ids[remaining_positions],
                    radius_mult=expanded_radius_mult,
                )
                ray_accept, ray_quality_stats = self._evaluate_reattach_accept_policy(
                    drifted_idx[remaining_positions],
                    ray_ids,
                    ray_dist,
                    radius_mult=expanded_radius_mult,
                    preaccept_mask=ray_preaccept,
                    pending_candidate_mask=pending_candidate_mask[remaining_positions],
                )
                _merge_reattach_quality_stats(ray_quality_stats)
                _merge_ray_guided_stats(late_ray_stats, int(ray_accept.sum().item()), prefix="late")
                if torch.any(ray_accept):
                    accepted_positions = remaining_positions[ray_accept]
                    final_nearest_ids[accepted_positions] = ray_ids[ray_accept]
                    final_nearest_dist[accepted_positions] = ray_dist[ray_accept]
                    accept[accepted_positions] = True
                    accept_tier[accepted_positions] = 3

            if torch.any(accept):
                accepted_idx = drifted_idx[accept]
                self._apply_atlas_reattach(
                    accepted_idx,
                    final_nearest_ids[accept],
                    state_cooldown_iters=state_cooldown_iters,
                    low_confidence=False,
                )

            forced_pending_count = 0
            forced_attach_count = 0
            pending_prune_count = 0
            prune_after_gc = 0
            pruned_candidate_mask = torch.zeros_like(accept, dtype=torch.bool)
            if torch.any(~accept):
                failed_positions = torch.nonzero(~accept, as_tuple=False).squeeze(-1)
                failed_idx = drifted_idx[failed_positions]
                failed_was_pending = pending_candidate_mask[failed_positions]
                self._record_pending_retry_metadata(failed_idx, prefer_current_anchor=True)
                self._atlas_gc_fail_count[failed_idx] = self._atlas_gc_fail_count[failed_idx] + 1
                if torch.any(failed_was_pending):
                    pending_failed_idx = failed_idx[failed_was_pending]
                    self._atlas_pending_retry_count[pending_failed_idx] = self._atlas_pending_retry_count[pending_failed_idx] + 1
                if int(max(state_cooldown_iters, 0)) > 0:
                    self._atlas_state_cooldown[failed_idx] = torch.maximum(
                        self._atlas_state_cooldown[failed_idx],
                        torch.full_like(self._atlas_state_cooldown[failed_idx], int(max(state_cooldown_iters, 0))),
                    )
                max_retry = int(max(max_reattach_failures, 1))
                exceeded_retry = (
                    (self._atlas_gc_fail_count[failed_idx] > max_retry)
                    | (self._atlas_pending_retry_count[failed_idx] > max_retry)
                )
                exceeded_idx = failed_idx[exceeded_retry]
                exceeded_positions = failed_positions[exceeded_retry]
                if exceeded_idx.numel() > 0:
                    opacity = self.get_opacity.detach().squeeze(-1)[exceeded_idx]
                    visibility = self._atlas_visibility_ema.detach()[exceeded_idx]
                    dead_opacity_threshold = max(float(forced_prune_opacity) * 1.5, 1e-5)
                    dead_visibility_threshold = max(float(min_visibility_ema) * 0.25, 0.0)
                    dead_mask = (
                        (opacity <= dead_opacity_threshold)
                        | (visibility <= dead_visibility_threshold)
                    )
                    live_exceeded_idx = exceeded_idx[~dead_mask]
                    live_exceeded_positions = exceeded_positions[~dead_mask]
                    if live_exceeded_idx.numel() > 0:
                        tier4_attempt_count += int(live_exceeded_idx.shape[0])
                        forced_ids, forced_dist = self._chunked_nearest_atlas_nodes(
                            self._xyz.detach()[live_exceeded_idx],
                            chunk_size=1024,
                        )
                        forced_accept, forced_quality_stats = self._evaluate_reattach_accept_policy(
                            live_exceeded_idx,
                            forced_ids,
                            forced_dist,
                            radius_mult=max(expanded_radius_mult * 1.5, float(reattach_radius_mult) + 2.0),
                            preaccept_mask=torch.ones_like(forced_ids, dtype=torch.bool),
                            pending_candidate_mask=torch.ones_like(forced_ids, dtype=torch.bool),
                            force_relaxed=True,
                        )
                        _merge_reattach_quality_stats(forced_quality_stats)
                        if torch.any(forced_accept):
                            forced_accept_idx = live_exceeded_idx[forced_accept]
                            forced_accept_positions = live_exceeded_positions[forced_accept]
                            self._apply_atlas_reattach(
                                forced_accept_idx,
                                forced_ids[forced_accept],
                                state_cooldown_iters=state_cooldown_iters,
                                low_confidence=True,
                            )
                            final_nearest_ids[forced_accept_positions] = forced_ids[forced_accept]
                            final_nearest_dist[forced_accept_positions] = forced_dist[forced_accept]
                            accept[forced_accept_positions] = True
                            accept_tier[forced_accept_positions] = 4
                            forced_attach_count = int(forced_accept_idx.shape[0])
                        if torch.any(~forced_accept):
                            still_pending_idx = live_exceeded_idx[~forced_accept]
                            self._atlas_state[still_pending_idx] = GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
                            self._atlas_high_residual_count[still_pending_idx] = 0
                            self._atlas_low_residual_count[still_pending_idx] = 0
                            self._atlas_promotion_streak[still_pending_idx] = 0
                            self._atlas_demotion_streak[still_pending_idx] = 0
                            self._atlas_last_transition_iter[still_pending_idx] = int(max(getattr(self, "_atlas_state_update_iter", 0), 0))
                            self._atlas_active_lifetime[still_pending_idx] = 0
                            forced_pending_count += int(still_pending_idx.shape[0])

                    dead_exceeded_idx = exceeded_idx[dead_mask]
                    dead_exceeded_positions = exceeded_positions[dead_mask]
                    if dead_exceeded_idx.numel() > 0:
                        if self.optimizer is not None:
                            prune_mask = torch.zeros((self._xyz.shape[0],), dtype=torch.bool, device=self._device())
                            prune_mask[dead_exceeded_idx] = True
                            prune_after_gc = self.prune_points(prune_mask)
                            pruned_candidate_mask[dead_exceeded_positions] = True
                            pending_prune_count = int(prune_after_gc)
                        else:
                            self._set_opacity_subset(dead_exceeded_idx, target_opacity=float(forced_prune_opacity))
                            self._atlas_state[dead_exceeded_idx] = GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
                            self._atlas_high_residual_count[dead_exceeded_idx] = 0
                            self._atlas_low_residual_count[dead_exceeded_idx] = 0
                            self._atlas_promotion_streak[dead_exceeded_idx] = 0
                            self._atlas_demotion_streak[dead_exceeded_idx] = 0
                            self._atlas_last_transition_iter[dead_exceeded_idx] = int(max(getattr(self, "_atlas_state_update_iter", 0), 0))
                            self._atlas_active_lifetime[dead_exceeded_idx] = 0
                            forced_pending_count += int(dead_exceeded_idx.shape[0])
            self.update_atlas_states(
                surface_stable_min=surface_stable_min,
                edge_stable_min=edge_stable_min,
                min_visibility_ema=min_visibility_ema,
                stable_residual_threshold=stable_residual_threshold,
                activate_threshold=activate_threshold,
                deactivate_threshold=deactivate_threshold,
                activate_min_high_residual_iters=activate_min_high_residual_iters,
                recover_low_residual_iters=recover_low_residual_iters,
                drift_activate_iters=drift_activate_iters,
                out_of_anchor_drift_iters=out_of_anchor_drift_iters,
                out_of_anchor_gc_failures=out_of_anchor_gc_failures,
                state_cooldown_iters=state_cooldown_iters,
                active_min_lifetime_iters=active_min_lifetime_iters,
                active_quota_ratio=active_quota_ratio,
                active_quota_min=active_quota_min,
                active_quota_max=active_quota_max,
                min_active_opacity=min_active_opacity,
                promote_to_active_threshold=promote_to_active_threshold,
                demote_to_passive_threshold=demote_to_passive_threshold,
                active_max_lifetime_iters=active_max_lifetime_iters,
                active_nonimprove_iters=active_nonimprove_iters,
                state_rebuild_after_gc=True,
            )
            gc_candidates = int(drifted_idx.shape[0])
            unresolved_mask = ~(accept | pruned_candidate_mask)
            reattach_success = int(accept.sum().item())
            reattach_fail = int(unresolved_mask.sum().item())
            pending_reattach_success = int((accept & pending_candidate_mask).sum().item())
            pending_reattach_fail = int((unresolved_mask & pending_candidate_mask).sum().item())
            pending_candidate_count = int(pending_candidate_mask.sum().item())
            pending_count = int((self._atlas_state == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).sum().item())
            mean_gc_fail = float(self._atlas_gc_fail_count.float().mean().item()) if self._atlas_gc_fail_count.numel() > 0 else 0.0
            return {
                "gc_ran": 1.0,
                "gc_candidates": gc_candidates,
                "gc_drift_candidates": int(gc_batch.get("selected_drift_candidate_count", drifted_idx.shape[0])),
                "gc_pending_candidates": int(gc_batch.get("selected_pending_candidate_count", 0)),
                "gc_retry_pending_enabled": 1.0 if bool(gc_batch.get("retry_pending_enabled", retry_pending)) else 0.0,
                "gc_compacted": int(drifted_xyz.shape[0]),
                "reattach_success": reattach_success,
                "reattach_fail": reattach_fail,
                "reattach_success_ratio": float(reattach_success / max(gc_candidates, 1)),
                "reattach_fail_ratio": float(reattach_fail / max(gc_candidates, 1)),
                "pending_reattach_success": pending_reattach_success,
                "pending_reattach_fail": pending_reattach_fail,
                "pending_reattach_success_ratio": float(pending_reattach_success / max(pending_candidate_count, 1)),
                "pending_reattach_fail_ratio": float(pending_reattach_fail / max(pending_candidate_count, 1)),
                "forced_pending": int(forced_pending_count),
                "pending_forced_attach_count": int((accept_tier[pending_candidate_mask] == 4).sum().item()) if pending_candidate_count > 0 else 0,
                "forced_attach_count": int(forced_attach_count),
                "pending_prune_count": int(pending_prune_count),
                "out_of_anchor_pending_count": pending_count,
                "mean_gc_fail_count": mean_gc_fail,
                "mean_gc_fail_count_after": mean_gc_fail,
                "prune_after_gc": int(prune_after_gc),
                "max_fail_count": int(self._atlas_gc_fail_count.max().item()) if self._atlas_gc_fail_count.numel() > 0 else 0,
                "probe_radius_cells": int(probe_radius),
                "expanded_probe_radius_cells": int(expanded_probe_radius),
                "reattach_tier1_attempt_count": int(tier1_attempt_count),
                "reattach_tier1_raw_accept_count": int(tier1_raw_accept_count),
                "reattach_tier1_success": int((accept_tier == 1).sum().item()),
                "reattach_tier2_attempt_count": int(tier2_attempt_count),
                "reattach_tier2_raw_accept_count": int(tier2_raw_accept_count),
                "reattach_tier2_success": int((accept_tier == 2).sum().item()),
                "reattach_tier3_attempt_count": int(ray_stats.get("ray_guided_queries", 0)),
                "reattach_tier3_raw_accept_count": int(ray_stats.get("ray_guided_preaccept_count", 0)),
                "reattach_tier3_success": int((accept_tier == 3).sum().item()),
                "reattach_tier4_attempt_count": int(tier4_attempt_count),
                "reattach_tier4_forced_success": int((accept_tier == 4).sum().item()),
                "reattach_candidate_starvation_count": int(hash_stats.get("candidate_starvation_count", 0))
                + int(expanded_hash_stats.get("expanded_candidate_starvation_count", 0))
                + int(ray_stats.get("ray_guided_empty_seed_count", 0)),
                "reattach_candidate_starvation_ratio": float(
                    (
                        int(hash_stats.get("candidate_starvation_count", 0))
                        + int(expanded_hash_stats.get("expanded_candidate_starvation_count", 0))
                        + int(ray_stats.get("ray_guided_empty_seed_count", 0))
                    )
                    / max(gc_candidates, 1)
                ),
                "hash_bucket_count": int(self._atlas_hash_bucket_count),
                "hash_source": self._atlas_hash_source,
                **reattach_quality_stats,
                **hash_stats,
                **expanded_hash_stats,
                **ray_stats,
            }

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _build_dead_prune_mask(
        self,
        min_opacity: float,
        visibility_threshold: float = 0.02,
        max_reattach_failures: int = 2,
        background_ref_score_min: float = 0.06,
        background_visibility_min: float = 0.003,
        background_guard_enabled: bool = True,
        fidelity_background_guard_strength: float = 0.0,
        return_metrics: bool = False,
    ):
        opacity = self.get_opacity.detach().squeeze(-1)
        low_opacity = opacity < float(min_opacity)
        if not self.has_atlas_bindings:
            return (low_opacity, {}) if return_metrics else low_opacity

        invisible_long = self._atlas_visibility_ema.detach() <= float(max(visibility_threshold, 0.0))
        reattach_failed = self._atlas_gc_fail_count.detach() >= max(int(max_reattach_failures), 1)
        out_of_anchor = self._atlas_state.detach() == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
        dead_mask = torch.logical_or(
            torch.logical_and(low_opacity, torch.logical_or(invisible_long, reattach_failed)),
            torch.logical_and(out_of_anchor, torch.logical_and(invisible_long, reattach_failed)),
        )
        guard_metrics = {
            "background_dead_prune_protected_count": 0.0,
            "background_dead_prune_guard_candidate_count": 0.0,
        }
        if background_guard_enabled and dead_mask.numel() == self._atlas_state.numel():
            guard_strength = float(max(min(fidelity_background_guard_strength, 1.0), 0.0))
            # Protect stable/passive nodes with valid reference and real (if thin) visibility
            # from being dead-pruned due to temporarily low opacity alone.
            is_stable_or_passive = (
                (self._atlas_state.detach() == GAUSSIAN_STATE_STABLE)
                | (self._atlas_state.detach() == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
            )
            support_consistency = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
            node_observed = self._atlas_node_metric_for_gaussians(
                self._atlas_node_observed_score_ema,
                default_value=0.0,
            ).clamp(0.0, 1.0)
            node_coverage = self._atlas_node_metric_for_gaussians(
                self._atlas_refresh_node_coverage_ratio,
                default_value=0.0,
            ).clamp(0.0, 1.0)
            reliability_effective = self.get_gaussian_atlas_reliability_effective.detach().clamp(0.0, 1.0)
            ref_score = self._atlas_ref_score.detach().clamp(0.0, 1.0)
            support_keep_min = max(0.18, 0.26 - 0.06 * guard_strength)
            reliability_keep_min = max(0.10, 0.14 - 0.04 * guard_strength)
            observed_keep_min = max(0.08, 0.12 - 0.03 * guard_strength)
            coverage_keep_min = max(0.07, 0.10 - 0.02 * guard_strength)
            ref_keep_min = max(float(background_ref_score_min) * (1.0 - 0.25 * guard_strength), 0.035)
            background_anchor_hint = (
                (self._atlas_ref_camera.detach() >= 0)
                | (ref_score >= ref_keep_min)
                | (node_observed >= 0.16)
            )
            ref_ready = (
                (ref_score >= ref_keep_min)
                | (node_observed >= 0.20)
                | (reliability_effective >= 0.20)
            ) & background_anchor_hint
            thin_but_supported = (
                (support_consistency >= support_keep_min)
                & (reliability_effective >= reliability_keep_min)
                & (
                    (self._atlas_visibility_ema.detach() >= float(max(background_visibility_min, 0.0)))
                    | (node_observed >= observed_keep_min)
                    | (node_coverage >= coverage_keep_min)
                )
            )
            strong_ref_supported = (
                (ref_score >= max(ref_keep_min, 0.05))
                & (support_consistency >= max(0.22, support_keep_min))
                & (reliability_effective >= max(0.12, reliability_keep_min))
            )
            background_keep = (
                is_stable_or_passive
                & (ref_ready | thin_but_supported | strong_ref_supported)
                & (
                    (self._atlas_visibility_ema.detach() >= float(max(background_visibility_min, 0.0)))
                    | thin_but_supported
                    | strong_ref_supported
                )
            )
            guard_metrics["background_dead_prune_guard_candidate_count"] = float((dead_mask & is_stable_or_passive).sum().item())
            guard_metrics["background_dead_prune_protected_count"] = float((dead_mask & background_keep).sum().item())
            dead_mask = dead_mask & (~background_keep)
        return (dead_mask, guard_metrics) if return_metrics else dead_mask

    def _build_gc_pending_prune_mask(self, max_reattach_failures: int = 2):
        if not self.has_atlas_bindings:
            return torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self._device())
        reattach_failed = self._atlas_gc_fail_count.detach() >= max(int(max_reattach_failures), 1)
        out_of_anchor = self._atlas_state.detach() == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING
        return torch.logical_and(out_of_anchor, reattach_failed)

    def _build_prune_priority(self, dead_mask: torch.Tensor, soft_mask: torch.Tensor):
        opacity = self.get_opacity.detach().squeeze(-1).clamp(0.0, 1.0)
        priority = (1.0 - opacity) * 4.0
        if self.has_atlas_bindings:
            visibility = self._atlas_visibility_ema.detach().clamp(0.0, 1.0)
            gc_fail = self._atlas_gc_fail_count.detach().to(dtype=torch.float32)
            drift_flag = self._atlas_drift_flag.detach().to(dtype=torch.float32)
            out_of_anchor = (self._atlas_state.detach() == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING).to(dtype=torch.float32)
            priority = priority + (1.0 - visibility) * 2.0 + gc_fail * 1.5 + drift_flag * 0.5 + out_of_anchor
        priority = priority + dead_mask.to(dtype=priority.dtype) * 4.0 + soft_mask.to(dtype=priority.dtype)
        return priority

    def _limit_prune_mask(self, prune_mask: torch.Tensor, min_points_to_keep: int, prune_priority: torch.Tensor):
        point_count = int(self.get_xyz.shape[0])
        min_points_to_keep = max(int(min_points_to_keep), 0)
        if point_count <= min_points_to_keep:
            return torch.zeros_like(prune_mask)

        candidate_idx = torch.nonzero(prune_mask, as_tuple=False).squeeze(-1)
        if candidate_idx.numel() == 0:
            return prune_mask

        max_prunable = max(point_count - min_points_to_keep, 0)
        if int(candidate_idx.shape[0]) <= max_prunable:
            return prune_mask
        if max_prunable <= 0:
            return torch.zeros_like(prune_mask)

        priority = prune_priority[candidate_idx]
        keep_idx = candidate_idx[torch.argsort(priority, descending=True)[:max_prunable]]
        limited_mask = torch.zeros_like(prune_mask)
        limited_mask[keep_idx] = True
        return limited_mask

    def prune_points(self, mask):
        if mask is None:
            return 0
        mask = mask.to(device=self._device(), dtype=torch.bool)
        if mask.numel() == 0 or not torch.any(mask):
            return 0

        pruned_count = int(mask.sum().item())
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._center_log_sigma_parallel = optimizable_tensors["center_sigma_parallel"]
        self._center_log_sigma_support = optimizable_tensors["center_sigma_support"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        tmp_radii = getattr(self, "tmp_radii", None)
        if tmp_radii is None or tmp_radii.shape[0] != valid_points_mask.shape[0]:
            self.tmp_radii = torch.zeros(
                (valid_points_mask.shape[0],),
                dtype=self.max_radii2D.dtype,
                device=self.max_radii2D.device,
            )
        self.tmp_radii = self.tmp_radii[valid_points_mask]
        self._prune_atlas_bindings(valid_points_mask)
        return pruned_count

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_center_sigma_parallel,
        new_center_sigma_support,
        new_tmp_radii,
        new_atlas_node_ids=None,
        new_atlas_states=None,
        new_photo_ema=None,
        new_visibility_ema=None,
        new_high_residual_count=None,
        new_low_residual_count=None,
        new_promotion_streak=None,
        new_demotion_streak=None,
        new_last_transition_iter=None,
        new_gc_fail_count=None,
        new_drift_flag=None,
        new_drift_count=None,
        new_state_cooldown=None,
        new_active_lifetime=None,
        new_ref_camera=None,
        new_ref_score=None,
        new_active_provenance=None,
    ):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "center_sigma_parallel": new_center_sigma_parallel,
        "center_sigma_support": new_center_sigma_support}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._center_log_sigma_parallel = optimizable_tensors["center_sigma_parallel"]
        self._center_log_sigma_support = optimizable_tensors["center_sigma_support"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if self._atlas_positions.numel() > 0 and new_atlas_node_ids is not None and new_atlas_states is not None:
            self._append_atlas_bindings(
                new_atlas_node_ids,
                new_atlas_states,
                photo_ema=new_photo_ema,
                visibility_ema=new_visibility_ema,
                high_residual_count=new_high_residual_count,
                low_residual_count=new_low_residual_count,
                promotion_streak=new_promotion_streak,
                demotion_streak=new_demotion_streak,
                last_transition_iter=new_last_transition_iter,
                gc_fail_count=new_gc_fail_count,
                drift_flag=new_drift_flag,
                drift_count=new_drift_count,
                state_cooldown=new_state_cooldown,
                active_lifetime=new_active_lifetime,
                ref_camera=new_ref_camera,
                ref_score=new_ref_score,
                active_provenance=new_active_provenance,
            )

    def _atlas_node_metric_for_gaussians(self, node_metric: torch.Tensor, default_value: float = 0.0):
        point_count = int(self.get_xyz.shape[0])
        values = torch.full((point_count,), float(default_value), dtype=torch.float32, device=self._device())
        if (
            (not self.has_atlas_bindings)
            or node_metric is None
            or node_metric.numel() == 0
            or self._atlas_node_ids.numel() != point_count
        ):
            return values
        node_metric = node_metric.detach().to(device=values.device, dtype=torch.float32).reshape(-1)
        valid_nodes = (self._atlas_node_ids >= 0) & (self._atlas_node_ids < int(node_metric.shape[0]))
        if torch.any(valid_nodes):
            values[valid_nodes] = node_metric.index_select(0, self._atlas_node_ids[valid_nodes])
        return values

    def compute_stable_split_candidates(
        self,
        grads,
        grad_threshold,
        scene_extent,
        stable_residual_threshold: float = 0.03,
        min_opacity: float = 0.005,
        allowed_mask=None,
        budget_override: int | None = None,
    ):
        point_count = int(self.get_xyz.shape[0])
        device = self._device()
        empty_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if not self.has_atlas_bindings or point_count == 0:
            return empty_mask, {"stable_split_candidate_count": 0.0}

        grad_norm = self._padded_grad_norm(grads)
        grad_gate = grad_norm >= float(grad_threshold)
        allowed_mask = self._pad_selection_mask(allowed_mask, point_count)
        stable = self._atlas_state.detach() == GAUSSIAN_STATE_STABLE
        if allowed_mask is not None:
            stable = stable & allowed_mask[:point_count]

        opacity = self.get_opacity.detach().squeeze(-1).clamp(0.0, 1.0)
        visibility = self._atlas_visibility_ema.detach().clamp(0.0, 1.0)
        photo = self._atlas_photo_ema.detach().clamp_min(0.0)
        high_residual = self._atlas_high_residual_count.detach()
        persistent_residual = (high_residual >= 1) | (photo >= float(stable_residual_threshold) * 0.75)

        support_score = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
        support_inconsistency = (1.0 - support_score).clamp(0.0, 1.0)
        node_obs = self._atlas_node_metric_for_gaussians(self._atlas_node_observed_score_ema, default_value=1.0).clamp(0.0, 1.0)
        node_coverage = self._atlas_node_metric_for_gaussians(self._atlas_refresh_node_coverage_ratio, default_value=1.0).clamp(0.0, 1.0)
        radius = self.get_gaussian_atlas_radius.detach().clamp_min(1e-6)
        sigma_parallel = self.get_center_sigma_parallel.detach().reshape(point_count, -1).max(dim=1).values
        sigma_support = self.get_center_sigma_support.detach().reshape(point_count, -1).max(dim=1).values
        ambiguity_ratio = torch.maximum(sigma_parallel, sigma_support) / radius
        support_ambiguity = (
            (ambiguity_ratio >= 0.55)
            | (support_inconsistency >= 0.22)
            | ((sigma_support / radius) >= 0.55)
        )
        support_ready = support_score >= 0.28
        coverage_not_thin = (
            (node_obs >= 0.12)
            | (node_coverage >= 0.18)
            | (visibility >= 0.015)
            | (self._atlas_ref_score.detach() >= 0.08)
        )
        fidelity_refine_signal = (
            (grad_gate & support_ambiguity & coverage_not_thin)
            | (
                (photo >= float(stable_residual_threshold) * 0.55)
                & support_ambiguity
                & support_ready
            )
        )

        support = self.get_gaussian_atlas_support.detach()
        support_valid = (
            torch.isfinite(support).reshape(point_count, -1).all(dim=1)
            & torch.isfinite(radius)
            & (radius > 0.0)
            & torch.isfinite(self._xyz.detach()).all(dim=1)
        )
        live_enough = opacity >= float(max(min_opacity, 1e-4))
        visible_enough = (
            (visibility >= 0.01)
            | (self._atlas_ref_camera.detach() >= 0)
            | (self._atlas_ref_score.detach() >= 0.05)
        )
        drift_heavy = (
            (self._atlas_state.detach() == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING)
            | (self._atlas_drift_count.detach() >= 2)
            | (self._atlas_gc_fail_count.detach() > 0)
        )
        max_scale = self.get_scaling.detach().max(dim=1).values
        size_signal = (max_scale > float(self.percent_dense) * float(scene_extent) * 0.50) | (ambiguity_ratio >= 0.95)
        candidate_prebudget = (
            stable
            & support_valid
            & live_enough
            & visible_enough
            & (~drift_heavy)
            & support_ready
            & coverage_not_thin
            & support_ambiguity
            & (persistent_residual | fidelity_refine_signal)
            & size_signal
            & (grad_gate | (high_residual >= 2) | (photo >= float(stable_residual_threshold)) | fidelity_refine_signal)
        )

        default_budget = int(min(max(64, int(point_count * 0.04)), 2048))
        budget = default_budget
        if budget_override is not None:
            budget = int(min(default_budget, max(int(budget_override), 0)))
        score = (
            grad_norm / max(float(grad_threshold), 1e-8)
            + (photo / max(float(stable_residual_threshold), 1e-4)).clamp(0.0, 4.0)
            + support_inconsistency
            + ambiguity_ratio.clamp(0.0, 4.0)
            + 0.25 * opacity
            + 0.15 * visibility
        )
        selected = self._select_topk_mask(candidate_prebudget, score, budget)
        metrics = {
            "stable_split_candidate_count": float(selected.sum().item()),
            "stable_split_candidate_prebudget_count": float(candidate_prebudget.sum().item()),
            "stable_split_budget": float(budget),
            "stable_split_default_budget": float(default_budget),
            "stable_split_grad_signal_count": float((stable & grad_gate).sum().item()),
            "stable_split_residual_signal_count": float((stable & persistent_residual).sum().item()),
            "stable_split_fidelity_refine_signal_count": float((stable & fidelity_refine_signal).sum().item()),
            "stable_split_ambiguity_signal_count": float((stable & support_ambiguity).sum().item()),
            "stable_split_support_ready_count": float((stable & support_ready).sum().item()),
            "stable_split_coverage_not_thin_count": float((stable & coverage_not_thin).sum().item()),
            "stable_split_block_drift_count": float((stable & drift_heavy).sum().item()),
            "stable_split_block_projector_count": float((stable & (~support_valid)).sum().item()),
        }
        return selected, metrics

    def compute_stable_clone_candidates(
        self,
        grads,
        grad_threshold,
        scene_extent,
        stable_residual_threshold: float = 0.03,
        min_opacity: float = 0.005,
        allowed_mask=None,
        budget_override: int | None = None,
    ):
        point_count = int(self.get_xyz.shape[0])
        device = self._device()
        empty_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if not self.has_atlas_bindings or point_count == 0:
            return empty_mask, {"stable_clone_candidate_count": 0.0}

        grad_norm = self._padded_grad_norm(grads)
        grad_gate = grad_norm >= float(grad_threshold) * 0.75
        allowed_mask = self._pad_selection_mask(allowed_mask, point_count)
        stable = self._atlas_state.detach() == GAUSSIAN_STATE_STABLE
        if allowed_mask is not None:
            stable = stable & allowed_mask[:point_count]

        opacity = self.get_opacity.detach().squeeze(-1).clamp(0.0, 1.0)
        visibility = self._atlas_visibility_ema.detach().clamp(0.0, 1.0)
        photo = self._atlas_photo_ema.detach().clamp_min(0.0)
        high_residual = self._atlas_high_residual_count.detach()
        residual_persists = (
            (photo >= float(stable_residual_threshold) * 0.70)
            | (high_residual >= 2)
            | (grad_gate & (photo >= float(stable_residual_threshold) * 0.35))
        )

        node_obs = self._atlas_node_metric_for_gaussians(self._atlas_node_observed_score_ema, default_value=1.0).clamp(0.0, 1.0)
        node_vis = self._atlas_node_metric_for_gaussians(self._atlas_node_visibility_ema, default_value=1.0).clamp(0.0, 1.0)
        node_coverage = self._atlas_node_metric_for_gaussians(self._atlas_refresh_node_coverage_ratio, default_value=1.0).clamp(0.0, 1.0)
        ref_score = self._atlas_ref_score.detach().clamp(0.0, 1.0)
        coverage_low = (
            (node_coverage <= 0.18)
            | ((node_obs <= 0.22) & (node_vis <= 0.10))
            | ((visibility <= 0.025) & (node_obs <= 0.35) & (node_vis <= 0.12))
        )
        clone_signal = residual_persists | (grad_gate & coverage_low)

        support_score = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
        support = self.get_gaussian_atlas_support.detach()
        radius = self.get_gaussian_atlas_radius.detach().clamp_min(1e-6)
        support_valid = (
            torch.isfinite(support).reshape(point_count, -1).all(dim=1)
            & torch.isfinite(radius)
            & (radius > 0.0)
            & torch.isfinite(self._xyz.detach()).all(dim=1)
            & (support_score >= 0.38)
        )
        delta = self._xyz.detach() - self.get_gaussian_atlas_positions.detach()
        projected_delta = torch.bmm(support, delta.unsqueeze(-1)).squeeze(-1)
        projected_energy = torch.sum(projected_delta * delta, dim=1).clamp_min(0.0)
        projected_drift = torch.sqrt(torch.where(projected_energy > 1e-12, projected_energy, torch.zeros_like(projected_energy)))
        normalized_projected_drift = projected_drift / radius.clamp_min(1e-6)
        projected_drift_small = torch.isfinite(normalized_projected_drift) & (normalized_projected_drift <= 0.75)
        thin_background_support = (
            ((ref_score >= 0.06) | (node_obs >= 0.16))
            & (visibility <= 0.035)
            & (node_coverage <= 0.32)
            & (support_score >= 0.34)
            & projected_drift_small
        )
        coverage_low = coverage_low | thin_background_support
        clone_signal = clone_signal | (
            thin_background_support
            & (
                grad_gate
                | (photo >= float(stable_residual_threshold) * 0.35)
                | (node_obs >= 0.24)
            )
        )
        live_enough = opacity >= float(max(min_opacity * 2.0, 0.01))
        ref_or_visible = (
            (self._atlas_ref_camera.detach() >= 0)
            | (self._atlas_ref_score.detach() >= 0.08)
            | (visibility >= 0.006)
            | (node_obs >= 0.12)
        )
        state_update_iter = int(max(getattr(self, "_atlas_state_update_iter", 0), 0))
        recent_transition = self._atlas_state_cooldown.detach() > 0
        if self._atlas_last_transition_iter.numel() == point_count:
            recent_transition = recent_transition | (
                self._atlas_last_transition_iter.detach() >= max(state_update_iter - 2, 0)
            )
        drift_heavy = (
            (self._atlas_state.detach() == GAUSSIAN_STATE_OUT_OF_ANCHOR_PENDING)
            | (self._atlas_drift_count.detach() >= 2)
            | (self._atlas_gc_fail_count.detach() > 0)
        )
        max_scale = self.get_scaling.detach().max(dim=1).values
        local_size_ok = (max_scale <= float(self.percent_dense) * float(scene_extent) * 1.25) | coverage_low
        candidate_prebudget = (
            stable
            & support_valid
            & live_enough
            & ref_or_visible
            & (~drift_heavy)
            & (~recent_transition)
            & coverage_low
            & clone_signal
            & projected_drift_small
            & local_size_ok
        )

        default_budget = int(min(max(64, int(point_count * 0.04)), 2048))
        budget = default_budget
        if budget_override is not None:
            budget = int(min(default_budget, max(int(budget_override), 0)))
        score = (
            grad_norm / max(float(grad_threshold), 1e-8)
            + (photo / max(float(stable_residual_threshold), 1e-4)).clamp(0.0, 4.0)
            + (1.0 - visibility).clamp(0.0, 1.0)
            + (1.0 - node_obs).clamp(0.0, 1.0)
            + (1.0 - node_coverage).clamp(0.0, 1.0)
            + (1.0 - normalized_projected_drift.clamp(0.0, 1.0))
            + 0.25 * opacity
        )
        selected = self._select_topk_mask(candidate_prebudget, score, budget)
        metrics = {
            "stable_clone_candidate_count": float(selected.sum().item()),
            "stable_clone_candidate_prebudget_count": float(candidate_prebudget.sum().item()),
            "stable_clone_budget": float(budget),
            "stable_clone_default_budget": float(default_budget),
            "stable_clone_coverage_low_count": float((stable & coverage_low).sum().item()),
            "stable_clone_thin_background_support_count": float((stable & thin_background_support).sum().item()),
            "stable_clone_residual_signal_count": float((stable & residual_persists).sum().item()),
            "stable_clone_grad_thin_signal_count": float((stable & grad_gate & coverage_low).sum().item()),
            "stable_clone_projected_drift_small_count": float((stable & projected_drift_small).sum().item()),
            "stable_clone_recent_transition_block_count": float((stable & recent_transition).sum().item()),
            "stable_clone_block_pose_ref_count": float((stable & (~ref_or_visible)).sum().item()),
            "stable_clone_block_projector_count": float((stable & (~support_valid)).sum().item()),
            "stable_clone_block_drift_count": float((stable & drift_heavy).sum().item()),
        }
        return selected, metrics

    def compute_active_explore_clone_candidates(
        self,
        grads,
        grad_threshold,
        stable_residual_threshold: float = 0.03,
        min_opacity: float = 0.005,
        allowed_mask=None,
        budget_override: int | None = None,
    ):
        point_count = int(self.get_xyz.shape[0])
        device = self._device()
        empty_mask = torch.zeros((point_count,), dtype=torch.bool, device=device)
        if not self.has_atlas_bindings or point_count == 0:
            return empty_mask, {"explore_candidate_count": 0.0}

        grad_norm = self._padded_grad_norm(grads)
        gradient_signal = grad_norm >= float(grad_threshold)
        allowed_mask = self._pad_selection_mask(allowed_mask, point_count)
        active = self._atlas_state.detach() == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        if allowed_mask is not None:
            active = active & allowed_mask[:point_count]

        opacity = self.get_opacity.detach().squeeze(-1).clamp(0.0, 1.0)
        live_opacity = opacity >= float(max(min_opacity * 0.05, 1e-5))
        finite_center = torch.isfinite(self._xyz.detach()).all(dim=1)
        structural = active & live_opacity & finite_center
        photo = self._atlas_photo_ema.detach().clamp_min(0.0)
        visibility = self._atlas_visibility_ema.detach().clamp(0.0, 1.0)
        high_residual = self._atlas_high_residual_count.detach()
        support_score = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
        support_inconsistency = (1.0 - support_score).clamp(0.0, 1.0)
        node_ref_consistency = self._atlas_node_metric_for_gaussians(
            self._atlas_node_ref_consistency_ema,
            default_value=0.0,
        ).clamp(0.0, 1.0)
        node_observed_score = self._atlas_node_metric_for_gaussians(
            self._atlas_node_observed_score_ema,
            default_value=0.0,
        ).clamp(0.0, 1.0)
        atlas_radius = self.get_gaussian_atlas_radius.detach().clamp_min(1e-6)
        anchor_distance_ratio = (
            torch.linalg.norm(self._xyz.detach() - self.get_gaussian_atlas_positions.detach(), dim=1)
            / atlas_radius
        ).clamp_min(0.0)
        view_evidence = self.get_gaussian_atlas_view_weights.detach()
        if (
            view_evidence.ndim == 2
            and view_evidence.shape[0] == point_count
            and view_evidence.shape[1] > 0
        ):
            view_evidence_score = view_evidence.clamp_min(0.0).max(dim=1).values.clamp(0.0, 1.0)
        else:
            view_evidence_score = torch.zeros((point_count,), dtype=photo.dtype, device=device)
        ref_ready = (
            (self._atlas_ref_camera.detach() >= 0)
            | (self._atlas_ref_score.detach() >= 0.05)
            | (view_evidence_score > 1e-6)
        )
        slab_evidence = ref_ready | (visibility >= 0.003)
        residual_persistent = (high_residual >= 1) | (photo >= float(stable_residual_threshold) * 0.75)
        gradient_rescue_signal = structural & gradient_signal & slab_evidence & ref_ready
        rescue_signal = structural & (residual_persistent | gradient_rescue_signal) & slab_evidence
        support_rescue = rescue_signal & (support_inconsistency >= 0.18)
        low_view_support = (
            (visibility < 0.006)
            & (view_evidence_score < 0.02)
            & (self._atlas_ref_score.detach() < 0.08)
            & (node_observed_score < 0.10)
        )
        low_texture_signal = grad_norm < float(grad_threshold) * 0.55
        weak_ref_consistency = (node_ref_consistency < 0.20) & (self._atlas_ref_score.detach() < 0.08)
        weak_geometric_residual = (
            (high_residual < 2)
            & (photo < float(stable_residual_threshold) * 1.25)
            & (~gradient_signal)
        )
        support_only_conflict = (support_inconsistency >= 0.35) & weak_geometric_residual
        far_anchor_weak = (anchor_distance_ratio > 1.75) & low_texture_signal & weak_geometric_residual
        background_like_ray = structural & (
            (low_view_support & low_texture_signal & weak_ref_consistency)
            | support_only_conflict
            | far_anchor_weak
        )
        hard_rescue_signal = (
            (high_residual >= 3)
            | (photo >= float(stable_residual_threshold) * 1.75)
            | (gradient_signal & (photo >= float(stable_residual_threshold)))
        )
        admissible_explore_ray = (~background_like_ray) | hard_rescue_signal
        candidate_prebudget = (
            structural
            & slab_evidence
            & (residual_persistent | gradient_rescue_signal)
            & admissible_explore_ray
            & (gradient_signal | support_rescue | (support_inconsistency >= 0.22) | (high_residual >= 2))
        )

        default_budget = int(min(max(16, int(structural.sum().item())), 256))
        budget = default_budget
        if budget_override is not None:
            budget = int(min(default_budget, max(int(budget_override), 0)))
        score = (
            grad_norm / max(float(grad_threshold), 1e-8)
            + (photo / max(float(stable_residual_threshold), 1e-4)).clamp(0.0, 4.0)
            + (high_residual.to(dtype=photo.dtype) / 3.0).clamp(0.0, 1.0)
            + support_inconsistency
            + 0.50 * self._atlas_ref_score.detach().clamp(0.0, 1.0)
            + 0.35 * view_evidence_score
            + 0.25 * visibility
            + 0.25 * opacity
            - 1.25 * background_like_ray.to(dtype=photo.dtype)
            - 0.35 * (self._atlas_active_lifetime.detach().to(dtype=photo.dtype) / 256.0).clamp(0.0, 1.0)
        )
        selected = self._select_topk_mask(candidate_prebudget, score, budget)
        metrics = {
            "explore_candidate_count": float(selected.sum().item()),
            "explore_candidate_prebudget_count": float(candidate_prebudget.sum().item()),
            "explore_budget": float(budget),
            "explore_default_budget": float(default_budget),
            "explore_live_active_count": float(structural.sum().item()),
            "explore_gradient_signal_count": float((structural & gradient_signal).sum().item()),
            "explore_gradient_rescue_signal_count": float(gradient_rescue_signal.sum().item()),
            "explore_rescue_signal_count": float(rescue_signal.sum().item()),
            "explore_support_rescue_count": float(support_rescue.sum().item()),
            "explore_score_fallback_count": float((selected & (~gradient_signal)).sum().item()),
            "explore_low_opacity_hard_block_count": float((active & (~live_opacity)).sum().item()),
            "explore_ref_or_visibility_ready_count": float((structural & slab_evidence).sum().item()),
            "explore_view_evidence_ready_count": float((structural & (view_evidence_score > 1e-6)).sum().item()),
            "explore_background_like_block_count": float((structural & background_like_ray & (~hard_rescue_signal)).sum().item()),
            "explore_background_like_selected_count": float((selected & background_like_ray).sum().item()),
            "explore_hard_rescue_signal_count": float((structural & hard_rescue_signal).sum().item()),
            "explore_low_view_support_count": float((structural & low_view_support).sum().item()),
            "explore_support_only_conflict_count": float((structural & support_only_conflict).sum().item()),
        }
        return selected, metrics

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, allowed_mask=None, candidate_mask=None):
        n_init_points = self.get_xyz.shape[0]
        if candidate_mask is not None:
            selected_pts_mask = self._pad_selection_mask(candidate_mask, n_init_points)
        else:
            padded_grad = self._padded_grad_norm(grads)
            selected_pts_mask = torch.where(padded_grad >= float(grad_threshold), True, False)
        allowed_mask = self._pad_selection_mask(allowed_mask, n_init_points)
        if allowed_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, allowed_mask)
        if candidate_mask is None:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent,
            )
        if not torch.any(selected_pts_mask):
            return {
                "split_count": 0.0,
                "split_source_count": 0.0,
                "split_parent_pruned_count": 0.0,
                "split_child_scale_ratio_mean": 0.0,
                "split_child_scale_ratio_max": 0.0,
                "split_child_log_anisotropy_delta_mean": 0.0,
            }

        source_count = int(selected_pts_mask.sum().item())
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self._uniform_rescale_log_scaling(self._scaling[selected_pts_mask].repeat(N, 1), factor=(1.0 / (0.8 * N)))
        child_scales = self.scaling_activation(new_scaling).detach()
        parent_scales = stds.detach()
        child_scale_ratio = child_scales.max(dim=1).values / parent_scales.max(dim=1).values.clamp_min(1e-8)
        child_aniso_delta = (
            self._scale_free_log_anisotropy(child_scales)
            - self._scale_free_log_anisotropy(parent_scales)
        ).abs().mean(dim=1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_center_sigma_parallel = self._center_log_sigma_parallel[selected_pts_mask].repeat(N,1)
        new_center_sigma_support = self._center_log_sigma_support[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_atlas_node_ids = None
        new_atlas_states = None
        new_photo_ema = None
        new_visibility_ema = None
        new_high_residual_count = None
        new_low_residual_count = None
        new_promotion_streak = None
        new_demotion_streak = None
        new_last_transition_iter = None
        new_gc_fail_count = None
        new_drift_flag = None
        new_drift_count = None
        new_state_cooldown = None
        new_active_lifetime = None
        new_ref_camera = None
        new_ref_score = None
        new_active_provenance = None
        if self.has_atlas_bindings:
            new_atlas_node_ids = self._atlas_node_ids[selected_pts_mask].repeat(N)
            new_atlas_states = self._atlas_state[selected_pts_mask].repeat(N)
            new_photo_ema = self._atlas_photo_ema[selected_pts_mask].repeat(N)
            new_visibility_ema = self._atlas_visibility_ema[selected_pts_mask].repeat(N)
            new_high_residual_count = self._atlas_high_residual_count[selected_pts_mask].repeat(N)
            new_low_residual_count = self._atlas_low_residual_count[selected_pts_mask].repeat(N)
            new_promotion_streak = self._atlas_promotion_streak[selected_pts_mask].repeat(N)
            new_demotion_streak = self._atlas_demotion_streak[selected_pts_mask].repeat(N)
            new_last_transition_iter = self._atlas_last_transition_iter[selected_pts_mask].repeat(N)
            new_gc_fail_count = self._atlas_gc_fail_count[selected_pts_mask].repeat(N)
            new_drift_flag = self._atlas_drift_flag[selected_pts_mask].repeat(N)
            new_drift_count = self._atlas_drift_count[selected_pts_mask].repeat(N)
            new_state_cooldown = self._atlas_state_cooldown[selected_pts_mask].repeat(N)
            new_active_lifetime = torch.where(
                self._atlas_state[selected_pts_mask] == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                self._atlas_active_lifetime[selected_pts_mask],
                torch.zeros_like(self._atlas_active_lifetime[selected_pts_mask]),
            ).repeat(N)
            new_ref_camera = self._atlas_ref_camera[selected_pts_mask].repeat(N)
            new_ref_score = self._atlas_ref_score[selected_pts_mask].repeat(N)
            new_active_provenance = self._atlas_active_provenance[selected_pts_mask].repeat(N)

        split_payload, split_nonfinite_discard_count = self._filter_nonfinite_clone_payload(
            {
                "xyz": new_xyz,
                "features_dc": new_features_dc,
                "features_rest": new_features_rest,
                "opacity": new_opacity,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "center_sigma_parallel": new_center_sigma_parallel,
                "center_sigma_support": new_center_sigma_support,
                "tmp_radii": new_tmp_radii,
                "atlas_node_ids": new_atlas_node_ids,
                "atlas_states": new_atlas_states,
                "photo_ema": new_photo_ema,
                "visibility_ema": new_visibility_ema,
                "high_residual_count": new_high_residual_count,
                "low_residual_count": new_low_residual_count,
                "promotion_streak": new_promotion_streak,
                "demotion_streak": new_demotion_streak,
                "last_transition_iter": new_last_transition_iter,
                "gc_fail_count": new_gc_fail_count,
                "drift_flag": new_drift_flag,
                "drift_count": new_drift_count,
                "state_cooldown": new_state_cooldown,
                "active_lifetime": new_active_lifetime,
                "ref_camera": new_ref_camera,
                "ref_score": new_ref_score,
                "active_provenance": new_active_provenance,
            }
        )
        split_count = int(split_payload["xyz"].shape[0]) if split_payload["xyz"] is not None else 0
        parent_pruned_count = 0
        if split_count > 0:
            self.densification_postfix(
                split_payload["xyz"],
                split_payload["features_dc"],
                split_payload["features_rest"],
                split_payload["opacity"],
                split_payload["scaling"],
                split_payload["rotation"],
                split_payload["center_sigma_parallel"],
                split_payload["center_sigma_support"],
                split_payload["tmp_radii"],
                new_atlas_node_ids=split_payload["atlas_node_ids"],
                new_atlas_states=split_payload["atlas_states"],
                new_photo_ema=split_payload["photo_ema"],
                new_visibility_ema=split_payload["visibility_ema"],
                new_high_residual_count=split_payload["high_residual_count"],
                new_low_residual_count=split_payload["low_residual_count"],
                new_promotion_streak=split_payload["promotion_streak"],
                new_demotion_streak=split_payload["demotion_streak"],
                new_last_transition_iter=split_payload["last_transition_iter"],
                new_gc_fail_count=split_payload["gc_fail_count"],
                new_drift_flag=split_payload["drift_flag"],
                new_drift_count=split_payload["drift_count"],
                new_state_cooldown=split_payload["state_cooldown"],
                new_active_lifetime=split_payload["active_lifetime"],
                new_ref_camera=split_payload["ref_camera"],
                new_ref_score=split_payload["ref_score"],
                new_active_provenance=split_payload["active_provenance"],
            )
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(split_count, device=self._device(), dtype=torch.bool)))
            parent_pruned_count = self.prune_points(prune_filter)
        return {
            "split_count": float(split_count),
            "split_source_count": float(source_count),
            "split_parent_pruned_count": float(parent_pruned_count),
            "split_nonfinite_discard_count": float(split_nonfinite_discard_count),
            "split_child_scale_ratio_mean": float(child_scale_ratio.mean().item()) if child_scale_ratio.numel() > 0 else 0.0,
            "split_child_scale_ratio_max": float(child_scale_ratio.max().item()) if child_scale_ratio.numel() > 0 else 0.0,
            "split_child_log_anisotropy_delta_mean": float(child_aniso_delta.mean().item()) if child_aniso_delta.numel() > 0 else 0.0,
        }

    def densify_and_clone(
        self,
        grads,
        grad_threshold,
        scene_extent,
        allowed_mask=None,
        candidate_mask=None,
        support_projected_jitter: bool = False,
        support_jitter_scale: float = 0.35,
    ):
        if candidate_mask is not None:
            selected_pts_mask = self._pad_selection_mask(candidate_mask, self.get_xyz.shape[0])
        else:
            grad_norm = self._padded_grad_norm(grads)
            selected_pts_mask = torch.where(grad_norm >= float(grad_threshold), True, False)
        allowed_mask = self._pad_selection_mask(allowed_mask, selected_pts_mask.shape[0])
        if allowed_mask is not None:
            selected_pts_mask = torch.logical_and(selected_pts_mask, allowed_mask[: selected_pts_mask.shape[0]])
        if candidate_mask is None:
            selected_pts_mask = torch.logical_and(
                selected_pts_mask,
                torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent,
            )
        if not torch.any(selected_pts_mask):
            return {"clone_count": 0.0}

        candidate_clone_count = int(selected_pts_mask.sum().item())
        new_xyz = self._xyz[selected_pts_mask]
        if support_projected_jitter and self.has_atlas_bindings:
            support = self.get_gaussian_atlas_support.detach()[selected_pts_mask]
            noise = torch.randn_like(new_xyz)
            support_noise = torch.bmm(support, noise.unsqueeze(-1)).squeeze(-1)
            support_norm = torch.linalg.norm(support_noise, dim=1, keepdim=True).clamp_min(1e-6)
            support_dir = support_noise / support_norm
            radius = self.get_gaussian_atlas_radius.detach()[selected_pts_mask].clamp_min(1e-6)
            max_scale = self.get_scaling.detach()[selected_pts_mask].max(dim=1).values.clamp_min(1e-6)
            step = torch.minimum(
                radius * float(max(support_jitter_scale, 0.0)),
                torch.maximum(max_scale * 0.50, radius * 0.05),
            )
            offset = support_dir * step[:, None]
            offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
            new_xyz = new_xyz + offset
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_center_sigma_parallel = self._center_log_sigma_parallel[selected_pts_mask]
        new_center_sigma_support = self._center_log_sigma_support[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_atlas_node_ids = None
        new_atlas_states = None
        new_photo_ema = None
        new_visibility_ema = None
        new_high_residual_count = None
        new_low_residual_count = None
        new_promotion_streak = None
        new_demotion_streak = None
        new_last_transition_iter = None
        new_gc_fail_count = None
        new_drift_flag = None
        new_drift_count = None
        new_state_cooldown = None
        new_active_lifetime = None
        new_ref_camera = None
        new_ref_score = None
        new_active_provenance = None
        if self.has_atlas_bindings:
            new_atlas_node_ids = self._atlas_node_ids[selected_pts_mask]
            new_atlas_states = self._atlas_state[selected_pts_mask]
            new_photo_ema = self._atlas_photo_ema[selected_pts_mask]
            new_visibility_ema = self._atlas_visibility_ema[selected_pts_mask]
            new_high_residual_count = self._atlas_high_residual_count[selected_pts_mask]
            new_low_residual_count = self._atlas_low_residual_count[selected_pts_mask]
            new_promotion_streak = self._atlas_promotion_streak[selected_pts_mask]
            new_demotion_streak = self._atlas_demotion_streak[selected_pts_mask]
            new_last_transition_iter = self._atlas_last_transition_iter[selected_pts_mask]
            new_gc_fail_count = self._atlas_gc_fail_count[selected_pts_mask]
            new_drift_flag = self._atlas_drift_flag[selected_pts_mask]
            new_drift_count = self._atlas_drift_count[selected_pts_mask]
            new_state_cooldown = self._atlas_state_cooldown[selected_pts_mask]
            new_active_lifetime = torch.where(
                self._atlas_state[selected_pts_mask] == GAUSSIAN_STATE_UNSTABLE_ACTIVE,
                self._atlas_active_lifetime[selected_pts_mask],
                torch.zeros_like(self._atlas_active_lifetime[selected_pts_mask]),
            )
            new_ref_camera = self._atlas_ref_camera[selected_pts_mask]
            new_ref_score = self._atlas_ref_score[selected_pts_mask]
            new_active_provenance = self._atlas_active_provenance[selected_pts_mask]

        clone_payload, clone_nonfinite_discard_count = self._filter_nonfinite_clone_payload(
            {
                "xyz": new_xyz,
                "features_dc": new_features_dc,
                "features_rest": new_features_rest,
                "opacity": new_opacities,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "center_sigma_parallel": new_center_sigma_parallel,
                "center_sigma_support": new_center_sigma_support,
                "tmp_radii": new_tmp_radii,
                "atlas_node_ids": new_atlas_node_ids,
                "atlas_states": new_atlas_states,
                "photo_ema": new_photo_ema,
                "visibility_ema": new_visibility_ema,
                "high_residual_count": new_high_residual_count,
                "low_residual_count": new_low_residual_count,
                "promotion_streak": new_promotion_streak,
                "demotion_streak": new_demotion_streak,
                "last_transition_iter": new_last_transition_iter,
                "gc_fail_count": new_gc_fail_count,
                "drift_flag": new_drift_flag,
                "drift_count": new_drift_count,
                "state_cooldown": new_state_cooldown,
                "active_lifetime": new_active_lifetime,
                "ref_camera": new_ref_camera,
                "ref_score": new_ref_score,
                "active_provenance": new_active_provenance,
            }
        )
        clone_count = int(clone_payload["xyz"].shape[0]) if clone_payload["xyz"] is not None else 0
        if clone_count == 0:
            return {
                "clone_count": 0.0,
                "clone_candidate_count": float(candidate_clone_count),
                "clone_nonfinite_discard_count": float(clone_nonfinite_discard_count),
            }

        self.densification_postfix(
            clone_payload["xyz"],
            clone_payload["features_dc"],
            clone_payload["features_rest"],
            clone_payload["opacity"],
            clone_payload["scaling"],
            clone_payload["rotation"],
            clone_payload["center_sigma_parallel"],
            clone_payload["center_sigma_support"],
            clone_payload["tmp_radii"],
            new_atlas_node_ids=clone_payload["atlas_node_ids"],
            new_atlas_states=clone_payload["atlas_states"],
            new_photo_ema=clone_payload["photo_ema"],
            new_visibility_ema=clone_payload["visibility_ema"],
            new_high_residual_count=clone_payload["high_residual_count"],
            new_low_residual_count=clone_payload["low_residual_count"],
            new_promotion_streak=clone_payload["promotion_streak"],
            new_demotion_streak=clone_payload["demotion_streak"],
            new_last_transition_iter=clone_payload["last_transition_iter"],
            new_gc_fail_count=clone_payload["gc_fail_count"],
            new_drift_flag=clone_payload["drift_flag"],
            new_drift_count=clone_payload["drift_count"],
            new_state_cooldown=clone_payload["state_cooldown"],
            new_active_lifetime=clone_payload["active_lifetime"],
            new_ref_camera=clone_payload["ref_camera"],
            new_ref_score=clone_payload["ref_score"],
            new_active_provenance=clone_payload["active_provenance"],
        )
        return {
            "clone_count": float(clone_count),
            "clone_candidate_count": float(candidate_clone_count),
            "clone_nonfinite_discard_count": float(clone_nonfinite_discard_count),
        }

    def explore_and_clone(
        self,
        grads,
        grad_threshold,
        camera_center,
        slab_radius_mult: float,
        jitter_scale: float,
        allowed_mask=None,
        all_camera_centers=None,
        active_min_lifetime_iters: int = 5,
        min_opacity: float = 0.005,
        stable_residual_threshold: float = 0.03,
        candidate_mask=None,
        candidate_metrics: dict | None = None,
    ):
        if not self.has_atlas_bindings:
            return {"explore_clone_count": 0.0, "explore_candidate_count": 0.0, "explore_valid_ref_count": 0.0}

        if candidate_mask is None:
            selected_pts_mask, candidate_metrics = self.compute_active_explore_clone_candidates(
                grads,
                grad_threshold,
                stable_residual_threshold=stable_residual_threshold,
                min_opacity=min_opacity,
                allowed_mask=allowed_mask,
            )
        else:
            selected_pts_mask = self._pad_selection_mask(candidate_mask, int(self.get_xyz.shape[0]))
            if allowed_mask is not None:
                selected_pts_mask = selected_pts_mask & self._pad_selection_mask(allowed_mask, int(self.get_xyz.shape[0]))
            candidate_metrics = dict(candidate_metrics or {})
        explore_candidate_count = int(selected_pts_mask.sum().item())
        explore_score_fallback_count = float(candidate_metrics.get("explore_score_fallback_count", 0.0))
        explore_rescue_signal_count = float(candidate_metrics.get("explore_rescue_signal_count", 0.0))
        explore_live_active_count = float(candidate_metrics.get("explore_live_active_count", 0.0))
        slab_admission_candidate_count = 0
        slab_admission_valid_count = 0
        slab_admission_added_count = 0
        slab_admission_repair_count = 0
        slab_admission_fallback_count = 0
        point_count = int(self.get_xyz.shape[0])
        if point_count > 0:
            active_allowed = self._atlas_state.detach() == GAUSSIAN_STATE_UNSTABLE_ACTIVE
            if allowed_mask is not None:
                active_allowed = active_allowed & self._pad_selection_mask(allowed_mask, point_count)
            opacity = self.get_opacity.detach().squeeze(-1).clamp(0.0, 1.0)
            live_opacity = opacity >= float(max(min_opacity * 0.05, 1e-5))
            finite_center = torch.isfinite(self._xyz.detach()).all(dim=1)
            photo = self._atlas_photo_ema.detach().clamp_min(0.0)
            visibility = self._atlas_visibility_ema.detach().clamp(0.0, 1.0)
            high_residual = self._atlas_high_residual_count.detach()
            residual_persistent = (high_residual >= 1) | (photo >= float(stable_residual_threshold) * 0.50)
            support_score = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
            support_inconsistency = (1.0 - support_score).clamp(0.0, 1.0)
            node_ref_consistency = self._atlas_node_metric_for_gaussians(
                self._atlas_node_ref_consistency_ema,
                default_value=0.0,
            ).clamp(0.0, 1.0)
            node_observed_score = self._atlas_node_metric_for_gaussians(
                self._atlas_node_observed_score_ema,
                default_value=0.0,
            ).clamp(0.0, 1.0)
            atlas_radius = self.get_gaussian_atlas_radius.detach().clamp_min(1e-6)
            anchor_distance_ratio = (
                torch.linalg.norm(self._xyz.detach() - self.get_gaussian_atlas_positions.detach(), dim=1)
                / atlas_radius
            ).clamp_min(0.0)
            view_evidence = self.get_gaussian_atlas_view_weights.detach()
            if (
                view_evidence.ndim == 2
                and view_evidence.shape[0] == point_count
                and view_evidence.shape[1] > 0
            ):
                view_evidence_score = view_evidence.clamp_min(0.0).max(dim=1).values.clamp(0.0, 1.0)
            else:
                view_evidence_score = torch.zeros((point_count,), dtype=photo.dtype, device=self._device())
            grad_norm = self._padded_grad_norm(grads)
            admission_score = (
                grad_norm / max(float(grad_threshold), 1e-8)
                + (photo / max(float(stable_residual_threshold), 1e-4)).clamp(0.0, 4.0)
                + (high_residual.to(dtype=photo.dtype) / 3.0).clamp(0.0, 1.0)
                + support_inconsistency
                + 0.50 * self._atlas_ref_score.detach().clamp(0.0, 1.0)
                + 0.35 * view_evidence_score
                + 0.25 * visibility
                + 0.25 * opacity
            )
            structural = active_allowed & live_opacity & finite_center
            low_view_support = (
                (visibility < 0.006)
                & (view_evidence_score < 0.02)
                & (self._atlas_ref_score.detach() < 0.08)
                & (node_observed_score < 0.10)
            )
            low_texture_signal = grad_norm < float(grad_threshold) * 0.55
            weak_ref_consistency = (node_ref_consistency < 0.20) & (self._atlas_ref_score.detach() < 0.08)
            weak_geometric_residual = (
                (high_residual < 2)
                & (photo < float(stable_residual_threshold) * 1.25)
                & (~(grad_norm >= float(grad_threshold)))
            )
            support_only_conflict = (support_inconsistency >= 0.35) & weak_geometric_residual
            far_anchor_weak = (anchor_distance_ratio > 1.75) & low_texture_signal & weak_geometric_residual
            background_like_ray = structural & (
                (low_view_support & low_texture_signal & weak_ref_consistency)
                | support_only_conflict
                | far_anchor_weak
            )
            hard_rescue_signal = (
                (high_residual >= 3)
                | (photo >= float(stable_residual_threshold) * 1.75)
                | ((grad_norm >= float(grad_threshold)) & (photo >= float(stable_residual_threshold)))
            )
            admission_score = admission_score - 1.25 * background_like_ray.to(dtype=admission_score.dtype)
            slab_admission_probe = (
                structural
                & residual_persistent
                & ((~background_like_ray) | hard_rescue_signal)
                & (admission_score >= 0.75)
            )
            slab_admission_candidate_count = int(slab_admission_probe.sum().item())
            if torch.any(slab_admission_probe):
                admission_slab = compute_point_slab_bounds(
                    self,
                    slab_admission_probe,
                    camera_centers=all_camera_centers,
                    fallback_camera_center=camera_center,
                    slab_radius_mult=float(slab_radius_mult),
                    detach_points=True,
                    require_valid_ref_camera=True,
                    min_reference_score=0.05,
                    repair_ref_camera=True,
                )
                if admission_slab is not None:
                    slab_valid_indices = admission_slab["point_indices"]
                    slab_valid_mask = torch.zeros((point_count,), dtype=torch.bool, device=self._device())
                    slab_valid_mask[slab_valid_indices] = True
                    slab_admission_valid_count = int(slab_valid_indices.numel())
                    slab_admission_repair_count = int(admission_slab["repaired_ref_mask"].sum().item())
                    slab_admission_fallback_count = int(admission_slab["fallback_slab_mask"].sum().item())
                    admission_budget = int(min(max(16, int(structural.sum().item())), 256))
                    slab_admitted = self._select_topk_mask(
                        slab_valid_mask & slab_admission_probe,
                        admission_score,
                        admission_budget,
                    )
                    added_mask = slab_admitted & (~selected_pts_mask)
                    slab_admission_added_count = int(added_mask.sum().item())
                    selected_pts_mask = selected_pts_mask | slab_admitted
                    candidate_metrics["explore_candidate_count"] = float(selected_pts_mask.sum().item())
            candidate_metrics["explore_slab_admission_background_like_block_count"] = float(
                (structural & background_like_ray & (~hard_rescue_signal)).sum().item()
            )
        candidate_metrics["explore_slab_admission_candidate_count"] = float(slab_admission_candidate_count)
        candidate_metrics["explore_slab_admission_valid_count"] = float(slab_admission_valid_count)
        candidate_metrics["explore_slab_admission_added_count"] = float(slab_admission_added_count)
        candidate_metrics["explore_slab_admission_ref_repair_count"] = float(slab_admission_repair_count)
        candidate_metrics["explore_slab_admission_fallback_count"] = float(slab_admission_fallback_count)
        explore_candidate_count = int(selected_pts_mask.sum().item())
        if not torch.any(selected_pts_mask):
            result = {
                "explore_clone_count": 0.0,
                "explore_candidate_count": float(explore_candidate_count),
                "explore_valid_ref_count": 0.0,
                "explore_score_fallback_count": float(explore_score_fallback_count),
                "explore_rescue_signal_count": float(explore_rescue_signal_count),
                "explore_live_active_count": float(explore_live_active_count),
            }
            result.update(candidate_metrics)
            return result

        slab = compute_point_slab_bounds(
            self,
            selected_pts_mask,
            camera_centers=all_camera_centers,
            fallback_camera_center=camera_center,
            slab_radius_mult=float(slab_radius_mult),
            detach_points=True,
            require_valid_ref_camera=True,
            min_reference_score=0.05,
            repair_ref_camera=True,
        )
        if slab is None:
            result = {
                "explore_clone_count": 0.0,
                "explore_candidate_count": float(explore_candidate_count),
                "explore_valid_ref_count": 0.0,
                "explore_score_fallback_count": float(explore_score_fallback_count),
                "explore_rescue_signal_count": float(explore_rescue_signal_count),
                "explore_live_active_count": float(explore_live_active_count),
            }
            result.update(candidate_metrics)
            return result

        selected_indices = slab["point_indices"]
        explore_clone_count = int(selected_indices.shape[0])
        if explore_clone_count == 0:
            result = {
                "explore_clone_count": 0.0,
                "explore_candidate_count": float(explore_candidate_count),
                "explore_valid_ref_count": 0.0,
                "explore_score_fallback_count": float(explore_score_fallback_count),
                "explore_rescue_signal_count": float(explore_rescue_signal_count),
                "explore_live_active_count": float(explore_live_active_count),
            }
            result.update(candidate_metrics)
            return result
        camera_centers = slab["ref_centers"]
        dirs = slab["ray_dirs"]
        tau_center = slab["tau"]
        tau_min = slab["tau_min"]
        tau_max = slab["tau_max"]
        anchor_radius = slab["anchor_radius"]
        background_like_slab = slab.get(
            "background_like_depth_mask",
            torch.zeros((selected_indices.shape[0],), dtype=torch.bool, device=selected_indices.device),
        )
        active_rescue_slab = slab.get(
            "active_rescue_slab_mask",
            torch.zeros((selected_indices.shape[0],), dtype=torch.bool, device=selected_indices.device),
        )
        adaptive_slab_mult = slab.get(
            "adaptive_slab_mult",
            torch.zeros((selected_indices.shape[0],), dtype=tau_center.dtype, device=selected_indices.device),
        )
        depth_delta_ratio = slab.get(
            "depth_delta_ratio",
            torch.zeros((selected_indices.shape[0],), dtype=tau_center.dtype, device=selected_indices.device),
        )
        neighbor_stable_slab = slab.get(
            "neighbor_stable_mask",
            torch.zeros((selected_indices.shape[0],), dtype=torch.bool, device=selected_indices.device),
        )
        resolved_ref_camera = slab.get(
            "resolved_ref_camera",
            torch.full((selected_indices.shape[0],), -1, dtype=torch.long, device=selected_indices.device),
        )
        finite_slab_mask = (
            torch.isfinite(camera_centers).all(dim=1)
            & torch.isfinite(dirs).all(dim=1)
            & torch.isfinite(tau_center)
            & torch.isfinite(tau_min)
            & torch.isfinite(tau_max)
            & ((tau_max - tau_min) > 1e-6)
            & (resolved_ref_camera >= 0)
            & ((~background_like_slab) | active_rescue_slab)
        )
        if not torch.any(finite_slab_mask):
            result = {
                "explore_clone_count": 0.0,
                "explore_candidate_count": float(explore_candidate_count),
                "explore_valid_ref_count": 0.0,
                "explore_score_fallback_count": float(explore_score_fallback_count),
                "explore_rescue_signal_count": float(explore_rescue_signal_count),
                "explore_live_active_count": float(explore_live_active_count),
                "explore_slab_discard_count": float(explore_clone_count),
            }
            result.update(candidate_metrics)
            return result
        if not torch.all(finite_slab_mask):
            selected_indices = selected_indices[finite_slab_mask]
            camera_centers = camera_centers[finite_slab_mask]
            dirs = dirs[finite_slab_mask]
            tau_center = tau_center[finite_slab_mask]
            tau_min = tau_min[finite_slab_mask]
            tau_max = tau_max[finite_slab_mask]
            anchor_radius = anchor_radius[finite_slab_mask]
            resolved_ref_camera = resolved_ref_camera[finite_slab_mask]
            background_like_slab = background_like_slab[finite_slab_mask]
            active_rescue_slab = active_rescue_slab[finite_slab_mask]
            adaptive_slab_mult = adaptive_slab_mult[finite_slab_mask]
            depth_delta_ratio = depth_delta_ratio[finite_slab_mask]
            neighbor_stable_slab = neighbor_stable_slab[finite_slab_mask]
        pre_slab_filter_count = explore_clone_count
        explore_clone_count = int(selected_indices.shape[0])
        jitter = (torch.rand_like(tau_center) * 2.0 - 1.0) * anchor_radius * float(jitter_scale)
        raw_tau_new = tau_center + jitter
        slab_upper_overrun = torch.relu(raw_tau_new - tau_max)
        slab_lower_overrun = torch.relu(tau_min - raw_tau_new)
        slab_soft_overrun = slab_upper_overrun + slab_lower_overrun
        slab_soft_clamp_count = int((slab_soft_overrun > 0.0).sum().item())
        slab_span = (tau_max - tau_min).clamp_min(1e-6)
        slab_margin = torch.minimum(slab_span * 1e-4, anchor_radius.clamp_min(1e-6) * 1e-3).clamp_min(1e-7)
        tau_low = tau_min + slab_margin
        tau_high = tau_max - slab_margin
        tau_new = torch.where(
            tau_high > tau_low,
            torch.minimum(torch.maximum(raw_tau_new, tau_low), tau_high),
            torch.clamp(raw_tau_new, min=tau_min, max=tau_max),
        )
        new_xyz = camera_centers + tau_new[:, None] * dirs
        tau_recheck = ((new_xyz - camera_centers) * dirs).sum(dim=1)
        finite_explore = (
            torch.isfinite(new_xyz).all(dim=1)
            & torch.isfinite(tau_recheck)
        )
        if not torch.any(finite_explore):
            result = {
                "explore_clone_count": 0.0,
                "explore_candidate_count": float(explore_candidate_count),
                "explore_valid_ref_count": 0.0,
                "explore_score_fallback_count": float(explore_score_fallback_count),
                "explore_rescue_signal_count": float(explore_rescue_signal_count),
                "explore_live_active_count": float(explore_live_active_count),
                "explore_slab_discard_count": float(pre_slab_filter_count),
                "explore_slab_soft_clamp_count": float(slab_soft_clamp_count),
                "explore_tau_soft_penalty_mean": float(slab_soft_overrun.mean().item()) if slab_soft_overrun.numel() > 0 else 0.0,
                "explore_tau_span_mean": float(slab_span.mean().item()) if slab_span.numel() > 0 else 0.0,
            }
            result.update(candidate_metrics)
            return result
        if not torch.all(finite_explore):
            selected_indices = selected_indices[finite_explore]
            new_xyz = new_xyz[finite_explore]
            resolved_ref_camera = resolved_ref_camera[finite_explore]
            slab_soft_overrun = slab_soft_overrun[finite_explore]
            slab_span = slab_span[finite_explore]
            jitter = jitter[finite_explore]
            adaptive_slab_mult = adaptive_slab_mult[finite_explore]
            depth_delta_ratio = depth_delta_ratio[finite_explore]
            background_like_slab = background_like_slab[finite_explore]
            active_rescue_slab = active_rescue_slab[finite_explore]
            neighbor_stable_slab = neighbor_stable_slab[finite_explore]
        slab_discard_count = pre_slab_filter_count - int(selected_indices.shape[0])
        new_features_dc = self._features_dc[selected_indices]
        new_features_rest = self._features_rest[selected_indices]
        parent_opacity = self.get_opacity.detach()[selected_indices].clamp(1e-4, 0.995)
        new_opacities = inverse_sigmoid((parent_opacity * 0.55).clamp(1e-4, 0.995))
        new_scaling = self._uniform_rescale_log_scaling(self._scaling[selected_indices], factor=0.80)
        new_rotation = self._rotation[selected_indices]
        new_center_sigma_parallel = self._center_log_sigma_parallel[selected_indices]
        new_center_sigma_support = self._center_log_sigma_support[selected_indices]
        new_tmp_radii = self.tmp_radii[selected_indices]
        new_atlas_node_ids = self._atlas_node_ids[selected_indices]
        new_atlas_states = torch.full_like(new_atlas_node_ids, GAUSSIAN_STATE_UNSTABLE_ACTIVE)
        new_photo_ema = self._atlas_photo_ema[selected_indices]
        new_visibility_ema = self._atlas_visibility_ema[selected_indices]
        new_high_residual_count = self._atlas_high_residual_count[selected_indices]
        new_low_residual_count = torch.zeros_like(self._atlas_low_residual_count[selected_indices])
        new_promotion_streak = self._atlas_promotion_streak[selected_indices]
        new_demotion_streak = torch.zeros_like(self._atlas_demotion_streak[selected_indices])
        new_last_transition_iter = torch.full_like(
            self._atlas_last_transition_iter[selected_indices],
            int(max(getattr(self, "_atlas_state_update_iter", 0), 0)),
        )
        new_gc_fail_count = torch.zeros_like(self._atlas_gc_fail_count[selected_indices])
        new_drift_flag = torch.zeros_like(self._atlas_drift_flag[selected_indices], dtype=torch.bool)
        new_drift_count = torch.zeros_like(self._atlas_drift_count[selected_indices])
        new_state_cooldown = torch.maximum(
            self._atlas_state_cooldown[selected_indices],
            torch.full_like(
                self._atlas_state_cooldown[selected_indices],
                int(max(active_min_lifetime_iters, 0)),
            ),
        )
        new_active_lifetime = torch.ones_like(self._atlas_active_lifetime[selected_indices])
        new_ref_camera = resolved_ref_camera.to(device=self._atlas_ref_camera.device, dtype=torch.long)
        new_ref_score = torch.maximum(
            self._atlas_ref_score[selected_indices],
            torch.full_like(self._atlas_ref_score[selected_indices], 0.5),
        )
        new_active_provenance = torch.where(
            active_rescue_slab,
            torch.full_like(new_atlas_node_ids, ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP),
            torch.full_like(new_atlas_node_ids, ACTIVE_PROVENANCE_FROM_ACTIVE_EXPLORE_CLONE),
        )
        explore_payload, explore_nonfinite_discard_count = self._filter_nonfinite_clone_payload(
            {
                "xyz": new_xyz,
                "features_dc": new_features_dc,
                "features_rest": new_features_rest,
                "opacity": new_opacities,
                "scaling": new_scaling,
                "rotation": new_rotation,
                "center_sigma_parallel": new_center_sigma_parallel,
                "center_sigma_support": new_center_sigma_support,
                "tmp_radii": new_tmp_radii,
                "atlas_node_ids": new_atlas_node_ids,
                "atlas_states": new_atlas_states,
                "photo_ema": new_photo_ema,
                "visibility_ema": new_visibility_ema,
                "high_residual_count": new_high_residual_count,
                "low_residual_count": new_low_residual_count,
                "promotion_streak": new_promotion_streak,
                "demotion_streak": new_demotion_streak,
                "last_transition_iter": new_last_transition_iter,
                "gc_fail_count": new_gc_fail_count,
                "drift_flag": new_drift_flag,
                "drift_count": new_drift_count,
                "state_cooldown": new_state_cooldown,
                "active_lifetime": new_active_lifetime,
                "ref_camera": new_ref_camera,
                "ref_score": new_ref_score,
                "active_provenance": new_active_provenance,
            }
        )
        explore_clone_count = int(explore_payload["xyz"].shape[0]) if explore_payload["xyz"] is not None else 0
        if explore_clone_count == 0:
            result = {
                "explore_clone_count": 0.0,
                "explore_candidate_count": float(explore_candidate_count),
                "explore_valid_ref_count": 0.0,
                "explore_nonfinite_discard_count": float(explore_nonfinite_discard_count),
                "explore_slab_discard_count": float(slab_discard_count),
                "explore_score_fallback_count": float(explore_score_fallback_count),
                "explore_rescue_signal_count": float(explore_rescue_signal_count),
                "explore_live_active_count": float(explore_live_active_count),
                "explore_slab_soft_clamp_count": float(slab_soft_clamp_count),
                "explore_tau_soft_penalty_mean": float(slab_soft_overrun.mean().item()) if slab_soft_overrun.numel() > 0 else 0.0,
                "explore_tau_span_mean": float(slab_span.mean().item()) if slab_span.numel() > 0 else 0.0,
            }
            result.update(candidate_metrics)
            return result

        self.densification_postfix(
            explore_payload["xyz"],
            explore_payload["features_dc"],
            explore_payload["features_rest"],
            explore_payload["opacity"],
            explore_payload["scaling"],
            explore_payload["rotation"],
            explore_payload["center_sigma_parallel"],
            explore_payload["center_sigma_support"],
            explore_payload["tmp_radii"],
            new_atlas_node_ids=explore_payload["atlas_node_ids"],
            new_atlas_states=explore_payload["atlas_states"],
            new_photo_ema=explore_payload["photo_ema"],
            new_visibility_ema=explore_payload["visibility_ema"],
            new_high_residual_count=explore_payload["high_residual_count"],
            new_low_residual_count=explore_payload["low_residual_count"],
            new_promotion_streak=explore_payload["promotion_streak"],
            new_demotion_streak=explore_payload["demotion_streak"],
            new_last_transition_iter=explore_payload["last_transition_iter"],
            new_gc_fail_count=explore_payload["gc_fail_count"],
            new_drift_flag=explore_payload["drift_flag"],
            new_drift_count=explore_payload["drift_count"],
            new_state_cooldown=explore_payload["state_cooldown"],
            new_active_lifetime=explore_payload["active_lifetime"],
            new_ref_camera=explore_payload["ref_camera"],
            new_ref_score=explore_payload["ref_score"],
            new_active_provenance=explore_payload["active_provenance"],
        )
        result = {
            "explore_clone_count": float(explore_clone_count),
            "explore_candidate_count": float(explore_candidate_count),
            "explore_valid_ref_count": float(explore_clone_count),
            "explore_nonfinite_discard_count": float(explore_nonfinite_discard_count),
            "explore_slab_discard_count": float(slab_discard_count),
            "explore_score_fallback_count": float(explore_score_fallback_count),
            "explore_rescue_signal_count": float(explore_rescue_signal_count),
            "explore_live_active_count": float(explore_live_active_count),
            "explore_slab_valid_count": float(pre_slab_filter_count),
            "explore_slab_soft_clamp_count": float(slab_soft_clamp_count),
            "explore_tau_soft_penalty_mean": float(slab_soft_overrun.mean().item()) if slab_soft_overrun.numel() > 0 else 0.0,
            "explore_tau_jitter_abs_mean": float(jitter.abs().mean().item()) if jitter.numel() > 0 else 0.0,
            "explore_tau_span_mean": float(slab_span.mean().item()) if slab_span.numel() > 0 else 0.0,
            "explore_adaptive_slab_mult_mean": float(adaptive_slab_mult.mean().item()) if adaptive_slab_mult.numel() > 0 else 0.0,
            "explore_depth_delta_ratio_mean": float(depth_delta_ratio.mean().item()) if depth_delta_ratio.numel() > 0 else 0.0,
            "explore_background_like_slab_count": float(background_like_slab.sum().item()) if background_like_slab.numel() > 0 else 0.0,
            "explore_active_rescue_slab_count": float(active_rescue_slab.sum().item()) if active_rescue_slab.numel() > 0 else 0.0,
            "explore_neighbor_stable_slab_count": float(neighbor_stable_slab.sum().item()) if neighbor_stable_slab.numel() > 0 else 0.0,
            "explore_ref_repair_count": float(
                slab_admission_repair_count
                + int(slab.get("repaired_ref_mask", torch.zeros_like(selected_indices, dtype=torch.bool)).sum().item())
            ),
            "explore_slab_fallback_count": float(
                slab_admission_fallback_count
                + int(slab.get("fallback_slab_mask", torch.zeros_like(selected_indices, dtype=torch.bool)).sum().item())
            ),
        }
        result.update(candidate_metrics)
        return result

    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        radii,
        prune_enabled: bool = True,
        min_points_to_keep: int = 0,
        visibility_threshold: float = 0.02,
        max_reattach_failures: int = 2,
        enable_soft_prune: bool = True,
    ):
        metrics = {
            "clone_count": 0.0,
            "split_count": 0.0,
            "split_source_count": 0.0,
            "split_parent_pruned_count": 0.0,
            "explore_clone_count": 0.0,
            "stable_clone_count": 0.0,
            "stable_split_count": 0.0,
            "active_explore_clone_count": 0.0,
            "pruned_count": 0.0,
            "prune_after_gc": 0.0,
            "pruned_dead_count": 0.0,
            "pruned_soft_count": 0.0,
            "pruned_dead_candidate_count": 0.0,
            "pruned_soft_candidate_count": 0.0,
            "prune_enabled": 1.0 if prune_enabled else 0.0,
            "prune_soft_enabled": 1.0 if enable_soft_prune else 0.0,
            "min_points_to_keep": float(max(int(min_points_to_keep), 0)),
            "nonfinite_clone_discard_count": 0.0,
            "invalid_gaussian_prune_count": 0.0,
        }
        grads = torch.zeros_like(self.xyz_gradient_accum)
        valid_grad_samples = self.denom > 0
        grads[valid_grad_samples] = self.xyz_gradient_accum[valid_grad_samples] / self.denom[valid_grad_samples]
        grads[~torch.isfinite(grads)] = 0.0

        self.tmp_radii = radii
        clone_metrics = self.densify_and_clone(grads, max_grad, extent)
        split_metrics = self.densify_and_split(grads, max_grad, extent)
        metrics.update(clone_metrics)
        metrics.update(split_metrics)
        metrics["nonfinite_clone_discard_count"] = float(
            clone_metrics.get("clone_nonfinite_discard_count", 0.0)
            + split_metrics.get("split_nonfinite_discard_count", 0.0)
        )

        dead_mask = self._build_dead_prune_mask(
            min_opacity=min_opacity,
            visibility_threshold=visibility_threshold,
            max_reattach_failures=max_reattach_failures,
        )
        invalid_mask = self._build_invalid_gaussian_mask()
        soft_mask = torch.zeros_like(dead_mask)
        gc_pending_mask = self._build_gc_pending_prune_mask(max_reattach_failures=max_reattach_failures)
        if max_screen_size and enable_soft_prune:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            soft_mask = torch.logical_or(big_points_vs, big_points_ws)
        prune_mask = torch.logical_or(torch.logical_or(dead_mask, soft_mask), invalid_mask)
        metrics["pruned_dead_candidate_count"] = float(dead_mask.sum().item())
        metrics["pruned_soft_candidate_count"] = float(torch.logical_and(soft_mask, ~dead_mask).sum().item())
        force_invalid_prune = bool(torch.any(invalid_mask).item()) if invalid_mask.numel() > 0 else False

        if (prune_enabled or force_invalid_prune) and torch.any(prune_mask):
            prune_priority = self._build_prune_priority(dead_mask, soft_mask)
            if force_invalid_prune:
                prune_priority[invalid_mask] = prune_priority[invalid_mask] + 1e6
            if prune_enabled:
                prune_mask = self._limit_prune_mask(prune_mask, min_points_to_keep=min_points_to_keep, prune_priority=prune_priority)
            if force_invalid_prune:
                prune_mask = torch.logical_or(prune_mask, invalid_mask)
            metrics["pruned_dead_count"] = float(torch.logical_and(prune_mask, dead_mask).sum().item())
            metrics["pruned_soft_count"] = float(torch.logical_and(torch.logical_and(prune_mask, soft_mask), ~dead_mask).sum().item())
            metrics["prune_after_gc"] = float(torch.logical_and(prune_mask, gc_pending_mask).sum().item())
            metrics["invalid_gaussian_prune_count"] = float(torch.logical_and(prune_mask, invalid_mask).sum().item())
            metrics["pruned_count"] = float(self.prune_points(prune_mask))
        self.tmp_radii = None

        torch.cuda.empty_cache()
        return metrics

    def densify_and_prune_with_atlas(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        radii,
        camera_center,
        explore_grad_scale: float,
        explore_slab_radius_mult: float,
        explore_jitter_scale: float,
        active_min_lifetime_iters: int = 5,
        stable_residual_threshold: float = 0.03,
        all_camera_centers=None,
        prune_enabled: bool = True,
        min_points_to_keep: int = 0,
        visibility_threshold: float = 0.02,
        max_reattach_failures: int = 2,
        enable_soft_prune: bool = True,
        densify_runtime_controls: dict | None = None,
    ):
        metrics = {
            "clone_count": 0.0,
            "split_count": 0.0,
            "split_source_count": 0.0,
            "split_parent_pruned_count": 0.0,
            "explore_clone_count": 0.0,
            "stable_clone_count": 0.0,
            "stable_split_count": 0.0,
            "active_explore_clone_count": 0.0,
            "pruned_count": 0.0,
            "prune_after_gc": 0.0,
            "pruned_dead_count": 0.0,
            "pruned_soft_count": 0.0,
            "pruned_dead_candidate_count": 0.0,
            "pruned_soft_candidate_count": 0.0,
            "prune_enabled": 1.0 if prune_enabled else 0.0,
            "prune_soft_enabled": 1.0 if enable_soft_prune else 0.0,
            "min_points_to_keep": float(max(int(min_points_to_keep), 0)),
            "nonfinite_clone_discard_count": 0.0,
            "invalid_gaussian_prune_count": 0.0,
            "background_fidelity_protected_count": 0.0,
            "fidelity_mode_enabled": 0.0,
            "active_noisy_pruned_count": 0.0,
            "unsupported_rescue_pruned_count": 0.0,
            "unsupported_explore_pruned_count": 0.0,
            "fidelity_handoff_unsupported_explore_prune_count": 0.0,
            "fidelity_handoff_unsupported_rescue_prune_count": 0.0,
            "fidelity_handoff_active_noisy_prune_count": 0.0,
        }
        grads = torch.zeros_like(self.xyz_gradient_accum)
        valid_grad_samples = self.denom > 0
        grads[valid_grad_samples] = self.xyz_gradient_accum[valid_grad_samples] / self.denom[valid_grad_samples]
        grads[~torch.isfinite(grads)] = 0.0

        self.tmp_radii = radii
        stable_mask = self._atlas_state == GAUSSIAN_STATE_STABLE
        active_mask = self._atlas_state == GAUSSIAN_STATE_UNSTABLE_ACTIVE
        controls = dict(densify_runtime_controls or {})
        budget_scale = float(np.clip(float(controls.get("budget_scale", 1.0)), 0.0, 1.0))
        max_new_ratio = float(max(controls.get("max_new_ratio", 0.012), 0.0))
        max_new_points = int(max(controls.get("max_new_points", 2048), 0))
        min_new_points = int(max(controls.get("min_new_points", 64), 0))
        point_count = int(self.get_xyz.shape[0])
        default_global_quota = int(point_count * max_new_ratio)
        if max_new_points > 0:
            default_global_quota = min(default_global_quota, max_new_points)
        if budget_scale > 0.0:
            default_global_quota = max(default_global_quota, min_new_points)
        else:
            default_global_quota = 0
        global_quota = int(max(0, np.floor(default_global_quota * budget_scale)))
        split_fraction = float(max(controls.get("split_quota_fraction", 0.55), 0.0))
        clone_fraction = float(max(controls.get("clone_quota_fraction", 0.30), 0.0))
        explore_fraction = float(max(controls.get("explore_quota_fraction", 0.15), 0.0))
        raw_split_fraction = split_fraction
        raw_clone_fraction = clone_fraction
        raw_explore_fraction = explore_fraction
        b2_unhealthy_gate = float(max(min(controls.get("b2_unhealthy_gate", 0.0), 1.0), 0.0))
        if b2_unhealthy_gate > 0.0:
            split_fraction *= float(max(min(controls.get("b2_unhealthy_split_scale", 0.35), 1.0), 0.0))
            explore_fraction *= float(max(min(controls.get("b2_unhealthy_explore_scale", 0.10), 1.0), 0.0))
            clone_fraction *= float(max(min(controls.get("b2_unhealthy_clone_scale", 1.15), 2.0), 0.0))
        fidelity_mode_strength = float(
            max(
                min(
                    controls.get(
                        "fidelity_mode_enabled",
                        controls.get("fidelity_mode_gate", 0.0),
                    ),
                    1.0,
                ),
                0.0,
            )
        )
        maintenance_mode_strength = 1.0 - fidelity_mode_strength
        if fidelity_mode_strength > 0.0:
            split_scale = float(max(min(controls.get("fidelity_mode_split_scale", 0.45), 1.0), 0.0))
            explore_scale = float(max(min(controls.get("fidelity_mode_explore_scale", 0.08), 1.0), 0.0))
            clone_boost = float(max(min(controls.get("fidelity_mode_clone_boost", 0.20), 1.0), 0.0))
            split_fraction *= 1.0 - fidelity_mode_strength * (1.0 - split_scale)
            explore_fraction *= 1.0 - fidelity_mode_strength * (1.0 - explore_scale)
            clone_fraction *= 1.0 + fidelity_mode_strength * clone_boost
        fraction_sum = split_fraction + clone_fraction + explore_fraction
        if fraction_sum <= 1e-8:
            split_fraction, clone_fraction, explore_fraction = 0.55, 0.30, 0.15
            fraction_sum = 1.0
        split_fraction /= fraction_sum
        clone_fraction /= fraction_sum
        explore_fraction /= fraction_sum
        split_quota = int(np.floor(global_quota * split_fraction))
        clone_quota = int(np.floor(global_quota * clone_fraction))
        explore_quota = max(global_quota - split_quota - clone_quota, 0)
        explore_quota = min(explore_quota, int(np.ceil(global_quota * explore_fraction)))
        metrics.update({
            "densify_budget_scale": float(budget_scale),
            "densify_global_quota": float(global_quota),
            "densify_default_global_quota": float(default_global_quota),
            "densify_split_quota": float(split_quota),
            "densify_clone_quota": float(clone_quota),
            "densify_explore_quota": float(explore_quota),
            "densify_raw_split_fraction": float(raw_split_fraction),
            "densify_raw_clone_fraction": float(raw_clone_fraction),
            "densify_raw_explore_fraction": float(raw_explore_fraction),
            "densify_effective_split_fraction": float(split_fraction),
            "densify_effective_clone_fraction": float(clone_fraction),
            "densify_effective_explore_fraction": float(explore_fraction),
            "densify_b2_unhealthy_gate": float(b2_unhealthy_gate),
            "densify_b2_recovery_unhealthy_gate": float(controls.get("b2_recovery_unhealthy_gate", 0.0)),
            "densify_atlas_recovery_event_count": float(controls.get("atlas_recovery_event_count", 0.0)),
            "densify_atlas_recovery_seen": float(controls.get("atlas_recovery_seen", 0.0)),
            "densify_b2_zero_grad_skip_delta": float(controls.get("b2_zero_grad_skip_delta", 0.0)),
            "maintenance_mode_enabled": float(maintenance_mode_strength),
            "densify_budget_phase_ramp": float(controls.get("phase_ramp", 1.0)),
            "densify_budget_b2_health": float(controls.get("b2_step_health", 1.0)),
            "densify_budget_floater_guard": float(controls.get("floater_guard", 1.0)),
            "densify_budget_quality_guard": float(controls.get("quality_guard", 1.0)),
            "fidelity_handoff_gate": float(controls.get("fidelity_handoff_gate", 0.0)),
            "fidelity_handoff_completion_gate": float(controls.get("fidelity_handoff_completion_gate", 0.0)),
            "fidelity_handoff_observed_gate": float(controls.get("fidelity_handoff_observed_gate", 0.0)),
            "fidelity_handoff_dark_gate": float(controls.get("fidelity_handoff_dark_gate", 0.0)),
            "fidelity_handoff_stable_gate": float(controls.get("fidelity_handoff_stable_gate", 0.0)),
            "fidelity_handoff_floater_gate": float(controls.get("fidelity_handoff_floater_gate", 0.0)),
            "fidelity_handoff_quality_gate": float(controls.get("fidelity_handoff_quality_gate", 0.0)),
            "fidelity_handoff_late_phase_boost": float(controls.get("fidelity_handoff_late_phase_boost", 0.0)),
            "fidelity_mode_gate": float(controls.get("fidelity_mode_gate", 0.0)),
            "fidelity_mode_enabled": float(fidelity_mode_strength),
            "fidelity_mode_dark_gate": float(controls.get("fidelity_mode_dark_gate", 0.0)),
            "fidelity_mode_l1_gate": float(controls.get("fidelity_mode_l1_gate", 0.0)),
            "fidelity_mode_floater_gate": float(controls.get("fidelity_mode_floater_gate", 0.0)),
            "fidelity_mode_reliability_gate": float(controls.get("fidelity_mode_reliability_gate", 0.0)),
            "fidelity_mode_pose_gate": float(controls.get("fidelity_mode_pose_gate", 0.0)),
            "fidelity_mode_recovery_gate": float(controls.get("fidelity_mode_recovery_gate", 0.0)),
            "fidelity_handoff_budget_scale": float(controls.get("fidelity_handoff_budget_scale", 1.0)),
            "fidelity_handoff_split_scale": float(controls.get("fidelity_handoff_split_scale", 1.0)),
            "fidelity_handoff_clone_scale": float(controls.get("fidelity_handoff_clone_scale", 1.0)),
            "fidelity_handoff_explore_scale": float(controls.get("fidelity_handoff_explore_scale", 1.0)),
            "fidelity_handoff_dark_region_completeness_ema": float(controls.get("dark_region_completeness_ema", 0.0)),
        })
        stable_split_mask, stable_split_candidate_metrics = self.compute_stable_split_candidates(
            grads,
            max_grad,
            extent,
            stable_residual_threshold=stable_residual_threshold,
            min_opacity=min_opacity,
            allowed_mask=stable_mask,
            budget_override=split_quota,
        )
        # Reallocate unused split budget to clone so quota doesn't go to waste.
        used_split = int(stable_split_mask.sum().item())
        clone_quota_effective = clone_quota + max(split_quota - used_split, 0)
        stable_clone_mask, stable_clone_candidate_metrics = self.compute_stable_clone_candidates(
            grads,
            max_grad,
            extent,
            stable_residual_threshold=stable_residual_threshold,
            min_opacity=min_opacity,
            allowed_mask=stable_mask,
            budget_override=clone_quota_effective,
        )
        metrics["densify_clone_quota_effective"] = float(clone_quota_effective)
        metrics["densify_split_unused_reallocated"] = float(max(split_quota - used_split, 0))
        clone_split_overlap = stable_clone_mask & stable_split_mask
        if torch.any(clone_split_overlap):
            stable_clone_mask = stable_clone_mask & (~clone_split_overlap)
        stable_clone_candidate_metrics["stable_clone_suppressed_by_split_count"] = float(clone_split_overlap.sum().item())
        stable_clone_candidate_metrics["stable_clone_candidate_count"] = float(stable_clone_mask.sum().item())
        active_explore_mask, active_explore_candidate_metrics = self.compute_active_explore_clone_candidates(
            grads,
            max_grad * float(explore_grad_scale),
            stable_residual_threshold=stable_residual_threshold,
            min_opacity=min_opacity,
            allowed_mask=active_mask,
            budget_override=explore_quota,
        )
        metrics.update(stable_split_candidate_metrics)
        metrics.update(stable_clone_candidate_metrics)
        metrics.update(active_explore_candidate_metrics)
        # Atlas-aware densification policies:
        # stable -> local split or support-projected clone
        # unstable_active -> ray/slab explore-clone only
        # unstable_passive -> no densification, wait for state transition
        stable_clone_metrics = self.densify_and_clone(
            grads,
            max_grad,
            extent,
            allowed_mask=stable_mask,
            candidate_mask=stable_clone_mask,
            support_projected_jitter=True,
            support_jitter_scale=0.35,
        )
        metrics.update(stable_clone_metrics)
        metrics["stable_clone_count"] = float(stable_clone_metrics.get("clone_count", 0.0))

        explore_metrics = self.explore_and_clone(
            grads,
            max_grad * float(explore_grad_scale),
            camera_center=camera_center,
            slab_radius_mult=float(explore_slab_radius_mult),
            jitter_scale=float(explore_jitter_scale),
            allowed_mask=active_mask,
            all_camera_centers=all_camera_centers,
            active_min_lifetime_iters=int(max(active_min_lifetime_iters, 0)),
            min_opacity=float(min_opacity),
            stable_residual_threshold=float(stable_residual_threshold),
            candidate_mask=active_explore_mask,
            candidate_metrics=active_explore_candidate_metrics,
        )
        metrics.update(explore_metrics)
        metrics["active_explore_clone_count"] = float(explore_metrics.get("explore_clone_count", 0.0))

        stable_split_metrics = self.densify_and_split(
            grads,
            max_grad,
            extent,
            allowed_mask=stable_mask,
            candidate_mask=stable_split_mask,
        )
        metrics.update(stable_split_metrics)
        metrics["stable_split_count"] = float(stable_split_metrics.get("split_count", 0.0))
        metrics["nonfinite_clone_discard_count"] = float(
            explore_metrics.get("explore_nonfinite_discard_count", 0.0)
            + stable_clone_metrics.get("clone_nonfinite_discard_count", 0.0)
            + stable_split_metrics.get("split_nonfinite_discard_count", 0.0)
        )

        dead_mask, dead_prune_guard_metrics = self._build_dead_prune_mask(
            min_opacity=min_opacity,
            visibility_threshold=visibility_threshold,
            max_reattach_failures=max_reattach_failures,
            background_ref_score_min=float(controls.get("background_ref_score_min", 0.06)),
            background_visibility_min=float(controls.get("background_visibility_min", 0.003)),
            background_guard_enabled=bool(controls.get("background_dead_prune_guard", True)),
            fidelity_background_guard_strength=fidelity_mode_strength * float(controls.get("fidelity_mode_background_guard_strength", 1.0)),
            return_metrics=True,
        )
        metrics.update(dead_prune_guard_metrics)
        metrics["background_fidelity_protected_count"] = float(
            metrics.get("background_fidelity_protected_count", 0.0)
            + dead_prune_guard_metrics.get("background_dead_prune_protected_count", 0.0)
        )
        invalid_mask = self._build_invalid_gaussian_mask()
        soft_mask = torch.zeros_like(dead_mask)
        gc_pending_mask = self._build_gc_pending_prune_mask(max_reattach_failures=max_reattach_failures)
        if max_screen_size and enable_soft_prune:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            soft_mask = torch.logical_or(big_points_vs, big_points_ws)
            if (
                bool(controls.get("background_soft_prune_guard", True))
                and self.has_atlas_bindings
                and self._atlas_state.numel() == soft_mask.numel()
            ):
                support_consistency = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
                node_observed = self._atlas_node_metric_for_gaussians(
                    self._atlas_node_observed_score_ema,
                    default_value=0.0,
                ).clamp(0.0, 1.0)
                node_coverage = self._atlas_node_metric_for_gaussians(
                    self._atlas_refresh_node_coverage_ratio,
                    default_value=0.0,
                ).clamp(0.0, 1.0)
                reliability_effective = self.get_gaussian_atlas_reliability_effective.detach().clamp(0.0, 1.0)
                ref_score = self._atlas_ref_score.detach().clamp(0.0, 1.0)
                background_ref_min = float(controls.get("background_ref_score_min", 0.06))
                background_visibility_min = float(controls.get("background_visibility_min", 0.003))
                thin_background_support = (
                    (support_consistency >= 0.26)
                    & (reliability_effective >= 0.14)
                    & (
                        (node_observed >= 0.12)
                        | (node_coverage >= 0.10)
                        | (self._atlas_visibility_ema.detach() >= background_visibility_min)
                    )
                )
                background_anchor_hint = (
                    (self._atlas_ref_camera.detach() >= 0)
                    | (ref_score >= background_ref_min)
                    | (node_observed >= 0.16)
                )
                background_keep = (
                    (
                        (self._atlas_state.detach() == GAUSSIAN_STATE_STABLE)
                        | (self._atlas_state.detach() == GAUSSIAN_STATE_UNSTABLE_PASSIVE)
                    )
                    & (
                        (ref_score >= background_ref_min)
                        | (node_observed >= 0.20)
                        | (support_consistency >= 0.32)
                        | thin_background_support
                    )
                    & background_anchor_hint
                    & (
                        (self._atlas_visibility_ema.detach() >= background_visibility_min)
                        | (node_observed >= 0.12)
                        | (support_consistency >= 0.32)
                        | thin_background_support
                    )
                )
                protected = soft_mask & background_keep
                metrics["background_fidelity_protected_count"] = float(
                    metrics.get("background_fidelity_protected_count", 0.0) + float(protected.sum().item())
                )
                soft_mask = soft_mask & (~background_keep)
        fidelity_handoff_gate = float(controls.get("fidelity_handoff_gate", 0.0))
        fidelity_prune_gate = max(fidelity_handoff_gate, fidelity_mode_strength)
        active_prune_boost = fidelity_mode_strength * float(max(min(controls.get("fidelity_mode_active_prune_boost", 1.0), 1.0), 0.0))
        active_noisy_prune = torch.zeros_like(dead_mask)
        unsupported_explore_active = torch.zeros_like(dead_mask)
        unsupported_rescue_active = torch.zeros_like(dead_mask)
        if (
            enable_soft_prune
            and fidelity_prune_gate >= float(controls.get("active_prune_min_gate", 0.65))
            and self._atlas_state.numel() == dead_mask.numel()
        ):
            current_active = self._atlas_state.detach() == GAUSSIAN_STATE_UNSTABLE_ACTIVE
            opacity = self.get_opacity.detach().squeeze(-1).clamp(0.0, 1.0)
            low_opacity_active = opacity <= max(float(min_opacity) * 2.0, 1e-4)
            active_photo_prune_scale = max(0.85, 1.0 - 0.20 * active_prune_boost)
            explore_support_max = min(0.28, 0.20 + 0.06 * active_prune_boost)
            rescue_support_max = min(0.26, 0.18 + 0.06 * active_prune_boost)
            active_visibility_max = max(float(visibility_threshold) * (0.75 + 0.35 * active_prune_boost), 0.0)
            long_lived_noisy_active = (
                (self._atlas_active_lifetime.detach() >= max(int(active_min_lifetime_iters) * 4, 1))
                & (self._atlas_visibility_ema.detach() <= max(float(visibility_threshold) * 0.50, 0.0))
                & (self._atlas_photo_ema.detach() >= float(stable_residual_threshold) * 1.25 * active_photo_prune_scale)
            )
            support_consistency = self._compute_support_consistency_score().detach().clamp(0.0, 1.0)
            active_provenance = self._atlas_active_provenance.detach()
            explore_origin = active_provenance == ACTIVE_PROVENANCE_FROM_ACTIVE_EXPLORE_CLONE
            rescue_or_transition_origin = (
                explore_origin
                | (active_provenance == ACTIVE_PROVENANCE_FROM_FORCED_RESCUE_BOOTSTRAP)
                | (active_provenance == ACTIVE_PROVENANCE_FROM_TRANSITION_PASSIVE_TO_ACTIVE)
            )
            unsupported_explore_active = (
                explore_origin
                & (self._atlas_active_lifetime.detach() >= max(int(active_min_lifetime_iters) * 2, 1))
                & (support_consistency <= explore_support_max)
                & (self._atlas_photo_ema.detach() >= float(stable_residual_threshold) * 1.10 * active_photo_prune_scale)
            )
            unsupported_rescue_active = (
                rescue_or_transition_origin
                & (self._atlas_active_lifetime.detach() >= max(int(active_min_lifetime_iters) * 3, 1))
                & (support_consistency <= rescue_support_max)
                & (self._atlas_photo_ema.detach() >= float(stable_residual_threshold) * 1.20 * active_photo_prune_scale)
                & (self._atlas_visibility_ema.detach() <= active_visibility_max)
            )
            active_noisy_prune = current_active & (
                low_opacity_active
                | long_lived_noisy_active
                | unsupported_explore_active
                | unsupported_rescue_active
            )
            soft_mask = torch.logical_or(soft_mask, active_noisy_prune)
            metrics["fidelity_handoff_unsupported_explore_prune_count"] = float(unsupported_explore_active.sum().item())
            metrics["fidelity_handoff_unsupported_rescue_prune_count"] = float(unsupported_rescue_active.sum().item())
        metrics["fidelity_prune_gate"] = float(fidelity_prune_gate)
        metrics["fidelity_handoff_active_noisy_prune_count"] = float(active_noisy_prune.sum().item())
        prune_mask = torch.logical_or(torch.logical_or(dead_mask, soft_mask), invalid_mask)
        metrics["pruned_dead_candidate_count"] = float(dead_mask.sum().item())
        metrics["pruned_soft_candidate_count"] = float(torch.logical_and(soft_mask, ~dead_mask).sum().item())
        force_invalid_prune = bool(torch.any(invalid_mask).item()) if invalid_mask.numel() > 0 else False

        if (prune_enabled or force_invalid_prune) and torch.any(prune_mask):
            prune_priority = self._build_prune_priority(dead_mask, soft_mask)
            if active_noisy_prune.numel() == prune_priority.numel():
                prune_priority = prune_priority.clone()
                prune_priority[active_noisy_prune] = prune_priority[active_noisy_prune] + 2.0 + 3.0 * float(fidelity_prune_gate)
            if force_invalid_prune:
                prune_priority[invalid_mask] = prune_priority[invalid_mask] + 1e6
            if prune_enabled:
                prune_mask = self._limit_prune_mask(prune_mask, min_points_to_keep=min_points_to_keep, prune_priority=prune_priority)
            if force_invalid_prune:
                prune_mask = torch.logical_or(prune_mask, invalid_mask)
            metrics["active_noisy_pruned_count"] = float(torch.logical_and(prune_mask, active_noisy_prune).sum().item())
            metrics["unsupported_explore_pruned_count"] = float(torch.logical_and(prune_mask, unsupported_explore_active).sum().item())
            metrics["unsupported_rescue_pruned_count"] = float(torch.logical_and(prune_mask, unsupported_rescue_active).sum().item())
            metrics["pruned_dead_count"] = float(torch.logical_and(prune_mask, dead_mask).sum().item())
            metrics["pruned_soft_count"] = float(torch.logical_and(torch.logical_and(prune_mask, soft_mask), ~dead_mask).sum().item())
            metrics["prune_after_gc"] = float(torch.logical_and(prune_mask, gc_pending_mask).sum().item())
            metrics["invalid_gaussian_prune_count"] = float(torch.logical_and(prune_mask, invalid_mask).sum().item())
            metrics["pruned_count"] = float(self.prune_points(prune_mask))
        self.tmp_radii = None

        torch.cuda.empty_cache()
        return metrics

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
