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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self.atlas_path = ""
        self.atlas_color_sample_limit = 50000
        self.atlas_surface_stable_reliability = 0.12
        self.atlas_edge_stable_reliability = 0.08
        self.atlas_scale_multiplier = 1.0
        self.atlas_surface_thickness = 0.15
        self.atlas_edge_thickness = 0.18
        self.atlas_unstable_scale = 0.35
        self.atlas_seed = 42
        self._images = "images"
        self._depths = ""
        self.depth_confidences = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.center_uncertainty_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.prune_from_iter = 500
        self.min_points_before_prune = 0
        self.prune_min_capacity_ratio = 1.25
        self.prune_min_capacity_extra = 1024
        self.prune_hard_floor_ratio = 0.4
        self.prune_visibility_threshold = 0.02
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.depth_confidence_exponent = 1.0
        self.depth_confidence_min = 0.05
        self.atlas_mean_weight = 0.1
        self.atlas_ori_weight = 0.02
        self.atlas_aniso_weight = 0.01
        self.atlas_kl_weight = 0.01
        self.atlas_mean_warmup_scale = 0.35
        self.atlas_kl_warmup_scale = 0.25
        self.atlas_shape_main_phase_ramp_iters = 500
        self.atlas_after_refresh_mean_boost = 0.20
        self.atlas_after_refresh_shape_boost = 0.12
        self.atlas_after_refresh_mean_max_scale = 1.20
        self.atlas_after_refresh_shape_max_scale = 1.12
        self.atlas_shape_post_refresh_floor = 0.08
        self.atlas_shape_floor_min_reliability = 0.06
        self.atlas_shape_floor_full_reliability = 0.16
        self.atlas_mean_ramp_stable_ratio = 0.30
        self.atlas_ori_ramp_stable_ratio = 0.45
        self.atlas_aniso_ramp_stable_ratio = 0.60
        self.atlas_reg_ramp_capacity_ratio = 1.15
        self.atlas_mean_passive_state_weight = 0.35
        self.atlas_mean_active_state_weight = 0.1
        self.atlas_reg_passive_state_weight = 0.35
        self.atlas_reg_active_state_weight = 0.03
        self.atlas_kl_passive_state_weight = 0.35
        self.atlas_kl_active_state_weight = 0.15
        self.atlas_kl_eps_perp = 0.002
        self.atlas_kl_eps_tangent = 0.01
        self.atlas_kl_lambda_parallel_base = 5.0
        self.atlas_kl_lambda_parallel_gain = 15.0
        self.atlas_kl_lambda_support_base = 10.0
        self.atlas_kl_lambda_support_gain = 20.0
        self.atlas_kl_lambda_perp_base = 40.0
        self.atlas_kl_lambda_perp_gain = 60.0
        self.atlas_obs_lambda = 0.001
        self.atlas_obs_max_cameras = 64
        self.atlas_obs_point_chunk = 2048
        self.atlas_huber_delta = 0.02
        self.atlas_reg_warmup_steps = 1000
        self.atlas_state_ema_decay = 0.95
        self.atlas_state_min_visibility = 0.12
        self.atlas_stable_residual_threshold = 0.03
        self.atlas_promote_to_active_threshold = 0.15
        self.atlas_demote_to_passive_threshold = 0.075
        self.atlas_activate_threshold = 0.15
        self.atlas_deactivate_threshold = 0.075
        self.atlas_activate_min_high_residual_iters = 30
        self.atlas_state_low_residual_iters = 3
        self.atlas_state_drift_iters = 2
        self.atlas_state_out_of_anchor_iters = 3
        self.atlas_state_out_of_anchor_gc_failures = 2
        self.atlas_state_cooldown_iters = 5
        self.atlas_state_active_min_iters = 25
        self.atlas_state_active_quota_ratio = 0.03
        self.atlas_state_active_quota_min = 8
        self.atlas_state_active_quota_max = 128
        self.atlas_state_active_max_iters = 600
        self.atlas_state_active_nonimprove_iters = 180
        self.atlas_state_min_active_opacity = 0.02
        self.atlas_passive_to_stable_reliability_min = 0.18
        self.atlas_passive_to_stable_support_consistency_min = 0.28
        self.atlas_passive_to_stable_drift_max = 1.15
        self.atlas_passive_to_stable_photo_ema_max = 0.045
        self.atlas_drift_radius_mult = 1.75
        self.atlas_gc_interval = 100
        self.atlas_reattach_radius_mult = 2.5
        self.atlas_gc_retry_pending = True
        self.atlas_gc_max_reattach_failures = 2
        self.atlas_out_of_anchor_prune_opacity = 0.01
        self.atlas_refresh_alpha = 1.0
        self.atlas_refresh_gamma = 4.0
        self.atlas_refresh_min_visibility = 0.05
        self.atlas_refresh_min_reliability = 0.05
        self.atlas_refresh_low_band_power = 0.65
        self.atlas_refresh_mid_band_power = 1.05
        self.atlas_refresh_support_consistency_weight = 0.12
        self.atlas_refresh_visibility_weight = 0.14
        self.atlas_refresh_override_min_evidence = 0.18
        self.atlas_explore_grad_scale = 0.75
        self.atlas_explore_slab_radius_mult = 2.0
        self.atlas_explore_jitter_scale = 0.45
        self.atlas_slab_weight = 0.01
        self.atlas_densify_ramp_iters = 2500
        self.atlas_densify_max_new_ratio = 0.012
        self.atlas_densify_max_new_points = 2048
        self.atlas_densify_min_new_points = 64
        self.atlas_densify_split_quota_fraction = 0.55
        self.atlas_densify_clone_quota_fraction = 0.30
        self.atlas_densify_explore_quota_fraction = 0.15
        self.atlas_densify_b2_unhealthy_scale = 0.65
        self.atlas_densify_floater_guard_scale = 0.55
        self.atlas_densify_quality_guard_scale = 0.70
        self.atlas_fidelity_handoff_enabled = True
        self.atlas_fidelity_handoff_min_observed_ratio = 0.82
        self.atlas_fidelity_handoff_min_dark_completeness = 0.94
        self.atlas_fidelity_handoff_min_stable_ratio = 0.30
        self.atlas_fidelity_handoff_budget_scale = 0.65
        self.atlas_fidelity_handoff_explore_scale = 0.20
        self.atlas_fidelity_handoff_clone_scale = 0.80
        self.atlas_fidelity_handoff_split_scale = 1.00
        self.atlas_fidelity_handoff_active_prune_min_gate = 0.65
        self.atlas_fidelity_handoff_dark_observed_override = True
        self.atlas_fidelity_mode_min_dark_completeness = 0.94
        self.atlas_fidelity_mode_max_l1 = 0.075
        self.atlas_fidelity_mode_max_floater = 0.070
        self.atlas_mc_pairs = 1
        self.atlas_mc_scale = 1.0
        self.atlas_mc_max_blend_weight = 0.35
        self.atlas_mc_active_fraction_start = 0.002
        self.atlas_mc_active_fraction_full = 0.08
        self.atlas_sigma_parallel_min_ratio = 0.03
        self.atlas_sigma_parallel_max_ratio = 0.45
        self.atlas_sigma_support_min_ratio = 0.01
        self.atlas_sigma_support_max_ratio = 0.20
        self.atlas_sigma_active_ray_max_fraction = 1.0
        self.atlas_sigma_active_ray_min_fraction = 0.10
        self.atlas_sigma_passive_parallel_max_mult = 1.35
        self.atlas_sigma_passive_support_max_mult = 1.25
        self.atlas_sigma_active_parallel_min_mult = 1.50
        self.atlas_sigma_active_parallel_max_mult = 2.25
        self.atlas_sigma_active_support_min_mult = 0.75
        self.atlas_sigma_active_support_max_mult = 1.50
        self.atlas_sigma_active_low_visibility_decay = 0.995
        self.atlas_sigma_decay = 0.98
        self.atlas_sigma_low_visibility_threshold = 0.05
        self.pose_lr = 0.0005
        self.pose_refine_after_warmup = True
        self.pose_update_interval = 5
        self.pose_geo_weight = 0.02
        self.pose_photo_weight = 0.01
        self.pose_photo_alpha = 0.5
        self.pose_gradient_weight = 0.05
        self.pose_patchfeat_weight = 0.05
        self.pose_b2_patchfeat_weight = 0.0
        self.pose_patch_radius = 1
        self.pose_sample_count = 1024
        self.pose_geo_mad_scale = 3.5
        self.pose_geo_percentile = 0.9
        self.pose_geo_min_corr = 32
        self.pose_translation_l2_weight = 0.01
        self.pose_rotation_l2_weight = 0.01
        self.pose_max_translation_ratio = 0.02
        self.pose_max_rotation_degrees = 3.0
        self.pose_b1_bootstrap_iters = 250
        self.pose_b1_bootstrap_min_corr = 16
        self.pose_b1_bootstrap_stable_ratio = 0.20
        self.pose_b1_bootstrap_max_active_ratio = 0.18
        self.pose_b1_bootstrap_min_capacity_ratio = 1.0
        self.pose_b1_bootstrap_update_interval = 10
        self.pose_b1_bootstrap_lr_scale = 0.35
        self.pose_b1_geometry_min_corr = 64
        self.pose_b1_geometry_min_corr_quality = 0.75
        self.pose_b1_geometry_min_in_frame_ratio = 0.20
        self.pose_b1_geometry_min_projected_ratio = 0.35
        self.pose_b1_geometry_max_drift_ratio = 0.02
        self.pose_b1_geometry_max_active_ratio = 0.18
        self.pose_b1_geometry_update_interval = 10
        self.pose_b1_geometry_lr_scale = 0.50
        self.pose_b1_corridor_min_corr = 24
        self.pose_b1_corridor_min_corr_quality = 0.35
        self.pose_b1_corridor_min_in_frame_ratio = 0.10
        self.pose_b1_corridor_min_projected_ratio = 0.15
        self.pose_b1_corridor_max_active_ratio = 0.24
        self.pose_b1_corridor_update_interval = 20
        self.pose_b1_corridor_lr_scale = 0.30
        self.pose_b1_success_min_px_reduction = 0.05
        self.pose_b1_no_improve_streak_for_slowdown = 3
        self.pose_b1_no_improve_interval_mult = 2.0
        self.pose_trust_min_scale = 0.35
        self.pose_trust_max_scale = 1.75
        self.pose_b2_trust_max_scale = 1.15
        self.pose_b2_corridor_trust_max_scale = 1.35
        self.pose_grad_norm_floor = 1e-8
        self.pose_trust_block_ratio = 2.0
        self.pose_exposure_mismatch_threshold = 0.04
        self.pose_enable_stable_ratio = 0.45
        self.pose_enable_max_drift_ratio = 0.01
        self.pose_enable_max_active_ratio = 0.12
        self.pose_enable_min_capacity_ratio = 1.25
        self.pose_b2_min_b1_updates = 2
        self.pose_b2_min_global_b1_updates = 2
        self.pose_b2_min_camera_b1_updates = 1
        self.pose_b2_update_interval = 15
        self.pose_b2_quality_update_interval = 10
        self.pose_b2_min_camera_b1_quality = 0.45
        self.pose_b2_max_camera_b1_median_px = 96.0
        self.pose_b2_bootstrap_after_iters = 50
        self.pose_b2_bootstrap_stable_ratio = 0.75
        self.pose_b2_bootstrap_max_active_ratio = 0.12
        self.pose_b2_b1_history_fresh_iters = 240
        self.pose_b2_max_quality_regression = 0.01
        self.pose_b2_max_drift_ratio = 0.01
        self.pose_b2_grad_norm_floor = 1e-10
        self.pose_b2_translation_grad_floor = 1e-10
        self.pose_b2_rotation_grad_floor = 1e-10
        self.pose_b2_early_grad_floor_scale = 0.10
        self.pose_b2_small_step_warmup_steps = 8
        self.pose_b2_photo_corridor_enabled = True
        self.pose_b2_photo_corridor_interval = 40
        self.pose_b2_photo_corridor_after_iters = 80
        self.pose_b2_photo_corridor_min_stable_ratio = 0.18
        self.pose_b2_photo_corridor_max_active_ratio = 0.22
        self.pose_b2_photo_corridor_min_quality_gap = 0.002
        self.pose_b2_photo_corridor_min_dark_l1 = 0.02
        self.pose_b2_photo_corridor_min_dark_completeness = 0.90
        self.pose_b2_photo_corridor_min_corr_ratio = 0.08
        self.pose_b2_corridor_step_floor_scale = 0.25
        self.pose_b2_min_mask_nonzero_ratio = 0.05
        self.pose_b2_force_step_if_photo_high_l1 = 0.015
        self.pose_b2_zero_grad_skip_threshold_rot = 1e-12
        self.pose_b2_zero_grad_skip_threshold_trans = 1e-12
        self.pose_b2_force_small_step_if_photo_high = True
        self.pose_b2_small_step_lr_scale = 0.25
        self.pose_b2_fullframe_stress_enable = False
        self.pose_b2_fullframe_stress_start_iter = 0
        self.pose_b2_fullframe_stress_end_iter = -1
        self.pose_b2_fullframe_downsample = 1
        self.pose_b2_fullframe_disable_patchfeat = True
        self.pose_b2_photo_corridor_max_drift_ratio = 0.01
        self.atlas_mean_weight_stress_enable = False
        self.atlas_mean_weight_stress_scale = 1.0
        self.atlas_mean_weight_stress_start_iter = 0
        self.atlas_mean_weight_stress_end_iter = -1
        self.pose_quality_ema_decay = 0.9
        self.pose_freeze_max_drift_ratio = 0.02
        self.pose_freeze_max_active_ratio = 0.15
        self.pose_freeze_bad_loss_iters = 3
        self.pose_freeze_quality_regression = 0.03
        self.pose_freeze_cooldown_iters = 50
        self.pose_freeze_recovery_good_iters = 5
        self.atlas_background_ref_score_min = 0.06
        self.atlas_background_visibility_min = 0.003
        self.atlas_background_dead_prune_guard = True
        self.atlas_background_soft_prune_guard = True
        self.atlas_late_phase_fidelity_boost = True
        self.random_background = False
        self.optimizer_type = "default"
        self.log_interval = 10
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
