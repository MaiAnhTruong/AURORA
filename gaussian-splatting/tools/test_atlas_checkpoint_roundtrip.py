import json
import shutil
import sys
import types
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "diff_gaussian_rasterization" not in sys.modules:
    stub = types.ModuleType("diff_gaussian_rasterization")

    class _DummyRasterizationSettings:
        def __init__(self, *args, **kwargs):
            pass

    class _DummyRasterizer:
        def __init__(self, *args, **kwargs):
            pass

    stub.GaussianRasterizationSettings = _DummyRasterizationSettings
    stub.GaussianRasterizer = _DummyRasterizer
    sys.modules["diff_gaussian_rasterization"] = stub

from scene.foundation_atlas import GAUSSIAN_STATE_UNSTABLE_ACTIVE, load_foundation_atlas  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from train import load_training_checkpoint, prepare_output_and_logger, save_training_checkpoint  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


def build_training_args():
    return SimpleNamespace(
        percent_dense=1.0,
        position_lr_init=1e-4,
        position_lr_final=1e-6,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=1000,
        feature_lr=1e-3,
        opacity_lr=1e-2,
        scaling_lr=1e-3,
        rotation_lr=1e-3,
        center_uncertainty_lr=1e-3,
        exposure_lr_init=1e-2,
        exposure_lr_final=1e-3,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        iterations=1000,
    )


class DummyScene:
    def __init__(self, pose_state):
        self._pose_state = {
            name: {
                "pose_delta_q": list(entry["pose_delta_q"]),
                "pose_delta_t": list(entry["pose_delta_t"]),
            }
            for name, entry in pose_state.items()
        }

    def export_pose_state(self, scale=1.0):
        return {
            name: {
                "pose_delta_q": list(entry["pose_delta_q"]),
                "pose_delta_t": list(entry["pose_delta_t"]),
            }
            for name, entry in self._pose_state.items()
        }

    def apply_pose_state(self, payload, scale=1.0):
        self._pose_state = {
            name: {
                "pose_delta_q": list(entry["pose_delta_q"]),
                "pose_delta_t": list(entry["pose_delta_t"]),
            }
            for name, entry in payload.items()
        }

    def get_pose_camera_order(self, scale=1.0):
        return list(self._pose_state.keys())


def populate_optimizer_state(gm):
    loss = gm.get_xyz.sum() + gm.get_exposure.sum()
    loss.backward()
    gm.optimizer.step()
    gm.exposure_optimizer.step()
    gm.optimizer.zero_grad(set_to_none=True)
    gm.exposure_optimizer.zero_grad(set_to_none=True)


def populate_pose_optimizer_state(pose_optimizer, pose_params):
    loss = sum(param.square().sum() for param in pose_params)
    loss.backward()
    pose_optimizer.step()
    pose_optimizer.zero_grad(set_to_none=True)


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_roundtrip"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        atlas_init = load_foundation_atlas(atlas_root)
        gm = GaussianModel(sh_degree=0)
        cams = [SimpleNamespace(image_name="cam_0"), SimpleNamespace(image_name="cam_1")]
        gm.create_from_atlas(atlas_init, cams, spatial_lr_scale=1.0)
        gm.training_setup(build_training_args())
        populate_optimizer_state(gm)

        with torch.no_grad():
            gm._atlas_state[-1] = GAUSSIAN_STATE_UNSTABLE_ACTIVE
            gm._atlas_visibility_ema[:] = 0.7
            gm._atlas_node_photo_ema[:] = torch.tensor([0.05, 0.10, 0.15, 0.20], dtype=torch.float32, device=gm.get_xyz.device)
            gm._atlas_node_visibility_ema[:] = torch.tensor([1.0, 1.0, 0.3, 0.0], dtype=torch.float32, device=gm.get_xyz.device)
            gm._atlas_refresh_observed_mask[:] = torch.tensor([True, True, False, False], dtype=torch.bool, device=gm.get_xyz.device)
            gm._atlas_refresh_node_photo_ema.copy_(gm._atlas_node_photo_ema)
            gm._atlas_refresh_node_visibility_ema.copy_(gm._atlas_node_visibility_ema)
            gm._atlas_refresh_obs_quality[:] = torch.tensor([0.9, 0.7, 0.0, 0.0], dtype=torch.float32, device=gm.get_xyz.device)
            gm._atlas_refresh_done = True
            gm._atlas_ref_camera[:] = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=gm.get_xyz.device)
            gm._atlas_ref_score[:] = 0.5
            gm._center_log_sigma_parallel += 0.1
            gm._center_log_sigma_support -= 0.2
            gm._exposure[0, 0, 3] = 1.2345
            gm._exposure[1, 1, 3] = -0.4321

        payload = gm.capture()
        gm_roundtrip = GaussianModel(sh_degree=0)
        gm_roundtrip.restore(payload, build_training_args())

        assert gm_roundtrip.get_xyz.shape == gm.get_xyz.shape
        assert torch.equal(gm_roundtrip.get_atlas_state, gm.get_atlas_state)
        assert torch.equal(gm_roundtrip.get_atlas_ref_camera, gm.get_atlas_ref_camera)
        assert torch.allclose(gm_roundtrip.get_atlas_view_weights, gm.get_atlas_view_weights)
        assert torch.equal(gm_roundtrip.get_atlas_view_counts, gm.get_atlas_view_counts)
        assert torch.equal(gm_roundtrip._atlas_refresh_observed_mask, gm._atlas_refresh_observed_mask)
        assert torch.allclose(gm_roundtrip._atlas_refresh_node_photo_ema, gm._atlas_refresh_node_photo_ema)
        assert torch.allclose(gm_roundtrip._atlas_refresh_node_visibility_ema, gm._atlas_refresh_node_visibility_ema)
        assert torch.allclose(gm_roundtrip._atlas_refresh_obs_quality, gm._atlas_refresh_obs_quality)
        assert torch.allclose(gm_roundtrip.get_center_sigma_parallel, gm.get_center_sigma_parallel)
        assert torch.allclose(gm_roundtrip.get_center_sigma_support, gm.get_center_sigma_support)
        assert torch.allclose(gm_roundtrip.get_exposure, gm.get_exposure)

        point_cloud_dir = tmp_root / "saved_iteration"
        gm.save_ply(point_cloud_dir / "point_cloud.ply")
        gm.save_atlas_state(point_cloud_dir / "atlas_state.npz")

        gm_iteration_roundtrip = GaussianModel(sh_degree=0)
        gm_iteration_roundtrip.load_ply(point_cloud_dir / "point_cloud.ply")
        gm_iteration_roundtrip.load_atlas_state(point_cloud_dir / "atlas_state.npz")

        assert torch.allclose(gm_iteration_roundtrip.get_atlas_view_weights, gm.get_atlas_view_weights)
        assert torch.equal(gm_iteration_roundtrip.get_atlas_view_counts, gm.get_atlas_view_counts)
        assert torch.equal(gm_iteration_roundtrip._atlas_refresh_observed_mask, gm._atlas_refresh_observed_mask)
        assert torch.allclose(gm_iteration_roundtrip._atlas_refresh_node_photo_ema, gm._atlas_refresh_node_photo_ema)
        assert torch.allclose(gm_iteration_roundtrip._atlas_refresh_node_visibility_ema, gm._atlas_refresh_node_visibility_ema)
        assert torch.allclose(gm_iteration_roundtrip._atlas_refresh_obs_quality, gm._atlas_refresh_obs_quality)
        assert torch.allclose(gm_iteration_roundtrip.get_center_sigma_parallel, gm.get_center_sigma_parallel)
        assert torch.allclose(gm_iteration_roundtrip.get_center_sigma_support, gm.get_center_sigma_support)

        pose_state = {
            "cam_0": {"pose_delta_q": [0.99, 0.01, 0.0, 0.0], "pose_delta_t": [0.1, -0.2, 0.3]},
            "cam_1": {"pose_delta_q": [0.98, 0.02, 0.0, 0.0], "pose_delta_t": [-0.3, 0.4, -0.1]},
        }
        source_scene = DummyScene(pose_state)
        target_scene = DummyScene(
            {
                "cam_0": {"pose_delta_q": [1.0, 0.0, 0.0, 0.0], "pose_delta_t": [0.0, 0.0, 0.0]},
                "cam_1": {"pose_delta_q": [1.0, 0.0, 0.0, 0.0], "pose_delta_t": [0.0, 0.0, 0.0]},
            }
        )
        pose_params = [
            torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32, device=gm.get_xyz.device)),
            torch.nn.Parameter(torch.tensor([-0.5], dtype=torch.float32, device=gm.get_xyz.device)),
        ]
        pose_optimizer = torch.optim.Adam(pose_params, lr=1e-3)
        populate_pose_optimizer_state(pose_optimizer, pose_params)

        checkpoint_path = tmp_root / "training_checkpoint.pth"
        checkpoint_args = Namespace(
            model_path=str(tmp_root / "logged_run"),
            source_path="synthetic_source",
            atlas_path="synthetic_atlas",
            checkpoint_iterations=[17],
            save_iterations=[17],
            test_iterations=[17],
            debug_from=-1,
            iterations=1000,
        )
        save_training_checkpoint(
            source_scene,
            gm,
            pose_optimizer,
            17,
            checkpoint_path,
            run_args=checkpoint_args,
        )

        gm_checkpoint_roundtrip = GaussianModel(sh_degree=0)
        target_pose_params = [
            torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=gm.get_xyz.device)),
            torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=gm.get_xyz.device)),
        ]
        target_pose_optimizer = torch.optim.Adam(target_pose_params, lr=1e-3)
        restored_iter = load_training_checkpoint(
            checkpoint_path,
            target_scene,
            gm_checkpoint_roundtrip,
            build_training_args(),
            target_pose_optimizer,
        )

        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        assert restored_iter == 17
        assert target_scene.export_pose_state() == source_scene.export_pose_state()
        assert torch.allclose(gm_checkpoint_roundtrip.get_exposure, gm.get_exposure)
        assert torch.allclose(gm_checkpoint_roundtrip.get_atlas_view_weights, gm.get_atlas_view_weights)
        assert torch.equal(gm_checkpoint_roundtrip.get_atlas_view_counts, gm.get_atlas_view_counts)
        assert torch.equal(gm_checkpoint_roundtrip._atlas_refresh_observed_mask, gm._atlas_refresh_observed_mask)
        assert torch.allclose(gm_checkpoint_roundtrip._atlas_refresh_node_photo_ema, gm._atlas_refresh_node_photo_ema)
        assert torch.allclose(gm_checkpoint_roundtrip._atlas_refresh_node_visibility_ema, gm._atlas_refresh_node_visibility_ema)
        assert torch.allclose(gm_checkpoint_roundtrip._atlas_refresh_obs_quality, gm._atlas_refresh_obs_quality)
        assert torch.allclose(gm_checkpoint_roundtrip.get_center_sigma_parallel, gm.get_center_sigma_parallel)
        assert torch.allclose(gm_checkpoint_roundtrip.get_center_sigma_support, gm.get_center_sigma_support)
        assert len(gm_checkpoint_roundtrip.exposure_optimizer.state_dict()["state"]) > 0
        assert len(target_pose_optimizer.state_dict()["state"]) > 0
        assert checkpoint_payload["args"]["atlas_path"] == "synthetic_atlas"

        tb_writer = prepare_output_and_logger(checkpoint_args)
        if tb_writer is not None:
            tb_writer.close()
        cfg_args_path = Path(checkpoint_args.model_path) / "cfg_args"
        cfg_args_json_path = Path(checkpoint_args.model_path) / "cfg_args.json"
        assert cfg_args_path.exists()
        assert cfg_args_json_path.exists()
        assert "atlas_path='synthetic_atlas'" in cfg_args_path.read_text(encoding="utf-8")
        cfg_args_payload = json.loads(cfg_args_json_path.read_text(encoding="utf-8"))
        assert cfg_args_payload["atlas_path"] == "synthetic_atlas"
        assert cfg_args_payload["checkpoint_iterations"] == [17]
        print("[OK] Atlas checkpoint roundtrip, iteration persistence, and config logging checks passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
