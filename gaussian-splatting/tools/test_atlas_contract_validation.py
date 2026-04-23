import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.foundation_atlas import load_foundation_atlas  # noqa: E402
from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402


def main():
    tmp_root = REPO_ROOT / ".tmp_atlas_contract_validation"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")

    try:
        scene_alignment = {
            "schema_version": 1,
            "applied": True,
            "contract_validation": {
                "schema_version": 1,
                "passed": False,
                "thresholds": {
                    "max_camera_center_rmse": 0.05,
                    "max_rotation_error_deg": 1.0,
                    "max_median_px_error": 12.0,
                    "min_in_frame_corr": 256,
                },
                "failures": [
                    "scene_alignment.camera_center_rmse=0.100000 > 0.050000",
                ],
            },
        }
        with open(atlas_root / "scene_alignment.json", "w", encoding="utf-8") as handle:
            json.dump(scene_alignment, handle, indent=2)

        try:
            load_foundation_atlas(atlas_root)
        except ValueError as exc:
            message = str(exc)
            assert "contract validation failed" in message.lower()
            assert "camera_center_rmse" in message
        else:
            raise AssertionError("Expected load_foundation_atlas() to reject a failed scene alignment contract.")

        scene_alignment["contract_validation"]["passed"] = True
        scene_alignment["contract_validation"]["failures"] = []
        with open(atlas_root / "scene_alignment.json", "w", encoding="utf-8") as handle:
            json.dump(scene_alignment, handle, indent=2)

        atlas_init = load_foundation_atlas(atlas_root)
        assert atlas_init.positions.shape[0] > 0
        print("[OK] Atlas contract validation check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
