import json
import shutil
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
GS_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(GS_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_ROOT))

from tools.test_atlas_backend_init import build_synthetic_atlas_run  # noqa: E402
from tools.validate_atlas_artifact import validate_atlas_artifact  # noqa: E402


def main():
    tmp_root = WORKSPACE_ROOT / ".tmp_validate_atlas_artifact"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    atlas_root = build_synthetic_atlas_run(tmp_root / "synthetic_atlas")
    try:
        report = validate_atlas_artifact(atlas_root)
        assert report["valid"]
        assert report["consumer_summary"] is not None
        assert report["consumer_summary"]["has_correspondence_manifest"]
        assert report["consumer_summary"]["reference_camera_source"] == "reference_camera_evidence.npz"
        print(json.dumps(report, indent=2))
        print("[OK] Atlas artifact validator check passed.")
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


if __name__ == "__main__":
    main()
