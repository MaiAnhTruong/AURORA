import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AURORA.mast3r.mast3r.foundation_atlas import backfill_foundation_atlas_sidecars  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Backfill standardized sidecars for an existing Foundation Geometry Atlas without replaying MASt3R."
    )
    parser.add_argument("--atlas_dir", type=str, required=True, help="Existing atlas directory containing atlas_nodes.npz")
    parser.add_argument("--camera_bundle_path", type=str, default="", help="Optional path to an existing camera_bundle.json")
    parser.add_argument("--image_dir", type=str, default="", help="Optional source image directory used to populate source_image_paths")
    parser.add_argument("--min_conf_thr", type=float, default=None, help="Optional override for confidence threshold used in correspondences")
    parser.add_argument("--hash_cell_size", type=float, default=None, help="Optional override for atlas_hash voxel cell size")
    parser.add_argument("--skip_update_summary", action="store_true", help="Do not merge backfill metadata into atlas_summary.json")
    args = parser.parse_args()

    summary = backfill_foundation_atlas_sidecars(
        atlas_dir=args.atlas_dir,
        camera_bundle_path=args.camera_bundle_path or None,
        image_dir=args.image_dir or None,
        min_conf_thr=args.min_conf_thr,
        hash_cell_size=args.hash_cell_size,
        update_summary=not args.skip_update_summary,
    )
    print(json.dumps(summary, indent=2))
    print("[DONE] Atlas sidecars backfilled.")


if __name__ == "__main__":
    main()
