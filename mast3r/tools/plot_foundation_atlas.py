import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from AURORA.mast3r.mast3r.foundation_atlas import load_atlas_npz, plot_foundation_atlas_report


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots for a saved Foundation Geometry Atlas.")
    parser.add_argument("--atlas_npz", type=str, required=True, help="Path to atlas_nodes.npz")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for the regenerated plots")
    parser.add_argument("--summary_json", type=str, default="", help="Optional atlas_summary.json to annotate the plots")
    parser.add_argument("--title", type=str, default="", help="Optional title override for the generated plots")
    args = parser.parse_args()

    atlas_path = Path(args.atlas_npz).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else atlas_path.parent
    summary_path = Path(args.summary_json).resolve() if args.summary_json else None

    atlas = load_atlas_npz(atlas_path)
    summary = None
    if summary_path:
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

    title = args.title if args.title else atlas_path.parent.name
    plot_foundation_atlas_report(atlas, output_dir, summary=summary, title=title)
    print(f"[DONE] Atlas plots written to {output_dir}")


if __name__ == "__main__":
    main()
