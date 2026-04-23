from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .validate_atlas_artifact import validate_atlas_artifact
except ImportError:  # pragma: no cover
    from validate_atlas_artifact import validate_atlas_artifact


REPO_ROOT = Path(__file__).resolve().parents[1]
MAST3R_BUILD_SCRIPT = REPO_ROOT / "mast3r" / "build_foundation_atlas.py"
WARNING = (
    "This wrapper runs the MASt3R dense replay foundation builder. "
    "It is NOT the current Kitchen12 VGGT/COLMAP sparse-track atlas pipeline. "
    "Use tools/build_vggt_colmap_atlas.py for the active VGGT sparse foundation branch."
)


def _resolve_output_dir(dataset_root: str | None, output_dir: str | None):
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    if dataset_root:
        return Path(dataset_root).expanduser().resolve() / "foundation_atlas"
    raise ValueError("Either --dataset_root or --output_dir must be provided.")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description=(
            "MASt3R dense replay Foundation Atlas wrapper. "
            "Do not use for the current Kitchen12 VGGT/COLMAP sparse-track atlas unless "
            "you intentionally want a dense-replay ablation."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--skip_validate", action="store_true")
    parser.add_argument("--validator_json_out", type=str, default="")
    parser.add_argument("--strict_validation", action=argparse.BooleanOptionalAction, default=True)
    args, passthrough = parser.parse_known_args(argv)

    builder_cmd = [sys.executable, str(MAST3R_BUILD_SCRIPT)]
    if args.dataset_root:
        builder_cmd.extend(["--dataset_root", args.dataset_root])
    if args.output_dir:
        builder_cmd.extend(["--output_dir", args.output_dir])
    builder_cmd.extend(passthrough)

    print(json.dumps({"stage": "warning", "message": WARNING}, indent=2))
    print(json.dumps({"stage": "build", "builder_mode": "mast3r_dense_replay_wrapper", "command": builder_cmd}, indent=2))
    result = subprocess.run(builder_cmd, cwd=str(REPO_ROOT), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    atlas_dir = _resolve_output_dir(args.dataset_root or None, args.output_dir or None)
    if args.skip_validate:
        print(json.dumps({"stage": "validate", "skipped": True, "atlas_dir": str(atlas_dir)}, indent=2))
        raise SystemExit(0)

    report = validate_atlas_artifact(atlas_dir, strict=bool(args.strict_validation))
    if args.validator_json_out:
        out_path = Path(args.validator_json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps({"stage": "validate", "atlas_dir": str(atlas_dir), "valid": bool(report["valid"])}, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
