from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _json_load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(root: Path, value):
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _is_atlas_root(path: Path) -> bool:
    return (path / "atlas_nodes.npz").exists() and (path / "camera_bundle.json").exists()


def _resolve_atlas_root(atlas_path: str | Path) -> Path:
    candidate = Path(atlas_path).expanduser().resolve()
    if _is_atlas_root(candidate):
        return candidate
    if _is_atlas_root(candidate.parent):
        return candidate.parent
    raise FileNotFoundError(
        "Expected an atlas artifact directory containing atlas_nodes.npz and camera_bundle.json: "
        f"{candidate}"
    )


def _append_shape_error(errors: list[str], key: str, shape, expected: str):
    errors.append(f"atlas_nodes.npz:{key} has shape {tuple(shape)}, expected {expected}")


def _validate_scene_alignment(root: Path, errors: list[str], warnings: list[str]):
    path = root / "scene_alignment.json"
    if not path.exists():
        warnings.append("scene_alignment.json missing")
        return None
    payload = _json_load(path)
    contract = payload.get("contract_validation")
    if isinstance(contract, dict) and not bool(contract.get("passed", False)):
        failures = contract.get("failures", []) or []
        errors.append("scene_alignment contract failed: " + ("; ".join(map(str, failures)) or "unknown"))
    return payload


def _validate_camera_bundle(root: Path, errors: list[str]):
    path = root / "camera_bundle.json"
    if not path.exists():
        errors.append("camera_bundle.json missing")
        return None
    payload = _json_load(path)
    image_names = payload.get("image_names", []) or []
    if not isinstance(image_names, list) or len(image_names) == 0:
        errors.append("camera_bundle.json:image_names is empty")
    num_cameras = int(payload.get("num_cameras", len(image_names)) or 0)
    if image_names and num_cameras != len(image_names):
        errors.append(f"camera_bundle.json:num_cameras={num_cameras} but image_names has {len(image_names)} entries")
    return payload


def _validate_manifest(root: Path, errors: list[str], warnings: list[str], strict: bool):
    path = root / "correspondence_manifest.json"
    if not path.exists():
        message = "correspondence_manifest.json missing"
        (errors if strict else warnings).append(message)
        return None

    payload = _json_load(path)
    views = payload.get("views", {}) or {}
    if not isinstance(views, dict) or len(views) == 0:
        errors.append("correspondence_manifest.json:views is empty")
        return payload

    missing_sparse = 0
    malformed_sparse = 0
    sparse_count = 0
    for image_name, view in views.items():
        if not isinstance(view, dict):
            errors.append(f"correspondence_manifest view {image_name!r} is not an object")
            continue
        sparse_path = _resolve_path(root, view.get("sparse_path"))
        if sparse_path is None:
            continue
        if not sparse_path.exists():
            missing_sparse += 1
            continue
        try:
            with np.load(sparse_path) as sparse:
                for key in ("xy", "xyz", "confidence"):
                    if key not in sparse:
                        malformed_sparse += 1
                        break
                else:
                    xy = np.asarray(sparse["xy"])
                    xyz = np.asarray(sparse["xyz"])
                    confidence = np.asarray(sparse["confidence"])
                    if xy.ndim != 2 or xy.shape[1] != 2 or xyz.ndim != 2 or xyz.shape[1] != 3:
                        malformed_sparse += 1
                    elif xy.shape[0] != xyz.shape[0] or xy.shape[0] != confidence.reshape(-1).shape[0]:
                        malformed_sparse += 1
                    else:
                        sparse_count += int(xy.shape[0])
        except Exception as exc:
            malformed_sparse += 1
            warnings.append(f"could not read sparse correspondence file {sparse_path}: {exc}")

    if missing_sparse:
        errors.append(f"{missing_sparse} sparse correspondence files referenced by manifest are missing")
    if malformed_sparse:
        errors.append(f"{malformed_sparse} sparse correspondence files are malformed")
    payload["_validated_sparse_correspondence_count"] = sparse_count
    return payload


def _validate_reference_evidence(root: Path, node_count: int, errors: list[str], warnings: list[str], strict: bool):
    npz_path = root / "reference_camera_evidence.npz"
    json_path = root / "reference_camera_evidence.json"
    if npz_path.exists():
        try:
            with np.load(npz_path) as payload:
                ids = np.asarray(payload["reference_camera_ids"]).reshape(-1)
                scores = np.asarray(payload["reference_camera_scores"]).reshape(-1)
            if ids.shape[0] != node_count or scores.shape[0] != node_count:
                errors.append(
                    "reference_camera_evidence.npz vectors must match node count "
                    f"({ids.shape[0]}, {scores.shape[0]} vs {node_count})"
                )
            return "reference_camera_evidence.npz", float(np.mean(ids >= 0)) if ids.size else 0.0
        except Exception as exc:
            errors.append(f"reference_camera_evidence.npz could not be read: {exc}")
            return "reference_camera_evidence.npz", 0.0
    if json_path.exists():
        payload = _json_load(json_path)
        ids = np.asarray(payload.get("reference_camera_ids", []), dtype=np.int64).reshape(-1)
        if ids.shape[0] != node_count:
            errors.append(f"reference_camera_evidence.json ids length {ids.shape[0]} does not match node count {node_count}")
        return "reference_camera_evidence.json", float(np.mean(ids >= 0)) if ids.size else 0.0

    message = "reference camera evidence missing"
    (errors if strict else warnings).append(message)
    return "missing", 0.0


def validate_atlas_artifact(atlas_path: str | Path, strict: bool = False):
    errors: list[str] = []
    warnings: list[str] = []
    root = _resolve_atlas_root(atlas_path)

    required_files = ["atlas_nodes.npz", "camera_bundle.json"]
    if strict:
        required_files.extend(["correspondence_manifest.json", "reference_camera_evidence.npz"])
    for name in required_files:
        if not (root / name).exists():
            errors.append(f"{name} missing")

    node_count = 0
    archive_keys = []
    archive_path = root / "atlas_nodes.npz"
    if archive_path.exists():
        try:
            with np.load(archive_path) as archive:
                archive_keys = list(archive.files)
                required_keys = {
                    "positions": "Nx3",
                    "support": "Nx3x3",
                    "basis": "Nx3x3",
                    "normal": "Nx3",
                    "radius": "N",
                    "reliability": "N",
                    "atlas_class": "N",
                    "anisotropy_ref": "Nx2",
                }
                missing = [key for key in required_keys if key not in archive]
                if missing:
                    errors.append("atlas_nodes.npz missing keys: " + ", ".join(missing))
                if "positions" in archive:
                    positions = np.asarray(archive["positions"])
                    if positions.ndim != 2 or positions.shape[1] != 3:
                        _append_shape_error(errors, "positions", positions.shape, "Nx3")
                    node_count = int(positions.shape[0]) if positions.ndim >= 1 else 0
                    if node_count <= 0:
                        errors.append("atlas_nodes.npz:positions is empty")
                    if not np.isfinite(positions).all():
                        errors.append("atlas_nodes.npz:positions contains non-finite values")
                for key in ("normal",):
                    if key in archive and node_count > 0:
                        value = np.asarray(archive[key])
                        if value.shape != (node_count, 3):
                            _append_shape_error(errors, key, value.shape, "Nx3")
                for key in ("support", "basis"):
                    if key in archive and node_count > 0:
                        value = np.asarray(archive[key])
                        if value.shape != (node_count, 3, 3):
                            _append_shape_error(errors, key, value.shape, "Nx3x3")
                for key in ("radius", "reliability", "atlas_class"):
                    if key in archive and node_count > 0:
                        value = np.asarray(archive[key]).reshape(-1)
                        if value.shape[0] != node_count:
                            _append_shape_error(errors, key, value.shape, "N")
                if "anisotropy_ref" in archive and node_count > 0:
                    value = np.asarray(archive["anisotropy_ref"])
                    if value.shape[0] != node_count:
                        _append_shape_error(errors, "anisotropy_ref", value.shape, "NxK")
        except Exception as exc:
            errors.append(f"atlas_nodes.npz could not be read: {exc}")

    camera_bundle = _validate_camera_bundle(root, errors)
    manifest = _validate_manifest(root, errors, warnings, strict)
    _validate_scene_alignment(root, errors, warnings)
    reference_source, reference_coverage = _validate_reference_evidence(root, node_count, errors, warnings, strict)

    camera_count = 0
    if isinstance(camera_bundle, dict):
        camera_count = int(camera_bundle.get("num_cameras", len(camera_bundle.get("image_names", []) or [])) or 0)
    sparse_count = 0
    if isinstance(manifest, dict):
        sparse_count = int(manifest.get("_validated_sparse_correspondence_count", 0) or 0)

    consumer_summary = {
        "node_count": int(node_count),
        "archive_keys": archive_keys,
        "has_camera_bundle": camera_bundle is not None,
        "camera_count": int(camera_count),
        "has_correspondence_manifest": manifest is not None,
        "correspondence_view_count": 0 if not isinstance(manifest, dict) else int(len(manifest.get("views", {}) or {})),
        "sparse_correspondence_count": int(sparse_count),
        "reference_camera_source": reference_source,
        "reference_camera_coverage": float(reference_coverage),
    }
    return {
        "valid": len(errors) == 0,
        "strict": bool(strict),
        "atlas_path": str(root),
        "errors": errors,
        "warnings": warnings,
        "consumer_summary": consumer_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a Foundation Atlas artifact.")
    parser.add_argument("atlas_path")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = validate_atlas_artifact(args.atlas_path, strict=args.strict)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
