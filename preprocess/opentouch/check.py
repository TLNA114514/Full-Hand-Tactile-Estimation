import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from preprocess.check_utils import (
    check_array_field,
    check_required_fields,
    file_nonempty,
    load_json,
    resolve_sample_dir_from_opentouch,
)
from preprocess.common import ARTIFACT_ROOT, REPO_ROOT


def check_sample(item, output_dir, strict):
    sample_dir = resolve_sample_dir_from_opentouch(item, output_dir)
    if sample_dir is None:
        return "cannot resolve sample_dir"
    missing = [name for name in ("image.jpg", "meta.json") if not file_nonempty(sample_dir / name)]
    if missing:
        return f"missing/empty files: {missing}"
    try:
        meta = load_json(sample_dir / "meta.json")
    except Exception as exc:
        return f"bad meta.json: {exc}"
    if not isinstance(meta, dict):
        return "meta.json root is not object"
    missing_fields = check_required_fields(meta, ["scene", "demo", "frame_idx", "is_right", "bbox"])
    if missing_fields:
        return f"missing meta fields: {missing_fields}"
    err = check_array_field(meta, "keypoints_3d_cam", expected_shape=(21, 3), allow_none=False)
    if err:
        return err
    valid_mask = meta.get("valid_mask")
    if not isinstance(valid_mask, list) or len(valid_mask) != 21:
        return "valid_mask shape is not (21,)"
    original = meta.get("original_hdf5_data")
    if not isinstance(original, dict):
        return "original_hdf5_data missing/not object"
    pressure_keys = [key for key in original if "pressure" in key]
    if not pressure_keys:
        return "no pressure fields in original_hdf5_data"
    if strict:
        try:
            if int(meta["frame_idx"]) != int(item["frame_idx"]):
                return f"frame_idx mismatch: meta={meta['frame_idx']}, registry={item['frame_idx']}"
            if int(meta["is_right"]) != int(item["is_right"]):
                return f"is_right mismatch: meta={meta['is_right']}, registry={item['is_right']}"
        except Exception as exc:
            return f"bad strict fields: {exc}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Check OpenTouch extracted-frame registry.")
    artifact_registry = ARTIFACT_ROOT / "opentouch/dataset_frames_registry.json"
    legacy_registry = REPO_ROOT / "hamer_tactile_ft/dataset_frames_registry.json"
    default_registry = artifact_registry if artifact_registry.exists() else legacy_registry
    parser.add_argument("--registry_json", default=default_registry)
    parser.add_argument("--output_dir", default="/data1/jiangrui/OpenTouch Data/full_dataset")
    parser.add_argument("--strict", action="store_true", help="Also check registry/meta consistency.")
    parser.add_argument("--workers", type=int, default=32, help="Threads for checking samples.")
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    registry_path = Path(args.registry_json)
    registry = load_json(registry_path)

    bad = []
    check_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=check_workers) as executor:
        results = executor.map(
            lambda item: (item, check_sample(item, args.output_dir, args.strict)),
            registry,
        )
        iterator = tqdm(results, total=len(registry), desc="Checking OpenTouch samples")
        for item, error in iterator:
            if error and len(bad) < args.max_errors:
                sample_dir = resolve_sample_dir_from_opentouch(item, args.output_dir)
                bad.append(f"{sample_dir}: {error}")

    print("OpenTouch check finished.")
    print(f"  registry entries: {len(registry)}")
    print(f"  first bad samples: {len(bad)}")
    for item in bad:
        print(f"    {item}")


if __name__ == "__main__":
    main()
