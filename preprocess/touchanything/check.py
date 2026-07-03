import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from preprocess.check_utils import check_array_field, check_required_fields, file_nonempty, load_json


def sample_files_ok(sample_dir):
    sample_dir = Path(sample_dir)
    required = ["chest.jpg", "left.jpg", "right.jpg", "meta.json"]
    missing = [name for name in required if not file_nonempty(sample_dir / name)]
    return missing


def check_sample(item, strict):
    sample_dir = item.get("sample_dir")
    if not sample_dir:
        return "missing sample_dir in registry"
    sample_dir = Path(sample_dir)
    missing = sample_files_ok(sample_dir)
    if missing:
        return f"missing/empty files: {missing}"
    try:
        meta = load_json(sample_dir / "meta.json")
    except Exception as exc:
        return f"bad meta.json: {exc}"
    if not isinstance(meta, dict):
        return "meta.json root is not object"

    missing_fields = check_required_fields(meta, ["dataset", "frame_idx", "views", "hands"])
    if missing_fields:
        return f"missing meta fields: {missing_fields}"
    if meta.get("dataset") != "TouchAnything":
        return f"dataset mismatch: {meta.get('dataset')}"
    hands = meta.get("hands")
    if not isinstance(hands, dict):
        return "hands is not object"
    for hand in ("left", "right"):
        hand_meta = hands.get(hand)
        if not isinstance(hand_meta, dict):
            return f"hands.{hand} missing/not object"
        err = check_array_field(
            hand_meta,
            "normalized_pressure_grid",
            expected_shape=(21, 21),
            min_value=0.0,
            max_value=1.0,
            allow_none=not meta.get("pressure_npz_available", True),
        )
        if err:
            return f"hands.{hand}.{err}"
        err = check_array_field(
            hand_meta,
            "gaussian_pressure",
            min_value=0.0,
            max_value=1.0,
            allow_none=not meta.get("pressure_npz_available", True),
        )
        if err:
            return f"hands.{hand}.{err}"
    if strict:
        try:
            frame_idx = int(meta["frame_idx"])
            if frame_idx != int(item["frame_idx"]):
                return f"frame_idx mismatch: meta={frame_idx}, registry={item['frame_idx']}"
        except Exception as exc:
            return f"bad frame_idx: {exc}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Check TouchAnything extracted-frame registry.")
    parser.add_argument("--output_dir", default=None, help="Extracted frames dir. Defaults to --registry_json parent.")
    parser.add_argument(
        "--registry_json",
        default=None,
        help="Registry path. Defaults to <output_dir>/touchanything_frames_registry.json.",
    )
    parser.add_argument("--strict", action="store_true", help="Also check registry/meta consistency.")
    parser.add_argument("--workers", type=int, default=32, help="Threads for checking samples.")
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    if args.registry_json is None:
        output_dir = Path(args.output_dir) if args.output_dir else Path("/data1/jiangrui/EgoTouch/extracted_frames")
        registry_path = output_dir / "touchanything_frames_registry.json"
    else:
        registry_path = Path(args.registry_json)

    with registry_path.open("r", encoding="utf-8") as f:
        registry = json.load(f)

    bad = []
    bad_pressure = 0
    check_workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=check_workers) as executor:
        results = executor.map(lambda item: (item, check_sample(item, args.strict)), registry)
        iterator = tqdm(results, total=len(registry), desc="Checking TouchAnything samples")
        for item, error in iterator:
            if item.get("pressure_npz_available") is False:
                bad_pressure += 1
            if error and len(bad) < args.max_errors:
                bad.append(f"{item.get('sample_dir')}: {error}")

    print("TouchAnything check finished.")
    print(f"  registry entries: {len(registry)}")
    print(f"  entries with missing/bad pressure npz: {bad_pressure}")
    print(f"  first bad samples: {len(bad)}")
    for item in bad:
        print(f"    {item}")


if __name__ == "__main__":
    main()
