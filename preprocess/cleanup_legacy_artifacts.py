import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from migrate_layout import ARTIFACTS, REPO_ROOT


LEGACY_CODE_FILES = [
    ("hamer_tactile_ft/extract_full_dataset.py", False),
    ("hamer_tactile_ft/extract_test_dataset.py", False),
    ("hamer_tactile_ft/extract_touchanything_dataset.py", False),
    ("hamer_tactile_ft/extract_egotactile_dataset.py", False),
    ("hamer_tactile_ft/add_continuous_pressure_fixed.py", False),
    ("scratch/repair_touchanything_pressure_npz.py", False),
    ("scratch/process_ta_gaussian.py", False),
    ("scratch/process_egotactile_gaussian.py", False),
    ("scratch/compare_pressure_dist.py", False),
    ("scratch/check_egotactile.py", False),
    ("scratch/check_egotactile_dist.py", False),
    ("scratch/check_egotouch.py", False),
    ("scratch/check_hdf5_pressure.py", False),
    ("scratch/check_mapping.py", False),
    ("scratch/check_nan_npz.py", False),
    ("scratch/check_opentouch_raw.py", False),
    ("scratch/clean_corrupted_json.py", False),
    ("scratch/visualize_egotactile_ta_mapping.py", False),
    ("scratch/visualize_touchanything_ta_mapping.py", False),
]


def remove_path(path):
    if path.is_dir() and not path.is_symlink():
        import shutil

        shutil.rmtree(path)
    else:
        path.unlink()


def cleanup_artifacts(repo_root, args):
    print("\n[Artifacts]")
    for old_rel, new_rel in ARTIFACTS:
        old_path = repo_root / old_rel
        new_path = repo_root / new_rel
        if not old_path.exists() and not old_path.is_symlink():
            print(f"{'missing':>18}: {old_rel}")
            continue
        if old_path.is_symlink():
            target_exists = new_path.exists()
            print(f"{'remove_link':>18}: {old_rel} -> {old_path.readlink()} (new exists={target_exists})")
            if args.apply:
                old_path.unlink()
            continue
        if args.delete_real_artifacts:
            print(f"{'delete_real':>18}: {old_rel} (new exists={new_path.exists()})")
            if args.apply:
                remove_path(old_path)
        else:
            print(f"{'keep_real':>18}: {old_rel} (pass --delete_real_artifacts to remove)")


def cleanup_code(repo_root, args):
    print("\n[Legacy code]")
    for rel_path, still_used_by_wrapper in LEGACY_CODE_FILES:
        path = repo_root / rel_path
        if not path.exists() and not path.is_symlink():
            print(f"{'missing':>18}: {rel_path}")
            continue
        if still_used_by_wrapper and not args.delete_dependent_code:
            print(f"{'keep_dependent':>18}: {rel_path} (pass --delete_dependent_code to remove)")
            continue
        print(f"{'delete_code':>18}: {rel_path}")
        if args.apply:
            remove_path(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Clean legacy preprocessing artifacts and, optionally, legacy preprocessing code files. "
            "Default mode is dry-run and conservative."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Actually remove paths. Default is dry-run.")
    parser.add_argument(
        "--delete_real_artifacts",
        action="store_true",
        help="Also delete real legacy artifact files/dirs. Symlinks are removed without this flag.",
    )
    parser.add_argument(
        "--include_code",
        action="store_true",
        help="Also clean legacy preprocessing code files listed in this script.",
    )
    parser.add_argument(
        "--delete_dependent_code",
        action="store_true",
        help=(
            "Kept for compatibility. preprocess is now self-contained, so legacy preprocess code is no longer dependent."
        ),
    )
    parser.add_argument("--repo_root", default=REPO_ROOT)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    print(f"Repo root: {repo_root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    if args.include_code and args.delete_dependent_code:
        print("Note: preprocess is self-contained; this flag is no longer required for listed legacy code.")

    cleanup_artifacts(repo_root, args)
    if args.include_code:
        cleanup_code(repo_root, args)
    else:
        print("\n[Legacy code]")
        print("skip_code_cleanup: pass --include_code to show/remove old preprocessing code files")

    print("Done.")
    if not args.apply:
        print("Re-run with --apply to perform cleanup.")


if __name__ == "__main__":
    main()
