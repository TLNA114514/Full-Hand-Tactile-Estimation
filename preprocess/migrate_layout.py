import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


ARTIFACTS = [
    ("hamer_tactile_ft/full_bboxes_cache", "preprocess/artifacts/opentouch/full_bboxes_cache"),
    ("hamer_tactile_ft/test_bboxes_cache", "preprocess/artifacts/opentouch/test_bboxes_cache"),
    ("hamer_tactile_ft/opentouch_all_bboxes.json", "preprocess/artifacts/opentouch/opentouch_all_bboxes.json"),
    ("hamer_tactile_ft/opentouch_test_bboxes.json", "preprocess/artifacts/opentouch/opentouch_test_bboxes.json"),
    ("hamer_tactile_ft/dataset_frames_registry.json", "preprocess/artifacts/opentouch/dataset_frames_registry.json"),
    ("hamer_tactile_ft/touchanything_bboxes_cache", "preprocess/artifacts/touchanything/bboxes_cache"),
    ("hamer_tactile_ft/touchanything_all_bboxes.json", "preprocess/artifacts/touchanything/touchanything_all_bboxes.json"),
    ("hamer_tactile_ft/egotactile_bboxes_cache", "preprocess/artifacts/egotactile/bboxes_cache"),
    ("hamer_tactile_ft/egotactile_all_bboxes.json", "preprocess/artifacts/egotactile/egotactile_all_bboxes.json"),
]


def describe_action(src, dst):
    if src.is_symlink():
        return "already_symlink"
    if not src.exists():
        return "missing_source"
    if dst.exists():
        return "destination_exists"
    return "move_and_link"


def move_and_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    rel_target = os.path.relpath(dst, src.parent)
    os.symlink(rel_target, src)


def link_existing(src, dst):
    rel_target = os.path.relpath(dst, src.parent)
    os.symlink(rel_target, src)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Move preprocessing cache/json artifacts into preprocess/artifacts and leave "
            "compatibility symlinks at the old hamer_tactile_ft paths."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Actually move files. Default is dry-run.")
    parser.add_argument(
        "--link_existing",
        action="store_true",
        help="If the new artifact already exists and the old path is missing, create a compatibility symlink.",
    )
    parser.add_argument("--repo_root", default=REPO_ROOT)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    print(f"Repo root: {repo_root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    for old_rel, new_rel in ARTIFACTS:
        src = repo_root / old_rel
        dst = repo_root / new_rel
        action = describe_action(src, dst)
        if action == "missing_source" and args.link_existing and dst.exists():
            action = "link_existing"
        print(f"{action:>18}: {old_rel} -> {new_rel}")
        if args.apply and action == "move_and_link":
            move_and_link(src, dst)
        elif args.apply and action == "link_existing":
            link_existing(src, dst)

    print("Done.")
    if not args.apply:
        print("Re-run with --apply to perform the migration.")


if __name__ == "__main__":
    main()
