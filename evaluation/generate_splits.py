#!/usr/bin/env python
import os
import json
import random
import h5py
from typing import Sequence, Tuple, Dict, List

def split_clips_into_train_val_test(
    clip_keys: Sequence[Tuple[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[str, str]]]:
    clips = list(clip_keys)
    if not clips:
        return {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    rng.shuffle(clips)
    n = len(clips)
    n_test = min(int(round(n * test_ratio)), n)
    n_val = min(int(round(n * val_ratio)), max(0, n - n_test))
    n_train = max(0, n - n_val - n_test)
    if n_train == 0:
        n_train = 1
        if n_val > n_test:
            n_val = max(0, n_val - 1)
        elif n_test > 0:
            n_test -= 1
    return {
        "train": clips[:n_train],
        "val": clips[n_train : n_train + n_val],
        "test": clips[n_train + n_val : n_train + n_val + n_test],
    }

def main():
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(eval_dir, "../opentouch/data"))
    output_path = os.path.join(eval_dir, "opentouch_splits.json")
    
    print(f"Scanning HDF5 files in: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return

    hdf5_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".hdf5")])
    print(f"Found {len(hdf5_files)} HDF5 files.")

    clip_keys = []
    for f_name in hdf5_files:
        f_path = os.path.join(data_dir, f_name)
        scene_name = os.path.splitext(f_name)[0]
        try:
            with h5py.File(f_path, "r") as f:
                if "data" in f:
                    clips_in_file = sorted(list(f["data"].keys()))
                    for clip_id in clips_in_file:
                        clip_keys.append((scene_name, clip_id))
        except Exception as e:
            print(f"Warning: Failed to read {f_name}: {e}")

    print(f"Total clips discovered: {len(clip_keys)}")
    if not clip_keys:
        print("No clips found. Cannot generate splits.")
        return

    # Deterministic sorting before shuffle
    clip_keys.sort()

    splits = split_clips_into_train_val_test(
        clip_keys=clip_keys,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )

    print("\nSplit statistics:")
    for split_name, items in splits.items():
        print(f"  {split_name.capitalize()}: {len(items)} clips ({len(items)/len(clip_keys)*100:.1f}%)")

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as json_f:
        json.dump(splits, json_f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully generated and saved splits to: {output_path}")

if __name__ == "__main__":
    main()
