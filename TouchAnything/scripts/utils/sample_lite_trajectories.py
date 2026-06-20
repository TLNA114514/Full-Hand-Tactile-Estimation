#!/usr/bin/env python3
"""
Lite inference: select one sample from each task

Usage:
    python scripts/utils/sample_lite_trajectories.py \
        --split_file datasets/TouchAnything_hdf5_clean/split.json \
        --split test_seen \
        --output /tmp/lite_trajectories.txt
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def extract_task_name(hdf5_path):
    """
    Extract the task name from an HDF5 path
    
    Examples:
        /path/to/Home/push_cart/20260319_211038_740.hdf5 -> push_cart
        /path/to/Lab/open_door/20260320_123456_789.hdf5 -> open_door
    """
    path = Path(hdf5_path)
    # The second-to-last directory name is the task name
    if len(path.parts) >= 2:
        return path.parts[-2]
    return "unknown"


def sample_lite_trajectories(split_file, split_name):
    """
    Select one sample per task from the specified split
    
    Args:
        split_file: split.json file path
        split_name: split name, e.g. test_seen or test_unseen
    
    Returns:
        list: lite trajectory path list
    """
    # Read split file
    with open(split_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    if split_name not in splits:
        raise ValueError(f"Split '{split_name}' not found in {split_file}")
    
    trajectories = splits[split_name]
    
    # Group by task
    task_groups = defaultdict(list)
    for traj_path in trajectories:
        task_name = extract_task_name(traj_path)
        task_groups[task_name].append(traj_path)
    
    # Select the first sample from each task
    lite_trajectories = []
    for task_name in sorted(task_groups.keys()):
        # Select the first sample
        selected = task_groups[task_name][0]
        lite_trajectories.append(selected)
        print(f"Task '{task_name}': selected 1/{len(task_groups[task_name])} samples")
    
    print(f"\nTotal: {len(lite_trajectories)} tasks, one sample per task")
    print(f"Original sample count: {len(trajectories)}")
    print(f"Lite sample count: {len(lite_trajectories)}")
    print(f"Reduction ratio: {(1 - len(lite_trajectories)/len(trajectories))*100:.1f}%")
    
    return lite_trajectories


def main():
    parser = argparse.ArgumentParser(description='Select one sample from each task for lite inference')
    parser.add_argument('--split_file', type=str, required=True,
                        help='split.json file path')
    parser.add_argument('--split', type=str, required=True,
                        help='split name, e.g. test_seen or test_unseen')
    parser.add_argument('--output', type=str, default=None,
                        help='output file path; optional, prints to screen if omitted')
    
    args = parser.parse_args()
    
    # Sample lite trajectories
    lite_trajectories = sample_lite_trajectories(args.split_file, args.split)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for traj in lite_trajectories:
                f.write(traj + '\n')
        print(f"\n✓ Lite trajectory list saved to: {args.output}")
    else:
        print("\nLite trajectory list:")
        for traj in lite_trajectories:
            print(f"  {traj}")


if __name__ == '__main__':
    main()
