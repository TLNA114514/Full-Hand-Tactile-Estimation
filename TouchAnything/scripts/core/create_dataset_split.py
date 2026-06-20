#!/usr/bin/env python3
"""
Automatic dataset split generation with test_seen/test_unseen support.

Features:
1. Scan the HDF5 dataset and run quality checks via ``check_hdf5_quality.py``.
2. Split data into train/val/test_seen/test_unseen at the task level.
3. test_seen: test trajectories from tasks that appear in the training set.
4. test_unseen: test trajectories from tasks that do NOT appear in the training set.
5. Generate detailed statistics report (trajectory count, total duration, etc.).
6. Generate ``datasets/split.json``.

Usage:
    # Automatic split (8:1:1) with test_seen/test_unseen
    python scripts/core/create_dataset_split.py --data_root datasets/TouchAnything_hdf5

    # Specify unseen test tasks (these tasks will NOT appear in train/val)
    python scripts/core/create_dataset_split.py --data_root datasets/TouchAnything_hdf5 \
        --test_unseen_tasks "pick_up_bottle" "open_toolbox"

    # Custom ratio
    python scripts/core/create_dataset_split.py --data_root datasets/TouchAnything_hdf5 \
        --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15
"""

import os
import sys
import json
import argparse
import random
import h5py
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import the quality-check module.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from maintenance.check_hdf5_quality import check_single_file, QualityReport


def _check_worker(args):
    """Parallel worker for validating a single HDF5 file."""
    hdf5_path, data_root, quality_thresholds = args
    report = check_single_file(str(hdf5_path), quality_thresholds)
    
    # New format: Scene/Task/Trajectory.hdf5
    # Use Scene/Task as the full task identifier.
    rel_path_obj = hdf5_path.relative_to(data_root)
    if len(rel_path_obj.parts) >= 2:
        # Use Scene/Task as the task identifier.
        task_key = str(Path(rel_path_obj.parts[0]) / rel_path_obj.parts[1])
    else:
        # Backward compatibility: Task/Trajectory.hdf5
        task_key = hdf5_path.parent.name
    
    rel_path = str(rel_path_obj)
    abs_path = str(hdf5_path.resolve())  # Absolute path.
    return (task_key, rel_path, abs_path, report)


def scan_hdf5_dataset(
    data_root: Path,
    quality_thresholds: dict,
    num_workers: int = 8
) -> Tuple[Dict, List]:
    """
    Scan an HDF5 dataset in parallel, group files by task, and run quality checks.

    Args:
        data_root: Root directory of the HDF5 dataset.
        quality_thresholds: Quality-check thresholds.
        num_workers: Number of worker processes.

    Returns:
        task_files: {task_key: [hdf5_path, ...]} where task_key = 'Scene/Task'
        bad_files:  [(hdf5_path, anomaly_reason), ...] for rejected files
    """
    task_files = defaultdict(list)
    bad_files = []  # [(rel_path, anomaly_reason), ...]
    
    print(f"Scanning dataset: {data_root}")
    all_hdf5 = sorted(data_root.rglob('*.hdf5'))
    print(f"Found {len(all_hdf5)} HDF5 files")
    
    if not all_hdf5:
        return dict(task_files), bad_files
    
    # Prepare parallel tasks.
    tasks = [(hdf5_path, data_root, quality_thresholds) for hdf5_path in all_hdf5]
    
    print(f"\nRunning parallel quality checks (workers={num_workers})...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_check_worker, task): task for task in tasks}
        
        with tqdm(total=len(futures), desc="Quality check") as pbar:
            for future in as_completed(futures):
                task_name, rel_path, abs_path, report = future.result()
                
                if report.ok:
                    # Store absolute paths so training can load files reliably.
                    task_files[task_name].append(abs_path)
                else:
                    # Store the file path and anomaly reason.
                    anomaly_str = report.anomalies if report.anomalies else report.error
                    bad_files.append((rel_path, anomaly_str))
                    tqdm.write(f"  [Skipped] {rel_path}: {anomaly_str}")
                
                pbar.update(1)
    
    print(f"\nQuality check finished:")
    print(f"  Passed: {sum(len(v) for v in task_files.values())}")
    print(f"  Failed: {len(bad_files)}")
    print(f"  Tasks: {len(task_files)}")
    
    # Summarize anomaly types.
    if bad_files:
        anomaly_stats = defaultdict(int)
        for _, anomaly in bad_files:
            # Extract the main anomaly type (remove parenthetical details).
            main_anomaly = anomaly.split('(')[0] if '(' in anomaly else anomaly
            for tag in main_anomaly.split(','):
                tag = tag.strip()
                if tag:
                    anomaly_stats[tag] += 1
        
        print(f"\nAnomaly summary:")
        for anomaly_type, count in sorted(anomaly_stats.items(), key=lambda x: -x[1]):
            print(f"  {anomaly_type}: {count}")
    
    return dict(task_files), bad_files


def split_by_task(
    task_files: Dict[str, List[str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    test_unseen_tasks: List[str] = None,
    test_unseen_ratio: float = 0.5,
    seed: int = 42,
    mini_tasks: List[str] = None,
    mini_max_trajs_per_task: int = 5
) -> Dict[str, List[str]]:
    """
    Advanced split strategy with test_seen/test_unseen and scene-aware distribution:
    - Train/val: share tasks, split at the trajectory level.
    - test_seen: test trajectories from tasks that appear in the training set.
    - test_unseen: test trajectories from tasks that do NOT appear in train/val.
    - Ensures all scenes are represented in the test set (test_seen + test_unseen).

    Args:
        task_files: {task_key: [hdf5_path, ...]} where task_key = 'Scene/Task'
        train_ratio, val_ratio, test_ratio: trajectory-level ratios for train/val/test
        test_unseen_tasks: user-specified unseen test tasks (will be excluded from train/val)
        test_unseen_ratio: ratio of test tasks to allocate to test_unseen (default: 0.5)
        seed: random seed
        mini_tasks: optional mini-mode task list
        mini_max_trajs_per_task: max trajectories to keep per mini-mode task

    Returns:
        {'train': [...], 'val': [...], 'test_seen': [...], 'test_unseen': [...]}
    """
    random.seed(seed)

    def _task_name_aliases(task_key: str) -> set:
        """Return task-name aliases for robust mini-task matching."""
        task_name = Path(task_key).name
        aliases = {task_name}
        if task_name.endswith('-origin'):
            aliases.add(task_name[:-7])
        return aliases
    
    # ========== Mini mode: filter tasks ==========
    if mini_tasks:
        print(f"\n🔍 Mini mode enabled")
        print(f"  Requested tasks: {mini_tasks}")
        print(f"  Max trajectories per task: {mini_max_trajs_per_task}")
        requested_tasks = set(mini_tasks)
        
        # Keep only the requested tasks.
        filtered_task_files = {}
        for task_key, files in task_files.items():
            task_aliases = _task_name_aliases(task_key)
            matched_names = sorted(task_aliases & requested_tasks)
            if matched_names:
                # Limit the number of trajectories per task.
                selected_files = files[:mini_max_trajs_per_task]
                filtered_task_files[task_key] = selected_files
                print(f"  ✓ {Path(task_key).name}: {len(selected_files)}/{len(files)} trajectories")
        
        if not filtered_task_files:
            print(f"\n⚠️  Warning: none of the requested tasks were found. Available tasks:")
            all_task_names = sorted(set(Path(k).name for k in task_files.keys()))
            for i, name in enumerate(all_task_names, 1):
                print(f"    {i}. {name}")
            print(f"\nPlease check the --mini_tasks arguments.")
        
        task_files = filtered_task_files
        print()
    
    all_tasks = list(task_files.keys())
    num_tasks = len(all_tasks)
    total_trajectories = sum(len(v) for v in task_files.values())
    
    # Group tasks by scene for scene-aware splitting
    scene_tasks = defaultdict(list)
    for task_key in all_tasks:
        scene_name = Path(task_key).parts[0] if '/' in task_key else 'Unknown'
        scene_tasks[scene_name].append(task_key)
    
    print(f"\nTask list ({num_tasks} tasks, {total_trajectories} trajectories):")
    print(f"Scenes: {len(scene_tasks)}")
    for scene in sorted(scene_tasks.keys()):
        print(f"\n  Scene: {scene} ({len(scene_tasks[scene])} tasks)")
        for task in sorted(scene_tasks[scene]):
            print(f"    - {Path(task).name}: {len(task_files[task])} trajectories")
    
    # ========== 1. Select test_unseen tasks (scene-aware, task-level isolation) ==========
    if test_unseen_tasks:
        # Support fuzzy matching: users may provide only the task name.
        test_unseen_set = set()
        for user_task in test_unseen_tasks:
            # Exact match.
            if user_task in all_tasks:
                test_unseen_set.add(user_task)
            else:
                # Fuzzy match the task-name part of Scene/Task.
                matched = [t for t in all_tasks if t.endswith('/' + user_task) or t == user_task]
                if matched:
                    test_unseen_set.update(matched)
                    print(f"  Matched '{user_task}' -> {matched}")
                else:
                    print(f"  Warning: no task matched '{user_task}'")
        
        if not test_unseen_set:
            print("Warning: no valid test_unseen task was provided; falling back to automatic split")
            test_unseen_task_list = []
        else:
            test_unseen_task_list = list(test_unseen_set)
            print(f"\nUser-specified test_unseen tasks ({len(test_unseen_task_list)}): {test_unseen_task_list}")
    else:
        # Automatically choose test_unseen tasks: select from each scene to ensure coverage
        test_unseen_task_list = []
        print(f"\nAutomatically selecting test_unseen tasks (scene-aware):")
        for scene_name, tasks in sorted(scene_tasks.items()):
            if len(tasks) < 2:
                # If a scene has only 1 task, keep it for train/val/test_seen
                print(f"  Scene '{scene_name}': only 1 task, skipping unseen split")
                continue
            
            # Select test_unseen_ratio of tasks from this scene
            num_unseen = max(1, int(len(tasks) * test_unseen_ratio))
            shuffled = tasks.copy()
            random.shuffle(shuffled)
            scene_unseen = shuffled[:num_unseen]
            test_unseen_task_list.extend(scene_unseen)
            print(f"  Scene '{scene_name}': {num_unseen}/{len(tasks)} tasks for test_unseen")
        
        print(f"\nTotal test_unseen tasks: {len(test_unseen_task_list)}")
    
    # The remaining tasks are used for train/val/test_seen trajectory-level splitting.
    trainval_seen_tasks = [t for t in all_tasks if t not in test_unseen_task_list]
    
    # ========== 2. Gather test_unseen trajectories ==========
    test_unseen_trajectories = []
    for task in test_unseen_task_list:
        test_unseen_trajectories.extend(task_files[task])
    
    # ========== 3. Split train/val/test_seen tasks at the trajectory level ==========
    all_trainval_seen_trajectories = []
    
    for task in trainval_seen_tasks:
        trajectories = task_files[task].copy()
        random.shuffle(trajectories)  # Shuffle trajectories within each task.
        all_trainval_seen_trajectories.extend(trajectories)
    
    # Split into train/val/test_seen according to ratios.
    num_total = len(all_trainval_seen_trajectories)
    if num_total > 0:
        # Adjust ratios to exclude test_unseen.
        adjusted_train_ratio = train_ratio / (train_ratio + val_ratio + test_ratio)
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio + test_ratio)
        adjusted_test_seen_ratio = test_ratio / (train_ratio + val_ratio + test_ratio)
        
        num_test_seen = max(1, int(num_total * adjusted_test_seen_ratio)) if num_total > 2 else 0
        num_val = max(1, int((num_total - num_test_seen) * (adjusted_val_ratio / (adjusted_train_ratio + adjusted_val_ratio)))) if num_total - num_test_seen > 1 else 0
        num_train = num_total - num_test_seen - num_val
        
        # Shuffle all trajectories.
        random.shuffle(all_trainval_seen_trajectories)
        test_seen_trajectories = all_trainval_seen_trajectories[:num_test_seen]
        val_trajectories = all_trainval_seen_trajectories[num_test_seen:num_test_seen + num_val]
        train_trajectories = all_trainval_seen_trajectories[num_test_seen + num_val:]
    else:
        train_trajectories = []
        val_trajectories = []
        test_seen_trajectories = []
    
    # ========== 4. Summarize task and scene distribution in each split ==========
    def get_task_distribution(trajectories):
        task_dist = defaultdict(int)
        scene_dist = defaultdict(int)
        for traj in trajectories:
            # Extract Scene/Task from absolute path
            parts = Path(traj).parts
            if len(parts) >= 2:
                scene = parts[-3]  # Scene
                task = parts[-2]   # Task
                task_key = f"{scene}/{task}"
                scene_dist[scene] += 1
            else:
                task = Path(traj).parent.name
                task_key = task
            task_dist[task_key] += 1
        return dict(task_dist), dict(scene_dist)
    
    train_dist, train_scenes = get_task_distribution(train_trajectories)
    val_dist, val_scenes = get_task_distribution(val_trajectories)
    test_seen_dist, test_seen_scenes = get_task_distribution(test_seen_trajectories)
    test_unseen_dist, test_unseen_scenes = get_task_distribution(test_unseen_trajectories)
    
    # ========== 5. Print split summary ==========
    print(f"\nDataset split summary (scene-aware test_seen/test_unseen strategy):")
    
    print(f"\n  Train: {len(train_trajectories)} trajectories")
    print(f"    Scenes: {sorted(train_scenes.keys())}")
    for task in sorted(train_dist.keys()):
        print(f"    - {task}: {train_dist[task]}")
    
    print(f"\n  Val:   {len(val_trajectories)} trajectories")
    print(f"    Scenes: {sorted(val_scenes.keys())}")
    for task in sorted(val_dist.keys()):
        print(f"    - {task}: {val_dist[task]}")
    
    print(f"\n  Test_seen:  {len(test_seen_trajectories)} trajectories (from training tasks)")
    print(f"    Scenes: {sorted(test_seen_scenes.keys())}")
    for task in sorted(test_seen_dist.keys()):
        print(f"    - {task}: {test_seen_dist[task]}")
    
    print(f"\n  Test_unseen:  {len(test_unseen_trajectories)} trajectories (unseen tasks, task-level isolation)")
    print(f"    Scenes: {sorted(test_unseen_scenes.keys())}")
    for task in sorted(test_unseen_dist.keys()):
        print(f"    - {task}: {test_unseen_dist[task]}")
    
    # Check scene coverage in test set
    all_scenes = set(scene_tasks.keys())
    test_scenes = set(test_seen_scenes.keys()) | set(test_unseen_scenes.keys())
    missing_scenes = all_scenes - test_scenes
    if missing_scenes:
        print(f"\n  ⚠️  Warning: Test set missing scenes: {sorted(missing_scenes)}")
    else:
        print(f"\n  ✓ Test set covers all {len(all_scenes)} scenes")
    
    # Report train/val task overlap.
    shared_tasks = set(train_dist.keys()) & set(val_dist.keys())
    if shared_tasks:
        print(f"\n  Shared Train/Val tasks ({len(shared_tasks)}): {sorted(shared_tasks)}")
    
    # Report task isolation.
    train_val_tasks = set(train_dist.keys()) | set(val_dist.keys())
    test_seen_tasks = set(test_seen_dist.keys())
    test_unseen_tasks = set(test_unseen_dist.keys())
    print(f"\n  Train/Val unique tasks: {len(train_val_tasks)}")
    print(f"  Test_seen tasks (overlap with train): {len(test_seen_tasks & train_val_tasks)}")
    print(f"  Test_unseen tasks (isolated): {len(test_unseen_tasks)}")
    
    return {
        'train': train_trajectories,
        'val': val_trajectories,
        'test_seen': test_seen_trajectories,
        'test_unseen': test_unseen_trajectories,
    }


def compute_dataset_statistics(split: Dict[str, List[str]], data_root: Path) -> Dict:
    """
    Compute dataset statistics such as total duration, total frames, and task count.

    Args:
        split: {'train': [...], 'val': [...], 'test': [...]}
        data_root: Dataset root directory.

    Returns:
        stats: Statistics dictionary.
    """
    stats = {}
    
    for split_name in ['train', 'val', 'test_seen', 'test_unseen']:
        files = split[split_name]
        total_frames = 0
        total_duration = 0.0
        tasks = set()
        file_count = len(files)
        
        print(f"  Computing statistics for {split_name}...")
        for file_path in tqdm(files, desc=f"{split_name}", leave=False):
            try:
                with h5py.File(file_path, 'r') as f:
                    num_frames = int(f['metadata'].attrs['num_frames'])
                    fps = float(f['metadata'].attrs.get('fps', 30))
                    task_name = str(f['metadata'].attrs.get('task_name', ''))
                    
                    total_frames += num_frames
                    total_duration += num_frames / fps
                    if task_name:
                        tasks.add(task_name)
            except Exception as e:
                print(f"    Warning: failed to read {file_path}: {e}")
        
        stats[split_name] = {
            'num_files': file_count,
            'num_tasks': len(tasks),
            'total_frames': total_frames,
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'total_duration_hours': total_duration / 3600,
            'tasks': sorted(tasks),
        }
    
    return stats


def print_dataset_statistics(stats: Dict):
    """Print dataset statistics."""
    print("\n" + "="*80)
    print("Dataset Statistics Report")
    print("="*80)
    
    for split_name in ['train', 'val', 'test_seen', 'test_unseen']:
        s = stats[split_name]
        print(f"\n[{split_name.upper()}]")
        print(f"  Files: {s['num_files']}")
        print(f"  Tasks: {s['num_tasks']}")
        print(f"  Total frames: {s['total_frames']:,}")
        print(f"  Total duration: {s['total_duration_hours']:.2f} hours ({s['total_duration_minutes']:.1f} minutes / {s['total_duration_seconds']:.1f} seconds)")
        print(f"  Average per file: {s['total_duration_seconds']/s['num_files']:.1f} seconds" if s['num_files'] > 0 else "")
        if s['num_tasks'] <= 10:
            print(f"  Tasks: {', '.join(s['tasks'])}")
    
    # Totals.
    total_files = sum(s['num_files'] for s in stats.values())
    total_frames = sum(s['total_frames'] for s in stats.values())
    total_hours = sum(s['total_duration_hours'] for s in stats.values())
    all_tasks = set()
    for s in stats.values():
        all_tasks.update(s['tasks'])
    
    print(f"\n[TOTAL]")
    print(f"  Total files: {total_files}")
    print(f"  Total tasks: {len(all_tasks)}")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Total duration: {total_hours:.2f} hours ({total_hours*60:.1f} minutes)")
    print("="*80)


def save_statistics_report(stats: Dict, output_file: Path):
    """Save the statistics report to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("TouchAnything Dataset Statistics Report\n")
        f.write("="*80 + "\n\n")
        
        from datetime import datetime
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for split_name in ['train', 'val', 'test_seen', 'test_unseen']:
            s = stats[split_name]
            f.write(f"[{split_name.upper()}]\n")
            f.write(f"  Files: {s['num_files']}\n")
            f.write(f"  Tasks: {s['num_tasks']}\n")
            f.write(f"  Total frames: {s['total_frames']:,}\n")
            f.write(f"  Total duration: {s['total_duration_hours']:.2f} hours ({s['total_duration_minutes']:.1f} minutes)\n")
            f.write(f"  Average per file: {s['total_duration_seconds']/s['num_files']:.1f} seconds\n" if s['num_files'] > 0 else "")
            f.write(f"  Task list:\n")
            for task in s['tasks']:
                f.write(f"    - {task}\n")
            f.write("\n")
        
        # Totals.
        total_files = sum(s['num_files'] for s in stats.values())
        total_frames = sum(s['total_frames'] for s in stats.values())
        total_hours = sum(s['total_duration_hours'] for s in stats.values())
        all_tasks = set()
        for s in stats.values():
            all_tasks.update(s['tasks'])
        
        f.write(f"[TOTAL]\n")
        f.write(f"  Total files: {total_files}\n")
        f.write(f"  Total tasks: {len(all_tasks)}\n")
        f.write(f"  Total frames: {total_frames:,}\n")
        f.write(f"  Total duration: {total_hours:.2f} hours ({total_hours*60:.1f} minutes)\n")
        f.write("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Automatic dataset split generation: quality checks + task-level split')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the HDF5 dataset')
    parser.add_argument('--output', type=str, default='datasets/split.json',
                        help='Output path for split.json')
    
    # Split ratios.
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test ratio (default: 0.1)')
    
    # User-specified test_unseen tasks.
    parser.add_argument('--test_unseen_tasks', type=str, nargs='*',
                        help='Specify test_unseen task names (will be excluded from train/val), e.g. --test_unseen_tasks "pick_up_bottle" "open_toolbox"')
    parser.add_argument('--test_unseen_ratio', type=float, default=0.5,
                        help='Ratio of test tasks to allocate to test_unseen (default: 0.5)')
    
    # Mini mode for fast smoke testing.
    parser.add_argument('--mini_tasks', type=str, nargs='*',
                        help='Mini mode: use only the specified tasks, e.g. --mini_tasks "pick_up_bottle" "pick_up_paper_box" "squeeze_toothpaste"')
    parser.add_argument('--mini_max_trajs_per_task', type=int, default=5,
                        help='Mini mode: maximum trajectories per task (default: 5)')
    
    # Quality-check thresholds.
    parser.add_argument('--min_frames', type=int, default=30,
                        help='Minimum frame count (default: 30, about 1 second)')
    parser.add_argument('--pressure_max_thresh', type=float, default=0.05,
                        help='Pressure max threshold (default: 0.05)')
    parser.add_argument('--pressure_std_thresh', type=float, default=0.001,
                        help='Pressure std threshold (default: 0.001)')
    parser.add_argument('--joint_delta_thresh', type=float, default=0.005,
                        help='Joint displacement threshold (default: 0.005m = 5mm, tolerant to HaMeR noise)')
    parser.add_argument('--joint_static_frames', type=int, default=120,
                        help='Joint-static frame threshold (default: 120 frames = 4 seconds)')
    parser.add_argument('--image_motion_thresh', type=float, default=0.5,
                        help='Image-motion threshold (default: 0.5)')
    parser.add_argument('--image_bright_thresh', type=float, default=3.0,
                        help='Image-brightness threshold (default: 3.0)')
    
    # WiLoR-specific options (replaced HaMeR)
    parser.add_argument('--wilor_min_valid_ratio', type=float, default=0.3,
                        help='Minimum valid-frame ratio for WiLoR (default: 0.3)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes (default: 8)')
    
    args = parser.parse_args()
    
    # Normalize ratios if needed.
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Warning: train/val/test ratios sum to {total_ratio:.3f}; normalizing to 1.0")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio
    
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Error: dataset root does not exist: {data_root}")
        sys.exit(1)
    
    # Quality-check thresholds.
    quality_thresholds = {
        'min_frames': args.min_frames,
        'pressure_max_thresh': args.pressure_max_thresh,
        'pressure_std_thresh': args.pressure_std_thresh,
        'joint_delta_thresh': args.joint_delta_thresh,
        'joint_static_frames': args.joint_static_frames,
        'image_motion_thresh': args.image_motion_thresh,
        'image_bright_thresh': args.image_bright_thresh,
        'wilor_min_valid_ratio': args.wilor_min_valid_ratio,
    }
    
    # 1. Scan and validate the dataset in parallel.
    task_files, bad_files = scan_hdf5_dataset(
        data_root, quality_thresholds, num_workers=args.num_workers)
    
    if not task_files:
        print("\nError: no valid HDF5 files were found")
        sys.exit(1)
    
    # 2. Split by task with test_seen/test_unseen.
    split = split_by_task(
        task_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        test_unseen_tasks=args.test_unseen_tasks,
        test_unseen_ratio=args.test_unseen_ratio,
        seed=args.seed,
        mini_tasks=args.mini_tasks,
        mini_max_trajs_per_task=args.mini_max_trajs_per_task
    )
    
    # 3. Compute dataset statistics.
    print("\nComputing dataset statistics...")
    dataset_stats = compute_dataset_statistics(split, data_root)
    
    # 4. Save split.json.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved split.json to: {output_path}")
    print(f"  Train:        {len(split['train'])} trajectories")
    print(f"  Val:          {len(split['val'])} trajectories")
    print(f"  Test_seen:    {len(split['test_seen'])} trajectories")
    print(f"  Test_unseen:  {len(split['test_unseen'])} trajectories")
    
    # 5. Print detailed statistics.
    print_dataset_statistics(dataset_stats)
    
    # 6. Save the statistics report.
    stats_file = data_root / 'dataset_statistics.txt'
    save_statistics_report(dataset_stats, stats_file)
    print(f"\nSaved statistics report to: {stats_file}")
    
    # 7. Save the quality report (optional).
    if bad_files:
        # Save rejected files together with anomaly reasons.
        bad_file_path = data_root / 'bad_files.txt'
        with open(bad_file_path, 'w', encoding='utf-8') as f:
            f.write("# Rejected files (format: file path | anomaly reason)\n")
            f.write("# " + "="*80 + "\n\n")
            for bad_file, anomaly_reason in bad_files:
                f.write(f"{bad_file} | {anomaly_reason}\n")
        print(f"\nSaved rejected-file list to: {bad_file_path}")


if __name__ == '__main__':
    main()
