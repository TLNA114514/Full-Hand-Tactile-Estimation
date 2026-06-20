#!/usr/bin/env python
"""
Batch-process the TouchAnything dataset and generate HaMeR hand-pose estimation results.
- Traverse all sequences containing chest.mp4
- Generate hamer_hands.json (hand joints)
- Generate hamer_visualization.mp4 (visualization video)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_hamer_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "third_party" / "hamer",
        project_root / "hamer",  # backward compatibility
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def find_all_sequences(dataset_root):
    """Find all sequences containing chest.mp4."""
    dataset_root = Path(dataset_root)
    sequences = []
    
    for chest_video in dataset_root.rglob("chest.mp4"):
        seq_dir = chest_video.parent
        sequences.append(seq_dir)
    
    return sorted(sequences)


def check_sequence_status(seq_dir):
    """Check the processing status of a sequence."""
    hamer_json = seq_dir / "hamer_hands.json"
    hamer_video = seq_dir / "hamer_visualization.mp4"
    
    status = {
        'has_json': hamer_json.exists(),
        'has_video': hamer_video.exists(),
        'completed': hamer_json.exists() and hamer_video.exists()
    }
    
    return status


def process_sequence(seq_dir, hamer_script, hamer_dir, args, gpu_id):
    """Process a single sequence on a single GPU."""
    cmd = [
        sys.executable,
        str(hamer_script),
        '--dataset_root', str(args.dataset_root),
        '--sequence', str(seq_dir),
        '--batch_size', str(args.batch_size),
        '--frame_batch_size', str(args.frame_batch_size),
        '--gpu_ids', '0',  # CUDA_VISIBLE_DEVICES is set, so the visible device index is always 0
        '--num_workers', '1',  # Single-process mode
    ]
    
    if args.max_frames:
        cmd.extend(['--max_frames', str(args.max_frames)])
    
    if args.overwrite:
        cmd.append('--overwrite')
    
    print(f"\n[GPU {gpu_id}] {seq_dir.relative_to(args.dataset_root)}")
    
    # Restrict the process to the selected GPU only.
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Run inside the hamer directory so that the _DATA/ path can be resolved.
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(hamer_dir), env=env)
    
    # Print error details on failure.
    if result.returncode != 0:
        print(f"\n[ERROR] GPU {gpu_id} failed on: {seq_dir.relative_to(args.dataset_root)}")
        if result.stderr:
            print(f"STDERR: {result.stderr[-500:]}")  # Print only the last 500 characters
        if result.stdout:
            print(f"STDOUT: {result.stdout[-500:]}")
    
    return result.returncode == 0, seq_dir


def main():
    parser = argparse.ArgumentParser(description='Batch-process the TouchAnything dataset and generate HaMeR results')
    parser.add_argument(
        '--dataset_root',
        type=str,
        default=str(PROJECT_ROOT / 'datasets' / 'TouchAnything_Datasets_clean'),
        help='Dataset root directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='HaMeR inference batch size per GPU (recommended: 256-512)'
    )
    parser.add_argument(
        '--frame_batch_size',
        type=int,
        default=1024,
        help='Number of frames processed at once per GPU (recommended: 512-1024)'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum number of frames to process per sequence (for testing)'
    )
    parser.add_argument(
        '--skip_completed',
        action='store_true',
        help='Skip sequences that are already completed'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing result files'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='List sequences only without processing'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help='GPU IDs to use, separated by commas, e.g. "0,1,2,3"'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker processes per GPU'
    )
    
    args = parser.parse_args()
    
    # Resolve paths.
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]
    hamer_dir = resolve_hamer_dir(project_root)
    hamer_script = hamer_dir / "process_touchanything_parallel.py"
    
    if not hamer_script.exists():
        print(f"[ERROR] HaMeR script does not exist: {hamer_script}")
        return 1
    
    if not hamer_dir.exists():
        print(f"[ERROR] HaMeR directory does not exist: {hamer_dir}")
        return 1
    
    # Find all sequences.
    print("="*80)
    print("Searching for sequences...")
    print("="*80)
    
    sequences = find_all_sequences(args.dataset_root)
    
    if not sequences:
        print(f"[ERROR] No sequences containing chest.mp4 were found")
        return 1
    
    print(f"\nFound {len(sequences)} sequences\n")
    
    # Summarize current status.
    total = len(sequences)
    completed = 0
    to_process = []
    
    for seq_dir in sequences:
        status = check_sequence_status(seq_dir)
        
        if status['completed']:
            completed += 1
            if args.skip_completed:
                print(f"[✓] {seq_dir.relative_to(args.dataset_root)} - completed")
                continue
        
        to_process.append(seq_dir)
        
        if args.dry_run:
            status_str = "completed" if status['completed'] else "pending"
            json_str = "✓" if status['has_json'] else "✗"
            video_str = "✓" if status['has_video'] else "✗"
            print(f"[{status_str}] {seq_dir.relative_to(args.dataset_root)}")
            print(f"    JSON: {json_str} | Video: {video_str}")
    
    print("\n" + "="*80)
    print(f"Summary:")
    print(f"  Total sequences: {total}")
    print(f"  Completed: {completed}")
    print(f"  Pending: {len(to_process)}")
    print("="*80)
    
    if args.dry_run:
        print("\n[DRY RUN] Listed sequences only. No processing was performed.")
        return 0
    
    if not to_process:
        print("\nAll sequences have already been processed.")
        return 0
    
    # Confirm processing.
    if not args.skip_completed and completed > 0:
        response = input(f"\nAbout to process {len(to_process)} sequences (including {completed} already completed ones). Continue? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Batch processing with multi-GPU parallelism.
    print("\n" + "="*80)
    print("Starting batch processing (multi-GPU parallel mode)")
    print("="*80)
    
    # Parse GPU IDs.
    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    print(f"Parallelism: {len(gpu_ids)} GPUs processing simultaneously")
    
    success_count = 0
    failed_sequences = []
    
    start_time = datetime.now()
    
    # Run tasks in parallel with ThreadPoolExecutor.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Assign GPUs to sequences in round-robin order.
    tasks = []
    for idx, seq_dir in enumerate(to_process):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        tasks.append((seq_dir, gpu_id))
    
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        # Submit all tasks.
        future_to_seq = {
            executor.submit(process_sequence, seq_dir, hamer_script, hamer_dir, args, gpu_id): (seq_dir, gpu_id)
            for seq_dir, gpu_id in tasks
        }
        
        # Show overall progress with tqdm.
        from tqdm import tqdm
        with tqdm(total=len(to_process), desc="Overall progress", unit="seq") as pbar:
            # Handle completed tasks.
            for future in as_completed(future_to_seq):
                seq_dir, gpu_id = future_to_seq[future]
                try:
                    success, seq_dir = future.result()
                    
                    if success:
                        success_count += 1
                        pbar.set_postfix_str(f"✓ GPU{gpu_id}: {seq_dir.name}")
                    else:
                        failed_sequences.append(seq_dir)
                        pbar.set_postfix_str(f"✗ GPU{gpu_id}: {seq_dir.name}")
                
                except Exception as e:
                    failed_sequences.append(seq_dir)
                    pbar.set_postfix_str(f"✗ GPU{gpu_id}: {seq_dir.name} - {str(e)[:30]}")
                
                pbar.update(1)
    
    # Final summary.
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("Batch processing finished")
    print("="*80)
    print(f"Total sequences: {len(to_process)}")
    print(f"Succeeded: {success_count}")
    print(f"Failed: {len(failed_sequences)}")
    print(f"Elapsed time: {duration}")
    print("="*80)
    
    if failed_sequences:
        print("\nFailed sequences:")
        for seq_dir in failed_sequences:
            print(f"  - {seq_dir.relative_to(args.dataset_root)}")
    
    # Save processing log.
    log_file = project_root / "hamer_batch_processing_log.json"
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'total': len(to_process),
        'success': success_count,
        'failed': len(failed_sequences),
        'duration_seconds': duration.total_seconds(),
        'failed_sequences': [str(s.relative_to(args.dataset_root)) for s in failed_sequences]
    }
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nProcessing log saved to: {log_file}")
    
    return 0 if len(failed_sequences) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
