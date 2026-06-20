#!/usr/bin/env python3
"""
Simple WiLoR batch processing script with frame-balanced multi-GPU assignment

Features:
1. Scan all chest.mp4 videos
2. Count frames in each video
3. Balance assignments across GPUs by total frame count
4. Run one independent process per GPU with separate logs
5. Estimate processing time at 4 FPS

Usage:
    python batch_process_wilor_simple.py \
        --dataset_root datasets/TouchAnything_Datasets \
        --gpus 0,1,2,3,4,5,6,7 \
        --skip_video
"""

import argparse
import json
import os
import subprocess
import cv2
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import time


def get_video_frame_count(video_path):
    """Get the number of video frames"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except:
        return 0


def find_all_videos_with_frames(dataset_root):
    """Scan all videos and count frames"""
    print("Scanning videos and counting frames...")
    videos_info = []
    
    for root, dirs, files in os.walk(dataset_root):
        if 'chest.mp4' in files:
            video_path = os.path.join(root, 'chest.mp4')
            json_path = os.path.join(root, 'wilor_hands.json')
            
            # Skip if already processed
            if os.path.exists(json_path):
                continue
            
            frame_count = get_video_frame_count(video_path)
            if frame_count > 0:
                videos_info.append({
                    'path': video_path,
                    'frames': frame_count,
                    'dir': root
                })
    
    print(f"Found {len(videos_info)} videos to process")
    total_frames = sum(v['frames'] for v in videos_info)
    print(f"Total frames: {total_frames:,}")
    
    return videos_info


def balance_videos_by_frames(videos_info, num_gpus):
    """Assign videos to GPUs with frame-count balancing"""
    # Sort videos by frame count (descending) for better load balancing
    videos_sorted = sorted(videos_info, key=lambda x: x['frames'], reverse=True)
    
    # Initialize GPU assignments
    gpu_assignments = [[] for _ in range(num_gpus)]
    gpu_frame_counts = [0] * num_gpus
    
    # Greedy assignment: always assign to GPU with least frames
    for video in videos_sorted:
        min_gpu = gpu_frame_counts.index(min(gpu_frame_counts))
        gpu_assignments[min_gpu].append(video)
        gpu_frame_counts[min_gpu] += video['frames']
    
    return gpu_assignments, gpu_frame_counts


def save_video_list(videos, output_file):
    """Save the video list to a file using absolute paths"""
    with open(output_file, 'w') as f:
        for video in videos:
            # Convert to absolute path
            abs_path = os.path.abspath(video['path'])
            f.write(abs_path + '\n')


def estimate_time(frame_count, fps=4.0):
    """Estimate processing time in seconds"""
    return frame_count / fps


def run_gpu_worker(gpu_id, video_list_file, log_file, skip_video):
    """Run the processing task on the specified GPU"""
    python_exe = "/home/intern10/anaconda3/envs/wilor/bin/python"
    batch_script = "/data_all/intern10/tmp/TouchAnything-Dev/third_party/wilor/demo_sam3_video_batch.py"
    wilor_dir = "/data_all/intern10/tmp/TouchAnything-Dev/third_party/wilor"
    
    # Use absolute path for video list file
    abs_video_list = str(Path(video_list_file).absolute())
    
    cmd = [
        python_exe,
        batch_script,
        "--video_list", abs_video_list,
        "--fast"
    ]
    
    if skip_video:
        cmd.append("--skip_video")
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"[GPU {gpu_id}] Starting worker, log: {log_file}")
    
    with open(log_file, 'w') as log:
        log.write(f"GPU {gpu_id} started at {datetime.now()}\n")
        log.write(f"Command: {' '.join(str(x) for x in cmd)}\n")
        log.write(f"Video list: {video_list_file}\n")
        log.write("="*80 + "\n\n")
        log.flush()
        
        process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=wilor_dir,
            env=env,
            text=True
        )
    
    return process


def main():
    parser = argparse.ArgumentParser(description='Simple WiLoR batch processing with load balancing')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root directory')
    parser.add_argument('--gpus', type=str, required=True, help='Comma-separated GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--skip_video', action='store_true', help='Skip video output, only save JSON')
    parser.add_argument('--output_dir', type=str, default='./wilor_batch_logs', 
                       help='Directory for logs and video lists')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    
    print("="*80)
    print("WiLoR Simple Batch Processing")
    print("="*80)
    print(f"Dataset: {args.dataset_root}")
    print(f"GPUs: {gpu_ids}")
    print(f"Skip video: {args.skip_video}")
    print("="*80 + "\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all videos and count frames
    videos_info = find_all_videos_with_frames(args.dataset_root)
    
    if not videos_info:
        print("No videos to process!")
        return
    
    # Balance videos across GPUs
    print(f"\nBalancing {len(videos_info)} videos across {num_gpus} GPUs...")
    gpu_assignments, gpu_frame_counts = balance_videos_by_frames(videos_info, num_gpus)
    
    # Print assignment summary
    print("\n" + "="*80)
    print("GPU Assignment Summary")
    print("="*80)
    total_frames = sum(gpu_frame_counts)
    
    for i, (gpu_id, videos, frame_count) in enumerate(zip(gpu_ids, gpu_assignments, gpu_frame_counts)):
        num_videos = len(videos)
        percentage = (frame_count / total_frames * 100) if total_frames > 0 else 0
        eta_seconds = estimate_time(frame_count, fps=4.0)
        eta_minutes = eta_seconds / 60
        eta_hours = eta_minutes / 60
        
        print(f"GPU {gpu_id}: {num_videos:4d} videos, {frame_count:8,d} frames ({percentage:5.1f}%) "
              f"- ETA: {eta_hours:.1f}h ({eta_minutes:.0f}min)")
    
    print("="*80 + "\n")
    
    # Save video lists for each GPU
    video_list_files = []
    for i, (gpu_id, videos) in enumerate(zip(gpu_ids, gpu_assignments)):
        if not videos:
            continue
        
        list_file = output_dir / f"gpu_{gpu_id}_videos.txt"
        save_video_list(videos, list_file)
        video_list_files.append((gpu_id, list_file))
        print(f"Saved GPU {gpu_id} video list: {list_file} ({len(videos)} videos)")
    
    print("\n" + "="*80)
    print("Starting GPU workers...")
    print("="*80 + "\n")
    
    # Start all GPU workers
    processes = []
    start_time = time.time()
    
    for gpu_id, list_file in video_list_files:
        log_file = output_dir / f"gpu_{gpu_id}_log.txt"
        process = run_gpu_worker(gpu_id, list_file, log_file, args.skip_video)
        processes.append((gpu_id, process, log_file))
    
    print(f"\nAll {len(processes)} GPU workers started!")
    print(f"Monitor logs in: {output_dir}/")
    print("\nTo monitor progress:")
    for gpu_id, _, log_file in processes:
        print(f"  tail -f {log_file}")
    
    # Wait for all processes to complete
    print("\nWaiting for all workers to complete...")
    print("(Press Ctrl+C to stop all workers)\n")
    
    try:
        while True:
            all_done = True
            for gpu_id, process, log_file in processes:
                if process.poll() is None:
                    all_done = False
            
            if all_done:
                break
            
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\nStopping all workers...")
        for gpu_id, process, _ in processes:
            process.terminate()
        print("All workers stopped.")
        return
    
    # Collect results
    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("Batch Processing Complete")
    print("="*80)
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    
    for gpu_id, process, log_file in processes:
        returncode = process.returncode
        status = "✓ Success" if returncode == 0 else f"✗ Failed (code {returncode})"
        print(f"GPU {gpu_id}: {status} - Log: {log_file}")
    
    print("="*80)


if __name__ == '__main__':
    main()
