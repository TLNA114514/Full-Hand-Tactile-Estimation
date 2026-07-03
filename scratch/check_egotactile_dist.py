import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

EGOTACTILE_PMIN = 5.0
EGOTACTILE_PMAX = 200.0
DEFAULT_SCAN_EXCLUDE_DIRS = {".git", "__pycache__", "extracted_frames", "metadata", "artifacts"}


def normalize_egotactile_sensor(values, pmin=EGOTACTILE_PMIN, pmax=EGOTACTILE_PMAX):
    if pmax <= pmin:
        raise ValueError(f"pmax must be greater than pmin, got pmin={pmin}, pmax={pmax}")
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, pmin, pmax)
    return (values - pmin) / (pmax - pmin)

def compute_frame_metrics(grid):
    if grid.size == 0:
        return {}
    active_mask = grid > 0.01
    contact_ratio = np.mean(active_mask, axis=1)
    max_pressure = np.max(grid, axis=1)
    active_sum = np.sum(grid * active_mask, axis=1)
    active_count = np.sum(active_mask, axis=1)
    mean_active = np.divide(active_sum, active_count, out=np.zeros_like(active_sum), where=active_count != 0)
    p50 = np.percentile(grid, 50, axis=1)
    p90 = np.percentile(grid, 90, axis=1)
    p99 = np.percentile(grid, 99, axis=1)
    return {
        'contact_ratio': contact_ratio,
        'max_pressure': max_pressure,
        'mean_active': mean_active,
        'p50': p50,
        'p90': p90,
        'p99': p99
    }

def process_egotactile(json_path):
    metrics_list = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list) or len(data) == 0:
            return metrics_list
            
        grid_list = []
        for frame in data:
            if 'RH' in frame and 'sensor_256' in frame['RH']:
                grid_list.append(frame['RH']['sensor_256'])
            if 'LH' in frame and 'sensor_256' in frame['LH']:
                grid_list.append(frame['LH']['sensor_256'])
                
        if len(grid_list) == 0:
            return metrics_list
            
        # EgoTactile normalization: [5N, 200N] -> [0, 1].
        grid = normalize_egotactile_sensor(grid_list)
        
        metrics_list.append(compute_frame_metrics(grid))
    except Exception as e:
        pass
    
    return metrics_list

def combine_metrics(metrics_lists):
    combined = {}
    if not metrics_lists: return combined
    keys = metrics_lists[0][0].keys() if metrics_lists and metrics_lists[0] else []
    for k in keys:
        arrs = [m[k] for mlist in metrics_lists for m in mlist if k in m]
        if arrs:
            combined[k] = np.concatenate(arrs)
    return combined


def find_data_json_files(root, exclude_dirs=DEFAULT_SCAN_EXCLUDE_DIRS):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in exclude_dirs]
        if "data.json" in filenames and "video.mp4" in filenames:
            files.append(os.path.join(dirpath, "data.json"))
    return sorted(files)

def main():
    egotactile_dir = "/data1/jiangrui/EgoTactile/Raw_data/"
    json_files = find_data_json_files(egotactile_dir)
    
    print(f"🔎 找到 {len(json_files)} 个 EgoTactile data.json 文件.")
    
    all_metrics = []
    
    print("\n🚀 开始并行处理 EgoTactile 归一化提取...")
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_egotactile, f) for f in json_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            m_list = future.result()
            if m_list:
                all_metrics.append(m_list)

    combined = combine_metrics(all_metrics)
    
    print("\n--- EgoTactile (RAW Normalized) Summary ---")
    if not combined or 'contact_ratio' not in combined:
        print("No data collected.")
        return
        
    print(f"Total frames evaluated: {len(combined['contact_ratio']):,}")
    for k, v in combined.items():
        print(f"  [{k:<15}] Mean: {np.mean(v):.4f} | Median (P50): {np.median(v):.4f} | P90: {np.percentile(v, 90):.4f} | Max: {np.max(v):.4f}")

if __name__ == "__main__":
    main()
