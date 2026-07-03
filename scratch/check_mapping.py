import os
import json
import numpy as np
import argparse

def check_mapping(clip_dir):
    print(f"==================================================")
    print(f"Analyzing {clip_dir}...")
    npz_path = os.path.join(clip_dir, "pressure_grids.npz")
    json_path = os.path.join(clip_dir, "jq_pressure.json")
    
    if not os.path.exists(npz_path) or not os.path.exists(json_path):
        print("Missing npz or json file.")
        return
    
    # 1. Load npz
    data = np.load(npz_path)
    grid_l = data['left_pressure_grid']  # [T, 21, 21]
    
    # 2. Load json
    raw_l_list = []
    with open(json_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            frame_data = json.loads(line)
            raw_l_list.append(frame_data['sensor_left'])
    raw_l = np.array(raw_l_list)  # [T, 256]
    
    T1 = grid_l.shape[0]
    T2 = raw_l.shape[0]
    T = min(T1, T2)
    grid_l = grid_l[:T]
    raw_l = raw_l[:T]
    
    print(f"Loaded {T} frames.")
    
    # 3. Find valid pixels in 21x21 grid
    # A pixel is valid if it's not NaN in the first frame
    valid_mask = ~np.isnan(grid_l[0])
    valid_coords = np.argwhere(valid_mask)
    print(f"Found {len(valid_coords)} valid pixels in 21x21 grid.")
    
    # 4. Compute correlation to find mapping
    mapping = {}
    for (r, c) in valid_coords:
        y = grid_l[:, r, c]
        
        # If y is constant, correlation will be NaN
        if np.max(y) - np.min(y) < 1e-6:
            mapping[f"{r},{c}"] = "Constant/Zero"
            continue
            
        best_idx = -1
        best_corr = -1.0
        
        for idx in range(256):
            x = raw_l[:, idx]
            if np.max(x) - np.min(x) < 1e-6:
                continue
            
            # Pearson correlation
            with np.errstate(divide='ignore', invalid='ignore'):
                corr = np.corrcoef(x, y)[0, 1]
            if np.isnan(corr):
                continue
            if corr > best_corr:
                best_corr = corr
                best_idx = idx
                
        if best_corr > 0.99:
            mapping[f"{r},{c}"] = best_idx
        else:
            mapping[f"{r},{c}"] = f"Max corr {best_corr:.2f} at {best_idx}"
            
    # Print the mapping as a 21x21 text grid
    grid_vis = np.full((21, 21), -1, dtype=int)
    for (r, c) in valid_coords:
        val = mapping.get(f"{r},{c}", -1)
        if isinstance(val, int):
            grid_vis[r, c] = val
            
    print("\nMapping from 21x21 (r, c) to 1D index (0-255):")
    print("      " + " ".join([f"{c:3d}" for c in range(21)]))
    print("    +" + "-" * (21 * 4))
    for r in range(21):
        row_str = []
        for c in range(21):
            if grid_vis[r, c] != -1:
                row_str.append(f"{grid_vis[r, c]:3d}")
            elif not valid_mask[r, c]:
                row_str.append("   ")  # empty space for NaN
            else:
                row_str.append(" ? ")
        print(f"{r:2d}  | " + " ".join(row_str))
        
    mapped_count = sum(1 for v in mapping.values() if isinstance(v, int))
    print(f"\nSuccessfully mapped {mapped_count} out of {len(valid_coords)} points with correlation > 0.99.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=str, default="/data1/jiangrui/EgoTouch/Home/arrange_pillow/20260412_101136_379")
    args = parser.parse_args()
    check_mapping(args.clip)
