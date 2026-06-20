import os
import json
import numpy as np
from glob import glob

def analyze_jsonl_file(filepath):
    try:
        keys_info = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                
                # If it's a dict like vive_poses or others
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k not in keys_info:
                            keys_info[k] = {'type': 'NoneType', 'shape': None, 'min': float('inf'), 'max': float('-inf'), 'has_num': False}
                        
                        if v is None: continue
                        keys_info[k]['type'] = type(v).__name__
                        
                        if isinstance(v, list):
                            if len(v) > 0 and isinstance(v[0], list):
                                shape = f"[{len(v)}, {len(v[0])}]"
                                flat = [item for sublist in v for item in sublist if isinstance(item, (int, float))]
                            else:
                                shape = f"[{len(v)}]"
                                flat = [item for item in v if isinstance(item, (int, float))]
                            keys_info[k]['shape'] = shape
                            
                            if len(flat) > 0:
                                keys_info[k]['has_num'] = True
                                keys_info[k]['min'] = min(keys_info[k]['min'], min(flat))
                                keys_info[k]['max'] = max(keys_info[k]['max'], max(flat))
                                
                        elif isinstance(v, (int, float)):
                            keys_info[k]['has_num'] = True
                            keys_info[k]['min'] = min(keys_info[k]['min'], v)
                            keys_info[k]['max'] = max(keys_info[k]['max'], v)
                else:
                    return f"Root is {type(data).__name__}, not a dict."
        
        # Format result
        res = {}
        for k, info in keys_info.items():
            desc = info['type']
            if info['shape']:
                desc += f" shape: {info['shape']}"
            if info['has_num'] and info['min'] != float('inf'):
                desc += f" | range: [{info['min']:.4f}, {info['max']:.4f}]"
            res[k] = desc
        return res
    except Exception as e:
        return f"Error reading {filepath}: {e}"

def analyze_npz(filepath):
    try:
        data = np.load(filepath)
        info = {}
        for k in data.files:
            arr = data[k]
            info[k] = {
                'shape': arr.shape,
                'dtype': str(arr.dtype),
                'min': float(arr.min()) if arr.size > 0 and np.issubdtype(arr.dtype, np.number) else None,
                'max': float(arr.max()) if arr.size > 0 and np.issubdtype(arr.dtype, np.number) else None,
            }
        return info
    except Exception as e:
        return f"Error reading {filepath}: {e}"

def main():
    base_dir = "/data/jiangrui/EgoTouch"
    categories = ["Home", "Office", "Outdoor", "Retail", "Workbench"]
    
    expected_files = {
        "chest.mp4", "left.mp4", "right.mp4", "visualization.mp4",
        "jq_pressure.json", "manual_contact_annotation.json", 
        "pressure_grids.npz", "rokoko_hands.json", "vive_poses.json", "wilor_hands.json"
    }
    
    missing_files_report = []
    total_clips = 0
    sample_clip = None
    
    for cat in categories:
        cat_dir = os.path.join(base_dir, cat)
        if not os.path.exists(cat_dir): continue
        for action in os.listdir(cat_dir):
            action_dir = os.path.join(cat_dir, action)
            if not os.path.isdir(action_dir) or action == "metadata": continue
            for clip in os.listdir(action_dir):
                clip_dir = os.path.join(action_dir, clip)
                if not os.path.isdir(clip_dir): continue
                
                total_clips += 1
                if sample_clip is None:
                    sample_clip = clip_dir
                    
                files_in_clip = set(os.listdir(clip_dir))
                missing = expected_files - files_in_clip
                if missing:
                    missing_files_report.append(f"{cat}/{action}/{clip} is missing: {missing}")
                    
    print(f"Total clips checked: {total_clips}")
    if not missing_files_report:
        print("✅ ALL clips contain ALL expected files!")
    else:
        print(f"❌ Found {len(missing_files_report)} clips with missing files.")
        print("First 10 missing reports:")
        for r in missing_files_report[:10]:
            print("  ", r)
            
    print("\n" + "="*50)
    print(f"Analyzing sample clip: {sample_clip}")
    print("="*50)
    
    if sample_clip:
        print("\n--- pressure_grids.npz ---")
        print(json.dumps(analyze_npz(os.path.join(sample_clip, "pressure_grids.npz")), indent=2))
        
        for json_file in ["jq_pressure.json", "manual_contact_annotation.json", "rokoko_hands.json", "vive_poses.json", "wilor_hands.json"]:
            path = os.path.join(sample_clip, json_file)
            if os.path.exists(path):
                print(f"\n--- {json_file} ---")
                if json_file == "manual_contact_annotation.json":
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                            print(json.dumps(data, indent=2))
                    except Exception as e:
                        print(f"Failed to parse {json_file}: {e}")
                else:
                    # Parse as JSONL
                    analysis = analyze_jsonl_file(path)
                    if isinstance(analysis, dict):
                        print(json.dumps(analysis, indent=2))
                    else:
                        print(analysis)

if __name__ == "__main__":
    main()
