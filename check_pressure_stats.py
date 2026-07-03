import os
import json
import glob
import numpy as np

def main():
    data_dir = "/data1/jiangrui/OpenTouch Data/extracted_dataset/train"
    meta_files = glob.glob(os.path.join(data_dir, "*", "meta.json"))
    
    print(f"Found {len(meta_files)} meta.json files.")
    
    # We will sample up to 1000 files to get a quick estimate
    np.random.seed(42)
    if len(meta_files) > 1000:
        meta_files = np.random.choice(meta_files, 1000, replace=False)
        
    global_max = -float('inf')
    global_min = float('inf')
    
    for mf in meta_files:
        with open(mf, 'r') as f:
            data = json.load(f)
            
        hdf5_data = data.get("original_hdf5_data", {})
        pressure = None
        if "right_pressure" in hdf5_data:
            pressure = hdf5_data["right_pressure"]
        elif "left_pressure" in hdf5_data:
            pressure = hdf5_data["left_pressure"]
            
        if pressure is not None:
            p_arr = np.array(pressure)
            p_max = p_arr.max()
            p_min = p_arr.min()
            
            if p_max > global_max:
                global_max = p_max
            if p_min < global_min:
                global_min = p_min

    print(f"Stats over sample: Global Min = {global_min}, Global Max = {global_max}")

if __name__ == "__main__":
    main()
