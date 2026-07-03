import glob
import h5py
import json
import numpy as np
import os

def check_h5(h5_dir):
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    if not h5_files:
        print(f"No H5 files found in {h5_dir}")
        return
        
    sample_file = h5_files[0]
    print(f"--- Checking HDF5: {sample_file} ---")
    
    with h5py.File(sample_file, 'r') as f:
        data_group = f['data']
        demo_name = list(data_group.keys())[0]
        demo = data_group[demo_name]
        
        for key in ["right_pressure", "right_pressure_continuous", "right_pressure_continuous_subdiv"]:
            if key in demo:
                p = demo[key][:]
                print(f"[{key}] shape: {p.shape}, dtype: {p.dtype}")
                if p.size > 0:
                    print(f"  min: {np.nanmin(p):.4f}, max: {np.nanmax(p):.4f}, mean: {np.nanmean(p):.4f}")
            else:
                print(f"[{key}] Not found in demo {demo_name}")

def check_extracted(ext_dir):
    meta_files = glob.glob(os.path.join(ext_dir, "*", "*", "meta.json"))
    if not meta_files:
        print(f"No meta.json found in {ext_dir}")
        return
        
    sample_file = meta_files[0]
    print(f"\n--- Checking JSON: {sample_file} ---")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    hdf5_data = data.get("original_hdf5_data", {})
    for key in ["right_pressure", "right_pressure_continuous", "right_pressure_continuous_subdiv"]:
        if key in hdf5_data:
            p = np.array(hdf5_data[key])
            print(f"[{key}] shape: {p.shape}, dtype: {p.dtype}")
            if p.size > 0:
                print(f"  min: {np.nanmin(p):.4f}, max: {np.nanmax(p):.4f}, mean: {np.nanmean(p):.4f}")
        else:
            print(f"[{key}] Not found")

if __name__ == "__main__":
    check_h5("/data1/jiangrui/OpenTouch Data/data/")
    check_extracted("/data1/jiangrui/OpenTouch Data/extracted_dataset/")
