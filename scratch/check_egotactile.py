import os
import json
import numpy as np

def check_json_structure(json_path):
    print(f"==================================================")
    print(f"Analyzing: {json_path}")
    print(f"==================================================")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    if isinstance(data, list):
        print(f"Root object is a list containing {len(data)} items (likely frames).")
        if len(data) == 0:
            print("The list is empty.")
            return
        print("Structure of the first item (data[0]):\n")
        sample = data[0]
    else:
        print("Root object is a dictionary.\n")
        sample = data
        
    for k, v in sample.items():
        if isinstance(v, list):
            arr = np.array(v)
            if np.issubdtype(arr.dtype, np.number):
                print(f"Key: {k:<20} | Type: list/array | Shape: {arr.shape} | Range: [{np.min(arr):.4f}, {np.max(arr):.4f}]")
            else:
                print(f"Key: {k:<20} | Type: list/array | Shape: {arr.shape} | Contains non-numeric data")
        elif isinstance(v, dict):
            print(f"Key: {k:<20} | Type: dict       | Keys: {list(v.keys())}")
            for sub_k, sub_v in v.items():
                full_k = f"{k}.{sub_k}"
                if isinstance(sub_v, list):
                    arr = np.array(sub_v)
                    if np.issubdtype(arr.dtype, np.number):
                        print(f"  └─ {full_k:<17} | Type: list/array | Shape: {arr.shape} | Range: [{np.min(arr):.4f}, {np.max(arr):.4f}]")
                    else:
                        print(f"  └─ {full_k:<17} | Type: list/array | Shape: {arr.shape} | Contains non-numeric data")
                elif isinstance(sub_v, dict):
                    print(f"  └─ {full_k:<17} | Type: dict       | Keys: {list(sub_v.keys())}")
                else:
                    print(f"  └─ {full_k:<17} | Type: {type(sub_v).__name__:<10} | Value: {sub_v}")
        else:
            print(f"Key: {k:<20} | Type: {type(v).__name__:<10} | Value: {v}")

if __name__ == "__main__":
    sample_path = "/data1/jiangrui/EgoTactile/Raw_data/bare_hand/p001/Apple/repeat0000/data.json"
    check_json_structure(sample_path)
