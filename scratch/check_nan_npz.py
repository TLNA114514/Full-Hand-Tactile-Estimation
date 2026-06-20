import numpy as np
import sys

npz_path = "/data/jiangrui/EgoTouch/Home/arrange_pillow/20260412_101136_379/pressure_grids.npz"
data = np.load(npz_path)

for key in ['left_pressure_grid', 'right_pressure_grid']:
    arr = data[key]
    valid_mask = ~np.isnan(arr)
    valid_arr = arr[valid_mask]
    
    print(f"\n--- {key} ---")
    print(f"Shape: {arr.shape}")
    print(f"Total elements: {arr.size}")
    print(f"NaN count: {np.isnan(arr).sum()}")
    print(f"Valid elements: {valid_arr.size}")
    if valid_arr.size > 0:
        print(f"Valid min: {valid_arr.min()}")
        print(f"Valid max: {valid_arr.max()}")
        print(f"Valid mean: {valid_arr.mean()}")
