import h5py
import numpy as np

file_path = "data/fablab_ml_p1.hdf5"
dataset_path = "data/demo_83"

with h5py.File(file_path, "r") as f:
    group = f[dataset_path]
    
    # Load datasets into numpy arrays
    landmarks = group["right_hand_landmarks"][:]
    palm_pos = group["right_palm_pos"][:]
    wrist_pos = group["right_wrist_pos"][:]
    
    print(f"=== Analysis for {dataset_path} ===")
    print("-" * 50)
    
    # 1. right_hand_landmarks
    print("Dataset: right_hand_landmarks")
    print(f"  Shape: {landmarks.shape} (Frames, Joint_Count, Channels)")
    print(f"  Min value: {np.min(landmarks):.4f}")
    print(f"  Max value: {np.max(landmarks):.4f}")
    print(f"  Mean value: {np.mean(landmarks):.4f}")
    # Optional: Per-axis (X, Y, Z) ranges
    min_xyz = np.min(landmarks, axis=(0, 1))
    max_xyz = np.max(landmarks, axis=(0, 1))
    print(f"  Range per axis (X, Y, Z):")
    print(f"    X: [{min_xyz[0]:.4f}, {max_xyz[0]:.4f}]")
    print(f"    Y: [{min_xyz[1]:.4f}, {max_xyz[1]:.4f}]")
    print(f"    Z: [{min_xyz[2]:.4f}, {max_xyz[2]:.4f}]")
    print("-" * 50)
    
    # 2. right_palm_pos
    print("Dataset: right_palm_pos")
    print(f"  Shape: {palm_pos.shape} (Frames, Channels)")
    print(f"  Min value: {np.min(palm_pos):.4f}")
    print(f"  Max value: {np.max(palm_pos):.4f}")
    min_palm_xyz = np.min(palm_pos, axis=0)
    max_palm_xyz = np.max(palm_pos, axis=0)
    print(f"  Range per axis (X, Y, Z):")
    print(f"    X: [{min_palm_xyz[0]:.4f}, {max_palm_xyz[0]:.4f}]")
    print(f"    Y: [{min_palm_xyz[1]:.4f}, {max_palm_xyz[1]:.4f}]")
    print(f"    Z: [{min_palm_xyz[2]:.4f}, {max_palm_xyz[2]:.4f}]")
    print("-" * 50)
    
    # 3. right_wrist_pos
    print("Dataset: right_wrist_pos")
    print(f"  Shape: {wrist_pos.shape} (Frames, Channels)")
    print(f"  Min value: {np.min(wrist_pos):.4f}")
    print(f"  Max value: {np.max(wrist_pos):.4f}")
    min_wrist_xyz = np.min(wrist_pos, axis=0)
    max_wrist_xyz = np.max(wrist_pos, axis=0)
    print(f"  Range per axis (X, Y, Z):")
    print(f"    X: [{min_wrist_xyz[0]:.4f}, {max_wrist_xyz[0]:.4f}]")
    print(f"    Y: [{min_wrist_xyz[1]:.4f}, {max_wrist_xyz[1]:.4f}]")
    print(f"    Z: [{min_wrist_xyz[2]:.4f}, {max_wrist_xyz[2]:.4f}]")
    print("-" * 50)