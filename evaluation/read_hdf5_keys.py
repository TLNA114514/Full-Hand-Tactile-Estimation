import h5py
import sys

def print_hdf5_sample(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        print(f"HDF5 File: {hdf5_path}")
        if "data/demo_05" in f:
            clip = f["data/demo_05"]
            if "right_hand_landmarks" in clip:
                landmarks = clip["right_hand_landmarks"][()]
                print(f"right_hand_landmarks shape: {landmarks.shape}")
                print(f"Sample of Frame 0 (Wrist & Thumb CMC & Index MCP):")
                print(f"  Wrist (0): {landmarks[0, 0]}")
                print(f"  Joint 1  : {landmarks[0, 1]}")
                print(f"  Joint 5  : {landmarks[0, 5]}")
                
                # Check Min/Max across the whole clip
                valid_mask = ~__import__('numpy').isnan(landmarks)
                if valid_mask.any():
                    valid_landmarks = landmarks[valid_mask]
                    print(f"\nStats across all frames:")
                    print(f"  Min Value: {valid_landmarks.min()}")
                    print(f"  Max Value: {valid_landmarks.max()}")
            else:
                print("No right_hand_landmarks found.")
        else:
            print("demo_05 not found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print_hdf5_sample(sys.argv[1])
    else:
        print_hdf5_sample("/remote-home/luojr/US.T/opentouch/data/fablab_ml_p1.hdf5")
