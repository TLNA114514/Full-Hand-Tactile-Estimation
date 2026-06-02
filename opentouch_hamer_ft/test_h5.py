import h5py
import sys
import os
import numpy as np

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

file_path = "/remote-home/luojr/US.T/opentouch/data/eat_mcdonalds.hdf5"
print(f"Reading {file_path}...")
try:
    with h5py.File(file_path, "r", swmr=True) as f:
        clip_group = f["data/demo_31"]
        rgb_bytes_seq = clip_group["rgb_images_jpeg"]
        raw_bytes = rgb_bytes_seq[32]
        
        byte_data = raw_bytes.tobytes() if isinstance(raw_bytes, np.ndarray) else raw_bytes
        
        print(f"Read {len(byte_data)} bytes.")
        if byte_data[:2] == b'\xff\xd8':
            print("It's a valid JPEG header!")
        else:
            print(f"Header is: {byte_data[:10]}")
except Exception as e:
    print(f"Error: {e}")
