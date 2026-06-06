import h5py

def print_structure(name, obj):

    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")

file_path = "/data/jiangrui/OpenTouch Data/data/fablab_ml_p1.hdf5"
with h5py.File(file_path, "r") as f:
    print("--- HDF5 ---")
    f.visititems(print_structure)