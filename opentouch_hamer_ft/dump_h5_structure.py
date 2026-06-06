import h5py

def dump_h5_structure(h5_path, output_txt):
    def print_structure(name, obj, file):
        if isinstance(obj, h5py.Dataset):
            file.write(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}\n")
        elif isinstance(obj, h5py.Group):
            file.write(f"Group: {name}\n")
            
    with h5py.File(h5_path, 'r') as f:
        with open(output_txt, 'w') as out_f:
            f.visititems(lambda name, obj: print_structure(name, obj, out_f))

if __name__ == "__main__":
    # 以 demo_85 为例，或者读取整个文件结构
    data_dir = "/data/jiangrui/OpenTouch Data/data"
    scene = "home_kitchen_p3"
    h5_path = f"{data_dir}/{scene}.hdf5"
    output_path = "/code/users/jiangrui/Full-Hand-Tactile-Estimation/opentouch_hamer_ft/opentouch_hdf5_structure.txt"
    
    print(f"Dumping HDF5 structure to {output_path}...")
    try:
        dump_h5_structure(h5_path, output_path)
        print("Done!")
    except Exception as e:
        print(f"Failed to dump structure: {e}")
