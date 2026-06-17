import os
import glob
import json
import shutil
from tqdm import tqdm

dataset_dir = "/data/jiangrui/OpenTouch Data/extracted_dataset/"
meta_files = glob.glob(os.path.join(dataset_dir, "*", "*", "meta.json"))

broken_count = 0
for mf in tqdm(meta_files, desc="Checking for corrupted meta.json files"):
    try:
        with open(mf, "r") as f:
            json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ 发现损坏文件: {mf}")
        folder_path = os.path.dirname(mf)
        broken_name = "_broken_" + os.path.basename(folder_path)
        new_folder_path = os.path.join(os.path.dirname(folder_path), broken_name)
        
        try:
            os.rename(folder_path, new_folder_path)
            print(f"✅ 已安全隔离为: {new_folder_path}")
            broken_count += 1
        except Exception as e:
            print(f"❌ 隔离失败: {e}")

print(f"\n检查完毕！共发现并隔离了 {broken_count} 个损坏的数据帧。")
