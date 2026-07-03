import os
import glob
import h5py
import numpy as np

def check_raw_opentouch():
    h5_dir = "/data1/jiangrui/OpenTouch Data/data/"
    h5_files = glob.glob(os.path.join(h5_dir, "*.h5")) + glob.glob(os.path.join(h5_dir, "*.hdf5"))
    
    total_values = 0
    negative_values = 0
    zero_values = 0
    val_lt_307 = 0
    max_raw_val = -float('inf')
    min_raw_val = float('inf')
    
    print(f"开始扫描 {len(h5_files)} 个 OpenTouch HDF5 文件中的原始压力数据...")
    
    for h5f in h5_files:
        try:
            with h5py.File(h5f, 'r') as f:
                if 'data' not in f:
                    continue
                for demo_name in f['data']:
                    demo = f['data'][demo_name]
                    for key in ["left_pressure", "right_pressure"]:
                        if key in demo:
                            p = demo[key][:]
                            # 过滤 NaN
                            p = p[~np.isnan(p)]
                            if p.size == 0:
                                continue
                                
                            total_values += p.size
                            negative_values += np.sum(p < 0)
                            zero_values += np.sum(p == 0)
                            val_lt_307 += np.sum(p < 307.2)
                            
                            max_raw_val = max(max_raw_val, float(np.max(p)))
                            min_raw_val = min(min_raw_val, float(np.min(p)))
        except Exception as e:
            pass
            
    if total_values == 0:
        print("没有找到有效的原始压力数据！")
        return
        
    print("\n--- 原始 OpenTouch 压力分布统计 ---")
    print(f"总计扫描的有效数值点数量: {total_values:,}")
    print(f"原始数值的最大值 (Max): {max_raw_val}")
    print(f"原始数值的最小值 (Min): {min_raw_val}")
    
    neg_ratio = (negative_values / total_values) * 100
    zero_ratio = (zero_values / total_values) * 100
    lt_307_ratio = (val_lt_307 / total_values) * 100
    
    print(f"\n小于 0 的数值数量 (归一化 > 1.0): {negative_values:,} (占比: {neg_ratio:.4f}%)")
    print(f"等于 0 的数值数量 (归一化 = 1.0): {zero_values:,} (占比: {zero_ratio:.4f}%)")
    print(f"小于 307.2 的数值数量 (归一化 > 0.9): {val_lt_307:,} (占比: {lt_307_ratio:.4f}%)")

if __name__ == "__main__":
    check_raw_opentouch()
