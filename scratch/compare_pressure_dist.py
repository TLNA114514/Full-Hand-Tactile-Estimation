import os
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

BINS = 100
RANGE = (0.0, 1.0)
BIN_EDGES = np.linspace(RANGE[0], RANGE[1], BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

def compute_frame_metrics(grid):
    """
    输入: grid (shape: [T, K])，代表一个 sequence 中 T 帧的 K 个有效传感器数值。
    输出: 各种帧级别的统计量 (shape: [T])
    """
    if grid.size == 0:
        return {}
        
    active_mask = grid > 0.01
    
    # 1. Contact ratio
    contact_ratio = np.mean(active_mask, axis=1)
    
    # 2. Max pressure
    max_pressure = np.max(grid, axis=1)
    
    # 3. Mean active pressure
    active_sum = np.sum(grid * active_mask, axis=1)
    active_count = np.sum(active_mask, axis=1)
    mean_active = np.divide(active_sum, active_count, out=np.zeros_like(active_sum), where=active_count != 0)
    
    # 4. Quantiles (基于整帧的分布，包含 0，这能反映整手受压的极值情况)
    p50 = np.percentile(grid, 50, axis=1)
    p90 = np.percentile(grid, 90, axis=1)
    p99 = np.percentile(grid, 99, axis=1)
    
    return {
        'contact_ratio': contact_ratio,
        'max_pressure': max_pressure,
        'mean_active': mean_active,
        'p50': p50,
        'p90': p90,
        'p99': p99
    }

def process_opentouch(h5_path):
    counts_all = np.zeros(BINS, dtype=np.int64)
    counts_active = np.zeros(BINS, dtype=np.int64)
    metrics_list = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'data' not in f: 
                return counts_all, counts_active, metrics_list
            for demo_name in f['data']:
                demo = f['data'][demo_name]
                for key in ["left_pressure_continuous", "right_pressure_continuous"]:
                    if key in demo:
                        # 形状通常为 [T, N] (例如 OpenTouch 预处理好的 1D 数据) 或者 [T, H, W]
                        vals = demo[key][:]
                        if vals.ndim == 3:
                            vals = vals.reshape(vals.shape[0], -1)
                        
                        # 清理 NaN
                        valid_mask = ~np.isnan(vals[0])
                        if not np.any(valid_mask): continue
                        grid = vals[:, valid_mask] # [T, K]
                        
                        if grid.size == 0: continue
                        
                        # 算直方图
                        c, _ = np.histogram(grid, bins=BINS, range=RANGE)
                        counts_all += c
                        c_act, _ = np.histogram(grid[grid > 0.01], bins=BINS, range=RANGE)
                        counts_active += c_act
                        
                        # 计算帧级 Metrics
                        metrics_list.append(compute_frame_metrics(grid))
    except Exception as e:
        pass
        
    return counts_all, counts_active, metrics_list

def process_touchanything(npz_path):
    counts_all = np.zeros(BINS, dtype=np.int64)
    counts_active = np.zeros(BINS, dtype=np.int64)
    metrics_list = []
    
    try:
        data = np.load(npz_path)
        for key in ['left_pressure_grid', 'right_pressure_grid']:
            if key in data:
                vals = data[key]
                if vals.ndim == 3:
                    vals = vals.reshape(vals.shape[0], -1)
                
                valid_mask = ~np.isnan(vals[0])
                if not np.any(valid_mask): continue
                grid = vals[:, valid_mask] # [T, K]
                
                if grid.size == 0: continue
                
                grid = np.clip(grid, 0.0, 1.0)
                
                c, _ = np.histogram(grid, bins=BINS, range=RANGE)
                counts_all += c
                c_act, _ = np.histogram(grid[grid > 0.01], bins=BINS, range=RANGE)
                counts_active += c_act
                
                metrics_list.append(compute_frame_metrics(grid))
    except Exception as e:
        pass
        
    return counts_all, counts_active, metrics_list

def combine_metrics(metrics_lists):
    combined = {}
    if not metrics_lists: return combined
    keys = metrics_lists[0][0].keys() if metrics_lists and metrics_lists[0] else []
    for k in keys:
        combined[k] = np.concatenate([m[k] for mlist in metrics_lists for m in mlist if k in m])
    return combined

def print_summary(name, metrics):
    print(f"\n--- {name} Dataset Summary ---")
    if not metrics:
        print("No data collected.")
        return
    print(f"Total frames evaluated: {len(metrics['contact_ratio'])}")
    for k, v in metrics.items():
        print(f"  [{k:<15}] Mean: {np.mean(v):.4f} | Median (P50): {np.median(v):.4f} | P90: {np.percentile(v, 90):.4f} | Max: {np.max(v):.4f}")

def main():
    opentouch_dir = "/data/jiangrui/OpenTouch Data/data/"
    touchanything_dir = "/data/jiangrui/EgoTouch/"
    
    print("🔎 正在扫描数据集文件...")
    ot_files = glob.glob(os.path.join(opentouch_dir, "*.h5")) + glob.glob(os.path.join(opentouch_dir, "*.hdf5"))
    ta_files = glob.glob(os.path.join(touchanything_dir, "**", "pressure_grids.npz"), recursive=True)
    
    print(f"找到 {len(ot_files)} 个 OpenTouch HDF5 文件.")
    print(f"找到 {len(ta_files)} 个 TouchAnything npz 文件.")
    
    ot_counts_all, ot_counts_active = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64)
    ta_counts_all, ta_counts_active = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64)
    
    ot_all_metrics = []
    ta_all_metrics = []
    
    print("\n🚀 开始并行处理 OpenTouch 数据...")
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_opentouch, f) for f in ot_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            c_all, c_act, m_list = future.result()
            ot_counts_all += c_all
            ot_counts_active += c_act
            if m_list: ot_all_metrics.append(m_list)
            
    print("\n🚀 开始并行处理 TouchAnything 数据...")
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_touchanything, f) for f in ta_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            c_all, c_act, m_list = future.result()
            ta_counts_all += c_all
            ta_counts_active += c_act
            if m_list: ta_all_metrics.append(m_list)

    # 汇总帧级 Metrics
    ot_combined = combine_metrics(ot_all_metrics)
    ta_combined = combine_metrics(ta_all_metrics)
    
    print_summary("OpenTouch", ot_combined)
    print_summary("TouchAnything", ta_combined)
    
    # 画图
    def to_pdf(counts):
        return counts / counts.sum() if counts.sum() > 0 else counts.astype(float)
        
    print("\n📊 正在生成图表...")
    # 第一张图：原始的整体直方图对比
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(BIN_CENTERS, to_pdf(ot_counts_all), label='OpenTouch', color='blue')
    plt.plot(BIN_CENTERS, to_pdf(ta_counts_all), label='TouchAnything', color='orange')
    plt.title('All Values Distribution')
    plt.yscale('log')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(BIN_CENTERS, to_pdf(ot_counts_active), label='OT Active(>0.01)', color='blue')
    plt.plot(BIN_CENTERS, to_pdf(ta_counts_active), label='TA Active(>0.01)', color='orange')
    plt.fill_between(BIN_CENTERS, to_pdf(ot_counts_active), alpha=0.2, color='blue')
    plt.fill_between(BIN_CENTERS, to_pdf(ta_counts_active), alpha=0.2, color='orange')
    plt.title('Active Values (>0.01)')
    plt.legend()
    plt.tight_layout()
    plt.savefig("pressure_dist_comparison.png", dpi=300)
    
    # 第二张图：帧级别的指标对比直方图 (Contact Ratio, Max Pressure, etc.)
    if ot_combined and ta_combined:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        metrics_keys = ['contact_ratio', 'max_pressure', 'mean_active', 'p50', 'p90', 'p99']
        titles = ['Contact Ratio (per frame)', 'Max Pressure (per frame)', 'Mean Active Pressure', 
                  'P50 Pressure', 'P90 Pressure', 'P99 Pressure']
        
        for ax, key, title in zip(axes, metrics_keys, titles):
            # 去除 NaN 以防绘图报错
            ot_data = ot_combined[key][~np.isnan(ot_combined[key])]
            ta_data = ta_combined[key][~np.isnan(ta_combined[key])]
            
            # 使用透明度重叠的 Histogram 直观对比两者分布
            ax.hist(ot_data, bins=50, density=True, alpha=0.5, color='blue', label='OpenTouch')
            ax.hist(ta_data, bins=50, density=True, alpha=0.5, color='orange', label='TouchAnything')
            ax.set_title(title)
            ax.legend()
            
        plt.tight_layout()
        plt.savefig("frame_metrics_comparison.png", dpi=300)
        print("✅ 指标统计分布图已保存至: frame_metrics_comparison.png")

if __name__ == "__main__":
    main()
