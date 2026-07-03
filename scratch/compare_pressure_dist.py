import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

BINS = 100
RANGE = (0.0, 1.0)
BIN_EDGES = np.linspace(RANGE[0], RANGE[1], BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

DEFAULT_OPENTOUCH_DIR = "/data1/jiangrui/OpenTouch Data/data/"
DEFAULT_TOUCHANYTHING_DIR = "/data1/jiangrui/EgoTouch/"
DEFAULT_EGOTACTILE_DIR = "/data1/jiangrui/EgoTactile/Raw_data/"
EGOTACTILE_PMIN = 5.0
EGOTACTILE_PMAX = 200.0
DEFAULT_SCAN_EXCLUDE_DIRS = (
    ".git",
    "__pycache__",
    "outputs",
    "touchanything_bboxes_cache",
    "touchanything_dataset",
    "extracted_frames",
    "touchanything_frames",
)


def normalize_egotactile_sensor(values, pmin=EGOTACTILE_PMIN, pmax=EGOTACTILE_PMAX):
    if pmax <= pmin:
        raise ValueError(f"pmax must be greater than pmin, got pmin={pmin}, pmax={pmax}")
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, pmin, pmax)
    return (values - pmin) / (pmax - pmin)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare normalized pressure distributions across tactile datasets."
    )
    parser.add_argument("--opentouch_dir", default=DEFAULT_OPENTOUCH_DIR)
    parser.add_argument("--touchanything_dir", default=DEFAULT_TOUCHANYTHING_DIR)
    parser.add_argument("--egotactile_dir", default=DEFAULT_EGOTACTILE_DIR)
    parser.add_argument("--egotactile_npz_name", default="pressure_grids_egotactile.npz")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--scan_workers", type=int, default=16)
    parser.add_argument("--scan_exclude_dirs", nargs="*", default=list(DEFAULT_SCAN_EXCLUDE_DIRS))
    parser.add_argument(
        "--touchanything_scan_depth",
        type=int,
        default=3,
        help="Directory depth below touchanything_dir where pressure_grids.npz is expected. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--egotactile_scan_depth",
        type=int,
        default=4,
        help="Directory depth below egotactile_dir where data.json / Gaussian npz is expected. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--touchanything_scan_split_depth",
        type=int,
        default=2,
        help="Split TouchAnything scanning into parallel jobs at this directory depth.",
    )
    parser.add_argument(
        "--egotactile_scan_split_depth",
        type=int,
        default=3,
        help="Split EgoTactile scanning into parallel jobs at this directory depth.",
    )
    parser.add_argument("--output_dir", default=".")
    return parser.parse_args()


def normalized_depth(value):
    return None if value is None or value < 0 else value


def rel_depth(path, root):
    rel = os.path.relpath(path, root)
    if rel == ".":
        return 0
    return len(rel.split(os.sep))


def collect_scan_roots(root, split_depth, exclude_dirs):
    root = os.path.abspath(root)
    exclude_dirs = set(exclude_dirs)
    if split_depth <= 0:
        return [root]

    current = [root]
    for _ in range(split_depth):
        next_dirs = []
        for parent in current:
            try:
                with os.scandir(parent) as it:
                    for entry in it:
                        if entry.name in exclude_dirs:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            next_dirs.append(entry.path)
            except OSError:
                continue
        if not next_dirs:
            return current
        current = next_dirs
    return current


def walk_named_files(start_dir, root, filename, exclude_dirs, max_depth):
    matches = []
    exclude_dirs = set(exclude_dirs)
    for dirpath, dirnames, filenames in os.walk(start_dir):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        depth = rel_depth(dirpath, root)
        if max_depth is not None and depth > max_depth:
            dirnames[:] = []
            continue
        if filename in filenames:
            matches.append(os.path.join(dirpath, filename))
        if max_depth is not None and depth >= max_depth:
            dirnames[:] = []
    return matches


def find_named_files(root, filename, scan_workers, exclude_dirs, max_depth, split_depth, desc):
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        return []

    scan_roots = collect_scan_roots(root, split_depth, exclude_dirs)
    if len(scan_roots) == 1:
        return sorted(walk_named_files(scan_roots[0], root, filename, exclude_dirs, max_depth))

    matches = []
    root_file = os.path.join(root, filename)
    if os.path.isfile(root_file):
        matches.append(root_file)

    workers = max(1, min(scan_workers, len(scan_roots)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(walk_named_files, scan_root, root, filename, exclude_dirs, max_depth)
            for scan_root in scan_roots
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            matches.extend(future.result())
    return sorted(set(matches))


def find_opentouch_files(root):
    if not os.path.isdir(root):
        return []
    files = []
    try:
        with os.scandir(root) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                lower = entry.name.lower()
                if lower.endswith(".h5") or lower.endswith(".hdf5"):
                    files.append(entry.path)
    except OSError:
        return []
    return sorted(files)

def compute_frame_metrics(grid):
    """
    输入: grid (shape: [T, K])，代表一个 sequence 中 T 帧的 K 个有效传感器数值。
    输出: 各种帧级别的统计量 (shape: [T])
    """
    if grid.size == 0:
        return {}

    active_mask = grid > 0.01

    contact_ratio = np.mean(active_mask, axis=1)
    max_pressure = np.max(grid, axis=1)

    active_sum = np.sum(grid * active_mask, axis=1)
    active_count = np.sum(active_mask, axis=1)
    mean_active = np.divide(active_sum, active_count, out=np.zeros_like(active_sum), where=active_count != 0)

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

def process_egotactile(json_path):
    import json
    res_raw = (np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), [])
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0: return res_raw

        grid_list = []
        for frame in data:
            if 'RH' in frame and 'sensor_256' in frame['RH']:
                grid_list.append(frame['RH']['sensor_256'])
            if 'LH' in frame and 'sensor_256' in frame['LH']:
                grid_list.append(frame['LH']['sensor_256'])
        if len(grid_list) == 0: return res_raw

        grid = normalize_egotactile_sensor(grid_list)

        c, _ = np.histogram(grid, bins=BINS, range=RANGE)
        res_raw[0][:] += c
        c_act, _ = np.histogram(grid[grid > 0.01], bins=BINS, range=RANGE)
        res_raw[1][:] += c_act
        res_raw[2].append(compute_frame_metrics(grid))
    except Exception as e:
        pass
    return res_raw


def process_egotactile_gaussian(npz_path):
    res_sub = (np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), [])
    try:
        data = np.load(npz_path)
        frame_masks = {}
        if "left_sensor_valid" in data:
            frame_masks["left_pressure_continuous_subdiv"] = data["left_sensor_valid"][:]
        if "right_sensor_valid" in data:
            frame_masks["right_pressure_continuous_subdiv"] = data["right_sensor_valid"][:]
        s_all, s_act, s_met = extract_from_dict(
            data,
            ["left_pressure_continuous_subdiv", "right_pressure_continuous_subdiv"],
            is_ot_raw=False,
            frame_masks=frame_masks,
        )
        res_sub[0][:] += s_all
        res_sub[1][:] += s_act
        res_sub[2].extend(s_met)
    except Exception:
        pass
    return res_sub

def extract_from_dict(data, keys, is_ot_raw=False, frame_masks=None):
    counts_all = np.zeros(BINS, dtype=np.int64)
    counts_active = np.zeros(BINS, dtype=np.int64)
    metrics_list = []
    frame_masks = frame_masks or {}

    for key in keys:
        if key in data:
            vals = data[key][:]
            if vals.ndim == 3:
                vals = vals.reshape(vals.shape[0], -1)

            if key in frame_masks:
                frame_mask = np.asarray(frame_masks[key], dtype=bool).reshape(-1)
                n = min(vals.shape[0], frame_mask.shape[0])
                vals = vals[:n][frame_mask[:n]]
                if vals.shape[0] == 0:
                    continue

            if is_ot_raw:
                # OpenTouch Raw 归一化公式
                vals = np.clip((3072.0 - vals) / 3072.0, 0.0, 1.0)
            else:
                vals = np.clip(vals, 0.0, 1.0)

            valid_mask = ~np.isnan(vals[0])
            if not np.any(valid_mask): continue
            grid = vals[:, valid_mask]

            if grid.size == 0: continue

            c, _ = np.histogram(grid, bins=BINS, range=RANGE)
            counts_all += c
            c_act, _ = np.histogram(grid[grid > 0.01], bins=BINS, range=RANGE)
            counts_active += c_act

            metrics_list.append(compute_frame_metrics(grid))

    return counts_all, counts_active, metrics_list

def process_opentouch(h5_path):
    import h5py

    res_raw = (np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), [])
    res_sub = (np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), [])
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'data' not in f: return res_raw, res_sub
            for demo_name in f['data']:
                demo = f['data'][demo_name]
                r_all, r_act, r_met = extract_from_dict(demo, ["left_pressure", "right_pressure"], is_ot_raw=True)
                s_all, s_act, s_met = extract_from_dict(demo, ["left_pressure_continuous_subdiv", "right_pressure_continuous_subdiv"], is_ot_raw=False)

                res_raw[0][:] += r_all; res_raw[1][:] += r_act; res_raw[2].extend(r_met)
                res_sub[0][:] += s_all; res_sub[1][:] += s_act; res_sub[2].extend(s_met)
    except Exception as e:
        pass
    return res_raw, res_sub

def process_touchanything(npz_path):
    res_raw = (np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), [])
    res_sub = (np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), [])
    try:
        data = np.load(npz_path)
        r_all, r_act, r_met = extract_from_dict(data, ["left_pressure_grid", "right_pressure_grid"], is_ot_raw=False)
        s_all, s_act, s_met = extract_from_dict(data, ["left_pressure_continuous_subdiv", "right_pressure_continuous_subdiv"], is_ot_raw=False)

        res_raw[0][:] += r_all; res_raw[1][:] += r_act; res_raw[2].extend(r_met)
        res_sub[0][:] += s_all; res_sub[1][:] += s_act; res_sub[2].extend(s_met)
    except Exception as e:
        pass
    return res_raw, res_sub

def combine_metrics(metrics_lists):
    combined = {}
    if not metrics_lists: return combined
    # find first non-empty
    keys = []
    for ml in metrics_lists:
        if ml:
            keys = ml[0].keys()
            break
    for k in keys:
        arrs = [m[k] for mlist in metrics_lists for m in mlist if k in m]
        if arrs:
            combined[k] = np.concatenate(arrs)
    return combined

def print_summary(name, metrics):
    print(f"\n--- {name} Summary ---")
    if not metrics or 'contact_ratio' not in metrics:
        print("No data collected.")
        return
    print(f"Total frames evaluated: {len(metrics['contact_ratio'])}")
    for k, v in metrics.items():
        print(f"  [{k:<15}] Mean: {np.mean(v):.4f} | Median (P50): {np.median(v):.4f} | P90: {np.percentile(v, 90):.4f} | Max: {np.max(v):.4f}")


def has_counts(counts):
    return counts is not None and np.sum(counts) > 0


def has_metric(metrics, key):
    return metrics is not None and key in metrics and len(metrics[key]) > 0


def clean_metric(metrics, key):
    vals = metrics[key]
    return vals[~np.isnan(vals)]


def plot_pdf(ax, counts, label, color, fill=False, alpha=0.15):
    if not has_counts(counts):
        return False
    pdf = counts / counts.sum()
    ax.plot(BIN_CENTERS, pdf, label=label, color=color)
    if fill:
        ax.fill_between(BIN_CENTERS, pdf, alpha=alpha, color=color)
    return True


def plot_metric_hist(ax, metrics, key, label, color):
    if not has_metric(metrics, key):
        return False
    data = clean_metric(metrics, key)
    if data.size == 0:
        return False
    ax.hist(data, bins=50, density=True, alpha=0.5, color=color, label=label)
    return True


def show_legend_if_plotted(ax, plotted):
    if plotted:
        ax.legend()


def warn_if_empty(name, path, files, counts, metrics):
    if files and has_counts(counts) and metrics:
        return
    print(f"⚠️  {name} 没有收集到可画的数据。路径: {path} | 文件数: {len(files)}")

def main():
    args = parse_args()
    opentouch_dir = args.opentouch_dir
    touchanything_dir = args.touchanything_dir
    egotactile_dir = args.egotactile_dir
    os.makedirs(args.output_dir, exist_ok=True)

    print("🔎 正在扫描数据集文件...")
    ot_files = find_opentouch_files(opentouch_dir)
    print(f"找到 {len(ot_files)} 个 OpenTouch HDF5 文件.")

    ta_files = find_named_files(
        touchanything_dir,
        "pressure_grids.npz",
        args.scan_workers,
        args.scan_exclude_dirs,
        normalized_depth(args.touchanything_scan_depth),
        args.touchanything_scan_split_depth,
        "Scanning TouchAnything",
    )
    print(f"找到 {len(ta_files)} 个 TouchAnything npz 文件.")

    et_files = find_named_files(
        egotactile_dir,
        "data.json",
        args.scan_workers,
        args.scan_exclude_dirs,
        normalized_depth(args.egotactile_scan_depth),
        args.egotactile_scan_split_depth,
        "Scanning EgoTactile JSON",
    )
    print(f"找到 {len(et_files)} 个 EgoTactile json 文件.")

    et_npz_files = find_named_files(
        egotactile_dir,
        args.egotactile_npz_name,
        args.scan_workers,
        args.scan_exclude_dirs,
        normalized_depth(args.egotactile_scan_depth),
        args.egotactile_scan_split_depth,
        "Scanning EgoTactile Gaussian",
    )
    print(f"找到 {len(et_npz_files)} 个 EgoTactile Gaussian npz 文件.")

    ot_raw_c_all, ot_raw_c_act, ot_raw_m = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), []
    ot_sub_c_all, ot_sub_c_act, ot_sub_m = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), []

    ta_raw_c_all, ta_raw_c_act, ta_raw_m = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), []
    ta_sub_c_all, ta_sub_c_act, ta_sub_m = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), []

    et_raw_c_all, et_raw_c_act, et_raw_m = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), []
    et_sub_c_all, et_sub_c_act, et_sub_m = np.zeros(BINS, dtype=np.int64), np.zeros(BINS, dtype=np.int64), []

    print("\n🚀 开始并行处理 OpenTouch 数据...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_opentouch, f) for f in ot_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res_raw, res_sub = future.result()
            ot_raw_c_all += res_raw[0]; ot_raw_c_act += res_raw[1]; ot_raw_m.append(res_raw[2])
            ot_sub_c_all += res_sub[0]; ot_sub_c_act += res_sub[1]; ot_sub_m.append(res_sub[2])

    print("\n🚀 开始并行处理 TouchAnything 数据...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_touchanything, f) for f in ta_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res_raw, res_sub = future.result()
            ta_raw_c_all += res_raw[0]; ta_raw_c_act += res_raw[1]; ta_raw_m.append(res_raw[2])
            ta_sub_c_all += res_sub[0]; ta_sub_c_act += res_sub[1]; ta_sub_m.append(res_sub[2])

    print("\n🚀 开始并行处理 EgoTactile 数据...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_egotactile, f) for f in et_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res_raw = future.result()
            et_raw_c_all += res_raw[0]; et_raw_c_act += res_raw[1]; et_raw_m.append(res_raw[2])

    print("\n🚀 开始并行处理 EgoTactile Gaussian 数据...")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_egotactile_gaussian, f) for f in et_npz_files]
        for future in tqdm(as_completed(futures), total=len(futures)):
            res_sub = future.result()
            et_sub_c_all += res_sub[0]; et_sub_c_act += res_sub[1]; et_sub_m.append(res_sub[2])

    ot_raw_combined = combine_metrics(ot_raw_m)
    ot_sub_combined = combine_metrics(ot_sub_m)
    ta_raw_combined = combine_metrics(ta_raw_m)
    ta_sub_combined = combine_metrics(ta_sub_m)
    et_raw_combined = combine_metrics(et_raw_m)
    et_sub_combined = combine_metrics(et_sub_m)

    print_summary("OpenTouch (RAW Normalized)", ot_raw_combined)
    print_summary("TouchAnything (RAW Grid)", ta_raw_combined)
    print_summary("EgoTactile (RAW Normalized)", et_raw_combined)
    print_summary("OpenTouch (Gaussian Subdiv)", ot_sub_combined)
    print_summary("TouchAnything (Gaussian Subdiv)", ta_sub_combined)
    print_summary("EgoTactile (Gaussian Subdiv)", et_sub_combined)

    warn_if_empty("OpenTouch RAW", opentouch_dir, ot_files, ot_raw_c_all, ot_raw_combined)
    warn_if_empty("TouchAnything RAW", touchanything_dir, ta_files, ta_raw_c_all, ta_raw_combined)
    warn_if_empty("EgoTactile RAW", egotactile_dir, et_files, et_raw_c_all, et_raw_combined)
    warn_if_empty("OpenTouch SUBDIV", opentouch_dir, ot_files, ot_sub_c_all, ot_sub_combined)
    warn_if_empty("TouchAnything SUBDIV", touchanything_dir, ta_files, ta_sub_c_all, ta_sub_combined)
    warn_if_empty("EgoTactile SUBDIV", egotactile_dir, et_npz_files, et_sub_c_all, et_sub_combined)

    print("\n📊 正在生成图表...")

    # --- 1. 基础像素直方图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plotted = False
    plotted |= plot_pdf(axes[0, 0], ot_raw_c_all, 'OT Raw', 'blue')
    plotted |= plot_pdf(axes[0, 0], ta_raw_c_all, 'TA Raw', 'orange')
    plotted |= plot_pdf(axes[0, 0], et_raw_c_all, 'ET Raw', 'green')
    axes[0, 0].set_title('[RAW] All Values')
    axes[0, 0].set_yscale('log')
    show_legend_if_plotted(axes[0, 0], plotted)

    plotted = False
    plotted |= plot_pdf(axes[0, 1], ot_raw_c_act, 'OT Raw Active', 'blue', fill=True, alpha=0.1)
    plotted |= plot_pdf(axes[0, 1], ta_raw_c_act, 'TA Raw Active', 'orange', fill=True, alpha=0.1)
    plotted |= plot_pdf(axes[0, 1], et_raw_c_act, 'ET Raw Active', 'green', fill=True, alpha=0.1)
    axes[0, 1].set_title('[RAW] Active Values (>0.01)')
    show_legend_if_plotted(axes[0, 1], plotted)

    plotted = False
    plotted |= plot_pdf(axes[1, 0], ot_sub_c_all, 'OT Subdiv', 'blue')
    plotted |= plot_pdf(axes[1, 0], ta_sub_c_all, 'TA Subdiv', 'orange')
    plotted |= plot_pdf(axes[1, 0], et_sub_c_all, 'ET Subdiv', 'green')
    axes[1, 0].set_title('[SUBDIV] All Values')
    axes[1, 0].set_yscale('log')
    show_legend_if_plotted(axes[1, 0], plotted)

    plotted = False
    plotted |= plot_pdf(axes[1, 1], ot_sub_c_act, 'OT Subdiv Active', 'blue', fill=True, alpha=0.2)
    plotted |= plot_pdf(axes[1, 1], ta_sub_c_act, 'TA Subdiv Active', 'orange', fill=True, alpha=0.2)
    plotted |= plot_pdf(axes[1, 1], et_sub_c_act, 'ET Subdiv Active', 'green', fill=True, alpha=0.2)
    axes[1, 1].set_title('[SUBDIV] Active Values (>0.01)')
    show_legend_if_plotted(axes[1, 1], plotted)

    plt.tight_layout()
    pressure_fig_path = os.path.join(args.output_dir, "pressure_dist_comparison.png")
    plt.savefig(pressure_fig_path, dpi=300)
    print(f"✅ 压力值分布图已保存至: {pressure_fig_path}")

    # --- 2. 帧级别聚合指标 ---
    if ot_raw_combined or ta_raw_combined or et_raw_combined or ot_sub_combined or ta_sub_combined or et_sub_combined:
        # 画一个 4x3 的超大图，上两行是 RAW，下两行是 SUBDIV
        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        metrics_keys = ['contact_ratio', 'max_pressure', 'mean_active', 'p50', 'p90', 'p99']
        titles = ['Contact Ratio', 'Max Pressure', 'Mean Active', 'P50', 'P90', 'P99']

        # Plot RAW (Top 2 rows)
        for i, (key, title) in enumerate(zip(metrics_keys, titles)):
            ax = axes[i // 3, i % 3]
            plotted = False
            plotted |= plot_metric_hist(ax, ot_raw_combined, key, 'OpenTouch', 'blue')
            plotted |= plot_metric_hist(ax, ta_raw_combined, key, 'TouchAnything', 'orange')
            plotted |= plot_metric_hist(ax, et_raw_combined, key, 'EgoTactile', 'green')
            ax.set_title(f"[RAW] {title}")
            show_legend_if_plotted(ax, plotted)

        # Plot SUBDIV (Bottom 2 rows)
        for i, (key, title) in enumerate(zip(metrics_keys, titles)):
            ax = axes[2 + (i // 3), i % 3]
            plotted = False
            plotted |= plot_metric_hist(ax, ot_sub_combined, key, 'OpenTouch (Subdiv)', 'blue')
            plotted |= plot_metric_hist(ax, ta_sub_combined, key, 'TouchAnything (Subdiv)', 'orange')
            plotted |= plot_metric_hist(ax, et_sub_combined, key, 'EgoTactile (Subdiv)', 'green')
            ax.set_title(f"[SUBDIV] {title}")
            show_legend_if_plotted(ax, plotted)

        plt.tight_layout()
        metrics_fig_path = os.path.join(args.output_dir, "frame_metrics_comparison.png")
        plt.savefig(metrics_fig_path, dpi=300)
        print(f"✅ 指标统计分布图已保存至: {metrics_fig_path}")
    else:
        print("⚠️  没有任何帧级指标数据，跳过 frame_metrics_comparison.png。")

if __name__ == "__main__":
    main()
