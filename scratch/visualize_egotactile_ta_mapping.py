import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_EGOTACTILE_DIR = "/data1/jiangrui/EgoTactile/Raw_data"
DEFAULT_SCAN_EXCLUDE_DIRS = {".git", "__pycache__", "extracted_frames", "metadata", "artifacts"}
HAND_TO_JSON_KEY = {
    "right": "RH",
    "left": "LH",
}
MANO_TRANSFORMS = [
    "auto",
    "none",
    "flip_lr",
    "flip_ud",
    "rot180",
    "transpose",
    "transpose_flip_lr",
    "transpose_flip_ud",
]


def load_frame_list(path):
    with open(path, "r") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        frames = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                frames.append(json.loads(line))
        return frames


def normalize_sensor(sensor, pmin=5.0, pmax=200.0):
    values = np.asarray(sensor, dtype=np.float32).reshape(-1)
    if values.size != 256:
        raise ValueError(f"Expected sensor_256 with 256 values, got {values.size}")

    if pmax <= pmin:
        raise ValueError(f"pmax must be greater than pmin, got pmin={pmin}, pmax={pmax}")
    values = np.clip(values, pmin, pmax)
    values = (values - pmin) / (pmax - pmin)
    return values


def frame_has_requested_hand(frame, requested_hand, pmin, pmax):
    for hand in available_hands(frame, requested_hand):
        key = HAND_TO_JSON_KEY[hand]
        if key not in frame or "sensor_256" not in frame[key]:
            continue
        try:
            normalize_sensor(frame[key]["sensor_256"], pmin=pmin, pmax=pmax)
        except ValueError:
            continue
        return True
    return False


def find_data_json_files(root, exclude_dirs=DEFAULT_SCAN_EXCLUDE_DIRS):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in exclude_dirs]
        if "data.json" in filenames and "video.mp4" in filenames:
            files.append(Path(dirpath) / "data.json")
    return sorted(files)


def resolve_random_data_json(root, rng, requested_frame=-1, requested_hand="auto", pmin=5.0, pmax=200.0):
    files = find_data_json_files(root)
    if not files:
        raise FileNotFoundError(f"No data.json found under {root}")

    if requested_frame < 0:
        return rng.choice(files)

    candidates = []
    skipped_unreadable = 0
    for path in files:
        try:
            frames = load_frame_list(path)
        except Exception:
            skipped_unreadable += 1
            continue
        if requested_frame >= len(frames):
            continue
        if frame_has_requested_hand(frames[requested_frame], requested_hand, pmin, pmax):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No sequence under {root} has frame {requested_frame} "
            f"with hand={requested_hand} sensor_256 data. "
            f"Skipped unreadable files: {skipped_unreadable}"
        )

    print(
        f"Random sequence candidates for frame {requested_frame}, "
        f"hand={requested_hand}: {len(candidates)}"
    )
    return rng.choice(candidates)


def available_hands(frame, requested_hand):
    if requested_hand == "auto":
        hands = []
        task_hand = str(frame.get("task_hand", "")).lower()
        if task_hand.startswith("r"):
            hands.append("right")
        elif task_hand.startswith("l"):
            hands.append("left")
        hands.extend(["right", "left"])
    else:
        hands = [requested_hand]

    deduped = []
    for hand in hands:
        if hand not in deduped:
            deduped.append(hand)
    return deduped


def choose_frame(frames, requested_frame, requested_hand, min_active, pmin, pmax, rng):
    if requested_frame >= 0:
        frame = frames[requested_frame]
        for hand in available_hands(frame, requested_hand):
            key = HAND_TO_JSON_KEY[hand]
            if key in frame and "sensor_256" in frame[key]:
                sensor = normalize_sensor(frame[key]["sensor_256"], pmin=pmin, pmax=pmax)
                return requested_frame, hand, frame, sensor
        raise ValueError(f"Frame {requested_frame} does not contain requested hand data")

    candidates = []
    for idx, frame in enumerate(frames):
        for hand in available_hands(frame, requested_hand):
            key = HAND_TO_JSON_KEY[hand]
            if key not in frame or "sensor_256" not in frame[key]:
                continue
            try:
                sensor = normalize_sensor(frame[key]["sensor_256"], pmin=pmin, pmax=pmax)
            except ValueError:
                continue
            if int(np.count_nonzero(sensor > 0.0)) >= min_active:
                candidates.append((idx, hand, frame, sensor))
                break

    if not candidates:
        raise ValueError(
            f"No frame with at least {min_active} active sensors found "
            f"for hand={requested_hand}"
        )
    return rng.choice(candidates)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def sensor_to_21_grid(sensor_norm, mapping_path):
    mapping = load_json(mapping_path)
    grid = np.full((21, 21), np.nan, dtype=np.float32)
    for key, sensor_idx in mapping.items():
        row, col = [int(x) for x in key.split(",")]
        sensor_idx = int(sensor_idx)
        if 0 <= row < 21 and 0 <= col < 21 and 0 <= sensor_idx < sensor_norm.size:
            grid[row, col] = sensor_norm[sensor_idx]
    return grid


def resolve_mano_transform(hand, requested_transform):
    if requested_transform != "auto":
        return requested_transform
    # TA->MANO visual mappings use display coordinates; the right-hand layout is
    # mirrored relative to the pressure-position mapping used to build the grid.
    return "flip_lr" if hand == "right" else "none"


def transform_grid_for_mano(grid, transform):
    if transform == "none":
        return grid.copy()
    if transform == "flip_lr":
        return np.fliplr(grid)
    if transform == "flip_ud":
        return np.flipud(grid)
    if transform == "rot180":
        return np.rot90(grid, 2)
    if transform == "transpose":
        return grid.T
    if transform == "transpose_flip_lr":
        return np.fliplr(grid.T)
    if transform == "transpose_flip_ud":
        return np.flipud(grid.T)
    raise ValueError(f"Unknown MANO grid transform: {transform}")


def load_obj_vertices(obj_path):
    vertices = []
    with open(obj_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not vertices:
        raise ValueError(f"No vertices found in {obj_path}")
    return np.asarray(vertices, dtype=np.float32)


def grid_to_mano_values(grid, mano_mapping_path, num_vertices):
    mapping_data = load_json(mano_mapping_path)
    positions = mapping_data.get("positions", {})

    vertex_values = np.zeros(num_vertices, dtype=np.float32)
    touched_vertices = np.zeros(num_vertices, dtype=bool)
    used_nodes = 0

    for node_id, info in positions.items():
        row, col = [int(x) for x in node_id.split(",")]
        if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
            continue
        value = grid[row, col]
        if np.isnan(value):
            continue

        vids = info.get("mano_vid", [])
        if not vids:
            continue
        used_nodes += 1
        for vid in vids:
            vid = int(vid)
            if 0 <= vid < num_vertices:
                vertex_values[vid] = max(vertex_values[vid], float(value))
                touched_vertices[vid] = True

    return vertex_values, touched_vertices, used_nodes


def resolve_video_path(data_json, requested_video):
    if requested_video:
        return Path(requested_video)
    return Path(data_json).with_name("video.mp4")


def read_video_frame(video_path, frame_index):
    video_path = Path(video_path)
    if not video_path.exists():
        return None, f"video not found: {video_path}"

    try:
        import cv2
    except ImportError:
        return None, "opencv-python/cv2 is not installed"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, f"failed to open video: {video_path}"

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0 and frame_index >= total:
        cap.release()
        return None, f"frame {frame_index} out of range for video with {total} frames"

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None, f"failed to read frame {frame_index} from {video_path}"

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, None


def active_grid_nodes(grid):
    coords = np.argwhere(~np.isnan(grid) & (grid > 0))
    return [(int(r), int(c), float(grid[r, c])) for r, c in coords]


def draw_grid(ax, grid, title, cmap="inferno"):
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, origin="upper", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    for row, col, value in active_grid_nodes(grid):
        ax.text(
            col,
            row,
            f"{row},{col}",
            color="white",
            fontsize=6,
            ha="center",
            va="center",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def draw_rgb_frame(ax, rgb_frame, error, title):
    if rgb_frame is not None:
        ax.imshow(rgb_frame)
        ax.set_title(title)
    else:
        ax.set_facecolor("#f3f3f3")
        ax.text(
            0.5,
            0.5,
            error or "RGB frame unavailable",
            ha="center",
            va="center",
            wrap=True,
            fontsize=9,
        )
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_projection(ax, vertices, vertex_values, touched_vertices, dims, title):
    x = vertices[:, dims[0]]
    y = vertices[:, dims[1]]
    ax.scatter(x, y, s=1.0, c="#d0d0d0", alpha=0.12, linewidths=0)

    positive = touched_vertices & (vertex_values > 0.0)
    if np.any(positive):
        sc = ax.scatter(
            x[positive],
            y[positive],
            s=5.0,
            c=vertex_values[positive],
            cmap="inferno",
            vmin=0,
            vmax=1,
            linewidths=0,
        )
    else:
        sc = None

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def make_figure(
    sensor_norm,
    grid21,
    mano_grid21,
    rgb_frame,
    rgb_error,
    vertices,
    vertex_values,
    touched_vertices,
    meta,
    output_path,
):
    grid16 = sensor_norm.reshape(16, 16)

    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    draw_rgb_frame(axes[0, 0], rgb_frame, rgb_error, "RGB video frame")
    im0 = draw_grid(axes[0, 1], grid16, "EgoTactile sensor_256 reshaped to 16x16")
    im1 = draw_grid(axes[0, 2], grid21, "21x21 from pressure_position_mapping")
    im2 = draw_grid(axes[0, 3], mano_grid21, f"21x21 sent to MANO ({meta['mano_grid_transform']})")
    sc0 = draw_projection(axes[0, 4], vertices, vertex_values, touched_vertices, (0, 1), "MANO vertex projection: X/Y")
    sc1 = draw_projection(axes[1, 0], vertices, vertex_values, touched_vertices, (0, 2), "MANO vertex projection: X/Z")
    draw_projection(axes[1, 1], vertices, vertex_values, touched_vertices, (1, 2), "MANO vertex projection: Y/Z")

    active = sensor_norm[sensor_norm > 0]
    axes[1, 2].hist(active, bins=40, range=(0, 1), color="#3366cc", alpha=0.85)
    axes[1, 2].set_title("Active normalized sensor values")
    axes[1, 2].set_xlabel("pressure")
    axes[1, 2].set_ylabel("count")

    axes[1, 3].axis("off")
    active_text = "\n".join(
        [
            "Active 21x21 nodes:",
            f"raw:  {meta['active_grid_nodes']}",
            f"mano: {meta['active_mano_grid_nodes']}",
            "",
            "If raw and MANO nodes fall on different",
            "finger columns, the grid transform is the",
            "coordinate-system correction being tested.",
        ]
    )
    axes[1, 3].text(0.0, 1.0, active_text, ha="left", va="top", family="monospace", fontsize=8)
    axes[1, 4].axis("off")
    axes[1, 4].text(
        0.0,
        1.0,
        f"Video:\n{meta['video_path']}\n\nRGB status:\n{meta['rgb_status']}",
        ha="left",
        va="top",
        wrap=True,
        fontsize=8,
    )

    fig.colorbar(im0, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    if sc0 is not None:
        fig.colorbar(sc0, ax=axes[0, 4], fraction=0.046, pad=0.04)
    if sc1 is not None:
        fig.colorbar(sc1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    title = (
        f"{meta['data_json']} | frame={meta['frame_index']} | hand={meta['hand']} | "
        f"active={meta['active_sensors']} | max={meta['max_pressure']:.4f}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Visualize a random EgoTactile sensor_256 frame using the "
            "TouchAnything 16x16->21x21 and 21x21->MANO mapping assumptions."
        )
    )
    parser.add_argument("--egotactile_dir", default=DEFAULT_EGOTACTILE_DIR)
    parser.add_argument("--data_json", default=None)
    parser.add_argument("--video", default=None, help="Defaults to video.mp4 next to data.json")
    parser.add_argument("--hand", choices=["auto", "right", "left"], default="auto")
    parser.add_argument("--frame", type=int, default=-1, help="-1 means random active frame")
    parser.add_argument("--seed", type=int, default=None, help="Set this for reproducible random choices")
    parser.add_argument("--min_active", type=int, default=1)
    parser.add_argument("--pmin", type=float, default=5.0)
    parser.add_argument("--pmax", type=float, default=200.0)
    parser.add_argument(
        "--mano_grid_transform",
        choices=MANO_TRANSFORMS,
        default="auto",
        help="Transform 21x21 pressure grid before TA->MANO mapping.",
    )
    parser.add_argument(
        "--output",
        default=str(repo_root / "scratch" / "egotactile_ta_mapping_preview.png"),
    )
    parser.add_argument(
        "--ta_mapping_dir",
        default=str(repo_root / "TouchAnything" / "configs"),
    )
    parser.add_argument(
        "--mano_mapping_dir",
        default=str(repo_root / "TouchAnything" / "scripts" / "tools" / "mano_visualization"),
    )
    parser.add_argument(
        "--mesh",
        default=str(
            repo_root
            / "TouchAnything"
            / "scripts"
            / "tools"
            / "mano_visualization"
            / "scratch"
            / "mano_right_neutral_subdiv.obj"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed) if args.seed is not None else random.Random()

    data_json = (
        Path(args.data_json)
        if args.data_json
        else resolve_random_data_json(
            args.egotactile_dir,
            rng,
            requested_frame=args.frame,
            requested_hand=args.hand,
            pmin=args.pmin,
            pmax=args.pmax,
        )
    )
    frames = load_frame_list(data_json)
    if not frames:
        raise ValueError(f"No frames loaded from {data_json}")

    frame_index, hand, frame, sensor_norm = choose_frame(
        frames,
        requested_frame=args.frame,
        requested_hand=args.hand,
        min_active=args.min_active,
        pmin=args.pmin,
        pmax=args.pmax,
        rng=rng,
    )

    ta_mapping_path = Path(args.ta_mapping_dir) / f"pressure_position_mapping_{hand}.json"
    mano_mapping_path = Path(args.mano_mapping_dir) / f"ta_to_mano_mapping_{hand}_visual.json"
    video_path = resolve_video_path(data_json, args.video)

    grid21 = sensor_to_21_grid(sensor_norm, ta_mapping_path)
    mano_grid_transform = resolve_mano_transform(hand, args.mano_grid_transform)
    mano_grid21 = transform_grid_for_mano(grid21, mano_grid_transform)
    rgb_frame, rgb_error = read_video_frame(video_path, frame_index)
    vertices = load_obj_vertices(args.mesh)
    vertex_values, touched_vertices, used_nodes = grid_to_mano_values(
        mano_grid21,
        mano_mapping_path,
        num_vertices=vertices.shape[0],
    )

    meta = {
        "data_json": str(data_json),
        "frame_index": frame_index,
        "hand": hand,
        "active_sensors": int(np.count_nonzero(sensor_norm > 0)),
        "max_pressure": float(sensor_norm.max()) if sensor_norm.size else 0.0,
        "mapped_21x21_nodes": int(np.count_nonzero(~np.isnan(grid21))),
        "used_mano_nodes": int(used_nodes),
        "active_mano_vertices": int(np.count_nonzero(vertex_values > 0)),
        "mano_grid_transform": mano_grid_transform,
        "active_grid_nodes": active_grid_nodes(grid21),
        "active_mano_grid_nodes": active_grid_nodes(mano_grid21),
        "video_path": str(video_path),
        "rgb_status": "ok" if rgb_error is None else rgb_error,
    }

    make_figure(
        sensor_norm=sensor_norm,
        grid21=grid21,
        mano_grid21=mano_grid21,
        rgb_frame=rgb_frame,
        rgb_error=rgb_error,
        vertices=vertices,
        vertex_values=vertex_values,
        touched_vertices=touched_vertices,
        meta=meta,
        output_path=args.output,
    )

    print("Saved visualization:", args.output)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
