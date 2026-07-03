import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TOUCHANYTHING_DIR = "/data1/jiangrui/EgoTouch"
MANO_TRANSFORMS = ["auto", "none", "flip_lr", "flip_ud", "rot180"]


def resolve_random_clip(root, rng, frame_index):
    npz_files = sorted(Path(root).glob("**/pressure_grids.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No pressure_grids.npz found under {root}")

    if frame_index < 0:
        return rng.choice(npz_files).parent

    candidates = []
    for npz_path in npz_files:
        try:
            data = np.load(npz_path)
            ok = any(
                key in data and data[key].ndim == 3 and frame_index < data[key].shape[0]
                for key in ("left_pressure_grid", "right_pressure_grid")
            )
            data.close()
        except Exception:
            ok = False
        if ok:
            candidates.append(npz_path.parent)

    if not candidates:
        raise FileNotFoundError(f"No TouchAnything clip has frame {frame_index} under {root}")
    print(f"Random clip candidates for frame {frame_index}: {len(candidates)}")
    return rng.choice(candidates)


def resolve_mano_transform(hand, requested_transform):
    if requested_transform != "auto":
        return requested_transform
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
    raise ValueError(f"Unknown transform: {transform}")


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


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
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), None


def active_grid_nodes(grid):
    coords = np.argwhere(~np.isnan(grid) & (grid > 0))
    return [(int(r), int(c), float(grid[r, c])) for r, c in coords]


def draw_rgb(ax, frame, error, title):
    if frame is not None:
        ax.imshow(frame)
    else:
        ax.set_facecolor("#f2f2f2")
        ax.text(0.5, 0.5, error or "unavailable", ha="center", va="center", wrap=True, fontsize=8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_grid(ax, grid, title):
    im = ax.imshow(np.ma.masked_invalid(grid), origin="upper", interpolation="nearest", cmap="inferno", vmin=0, vmax=1)
    for row, col, value in active_grid_nodes(grid):
        ax.text(col, row, f"{row},{col}", color="white", fontsize=6, ha="center", va="center")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def draw_projection(ax, vertices, vertex_values, touched_vertices, dims, title):
    x = vertices[:, dims[0]]
    y = vertices[:, dims[1]]
    ax.scatter(x, y, s=1.0, c="#d0d0d0", alpha=0.12, linewidths=0)
    positive = touched_vertices & (vertex_values > 0)
    sc = None
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
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return sc


def make_figure(meta, rgb_frames, raw_grid, mano_grid, vertices, vertex_values, touched_vertices, output):
    fig, axes = plt.subplots(2, 5, figsize=(23, 9))
    draw_rgb(axes[0, 0], rgb_frames["chest"][0], rgb_frames["chest"][1], "chest.mp4")
    draw_rgb(axes[0, 1], rgb_frames["left"][0], rgb_frames["left"][1], "left.mp4")
    draw_rgb(axes[0, 2], rgb_frames["right"][0], rgb_frames["right"][1], "right.mp4")
    im0 = draw_grid(axes[0, 3], raw_grid, f"{meta['hand']} pressure_grid raw")
    im1 = draw_grid(axes[0, 4], mano_grid, f"grid sent to MANO ({meta['mano_grid_transform']})")

    sc0 = draw_projection(axes[1, 0], vertices, vertex_values, touched_vertices, (0, 1), "MANO X/Y")
    sc1 = draw_projection(axes[1, 1], vertices, vertex_values, touched_vertices, (0, 2), "MANO X/Z")
    draw_projection(axes[1, 2], vertices, vertex_values, touched_vertices, (1, 2), "MANO Y/Z")

    active = raw_grid[~np.isnan(raw_grid) & (raw_grid > 0)]
    axes[1, 3].hist(active, bins=40, range=(0, 1), color="#3366cc", alpha=0.85)
    axes[1, 3].set_title("Active pressure values")
    axes[1, 3].set_xlabel("pressure")
    axes[1, 3].set_ylabel("count")

    axes[1, 4].axis("off")
    axes[1, 4].text(
        0,
        1,
        "\n".join(
            [
                f"clip: {meta['clip_dir']}",
                f"frame: {meta['frame_index']}",
                f"hand: {meta['hand']}",
                f"raw active nodes: {meta['active_grid_nodes']}",
                f"mano active nodes: {meta['active_mano_grid_nodes']}",
                f"used MANO nodes: {meta['used_mano_nodes']}",
                f"active vertices: {meta['active_mano_vertices']}",
            ]
        ),
        ha="left",
        va="top",
        family="monospace",
        fontsize=8,
        wrap=True,
    )

    fig.colorbar(im0, ax=axes[0, 3], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 4], fraction=0.046, pad=0.04)
    if sc0 is not None:
        fig.colorbar(sc0, ax=axes[1, 0], fraction=0.046, pad=0.04)
    if sc1 is not None:
        fig.colorbar(sc1, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(f"{meta['clip_dir']} | frame={meta['frame_index']} | hand={meta['hand']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Visualize TouchAnything grid-to-MANO mapping on one frame.")
    parser.add_argument("--touchanything_dir", default=DEFAULT_TOUCHANYTHING_DIR)
    parser.add_argument("--clip_dir", default=None)
    parser.add_argument("--frame", type=int, default=-1, help="-1 means random frame")
    parser.add_argument("--hand", choices=["left", "right"], default="right")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_active", type=int, default=1)
    parser.add_argument("--mano_grid_transform", choices=MANO_TRANSFORMS, default="auto")
    parser.add_argument("--output", default=str(repo_root / "scratch" / "touchanything_ta_mapping_preview.png"))
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


def choose_frame_index(grid, requested_frame, min_active, rng):
    if requested_frame >= 0:
        if requested_frame >= grid.shape[0]:
            raise ValueError(f"Requested frame {requested_frame}, but grid only has {grid.shape[0]} frames")
        return requested_frame

    active_counts = np.sum(np.nan_to_num(grid, nan=0.0) > 0, axis=(1, 2))
    candidates = np.flatnonzero(active_counts >= min_active)
    if candidates.size == 0:
        raise ValueError(f"No frame has at least {min_active} active nodes")
    return int(rng.choice(candidates.tolist()))


def main():
    args = parse_args()
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    clip_dir = Path(args.clip_dir) if args.clip_dir else resolve_random_clip(args.touchanything_dir, rng, args.frame)
    npz_path = clip_dir / "pressure_grids.npz"
    data = np.load(npz_path)
    grid_key = f"{args.hand}_pressure_grid"
    if grid_key not in data:
        raise KeyError(f"{grid_key} not found in {npz_path}")
    pressure_grid = np.asarray(data[grid_key], dtype=np.float32)
    frame_index = choose_frame_index(pressure_grid, args.frame, args.min_active, rng)
    raw_grid = pressure_grid[frame_index]
    transform = resolve_mano_transform(args.hand, args.mano_grid_transform)
    mano_grid = transform_grid_for_mano(raw_grid, transform)

    rgb_frames = {}
    for name in ("chest", "left", "right"):
        rgb_frames[name] = read_video_frame(clip_dir / f"{name}.mp4", frame_index)

    vertices = load_obj_vertices(args.mesh)
    mano_mapping_path = Path(args.mano_mapping_dir) / f"ta_to_mano_mapping_{args.hand}_visual.json"
    vertex_values, touched_vertices, used_nodes = grid_to_mano_values(
        mano_grid,
        mano_mapping_path,
        num_vertices=vertices.shape[0],
    )

    meta = {
        "clip_dir": str(clip_dir),
        "frame_index": int(frame_index),
        "hand": args.hand,
        "mano_grid_transform": transform,
        "active_grid_nodes": active_grid_nodes(raw_grid),
        "active_mano_grid_nodes": active_grid_nodes(mano_grid),
        "used_mano_nodes": int(used_nodes),
        "active_mano_vertices": int(np.count_nonzero(vertex_values > 0)),
    }
    make_figure(meta, rgb_frames, raw_grid, mano_grid, vertices, vertex_values, touched_vertices, args.output)
    print("Saved visualization:", args.output)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
