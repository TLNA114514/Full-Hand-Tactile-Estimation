#!/usr/bin/env python3
"""
Visualize saved WiLoR hand poses on top of chest videos.

This script does not run WiLoR inference. It only reads an existing
`wilor_hands.json` JSONL file and projects `left_pos` / `right_pos` onto the
corresponding `chest.mp4` frames with the same assumed pinhole intrinsics used
by our WiLoR processing script:

    focal_length = 5000 / 256 * max(image_width, image_height)
    principal_point = (image_width / 2, image_height / 2)

This is an approximate visualization of WiLoR-frame predictions, not a
calibrated physical camera projection. It does not use the dataset `chest_pose`
extrinsics.

Examples:
    python scripts/utils/visualize_wilor_from_json.py \
        --traj /path/to/trajectory

    python scripts/utils/visualize_wilor_from_json.py \
        --video /path/to/chest.mp4 \
        --json /path/to/wilor_hands.json \
        --out /path/to/wilor_annotated_from_json.mp4

    python scripts/utils/visualize_wilor_from_json.py \
        --root /path/to/TouchAnything_Datasets_opensource_en \
        --workers 8
"""

import argparse
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGER_EDGE_COLORS = [
    (80, 220, 255),   # thumb
    (80, 255, 120),   # index
    (255, 210, 80),   # middle
    (255, 130, 80),   # ring
    (210, 120, 255),  # pinky
]


def iter_json_records(path: Path) -> Iterable[dict]:
    """Read JSONL or a JSON list from `path`."""
    with path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                yield item


def load_wilor_frames(json_path: Path) -> Dict[int, dict]:
    """Load WiLoR records keyed by frame index."""
    frames: Dict[int, dict] = {}
    for sequential_idx, item in enumerate(iter_json_records(json_path)):
        frame_idx = item.get("frame_index", sequential_idx)
        try:
            frame_idx = int(frame_idx)
        except (TypeError, ValueError):
            frame_idx = sequential_idx
        frames[frame_idx] = item
    return frames


def parse_pose(value: object) -> Optional[np.ndarray]:
    """Return a valid (21, 3) pose array, or None."""
    if value is None or value == []:
        return None
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if pose.shape != (21, 3):
        return None
    if not np.isfinite(pose).all():
        return None
    if np.any(pose[:, 2] <= 1e-8):
        return None
    if np.abs(pose[:, :2]).max() > 1000 or pose[:, 2].max() > 1000:
        return None
    return pose


def default_focal_length(width: int, height: int) -> float:
    """Match WiLoR demo_sam3_video_fast.py projection scaling: 5000 / 256 * image_size."""
    return 5000.0 / 256.0 * max(width, height)


def project_points(points_3d_cam: np.ndarray, focal_length: float, width: int, height: int) -> np.ndarray:
    z = np.clip(points_3d_cam[:, 2], 1e-8, None)
    u = focal_length * (points_3d_cam[:, 0] / z) + width / 2.0
    v = focal_length * (points_3d_cam[:, 1] / z) + height / 2.0
    return np.stack([u, v], axis=1)


def point_is_drawable(point: np.ndarray, width: int, height: int, margin: int = 80) -> bool:
    if not np.isfinite(point).all():
        return False
    x, y = point
    return -margin <= x <= width + margin and -margin <= y <= height + margin


def draw_text_with_outline(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_hand(
    frame: np.ndarray,
    joints_2d: np.ndarray,
    is_right: bool,
    draw_labels: bool = True,
) -> None:
    height, width = frame.shape[:2]
    joint_color = (255, 80, 80) if is_right else (80, 80, 255)
    label = "R" if is_right else "L"

    for edge_idx, (i, j) in enumerate(HAND_EDGES):
        p1 = joints_2d[i]
        p2 = joints_2d[j]
        if not (point_is_drawable(p1, width, height) and point_is_drawable(p2, width, height)):
            continue
        color = FINGER_EDGE_COLORS[min(edge_idx // 4, len(FINGER_EDGE_COLORS) - 1)]
        pt1 = tuple(np.round(p1).astype(int))
        pt2 = tuple(np.round(p2).astype(int))
        cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)

    for idx, point in enumerate(joints_2d):
        if not point_is_drawable(point, width, height):
            continue
        center = tuple(np.round(point).astype(int))
        radius = 5 if idx == 0 else 3
        cv2.circle(frame, center, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, joint_color, -1, cv2.LINE_AA)

    if draw_labels and point_is_drawable(joints_2d[0], width, height):
        wrist = np.round(joints_2d[0]).astype(int)
        draw_text_with_outline(frame, label, (int(wrist[0]) + 8, int(wrist[1]) - 8), joint_color)


class FFmpegVideoWriter:
    """Encode BGR frames to browser/VS Code friendly H.264 MP4."""

    def __init__(self, path: Path, fps: float, width: int, height: int, crf: int = 18) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.width = width
        self.height = height
        fps_arg = f"{fps:.6f}".rstrip("0").rstrip(".")
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", fps_arg,
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-movflags", "+faststart",
            str(path),
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.proc.stdin is None:
            raise RuntimeError("Failed to open ffmpeg stdin")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape[:2]} does not match writer size {(self.height, self.width)}"
            )
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            stderr = self._read_stderr()
            raise RuntimeError(f"ffmpeg pipe closed early while writing {self.path}: {stderr}") from exc

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        returncode = self.proc.wait()
        stderr = self._read_stderr()
        if returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path} with code {returncode}: {stderr}")

    def _read_stderr(self) -> str:
        if self.proc.stderr is None:
            return ""
        data = self.proc.stderr.read()
        if not data:
            return ""
        return data.decode("utf-8", errors="replace").strip()


def make_writer(path: Path, fps: float, width: int, height: int) -> FFmpegVideoWriter:
    return FFmpegVideoWriter(path, fps, width, height)


def visualize_one(
    video_path: Path,
    json_path: Path,
    out_path: Path,
    focal_length: Optional[float] = None,
    max_frames: Optional[int] = None,
    force: bool = False,
    draw_labels: bool = True,
) -> Tuple[bool, str]:
    if out_path.exists() and not force:
        return True, f"exists: {out_path}"
    if not video_path.exists():
        return False, f"missing video: {video_path}"
    if not json_path.exists():
        return False, f"missing json: {json_path}"

    poses_by_frame = load_wilor_frames(json_path)
    if not poses_by_frame:
        return False, f"empty json: {json_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, f"failed to open video: {video_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = total_frames
    if max_frames is not None:
        frame_limit = min(frame_limit, max_frames)

    if focal_length is None:
        focal_length = default_focal_length(width, height)

    writer = make_writer(out_path, fps, width, height)
    valid_left = 0
    valid_right = 0
    written = 0

    try:
        for frame_idx in range(frame_limit):
            ok, frame = cap.read()
            if not ok:
                break

            item = poses_by_frame.get(frame_idx)
            if item is not None:
                left_pose = parse_pose(item.get("left_pos"))
                right_pose = parse_pose(item.get("right_pos"))

                if left_pose is not None:
                    left_2d = project_points(left_pose, focal_length, width, height)
                    draw_hand(frame, left_2d, is_right=False, draw_labels=draw_labels)
                    valid_left += 1

                if right_pose is not None:
                    right_2d = project_points(right_pose, focal_length, width, height)
                    draw_hand(frame, right_2d, is_right=True, draw_labels=draw_labels)
                    valid_right += 1

            draw_text_with_outline(
                frame,
                f"frame {frame_idx}",
                (12, 24),
                (255, 255, 255),
                scale=0.55,
                thickness=1,
            )
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.close()

    if written == 0:
        return False, f"no frames written: {video_path}"

    return True, (
        f"saved: {out_path} "
        f"(frames={written}, left_valid={valid_left}, right_valid={valid_right}, focal={focal_length:.1f})"
    )


def find_trajectories(root: Path) -> List[Path]:
    trajs = []
    for chest in root.rglob("chest.mp4"):
        traj_dir = chest.parent
        if (traj_dir / "wilor_hands.json").exists():
            trajs.append(traj_dir)
    return sorted(trajs)


def visualize_traj(args_tuple: tuple) -> Tuple[Path, bool, str]:
    traj_dir, output_name, focal_length, max_frames, force, draw_labels = args_tuple
    video_path = traj_dir / "chest.mp4"
    json_path = traj_dir / "wilor_hands.json"
    out_path = traj_dir / output_name
    ok, msg = visualize_one(
        video_path=video_path,
        json_path=json_path,
        out_path=out_path,
        focal_length=focal_length,
        max_frames=max_frames,
        force=force,
        draw_labels=draw_labels,
    )
    return traj_dir, ok, msg


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize existing WiLoR JSON poses on chest videos using the same "
            "assumed intrinsics as the WiLoR processing script."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--traj", type=Path, help="Trajectory directory containing chest.mp4 and wilor_hands.json")
    mode.add_argument("--root", type=Path, help="Dataset root; process all trajectory dirs recursively")
    mode.add_argument("--video", type=Path, help="Input video path for explicit single-file mode")

    parser.add_argument("--json", type=Path, help="WiLoR JSON path for --video mode")
    parser.add_argument("--out", type=Path, help="Output video path for --traj or --video mode")
    parser.add_argument("--output_name", type=str, default="wilor_annotated_from_json.mp4",
                        help="Output file name inside each trajectory for --root/--traj")
    parser.add_argument("--focal_length", type=float, default=None,
                        help=(
                            "Projection focal length. Default matches WiLoR: "
                            "5000/256*max(width,height). This is not a calibrated "
                            "camera intrinsic."
                        ))
    parser.add_argument("--max_frames", type=int, default=None, help="Visualize only the first N frames")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for --root")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--no_labels", action="store_true", help="Do not draw L/R wrist labels")
    args = parser.parse_args()

    draw_labels = not args.no_labels

    if args.video is not None:
        if args.json is None:
            parser.error("--json is required when using --video")
        out_path = args.out
        if out_path is None:
            out_path = args.video.with_name("wilor_annotated_from_json.mp4")
        ok, msg = visualize_one(
            video_path=args.video,
            json_path=args.json,
            out_path=out_path,
            focal_length=args.focal_length,
            max_frames=args.max_frames,
            force=args.force,
            draw_labels=draw_labels,
        )
        print(msg)
        return 0 if ok else 1

    if args.traj is not None:
        out_path = args.out if args.out is not None else args.traj / args.output_name
        ok, msg = visualize_one(
            video_path=args.traj / "chest.mp4",
            json_path=args.traj / "wilor_hands.json",
            out_path=out_path,
            focal_length=args.focal_length,
            max_frames=args.max_frames,
            force=args.force,
            draw_labels=draw_labels,
        )
        print(msg)
        return 0 if ok else 1

    trajs = find_trajectories(args.root)
    if not trajs:
        print(f"No trajectories with chest.mp4 and wilor_hands.json found under {args.root}")
        return 1

    print(f"Found {len(trajs)} trajectories")
    tasks = [
        (traj, args.output_name, args.focal_length, args.max_frames, args.force, draw_labels)
        for traj in trajs
    ]

    failed = 0
    if args.workers <= 1:
        for task in tasks:
            traj, ok, msg = visualize_traj(task)
            print(f"[{'OK' if ok else 'FAIL'}] {traj}: {msg}")
            failed += 0 if ok else 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(visualize_traj, task) for task in tasks]
            for future in as_completed(futures):
                traj, ok, msg = future.result()
                print(f"[{'OK' if ok else 'FAIL'}] {traj}: {msg}")
                failed += 0 if ok else 1

    print(f"Done. failed={failed}, total={len(trajs)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
