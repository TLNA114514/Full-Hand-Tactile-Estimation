#!/usr/bin/env python3
"""Render deterministic contiguous AceData SAM3 bbox review clips."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import cv2


COLORS = {
    "left": (48, 210, 72),
    "right": (225, 80, 225),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def draw_label(
    image,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x - 4, y - height - 6),
        (x + width + 4, y + baseline + 4),
        (16, 16, 16),
        -1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_hand_box(image, hand: str, item: dict[str, Any]) -> bool:
    bbox = item.get("bbox")
    if bbox is None:
        return False
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return False
    color = COLORS[hand]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    score = float(item.get("score", 0.0))
    label = hand.upper()
    if score > 0.0:
        label += f"  score={score:.3f}"
    draw_label(image, label, (x1 + 4, max(y1 - 7, 24)), color)
    return True


def render_clip(
    row: dict[str, Any],
    bbox_root: Path,
    output_path: Path,
    sampled_indices: list[int],
    requested_output_fps: float,
) -> dict[str, Any]:
    video_path = Path(row["resource_path"])
    bbox_path = bbox_root / row["old_bbox_relpath"]
    boxes = json.loads(bbox_path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    output_fps = requested_output_fps if requested_output_fps > 0.0 else source_fps
    if output_fps <= 0.0:
        capture.release()
        raise RuntimeError(f"Invalid source/output FPS for {video_path}: {output_fps}")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {output_path}")
    wanted = set(sampled_indices)
    rendered = []
    left_count = 0
    right_count = 0
    both_count = 0
    frame_index = 0
    try:
        while frame_index <= sampled_indices[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                item = boxes[str(frame_index)]
                left_ok = draw_hand_box(frame, "left", item["left"])
                right_ok = draw_hand_box(frame, "right", item["right"])
                left_count += int(left_ok)
                right_count += int(right_ok)
                both_count += int(left_ok and right_ok)
                missing = []
                if not left_ok:
                    missing.append("LEFT")
                if not right_ok:
                    missing.append("RIGHT")
                draw_label(
                    frame,
                    f"{row['sequence_key']}  source frame {frame_index}",
                    (18, 30),
                    (245, 245, 245),
                )
                if missing:
                    draw_label(
                        frame,
                        "NO ACCEPTED SAM3 BOX: " + ", ".join(missing),
                        (18, height - 20),
                        (40, 40, 245),
                    )
                writer.write(frame)
                rendered.append(frame_index)
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    if rendered != sampled_indices:
        missing = sorted(set(sampled_indices) - set(rendered))
        raise RuntimeError(
            f"{video_path}: rendered {len(rendered)}/{len(sampled_indices)} sampled "
            f"frames; missing={missing[:10]}"
        )
    return {
        "sequence_key": row["sequence_key"],
        "video_path": str(video_path),
        "bbox_path": str(bbox_path),
        "output_video": str(output_path),
        "source_frame_count": int(row["frame_count"]),
        "source_fps": source_fps,
        "output_fps": output_fps,
        "sample_count": len(rendered),
        "sample_window_start": rendered[0],
        "sample_window_end_inclusive": rendered[-1],
        "sampled_frame_indices": rendered,
        "sample_left_box_count": left_count,
        "sample_right_box_count": right_count,
        "sample_both_box_count": both_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbox-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-count", type=int, default=5)
    parser.add_argument("--frames-per-clip", type=int, default=200)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="Output FPS. Zero preserves each source video's original FPS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clip_count < 1 or args.frames_per_clip < 1:
        raise ValueError("clip-count and frames-per-clip must be positive")
    if args.output_fps < 0.0:
        raise ValueError("output-fps must be zero or positive")
    rows = [
        row
        for row in read_jsonl(args.manifest.expanduser().resolve())
        if int(row["frame_count"]) >= args.frames_per_clip
    ]
    if len(rows) < args.clip_count:
        raise RuntimeError(
            f"Only {len(rows)} clips contain at least {args.frames_per_clip} frames"
        )
    generator = random.Random(args.seed)
    selected = generator.sample(rows, args.clip_count)
    output_dir = args.output_dir.expanduser().resolve()
    bbox_root = args.bbox_root.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for order, row in enumerate(selected, start=1):
        frame_count = int(row["frame_count"])
        start = generator.randrange(frame_count - args.frames_per_clip + 1)
        indices = list(range(start, start + args.frames_per_clip))
        output_path = output_dir / (
            f"{order:02d}_{safe_name(row['sequence_key'])}_sam3_bbox_review.mp4"
        )
        result = render_clip(row, bbox_root, output_path, indices, args.output_fps)
        records.append(result)
        print(
            f"[{order}/{len(selected)}] {row['sequence_key']}: "
            f"L={result['sample_left_box_count']}/{args.frames_per_clip}, "
            f"R={result['sample_right_box_count']}/{args.frames_per_clip}, "
            f"both={result['sample_both_box_count']}/{args.frames_per_clip}",
            flush=True,
        )
    summary = {
        "schema": "acedata_sam3_contiguous_bbox_review_v2",
        "sampling_policy": "random_clip_and_random_contiguous_window",
        "seed": args.seed,
        "clip_count": args.clip_count,
        "frames_per_clip": args.frames_per_clip,
        "output_fps_policy": "source" if args.output_fps <= 0.0 else args.output_fps,
        "manifest": str(args.manifest.expanduser().resolve()),
        "bbox_root": str(bbox_root),
        "clips": records,
    }
    (output_dir / "selected_clips.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
