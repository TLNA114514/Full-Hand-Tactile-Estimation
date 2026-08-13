"""Input staging, SAM bbox parsing, crop construction, and tactile rendering."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class FrameSet:
    paths: tuple[Path, ...]
    fps: float
    is_image: bool
    width: int
    height: int
    source: Path
    rotation: int


def _video_rotation(path: Path) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(result.stdout).get("streams", [{}])[0]
        tag = stream.get("tags", {}).get("rotate")
        matrix = next(
            (
                item.get("rotation")
                for item in stream.get("side_data_list", [])
                if item.get("rotation") is not None
            ),
            None,
        )
        rotation = int(round(float(tag))) if tag is not None else -int(round(float(matrix or 0)))
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, TypeError, IndexError):
        rotation = 0
    rotation %= 360
    return rotation if rotation in {0, 90, 180, 270} else 0


def _rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def _write_frame(path: Path, frame: np.ndarray, jpeg_quality: int) -> None:
    options = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if path.suffix == ".jpg" else []
    if not cv2.imwrite(str(path), frame, options):
        raise RuntimeError(f"Failed to write staged frame: {path}")


def stage_input(source: Path, frame_dir: Path, jpeg_quality: int = 95) -> FrameSet:
    source = source.expanduser().resolve(strict=True)
    if not 1 <= int(jpeg_quality) <= 100:
        raise ValueError("frame_jpeg_quality must lie in [1, 100]")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)
    is_single_image = source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES
    rotation = 0
    fps = 30.0
    paths: list[Path] = []
    expected_shape: tuple[int, int] | None = None

    def append_frame(frame: np.ndarray, *, lossless: bool = False) -> None:
        nonlocal expected_shape
        if expected_shape is None:
            expected_shape = frame.shape[:2]
        elif frame.shape[:2] != expected_shape:
            raise ValueError("All input frames must have the same dimensions")
        suffix = ".png" if lossless else ".jpg"
        path = frame_dir / f"frame_{len(paths):08d}{suffix}"
        _write_frame(path, frame, jpeg_quality)
        paths.append(path)

    if is_single_image:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image: {source}")
        append_frame(image, lossless=True)
    elif source.is_dir():
        source_paths = sorted(
            path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not source_paths:
            raise ValueError(f"No images found under {source}")
        for path in source_paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not decode image: {path}")
            append_frame(image)
    else:
        rotation = _video_rotation(source)
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source}")
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                append_frame(_rotate(frame, rotation))
        finally:
            capture.release()
        if not paths:
            raise ValueError(f"Video contains no decodable frames: {source}")
    if expected_shape is None:
        raise ValueError(f"Input contains no decodable frames: {source}")
    height, width = expected_shape
    return FrameSet(
        paths=tuple(paths),
        fps=float(fps),
        is_image=len(paths) == 1,
        width=int(width),
        height=int(height),
        source=source,
        rotation=rotation,
    )


def load_sam_tracks(path: Path, frame_count: int) -> dict[int, np.ndarray]:
    path = path.expanduser().resolve(strict=True)
    tracks: dict[int, np.ndarray] = {}
    seen_frames: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            frame_index = int(row.get("frame_index", -1))
            if not 0 <= frame_index < frame_count:
                raise ValueError(f"Invalid frame_index={frame_index} at {path}:{line_number}")
            if frame_index in seen_frames:
                raise ValueError(f"Duplicate frame_index={frame_index} in {path}")
            seen_frames.add(frame_index)
            for track in row.get("tracks", []):
                track_id = int(track["track_id"])
                bbox = np.asarray(track.get("bbox", []), dtype=np.float32)
                if bbox.shape != (4,) or not np.isfinite(bbox).all():
                    raise ValueError(f"Invalid bbox for track={track_id}, frame={frame_index}")
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ValueError(f"Degenerate bbox for track={track_id}, frame={frame_index}")
                if track_id not in tracks:
                    tracks[track_id] = np.full((frame_count, 4), np.nan, dtype=np.float32)
                if np.isfinite(tracks[track_id][frame_index]).all():
                    raise ValueError(f"Duplicate track={track_id}, frame={frame_index}")
                tracks[track_id][frame_index] = bbox
    if not tracks:
        raise RuntimeError(f"SAM3 produced no accepted tracks: {path}")
    return tracks


def assign_track_sides(
    tracks: dict[int, np.ndarray],
    mode: str,
    single_hand_default: str,
) -> list[dict[str, Any]]:
    if mode not in {"auto", "left", "right", "both"}:
        raise ValueError(f"Unsupported handedness mode: {mode}")
    if single_hand_default not in {"left", "right"}:
        raise ValueError("single_hand_default must be left or right")
    centers = {}
    for track_id, boxes in tracks.items():
        valid = np.isfinite(boxes).all(axis=1)
        centers[track_id] = float(np.median((boxes[valid, 0] + boxes[valid, 2]) * 0.5))
    if mode == "both":
        return [
            {
                "name": f"query_{track_id}_{side}",
                "track_id": track_id,
                "side": side,
                "side_source": "dual_orientation",
                "bboxes": tracks[track_id],
            }
            for track_id in sorted(tracks)
            for side in ("left", "right")
        ]
    if mode in {"left", "right"}:
        side_by_track = {track_id: mode for track_id in tracks}
        source = "explicit"
    elif len(tracks) == 1:
        side_by_track = {next(iter(tracks)): single_hand_default}
        source = "single_hand_default"
    elif len(tracks) == 2:
        ordered = sorted(tracks, key=lambda track_id: centers[track_id])
        side_by_track = {ordered[0]: "right", ordered[1]: "left"}
        source = "egocentric_screen_order"
    else:
        raise ValueError(
            "Auto handedness supports at most two tracks; use --handedness left/right/both"
        )
    return [
        {
            "name": f"query_{track_id}",
            "track_id": track_id,
            "side": side_by_track[track_id],
            "side_source": source,
            "bboxes": tracks[track_id],
        }
        for track_id in sorted(tracks)
    ]


def write_track_preview(
    frames: FrameSet,
    tracks: dict[int, np.ndarray],
    output_path: Path,
) -> Path:
    """Write one representative, numbered panel per anonymous SAM3 track."""

    colors = ((70, 220, 100), (70, 180, 245), (220, 120, 245), (245, 190, 70))
    panels = []
    for panel_index, track_id in enumerate(sorted(tracks)):
        boxes = tracks[track_id]
        valid_indices = np.flatnonzero(np.isfinite(boxes).all(axis=1))
        if not len(valid_indices):
            continue
        frame_index = int(valid_indices[len(valid_indices) // 2])
        image = cv2.imread(str(frames.paths[frame_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode preview frame: {frames.paths[frame_index]}")
        color = colors[panel_index % len(colors)]
        for other_id, other_boxes in sorted(tracks.items()):
            other_bbox = other_boxes[frame_index]
            if not np.isfinite(other_bbox).all():
                continue
            x0, y0, x1, y1 = np.rint(other_bbox).astype(int)
            selected = other_id == track_id
            box_color = color if selected else (130, 130, 130)
            cv2.rectangle(image, (x0, y0), (x1, y1), box_color, 4 if selected else 2)
            cv2.putText(
                image,
                f"track {other_id}",
                (max(4, x0), max(24, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                box_color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f"SELECT TRACK {track_id} | frame {frame_index} | visible {len(valid_indices)}/{len(frames.paths)}",
            (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
        if image.shape[1] > 1280:
            scale = 1280.0 / float(image.shape[1])
            image = cv2.resize(
                image,
                (1280, max(1, int(round(image.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        panels.append(image)
    if not panels:
        raise RuntimeError("No valid SAM3 track is available for handedness preview")
    target_width = max(panel.shape[1] for panel in panels)
    padded = []
    for panel in panels:
        if panel.shape[1] < target_width:
            panel = cv2.copyMakeBorder(
                panel,
                0,
                0,
                0,
                target_width - panel.shape[1],
                cv2.BORDER_CONSTANT,
                value=(24, 28, 32),
            )
        padded.append(panel)
    preview = np.concatenate(padded, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f"Failed to write track preview: {output_path}")
    return output_path


def assign_track_sides_interactively(
    tracks: dict[int, np.ndarray],
    preview_path: Path,
) -> list[dict[str, Any]]:
    """Ask the terminal user which canonical orientation each query should use."""

    import sys

    if not sys.stdin.isatty():
        raise RuntimeError(
            "--handedness interactive requires an attached terminal; use "
            "--handedness left/right/both for a non-interactive run"
        )
    print(f"Handedness preview: {preview_path}", flush=True)
    print("Assign each anonymous SAM3 track: [l]eft, [r]ight, [b]oth, [s]kip", flush=True)
    query_specs = []
    for track_id in sorted(tracks):
        boxes = tracks[track_id]
        detected = int(np.isfinite(boxes).all(axis=1).sum())
        while True:
            choice = input(
                f"track {track_id} ({detected}/{len(boxes)} frames) [l/r/b/s]: "
            ).strip().lower()
            aliases = {
                "l": ("left",),
                "left": ("left",),
                "r": ("right",),
                "right": ("right",),
                "b": ("left", "right"),
                "both": ("left", "right"),
                "s": (),
                "skip": (),
            }
            if choice in aliases:
                break
            print("Please enter l, r, b, or s.", flush=True)
        sides = aliases[choice]
        for side in sides:
            suffix = f"_{side}" if len(sides) == 2 else ""
            query_specs.append(
                {
                    "name": f"query_{track_id}{suffix}",
                    "track_id": track_id,
                    "side": side,
                    "side_source": "interactive",
                    "bboxes": boxes,
                }
            )
    if not query_specs:
        raise RuntimeError("Every SAM3 track was skipped; there is nothing to infer")
    return query_specs


def crop_bbox(
    bbox: np.ndarray,
    scale: float,
    resolution: tuple[int, int],
) -> np.ndarray:
    bbox = np.asarray(bbox, dtype=np.float32)
    center = (bbox[:2] + bbox[2:]) * 0.5
    output_height, output_width = (int(value) for value in resolution)
    crop_height = float(np.max(bbox[2:] - bbox[:2])) * float(scale)
    crop_width = crop_height * float(output_width) / float(output_height)
    return np.asarray(
        [
            center[0] - crop_width * 0.5,
            center[1] - crop_height * 0.5,
            center[0] + crop_width * 0.5,
            center[1] + crop_height * 0.5,
        ],
        dtype=np.float32,
    )


def tactile_crop(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    side: str,
    resolution: tuple[int, int],
    bbox_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    output_height, output_width = resolution
    bbox = np.asarray(bbox, dtype=np.float32)
    center = (bbox[:2] + bbox[2:]) * 0.5
    bbox_size = float(np.max(bbox[2:] - bbox[:2])) * float(bbox_scale)
    if not math.isfinite(bbox_size) or bbox_size <= 1.0:
        raise ValueError(f"Invalid model crop from bbox={bbox.tolist()}")
    transform = np.zeros((2, 3), dtype=np.float32)
    transform[0, 0] = float(output_height) / bbox_size
    transform[1, 1] = float(output_height) / bbox_size
    transform[0, 2] = -output_height * float(center[0]) / bbox_size + output_width * 0.5
    transform[1, 2] = output_height * (-float(center[1]) / bbox_size + 0.5)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    patch = cv2.warpAffine(
        image_rgb,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
    ).astype(np.float32) / 255.0
    if side == "left":
        patch = cv2.flip(patch, 1)
    patch = ((patch - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return (
        np.ascontiguousarray(patch, dtype=np.float32),
        crop_bbox(bbox, bbox_scale, resolution),
    )


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    with path.expanduser().resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(token.split("/", 1)[0]) - 1 for token in line.split()[1:]]
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) > 3:
                    faces.extend([indices[:1] + indices[i : i + 2] for i in range(1, len(indices) - 1)])
    if not vertices or not faces:
        raise ValueError(f"OBJ contains no usable mesh: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def load_palm_support(path: Path, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    group = data["group_negative"]
    palm_vertices = {
        int(vertex)
        for face in group["face_triplets"]
        for vertex in face
        if 0 <= int(vertex) < vertex_count
    }
    mask = np.zeros(vertex_count, dtype=np.float32)
    mask[list(palm_vertices)] = 1.0
    return mask, np.asarray(group["face_indices"], dtype=np.int32)


def _axis_rotation(degrees: float, axis: str) -> np.ndarray:
    angle = math.radians(degrees)
    sine, cosine = math.sin(angle), math.cos(angle)
    if axis == "x":
        return np.asarray([[1, 0, 0, 0], [0, cosine, -sine, 0], [0, sine, cosine, 0], [0, 0, 0, 1]], dtype=np.float32)
    if axis == "y":
        return np.asarray([[cosine, 0, sine, 0], [0, 1, 0, 0], [-sine, 0, cosine, 0], [0, 0, 0, 1]], dtype=np.float32)
    raise ValueError(axis)


class CanonicalRasterizer:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        palm_face_indices: np.ndarray,
        size: tuple[int, int],
        mirror: bool,
    ):
        self.width, self.height = (int(value) for value in size)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.vertex_count = len(vertices)
        rotation = _axis_rotation(-90, "x") @ _axis_rotation(90, "y")
        homogeneous = np.concatenate(
            [np.asarray(vertices, dtype=np.float32), np.ones((len(vertices), 1), np.float32)],
            axis=1,
        )
        base = (rotation @ homogeneous.T).T[:, :3]
        candidates = [base, base * np.asarray([-1.0, 1.0, -1.0], np.float32)]
        rendered = []
        valid_palm_faces = np.asarray(palm_face_indices, dtype=np.int32)
        valid_palm_faces = valid_palm_faces[
            (valid_palm_faces >= 0) & (valid_palm_faces < len(self.faces))
        ]
        for camera_vertices in candidates:
            camera_vertices = camera_vertices.copy()
            if mirror:
                camera_vertices[:, 0] *= -1.0
            camera_vertices[:, 2] += 2.0
            screen = self._project(camera_vertices)
            face_ids = self._rasterize_face_ids(screen, camera_vertices)
            visible = face_ids[face_ids >= 0]
            score = int(np.count_nonzero(np.isin(visible, valid_palm_faces)))
            rendered.append((score, screen, face_ids))
        _score, screen, face_ids = max(rendered, key=lambda item: item[0])
        self._build_lookup(screen, face_ids)

    def _project(self, vertices: np.ndarray) -> np.ndarray:
        focal = 8000.0 * (self.width / 1280.0)
        screen = np.empty((len(vertices), 2), dtype=np.float32)
        screen[:, 0] = focal * vertices[:, 0] / vertices[:, 2] + self.width / 2.0
        screen[:, 1] = -focal * vertices[:, 1] / vertices[:, 2] + self.height / 2.0
        return screen

    def _rasterize_face_ids(self, screen: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        depth = vertices[self.faces, 2].mean(axis=1)
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for face_index in np.argsort(depth)[::-1]:
            triangle = screen[self.faces[face_index]]
            if not np.isfinite(triangle).all():
                continue
            encoded = int(face_index) + 1
            cv2.fillConvexPoly(
                image,
                np.rint(triangle).astype(np.int32),
                (encoded & 255, (encoded >> 8) & 255, (encoded >> 16) & 255),
            )
        return (
            image[:, :, 0].astype(np.int32)
            + (image[:, :, 1].astype(np.int32) << 8)
            + (image[:, :, 2].astype(np.int32) << 16)
            - 1
        )

    def _build_lookup(self, screen: np.ndarray, face_ids: np.ndarray) -> None:
        rows, cols = np.nonzero(face_ids >= 0)
        pixel_faces = self.faces[face_ids[rows, cols]]
        triangles = screen[pixel_faces]
        points = np.stack([cols, rows], axis=1).astype(np.float32)
        denominator = (
            (triangles[:, 1, 1] - triangles[:, 2, 1]) * (triangles[:, 0, 0] - triangles[:, 2, 0])
            + (triangles[:, 2, 0] - triangles[:, 1, 0]) * (triangles[:, 0, 1] - triangles[:, 2, 1])
        )
        valid = np.abs(denominator) > 1e-8
        rows, cols = rows[valid], cols[valid]
        pixel_faces, triangles, points, denominator = (
            pixel_faces[valid], triangles[valid], points[valid], denominator[valid]
        )
        w0 = (
            (triangles[:, 1, 1] - triangles[:, 2, 1]) * (points[:, 0] - triangles[:, 2, 0])
            + (triangles[:, 2, 0] - triangles[:, 1, 0]) * (points[:, 1] - triangles[:, 2, 1])
        ) / denominator
        w1 = (
            (triangles[:, 2, 1] - triangles[:, 0, 1]) * (points[:, 0] - triangles[:, 2, 0])
            + (triangles[:, 0, 0] - triangles[:, 2, 0]) * (points[:, 1] - triangles[:, 2, 1])
        ) / denominator
        self.rows, self.cols = rows, cols
        self.pixel_vertices = pixel_faces
        self.weights = np.stack([w0, w1, 1.0 - w0 - w1], axis=1).astype(np.float32)

    def render(self, vertex_rgb: np.ndarray) -> np.ndarray:
        sampled = np.asarray(vertex_rgb, dtype=np.float32)[self.pixel_vertices]
        colors = np.einsum("nij,ni->nj", sampled, self.weights)
        image = np.full((self.height, self.width, 3), (24, 28, 32), dtype=np.uint8)
        image[self.rows, self.cols] = np.clip(colors, 0, 255).astype(np.uint8)
        return image


def pressure_colors(pressure: np.ndarray, display_floor: float) -> np.ndarray:
    from matplotlib import cm

    pressure = np.asarray(pressure, dtype=np.float32).clip(0.0, 1.0)
    pressure = np.where(pressure >= float(display_floor), pressure, 0.0)
    colormap = cm.get_cmap("gnuplot2")
    return np.round(colormap(1.0 - pressure)[..., :3] * 255.0).astype(np.uint8)


def _draw_boxes(
    frame: np.ndarray,
    tight: np.ndarray,
    crop: np.ndarray,
    label: str,
) -> np.ndarray:
    image = frame.copy()
    if np.isfinite(crop).all():
        x0, y0, x1, y1 = np.rint(crop).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), (245, 184, 55), 2)
    if np.isfinite(tight).all():
        x0, y0, x1, y1 = np.rint(tight).astype(int)
        cv2.rectangle(image, (x0, y0), (x1, y1), (60, 220, 100), 3)
        cv2.putText(
            image,
            label,
            (max(4, x0), max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (60, 220, 100),
            2,
            cv2.LINE_AA,
        )
    return image


def _fit_height(image: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def render_query_output(
    query_dir: Path,
    frames: FrameSet,
    pressure: np.ndarray,
    bboxes: np.ndarray,
    crop_boxes: np.ndarray,
    detected: np.ndarray,
    side: str,
    label: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    palm_faces: np.ndarray,
    render_size: tuple[int, int],
    display_floor: float,
    temporal_alpha: float,
) -> None:
    query_dir.mkdir(parents=True, exist_ok=True)
    tactile_dir = query_dir / "tactile_frames"
    combined_dir = query_dir / "combined_frames"
    tactile_dir.mkdir(exist_ok=True)
    combined_dir.mkdir(exist_ok=True)
    renderer = CanonicalRasterizer(
        vertices,
        faces,
        palm_faces,
        render_size,
        mirror=side == "left",
    )
    previous = None
    combined_paths = []
    for index, frame_path in enumerate(frames.paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not reread staged frame: {frame_path}")
        current = pressure[index]
        if not detected[index]:
            shown = np.zeros_like(current)
            previous = None
        elif previous is None or temporal_alpha <= 0:
            shown = current
            previous = current
        else:
            shown = temporal_alpha * current + (1.0 - temporal_alpha) * previous
            previous = shown
        tactile_rgb = renderer.render(pressure_colors(shown, display_floor))
        tactile_bgr = cv2.cvtColor(tactile_rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(
            tactile_bgr,
            f"{label} | {side.upper()}",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        if not detected[index]:
            cv2.putText(
                tactile_bgr,
                "NO SAM3 QUERY",
                (18, render_size[1] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (60, 80, 240),
                2,
                cv2.LINE_AA,
            )
        tactile_path = tactile_dir / f"frame_{index:08d}.png"
        cv2.imwrite(str(tactile_path), tactile_bgr)
        boxed = _draw_boxes(frame, bboxes[index], crop_boxes[index], label)
        fitted_tactile = _fit_height(tactile_bgr, boxed.shape[0])
        combined = np.concatenate([boxed, fitted_tactile], axis=1)
        combined_path = combined_dir / f"frame_{index:08d}.jpg"
        _write_frame(combined_path, combined, 95)
        combined_paths.append(combined_path)
    if frames.is_image:
        shutil.copy2(tactile_dir / "frame_00000000.png", query_dir / "tactile.png")
        shutil.copy2(combined_dir / "frame_00000000.jpg", query_dir / "combined.jpg")
        return
    first = cv2.imread(str(combined_paths[0]), cv2.IMREAD_COLOR)
    writer = cv2.VideoWriter(
        str(query_dir / "combined.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frames.fps,
        (first.shape[1], first.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {query_dir / 'combined.mp4'}")
    try:
        for path in combined_paths:
            writer.write(cv2.imread(str(path), cv2.IMREAD_COLOR))
    finally:
        writer.release()
