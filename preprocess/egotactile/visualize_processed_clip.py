#!/usr/bin/env python3
"""Render one processed EgoTactile clip as RGB+bbox | canonical MANO pressure.

The script is deliberately read-only with respect to the processed dataset. It
uses ``frame_idx`` as the sole alignment key between the original source video,
the current extracted-frame bbox registry, and the processed pressure archive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import trimesh


if os.environ.get("FULL_HAND_TACTILE_ROOT"):
    REPO_ROOT = Path(os.environ["FULL_HAND_TACTILE_ROOT"]).expanduser().resolve()
else:
    REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_ROOT = Path(
    "/home/ma-user/work/cfzhao/EgoTactile/processed-data"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ma-user/work/cfzhao/EgoTactile/processed-data-audits/clip_visualization"
)
PRESSURE_KEY = "right_pressure_continuous_subdiv"


def _load_subdiv_palm_support(tactile_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    support_path = (
        REPO_ROOT
        / "opentouch/preprocess/scratch/auto_calibrated_palm_subdiv_faces.json"
    )
    palm_data = _load_json(support_path)
    if not isinstance(palm_data, dict):
        raise ValueError(f"Invalid palm support file: {support_path}")
    negative = palm_data["group_negative"]
    palm_vertices = {
        int(vertex_id)
        for triplet in negative["face_triplets"]
        for vertex_id in triplet
        if 0 <= int(vertex_id) < tactile_dim
    }
    palm_mask = np.zeros(tactile_dim, dtype=np.float32)
    palm_mask[list(palm_vertices)] = 1.0
    return palm_mask, np.asarray(negative["face_indices"], dtype=np.int32)


def _pressure_vertex_colors(pressure: np.ndarray, display_floor: float) -> np.ndarray:
    from matplotlib import cm

    clipped = np.clip(np.asarray(pressure, dtype=np.float32), 0.0, 1.0)
    clipped = np.where(clipped < display_floor, 0.0, clipped)
    rgb = np.asarray(cm.gnuplot2(1.0 - clipped)[..., :3], dtype=np.float32)
    return np.concatenate(
        [
            np.round(rgb * 255.0).astype(np.uint8),
            np.full((len(clipped), 1), 255, dtype=np.uint8),
        ],
        axis=1,
    )


def _draw_pressure_colorbar(
    image_rgb: np.ndarray, display_floor: float
) -> np.ndarray:
    from matplotlib import cm

    image = np.asarray(image_rgb, dtype=np.uint8).copy()
    height, width = image.shape[:2]
    bar_width = max(18, width // 36)
    margin_right = max(20, width // 32)
    x1 = width - margin_right
    x0 = x1 - bar_width
    y0 = max(50, height // 12)
    y1 = height - y0
    physical = np.linspace(1.0, 0.0, y1 - y0 + 1, dtype=np.float32)
    displayed = np.where(physical < display_floor, 0.0, physical)
    bar_rgb = np.asarray(cm.gnuplot2(1.0 - displayed)[..., :3], dtype=np.float32)
    image[y0 : y1 + 1, x0:x1] = np.round(bar_rgb[:, None, :] * 255.0).astype(
        np.uint8
    )
    cv2.rectangle(image, (x0 - 1, y0 - 1), (x1, y1 + 1), (230, 230, 230), 1)
    labels = [(1.0, y0 + 6), (0.5, (y0 + y1) // 2 + 6), (0.0, y1)]
    if 0.0 < display_floor < 0.5:
        labels.insert(
            2,
            (
                display_floor,
                int(y0 + (1.0 - display_floor) * (y1 - y0)),
            ),
        )
    for value, y in labels:
        text = f"{value:.2f}" if value < 0.1 else f"{value:.1f}"
        cv2.putText(
            image,
            text,
            (max(4, x0 - 68), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return image


def _draw_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.65, min(1.15, image.shape[0] / 900.0))
    thickness = max(2, int(round(font_scale * 2.0)))
    (text_width, text_height), baseline = cv2.getTextSize(
        title, font, font_scale, thickness
    )
    x0, y0 = 18, 18
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + text_width + 24, y0 + text_height + baseline + 20),
        (255, 255, 255),
        -1,
    )
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + text_width + 24, y0 + text_height + baseline + 20),
        (32, 32, 32),
        2,
    )
    cv2.putText(
        image,
        title,
        (x0 + 12, y0 + text_height + 8),
        font,
        font_scale,
        (24, 24, 24),
        thickness,
        cv2.LINE_AA,
    )
    return image


class CanonicalSubdivRasterizer:
    """Precompute a static canonical MANO rasterization lookup."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        image_size: Tuple[int, int],
        palm_face_indices: np.ndarray,
    ) -> None:
        self.width, self.height = [int(value) for value in image_size]
        self.faces = np.asarray(faces, dtype=np.int32)
        self.vertex_count = int(len(vertices))
        self.background_rgb = np.asarray((24, 28, 32), dtype=np.uint8)

        rotation = (
            trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
            @ trimesh.transformations.rotation_matrix(np.radians(90), [0, 1, 0])
        )
        vertices_h = np.concatenate(
            [
                np.asarray(vertices, dtype=np.float32),
                np.ones((len(vertices), 1), dtype=np.float32),
            ],
            axis=1,
        )
        base = (rotation @ vertices_h.T).T[:, :3]
        opposite = base.copy()
        opposite[:, (0, 2)] *= -1.0
        palm_face_indices = np.asarray(palm_face_indices, dtype=np.int32)
        palm_face_indices = palm_face_indices[
            (palm_face_indices >= 0) & (palm_face_indices < len(self.faces))
        ]

        candidates = []
        for name, camera_vertices in (("legacy", base), ("opposite", opposite)):
            camera_vertices = camera_vertices.copy()
            camera_vertices[:, 2] += 2.0
            screen_vertices = self._project(camera_vertices)
            face_ids = self._rasterize_face_ids(screen_vertices, camera_vertices)
            visible = face_ids[face_ids >= 0]
            palm_pixels = int(np.count_nonzero(np.isin(visible, palm_face_indices)))
            candidates.append((palm_pixels, name, screen_vertices, face_ids))
        palm_pixels, view_name, screen_vertices, face_ids = max(
            candidates, key=lambda item: item[0]
        )
        print(f"Canonical tactile view: {view_name} (visible palm pixels={palm_pixels})")
        self._build_barycentric_lookup(screen_vertices, face_ids)

    def _project(self, camera_vertices: np.ndarray) -> np.ndarray:
        focal_length = 8000.0 * (self.width / 1280.0)
        screen = np.empty((len(camera_vertices), 2), dtype=np.float32)
        screen[:, 0] = (
            focal_length * camera_vertices[:, 0] / camera_vertices[:, 2]
            + self.width / 2.0
        )
        screen[:, 1] = (
            -focal_length * camera_vertices[:, 1] / camera_vertices[:, 2]
            + self.height / 2.0
        )
        return screen

    def _rasterize_face_ids(
        self, screen_vertices: np.ndarray, camera_vertices: np.ndarray
    ) -> np.ndarray:
        face_depth = camera_vertices[self.faces, 2].mean(axis=1)
        id_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for face_index in np.argsort(face_depth)[::-1]:
            triangle = screen_vertices[self.faces[face_index]]
            if not np.isfinite(triangle).all():
                continue
            encoded = int(face_index) + 1
            cv2.fillConvexPoly(
                id_image,
                np.rint(triangle).astype(np.int32),
                (encoded & 255, (encoded >> 8) & 255, (encoded >> 16) & 255),
                lineType=cv2.LINE_8,
            )
        return (
            id_image[:, :, 0].astype(np.int32)
            + (id_image[:, :, 1].astype(np.int32) << 8)
            + (id_image[:, :, 2].astype(np.int32) << 16)
            - 1
        )

    def _build_barycentric_lookup(
        self, screen_vertices: np.ndarray, face_ids: np.ndarray
    ) -> None:
        rows, cols = np.nonzero(face_ids >= 0)
        if not len(rows):
            raise RuntimeError("Canonical mesh rasterization produced no visible pixels")
        pixel_faces = self.faces[face_ids[rows, cols]]
        triangles = screen_vertices[pixel_faces]
        points = np.stack([cols, rows], axis=1).astype(np.float32)
        denominator = (
            (triangles[:, 1, 1] - triangles[:, 2, 1])
            * (triangles[:, 0, 0] - triangles[:, 2, 0])
            + (triangles[:, 2, 0] - triangles[:, 1, 0])
            * (triangles[:, 0, 1] - triangles[:, 2, 1])
        )
        valid = np.abs(denominator) > 1e-8
        triangles = triangles[valid]
        self.pixel_vertices = pixel_faces[valid]
        points = points[valid]
        self.rows = rows[valid]
        self.cols = cols[valid]
        denominator = denominator[valid]
        w0 = (
            (triangles[:, 1, 1] - triangles[:, 2, 1])
            * (points[:, 0] - triangles[:, 2, 0])
            + (triangles[:, 2, 0] - triangles[:, 1, 0])
            * (points[:, 1] - triangles[:, 2, 1])
        ) / denominator
        w1 = (
            (triangles[:, 2, 1] - triangles[:, 0, 1])
            * (points[:, 0] - triangles[:, 2, 0])
            + (triangles[:, 0, 0] - triangles[:, 2, 0])
            * (points[:, 1] - triangles[:, 2, 1])
        ) / denominator
        self.barycentric_weights = np.stack(
            [w0, w1, 1.0 - w0 - w1], axis=1
        ).astype(np.float32)

    def render(self, vertex_colors: np.ndarray) -> np.ndarray:
        colors = np.asarray(vertex_colors, dtype=np.uint8)
        if colors.shape != (self.vertex_count, 4):
            raise ValueError(
                f"Expected vertex colors {(self.vertex_count, 4)}, got {colors.shape}"
            )
        sampled = colors[self.pixel_vertices, :3].astype(np.float32)
        pixel_colors = np.einsum(
            "nij,ni->nj", sampled, self.barycentric_weights
        )
        image = np.broadcast_to(
            self.background_rgb, (self.height, self.width, 3)
        ).copy()
        image[self.rows, self.cols] = np.clip(pixel_colors, 0, 255).astype(np.uint8)
        return image


def _load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scalar(array: np.ndarray):
    value = np.asarray(array)
    if value.shape != ():
        raise ValueError(f"Expected scalar array, got {value.shape}")
    return value.item()


def _safe_component(value: str) -> str:
    return "__".join(
        part.replace(" ", "_").replace(os.sep, "_")
        for part in str(value).split("/")
        if part
    )


def _consecutive_runs(indices: Iterable[int]) -> List[Tuple[int, int]]:
    values = sorted(set(int(value) for value in indices))
    if not values:
        return []
    runs: List[Tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _best_window(
    pressure: np.ndarray,
    run: Tuple[int, int],
    max_frames: int,
) -> Tuple[int, int, float, Dict[str, float]]:
    run_start, run_end = run
    run_length = run_end - run_start + 1
    frame_count = min(max_frames, run_length)
    values = np.asarray(pressure[run_start : run_end + 1], dtype=np.float32)
    frame_max = values.max(axis=1)
    frame_volume = values.sum(axis=1)
    active = (values >= 0.05).sum(axis=1).astype(np.float32)

    def summarize(offset: int) -> Tuple[float, Dict[str, float]]:
        selection = slice(offset, offset + frame_count)
        selected_max = frame_max[selection]
        selected_volume = frame_volume[selection]
        selected_active = active[selection]
        max_span = float(np.quantile(selected_max, 0.95) - np.quantile(selected_max, 0.05))
        volume_p95 = float(np.quantile(selected_volume, 0.95))
        volume_span = float(
            (volume_p95 - np.quantile(selected_volume, 0.05))
            / max(volume_p95, 1e-6)
        )
        active_p95 = float(np.quantile(selected_active, 0.95))
        active_span = float(
            (active_p95 - np.quantile(selected_active, 0.05))
            / max(active_p95, 1.0)
        )
        score = max_span + 0.20 * volume_span + 0.10 * active_span
        return score, {
            "pressure_max_min": float(selected_max.min()),
            "pressure_max_median": float(np.median(selected_max)),
            "pressure_max_max": float(selected_max.max()),
            "pressure_volume_min": float(selected_volume.min()),
            "pressure_volume_median": float(np.median(selected_volume)),
            "pressure_volume_max": float(selected_volume.max()),
            "active_vertices_0.05_median": float(np.median(selected_active)),
            "dynamic_score": float(score),
        }

    if run_length <= frame_count:
        score, summary = summarize(0)
        return run_start, run_end, score, summary

    last_offset = run_length - frame_count
    offsets = sorted(
        set(
            np.linspace(0, last_offset, min(last_offset + 1, 41))
            .round()
            .astype(np.int64)
            .tolist()
        )
    )
    best_offset = 0
    best_score = -math.inf
    best_summary: Dict[str, float] = {}
    for offset in offsets:
        score, summary = summarize(int(offset))
        if score > best_score:
            best_offset = int(offset)
            best_score = score
            best_summary = summary
    return (
        run_start + best_offset,
        run_start + best_offset + frame_count - 1,
        best_score,
        best_summary,
    )


def _group_registry(
    manifest: Mapping[str, object],
) -> Dict[str, List[Mapping[str, object]]]:
    registry = manifest.get("registry")
    if not isinstance(registry, list):
        raise ValueError("Processed extracted-frame manifest has no list-valued registry")
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in registry:
        if not isinstance(row, dict):
            continue
        if row.get("hand") != "right" or not bool(row.get("has_bbox", False)):
            continue
        rel_seq = str(row.get("rel_seq", ""))
        if rel_seq:
            grouped[rel_seq].append(row)
    return dict(grouped)


def _select_sequence(
    processed_root: Path,
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    max_frames: int,
    top_k: int,
) -> Tuple[str, int, int, Dict[str, float]]:
    coverage_candidates = []
    for rel_seq, rows in grouped.items():
        runs = _consecutive_runs(int(row["frame_idx"]) for row in rows)
        if not runs:
            continue
        longest = max(runs, key=lambda value: value[1] - value[0])
        coverage_candidates.append(
            (longest[1] - longest[0] + 1, len(rows), rel_seq, longest)
        )
    if not coverage_candidates:
        raise RuntimeError("No right-hand bbox timelines were found in the current manifest")
    coverage_candidates.sort(reverse=True)

    scored = []
    for run_length, bbox_count, rel_seq, run in coverage_candidates[:top_k]:
        archive_path = processed_root / rel_seq / "pressure_grids.npz"
        if not archive_path.is_file():
            continue
        with np.load(archive_path, allow_pickle=False) as archive:
            if PRESSURE_KEY not in archive:
                continue
            pressure = np.asarray(archive[PRESSURE_KEY], dtype=np.float32)
        if run[1] >= len(pressure):
            continue
        start, end, dynamic_score, summary = _best_window(
            pressure, run, max_frames
        )
        summary.update(
            {
                "bbox_registry_count": int(bbox_count),
                "longest_consecutive_bbox_run": int(run_length),
            }
        )
        scored.append((dynamic_score, run_length, rel_seq, start, end, summary))
    if not scored:
        raise RuntimeError("No bbox-covered sequence had a compatible pressure archive")
    _score, _run_length, rel_seq, start, end, summary = max(scored)
    return rel_seq, start, end, summary


def _resolve_explicit_range(
    rows: Sequence[Mapping[str, object]],
    frame_start: int | None,
    frame_end: int | None,
    max_frames: int,
) -> Tuple[int, int]:
    runs = _consecutive_runs(int(row["frame_idx"]) for row in rows)
    if not runs:
        raise RuntimeError("The requested sequence has no current right-hand bbox rows")
    if frame_start is None:
        start, end = max(runs, key=lambda value: value[1] - value[0])
    else:
        containing = [run for run in runs if run[0] <= frame_start <= run[1]]
        if not containing:
            raise ValueError(
                f"frame_start={frame_start} is not inside a consecutive bbox run: {runs[:8]}"
            )
        start, end = containing[0]
        start = int(frame_start)
    if frame_end is not None:
        end = min(end, int(frame_end))
    end = min(end, start + max_frames - 1)
    if end < start:
        raise ValueError(f"Invalid selected frame range: {start}..{end}")
    return start, end


def _load_bbox_rows(
    rows: Sequence[Mapping[str, object]], start: int, end: int
) -> Dict[int, Dict[str, object]]:
    selected: Dict[int, Dict[str, object]] = {}
    for row in rows:
        frame_index = int(row["frame_idx"])
        if frame_index < start or frame_index > end:
            continue
        sample_dir = Path(str(row["sample_dir"]))
        meta_path = sample_dir / "meta.json"
        meta = _load_json(meta_path)
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid bbox metadata object: {meta_path}")
        bbox = np.asarray(meta.get("bbox"), dtype=np.float32)
        if (
            bbox.shape != (4,)
            or not np.isfinite(bbox).all()
            or not bool(np.all(bbox[2:] > bbox[:2]))
        ):
            raise ValueError(f"Invalid bbox in {meta_path}: {bbox.tolist()}")
        selected[frame_index] = {
            "bbox": bbox,
            "bbox_score": float(meta.get("bbox_score", float("nan"))),
            "meta_path": str(meta_path),
        }
    expected = set(range(start, end + 1))
    missing = sorted(expected - set(selected))
    if missing:
        raise RuntimeError(
            f"Selected interval is not fully covered by current bboxes; first missing={missing[:8]}"
        )
    return selected


def _load_sam3_bbox_rows(
    path: Path, start: int, end: int
) -> Dict[int, Dict[str, object]]:
    selected: Dict[int, Dict[str, object]] = {}
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame_index = int(row.get("frame_index", -1))
            if frame_index < start or frame_index > end:
                continue
            tracks = row.get("tracks", [])
            if not isinstance(tracks, list) or not tracks:
                continue
            tracks = sorted(
                tracks,
                key=lambda track: float(
                    track.get("prompt_score")
                    if track.get("prompt_score") is not None
                    else -math.inf
                ),
                reverse=True,
            )
            track = tracks[0]
            bbox = np.asarray(track.get("bbox"), dtype=np.float32)
            if (
                bbox.shape != (4,)
                or not np.isfinite(bbox).all()
                or not bool(np.all(bbox[2:] > bbox[:2]))
            ):
                raise ValueError(
                    f"Invalid SAM3 bbox at {path}:{line_number}: {bbox.tolist()}"
                )
            selected[frame_index] = {
                "bbox": bbox,
                "bbox_score": float(
                    track.get("prompt_score")
                    if track.get("prompt_score") is not None
                    else float("nan")
                ),
                "track_id": int(track.get("track_id", -1)),
                "prompt": str(track.get("prompt", "")),
                "bbox_source": str(track.get("bbox_source", "sam3")),
                "jsonl_path": str(path),
            }
    expected = set(range(start, end + 1))
    missing = sorted(expected - set(selected))
    if missing:
        raise RuntimeError(
            f"SAM3 bbox JSONL does not fully cover the selected interval; "
            f"first missing={missing[:8]}"
        )
    return selected


def _sam3_bbox_frame_indices(path: Path) -> List[int]:
    indices: List[int] = []
    with path.expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tracks = row.get("tracks", [])
            if not isinstance(tracks, list) or not tracks:
                continue
            frame_index = int(row.get("frame_index", -1))
            if frame_index < 0:
                raise ValueError(
                    f"Invalid SAM3 frame index at {path}:{line_number}: {frame_index}"
                )
            indices.append(frame_index)
    if not indices:
        raise RuntimeError(f"SAM3 bbox JSONL has no tracked frames: {path}")
    return indices


def _load_sequence_meta_bbox_rows(
    meta_root: Path,
    rel_seq: str,
    start: int,
    end: int,
    *,
    allow_missing: bool,
) -> Tuple[Dict[int, Dict[str, object]], List[int]]:
    sequence_parts = Path(rel_seq).parts
    if not sequence_parts:
        raise ValueError("A non-empty sequence is required for direct bbox metadata")
    split_root = meta_root.expanduser().resolve() / sequence_parts[0]
    sample_prefix = _safe_component(rel_seq)
    selected: Dict[int, Dict[str, object]] = {}
    missing: List[int] = []
    for frame_index in range(start, end + 1):
        meta_path = (
            split_root
            / f"{sample_prefix}__{frame_index:06d}__right"
            / "meta.json"
        )
        if not meta_path.is_file():
            missing.append(frame_index)
            continue
        meta = _load_json(meta_path)
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid bbox metadata object: {meta_path}")
        stored_frame_index = int(meta.get("frame_idx", -1))
        if stored_frame_index != frame_index:
            raise ValueError(
                f"Frame index mismatch in {meta_path}: "
                f"expected {frame_index}, got {stored_frame_index}"
            )
        bbox = np.asarray(meta.get("bbox"), dtype=np.float32)
        if (
            bbox.shape != (4,)
            or not np.isfinite(bbox).all()
            or not bool(np.all(bbox[2:] > bbox[:2]))
        ):
            raise ValueError(f"Invalid bbox in {meta_path}: {bbox.tolist()}")
        selected[frame_index] = {
            "bbox": bbox,
            "bbox_score": float(meta.get("bbox_score", float("nan"))),
            "bbox_source": str(meta.get("bbox_source", "processed frame meta.json")),
            "meta_path": str(meta_path),
        }
    if missing and not allow_missing:
        raise RuntimeError(
            "Direct processed bbox metadata does not fully cover the selected "
            f"interval; first missing={missing[:8]}"
        )
    return selected, missing


def _mesh_path() -> Path:
    candidates = [
        REPO_ROOT / "opentouch/preprocess/scratch/mano_right_neutral_subdiv.obj",
        REPO_ROOT / "TouchAnything/src/resources/mano_right_neutral_subdiv.obj",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Could not find mano_right_neutral_subdiv.obj")


def _draw_bbox(
    image: np.ndarray, bbox: np.ndarray, score: float, label: str
) -> None:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = np.rint(bbox).astype(np.int64).tolist()
    x0, x1 = np.clip([x0, x1], 0, width - 1).tolist()
    y0, y1 = np.clip([y0, y1], 0, height - 1).tolist()
    color = (65, 225, 90)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 3, cv2.LINE_AA)
    if np.isfinite(score):
        label += f"  score={score:.3f}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2
    )
    label_top = max(0, y0 - text_height - baseline - 10)
    cv2.rectangle(
        image,
        (x0, label_top),
        (min(width - 1, x0 + text_width + 12), y0),
        (18, 24, 20),
        -1,
    )
    cv2.putText(
        image,
        label,
        (x0 + 6, max(text_height + 2, y0 - baseline - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_missing_bbox(image: np.ndarray) -> None:
    label = "ORIGINAL BBOX MISSING"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(label, font, 0.72, 2)
    x0 = max(16, image.shape[1] - text_width - 44)
    y0 = 18
    cv2.rectangle(
        image,
        (x0, y0),
        (x0 + text_width + 24, y0 + text_height + baseline + 20),
        (28, 28, 180),
        -1,
    )
    cv2.putText(
        image,
        label,
        (x0 + 12, y0 + text_height + 8),
        font,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_status(
    image: np.ndarray,
    frame_index: int,
    pressure: np.ndarray,
) -> None:
    overlay = image.copy()
    bar_height = 48
    cv2.rectangle(
        overlay,
        (0, image.shape[0] - bar_height),
        (image.shape[1], image.shape[0]),
        (12, 15, 18),
        -1,
    )
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
    text = (
        f"frame {frame_index}   max={float(pressure.max()):.3f}   "
        f"volume={float(pressure.sum()):.1f}   "
        f"active@.05={int(np.count_nonzero(pressure >= 0.05))}"
    )
    cv2.putText(
        image,
        text,
        (18, image.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )


def _letterbox(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_width, target_height = target_size
    scale = min(target_width / image.shape[1], target_height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((target_height, target_width, 3), 24, dtype=np.uint8)
    x0 = (target_width - resized.shape[1]) // 2
    y0 = (target_height - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def _encode_h264(raw_path: Path, output_path: Path, fps: float) -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raw_path.replace(output_path)
        return "opencv-mp4v"
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(raw_path),
        "-an",
        "-r",
        f"{fps:.8g}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    raw_path.unlink()
    return "h264"


def _write_contact_sheet(frames: Sequence[np.ndarray], path: Path) -> None:
    if not frames:
        return
    count = min(8, len(frames))
    indices = np.linspace(0, len(frames) - 1, count).round().astype(np.int64)
    thumbs = [
        cv2.resize(frames[int(index)], (750, 270), interpolation=cv2.INTER_AREA)
        for index in indices
    ]
    if len(thumbs) % 2:
        thumbs.append(np.full_like(thumbs[0], 24))
    rows = [np.concatenate(thumbs[index : index + 2], axis=1) for index in range(0, len(thumbs), 2)]
    sheet = np.concatenate(rows, axis=0)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {path}")


def render(args: argparse.Namespace) -> Path:
    processed_root = args.processed_root.expanduser().resolve()
    manifest_path = processed_root / "extracted_frames/manifest.json"
    grouped: Dict[str, List[Mapping[str, object]]] = {}
    needs_manifest = not args.sequence or (
        args.bbox_jsonl is None and args.bbox_meta_root is None
    )
    if needs_manifest:
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError(f"Expected object manifest: {manifest_path}")
        grouped = _group_registry(manifest)

    selection_summary: Dict[str, float] = {}
    if args.sequence:
        rel_seq = args.sequence.strip("/")
        if args.bbox_meta_root is not None:
            frame_start = 0 if args.frame_start is None else int(args.frame_start)
            frame_end = (
                frame_start + args.max_frames - 1
                if args.frame_end is None
                else min(int(args.frame_end), frame_start + args.max_frames - 1)
            )
            if frame_start < 0 or frame_end < frame_start:
                raise ValueError(
                    f"Invalid selected frame range: {frame_start}..{frame_end}"
                )
        elif rel_seq in grouped:
            range_rows = grouped[rel_seq]
            frame_start, frame_end = _resolve_explicit_range(
                range_rows, args.frame_start, args.frame_end, args.max_frames
            )
        elif args.bbox_jsonl is not None:
            range_rows = [
                {"frame_idx": frame_index}
                for frame_index in _sam3_bbox_frame_indices(args.bbox_jsonl)
            ]
            frame_start, frame_end = _resolve_explicit_range(
                range_rows, args.frame_start, args.frame_end, args.max_frames
            )
        else:
            raise KeyError(
                f"Requested sequence has no current right-hand bbox registry: {rel_seq}"
            )
    else:
        rel_seq, frame_start, frame_end, selection_summary = _select_sequence(
            processed_root,
            grouped,
            max_frames=args.max_frames,
            top_k=args.selection_top_k,
        )

    archive_path = processed_root / rel_seq / "pressure_grids.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        pressure_all = np.asarray(archive[PRESSURE_KEY], dtype=np.float32)
        source_video = Path(str(_scalar(archive["source_video"]))).expanduser().resolve()
        mapping_schema = str(_scalar(archive["mapping_schema"]))
        normalization_schema = str(_scalar(archive["normalization_schema"]))
        normalization_formula = str(_scalar(archive["normalization_formula"]))
        mapping_sha256 = str(_scalar(archive["right_mapping_sha256"]))
        bend_mixed = bool(_scalar(archive["bend_mixed_into_pressure"]))
    if pressure_all.ndim != 2 or pressure_all.shape[1] != 13614:
        raise ValueError(
            f"Expected {PRESSURE_KEY} shape [T,13614], got {pressure_all.shape}"
        )
    if not np.isfinite(pressure_all).all():
        raise ValueError(f"Non-finite values found in {archive_path}:{PRESSURE_KEY}")
    if frame_end >= len(pressure_all):
        raise IndexError(
            f"Selected frame {frame_end} exceeds pressure length {len(pressure_all)}"
        )
    missing_bbox_frames: List[int] = []
    if args.bbox_meta_root is not None:
        bbox_rows, missing_bbox_frames = _load_sequence_meta_bbox_rows(
            args.bbox_meta_root,
            rel_seq,
            frame_start,
            frame_end,
            allow_missing=args.allow_missing_bboxes,
        )
        bbox_label = str(args.bbox_label or "original processed bbox")
        bbox_source = str(args.bbox_meta_root.expanduser().resolve())
        bbox_suffix = _safe_component(bbox_label.lower())
    elif args.bbox_jsonl is None:
        bbox_rows = _load_bbox_rows(grouped[rel_seq], frame_start, frame_end)
        bbox_label = "current bbox"
        bbox_source = "current extracted-frame registry meta.json"
        bbox_suffix = "current_bbox"
    else:
        bbox_rows = _load_sam3_bbox_rows(
            args.bbox_jsonl, frame_start, frame_end
        )
        bbox_label = str(args.bbox_label or "SAM3 bbox")
        bbox_source = str(args.bbox_jsonl.expanduser().resolve())
        bbox_suffix = _safe_component(bbox_label.lower())
    pressure = pressure_all[frame_start : frame_end + 1]

    mesh_path = _mesh_path()
    mesh = trimesh.load(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) != pressure.shape[1]:
        raise ValueError(
            f"Mesh/pressure vertex mismatch: {len(vertices)} vs {pressure.shape[1]}"
        )
    _palm_mask, palm_faces = _load_subdiv_palm_support(len(vertices))
    renderer = CanonicalSubdivRasterizer(
        vertices,
        faces,
        image_size=(args.panel_height, args.panel_height),
        palm_face_indices=palm_faces,
    )

    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open source video: {source_video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if source_frame_count > 0 and frame_end >= source_frame_count:
        raise IndexError(
            f"Selected frame {frame_end} exceeds source video length {source_frame_count}"
        )
    fps = float(args.fps if args.fps is not None else source_fps)
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    clip_name = (
        f"{_safe_component(rel_seq)}__frames_{frame_start:06d}_{frame_end:06d}"
        f"__{bbox_suffix}"
    )
    output_dir = args.output_root.expanduser().resolve() / clip_name
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = output_dir / "rgb_bbox_mano_pressure.raw.mp4"
    output_video = output_dir / "rgb_bbox_mano_pressure.mp4"
    combined_size = (args.rgb_width + args.panel_height, args.panel_height)
    writer = cv2.VideoWriter(
        str(raw_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        combined_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not initialize video writer: {raw_video}")

    contact_frames: List[np.ndarray] = []
    frame_indices = list(range(frame_start, frame_end + 1))
    contact_target_indices = set(
        np.linspace(0, len(frame_indices) - 1, min(8, len(frame_indices)))
        .round()
        .astype(np.int64)
        .tolist()
    )
    try:
        for output_index, frame_index in enumerate(frame_indices):
            ok, rgb = capture.read()
            if not ok or rgb is None:
                raise RuntimeError(
                    f"Could not decode source video frame {frame_index}: {source_video}"
                )
            current_pressure = pressure[output_index]
            bbox_record = bbox_rows.get(frame_index)
            if bbox_record is None:
                _draw_missing_bbox(rgb)
            else:
                _draw_bbox(
                    rgb,
                    np.asarray(bbox_record["bbox"]),
                    float(bbox_record["bbox_score"]),
                    bbox_label,
                )
            rgb = _letterbox(rgb, (args.rgb_width, args.panel_height))
            rgb = _draw_panel_title(
                rgb,
                "ORIGINAL RGB + "
                + bbox_label.upper(),
            )
            _draw_status(rgb, frame_index, current_pressure)

            pressure_rgb = renderer.render(
                _pressure_vertex_colors(
                    current_pressure,
                    display_floor=args.display_floor,
                )
            )
            pressure_rgb = _draw_pressure_colorbar(
                pressure_rgb,
                display_floor=args.display_floor,
            )
            pressure_rgb = _draw_panel_title(
                pressure_rgb, "STORED MANO PRESSURE (RIGHT)"
            )
            pressure_bgr = np.ascontiguousarray(pressure_rgb[:, :, ::-1])
            combined = np.concatenate([rgb, pressure_bgr], axis=1)
            writer.write(combined)
            if output_index in contact_target_indices:
                contact_frames.append(combined.copy())
    finally:
        writer.release()
        capture.release()

    codec = _encode_h264(raw_video, output_video, fps)
    contact_sheet = output_dir / "contact_sheet.jpg"
    _write_contact_sheet(contact_frames, contact_sheet)

    selected_max = pressure.max(axis=1)
    selected_volume = pressure.sum(axis=1)
    selected_active = (pressure >= 0.05).sum(axis=1)
    metadata = {
        "schema": "egotactile_processed_clip_visualization_v1",
        "read_only_inputs": True,
        "sequence": rel_seq,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": len(frame_indices),
        "source_video": str(source_video),
        "source_video_frame_count": source_frame_count,
        "source_fps": source_fps,
        "output_fps": fps,
        "pressure_archive": str(archive_path),
        "pressure_key": PRESSURE_KEY,
        "pressure_shape": list(pressure_all.shape),
        "bbox_manifest": str(manifest_path) if needs_manifest else None,
        "bbox_source": bbox_source,
        "bbox_label": bbox_label,
        "bbox_coverage_in_rendered_range": len(bbox_rows) / len(frame_indices),
        "bbox_missing_frames": missing_bbox_frames,
        "mapping_schema": mapping_schema,
        "mapping_sha256": mapping_sha256,
        "normalization_schema": normalization_schema,
        "normalization_formula": normalization_formula,
        "bend_mixed_into_pressure": bend_mixed,
        "pressure_visualization": {
            "stored_values_modified": False,
            "temporal_smoothing": False,
            "display_floor": args.display_floor,
            "palm_mask_applied": False,
            "colormap": "OpenTouch gnuplot2(1-pressure)",
        },
        "selected_pressure_statistics": {
            "max_min": float(selected_max.min()),
            "max_median": float(np.median(selected_max)),
            "max_max": float(selected_max.max()),
            "volume_min": float(selected_volume.min()),
            "volume_median": float(np.median(selected_volume)),
            "volume_max": float(selected_volume.max()),
            "active_vertices_0.05_min": int(selected_active.min()),
            "active_vertices_0.05_median": float(np.median(selected_active)),
            "active_vertices_0.05_max": int(selected_active.max()),
        },
        "automatic_selection": selection_summary,
        "mano_mesh": str(mesh_path),
        "video_codec": codec,
        "output_video": str(output_video),
        "contact_sheet": str(contact_sheet),
    }
    metadata_path = output_dir / "manifest.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=True))
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--bbox-jsonl",
        type=Path,
        help="Optional SAM3 bboxes.jsonl; replaces current manifest bboxes for rendering",
    )
    parser.add_argument(
        "--bbox-meta-root",
        type=Path,
        help=(
            "Read original per-frame meta.json bboxes directly below this "
            "extracted_frames root, without consulting the global manifest"
        ),
    )
    parser.add_argument(
        "--allow-missing-bboxes",
        action="store_true",
        help="Render and label frames whose direct bbox metadata is absent",
    )
    parser.add_argument(
        "--bbox-label",
        help="Display label for an external bbox JSONL, e.g. 'SAM3 bare bbox'",
    )
    parser.add_argument(
        "--sequence",
        help="Processed relative sequence, e.g. gloved_hand/p002/Pepsi-330ml/repeat0000",
    )
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--selection-top-k", type=int, default=12)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--rgb-width", type=int, default=1280)
    parser.add_argument(
        "--display-floor",
        type=float,
        default=0.0,
        help="Visualization-only cutoff; default 0 shows all stored pressure",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bbox_jsonl is not None and args.bbox_meta_root is not None:
        raise ValueError("--bbox-jsonl and --bbox-meta-root are mutually exclusive")
    if args.bbox_meta_root is not None and not args.sequence:
        raise ValueError("--bbox-meta-root requires an explicit --sequence")
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.selection_top_k <= 0:
        raise ValueError("--selection-top-k must be positive")
    if args.panel_height <= 0 or args.rgb_width <= 0:
        raise ValueError("Panel dimensions must be positive")
    if not 0.0 <= args.display_floor <= 1.0:
        raise ValueError("--display-floor must lie in [0,1]")
    output_dir = render(args)
    print(f"EgoTactile clip audit ready: {output_dir}")


if __name__ == "__main__":
    main()
