#!/usr/bin/env python3
"""Render deterministic random and low-quality HaMeR sidecar samples."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping

import cv2
import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hand_pose_priors.pose_sidecar import (  # noqa: E402
    HaMeRPoseSidecar,
    tactile_crop_affine,
    transform_uv,
)


VALID_STATUS = 1
DEFAULT_ROOT = Path(
    "/home/ma-user/work/cfzhao/hand_pose_sidecars/touchanything_hamer_v1"
)
DEFAULT_MANO = REPO_ROOT / "hamer/_DATA/data/mano/MANO_RIGHT.pkl"


def _json_row(path: Path, offset: int) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        handle.seek(int(offset))
        row = json.loads(handle.readline())
    if not isinstance(row, Mapping):
        raise ValueError(f"Invalid manifest row at byte {offset}: {path}")
    return row


def _decode_frame(path: Path, frame_row: int) -> np.ndarray:
    with h5py.File(path, "r", libver="latest") as handle:
        offsets = handle["images/rgb/jpeg_offsets"]
        start, stop = np.asarray(offsets[frame_row : frame_row + 2], dtype=np.uint64)
        encoded = np.asarray(
            handle["images/rgb/jpeg_data"][int(start) : int(stop)], dtype=np.uint8
        )
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode frame_row={frame_row} from {path}")
    return image


def _load_edges(mano_path: Path) -> np.ndarray:
    with mano_path.open("rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    faces = np.asarray(model["f"], dtype=np.int32)
    edges = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _draw_mesh(
    image: np.ndarray,
    uv: np.ndarray,
    edges: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 190, 255),
) -> np.ndarray:
    result = image.copy()
    overlay = image.copy()
    height, width = image.shape[:2]
    finite = np.isfinite(uv).all(axis=1)
    inside = (
        finite
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    for first, second in edges:
        if inside[first] and inside[second]:
            point_a = tuple(np.rint(uv[first]).astype(int))
            point_b = tuple(np.rint(uv[second]).astype(int))
            cv2.line(overlay, point_a, point_b, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, result, 0.28, 0.0, result)
    for point in uv[inside][::4]:
        cv2.circle(result, tuple(np.rint(point).astype(int)), 1, (255, 255, 255), -1)
    return result


def _label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    header_height = 27 * len(lines) + 10
    output = np.zeros((image.shape[0] + header_height, image.shape[1], 3), np.uint8)
    output[header_height:] = image
    for index, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (8, 24 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return output


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _render_record(
    reader: HaMeRPoseSidecar,
    split: str,
    source_row: int,
    edges: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    record = reader.get(split, source_row, derive_uv=True)
    source = reader.sources[split]
    offsets_path = reader.root / source["index_files"]["offsets"]["path"]
    offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
    manifest_row = _json_row(Path(source["path"]), int(offsets[source_row]))
    h5_path = Path(reader.config["request"]["processed_root"]) / str(
        manifest_row["h5_relpath"]
    )
    image = _decode_frame(h5_path, int(record["frame_row"]))
    full = _draw_mesh(image, record["vertices_uv_full"], edges)
    bbox = np.asarray(record["bbox_xyxy"], dtype=np.float32)
    cv2.rectangle(
        full,
        tuple(np.rint(bbox[:2]).astype(int)),
        tuple(np.rint(bbox[2:]).astype(int)),
        (60, 255, 60),
        2,
    )

    affine = tactile_crop_affine(bbox, scale=1.2, output_hw=(256, 192))
    crop = cv2.warpAffine(image, affine, (192, 256), flags=cv2.INTER_LINEAR)
    crop_uv = transform_uv(record["vertices_uv_full"], affine)
    if not record["is_right"]:
        crop = cv2.flip(crop, 1)
        crop_uv[:, 0] = crop.shape[1] - 1 - crop_uv[:, 0]
    crop = _draw_mesh(crop, crop_uv, edges)

    full_panel = _fit(full, 640, 480)
    crop_panel = _fit(crop, 360, 480)
    panel = np.concatenate((full_panel, crop_panel), axis=1)
    hand = "right" if record["is_right"] else "left"
    panel = _label(
        panel,
        [
            f"{split} row={source_row:,} {hand} status={record['status']}",
            "in_frame="
            f"{record['in_frame_fraction']:.3f}  "
            f"depth={record['positive_depth_fraction']:.3f}",
            str(record["sample_uid"]),
        ],
    )
    summary = {
        "split": split,
        "source_row": int(source_row),
        "sample_uid": str(record["sample_uid"]),
        "is_right": bool(record["is_right"]),
        "status": int(record["status"]),
        "in_frame_fraction": float(record["in_frame_fraction"]),
        "positive_depth_fraction": float(record["positive_depth_fraction"]),
        "bbox_xyxy": bbox.tolist(),
        "focal_length": float(record["focal_length"]),
        "camera_translation": np.asarray(record["camera_translation"]).tolist(),
    }
    return panel, summary


def _random_rows(
    reader: HaMeRPoseSidecar,
    split: str,
    count_per_hand: int,
    rng: np.random.Generator,
) -> list[int]:
    selected: list[int] = []
    used: set[int] = set()
    for is_right in (False, True):
        attempts = 0
        while (
            sum(
                bool(reader.get(split, row)["is_right"]) == is_right
                for row in selected
            )
            < count_per_hand
        ):
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(f"Could not sample enough valid {split}/{is_right} rows")
            row = int(rng.integers(0, reader.split_count(split)))
            if row in used:
                continue
            record = reader.get(split, row)
            if record["status"] != VALID_STATUS or record["is_right"] != is_right:
                continue
            used.add(row)
            selected.append(row)
    return selected


def _lowest_in_frame_rows(reader: HaMeRPoseSidecar, per_split: int) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for split in reader.sources:
        candidates: list[tuple[float, int]] = []
        for (item_split, shard_index), item in reader.work_items.items():
            if item_split != split:
                continue
            handle = reader._handle(split, shard_index)
            status = handle["quality/status"][:]
            quality = handle["quality/in_frame_fraction"][:].astype(
                np.float32, copy=False
            )
            valid_indices = np.flatnonzero(status == VALID_STATUS)
            if not len(valid_indices):
                continue
            take = valid_indices[
                np.argsort(quality[valid_indices])[: min(per_split, len(valid_indices))]
            ]
            for local_row in take:
                entry = (-float(quality[local_row]), int(item["source_start"]) + int(local_row))
                if len(candidates) < per_split:
                    heapq.heappush(candidates, entry)
                elif entry > candidates[0]:
                    heapq.heapreplace(candidates, entry)
        result[split] = [row for _, row in sorted(candidates, reverse=True)]
    return result


def _write_sheet(path: Path, panels: list[np.ndarray], columns: int = 2) -> None:
    if not panels:
        return
    tile_width, tile_height = 700, 410
    rows = (len(panels) + columns - 1) // columns
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), np.uint8)
    for index, panel in enumerate(panels):
        tile = _fit(panel, tile_width, tile_height)
        y, x = divmod(index, columns)
        sheet[
            y * tile_height : (y + 1) * tile_height,
            x * tile_width : (x + 1) * tile_width,
        ] = tile
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--mano", default=str(DEFAULT_MANO))
    parser.add_argument("--random-per-hand", type=int, default=2)
    parser.add_argument("--low-quality-per-split", type=int, default=1)
    parser.add_argument("--seed", type=int, default=521)
    args = parser.parse_args()

    root = Path(args.sidecar_root).expanduser().resolve(strict=True)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else root / "audits/random_visualization"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = _load_edges(Path(args.mano).expanduser().resolve(strict=True))
    rng = np.random.default_rng(args.seed)
    summaries: dict[str, list[dict[str, Any]]] = {"random": [], "low_in_frame": []}

    with HaMeRPoseSidecar(root, max_open_shards=8) as reader:
        random_spec = {
            split: _random_rows(reader, split, args.random_per_hand, rng)
            for split in reader.sources
        }
        low_spec = _lowest_in_frame_rows(reader, args.low_quality_per_split)
        for group, spec in (("random", random_spec), ("low_in_frame", low_spec)):
            panels = []
            group_root = output_dir / group
            group_root.mkdir(parents=True, exist_ok=True)
            for split, rows in spec.items():
                for source_row in rows:
                    panel, summary = _render_record(reader, split, source_row, edges)
                    filename = f"{split}_row-{source_row:08d}.jpg"
                    if not cv2.imwrite(
                        str(group_root / filename), panel, [cv2.IMWRITE_JPEG_QUALITY, 95]
                    ):
                        raise RuntimeError(f"Could not write {group_root / filename}")
                    summary["image"] = str((group_root / filename).relative_to(output_dir))
                    summaries[group].append(summary)
                    panels.append(panel)
            _write_sheet(output_dir / f"{group}_contact_sheet.jpg", panels)

    summary_path = output_dir / "samples.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "random_samples": len(summaries["random"]),
                "low_in_frame_samples": len(summaries["low_in_frame"]),
                "random_contact_sheet": str(output_dir / "random_contact_sheet.jpg"),
                "low_in_frame_contact_sheet": str(output_dir / "low_in_frame_contact_sheet.jpg"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
