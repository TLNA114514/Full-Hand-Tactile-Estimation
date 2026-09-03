"""Online HDF5 temporal sampling without materialized feature tensors."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_cache import sha256_file, sha256_json
from .temporal_flow import (
    TEMPORAL_PAIR_SCHEMA,
    _history_crop_transforms,
    _pair_control_seed,
    _pair_crop_affine_lookup,
    build_temporal_pair_index,
    current_to_history_crop_affine,
    strict_history_control_pair_indices,
    strict_lag_history_metadata,
    temporal_manifest_key,
)


class OnlineTemporalRecordIndex:
    """Expose normalized HDF5 rows to the strict temporal pair builder.

    The object intentionally implements only the metadata contract consumed by
    ``build_temporal_pair_index``. It never stores RGB, DINO features, logits,
    or tactile targets.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        input_resolution: Sequence[int],
        bbox_rescale_factor: float,
    ):
        self.dataset = dataset
        self.input_resolution = tuple(int(value) for value in input_resolution)
        self.bbox_rescale_factor = float(bbox_rescale_factor)
        self.config_sha256 = sha256_json(
            {
                "schema": "tactile_online_temporal_index_v1",
                "sample_count": len(dataset),
                "input_resolution": list(self.input_resolution),
                "bbox_rescale_factor": self.bbox_rescale_factor,
                "query_manifest_sha256": dict(
                    getattr(dataset, "query_manifest_sha256", {})
                ),
                "bbox_manifest_sha256": dict(
                    getattr(dataset, "bbox_manifest_sha256", {})
                ),
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def sample_records(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        for index in range(len(self.dataset)):
            row = self.dataset.samples[index]
            uid = str(row.get("sample_uid") or "")
            if not uid:
                raise RuntimeError(f"Online temporal row {index} has no sample_uid")
            if uid in records:
                raise RuntimeError(f"Duplicate online temporal sample_uid: {uid}")
            source_frame = row.get("source_frame_idx")
            if source_frame is None:
                source_frame = row.get("jq_pressure_frame_index")
            timestamp = row.get("timestamp")
            if timestamp is None:
                timestamp = row.get("frame_timestamp")
            records[uid] = {
                "cache_index": index,
                "dataset": str(row.get("dataset") or ""),
                "sequence_key": str(row.get("sequence_key") or ""),
                "query_alias": str(row.get("query_alias") or "query"),
                "frame_idx": int(row.get("frame_idx", row.get("frame_row", 0))),
                "source_frame_idx": (
                    None if source_frame is None else int(source_frame)
                ),
                "timestamp": None if timestamp is None else float(timestamp),
                "is_right": int(row.get("is_right", row.get("right", 1))),
                "bbox_xyxy": [
                    float(value) for value in row.get("bbox_xyxy", row.get("bbox", ()))
                ],
                "bbox_association_id": str(
                    row.get("bbox_association_id")
                    or row.get("bbox_track_id")
                    or row.get("bbox_instance_id")
                    or ""
                ),
            }
        return records


def build_online_temporal_pair_index(
    record_index: OnlineTemporalRecordIndex,
    manifests: Iterable[os.PathLike[str] | str],
    root: os.PathLike[str] | str,
    split: str,
    *,
    seed: int = 521,
    lock_timeout_seconds: int = 21600,
) -> Path:
    """Build/reuse a small metadata-only temporal pair index."""

    manifest_paths = tuple(
        str(Path(path).expanduser().resolve(strict=True)) for path in manifests
    )
    key = temporal_manifest_key(manifest_paths)
    root = Path(root).expanduser().resolve(strict=False)
    path = root / f"{split}-online-{record_index.config_sha256[:12]}-{key}.npz"
    lock = path.with_suffix(".lock")
    deadline = time.monotonic() + float(lock_timeout_seconds)
    acquired = False
    while time.monotonic() < deadline:
        try:
            lock.mkdir(parents=True)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > lock_timeout_seconds:
                    lock.rmdir()
                    continue
            except (FileNotFoundError, OSError):
                pass
            time.sleep(1.0)
    if not acquired:
        raise TimeoutError(f"Timed out waiting for temporal pair index lock: {lock}")
    try:
        return build_temporal_pair_index(
            record_index,
            manifest_paths,
            path,
            seed=seed,
            label_free_controls=True,
        )
    finally:
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass


class OnlineTemporalDataset(Dataset):
    """Read current/history RGB directly from sequence HDF5 containers.

    Frozen-adapter probes use ``pair_only=True``. Fresh temporal trunks and
    evaluation use ``pair_only=False`` so the complete frame distribution is
    retained and cold starts emit exact RGB-reset inputs. Batched fetching
    deduplicates source rows before delegating to the grouped HDF5 reader, so
    one batch does not decode the same JPEG more than once.
    """

    def __init__(
        self,
        dataset: Dataset,
        pair_index: os.PathLike[str] | str,
        *,
        palm_vertex_indices: Sequence[int],
        history_lags: Sequence[int] = (1, 2),
        include_control: bool = False,
        include_contralateral: bool = False,
        pair_only: bool = True,
        control_pressure_bins: Sequence[int] | None = None,
    ):
        self.dataset = dataset
        self.pair_only = bool(pair_only)
        self.include_control = bool(include_control)
        self.include_contralateral = bool(include_contralateral)
        self.history_lags = tuple(int(value) for value in history_lags)
        self.palm_vertex_indices = np.asarray(
            tuple(int(value) for value in palm_vertex_indices), dtype=np.int64
        )
        if self.palm_vertex_indices.size == 0:
            raise ValueError("palm_vertex_indices cannot be empty")
        self.pair_index_path = Path(pair_index).expanduser().resolve(strict=True)
        with np.load(self.pair_index_path, allow_pickle=False) as payload:
            self.arrays = {
                name: np.asarray(payload[name]) for name in payload.files
            }
        pair_count = len(self.arrays["current_index"])
        if any(len(value) != pair_count for value in self.arrays.values()):
            raise RuntimeError("Temporal pair-index arrays have different lengths")
        self.history_metadata = strict_lag_history_metadata(
            len(self.dataset),
            self.arrays["current_index"],
            self.arrays["previous_index"],
            self.history_lags,
            time_gap=self.arrays["time_gap"],
            bbox_iou=self.arrays["bbox_iou"],
            bbox_center_jump=self.arrays["bbox_center_jump"],
            bbox_abs_log_area_ratio=self.arrays["bbox_abs_log_area_ratio"],
            contralateral_previous_indices=self.arrays.get(
                "contralateral_previous_index"
            ),
        )
        self.history_indices = self.history_metadata["history_indices"]
        self.crop_affines = _pair_crop_affine_lookup(len(self.dataset), self.arrays)
        self.pair_lookup = np.full(len(self.dataset), -1, dtype=np.int64)
        current = np.asarray(self.arrays["current_index"], dtype=np.int64)
        if len(np.unique(current)) != len(current):
            raise RuntimeError("Temporal pair index contains duplicate current rows")
        self.pair_lookup[current] = np.arange(pair_count, dtype=np.int64)
        self.pair_sequence_ids = np.unique(
            np.asarray(self.arrays["sequence_key"], dtype=np.str_),
            return_inverse=True,
        )[1].astype(np.int64)
        self.sequence_count = (
            int(self.pair_sequence_ids.max()) + 1
            if self.pair_sequence_ids.size
            else 0
        )
        self.control_pair_indices = None
        self.control_history_indices = None
        self.control_pressure_bins = None
        if self.include_control:
            if control_pressure_bins is None:
                # No target-derived matching is used online. Availability, side,
                # and different-sequence constraints remain exact.
                pressure_bins = np.zeros(pair_count, dtype=np.int64)
                self.control_bin_source = "none_label_free"
            else:
                pressure_bins = np.asarray(control_pressure_bins, dtype=np.int64)
                if pressure_bins.shape != (pair_count,):
                    raise ValueError(
                        "control_pressure_bins must contain one value per temporal pair"
                    )
                self.control_bin_source = "external_label_free"
            self.control_pressure_bins = pressure_bins
            self.control_pair_indices = strict_history_control_pair_indices(
                self.arrays["sequence_key"],
                self.arrays["side"],
                pressure_bins,
                self.history_indices >= 0,
                seed=_pair_control_seed(self.pair_index_path),
            )
            self.control_history_indices = self.history_indices[
                self.control_pair_indices
            ]

    @property
    def pair_count(self) -> int:
        return len(self.arrays["current_index"])

    def __len__(self) -> int:
        return self.pair_count if self.pair_only else len(self.dataset)

    def _position(self, index: int) -> tuple[int, int, bool]:
        if self.pair_only:
            pair_position = int(index)
            return (
                int(self.arrays["current_index"][pair_position]),
                pair_position,
                True,
            )
        current_index = int(index)
        pair_position = int(self.pair_lookup[current_index])
        return current_index, pair_position, pair_position >= 0

    def _requested_source_indices(self, index: int) -> tuple[int, ...]:
        current_index, pair_position, eligible = self._position(index)
        requested = [current_index]
        if eligible:
            requested.extend(
                int(value)
                for value in self.history_indices[pair_position]
                if int(value) >= 0
            )
            if self.include_control:
                requested.extend(
                    int(value)
                    for value in self.control_history_indices[pair_position]
                    if int(value) >= 0
                )
            if self.include_contralateral:
                requested.extend(
                    int(value)
                    for value in self.history_metadata[
                        "contralateral_history_indices"
                    ][pair_position]
                    if int(value) >= 0
                )
        return tuple(requested)

    @staticmethod
    def _stack_images(
        indices: Sequence[int],
        current_index: int,
        samples: Mapping[int, Mapping[str, Any]],
    ) -> torch.Tensor:
        current = samples[current_index]
        return torch.stack(
            [
                samples[int(value)]["img"] if int(value) >= 0 else current["img"]
                for value in indices
            ]
        )

    def _assemble(
        self,
        index: int,
        samples: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        current_index, pair_position, eligible = self._position(index)
        current = samples[current_index]
        if eligible:
            history_indices = self.history_indices[pair_position]
        else:
            history_indices = np.full(len(self.history_lags), -1, dtype=np.int64)
        history_available = torch.as_tensor(
            history_indices >= 0, dtype=torch.float32
        )
        palm = torch.as_tensor(self.palm_vertex_indices, dtype=torch.long)
        history_targets = []
        for history_index in history_indices:
            sample = samples[int(history_index)] if int(history_index) >= 0 else current
            history_targets.append(sample["tactile_signal"].index_select(0, palm))

        def actual_crop_transforms(indices: Sequence[int]) -> np.ndarray:
            current_affine = current.get("crop_affine")
            if current_affine is None:
                return _history_crop_transforms(
                    self.crop_affines, current_index, indices
                )
            current_array = np.asarray(current_affine, dtype=np.float32)
            transforms = []
            for source_index in indices:
                if int(source_index) < 0:
                    transforms.append(np.eye(3, dtype=np.float32))
                    continue
                history_affine = samples[int(source_index)].get("crop_affine")
                if history_affine is None:
                    return _history_crop_transforms(
                        self.crop_affines, current_index, indices
                    )
                transforms.append(
                    current_to_history_crop_affine(
                        current_array,
                        np.asarray(history_affine, dtype=np.float32),
                    )
                )
            return np.stack(transforms).astype(np.float32, copy=False)
        result: dict[str, Any] = {
            "current_image": current["img"],
            "history_images": self._stack_images(
                history_indices, current_index, samples
            ),
            "history_available": history_available,
            "history_time_gap": torch.from_numpy(
                np.array(
                    self.history_metadata["history_time_gap"][pair_position]
                    if eligible
                    else np.zeros(len(self.history_lags), dtype=np.float32),
                    copy=True,
                )
            ).float(),
            "history_tactile_signal": torch.stack(history_targets),
            "tactile_signal": current["tactile_signal"].index_select(0, palm),
            "has_tactile": current["has_tactile"].float().reshape(()),
            "current_index": torch.tensor(current_index, dtype=torch.long),
            "sequence_id": torch.tensor(
                int(self.pair_sequence_ids[pair_position]) if eligible else -1,
                dtype=torch.long,
            ),
            "history_crop_transform": torch.from_numpy(
                actual_crop_transforms(history_indices)
            ).float(),
        }
        result["previous_tactile_signal"] = result["history_tactile_signal"][0]
        if self.include_control:
            if eligible:
                control_indices = self.control_history_indices[pair_position]
                control_pair = int(self.control_pair_indices[pair_position])
                control_time = self.history_metadata["history_time_gap"][control_pair]
                control_available = control_indices >= 0
            else:
                control_indices = np.full(
                    len(self.history_lags), -1, dtype=np.int64
                )
                control_time = np.zeros(len(self.history_lags), dtype=np.float32)
                control_available = control_indices >= 0
            result.update(
                {
                    "control_history_images": self._stack_images(
                        control_indices, current_index, samples
                    ),
                    "control_history_available": torch.as_tensor(
                        control_available, dtype=torch.float32
                    ),
                    "control_history_time_gap": torch.from_numpy(
                        np.array(control_time, copy=True)
                    ).float(),
                    "control_history_crop_transform": torch.from_numpy(
                        actual_crop_transforms(control_indices)
                    ).float(),
                }
            )
        if self.include_contralateral:
            if eligible:
                contra_indices = self.history_metadata[
                    "contralateral_history_indices"
                ][pair_position]
            else:
                contra_indices = np.full(
                    len(self.history_lags), -1, dtype=np.int64
                )
            result.update(
                {
                    "contralateral_history_images": self._stack_images(
                        contra_indices, current_index, samples
                    ),
                    "contralateral_history_available": torch.as_tensor(
                        contra_indices >= 0, dtype=torch.float32
                    ),
                    "contralateral_history_crop_transform": torch.from_numpy(
                        actual_crop_transforms(contra_indices)
                    ).float(),
                }
            )
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        requested = self._requested_source_indices(int(index))
        unique = tuple(dict.fromkeys(requested))
        samples = {source: self.dataset[source] for source in unique}
        return self._assemble(int(index), samples)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        positions = [int(value) for value in indices]
        requested = []
        for index in positions:
            requested.extend(self._requested_source_indices(index))
        unique = tuple(dict.fromkeys(requested))
        batched_getter = getattr(self.dataset, "__getitems__", None)
        if callable(batched_getter):
            loaded = batched_getter(list(unique))
        else:
            loaded = [self.dataset[index] for index in unique]
        samples = dict(zip(unique, loaded))
        return [self._assemble(index, samples) for index in positions]


class OnlineTemporalClipDataset(Dataset):
    """Non-overlapping causal clips built from the strict temporal edge graph.

    Every source row appears in exactly one clip.  Clips never cross an edge
    rejected by the existing strict pair builder, so association switches,
    side changes, gaps, and invalid bbox motion naturally start a new clip.
    Short terminal clips are padded with their final image but expose an exact
    validity mask; padded frames are never used as targets or attention keys.
    """

    def __init__(
        self,
        dataset: Dataset,
        pair_index: os.PathLike[str] | str,
        *,
        palm_vertex_indices: Sequence[int],
        clip_length: int = 8,
        include_control: bool = False,
        seed: int = 521,
    ):
        self.dataset = dataset
        self.clip_length = int(clip_length)
        self.include_control = bool(include_control)
        self.seed = int(seed)
        if self.clip_length < 2:
            raise ValueError("clip_length must be at least 2")
        self.palm_vertex_indices = np.asarray(
            tuple(int(value) for value in palm_vertex_indices), dtype=np.int64
        )
        if self.palm_vertex_indices.size == 0:
            raise ValueError("palm_vertex_indices cannot be empty")
        self.pair_index_path = Path(pair_index).expanduser().resolve(strict=True)
        with np.load(self.pair_index_path, allow_pickle=False) as payload:
            self.arrays = {name: np.asarray(payload[name]) for name in payload.files}
        self.clips, self.clip_lengths, self.clip_sequence_ids = self._build_clips()
        self.sequence_count = (
            int(self.clip_sequence_ids.max()) + 1
            if self.clip_sequence_ids.size
            else 0
        )
        self.control_clip_indices = (
            self._build_control_clips() if self.include_control else None
        )
        self.config_sha256 = sha256_json(
            {
                "schema": "tactile_online_causal_clip_v1",
                "pair_index": sha256_file(self.pair_index_path),
                "sample_count": len(self.dataset),
                "clip_count": len(self.clips),
                "clip_length": self.clip_length,
                "seed": self.seed,
            }
        )

    def _build_clips(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_count = len(self.dataset)
        previous = np.full(sample_count, -1, dtype=np.int64)
        edge_gap = np.zeros(sample_count, dtype=np.float32)
        current_rows = np.asarray(self.arrays["current_index"], dtype=np.int64)
        previous_rows = np.asarray(self.arrays["previous_index"], dtype=np.int64)
        gaps = np.asarray(self.arrays["time_gap"], dtype=np.float32)
        for current, prior, gap in zip(current_rows, previous_rows, gaps):
            if not (0 <= int(current) < sample_count):
                raise RuntimeError(f"Temporal current index is out of range: {current}")
            if int(prior) >= sample_count:
                raise RuntimeError(f"Temporal previous index is out of range: {prior}")
            previous[int(current)] = int(prior)
            edge_gap[int(current)] = max(float(gap), 0.0)

        successor = np.full(sample_count, -1, dtype=np.int64)
        for current, prior in enumerate(previous):
            if prior < 0:
                continue
            if successor[prior] >= 0 and successor[prior] != current:
                # Duplicate/current-row anomalies must not make a full run
                # unrecoverable. Keep one strict continuation and make the
                # additional child an explicit cold-start clip boundary.
                previous[current] = -1
                edge_gap[current] = 0.0
                continue
            successor[prior] = current

        visited = np.zeros(sample_count, dtype=np.bool_)
        segments: list[list[int]] = []
        roots = np.flatnonzero(previous < 0)
        for root in roots:
            segment = []
            cursor = int(root)
            while cursor >= 0:
                if visited[cursor]:
                    raise RuntimeError("Strict temporal graph contains a cycle")
                visited[cursor] = True
                segment.append(cursor)
                cursor = int(successor[cursor])
            segments.append(segment)
        if not bool(visited.all()):
            missing = np.flatnonzero(~visited)
            raise RuntimeError(
                "Strict temporal graph contains a cycle or disconnected loop; "
                f"first unresolved rows={missing[:8].tolist()}"
            )

        sequence_keys = [str(row.get("sequence_key") or "") for row in self.dataset.samples]
        sequence_values = sorted(set(sequence_keys))
        sequence_lookup = {value: index for index, value in enumerate(sequence_values)}
        clips: list[np.ndarray] = []
        lengths: list[int] = []
        sequence_ids: list[int] = []
        for segment in segments:
            for start in range(0, len(segment), self.clip_length):
                values = segment[start : start + self.clip_length]
                padded = np.full(self.clip_length, -1, dtype=np.int64)
                padded[: len(values)] = values
                clips.append(padded)
                lengths.append(len(values))
                sequence_ids.append(sequence_lookup[sequence_keys[values[0]]])
        clip_array = np.stack(clips) if clips else np.empty((0, self.clip_length), np.int64)
        length_array = np.asarray(lengths, dtype=np.int64)
        sequence_array = np.asarray(sequence_ids, dtype=np.int64)
        valid_rows = clip_array[clip_array >= 0]
        if len(valid_rows) != sample_count or len(np.unique(valid_rows)) != sample_count:
            raise RuntimeError("Non-overlapping clip partition lost or duplicated samples")
        self._edge_gap = edge_gap
        return clip_array, length_array, sequence_array

    def _build_control_clips(self) -> np.ndarray:
        if len(self.clips) < 2:
            return np.zeros(len(self.clips), dtype=np.int64)
        side = np.asarray(
            [
                int(
                    self.dataset.samples[int(clip[0])].get(
                        "is_right", self.dataset.samples[int(clip[0])].get("right", 1)
                    )
                )
                for clip in self.clips
            ],
            dtype=np.int64,
        )
        controls = np.arange(len(self.clips), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        for hand_side in (0, 1):
            for clip_length in range(1, self.clip_length + 1):
                candidates = np.flatnonzero(
                    (side == hand_side) & (self.clip_lengths == clip_length)
                )
                if len(candidates) < 2:
                    continue
                candidates = candidates[rng.permutation(len(candidates))]
                for position, clip_index in enumerate(candidates):
                    donor = int(candidates[(position + 1) % len(candidates)])
                    for offset in range(1, len(candidates)):
                        proposal = int(candidates[(position + offset) % len(candidates)])
                        if (
                            self.clip_sequence_ids[proposal]
                            != self.clip_sequence_ids[clip_index]
                        ):
                            donor = proposal
                            break
                    controls[int(clip_index)] = donor
        return controls

    def __len__(self) -> int:
        return len(self.clips)

    def _requested_source_indices(self, clip_index: int) -> tuple[int, ...]:
        requested = [int(value) for value in self.clips[clip_index] if int(value) >= 0]
        if self.include_control:
            donor = int(self.control_clip_indices[clip_index])
            requested.extend(
                int(value) for value in self.clips[donor] if int(value) >= 0
            )
        return tuple(requested)

    def _assemble_clip(
        self,
        clip_index: int,
        samples: Mapping[int, Mapping[str, Any]],
        *,
        source_clip_index: int | None = None,
    ) -> dict[str, torch.Tensor]:
        source = clip_index if source_clip_index is None else int(source_clip_index)
        indices = self.clips[source]
        length = int(self.clip_lengths[source])
        valid_indices = [int(value) for value in indices[:length]]
        if not valid_indices:
            raise RuntimeError("Temporal clip cannot be empty")
        padding_sample = samples[valid_indices[-1]]
        palm = torch.as_tensor(self.palm_vertex_indices, dtype=torch.long)
        images = []
        targets = []
        has_tactile = []
        affines = []
        times = []
        elapsed = 0.0
        for position in range(self.clip_length):
            if position < length:
                row_index = valid_indices[position]
                sample = samples[row_index]
                if position > 0:
                    elapsed += float(self._edge_gap[row_index])
            else:
                row_index = -1
                sample = padding_sample
            images.append(sample["img"])
            targets.append(sample["tactile_signal"].index_select(0, palm))
            has_tactile.append(
                sample["has_tactile"].float().reshape(())
                if position < length
                else torch.zeros((), dtype=torch.float32)
            )
            affines.append(sample["crop_affine"].float())
            times.append(elapsed)
        valid = torch.arange(self.clip_length) < length
        result = {
            "clip_images": torch.stack(images),
            "clip_tactile_signal": torch.stack(targets),
            "clip_has_tactile": torch.stack(has_tactile),
            "clip_valid": valid.float(),
            "clip_time": torch.tensor(times, dtype=torch.float32),
            "clip_crop_affine": torch.stack(affines),
            "clip_source_indices": torch.as_tensor(indices, dtype=torch.long),
        }
        return result

    def _assemble(
        self, clip_index: int, samples: Mapping[int, Mapping[str, Any]]
    ) -> dict[str, Any]:
        result: dict[str, Any] = self._assemble_clip(clip_index, samples)
        result["clip_index"] = torch.tensor(clip_index, dtype=torch.long)
        result["sequence_id"] = torch.tensor(
            int(self.clip_sequence_ids[clip_index]), dtype=torch.long
        )
        if self.include_control:
            donor = int(self.control_clip_indices[clip_index])
            control = self._assemble_clip(
                clip_index, samples, source_clip_index=donor
            )
            result.update({f"control_{name}": value for name, value in control.items()})
            result["control_is_cross_sequence"] = torch.tensor(
                self.clip_sequence_ids[donor] != self.clip_sequence_ids[clip_index],
                dtype=torch.float32,
            )
        return result

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip_index = int(index)
        requested = tuple(dict.fromkeys(self._requested_source_indices(clip_index)))
        previous_seeds = getattr(
            self.dataset, "_augmentation_group_seed_by_index", None
        )
        if previous_seeds is not None and bool(getattr(self.dataset, "train", False)):
            seed = int(np.random.randint(0, np.iinfo(np.int32).max))
            self.dataset._augmentation_group_seed_by_index = {
                source: seed for source in requested
            }
        try:
            samples = {source: self.dataset[source] for source in requested}
        finally:
            if previous_seeds is not None:
                self.dataset._augmentation_group_seed_by_index = previous_seeds
        return self._assemble(clip_index, samples)

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        clip_indices = [int(value) for value in indices]
        requested: list[int] = []
        for clip_index in clip_indices:
            requested.extend(self._requested_source_indices(clip_index))
        unique = tuple(dict.fromkeys(requested))
        batched_getter = getattr(self.dataset, "__getitems__", None)
        previous_seeds = getattr(
            self.dataset, "_augmentation_group_seed_by_index", None
        )
        if previous_seeds is not None and bool(getattr(self.dataset, "train", False)):
            seed_by_source_clip: dict[int, int] = {}
            augmentation_seeds: dict[int, int] = {}
            for clip_index in clip_indices:
                source_clips = [clip_index]
                if self.include_control:
                    source_clips.append(int(self.control_clip_indices[clip_index]))
                for source_clip in source_clips:
                    if source_clip not in seed_by_source_clip:
                        seed_by_source_clip[source_clip] = int(
                            np.random.randint(0, np.iinfo(np.int32).max)
                        )
                    seed = seed_by_source_clip[source_clip]
                    for source in self.clips[source_clip]:
                        if int(source) >= 0:
                            augmentation_seeds[int(source)] = seed
            self.dataset._augmentation_group_seed_by_index = augmentation_seeds
        try:
            loaded = (
                batched_getter(list(unique))
                if callable(batched_getter)
                else [self.dataset[index] for index in unique]
            )
        finally:
            if previous_seeds is not None:
                self.dataset._augmentation_group_seed_by_index = previous_seeds
        samples = dict(zip(unique, loaded))
        return [self._assemble(clip_index, samples) for clip_index in clip_indices]


def online_temporal_contract(
    dataset: OnlineTemporalDataset,
    manifests: Sequence[str],
) -> dict[str, Any]:
    metadata_path = dataset.pair_index_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != TEMPORAL_PAIR_SCHEMA:
        raise RuntimeError("Online temporal pair metadata schema is not current")
    return {
        "mode": "online",
        "query_manifests": [str(Path(value).resolve()) for value in manifests],
        "query_manifest_sha256": [sha256_file(value) for value in manifests],
        "pair_index": str(dataset.pair_index_path),
        "pair_index_sha256": sha256_file(dataset.pair_index_path),
        "pair_contract_sha256": str(metadata.get("contract_sha256") or ""),
        "control_bin_source": getattr(dataset, "control_bin_source", "none"),
        "feature_cache": None,
    }
