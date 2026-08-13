import os
import sys
import json
import cv2
import numpy as np
import torch
import hashlib
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from torch.utils.data import Dataset
from yacs.config import CfgNode

if __package__:
    from .process_lifecycle import initialize_worker_parent_death_signal
else:
    from process_lifecycle import initialize_worker_parent_death_signal

_workspace_import_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _workspace_import_root not in sys.path:
    sys.path.append(_workspace_import_root)

try:
    from tactile_input_priors.depth_sidecar import (
        SequenceSidecarReader,
        sequence_sidecar_filename,
        warp_record_pointnormal,
    )
except ImportError:
    SequenceSidecarReader = None
    sequence_sidecar_filename = None
    warp_record_pointnormal = None

try:
    import orjson
except ImportError:
    orjson = None

try:
    if __package__:
        from . import hdf5_storage as _hdf5_storage
    else:
        import hdf5_storage as _hdf5_storage
except ImportError:
    _hdf5_storage = None

if __package__:
    from .data.indexing import *
else:
    from data.indexing import *

# Add paths
ft_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(ft_dir, ".."))
sys.path.append(os.path.join(workspace_dir, "hamer"))

# Global keypoint permutation for hand flipping (MediaPipe format)
FLIP_KEYPOINT_PERMUTATION = list(range(21))
HDF5_STORAGE_SCHEMA_VERSION = str(
    getattr(
        _hdf5_storage,
        "HDF5_SCHEMA_VERSION",
        getattr(_hdf5_storage, "SCHEMA_VERSION", "tactile_sequence_hdf5_v1"),
    )
)

DATASET_ROOTS = {
    "opentouch": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "open_touch": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "ot": "/data1/jiangrui/OpenTouch Data/full_dataset",
    "touchanything": "/data1/jiangrui/EgoTouch/extracted_frames",
    "touch_anything": "/data1/jiangrui/EgoTouch/extracted_frames",
    "egotouch": "/data1/jiangrui/EgoTouch/extracted_frames",
    "ego_touch": "/data1/jiangrui/EgoTouch/extracted_frames",
    "ta": "/data1/jiangrui/EgoTouch/extracted_frames",
    "egotactile": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
    "ego_tactile": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
    "ego": "/data1/jiangrui/EgoTactile/Raw_data/extracted_frames",
}


class OpenTouchTactileDataset(DatasetIndexingMixin, Dataset):
    def __init__(self, cfg: CfgNode, split: str = "train", 
                 data_dir: str = None, train: bool = True, index_workers: int = 1,
                 index_chunksize: int = 256, index_cache_dir: str = None,
                 rebuild_index: bool = False, index_cache_timeout: int = 3600,
                 index_backend: str = "process",
                 sample_records=None, tactile_only: bool = False,
                 input_resolution=None,
                 bbox_rescale_factor: float = 2.0,
                 bbox_source_policy: str = "any",
                 bbox_manifests=None,
                 lazy_index_records: bool = False,
                 augmentation_enabled: bool = True,
                 index_process_worker_cap: int = 64,
                 index_manifest: str = None,
                 expected_datasets=None,
                 data_backend: str = "auto",
                 query_manifests=None,
                 hdf5_handle_cache_size: int = 4,
                 hdf5_manifest_cache_dir: str = None,
                 depth_sidecar_root: str = None,
                 depth_control: str = "none",
                 depth_output_hw=(32, 24),
                 io_debug_enabled: bool = False,
                 hdf5_batch_read_mode: str = "grouped"):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.train = train
        self.augmentation_enabled = bool(augmentation_enabled)
        self.tactile_only = bool(tactile_only)
        self.index_workers = max(1, int(index_workers))
        self.index_chunksize = max(1, int(index_chunksize))
        self.index_backend = str(index_backend or "process").lower()
        if self.index_backend not in ("process", "thread"):
            raise ValueError(f"Unsupported index_backend: {index_backend!r}. Use 'process' or 'thread'.")
        self.index_cache_dir = index_cache_dir
        self.hdf5_manifest_cache_dir = (
            os.path.realpath(
                os.path.abspath(os.path.expanduser(str(hdf5_manifest_cache_dir)))
            )
            if hdf5_manifest_cache_dir and str(hdf5_manifest_cache_dir).strip()
            else None
        )
        self.depth_sidecar_root = (
            os.path.realpath(os.path.abspath(os.path.expanduser(str(depth_sidecar_root))))
            if depth_sidecar_root and str(depth_sidecar_root).strip()
            else None
        )
        self.depth_control = str(depth_control or "none")
        self.io_debug_enabled = bool(io_debug_enabled)
        self.hdf5_batch_read_mode = str(hdf5_batch_read_mode).strip().lower()
        if self.hdf5_batch_read_mode not in ("grouped", "streaming"):
            raise ValueError("hdf5_batch_read_mode must be grouped or streaming")
        if self.depth_control not in ("none", "sample_spatial_shuffle"):
            raise ValueError(
                "depth_control must be none or sample_spatial_shuffle"
            )
        self.depth_output_hw = tuple(int(value) for value in depth_output_hw)
        if len(self.depth_output_hw) != 2 or min(self.depth_output_hw) <= 0:
            raise ValueError(
                f"depth_output_hw must contain two positive integers, got "
                f"{depth_output_hw!r}"
            )
        if self.depth_sidecar_root is not None:
            if SequenceSidecarReader is None:
                raise ImportError(
                    "Depth sidecars require tactile_input_priors.depth_sidecar"
                )
            if not os.path.isdir(self.depth_sidecar_root):
                raise FileNotFoundError(
                    f"Depth sidecar root does not exist: {self.depth_sidecar_root}"
                )
        self._depth_sidecar_pid = None
        self._depth_sidecar_readers = OrderedDict()
        self.depth_sidecar_contract = {}
        self.index_process_worker_cap = max(0, int(index_process_worker_cap))
        self.index_manifest = (
            os.path.abspath(os.path.expanduser(str(index_manifest)))
            if index_manifest and str(index_manifest).strip()
            else None
        )
        self.expected_datasets = canonical_dataset_filter(expected_datasets)
        self.index_manifest_sha256 = (
            sha256_file(self.index_manifest)
            if self.index_manifest and os.path.isfile(self.index_manifest)
            else ""
        )
        self.rebuild_index = bool(rebuild_index)
        self.index_cache_timeout = int(index_cache_timeout)
        self._active_cache_lock = None
        self.lazy_index_records = bool(lazy_index_records)
        
        if input_resolution is None:
            self.input_resolution = (int(cfg.MODEL.IMAGE_SIZE), int(cfg.MODEL.IMAGE_SIZE))
        else:
            if len(input_resolution) != 2:
                raise ValueError("input_resolution must contain height and width")
            self.input_resolution = tuple(int(value) for value in input_resolution)
        self.img_size = self.input_resolution[0]
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.rescale_factor = float(bbox_rescale_factor)
        if not 1.0 <= self.rescale_factor <= 4.0:
            raise ValueError("bbox_rescale_factor must lie in [1.0, 4.0]")
        self.bbox_source_policy = str(bbox_source_policy or "any").lower()
        if self.bbox_source_policy not in BBOX_SOURCE_POLICIES:
            raise ValueError(
                f"bbox_source_policy must be one of {BBOX_SOURCE_POLICIES}, "
                f"got {bbox_source_policy!r}"
            )
        if isinstance(bbox_manifests, str):
            bbox_manifests = bbox_manifests.split(",")
        self.bbox_manifests = tuple(
            os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))
            for path in (bbox_manifests or ())
            if str(path).strip()
        )
        missing_bbox_manifests = [
            path for path in self.bbox_manifests if not os.path.isfile(path)
        ]
        if missing_bbox_manifests:
            raise FileNotFoundError(
                "SAM3 bbox manifest(s) are missing: " + ", ".join(missing_bbox_manifests)
            )
        self.bbox_manifest_sha256 = {
            path: persistent_sha256_file(path, self.hdf5_manifest_cache_dir)
            for path in self.bbox_manifests
        }
        self._bbox_manifest_overlay_index = None

        if data_dir is None:
            data_dirs = ["/data1/jiangrui/OpenTouch Data/extracted_dataset"]
        elif isinstance(data_dir, (list, tuple)):
            data_dirs = [str(d) for d in data_dir if str(d).strip()]
        else:
            data_dirs = [d.strip() for d in str(data_dir).split(",") if d.strip()]
        self.data_dirs = data_dirs

        requested_backend = str(data_backend or "legacy_dirs").strip().lower()
        if requested_backend not in ("legacy_dirs", "sequence_hdf5", "auto"):
            raise ValueError(
                "data_backend must be one of legacy_dirs|sequence_hdf5|auto, "
                f"got {data_backend!r}"
            )
        self.query_manifest_specs = self._normalize_query_manifest_specs(query_manifests)
        if not self.query_manifest_specs and requested_backend in ("auto", "sequence_hdf5"):
            self.query_manifest_specs = self._discover_query_manifest_specs()
        if requested_backend == "auto":
            has_hdf5_records = False
            if isinstance(sample_records, (list, tuple)) and sample_records:
                first_record = sample_records[0]
                has_hdf5_records = (
                    isinstance(first_record, dict)
                    and ("h5_path" in first_record or "h5_relpath" in first_record)
                )
            self.data_backend = (
                "sequence_hdf5"
                if self.query_manifest_specs or has_hdf5_records
                else "legacy_dirs"
            )
        else:
            self.data_backend = requested_backend
        if self.data_backend == "sequence_hdf5" and sample_records is None and not self.query_manifest_specs:
            raise ValueError(
                "data_backend='sequence_hdf5' requires query_manifests; "
                "HDF5 mode never scans directories or reuses the legacy index cache"
            )
        if self.data_backend == "sequence_hdf5":
            print(
                f"[{self.split}] HDF5 query manifests are the authoritative sample "
                "index; legacy index_workers/index_manifest/rebuild_index are ignored.",
                flush=True,
            )
            if self.lazy_index_records:
                cache_label = self.hdf5_manifest_cache_dir or "<disabled>"
                print(
                    f"[{self.split}] Normalized HDF5 manifest mmap cache: {cache_label}",
                    flush=True,
                )
        self.hdf5_handle_cache_size = max(1, int(hdf5_handle_cache_size))
        self._hdf5_handle_pid = None
        self._hdf5_handles = OrderedDict()
        self._hdf5_validated_paths = set()
        self._pending_batched_loaded = None
        self._batched_hdf5_jpeg_cache = None
        self._resolved_hdf5_paths = {}
        self.query_manifest_sha256 = {
            spec["path"]: persistent_sha256_file(
                spec["path"],
                self.hdf5_manifest_cache_dir,
            )
            for spec in self.query_manifest_specs
        }
        self.hdf5_schema_versions = set()

        self.tactile_dim = count_obj_vertices(SUBDIV_OBJ_PATH)
        print(f"[{split}] Loading subdiv palm mask for evaluation and loss masking...")
        self.palm_mask = self._load_palm_mask()

        if sample_records is None:
            if self.data_backend == "sequence_hdf5":
                self.samples = self._load_hdf5_query_manifests()
            else:
                self.samples = self._load_or_build_index()
        else:
            self.samples = list(sample_records)
            if self.data_backend == "sequence_hdf5":
                self.samples = [
                    self._normalize_hdf5_manifest_record(
                        sample,
                        manifest_path=None,
                        root_hint=None,
                        dataset_hint=None,
                        line_number=index + 1,
                    )
                    for index, sample in enumerate(self.samples)
                ]
            elif self.bbox_source_policy != "any" and not all(
                sample.get("bbox_source_policy") == self.bbox_source_policy
                for sample in self.samples
            ):
                self.samples = self._filter_samples_by_bbox_source(self.samples)
        self._validate_dataset_filter()
        if self.depth_sidecar_root is not None:
            self._initialize_depth_sidecar_contract()
        source_counts = {}
        if isinstance(self.samples, MMapJsonlRecords):
            source_counts = {"mmap_records": len(self.samples)}
        else:
            for sample in self.samples:
                source_counts[sample["dataset"]] = source_counts.get(sample["dataset"], 0) + 1
        print(
            f"[{split}] Loaded {len(self.samples)} hand samples with "
            f"backend={self.data_backend} from {len(self.data_dirs)} root(s): {source_counts}"
        )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _sample_error(sample_ref, reason):
        raise RuntimeError(f"Invalid indexed tactile sample at {sample_ref}: {reason}")

    def _close_hdf5_handles(self):
        handles = getattr(self, "_hdf5_handles", None)
        if handles is not None:
            for handle in handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            handles.clear()
        self._hdf5_handle_pid = None

    def _close_depth_sidecars(self):
        readers = getattr(self, "_depth_sidecar_readers", None)
        if readers is not None:
            for reader in readers.values():
                try:
                    reader.close()
                except Exception:
                    pass
            readers.clear()
        self._depth_sidecar_pid = None

    def _initialize_depth_sidecar_contract(self):
        if not self.samples:
            raise RuntimeError(
                f"[{self.split}] Cannot validate depth sidecars for an empty dataset"
            )
        sample = self.samples[0]
        sequence_key = str(sample.get("sequence_key", "")).strip()
        if not sequence_key:
            raise RuntimeError(
                f"[{self.split}] Depth sidecars require sequence_key in every sample"
            )
        path = os.path.join(
            self.depth_sidecar_root,
            sequence_sidecar_filename(sequence_key),
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[{self.split}] Depth sidecar contract shard is missing: {path}"
            )
        reader = SequenceSidecarReader(path)
        try:
            config = reader.config
            manifest_hashes = set(self.query_manifest_sha256.values())
            if manifest_hashes and config.manifest_sha256 not in manifest_hashes:
                raise RuntimeError(
                    f"[{self.split}] Depth sidecar manifest SHA does not match the "
                    "authoritative query manifest: "
                    f"sidecar={config.manifest_sha256}, "
                    f"query_manifests={sorted(manifest_hashes)}"
                )
            self.depth_sidecar_contract = {
                "teacher_model": str(config.teacher_model),
                "model_sha256": str(config.model_sha256),
                "config_sha256": str(config.config_sha256),
                "semantic_config_sha256": str(config.semantic_config_sha256),
                "manifest_sha256": str(config.manifest_sha256),
                "teacher_input_hw": list(config.teacher_input_hw),
                "stored_grid_hw": list(config.stored_grid_hw),
                "teacher_bbox_scale": float(config.teacher_bbox_scale),
                "coordinate_convention": str(config.coordinate_convention),
                "extra": dict(config.extra),
            }
        finally:
            reader.close()
        print(
            f"[{self.split}] Locked depth sidecar contract: "
            f"model={self.depth_sidecar_contract['model_sha256'][:12]}, "
            f"config={self.depth_sidecar_contract['config_sha256'][:12]}, "
            f"manifest={self.depth_sidecar_contract['manifest_sha256'][:12]}",
            flush=True,
        )

    def _get_depth_sidecar_reader(self, sequence_key):
        if self.depth_sidecar_root is None:
            return None
        pid = os.getpid()
        if self._depth_sidecar_pid != pid:
            self._close_depth_sidecars()
            self._depth_sidecar_pid = pid
        key = str(sequence_key)
        reader = self._depth_sidecar_readers.pop(key, None)
        if reader is not None:
            self._depth_sidecar_readers[key] = reader
            return reader
        path = os.path.join(
            self.depth_sidecar_root,
            sequence_sidecar_filename(key),
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Depth sidecar missing for sequence {key!r}: {path}"
            )
        contract = self.depth_sidecar_contract
        reader = SequenceSidecarReader(
            path,
            expected_model_sha256=contract.get("model_sha256"),
            expected_config_sha256=contract.get("config_sha256"),
            expected_manifest_sha256=contract.get("manifest_sha256"),
        )
        if reader.sequence_key != key:
            reader.close()
            raise RuntimeError(
                f"Depth sidecar sequence mismatch: expected={key!r}, "
                f"actual={reader.sequence_key!r}, path={path}"
            )
        self._depth_sidecar_readers[key] = reader
        while len(self._depth_sidecar_readers) > self.hdf5_handle_cache_size:
            _, evicted = self._depth_sidecar_readers.popitem(last=False)
            evicted.close()
        return reader

    def __getstate__(self):
        self._close_hdf5_handles()
        self._close_depth_sidecars()
        state = self.__dict__.copy()
        state["_hdf5_handles"] = OrderedDict()
        state["_hdf5_handle_pid"] = None
        state["_hdf5_validated_paths"] = set()
        state["_pending_batched_loaded"] = None
        state["_batched_hdf5_jpeg_cache"] = None
        state["_depth_sidecar_readers"] = OrderedDict()
        state["_depth_sidecar_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._hdf5_handles = OrderedDict()
        self._hdf5_handle_pid = None
        self._hdf5_validated_paths = set()
        self._pending_batched_loaded = None
        self._batched_hdf5_jpeg_cache = None
        self._depth_sidecar_readers = OrderedDict()
        self._depth_sidecar_pid = None

    def __del__(self):
        try:
            self._close_hdf5_handles()
            self._close_depth_sidecars()
        except Exception:
            pass

    @staticmethod
    def _import_h5py():
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "data_backend='sequence_hdf5' requires h5py in every DataLoader worker"
            ) from exc
        return h5py

    def _get_hdf5_handle(self, path):
        pid = os.getpid()
        if self._hdf5_handle_pid != pid:
            self._close_hdf5_handles()
            self._hdf5_handle_pid = pid
        handle = self._hdf5_handles.pop(path, None)
        if handle is not None:
            try:
                if handle.id.valid:
                    self._hdf5_handles[path] = handle
                    return handle
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass

        h5py = self._import_h5py()
        try:
            opener = getattr(_hdf5_storage, "open_readonly", None)
            handle = opener(path) if callable(opener) else h5py.File(path, "r")
        except Exception as exc:
            raise RuntimeError(f"Could not open HDF5 container read-only: {path}: {exc}") from exc
        self._hdf5_handles[path] = handle
        while len(self._hdf5_handles) > self.hdf5_handle_cache_size:
            _, evicted = self._hdf5_handles.popitem(last=False)
            try:
                evicted.close()
            except Exception:
                pass
        return handle

    @staticmethod
    def _decode_hdf5_attr(value):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _hdf5_dataset_candidates(self, logical_name, defaults):
        candidates = []
        custom_paths = getattr(_hdf5_storage, "HDF5_DATASET_PATHS", None)
        if isinstance(custom_paths, dict):
            custom = custom_paths.get(logical_name)
            if isinstance(custom, str):
                candidates.append(custom)
            elif custom:
                candidates.extend(custom)
        candidates.extend(defaults)
        return tuple(dict.fromkeys(str(path).lstrip("/") for path in candidates))

    def _find_hdf5_dataset(self, handle, logical_name, defaults, required=True):
        candidates = self._hdf5_dataset_candidates(logical_name, defaults)
        for path in candidates:
            if path in handle:
                value = handle[path]
                if hasattr(value, "shape"):
                    return value
        if required:
            raise KeyError(
                f"HDF5 dataset {logical_name!r} is missing; expected one of {candidates}"
            )
        return None

    @staticmethod
    def _hdf5_hand_slot(record):
        if "hand_slot" in record:
            return int(record["hand_slot"])
        return int(record["is_right"])

    def _read_hdf5_query_array(
        self,
        handle,
        record,
        logical_name,
        candidates,
        required=True,
    ):
        dataset = self._find_hdf5_dataset(
            handle,
            logical_name,
            candidates,
            required=required,
        )
        if dataset is None:
            return None
        query_row = int(record["query_row"])
        frame_row = int(record["frame_row"])
        hand_slot = self._hdf5_hand_slot(record)
        try:
            if dataset.ndim == 1:
                return np.asarray(dataset[...])
            if (
                logical_name == "pressure"
                and dataset.ndim >= 3
                and int(dataset.shape[1]) == 2
            ):
                return np.asarray(dataset[frame_row, hand_slot])
            return np.asarray(dataset[query_row])
        except (IndexError, ValueError, TypeError) as exc:
            raise IndexError(
                f"Cannot read {logical_name} for query_row={query_row}, "
                f"frame_row={frame_row}, hand_slot={hand_slot}, shape={dataset.shape}"
            ) from exc
        raise ValueError(f"Unsupported {logical_name} dataset shape: {dataset.shape}")

    def _read_hdf5_jpeg(self, handle, record, timing=None):
        data = self._find_hdf5_dataset(
            handle,
            "jpeg_data",
            (
                "images/rgb/jpeg_data",
                "images/rgb/jpeg_bytes",
                "images/chest/jpeg_data",
                "images/chest/jpeg_bytes",
                "frames/rgb_jpeg/data",
            ),
        )
        offsets = self._find_hdf5_dataset(
            handle,
            "jpeg_offsets",
            (
                "images/rgb/jpeg_offsets",
                "images/rgb/offsets",
                "images/chest/jpeg_offsets",
                "images/chest/offsets",
                "frames/rgb_jpeg/offsets",
            ),
            required=False,
        )
        frame_row = int(record["frame_row"])
        read_started = time.perf_counter_ns() if timing is not None else 0
        try:
            if offsets is not None:
                if frame_row + 1 >= len(offsets):
                    raise IndexError(
                        f"frame_row={frame_row} exceeds JPEG offsets length={len(offsets)}"
                    )
                bounds = np.asarray(
                    offsets[frame_row : frame_row + 2],
                    dtype=np.uint64,
                )
                if bounds.shape != (2,):
                    raise IndexError(
                        f"frame_row={frame_row} could not read two JPEG offsets"
                    )
                start, end = (int(bounds[0]), int(bounds[1]))
                if start < 0 or end <= start or end > int(data.shape[0]):
                    raise ValueError(
                        f"invalid JPEG byte range [{start}, {end}) for data length={data.shape[0]}"
                    )
                encoded = np.asarray(data[start:end], dtype=np.uint8)
            else:
                encoded = np.asarray(data[frame_row], dtype=np.uint8).reshape(-1)
        except Exception as exc:
            raise RuntimeError(
                f"Could not read JPEG for frame_row={frame_row}: {exc}"
            ) from exc
        if timing is not None:
            timing["jpeg_hdf5_ms"] = (
                time.perf_counter_ns() - read_started
            ) / 1e6
        decode_started = time.perf_counter_ns() if timing is not None else 0
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(
                f"OpenCV could not decode HDF5 JPEG for frame_row={frame_row}"
            )
        if timing is not None:
            timing["jpeg_decode_ms"] = (
                time.perf_counter_ns() - decode_started
            ) / 1e6
        return image

    def _load_legacy_raw_sample(self, sample_record):
        sample_dir = sample_record["sample_dir"]
        meta_path = os.path.join(sample_dir, "meta.json")
        if not os.path.exists(meta_path):
            self._sample_error(sample_dir, "meta.json disappeared after index construction")
        meta = load_json_file(meta_path)

        dataset_name = sample_record.get("dataset", self._infer_dataset_name(meta))
        hand = sample_record.get("hand")
        is_right = int(sample_record.get("is_right", meta.get("is_right", 1)))
        if not bbox_source_allowed(
            meta,
            dataset_name,
            hand=hand,
            policy=self.bbox_source_policy,
        ):
            self._sample_error(
                sample_dir,
                f"bbox source no longer satisfies policy={self.bbox_source_policy}",
            )

        if dataset_name == "TouchAnything":
            hand_meta = meta.get("hands", {}).get(hand, {})
            image_name = meta.get("views", {}).get("chest", "chest.jpg")
            bbox = np.array(hand_meta["bbox_chest"], dtype=np.float32)
            pressure_data = hand_meta.get("gaussian_pressure")
            landmarks_cam = np.zeros((21, 3), dtype=np.float32)
            valid_mask = np.zeros(21, dtype=bool)
        else:
            image_name = meta.get("image", "image.jpg")
            bbox = np.array(meta["bbox"], dtype=np.float32)
            landmarks_cam = np.array(
                meta.get("keypoints_3d_cam", np.zeros((21, 3))),
                dtype=np.float32,
            )
            valid_mask = np.array(
                meta.get("valid_mask", np.zeros(21, dtype=bool)),
                dtype=bool,
            )
            side = "right" if is_right else "left"
            tactile_key = f"{side}_pressure_continuous_subdiv"
            pressure_data = meta.get("original_hdf5_data", {}).get(tactile_key)
            if pressure_data is None:
                pressure_data = meta.get("gaussian_pressure")

        img_path = os.path.join(sample_dir, image_name)
        if not os.path.exists(img_path):
            self._sample_error(
                sample_dir,
                f"image disappeared after index construction: {img_path}",
            )
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            self._sample_error(sample_dir, f"OpenCV could not decode image: {img_path}")
        return {
            "sample_ref": sample_dir,
            "sample_dir": sample_dir,
            "sample_uid": str(sample_record.get("sample_uid") or sample_dir),
            "h5_path": "",
            "frame_row": -1,
            "query_row": -1,
            "dataset_name": dataset_name,
            "hand": hand,
            "is_right": is_right,
            "bbox": bbox,
            "pressure_data": pressure_data,
            "landmarks_cam": landmarks_cam,
            "valid_mask": valid_mask,
            "img_bgr": img_bgr,
            "frame_idx": int(sample_record.get("frame_idx", meta.get("frame_idx", 0) or 0)),
        }

    def _load_hdf5_raw_sample(self, sample_record):
        sample_ref = sample_record["sample_ref"]
        h5_path = sample_record["h5_path"]
        timing = {} if self.io_debug_enabled else None
        raw_started = time.perf_counter_ns() if timing is not None else 0
        try:
            source_cached = h5_path in self._hdf5_handles
            handle_started = time.perf_counter_ns() if timing is not None else 0
            handle = self._get_hdf5_handle(h5_path)
            if timing is not None:
                timing["source_handle_ms"] = (
                    time.perf_counter_ns() - handle_started
                ) / 1e6
                timing["source_handle_hit"] = float(source_cached)
            if h5_path not in self._hdf5_validated_paths:
                expected_schema = str(
                    sample_record.get("_expected_hdf5_schema_version", "")
                )
                actual_schema = self._decode_hdf5_attr(
                    handle.attrs.get(
                        "schema_version",
                        handle.attrs.get("schema_name", expected_schema),
                    )
                )
                if expected_schema and actual_schema and str(actual_schema) != expected_schema:
                    self._sample_error(
                        sample_ref,
                        f"HDF5 schema mismatch: manifest={expected_schema!r}, "
                        f"container={actual_schema!r}",
                    )
                actual_dataset = self._decode_hdf5_attr(
                    handle.attrs.get("dataset", "")
                )
                if (
                    actual_dataset
                    and canonical_dataset_name(actual_dataset)
                    != sample_record["dataset"]
                ):
                    self._sample_error(
                        sample_ref,
                        f"HDF5 dataset mismatch: manifest={sample_record['dataset']!r}, "
                        f"container={actual_dataset!r}",
                    )
                actual_split = self._decode_hdf5_attr(
                    handle.attrs.get("split", "")
                )
                if actual_split and str(actual_split) != self.split:
                    self._sample_error(
                        sample_ref,
                        f"HDF5 split mismatch: requested={self.split!r}, "
                        f"container={actual_split!r}",
                    )
                self._hdf5_validated_paths.add(h5_path)
            jpeg_cache = getattr(self, "_batched_hdf5_jpeg_cache", None)
            jpeg_key = (h5_path, int(sample_record["frame_row"]))
            img_bgr = jpeg_cache.get(jpeg_key) if jpeg_cache is not None else None
            if img_bgr is None:
                img_bgr = self._read_hdf5_jpeg(handle, sample_record, timing=timing)
                if jpeg_cache is not None:
                    jpeg_cache[jpeg_key] = img_bgr
            elif timing is not None:
                timing["jpeg_hdf5_ms"] = 0.0
                timing["jpeg_decode_ms"] = 0.0
            pressure_started = time.perf_counter_ns() if timing is not None else 0
            pressure_data = self._read_hdf5_query_array(
                handle,
                sample_record,
                "pressure",
                (
                    "targets/pressure",
                    "queries/pressure/gaussian_subdiv",
                    "tactile/pressure",
                ),
            )
            if timing is not None:
                timing["pressure_hdf5_ms"] = (
                    time.perf_counter_ns() - pressure_started
                ) / 1e6

            bbox_value = sample_record.get("bbox_xyxy", sample_record.get("bbox"))
            if bbox_value is None:
                bbox_value = self._read_hdf5_query_array(
                    handle,
                    sample_record,
                    "bbox",
                    (
                        "queries/bbox_xyxy",
                        "queries/bbox",
                        "queries/bbox_chest",
                    ),
                )
            bbox = np.asarray(bbox_value, dtype=np.float32)

            if self.tactile_only:
                landmarks_cam = np.zeros((21, 3), dtype=np.float32)
                valid_mask = np.zeros(21, dtype=bool)
            else:
                landmarks_cam = self._read_hdf5_query_array(
                    handle,
                    sample_record,
                    "keypoints_3d_cam",
                    (
                        "queries/keypoints_3d_cam",
                        "queries/keypoints_3d",
                    ),
                    required=False,
                )
                valid_mask = self._read_hdf5_query_array(
                    handle,
                    sample_record,
                    "keypoints_valid",
                    (
                        "queries/keypoints_valid",
                        "queries/valid_mask",
                    ),
                    required=False,
                )
                if landmarks_cam is None:
                    landmarks_cam = np.zeros((21, 3), dtype=np.float32)
                else:
                    landmarks_cam = np.asarray(landmarks_cam, dtype=np.float32)
                    if landmarks_cam.shape != (21, 3):
                        raise ValueError(
                            f"keypoints_3d_cam must have shape (21, 3), got "
                            f"{landmarks_cam.shape}"
                        )
                if valid_mask is None:
                    valid_mask = np.zeros(21, dtype=bool)
                else:
                    valid_mask = np.asarray(valid_mask, dtype=bool)
                    if valid_mask.shape != (21,):
                        raise ValueError(
                            f"keypoints_valid must have shape (21,), got "
                            f"{valid_mask.shape}"
                        )
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith(
                "Invalid indexed tactile sample"
            ):
                raise
            self._sample_error(sample_ref, f"HDF5 read failed: {exc}")

        if timing is not None:
            timing["source_raw_ms"] = (
                time.perf_counter_ns() - raw_started
            ) / 1e6

        return {
            "sample_ref": sample_ref,
            "sample_dir": sample_ref,
            "sample_uid": sample_record["sample_uid"],
            "h5_path": h5_path,
            "frame_row": int(sample_record["frame_row"]),
            "query_row": int(sample_record["query_row"]),
            "dataset_name": sample_record["dataset"],
            "hand": sample_record["hand"],
            "is_right": int(sample_record["is_right"]),
            "bbox": bbox,
            "pressure_data": pressure_data,
            "landmarks_cam": landmarks_cam,
            "valid_mask": valid_mask,
            "img_bgr": img_bgr,
            "frame_idx": int(sample_record.get("frame_idx", sample_record["frame_row"])),
            "_runtime_io_debug": timing,
        }

    def __getitems__(self, indices):
        """Read one auto-collated batch in grouped or memory-bounded mode.

        Grouped mode reads raw records by container before item construction.
        Streaming mode retains only one decoded source frame at a time. Both
        preserve batch membership, collation order, and NumPy augmentation RNG.
        """
        normalized_indices = [int(index) for index in indices]
        if self.data_backend != "sequence_hdf5" or len(normalized_indices) <= 1:
            return [self[index] for index in normalized_indices]
        if self.hdf5_batch_read_mode == "streaming":
            # Avoid retaining an entire batch of decoded full-resolution images.
            # Item order and NumPy augmentation RNG order remain unchanged.
            return [self[index] for index in normalized_indices]

        grouped_positions = OrderedDict()
        for position, index in enumerate(normalized_indices):
            sample_record = self.samples[index]
            grouped_positions.setdefault(sample_record["h5_path"], []).append(
                (position, index, sample_record)
            )

        loaded_by_position = [None] * len(normalized_indices)
        self._batched_hdf5_jpeg_cache = {}
        try:
            for entries in grouped_positions.values():
                for position, _, sample_record in entries:
                    loaded_by_position[position] = self._load_hdf5_raw_sample(
                        sample_record
                    )

            items = []
            for position, index in enumerate(normalized_indices):
                self._pending_batched_loaded = (
                    index,
                    loaded_by_position[position],
                )
                items.append(self[index])
            return items
        finally:
            self._pending_batched_loaded = None
            self._batched_hdf5_jpeg_cache = None

    def __getitem__(self, idx):
        idx = int(idx)
        sample_record = self.samples[idx]
        pending = getattr(self, "_pending_batched_loaded", None)
        if pending is not None and int(pending[0]) == idx:
            loaded = pending[1]
            self._pending_batched_loaded = None
        else:
            loaded = (
                self._load_hdf5_raw_sample(sample_record)
                if self.data_backend == "sequence_hdf5"
                else self._load_legacy_raw_sample(sample_record)
            )
        sample_ref = loaded["sample_ref"]
        sample_dir = loaded["sample_dir"]
        dataset_name = loaded["dataset_name"]
        hand = loaded["hand"]
        is_right = loaded["is_right"]
        bbox = loaded["bbox"]
        pressure_data = loaded["pressure_data"]
        landmarks_cam = loaded["landmarks_cam"]
        valid_mask = loaded["valid_mask"]
        img_bgr = loaded["img_bgr"]
        timing = loaded.get("_runtime_io_debug")
        transform_started = time.perf_counter_ns() if timing is not None else 0
        
        # Extract tactile pressure signal on the subdiv MANO mesh.
        tactile_signal = np.zeros(self.tactile_dim, dtype=np.float32)
        has_tactile = 0.0
        if pressure_data is not None:
            raw_signal = np.array(pressure_data, dtype=np.float32)
            if raw_signal.shape == (self.tactile_dim,) and np.isfinite(raw_signal).all():
                tactile_signal = np.clip(raw_signal, 0.0, 1.0)
                has_tactile = 1.0
            else:
                self._sample_error(
                    sample_dir,
                    "tactile signal must be finite with shape "
                    f"({self.tactile_dim},), got shape={raw_signal.shape}",
                )
        else:
            self._sample_error(sample_dir, "pressure target disappeared after index construction")

        if not self.tactile_only:
            keypoints_3d = np.zeros((21, 4), dtype=np.float32)
            keypoints_3d[valid_mask, :3] = landmarks_cam[valid_mask]
            keypoints_3d[valid_mask, 3] = 1.0
            keypoints_2d = np.zeros((21, 3), dtype=np.float32)
            num_pose = 3 * (self.cfg.MANO.NUM_HAND_JOINTS + 1)
            mano_params = {
                'global_orient': np.zeros(3, dtype=np.float32),
                'hand_pose': np.zeros(num_pose - 3, dtype=np.float32),
                'betas': np.zeros(10, dtype=np.float32)
            }
            has_mano_params = {k: 0.0 for k in mano_params.keys()}
            mano_params_is_axis_angle = {'global_orient': True, 'hand_pose': True, 'betas': False}

        # Calculate bounding box parameters.
        if np.isnan(bbox).any() or len(bbox) < 4:
            self._sample_error(sample_dir, "bbox is missing or non-finite")
            
        center = (bbox[2:4] + bbox[0:2]) / 2.0
        center_x, center_y = center[0], center[1]
        
        scale_pixels = np.max(bbox[2:4] - bbox[0:2])
        if not valid_bbox(bbox) or np.isnan(scale_pixels) or scale_pixels <= 1.0:
            self._sample_error(sample_dir, f"bbox is invalid: {bbox.tolist()}")
            
        bbox_size = self.rescale_factor * scale_pixels
        
        # Add basic augmentation during training
        if self.train and self.augmentation_enabled:
            augm_config = self.cfg.DATASETS.CONFIG
            scale_aug = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.SCALE_FACTOR + 1.0
            tx = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            ty = np.clip(np.random.randn(), -1.0, 1.0) * augm_config.TRANS_FACTOR * bbox_size
            
            bbox_size = bbox_size * scale_aug
            center_x += tx
            center_y += ty
            
        # Crop and resize image using affine transform
        output_height, output_width = self.input_resolution
        res = output_height
        t = np.zeros((2, 3), dtype=np.float32)
        t[0, 0] = float(res) / bbox_size
        t[1, 1] = float(res) / bbox_size
        t[0, 2] = -res * float(center_x) / bbox_size + output_width * 0.5
        t[1, 2] = res * (-float(center_y) / bbox_size + 0.5)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_patch = cv2.warpAffine(
            img_rgb,
            t,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
        )
        
        # Normalize and convert to CHW
        img_patch = img_patch.astype(np.float32) / 255.0
        
        if is_right == 0:
            # Flip left hands to right hands for Hamer
            img_patch = cv2.flip(img_patch, 1)
            if not self.tactile_only:
                keypoints_3d[:, 0] = -keypoints_3d[:, 0]
            # Continuous pressure is already generated on the canonical MANO topology.
            
        # Standard mean/std normalization
        img_patch = (img_patch - self.cfg.MODEL.IMAGE_MEAN) / self.cfg.MODEL.IMAGE_STD
        img_patch = img_patch.transpose(2, 0, 1)
        
        item = {
            'dataset': dataset_name,
            'sample_dir': sample_dir,
            'sample_uid': str(loaded['sample_uid']),
            'sample_ref': sample_ref,
            'h5_path': str(loaded['h5_path']),
            'frame_row': torch.tensor(int(loaded['frame_row'])),
            'query_row': torch.tensor(int(loaded['query_row'])),
            'hand': str(hand),
            'sequence_key': str(sample_record.get('sequence_key', '')),
            'query_alias': str(sample_record.get('query_alias', hand or 'query')),
            'frame_idx': torch.tensor(int(loaded['frame_idx'])),
            'bbox_score': torch.tensor(float(sample_record.get('bbox_score', 0.0))).float(),
            'pressure_source_key': str(sample_record.get('pressure_source_key', '')),
            'query_bbox': torch.from_numpy(bbox.copy()).float(),
            'image_width': torch.tensor(int(img_bgr.shape[1])),
            'image_height': torch.tensor(int(img_bgr.shape[0])),
            'img': torch.from_numpy(img_patch).float(),
            'tactile_signal': torch.from_numpy(tactile_signal).float(),
            'has_tactile': torch.tensor(has_tactile).float(),
            'palm_mask': torch.from_numpy(self.palm_mask).float(),
            'right': torch.tensor(float(is_right)).float(),
        }
        if timing is not None:
            timing["rgb_transform_ms"] = (
                time.perf_counter_ns() - transform_started
            ) / 1e6
        if self.depth_sidecar_root is not None:
            sequence_key = str(sample_record.get('sequence_key', ''))
            if not sequence_key:
                self._sample_error(sample_ref, "depth sidecar requires sequence_key")
            depth_cached = sequence_key in self._depth_sidecar_readers
            depth_handle_started = time.perf_counter_ns() if timing is not None else 0
            reader = self._get_depth_sidecar_reader(sequence_key)
            if timing is not None:
                timing["depth_handle_ms"] = (
                    time.perf_counter_ns() - depth_handle_started
                ) / 1e6
                timing["depth_handle_hit"] = float(depth_cached)
            try:
                depth_read_started = time.perf_counter_ns() if timing is not None else 0
                geometry_record = reader.read(
                    sample_uid=str(loaded['sample_uid']),
                    query_row=int(loaded['query_row']),
                )
                if timing is not None:
                    timing["depth_hdf5_ms"] = (
                        time.perf_counter_ns() - depth_read_started
                    ) / 1e6
                depth_warp_started = time.perf_counter_ns() if timing is not None else 0
                depth_prior = warp_record_pointnormal(
                    geometry_record,
                    rgb_affine=t,
                    rgb_output_hw=self.input_resolution,
                    output_hw=self.depth_output_hw,
                    flip_left_to_right=(is_right == 0),
                    spatial_shuffle_seed=(
                        521
                        if self.depth_control == "sample_spatial_shuffle"
                        else None
                    ),
                )
                if timing is not None:
                    timing["depth_warp_ms"] = (
                        time.perf_counter_ns() - depth_warp_started
                    ) / 1e6
            except Exception as exc:
                self._sample_error(
                    sample_ref,
                    f"depth sidecar read/warp failed: {exc}",
                )
            if not np.isfinite(depth_prior).all():
                self._sample_error(
                    sample_ref,
                    "depth sidecar warp produced non-finite values",
                )
            item['depth_prior'] = torch.from_numpy(depth_prior).float()
            item['crop_affine'] = torch.from_numpy(t.copy()).float()
        if timing is not None:
            timing.setdefault("depth_handle_ms", 0.0)
            timing.setdefault("depth_handle_hit", 0.0)
            timing.setdefault("depth_hdf5_ms", 0.0)
            timing.setdefault("depth_warp_ms", 0.0)
            timing["getitem_total_ms"] = sum(
                float(timing.get(name, 0.0))
                for name in ("source_raw_ms", "rgb_transform_ms", "depth_handle_ms", "depth_hdf5_ms", "depth_warp_ms")
            )
            worker = torch.utils.data.get_worker_info()
            timing["worker_id"] = -1 if worker is None else int(worker.id)
            timing["worker_pid"] = int(os.getpid())
            item['_runtime_io_debug'] = {
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in timing.items()
            }
        if not self.tactile_only:
            img_size_array = np.array([img_bgr.shape[1], img_bgr.shape[0]])
            item.update({
                'keypoints_3d': torch.from_numpy(keypoints_3d).float(),
                'keypoints_2d': torch.from_numpy(keypoints_2d).float(),
                'box_center': torch.tensor([center_x, center_y]).float(),
                'box_size': torch.tensor(bbox_size).float(),
                'img_size': torch.from_numpy(img_size_array).float(),
                'mano_params': {k: torch.from_numpy(v).float() for k, v in mano_params.items()},
                'has_mano_params': {k: torch.tensor(float(v)).float() for k, v in has_mano_params.items()},
                'mano_params_is_axis_angle': {k: torch.tensor(v).bool() for k, v in mano_params_is_axis_angle.items()},
            })
        return item
