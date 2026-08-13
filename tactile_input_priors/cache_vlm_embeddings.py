#!/usr/bin/env python3
"""Build resumable full-frame Qwen3-VL embedding caches keyed by sample_uid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tactile_input_priors.feature_cache import (  # noqa: E402
    FeatureCacheBuilder,
    FeatureSpec,
    atomic_write_json,
    canonical_json,
    iter_jsonl,
    sha256_file,
)
from tactile_input_priors.hdf5_manifest import HDF5ImageReader  # noqa: E402
from tactile_input_priors.resolve_depth_manifests import (  # noqa: E402
    _manifest_paths,
    _resolve_root,
)


DEFAULT_INSTRUCTION = (
    "Represent the full scene for estimating the contact state between the "
    "queried hand and nearby objects."
)


def _directory_fingerprint(path: Path) -> str:
    """Fingerprint a local model tree without hashing multi-GB weights repeatedly."""

    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        if child.name in {
            "config.json",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
        }:
            digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _absolute_h5_path(row: Mapping[str, Any], processed_root: Path) -> Path:
    raw = str(row.get("h5_path", "")).strip()
    if raw:
        path = Path(raw).expanduser()
    else:
        relative = str(row.get("h5_relpath", "")).strip()
        if not relative:
            raise KeyError(f"sample_uid={row.get('sample_uid')!r} has no HDF5 path")
        path = processed_root / relative
    return path.resolve(strict=True)


def _write_partition_manifest(
    source_manifest: Path,
    processed_root: Path,
    output_path: Path,
    *,
    partition_index: int,
    num_partitions: int,
) -> list[dict[str, Any]]:
    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.partial-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for global_index, (_, original) in enumerate(iter_jsonl(source_manifest)):
                if global_index % num_partitions != partition_index:
                    continue
                row = dict(original)
                row["h5_path"] = str(_absolute_h5_path(row, processed_root))
                row["metadata"] = {
                    "dataset": str(row.get("dataset", "")),
                    "sequence_key": str(row.get("sequence_key", "")),
                    "query_alias": str(row.get("query_alias", "query")),
                    "frame_idx": int(row.get("frame_idx", 0)),
                    "sample_ref": str(row.get("source_sample_relpath", "")),
                }
                handle.write(canonical_json(row) + "\n")
                rows.append(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    if not rows:
        raise RuntimeError(
            f"VLM cache partition {partition_index}/{num_partitions} is empty"
        )
    return rows


class QwenEmbeddingCallback:
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        model_path: Path,
        code_root: Path,
        instruction: str,
        batch_size: int,
        embedding_dim: int,
        max_pixels: int,
        max_handles: int,
        attn_implementation: str,
    ):
        if str(code_root) not in sys.path:
            sys.path.insert(0, str(code_root))
        try:
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError as exc:
            raise ImportError(
                f"Could not import Qwen3VLEmbedder from {code_root}. "
                "Set QWEN_EMBED_CODE_ROOT to the official Qwen3-VL-Embedding checkout."
            ) from exc

        kwargs = {
            "model_name_or_path": str(model_path),
            "max_pixels": int(max_pixels),
            "torch_dtype": torch.bfloat16,
            "attn_implementation": str(attn_implementation),
        }
        self.embedder = Qwen3VLEmbedder(**kwargs)
        self.rows = tuple(rows)
        self.reader = HDF5ImageReader(max_handles=max_handles)
        self.instruction = str(instruction)
        self.batch_size = max(1, int(batch_size))
        self.embedding_dim = int(embedding_dim)
        self.start = -1
        self.stop = -1
        self.values = np.empty((0, self.embedding_dim), dtype=np.float16)

    def _fill(self, source_index: int) -> None:
        self.start = int(source_index)
        self.stop = min(self.start + self.batch_size, len(self.rows))
        rows = self.rows[self.start : self.stop]

        # A frame can contain multiple anonymous hand queries. Encode its full
        # image once, then replicate the context vector for those query UIDs.
        unique_inputs = []
        key_to_position = {}
        row_positions = []
        for row in rows:
            key = (str(row["h5_path"]), int(row["frame_row"]))
            position = key_to_position.get(key)
            if position is None:
                bgr = self.reader.read_bgr(row)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                position = len(unique_inputs)
                key_to_position[key] = position
                unique_inputs.append(
                    {"image": image, "instruction": self.instruction}
                )
            row_positions.append(position)

        with torch.inference_mode():
            encoded = self.embedder.process(unique_inputs)
        if isinstance(encoded, torch.Tensor):
            array = encoded.detach().float().cpu().numpy()
        else:
            array = np.asarray(encoded, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.embedding_dim:
            raise RuntimeError(
                f"Qwen embedding shape mismatch: got {array.shape}, expected "
                f"[N,{self.embedding_dim}]"
            )
        if not np.isfinite(array).all():
            raise FloatingPointError("Qwen returned a non-finite embedding")
        self.values = np.asarray(array[row_positions], dtype=np.float16)

    def __call__(self, row: Mapping[str, Any], source_index: int):
        del row
        if not self.start <= source_index < self.stop:
            self._fill(source_index)
        return {"vlm_embedding": self.values[source_index - self.start]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--qwen-code-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--dataset", choices=("touchanything", "opentouch"), default="touchanything")
    parser.add_argument("--split", required=True)
    parser.add_argument("--processed-root", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--embedding-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)
    parser.add_argument("--max-hdf5-handles", type=int, default=4)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--max-new-shards", type=int)
    parser.add_argument("--repair-invalid-shards", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.partition_index < args.num_partitions:
        raise ValueError("partition-index must lie in [0,num-partitions)")
    model_path = Path(args.model).expanduser().resolve(strict=True)
    code_root = Path(args.qwen_code_root).expanduser().resolve(strict=True)
    processed_root = _resolve_root(args.dataset, args.processed_root)
    if args.manifest:
        source_manifest = Path(args.manifest).expanduser().resolve(strict=True)
    else:
        source_manifest = _manifest_paths(
            processed_root, args.dataset, args.split
        )[0].resolve(strict=True)

    cache_root = Path(args.cache_dir).expanduser().resolve(strict=False)
    cache_dir = (
        cache_root / f"part-{args.partition_index:02d}-of-{args.num_partitions:02d}"
        if args.num_partitions > 1
        else cache_root
    )
    partition_manifest = cache_dir.parent / "source_manifests" / (
        f"{args.split}.part-{args.partition_index:02d}-of-"
        f"{args.num_partitions:02d}.jsonl"
    )
    rows = _write_partition_manifest(
        source_manifest,
        processed_root,
        partition_manifest,
        partition_index=args.partition_index,
        num_partitions=args.num_partitions,
    )
    provenance = {
        "producer": "tactile_input_priors.cache_vlm_embeddings",
        "backend": "qwen3_vl_embedding",
        "model_path": str(model_path),
        "model_fingerprint": _directory_fingerprint(model_path),
        "official_code_root": str(code_root),
        "source_manifest_sha256": sha256_file(source_manifest),
        "dataset": args.dataset,
        "split": args.split,
        "context": "full_frame",
        "instruction": args.instruction,
        "embedding_dim": args.embedding_dim,
        "max_pixels": args.max_pixels,
        "partition_index": args.partition_index,
        "num_partitions": args.num_partitions,
    }
    atomic_write_json(cache_dir / "cache_request.json", vars(args))
    callback = QwenEmbeddingCallback(
        rows,
        model_path=model_path,
        code_root=code_root,
        instruction=args.instruction,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        max_pixels=args.max_pixels,
        max_handles=args.max_hdf5_handles,
        attn_implementation=args.attn_implementation,
    )
    builder = FeatureCacheBuilder(
        cache_dir,
        partition_manifest,
        (FeatureSpec("vlm_embedding", (args.embedding_dim,), "float16"),),
        provenance=provenance,
        shard_size=args.shard_size,
        sample_id_key="sample_uid",
        repair_invalid_shards=args.repair_invalid_shards,
    )
    result = builder.build(callback, max_new_shards=args.max_new_shards)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
