#!/usr/bin/env python3
"""Export one audited geodesic surface basis as a training runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_canonical_localization import (
    DEFAULT_MESH,
    DEFAULT_PALM_FACES,
    _load_mesh_and_palm_graph,
    _surface_basis_banks,
    _write_json,
)
from tactile_input_priors.feature_cache import sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _resolve_asset(recorded: Any, fallback: Path) -> Path:
    candidate = Path(str(recorded or "")).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    return fallback.resolve(strict=True)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def _sparse_basis_sha256(
    support_indices: torch.Tensor, support_weights: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    for name, value in (
        ("indices", support_indices.detach().cpu().long().contiguous()),
        ("weights", support_weights.detach().cpu().float().contiguous()),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _obj_vertex_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                count += 1
    if count <= 0:
        raise RuntimeError(f"Mesh contains no vertices: {path}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--coefficient-dim", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_dir = Path(args.audit_dir).expanduser().resolve(strict=True)
    output_path = Path(args.output).expanduser().resolve()
    coefficient_dim = int(args.coefficient_dim)
    if coefficient_dim <= 0:
        raise ValueError("--coefficient-dim must be positive")
    if output_path.is_file() and not args.force:
        print(f"Surface basis runtime artifact already exists: {output_path}")
        return

    summary_path = audit_dir / "summary.json"
    source_npz_path = audit_dir / "canonical_surface_basis_cleanup.npz"
    summary = _read_json(summary_path)
    run_config = dict(summary.get("run_config", {}))
    if run_config.get("basis_bandwidth_policy") != "target_overlap":
        raise RuntimeError("Runtime export requires a target_overlap audit")
    target_support = int(run_config.get("basis_target_support_count", -1))
    if target_support != 4:
        raise RuntimeError(
            f"Stage 1 requires target support 4, got {target_support}"
        )
    audited_counts = tuple(int(value) for value in run_config.get("anchor_counts", ()))
    if coefficient_dim not in audited_counts:
        raise ValueError(
            f"Coefficient dimension {coefficient_dim} was not audited: {audited_counts}"
        )

    with np.load(source_npz_path) as payload:
        valid_indices = np.asarray(payload["valid_vertex_indices"], dtype=np.int64)
        anchor_prefix = np.asarray(
            payload["anchor_prefix_valid_indices"], dtype=np.int64
        )
        audited_bandwidth = float(payload[f"bandwidth_{coefficient_dim}"])
    mesh_path = _resolve_asset(run_config.get("mesh"), DEFAULT_MESH)
    tactile_dim = int(run_config.get("tactile_dim") or _obj_vertex_count(mesh_path))
    if tactile_dim <= int(valid_indices.max()):
        raise RuntimeError("Recorded tactile_dim does not contain every valid vertex")
    valid_mask = np.zeros(tactile_dim, dtype=bool)
    valid_mask[valid_indices] = True
    palm_faces_path = _resolve_asset(
        run_config.get("palm_faces"), DEFAULT_PALM_FACES
    )
    coordinates, loaded_valid, adjacency = _load_mesh_and_palm_graph(
        mesh_path, palm_faces_path, valid_mask
    )
    if not np.array_equal(valid_indices, loaded_valid):
        raise RuntimeError("Runtime valid vertices differ from the audited basis")

    diagnostics: dict[int, dict[str, Any]] = {}
    banks, bandwidths = _surface_basis_banks(
        coordinates[valid_indices],
        adjacency,
        anchor_prefix,
        (coefficient_dim,),
        bandwidth_scale=float(run_config.get("basis_bandwidth_scale", 1.0)),
        support_sigma=float(run_config.get("basis_support_sigma", 3.0)),
        bandwidth_policy="target_overlap",
        target_support_count=target_support,
        diagnostics=diagnostics,
    )
    actual_bandwidth = float(bandwidths[coefficient_dim])
    if not np.isclose(actual_bandwidth, audited_bandwidth, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Rebuilt basis bandwidth differs from the audit: "
            f"actual={actual_bandwidth}, audited={audited_bandwidth}"
        )
    expected_diagnostics = next(
        (
            dict(row)
            for row in summary.get("bank_diagnostics", ())
            if int(row.get("anchor_count", -1)) == coefficient_dim
        ),
        None,
    )
    if expected_diagnostics is None:
        raise RuntimeError(f"Audit lacks diagnostics for {coefficient_dim}")
    actual_diagnostics = diagnostics[coefficient_dim]
    for key in (
        "fallback_vertex_count",
        "support_count_min",
        "support_count_median",
        "support_count_p90",
    ):
        if not np.isclose(
            float(actual_diagnostics[key]),
            float(expected_diagnostics[key]),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                f"Rebuilt basis diagnostic {key} differs: "
                f"actual={actual_diagnostics[key]}, audited={expected_diagnostics[key]}"
            )

    basis = torch.from_numpy(banks[coefficient_dim]).contiguous()
    if basis.shape != (len(valid_indices), coefficient_dim):
        raise RuntimeError(f"Unexpected runtime basis shape {tuple(basis.shape)}")
    basis_sha256 = _tensor_sha256(basis)
    support_counts = (basis > 0.0).sum(dim=1)
    maximum_support = int(support_counts.max().item())
    support_indices = torch.zeros(
        (basis.shape[0], maximum_support), dtype=torch.long
    )
    support_weights = torch.zeros(
        (basis.shape[0], maximum_support), dtype=torch.float32
    )
    for row_index in range(basis.shape[0]):
        columns = torch.nonzero(
            basis[row_index] > 0.0, as_tuple=False
        ).flatten()
        count = int(columns.numel())
        support_indices[row_index, :count] = columns
        support_weights[row_index, :count] = basis[row_index, columns]
    sparse_sha256 = _sparse_basis_sha256(
        support_indices, support_weights
    )
    metadata = {
        "schema_version": 1,
        "basis_method": "weighted_geodesic_rbf_target_overlap_v1",
        "coefficient_dim": coefficient_dim,
        "target_support_count": target_support,
        "valid_vertex_count": len(valid_indices),
        "tactile_dim": tactile_dim,
        "basis_shape": list(basis.shape),
        "basis_dtype": str(basis.dtype),
        "basis_sha256": basis_sha256,
        "sparse_basis_sha256": sparse_sha256,
        "maximum_support_count": maximum_support,
        "nonzero_basis_count": int(support_counts.sum().item()),
        "bandwidth": actual_bandwidth,
        "bank_diagnostics": actual_diagnostics,
        "audit_dir": str(audit_dir),
        "audit_summary_sha256": sha256_file(summary_path),
        "audit_basis_config_sha256": sha256_file(source_npz_path),
        "mesh": str(mesh_path),
        "mesh_sha256": sha256_file(mesh_path),
        "palm_faces": str(palm_faces_path),
        "palm_faces_sha256": sha256_file(palm_faces_path),
    }
    artifact = {
        "format": "canonical_surface_basis_v1",
        "support_indices": support_indices,
        "support_weights": support_weights,
        "valid_vertex_indices": torch.from_numpy(valid_indices.copy()),
        "anchor_prefix_valid_indices": torch.from_numpy(
            anchor_prefix[:coefficient_dim].copy()
        ),
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    torch.save(artifact, temporary)
    os.replace(temporary, output_path)
    metadata["artifact"] = str(output_path)
    metadata["artifact_sha256"] = sha256_file(output_path)
    _write_json(output_path.with_suffix(output_path.suffix + ".json"), metadata)
    print(f"Surface basis runtime artifact complete: {output_path}")
    print(
        f"shape={tuple(basis.shape)} support={target_support} "
        f"basis_sha256={basis_sha256}"
    )


if __name__ == "__main__":
    main()
