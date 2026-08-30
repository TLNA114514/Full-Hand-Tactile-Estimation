#!/usr/bin/env python3
"""Aggregate the Stage 0.4b target-overlap surface-basis sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_canonical_localization import _write_csv, _write_json
from hamer_tactile_ft.audit_surface_basis_cleanup import _read_json, _tradeoff_rows
from tactile_input_priors.feature_cache import sha256_file


BASE_VARIANT = "base_basis_sampled_all"
BASE_METRICS = (
    "contact_iou_010_frame_macro",
    "volumetric_iou_frame_macro",
    "distribution_viou_frame_macro",
    "core_distribution_viou_frame_macro",
    "rmse_vertex_micro",
    "false_high_excess_mean",
    "gt_ge_070_mean_prediction",
)


def _base_row(summary: Mapping[str, Any], source: Path) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in summary.get("reconstruction", [])
        if row.get("variant") == BASE_VARIANT
    ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {BASE_VARIANT!r} row in {source}")
    return rows[0]


def _assert_same_base(
    expected: Mapping[str, Any], current: Mapping[str, Any], source: Path
) -> None:
    for key in BASE_METRICS:
        left = float(expected[key])
        right = float(current[key])
        if abs(left - right) > 1e-10 * max(1.0, abs(left), abs(right)):
            raise RuntimeError(
                f"Base metric {key!r} differs in {source}: {right} vs {left}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).expanduser().resolve(strict=True)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    child_dirs = sorted(
        path
        for path in input_root.glob("support_*")
        if path.is_dir() and (path / "AUDIT_DONE.json").is_file()
    )
    if not child_dirs:
        raise RuntimeError(f"No completed support_* audits under {input_root}")

    expected_contract: tuple[Any, ...] | None = None
    base_row: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    for child_dir in child_dirs:
        summary_path = child_dir / "summary.json"
        summary = _read_json(summary_path)
        config = dict(summary.get("run_config", {}))
        contract = (
            config.get("stage1_summary_sha256"),
            config.get("stage1_samples_sha256"),
            int(config.get("sample_count", -1)),
            int(config.get("valid_vertex_count", -1)),
            config.get("mesh_sha256"),
            config.get("palm_faces_sha256"),
        )
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise RuntimeError(
                f"Sweep child contract differs in {child_dir}: {contract} vs "
                f"{expected_contract}"
            )
        current_base = _base_row(summary, summary_path)
        if base_row is None:
            base_row = current_base
        else:
            _assert_same_base(base_row, current_base, summary_path)

        support = int(config["basis_target_support_count"])
        if config.get("basis_bandwidth_policy") != "target_overlap":
            raise RuntimeError(f"Child is not a target-overlap audit: {child_dir}")
        child_candidates = [
            dict(row)
            for row in summary.get("tradeoff", [])
            if str(row.get("basis_family", "")).startswith("weighted_target_")
        ]
        for row in child_candidates:
            row["sweep_target_support_count"] = support
            row["sweep_child"] = child_dir.name
        candidates.extend(child_candidates)
        for row in summary.get("bank_diagnostics", []):
            bank_rows.append(
                {
                    **dict(row),
                    "sweep_target_support_count": support,
                    "sweep_child": child_dir.name,
                }
            )
        for row in summary.get("matrix_diagnostics", []):
            matrix_rows.append(
                {
                    **dict(row),
                    "sweep_target_support_count": support,
                    "sweep_child": child_dir.name,
                }
            )
        children.append(
            {
                "target_support_count": support,
                "directory": str(child_dir),
                "summary_sha256": sha256_file(summary_path),
                "candidate_count": len(child_candidates),
            }
        )

    if base_row is None or expected_contract is None:
        raise AssertionError("Completed sweep unexpectedly contained no base row")
    if not candidates:
        raise RuntimeError("Completed sweep contained no target-overlap candidates")
    tradeoff_rows, decision = _tradeoff_rows(candidates, base_row)
    tradeoff_rows.sort(
        key=lambda row: (
            int(row["coefficient_dimension"]),
            int(row["sweep_target_support_count"]),
            str(row["variant"]),
        )
    )
    matrix_rows.sort(
        key=lambda row: (
            int(row.get("coefficient_dimension", 0)),
            int(row["sweep_target_support_count"]),
        )
    )
    bank_rows.sort(
        key=lambda row: (
            int(row.get("anchor_count", 0)),
            int(row["sweep_target_support_count"]),
        )
    )
    selected_variant = decision["complexity_selection"]["smallest_near_best_variant"]
    selected_row = next(
        (row for row in tradeoff_rows if row["variant"] == selected_variant), None
    )
    decision["selected_target_support_count"] = (
        int(selected_row["sweep_target_support_count"]) if selected_row else None
    )
    decision["selected_child"] = selected_row.get("sweep_child") if selected_row else None
    decision["interpretation"] = (
        "Use the selected adaptive basis for learned-decoder experiments only if "
        "it remains numerically full-rank and materially improves the fixed-bandwidth "
        "capacity frontier; otherwise retain the lower-dimensional basis."
    )
    summary = {
        "schema_version": 1,
        "purpose": "Stage 0.4b high-density adaptive-overlap basis confirmation",
        "input_root": str(input_root),
        "children": children,
        "shared_contract": {
            "stage1_summary_sha256": expected_contract[0],
            "stage1_samples_sha256": expected_contract[1],
            "sample_count": expected_contract[2],
            "valid_vertex_count": expected_contract[3],
            "mesh_sha256": expected_contract[4],
            "palm_faces_sha256": expected_contract[5],
        },
        "base": base_row,
        "decision": decision,
        "tradeoff": tradeoff_rows,
        "bank_diagnostics": bank_rows,
        "matrix_diagnostics": matrix_rows,
    }
    _write_csv(output_dir / "basis_density_curve.csv", tradeoff_rows)
    _write_csv(output_dir / "basis_density_bank_diagnostics.csv", bank_rows)
    _write_csv(output_dir / "basis_density_matrix_diagnostics.csv", matrix_rows)
    _write_json(output_dir / "basis_density_tradeoff.json", decision)
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "AUDIT_DONE.json",
        {
            "schema_version": 1,
            "completed_support_count": len(children),
            "candidate_count": len(tradeoff_rows),
            "smallest_near_best_variant": selected_variant,
            "smallest_near_best_dimension": decision["complexity_selection"][
                "smallest_near_best_dimension"
            ],
            "selected_target_support_count": decision[
                "selected_target_support_count"
            ],
            "summary": "summary.json",
        },
    )
    print(f"Basis density sweep aggregate complete: {output_dir}", flush=True)
    print(
        "selected="
        f"{selected_variant or 'none'} support="
        f"{decision['selected_target_support_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
