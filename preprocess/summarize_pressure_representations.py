#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    "outputs/representation_eval_opentouch/summary.csv",
    "outputs/representation_eval_egotactile/summary.csv",
    "outputs/representation_eval_touchanything/summary.csv",
    "outputs/representation_eval_egotactile_raw_direct/summary.csv",
    "outputs/representation_eval_touchanything_raw_direct/summary.csv",
)
METHOD_ALIASES = {
    "ot_raw_heatmap": "raw_to_mano_direct",
}
METHOD_ORDER = {
    "raw_to_mano_direct": 0,
    "ot_discrete_heatmap": 1,
    "ot_centered_mano": 2,
    "egotactile_heatmap": 3,
    "preprocess_gaussian": 4,
}
CSV_COLUMNS = (
    "dataset",
    "split",
    "hand",
    "method",
    "source_method",
    "sequences",
    "frames_evaluated",
    "mapped_sensor_count",
    "raw_sensor_count",
    "sensor_mapping_coverage",
    "spatial_laplacian_mean",
    "spatial_laplacian_p50",
    "spatial_laplacian_p90",
    "native_spatial_laplacian_mean",
    "temp_1st_mean",
    "temp_2nd_mean",
    "centroid_error_l2_mean",
    "peak_abs_error_mean",
    "peak_overshoot_rate",
    "emd_mean",
    "emd_p50",
    "emd_p90",
    "emd_mode",
    "emd_solver",
    "source_summary",
    "source_summary_sha256",
)
MARKDOWN_COLUMNS = (
    "dataset",
    "hand",
    "method",
    "frames_evaluated",
    "sensor_mapping_coverage",
    "spatial_laplacian_mean",
    "temp_1st_mean",
    "temp_2nd_mean",
    "centroid_error_l2_mean",
    "peak_abs_error_mean",
    "peak_overshoot_rate",
    "emd_mean",
)
DIRECT_GAUSSIAN_METHODS = frozenset(
    {
        "raw_to_mano_direct",
        "preprocess_gaussian",
    }
)
DIRECT_GAUSSIAN_COLUMNS = (
    "dataset",
    "hand",
    "method",
    "spatial_laplacian_mean",
    "temp_1st_mean",
    "temp_2nd_mean",
    "centroid_error_l2_mean",
    "peak_abs_error_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge historical and new pressure-representation summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=list(DEFAULT_INPUTS),
        help="Summary CSV files or output directories containing summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/representation_eval_master",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing inputs. The default fails to prevent an incomplete main table.",
    )
    return parser.parse_args()


def _resolve_input(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        path = path / "summary.csv"
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_row(row: dict[str, str], path: Path, checksum: str) -> dict[str, str]:
    source_method = str(row.get("method", "")).strip()
    normalized = {column: str(row.get(column, "")).strip() for column in CSV_COLUMNS}
    normalized.update(
        {
            "method": METHOD_ALIASES.get(source_method, source_method),
            "source_method": source_method,
            "source_summary": str(path),
            "source_summary_sha256": checksum,
        }
    )
    return normalized


def load_rows(paths: Iterable[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: dict[tuple[str, str, str, str], dict[str, str]] = {}
    provenance = []
    for path in paths:
        checksum = _sha256(path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            current = [_normalized_row(row, path, checksum) for row in csv.DictReader(handle)]
        provenance.append(
            {
                "path": str(path),
                "sha256": checksum,
                "rows": len(current),
                "run_config_path": (
                    str(path.parent / "run_config.json")
                    if (path.parent / "run_config.json").is_file()
                    else None
                ),
                "run_config_sha256": (
                    _sha256(path.parent / "run_config.json")
                    if (path.parent / "run_config.json").is_file()
                    else None
                ),
            }
        )
        for row in current:
            key = (row["dataset"], row["split"], row["hand"], row["method"])
            previous = rows.get(key)
            if previous is not None:
                raise RuntimeError(
                    "Duplicate canonical main-table row for "
                    f"{key}: {previous['source_summary']} and {path}"
                )
            rows[key] = row
    ordered = sorted(
        rows.values(),
        key=lambda row: (
            row["dataset"],
            row["split"],
            row["hand"],
            METHOD_ORDER.get(row["method"], 100),
            row["method"],
        ),
    )
    return ordered, provenance


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown(value: str) -> str:
    if value == "":
        return "-"
    try:
        number = float(value)
    except ValueError:
        return value
    if not number.is_integer() or any(marker in value.lower() for marker in (".", "e")):
        return f"{number:.6g}"
    return str(int(number))


def _write_markdown(path: Path, rows: list[dict[str, str]], provenance: list[dict[str, str]]) -> None:
    labels = {
        "dataset": "Dataset",
        "hand": "Hand",
        "method": "Representation",
        "frames_evaluated": "Frames",
        "sensor_mapping_coverage": "Mapped coverage",
        "spatial_laplacian_mean": "Spatial Lap. down",
        "temp_1st_mean": "Temporal-1 down",
        "temp_2nd_mean": "Temporal-2 down",
        "centroid_error_l2_mean": "Centroid err. down",
        "peak_abs_error_mean": "Peak err. down",
        "peak_overshoot_rate": "Peak over. down",
        "emd_mean": "EMD down",
    }
    lines = [
        "# Pressure Representation Main Table",
        "",
        "All fidelity metrics use the normalized raw taxels as the reference. "
        "`raw_to_mano_direct` scatters only calibrated taxels to their assigned "
        "MANO vertices and applies no spatial smoothing.",
        "",
        "| " + " | ".join(labels[column] for column in MARKDOWN_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(MARKDOWN_COLUMNS)) + " |",
    ]
    for row in rows:
        values = []
        for column in MARKDOWN_COLUMNS:
            value = _format_markdown(row[column])
            if column == "sensor_mapping_coverage" and value != "-":
                value = f"{100.0 * float(row[column]):.1f}%"
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "The complete schema, percentiles, solver settings, original method names, "
            "and SHA256 provenance are retained in `master_table.csv`.",
            "",
        ]
    )
    for item in provenance:
        detail = f"{item['rows']} rows, summary SHA256 `{item['sha256']}`"
        if item.get("run_config_sha256"):
            detail += f", config SHA256 `{item['run_config_sha256']}`"
        lines.append(f"- `{item['path']}`: {detail}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _direct_gaussian_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["method"] in DIRECT_GAUSSIAN_METHODS]


def _write_direct_gaussian_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DIRECT_GAUSSIAN_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in DIRECT_GAUSSIAN_COLUMNS})


def _write_direct_gaussian_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    labels = {
        "dataset": "Dataset",
        "hand": "Hand",
        "method": "Representation",
        "spatial_laplacian_mean": "Spatial Lap. down",
        "temp_1st_mean": "Temporal-1 down",
        "temp_2nd_mean": "Temporal-2 down",
        "centroid_error_l2_mean": "Centroid err. down",
        "peak_abs_error_mean": "Peak err. down",
    }
    lines = [
        "# Raw-to-MANO Direct vs. Gaussian",
        "",
        "All metrics use normalized raw taxels as the reference; lower is better.",
        "",
        "| " + " | ".join(labels[column] for column in DIRECT_GAUSSIAN_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(DIRECT_GAUSSIAN_COLUMNS)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_format_markdown(row[column]) for column in DIRECT_GAUSSIAN_COLUMNS)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    resolved = [_resolve_input(value) for value in args.inputs]
    missing = [path for path in resolved if not path.is_file()]
    if missing and not args.allow_missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing representation summaries:\n{formatted}")
    resolved = [path for path in resolved if path.is_file()]
    if not resolved:
        raise RuntimeError("No representation summaries were found")

    rows, provenance = load_rows(resolved)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "master_table.csv", rows)
    _write_markdown(output_dir / "master_table.md", rows, provenance)
    direct_gaussian_rows = _direct_gaussian_rows(rows)
    _write_direct_gaussian_csv(
        output_dir / "raw_direct_vs_gaussian.csv", direct_gaussian_rows
    )
    _write_direct_gaussian_markdown(
        output_dir / "raw_direct_vs_gaussian.md", direct_gaussian_rows
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "inputs": provenance,
                "canonical_method_aliases": METHOD_ALIASES,
                "row_count": len(rows),
                "direct_gaussian_row_count": len(direct_gaussian_rows),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
