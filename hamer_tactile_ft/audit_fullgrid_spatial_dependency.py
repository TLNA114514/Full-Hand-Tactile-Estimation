#!/usr/bin/env python3
"""Audit whether the trained FullGrid decoder uses spatial token placement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hamer_tactile_ft.audit_canonical_localization import (
    MetricAccumulator,
    _validate_checkpoint_provenance,
)
from hamer_tactile_ft.audit_local_controllability import (
    CacheGroup,
    _available_fields,
    _load_decoder,
    _palm_mask,
)
from tactile_input_priors.feature_cache import sha256_file


SCHEMA = "fullgrid_spatial_dependency_v1"
SUPPORTED_VARIANTS = (
    "identity",
    "global_mean",
    "spatial_shuffle",
    "block_shuffle",
    "cyclic_shift",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty spatial-dependency table")
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_variants(raw: str) -> tuple[str, ...]:
    variants = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    unknown = sorted(set(variants) - set(SUPPORTED_VARIANTS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unsupported variants {unknown}; choose from {SUPPORTED_VARIANTS}"
        )
    if "identity" not in variants:
        raise argparse.ArgumentTypeError("variants must include identity")
    if len(set(variants)) != len(variants):
        raise argparse.ArgumentTypeError("variants must not contain duplicates")
    return variants


def _stable_seed(sample_id: str, seed: int, namespace: str) -> int:
    payload = f"{int(seed)}:{namespace}:{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _permutations(
    sample_ids: Sequence[str],
    count: int,
    *,
    seed: int,
    namespace: str,
    device: torch.device,
) -> torch.Tensor:
    values = []
    for sample_id in sample_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_seed(sample_id, seed, namespace))
        values.append(torch.randperm(int(count), generator=generator))
    return torch.stack(values).to(device=device, non_blocking=True)


def perturb_grid(
    grid: torch.Tensor,
    variant: str,
    sample_ids: Sequence[str],
    *,
    seed: int,
    block_size: tuple[int, int],
    shift: tuple[int, int],
) -> torch.Tensor:
    """Apply one content-preserving or information-removing spatial control."""

    if grid.ndim != 4:
        raise ValueError(f"Expected grid [B,C,H,W], got {tuple(grid.shape)}")
    variant = str(variant).strip().lower()
    if variant == "identity":
        return grid
    if variant == "global_mean":
        return grid.mean(dim=(2, 3), keepdim=True).expand_as(grid)
    if variant == "cyclic_shift":
        return torch.roll(grid, shifts=shift, dims=(2, 3))

    batch, channels, height, width = grid.shape
    if len(sample_ids) != batch:
        raise ValueError("sample_ids length must equal the grid batch size")
    if variant == "spatial_shuffle":
        token_count = height * width
        order = _permutations(
            sample_ids,
            token_count,
            seed=seed,
            namespace=variant,
            device=grid.device,
        )
        flat = grid.flatten(2)
        return torch.gather(
            flat,
            2,
            order[:, None].expand(-1, channels, -1),
        ).reshape_as(grid)
    if variant == "block_shuffle":
        block_height, block_width = block_size
        if height % block_height or width % block_width:
            raise ValueError(
                f"Grid {(height, width)} is not divisible by block size {block_size}"
            )
        height_blocks = height // block_height
        width_blocks = width // block_width
        block_count = height_blocks * width_blocks
        order = _permutations(
            sample_ids,
            block_count,
            seed=seed,
            namespace=variant,
            device=grid.device,
        )
        blocks = (
            grid.reshape(
                batch,
                channels,
                height_blocks,
                block_height,
                width_blocks,
                block_width,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(batch, channels, block_count, block_height, block_width)
        )
        shuffled = torch.gather(
            blocks,
            2,
            order[:, None, :, None, None].expand(
                -1, channels, -1, block_height, block_width
            ),
        )
        return (
            shuffled.reshape(
                batch,
                channels,
                height_blocks,
                width_blocks,
                block_height,
                block_width,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape_as(grid)
        )
    raise ValueError(f"Unsupported variant={variant!r}")


class DeltaAccumulator:
    def __init__(self) -> None:
        self.point_count = 0
        self.abs_sum = 0.0
        self.square_sum = 0.0
        self.contact_flip_count = 0
        self.cosine_sum = 0.0
        self.frame_count = 0
        self.feature_square_sum = 0.0
        self.feature_base_square_sum = 0.0
        self.feature_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        baseline: torch.Tensor,
        perturbed_grid: torch.Tensor,
        original_grid: torch.Tensor,
    ) -> None:
        delta = prediction.detach().float() - baseline.detach().float()
        self.point_count += int(delta.numel())
        self.abs_sum += float(delta.abs().sum().item())
        self.square_sum += float(delta.square().sum().item())
        self.contact_flip_count += int(
            ((prediction >= 0.10) != (baseline >= 0.10)).sum().item()
        )
        self.cosine_sum += float(
            F.cosine_similarity(prediction.float(), baseline.float(), dim=1)
            .sum()
            .item()
        )
        self.frame_count += int(prediction.shape[0])
        feature_delta = perturbed_grid.detach().float() - original_grid.detach().float()
        self.feature_square_sum += float(feature_delta.square().sum().item())
        self.feature_base_square_sum += float(
            original_grid.detach().float().square().sum().item()
        )
        self.feature_count += int(feature_delta.numel())

    def summary(self) -> dict[str, float]:
        point_count = max(self.point_count, 1)
        feature_count = max(self.feature_count, 1)
        feature_rms = (self.feature_square_sum / feature_count) ** 0.5
        feature_base_rms = (self.feature_base_square_sum / feature_count) ** 0.5
        return {
            "prediction_delta_abs_mean": self.abs_sum / point_count,
            "prediction_delta_rms": (self.square_sum / point_count) ** 0.5,
            "contact_flip_fraction": self.contact_flip_count / point_count,
            "prediction_cosine_mean": self.cosine_sum / max(self.frame_count, 1),
            "feature_delta_rms": feature_rms,
            "feature_delta_to_base_rms": feature_rms / max(feature_base_rms, 1e-12),
        }


def _contract(args: argparse.Namespace, cache: CacheGroup) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "feature_cache": str(Path(args.feature_cache).expanduser().resolve()),
        "cache_config_sha256s": list(cache.config_sha256s),
        "base_checkpoint": str(Path(args.base_checkpoint).expanduser().resolve()),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "sample_limit": int(args.sample_limit),
        "batch_size": int(args.batch_size),
        "seed": int(args.seed),
        "variants": list(args.variants),
        "block_size": list(args.block_size),
        "shift": list(args.shift),
    }


def _contract_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = _available_fields(Path(args.feature_cache).expanduser().resolve(strict=True))
    required = {"z_rgb", "tactile_signal"}
    missing = sorted(required - fields)
    if missing:
        raise RuntimeError(f"Feature cache lacks required fields: {missing}")
    requested_fields = ["z_rgb", "tactile_signal"]
    if "palm_mask" in fields:
        requested_fields.append("palm_mask")
    cache = CacheGroup(args.feature_cache, requested_fields)
    try:
        _validate_checkpoint_provenance(
            cache,
            Path(args.base_checkpoint).expanduser().resolve(strict=True),
        )
        contract = _contract(args, cache)
        contract_sha = _contract_sha256(contract)
        done_path = output_dir / "AUDIT_DONE.json"
        if done_path.is_file() and not args.force:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("contract_sha256") == contract_sha:
                print(f"[fullgrid-spatial] reusing completed audit: {output_dir}")
                return
            raise RuntimeError(
                f"Existing audit contract differs under {output_dir}; pass --force to replace outputs"
            )

        sample_count = min(int(args.sample_limit), len(cache))
        if sample_count <= 0:
            raise ValueError("sample_limit must select at least one sample")
        rng = np.random.default_rng(int(args.seed))
        if sample_count == len(cache):
            selected = np.arange(len(cache), dtype=np.int64)
        else:
            selected = np.sort(
                rng.choice(len(cache), size=sample_count, replace=False)
            ).astype(np.int64)

        first = cache[int(selected[0])]
        first_grid = np.asarray(first["z_rgb"])
        if first_grid.ndim != 3:
            raise ValueError(f"z_rgb must be [C,H,W], got {first_grid.shape}")
        first_target = np.asarray(first["tactile_signal"]).reshape(-1)
        palm_mask = _palm_mask(first, cache.provenance)
        decoder, decoder_config = _load_decoder(
            Path(args.base_checkpoint).expanduser().resolve(strict=True),
            grid_size=tuple(int(value) for value in first_grid.shape[-2:]),
            tactile_dim=int(first_target.size),
            device=torch.device(args.device),
        )
        device = torch.device(args.device)
        metrics = {
            variant: MetricAccumulator(variant) for variant in args.variants
        }
        deltas = {variant: DeltaAccumulator() for variant in args.variants}
        selected_rows: list[dict[str, Any]] = []

        for start in range(0, sample_count, int(args.batch_size)):
            batch_indices = selected[start : start + int(args.batch_size)]
            items = [cache[int(index)] for index in batch_indices]
            sample_ids = [str(item["sample_id"]) for item in items]
            for cache_index, item in zip(batch_indices.tolist(), items):
                current_mask = _palm_mask(item, cache.provenance)
                if not np.array_equal(current_mask, palm_mask):
                    raise RuntimeError("The canonical palm mask differs across cached samples")
                selected_rows.append(
                    {"cache_index": int(cache_index), "sample_id": str(item["sample_id"])}
                )
            grid = torch.from_numpy(
                np.stack(
                    [np.asarray(item["z_rgb"], dtype=np.float32) for item in items]
                )
            ).to(device=device, non_blocking=True)
            target = np.stack(
                [
                    np.asarray(item["tactile_signal"], dtype=np.float32)[palm_mask]
                    for item in items
                ]
            )
            with torch.inference_mode():
                base_prediction_full = torch.sigmoid(decoder(grid))
                base_prediction = base_prediction_full[:, palm_mask]
                for variant in args.variants:
                    perturbed = perturb_grid(
                        grid,
                        variant,
                        sample_ids,
                        seed=int(args.seed),
                        block_size=tuple(args.block_size),
                        shift=tuple(args.shift),
                    )
                    prediction = (
                        base_prediction
                        if variant == "identity"
                        else torch.sigmoid(decoder(perturbed))[:, palm_mask]
                    )
                    metrics[variant].update(
                        prediction.float().cpu().numpy(), target
                    )
                    deltas[variant].update(
                        prediction,
                        base_prediction,
                        perturbed,
                        grid,
                    )
            print(
                f"[fullgrid-spatial] audited {min(start + len(items), sample_count):,}/{sample_count:,}",
                flush=True,
            )

        rows: list[dict[str, Any]] = []
        identity_summary = metrics["identity"].summary()
        comparison_keys = (
            "rmse_vertex_micro",
            "contact_iou_010_frame_macro",
            "volumetric_iou_frame_macro",
            "core_distribution_viou_frame_macro",
            "false_high_excess_mean",
        )
        for variant in args.variants:
            row = dict(metrics[variant].summary())
            row.update(deltas[variant].summary())
            for key in comparison_keys:
                row[f"delta_{key}_vs_identity"] = float(row[key]) - float(
                    identity_summary[key]
                )
            rows.append(row)

        _write_csv(output_dir / "metrics.csv", rows)
        _write_jsonl(output_dir / "selected_samples.jsonl", selected_rows)
        summary = {
            "schema": SCHEMA,
            "contract": contract,
            "contract_sha256": contract_sha,
            "decoder_config": decoder_config,
            "valid_vertex_count": int(palm_mask.sum()),
            "metrics": rows,
            "interpretation": {
                "global_mean": "tests whether spatial variation is needed beyond per-channel global means",
                "spatial_shuffle": "preserves token content but destroys content-to-position assignment",
                "block_shuffle": "preserves within-block structure but destroys global block placement",
                "cyclic_shift": "measures sensitivity to a fixed one-grid displacement without changing token values",
            },
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(
            done_path,
            {
                "schema": SCHEMA,
                "contract_sha256": contract_sha,
                "summary_sha256": sha256_file(output_dir / "summary.json"),
                "metrics_sha256": sha256_file(output_dir / "metrics.csv"),
                "selected_samples_sha256": sha256_file(
                    output_dir / "selected_samples.jsonl"
                ),
            },
        )
        print(f"[fullgrid-spatial] complete: {output_dir}")
    finally:
        cache.close()


def self_test() -> None:
    grid = torch.arange(3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)
    sample_ids = ["sample-a"]
    identity = perturb_grid(
        grid, "identity", sample_ids, seed=521, block_size=(2, 2), shift=(1, 1)
    )
    if not torch.equal(identity, grid):
        raise AssertionError("identity perturbation changed the grid")
    mean = perturb_grid(
        grid, "global_mean", sample_ids, seed=521, block_size=(2, 2), shift=(1, 1)
    )
    if not torch.allclose(mean.mean(dim=(2, 3)), grid.mean(dim=(2, 3))):
        raise AssertionError("global_mean did not preserve channel means")
    for variant in ("spatial_shuffle", "block_shuffle", "cyclic_shift"):
        value = perturb_grid(
            grid,
            variant,
            sample_ids,
            seed=521,
            block_size=(2, 2),
            shift=(1, 1),
        )
        if not torch.equal(
            value.flatten(2).sort(dim=2).values,
            grid.flatten(2).sort(dim=2).values,
        ):
            raise AssertionError(f"{variant} changed the token multiset")
    print("FullGrid spatial-dependency self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-limit", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument(
        "--variants",
        type=_parse_variants,
        default=SUPPORTED_VARIANTS,
    )
    parser.add_argument("--block-size", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--shift", type=int, nargs=2, default=(1, 1))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    for name in ("feature_cache", "base_checkpoint", "output_dir"):
        if not getattr(args, name):
            raise ValueError(f"--{name.replace('_', '-')} is required")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")
    if any(int(value) <= 0 for value in args.block_size):
        raise ValueError("--block-size values must be positive")
    run_audit(args)


if __name__ == "__main__":
    main()
