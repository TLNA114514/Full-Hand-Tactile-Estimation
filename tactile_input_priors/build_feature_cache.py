#!/usr/bin/env python3
"""CLI and generic input adapters for :mod:`tactile_input_priors.feature_cache`.

Examples::

    python tactile_input_priors/build_feature_cache.py build \
      --manifest samples.jsonl --cache-dir cache \
      --field z_rgb:float16:256x16x12

Manifest rows may contain inline arrays under ``features`` or descriptors such
as ``{"path": "sample.npz", "key": "z_rgb"}``.  Alternatively, one aligned
NPZ may provide a leading sample dimension for every configured field.  Python
model builders can use ``FeatureCacheBuilder`` directly or pass
``--callback package.module:function``; the callable receives
``(manifest_row, source_index, callback_kwargs)``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

try:
    from tactile_input_priors.feature_cache import (
        ALLOWED_FEATURE_FIELDS,
        FeatureCacheBuilder,
        FeatureCacheDataset,
        FeatureCacheMismatchError,
        FeatureSpec,
        atomic_write_json,
        canonical_json,
        inspect_source_manifest,
        normalize_feature_specs,
        sha256_file,
        verify_feature_cache,
    )
except ImportError:  # Direct execution from tactile_input_priors/.
    from feature_cache import (  # type: ignore
        ALLOWED_FEATURE_FIELDS,
        FeatureCacheBuilder,
        FeatureCacheDataset,
        FeatureCacheMismatchError,
        FeatureSpec,
        atomic_write_json,
        canonical_json,
        inspect_source_manifest,
        normalize_feature_specs,
        sha256_file,
        verify_feature_cache,
    )


def _parse_json_or_path(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    candidate = Path(value).expanduser()
    if candidate.is_file():
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def _parse_field(value: str) -> FeatureSpec:
    try:
        name, dtype, shape_text = value.split(":", 2)
        shape = tuple(int(item) for item in shape_text.lower().replace(",", "x").split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid field {value!r}; expected NAME:DTYPE:DIMxDIM..."
        ) from exc
    try:
        return FeatureSpec(name=name, dtype=dtype, shape=shape)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _load_fields_json(path: str | None) -> list[FeatureSpec]:
    if not path:
        return []
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if isinstance(value, dict):
        rows = [
            {"name": name, "dtype": config["dtype"], "shape": config["shape"]}
            for name, config in value.items()
        ]
    elif isinstance(value, list):
        rows = value
    else:
        raise ValueError("--fields-json must contain an object or list")
    return [FeatureSpec.from_json(item) for item in rows]


def _feature_specs(args: argparse.Namespace) -> tuple[FeatureSpec, ...]:
    return normalize_feature_specs([*args.field, *_load_fields_json(args.fields_json)])


def _resolve_descriptor(value: Any, *, base_dir: Path, field_name: str) -> np.ndarray:
    if not isinstance(value, Mapping) or "path" not in value:
        return np.asarray(value)
    path = Path(os.path.expandvars(os.path.expanduser(str(value["path"]))))
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve(strict=True)
    expected_sha = str(value.get("sha256", "")).strip().lower()
    if len(expected_sha) != 64:
        raise ValueError(
            f"External feature descriptor for {field_name!r} must include a "
            "64-character sha256"
        )
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"External feature SHA mismatch for {path}: "
            f"expected={expected_sha}, actual={actual_sha}"
        )
    key = str(value.get("key", field_name))
    row_index = value.get("index")
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{path} has no key {key!r}")
            array = np.asarray(archive[key])
    else:
        raise ValueError(f"Unsupported feature descriptor path: {path}")
    return np.asarray(array[int(row_index)] if row_index is not None else array)


class InlineJsonlCallback:
    def __init__(self, manifest: Path, specs: Sequence[FeatureSpec]):
        self.base_dir = manifest.parent
        self.specs = tuple(specs)

    def __call__(self, row: Mapping[str, Any], source_index: int) -> Mapping[str, Any]:
        del source_index
        values = row.get("features", row)
        if not isinstance(values, Mapping):
            raise TypeError("Manifest row 'features' must be an object")
        result = {}
        for spec in self.specs:
            if spec.name in values:
                value = values[spec.name]
            elif f"{spec.name}_path" in values:
                sha_key = f"{spec.name}_sha256"
                if sha_key not in values:
                    raise KeyError(
                        f"Manifest shorthand {spec.name + '_path'!r} requires {sha_key!r}"
                    )
                value = {
                    "path": values[f"{spec.name}_path"],
                    "key": spec.name,
                    "sha256": values[sha_key],
                }
            else:
                raise KeyError(f"Manifest row has no feature {spec.name!r}")
            result[spec.name] = _resolve_descriptor(
                value,
                base_dir=self.base_dir,
                field_name=spec.name,
            )
        return result


class AlignedNpzCallback:
    def __init__(
        self,
        path: Path,
        specs: Sequence[FeatureSpec],
        *,
        sample_id_key: str,
        npz_sample_id_key: str,
        expected_count: int,
    ):
        self.path = path.resolve(strict=True)
        self.archive = np.load(self.path, allow_pickle=False)
        self.specs = tuple(specs)
        self.sample_id_key = sample_id_key
        self.ids = self.archive[npz_sample_id_key] if npz_sample_id_key in self.archive else None
        for spec in self.specs:
            if spec.name not in self.archive:
                raise KeyError(f"{self.path} has no field {spec.name!r}")
            if len(self.archive[spec.name]) != int(expected_count):
                raise ValueError(
                    f"{self.path}:{spec.name} has {len(self.archive[spec.name])} rows, "
                    f"expected {expected_count}"
                )
        if self.ids is not None and len(self.ids) != int(expected_count):
            raise ValueError(
                f"{self.path}:{npz_sample_id_key} has {len(self.ids)} rows, "
                f"expected {expected_count}"
            )

    def __call__(self, row: Mapping[str, Any], source_index: int) -> Mapping[str, Any]:
        if self.ids is not None:
            manifest_id = str(row[self.sample_id_key])
            npz_id = str(self.ids[source_index])
            if manifest_id != npz_id:
                raise ValueError(
                    f"NPZ/manifest sample ID mismatch at {source_index}: "
                    f"{npz_id!r} != {manifest_id!r}"
                )
        return {spec.name: self.archive[spec.name][source_index] for spec in self.specs}

    def close(self) -> None:
        self.archive.close()


def _load_external_callback(
    reference: str,
    callback_kwargs: Mapping[str, Any],
) -> Callable[[Mapping[str, Any], int], Mapping[str, Any]]:
    if ":" not in reference:
        raise ValueError("--callback must be MODULE:CALLABLE")
    module_name, attribute = reference.rsplit(":", 1)
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise TypeError(f"{reference} is not callable")

    def wrapped(row: Mapping[str, Any], source_index: int) -> Mapping[str, Any]:
        return callback(row, source_index, dict(callback_kwargs))

    return wrapped


def command_build(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest).expanduser().resolve(strict=True)
    specs = _feature_specs(args)
    provenance = _parse_json_or_path(args.provenance_json)
    provenance.setdefault("feature_source", "inline_jsonl")
    closer = None
    if args.input_npz:
        npz_path = Path(args.input_npz).expanduser().resolve(strict=True)
        provenance.update(
            {
                "feature_source": "aligned_npz",
                "input_npz_name": npz_path.name,
                "input_npz_sha256": sha256_file(npz_path),
            }
        )
        callback = AlignedNpzCallback(
            npz_path,
            specs,
            sample_id_key=args.sample_id_key,
            npz_sample_id_key=args.npz_sample_id_key,
            expected_count=inspect_source_manifest(
                manifest, args.sample_id_key
            ).sample_count,
        )
        closer = callback.close
    elif args.callback:
        module_name = args.callback.rsplit(":", 1)[0]
        module = importlib.import_module(module_name)
        module_file = getattr(module, "__file__", None)
        if not module_file:
            raise ValueError(
                f"Callback module {module_name!r} has no hashable source file"
            )
        module_path = Path(module_file).resolve(strict=True)
        provenance.update(
            {
                "feature_source": "python_callback",
                "callback": args.callback,
                "callback_module_sha256": sha256_file(module_path),
                "callback_kwargs": _parse_json_or_path(args.callback_kwargs),
            }
        )
        callback = _load_external_callback(
            args.callback,
            _parse_json_or_path(args.callback_kwargs),
        )
    else:
        callback = InlineJsonlCallback(manifest, specs)
    builder = FeatureCacheBuilder(
        args.cache_dir,
        manifest,
        specs,
        provenance=provenance,
        shard_size=args.shard_size,
        sample_id_key=args.sample_id_key,
        lock_timeout_seconds=args.lock_timeout,
        break_stale_lock=args.break_stale_lock,
        deep_verify_existing=args.deep_verify_existing,
        repair_invalid_shards=args.repair_invalid_shards,
    )
    try:
        result = builder.build(callback, max_new_shards=args.max_new_shards)
    finally:
        if closer is not None:
            closer()
    print(json.dumps(result, indent=2, sort_keys=True))


def command_verify(args: argparse.Namespace) -> None:
    result = verify_feature_cache(
        args.cache_dir,
        deep=args.deep,
        expected_manifest_path=args.manifest,
        expected_provenance=(
            _parse_json_or_path(args.expected_provenance_json)
            if args.expected_provenance_json
            else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_inspect(args: argparse.Namespace) -> None:
    with FeatureCacheDataset(args.cache_dir, fields=args.fields or None) as dataset:
        index: int | str
        try:
            index = int(args.sample)
        except ValueError:
            index = args.sample
        row = dataset[index]
        summary = {
            key: (
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "min": float(np.min(value)),
                    "max": float(np.max(value)),
                }
                if isinstance(value, np.ndarray)
                else value
            )
            for key, value in row.items()
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_self_check(args: argparse.Namespace) -> None:
    del args
    with tempfile.TemporaryDirectory(prefix="tactile-feature-cache-") as directory:
        root = Path(directory)
        manifest = root / "samples.jsonl"
        rows = []
        for index in range(7):
            rows.append(
                {
                    "sample_id": f"sample-{index}",
                    "features": {
                        "z_rgb": [[index, index + 1], [index + 2, index + 3]],
                        "h_rgb": [index, index + 0.25, -index],
                    },
                }
            )
        manifest.write_text(
            "".join(canonical_json(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        specs = (
            FeatureSpec("z_rgb", (2, 2), "float32"),
            FeatureSpec("h_rgb", (3,), "float16"),
        )
        callback = InlineJsonlCallback(manifest, specs)
        provenance = {"self_check": True, "model_sha256": "0" * 64}
        builder = FeatureCacheBuilder(
            root / "cache",
            manifest,
            specs,
            provenance=provenance,
            shard_size=3,
        )
        first = builder.build(callback, max_new_shards=1)
        if first["complete"] or first["new_shards"] != 1:
            raise AssertionError(f"Bounded build did not stop after one shard: {first}")
        fake_partial = root / "cache/shards/shard-000001.partial.interrupted"
        fake_partial.mkdir()
        (fake_partial / "garbage").write_text("incomplete", encoding="utf-8")
        resumed = builder.build(callback)
        if not resumed["complete"] or resumed["new_shards"] != 2:
            raise AssertionError(f"Resume did not reuse exactly one shard: {resumed}")
        verified = verify_feature_cache(
            root / "cache",
            deep=True,
            expected_manifest_path=manifest,
            expected_provenance=provenance,
        )
        with FeatureCacheDataset(root / "cache", copy_arrays=True) as dataset:
            if len(dataset) != 7:
                raise AssertionError("Dataset length mismatch")
            by_id = dataset.get_by_id("sample-5")
            by_index = dataset[5]
            np.testing.assert_array_equal(by_id["z_rgb"], by_index["z_rgb"])
            np.testing.assert_allclose(by_id["h_rgb"], [5, 5.25, -5])
            if dataset[-1]["sample_id"] != "sample-6":
                raise AssertionError("Negative indexing failed")
        try:
            FeatureCacheDataset(
                root / "cache",
                expected_provenance={"self_check": False},
            )
        except FeatureCacheMismatchError:
            pass
        else:
            raise AssertionError("Provenance mismatch was not rejected")
        print(json.dumps({"self_check": "passed", **verified}, indent=2, sort_keys=True))


def _add_field_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--field",
        action="append",
        type=_parse_field,
        default=[],
        help=(
            "Fixed field NAME:DTYPE:DIMxDIM..., repeatable. Allowed names: "
            + ", ".join(ALLOWED_FEATURE_FIELDS)
        ),
    )
    parser.add_argument(
        "--fields-json",
        help="JSON file mapping field names to {dtype,shape}, or a list of specs.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build or resume a feature cache.")
    build.add_argument("--manifest", required=True, help="Authoritative source JSONL.")
    build.add_argument("--cache-dir", required=True)
    _add_field_arguments(build)
    build.add_argument("--sample-id-key", default="sample_id")
    build.add_argument("--shard-size", type=int, default=4096)
    build.add_argument("--input-npz", help="Aligned NPZ with one leading sample axis.")
    build.add_argument("--npz-sample-id-key", default="sample_id")
    build.add_argument("--callback", help="Python callback as MODULE:CALLABLE.")
    build.add_argument("--callback-kwargs", help="Inline JSON object or JSON file.")
    build.add_argument("--provenance-json", help="Inline JSON object or JSON file.")
    build.add_argument("--max-new-shards", type=int)
    build.add_argument("--lock-timeout", type=float, default=600.0)
    build.add_argument("--break-stale-lock", action="store_true")
    build.add_argument("--deep-verify-existing", action="store_true")
    build.add_argument("--repair-invalid-shards", action="store_true")
    build.set_defaults(func=command_build)

    verify = subparsers.add_parser("verify", help="Validate a finalized cache.")
    verify.add_argument("--cache-dir", required=True)
    verify.add_argument("--manifest", help="Expected source manifest; checked by SHA.")
    verify.add_argument("--expected-provenance-json")
    verify.add_argument("--deep", action="store_true", help="Hash every feature file.")
    verify.set_defaults(func=command_verify)

    inspect = subparsers.add_parser("inspect", help="Inspect one cached sample.")
    inspect.add_argument("--cache-dir", required=True)
    inspect.add_argument("--sample", default="0", help="Integer ordinal or sample ID.")
    inspect.add_argument("--fields", nargs="*")
    inspect.set_defaults(func=command_inspect)

    self_check = subparsers.add_parser("self-check", help="Run a tiny resume/read check.")
    self_check.set_defaults(func=command_self_check)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build" and args.input_npz and args.callback:
        raise SystemExit("--input-npz and --callback are mutually exclusive")
    args.func(args)


if __name__ == "__main__":
    main()
