"""Resolve and self-heal the small HaMeR model configuration asset."""

from __future__ import annotations

import os
from pathlib import Path
import warnings


REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_HAMER_MODEL_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "hamer_model_config.yaml"
)


def resolve_hamer_model_config_path(
    workspace_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the HaMeR config, restoring the ignored asset when possible.

    ``hamer/_DATA`` is intentionally ignored because it normally contains
    multi-gigabyte model assets. The tiny YAML is nevertheless required by the
    shared dataset preprocessing path, so keep a synchronized canonical copy
    and recreate the original layout after accidental data cleanup.
    """

    explicit = os.environ.get("HAMER_MODEL_CONFIG", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(
                f"HAMER_MODEL_CONFIG does not point to a file: {path}"
            )
        return path.resolve()

    root = (
        Path(workspace_dir).expanduser().resolve(strict=False)
        if workspace_dir is not None
        else REPO_ROOT
    )
    expected = root / "hamer" / "_DATA" / "hamer_ckpts" / "model_config.yaml"
    if expected.is_file():
        return expected.resolve()
    if not BUNDLED_HAMER_MODEL_CONFIG.is_file():
        raise FileNotFoundError(
            "HaMeR model_config.yaml is missing from both the model assets and "
            f"the bundled fallback: {expected}, {BUNDLED_HAMER_MODEL_CONFIG}"
        )

    try:
        expected.parent.mkdir(parents=True, exist_ok=True)
        temporary = expected.with_name(f".{expected.name}.partial.{os.getpid()}")
        temporary.write_bytes(BUNDLED_HAMER_MODEL_CONFIG.read_bytes())
        os.replace(temporary, expected)
        print(f"Restored missing HaMeR model config: {expected}", flush=True)
        return expected.resolve()
    except OSError as error:
        warnings.warn(
            f"Could not restore {expected}: {error}; using bundled config "
            f"{BUNDLED_HAMER_MODEL_CONFIG}",
            RuntimeWarning,
            stacklevel=2,
        )
        return BUNDLED_HAMER_MODEL_CONFIG.resolve()
