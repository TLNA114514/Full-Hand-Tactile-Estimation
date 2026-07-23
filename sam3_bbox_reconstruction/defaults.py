"""Machine defaults for the SAM bbox reconstruction workflow.

The two training servers share the same filesystem layout.  Keeping those
defaults here makes the normal commands short while every CLI still accepts an
explicit override for a relocated checkout or dataset.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
LOCAL_REPO_ROOT = PACKAGE_ROOT.parent
SERVER_REPO_ROOT = Path(
    os.environ.get(
        "FULL_HAND_TACTILE_ROOT",
        "/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation",
    )
).expanduser()

DEFAULT_OPENTOUCH_DATA_ROOT = Path(
    os.environ.get("OPENTOUCH_DATA_ROOT", "/home/ma-user/work/cfzhao/OpenTouch Data/data")
).expanduser()
DEFAULT_OPENTOUCH_EXTRACTED_ROOT = Path(
    os.environ.get(
        "OPENTOUCH_EXTRACTED_ROOT",
        "/home/ma-user/work/cfzhao/OpenTouch Data/full_dataset",
    )
).expanduser()
_LOCAL_OPENTOUCH_SPLITS = LOCAL_REPO_ROOT / "evaluation" / "opentouch_splits.json"
DEFAULT_OPENTOUCH_SPLITS = Path(
    os.environ.get(
        "OPENTOUCH_SPLITS",
        str(
            _LOCAL_OPENTOUCH_SPLITS
            if _LOCAL_OPENTOUCH_SPLITS.is_file()
            else SERVER_REPO_ROOT / "evaluation" / "opentouch_splits.json"
        ),
    )
).expanduser()
DEFAULT_TOUCHANYTHING_ROOT = Path(
    os.environ.get("TOUCHANYTHING_ROOT", "/home/ma-user/work/cfzhao/EgoTouch")
).expanduser()
DEFAULT_TOUCHANYTHING_SPLIT_JSON = Path(
    os.environ.get(
        "TOUCHANYTHING_SPLIT_JSON",
        str(DEFAULT_TOUCHANYTHING_ROOT / "split.json"),
    )
).expanduser()
DEFAULT_TOUCHANYTHING_EXTRACTED_ROOT = Path(
    os.environ.get(
        "TOUCHANYTHING_EXTRACTED_ROOT",
        str(DEFAULT_TOUCHANYTHING_ROOT / "extracted_frames"),
    )
).expanduser()

_CHECKPOINT_FILENAMES = {
    "sam3": "sam3.pt",
    "sam3.1": "sam3.1_multiplex.pt",
}


def default_checkpoint_candidates(sam_version: str) -> tuple[Path, ...]:
    """Return deterministic local checkpoint candidates, preferred first."""

    try:
        filename = _CHECKPOINT_FILENAMES[sam_version]
    except KeyError as exc:
        raise ValueError(f"Unsupported SAM version: {sam_version!r}") from exc
    env_key = "SAM31_CHECKPOINT" if sam_version == "sam3.1" else "SAM3_CHECKPOINT"
    candidates = []
    if os.environ.get(env_key):
        candidates.append(Path(os.environ[env_key]).expanduser())
    candidates.extend(
        [
            SERVER_REPO_ROOT / "_DATA" / filename,
            SERVER_REPO_ROOT / "_DATA" / "sam3" / filename,
            LOCAL_REPO_ROOT / "_DATA" / filename,
            LOCAL_REPO_ROOT / "_DATA" / "sam3" / filename,
            PACKAGE_ROOT / "checkpoints" / filename,
            PACKAGE_ROOT / "third_party" / "sam3" / "checkpoints" / filename,
            PACKAGE_ROOT / "third_party" / "sam3" / filename,
        ]
    )
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hf_repo = (
        "models--facebook--sam3.1" if sam_version == "sam3.1" else "models--facebook--sam3"
    )
    candidates.extend(sorted((hf_home / "hub" / hf_repo / "snapshots").glob(f"*/{filename}")))
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def resolve_checkpoint(sam_version: str, checkpoint: Path | None) -> Path:
    """Resolve an explicit checkpoint or the first existing machine default.

    We fail locally instead of silently triggering a gated Hugging Face download.
    This keeps long multi-GPU pilots deterministic and gives the user the exact
    path at which the weight file is expected.
    """

    if checkpoint is not None:
        resolved = checkpoint.expanduser().resolve(strict=False)
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM checkpoint does not exist: {resolved}")
        return resolved
    candidates = default_checkpoint_candidates(sam_version)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        if resolved.is_file():
            return resolved
    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No local {sam_version} checkpoint was found. Searched:\n  - {searched}\n"
        "Place the checkpoint at the first path, set SAM3_CHECKPOINT/SAM31_CHECKPOINT, "
        "or pass --checkpoint explicitly."
    )
