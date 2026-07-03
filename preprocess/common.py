from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "preprocess" / "artifacts"


def append_default_arg(argv, flag, value):
    if flag not in argv:
        argv.extend([flag, str(value)])


def append_default_bool(argv, flag):
    if flag not in argv:
        argv.append(flag)
