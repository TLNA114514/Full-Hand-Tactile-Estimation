"""TTY-aware progress helpers shared by the reconstruction stages."""

from __future__ import annotations

import sys

from tqdm.auto import tqdm


def progress(iterable=None, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("disable", not sys.stderr.isatty())
    return tqdm(iterable, **kwargs)
