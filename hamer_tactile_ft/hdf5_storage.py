#!/usr/bin/env python3
"""Compatibility facade for the sequence-level HDF5 backend.

Both ``import hdf5_storage`` (script execution) and
``import hamer_tactile_ft.hdf5_storage`` (package import) remain supported.
"""

if __package__:
    from .data.hdf5_backend import *  # noqa: F401,F403
else:
    from data.hdf5_backend import *  # noqa: F401,F403
