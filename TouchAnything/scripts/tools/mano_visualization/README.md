# MANO Visualization Assets

This directory stores the minimal MANO-visualization assets used by TouchAnything.

## Included Files

- `pyrenderer.py`
- `ta_to_mano_mapping_left_visual.json`
- `ta_to_mano_mapping_right_visual.json`
- `scratch/mano_right_neutral_subdiv.obj`

## Usage

These files are only used by the optional MANO 3D visualization branches:

- `scripts/core/visualize_hdf5.py`
- `scripts/core/inference_tactile_parallel_mano_style.py` with `pressure_style=mano_3d`
- MANO rendering helpers in `scripts/core/load_data.py`

## EasyMocap Dependency

- `third_party/EasyMocap/` is the primary EasyMocap source.
- `J_regressor_mano_LEFT.txt` and `J_regressor_mano_RIGHT.txt` are read from:

```text
third_party/EasyMocap/data/smplx/
```

## Notes

- This directory only keeps the small subset of assets currently required by TouchAnything.
- It is intentionally named by function rather than by the upstream project name.
