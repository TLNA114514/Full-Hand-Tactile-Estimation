#!/bin/bash
# Convert the dataset to HDF5 format (task-level normalization enabled by default)
#
# Task-level normalization:
#   - Automatically scans pressure data from all trajectories under each task
#   - Uses the 99th percentile as task_vmax and normalizes values to [0, 1]
#   - Stores the result in HDF5 as pressure.attrs['normalized']=True and pressure.attrs['task_vmax']=xxx
#   - Avoids per-clip normalization during training and preserves temporal consistency
#
# To disable normalization, add --no_task_normalize

# Usage:
#   1) Full conversion (default):
#        bash scripts/run_convert_to_hdf5.sh
#   2) Regenerate trajectories listed in bad_files.txt (only read_error by default):
#        bash scripts/run_convert_to_hdf5.sh ./datasets/TouchAnything_hdf5_opensource_new/bad_files.txt
#   3) Regenerate all reasons from bad_files.txt:
#        BAD_REASON_FILTER="read_error,too_short,low_wilor_valid_ratio" bash scripts/run_convert_to_hdf5.sh ./datasets/TouchAnything_hdf5_opensource_new/bad_files.txt

BAD_FILES_LIST=${1:-""}
BAD_REASON_FILTER=${BAD_REASON_FILTER:-"read_error"}

EXTRA_ARGS=""
if [ -n "$BAD_FILES_LIST" ]; then
  EXTRA_ARGS="--bad_files_list $BAD_FILES_LIST --bad_reason_filter $BAD_REASON_FILTER --delete_before_regen"
fi

python scripts/core/convert_to_hdf5.py \
    --root_path ./datasets/EgoTouch \
    --batch \
    --output ./datasets/EgoTouch_hdf5 \
    --traj_workers 64 \
    --compression_level 4 \
    --skip_existing \
    $EXTRA_ARGS \
