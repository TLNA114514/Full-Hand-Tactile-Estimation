# Tactile Fine-Tuning

This directory contains the active single-frame tactile training and evaluation
pipeline. Historical experimental decoders are intentionally not kept in the
runtime surface; their reports and Git history remain available.

## Active Entry Points

```text
run_tactile_experiment.sh -> run_tactile_ft.sh -> train.py
run_eval_matrix.sh        -> eval_tactile_fast.py
demo_tactile_video.py     -> video/demo inference
```

The maintained model surface is deliberately small:

```text
dense_v2                 final DINO feature control
dense_v2_dino_rezero     multilevel DINO ReZero baseline

pool_layout=legacy5|fullgrid32
```

FullGrid32, CoreLoc, and the `hump|plateau|capped_linear` pressure-weight
choices remain supported. OpenTouch, TouchAnything, and mixed datasets can be
trained and evaluated independently; dataset identity is never a model input.

## Supporting Modules

- `data/`: index/manifest/cache and sequence-HDF5 infrastructure.
- `dataset.py`: dataset assembly, query crop, augmentation, and target loading.
- `losses.py`: the maintained dense tactile objective.
- `tactile_metrics.py`: shared train/eval metrics.
- `process_supervisor.py`: process-group lifecycle and orphan cleanup.
- `wandb_epoch_sync.py`: resumable epoch-level WandB synchronization.
- `audit_sequence_failures.py`: maintained prediction/data-integrity audit.

Data preprocessing lives in [`preprocess/`](../preprocess/). SAM3 query bbox
construction lives in
[`sam3_bbox_reconstruction/`](../sam3_bbox_reconstruction/). Reusable offline
depth sidecars live in [`tactile_input_priors/`](../tactile_input_priors/).

## Removed Runtime Routes

The following completed experiments were removed from the active code after
their controlled comparisons failed to beat the FullGrid baseline:

```text
region/source/ordinal heads
DPT/progressive fusion variants
vertex cross-attention (all versions)
deformable attention and its custom CUDA extension
CSE scatter
DepthPN/DepthLocal/DepthContact adapters
Tail L1 auxiliary loss
legacy input-prior Step-0/VLM probe launchers
```

Old checkpoints that require those implementations are no longer guaranteed to
reconstruct. Existing report directories are unaffected.
