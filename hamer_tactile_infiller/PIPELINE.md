# Tactile Infiller Pipeline (Historical)

> This directory is documentation-only. The original prototype depended on a
> retired MLP/legacy5 interface, used identity-like inputs that conflict with
> the anonymous-query contract, and did not model real timestamp deltas. Its
> executable code was removed during the 2026-08 cleanup. Any future temporal
> model should be rebuilt on the current FullGrid/ReZero baseline.

This directory is intentionally isolated from `hamer_tactile_ft`. The old path still trains a single-frame bbox-crop tactile regressor. This path trains a sequence-level infiller that can learn from sparse hand detections.

## 1. Data Indexing

`build_sequence_manifest.py` scans extracted frame folders and writes hand-level sequence manifests:

```text
sequence_manifest_train.jsonl
sequence_manifest_val.jsonl
sequence_manifest_test.jsonl
```

Each row is one hand sequence:

```text
dataset / split / clip-or-sequence / hand
  frame 0: image path, bbox, bbox_valid, tactile key, tactile_valid
  frame 1: image path, bbox, bbox_valid, tactile key, tactile_valid
  ...
```

This means manifest row count is expected to be much smaller than extracted-frame directory count. For example, tens of thousands of per-frame sample dirs may become hundreds of clip/hand sequences, and then the dataset cuts those sequences into thousands of training windows.

Dataset grouping:

- OpenTouch: `scene + demo/clip + is_right`
- TouchAnything: `scene + task + clip + left/right`
- EgoTactile: `rel_seq + left/right`

EgoPressureDiff has no official train/val/test split in code. It uses `V2P_data/motion_list.txt` as a training sequence list. The default infiller path does not use that list; it scans the extracted-frame roots directly. The manifest builder only uses `motion_list.txt` when `--egotactile_split_source motion_list` or `derived` is explicitly selected.

The extracted scanner preserves every split it sees, including `train`, `val`, `test`, `test_seen`, and `test_unseen`. Manifest building can be parallelized with `--manifest_workers N`; this only parallelizes reading/parsing `meta.json` files and does not change the output format.

During training, validation uses the first non-empty split in this order:

```text
val -> test_seen -> test_unseen -> test
```

If none of those manifests produce valid sequence windows, training continues without validation.

## 2. Sequence Dataset

`TactileSequenceDataset` cuts long hand sequences into windows.

Important args:

- `seq_len`: frames per training window. Default: `16`.
- `seq_stride`: train window stride. Default: `8`.
- `eval_seq_stride`: eval window stride. Default: `16`.
- `sample_frame_rate`: temporal sampling interval. `1` means consecutive frames; `5` matches EgoPressureDiff-style sparse sampling.
- `min_observed_bbox`: minimum valid bbox frames inside a window. Default: `1`.
- `allow_missing_bbox`: keep frames without hand bbox. Default: enabled.
- `mask_prob`: randomly hide bbox-valid frames during training. Default: `0.5`.
- `target_policy`: frames used for tactile loss. Default: `has_tactile`.
- `missing_bbox_weight`: loss weight for naturally missing-bbox target frames. Default: `1.0`.
- `observed_bbox_weight`: loss weight for bbox-observed target frames. Default: `0.5`.
- `pressure_key_priority`: target pressure lookup order.

For a frame with valid bbox:

```text
RGB frame + bbox -> hand crop -> HaMeR frame tactile encoder
```

For a frame without valid bbox:

```text
img = 0
bbox_valid = 0
observed_mask = 0
target_mask = has_tactile
```

So missing-bbox frames do not run the frame encoder, but can still supervise the temporal infiller if tactile GT exists.

## 3. Model

The model has two parts:

```text
valid bbox frames
  -> frozen HAMER_Tactile backbone + tactile head
  -> observed tactile token + visual token

missing bbox frames
  -> zero observed tactile token + zero visual token

all frames
  -> add bbox/observed/target masks + dataset embedding + hand embedding + time embedding
  -> Transformer temporal encoder
  -> subdiv MANO tactile prediction for every frame
```

Training default:

1. Load an existing frame checkpoint.
2. Freeze HaMeR backbone, MANO head, and frame tactile head.
3. Train only the temporal infiller.
4. Optionally pass `--joint_finetune` to unfreeze the frame tactile head.

## 4. Loss and Metrics

Loss is computed on `target_mask == 1` frames:

```text
SmoothL1(pred, target) + 0.1 * BCEWithLogits(logits, target)
```

Then it applies:

- palm mask
- dataset-specific OpenTouch high-pressure downweighting
- missing/observed bbox frame weights
- optional temporal smoothness on adjacent target-valid frames

Validation reports:

- `all_*`: all target-valid frames
- `observed_bbox_*`: frames with bbox
- `missing_bbox_*`: frames without bbox

`missing_bbox_mae` is the main infiller metric.

## 5. What If A Clip Has No BBox At All?

Current v1 behavior:

- Manifest builder keeps the sequence frames if they exist in extracted data.
- Dataset window construction requires `min_observed_bbox >= 1` by default.
- Therefore a clip/window with zero bbox frames is skipped during infiller training.

Why this default is intentional:

- The current infiller is conditional on at least one visual hand observation.
- If a whole clip has no bbox, there is no anchor for hand identity, pose, location, or visible context.
- Training on such windows would push the model toward learning only dataset priors or average pressure dynamics.

Recommended handling:

1. Keep `min_observed_bbox=1` for the main infiller model.
2. Re-run extraction with a lower bbox threshold or a tracker/interpolator to recover at least sparse bbox anchors.
3. For EgoTactile, use `--keep_no_bbox` during preprocessing so no-bbox frames with tactile GT are kept, but still rely on neighboring bbox-valid frames in the same sequence.
4. If you specifically want zero-bbox clips, train a separate unconditional/metadata-conditioned prior model with `--min_observed_bbox 0`; do not mix it silently into the visual infiller objective.

Practical policy:

```text
clip has some bbox frames:
  use it; missing frames become infiller targets

clip has zero bbox frames but tactile GT:
  keep in manifest for audit
  skip by default in TactileSequenceDataset
  optionally train a separate prior-only model later

clip has zero bbox and no tactile GT:
  ignore
```

## 6. ASCII Pipeline

```text
Extracted frame folders
        |
        v
build_sequence_manifest.py
        |
        v
hand-level sequence JSONL
        |
        v
TactileSequenceDataset
        |
        +--> bbox-valid frames ----> HAMER_Tactile frame encoder ----+
        |                                                            |
        +--> bbox-missing frames --> zero observation tokens --------+
                                                                     |
                                                                     v
                                                       Temporal Transformer Infiller
                                                                     |
                                                                     v
                                               subdiv MANO tactile prediction per frame
                                                                     |
                                                                     v
                                    loss on all tactile-valid frames, including missing bbox
```
