# AceData SAM3 BBox Reconstruction

This pipeline tracks both gloved hands in every AceData `stereo1.mp4` using
SAM3. It assigns the first reliable screen-left track to `left` and the
screen-right track to `right`; SAM object IDs and legacy boxes do not determine
hand identity.

```bash
./preprocess/acedata/run_sam3_reconstruction.sh all
```

The run is resumable per clip across all eight GPUs. Outputs are first written
to `acedata-processed/bboxes_sam3_gloved_screen_order_v1`. Activation only
occurs after every tracker output can be associated and audited. The original
`bboxes` directory is moved to `bboxes_vitdet_kcf_v1`, then `bboxes` becomes a
symlink to the versioned SAM3 result.

Missing or ambiguous hand boxes stay null. They are not filled from the old
detector and must be marked non-trainable when HDF5/query manifests are built.
The existing `samples/all/*/meta.json` files embed the old boxes and are not
rewritten by this pipeline; do not use those files for the next manifest.

## Train-only sequence HDF5

After SAM3 boxes have been reviewed and activated, build the training backend:

```bash
./preprocess/acedata/run_hdf5_conversion.sh build
./preprocess/acedata/run_hdf5_conversion.sh status
```

This is a CPU/shared-filesystem conversion; `ACEDATA_HDF5_WORKERS=8` controls
parallel sequence writers and does not allocate GPUs. The default
`ACEDATA_IMAGE_SOURCE=video` sequentially decodes each canonical MP4 and avoids
random metadata lookups in the multi-million-entry legacy sample directory.
`legacy_jpeg` remains available only as a reproducibility control.

The conversion is resumable across 494 sequences. It rebuilds every query from
the synchronized pressure NPZ and current SAM3 bbox JSON, while reusing legacy
`image.jpg` only as an encoded RGB cache. A query is included only when both
its pressure sensor row and SAM3 bbox are valid. Original source frame indices
and timestamps are retained even though stored HDF5 frame rows are compact.
Publication additionally requires the audited `1,987,236` train-query count;
if it differs, no official manifest is left behind.

Published files are:

```text
acedata-processed-hdf5/manifests/acedata_train.queries.jsonl
acedata-processed-hdf5/manifests/acedata_train.sequences.jsonl
acedata-processed-hdf5/manifests/acedata_train.summary.json
```

No AceData validation or test manifest is created. To train OT + TA +
EgoTactile + AceData while keeping validation strictly OT + TA:

```bash
./hamer_tactile_ft/run_tactile_experiment.sh fullgrid-coreloc-four-domain \
  --gpus 0,1,2,3,4,5,6,7
```

Useful commands:

```bash
./preprocess/acedata/run_sam3_reconstruction.sh status
./preprocess/acedata/run_sam3_reconstruction.sh track
./preprocess/acedata/run_sam3_reconstruction.sh audit
```
