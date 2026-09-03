# HumanTouch SAM3 Head-Camera BBoxes

This pipeline discovers only:

```text
X*/videos/chunk-*/observation.images.cam_head/episode_*.mp4
```

Left/right wrist-camera videos are never added to the tracking manifest. The
head videos contain two instrumented gloves, so SAM3 uses the `gloved` prompt
without an additional glove-vs-bare semantic filter. At the first reliable
two-track observation, the screen-left track is assigned to `left` and the
screen-right track to `right`; track identity is retained afterward.

Run the complete bbox-only pipeline on eight GPUs:

```bash
./preprocess/humantouch/run_sam3_head_bboxes.sh all
```

This command is resumable per episode and writes tracking state under
`/home/ma-user/work/hy/humantouch-sam3-head-v1`. Materialized bbox JSON files
are written under:

```text
/home/ma-user/work/hy/humantouch-processed/
  bboxes_sam3_head_gloved_screen_order_v1/X*/episode_*.json
```

Missing or ambiguous frame-level boxes remain null and must be excluded by a
future HDF5/query-manifest builder. This stage neither creates HDF5 files nor
changes an active dataset path.

Useful individual commands:

```bash
./preprocess/humantouch/run_sam3_head_bboxes.sh build
./preprocess/humantouch/run_sam3_head_bboxes.sh track
./preprocess/humantouch/run_sam3_head_bboxes.sh status
./preprocess/humantouch/run_sam3_head_bboxes.sh audit
./preprocess/humantouch/run_sam3_head_bboxes.sh materialize
```
