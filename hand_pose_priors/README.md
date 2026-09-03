# Hand Pose Priors

This directory owns the reproducible HaMeR/Dyn-HaMR preparation and export
code. Large weights, generated sidecars, previews, and logs remain machine-local
under ignored `_DATA`, `outputs`, `sidecars`, and `logs` directories.

## HaMeR Stage 0

The remote runtime reuses the existing CUDA-compatible Python environment:

```text
/home/ma-user/work/cfzhao/tactile/bin/python
```

The numerical inference path deliberately disables HaMeR's internal renderer,
so EGL/OSMesa is not required. Existing SAM3 boxes and handedness are passed
directly to HaMeR; ViTPose and the person detector are not rerun.

Prepare or validate model assets:

```bash
./hand_pose_priors/run.sh setup-hamer
```

Run an end-to-end TouchAnything HDF5 + SAM3 smoke test:

```bash
CUDA_VISIBLE_DEVICES=7 ./hand_pose_priors/run.sh smoke-hamer
```

The smoke test writes:

```text
hand_pose_priors/outputs/hamer_smoke/summary.json
hand_pose_priors/outputs/hamer_smoke/hamer_sam3_smoke.npz
hand_pose_priors/outputs/hamer_smoke/hamer_sam3_overlay.jpg
```

The NPZ preserves camera-space MANO vertices/joints, full-image UV, crop-1.2
UV, camera parameters, and raw right-canonical MANO parameters. Left-hand
geometry is mirrored back into the source camera view, while raw MANO pose is
explicitly labelled as right-hand canonical.

## Full TouchAnything HaMeR Sidecar

The full exporter is designed for the 3.65M TouchAnything hand queries. It:

- consumes the four HDF5 query manifests and existing SAM3 hand boxes;
- preserves the fixed MANO-778 vertex order needed for later image-to-vertex routing;
- stores right-canonical MANO rotations/betas and source-camera translation;
- stores source-camera MANO-778 vertices as FP16 by default;
- does not store projected UV or expanded 13,614-vertex geometry;
- preserves rows without a SAM3 box as explicit `missing_sam3_bbox` records;
- uses atomic 8,192-record HDF5 shards with SHA-256 completion markers;
- reuses completed shards when the same command is rerun after interruption;
- runs one persistent HaMeR model per GPU and persistent CPU preprocessing workers.

The default machine-local destination is:

```text
/home/ma-user/work/cfzhao/hand_pose_sidecars/touchanything_hamer_v1
```

This path is outside the repository and is therefore unaffected by source-code
sync. Override it with `HAMER_SIDECAR_ROOT` when needed.

### Preflight

Run a small two-GPU export before committing all eight GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
HAMER_GPUS=2 \
HAMER_SIDECAR_ROOT=/home/ma-user/work/cfzhao/hand_pose_sidecars/hamer_preflight \
./hand_pose_priors/run.sh build-hamer \
  --max-samples 512 \
  --shard-size 256 \
  --batch-size 64

HAMER_SIDECAR_ROOT=/home/ma-user/work/cfzhao/hand_pose_sidecars/hamer_preflight \
./hand_pose_priors/run.sh verify-hamer --deep
```

### Full eight-GPU export

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HAMER_GPUS=8 \
./hand_pose_priors/run.sh build-hamer
```

The exact same command resumes from finalized shards. Runtime-only knobs such
as `--batch-size` and `--workers` may be changed between resumes; semantic
knobs such as precision, bbox scale, shard size, or stored fields may not.

Rows with no usable SAM3 box do not abort the full export. By default they keep
their original `(split, source_row, sample_uid)` position, receive
`quality/status=missing_sam3_bbox`, and bypass image decoding and HaMeR. This
keeps the sidecar exactly aligned with the source manifests without silently
using a different crop. Two explicit alternatives are available:

```bash
# Audit mode: stop during preparation if even one SAM3 box is missing.
./hand_pose_priors/run.sh build-hamer --unresolved-bbox-policy error

# Use a valid manifest bbox even when its SAM3 provenance cannot be verified.
./hand_pose_priors/run.sh build-hamer --unresolved-bbox-policy manifest_fallback
```

`manifest_fallback` is intentionally opt-in because it weakens the bbox
provenance contract.

Useful variants:

```bash
# Lower peak GPU memory.
./hand_pose_priors/run.sh build-hamer --batch-size 64

# Repair and rebuild a damaged finalized shard.
./hand_pose_priors/run.sh build-hamer --repair-invalid-shards

# Compact 3-5 GB form: omit per-frame MANO-778 camera vertices.
HAMER_SIDECAR_ROOT=/home/ma-user/work/cfzhao/hand_pose_sidecars/touchanything_hamer_compact_v1 \
./hand_pose_priors/run.sh build-hamer --no-store-camera-vertices
```

Inspect progress or perform final validation:

```bash
./hand_pose_priors/run.sh status-hamer
./hand_pose_priors/run.sh verify-hamer
./hand_pose_priors/run.sh verify-hamer --deep
./hand_pose_priors/run.sh visualize-hamer
```

The visualization command deterministically samples both hands from every split,
renders the stored MANO mesh in the source frame and crop-1.2 view, and also
renders the lowest in-frame-quality cases. Outputs default to
`$HAMER_SIDECAR_ROOT/audits/random_visualization`.

`verify-hamer --deep` rereads and hashes every shard, so it is intended for the
final integrity pass rather than every resume.

### Stored schema

Each shard contains:

```text
queries/sample_uid, source_row, query_row, frame_row
queries/is_right, bbox_xyxy, bbox_score, bbox_source_code
camera/image_wh, focal_length, translation
mano/global_orient [1,3,3], hand_pose [15,3,3], betas [10]
geometry/vertices_camera [778,3] float16       # default, optional
quality/status, positive_depth_fraction, in_frame_fraction
```

Status values are `1=valid`, `2=nonfinite`, and `3=missing_sam3_bbox`.

The MANO parameters remain in HaMeR's right-hand canonical convention. For a
left-hand sample, `vertices_camera` is already mirrored into the original
source-image camera. This distinction is recorded in every shard.

Use `hand_pose_priors.pose_sidecar.HaMeRPoseSidecar` to access a record by
`(split, source_row)`. Passing `derive_uv=True` reconstructs full-image UV and
crop-1.2 UV in memory. No per-sample NPZ, OBJ, preview, projected-UV cache, or
dense 13,614-point cache is created.

At 3,653,778 records, the default uncompressed numeric payload is about 20 GB;
allowing for HDF5 metadata and shard markers, budget roughly 22-27 GB. The
compact `--no-store-camera-vertices` variant is expected to occupy 3-5 GB.

## Dyn-HaMR Short-Sequence Trial

The Dyn-HaMR integration deliberately starts with a compact, inspectable trial
instead of materializing TouchAnything as millions of image files. It pins the
official source commit, reuses the HaMeR sidecar as initialization, derives the
21 MANO/OpenPose joints from stored MANO-778 camera vertices, and extracts only
one 120-frame bilateral sequence from the source HDF5.

Setup normally uses a sparse Git checkout. Servers with older Git may instead
use a source-only snapshot carrying `.dynhamr_commit`; the marker must exactly
match the pinned official commit before any code runs.

The compatibility patch removes Dyn-HaMR's unused top-level `mano` import and
makes visualization imports lazy. The active hand geometry path uses the
repository's `body_model.MANO`, and headless optimization no longer requires a
system EGL library.

HaMeR's MANO assets are linked at both checkout-relative locations used by the
upstream Hydra path resolver; model files are never duplicated.

The upstream biomechanical loss is also constructed only when its configured
weight is nonzero. The pinned default uses `bio=0`, so the trial does not
silently require unavailable BMC statistics for an inactive objective.

The first run uses identity camera extrinsics and the same focal convention as
the HaMeR initialization. This isolates Dyn-HaMR's temporal hand optimization
from camera-SLAM errors. It is a temporal-refinement smoke test, not yet the
full dynamic-camera claim from the paper. A VIPE-camera pass should follow only
after this input and optimization contract passes.

Default machine-local paths:

```text
checkout: /home/ma-user/work/cfzhao/hand_pose_dynhamr/code/Dyn-HaMR
trial:    /home/ma-user/work/cfzhao/hand_pose_dynhamr/trials/touchanything_arrange_pillow_v3
run:      <trial>/outputs/static_focal_standard_v1
audit:    <run>/audit
```

The `v3` trial contract is intentional. It mirrors only canonical left-hand
MANO geometry and then adds the shared camera/world translation. Earlier trial
directories were prepared before that left-hand translation convention was
corrected and must not be reused.

Prepare the pinned checkout and the short HDF5-backed trial without using a
GPU:

```bash
./hand_pose_priors/run.sh setup-dynhamr
./hand_pose_priors/run.sh prepare-dynhamr
./hand_pose_priors/run.sh check-dynhamr
```

Run the official root and smooth optimization when a GPU is free, then audit
it against the exact per-frame HaMeR initialization:

```bash
DYNHAMR_GPU=0 ./hand_pose_priors/run.sh run-dynhamr
./hand_pose_priors/run.sh audit-dynhamr
```

The optimizer is resumable through Dyn-HaMR's `root_fit_*` and
`smooth_fit_*` checkpoints in the fixed run directory. The audit writes
`summary.json`, `per_frame.csv`, seven frame strips, and a full comparison
video. The key checks are source-view reprojection drift, absolute root/depth
drift, root-relative articulation drift, shape drift, and separate root versus
articulation temporal ratios. A large reduction in total 3D acceleration is
not evidence of better hand articulation when it comes only from root motion.

The first completed `5 + 20` iteration diagnostic is recorded in
[`DYNHAMR_TOUCHANYTHING_TRIAL.md`](DYNHAMR_TOUCHANYTHING_TRIAL.md). It validates
the adapter and rendering path, but it does not yet support replacing raw HaMeR
with Dyn-HaMR geometry.

Source: [official Dyn-HaMR repository](https://github.com/ZhengdiYu/Dyn-HaMR)
and [project page](https://dyn-hamr.github.io/).
