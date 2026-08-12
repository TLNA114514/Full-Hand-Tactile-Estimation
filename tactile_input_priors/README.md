# Tactile Input Priors

This directory now contains only the reusable, offline MoGe depth-sidecar
pipeline. The completed VLM V1-V6 probes and the failed tactile depth adapters
have been retired from source; their generated reports, sidecars, data, and
presentation remain untouched.

Research conclusions and the reason for this cleanup are preserved in
[HISTORY.md](HISTORY.md). The broader image-to-canonical diagnosis remains in
[IMAGE_TO_CANONICAL_CORRESPONDENCE_DIAGNOSIS.md](../hamer_tactile_ft/IMAGE_TO_CANONICAL_CORRESPONDENCE_DIAGNOSIS.md).

## Source Layout

- `depth_teacher.py`: lazy offline MoGe teacher wrapper.
- `hdf5_manifest.py`: source HDF5 reader, manifest normalization, hashing,
  atomic JSON/JSONL publication, and finalized-container verification.
- `resolve_depth_manifests.py`: discover or rebuild authoritative query
  manifests from finalized sequence HDF5 files.
- `build_depth_sidecars.py`: resumable, sequence-sharded MoGe point/normal
  extraction.
- `depth_sidecar.py`: versioned sidecar schema, reader, warp, inspection, and
  coverage validation.
- `run.sh`: the only supported command entry point.

`run.sh` deliberately has no tactile training or evaluation modes. It does not
import `train.py`, `hamer_tactile.py`, or `eval_tactile_fast.py`.

## Runtime Data Safety

Keep expensive models, sidecars, and logs outside the source checkout:

```bash
export INPUT_PRIOR_ROOT=/home/ma-user/work/cfzhao/input_prior_full
export DEPTH_SIDECAR_ROOT="$INPUT_PRIOR_ROOT/depth_sidecars"
export DEPTH_BUILD_LOG_DIR="$INPUT_PRIOR_ROOT/logs/depth_build"
mkdir -p "$DEPTH_SIDECAR_ROOT" "$DEPTH_BUILD_LOG_DIR"
```

The launcher executes from a private temporary copy so source synchronization
cannot replace the running shell script. Sidecar shards are written atomically
and are resumable at sequence granularity. Source synchronization must continue
to exclude runtime data roots; the repository-level `sync.sh` and
`sync_to_server.sh` remain the synchronization authorities.

## Environment

Use the existing Depth environment and MoGe checkpoint:

```bash
export DEPTH_PYTHON=/home/ma-user/work/cfzhao/input_prior_step0/envs/depth/bin/python
export MOGE_MODEL=/home/ma-user/work/cfzhao/input_prior_step0/models/moge-2-vitl-normal/model.pt
```

The Python environment needs MoGe, PyTorch, NumPy, OpenCV, and h5py. MoGe is
imported only by the builder; manifest discovery and sidecar validation do not
initialize CUDA.

## 1. Resolve Manifests

Inspect or atomically create manifests from finalized sequence HDF5 metadata:

```bash
./tactile_input_priors/run.sh depth-manifests \
  --dataset touchanything \
  --splits auto \
  --create-missing \
  --print-paths
```

Supported datasets are `touchanything` and `opentouch`. Auto-detection checks
the environment-specific root first, followed by the known shared roots:

```text
TOUCHANYTHING_DATA_ROOT
OPENTOUCH_DATA_ROOT
```

Pass `--processed-root /path/to/root` when discovery is ambiguous. Manifest
publication uses an atomic lock directory and never scans legacy JPEG sample
trees.

## 2. Build Sidecars

Build one manifest on one GPU:

```bash
./tactile_input_priors/run.sh depth-build \
  --manifest /path/to/touchanything_train.queries.jsonl \
  --model "$MOGE_MODEL" \
  --output-dir "$DEPTH_SIDECAR_ROOT" \
  --device cuda:0
```

Build one explicit manifest over eight GPUs:

```bash
DEPTH_GPUS=0,1,2,3,4,5,6,7 \
./tactile_input_priors/run.sh depth-build-8gpu \
  --manifest /path/to/touchanything_train.queries.jsonl \
  --model "$MOGE_MODEL" \
  --output-dir "$DEPTH_SIDECAR_ROOT"
```

Resolve and build every requested split in one command:

```bash
DEPTH_DATASET=touchanything \
DEPTH_SPLITS=train,val,test_seen,test_unseen \
DEPTH_GPUS=0,1,2,3,4,5,6,7 \
./tactile_input_priors/run.sh depth-build-auto-8gpu \
  --model "$MOGE_MODEL" \
  --output-dir "$DEPTH_SIDECAR_ROOT"
```

Set `DEPTH_DATA_ROOT=/custom/processed/root` if needed. Complete compatible
sequence shards are reused. Incomplete or provenance-incompatible shards fail
explicitly unless `--overwrite-incompatible` is supplied.

## 3. Validate Sidecars

Validate coverage and provenance against an authoritative manifest:

```bash
./tactile_input_priors/run.sh depth-validate \
  "$DEPTH_SIDECAR_ROOT" \
  --manifest /path/to/touchanything_train.queries.jsonl
```

Add `--deep` to read every stored array and verify finite values. Sidecar
contracts include the sequence identity, query rows, model SHA256, manifest
SHA256, extraction config SHA256, affine convention, and completion state.

## Removed Interfaces

The following are intentionally unsupported:

```text
VLM V1-V6 extraction/probe commands
depth adapter training/evaluation commands
legacy run_input_prior_step0*.sh launchers
```

Existing outputs and the checked-in `VLM_Depth_Probe_Summary_CN.pptx` are
historical artifacts, not active code dependencies.
