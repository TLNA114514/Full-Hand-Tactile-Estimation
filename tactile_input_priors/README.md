# Tactile Input Priors

This directory owns the complete input-prior subsystem: reusable MoGe
sidecars, immutable frozen-feature caches, feature-level Depth/VLM adapters,
adapter-only checkpoints, training, and evaluation. The mature RGB tactile
model in `hamer_tactile_ft` is imported as a frozen dependency and is not
modified by these experiments.

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
- `prior_adapters.py`: bounded Depth spatial and VLM low-rank feature adapters.
- `prior_model.py`: permanently frozen RGB base plus one trainable adapter.
- `feature_cache.py`: atomic, resumable mmap cache with SHA provenance.
- `cache_tactile_features.py`: precompute frozen RGB grids, bottlenecks, logits,
  and aligned depth grids.
- `runtime.py`: the sole bridge for reconstructing the frozen tactile base and
  HDF5 datasets.
- `train_prior_adapter.py`: independent DDP adapter training.
- `eval_prior_adapter.py`: independent base-vs-fused evaluation.
- `run.sh`: the only supported command entry point.

New prior implementation never lives in `hamer_tactile_ft`. Shared loss,
dataset, frozen model, and metric code are treated as stable dependencies.

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
legacy direct-logit Depth/VLM adapter commands
legacy run_input_prior_step0*.sh launchers
```

Existing outputs and the checked-in `VLM_Depth_Probe_Summary_CN.pptx` are
historical artifacts, not active code dependencies.

## Feature-Level Adapters

Both adapters load a compact `tactile_trainable_v2` checkpoint, permanently
freeze DINO/ReZero/FullGrid/decoder, and optimize only the adapter:

```text
Depth: aligned point/normal grid + z_rgb -> multiplicative spatial delta
VLM:   cached semantic vector + h_rgb   -> low-rank multiplicative delta
```

Feature deltas have a per-sample RMS budget of `0.05`; the resulting tactile
logit delta is smoothly capped at `+/-0.5`. Evaluation always reports the
unchanged frozen base and fused prediction from the same forward pass.

## One-Command Pipelines

The high-level interface is split into preparation, training, and evaluation.
Completed sidecars and caches are reused, and an atomic preparation lock
prevents two shared servers from building the same inputs concurrently.
Training updates an adapter-only exact `resume.ckpt` after validation each
epoch and reconnects to the same WandB run after restart. Epoch metrics first
enter a durable local queue; an independent uploader retries WandB without
letting a network failure interrupt training. Evaluation can read the
atomically written `loss-best.ckpt` while training continues.

Prior training defaults to `8` train workers, `4` validation workers, and one
prefetched batch per worker. With batch 128 this avoids the large pinned-memory
and HDF5 contention caused by the main trainer's older `32/16 x prefetch 2`
loader settings. Explicit CLI values still override these defaults. Every
training entry is supervised as one process group, so Ctrl+C terminates DDP
ranks, DataLoader workers, and the WandB uploader together while preserving the
last completed atomic `resume.ckpt` and queued epoch metrics.

Runtime I/O diagnostics can be enabled with `--runtime-debug`. They are written
under `<experiment>/runtime_debug/` and never introduce DDP collectives or
per-step WandB traffic:

```text
batch_timing_rank_XX.csv  loader gaps, step time, unique HDF5/sequence count,
                         source/depth HDF5, JPEG, crop, and warp latency
system_io.csv             process-state counts, Linux I/O PSI, dirty/writeback
process_d_waits.csv       every observed D-state PID, kernel wchan, and I/O bytes
system_io_summary.json    top D-state wait channels and processes
```

The rank CSVs buffer 64 steps before writing. The system monitor samples every
two seconds and flushes every 30 seconds, so its own storage traffic remains
small. Diagnostics are opt-in because per-sample stage timers still have some
CPU cost. Tune them with `--runtime-debug-interval` and
`--runtime-debug-flush-steps`.

Online prior loading also defaults to `--hdf5-batch-read-mode streaming`.
Streaming reads, decodes, crops, and releases one full-resolution frame at a
time while retaining the final cropped batch. The historical grouped mode first
materializes every decoded source frame in a worker batch; with batch 128 this
can evict useful filesystem cache pages and multiply resident memory across 64
workers. `--hdf5-batch-read-mode grouped` remains available as an explicit
locality benchmark and does not change model inputs or targets.

With the online defaults, `8 workers x prefetch_factor 1` means exactly eight
batches can be in flight per rank. A stall repeating every eight steps together
with DataLoader workers in Linux `D` state is therefore strong evidence that
the prefetch queue is draining while workers wait in kernel I/O. The debug CSVs
distinguish source sequence HDF5/JPEG latency from Depth-sidecar open/read and
CPU warp latency before changing worker counts or prefetch depth.

Inspect or terminate registered prior-training process groups with:

```bash
./tactile_input_priors/run.sh prior-process-list
./tactile_input_priors/run.sh prior-process-stop
```

An already-running job can be sampled without restarting it. This auto-attaches
to the newest active supervised training session on the current host and runs
until Ctrl+C:

```bash
./tactile_input_priors/run.sh prior-debug-monitor
```

Depth coverage is fully validated once per manifest and then recorded in an
atomic stamp outside the source tree. Unchanged manifest metadata, summary,
sidecar root, and validator code produce a fast cache hit on later training
starts. Set `FORCE_DEPTH_VALIDATE=1` for an intentional full coverage recheck.

The current TouchAnything crop-1.2 FullGrid `best_loss.ckpt` and the known
server paths for DINO, MoGe, Qwen, Python environments, cache, experiments,
and reports have defaults. Override the checkpoint only when intentionally
testing a different frozen base:

```bash
export TACTILE_BASE_CHECKPOINT=/path/to/another/fullgrid32/best_loss.ckpt
```

Online RGB/DINO workflow:

```bash
./tactile_input_priors/run.sh prepare-online depth-real
./tactile_input_priors/run.sh train-online depth-real
./tactile_input_priors/run.sh eval-online depth-real loss-best
```

Deterministic cache-only workflow, which disables crop augmentation and skips
JPEG/HDF5/DINO work during training and evaluation:

```bash
./tactile_input_priors/run.sh prepare-cache-only vlm-real
./tactile_input_priors/run.sh train-cache-only vlm-real
./tactile_input_priors/run.sh eval-cache-only vlm-real loss-best,last
```

Run one command per server, such as real on Server A and its shuffled control
on Server B. Runtime outputs stay under
`/home/ma-user/work/cfzhao/input_prior_full`, outside source synchronization.
The original `pipeline-online` and `pipeline-cache-only` modes remain as
one-command prepare/train/evaluate shortcuts. The low-level commands below
remain available for inspection and partial runs. WandB defaults to project
`tactile-priors-v2`.

Required common paths:

```bash
export TACTILE_PYTHON=/home/ma-user/work/cfzhao/tactile/bin/python
export DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
export TACTILE_BASE_CHECKPOINT=/path/to/fullgrid/loss-best.ckpt
export PRIOR_EXPERIMENT_ROOT=/home/ma-user/work/cfzhao/input_prior_full/experiments
```

Depth real/shuffle pair:

```bash
export DEPTH_SIDECAR_ROOT=/home/ma-user/work/cfzhao/input_prior_full/depth_sidecars

./tactile_input_priors/run.sh train-depth-real \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 128

./tactile_input_priors/run.sh train-depth-shuffle \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 128
```

VLM real/context-shuffle pair consumes a finalized feature cache containing
`vlm_embedding`:

```bash
export VLM_PYTHON=/home/ma-user/work/cfzhao/input_prior_step0/envs/vlm/bin/python
export VLM_MODEL=/home/ma-user/work/cfzhao/input_prior_step0/models/Qwen3-VL-Embedding-2B
export QWEN_EMBED_CODE_ROOT=/home/ma-user/work/cfzhao/input_prior_step0/code/Qwen3-VL-Embedding
export VLM_CACHE_ROOT=/home/ma-user/work/cfzhao/input_prior_full/cache/qwen_fullframe
export VLM_PRIOR_DIM=2048

./tactile_input_priors/run.sh cache-vlm-auto-8gpu

export VLM_FEATURE_CACHE_TRAIN="$VLM_CACHE_ROOT/train"
export VLM_FEATURE_CACHE_VAL="$VLM_CACHE_ROOT/val"

./tactile_input_priors/run.sh train-vlm-real \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 128

./tactile_input_priors/run.sh train-vlm-control \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 128
```

The builder reads full RGB frames from the authoritative HDF5 query manifests,
deduplicates repeated frame queries within each inference batch, and stores one
2048-dimensional Qwen vector per `sample_uid`. It is resumable at cache-shard
granularity. Online adapter training still uses this semantic cache: "online"
means that the RGB crop augmentation and frozen DINO path are recomputed, not
that the 2B VLM is run inside every training step.

The VLM dimension must match the cached embedding exactly. A generic aligned
NPZ can alternatively be converted without adding source code:

```bash
./tactile_input_priors/run.sh cache-build \
  --manifest /path/to/sample_uids.jsonl \
  --sample-id-key sample_uid \
  --input-npz /path/to/aligned_vlm_embeddings.npz \
  --npz-sample-id-key sample_uid \
  --cache-dir /path/to/vlm_cache \
  --field vlm_embedding:float16:1536
```

## Frozen Feature Cache

Online mode retains crop augmentation and recomputes frozen DINO features.
Fixed-cache mode removes DINO from every training step and avoids repeated
decoder-base work, but must use deterministic crops:

```bash
export BASE_FEATURE_CACHE=/path/to/base_cache

CACHE_GPUS=0,1,2,3,4,5,6,7 \
./tactile_input_priors/run.sh cache-tactile-8gpu \
  --cache-dir "$BASE_FEATURE_CACHE" \
  --datasets touchanything \
  --split train \
  --fields z_rgb,depth_grid,tactile_signal,has_tactile \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT" \
  --batch-size 128
```

The command writes `part-00-of-08` through `part-07-of-08`; the partition root
is detected automatically. Build separate roots for `train`, `val`, and each
test split. With the target/mask fields above, `--cache-only` avoids both DINO
and JPEG/HDF5 reads. Training with a base cache requires deterministic crops,
because pairing a cached RGB grid with a new random crop would be geometrically
invalid; the short presets enforce this automatically.

Cache integrity can be checked independently:

```bash
./tactile_input_priors/run.sh cache-self-check
./tactile_input_priors/run.sh cache-verify --cache-dir /path/to/one/partition --deep
```

## Evaluation

Each experiment saves only adapter parameters under
`$PRIOR_EXPERIMENT_ROOT/EXP/checkpoints/{loss-best,last}.ckpt` (default:
`/home/ma-user/work/cfzhao/input_prior_full/experiments`). Runtime data stays
outside the source checkout so code synchronization cannot delete it.
Evaluate one checkpoint/split with:

```bash
./tactile_input_priors/run.sh eval-prior \
  --checkpoint "$PRIOR_EXPERIMENT_ROOT/ta_depth_feature_real/checkpoints/loss-best.ckpt" \
  --split test_seen \
  --query-manifests /path/to/touchanything_test_seen.queries.jsonl \
  --output-dir /home/ma-user/work/cfzhao/input_prior_full/reports/ta_depth_feature_real/loss-best/test_seen \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT" \
  --device cuda:0
```

The report contains `metrics.json`, a concise `eval.txt`, optional per-frame
CSV, and an automatic copy of the experiment's `val_metrics.csv`. No legacy
audit suite is run by default.
