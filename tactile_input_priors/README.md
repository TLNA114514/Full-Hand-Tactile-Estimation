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
- `prior_adapters.py`: legacy adapters plus causal Depth FiLM/local-attention adapters.
- `prior_model.py`: permanently frozen RGB base plus one trainable adapter.
- `selector_prior_adapters.py`: causal Depth/VLM contact-selector adapters.
- `selector_prior_model.py`: frozen pressure plus frozen Binary Grid selector contract.
- `train_prior_selector.py`: contact-only paired-control training.
- `eval_prior_selector.py`: aligned/control contact-information evaluation.
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

## Temporal Tactile Data Audit

Before adding a temporal backbone, audit whether the current HDF5 data has
enough reliable short-gap pairs and whether previous pressure contains useful
state. This scan never decodes RGB and does not rebuild a legacy index:

```bash
./tactile_input_priors/run.sh audit-tactile-dynamics \
  --dataset touchanything \
  --splits train,val,test_seen,test_unseen
```

The default deterministically reads 50,000 pressure pairs after a full
manifest-structure scan. Use `--max-pressure-pairs 0` for every eligible pair,
or `--stable-bbox-only` for the stricter crop-stability subset. Reports are
written under `$TACTILE_FLOW_AUDIT_ROOT` (by default
`/home/ma-user/work/cfzhao/input_prior_full/tactile_flow_audits`). They separate
pressure persistence, normalized-location persistence, contact birth/death,
transport candidates, and loading/release. These are data-usability and oracle
baselines; they do not by themselves establish a gain over the RGB model.

For the bimanual follow-up, compare exact source-frame lags and use the other
hand at the same lag as a counterfactual history:

```bash
./tactile_input_priors/run.sh audit-bilateral-tactile-dynamics \
  --dataset touchanything \
  --splits train,val,test_seen,test_unseen \
  --controlled-lags 1,2,4,8
```

This reports left/right canonical-query marginals, same-query versus
contralateral pressure persistence, raw SAM3 track/association continuity,
and bbox-based anonymous association ambiguity. The side field is used only to
construct the audit: the current input pipeline already flips left-hand RGB to
the canonical right-hand view. A future temporal state should therefore be
routed by an anonymous tracked instance A/B and reset on uncertain association,
instead of feeding semantic handedness into the tactile model. The default
samples 200,000 controlled-lag pairs; use
`--max-bilateral-pressure-pairs 0` only for a full pressure pass.

Pressure rows are pre-masked, converted once to the selected metric dtype, and
retained in a 512-row LRU by default. The scalar CPU path retains the historical
FP64 calculations. This avoids rereading the same
current/history rows for every lag and contralateral control without changing
any metric definition. `summary.json` records requests, cache hits, and actual
HDF5 row reads under `pressure_reader`. Use `--pressure-row-cache-size 0` to
disable it, or raise the value when sequence interleaving is unusually wide.

The full bilateral pressure pass also has an optional batched Torch backend.
It leaves manifest/HDF5 traversal sequential, but evaluates thousands of
6623-vertex pressure pairs at once on one GPU and merges all report marginals
with batched NumPy reductions instead of issuing per-pair metric updates:

```bash
CUDA_VISIBLE_DEVICES=0 \
./tactile_input_priors/run.sh audit-bilateral-tactile-dynamics-fast \
  --dataset touchanything \
  --splits train,val,test_seen,test_unseen \
  --controlled-lags 1,2,4,8
```

The fast wrapper defaults to outer HDF5 batches of 8192 pairs and GPU metric
microbatches of 32768 combined self/contralateral rows. The two controls are
independent: a large outer batch improves contiguous HDF5 reads, while a
smaller metric chunk prevents oversized 6623-vertex GPU intermediates. For the
current full TouchAnything audit, use:

```bash
CUDA_VISIBLE_DEVICES=0 \
BILATERAL_PRESSURE_BATCH_SIZE=32768 \
BILATERAL_PRESSURE_METRIC_CHUNK_SIZE=32768 \
./tactile_input_priors/run.sh audit-bilateral-tactile-dynamics-fast \
  --dataset touchanything \
  --splits train,val,test_seen,test_unseen \
  --controlled-lags 1,2,4,8
```

The outer batch is measured in pairs, while the metric chunk is measured in
combined self/contralateral rows. Thus a 32768-row metric chunk reproduces the
effective GPU batch of the older 16384-pair configuration. If it is slower or
uses too much memory, keep the outer batch at 32768 and benchmark metric chunks
of 24576 and 16384; do not infer their pair count directly from the number.

After selecting a signed-additive temporal checkpoint, run the cache-only
long-horizon audit before adding more history to training:

```bash
./tactile_input_priors/run.sh audit-tflow-long-horizon
```

The default uses the lag-1/2/4 `temporal-best` checkpoint and validation split.
An explicit checkpoint and split may be supplied as the first two arguments.
The audit chains only strict lag-1 edges to obtain lags `1/2/4/8/16/32` and
records the real cumulative time, minimum bbox IoU, maximum center jump, and
maximum area change for every lag. It additionally writes:

```text
lag_metadata.csv
conditional_incremental_gain.csv
history_direction.csv
trained_model_sweep.csv
```

`trained_model_sweep.csv` replays every non-empty subset of the checkpoint's
own lags for real history at residual scales `0/.25/.5/.75/1`. The full trained
lag set is then compared with an availability-matched cross-sequence history,
the opposite anonymous hand, and an explicit reset at the same scales. The
reset path must reproduce the frozen RGB prediction.
Long lags are audited as evidence only; this command does not silently add them
to the trained temporal model.

Use `--pressure-metric-device cpu` for the historical scalar FP64 path, or
`--pressure-metric-dtype float64` for a closer numerical cross-check on CUDA.
The GPU backend is intentionally batched rather than threaded h5py: h5py file
access is serialized internally, and parallel random readers can increase
shared-storage blocking. Within that one reader, missing rows are grouped by
HDF5 file and contiguous `query_row` ranges, so one slice replaces many Python
`dataset[row]` calls. A one-batch prefetch pipeline overlaps the next bulk read
with current CUDA metric computation. Disable only for diagnosis with
`--no-pressure-prefetch`.

Progress and `summary.json` report HDF5 rows versus actual read calls, bulk
fallbacks, and active `read / metric / aggregate` time. These active durations
may overlap when prefetch is enabled. This distinguishes HDF5/API preparation
from GPU work before adding more processes or GPUs.

### Predicted-History Replay

The GT persistence audit is an upper bound. Before training a temporal model,
export the frozen RGB baseline's own palm predictions and ask whether those
predictions contain usable state:

```bash
export TACTILE_BASE_CHECKPOINT=/path/to/fullgrid32/loss-best.ckpt
export DINO_WEIGHTS=/path/to/dinov3_vith16plus.pth

./tactile_input_priors/run.sh pipeline-tactile-history-replay
```

The pipeline resolves or creates the authoritative TouchAnything
`val/test_seen/test_unseen` HDF5 manifests, exports only the 6,623 valid palm
vertices on eight GPUs, fits one convex history coefficient per lag on
validation, and replays those fixed coefficients on seen and unseen. Prediction
artifacts are addressed by checkpoint SHA256 and manifest SHA256, so unchanged
inputs are reused while stale predictions cannot be selected silently.

Training and auditing can also be separated:

```bash
./tactile_input_priors/run.sh prepare-tactile-history-replay
./tactile_input_priors/run.sh audit-tactile-history-replay
```

The pair audit compares the current RGB prediction with previous predicted
pressure, the validation-selected blend, an all-zero history, a strict global
same-lag cross-sequence shuffle, and the previous GT oracle. On frames where
both anonymous hand queries exist, RGB-only, same-hand history, and opposite-
hand history are evaluated on the exact same rows. Two explicitly nondeployable
oracles report the upper bound from perfect per-frame gating and GT dynamics-
class gating. The continuous audit rolls the predicted state forward with an
EMA and resets it on frame/time gaps, association changes, or unstable boxes.
A useful temporal signal must beat both RGB-only and the matched zero/shuffle
controls; improvement over RGB alone can otherwise be explained by generic
smoothing or global pressure suppression.

Reports live under `$TACTILE_HISTORY_REPLAY_ROOT` and include
`alpha_selection.json`, `pair_replay_metrics.csv`, `rollout_metrics.csv`,
`target_dynamics_classes.csv`, `sequence_bootstrap.csv`, and `summary.json`.
The bootstrap table contains sequence-clustered confidence intervals for
history-versus-control deltas, including false-high excess and catastrophic
over/under rates. Use
`FORCE_TACTILE_HISTORY_AUDIT=1` to recompute metrics while retaining exported
predictions. Export and metric batch sizes are controlled by
`TACTILE_HISTORY_EVAL_BATCH_SIZE` and `TACTILE_HISTORY_PAIR_BATCH_SIZE`;
`TACTILE_HISTORY_BOOTSTRAP_ITERATIONS` controls confidence-interval cost.

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

All adapters load a compact `tactile_trainable_v2` checkpoint, permanently
freeze DINO/ReZero/FullGrid/decoder, and optimize only the adapter:

```text
Depth: aligned point/normal grid + z_rgb -> multiplicative spatial delta
VLM:   cached semantic vector + h_rgb   -> low-rank multiplicative delta
```

Feature deltas have a per-sample RMS budget of `0.05`; the resulting tactile
logit delta is smoothly capped at `+/-0.5`. Evaluation always reports the
unchanged frozen base and fused prediction from the same forward pass.

The causal Depth presets use a stricter contract: the conditioner cannot emit
an RGB-only correction, invalid Depth produces an exact zero delta, feature RMS
is limited to `0.02`, and the palm logit correction is zero-mean and capped at
`+/-0.25`. Training pairs aligned Depth with a deterministic per-sample spatial
shuffle. The aligned branch receives the unchanged tactile loss; the shuffled
branch is regularized back to the frozen base. This distinguishes aligned
geometry from adapter capacity or global pressure suppression.

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

Causal Depth dual-server workflow:

```bash
# Server A
./tactile_input_priors/run.sh train-online depth-film-cf

# Server B
./tactile_input_priors/run.sh train-online depth-xattn-cf

# After training, one command per experiment
./tactile_input_priors/run.sh eval-online depth-film-cf loss-best,last
./tactile_input_priors/run.sh eval-online depth-xattn-cf loss-best,last
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

The newer causal presets do not require a separately trained shuffle model.
Real and shuffled Depth are paired inside every training batch:

```bash
./tactile_input_priors/run.sh train-depth-film-cf --gpus 0,1,2,3,4,5,6,7
./tactile_input_priors/run.sh train-depth-xattn-cf --gpus 0,1,2,3,4,5,6,7
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
audit suite is run by default. Multi-GPU evaluation prints global sample
progress, throughput, and ETA every five seconds. Rank-local protocol results
are merged through atomic files rather than large NCCL object collectives;
evaluation therefore defaults to 8 loader workers and a prefetch factor of 1
per rank. Override these with `--num-workers`, `--prefetch-factor`, and
`--progress-interval` when needed.

Run all same-checkpoint Depth controls without retraining:

```bash
./tactile_input_priors/run.sh eval-depth-controls-8gpu \
  "$PRIOR_EXPERIMENT_ROOT/ta_dfilm_cf_r256/checkpoints/loss-best.ckpt" \
  "$PRIOR_REPORT_ROOT/ta_dfilm_cf_r256_controls" \
  test_seen \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT"
```

This evaluates `real`, per-sample `spatial_shuffle`, `sample_shuffle`,
`global_mean`, and `zero`; every report also contains the same frozen-base
metrics.

## Prior-Aware Contact Selector

The current route does **not** decode pressure from prior-modified features.
It loads the frozen Binary Grid selector checkpoint and trains only a contact
adapter. Pressure logits remain byte-for-byte identical to that checkpoint.

Depth now has three deliberately different paths:

```text
depth_mapping_rectifier: Depth -> bounded selector-neck correction -> frozen anchor mapping
depth_anchor_residual:   [frozen RGB neck, Depth] -> bounded anchor-logit residual
depth_anchor_query:      512 canonical XYZ queries -> aligned RGB/Depth tokens ->
                         contact residual + independent false-high score
```

The query path uses the canonical mesh only as the output coordinate system.
It does not consume per-frame MANO, pose, hand identity, or an image-to-MANO
calibration. Pressure remains frozen; the direct false-high score is trained
only on vertices where the frozen pressure model already predicts contact.

VLM context uses one shared low-rank FiLM calibrator over existing local
contact/pressure evidence. It has no vertex embedding and cannot create an
independent canonical contact template.

Set the frozen selector and common inputs once:

```bash
export TACTILE_SELECTOR_CHECKPOINT=/path/to/ta_selector_grid_r256/best_selector.ckpt
export DINO_WEIGHTS=/path/to/dinov3_weights.pth
export DEPTH_SIDECAR_ROOT=/home/ma-user/work/cfzhao/input_prior_full/depth_sidecars
```

Selector Depth runs are cache-only by default. The first command computes a
content-addressed cache key from the selector/DINO hashes, authoritative query
and SAM3 manifest hashes, Depth sidecar contract, crop, resolution, field set,
and storage dtype. Missing partitions are built over eight GPUs; finalized
partitions are reused on later runs and on the second server:

```bash
# Optional explicit preparation of train/val/seen/unseen.
./tactile_input_priors/run.sh prepare-selector-cache

# Show the persistent paths selected for each split.
./tactile_input_priors/run.sh selector-cache-paths
```

Before a new cache identity is built, the runner samples the current SAM3
train crops and audits validity after reprojecting the stored Depth sidecars.
The report is content-addressed and reused at
`$INPUT_PRIOR_ROOT/audits/depth_crop_coverage/train`. Run it explicitly with:

```bash
./tactile_input_priors/run.sh audit-depth-crop-coverage \
  --datasets touchanything --split train \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT" \
  --output-dir "$INPUT_PRIOR_ROOT/audits/depth_crop_coverage/train" \
  --reuse-if-current
```

The default cache lives under
`/home/ma-user/work/cfzhao/input_prior_full/cache/selector_evidence`, outside
the source checkout. It stores the FP32 512-D frozen decoder bottleneck,
selector neck and anchor logits, aligned Depth, targets, and availability. The
cheap frozen decoder tail reconstructs pressure logits during training. It
deliberately omits both the 42 GB fused RGB grid and the much larger per-vertex
logit array: the selector adapters consume the frozen neck/anchor evidence
directly. With eight cache partitions and a 65,536-row shard size,
each TA split has one mmap shard per partition, avoiding random small-shard
open/close churn. Crop augmentation is disabled because one immutable feature
record cannot represent a new random crop every epoch.

Cache-only training defaults to batch 256, two train workers and one
validation worker per rank. A deterministic 8,192-sample block shuffle keeps
each rank on the same mmap shard for useful stretches instead of opening pages
at globally random offsets. This is intentional: `32 workers x 8 ranks`
amplifies page faults and shared-storage queueing, which appears as many Linux
`D` tasks and periodic GPU starvation. Override with
`SELECTOR_TRAIN_BATCH_SIZE`, `SELECTOR_CACHE_NUM_WORKERS`,
`SELECTOR_CACHE_VAL_NUM_WORKERS`, or CLI options after measuring the server.

Concurrent servers wait up to six hours for the host already building a cache
partition, then reuse its finalized files. If a host was SIGKILLed and its
cross-host lock is known to be abandoned, reclaim it for one invocation with
`SELECTOR_CACHE_BREAK_STALE_LOCK=1`; never use that override while another
server is still building the same cache.

Run the two Depth controls on separate servers:

```bash
./tactile_input_priors/run.sh train-selector-depth-map
./tactile_input_priors/run.sh train-selector-depth-anchor
```

Run the new aligned/query causal comparison on two servers:

```bash
./tactile_input_priors/run.sh train-selector-depth-query
./tactile_input_priors/run.sh train-selector-depth-query-shuffle
```

These write `ta_dquery_real_r256_cached` and
`ta_dquery_shuffle_r256_cached`. The shuffled run trains the same architecture
with deterministically permuted Depth tokens; it is not merely an evaluation
ablation.

For the strict follow-up, use identical regularization and a short schedule on
both servers:

```bash
./tactile_input_priors/run.sh train-selector-depth-query-clean-real
./tactile_input_priors/run.sh train-selector-depth-query-clean-shuffle
```

These runs both use `control_identity_weight=0`, 10 epochs, and one warmup
epoch. They save `best_loss.ckpt`, `best_false_high.ckpt`,
`prior-info-best.ckpt`, and `last.ckpt`. Compare `best_loss` and
`best_false_high` formally; the composite information score remains available
only for historical continuity because its control term changes meaning when
the primary input is shuffled.

Both commands prepare missing train/val partitions and then remove DINO from
the training process. Cached presets write to `ta_dsel_map_r256_cached` and
`ta_dsel_anchor_r256_cached`, keeping their resume checkpoints separate from
the older online/augmented experiments. Set `SELECTOR_CACHE_MODE=online` only
for historical online-augmentation reproduction. `BASE_FEATURE_CACHE_TRAIN/VAL`
still take precedence when an explicit cache is supplied.

Evaluate aligned and counterfactual priors from the same checkpoint:

```bash
./tactile_input_priors/run.sh eval-prior-selector-8gpu \
  --checkpoint "$PRIOR_EXPERIMENT_ROOT/ta_dsel_map_r256_cached/checkpoints/prior-info-best.ckpt" \
  --split test_seen \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT" \
  --output-dir "$PRIOR_REPORT_ROOT/ta_dsel_map_r256_cached/prior-info-best/test_seen"
```

Selector evaluation and `audit-selector-pressure-8gpu` also resolve/build the
matching split cache automatically. They can recover the frozen selector and
DINO paths from the prior checkpoint when the environment variables are not
set. Set `SELECTOR_CACHE_MODE=online` to bypass this behavior.

The report includes contact AP, sequence-macro AP, calibration, false-high
candidate AP, recall at precision 0.50/0.60/0.70/0.80/0.85/0.90, per-frame
top-1/4/16/64 precision and recall, residual/global-shift diagnostics, and
aligned-minus-control gaps. The full histogram PR curve is written to
`false_high_pr_curve.csv`. Pressure RMSE/V-IoU are intentionally not optimized
at this stage.

### Selector pressure decision audit

AP measures ranking quality, but it does not answer whether acting on a
selector is beneficial. Run the causal pressure audit after selector training:

```bash
./tactile_input_priors/run.sh audit-selector-pressure-8gpu \
  "$PRIOR_EXPERIMENT_ROOT/ta_dsel_anchor_r256_cached/checkpoints/prior-info-best.ckpt" \
  "$PRIOR_REPORT_ROOT/ta_dsel_anchor_r256_cached/pressure_policy_audit" \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT"
```

The validation pass sweeps a bounded downward correction over selector score,
correction strength, and target floor. It writes `pressure_policy_sweep.csv`,
the non-dominated `pressure_policy_pareto.csv`, and three benefit/risk profiles
to `policy_selection.json`. Seen and unseen test passes load those validation
policies unchanged; test data never selects a threshold.

For `depth_anchor_query`, the policy score defaults to the dedicated
`false_high_head`; older adapters continue to use `sigmoid(-contact_logits)`.
Override this explicitly with `--policy-score-source contact|false_high` when
performing a score-source ablation. The selected source is stored in
`policy_selection.json` and enforced unchanged on test splits.

After selection, the runner also evaluates the fixed policies on validation as
`val_control_replay`, then repeats that replay on seen and unseen. The replay
applies the exact same pressure threshold, alpha, and floor to scores from the
RGB-only base selector, aligned Depth, and every counterfactual Depth control.
It writes `pressure_policy_control_replay.csv/json` with:

```text
absolute pressure metrics for every policy/source
source-minus-matched-aligned metric gaps
aligned/control action-set Jaccard and disagreement
false-high benefit, protected-contact harm, and added-under differences
sequence-clustered 95% bootstrap intervals for net utility differences
```

For mapped controls, both sides use only rows where that mapping exists. A
positive `aligned_minus_control_net_utility` favors aligned Depth. For the raw
harm columns, lower is better: negative aligned-minus-control protected removal
or added under-prediction favors aligned Depth.

The default counterfactual controls are deliberately stronger than the legacy
batch roll:

```text
cross_sequence    prior comes from a deterministically paired other sequence
same_sequence_far prior comes from the same query at least 30 frames away
wrong_query       prior comes from another query in the same frame, when present
spatial_shuffle   spatial arrangement is destroyed within the same prior
global_mean/zero  spatial or all prior information is removed
```

Every mapped control reports coverage and pairing provenance. Its aligned-real
reference is recomputed on exactly the same available subset, so a sparse
wrong-query control is not compared against the full split. The historical
`sample_shuffle` remains available only for report reproduction and is labeled
as a batch-local cyclic roll.

The pressure policy acts only when the frozen baseline is above the contact
threshold. It can lower pressure toward a finite floor, but never raises it or
changes the frozen checkpoint. Reports separate strict no-contact points
(`GT<=.02`), a sub-threshold band (`GT<=.08`), gray points, and protected true
contact (`GT>=.10`). Selection is based on false-high volume removed versus
protected pressure removed, added under-prediction, and changes to the normal
RMSE/Contact/V-IoU metrics. `precision>=.9` remains a diagnostic rather than a
hard gate.

To determine whether aligned Depth has a better intervention frontier than the
frozen RGB selector, run the matched-budget audit:

```bash
./tactile_input_priors/run.sh audit-selector-matched-pareto-8gpu \
  "$PRIOR_EXPERIMENT_ROOT/ta_dquery_contact_real_r256_cached/checkpoints/best_contact.ckpt" \
  "$PRIOR_REPORT_ROOT/ta_dquery_contact_real_r256_cached/matched_pareto"
```

This independently sweeps the full policy grid on val, seen, and unseen. It is
diagnostic only and never selects a test policy. Coverage matching keeps
`alpha` and `target_floor` fixed while choosing the nearest RGB score threshold;
removed-volume matching chooses the nearest RGB policy at the same normalized
strict false-high volume removed. Each split writes:

```text
pressure_policy_aligned_sweep.csv
pressure_policy_rgb_base_sweep.csv
pressure_policy_matched_coverage.csv
pressure_policy_matched_removal.csv
pressure_policy_matched_pareto.json
pressure_policy_matched_summary.txt
```

Only rows within the configured relative budget tolerance contribute to the
summary. The default tolerance is 10 percent and can be changed with
`--matched-coverage-relative-tolerance` and
`--matched-removal-relative-tolerance`. Positive `depth_minus_rgb` is favorable
for precision, removed fraction, utility, and IoU deltas; negative is favorable
for protected removal, added under-prediction, RMSE, and false-high excess.

For strict causal attribution, use the exact per-frame action-count audit:

```bash
./tactile_input_priors/run.sh audit-selector-exact-topk-8gpu \
  "$PRIOR_EXPERIMENT_ROOT/ta_dquery_contact_real_r256_cached/checkpoints/best_contact.ckpt" \
  "$PRIOR_REPORT_ROOT/ta_dquery_contact_real_r256_cached/exact_topk"
```

Within each frame, all sources rank the identical frozen-RGB candidate pool and
select exactly `min(k, candidate_count)` vertices. The aligned Depth selector,
frozen RGB selector, spatially shuffled Depth, global-mean Depth, and zero Depth
therefore share the same action count, alpha, floor, pressure prediction,
target, and palm mask. Raw risk logits are stably sorted; lower canonical vertex
indices break exact ties. Validation sweeps `k=0,1,2,4,8,16,32,64`, then writes
`exact_topk_selection.json`. Seen and unseen replay only the validation-selected
values and cannot select another budget.

Each split writes:

```text
exact_topk_source_metrics.csv
exact_topk_real_vs_controls.csv
exact_topk_selection.json
exact_topk_audit.json
exact_topk_summary.txt
```

The pairwise table recomputes ratio metrics from sequence-level sufficient
statistics on every paired bootstrap draw. It reports confidence intervals for
precision, recall, false-high removal, protected pressure removal, added under-
prediction, balanced utility, RMSE, Contact-IoU, V-IoU, and CoreLoc. This audit
does not use approximate threshold matching or change alpha/floor to match
removed pressure volume.

Run deterministic mapping and pressure-metric checks in the tactile environment
before a remote audit:

```bash
./tactile_input_priors/run.sh selector-pressure-tiny-check
```

For cache-only training, build the base cache from the selector checkpoint,
not the earlier pressure-only checkpoint:

```bash
TACTILE_BASE_CHECKPOINT="$TACTILE_SELECTOR_CHECKPOINT" \
./tactile_input_priors/run.sh cache-tactile-8gpu \
  --cache-dir /path/to/selector_cache/train \
  --split train \
  --fields z_rgb,base_logits,contact_neck,contact_anchor_logits,contact_logits,depth_grid,tactile_signal,has_tactile \
  --depth-sidecar-root "$DEPTH_SIDECAR_ROOT"
```

The cache records the selector checkpoint SHA. Training/evaluation rejects a
cache built from another checkpoint instead of silently mixing evidence.
# Query-Aware Temporal Flow

The trainable temporal path is deliberately separate from the mature RGB
baseline. It caches only valid-palm baseline logits, targets, and validity, so
the epoch loop never runs DINO or reads pressure from HDF5. Pair construction
requires the same dataset, sequence, anonymous query/hand side, source HDF5,
exact adjacent source frame, timestamp gap at most 50 ms, stable SAM3 box, and
unchanged association ID. Lag 2 and lag 4 are formed only by chaining those
validated adjacent edges. Cross-sequence controls preserve side and pressure
strata while replacing the complete requested history.

The historical `train-temporal-flow` entry retains the lag-1 product-gated
model for reproduction. The current comparison uses a zero-initialized signed
additive residual:

```text
delta = bounded_sum_k(alpha_k(anchor) * (history_k - current))
alpha_k = max_alpha * tanh(local_coefficient_k)
```

There is no shared global scalar and no multiplication by transition/history
classifier probabilities. Positive and negative corrections can coexist
across anchors and lags, and pressure gradients reach every local coefficient
head at the first backward pass. The classifier heads read a detached trunk
and remain diagnostics rather than pressure gates.

```bash
export TACTILE_BASE_CHECKPOINT=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/hamer_tactile_ft/checkpoints/touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12/best_loss.ckpt
export DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth

# Server A: signed lag-1 control
./tactile_input_priors/run.sh train-tflow-signed-l1

# Server B: signed lag-1/2/4
./tactile_input_priors/run.sh train-tflow-signed-l124

./tactile_input_priors/run.sh eval-temporal-flow-8gpu \
  /home/ma-user/work/cfzhao/input_prior_full/temporal_experiments/ta_tflow_sadd_l1_r256/checkpoints/best_loss.ckpt \
  test_seen

# One-command best_loss evaluation for test_seen and test_unseen
./tactile_input_priors/run.sh eval-tflow-signed-l1
./tactile_input_priors/run.sh eval-tflow-signed-l124

# Frozen validation-selected Step-3 candidates, both test splits, paired
# sequence-clustered bootstrap. Uses temporal-best by default.
./tactile_input_priors/run.sh eval-tflow-confirmatory
```

The epoch loop never runs DINO or reads HDF5. Defaults are `512` samples per
GPU, `2` train workers and `1` validation worker per rank, one prefetched
batch, persistent cache-only workers, 8,192-pair locality blocks, and `1024`
validation samples per GPU. Evaluation reports both complete-split metrics
(every unavailable lag falls back exactly to RGB) and conditional
all-requested-lags metrics. It also records lag-1/all-lag coverage, per-lag
signed coefficients, strict cross-sequence histories, and the available
contralateral lag-1 control. Repeat the evaluation for `test_unseen` and for
`ta_tflow_sadd_l124_r256`; the formal checkpoint is `best_loss`.
Set `TEMPORAL_CACHE_SPLITS=train,val` to prepare only training dependencies.

The maintained temporal roadmap and current decision order live in
[`TACTILE_FLOW_ROADMAP.md`](TACTILE_FLOW_ROADMAP.md). The confirmatory command
writes `confirmatory_metrics.csv`, `sequence_bootstrap.csv`, and compact
JSON/text summaries under `temporal_reports/.../confirmatory_step3/<split>`.

After Step 3, train the independent action selector rather than another
pressure residual. It predicts `down/hold/up` at 512 canonical anchors and
never changes the tactile output. The quality arm consumes actual per-lag
elapsed time, availability, minimum bbox IoU, maximum center jump, and maximum
area change; the second arm is a parameter-matched no-quality control.

```bash
# Server A
./tactile_input_priors/run.sh train-tflow-selector-quality

# Server B
./tactile_input_priors/run.sh train-tflow-selector-noquality

# selector-best on both TouchAnything test splits
./tactile_input_priors/run.sh eval-tflow-selector-quality
./tactile_input_priors/run.sh eval-tflow-selector-noquality
```

Evaluation writes `selector_metrics.csv`, `risk_coverage.csv`, `pr_curves.csv`,
`calibration_curves.csv`, `metrics.json`, and `eval_selector.txt`. Real,
cross-sequence, contralateral, and reset evidence
are measured on full, source-available, and common matched subsets. Balanced
training probabilities are corrected with the measured train class prior.

The quality arm did not improve the real-history selector, so the formal next
step uses the NoQ `selector-best` and keeps time/bbox quality only for masking,
reset, and abstention. Run the complete down-only pressure-policy audit with:

```bash
export DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth
./tactile_input_priors/run.sh audit-tflow-selector-pressure
```

This first sweeps down-score thresholds and bounded sink strengths on `val`.
It then freezes `policy_selection.json` and replays the same policies on val,
test-seen, and test-unseen. No upward correction is enabled. Missing real
history produces exact RGB output. Results are written below
`temporal_reports/ta_tsel_l12_noq_r256/selector-best/down_policy_v1/`:

```text
val_selection/pressure_policy_sweep.csv
val_selection/policy_selection.json
val/pressure_policy_replay.csv
test_seen/pressure_policy_replay.csv
test_unseen/pressure_policy_replay.csv
*/pressure_policy_pairs.csv
*/metrics.json
*/summary.txt
```

The replay includes real, pressure-bin-matched cross-sequence,
contralateral, and RGB-reset evidence. `pressure_policy_pairs.csv` compares
real and each control under the identical val-selected policy and includes a
sequence-clustered bootstrap interval for net utility. Use
`TEMPORAL_POLICY_BATCH_SIZE`, `TEMPORAL_POLICY_WORKERS`, and
`TEMPORAL_SELECTOR_PRESSURE_OUTPUT_ROOT` only when runtime or output placement
needs an override.

The down-only audit selected exact RGB output under every validation utility.
Before adding historical DINO features, attribute that failure to the
anchor-to-vertex mapping:

```bash
./tactile_input_priors/run.sh audit-tflow-selector-mapping
```

This cache-only multi-GPU V3 audit compares `rbf4`, `euclidean_nearest`,
`geodesic_nearest`, and the historical zero-filled `anchor_only` projection.
It additionally measures the selector directly on its native 512 anchors and
projects oracle GT anchor labels through every mapping. These two diagnostics
separate selector ranking quality from information lost while expanding 512
anchors to 6,623 palm vertices. Generic down-action AP is reported separately
from strict (`Pred>=.10, GT<=.02`) and formal (`Pred>=.30, GT<.005`)
false-high AP.

The cross-sequence control is matched by the frozen RGB maximum prediction
rather than a GT pressure bin. V3 reports both the complete control and a
strict subset where the RGB bin matches exactly, without broad-bin fallback.
Its compact sidecar is content-addressed under `TEMPORAL_PAIR_ROOT` and reused
on later runs. It also performs sequence-clustered paired bootstrap on the
exact subset. Outputs are written below
`mapping_attribution_v3/{val,test_seen,test_unseen}`:

```text
vertex_score_metrics.csv
vertex_score_pr_curves.csv
vertex_score_budget_points.csv
mapping_policy_sweep.csv
sequence_score_bootstrap.csv
exact_policy_real_vs_cross.csv
metrics.json
summary.txt
```

`vertex_score_budget_points.csv` is the fair comparison when score scales
differ across mappings. Rows with `mapping=anchor_native` use only the 512
native anchor locations. Rows with `source=oracle_anchor_gt` are mapping
diagnostics, not deployable model results. Do not choose a test-only policy
from `mapping_policy_sweep.csv`; it contains representative interventions only.

`sequence_score_bootstrap.csv` uses compact per-sequence score histograms. Its
AP interval is sequence-macro, while its precision/recall intervals aggregate
sequence-resampled sufficient statistics at source-specific equal action
budgets. `exact_policy_real_vs_cross.csv` uses identical current RGB/GT frames
and changes only the history source. The primary gate is validation transfer;
test-local best policies remain diagnostic oracles.

The next diagnostic adds frozen historical DINO grids without changing the
pressure prediction. It uses an independent content-addressed cache because
the earlier selector cache contains only palm logits and targets:

```bash
export TACTILE_BASE_CHECKPOINT=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/hamer_tactile_ft/checkpoints/touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12/best_loss.ckpt
export DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth

# Server A: warp historical FullGrid features into current crop coordinates.
./tactile_input_priors/run.sh train-tflow-selector-dino-aligned

# Server B: parameter-matched control without historical grid warping.
./tactile_input_priors/run.sh train-tflow-selector-dino-unwarped
```

The train commands automatically build or reuse the required
`z_rgb,palm_base_logits,palm_tactile_signal,has_tactile` cache. DINO itself is
never executed during epochs. DINO caches use 8,192-sample shards by default
instead of the logits-only 65,536-sample layout; override only with
`TEMPORAL_DINO_CACHE_SHARD_SIZE`. Canonical anchors read current and historical
feature motion through cross-attention, and a zero-initialized residual gate
adds that evidence to the existing action selector. This branch emits only
diagnostic `down/hold/up` scores; it cannot alter RGB pressure.

```bash
./tactile_input_priors/run.sh eval-tflow-selector-dino-aligned
./tactile_input_priors/run.sh eval-tflow-selector-dino-unwarped
```

These commands default to `strict-clear-best.ckpt` and evaluate
val/seen/unseen.
`real` means the checkpoint's trained alignment mode. Reports also contain the
opposite alignment, a fixed spatial-content shuffle, RGB-matched
cross-sequence, contralateral, and reset controls. The new outputs are
`strict_clear_metrics.csv`, `strict_clear_paired_bootstrap.csv`, and
`dino_diagnostics.csv`. A nonzero gate without aligned-over-control gain is not
evidence of image-motion correspondence.

Before changing the temporal architecture, run the cache-only action-space and
gradient audit on validation:

```bash
TEMPORAL_AUDIT_GPUS=0,1,2,3,4,5,6,7 \
./tactile_input_priors/run.sh audit-temporal-flow-cache \
  /home/ma-user/work/cfzhao/input_prior_full/temporal_experiments/ta_tflow_r256/checkpoints/best_loss.ckpt \
  val \
  --gradient-batches 32
```

It chains already validated lag-1 edges to form strict lag-2/4 histories, so it
does not rebuild DINO features or read HDF5 pressure. Outputs include
`alpha_sweep.csv`, `gate_algebra_ablation.csv`, `oracle_selector.csv`,
`selector_diagnostics.json`, `gradient_cancellation.csv`, and `summary.json`.
The audit uses exact non-padding stride shards across the selected GPUs and
reduces sufficient statistics globally. Gradient jobs are split by gate
scenario and loss term instead of being repeated on every rank. Set
`TEMPORAL_AUDIT_GPUS=0` for the equivalent single-GPU path.
The architecture decision tree is documented in
[`TACTILE_FLOW_V2_PLAN.md`](TACTILE_FLOW_V2_PLAN.md).
