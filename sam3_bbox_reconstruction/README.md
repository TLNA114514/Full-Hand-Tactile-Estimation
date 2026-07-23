# SAM3 BBox Reconstruction

This folder is an isolated, local-only hand detection/tracking pilot. It does not
modify the existing HaMeR tactile training data and it does not assign SAM track
IDs to left/right pressure targets.

## What "official SAM3 API" means

It is the Python interface shipped in the local `facebookresearch/sam3` checkout:

```python
predictor = build_sam3_predictor(...)
predictor.handle_request({"type": "start_session", ...})
predictor.handle_request({"type": "add_prompt", ...})
predictor.handle_stream_request({"type": "propagate_in_video", ...})
```

It is not a cloud service. The scripts in this folder wrap that interface, so the
normal user-facing command is `run_pilot.sh`. SAM3 cannot be run without some
model API internally, but callers do not need to use it directly.

Original SAM3 already supports text-prompted video tracking. SAM3.1 primarily
adds Object Multiplex and optimized multi-object inference. Since this project
usually tracks at most two hands, start with `--sam-version sam3`; use 3.1 only
after its compatibility smoke test passes.

## CUDA compatibility

Current upstream documentation asks for Python 3.12+, PyTorch 2.7+, and CUDA
12.6+. A host whose driver only supports CUDA 12.4 cannot use that exact official
stack. The default installer therefore creates a separate compatibility
environment with Python 3.12 and PyTorch 2.6/cu124:

```bash
./sam3_bbox_reconstruction/install_env.sh --profile compat-cu124
```

The installer defaults to Tsinghua mirrors for Conda and ordinary PyPI
packages, and to Aliyun's dedicated CUDA PyTorch wheel mirror. The Aliyun CUDA
directory is consumed with pip `--find-links` because it is a flat wheel
listing, not a package-name index. Use `--mirror none` to return to the
official PyTorch wheel index, or override only the CUDA wheel source:

```bash
./sam3_bbox_reconstruction/install_env.sh --profile compat-cu124 --mirror aliyun
./sam3_bbox_reconstruction/install_env.sh --profile compat-cu124 --mirror huawei
./sam3_bbox_reconstruction/install_env.sh --profile compat-cu124 --mirror none
./sam3_bbox_reconstruction/install_env.sh --profile compat-cu124 \
  --pytorch-index-url https://your.mirror.example/pytorch-wheels/cu124
./sam3_bbox_reconstruction/install_env.sh --profile compat-cu124 \
  --pytorch-find-links https://your.mirror.example/flat-cu124-directory
```

The `aliyun` preset is intentionally hybrid: Aliyun is used for PyPI and CUDA
PyTorch wheels, while Conda uses Tsinghua because Aliyun no longer publishes
the required Anaconda channels. The `huawei` preset uses Huawei Cloud for
ordinary PyPI packages, Tsinghua for Conda, and the verified Aliyun flat CUDA
wheel directory. Huawei Cloud publishes a matching ModelArts container but no
pip-compatible CUDA wheel repository was verified. The installer probes Conda
metadata before environment creation and falls back to SUSTech when necessary.

This compatibility profile disables FlashAttention 3, `torch.compile`, and
asynchronous frame loading. It is intentionally conservative but is not an
upstream-supported SAM3.1 combination. A real-video smoke test is mandatory.
The existing tactile environment (`torch 2.1.0+cu121`) is left untouched.
`numpy` and `h5py` are installed from conda-forge so h5py uses a compatible
binary HDF5 library instead of compiling against the host's older system HDF5.

Before editable installation, the installer verifies that the SAM3 checkout
contains both `pyproject.toml` and the `sam3/` package. An interrupted, sparse,
or wrong checkout is replaced only after a fresh clone passes validation; the
old directory is preserved beside it with an `.invalid.<timestamp>.<pid>` suffix.

When the host driver is upgraded later, the upstream-style environment can be
created explicitly:

```bash
./sam3_bbox_reconstruction/install_env.sh --profile official
```

SAM3/SAM3.1 checkpoints are gated on Hugging Face. Set `HF_TOKEN` before setup or
authenticate inside `sam3bbox`; a local checkpoint can also be passed to every
runner.

## Prompt presets

The presets live in `prompt_presets.json`. Neither preset asks SAM3 for left or
right. Handedness is a later pressure-association problem, not a segmentation
prompt.

Gloved, in recommended order:

```text
gloved hand
black gloved hand
hand wearing a glove
human hand wearing a glove
fingerless gloved hand
tactile sensing glove
white gloved hand
```

Bare, in recommended order:

```text
bare hand
human bare hand
bare human hand with visible fingers
human palm and fingers
human hand
```

The primary SAM session still uses only the preset's first prompt. This avoids
duplicate IDs from several overlapping prompts. In the bare-hand demo preset,
two anatomy prompts and two common held-object prompts are then run in
independent sessions and spatially matched to the primary track. Their scores
are never subtracted across prompts. Use `--prompt "..."` for a controlled
primary-prompt comparison.

Dataset box generation still treats only gloved hands as tactile queries. The pilot enforces the dataset contract:
OpenTouch keeps at most one gloved-hand instance per frame and TouchAnything
keeps at most two. Each SAM mask is reduced to its largest connected component
and filtered by an absolute area floor. The `bare` preset is intended for
standalone demo localization and diagnostics; it must not be used to generate
tactile training boxes.

`hamer_tactile_ft/run_demo_sam3_bbox.sh --prompt_preset bare` also supports an
optional MediaPipe frame selector. MediaPipe only chooses a text-anchor frame
from a sparse sample. Its landmarks, handedness and bbox never participate in
SAM track ranking or tactile inference. If MediaPipe is absent or detects no
hand, the launcher falls back to frame 0; an explicit `--prompt_frame N` always
disables this helper.

### Prompt-aware temporal policy

SAM3 is first run with the configured primary prompt, then all of its native
video object IDs are assessed over the complete sequence. A tactile query must
have a finite primary-prompt score (`out_probs`), a median score of at least
`0.5`, and at least two observed frames. That score is SAM's first-frame
detector score for *that same text prompt*: it is neither an area score nor a
calibrated `glove - bare` probability, so the pilot never ranks a `gloved hand`
candidate against a `bare hand` candidate by comparing their `out_probs`.

The pilot uses a dataset-specific semantic policy. The primary preset and its
candidate order remain untouched, while the preset separately declares a
curated verifier subset. **OpenTouch defaults to `filter`**: it can contain a
nearby bare hand, so a primary track needs matched positive glove evidence on
at least 10% of its observations. **TouchAnything defaults to `off`**: both
expected query hands are gloved, and independent glove/bare text sessions can
turn an unreliable `neither` vote into an unnecessary recall failure. In that
mode SAM's native primary-prompt score, mask-area floor, global track selection
and temporal anti-jump filter still run; only the extra semantic replay is
skipped.

Because these datasets use black tactile gloves, the OpenTouch positive set
includes `black glove`, `black gloved hand`, and `hand wearing a black glove`,
followed by the two colour-agnostic glove phrases. A frame labelled `neither`
means only that no independent verifier matched that primary observation; it
is not treated as a bare hand, and a selected stable track keeps those frames.
Bare-hand text prompts remain diagnostic only by default: SAM3 can segment the
same hand for both `gloved hand` and `hand without a glove`, so its per-prompt
`out_probs` cannot act as a calibrated glove-vs-bare classifier. A vote
requires the same frame, bbox IoU at least `0.70`, and a compatible
mask-centroid location, so two nearby hands do not automatically share
evidence. The complete evidence is written to `track_audit.json` and can be
tuned explicitly:

```bash
--bare-verification-mode filter \
--opentouch-semantic-verification-mode filter \
--touchanything-semantic-verification-mode off \
--glove-verification-prompts auto \
--bare-verification-prompts auto \
--bare-match-iou-floor 0.70 \
--semantic-match-centroid-ratio 0.25 \
--min-glove-verifier-fraction 0.10 \
--max-bare-evidence-fraction 0.0 \
--bare-rejection-policy off
```

`--bare-verification-mode` is the legacy fallback when a per-dataset option is
set to `inherit`. Use `--opentouch-semantic-verification-mode report` to inspect
OpenTouch semantic votes without the positive glove gate. For a deliberate TA
ablation, pass `--touchanything-semantic-verification-mode filter`; it is not
the default. `--bare-rejection-policy bare_only` and `hard` remain available
only as explicit diagnostic ablations; neither should be used to write training
boxes until a dedicated visual glove/bare verifier has been validated.

When a visually gloved hand is unexpectedly filtered, rerun the small pilot
with `--semantic-debug`. Each job then writes
`semantic_debug/semantic_match_evidence.jsonl`, which contains every matched
verifier candidate's IoU, centroid distance, mask area, score, selection result,
and frame-level rejection reason. It also writes raw mask videos for each glove
and bare verifier prompt. This makes it possible to distinguish a real bare
track, a nearby-instance association error, and a bare-prompt false positive
before changing thresholds.

After propagation finishes, the pilot reviews the whole track and rejects a
middle frame when its two temporal neighbours agree but the current box jumps
away and immediately returns. Rejected frames remain empty rather than being
filled from another detection. The thresholds are intentionally visible:

```bash
--min-prompt-score 0.5 \
--min-track-frames 2 \
--temporal-center-residual-ratio 0.75 \
--temporal-area-ratio 3.0
```

An optional multi-frame return-excursion check can remove a short middle run
when independently verified glove observations before and after agree, every
middle observation jumps away, and none has positive glove evidence. It is
disabled by default because it cannot prove that an unbracketed post-occlusion
track is still a glove. Enable it only for a controlled comparison with
`--temporal-return-excursion-frames 12`.

The same retrospective pass also clusters duplicate native SAM object IDs
before applying the one-hand/two-hand query cap. Two candidates are aliases
only when they overlap through time, have similar mask area, and have compatible
mask centroids. The best semantic/prompt track becomes the representative and
an alias may fill a missing representative frame; two nearby but genuinely
distinct hands are not merged merely because their boxes overlap. This prevents
two almost identical boxes around one hand from consuming both TouchAnything
query slots. Evidence is preserved in `duplicate_track_clusters`,
`duplicate_track_pairs`, and `duplicate_track_aliases` inside
`track_audit.json`. The defaults can be inspected or overridden with:

```bash
--duplicate-track-iou-floor 0.80 \
--duplicate-track-overlap-fraction 0.60 \
--duplicate-track-match-fraction 0.80 \
--duplicate-track-centroid-ratio 0.18 \
--duplicate-track-area-ratio 1.50 \
--duplicate-track-min-frames 2
```

Each job now writes `track_audit.json` beside `bboxes.jsonl`. It contains the
SAM response schema, every global track's primary score, independent glove/bare
semantic votes, the locked IDs, and rejection counts. `raw_sam_preview.mp4`
renders the original thresholded gloved-prompt SAM masks for every raw
candidate. `preview.mp4` replays and overlays the SAM masks for only the final
prompt-validated tracks. Thus an empty final frame is an explicit uncertainty
signal, not an implicit fallback to a different hand, while the raw video makes
it clear whether SAM itself or the offline filter made the decision.

For SAM3.1, the internal candidate capacity is intentionally separate from the
dataset query cap: it defaults to four candidates even though OpenTouch finally
keeps one query and TouchAnything keeps two. This gives the global selector
enough candidates to verify semantic prompt agreement rather than accepting
whichever single object happened to be largest in a close interaction.

The default propagation direction is explicitly `forward`: the default text
anchor is frame 0, so a second backwards pass would only duplicate frames. For
a deliberate anchor in the middle of a clip, use `--propagation-direction both`;
duplicate `(frame, object_id)` observations are then deduplicated by native
prompt confidence before the retrospective filter runs.

## 21-sequence pilot

The pilot deterministically selects three sequences from each split:

- OpenTouch: `train`, `val`, `test`
- TouchAnything: `train`, `val`, `test_seen`, `test_unseen`

OpenTouch clips are read directly from the raw HDF5. Only the active physical
tracking chunk is copied as encoded JPEG bytes into a RAM-backed directory.
TouchAnything clips use raw `chest.mp4` and decode only the active chunk into
the same RAM staging area. Existing bbox availability is not a selection
requirement, so the pilot can test sequences with sparse or no legacy bbox.

OpenTouch RAM staging copies the encoded HDF5 JPEG bytes without decoding or
re-encoding them. Each tracking result also writes
unmasked first/middle/last frames to `input_rgb_samples/` and an
`input_color_audit.json` report containing RGB channel statistics and an
OpenCV-versus-PIL decode check. The render contract is explicit: OpenCV frames
remain BGR while drawing and are passed as BGR to `VideoWriter`; SAM reads the
original image resource independently.

The manifest stores `scene.hdf5::data/<demo>/rgb_images_jpeg`; no complete
OpenTouch sequence is materialized by default. Temporary chunk directories are
deleted after every session and stale `/dev/shm/sam3_bbox_chunks` directories
from dead workers are reclaimed on the next run. Full runs also keep the
numeric color audit but do not write its three RGB sample images. The legacy
`--opentouch-materialization lazy|eager` modes remain available for debugging;
the default is `stream`.

The shared-server paths are built in:

```text
OpenTouch raw       /home/ma-user/work/cfzhao/OpenTouch Data/data
OpenTouch splits    <repo>/evaluation/opentouch_splits.json
TouchAnything raw   /home/ma-user/work/cfzhao/EgoTouch
TouchAnything split /home/ma-user/work/cfzhao/EgoTouch/split.json
TA extracted frames /home/ma-user/work/cfzhao/EgoTouch/extracted_frames
SAM3 checkpoint     <repo>/_DATA/sam3.pt
SAM3.1 checkpoint   <repo>/_DATA/sam3.1_multiplex.pt
```

The normal pilot therefore needs no dataset or checkpoint arguments. Run the
two domains independently so detection quality and TA association quality are
reported separately:

```bash
./sam3_bbox_reconstruction/run_domain_pilot.sh opentouch \
  --gpus 0,1,2,3,4,5,6,7

./sam3_bbox_reconstruction/run_domain_pilot.sh touchanything \
  --gpus 0,1,2,3,4,5,6,7
```

Each command defaults to three sequences from every split. `opentouch` covers
`train/val/test`; `touchanything` covers
`train/val/test_seen/test_unseen`. Use, for example, `--splits test` or
`--splits test_seen,test_unseen` for a narrower rerun. The wrapper runs the
tracker, writes the dataset/split quality report, and for TA also runs the
offline pressure-channel association.

To inspect more sequences, set the count per selected split. For example, this
selects 10 sequences from each of the four TouchAnything splits (40 jobs):

```bash
./sam3_bbox_reconstruction/run_domain_pilot.sh touchanything \
  --samples-per-split 10 \
  --gpus 0,1,2,3,4,5,6,7
```

Restricting to `--splits test_seen,test_unseen --samples-per-split 10` instead
creates 20 jobs. Selection remains deterministic for the configured seed.

Use `--all-sequences` for the full reconstruction after the pilot gate passes.
Full runs should use dedicated output roots and omit `--overwrite`, so rerunning
the same command resumes by skipping completed sequence jobs.

The one-command full workflow runs tracking, quality reports, association and
compact manifest export. OpenTouch defaults to two persistent workers per A800
GPU, while TouchAnything keeps one because its dual carry sessions have a
higher peak. After reconstruction and review, `--apply` performs only the fast
manifest writeback and does not rerun tracking, evaluation, or association. It
creates a rollback backup before updating existing extracted metadata. Use
`--run-and-apply` only when reconstruction and writeback deliberately belong to
the same invocation.

```bash
./sam3_bbox_reconstruction/run_full_reconstruction.sh opentouch
./sam3_bbox_reconstruction/run_full_reconstruction.sh touchanything

# Later, write existing manifests back without touching GPUs:
./sam3_bbox_reconstruction/run_full_reconstruction.sh opentouch --apply
./sam3_bbox_reconstruction/run_full_reconstruction.sh touchanything --apply
```

Run those two commands on separate shared-filesystem servers, or use `both` to
run them sequentially on one server. `--workers-per-gpu` is also available on
the lower-level pilot runner; every worker owns one model copy and handles a
different sequence, so output semantics are unchanged. For example, an A800
with sufficient memory can use `--opentouch-workers-per-gpu 8` on the full-run
wrapper. Each worker defaults to one CPU thread (`--cpu-threads-per-worker 1`)
to avoid OpenCV/OpenMP/BLAS oversubscription.

OpenTouch is intrinsically more CPU/GPU intensive than TouchAnything under the
default policy: its primary prompt is followed by five positive glove and two
bare-hand verifier propagations, while TouchAnything disables this semantic
gate by default. More GPU workers help only while CPU cores and storage can feed
those passes; compare completed sequences per hour when increasing from 2 to 4
and then 8 workers per GPU.

Sequence completion is transactional. A launched job has an `.in_progress`
marker; bbox JSONL, track audit, and final summary are atomically replaced, and
the marker is removed only after the complete summary is committed. Therefore
`Ctrl+C` can leave temporary or partial artifacts, but the next identical run
will rebuild that sequence rather than skip it. Only jobs with a matching
summary and all required outputs are resumed as complete.

Full runs automatically use compact storage defaults: mask-preview videos,
TouchAnything association videos, and per-sequence RGB sample JPEGs are
disabled. Chunk staging uses `/dev/shm` by default, so it consumes RAM rather
than the dataset filesystem. Bbox JSONL, track audits, numeric color
audits, quality reports, and writeback manifests are still produced. Full-run
writeback manifests omit repeated raw-ID/source/evidence fields while retaining
every field required by `apply_bbox_manifest.py`. A normal small pilot keeps all
review artifacts.

The compact defaults can be overridden explicitly with `--mask-previews`,
`--input-rgb-samples`, `--association-previews`,
`--no-compact-manifests`, `--keep-materialized-opentouch`, and
`--opentouch-materialization`. These
overrides are not recommended for a full run on a nearly full filesystem.

The full wrapper keeps domain-specific memory defaults. OpenTouch uses 64-frame
sessions because it deliberately re-detects frequently. TouchAnything defaults
to one uninterrupted session (`--video-chunk-frames 0`) with bounded continuous
memory: the original video is decoded lazily, four input frames are cached, and
already-consumed full-resolution outputs plus unreachable tracker history are
discarded while SAM's finite memory bank remains intact. The session and object
IDs are not reset at artificial boundaries. Override the chunk schedule without
changing the output root using `--touchanything-chunk-frames` and
`--touchanything-chunk-overlap` (or the corresponding `opentouch` options).

All defaults can still be overridden with the existing CLI flags. Environment
variables `FULL_HAND_TACTILE_ROOT`, `OPENTOUCH_DATA_ROOT`,
`TOUCHANYTHING_ROOT`, `SAM3_CHECKPOINT`, and `SAM31_CHECKPOINT` provide a
machine-wide alternative without editing code. A missing local checkpoint is a
hard error; the pilot does not silently start a gated Hugging Face download.

`--max-frames 0` is the default and evaluates complete sequences. For a quick
environment check first:

```bash
./sam3_bbox_reconstruction/run_domain_pilot.sh opentouch \
  --gpus 0 \
  --max-frames 32
```

`run_pilot.py` starts one persistent spawned worker for each requested GPU. A
worker loads the SAM predictor once, keeps its weights resident for its later
sequence jobs, and closes only the per-video SAM session and frame reader after
each pass. The primary prompt, glove verifiers, bare-hand verifiers, and mask
preview for one video also share that same predictor. This avoids repeated model
initialization without retaining video-specific tracking state. Use
`--reload-predictor-per-job` only as a compatibility fallback if a particular
SAM build cannot safely reuse its predictor across videos.

One active SAM session owns decoded frames, feature caches and temporal tracker
state. CPU offload alone does not bound RAM because native SAM loaders and
interactive output caches retain history. For an unchunked forward pass, use the
bounded continuous policy (the TouchAnything full wrapper already does this):

```bash
./sam3_bbox_reconstruction/run_domain_pilot.sh touchanything \
  --video-chunk-frames 0 \
  --video-chunk-overlap 0 \
  --continuous-state-memory bounded \
  --continuous-state-retain-frames 64 \
  --continuous-input-cache-frames 4 \
  --offload-state-to-cpu always \
  --gpus 0,1,2,3,4,5,6,7
```

Use `--offload-state-to-cpu never` only for a measured speed comparison.
`summary.json` records the resolved offload policy and per-job CUDA allocated,
reserved and peak memory in `predictor_runtime.cuda_memory_mb`. The model stays
resident between jobs; only dead session state is collected.

Physical session chunking remains available and is still the OpenTouch default.
When selected, each chunk gets a fresh SAM session whose resource contains only
that range. The overlap is used for one-to-one bbox/centroid track stitching; SAM
object IDs are namespaced per chunk before matching, so ID reuse cannot join two
hands accidentally. Primary tracking, semantic verifier passes and mask-preview
replays all use the same bounded-session policy:

```bash
--video-chunk-frames 256 \
--video-chunk-overlap 32 \
--offload-state-to-cpu always
```

`--chunk-staging-root auto` prefers `/dev/shm/sam3_bbox_chunks`. Set an
explicit RAM-backed path when `/dev/shm` is unavailable. Every chunk close logs
CUDA allocated, reserved and free MiB. `--empty-cache-between-chunks` is on by
default, returning dead session cache while keeping the SAM weights resident.
TouchAnything overlaps MP4 decoding with four bounded JPEG encoding workers per
GPU process (`--chunk-encode-workers 4`); OpenTouch directly copies its existing
JPEG bytes and skips encoding.

Session cleanup is defensive across SAM3 checkouts. If native `close_session`
fails or leaves an entry in `_all_inference_states`, the worker removes and
clears that state directly before collecting CUDA memory. At each sequence end,
`run.log` records `[sam-runtime] end_job` with active-session counts and CUDA
memory. A worker automatically recycles the resident predictor when a session
survives cleanup or post-job allocated memory grows more than 2 GiB above its
observed low-water mark; normal stable jobs continue reusing the loaded model.

Chunk boundaries retain online continuity. At the global frame where the next
overlap begins, up to the expected number of nonduplicate masks with native text
score at least `max(--min-prompt-score, --chunk-carry-min-score)` are converted
to normalized positive boxes. SAM3 permits only one initial visual box per
session, so two surviving TouchAnything hands open two bounded sessions that
share the resident model. Each session receives the unchanged glove text prompt
plus one distinct visual box. The prompt-frame object with the highest IoU to
that box is locked to the session, then the same-frame CPU masks are merged
before ordinary stitching. This keeps semantic confidence while preventing both
sessions from emitting both hands. If only one hand survives, its box remains
paired with unrestricted text discovery so another hand can re-enter.
`--chunk-carry-sessions 2` controls the bound. Disable
continuity only for an ablation with `--no-chunk-continuity`; the default carry
threshold is `--chunk-carry-min-score 0.60` and every carry decision is written
to `track_audit.json:chunk_continuity`.

If overlap stitching cannot join a hand that was absent at one boundary and
later reappears, `--chunk-fragment-reentry` retains the later
prompt-conformant fragment only when those frames still have an unused hand
slot. It never raises the configured per-frame cardinality. The quality report
records boundary/interior coverage and retained re-entry fragment counts so
chunk-induced loss can be measured directly.

OpenTouch uses a stricter identity policy in addition to the general memory
bound: a fresh gloved-hand text detection starts every 96 frames with a
24-frame overlap. This prevents a SAM propagation ID from carrying its old
glove identity indefinitely after occlusion and attaching to a dark plate or
another object. Tracks are connected across sessions only by one-to-one overlap
matching; verifier evidence and native object IDs are not inherited blindly.
TouchAnything keeps the general long-video policy because both of its target
hands are gloved and require a separate two-query association pass. Configure
or disable the OpenTouch schedule with:

```bash
--opentouch-redetect-frames 96 \
--opentouch-redetect-overlap 24

# Controlled comparison with the old continuous propagation behavior:
--opentouch-redetect-frames 0
```

For an unchunked run, pair `--video-chunk-frames 0` with
`--continuous-state-memory bounded`; native unchunked mode intentionally retains
SAM's full interactive history and can exhaust host RAM. Rolling memory and RSS
snapshots are written to `summary.json:resolved_session_policy.continuous_state`
and `track_audit.json:continuous_state`. Resolved ranges and cross-chunk aliases
are saved under
`summary.json:resolved_session_policy.chunks` and
`track_audit.json:chunk_stitching`.

For SAM3.1:

```bash
./sam3_bbox_reconstruction/run_pilot.sh \
  --sam-version sam3.1 \
  --gpus 0,1,2,3,4,5,6,7
```

Each sequence produces:

```text
results/<dataset>/<split>/<job>/
  raw_sam_preview.mp4
  preview.mp4
  association_preview.mp4  # TouchAnything after offline association
  bboxes.jsonl
  summary.json
  run.log
```

The top-level `index.html` shows the selected preview videos. Labels such as `query_0`
are anonymous track IDs. They must not be interpreted as left/right until a
separate pressure-association stage has been validated.

For TouchAnything, `association_preview.mp4` overlays `left`, `right`, or an
explicit `?` confidence marker on every associated box. The top-level
`association_index.html` collects those videos, so handedness can be reviewed
visually without opening the CSV/JSON manifests. The default reviewed domain
rule is `screen_order`: at the earliest frame containing the two main tracks,
the left image instance is assigned to the left pressure channel and the right
image instance to the right channel. SAM query/object IDs do not participate.
A clip with only one detected track remains unassigned instead of guessing from
the image midpoint. This label is offline preprocessing metadata and is never
passed to the tactile model.

Association is a CPU postprocess and normally starts after all GPU tracking
jobs finish. The domain wrapper now also runs it over every completed job when
some other job fails, so one OOM no longer suppresses all handedness videos.
It is still normal not to see `association_preview.mp4` while the GPU stage is
actively running. Existing completed outputs can be rendered without rerunning
SAM:

```bash
python sam3_bbox_reconstruction/associate_tracks.py \
  --pilot-dir sam3_bbox_reconstruction/outputs/pilot_touchanything
```

Use `--touchanything-association legacy_anchor` only to reproduce the previous
sparse-bbox control. It is not the default and legacy boxes do not affect the
screen-order manifest.

The numeric query ID comes from SAM's internal detector/tracker ordering. An
instance that is larger, clearer, or detected earlier can therefore become
`query_0` more often; this is not a learned right-hand label. The generated bbox
is the tight largest-component mask box, not the final DINO input crop. When
these boxes are later integrated into `hamer_tactile_ft`, the dataset forms a
square crop centered on the box and expands its maximum side by the checkpoint's
`bbox_rescale_factor` (currently `2.0`) before resizing for DINO.

## Pilot gate before full reconstruction

Review all seven splits for:

1. hand recall during fast motion and occlusion;
2. track fragmentation or ID switches when two hands cross;
3. glove-to-object leakage and masks that absorb the held object;
4. border clipping and long no-detection gaps;
5. performance on sequences that had no usable legacy bbox.

The wrapper also writes `reports/track_quality/sequence_metrics.csv` and
`summary.json`. These report coverage, expected-cardinality rate, prompt score,
fragmentation, normalized centre jumps, and area changes. They are deliberately
identity-free: OpenTouch and TouchAnything can be compared as detectors before
TA left/right pressure pairing is considered.

## Downstream bbox integration

Do not directly overwrite the legacy training metadata with anonymous SAM IDs.
The current tactile loader reads `meta.json`: OpenTouch expects `bbox` plus the
existing target-side `is_right`, while TouchAnything expects
`hands.<left|right>.bbox_chest`. The latter names a pressure target, not a SAM
track. A detected `query_0` therefore needs a separate, verified association to
the corresponding pressure channel before it can supervise training.

For OpenTouch, the user-confirmed single detected hand already identifies the
only tactile target, so there is no left/right assignment problem. For
TouchAnything, semantic filtering is intentionally disabled at detection time,
and `associate_tracks.py` first splits raw IDs at implausible jumps and
reconnects spatially compatible fragments. It then applies the reviewed
screen-order rule to the earliest visible pair. This intentionally replaces the
sparse legacy bbox association, which remains available only as an explicit
control.

The immutable outputs are:

```text
manifests/opentouch_sam3_v1.jsonl
manifests/touchanything_sam3_v1_highconf.jsonl
manifests/touchanything_sam3_v1_uncertain.jsonl
manifests/touchanything_association_audit.json
```

Only `highconf` is eligible for supervised expansion. `uncertain` retains clips
where a two-track ordering could not be established or where duplicate
fragments collided. The resulting manifest contains target-side `left`/`right`
for writing the box into the matching pressure channel, but the model still
receives only one anonymous crop and never sees that identity.

After that association is validated, export an immutable per-split bbox manifest
that records the source sample, target hand, `is_right`, tight SAM box, score,
track ID, and SAM-run fingerprint. Build the training compact index from that
manifest and store the resolved box in each index row; include the manifest hash
and bbox-source version in the cache key. This is faster and safer than parsing
the full SAM JSONL in every data-loader worker, preserves legacy boxes for
rollback, and forces a new index only when the sample set or active bbox source
changes. Newly recovered raw frames must first be materialized with their RGB
and pressure target; an index cache alone cannot create missing examples.

For already extracted TouchAnything samples, first generate a writeback plan:

```bash
python sam3_bbox_reconstruction/apply_bbox_manifest.py \
  --manifest sam3_bbox_reconstruction/outputs/pilot_touchanything/manifests/touchanything_sam3_v1_highconf.jsonl
```

Inspect `manifests/writeback/bbox_writeback_plan.jsonl` and the skipped rows.
Then apply the reviewed plan atomically:

```bash
python sam3_bbox_reconstruction/apply_bbox_manifest.py \
  --manifest sam3_bbox_reconstruction/outputs/pilot_touchanything/manifests/touchanything_sam3_v1_highconf.jsonl \
  --apply
```

Writeback groups left/right preflight checks targeting the same TouchAnything
`meta.json`, defaults to 64 concurrent readers and 32 atomic writers, and uses
temporary-file plus atomic-rename safety. Tune the phases independently with
`--preflight-workers` and `--apply-workers`; `--workers` remains a shorthand
that sets both. Shared filesystems often benefit from more readers but peak at
16 to 32 concurrent writers. The optional `orjson` package is used automatically
when installed. Add
`--fsync-each-file` only when every individual metadata update must survive a
host power loss, since per-file fsync is substantially slower. Process
interruption never exposes a half-written JSON in either mode.

The command writes a timestamped `bbox_writeback_backup_*.jsonl` before changing
the first `meta.json`. Roll back with:

```bash
python sam3_bbox_reconstruction/apply_bbox_manifest.py \
  --restore-backup path/to/bbox_writeback_backup_TIMESTAMP.jsonl
```

Writeback changes only `hands.<left|right>.bbox_chest`, `bbox_score`, and a
provenance-only `bbox_source` object. Pressure arrays and images are untouched.
It refuses a partial apply when selected rows lack an extracted `meta.json`;
those raw frames must be materialized before they can become training samples.

Current tactile training defaults to `--bbox_source_policy sam3_only`. OpenTouch
requires the top-level `bbox_source.schema` to be `sam3_bbox_source_v1`;
TouchAnything checks that schema independently under each target hand. A hand
without a new SAM3 association is excluded instead of falling back to its
legacy bbox. The policy is part of index-cache schema/key and compact checkpoint
provenance. Use `--bbox_source_policy any` only for an explicit legacy control.

## Optional optical-flow assistance

`--flow-assist` adds a CPU-side, bidirectional pyramidal-LK pass after SAM
semantic selection. It does not detect hands or decide glove identity. Native
SAM boxes remain authoritative; flow labels adjacent observations as
`sam3_flow_agreed`, `semantic_motion_conflict`, or indeterminate. With
`--flow-bridge-policy short_bridge`, a missing run of at most five frames is
filled only when projections from two prompt-accepted SAM anchors agree.

Run a review pilot before changing a full writeback manifest:

```bash
./sam3_bbox_reconstruction/run_domain_pilot.sh both \
  --samples-per-split 4 \
  --gpus 0,1,2,3,4,5,6,7 \
  --flow-assist \
  --flow-bridge-policy short_bridge \
  --flow-max-gap 5 \
  --overwrite
```

Each retained track records `bbox_source`, `flow_confidence`,
`flow_bbox_iou`, and `flow_anchor_frames`. Aggregate counts are stored under
`track_audit.json:optical_flow_assist`. A `flow_short_bridge` row has a bbox but
no invented segmentation mask, so mask previews continue to show only actual
SAM masks.
