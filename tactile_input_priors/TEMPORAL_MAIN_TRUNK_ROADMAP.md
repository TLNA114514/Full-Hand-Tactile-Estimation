# Temporal Main-Trunk Roadmap

Last updated: 2026-09-02

This document is the maintained source of truth for the active temporal RGB
tactile model. The full-width 6144-D frame-token Transformer remains the main
route. One OnlineHMR-style causal patch-KV branch is now an explicit comparison
experiment, not a replacement main line. Older temporal-grid, local-memory,
Tactile Flow, and other patch-memory experiments remain evidence archives.

## Status

```text
Active route: FullGrid6144 bidirectional spatiotemporal Transformer
Stage: implementation complete; full training pending
Primary task: canonical full-hand pressure prediction
Auxiliary task: per-vertex binary contact prediction
Deployment setting: fixed-window offline video
```

Parallel control: OnlineHMR-style 512-D causal patch query / two-frame KV memory
Control purpose: test causal topology and efficient online inference

## Non-Negotiable Contract

```text
Dataset: TouchAnything only
BBox: reviewed SAM3, crop scale 1.2
Input: 256x192
Reference: frozen DINOv3 + ReZero + FullGrid32 MLP + CoreLoc
Reference checkpoint: crop12 loss-best, used only as an evaluation baseline
Hand representation: anonymous canonical hand
Pose/MANO/handedness input: none
VLM/Depth input: none
Temporal direction: bidirectional inside a fixed clip
Checkpoint initialization: fresh tactile trunk, no crop12 weight loading
```

The model must preserve the sensor-independent 13,614-vertex canonical output.
It must not predict in a sensor coordinate system, construct pose-derived
vertex queries, average frame-level pressure predictions, or share state
between two hands.

The contact head is an auxiliary perception head. It does not gate, multiply,
clip, or post-process the tactile prediction in this experiment. This avoids
repeating the previous selector failure mode where broad suppression improved
some false-high statistics while damaging the full pressure field.

## Sole Active Architecture

### End-to-end path

```text
T RGB frames
  -> SAM3 crop1.2 independently per tracked hand
  -> frozen DINOv3 multilevel patch features
  -> shared trainable ReZero fusion
  -> fused grid [B,T,256,16,12]
  -> shared FullGrid projection 256 -> 32
  -> patch tokens [B,T,192,32]

  -> one full inner-frame spatial Transformer block
       192 patches attend to all 192 patches in the same frame
       2D RoPE, pre-norm, residual attention, residual FFN

  -> flatten each frame without pooling
  -> frame tokens [B,T,6144]

  -> one full-width bidirectional temporal Transformer block
       every frame attends to every valid frame in the clip
       actual-timestamp 1D RoPE, padding mask, pre-norm

  -> temporally enhanced frame tokens [B,T,6144]
       -> independent tactile head -> 13,614 pressure logits
       -> independent contact head -> 13,614 binary contact logits
```

"Full-width" means that the persistent temporal state and the standard Q/K/V/O
projections have width 6144. The active route does not replace that state with
a pooled 64/256/512-D frame summary. The existing FullGrid idea is therefore
preserved exactly: every one of the `192 x 32` ordered spatial features remains
available to the final heads.

### Spatial block

For each frame independently:

```text
x: [B*T,192,32]
x <- x + MHA(LN(x), 2D-RoPE)
x <- x + FFN(LN(x))
```

The spatial block uses full attention, not the depthwise 3x3 convolution used
by `causal_clip_transformer_v4`. It explicitly lets every image patch exchange
evidence with every other patch before the frame is flattened.

Default:

```text
layers=1
d_model=32
heads=4
head_dim=8
ffn_ratio=2
attention_dropout=0
```

Although `head_dim=8` is small, this block is deliberately lightweight. Its
job is spatial context mixing, while the ordered 6144-D vector and dense heads
retain the high-capacity image-to-canonical mapping.

### Temporal block

For one fixed clip:

```text
f: [B,T,6144]
f <- f + LayerScale_t * MHA(LN(f), bidirectional, timestamp-RoPE)
f <- f + LayerScale_f * FFN(LN(f))
```

Default:

```text
clip_length=8
layers=1
d_model=6144
heads=48
head_dim=128
ffn_ratio=2
attention_dropout=0
residual_dropout=0.10
LayerScale initialization=1e-3
```

The clip is bidirectional, so the method is an offline fixed-window model and
must be reported separately from causal online methods. Clips remain inside a
strict `sequence_id + bbox_association_id` hand track. Padding cannot become a
key or value. Actual timestamp deltas, rather than only integer frame offsets,
drive temporal RoPE. Crop-affine deltas remain available as an additive
conditioning term because independently generated SAM3 crops can move even
when the physical hand does not.

There is no old `causal_state - reset_state` subtraction and no hard 5% feature
RMS clamp. Both previously made the temporal path fragile or too easy to shut
off. Small nonzero LayerScale provides a stable residual start while allowing
the temporal trunk to become part of the representation.

## Joint Contact And Tactile Heads

Both heads read the same temporally enhanced 6144-D frame token, but their
prediction parameters are independent and their decoder topology is the same:

```text
tactile head:
  6144 -> 512 -> residual 512-D block -> 13,614 pressure logits

contact head:
  6144 -> 512 -> residual 512-D block -> 13,614 contact logits
```

The tactile head keeps the current FullGrid MLP topology and pressure loss.
The contact branch is an independent copy of the same decoder tail. It changes
only the target, output interpretation, and loss. It does not reuse the old
512-anchor/RBF selector: that module was useful as a low-parameter diagnostic
on a frozen feature, but its fixed interpolation would impose unnecessary
smoothness and a low-rank spatial bottleneck on small or sharply bounded
contact regions. The two heads share the complete ReZero, FullGrid projection,
spatial-attention, and temporal-attention representation, but no 512-D head
weights. The contact head predicts:

```text
y_contact = 1[ground_truth_pressure > 0.10]
```

Only valid palm vertices enter either loss. Contact optimization uses
class-balanced binary cross entropy plus a small soft-Jaccard term so that the
sparse positive class and the frame-level support are both represented:

```text
L_contact = balanced_BCE + 0.25 * soft_Jaccard
L_total = L_tactile + lambda_contact * L_contact
lambda_contact = 0.10 initially
```

The contact term follows the same five-epoch loss ramp as other auxiliary
losses. Its gradient norm and cosine similarity with the tactile gradient at
the shared temporal representation must be logged. If it persistently opposes
the tactile objective or dominates its norm, reduce `lambda_contact`; do not
connect contact probabilities to pressure outputs in response.

Formal pressure comparison uses the validation tactile-loss-best checkpoint.
Also save joint-loss-best and contact-AP-best for diagnosis, but neither may
replace tactile-loss-best in the main pressure table. Contact AP, precision,
recall, F1, calibration, and threshold-stratified errors are reported from the
same tactile-selected checkpoint.

## Initialization And Training

This is a fresh temporal trunk, not a crop12 fine-tune. Follow
`hamer_tactile_ft/INITIALIZATION_CONTRACT.md`:

```text
1. seed 521
2. construct the canonical ReZero + FullGrid tactile head using
   model_initialization_order=legacy_decoder_first
3. record initial_tactile_head_sha256
4. append the spatial and temporal Transformer blocks
5. append the independent contact head with the same decoder-tail topology
6. record temporal/contact initialization hashes
7. load only the released DINOv3 weights and freeze DINO
```

Trainable modules:

```text
ReZero fusion
FullGrid 256->32 projection
spatial Transformer
6144-D temporal Transformer
tactile head
contact head
```

Default run:

```text
hardware target: 8 x NVLink A800; assume 80 GiB/GPU and 1 TiB host RAM
clip partition: non-overlapping strict eight-frame clips
supervision: every valid frame in every clip
augmentation: crop12 scale/translation sampled once and shared inside each clip
clips/GPU: 16 (128 frames/GPU before memory fallback)
global target-frame batch: 1024
epochs/warmup/loss ramp: 60/3/5
base LR/effective 8-GPU LR: 5e-5/4e-4
precision: bf16-mixed
gradient clip: 1.0
validation: every epoch
```

The implemented one-click entry points are:

```bash
./tactile_input_priors/run.sh train-tfull6144
./tactile_input_priors/run.sh eval-tfull6144
```

Outputs are rooted at:

```text
$TEMPORAL_EXPERIMENT_ROOT/ta_tfull6144_bi_r256
$TEMPORAL_REPORT_ROOT/ta_tfull6144_bi_r256/best_loss
```

Resume is allowed only within the same experiment and must restore optimizer,
scheduler, sampler/RNG state, and W&B run ID. It is not an initialization path
for a new configuration.

## OnlineHMR-Style Comparison

This branch keeps the same data, frozen DINO, trainable ReZero/FullGrid
projection, fresh initialization, 6144-D decoder input, twin heads, losses, and
checkpoint selection. Only the temporal fusion topology changes:

```text
current FullGrid patches [B,192,32]
  -> project to 512
  -> current-frame patch self-attention
  -> current patch queries cross-attend to the previous two frames' patch KV
  -> residual project 512 -> 32
  -> restore ordered 6144-D FullGrid vector
  -> independent tactile/contact dense heads
```

Defaults follow the official OnlineHMR decoder scale where applicable:

```text
layers=4
hidden_dim=512
heads=4
ffn_ratio=4
history memory=2 frames / 384 patch tokens
clip_length=8, causal supervision on every valid frame
2D patch RoPE + actual-time RoPE
direct 512 -> 32 projection into the ordered FullGrid decoder input
```

The implementation retains separate self- and cross-attention Q/K/V
projections and post-norm residual blocks. It adds the positional encoding that
is present but commented out in the public OnlineHMR code, because temporal
order must be observable in this tactile setting. The transformed patch
representation directly replaces the pre-head patch representation, matching
OnlineHMR's trunk semantics. There is no small output gate or frozen RGB
prediction to fall back to. ReZero, projection, temporal stack, and both heads
are jointly trained from scratch; this is not a frozen-base correction adapter.

Training and evaluation:

```bash
./tactile_input_priors/run.sh train-tonlinehmr
./tactile_input_priors/run.sh eval-tonlinehmr
```

Outputs:

```text
$TEMPORAL_EXPERIMENT_ROOT/ta_tonlinehmr_kv_r256
$TEMPORAL_REPORT_ROOT/ta_tonlinehmr_kv_r256/best_loss
```

Evaluation includes RGB reset, real history, one-frame memory, cross-sequence
history, frame shuffle, lag reversal, spatial shuffle, and repeated-current
controls. It also verifies that learned batched causal inference and FIFO KV
inference agree to BF16 `max_abs <= 2e-3` before accepting a report. The branch wins
only if real history improves the RGB-reset result and the history controls
show a corresponding causal degradation; raw speed or parameter savings alone
do not establish useful temporal evidence.

## Required Controls

All controls use the same trained checkpoint and current RGB frames:

```text
real_bidirectional       chronological frames and timestamps
single_frame_reset       current frame only; all other frames masked
repeat_current           repeat current image at every timestamp
frame_content_shuffle    fixed token/frame content permutation, timestamps fixed
cross_sequence           matched foreign-hand clip
past_only                mask future frames
future_only              mask past frames
spatial_token_shuffle    shuffle the 192 patch contents before spatial attention
```

`past_only` and `future_only` are reported over all valid frames because the
last clip frame has no future: at that endpoint, bidirectional attention and
past-only attention have the same key set. Controls that replace or permute
historical image content are scored at the unchanged endpoint.

The model is useful only if real temporal context improves the pressure task
over both crop12 and its own single-frame reset, and the gain is materially
reduced by destructive history controls. A lower RMSE caused by broad pressure
suppression is not evidence of successful temporal reasoning.

Evaluation must report, per split and on common matched rows:

```text
RMSE, Contact-IoU, V-IoU, Temporal Accuracy, CoreLoc
false-high, catastrophic-over, catastrophic-under
pressure-volume ratio and up/down output volume
contact AP/precision/recall/F1/calibration
real-minus-reset paired deltas
control-minus-real paired deltas
onset/stable-contact/release stratification
per-frame position within the eight-frame clip
```

## Width, Heads, And FFN Cost

Let:

```text
D = model width
T = number of frame tokens
h = number of attention heads
r = FFN ratio, so FFN hidden width = rD
```

For standard full-width multi-head self-attention:

```text
attention parameters ~= 4D^2
FFN parameters       ~= 2rD^2
block parameters     ~= (4 + 2r)D^2

attention projection MACs ~= 4TD^2
attention mixing MACs     ~= 2T^2D
FFN MACs                  ~= 2rTD^2
```

Biases and LayerNorm parameters are negligible at this scale. With fixed `D`,
changing `h` does not materially change parameter count or total arithmetic:
it changes `head_dim=D/h`, kernel efficiency, head specialization, and the
size of an explicitly materialized `[B,h,T,T]` attention matrix.

Changing `r` can affect accuracy because the FFN is the per-frame nonlinear
channel mixer after temporal evidence has been collected. Too small an FFN can
underfit interactions among the 6144 ordered spatial channels. A wider FFN can
represent more conditional combinations, but improvement is not monotonic: it
also increases optimization difficulty, overfitting capacity, and the ability
to learn broad whole-hand pressure shortcuts. In this model the latter risk is
important because flattening makes every spatial location visible to every FFN
unit.

For `D=6144`:

| FFN ratio | FFN hidden | MHA params | FFN params | Approx block params |
|---:|---:|---:|---:|---:|
| 0.5 | 3072 | 151.0M | 37.7M | 188.7M |
| 1 | 6144 | 151.0M | 75.5M | 226.5M |
| 2 | 12288 | 151.0M | 151.0M | 302.0M |
| 4 | 24576 | 151.0M | 302.0M | 453.0M |

For `T=8`, one `r=2` block costs approximately 2.42 billion MACs per clip;
only about 0.8 million of those MACs are the actual `T x T` attention mixing.
Almost all cost comes from the dense Q/K/V/O and FFN projections. Increasing
the clip from 8 to 16 therefore does not create an attention-memory crisis,
but it approximately doubles the dominant projection cost and DINO frame
count.

Recommended temporal heads:

| Heads | Head dim | Assessment |
|---:|---:|---|
| 24 | 256 | Valid but at the upper end for Flash-SDPA kernels |
| 32 | 192 | Reasonable; kernel support must be preflighted |
| 48 | 128 | Default; conventional Flash-friendly head dimension |
| 64 | 96 | Valid but adds head-launch/score overhead without clear need |

Therefore the first full run uses `heads=48`, `ffn_ratio=2`, and one temporal
block. More heads do not make the model larger. Ratio 2 gives the jointly
trained pressure/contact trunk extra nonlinear capacity without paying for the
302M-parameter FFN of ratio 4. The existing tactile head also supplies another
substantial nonlinear stage after temporal attention.

Under ordinary DDP every A800 stores a complete model replica; NVLink speeds
gradient synchronization but does not automatically pool eight GPUs into one
640-GiB address space. The 1-TiB host memory helps HDF5 caching and workers, not
CUDA activations. Conservatively counting FP32 parameters, gradients, and two
AdamW moments, the ratio-2 temporal block occupies about 4.8 GB/GPU before
temporary optimizer buffers; ratio 4 occupies about 7.2 GB/GPU. Both fit an
80-GiB A800, so compute, all-reduce traffic, and generalization are stronger
reasons than memory to stop at ratio 2.

If the main configuration exceeds memory, use this fallback order without
changing the 6144-D route:

```text
1. clips/GPU 16 -> 8, increase gradient accumulation to preserve global frames
2. activation-checkpoint the temporal block
3. FFN ratio 2 -> 1
4. FFN ratio 1 -> 0.5
5. only then use low-rank Q/K/V projections with 6144-D residual state
```

Reducing the persistent frame state to 64/256/512 is not a memory fallback for
the active experiment; it is the separate archived backup architecture.

## Archived Evidence

- Output-level Tactile Flow found small causal L1/L2 gains but remained a
  pressure regulator and could improve RMSE through broad suppression.
- Temporal Grid V1 failed to prove spatial use of history.
- Local Memory V2 showed that counterfactual rejection can keep a history path
  from collapsing, but it was still a frozen-decoder mechanism probe.
- Hierarchical lag writers and the old causal clip model did not beat the
  single-frame crop12 baseline reliably.
- The old causal clip used a depthwise 3x3 spatial operator and same-location
  temporal attention, then subtracted a reset path. It did not implement the
  full spatial-attention/full-frame-temporal design specified here.

## Archived 64-D Backup

This design is retained for reproducibility only:

```text
[B,T,192,32]
  -> project patches 32 -> 64
  -> per-frame full spatial attention
  -> current patches cross-attend all patches in other frames
  -> project 64 -> 32
  -> residual to FullGrid32 features
```

This older 64-D residual adapter is not the active 512-D direct-trunk
OnlineHMR comparison above. It remains archived and must not be substituted
for the implemented `onlinehmr_patch_kv_v6` experiment.

## Engineering Contract

- Temporal train/eval defaults to grouped online HDF5 reads; no feature cache
  is generated implicitly.
- Metadata-only HDF5 and clip indexes remain persistent.
- Flash-SDPA must be verified in BF16; math fallback is not accepted silently.
- Compact checkpoints store all Transformer/head configuration, DINO SHA,
  initialization hashes, clip-index SHA, replay contracts, and W&B run ID.
- Evaluation copies validation metrics into the report directory.
- `real/all_frames` evaluation covers every source row exactly once, including
  cold-start and short terminal-clip frames. Padding is excluded. Evaluation
  hard-fails on a clip/frame/labeled-metric count mismatch and records the
  coverage counts in `summary.json`.
- Ctrl+C and worker failures stay under `process_supervisor.py` so DDP workers
  and W&B services are reaped together.

## Changelog

### 2026-09-02: FullGrid6144 implementation

- Implemented full 192-token spatial attention with 2D RoPE and persistent
  6144-D bidirectional temporal attention with timestamp RoPE.
- Added independent same-topology tactile/contact decoder tails, balanced BCE
  plus soft-Jaccard contact training, shared-gradient diagnostics, and separate
  diagnostic best checkpoints.
- Added clip-consistent crop12 scale/translation augmentation, exact resume
  provenance, Flash-only CUDA SDPA, and all required temporal controls.
- Added one-click `train-tfull6144` and `eval-tfull6144` entry points.

### 2026-09-02: FullGrid6144 route lock

- Replaced the causal 128-D same-patch clip model as the active architecture.
- Locked the sole active route to full per-frame spatial attention followed by
  bidirectional temporal attention over persistent 6144-D frame tokens.
- Added independent jointly trained binary-contact and pressure heads.
- Explicitly removed VLM, Depth, pose, selector gating, and output correction.
- Added full-width attention/FFN cost formulas and the A800-adjusted initial
  `heads=48, ffn_ratio=2, layers=1` configuration.
- Retained the 64-D patch-memory model only as an inactive backup copy.

### 2026-09-02: Historical causal clip

- Added the non-overlapping eight-frame causal clip experiment with all-frame
  supervision, exact reset controls, resume provenance, and per-epoch
  validation. Its evidence is archived above; it is no longer the active
  architecture.

### 2026-08-30 to 2026-09-01

- Moved temporal work from output-level correction into the RGB feature trunk.
- Added Temporal Grid V1, Local Memory V2, fresh lag-writer experiments,
  counterfactual controls, bilateral reset rules, and online grouped HDF5
  loading.
