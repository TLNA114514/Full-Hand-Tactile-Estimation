# Local Controllability, Two-Head, and Depth Roadmap

## Baseline

The fixed reference is:

```text
TouchAnything only
SAM3 bbox, bbox scale 1.2
256x192 input
DINO multilevel ReZero + FullGrid32 + CoreLoc
loss-best checkpoint
```

The current failure pattern is local: a small subset of palm vertices should
move up or down while most outputs should remain stable. The existing decoder
compresses the complete image grid into one 512-D vector before producing all
13,614 logits. This makes globally correlated corrections easy and selective
corrections difficult. Depth and VLM adapters therefore cannot be judged fairly
until the decoder's local controllability is measured.

## Experiment Order

1. Measure decoder local controllability.
2. Measure the upper bound of perfect support and ordinal information.
3. Add a bounded local output-correction carrier only if the decoder is the
   limiting factor.
4. Train an independent Contact/Ordinal head.
5. Use frozen, detached Contact/Ordinal predictions to control the pressure
   residual.
6. Run a joint two-head diagnostic only after the detached experiment works.
7. Test Depth as input to the Contact/Ordinal head.
8. Test Depth as input to the local pressure residual.
9. Optionally perform low-LR joint fine-tuning.
10. Return to magnitude and high-pressure calibration after position is stable.

## Stage 0: Oracle Diagnostics

Use a fixed, deterministic sample set containing false-high, false-low,
true-positive, and background frames. Run all oracles from immutable feature
caches so DINO, JPEG, HDF5, crop, and augmentation do not confound the result.

### Feature-Space Oracle

Optimize one bounded correction per cached RGB grid:

```text
z' = z_rgb + RMSClamp(delta_grid)
pred' = frozen_decoder(z')
```

Only `delta_grid` is optimized. DINO, ReZero, FullGrid projection, and the dense
decoder remain frozen.

### Output-Space Oracle

Optimize a bounded correction directly on the base logits:

```text
logits' = base_logits + max_delta * tanh(raw_delta / max_delta)
```

This is the upper bound for any local residual that bypasses the shared 512-D
bottleneck.

### GT Support Oracle

Use ground-truth support only as an offline oracle:

```text
GT <= 0.02  : downward correction is allowed
GT >= 0.10  : downward correction is blocked and upward correction is allowed
otherwise   : prediction is unchanged
```

It must never be used by a deployable model.

### GT Ordinal Oracle

Use cumulative thresholds:

```text
0.02, 0.05, 0.10, 0.20, 0.50
```

Clamp each base prediction into the interval defined by its true ordinal bin.
The extra gain over binary support estimates the value of pressure-range
supervision.

### Required Metrics

```text
target_error_reduction
off_target_delta_ratio
changed_vertex_fraction
up_correction_precision
down_correction_precision
delta_pca_first_component_ratio
RMSE / MAE / Contact-IoU / V-IoU
distribution V-IoU / core distribution V-IoU
```

Mesh-geodesic leakage is reported when a canonical edge/face file is supplied;
the core decision does not depend on optional mesh metadata.

### Stage 0 Decision

- Feature oracle approximately matches output oracle: the frozen decoder is
  locally controllable; keep it and train the Contact/Ordinal route first.
- Output oracle succeeds but feature oracle does not: the shared decoder is the
  bottleneck; add a local output residual that bypasses the 512-D vector.
- Both fail under reasonable bounds: do not add another head; revisit
  query/target association, labels, and single-frame observability.
- Perfect support gives little gain: a Contact head cannot solve the main error.
- Ordinal adds substantial gain over support: prefer cumulative ordinal targets
  over a binary-only Contact head.

### Stage 0 Commands

No pre-existing cache is required. Prepare it once on either server:

```bash
./hamer_tactile_ft/run_local_controllability.sh prepare
```

The wrapper uses eight GPUs to cache the audited TouchAnything validation split
under `/home/ma-user/work/cfzhao/input_prior_full/cache/local_control/` by
default. The path is outside the source checkout, is resumable, is protected by
a shared preparation lock, and is reused by both servers.

Then run Server A:

```bash
./hamer_tactile_ft/run_local_controllability.sh feature
```

Run Server B:

```bash
./hamer_tactile_ft/run_local_controllability.sh output
```

After the initial result, run the sequence-balanced Stage 0.1 controls. These
reuse the same cache but create new selections and reports:

```bash
# Server A: feature RMS budgets 0.025 / 0.05 / 0.10
./hamer_tactile_ft/run_local_controllability.sh stage01-feature

# Server B: output logit caps 1 / 2 / 4 / 6, exact output, support, ordinal
./hamer_tactile_ft/run_local_controllability.sh stage01-output
```

False-high and false-low are independently stratified into `1-32`, `33-256`,
and `257+` error vertices. Each category/stratum takes at most four frames from
one sequence. PCA is recomputed inside each category and stratum instead of
repeating one global PCA value.

Override `TACTILE_BASE_CHECKPOINT`, `DINO_WEIGHTS`, or
`TACTILE_FEATURE_CACHE` only when the server layout differs from the defaults.
The deterministic selector uses the same cache order and thresholds on both
servers. Each report contains `selected_samples.jsonl`, `sample_metrics.csv`,
`summary.csv`, `summary.json`, and full provenance in `run_config.json`.

## Stage 1: Local Correction Carrier

Stage 0.1 showed that the feature-space oracle still spreads roughly 89-95%
of its correction off target, while a direct output residual becomes selective
at logit caps 4-6. Stage 1 therefore bypasses the frozen 512-D pressure decoder
instead of adding another feature residual.

Keep the crop1.2 loss-best baseline logits and add a zero-initialized bounded
residual:

```text
frozen FullGrid32 features [B,6144]
  -> 512 independent canonical anchor up/down coefficients
  -> fixed four-neighbor canonical RBF interpolation
  -> valid-palm-only local_logit_delta in [-6,6]

final_logits = frozen_base_logits + local_logit_delta
```

The DINO backbone, ReZero fusion, FullGrid projection, and dense decoder are
loaded from the mature baseline and frozen. Only the local LayerNorm and the
two anchor heads are optimized. Equal zero-initialized up/down logits make the
initial residual exactly zero, while convex RBF interpolation preserves the
strict logit bound and zeros every non-palm vertex.

Train Stage 1:

```bash
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh local-residual-r256 \
  --index_workers 256 \
  --gpus 0,1,2,3,4,5,6,7
```

Override the frozen source only when necessary:

```bash
TACTILE_BASE_CHECKPOINT=/path/to/best_loss.ckpt \
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh local-residual-r256 \
  --index_workers 256 --gpus 0,1,2,3,4,5,6,7
```

Evaluate the formal loss-best checkpoint and the final drift control:

```bash
EVAL_TASKS_SPEC='touchanything:test_seen;touchanything:test_unseen' \
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
bash hamer_tactile_ft/run_eval_matrix.sh ta_localres_r256 loss-best last
```

Each split writes `local_base_vs_fused_summary.csv`, so the residual can be
judged against the exact frozen base from the same forward pass.

## Stage 2: Independent Contact/Ordinal Head

The first Stage 2 controls read the ordered FullGrid32 vector before the
pressure decoder's 512-D bottleneck and predict 512 canonical anchor logits.
Fixed four-neighbor RBF interpolation maps those logits to the 13,614 canonical
vertices. The complete crop1.2 pressure baseline is restored from its loss-best
checkpoint and frozen. The pressure prediction therefore remains an exact
same-forward reference and receives no selector gradient.

Targets:

```text
GT <= 0.02 : clear no-contact
0.02-0.10  : gray/weak-contact region
GT >= 0.10 : clear contact
```

Ordinal cumulative probabilities predict:

```text
P(p > 0.02), P(p > 0.05), P(p > 0.10), P(p > 0.20), P(p > 0.50)
```

First evaluate the head by itself against thresholding the baseline pressure.
It must add complementary information, not merely reproduce the same mistakes.

Contact training uses class-balanced BCE over only the clear labels. Ordinal
training uses class-balanced BCE independently at all five thresholds plus a
small cumulative-monotonicity penalty. Neither selector is connected to the
pressure residual in Stage 2.

Train both servers:

```bash
# Server A: binary clear-contact selector
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh selector-contact-r256 \
  --index_workers 256 --gpus 0,1,2,3,4,5,6,7

# Server B: five-threshold cumulative ordinal selector
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh selector-ordinal-r256 \
  --index_workers 256 --gpus 0,1,2,3,4,5,6,7
```

Override `TACTILE_BASE_CHECKPOINT` only if the crop1.2 loss-best checkpoint is
stored elsewhere. Exact resume continues to use `--auto_resume` or
`--resume_from_checkpoint`.

Evaluate loss-best formally and last as a drift control:

```bash
EVAL_TASKS_SPEC='touchanything:test_seen;touchanything:test_unseen' \
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
bash hamer_tactile_ft/run_eval_matrix.sh \
  ta_selector_contact_r256 loss-best last

EVAL_TASKS_SPEC='touchanything:test_seen;touchanything:test_unseen' \
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
bash hamer_tactile_ft/run_eval_matrix.sh \
  ta_selector_ordinal_r256 loss-best last
```

### Stage 2.1: Validation Calibration

Balanced selector BCE does not make `sigmoid(logit)=0.5` a calibrated contact
threshold. Existing Stage 2 checkpoints must therefore fit thresholds on val
before test evaluation. For example, calibrate Contact `loss-best/last`:

```bash
OUTPUT_ROOT=hamer_tactile_ft/eval_reports_ta_selector_contact_r256_calibrated \
SELECTOR_CALIBRATION_FIT=1 \
EVAL_TASKS_SPEC='touchanything:val' \
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
bash hamer_tactile_ft/run_eval_matrix.sh \
  ta_selector_contact_r256 loss-best last

OUTPUT_ROOT=hamer_tactile_ft/eval_reports_ta_selector_contact_r256_calibrated \
SELECTOR_CALIBRATION_ROOT=hamer_tactile_ft/eval_reports_ta_selector_contact_r256_calibrated \
EVAL_TASKS_SPEC='touchanything:test_seen;touchanything:test_unseen' \
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
bash hamer_tactile_ft/run_eval_matrix.sh \
  ta_selector_contact_r256 loss-best last
```

Use the same two commands with `contact` replaced by `ordinal` for the ordinal
experiment. Calibration JSON is bound to the exact checkpoint SHA256. Test
reports separate `validation_calibrated_*` formal metrics from explicitly
labeled `test_oracle_*` diagnostic upper bounds.

New selector training also saves `best_selector.ckpt`, selected by validation
calibrated clear Contact-IoU. It embeds that epoch's calibration in the compact
checkpoint and can be evaluated with the `selector-best` selector.

Each split writes `support_selector_summary.csv` and
`support_selector_threshold_curve.csv`. The decisive fields are the
selector/base clear-label IoU and F1, false-high detect rate, false-low recovery
rate, disagreement rate, and the per-threshold cumulative metrics. Pressure
RMSE/V-IoU remain the frozen baseline and must not be interpreted as Stage 2
improvements.

Proceed to Stage 3 only when a selector improves clear-label Contact-IoU over
the baseline threshold and its disagreements recover both some baseline
false-high and false-low vertices. A selector that only predicts less contact,
or copies the baseline support, does not qualify even if its balanced BCE is
lower. Prefer Ordinal only when its `.10` support is at least as good as Contact
and the additional thresholds materially reduce ordinal-bin MAE.

### Stage 2.2: Contact-Specific Feature Controls

The calibrated binary selector separates contact well in aggregate, but has
almost no high-precision coverage of the frozen pressure model's local
false-high/false-low errors. Before concluding that frozen DINO features are
insufficient, isolate whether the tactile-trained `256->32` FullGrid projection
discarded information needed by contact supervision.

Both controls retain the same nonlinear spatial selector:

```text
contact grid [B,256,16,12]
  -> independent 1x1 contact neck (256->64)
  -> spatial residual block
  -> ordered full-grid flatten
  -> 512-D residual MLP
  -> 512 canonical anchor logits
  -> fixed four-neighbor RBF interpolation
```

The final anchor layer remains zero initialized. DINO, the pressure ReZero
path, FullGrid projection, and pressure decoder remain frozen and are excluded
from the optimizer.

Run the strict two-server comparison:

```bash
# Server A: contact-specific neck on the frozen ReZero grid
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh selector-grid-r256 \
  --index_workers 256 --gpus 0,1,2,3,4,5,6,7

# Server B: independent projections/fusion from frozen DINO blocks 8/16/24/32
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh selector-raw-r256 \
  --index_workers 256 --gpus 0,1,2,3,4,5,6,7
```

Fit validation calibration and then evaluate test splits exactly as in Stage
2.1, using experiment names `ta_selector_grid_r256` and
`ta_selector_raw_r256`. Compare `selector-best` formally; keep `loss-best` only
as the frozen pressure reference and `last` as a drift check.

Interpretation:

- Grid succeeds, raw DINO does not: tactile fusion retains useful contact
  evidence; the old FullGrid32 channel compression was the bottleneck.
- Raw DINO succeeds, grid does not: tactile ReZero fusion discarded or biased
  contact evidence; keep an independent contact feature path.
- Both improve Contact-IoU but still lack high-precision correction coverage:
  the limitation is error complementarity or label/query ambiguity, not merely
  selector capacity.
- Both fail: do not unfreeze DINO immediately. Audit ambiguous labels and query
  association before trying a contact-specific DINO adapter.

### Stage 2.3: Base-Conditioned Down-Error Selector

Stage 2.2 selects the frozen ReZero grid as the contact representation, but a
generic contact target still has almost no validation-calibrated coverage of
the pressure base's false highs. Train a separate validity selector instead:

```text
frozen ReZero grid -> contact-specific spatial selector evidence
detached frozen base logit/probability ------------------------+
                                                               |
                   shared per-vertex validity MLP <------------+
```

The output means "retain this frozen-base contact". Training is restricted to
clear-label vertices for which the frozen base already predicts contact:

```text
candidate = palm && base_pred >= 0.10 && (GT <= 0.02 || GT >= 0.10)
label     = GT >= 0.10
```

Low validity is therefore a down-veto. Base-negative vertices do not enter the
loss and this stage never learns or applies upward correction. The pressure
base, DINO, ReZero fusion, FullGrid projection, and pressure decoder remain
frozen.

Checkpoint and threshold selection remain test-independent. Every epoch fits
the down threshold on validation only; `selector-best` maximizes validation
false-high coverage subject to precision `>=0.90`. The compact checkpoint
embeds that epoch's validation threshold, which is applied unchanged to seen
and unseen test splits. `test_oracle_*` remains diagnostic only.

Train:

```bash
NUM_WORKERS=32 VAL_NUM_WORKERS=16 \
./hamer_tactile_ft/run_tactile_experiment.sh selector-down-r256 \
  --index_workers 256 --gpus 0,1,2,3,4,5,6,7
```

Evaluate the formal checkpoint and drift control:

```bash
EVAL_TASKS_SPEC='touchanything:test_seen;touchanything:test_unseen' \
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
bash hamer_tactile_ft/run_eval_matrix.sh \
  ta_selector_down_r256 selector-best last
```

Adopt only if the validation-calibrated threshold transfers with precision at
least `0.90` and materially improves coverage over the generic Grid selector.
Do not proceed to an upward head unless its own error-conditioned oracle first
shows a useful high-precision operating point.

## Stage 2.4: RGB Contact/Ordinal Sufficiency Audit

The next formal step evaluates Contact and Ordinal as independent perceptual
tasks. It does not modify pressure. Exact artifacts contain frozen-base
pressure, GT, Grid Contact logits, Ordinal logits, Down-error logits, and the
parameter-matched Down control, aligned by `sample_uid + vertex_index`.

Export all validation and test artifacts, then run the CPU audit:

```bash
DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
./hamer_tactile_ft/run_selector_sufficiency_audit.sh all
```

The work can be split across servers or resumed one signal at a time:

```bash
./hamer_tactile_ft/run_selector_sufficiency_audit.sh export contact
./hamer_tactile_ft/run_selector_sufficiency_audit.sh export ordinal
./hamer_tactile_ft/run_selector_sufficiency_audit.sh export down
./hamer_tactile_ft/run_selector_sufficiency_audit.sh export down_control
./hamer_tactile_ft/run_selector_sufficiency_audit.sh analyze
```

The historical Ordinal run did not save `best_selector.ckpt`. Its existing
validation comparison favors `last` over `loss-best` for clear AP/IoU and
ordinal-bin MAE, so the runner uses `last` for Ordinal only. Contact and both
Down variants continue to use `selector-best`. These defaults can be overridden
with `ORDINAL_CKPT`, `CONTACT_CKPT`, `DOWN_CKPT`, or
`DOWN_CONTROL_CKPT`. Complete compatible artifacts are skipped automatically.

The audit fits L2-regularized logistic stackers on validation only. There is no
class weighting; the only weights undo the deterministic per-frame vertex
subsampling so that calibration still represents the validation vertex mass:

```text
B                 base pressure logit
C                 Grid Contact logit
O                 all cumulative Ordinal logits
B+C / B+O / B+C+O conditional RGB combinations
B+C+O+D           real Down-error increment
B+C+O+Dctl        parameter-matched Down control
```

Locked models are applied unchanged to seen/unseen for clear Contact,
base-positive false-high, base-negative false-low, and every Ordinal threshold.
Report AP/AUC, NLL/Brier/ECE, val-selected precision-coverage, frame/sequence
macro metrics, paired sequence bootstrap intervals, and score correlations.
Down-error is only compared inside its base-positive candidate set.

Interpretation:

- Strong Contact/Ordinal but no gain conditional on `B`: useful RGB readout,
  not independent correction evidence.
- `B+C+O` improves locked seen/unseen high-precision error recall: retain as a
  future uncertainty component, but still do not modify pressure yet.
- `D` does not improve over `Dctl` after `B+C+O`: stop the Down-error route.
- All RGB combinations fail while label-oracle support remains strong: move to
  additional geometry/view/pose evidence rather than more selector capacity.

Depth, VLM, topology, pose, multi-view, or temporal evidence must first improve
the independent Contact/Ordinal task with aligned/shuffled/wrong-query controls.
Only then may a detached, bounded local pressure residual be reconsidered.

## Stage 3: Detached Two-Head Correction

Stage 3 is paused until the RGB-only sufficiency audit demonstrates stable
conditional information beyond the frozen pressure prediction. A second head
reading the same RGB evidence is not treated as an independent sensor.

Freeze and detach the Contact/Ordinal head before using it to control pressure:

```text
delta = allow_up(q) * bounded_up - allow_down(q) * bounded_down
final_logits = base_logits + delta
```

Do not use a hard mask or direct `q * magnitude`. Required controls are:

```text
base only
GT oracle
predicted Contact/Ordinal
shuffled Contact/Ordinal
joint end-to-end
```

Only after detached predictions help should a joint experiment be attempted.

## Stage 4: Depth

Depth first enters the local Contact/Ordinal head, where its hand-object geometry
can affect support. Compare:

```text
RGB only
aligned real Depth
spatially shuffled Depth
wrong-sequence Depth
zero Depth
```

Depth may enter the local pressure residual only after it improves support with
the required controls. Do not send it through the current frozen global decoder
and interpret broad downward calibration as spatial understanding.

## Two-Server Schedule

### Round 1

- Server A: Feature-space oracle and local Jacobian diagnostics.
- Server B: Output-space, GT Support, and GT Ordinal oracles.

### Round 2

- Server A: independent binary Contact head.
- Server B: independent Ordinal head, only if the oracle supports it.

### Round 3

- Server A: export Grid Contact and Ordinal sufficiency artifacts.
- Server B: export Down-error real/control artifacts.
- Merge artifacts and run the CPU sufficiency audit before any correction.

### Round 4

- Server A: aligned real additional evidence in the support head.
- Server B: shuffled/wrong-sequence/zero evidence controls.

### Round 5

Detached bounded pressure correction is allowed only after a support head shows
locked-test conditional gain. Joint training and final ablations remain last.

## Deferred Work

Keep the following paused during this route:

```text
direct Depth through the frozen global decoder
hard contact masks and direct q*m prediction
all-head end-to-end training from the first step
simultaneous crop/resolution/weight changes
VLM or Depth changes without aligned counterfactual controls
additional decoder families before Stage 0 is resolved
```
