# Tactile Flow Roadmap

Last updated: 2026-08-27

This document is the maintained source of truth for the temporal tactile branch.
Update the status table, evidence, and changelog whenever an experiment changes
the order below. Do not infer the current plan from old chat logs.

**Branch status: paused after Step 6.** All implementations, checkpoints, and
reports are retained for future use. The active research priority has returned
to single-frame image-to-canonical pressure localization. Tactile Flow should
only be reopened after that spatial mapping is materially stronger, or when a
new temporal signal can pass the causal gates recorded below.

## Fixed Baseline

```text
Dataset: TouchAnything only
BBox: reviewed SAM3, crop scale 1.2
Input: 256x192
RGB model: frozen DINO + FullGrid32 + CoreLoc
Checkpoint selection: loss-best
Temporal base checkpoint: ta_tflow_sadd_l124_r256/temporal-best
```

Every temporal result must retain an exact RGB/reset path and report real
same-hand, cross-sequence, and contralateral histories on matched records.
Lower RMSE or fewer false-high errors alone is not causal evidence because an
incorrect history can obtain those gains through broad pressure suppression.

## Current Evidence

### Completed: L1/L2/L4 mask and residual-scale audit

- `L1+2, scale=0.75` is the current balanced trained candidate.
- `L1+2+4, scale=0.50` is the current CoreLoc/safety candidate.
- Full-strength L4 improves RMSE/CoreLoc/safety but starts to reduce Contact-IoU
  and temporal accuracy.

### Completed: L1/L2/L4/L8/L16/L32 long-horizon audit

- Strict val coverage remains 84.75% at lag32 (about 1.067 seconds).
- L1/L2 preserve current contact location best.
- L4/L8 carry transition evidence.
- L16/L32 primarily carry slow pressure-state and false-high downward evidence.
- Direct long-history averaging improves RMSE while eventually damaging
  Contact-IoU and V-IoU.
- The deployable history selector remains weak (AP 0.516, AUROC 0.564), while a
  GT oracle is much stronger. Information exists; selecting its action is the
  present bottleneck.

Detailed audit: `long_horizon_val/`.

### Completed: frozen Step-3 seen/unseen confirmation

- Real `L1+L2, scale=0.75` consistently improves RMSE, V-IoU, CoreLoc, and
  false-high over RGB on the common matched subset, but the absolute gain is
  small (RMSE about `-0.00037/-0.00020` on seen/unseen; CoreLoc about
  `+0.00074/+0.00085`).
- Its Contact-IoU gain is only `+0.00038/+0.00053`; the seen confidence
  interval touches zero.
- Cross-sequence history lowers RMSE and false-high more aggressively while
  losing about one Contact-IoU point. This confirms broad suppression remains
  an attractive shortcut and cannot be used as causal evidence.
- The fixed `L1/2/4/8, alpha=0.10` baseline is competitive with the trained
  residual. Temporal evidence is real, but action selection is the bottleneck.

Detailed reports: `confirmatory_step3/test_seen` and
`confirmatory_step3/test_unseen`.

### Completed: Selector V2 quality/no-quality comparison

- Real L1/L2 history raises matched macro AP by about `0.087/0.096` over RGB
  reset on seen/unseen, with most of the gain in the `down` action.
- No-quality down AP is `0.549/0.544`; up AP remains only `0.329/0.354`.
- Per-lag time and bbox-quality inputs change real macro AP by only
  `+0.00029/+0.00001` and slightly reduce macro F1. They are not retained as
  learned action evidence.
- Quality metadata remains mandatory for lag masking, track reset, and
  abstention. The no-quality selector is the pressure-policy baseline.
- Global confidence risk coverage is hold-dominated. Pressure intervention must
  use an action-specific down score selected against actual pressure utility.

Detailed reports: `ta_tsel_l12_noq_r256/selector-best/` and
`ta_tsel_l12_q_r256/selector-best/`.

### Completed, negative: down-only pressure policy

- Validation selected exact RGB output under aggressive, balanced, and
  conservative pressure utilities.
- The safest nonzero policy acted on only 20.8 ppm of validation vertices and
  removed 0.019% of strict false-high volume.
- Lower thresholds reduced RMSE through broad suppression while damaging
  Contact-IoU, CoreLoc, temporal accuracy, and high-pressure predictions.
- Real history was safer than contralateral history, but did not establish a
  useful absolute pressure correction over RGB/reset.

The first mapping audit found that RBF4 is the strongest practical full-palm
mapping. It preserved useful generic down-action ranking (AP 0.46-0.49) but
strict false-high AP remained only 0.15-0.20, and no mapping produced a useful
validation pressure policy. Real history consistently ranked above
cross-sequence and contralateral controls, so the temporal signal is not wholly
spurious. However, the old `anchor_only` control zero-filled every non-anchor
vertex, making it unsuitable for deciding whether the native 512-anchor
selector itself is weak.

The V2 attribution resolved that ambiguity. Native down AP is `0.53-0.55`, but
native strict false-high AP is only `0.16-0.21` and formal false-high AUROC is
near random. GT-anchor RBF4 projection retains `0.83-0.90` AP, so output
mapping is not the main bottleneck. On exact RGB-bin controls, real history
improves strict AP by `0.019-0.035`, while formal false-high does not transfer.
The selector has weak causal temporal evidence, but it predicts generic down
actions rather than reliable no-contact errors.

V3 adds sequence-clustered paired intervals around that exact-control result.
It bootstraps native-anchor AP for every label, RBF4 AP for strict/clear/formal
labels, equal-budget precision/recall, and real-versus-cross pressure utility.
It does not train a model or modify pressure output.

### Completed, negative for pressure control: historical DINO decomposition

- Exact affine warping has effectively zero causal effect. The unwarped model is
  at least as useful as the aligned model and is simpler.
- Spatial token shuffle barely changes the result: residual cosine remains
  above `0.997` and anchor argmax flips are only about `0.6-1.3%`. The branch
  did not establish image-token-to-anchor correspondence.
- At `selector-best`, unwarped DINO improves generic action macro AP by about
  `0.0044`, down AP by `0.0089`, and macro F1 by `0.0075` on average versus the
  no-DINO selector. This is real but small action evidence.
- The same DINO branch worsens calibration by about `0.0040` ECE and degrades
  strict top-ranked false-high precision. For example, unseen P@1 falls from
  about `0.514` with the DINO gate disabled to `0.361` with real unwarped DINO.
- `strict-clear-best` selects epoch 0 while generic selector performance peaks
  later. The objective conflict is structural: training learns broad action
  state more readily than safe local pressure correction.
- No pressure correction was applied in this audit. These results do not claim
  an RMSE, Contact-IoU, or V-IoU improvement.

Retained candidate: the unwarped `selector-best` branch may be reused as a
generic temporal-state feature. It is not a deployable pressure controller.

## Ordered Work

| Step | Status | Work | Decision gate |
|---:|---|---|---|
| 1 | Complete | L1/L2/L4 mask and residual-scale audit | Do not repeat the sweep. |
| 2 | Complete | Conditional lag 1/2/4/8/16/32 audit | Keep long lags as optional evidence, not direct full-strength transport. |
| 3 | Complete | Freeze val-selected candidates and evaluate seen/unseen with sequence-clustered bootstrap | Signal transfers, but gains are small and wrong histories exploit broad suppression. |
| 4 | Complete, negative | Add actual per-lag time, availability, and cumulative bbox-quality inputs | Quality did not beat the parameter-matched no-quality control; retain it only for reliability logic. |
| 5 | Complete, negative | Selector V2 down-only pressure-policy audit | No nonzero policy passed validation utility; do not continue threshold/alpha sweeps. |
| 5A | Complete, negative for pressure correction | Full-palm mapping attribution and label-free cross control | RBF4 is retained; mapping swaps did not rescue validation pressure utility. |
| 5B | Complete | Native-anchor AP, GT-anchor projection oracle, and exact RGB-bin control | Keep RBF4; selector quality, not output mapping, is the bottleneck. |
| 5C | Complete | Sequence-clustered exact-control AP, budget, and pressure-utility bootstrap | Real history contains weak action evidence, but it is not yet a safe pressure intervention. |
| 6 | Complete, negative for pressure control | Separate current-frame DINO capacity from historical DINO motion | Weak generic action evidence exists, but spatial correspondence, calibration, and strict false-high precision failed. Remove affine warping from any future restart. |
| 7 | Paused | Additive Tactile Flow V3 with transport/source/sink branches | Reopen only after single-frame localization improves and a selector beats causal controls on local pressure utility. |
| 8 | Paused | Query-keyed Anchor-GRU state | Reopen only with reliable sequence/query association and a useful Step-7 selector. |
| 9 | Paused | Low-LR FullGrid joint fine-tuning; DINO remains frozen | Frozen-base temporal model must first provide causal gains. |
| 10 | Paused | Calibration, uncertainty, and risk-coverage | Resume together with a validated temporal pressure branch. |
| 11 | Deferred | Mamba or Neural CDE | Mamba only after GRU capacity failure; CDE only if irregular time becomes real. |
| 12 | Deferred | Latent Flow Matching | Require demonstrated conditional multimodality after deterministic V3 and uncertainty audits. |

## Step 3 Confirmatory Candidates

Chosen on validation and frozen before test evaluation:

```text
rgb_reset
fixed_mean_l1248_probability_alpha0.10
trained_real_l12_scale0.75
trained_cross_sequence_l12_scale0.75
trained_contralateral_l12_scale0.75
trained_real_l124_scale0.50
trained_cross_sequence_l124_scale0.50
trained_contralateral_l124_scale0.50
```

Report three subsets:

```text
full_split       Candidate falls back to RGB when its required history is absent.
available        Candidate-specific required-history records.
matched          One common record set for paired causal comparison.
```

Bootstrap sequences, not frames. Masks, residual scales, and fixed alpha may not
be reselected on seen/unseen.

Run the frozen confirmation on both test splits:

```bash
./tactile_input_priors/run.sh eval-tflow-confirmatory
```

Default output:

```text
/home/ma-user/work/cfzhao/input_prior_full/temporal_reports/
  ta_tflow_sadd_l124_r256/temporal-best/confirmatory_step3/
    test_seen/
    test_unseen/
```

Each split contains `confirmatory_metrics.csv`, `sequence_bootstrap.csv`,
`confirmatory_summary.json`, and `confirmatory_summary.txt`. Step 3 is closed;
do not reselect masks or scales from its test reports.

## Selector V2 Contract

The selector is initially diagnostic and does not modify pressure:

```text
down: a bounded downward residual improves the current RGB prediction
hold: correction is too small, ambiguous, or unsafe
up:   a bounded upward residual improves the current RGB prediction
```

Use per-class AP/AUROC, calibrated PR curves, and risk-coverage. Correct the
class prior after balanced training. Train and validate the selector first,
freeze it, and only then train a bounded pressure residual behind it.

The implemented first comparison is:

```text
ta_tsel_l12_q_r256:   real per-lag time and cumulative bbox quality
ta_tsel_l12_noq_r256: identical model without those quality channels
```

Both models use cached frozen RGB/history logits and predict three actions at
512 canonical anchors. Labels are defined relative to the frozen RGB error
with a probability dead zone of `0.02`. Balanced training logits are corrected
with the measured train class prior for validation/test. Evaluation reports
real, cross-sequence, contralateral, and RGB-reset evidence on full, available,
and common matched subsets. No action is applied to pressure in this phase.

```bash
# Server A
./tactile_input_priors/run.sh train-tflow-selector-quality

# Server B
./tactile_input_priors/run.sh train-tflow-selector-noquality

./tactile_input_priors/run.sh eval-tflow-selector-quality
./tactile_input_priors/run.sh eval-tflow-selector-noquality
```

The next pressure-policy audit uses the no-quality `selector-best`. It maps the
512 canonical anchor down probabilities back to palm vertices with the fixed
RBF assignment, applies no action when history is missing, and sweeps only a
bounded downward correction. Validation selects the policy; all later splits
reuse that exact selection.

```bash
./tactile_input_priors/run.sh audit-tflow-selector-pressure
```

The one-button run writes `val_selection/`, a frozen `val/` replay, and fixed
`test_seen/` and `test_unseen/` replays. Every replay reports real,
cross-sequence, contralateral, and reset histories on full, available, and
matched subsets. `pressure_policy_pairs.csv` includes sequence-clustered
real-minus-control utility intervals. This audit does not train or modify the
RGB/temporal models.

Run the mapping attribution before adding another temporal model:

```bash
./tactile_input_priors/run.sh audit-tflow-selector-mapping
```

It reuses the NoQ selector/cache and writes `mapping_attribution_v3/{val,
test_seen,test_unseen}`. `vertex_score_metrics.csv` separates the generic
down-action label from strict and formal false-high labels, includes native
512-anchor metrics, and records GT-anchor projection diagnostics.
`vertex_score_budget_points.csv` compares mappings at the same global action
coverage. `mapping_policy_sweep.csv` applies only representative policies; it
is an attribution check, not another test-set hyperparameter search. The exact
RGB-bin subset excludes cross-sequence controls that required fallback to a
broader bin. `sequence_score_bootstrap.csv` reports sequence-macro AP and
equal-budget precision/recall intervals. `exact_policy_real_vs_cross.csv`
reports absolute aligned utility and paired aligned-minus-cross intervals.

## Tactile Flow V3 Contract

```text
final_logits = rgb_logits + transport_delta + source_delta + sink_delta

transport_delta: bounded and approximately zero-sum on the valid palm
source_delta:    bounded non-negative addition
sink_delta:      bounded non-positive subtraction
```

- Zero initialization must make the initial output exactly RGB.
- Do not multiply several uncertain gates.
- Short, medium, and long lag groups have separate reliability estimates.
- Runtime side/track metadata may route state but must not enter anonymous image
  features as an identity shortcut.

## Historical DINO Evidence

Do not subtract unaligned crops. Cache frozen DINO grids, map the historical
crop into current crop coordinates using known affine metadata, and add optical
flow or token correspondence only if needed. Initially feed current/history
features and their aligned difference only to the action selector. Direct
pressure modulation belongs after this evidence passes causal controls.

Step 6 implements this as a pressure-inert branch. The frozen cache stores the
FullGrid fused tensor `z_rgb [256,16,12]`. For each lag, the known SAM3 crop
affines map current patch centers into the historical crop before bilinear
sampling. The token content contains current appearance, aligned signed and
absolute feature motion, cosine similarity, spatial validity, lag availability,
and a fixed lag encoding. Canonical anchors read those tokens through one
cross-attention layer. A scalar zero-initialized ReZero gate adds the result to
the existing logits/graph selector hidden state; the module still outputs only
`down/hold/up` probabilities and never changes pressure.

Run the two parameter-matched servers with:

```bash
# Both servers
export TACTILE_BASE_CHECKPOINT=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/hamer_tactile_ft/checkpoints/touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12/best_loss.ckpt
export DINO_WEIGHTS=/home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth

# Server A
./tactile_input_priors/run.sh train-tflow-selector-dino-aligned

# Server B
./tactile_input_priors/run.sh train-tflow-selector-dino-unwarped
```

The first launch creates a content-addressed cache containing `z_rgb`; the
second server reuses it when storage is shared, otherwise it builds the same
contract locally. Evaluate the validation-selected strict-clear checkpoints:

```bash
./tactile_input_priors/run.sh eval-tflow-selector-dino-aligned
./tactile_input_priors/run.sh eval-tflow-selector-dino-unwarped
```

The evaluator runs val/seen/unseen and reports the checkpoint's configured
`real` path plus the opposite alignment, a fixed spatial-content shuffle,
RGB-matched cross-sequence history, contralateral history, and reset. It now
also reports three isolated controls:

```text
dino_gate_zero:     same trained selector with the complete DINO residual disabled
dino_zero_motion:   current DINO and reliability retained; historical image motion removed
dino_cross_history: real pressure history retained; only historical DINO content is replaced
```

Primary evidence is
`strict_clear_metrics.csv` together with sequence-clustered paired intervals in
`strict_clear_paired_bootstrap.csv`. `dino_diagnostics.csv` records gate,
residual, valid-token, and motion statistics. `dino_paired_diagnostics.csv`
records action-logit RMS change, argmax flips, residual RMS change, and residual
cosine against the real path. The pressure output remains exactly the frozen
RGB baseline throughout Step 6.

Run the original strict-clear checkpoints and the later generic selector
checkpoints separately:

```bash
./tactile_input_priors/run.sh eval-tflow-selector-dino-aligned
./tactile_input_priors/run.sh eval-tflow-selector-dino-unwarped

./tactile_input_priors/run.sh eval-tflow-selector-dino-aligned-selector-best
./tactile_input_priors/run.sh eval-tflow-selector-dino-unwarped-selector-best
```

Decision rule:

```text
real > gate-zero, but real ~= zero-motion/cross-history:
    keep only current-frame DINO capacity; remove historical DINO and warp
real > gate-zero, zero-motion, and cross-history:
    historical DINO is independently useful; redesign a bounded motion-only branch
real ~= gate-zero:
    remove the complete DINO branch
```

## Deferred And Paused

```text
Unselected full-strength long-history averaging
Unrestricted upward correction
Cross-hand temporal state sharing
Direct unvalidated DINO feature residuals
VLM or Depth directly suppressing pressure
Further decoder replacement
Mamba / Neural CDE / Flow Matching before selector causality
```

Depth may later provide local geometric evidence to the selector. VLM remains a
low-priority global state or uncertainty prior rather than a vertex pressure
controller.

## Changelog

- 2026-08-27: Paused the Tactile Flow branch after the final aligned/unwarped
  causal decomposition. Retained all code and reports. Historical DINO carries
  weak generic action information, but affine alignment and spatial layout were
  not causal, calibration worsened, and strict false-high precision regressed.
  Future work returns to single-frame image-to-canonical localization.

- 2026-08-27: Initial aligned/unwarped DINO reports failed to establish a
  historical-motion effect. Added pressure-history-preserving DINO gate-zero,
  zero-motion, and cross-history controls, paired logit/residual diagnostics,
  and explicit `selector-best` evaluation commands. Step 6 remains
  pressure-inert and is blocked on this causal decomposition.

- 2026-08-26: Closed the V3 mapping attribution as evidence for a weak but
  non-actionable temporal signal. Implemented Step 6 aligned historical DINO
  selector evidence, the unwarped/spatial/cross/contralateral/reset controls,
  strict-clear checkpointing, and sequence-clustered evaluation. No pressure
  correction is enabled.

- 2026-08-25: Reordered the roadmap after the long-horizon audit. Marked mask
  and long-lag audits complete, inserted frozen test confirmation and selector
  calibration before V3, and moved feature-level pressure modulation, Mamba,
  Neural CDE, and Flow Matching later.
- 2026-08-25: Added the one-pass Step-3 confirmatory evaluator, matched
  real/cross/contralateral subsets, and paired sequence-clustered bootstrap.
- 2026-08-25: Closed Step 3 after seen/unseen analysis. Added normalized actual
  per-lag time and cumulative bbox-quality inputs plus an independent,
  pressure-inert Selector V2 with a matched no-quality control.
- 2026-08-25: Closed the learned quality branch after it tied NoQ on AP and
  slightly lost F1. Added a val-selected, test-frozen down-only pressure-policy
  audit with exact RGB fallback, canonical anchor-to-vertex mapping, causal
  controls, and sequence-clustered paired utility intervals.
