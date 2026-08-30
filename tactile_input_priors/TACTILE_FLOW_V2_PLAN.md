# Tactile Flow V2: Audit-First Multi-Scale Residual Plan

## Current Diagnosis

The first temporal model is not failing merely because its effective alpha is
small. Its pressure path is

```text
alpha = max_alpha * global_gate * stable_probability * history_probability
delta = alpha * (previous_logits - current_logits)
```

At the exact zero initialization of `global_gate`, the pressure objective has
zero gradient to the local transition head, history head, and graph trunk. The
only initial pressure gradient reaches one scalar shared by every frame and
vertex, so conflicting local correction directions can cancel. The auxiliary
heads may still learn while the pressure output remains the RGB baseline.

The action space has a separate limitation: if current, lag-1, lag-2, and lag-4
all repeat the same false-high error, every history direction is approximately
zero. No interpolation gate can remove that error.

## Stage A: Cache-Only Audit

Run `audit_temporal_flow_cache.py` before training another temporal model. It
does not run DINO or read HDF5 pressure.

1. Build lag-2 and lag-4 only by chaining strict validated lag-1 edges. Every
   intermediate timestamp, bbox, association, source, anonymous query, and hand
   route therefore remains valid.
2. Compare lag-1, lag-2, lag-4, and their mean on the same matched subset.
3. Sweep fixed alpha in probability space and the current bounded-logit space.
4. Record the full-ramp Dense V2 loss as well as RMSE, Contact IoU, V-IoU,
   CoreLoc, false-high, and catastrophic errors.
5. Measure the per-frame oracle over lag/space/alpha candidates.
6. Compare the learned multiplicative gate with magnitude-rescaled product,
   history-only, stable-only, and residual-history gates without retraining.
7. Report history-selector AP/calibration, transition macro-F1, source/sink
   dynamics, and the fraction of persistent errors for which all history
   directions are too small to act.
8. Backpropagate pressure and auxiliary terms under learned, zero, `0.05`, and
   `0.10` effective global gates. Report parameter-group gradient norms and
   batch-gradient cancellation ratios.

Interpretation:

```text
all V2/metric optima at alpha=0
    -> prediction-history interpolation has no useful action space

metrics prefer alpha>0 but V2 prefers alpha=0
    -> objective mismatch

both prefer alpha>0 but trained gate is closed
    -> optimization/gating failure

rescaled/history-only gate beats learned product
    -> selector ordering may be useful but multiplication suppresses it

many persistent errors have no history direction
    -> add a separately bounded source/sink residual or move to feature flow
```

## Stage B: Local Additive Multi-Lag Residual

Only proceed if Stage A shows a useful fixed or oracle ceiling.

```text
current / lag-1 / lag-2 / lag-4 anchor logits
+ per-lag availability and lag identity, pair time/bbox quality
+ canonical anchor XYZ
                    |
             shared graph trunk
                    |
        zero-init local coefficient head per lag

a_k = alpha_max * tanh(head_k(hidden))
d_k = history_logits_k - current_logits
delta_history = sum_k available_k * a_k * d_k
```

There is no global scalar and no multiplication of classifier probabilities.
The signed `tanh` coefficient is exactly zero at initialization but has nonzero
derivative there, so the final pressure loss immediately reaches each local
coefficient head. Transition and history classification remain auxiliary
representations, not hard gates.

The implemented first comparison deliberately keeps one shared maximum budget
and lets each signed local coefficient choose its own magnitude. The summed
logit residual is capped again, so adding lag 2/4 cannot make the output
unbounded. Each lag carries its exact lag identity and availability; a missing
lag contributes exactly zero while shorter available histories remain usable.

Run the strict two-server comparison with:

```bash
./tactile_input_priors/run.sh train-tflow-signed-l1
./tactile_input_priors/run.sh train-tflow-signed-l124
```

Both start exactly at the frozen RGB prediction, use the same lag-1 pair
population, and select `best_loss` by the pressure objective only. Transition
and history-utility losses have weight `0.01`, consume detached graph features,
and therefore cannot steer the pressure trunk. This follows the general lesson
of time-aware recurrent models such as
[GRU-D](https://doi.org/10.1038/S41598-018-24271-9): elapsed time and missingness
are model inputs, not preprocessing details.

## Stage B.5: Long-Horizon Conditional Audit

Before extending the trained input beyond lag 1/2/4, chain the same validated
lag-1 edges to audit lag 8/16/32. Long histories are compared on one matched
population and carry cumulative elapsed time plus the worst bbox quality along
the chain. They are not silently treated as additional pressure templates.

```bash
./tactile_input_priors/run.sh audit-tflow-long-horizon
```

The audit replays every subset of the checkpoint's existing lags and scales its
bounded residual by `0/.25/.5/.75/1`. It also substitutes cross-sequence and
contralateral histories and performs an explicit state reset. Only a long lag
with positive conditional gain beyond shorter prefixes, a useful correction
direction, and adequate strict coverage should enter the next model. Lag 8/16
may instead become a phase/trend input while lag 1/2/4 retain fast correction.

## Stage C: Bounded Source/Sink Innovation

If Stage A finds a material persistent-error fraction, add a smaller independent
innovation path:

```text
delta_free = free_max * tanh(source_sink_head(hidden))
delta = delta_max * tanh((delta_history + delta_free) / delta_max)
```

The head is zero-initialized and locally predicted. Start with `free_max` much
smaller than the history budget. This is the only path allowed to create a new
loading/release correction when all historical prediction differences are zero.
Audit its up/down volume, false-high creation, and saturation separately.

## Stage D: Recurrent State Instead of Literal Lags

If lag-1/2/4 are jointly useful, replace literal storage with a small bank of
time-decayed states rather than a large sequence model:

```text
fast state   <- lag scale near 1
medium state <- lag scale near 2
slow state   <- lag scale near 4
```

A GRU-style carry/update equation is a residual state update rather than a
product of several independent confidence probabilities; the original GRU
formulation is described in
[Cho et al.](https://arxiv.org/abs/1406.1078). State is keyed by anonymous
tracked query. Hand side may route and reset state, but is not a model input.
Evaluation must be sequence-ordered and compare teacher-forced RGB-base state
against free-running fused state.

## Stage E: Feature-Level Tactile Flow

Prediction history can only move along existing pressure differences. If its
oracle ceiling is weak but image motion remains informative, cache frozen DINO
grids for several frames and model temporal feature differences before the
frozen pressure decoder.

Start with explicit multi-scale feature differences, inspired by the short- and
long-range difference design of
[TDN](https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html).
If crop motion makes same-position tokens unreliable, alignment must precede
fusion; local deformable inter-frame attention is preferable to global temporal
attention, consistent with
[SIFA](https://openaccess.thecvf.com/content/CVPR2022/html/Long_Stand-Alone_Inter-Frame_Attention_in_Video_Models_CVPR_2022_paper.html)
and
[TDAN](https://openaccess.thecvf.com/content_CVPR_2020/html/Tian_TDAN_Temporally-Deformable_Alignment_Network_for_Video_Super-Resolution_CVPR_2020_paper.html).

This stage is substantially more expensive and should not begin unless the
cache-only audit shows that temporal evidence has a useful ceiling that output
history cannot realize.

## Order

1. Run Stage A on validation.
2. Freeze all audit choices, then confirm only selected candidates on seen and
   unseen tests.
3. If optimization failed, train Stage B from the frozen RGB baseline.
4. Run Stage B.5 before adding lag 8/16/32 to a trained model.
5. Add Stage C only if persistent no-direction errors are a measured ceiling.
6. Move to Stage D only after multi-lag gains survive strict controls.
7. Move to Stage E only if prediction-history action space is the bottleneck.
