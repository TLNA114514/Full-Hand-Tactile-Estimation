# Input-Prior Research History

## 2026-08-26 Native Anchor And Mapping Oracle Audit

The completed mapping-attribution V1 audit retained RBF4 as the strongest
full-palm expansion. Across validation, seen, and unseen splits it produced
generic down-action AP around 0.46-0.49, while strict false-high AP remained
around 0.15-0.20. Euclidean nearest was close on strict ranking but worse on
generic down ranking; geodesic nearest and the old zero-filled `anchor_only`
control were weaker. No nonzero validation pressure policy became useful.

V1 also showed that real history generally outranked frozen-RGB-bin-matched
cross-sequence, contralateral, and reset controls. This supports a weak real
temporal signal, but does not prove that its 512-anchor predictions survive the
anchor-to-vertex expansion.

The V2 audit addresses that ambiguity without retraining. It measures AP on
the 512 native anchor locations, projects GT anchor labels through each mapping
as an expansion diagnostic, and repeats real-versus-cross comparisons only on
the exact RGB-bin matched subset. The output root is
`mapping_attribution_v2`; V1 artifacts remain untouched.

V2 showed that RBF4 is not the principal failure. With GT anchor labels, RBF4
retains roughly `0.83-0.90` AP on the full palm. The learned selector reaches
native down AP around `0.53-0.55`, but strict false-high AP only `0.16-0.21`;
formal false-high is unstable. Exact RGB-bin matching leaves a consistent
real-history strict AP advantage of about `0.019-0.035`, while formal
false-high does not transfer.

Mapping V3 adds the missing sequence-level uncertainty check. It stores compact
per-sequence histograms rather than raw vertex scores, reports paired
sequence-macro AP and equal-budget precision/recall intervals, and reuses the
pressure-policy sufficient statistics for aligned-versus-cross sequence
utility intervals. V3 writes a new `mapping_attribution_v3` root and leaves V1
and V2 unchanged.

## 2026-08-25 Temporal Selector Mapping Attribution

The NoQ L1/L2 selector retained useful diagnostic down-action evidence, but
its first bounded down-only pressure policy was negative: every validation
utility selected exact RGB output. More aggressive policies reduced RMSE by
broadly suppressing pressure and damaged contact localization.

The next implemented audit freezes that selector and separates two possible
causes. It compares RBF4, Euclidean-nearest, mesh-geodesic-nearest, and
anchor-only mappings while measuring generic down-action, strict false-high,
and formal false-high ranking independently. Its cross-sequence history is
matched by the frozen RGB prediction, removing the previous GT-pressure-bin
dependency. No model is trained and no pressure architecture is changed in
this attribution step.

## Status

The VLM V1-V6 probes and the first formal depth adapters are complete research
branches. Their executable code was retired during the input-prior cleanup,
while every generated result, PPTX, sidecar, model, and data artifact was kept.
This document preserves the conclusions; commands from the retired launchers
are intentionally not reproducible from the current source tree.

The only active implementation in this directory is the reusable offline MoGe
sidecar pipeline described in [README.md](README.md).

## Original Questions

The input-prior work tested two hypotheses around a frozen tactile baseline:

1. VLM features might add object, action, material, and coarse contact-state
   context missing from a tight RGB query crop.
2. Monocular geometry might reduce RGB shortcuts by adding hand-object depth,
   boundary, surface-normal, and occlusion evidence.

Both branches were required to use matched controls. Dataset ID, identity,
handedness, MANO pose, and raw pressure were never intended as model inputs.
Generated priors were cached offline rather than computed in DataLoader workers.

## VLM V1-V6

The VLM studies covered:

- Qwen and SigLIP global embeddings, structured contact prompts, and expanded
  object/action/material/ambiguity prompts.
- Tight crop, wide crop, full frame, marked full frame, dual view, wrong-query,
  and shuffled-context inputs.
- DINO-wide, constant, random, within-sequence, and wrong-query controls.
- Frame volume, active area, pressure bins, contact state, baseline residual,
  false-high risk, and coarse anatomical-region probes.
- Query-pair interventions designed to distinguish query grounding from scene
  recognition.

The stable conclusion was not that VLMs contain no tactile-relevant semantics.
They did recover weak object/action/contact-state information. The problem was
that this information did not provide a robust gain over strong DINO visual
controls and did not map reliably to canonical pressure location. Similar
semantics can correspond to fingertip contact, palm contact, support, release,
or no contact, so a global semantic vector is fundamentally many-to-many with
the 13,614-dimensional target.

The experiments therefore did not justify a production VLM adapter. In
particular, unconstrained FiLM could amplify the existing shortcut
"grasp-like appearance implies broad palm pressure". A future VLM revisit
would need a new signal or task, such as calibrated ambiguity/risk prediction,
and must still beat DINO-wide and query-shuffle controls.

## Depth Probes And Adapters

The geometry studies covered:

- MoGe-2 and Depth Anything V2 outputs, global statistics, dense maps, normals,
  point maps, latent features, RGB-edge controls, and teacher ensembles.
- Spatial shuffle, sequence shuffle, global-only, aligned, and constant-map
  controls.
- Pressure-PCA reconstruction, contact-state probes, baseline-error probes,
  and local image/token-space residuals.
- Frozen-base depth suppression and spatial/global parameter-matched adapters.

Depth contained weak contact-relevant geometry, but the aligned spatial signal
did not consistently beat shuffle controls or the frozen FullGrid baseline.
The adapters mostly learned calibration or train-set suppression. Validation
often peaked immediately while residual magnitude grew later, which is
consistent with overfitting and base/residual cancellation rather than learned
image-to-mesh correspondence.

This does not invalidate monocular depth as an offline asset. Point, normal,
validity, affine, and provenance sidecars remain useful for future diagnostics
or a differently supervised correspondence model. It does invalidate the
retired free depth-to-vertex adapters as the current training direction.

## Shared Diagnosis

SAM3 bbox and masks identify the anonymous image query, but they do not tell the
model which image patches correspond to canonical fingers, palm regions, or
vertices. VLM semantics are global, and depth geometry is camera-centric. Both
were being asked to correct canonical pressure without an explicit coordinate
bridge.

This explains why extra capacity, attention, CSE-like decoding, spatial depth,
and semantic conditioning could fit training data yet fail matched controls or
unseen evaluation. The stronger next question is not "which prior is larger?"
but "what supervises image-to-canonical registration?"

The detailed six-part diagnosis is maintained in
[IMAGE_TO_CANONICAL_CORRESPONDENCE_DIAGNOSIS.md](../hamer_tactile_ft/IMAGE_TO_CANONICAL_CORRESPONDENCE_DIAGNOSIS.md).

## Evidence Standards Retained

Any future restart should preserve these rules:

- Compare aligned features with spatial, sequence, constant, and RGB controls.
- Use sequence-held-out and unseen evaluation, not random-frame leakage.
- Treat gains that survive shuffling as capacity/calibration gains, not spatial
  or semantic causality.
- Keep auxiliary corrections bounded and report fused-minus-base behavior.
- Do not let a prior freely create high pressure before localization is sound.
- Keep teacher inference offline with model, manifest, bbox, and affine hashes.

## Retained And Retired Assets

Retained source:

```text
depth_teacher.py
hdf5_manifest.py
resolve_depth_manifests.py
build_depth_sidecars.py
depth_sidecar.py
run.sh
```

Retired source:

```text
hamer_tactile_ft/audit_input_priors.py
hamer_tactile_ft/run_input_prior_step0.sh
hamer_tactile_ft/run_input_prior_step0_click.sh
tactile_input_priors/vlm_exhaust.py
tactile_input_priors/build_vlm_pair_artifacts.py
tactile_input_priors/depth_adapter.py
tactile_input_priors/make_probe_summary_ppt.py
```

The generated VLM/depth result directories and
`VLM_Depth_Probe_Summary_CN.pptx` remain available as immutable evidence.

## Temporal Selector V2

Frozen seen/unseen confirmation found a small transferable benefit from real
L1/L2 history, while wrong histories could improve RMSE through broad pressure
suppression and damage contact localization. The next temporal stage therefore
does not alter pressure. It trains an independent anchor-level `down/hold/up`
classifier and compares actual per-lag time/bbox quality against an otherwise
identical no-quality control. Missing histories remain exact RGB evidence, and
evaluation retains cross-sequence, contralateral, and reset controls.

The quality/no-quality comparison subsequently showed that elapsed-time and
bbox-quality channels did not improve real-history AP and slightly reduced
macro F1. The no-quality selector is therefore retained. Its useful signal is
concentrated in the `down` action, so the next audit maps anchor down scores
back to palm vertices and selects a bounded sink policy on validation only.
Test replay keeps upward correction disabled, falls back exactly to RGB when
history is missing, and reports identical real/cross/contralateral/reset
policies with sequence-clustered paired utility intervals.

## Historical DINO Selector Evidence

Mapping attribution V3 confirmed that the retained temporal selector contains
weak action evidence but does not support a safe pressure policy. The next
stage therefore remains pressure-inert. It caches the frozen FullGrid `z_rgb`,
reconstructs query-crop affines from audited SAM3 boxes, aligns historical
feature grids into current crop coordinates, and lets canonical anchor queries
read current/history motion through one residual cross-attention layer.

The residual gate starts at zero. Evaluation compares the trained alignment
mode with the opposite alignment, fixed spatial shuffle, RGB-matched
cross-sequence, contralateral, and reset controls. Strict false-high/clear
ranking is reported both globally and with sequence-clustered paired
intervals. Pressure remains exactly the frozen RGB output until this evidence
passes its causal controls.
