# HaMeR Feature Routing Roadmap

Last updated: 2026-08-31

This document is the source of truth for testing HaMeR image features as an
implicit geometry prior for the crop1.2 FullGrid tactile baseline. It defines
the hypothesis, stage order, controls, cache contract, promotion rules, and
stop conditions. Update the status table and changelog whenever evidence or
implementation changes.

The parent roadmap is
[`CANONICAL_LOCALIZATION_ROADMAP.md`](CANONICAL_LOCALIZATION_ROADMAP.md).
This branch is Stage 2.2 there. H0 now has a runnable integrity audit; H1-H5
remain design-only and must not be documented as runnable until implemented.

## Scope And Terminology

This is a **pose-informed, pose-output-free canonical routing** experiment.
It is not fully pose-free: HaMeR learned hand geometry from MANO and keypoint
supervision. At tactile inference, however, this route must not consume:

```text
MANO pose or shape
predicted mesh or vertices
2D/3D joints
weak-perspective camera
projected canonical vertices
hand identity or dataset identity
```

Only a frozen HaMeR image-feature grid may enter the tactile router. Explicit
pose outputs may be used later as audit-only oracle diagnostics, never as a
hidden input to the main experiment.

HaMeR features are also not assumed to be dense correspondences. HaMeR uses a
single decoder query to regress global hand parameters from image tokens. Its
features can carry articulation and hand-part information without identifying
which pixel belongs to each canonical surface point. The experiments below
must measure that distinction rather than assume it.

## Fixed Baseline And Constraints

```text
Dataset: TouchAnything only
BBox: SAM3, scale 1.2
Input: 256x192
Appearance encoder: frozen DINOv3 H+/16
Base head: multilevel ReZero + FullGrid32 + CoreLoc
Formal base checkpoint: loss-best
Output: one shared continuous canonical hand surface
Evaluation tensor: 13,614 vertices; valid palm subset: 6,623
Formal checkpoint comparison: loss-best
```

Non-negotiable constraints:

1. Keep the output independent of sensor layout, count, spacing, and dataset.
2. Do not restore wrist-view input.
3. Do not change crop, resolution, pressure weighting, augmentation, sampling,
   or base checkpoint while testing the HaMeR feature hypothesis.
4. DINO, HaMeR, and the crop1.2 FullGrid base remain frozen through H3.
5. Route into canonical surface anchors, not native sensor cells and not
   13,614 independent vertex queries.
6. A gain must depend on aligned HaMeR spatial content. Extra parameters or a
   dataset-level pressure shift are not evidence of correspondence.

## Why This Route Exists

The accumulated evidence points to a routing bottleneck rather than a simple
capacity bottleneck:

1. The K4096 ground-truth surface-basis oracle has ample capacity on every
   official split.
2. FullGrid depends on coarse DINO token arrangement, but its canonical anchor
   influence is nearly global: normalized entropy is about `.993`, and the
   effective token count is about `185/192`.
3. Stage 2 V1 and Stage 2.1 failed to establish DINO-only canonical routing.
   Stage 2.1 reduces Contact-IoU by about `.0045/.0076` for Projected32 and
   `.0041/.0072` for ReZero256 on Seen/Unseen. Real-versus-shuffle differences
   are generally below `.001` and inconsistent across metrics and seeds.
4. Earlier Depth and VLM residuals often reduced pressure broadly instead of
   correcting selected canonical regions. A prior needs a reliable local route
   before its content can be used safely.

HaMeR is therefore tested for one narrow role: supply geometry-aware spatial
keys that may be more useful for routing than DINO appearance keys. DINO still
supplies the contact and object evidence values. HaMeR does not predict tactile
pressure.

## Falsifiable Hypothesis

Primary hypothesis:

```text
HaMeR spatial features contain frame-specific hand-part/articulation cues
that improve the assignment of DINO contact evidence to canonical anchors,
even when all explicit pose outputs are excluded.
```

The hypothesis is supported only if all three links are observed:

1. `DINO + real HaMeR` beats a parameter-matched DINO-only router.
2. Real HaMeR beats spatial shuffle, matched wrong-frame, global-repeat, and
   position-only controls.
3. The benefit appears in localization metrics without broad pressure-volume
   suppression, high-pressure erasure, or unseen-split collapse.

If HaMeR beats DINO but not the controls, the gain is attributable to capacity
or regularization, not frame-specific image-to-canonical geometry.

## Source Architecture Audit

Local HaMeR facts that constrain the implementation:

| Component | Local behavior | Consequence |
|---|---|---|
| Backbone | ViT-H, patch 16, width 1280, depth 32 | Candidate spatial grid is `16x12` for `256x192` input |
| Intermediate blocks | Blocks 8/16/24 are pre-final-norm; block 32 receives final norm | Normalize layer candidates before comparison |
| MANO head | One 1024-D query, six decoder layers, global parameter regression | Decoder query is a global control, not a localization feature |
| Public loading | Checkpoint load may use `strict=False` | Record missing/unexpected keys and fail on material mismatch |
| Stochastic depth | Backbone drop-path reaches a high rate | Frozen extractor must stay in `eval()` and pass determinism checks |

The current tactile crop is already `[B,3,256,192]`. Do not call the complete
HaMeR `forward_step`, which applies an additional horizontal crop intended for
its old square-input path. Call a dedicated rectangular backbone extractor on
the existing crop pixels.

The DINO and HaMeR grids both have 192 cells, but equal shape does not prove
equal patch centers. HaMeR uses a padded patch convolution. H0 must establish
the two lattice definitions and compare native-index fusion with one fixed,
geometry-derived resampling transform.

## Proposed Model Contract

The first deployable candidate is deliberately small and local:

```text
frozen HaMeR spatial tokens -> routing keys and route confidence
frozen DINO ReZero256 tokens -> contact/appearance values
256 canonical XYZ anchors -> routing queries
one local image-to-anchor routing block
-> support-4 canonical surface basis
-> pressure-space mass-preserving source/sink redistribution
-> frozen crop1.2 FullGrid prediction
```

Initial router defaults:

```text
canonical anchors: 256
router dimension: 128
attention heads: 4
routing layers: 1
FFN expansion: 2
anchor/vertex self-attention: none
full-vertex queries: none
HaMeR layer candidate: block 32 after normalization, subject to H0/H1
```

The geometry/content split is intentional:

- HaMeR keys answer **where to read**.
- DINO values answer **what contact evidence is present**.
- Canonical anchors answer **where the evidence may act on the hand surface**.
- The frozen FullGrid model remains the magnitude and fallback predictor.

A fixed zero/null evidence option may absorb occluded or unsupported anchors.
It must not be a learned value capable of emitting a static correction. Query
residuals, coefficient biases, and other image-independent output paths remain
disabled. Spatial values are centered and bias-free so global-repeat controls
are structurally unable to create a residual.

### First Residual: Mass-Preserving Redistribution

The first router must solve localization rather than recalibrate magnitude.
It predicts local source and sink distributions and a bounded transported
amount:

```text
base pressure p
source/sink fields over valid palm vertices
transport <= available source pressure and destination headroom
sum(delta_up) == sum(delta_down)
prediction = p + delta_up - delta_down
```

This prevents the broad downward shift seen in earlier residual branches and
gives a direct test of whether HaMeR improves placement. A bounded logit
residual is a secondary variant only after the mass-preserving route is
understood. A separate magnitude residual is postponed until localization is
demonstrably useful.

The final transport gate is zero-initialized. Before training, the new model
must equal the frozen FullGrid base element by element.

## Stage Status

| Stage | Status | Question | Promotion decision |
|---|---|---|---|
| H0 Integrity and lattice audit | Implemented; pending remote run | Can HaMeR features be extracted reproducibly from the exact crop1.2 pixels? | Select one normalized layer and one fixed lattice contract |
| H1 Frozen feature sufficiency | Planned | Do real HaMeR features add frame-specific canonical information on fixed subsets? | Continue only if real beats DINO-only and causal controls |
| H2 Frozen-base local router | Blocked on H1 | Can HaMeR keys route DINO values into local surface corrections? | Promote only a selective mass-preserving route |
| H3 Full-data confirmation | Blocked on H2 | Does the result survive official splits and full training data? | Adopt as the new localization branch or reject |
| H4 Low-LR refinement | Blocked on H3 | Does limited co-adaptation improve a successful frozen route? | Keep only if controls and unseen behavior remain valid |
| H5 Distillation and efficiency | Blocked on H3 | Can the useful geometry signal be made cheaper? | Prefer a student only when it preserves the causal gap |

Planned artifacts live under:

```text
hamer_tactile_ft/reports/hamer_feature_routing/h0_integrity
hamer_tactile_ft/reports/hamer_feature_routing/h1_probe
hamer_tactile_ft/reports/hamer_feature_routing/h2_router
hamer_tactile_ft/reports/hamer_feature_routing/h3_full
hamer_tactile_ft/reports/hamer_feature_routing/h4_refine
hamer_tactile_ft/reports/hamer_feature_routing/h5_distill
```

## H0: Integrity And Lattice Audit

H0 is mandatory because a silent crop or lattice mismatch would invalidate
every later control.

Implementation:

- [`hamer_feature_routing.py`](hamer_feature_routing.py)
- [`audit_hamer_feature_routing.py`](audit_hamer_feature_routing.py)
- [`run_canonical_localization.sh`](run_canonical_localization.sh)

Run the synthetic contract check without model weights:

```bash
./hamer_tactile_ft/run_canonical_localization.sh hamer-h0-self-test
```

Run the real crop1.2 audit after exposing the official checkpoint:

```bash
HAMER_CHECKPOINT=/path/to/hamer.ckpt \
./hamer_tactile_ft/run_canonical_localization.sh hamer-h0
```

The runner auto-detects the adjacent `model_config.yaml`; set
`HAMER_MODEL_CONFIG` only when the checkpoint and config are stored separately.

Checks:

1. Validate HaMeR checkpoint size, SHA256, model config, and expected backbone
   architecture before loading. The local archived demo data is not assumed to
   contain a valid checkpoint.
2. Load with an explicit report of missing and unexpected keys. Abort on any
   material backbone mismatch.
3. Extract directly from `[B,3,256,192]`; prove that no second center crop is
   applied.
4. Verify exact RGB normalization, SAM3 crop scale 1.2, affine provenance, and
   the existing left-to-right canonical flip.
5. Verify shapes, finite values, frozen parameters, `eval()` mode, and repeated
   forward determinism.
6. Compare normalized blocks `16/24/32` on a tiny balanced set. Block 32 is the
   default; cache only one raw layer after this decision.
7. Write explicit DINO and HaMeR patch-center coordinates. Compare native index
   alignment with a fixed geometric resampling transform; do not learn this
   alignment in H0.
8. Compare the old square-plus-crop and current rectangular path where they are
   expected to be pixel-equivalent.

H0 output must include a machine-readable contract containing crop, flip,
lattice, checkpoint, normalization, and layer hashes.

Implemented artifacts:

```text
h0_integrity/H0_DONE.json
h0_integrity/integrity_contract.json
h0_integrity/lattice_contract.json
h0_integrity/crop_equivalence.json
h0_integrity/sample_manifest.jsonl
h0_integrity/feature_statistics.csv
```

## H1: Frozen Feature Sufficiency Probe

Reuse immutable, sequence-disjoint subsets rather than launching full training:

```text
train: up to 131,072 frames
val: up to 32,768 frames
test_seen: up to 32,768 frames
test_unseen: up to 32,768 frames
```

Use the same sample IDs across every branch and reuse the balanced subset from
the mapping-attribution audits when possible.

Parameter-matched feature branches:

```text
DINO only
HaMeR only
DINO + HaMeR
HaMeR MANO-decoder global token control
```

Required HaMeR controls:

```text
real aligned spatial grid
deterministic per-sample spatial shuffle
same-query/same-sequence far wrong frame, matched for crop scale and pressure mass
global mean repeated over 192 positions
position-only
```

The wrong-frame sampler must never cross split boundaries and must record time
distance, query identity, sequence identity, and matching tolerances.

Probe targets emphasize placement rather than total pressure:

- GT-mass-matched normalized pressure distribution.
- Contact support at the formal thresholds.
- Coarse 32-region and 128-region canonical support.
- Core distribution and false-high/false-low localization.

Report Contact-IoU, V-IoU, CoreLoc, Distribution V-IoU, false-high,
false-low/catastrophic-under, and sequence/task bootstrap intervals. Frame-only
bootstrap is insufficient because adjacent frames are correlated.

Optional attribution after a positive result: compare the HaMeR-finetuned
backbone with its ViTPose initialization under the same probe. This asks whether
the gain comes from HaMeR geometry supervision rather than merely adding a
second large ViT. It is not part of the first required matrix.

H1 stops the route when real HaMeR is statistically indistinguishable from
shuffle/wrong/global controls or when any gain is confined to train/Seen.

## H2: Frozen-Base Geometry Router

Primary and parameter-matched branches:

```text
primary: HaMeR keys + DINO values
control: DINO keys + DINO values
```

Required counterfactual evaluations:

```text
shuffle HaMeR keys only
matched wrong-frame HaMeR keys
global-repeat HaMeR keys
position-only HaMeR keys
shuffle DINO values only
jointly shuffle HaMeR keys and DINO values
```

All branches use the same 256 anchors, support-4 basis, router width, optimizer,
seed, local transport head, and frozen base. Do not add more layers, heads, or
anchors during this attribution stage.

Diagnostics:

- Real-versus-control metric gaps per sequence and task.
- Source/sink transported volume and equality error.
- Geodesic distance between removed and added pressure.
- Anchor load, null use, attention entropy, and pairwise anchor diversity.
- Fraction of corrections inside the intended local support.
- False-high removed/created and false-low corrected/created.
- Base versus fused high-pressure mean prediction.

Attention maps are explanatory diagnostics, not proof of correspondence.

## H3: Full-Data Frozen-Base Confirmation

Only the H2 winner and its strongest DINO-only control move to full
TouchAnything training. Use two servers in parallel when available. Preserve:

```text
crop1.2, 256x192, current loss and sampling
frozen DINO, HaMeR, ReZero, FullGrid, and base decoder
loss-best for formal comparison
last only for drift diagnosis
```

Cache-only training is valid for a deterministic feature experiment, but it is
not automatically equivalent to the original online random crop augmentation.
A fixed seed does not make one cached feature per sample reproduce all epoch
augmentations. A successful cache-only result therefore needs one online or
strictly augmentation-matched confirmation before a formal claim.

Use sequence/task bootstrap intervals and inspect both official splits. A small
mean gain with broad task regressions is not promotion evidence.

## H4: Low-LR Refinement

Do not unfreeze from the start. If H3 succeeds, resume its `loss-best` and test:

```text
router LR: 1.0x
ReZero + FullGrid LR: 0.05x to 0.10x
DINO: frozen
HaMeR: frozen
short refinement only
```

Only after that controlled refinement succeeds may the last one or two HaMeR
blocks be considered at `0.01x` to `0.05x` router LR. Preserve a frozen-HaMeR
feature/affinity consistency target so the geometry prior cannot collapse into
a global pressure shortcut.

Training HaMeR jointly from the beginning is rejected for the first route: the
tactile labels do not directly supervise pose or correspondence, the dataset
can reward shortcut calibration, and a moving geometry encoder would make the
real-versus-control interpretation much weaker.

## H5: Distillation And Efficiency

HaMeR ViT-H is expensive. Optimize it only after H3 proves that aligned HaMeR
features have unique value.

Candidate distillation targets, in order:

1. Compressed normalized HaMeR spatial features.
2. Anchor-to-token affinity and visibility/null confidence.
3. The local source/sink transport field.

The student may share DINO-side features, but it must not distill MANO outputs
or final tactile pressure. Compare teacher and student real-versus-control gaps,
not just final pressure metrics. If distillation erases the causal gap, it has
learned a dataset prior rather than the desired geometry signal.

## Cache And Compute Plan

For `2,640,078` samples and 192 FP16 tokens, approximate uncompressed sizes are:

| Channels | Full cache |
|---:|---:|
| 1280 | 1,208.5 GiB |
| 256 | 241.7 GiB |
| 128 | 120.9 GiB |
| 64 | 60.4 GiB |
| 32 | 30.2 GiB |

For the maximum H1 subset of 229,376 samples:

| Channels | Probe cache |
|---:|---:|
| 1280 | 105.0 GiB |
| 128 | 10.5 GiB |
| 64 | 5.2 GiB |

Cache policy:

1. H0/H1 may cache one selected raw HaMeR layer on the probe subset.
2. Fit PCA or another fixed linear projection on train samples only. Save its
   mean, components, dtype, and SHA; never fit on val/test.
3. Freeze that projection before building a full-data 64/128-channel cache. A
   trainable projection cannot be cached without changing the experiment.
4. Extend the existing atomic, resumable feature-cache schema instead of
   creating loose per-frame files.
5. Align records by `sample_uid` and provenance hashes, never by row index.

Every cache shard must record:

```text
sample_uid
source manifest/index hash
SAM3 bbox manifest/hash
crop affine, bbox scale, and input resolution
left/right source and canonical flip
HaMeR checkpoint/config SHA
feature block and normalization state
patch lattice and alignment transform
dtype and projection SHA
code/schema version
```

## Decision Rules

There is no single magic acceptance threshold. Promotion requires a coherent
result across splits, tasks, seeds, and causal controls.

Primary evidence:

1. Better Contact-IoU and CoreLoc/Distribution localization than the frozen
   crop1.2 base and DINO-only router.
2. Real HaMeR materially better than shuffle, wrong-frame, global-repeat, and
   position-only controls.
3. No broad loss of high-pressure prediction, Temporal Accuracy, or unseen
   task performance.
4. Source/sink volumes remain balanced and local corrections remove more
   false-high/false-low errors than they create.
5. Sequence/task bootstrap intervals support the direction of the gain.

RMSE or pressure-loss improvement alone is insufficient because previous
branches achieved it by suppressing pressure globally. Attention entropy or a
visually appealing heatmap is also insufficient without metric and control
evidence.

Formal tables continue to use `loss-best`. `last` remains a drift diagnostic.

## Failure Interpretation And Next Action

| Observation | Interpretation | Next action |
|---|---|---|
| H1 real equals all controls | Frozen HaMeR tokens do not expose usable frame-specific geometry | Stop HaMeR branch; move to weak dense correspondence supervision |
| H1 passes, H2 fails | Feature is informative but the local transport/router is inadequate | Test one supervised local-association objective, not a larger generic attention stack |
| H2 improves RMSE but harms localization | Broad calibration leaked into the route or transport is incorrect | Reject and repair conservation/identity before any new architecture |
| Seen improves, Unseen fails | Domain/task shortcut or weak geometry generalization | Do not promote; inspect per-task failure and HaMeR confidence |
| Shuffle or wrong-frame has no effect | No evidence of image-to-canonical correspondence | Stop the claim and branch |
| H3 succeeds | HaMeR provides useful implicit geometry | Run H4, then H5; only then revisit Depth/VLM through this route |

If this branch fails, the next pose-free step is explicit training-time
correspondence supervision: synthetic render correspondences, cycle-consistent
surface embeddings, sparse hand-part/keypoint regularization, or teacher-student
dense association. Estimated pose remains a later confidence-gated fallback;
GT pose remains an oracle only.

## Paused Variants

```text
Using MANO pose, mesh, joints, or camera as input
Using HaMeR's single decoder query as the primary spatial feature
Predicting pressure directly from HaMeR
Full 13,614-vertex cross-attention
Anchor or vertex self-attention
Multiple HaMeR layers in the full cache
Learned crop/lattice alignment before H0 is resolved
HaMeR-native crop2.0 in the first controlled experiment
Unfreezing DINO or HaMeR from scratch
Depth/VLM fusion before a geometry route passes controls
```

## Primary References

- [HaMeR: Reconstructing Hands in 3D with Transformers](https://arxiv.org/abs/2312.05251)
- [Official HaMeR implementation](https://github.com/geopavlakos/hamer)
- [HMR 2.0 / Humans in 4D](https://arxiv.org/abs/2305.20091)
- [HandOccNet](https://arxiv.org/abs/2203.14564)
- [Continuous Surface Embeddings](https://arxiv.org/abs/2011.12438)
- [SurfEmb](https://arxiv.org/abs/2111.13489)
- [Keypoint Transformer](https://arxiv.org/abs/2104.14639)

The correspondence papers are methodological warnings as much as inspiration:
dense image-to-surface correspondence normally requires an explicit learning
signal. HaMeR features can be a useful prior, but controls must decide whether
they are sufficient for this dataset.

## Update Protocol

After each stage:

1. Update the status table and date.
2. Record exact sample IDs, commands, configs, checkpoint selectors, and output
   paths once they exist.
3. Separate base-versus-fused results from real-versus-control attribution.
4. State the falsified hypothesis and next single decision.
5. Preserve failed artifacts and move rejected variants to Paused Variants.
6. Add a dated changelog entry below.

## Changelog

- **2026-08-31:** Created Stage 2.2 after DINO-only evidence routing failed
  across four seeds. Defined a pose-informed but pose-output-free HaMeR feature
  route, exact crop/lattice integrity gate, frozen feature probes, causal
  controls, mass-preserving local transport, cache provenance, staged
  refinement, distillation, and explicit stopping conditions.
- **2026-08-31:** Implemented H0 without importing or instantiating the MANO
  head. Added strict `backbone.*` checkpoint validation, normalized HaMeR
  intermediate extraction on the existing `256x192` crop, deterministic
  real-sample feature checks, square-to-rectangle pixel equivalence, explicit
  DINO/HaMeR patch-center geometry, fixed lattice resampling, atomic artifacts,
  and click-to-run/self-test runner modes.
