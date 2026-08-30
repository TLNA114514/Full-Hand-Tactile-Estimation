# Sensor-Independent Canonical Tactile Localization Roadmap

Last updated: 2026-08-29

This document tracks the work needed to place pressure on the correct
canonical MANO vertices after the single-frame FullGrid baseline has already
estimated a plausible total pressure. It is the source of truth for ordering,
evidence, decisions, and paused branches. Update the status table and changelog
whenever a stage finishes or the order changes.

## Fixed Baseline

```text
Dataset: TouchAnything only
BBox: SAM3, scale 1.2
Input: 256x192
Visual encoder: frozen DINOv3 H+/16
Head: multilevel ReZero + FullGrid32 + CoreLoc
Formal checkpoint: loss-best
Output: 13,614 canonical vertices; metrics use the valid palm subset
```

The `[B,13,614]` tensor is an evaluation-time sampling of a canonical surface
pressure field. It is not a sensor grid and must not become the semantic output
space of a new decoder.

## Non-Negotiable Design Constraints

```text
Input: one SAM3 query crop from the main RGB view
No wrist view
Output: pressure on a shared continuous canonical hand surface
No sensor index, layout, count, spacing, or dataset ID in model features
Prefer no GT or estimated hand pose at inference
```

Dataset-specific sensor geometry may only appear on the label/loss side as an
observation operator, coverage mask, or calibration record. A native sensor
grid must never be the formal model output, a required latent coordinate, or a
deployable intermediate representation. Left/right metadata may be used in
preprocessing to mirror both hands into one canonical surface; it is not a pose
input.

Sensor-layout independence is not calibration independence. Cross-dataset
training still needs the physical support, response curve, normalization,
unknown regions, and effective point-spread function of each measurement
system. Unobserved surface regions are unknown supervision, not zero pressure.

Do not change crop, input resolution, pressure weighting, sampling, DINO
training state, or checkpoint policy while diagnosing localization. A changed
baseline would confound representation and supervision effects.

## Established Evidence

1. Strict-fit memorization reaches approximately `0.953/0.944` frame-macro
   V-IoU on OT/TA 1024. The model has enough capacity to fit the mapping when
   examples can be memorized.
2. The FullGrid head is materially better than MeshQuery and the tested
   cross-attention, CSE, depth, and VLM decoders. More decoder complexity alone
   has not established image-to-canonical correspondence.
3. The current decoder globally mixes the `16x12x32` projected grid through a
   `6144 -> 512 -> 13,614` dense path. Each output vertex can depend on every
   image token, but locality and mesh topology are not structural constraints.
4. Direct output oracles can repair sparse false-high/false-low errors with
   changes to roughly one percent of vertices. Comparable feature-space
   corrections leak broadly through the frozen dense decoder. This is evidence
   of weak local controllability, not proof that the input has no information.
5. Support/ordinal oracles have useful upper bounds, while RGB-only selectors
   have not safely converted that information into pressure corrections.
6. Depth/VLM experiments generally changed pressure too globally. Their failure
   does not establish that the priors are useless; it establishes that the
   tested fusion paths lacked a reliable canonical routing mechanism.
7. The tactile-flow branch produced useful temporal diagnostics but has not
   solved the single-frame canonical mapping. Its state is preserved in
   [`TACTILE_FLOW_ROADMAP.md`](../tactile_input_priors/TACTILE_FLOW_ROADMAP.md).
8. Stage 0.1 confirms localization is the larger bottleneck. On the common
   `GT volume >= 1` population, replacing only mass raises Contact-IoU from
   `.4415` to `.4958`, while replacing only normalized distribution raises it
   to `.6267` and reduces false-high excess by about `96.8%`.
9. Hard canonical patch means fail even at 512 patches (`Contact-IoU .2784`).
   A piecewise-constant patch output is therefore rejected. Pressure mass is
   low-rank at coarse scale, but binary support retains a long high-dimensional
   tail, motivating overlapping multiscale functions plus local residuals.
10. The original component and ambiguity subaudits took the first eligible
    12,000 records. Because the cache is ordered, both covered only `Home`.
    Stage 0.2 removed this prefix bias with balanced scene/task/sequence
    sampling over 5 scenes, 103 scene-task groups, and 174 sequences.
11. The balanced component audit strengthens the high-frequency conclusion.
    At pressure `>=.10`, active frames contain a mean/median of `99.1/105`
    connected components, and the largest 16 cover only `75.7%` of pressure
    mass. These are label-space components and may include mapping or threshold
    artifacts; they are not 99 literal physical contact islands.
12. The cumulative 1696-D (`32+128+512+1024`) GT basis oracle exceeds the
    same-subset FullGrid baseline by `.0933` Contact-IoU, `.0905` V-IoU, and
    `.0760` CoreLoc, but worsens false-high by about `15%`. Smaller 32/160/672-D
    dictionaries do not preserve enough localization detail.
13. The 1696-D result is not yet a trainable parameterization: `51.8%` of
    coefficients are negative and `35.1%` of pre-clamp vertex values are below
    zero. Repeated partition-of-unity directions also make the cumulative basis
    rank deficient. Stage 0.3 must resolve this before decoder training.
14. The K4096 scratch run confirms that reusing decoder-specialized crop1.2
   fusion features was part of the failure: unlike the frozen-feature run, it
   continues reducing train loss. It nevertheless plateaus below the original
   FullGrid validation baseline while the ReZero gate reaches its bound. The
   remaining failure must therefore be split into coefficient-head capacity,
   basis-loss conditioning, memorization, and sequence-disjoint generalization.
15. Stage 0.6 confirms that FullGrid uses image-space arrangement. On the fixed
    32,768-frame validation subset, spatial token shuffle lowers Contact-IoU by
    `.0515` and CoreLoc by `.0458`; a one-cell cyclic shift lowers them by only
    `.0092/.0085`. The frozen representation therefore contains useful but
    mostly coarse spatial evidence. It does not by itself establish a stable
    image-to-canonical correspondence.
16. The nonlinear K4096 full-data run is rejected as a FullGrid replacement.
    At `loss-best`, Seen/Unseen Contact-IoU changes by `-.0119/-.0543`, CoreLoc
    by `-.0122/-.0221`, and RMSE by `+.00194/+.00203`. Its modest false-high
    reduction comes with lower active recall and substantially more
    catastrophic-under. Official val was already below FullGrid; the extra
    failure on test_unseen is task-level generalization, not merely checkpoint
    selection. Increasing coefficient count or global MLP width is paused.
17. Stage 0.7 exonerates the surface basis but rejects the learned global
    mapping. The split-wise K4096 GT oracle reaches approximately
    `.8533-.8655` Contact-IoU and `.8189-.8218` CoreLoc on all four official
    splits. The parameter-matched direct valid-vertex control recovers part of
    the learned-basis loss, yet remains below FullGrid on Seen/Unseen Contact by
    `.0153/.0198`. Fixed-basis coupling is therefore a secondary cost; the
    dominant learned bottleneck is RGB-to-canonical assignment.
18. Stage 0.7 token influence is highly global: normalized influence entropy is
    about `.993`, effective token count is about `185/192`, canonical anchor
    maps have cosine similarity about `.972`, and each anchor uses roughly
    `12-14` distinct top tokens with near-zero canonical-distance correlation.
    FullGrid uses spatial arrangement in aggregate, but it does not expose a
    selective image-token-to-canonical-region route. Stage 2 must test that
    route directly instead of adding another global output MLP.
19. Stage 2 V1 does not establish canonical correspondence. All eight
    `loss-best` variants slightly reduce pressure loss/RMSE, but they also
    reduce Contact-IoU, CoreLoc, and high-pressure prediction through a broad
    downward correction. Test-time token shuffle changes Contact/CoreLoc by
    less than roughly `.0006`, so the apparent gain does not depend on the
    alignment between image content and token position. The 512-anchor geometry
    is more local than 256 anchors, but that geometric improvement does not
    become a metric improvement. V1 also permits learned query, null, and bias
    paths to produce a residual without frame-specific image evidence.

## Working Hypothesis

The dominant problem is an underconstrained registration bottleneck:

```text
image-space evidence
-> globally mixed FullGrid bottleneck
-> fixed canonical vertex vector
```

The dataset supervises pressure only after it has already been mapped to a
canonical mesh. It does not explicitly say which image region corresponds to
which canonical vertex. Crop normalization, left/right anonymization,
occlusion, object contact, and articulation make that correspondence
many-to-one or visually ambiguous. A dense MLP can therefore learn a strong
dataset prior and total-pressure estimate while averaging uncertain local
placements.

This hypothesis has four separable parts:

1. **Magnitude:** Is total pressure already accurate enough?
2. **Target representation:** Are contact maps low-complexity components, or do
   they require hundreds of local degrees of freedom?
3. **Visual observability:** Do similar frozen image features map to incompatible
   canonical pressure layouts even after matching total pressure?
4. **Decoder controllability:** Can a local feature change affect a local mesh
   region without moving the rest of the palm?

## Status

| Stage | Status | Purpose | Output / Decision |
|---|---|---|---|
| 0.0 Existing evidence | Complete | Establish capacity and local-control failure | Evidence above |
| 0.1 Canonical diagnosis | Complete | Separate magnitude, topology, patch resolution, and visual ambiguity | `canonical_localization_audits/stage0` |
| 0.2 Stratified basis audit | Complete | Remove prefix bias and test a sensor-independent continuous basis | 1024-level basis has capacity; raw cumulative coordinates rejected |
| 0.3 Basis cleanup | Complete | Make the basis identifiable, smooth to optimize, and safe | Standalone 1536 sigmoid is the only strict candidate; multiscale concatenation rejected |
| 0.4 Capacity ceiling | Complete | Locate saturation before committing to a learned decoder | 4096 is the useful knee, but the fixed bandwidth becomes rank deficient |
| 0.4b Density cleanup | Complete | Test whether high-dimensional rank loss is caused by the mesh-edge bandwidth floor | Support 4 restores rank; compare 4096/5120 |
| 0.5 Decoder learnability | Complete | Separate linear capacity, coefficient supervision, 1K fit, and sequence-disjoint generalization | Nonlinear pressure-only wins the surface family; coefficient auxiliary rejected |
| 0.6 FullGrid spatial dependency | Complete | Test whether the successful baseline truly depends on token placement | Spatial arrangement matters, but one-cell displacement sensitivity is modest |
| 0.7 Mapping attribution | Complete | Separate basis expressivity, basis coupling, and global token routing on official splits | Basis capacity passes; learned global mapping fails |
| 1 Continuous field decoder | Rejected in global form | Replace the shared 512-D vertex decoder with the selected standalone canonical basis | K4096 nonlinear global coefficient MLP loses to FullGrid, especially on unseen tasks |
| 2 Canonical routing V1 | Complete; rejected for attribution | Route projected FullGrid evidence into canonical anchors | `canonical_anchor_routing`; pressure loss improves slightly, correspondence controls fail |
| 2.1 Evidence-only routing | Implemented, pending run | Remove static residual bypasses and compare projected32 with pre-projection ReZero256 | `run_canonical_localization.sh routing-v2` |
| 3 Mass/location split | Pending | Predict total mass separately from normalized canonical distribution | Use after continuous representation is validated |
| 4 Extra priors | Paused | Add depth, VLM, or temporal evidence after routing exists | Real prior must beat shuffled/mismatched control |
| 5 Weak supervision | Fallback | Add correspondences only if RGB supervision is underdetermined | Synthetic/weak labels before full pose dependency |

## Stage 0.1: Canonical Diagnosis

Implementation:

- [`audit_canonical_localization.py`](audit_canonical_localization.py)
- [`run_canonical_localization.sh`](run_canonical_localization.sh)

The audit reuses the frozen crop1.2 validation feature cache. It does not read
images, rerun DINO, train parameters, or modify a checkpoint.

Historical artifacts:

```text
canonical_localization_audits/stage0
```

Artifacts:

| File | Meaning |
|---|---|
| `mass_distribution.csv` | All-frame base plus matched-population GT-mass/base-distribution and base-mass/GT-distribution comparison |
| `component_summary.csv` | Contact component counts and top-K mass/vertex coverage at several thresholds |
| `component_per_sample.csv` | Per-sample topology records for distribution plots and stratification |
| `patch_reconstruction.csv` | Oracle error from 32/128/512 topology-respecting canonical patch means |
| `label_rank.csv` | Effective rank of normalized patch pressure and binary patch contact |
| `ambiguity_pairs.csv` | Different-sequence, matched-mass visual nearest neighbors and pressure-layout agreement |
| `canonical_patch_partitions.npz` | Reusable geodesic-FPS anchors and hard canonical ownership maps |
| `summary.json` | Machine-readable aggregate and full provenance |

Interpretation:

- Canonical placement is the primary bottleneck; per-frame magnitude remains a
  meaningful secondary error despite cancellation in the dataset mean.
- A small component set is rejected by the observed component count and top-K
  coverage. Stage 0.2 will confirm that result without the `Home`-only bias.
- Hard 32/128/512 patch means are rejected because they erase peaks and create
  broad false-high pressure. Overlapping continuous functions remain untested.
- The original visual-ambiguity trend is provisional because the first audit
  sampled only `Home`. Do not claim an RGB information limit until the
  stratified rerun and a stronger spatial descriptor agree.
- The patch reconstruction rows are representation upper bounds, not model
  results. They must never be presented as deployable performance.

## Stage 0.2: Stratified Sensor-Independent Basis Audit

Run:

```bash
./hamer_tactile_ft/run_canonical_localization.sh stage1
```

The component, ambiguity, and basis subsets are independently balanced over
`scene/task/sequence`; no subaudit may consume a source-order prefix. The new
oracle builds `32/128/512/1024` geodesic-FPS anchor banks on the canonical palm.
Each bank uses overlapping physical-geodesic RBFs normalized as a partition of
unity. Anchors depend only on canonical geometry, never on sensor placement.

The nested dictionaries are fitted to held-out GT with deterministic ridge
projection. This is a representation diagnostic, not a deployable result.
Inspect:

```text
surface_basis_reconstruction.csv
subaudit_sampling.csv
canonical_surface_basis.npz
```

Result:

- `32/160/672` cumulative dimensions are below the same-subset FullGrid
  localization baseline.
- `1696` dimensions provide useful representation headroom, especially for
  contact support, CoreLoc, and high-pressure recovery.
- The result relies on signed cancellation and hard lower clipping, and it
  retains a false-high tradeoff. It passes representation capacity but fails
  direct trainability/identifiability.

## Stage 0.3: Basis Identifiability And Metric Cleanup

Run:

```bash
./hamer_tactile_ft/run_canonical_localization.sh cleanup
```

Implementation:

- [`audit_surface_basis_cleanup.py`](audit_surface_basis_cleanup.py)
- [`run_canonical_localization.sh`](run_canonical_localization.sh)

The runner automatically finds the completed Stage 0.2 directory and reuses
its exact 2,048 balanced samples. It reads cached labels only; it does not run
DINO, rebuild the 50,000-frame audit, or train a model.

The audit compares:

```text
physical-geodesic standalone RBF banks: 512/768/1024/1536
physical-geodesic cumulative 32+128+512+1024
local hierarchical zero-mass signed details
pressure-space ridge + hard clip (diagnostic only)
logit-space ridge + sigmoid
1024-anchor nonnegative coefficient control
```

It records natural support/fallback counts, rank, conditioning, signed
coefficient statistics, negative pre-link mass, saturation, bootstrap intervals,
and all localization/safety metrics. Selection is deliberately not a weighted
scalar score. Candidates first pass transparent guardrails, then retain every
non-dominated Pareto option.

Metric roles:

| Role | Metrics | Interpretation |
|---|---|---|
| Primary location | Contact-IoU, V-IoU, CoreLoc | Must provide enough oracle headroom for learned prediction |
| Distribution/calibration | Distribution V-IoU, vertex RMSE | Must not buy threshold gains by damaging the full field |
| Safety | False-high, high-pressure recovery | Suppress halo without erasing peaks |
| Trainability | Rank, condition, fallback, smooth link | Hard-clipped or non-identifiable coordinates cannot be promoted |

Default capacity guardrails require at least `+.03` Contact, V-IoU, and CoreLoc
headroom over the same 2,048-frame FullGrid baseline, no RMSE degradation, and
no more than `.005` Distribution V-IoU degradation. Safety requires false-high
within `1.10x` baseline and no loss of high-pressure recovery. These thresholds
are diagnostic guardrails, not a paper score or a substitute for inspecting
the Pareto front and confidence intervals.

Proceed to Stage 1 only when at least one smooth-link, naturally supported,
numerically usable candidate satisfies both capacity and safety. If no variant
passes, refine the representation rather than compensating with Depth, VLM,
temporal input, pose, or additional loss terms.

Stage 0.3 result:

- Physical weighted-geodesic FPS removed every support fallback.
- Standalone 1536 is full-rank and better conditioned than the lower-dimensional banks.
- Standalone 1536 pressure-clamp is the representation ceiling; it is diagnostic only.
- Standalone 1536 logit-sigmoid is the only smooth strict candidate.
- Hierarchical and cumulative variants remain rank deficient and leave the main line.
- NNLS shows that forcing every coefficient nonnegative creates excessive halo.

The unsaturated standalone curve was measured with:

```bash
./hamer_tactile_ft/run_canonical_localization.sh capacity
```

This evaluated `1536/2048/3072/4096/5120/6144`, wrote each completed
dimension to a resumable partial artifact, and skipped the rejected multiscale
and NNLS controls. Clamp remains an oracle ceiling and sigmoid remains the
deployable link. Oracle quality continued to improve through 6144, with the
largest practically useful step ending around 4096. However, the old bandwidth
hit the median mesh-edge floor at 2048. Median support then grew from 5 at 2048
to 17 at 6144; rank deficiency grew from 0 at 3072 to 5/41/407 at
4096/5120/6144. Therefore the high-dimensional curve confounds basis capacity
with increasingly redundant overlap.

Run one final density-controlled confirmation before Stage 1:

```bash
./hamer_tactile_ft/run_canonical_localization.sh density
```

This runs target support `4/6/8` concurrently on GPUs `0/1/2`, restricted to
`3072/4096/5120/6144`. Each child remains resumable under
`stage0_4b_basis_density/support_N`; the root aggregate validates the shared
sample/base contract and writes `basis_density_curve.csv` plus `summary.json`.
The adaptive support radius follows anchor density while retaining natural
coverage, instead of imposing the old edge-length bandwidth floor.

Decision after Stage 0.4b:

- Prefer 4096 only if adaptive overlap removes its five missing rank directions,
  keeps regularized condition within the existing numerical guardrail, and
  preserves its material oracle gain over 3072.
- Consider 5120/6144 only if redundancy falls sharply and they remain materially
  better than 4096 after the cross-support near-best comparison.
- Otherwise use 3072 for the first learned decoder; do not buy oracle capacity
  with an ill-conditioned coefficient space.

Observed Stage-0.4b decision:

- Target support 4 removed the old high-dimensional rank deficiency at both
  4096 and 5120 while preserving exact partition of unity and zero fallback
  vertices.
- 5120 was the only smooth strict near-best oracle under the registered
  tolerances; 4096 remained the lower-cost, numerically steadier eligible
  control.
- Stage 1 therefore compares 4096 and 5120 directly. Support 6/8 and 6144 do
  not enter the first learned experiment.

## Stage 1: Continuous Surface Field Control

Use a shared function over the canonical hand surface. Stage 0.3 rejected
concatenated multiscale coordinates, so the first decoder uses one selected
standalone weighted-geodesic bank:

```text
p(I, u) = sigmoid(sum_k coefficient_k(I) * basis_k(u))
```

`u` is an arbitrary canonical surface point. Evaluation samples this function
at 13,614 vertices, but training and inference semantics do not depend on that
mesh resolution. Pure spectral output, hard patches, cumulative/hierarchical
basis concatenation, and native sensor cells are not candidates. A bounded
local residual remains a later option only if the learned standalone field
shows a localized, systematic representation error.

The implemented first comparison trains a new coefficient head from scratch
while loading and freezing the crop1.2 loss-best DINO/ReZero/FullGrid feature
extractor:

```text
16x12x32 FullGrid
-> direct standalone coefficient head
-> fixed continuous canonical basis
-> sampled pressure field
```

Do not attach it as a residual to the frozen 512-D decoder. The experiment must
test whether removing `6144 -> 512 -> 13,614` improves local controllability.
The old `6144 -> 512 -> 13,614` decoder is absent. The exact audited basis is
stored as sparse per-vertex support indices and weights, so it retains the
support-4 field while avoiding dense zero multiplication and large DDP buffer
broadcasts.

Run the two-server comparison with:

```bash
./hamer_tactile_ft/run_tactile_experiment.sh surface-s4-k4096-r256
./hamer_tactile_ft/run_tactile_experiment.sh surface-s4-k5120-r256
```

The first K=4096 run reached its best gains during warmup while coefficient
RMS continued to grow after validation gains flattened. Preserve those runs
as the original `5e-5` base-LR controls and use fresh experiment directories
for the half-LR optimization diagnostic:

```bash
./hamer_tactile_ft/run_tactile_experiment.sh surface-s4-k4096-lrhalf-r256
./hamer_tactile_ft/run_tactile_experiment.sh surface-s4-k5120-lrhalf-r256
```

These presets use base LR `2.5e-5`, hence effective LR `2e-4` on eight GPUs.
They still initialize only the frozen feature extractor from crop1.2; they do
not resume the original surface-head optimizer or coefficient weights.

If reducing LR does not explain the early plateau, test whether the crop1.2
ReZero/FullGrid features were specialized for the removed dense decoder. Run a
single K=4096 from-scratch control:

```bash
./hamer_tactile_ft/run_tactile_experiment.sh surface-s4-k4096-scratch-r256
```

This control restores base LR `5e-5` (effective `4e-4` on eight GPUs), does not
load the crop1.2 tactile checkpoint, and jointly trains fresh ReZero fusion,
FullGrid projection, and the coefficient head. The pretrained DINOv3 backbone
remains frozen, preserving the mainline backbone assumption.

Stage 0.5 shows that the direct linear coefficient map is itself a material
bottleneck. The next formal experiment keeps the same K4096 support-4 field,
pressure loss, crop, data, optimizer, and frozen DINO, but replaces only the
coefficient predictor:

```text
6144 FullGrid features
-> LayerNorm
-> 1024-D projection + LayerNorm + GELU + dropout
-> two 1024-D residual MLP blocks
-> 4096 surface coefficients
```

Run it from scratch rather than initializing the linear surface model:

```bash
./hamer_tactile_ft/run_tactile_experiment.sh surface-nl-k4096-scratch-r256
```

The output layer remains zero-initialized at the same background probability.
This preserves stable initial pressure while jointly adapting the fresh ReZero
fusion, FullGrid32 projection, and nonlinear coefficient decoder. Loading a
linear surface checkpoint would confound the architecture comparison and is
not shape-compatible with the new hidden path.

Compare both against the current FullGrid baseline using `loss-best`; use
`last` only for drift. A parameter-matched direct-vertex MLP remains a
follow-up attribution control only if one learned surface field is competitive.
Keep loss, crop, sampling, seed, and checkpoint policy fixed.

### Stage 0.5: Layered Decoder Learnability

Run:

```bash
./hamer_tactile_ft/run_canonical_localization.sh learnability
```

The entry prepares one immutable, sequence-disjoint probe dataset and then
uses eight GPUs for this matrix:

```text
linear vs nonlinear coefficient predictor
x pressure-only vs pressure + ridge-coefficient auxiliary
x generalization split vs no-dropout 1K memorization
```

Every cell receives identical frozen crop1.2 FullGrid features, targets,
surface basis, initialization policy, and sample split. Ridge coefficient
teachers are computed once and cached; they are diagnostic training targets,
not deployable model inputs. Read `comparison.csv`, `contrasts.csv`, and
`interpretation.json` together:

- nonlinear over linear isolates decoder capacity;
- coefficient auxiliary over pressure-only isolates basis-gradient
  conditioning;
- good 1K fit but poor held-out results identifies observability or
  generalization;
- poor 1K fit in all four cells identifies parameterization or optimization.

Observed result:

- On the sequence-disjoint split, nonlinear pressure-only improves Contact-IoU
  over linear pressure-only by `.0410`, CoreLoc by `.0114`, Distribution V-IoU
  by `.0053`, and lowers vertex RMSE by `.00343`, despite using fewer parameters
  (`14.71M` versus `25.18M`).
- Direct ridge-coefficient supervision worsens pressure metrics for both
  architectures. It is rejected from formal training; basis coefficients are
  non-unique enough that matching one ridge solution is not the task objective.
- The nonlinear pressure-only head reaches `.8661` Contact-IoU on the 1K
  memorization set but only `.3643` on the held-out split. The representation
  and optimizer can fit local fields; generalization/correspondence remains the
  larger bottleneck.

### Stage 0.6: FullGrid Spatial Dependency

Before adding canonical routing or pose, audit what the successful crop1.2
FullGrid baseline already extracts from its `16x12` token arrangement:

```bash
./hamer_tactile_ft/run_canonical_localization.sh spatial-dependency
```

The audit uses cached fused grids and the frozen baseline decoder. It does not
rerun DINO or train a model. On identical samples it compares:

```text
identity         unchanged baseline grid
global_mean      remove all spatial variation
spatial_shuffle  preserve the token multiset, destroy token-to-position mapping
block_shuffle    preserve each 2x2 local block, destroy global block placement
cyclic_shift     preserve all token values, move them by one grid cell
```

`metrics.csv` records task metrics and prediction deltas; `summary.json` and
`AUDIT_DONE.json` bind the result to exact cache/checkpoint hashes. A large
shuffle/shift penalty means FullGrid already uses image-space position and the
next problem is canonical routing. A small penalty means its gain is mostly
global appearance/statistics, so adding a more local output head alone cannot
solve correspondence.

Observed result:

- Global-mean replacement lowers Contact-IoU by `.0933`.
- Spatial/block shuffle lowers Contact-IoU by `.0515/.0586` and CoreLoc by
  `.0458/.0518`.
- A one-cell cyclic shift lowers Contact-IoU/CoreLoc by only `.0092/.0085`.

The next problem is therefore not whether FullGrid has any spatial signal. It
is whether that signal can be routed selectively into canonical regions.

### Stage 0.7: Mapping Attribution

Run:

```bash
./hamer_tactile_ft/run_canonical_localization.sh attribution
```

The first run creates persistent, resumable frozen-feature subsets before any
probe training:

```text
train:       up to 131,072 rows, max 64 per sequence/query
val:         up to  32,768 rows, max 384 per sequence/query
test_seen:   up to  32,768 rows, max 384 per sequence/query
test_unseen: up to  32,768 rows, max 384 per sequence/query
```

Sampling uses stable sample-UID hashes and occurs before image/DINO reads. The
same manifest, seed, and limits produce the same set regardless of GPU count.
Existing cache behavior remains unchanged when the new limits are zero.

The audit then answers three separate questions:

1. **Split-wise basis oracle:** solve ridge-optimal K4096 coefficients from GT
   on a stable maximum of 8192 rows independently on train/val/seen/unseen. A
   strong unseen oracle exonerates basis expressivity and places the failure in
   RGB-to-canonical mapping. The larger prepared arrays remain available to the
   learned-head comparison without paying for unnecessary per-frame solves.
2. **Parameter-matched output control:** train the same nonlinear family on
   official train features as either `4096 -> fixed basis` or direct 6623 valid
   logits. The direct hidden width is selected automatically to match the
   K4096/1024 model within about `0.1%` parameters. Both use official val loss
   for checkpoint selection; test never selects a checkpoint.
3. **Token influence:** replace each of the 192 FullGrid tokens by its
   per-channel spatial mean and measure anchor-logit changes. Entropy,
   effective token count, top-token diversity, anchor-map cosine similarity,
   and canonical-distance correlation diagnose whether output regions read
   selective or nearly identical global evidence.

Primary artifacts:

```text
surface_mapping_attribution/comparison.csv
surface_mapping_attribution/basis_oracle_by_split.csv
surface_mapping_attribution/token_influence_comparison.csv
surface_mapping_attribution/interpretation.json
```

Do not use the oracle as supervision and do not promote direct valid-vertex
output as the final sensor-independent architecture. It is an attribution
control. If both learned heads fail while all split oracles remain strong,
proceed to Stage 2 local canonical routing. If direct clearly wins, quantify
how much fixed-basis coupling contributes before designing that route.

### SAM3 Mask Scope

The hand mask remains paused through Stage 0.6 and the nonlinear K4096 run.
It can add image-plane foreground, silhouette, boundary, and occlusion cues
even though it does not identify canonical mesh vertices. It
must not hard-mask RGB, because object pixels and the hand-object boundary are
contact evidence. When the mask branch resumes, test it as auxiliary spatial
metadata at the DINO grid: occupancy, signed distance to the boundary, and
optionally a boundary channel. Keep ordinary RGB/object tokens intact.

The required controls are `real mask`, `spatially shuffled mask`, and a
`bbox rectangle/all-ones` control. This matters because the current bbox is
already derived from the same SAM3 mask; a real-mask gain must exceed what its
box has already supplied. The compact SAM3 reconstruction currently persists
bboxes, area, and centroid rather than a reusable per-frame pixel mask, so a
mask experiment also requires an explicit resumable mask/RLE export instead of
reading preview videos.

## Stage 2: Canonical Anchor Routing

Stage 0.7 supplies the gate for this stage: the K4096 representation has ample
split-wise oracle capacity, while the learned global heads and FullGrid token
influence fail to provide selective canonical routing.

Implementation:

- [`audit_canonical_anchor_routing.py`](audit_canonical_anchor_routing.py)
- [`run_canonical_localization.sh`](run_canonical_localization.sh)

The model keeps the successful frozen FullGrid prediction as an exact base and
learns a zero-initialized local surface residual:

```text
frozen FullGrid 16x12x32 tokens + fixed 2D position
-> 256/512 canonical XYZ queries
-> two image-to-anchor routing blocks with explicit null state
-> one locally owned coefficient group per anchor
-> fixed support-4 K4096 surface basis
-> bounded logit residual + frozen FullGrid logits
```

This is deliberately not another replacement decoder. At initialization its
output equals FullGrid, and the trainable branch can only move pressure through
local basis functions owned by one canonical anchor. There is no vertex or
anchor self-attention.

The strict 8-run matrix is:

```text
anchors: 256, 512
routing: competitive, independent
source:  spatial, global_control
```

`competitive` normalizes token ownership across anchors before per-anchor
aggregation, while `independent` is the standard per-anchor token softmax.
Spatial and global controls have identical trainable parameters and
initialization. Evaluation additionally applies deterministic spatial token
shuffle to spatial checkpoints without using it for selection.

Run the whole stage on eight GPUs:

```bash
./hamer_tactile_ft/run_canonical_localization.sh routing
```

Or keep training and evaluation separate:

```bash
./hamer_tactile_ft/run_canonical_localization.sh routing-train
./hamer_tactile_ft/run_canonical_localization.sh routing-eval
```

Training resumes atomically from each run's `last.pt`; official validation
pressure loss selects `best_loss.pt`. The immutable prepared arrays from Stage
0.7 are reused and rebuilt only when missing. Geometry ownership and geodesic
locality are precomputed once per anchor count before parallel GPU jobs start.

Primary artifacts:

```text
canonical_anchor_routing/comparison.csv
canonical_anchor_routing/control_comparison.csv
canonical_anchor_routing/locality_comparison.csv
canonical_anchor_routing/interpretation.json
canonical_anchor_routing/<config>/checkpoints/{last,best_loss}.pt
canonical_anchor_routing/<config>/evaluation/summary.json
```

Interpret routing as successful only when the spatial model improves the frozen
base and beats both its global-token training control and its test-time token
shuffle. Attention selectivity, null use, anchor-load balance, canonical
distance correlation, and geodesic leakage are attribution evidence, not
standalone success criteria.

### Stage 2 V1 Result

The V1 matrix is complete under:

```text
canonical_anchor_routing
```

It is rejected as evidence of image-to-canonical routing. The learned branch
mostly discovers a useful dataset-level downward calibration: pressure loss
and RMSE move slightly in the desired direction, while Contact-IoU, CoreLoc,
and high-pressure prediction move in the wrong direction. Real and
`shuffle_spatial` evaluations are nearly identical. Increasing the number of
geometry anchors improves ownership locality but does not improve the formal
metrics.

This result does not yet prove that the frozen visual representation lacks the
needed evidence. V1 reads only the projected 32-channel FullGrid tokens and
contains paths through learned queries, null values, and coefficient biases
that can generate a mostly static correction. Stage 2.1 isolates these two
remaining explanations.

### Stage 2.1: Evidence-Only Routing Rescue

Stage 2.1 fixes the geometry and routing family and changes only the evidence
contract and feature source:

```text
frozen FullGrid base prediction

image content -> attention keys
spatially centered image content -> bias-free values
canonical query + XYZ -> attention weights only
last routed image evidence -> bias-free local K4096 coefficient readout
-> bounded local surface-logit residual + frozen base logits
```

There is no learned null value, value/output bias, coefficient bias, or query
readout. A spatially constant/global-repeat feature field therefore produces
an exactly zero residual even after training. This is a structural identity
control rather than another separately trained model.

The comparison is:

```text
architecture: evidence_only
anchors: 256
routing: competitive
layers/dimension/heads: 2/128/4
feature source A: projected32 [32,16,12]
feature source B: rezero256 [256,16,12]
seeds per source: 521, 2029, 3407, 4099
formal selection: best official-validation pressure loss
```

The source-specific 32/256-channel projections use an isolated deterministic
random stream. Consequently, for a given seed every shared anchor, routing
block, and coefficient-readout parameter starts identically across the two
feature sources even though the input projections have different shapes.

The persistent attribution cache already contains the ReZero fused grid, so
the preparation step aligns and stores it once without rerunning DINO. The
four official splits require about 21 GiB of FP16 storage. Completed alignment,
training checkpoints, and evaluations are all reusable and resumable.

Run training and evaluation separately:

```bash
./hamer_tactile_ft/run_canonical_localization.sh routing-v2-train
./hamer_tactile_ft/run_canonical_localization.sh routing-v2-eval
```

Or run the complete eight-GPU workflow:

```bash
./hamer_tactile_ft/run_canonical_localization.sh routing-v2
```

Artifacts are written to:

```text
canonical_anchor_routing_v2/raw_prepared/{train,val,test_seen,test_unseen}
canonical_anchor_routing_v2/{p32,r256}_s{seed}
canonical_anchor_routing_v2/comparison.csv
canonical_anchor_routing_v2/control_comparison.csv
canonical_anchor_routing_v2/seed_comparison.csv
canonical_anchor_routing_v2/seed_control_comparison.csv
```

The decision is evidence-based rather than threshold-only. A viable route must
improve the frozen base consistently across seeds, make aligned real tokens
better than shuffled tokens, and show materially more selective per-frame
attention. If `rezero256` succeeds where `projected32` fails, the FullGrid32
projection discarded correspondence. If both fail and shuffle remains
irrelevant, stop decoder-only pose-free routing and move to weak
correspondence/part supervision before adding depth, VLM, or pose.

The implementation follows these constraints:

1. Keep DINO frozen and preserve the Stage-1 continuous output field.
2. Retain spatial DINO tokens until after canonical routing; do not collapse to
   one 512-D vector first.
3. Route into 256/512 geometry-defined anchors that emit local basis
   coefficients, never sensor values or independent vertex logits.
4. Include a null/occluded state so invisible anchors are not forced to read an
   unrelated image token.
5. Compare real tokens with global-token and token-shuffle controls using equal
   parameter counts.
6. Add a mesh-neighborhood leakage audit to every evaluation.

Cross-attention remains optional in the longer route. This first controlled
implementation uses compact image-to-anchor attention because it exposes an
exact global-content control and explicit token assignments. Sparse assignment,
deformable sampling, or dynamic convolution remain alternatives only if the
controlled result shows useful spatial information but inadequate routing.

## Stage 3: Magnitude and Distribution

Only activate this stage if Stage 0 and the selected local decoder show that
total pressure and normalized location have different error modes.

```text
M = nonnegative frame total pressure
pi = normalized pressure distribution over valid vertices
prediction = bounded_composition(M, pi)
```

Train `M` with a scalar volume objective and `pi` with a distributional loss.
Retain the direct pressure loss for calibration. Compare soft V-IoU, CoreLoc,
and a bounded mesh-geodesic transport loss one at a time. Do not reintroduce a
large collection of simultaneous losses.

## Stage 4: Priors and Temporal Evidence

Depth, VLM, and tactile history return only after a local canonical route exists.

1. **Depth:** modulate local image tokens or patch evidence before canonical
   routing. Require aligned depth to beat spatial shuffle and wrong-frame depth.
2. **VLM:** condition component/patch existence and uncertainty, not individual
   vertices. Require meaningful prompt variation to beat embedding shuffle.
3. **Temporal:** use the preserved tactile-flow findings to update component or
   patch state, with true `delta_t`, lag quality, and anonymous A/B query state.

Do not use any prior merely because it produces a residual close to zero. The
real prior must produce a selective benefit unavailable to its control.

## Fallback Supervision Ladder

If matched-mass ambiguity is high and all controlled RGB routing models fail:

1. Add hand/object segmentation and synthetic render correspondence as weak
   image-to-canonical supervision.
2. Use cycle-consistent canonical surface embeddings or dense contrastive
   matching without requiring pose at inference.
3. Use sparse keypoints or a coarse hand-part parser only as training-time
   regularization.
4. Add estimated MANO/hand pose as a confidence-gated optional prior.
5. Use GT pose only as an oracle to quantify remaining headroom, never as the
   first deployable solution.

The pose-free preference is therefore preserved, but pose can still serve as a
diagnostic upper bound if the data prove underdetermined.

## Paused Work

```text
Native sensor-grid output or latent prediction
Wrist-view input or evaluation dependency
Hard piecewise-constant canonical patches
Further full-vertex cross-attention variants
CSE-style decoder variants
Standalone FullGrid resolution increases
Depth/VLM pressure-level modulation
RGB-only selector pressure policies
Uncontrolled or hard SAM3 mask token gating
Vertex self-attention
Flow matching before conditional multimodal evidence is established
Mamba/Neural CDE before long, reliable trajectories exist
```

## Update Protocol

For each completed stage:

1. Add the experiment path and checkpoint selector to the status table.
2. Record real-vs-control results separately from base-vs-fused results.
3. State what was falsified, what remains possible, and the next single decision.
4. Preserve exact commands and output schema when an audit becomes canonical.
5. Move failed branches to Paused Work; do not delete their evidence.
6. Add a dated changelog entry below.

## Changelog

- **2026-08-27:** Paused tactile-flow work and established canonical
  localization as the primary single-frame problem. Implemented Stage 0.1 with
  mass/distribution, component topology, geodesic patch, label-rank, and
  matched-mass visual ambiguity audits.
- **2026-08-27:** Made sensor-layout independence and single-main-view inference
  hard constraints. Rejected native sensor grids as output/latent semantics,
  recorded Stage-0 findings and prefix-sampling bias, and implemented the
  stratified multiscale canonical surface-basis audit as Stage 0.2.
- **2026-08-27:** Completed the Stage-0.4b density audit, fixed target support
  to 4, and selected 4096/5120 for the learned comparison. Implemented the
  direct continuous surface coefficient head, exact sparse runtime basis,
  frozen crop1.2 feature initialization, compact-checkpoint recovery, and
  two short training presets.
- **2026-08-28:** Recorded the K4096 scratch result and added Stage 0.5. The
  new reusable probe cache and eight-GPU matrix separate decoder capacity,
  coefficient supervision, memorization, and held-out generalization. Scoped
  SAM3 masks to controlled auxiliary 2D geometry rather than canonical
  correspondence.
- **2026-08-28:** Completed Stage 0.5. Promoted the nonlinear pressure-only
  K4096 coefficient decoder to the next from-scratch experiment, rejected the
  ridge-coefficient auxiliary, and added a cache-only FullGrid spatial
  dependency audit before canonical routing.
- **2026-08-29:** Completed Stage 0.6 and rejected the global nonlinear K4096
  coefficient decoder after its official loss-best result degraded both splits,
  especially unseen tasks. Added Stage 0.7 with deterministic split-subset
  caches, split-wise basis oracles, parameter-matched direct/basis heads, and
  token-influence diagnostics. Stage 2 remains gated on this attribution.
- **2026-08-29:** Completed Stage 0.7. The split-wise basis oracle remains
  strong, the parameter-matched direct head only partially recovers FullGrid,
  and token influence is nearly global across canonical anchors. Implemented
  Stage 2 as an eight-run 256/512-anchor competitive/independent routing matrix
  with spatial/global controls, test-time token shuffle, explicit null states,
  local K4096 ownership, frozen FullGrid base, resumable training, and separate
  evaluation/aggregation.
- **2026-08-29:** Completed Stage 2 V1 and rejected it as correspondence
  evidence: all variants trade broad pressure suppression for slightly lower
  pressure loss, while token shuffle is nearly inert. Implemented Stage 2.1
  evidence-only routing, exact global-repeat identity, per-frame selectivity
  diagnostics, aligned ReZero256 cache reuse, and an eight-GPU two-source by
  four-seed matrix with separate resumable training and evaluation.
