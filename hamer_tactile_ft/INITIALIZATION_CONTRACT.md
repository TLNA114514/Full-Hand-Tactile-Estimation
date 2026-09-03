# Tactile Initialization Contract

## Canonical rule

Every newly constructed RGB tactile base uses:

```text
model_initialization_order=legacy_decoder_first
```

This is the RNG assignment order of the July 2026 TouchAnything crop1.2
reference model. It applies to all `run_tactile_experiment.sh` presets,
including the FullGrid channel/hidden-width matrix and future presets that
inherit `DINO_COMMON`.

The same preset family also fixes the other replay-sensitive defaults found in
the baseline audit:

```text
worker_seed_mode=lightning_legacy
hdf5_sample_order=legacy_sample_dir_hand
crop_pipeline=legacy_square_center
optimizer_backend_mode=legacy_default
```

These are control variables, not ablation variables. Dataset composition,
crop scale, loss, resolution, and model dimensions may still differ when that
is the stated purpose of an experiment.

The official `tactile_input_priors.runtime.build_dataset` path uses the same
crop pipeline and HDF5 ordering. Cache-only experiments remain tied to the
provenance of the cache they load and must not mix caches from another contract.

The Python constructors and `train.py` use the same canonical default. The
`projection_first` option remains available only for explicit reproduction of
experiments that recorded that order. Such a replay must provide both:

```text
--model_initialization_order projection_first
--allow_noncanonical_model_initialization
```

Without the explicit opt-in, training fails before constructing the model.

## Checkpoint-based experiments

An adapter, selector, temporal model, evaluator, or cache builder that loads a
tactile checkpoint must reconstruct the base using the order recorded in that
checkpoint. A checkpoint with no order field falls back to the canonical rule.
The loaded state dict, rather than random initialization, remains authoritative
for the frozen base weights.

## Provenance and resume

Training records both:

```text
model_initialization_order
initial_tactile_head_sha256
```

in provenance, model config, compact checkpoints, and resume contracts. Resume
must reject an explicitly conflicting order. Evaluation restores the recorded
order.

## Ablation interpretation

The eight channel/hidden-width configurations use the same construction rule,
seed, and preset family. Their initial state hashes are not expected to be
identical because tensor shapes and parameter counts differ. The invariant is
the RNG assignment protocol, not byte-identical weights across different
architectures.

Do not add a new tactile-base preset with a private initialization order. Add it
through `DINO_COMMON`; if a noncanonical replay is genuinely required, name it
as a historical replay and keep its order explicit.

## Fresh temporal trunks

`hierarchical_memory_v3` is a fresh tactile-base experiment, not a checkpoint
adapter. It constructs the complete canonical FullGrid/ReZero base first,
records `initial_tactile_head_sha256`, then appends temporal writers. L12 and
L124816 construct the shared fast writer before the long model appends its
medium writer and record `initial_fast_writer_sha256`. A tactile base checkpoint
is rejected for these presets. Same-experiment `resume.ckpt` remains valid for
recovering optimizer/scheduler/RNG progress after interruption.

`causal_clip_transformer_v4` follows the same fresh-base rule. It constructs
the complete canonical FullGrid/ReZero tactile head first, records
`initial_tactile_head_sha256`, and only then appends the two-layer causal clip
Transformer. `initial_temporal_module_sha256` records the whole clip module at
construction time. It never initializes from crop12 or another temporal
checkpoint; only the same experiment's `resume.ckpt` may restore training
state. Frozen DINO weights are external and verified by SHA256.

The active `fullgrid6144_bidirectional_v5` route also starts from a fresh
canonical base. Construction order is fixed as:

```text
1. canonical ReZero + FullGrid32 tactile head
2. full per-frame spatial Transformer
3. full-width 6144-D bidirectional temporal Transformer
4. independent binary contact head with the tactile decoder-tail topology
```

Record `initial_tactile_head_sha256`, `initial_spatial_module_sha256`,
`initial_temporal_module_sha256`, and `initial_contact_head_sha256` immediately
after construction. It may load only the released frozen DINOv3 weights. The
crop1.2 checkpoint is an evaluation reference, not an initialization source.
The tactile head, contact head, ReZero fusion, FullGrid projection, and both
Transformer stages are jointly trainable. Only an exact same-experiment resume
may restore these tensors from a checkpoint.

`onlinehmr_patch_kv_v6` is a parameter-efficient causal comparison under the
same fresh-base contract. Construction order is canonical ReZero + FullGrid32,
then the four-layer patch self/cross-attention stack, then an independently
reset contact decoder tail. It loads no crop1.2 or V5 tactile weights. DINO is
the only frozen pretrained component; ReZero, FullGrid projection, patch-KV
stack, tactile head, and contact head are jointly trainable. Its compact and
resume formats are distinct from V5, so neither experiment can accidentally
resume from the other.
