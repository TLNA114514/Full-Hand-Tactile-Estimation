# Historical TouchAnything Crop1.2 Replay

## What Changed

The July 24 crop1.2 baseline and current C32/H512 training do not share the
same stochastic trajectory, even when their visible hyperparameters match.
Three independent changes are responsible:

1. The historical head constructed `decoder` before `base_projection`; the
   current default constructs them in the opposite order. The parameter names
   and shapes are unchanged, but the same seed is assigned to different
   tensors.
2. Historical Lightning 2.1.4 initialized every DataLoader worker with
   `pl_worker_init_function`. The current parent-death initializer suppressed
   that callback, so bbox augmentation used a different deterministic NumPy
   stream.
3. The legacy index sorted by `(sample_dir, hand)`. Sequence-HDF5 normally
   follows manifest order, so the same sampler permutation addresses different
   samples.

The model-construction boundary is commit `11696f01` (after the historical
`92de6b73` implementation). Both heads contain 17,291,666 parameters, but an
isolated seed-521 audit gives different ordered initial-state fingerprints:

```text
historical decoder-first: 1fc07a46eff6c8ce40312c8cfea8eac5cba48bb5e2fa336287d5085a29dc8235
current projection-first: 34e65c289e225952ddffec7fff2e38ac1b1941a8880c28ad381eec6fdabeb4b3
```

This is why an interrupted current C32/H512 run resumes consistently with
itself but does not reproduce the July curve: resume preserves the current
trajectory; it does not turn it back into the historical trajectory.

The direct rectangular crop is mathematically and pixel-wise equivalent to the
historical square crop in the audited cases. The replay profile nevertheless
recreates `square256 -> flip/normalize -> center crop` to remove that remaining
implementation difference.

## Experiment Inventory

### Historical complete pipeline

The following 51 training identities use decoder-first initialization,
Lightning worker seeding, and legacy sample ordering:

- `touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop{10,12,14,16,18,20}`
- `opentouch_dense_v2_dinov3_rezero_strictcontrol`
- `touchanything_dense_v2_dinov3_rezero_strictcontrol`
- all eight experiments under `reports/memorization_1`
- all sixteen strict-fit experiments under `reports/memorize`
- the following pre-crop mixed controls:

```text
mixed_v3.2_woego
mixed_contact_volume_gated_v15
mixed_contact_volume_gated_v15b_auxvol
mixed_dense_v2_repro
mixed_dense_v2_multilevel
mixed_dense_v2_multilevel_concat
mixed_dense_v2_dinov3_hplus
mixed_dense_v2_dinov3_taill1
mixed_dense_v2_dinov3_progressive
mixed_dense_v2_dinov3_dpt_rezero
mixed_dense_v2_dinov3_rezero_baseanchor
mixed_dense_v2_dinov3_rezero_finalcontrol
mixed_dense_v2_dinov3_rezero_fullgrid32
mixed_dense_v2_dinov3_rezero_fullgrid32_coreloc
mixed_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3expanded
mixed_dense_v2_dinov3_rezero_multilevel
mixed_dense_v2_dinov3_rezero_seqsqrt_stratified
mixed_dense_v2_dinov3_rezero_strictcontrol
mixed_dense_v2_dinov3_rezero_targetcrop18
```

The formal historical baseline is:

```text
touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12
```

### New head/worker, legacy directory data

These six experiments already use projection-first initialization and the
custom worker callback, although they still read legacy directories:

```text
touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12_res320
touchanything_dense_v2_dinov3_rezero_fullgrid32_coreloc_sam3_crop12_res384
ta_xattn_spatial_r256
ta_xattn_global_r256
ta_wplateau2_r256
ta_wlinear3_r256
```

### New head/worker and sequence-HDF5 data

The following 18 training identities additionally use manifest-order HDF5:

```text
ta_xattn_direct_sp_r256
ta_xattn_direct_glb_r256
ta_xattn_glocal_r256
ta_xattn_deform_fused_r256
ta_cse_sp_r256
ta_cse_glb_r256
ta_depthpn_real
ta_depthpn_shuffle
ta_dlocal_sp_r256
ta_dlocal_glb_r256
ta_localres_r256
ta_selector_contact_r256
ta_selector_ordinal_r256
ta_selector_grid_r256
ta_selector_raw_r256
ta_selector_down_r256
ta_selector_down_ctl_r256
ta_surface_nl_k4096_r256
```

Shuffle/calibration reports are evaluation controls, not extra trained models.
Later baseline-trainfit reports that load the July crop1.2 checkpoint inherit
the historical base weights despite their newer report timestamp.

Frozen-base LocalResidual, selector, Depth/VLM prior, and related
`tactile_input_priors` runs are **hybrid** experiments: their loaded crop1.2
base prediction is historical, while their newly trained adapter/selector sees
the newer worker and HDF5 trajectory. They must not be counted as fresh
baseline reproductions, but the initialization-order change does not rewrite
their frozen base weights.

The eight `ta_fg_c{32,64,128,256}_h{512,1024}` presets have no synced standard
report in this workspace. With the ordinary presets they use the current
projection-first/worker/HDF5 trajectory. Therefore C32/H512 is a valid control
for that eight-way matrix, but is not a strict replay of the July baseline.

## Strict Reconstruction Contract

Use:

```bash
AUTO_RESUME=0 ./hamer_tactile_ft/run_tactile_experiment.sh \
  replay-ta-crop12-20260724 \
  --dino_weights /home/ma-user/work/cfzhao/Full-Hand-Tactile-Estimation/_DATA/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth \
  --gpus 0,1,2,3,4,5,6,7
```

The preset writes to the independent experiment name:

```text
ta_crop12_legacy_replay_20260724
```

It fails before training if any locked field differs, including model shape,
crop/loss configuration, eight-GPU batch topology, worker count, runtime
PyTorch/Lightning versions, DINO SHA, bbox SHA, query-manifest SHA, or sample
count. The historical AdamW constructor path is also restored instead of
forcing the newer explicit foreach/fused choice. It records train/val
ordered-sample SHA256 values and the initial tactile-head SHA256 in every
provenance/checkpoint artifact.

The original millions of `meta.json` sample directories are no longer a safe
runtime source. The replay therefore uses their audited sequence-HDF5 archive,
reconstructs the old `(source_sample_relpath, hand)` order, and requires the
historically verified sample counts and source fingerprints. This is the
strictest reproducible reconstruction available from the retained artifacts.

Use the archived validation trajectory as an early replay check:

| Epoch | Global step | Train loss | Val loss | Val RMSE | Val V-IoU | Core V-IoU |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,579 | 0.330526 | 0.111095 | 0.057345 | 0.328277 | 0.300956 |
| 16 (loss-best in retained log) | 43,843 | 0.082239 | 0.095841 | 0.047145 | 0.461246 | 0.435934 |
| 40 (last retained epoch) | 105,739 | 0.078735 | 0.096902 | 0.047961 | 0.463891 | 0.440259 |

The historical command requested 60 epochs, while the retained log ends at
epoch 40. The replay keeps the requested 60-epoch contract; compare epoch 0
first and use the retained epoch-16 checkpoint as the primary trajectory
landmark. CUDA kernels were not run in deterministic mode historically, so
exact bit identity is not promised even when every retained input is locked.

Do not reuse `resume.ckpt` from another mode. Cross-mode resume is rejected by
the resume contract because optimizer state is positional. A fresh replay must
start with `AUTO_RESUME=0` and the independent output directory above.

## Pressure-Weight Ablations

Two TouchAnything crop1.2 controls isolate only the pointwise pressure weight:

| Preset | GT pressure `<0.10` | GT pressure `>=0.10` |
|---|---:|---:|
| `weight-flat1-r256` | 1.0 | 1.0 |
| `weight-contact2-r256` | 1.0 | 2.0 after the five-epoch ramp |

The contact boundary is the TouchAnything Contact-IoU threshold (`0.10`). The
second preset uses the historical hump's total peak weight:

```text
1.0 base weight + active_pressure_weight 1.0 = 2.0
```

These modes do not remove the baseline background penalty, logit BCE term, or
CoreLoc objective. They change only `pressure_weight_like`, so the comparison
remains a pressure-weight ablation rather than a different loss stack.
