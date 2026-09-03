# Dyn-HaMR TouchAnything Trial

## Status

This report covers one bilateral 120-frame sequence:

```text
train/Home/arrange_pillow/20260412_101243_753.h5
source frame rows 1001:1121
```

The official Dyn-HaMR source is pinned to commit
`fa9cd7412c205fd15ee4139c8caacf79bf6167e6`. The trial uses HaMeR MANO output
as initialization, identity camera extrinsics, HaMeR-derived focal length, and
no HMP prior. It therefore tests the temporal optimizer and integration, not
the full VIPE camera-aware Dyn-HaMR system.

## Integration Validation

The corrected `v3` conversion reconstructs the HaMeR input with:

| Metric | Initialization |
| --- | ---: |
| 2D reprojection RMSE | 1.683 px |
| 2D reprojection median | 1.022 px |
| Left / right RMSE | 1.716 / 1.650 px |
| 3D joint RMSE | 2.633 mm |
| Root-relative joint RMSE | 2.370 mm |
| Positive-depth fraction | 100% |

This confirms that handedness, MANO pose, translation, intrinsics, and joint
ordering are mutually consistent. In particular, canonical left-hand geometry
is mirrored before adding the common camera/world translation.

## Quick Optimization

The diagnostic run used 5 root iterations and 20 smooth iterations. This is
long enough to expose the direction of the objective, but is not the official
50 + 300 iteration schedule.

| Metric | HaMeR-compatible init | Dyn-HaMR 5 + 20 |
| --- | ---: | ---: |
| 2D reprojection RMSE | 1.683 px | 1.952 px |
| 2D reprojection median | 1.022 px | 0.937 px |
| 2D reprojection p95 | 3.403 px | 3.543 px |
| Positive-depth fraction | 100% | 100% |
| Root-relative joint RMSE vs HaMeR | 2.370 mm | 9.105 mm |
| Median absolute wrist-depth shift | 0.443 mm | 190.295 mm |
| p95 absolute wrist-depth shift | 0.823 mm | 956.596 mm |
| Median relative depth shift | 0.003% | 1.425% |
| Median MANO beta L2 shift | ~0 | 0.932 |

Total 3D velocity, acceleration, and jerk ratios fall to `0.471`, `0.373`, and
`0.350`. That apparent improvement is almost entirely root-trajectory
smoothing: root ratios are `0.471`, `0.373`, and `0.349`, while root-relative
articulation ratios are `1.035`, `1.043`, and `1.046`. In other words, finger
articulation becomes slightly less smooth while weakly constrained absolute
depth absorbs the optimization.

## Decision

The runtime integration is operational, bilateral geometry is correctly
converted, and source-view overlays remain visually plausible. However, this
configuration does not yet demonstrate a better MANO prior for tactile
localization. It should not be exported over all TouchAnything records or used
as image-to-vertex supervision in its current form.

The next meaningful Dyn-HaMR test needs both:

1. A real dynamic-camera trajectory (the VIPE path or an equivalent calibrated
   camera estimate), rather than identity extrinsics.
2. Independent image evidence, such as detected 2D joints, plus explicit
   initialization/depth and shape anchors. Reprojecting HaMeR's own joints only
   asks Dyn-HaMR to preserve HaMeR in 2D while leaving depth underconstrained.

Run artifacts are stored remotely at:

```text
/home/ma-user/work/cfzhao/hand_pose_dynhamr/trials/touchanything_arrange_pillow_v3/outputs/static_focal_quick
```

The pulled audit is under `outputs/dynhamr_touchanything_v3/quick`.
