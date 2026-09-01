# The decode contract

Everything the Swift side has to get exactly right. Each item names the Python
file that is authoritative; where the two disagree, Python wins and this
document is the bug.

## 1. Input

| | |
|---|---|
| tensor | `[1, 3, 256, 512]`, float32, NCHW, RGB |
| normalisation | `(pixel - MEAN) / STD`, per channel, on 0-255 values |
| MEAN | `[123.675, 116.28, 103.53]` |
| STD | `[58.395, 57.12, 57.375]` |

Source: `periphery/perception/fastbev.py`.

Note the scale: MEAN and STD are in **0-255 units**, not 0-1. Divide by 255
first and every activation is wrong by two orders of magnitude.

Letterboxing matters. The reference resizes preserving aspect ratio onto a
canvas pre-filled with MEAN, so the padding is exactly zero after
normalisation. Padding with black instead injects a strong negative signal at
the edges.

## 2. Focal matching

The backbone never sees the intrinsic matrix. It only sees pixels, and apparent
object scale is `f * W / d`. So shrinking the focal length by `s` is
indistinguishable from moving every object `1/s` times farther away.

The model was trained at focal **565.6 px** for a 512-wide input. Feeding a
different camera at its native crop silently rescales every distance. On comma
data, a naive resize to 512 gives focal 400 — every vehicle reads 1.41x too
far.

The fix is a centre crop chosen so the crop-then-resize lands on the training
focal, then resized to 512x256. See `focal_matched_crop` in
`periphery/sources/comma2k19.py`. For the iPhone this must be computed from the
live intrinsics, which means **video stabilisation has to be off** — with EIS
or OIS active the per-frame geometry changes unreported and
`cameraIntrinsicMatrixDeliveryEnabled` is unavailable anyway.

## 3. The projection LUT

Between the two models. Given a 3x4 projection matrix, for every voxel centre:

1. project to feature-plane coordinates, `u = round(x/z)`, `v = round(y/z)`
2. mark visible if `z > 0` and `u, v` are in bounds of the 128x64 feature map
3. index is `v * feature_width + u`, clamped to bounds
4. gather that feature column; multiply by the visibility mask (invisible
   voxels become zero, they are not skipped)

Output volume `[1, 64, 41, 80, 2]`.

Source: `periphery/training/model.py`, `_projection_lut` and `backproject`.

The LUT depends only on calibration, not on the image. Recompute it when the
mount pose estimate updates, not per frame.

## 4. Calibration

The projection matrix comes from the mount pose. Two angles, three timescales:

| timescale | what moves | source |
|---|---|---|
| ~33 ms | rattle | gyro (CoreMotion), drift-free over one frame interval |
| seconds to minutes | road grade, suspension, mount sag | rolling median of direction-of-travel pitch |
| cold start | initial pose, roll | tape measure and gravity |

Pitch is estimated against the **direction of travel**, not gravity: rotate the
frame-to-frame displacement into device axes and take
`pitch = -atan2(-down, forward)`, median over samples above 5 m/s. Referenced
this way, road grade cancels identically. Measured on comma data across four
segments: travel-referenced pitch sd 0.24-0.45 deg while grade ranged over
6-8.7 deg, and gravity-referenced pitch tracked the grade almost perfectly
(slope ~1.0). Seven independent drives agree on mount pitch within 0.43 deg.

Source: `periphery/sources/comma2k19.py`, `mount_pitch` and `sensor_T_vehicle`.

**The sign convention in `sensor_T_vehicle` is the single most dangerous line
in the pipeline.** Its own docstring says so. Port it literally.

### Why this deserves care

Sensitivity is wildly asymmetric. With `d = h / tan(theta)`:

- `d(d)/d(h) = d/h` — a 1 cm camera-height error is a pure scale factor,
  33 cm at 40 m. Forgiving.
- `d(d)/d(theta) = h / sin^2(theta)` ~ 23 m per degree — 0.1 deg is 2.3 m
  at 40 m. Not forgiving.

Measure height casually. Do not guess pitch.

### Tolerance budget

Per-frame zero-mean pitch jitter, 555 holdout frames, threshold 0.50:

| sigma | F1 | vs baseline |
|---|---|---|
| 0 deg | 0.662 | — |
| 0.1 deg | 0.653 | -1% |
| 0.25 deg | 0.643 | -3% |
| 0.5 deg | 0.602 | -9% |
| 1.0 deg | 0.515 | -22% |
| 2.0 deg | 0.399 | -40% |

A constant bias of the same magnitude costs roughly twice as much (-32% at
1 deg), because it moves every frame the same way instead of averaging out.
comma's rigid mount measures 0.24 deg rms including road noise, so a decent
clamp at 2-3x that is inside budget. Prefer a short arm over an expensive one:
cantilever natural frequency falls roughly as length^1.5, straight into the
1-10 Hz road-input band.

## 5. Head outputs and decode

Head returns three tensors: `classes`, `boxes`, `directions`.

Order of operations, and it is not negotiable:

1. sigmoid on `classes`
2. threshold at **0.50** (decode threshold 0.05 if you want the full curve)
3. anchor decode to grid boxes
4. direction decode: fold rotation by `pi` with `DIR_OFFSET = 0.7854`, then add
   `pi * direction`
5. grid to vehicle via the `GRID_TO_VEHICLE` affine
6. circular NMS, radius **2.0 m**, greedy in descending score, strict
   `< radius` suppression

Source: `periphery/perception/portable_postprocess.py`.

Do not substitute CoreML's built-in NMS without checking it against the
reference first — it is IoU-based and this is a distance rule on BEV centres,
which is a different suppression policy, not a faster version of the same one.

The constants in `portable_postprocess.py` were written for the earlier
nuScenes frontend. **Re-read `GRID_TO_VEHICLE` against the Safety40 config
before porting it** — the z offset in particular encodes a sensor height that
is not this camera's.

## 6. Evaluation region

Forward 0-40 m, lateral -10 to +10 m. Two height slices at metric z -0.66 m and
0.84 m. Anything outside is not a detection, it is a decode artefact.

Source: `configs/fastbev_cityscapes/D0_safety40_2h_512x256.json`.

## 7. What is deliberately not here

- **Tracking.** No temporal state. Each frame is independent. A tracker exists
  on the `frontend-and-lanes` branch of the `periphery` repo but its noise
  model was fitted to a different detector and would have to be refitted.
- **Occlusion filtering.** Geometrically justified, implemented on that same
  branch, never measured. See `notes/34`.
- **Lanes.** Different problem.

Ship the per-frame detector first. Everything above is an increment on a thing
that works, and none of it is needed to answer the question this port exists to
answer: what frame rate does this hold on a phone, and for how long before it
throttles.
