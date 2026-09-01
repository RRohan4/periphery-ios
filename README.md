# periphery-ios

iOS deployment of **Safety40**, a monocular BEV vehicle detector that runs on a
windshield-mounted phone.

This repository is deliberately small. The training code, datasets and
evaluation harness live in the `periphery` repo and have no business on a
rented Mac. What is here is the compiled model, the contract the Swift side has
to honour, and nothing else.

## What is in here

```
models/backbone_static.mlpackage    23 MB, 15.21 M params
models/head_static.mlpackage        7.1 MB,  3.67 M params
models/coreml_export_report.json    conversion + ONNX numeric agreement
docs/decode-contract.md             the exact handoff the Swift port must match
Periphery/                          the Xcode project (app target: Periphery)
Periphery/Periphery/Periphery/      the Swift port of everything CoreML does not do
  .../Resources/                    golden vectors the port checks itself against
tools/make_selfcheck.py             regenerates those goldens from the periphery repo
```

## The shape of the thing

The model is split in two, and the split is the whole reason this is portable:

```
image  [1,3,256,512]
   |  backbone_static.mlpackage
features [1,64,64,128]
   |  native gather through a calibration LUT   <-- Swift, not CoreML
volume [1,64,41,80,2]
   |  head_static.mlpackage
classes / boxes / directions
   |  decode + threshold + circular NMS         <-- Swift, not CoreML
detections in vehicle coordinates
```

The geometry step in the middle is an index gather, not learned. It never
enters the converted graph, which is why both halves are plain convolution
stacks that convert without complaint. It is also cheap: measured at 0.3-0.5 ms
on desktop, because it is a memory copy.

Decode stays in Swift for a measured reason, not a stylistic one. An export
with decode, argmax and top-k baked in (`head_full`) disagreed with PyTorch by
**24.7 max abs** — top-k tie-breaking is not portable across runtimes. The two
shipped halves agree to 1.7e-4 and 1.2e-4.

## Status

Measured on an **iPhone 16 (A18)**, release build, sideloaded.

| | |
|---|---|
| CoreML conversion | passes, both halves, iOS 17 target |
| ONNX numeric agreement | 1.7e-4 backbone, 1.2e-4 head |
| Swift port vs Python | 6/6 golden checks pass on device |
| ANE residency | every operation planned on the Neural Engine, both halves |
| On-device latency | **3.9 ms median**, p95 7.9 ms, worst 9.0 ms — 230 fps |
| Sustained / thermal FPS | **unmeasured** — 10 min run pending |

### Latency, 200 frames after 10 warm-up passes

| | backbone | gather | head | decode | total |
|---|---|---|---|---|---|
| iPhone 16 (A18) | 1.2 ms | 0.2 ms | 1.8 ms | 0.6 ms | **3.9 ms** |
| desktop CUDA | 5.1 ms | 0.5 ms | 2.0 ms | — | 7.6 ms |
| desktop CPU | 33.7 ms | 0.3 ms | 24.8 ms | — | 58.8 ms |

The phone beats the desktop GPU by 2x and the CPU floor by 15x. Two details
worth keeping attached to that number:

* **The gather is 0.2 ms on the phone against 0.3 ms on a desktop CPU.** It is a
  memory copy and it does not care what silicon it runs on. That is the
  strongest evidence that splitting the model at the geometry op cost nothing.
* **Core ML runs this in float16.** `image float16, volume float16` -- the ANE's
  native precision, chosen by Core ML, not requested. Every tensor crossing the
  boundary is converted through a float32 scratch buffer so the decode
  arithmetic stays where the goldens were computed. On-device detections are
  therefore not bit-identical to the Python; the export was gated against ONNX
  at 1.2e-4, which is float16-scale error, but the end-to-end difference on a
  real frame has not been quantified and would need the camera path first.

Preprocessing is excluded: it is camera plumbing, and folding it in would hide
the model cost behind vImage.

The export script's op-count extraction returned empty, so the original ANE
claim was inspection rather than measurement. It is now measured, twice over.
Xcode's model inspector gives the inventory -- 65 non-const operations in the
backbone (29 conv, 17 relu, 11 add, 3 nearest + 3 bilinear upsample, 1 concat,
1 max_pool) and 39 in the head (16 conv, 9 relu, 6 reshape, 4 transpose, 3 add,
1 stack) -- and the app's Compute tab runs `MLComputePlan` on device, which
reports **every one of them planned for the Neural Engine**.

State it as *planned* placement, not executed work: `MLComputePlan` reports what
Core ML intends before anything runs. Executed per-layer timings need
Instruments' Core ML template, which requires a USB-tethered device and is
therefore unavailable when the Mac is rented. The 3.9 ms measured end to end is
consistent with the plan being honoured.

## Latency budget

Desktop reference, batch 1:

| | backbone | gather | head | total |
|---|---|---|---|---|
| CUDA | 5.1 ms | 0.5 ms | 2.0 ms | **7.6 ms** |
| CPU | 33.7 ms | 0.3 ms | 24.8 ms | **58.8 ms** |

The CPU column is the conservative floor for the phone. If the ANE engages,
expect substantially better; if the measured number lands near 58.8 ms,
something fell off the ANE and `MLComputePlan` will say what.

## The Swift side

All of it lives in `Periphery/Periphery/Periphery/`, the group Xcode created
when the folder was added to the target:

```
Contract.swift       grid, anchors, frames, operating point
Calibration.swift    mount pose, focal matching, projection
ProjectionLUT.swift  the gather between the two models
Decode.swift         sigmoid, anchor decode, direction fold, circular NMS
Preprocess.swift     camera frame -> [1,3,256,512], vImage
Detector.swift       backbone -> gather -> head -> decode, with timings
SelfCheck.swift      runs the goldens against all of the above
Resources/           the goldens themselves
```

`Periphery/Periphery/*.mlpackage` are the copies Xcode compiles into the app.
`models/` stays the canonical export artefact -- the pair that
`coreml_export_report.json` describes. Re-export replaces `models/`, then
those get copied across.

Models are loaded by resource name and the head's three outputs are identified
by their trailing dimension (4 classes, 9 box codes, 2 direction logits), not by
the auto-generated coremltools output names, so a re-export cannot silently
rename a call site.

### Self-check

The two `.mlpackage`s are gated against ONNX at export time. The Swift half is
gated by `SelfCheck.run()`, which recomputes the anchor table, the voxel
centres, the focal-matched crop, the projection matrix, the LUT and a full
decode of 6720 fixed candidates, and diffs each against `Resources/`. Against
the Python reference the port currently reproduces:

| check | result |
|---|---|
| voxel centres | 6560, exact |
| anchors | 6720 rows, max diff 3.8e-6 (float32 rounding) |
| focal-matched crop | 824x412 at (170, 231), focal 565.6 |
| projection matrix | exact |
| projection LUT | 0 visibility and 0 index mismatches, 66.97% visible |
| decode + NMS | 63 detections, every field within 2e-4 |

That comparison has been run host-side against the Python; running it on the
phone is what confirms the arithmetic survives the trip. Regenerate the goldens
after any change to the geometry, from the `periphery` repo root:

```
PYTHONPATH=. .venv/bin/python ../periphery-ios/tools/make_selfcheck.py \
    --out ../periphery-ios/Periphery/Periphery/Periphery/Resources
```

## Building

Requires Xcode with an iOS 17 SDK or newer (Xcode 15+; Xcode 26 is the last
version that runs on Intel Macs). Drag both `.mlpackage` directories into the
Xcode project and let it generate the Swift interfaces. The Swift sources and
`Resources` are already members of the target in the committed project, so a
`git pull` is enough; only a *new* file needs File -> Add Files. `Resources`
must sit in *Copy Bundle Resources* or the self-check reports the goldens
missing.

The app launches straight into the self-check screen (`ContentView.swift`) --
six rows, all expected green, no camera and no model load involved.

Video stabilisation has to be off on the capture session. EIS and OIS change
per-frame geometry unreported, which breaks both the fixed intrinsics the crop
is computed from and the known extrinsics the LUT is built from --
`cameraIntrinsicMatrixDeliveryEnabled` is unavailable while it is on anyway.

## Where the truth lives

Constants are **not** transcribed into this README, on purpose — a copied
constant that drifts is worse than no constant. The Swift port must be written
against these files in the `periphery` repo:

| what | file |
|---|---|
| image normalisation | `periphery/perception/fastbev.py` (MEAN, STD) |
| projection LUT | `periphery/training/model.py` (`_projection_lut`, `backproject`) |
| decode + NMS | `periphery/perception/portable_postprocess.py` |
| mount calibration | `periphery/sources/comma2k19.py` (`mount_pitch`, `sensor_T_vehicle`) |
| architecture contract | `configs/fastbev_cityscapes/D0_safety40_2h_512x256.json` |

`docs/decode-contract.md` records the handoff precisely.

## Checkpoint

`safety40-locked-v1` — `out/safety40/D0_full/epoch_11.pt`. Holdout F1 0.662 at
score threshold 0.50, 2.0 m circular NMS, 555 frames. Do not re-export from a
different checkpoint without updating this line.
