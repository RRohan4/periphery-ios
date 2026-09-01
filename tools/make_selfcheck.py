#!/usr/bin/env python3
"""Emit golden vectors so the Swift port can be checked against the Python.

The two .mlpackages are already numerically gated against ONNX. What is NOT
gated is everything hand-ported into Swift: the anchor table, the projection
LUT, the box decode, the direction fold, the grid-to-vehicle transform and the
circular NMS. Those are exactly the places a sign error hides, so this script
dumps the reference answer for a fixed calibration and a fixed set of head
tensors, and Sources/Periphery/SelfCheck.swift recomputes them on the phone.

Run it from the periphery repo:

    .venv/bin/python ../periphery-ios/tools/make_selfcheck.py \
        --out ../periphery-ios/Resources

Nothing here touches the checkpoint: the head tensors are seeded noise. The
point is agreement between two implementations of the same arithmetic, not
realism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from periphery.perception.fastbev import cityscapes_projection, build_lut
from periphery.perception.portable_postprocess import (
    _decode_vehicle, circular_nms_torch)
from periphery.sources.comma2k19 import (
    CAM_FORWARD_OF_ORIGIN, CAM_HEIGHT, COMMA_FRAME_SIZE, COMMA_K,
    focal_matched_crop, sensor_T_vehicle)
from periphery.training.contract import (
    decode_boxes, grid_from_config, make_anchors, read_config)

CONFIG = "configs/fastbev_cityscapes/D0_safety40_2h_512x256.json"
TRAINED_FOCAL = 565.6
# A real measured mount: the median of the seven comma drives, nose-down.
PITCH_DEG = -3.659
SEED = 20260831


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(CONFIG))
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--nms-radius", type=float, default=2.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = grid_from_config(read_config(args.config))
    anchors = make_anchors(grid)
    points = grid.points_np()

    # --- geometry --------------------------------------------------------
    pitch = float(np.deg2rad(PITCH_DEG))
    (x0, y0, crop_w, crop_h), cropped_K = focal_matched_crop(
        TRAINED_FOCAL, grid.input_width, grid.input_height)
    extrinsic = sensor_T_vehicle(pitch, CAM_HEIGHT, CAM_FORWARD_OF_ORIGIN)
    projection = cityscapes_projection(
        cropped_K, extrinsic, crop_w, crop_h,
        input_width=grid.input_width, input_height=grid.input_height)
    feature_h = grid.input_height // 4
    feature_w = grid.input_width // 4
    visible, indices = build_lut(projection, feature_h, feature_w, points=points)

    # --- head tensors ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    count = anchors.shape[0]
    # Logits centred well below zero so only a plausible handful of candidates
    # survive the 0.50 threshold, with a few forced winners.
    classes = rng.normal(-4.0, 1.5, size=(count, 4)).astype(np.float32)
    winners = rng.choice(count, size=64, replace=False)
    classes[winners, rng.integers(0, 4, size=winners.size)] += 6.0
    boxes = rng.normal(0.0, 0.3, size=(count, 9)).astype(np.float32)
    directions = rng.normal(0.0, 1.0, size=(count, 2)).astype(np.float32)

    # --- reference decode, exactly the eval path -------------------------
    class_t = torch.from_numpy(classes).sigmoid()
    code_t = torch.from_numpy(boxes)
    direction_t = torch.from_numpy(directions).argmax(dim=-1)
    decoded = decode_boxes(code_t, anchors)
    scores, labels = class_t.max(dim=1)
    vehicle = _decode_vehicle(decoded, direction_t)
    keep = (
        (scores >= args.threshold)
        & torch.isin(labels, torch.as_tensor((0, 1, 2)))
        & torch.isfinite(scores)
        & torch.isfinite(vehicle[:, :7]).all(dim=1)
        & (vehicle[:, 3:6] > 0.0).all(dim=1)
        & (vehicle[:, 0] >= 0.0) & (vehicle[:, 0] <= 40.0)
        & (vehicle[:, 1] >= -10.0) & (vehicle[:, 1] <= 10.0)
    )
    valid = torch.nonzero(keep, as_tuple=False).flatten()
    nms_keep = circular_nms_torch(vehicle[valid, :2], scores[valid],
                                 radius=args.nms_radius)
    final = valid[nms_keep]
    order = torch.argsort(scores[final], descending=True, stable=True)
    final = final[order]

    detections = [
        {
            "score": round(float(scores[i]), 6),
            "label": int(labels[i]),
            "x": round(float(vehicle[i, 0]), 5),
            "y": round(float(vehicle[i, 1]), 5),
            "z": round(float(vehicle[i, 2]), 5),
            "length": round(float(vehicle[i, 3]), 5),
            "width": round(float(vehicle[i, 4]), 5),
            "height": round(float(vehicle[i, 5]), 5),
            "yaw": round(float(vehicle[i, 6]), 5),
        }
        for i in final.tolist()
    ]

    # --- write -----------------------------------------------------------
    (args.out / "selfcheck_anchors.bin").write_bytes(
        anchors.numpy().astype(np.float32).tobytes())
    (args.out / "selfcheck_points.bin").write_bytes(
        points.astype(np.float32).tobytes())
    (args.out / "selfcheck_lut.bin").write_bytes(
        indices.astype(np.int32).tobytes() + visible.astype(np.uint8).tobytes())
    (args.out / "selfcheck_head.bin").write_bytes(
        classes.tobytes() + boxes.tobytes() + directions.tobytes())

    manifest = {
        "generated_by": "periphery-ios/tools/make_selfcheck.py",
        "config": str(args.config),
        "seed": SEED,
        "calibration": {
            "pitch_deg": PITCH_DEG,
            "camera_height_m": CAM_HEIGHT,
            "camera_forward_of_origin_m": CAM_FORWARD_OF_ORIGIN,
            "frame_size": list(COMMA_FRAME_SIZE),
            "K": [[float(v) for v in row] for row in COMMA_K],
            "trained_focal_px": TRAINED_FOCAL,
        },
        "crop": {"x": int(x0), "y": int(y0), "width": int(crop_w), "height": int(crop_h)},
        "projection": [[float(v) for v in row] for row in projection],
        "counts": {
            "anchors": int(count),
            "voxels": int(points.shape[0]),
            "visible_voxels": int(visible.sum()),
            "visible_fraction": round(float(visible.mean()), 6),
            "detections": len(detections),
        },
        "decode": {
            "threshold": args.threshold,
            "nms_radius": args.nms_radius,
            "labels": [0, 1, 2],
        },
        "detections": detections,
    }
    (args.out / "selfcheck.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))
    print(f"wrote 5 files to {args.out}")


if __name__ == "__main__":
    main()
