#!/usr/bin/env python3
"""Generate the compact iOS tracker parity fixture from the flagged drive."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


IOS = Path(__file__).resolve().parents[1]
PYTHON_REPO = IOS.parent / "periphery"
sys.path.insert(0, str(PYTHON_REPO))

from periphery.perception.associate import Tracker, TrackerConfig  # noqa: E402
from periphery.perception.heading_state import track_heading  # noqa: E402
from periphery.sources.comma2k19 import load_segment  # noqa: E402
from periphery.types import Detection, EgoDelta  # noqa: E402


LABELS = ["light_vehicle", "large_vehicle", "two_wheeler", "pedestrian"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detections",
        default=PYTHON_REPO / "data/comma2k19/detections/"
        "safety40_drive_2026-09-04_11-47-39.json",
        type=Path,
    )
    parser.add_argument(
        "--segment",
        default=PYTHON_REPO / "data/iphone/drive_2026-09-04_11-47-39",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default=IOS / "Periphery/Periphery/Periphery/Resources/tracker_selfcheck.json",
        type=Path,
    )
    args = parser.parse_args()

    payload = json.loads(args.detections.read_text())
    segment = load_segment(str(args.segment))
    times = payload["frame_times"]
    cfg = TrackerConfig(
        confirm_hits=3,
        hold_s=0.5,
        coast_min_hits=8,
        state_observations=5,
        prediction_observations=4,
        parked_below=1.0,
        parked_confirm_frames=2,
        output_velocity_alpha=0.35,
        coast_freeze_s=0.2,
        trail_s=1.5,
        gate_floor=2.0,
        gate_speed=12.0,
        gate_sigmas=4.0,
        state_max_range_m=40.0,
        overlap_shrink=1.0,
        range_aware_gate=True,
    )
    tracker = Tracker(cfg)
    median_dt = sorted(b - a for a, b in zip(times, times[1:]))[len(times) // 2 - 1]
    frames = []
    watched = {41, 43, 44, 67, 68}

    for index, t in enumerate(times):
        rows = payload["detections"].get(str(index), [])
        detections = [Detection(**{key: row[key] for key in
                      ("x", "y", "z", "yaw", "length", "width", "height", "label", "score")})
                      for row in rows]
        ego = segment.ego_delta(index - 1) if index > 0 else EgoDelta(dt=median_dt)
        before_detection = tracker.detection_suppressed
        before_track = tracker.track_suppressed
        confirmed = tracker.step(detections, ego, float(t))
        suppression_changed = (tracker.detection_suppressed != before_detection
                               or tracker.track_suppressed != before_track)
        checkpoint = index % 25 == 0 or suppression_changed or any(
            track.id in watched for track in confirmed)

        expected = None
        if checkpoint:
            expected = []
            for track in confirmed:
                velocity = tracker.output_velocity(track)
                heading, source = track_heading(track.yaw, velocity)
                expected.append([
                    track.id, track.x, track.y, track.z,
                    track.length, track.width, track.height,
                    track.yaw, heading, source, LABELS.index(track.label), track.score,
                    track.hits, track.observed, track.state,
                    None if velocity is None else [velocity[0], velocity[1]],
                ])

        packed_detections = [[
            row["score"], LABELS.index(row["label"]), row["x"], row["y"],
            row.get("z", 0.0), row.get("length", 4.5), row.get("width", 1.9),
            row.get("height", 1.5), row.get("yaw", 0.0),
        ] for row in rows]
        frames.append([
            t, ego.dx, ego.dy, ego.dyaw, ego.dt, packed_detections,
            tracker.detection_suppressed, tracker.track_suppressed,
            tracker._next_id, expected,
        ])

    document = {
        "schema": 1,
        "source": str(args.detections.relative_to(PYTHON_REPO)),
        "frames": frames,
        "final": {
            "births": tracker.births,
            "matches": tracker.matches,
            "detection_suppressed": tracker.detection_suppressed,
            "track_suppressed": tracker.track_suppressed,
            "ids_issued": tracker._next_id,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, separators=(",", ":")))
    print(f"{len(frames)} frames, {sum(f[9] is not None for f in frames)} checkpoints, "
          f"{args.out.stat().st_size / 1024:.1f} KiB -> {args.out}")


if __name__ == "__main__":
    main()
