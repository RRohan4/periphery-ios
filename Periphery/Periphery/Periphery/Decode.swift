//  Decode.swift
//  Head outputs -> vehicle-frame boxes.
//
//  Ports periphery/training/contract.py (decode_boxes),
//  periphery/perception/portable_postprocess.py (_grid_yaw, _decode_vehicle,
//  circular_nms_torch) and the reference decode in
//  scripts/eval_fastbev_training.py:_training_predictions, which is the code
//  path that produced the measured operating point (P 0.647 / R 0.678 /
//  F1 0.662 at threshold 0.50, circular NMS radius 2.0 m).
//
//  The order of operations is not negotiable.

import Foundation
import simd

/// One detection in vehicle coordinates: x forward, y left, z up, metres.
struct Detection {
    var score: Float
    var label: Int
    /// Box centre.
    var x: Double
    var y: Double
    var z: Double
    /// Extents along the box's own axes, metres.
    var length: Double
    var width: Double
    var height: Double
    /// Heading, radians, wrapped to [-pi, pi).
    var yaw: Double

    var range: Double { (x * x + y * y).squareRoot() }
    var className: String { Contract.classNames[label] }
}

enum Decode {

    /// Full decode of one frame's head outputs.
    ///
    /// `classes` is [6720, 4] LOGITS, `boxes` is [6720, 9] regression codes,
    /// `directions` is [6720, 2] logits. All contiguous, candidate-major.
    static func detections(classes: UnsafePointer<Float>,
                           boxes: UnsafePointer<Float>,
                           directions: UnsafePointer<Float>,
                           anchors: [Contract.Anchor],
                           scoreThreshold: Double = Contract.scoreThreshold,
                           nmsRadius: Double = Contract.nmsRadius) -> [Detection] {
        var candidates = [Detection]()
        let numClasses = Contract.classNames.count

        for i in 0..<anchors.count {
            // 1. sigmoid on the class logits, best class wins.
            var best = -Float.greatestFiniteMagnitude
            var label = 0
            for c in 0..<numClasses {
                let score = sigmoid(classes[i * numClasses + c])
                if score > best { best = score; label = c }
            }
            // 2. threshold, and drop the frozen pedestrian head.
            guard Double(best) >= scoreThreshold,
                  Contract.vehicleLabels.contains(label) else { continue }

            // 3. anchor decode to a grid box.
            let code = boxes + i * 9
            let anchor = anchors[i]
            let grid = decodeBox(code: code, anchor: anchor)

            // 4. direction decode: argmax of the two logits, then the fold.
            let direction = directions[i * 2 + 1] > directions[i * 2] ? 1.0 : 0.0

            // 5. grid to vehicle.
            let detection = toVehicle(grid: grid, direction: direction,
                                      score: best, label: label)

            guard detection.length > 0, detection.width > 0, detection.height > 0,
                  detection.x.isFinite, detection.y.isFinite, detection.z.isFinite,
                  detection.yaw.isFinite else { continue }
            // 6. evaluation region. Outside it, a box is a decode artefact.
            guard detection.x >= Contract.forwardRange.min,
                  detection.x <= Contract.forwardRange.max,
                  detection.y >= Contract.lateralRange.min,
                  detection.y <= Contract.lateralRange.max else { continue }

            candidates.append(detection)
        }

        return circularNMS(candidates, radius: nmsRadius)
    }

    // MARK: - Steps

    @inline(__always)
    static func sigmoid(_ value: Float) -> Float {
        1.0 / (1.0 + exp(-value))
    }

    /// A grid-frame box: [x, y, z_bottom, w, l, h, yaw].
    struct GridBox {
        var x, y, zBottom, w, l, h, yaw: Double
    }

    /// DeltaXYZWLHRBBoxCoder, matching contract.decode_boxes. Codes 7 and 8 are
    /// the unused velocity slots and are never read.
    static func decodeBox(code: UnsafePointer<Float>, anchor: Contract.Anchor) -> GridBox {
        let diagonal = anchor.diagonal
        let centerZ = anchor.z + anchor.h / 2.0
        let boxZ = centerZ + Double(code[2]) * anchor.h
        let h = exp(Double(code[5])) * anchor.h
        return GridBox(x: Double(code[0]) * diagonal + anchor.x,
                       y: Double(code[1]) * diagonal + anchor.y,
                       zBottom: boxZ - h / 2.0,
                       w: exp(Double(code[3])) * anchor.w,
                       l: exp(Double(code[4])) * anchor.l,
                       h: h,
                       yaw: Double(code[6]) + anchor.yaw)
    }

    /// mmdet3d's half-period fold, then the predicted half turn.
    static func gridYaw(rotation: Double, direction: Double) -> Double {
        let period = Double.pi
        var folded = rotation - Contract.dirOffset
        folded -= (folded / period).rounded(.down) * period
        return folded + Contract.dirOffset + period * direction
    }

    /// GRID_TO_VEHICLE applied to the box centre, plus the yaw convention flip.
    /// Note the deliberate swap: the grid's l becomes the vehicle box's length
    /// because grid y is vehicle forward.
    static func toVehicle(grid: GridBox, direction: Double,
                          score: Float, label: Int) -> Detection {
        let centerZ = grid.zBottom + grid.h / 2.0
        var yaw = -gridYaw(rotation: grid.yaw, direction: direction) - Double.pi
        yaw -= ((yaw + Double.pi) / (2.0 * Double.pi)).rounded(.down) * (2.0 * Double.pi)
        return Detection(score: score,
                         label: label,
                         x: grid.y + Contract.gridToVehicle[3][0],
                         y: -grid.x,
                         z: centerZ + Contract.gridToVehicle[3][2],
                         length: grid.l,
                         width: grid.w,
                         height: grid.h,
                         yaw: yaw)
    }

    /// Greedy circular NMS on BEV centres, descending score, strict `< radius`
    /// suppression. This is a distance rule, not an IoU rule: CoreML's built-in
    /// NMS is a different suppression policy, not a faster version of this one.
    static func circularNMS(_ candidates: [Detection], radius: Double) -> [Detection] {
        guard !candidates.isEmpty else { return [] }
        let order = candidates.indices.sorted { lhs, rhs in
            let a = candidates[lhs].score, b = candidates[rhs].score
            return a == b ? lhs < rhs : a > b        // stable, as in the reference
        }
        let radiusSquared = radius * radius
        var suppressed = [Bool](repeating: false, count: candidates.count)
        var kept = [Detection]()
        for index in order {
            if suppressed[index] { continue }
            let keeper = candidates[index]
            kept.append(keeper)
            for other in candidates.indices where !suppressed[other] {
                let dx = candidates[other].x - keeper.x
                let dy = candidates[other].y - keeper.y
                if dx * dx + dy * dy < radiusSquared { suppressed[other] = true }
            }
        }
        return kept
    }
}
