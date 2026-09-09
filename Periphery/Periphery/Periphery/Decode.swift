// Decode detector head outputs into vehicle-frame boxes. The operation order
// mirrors the Python training and post-processing paths.

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
}

enum Decode {

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
            //the small adjustments to the anchors happen here btw
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
            // 6. trained detection region; outside it is a decode artefact.
            guard detection.x >= Contract.forwardRange.min,
                  detection.x <= Contract.forwardRange.max,
                  detection.y >= Contract.lateralRange.min,
                  detection.y <= Contract.lateralRange.max else { continue }

            candidates.append(detection)
        }

        // 7. circular NMS at the contract radius.
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

    ///  matching contract.decode_boxes. Codes 7 and 8 are
    /// the unused velocity slots and are never read.

    //this just applys the tiny adjustments of the anchors to the box
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
    //combine everything into one yaw from heading direction to tiny adjustment
    static func gridYaw(rotation: Double, direction: Double) -> Double {
        let period = Double.pi
        var folded = rotation - Contract.dirOffset
        folded -= (folded / period).rounded(.down) * period
        return folded + Contract.dirOffset + period * direction
    }

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
