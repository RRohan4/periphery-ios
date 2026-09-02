//  SelfCheck.swift
//  Runs the golden vectors from tools/make_selfcheck.py against this port.
//
//  The CoreML halves are gated against ONNX at export time. This gates the
//  half that was hand-written: anchors, voxel centres, the projection LUT, box
//  decode, the direction fold, grid-to-vehicle, and circular NMS. It needs no
//  camera and no model, so run it at launch and again after any change to
//  Contract.swift, Calibration.swift or Decode.swift.
//
//  Regenerate the resources from the periphery repo:
//    PYTHONPATH=. .venv/bin/python ../periphery-ios/tools/make_selfcheck.py \
//        --out ../periphery-ios/Resources

import Foundation
import simd

struct SelfCheck {

    struct Result {
        var name: String
        var passed: Bool
        var detail: String
    }

    /// Tolerances. Anchors are float32 in the golden file, so a few ulps of
    /// disagreement is expected; anything larger is a real difference.
    private static let geometryTolerance = 1e-4
    private static let decodeTolerance = 2e-4

    static func run(bundle: Bundle = .main) -> [Result] {
        guard let manifestURL = bundle.url(forResource: "selfcheck", withExtension: "json"),
              let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONDecoder().decode(Manifest.self, from: manifestData) else {
            return [Result(name: "resources", passed: false,
                           detail: "selfcheck.json missing from the bundle")]
        }
        var results = [Result]()
        results.append(checkVoxelPoints(bundle))
        results.append(checkAnchors(bundle))
        let calibration = manifest.calibration()
        results.append(checkCrop(calibration, manifest))
        results.append(checkProjection(calibration, manifest))
        results.append(checkLUT(bundle, calibration, manifest))
        results.append(checkDecode(bundle, manifest))
        results.append(checkMountAxes(calibration))
        results.append(checkFocusOfExpansion())
        return results
    }

    static func summary(_ results: [Result]) -> String {
        let failed = results.filter { !$0.passed }
        let header = failed.isEmpty
            ? "self-check PASS (\(results.count) checks)"
            : "self-check FAIL (\(failed.count) of \(results.count))"
        return ([header] + results.map {
            "\($0.passed ? "  ok  " : "  FAIL") \($0.name): \($0.detail)"
        }).joined(separator: "\n")
    }

    // MARK: - Checks

    /// Round-trip the focus-of-expansion inversion against the forward
    /// projection it inverts.
    ///
    /// `Calibration.sourceVanishingPoint((1,0,0))` says where straight-ahead
    /// lands for a known mount. The estimator measures that pixel and runs the
    /// map backwards. Feeding the forward answer into the backward map must
    /// return the mount it started from -- and must do so at nonzero ROLL,
    /// which is the term that mixes the vertical and horizontal offsets into
    /// each other and the one a sign error hides in.
    ///
    /// Without this, a flipped sign produces a confident, plausible, wrong
    /// pitch: no crash, no NaN, just every range off by a fixed factor.
    private static func checkFocusOfExpansion() -> Result {
        // A representative windshield rig; the numbers only have to be
        // self-consistent, since this checks a round trip.
        let K = simd_double3x3(rows: [
            SIMD3<Double>(910.0, 0.0, 582.0),
            SIMD3<Double>(0.0, 910.0, 437.0),
            SIMD3<Double>(0.0, 0.0, 1.0),
        ])
        let cases: [(pitch: Double, roll: Double, yaw: Double)] = [
            (0, 0, 0),
            (-3.659, 0, 0),
            (2.5, 0, 0),
            (-3.0, 6.0, 0),
            (-3.0, 0, 4.0),
            (1.5, -8.0, -5.0),
            (-6.0, 12.0, 3.5),
        ]
        var worst = 0.0
        var worstCase = "-"
        for c in cases {
            var pose = MountPose.fallback
            pose.pitchDegrees = c.pitch
            pose.rollDegrees = c.roll
            pose.yawDegrees = c.yaw
            let calibration = Calibration(pose: pose, K: K,
                                          frameWidth: 1164, frameHeight: 874)
            guard let point = calibration
                .sourceVanishingPoint(SIMD3<Double>(1.0, 0.0, 0.0)) else {
                return Result(name: "focus of expansion", passed: false,
                              detail: "no vanishing point at "
                                    + "pitch \(c.pitch) roll \(c.roll) yaw \(c.yaw)")
            }
            let recovered = FocusOfExpansion.mountAngles(
                foe: point,
                fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                roll: pose.roll)
            let dPitch = abs(recovered.pitch * 180.0 / .pi - c.pitch)
            let dYaw = abs(recovered.yaw * 180.0 / .pi - c.yaw)
            if max(dPitch, dYaw) > worst {
                worst = max(dPitch, dYaw)
                worstCase = "pitch \(c.pitch) roll \(c.roll) yaw \(c.yaw)"
            }
        }
        // Degrees. Pure float arithmetic on a round trip, so this should be
        // machine-epsilon small; 1e-6 deg is 0.0000004% of the 0.25 deg budget.
        return Result(name: "focus of expansion", passed: worst < 1e-6,
                      detail: "7 mounts round-tripped, worst \(format(worst))° at \(worstCase)")
    }

    private static func checkVoxelPoints(_ bundle: Bundle) -> Result {
        guard let golden = floats(bundle, "selfcheck_points") else {
            return Result(name: "voxel points", passed: false, detail: "resource missing")
        }
        let points = Contract.voxelPoints()
        guard golden.count == points.count * 3 else {
            return Result(name: "voxel points", passed: false,
                          detail: "\(points.count) here, \(golden.count / 3) in the golden file")
        }
        var worst = 0.0
        for (i, p) in points.enumerated() {
            worst = max(worst, abs(p.x - Double(golden[i * 3])))
            worst = max(worst, abs(p.y - Double(golden[i * 3 + 1])))
            worst = max(worst, abs(p.z - Double(golden[i * 3 + 2])))
        }
        return Result(name: "voxel points", passed: worst < geometryTolerance,
                      detail: "\(points.count) centres, max diff \(format(worst))")
    }

    private static func checkAnchors(_ bundle: Bundle) -> Result {
        guard let golden = floats(bundle, "selfcheck_anchors") else {
            return Result(name: "anchors", passed: false, detail: "resource missing")
        }
        let anchors = Contract.anchors()
        guard golden.count == anchors.count * 7 else {
            return Result(name: "anchors", passed: false,
                          detail: "\(anchors.count) here, \(golden.count / 7) in the golden file")
        }
        var worst = 0.0
        for (i, a) in anchors.enumerated() {
            let mine = [a.x, a.y, a.z, a.w, a.l, a.h, a.yaw]
            for (j, value) in mine.enumerated() {
                worst = max(worst, abs(value - Double(golden[i * 7 + j])))
            }
        }
        return Result(name: "anchors", passed: worst < geometryTolerance,
                      detail: "\(anchors.count) rows, max diff \(format(worst))")
    }

    private static func checkCrop(_ calibration: Calibration, _ manifest: Manifest) -> Result {
        let crop = calibration.focalMatchedCrop(targetFocal: manifest.calibrationBlock.trainedFocal)
        let matches = crop.x == manifest.crop.x && crop.y == manifest.crop.y
            && crop.width == manifest.crop.width && crop.height == manifest.crop.height
        return Result(name: "focal-matched crop", passed: matches,
                      detail: "\(crop.width)x\(crop.height) at (\(crop.x), \(crop.y)), "
                            + "focal \(format(calibration.achievedFocal(crop)))")
    }

    private static func checkProjection(_ calibration: Calibration, _ manifest: Manifest) -> Result {
        let crop = calibration.focalMatchedCrop(targetFocal: manifest.calibrationBlock.trainedFocal)
        let projection = calibration.projection(crop: crop)
        var worst = 0.0
        for row in 0..<3 {
            for column in 0..<4 {
                worst = max(worst, abs(projection[column][row] - manifest.projection[row][column]))
            }
        }
        return Result(name: "projection", passed: worst < geometryTolerance,
                      detail: "max diff \(format(worst))")
    }

    private static func checkLUT(_ bundle: Bundle, _ calibration: Calibration,
                                 _ manifest: Manifest) -> Result {
        guard let url = bundle.url(forResource: "selfcheck_lut", withExtension: "bin"),
              let data = try? Data(contentsOf: url) else {
            return Result(name: "projection LUT", passed: false, detail: "resource missing")
        }
        let count = Contract.voxelCount
        guard data.count == count * 5 else {
            return Result(name: "projection LUT", passed: false,
                          detail: "golden file is \(data.count) bytes, expected \(count * 5)")
        }
        let goldenIndices: [Int32] = data.prefix(count * 4).withUnsafeBytes {
            Array($0.bindMemory(to: Int32.self))
        }
        let goldenVisible: [UInt8] = Array(data.suffix(count))

        let crop = calibration.focalMatchedCrop(targetFocal: manifest.calibrationBlock.trainedFocal)
        let lut = ProjectionLUT(projection: calibration.projection(crop: crop))
        var visibilityMismatches = 0
        var indexMismatches = 0
        for i in 0..<count {
            let visible = goldenVisible[i] != 0
            if (lut.visibility[i] != 0) != visible { visibilityMismatches += 1 }
            // An invisible voxel's index is never read, so only gate the ones
            // that are actually gathered.
            if visible && lut.indices[i] != goldenIndices[i] { indexMismatches += 1 }
        }
        let passed = visibilityMismatches == 0 && indexMismatches == 0
        return Result(name: "projection LUT", passed: passed,
                      detail: "\(visibilityMismatches) visibility, \(indexMismatches) index "
                            + "mismatches; visible fraction \(format(lut.visibleFraction)) "
                            + "(golden \(format(manifest.counts.visibleFraction)))")
    }

    private static func checkDecode(_ bundle: Bundle, _ manifest: Manifest) -> Result {
        guard let head = floats(bundle, "selfcheck_head") else {
            return Result(name: "decode", passed: false, detail: "resource missing")
        }
        let n = Contract.candidateCount
        guard head.count == n * 15 else {
            return Result(name: "decode", passed: false,
                          detail: "golden head file has \(head.count) floats, expected \(n * 15)")
        }
        var detections = [Detection]()
        head.withUnsafeBufferPointer { buffer in
            let base = buffer.baseAddress!
            detections = Decode.detections(classes: base,
                                           boxes: base + n * 4,
                                           directions: base + n * 13,
                                           anchors: Contract.anchors(),
                                           scoreThreshold: manifest.decode.threshold,
                                           nmsRadius: manifest.decode.nmsRadius)
        }
        guard detections.count == manifest.detections.count else {
            return Result(name: "decode", passed: false,
                          detail: "\(detections.count) detections, golden has "
                                + "\(manifest.detections.count)")
        }
        var worst = 0.0
        var worstField = "-"
        for (mine, golden) in zip(detections, manifest.detections) {
            if mine.label != golden.label {
                return Result(name: "decode", passed: false,
                              detail: "label \(mine.label) != \(golden.label)")
            }
            let pairs: [(String, Double, Double)] = [
                ("score", Double(mine.score), golden.score),
                ("x", mine.x, golden.x), ("y", mine.y, golden.y), ("z", mine.z, golden.z),
                ("length", mine.length, golden.length), ("width", mine.width, golden.width),
                ("height", mine.height, golden.height), ("yaw", mine.yaw, golden.yaw),
            ]
            for (name, a, b) in pairs where abs(a - b) > worst {
                worst = abs(a - b); worstField = name
            }
        }
        return Result(name: "decode", passed: worst < decodeTolerance,
                      detail: "\(detections.count) detections, worst \(worstField) diff "
                            + format(worst))
    }

    /// Roll and yaw have no golden -- the Python rig had neither axis, so
    /// `sensor_T_vehicle` there is pitch-only and the committed goldens exercise
    /// the new terms at exactly zero, where they vanish. That is a test that
    /// passes for the wrong reason.
    ///
    /// So this asserts the two things a golden would have caught anyway: that
    /// the rotation still collapses to the Python's Ry(pitch) at zero roll and
    /// yaw, and that each axis moves the image the way the physical mount does.
    /// A flipped sign applies twice the mount angle instead of cancelling it,
    /// and the failure looks like a bad detector rather than a bad transform.
    private static func checkMountAxes(_ calibration: Calibration) -> Result {
        var failures = [String]()

        // 1. Zero roll and yaw must reproduce the Python's Ry(pitch) exactly.
        let pitch = calibration.pitch
        let mine = Calibration.vehicleToSensor(pitch: pitch, roll: 0, yaw: 0)
        let python = simd_double3x3(rows: [
            SIMD3<Double>(cos(pitch), 0.0, sin(pitch)),
            SIMD3<Double>(0.0, 1.0, 0.0),
            SIMD3<Double>(-sin(pitch), 0.0, cos(pitch)),
        ])
        var worst = 0.0
        for column in 0..<3 {
            for row in 0..<3 {
                worst = max(worst, abs(mine[column][row] - python[column][row]))
            }
        }
        if worst > geometryTolerance { failures.append("Ry(pitch) diff \(format(worst))") }

        // 2. Every rotation must be orthonormal with determinant +1. A typo in
        //    a matrix literal shows up here before it shows up as a bad range.
        let tilted = Calibration.vehicleToSensor(pitch: 0.07, roll: -0.05, yaw: 0.03)
        let identity = tilted.transpose * tilted
        var orthoError = 0.0
        for column in 0..<3 {
            for row in 0..<3 {
                orthoError = max(orthoError,
                                 abs(identity[column][row] - (column == row ? 1.0 : 0.0)))
            }
        }
        if orthoError > 1e-12 { failures.append("not orthonormal, \(format(orthoError))") }
        if abs(tilted.determinant - 1.0) > 1e-12 {
            failures.append("det \(format(tilted.determinant)), expected +1")
        }

        // 3. Each axis, one at a time, against the physical mount. Image axes
        //    are x right, y down, so "further down the image" is +y.
        let crop = calibration.focalMatchedCrop()
        let five = 5.0 * Double.pi / 180.0
        func imagePoint(_ vehicle: SIMD3<Double>,
                        pitch: Double, roll: Double, yaw: Double) -> SIMD2<Double>? {
            var probe = calibration
            probe.pose.pitch = pitch
            probe.pose.roll = roll
            probe.pose.yaw = yaw
            let rotation = Calibration.vehicleToSensor(pitch: pitch, roll: roll, yaw: yaw)
            let camera = SIMD3<Double>(probe.forwardOfOrigin, 0.0, probe.height)
            let sensor = rotation * (vehicle - camera)
            let image = Contract.sensorToImageAxes * sensor
            guard image.z > 1e-9 else { return nil }
            let adjusted = probe.adjustedIntrinsics(for: crop)
            let projected = adjusted * image
            return SIMD2<Double>(projected.x / projected.z, projected.y / projected.z)
        }

        let ahead = SIMD3<Double>(40.0, 0.0, 0.0)
        // Nose up must push the road FURTHER DOWN the image.
        if let level = imagePoint(ahead, pitch: 0, roll: 0, yaw: 0),
           let up = imagePoint(ahead, pitch: five, roll: 0, yaw: 0) {
            if !(up.y > level.y + 1.0) {
                failures.append(String(format: "pitch +5 deg moved row %.1f -> %.1f, expected down",
                                       level.y, up.y))
            }
        } else {
            failures.append("pitch probe fell behind the camera")
        }
        // Camera yawed LEFT must move a straight-ahead point RIGHT in the image.
        if let level = imagePoint(ahead, pitch: 0, roll: 0, yaw: 0),
           let left = imagePoint(ahead, pitch: 0, roll: 0, yaw: five) {
            if !(left.x > level.x + 1.0) {
                failures.append(String(format: "yaw +5 deg moved column %.1f -> %.1f, expected right",
                                       level.x, left.x))
            }
        } else {
            failures.append("yaw probe fell behind the camera")
        }
        // Camera rolled clockwise from behind (left side higher) must DROP the
        // left end of the horizon relative to the right.
        var rolled = calibration
        rolled.pose.pitch = 0
        rolled.pose.yaw = 0
        rolled.pose.roll = five
        if let onLeft = rolled.vanishingPoint(SIMD3<Double>(1.0, 0.3, 0.0), crop: crop),
           let onRight = rolled.vanishingPoint(SIMD3<Double>(1.0, -0.3, 0.0), crop: crop) {
            if !(onLeft.y > onRight.y) {
                failures.append(String(format: "roll +5 deg put the left horizon at %.1f, right at %.1f",
                                       onLeft.y, onRight.y))
            }
        } else {
            failures.append("roll probe has no vanishing point")
        }

        return Result(name: "mount axes (pitch, roll, yaw)",
                      passed: failures.isEmpty,
                      detail: failures.isEmpty
                          ? "collapses to Ry(pitch); all three signs correct"
                          : failures.joined(separator: "; "))
    }


    // MARK: - Golden files

    private static func floats(_ bundle: Bundle, _ name: String) -> [Float]? {
        guard let url = bundle.url(forResource: name, withExtension: "bin"),
              let data = try? Data(contentsOf: url) else { return nil }
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    // MARK: - Manifest

    struct Manifest: Decodable {
        struct CalibrationBlock: Decodable {
            let pitchDeg: Double
            let cameraHeight: Double
            let cameraForward: Double
            let frameSize: [Int]
            let K: [[Double]]
            let trainedFocal: Double

            enum CodingKeys: String, CodingKey {
                case pitchDeg = "pitch_deg"
                case cameraHeight = "camera_height_m"
                case cameraForward = "camera_forward_of_origin_m"
                case frameSize = "frame_size"
                case K
                case trainedFocal = "trained_focal_px"
            }
        }
        struct Crop: Decodable { let x, y, width, height: Int }
        struct Counts: Decodable {
            let visibleFraction: Double
            enum CodingKeys: String, CodingKey { case visibleFraction = "visible_fraction" }
        }
        struct DecodeBlock: Decodable {
            let threshold: Double
            let nmsRadius: Double
            enum CodingKeys: String, CodingKey {
                case threshold
                case nmsRadius = "nms_radius"
            }
        }
        struct GoldenDetection: Decodable {
            let score: Double
            let label: Int
            let x, y, z, length, width, height, yaw: Double
        }

        let calibrationBlock: CalibrationBlock
        let crop: Crop
        let projection: [[Double]]
        let counts: Counts
        let decode: DecodeBlock
        let detections: [GoldenDetection]

        enum CodingKeys: String, CodingKey {
            case calibrationBlock = "calibration"
            case crop, projection, counts, decode, detections
        }

        func calibration() -> Calibration {
            let k = calibrationBlock.K
            let intrinsics = simd_double3x3(rows: [
                SIMD3<Double>(k[0][0], k[0][1], k[0][2]),
                SIMD3<Double>(k[1][0], k[1][1], k[1][2]),
                SIMD3<Double>(k[2][0], k[2][1], k[2][2]),
            ])
            var pose = MountPose.fallback
            pose.pitch = calibrationBlock.pitchDeg * .pi / 180.0
            pose.roll = 0.0
            pose.yaw = 0.0
            pose.height = calibrationBlock.cameraHeight
            pose.forwardOfOrigin = calibrationBlock.cameraForward
            return Calibration(pose: pose,
                               K: intrinsics,
                               frameWidth: calibrationBlock.frameSize[0],
                               frameHeight: calibrationBlock.frameSize[1])
        }
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.6g", value)
    }
}
