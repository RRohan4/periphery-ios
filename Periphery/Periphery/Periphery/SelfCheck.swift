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
            return Calibration(pitch: calibrationBlock.pitchDeg * .pi / 180.0,
                               height: calibrationBlock.cameraHeight,
                               forwardOfOrigin: calibrationBlock.cameraForward,
                               K: intrinsics,
                               frameWidth: calibrationBlock.frameSize[0],
                               frameHeight: calibrationBlock.frameSize[1])
        }
    }

    private static func format(_ value: Double) -> String {
        String(format: "%.6g", value)
    }
}
