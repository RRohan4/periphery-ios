// Image -> calibration -> detector pipeline, fed one PerceptionFrame at a
// time by FramePipeline.

import CoreVideo
import Foundation
import simd

struct PerceptionFrame {
    let pixelBuffer: CVPixelBuffer
    /// Seconds on the source's monotonic media timeline.
    let timestamp: TimeInterval
    /// Intrinsics for this exact source image, before crop or resize.
    let intrinsics: simd_double3x3
    let width: Int
    let height: Int
    let mountPose: MountPose
    let egoMotion: EgoDelta
}

struct PerceptionConfiguration {
    var scoreThreshold = Contract.scoreThreshold
    var rejectImplausible = true
}

struct PerceptionCalibrationSnapshot {
    var calibration: Calibration
    var crop: ImageCrop
    var focal: Double
    var focalMatched: Bool
    var visibleVoxelFraction: Double
}

struct PerceptionResult {
    var rawDetections: [Detection]
    var trackedObjects: [TrackedVehicle]
    var calibration: PerceptionCalibrationSnapshot
}

final class PerceptionEngine {
    private let preprocessor: Preprocessor
    private var detector: Detector?
    private(set) var currentCalibration: Calibration?
    private let tracker = VehicleTracker()
    private var lastTimestamp: Double?

    init() throws {
        preprocessor = try Preprocessor()
    }

    func process(frame: PerceptionFrame,
                 configuration: PerceptionConfiguration = PerceptionConfiguration()) throws
        -> PerceptionResult {
        let calibration = try calibrate(for: frame)
        let crop = calibration.focalMatchedCrop()
        guard let detector else {
            throw DetectorError.modelMissing("detector was not initialized")
        }

        let input = try preprocessor.fill(from: frame.pixelBuffer, crop: crop)
        let detections = try detector.detect(
            image: input,
            scoreThreshold: configuration.scoreThreshold,
            rejectImplausible: configuration.rejectImplausible)

        let frameInterval = lastTimestamp.map { frame.timestamp - $0 }
        let discontinuous = !frame.timestamp.isFinite
            || !frame.egoMotion.valid
            || frameInterval.map { $0 <= 0 || $0 > 1
                || abs(frame.egoMotion.dt - $0) > 0.02 } ?? true
        if discontinuous { tracker.reset() }
        let ego = discontinuous ? EgoDelta(dt: 0) : frame.egoMotion
        let tracked = tracker.step(detections: detections, ego: ego, timestamp: frame.timestamp)
        lastTimestamp = frame.timestamp

        return PerceptionResult(
            rawDetections: detections,
            trackedObjects: tracked,
            calibration: PerceptionCalibrationSnapshot(
                calibration: calibration,
                crop: crop,
                focal: calibration.achievedFocal(crop),
                focalMatched: calibration.focalIsMatched(crop),
                visibleVoxelFraction: detector.visibleVoxelFraction))
    }

    func resetTemporalState() {
        tracker.reset()
        lastTimestamp = nil
    }

    /// Rebuild the LUT only when its calibration inputs move. This preserves
    /// the previous Live behavior while making the lifecycle source-neutral.
    private func calibrate(for frame: PerceptionFrame) throws -> Calibration {
        let updated = Calibration(
            pose: frame.mountPose,
            K: frame.intrinsics,
            frameWidth: frame.width,
            frameHeight: frame.height)

        if let existing = currentCalibration {
            let poseMoved = abs(existing.pitch - updated.pitch) > 0.0005
                || abs(existing.roll - updated.roll) > 0.0005
                || abs(existing.yaw - updated.yaw) > 0.0005
                || abs(existing.height - updated.height) > 0.005
            let opticsMoved = existing.K[0][0] != updated.K[0][0]
                || existing.frameWidth != updated.frameWidth
            if !poseMoved && !opticsMoved {
                return existing
            }
        }

        if detector == nil {
            detector = try Detector(calibration: updated)
        } else {
            detector?.updateCalibration(updated)
        }
        currentCalibration = updated
        return updated
    }
}
