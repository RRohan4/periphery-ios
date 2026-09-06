//  PerceptionEngine.swift
//  Source-neutral image -> calibration -> detector backend.
//
//  Live capture and recorded-drive replay both adapt their inputs into a
//  PerceptionFrame. Nothing in this file knows which source produced it.

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

struct PerceptionTimings: Sendable {
    var preprocessMS: Double
    var inferenceMS: Double
    /// The four stages that make up `inferenceMS`, in milliseconds.
    ///
    /// `Detector` has always measured these -- they were simply never carried out
    /// of it, which left Replay able to see that a frame was slow but not which
    /// stage was slow. Without this split, "inference is 70 ms" is not actionable.
    var backboneMS: Double = 0
    var gatherMS: Double = 0
    var headMS: Double = 0
    var decodeMS: Double = 0

    /// What the four stages account for.
    var stagesMS: Double { backboneMS + gatherMS + headMS + decodeMS }
    /// The rest of `inferenceMS`. This is not noise: the image precision
    /// conversion in `Detector.detect` happens BEFORE the first stage mark, so it
    /// lands here and nowhere else.
    var unaccountedMS: Double { inferenceMS - stagesMS }
}

struct PerceptionCalibrationSnapshot {
    var calibration: Calibration
    var crop: ImageCrop
    var focal: Double
    var focalMatched: Bool
    var visibleVoxelFraction: Double
    var guides: GroundGuides
}

struct PerceptionDiagnostics {
    /// Actual Core ML boundary precisions selected on this device.
    var tensorPrecision: String
    var temporalReset: Bool
}

struct PerceptionResult {
    var timestamp: TimeInterval
    var rawDetections: [Detection]
    var trackedObjects: [TrackedVehicle]
    var egoMotion: EgoDelta
    var calibration: PerceptionCalibrationSnapshot
    var timings: PerceptionTimings
    var diagnostics: PerceptionDiagnostics
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

        var mark = DispatchTime.now()
        let input = try preprocessor.fill(from: frame.pixelBuffer, crop: crop)
        let preprocessMS = Self.ms(since: mark)

        mark = DispatchTime.now()
        let detections = try detector.detect(
            image: input,
            scoreThreshold: configuration.scoreThreshold,
            rejectImplausible: configuration.rejectImplausible)
        let inferenceMS = Self.ms(since: mark)
        // Read immediately: `lastTiming` describes the call that just returned.
        let stages = detector.lastTiming

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
            timestamp: frame.timestamp,
            rawDetections: detections,
            trackedObjects: tracked,
            egoMotion: ego,
            calibration: PerceptionCalibrationSnapshot(
                calibration: calibration,
                crop: crop,
                focal: calibration.achievedFocal(crop),
                focalMatched: calibration.focalIsMatched(crop),
                visibleVoxelFraction: detector.visibleVoxelFraction,
                guides: calibration.groundGuides()),
            timings: PerceptionTimings(
                preprocessMS: preprocessMS,
                inferenceMS: inferenceMS,
                backboneMS: stages.backbone * 1000,
                gatherMS: stages.gather * 1000,
                headMS: stages.head * 1000,
                decodeMS: stages.decode * 1000),
            diagnostics: PerceptionDiagnostics(
                tensorPrecision: detector.precisionNote,
                temporalReset: discontinuous))
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

    private static func ms(since mark: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - mark.uptimeNanoseconds) / 1e6
    }
}
