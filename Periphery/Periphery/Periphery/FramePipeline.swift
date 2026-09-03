//  FramePipeline.swift
//  Camera -> preprocess -> detect, at whatever rate the phone holds.
//
//  Confined to the capture queue after `start()`. Only value types cross to the
//  main actor, once per frame, for drawing. Nothing here keeps temporal state --
//  each frame is independent, exactly as the contract says.
//
//  Split out of LiveView.swift, which had grown to hold the pipeline, the
//  CoreMotion client, the renderer, the screen and the view model. This file is
//  the pipeline and nothing else.

import AVFoundation
import Foundation
import simd

/// Confined to the capture queue after `start()`. The published snapshot is a
/// plain value type handed to the main actor once per frame.
final class FramePipeline: @unchecked Sendable {

    struct Snapshot {
        var detections: [Detection] = []
        var inferenceMS: Double = 0
        var preprocessMS: Double = 0
        var fps: Double = 0
        var pitchDegrees: Double = 0
        var focal: Double = 0
        var cropDescription: String = ""
        var thermal: String = "nominal"
        var dropped: Int = 0
        var note: String = ""
        var pose = MountPose.fallback
        /// Raw camera roll from gravity, degrees, before the plausibility gate.
        var measuredRollDegrees: Double = 0
        /// Mount yaw from heading minus course, degrees, when both are
        /// trustworthy. Computed and shown, not yet applied -- it wants the
        /// estimator's windowing, not a per-sample difference.
        var measuredYawDegrees: Double?
        var speed: Double = -1
        var relativeAltitude: Double?
        /// Set when the mount is somewhere the projection cannot follow.
        var mountWarning: String = ""
        var recording = DriveRecorder.Status()
        /// Raw gravity pitch, degrees, always -- even when pitch is locked to a
        /// manual or estimated value, so the two can be compared.
        var gravityPitchDegrees: Double = 0
        /// Fraction of voxels landing on the feature map. A sane windshield
        /// mount sits around 0.5-0.7; near zero means the pitch sign is flipped.
        var visibleFraction: Double = 0
        /// False when the frame is too narrow to reach the trained focal, in
        /// which case every range carries a scale error that must be stated.
        var focalMatched = true
        /// Lens position 0-1, whether focus is pinned, and whether the camera is
        /// still hunting. A soft frame costs the flow estimator far more than it
        /// costs the detector, so this is not a nicety.
        var lensPosition: Float = 0
        var focusLocked = false
        var focusHunting = false
        /// Horizon and ground-distance lines in source pixels, for the overlay
        /// on the camera preview. Computed here because this is where the
        /// calibration lives.
        var guides = GroundGuides()
        /// The drive-time camera estimator. Grade-immune, unlike gravity, and
        /// self-announcing when it fails -- see FocusOfExpansion.
        var foe = FocusOfExpansion.Estimate()
        /// Score below which a candidate is not drawn. 0.50 is the measured
        /// operating point; above it, precision rises and recall falls.
        var scoreThreshold = Contract.scoreThreshold
    }

    let camera = CameraSession()
    let motion = MotionSource()
    let recorder = DriveRecorder()
    let foe = FocusOfExpansion()
    private var preprocessor: Preprocessor?
    private var detector: Detector?
    private var calibration: Calibration?
    private var busy = false
    private var dropped = 0
    private var lastFrameTime: DispatchTime?
    private var smoothedFPS = 0.0
    /// The mount pose the LUT is built from. Pitch is gravity-referenced for
    /// now -- written by CoreMotion, read on the capture queue.
    private var pose = MountPose.load()
    private var smoothedRoll: Double?
    private var measuredRoll: Double = 0
    private var measuredYaw: Double?
    private var gravityPitch: Double = 0
    /// Live operating point. Written from the Calibrate tab, read here.
    private var scoreThreshold = Contract.scoreThreshold
    /// Reject boxes whose decoded dimensions are not a vehicle. Off in the
    /// goldens, on live -- see Decode.plausible.
    private var rejectImplausible = true
    /// Hand the camera estimate to the pose without a person asking, once it
    /// has converged.
    private var autoApplyFOE = true

    /// Beyond this the mount is not one the projection can follow.
    private static let rollLimit = 25.0 * Double.pi / 180.0

    var onSnapshot: ((Snapshot) -> Void)?
    /// The pose as of the last frame, for the recorder's manifest and the
    /// Calibrate tab. Read off the capture queue; a torn Double here would only
    /// mis-stamp a manifest, never the pipeline.
    var currentPose: MountPose { pose }

    /// Raw gravity pitch in radians, whether or not pitch is locked to it.
    var currentGravityPitch: Double { gravityPitch }
    var currentMeasuredYaw: Double? { measuredYaw }

    // MARK: - Pose edits
    //
    // Called from the Calibrate tab on the main actor, against a `pose` the
    // capture queue also writes. The next frame rebuilds the LUT from a whole
    // copy of the struct, so the worst case is one frame of mixed geometry --
    // against which the alternative, a lock on the 100 Hz attitude path, is a
    // poor trade.

    /// An explicit choice by a person; always accepted.
    func setPitch(degrees: Double, from provenance: MountPose.Provenance = .manual) {
        pose.pitchDegrees = degrees
        pose.pitchFrom = provenance
        pose.save()
    }

    /// An automatic update, subject to precedence. This is how the drive-time
    /// estimator supersedes a typed-in guess without a manual entry being able
    /// to freeze the pose forever.
    @discardableResult
    func offerPitch(_ radians: Double, from provenance: MountPose.Provenance) -> Bool {
        guard provenance.mayOverwrite(pose.pitchFrom) else { return false }
        pose.pitch = radians
        pose.pitchFrom = provenance
        return true
    }

    /// Hand pitch back to gravity. Not "the answer" -- gravity measures
    /// mount + road grade -- but the only thing available while stopped.
    func releasePitchToGravity() {
        pose.pitch = gravityPitch
        pose.pitchFrom = .gravity
        pose.save()
    }

    func setHeight(_ metres: Double, from provenance: MountPose.Provenance = .manual) {
        pose.height = metres
        pose.heightFrom = provenance
        pose.save()
    }

    func setForwardOfOrigin(_ metres: Double) {
        pose.forwardOfOrigin = metres
        pose.save()
    }

    /// Apply the measured yaw offset. Gated by the caller on course accuracy
    /// and speed; 1 degree is 0.70 m of lateral error at 40 m, on every box.
    func applyMeasuredYaw() {
        guard let yaw = measuredYaw else { return }
        pose.yaw = yaw
        pose.yawFrom = .estimated
        pose.save()
    }

    func clearYaw() {
        pose.yaw = 0
        pose.yawFrom = .fallback
        pose.save()
    }

    /// Apply the camera estimator's pitch now, as an explicit choice. Tagged
    /// `.estimated` rather than `.manual` because it IS the estimator's number,
    /// and tagging it manual would freeze out later, better windows.
    func applyEstimatedPitch() {
        let estimate = foe.estimate
        guard estimate.gates.writesToPose, estimate.reportable else { return }
        pose.pitchDegrees = estimate.pitchDegrees
        pose.pitchFrom = .estimated
        pose.pitchSigmaDegrees = estimate.sigmaDegrees
        pose.save()
    }

    /// The mount yaw the same fit gives for free. Independent of the compass
    /// path in `applyMeasuredYaw`, and available without a course fix.
    func applyEstimatedYaw() {
        let estimate = foe.estimate
        guard estimate.gates.writesToPose, estimate.reportable else { return }
        pose.yawDegrees = estimate.yawDegrees
        pose.yawFrom = .estimated
        pose.save()
    }

    /// Live operating point for the decoder. 0.50 is where precision and recall
    /// were measured (P 0.647 / R 0.678); raising it trades recall for
    /// precision and is the honest lever on a cluttered view.
    func setScoreThreshold(_ value: Double) { scoreThreshold = value }
    var currentScoreThreshold: Double { scoreThreshold }

    func setRejectImplausible(_ value: Bool) { rejectImplausible = value }
    var currentRejectImplausible: Bool { rejectImplausible }

    func setAutoApplyFOE(_ value: Bool) { autoApplyFOE = value }
    var currentAutoApplyFOE: Bool { autoApplyFOE }

    func resetPose() {
        pose = .fallback
        MountPose.clear()
    }
    var session: AVCaptureSession { camera.session }

    /// The validity flags a recorded drive has to carry, gathered from the two
    /// objects that actually know them.
    var captureFlags: DriveRecorder.Capture {
        DriveRecorder.Capture(
            referenceFrame: motion.headingIsTrueNorth
                ? "xTrueNorthZVertical" : "xArbitraryZVertical",
            headingIsTrueNorth: motion.headingIsTrueNorth,
            stabilizationDisabled: camera.stabilizationDisabled,
            intrinsicsAvailable: camera.intrinsicsAvailable,
            altimeterAvailable: motion.altimeterAvailable,
            attitudeRateHz: motion.attitudeRate)
    }

    func start() async throws {
        guard await CameraSession.requestAccess() else { throw CameraSession.CameraError.denied }
        try camera.configure()
        preprocessor = try Preprocessor()
        startMotion()
        camera.onFrame = { [weak self] frame in self?.handle(frame) }
        camera.start()
    }

    func stop() {
        camera.stop()
        motion.stop()
    }

    // MARK: Cold-start pose

    private func startMotion() {
        motion.onAttitude = { [weak self] attitude in
            guard let self else { return }
            self.gravityPitch = attitude.gravityPitch

            // Heavy smoothing: this is the seconds-to-minutes timescale, and
            // per-frame rattle belongs to the gyro, not here. MotionSource runs
            // at 100 Hz, so tau is about 0.5 s.
            //
            // Gravity may only write pitch where it outranks what is already
            // there. It beats the built-in default and itself, and nothing
            // else: a typed-in guess or an estimator output must not be walked
            // back over the next few seconds by a source that cannot tell mount
            // angle from road grade.
            //
            // A typed-in pitch is still only an INITIAL GUESS. It outranks
            // gravity so it survives, and is outranked by the drive-time
            // estimator, which is the thing it exists to seed.
            if MountPose.Provenance.gravity.mayOverwrite(self.pose.pitchFrom) {
                self.pose.pitch = self.pose.pitch * 0.98 + attitude.gravityPitch * 0.02
                self.pose.pitchFrom = .gravity
            }

            // Roll IS honestly measurable from gravity -- its contaminant is
            // road camber at ~0.6 deg and mean-zero, not the 2.45 deg of grade
            // that makes the same trick fail for pitch. So it is applied, not
            // merely displayed.
            self.smoothedRoll = self.smoothedRoll == nil
                ? attitude.roll
                : self.smoothedRoll! * 0.98 + attitude.roll * 0.02
            let roll = self.smoothedRoll ?? 0
            self.measuredRoll = roll
            if abs(roll) <= Self.rollLimit {
                self.pose.roll = roll
                self.pose.rollFrom = .gravity
            } else {
                // Not a windshield mount the projection can follow. The capture
                // buffer is landscape however the phone is held, so a portrait
                // mount does not rotate the image -- it lays the road sideways
                // across a crop computed for the other axis. Say so; do not
                // quietly reproject.
                self.pose.roll = 0
                self.pose.rollFrom = .fallback
            }

            self.recorder.append(attitude: attitude)
            // The camera estimator needs rotation rate on its own timeline, at
            // CoreMotion's rate rather than the frame rate: it averages over
            // each frame pair's interval.
            self.foe.append(rotationRate: attitude.rotationRate, at: attitude.timestamp)

            if let heading = attitude.cameraHeading, let fix = self.motion.latestLocation,
               fix.course >= 0, fix.courseAccuracy >= 0, fix.courseAccuracy < 5.0,
               fix.speed > 5.0 {
                self.measuredYaw = MotionSource.mountYaw(cameraHeading: heading,
                                                         course: fix.course)
            }
        }
        motion.onLocation = { [weak self] location in
            self?.recorder.append(location: location)
        }
        motion.onAltitude = { [weak self] altitude in
            self?.recorder.append(altitude: altitude)
        }
        motion.onHeading = { [weak self] heading in
            self?.recorder.append(heading: heading)
        }
        motion.start()
    }

    // MARK: Per frame

    /// Take the camera estimate once it has converged.
    ///
    /// Precedence does the rest: `.estimated` outranks everything automatic, so
    /// this supersedes gravity and a typed-in starting guess -- which is what a
    /// starting guess is for -- and a person's later explicit action still wins.
    private func acceptFOE(_ estimate: FocusOfExpansion.Estimate) {
        // `converged` is already false for any profile that may not write the
        // pose, but state it here too: a handheld reading is the angle of a
        // HAND, and silently installing it as the mount angle would be the
        // worst kind of bug -- plausible, persistent, and saved to disk.
        guard estimate.gates.writesToPose else { return }
        guard autoApplyFOE, estimate.converged else { return }
        guard MountPose.Provenance.estimated.mayOverwrite(pose.pitchFrom) else { return }
        pose.pitchDegrees = estimate.pitchDegrees
        pose.pitchFrom = .estimated
        pose.pitchSigmaDegrees = estimate.sigmaDegrees
    }

    private func handle(_ frame: CameraSession.Frame) {
        // Recording is deliberately AHEAD of the busy guard. The recorded
        // drive is the artefact; detection is a passenger on it. A detector
        // that falls behind must not punch holes in the video.
        recorder.append(frame: frame)

        // Also ahead of the busy guard, and for the same reason: the estimator
        // wants an even sample of the drive, not whatever the detector happened
        // to leave time for. It drops its own frames internally.
        if let calibration {
            foe.feed(frame: frame, calibration: calibration,
                     speed: motion.latestLocation?.speed ?? -1)
        }
        let estimate = foe.estimate
        acceptFOE(estimate)

        guard !busy else { dropped += 1; return }
        busy = true
        defer { busy = false }

        let now = DispatchTime.now()
        if let last = lastFrameTime {
            let delta = Double(now.uptimeNanoseconds - last.uptimeNanoseconds) / 1e9
            if delta > 0 {
                let instant = 1.0 / delta
                smoothedFPS = smoothedFPS == 0 ? instant : smoothedFPS * 0.9 + instant * 0.1
            }
        }
        lastFrameTime = now

        var snapshot = Snapshot()
        snapshot.fps = smoothedFPS
        snapshot.dropped = dropped
        snapshot.pitchDegrees = pose.pitchDegrees
        snapshot.pose = pose
        snapshot.measuredRollDegrees = measuredRoll * 180.0 / .pi
        snapshot.measuredYawDegrees = measuredYaw.map { $0 * 180.0 / .pi }
        snapshot.speed = motion.latestLocation?.speed ?? -1
        snapshot.relativeAltitude = motion.latestAltitude?.relativeAltitude
        if abs(measuredRoll) > Self.rollLimit {
            snapshot.mountWarning = String(
                format: "mount rolled %.0f deg — the capture buffer is landscape "
                      + "however the phone is held, so this is not recoverable",
                measuredRoll * 180.0 / .pi)
        }
        snapshot.thermal = Benchmark.describe(ProcessInfo.processInfo.thermalState)
        snapshot.recording = recorder.status
        snapshot.gravityPitchDegrees = gravityPitch * 180.0 / .pi
        snapshot.foe = estimate
        snapshot.scoreThreshold = scoreThreshold
        snapshot.lensPosition = camera.lensPosition
        snapshot.focusLocked = camera.focusIsLocked
        snapshot.focusHunting = camera.isAdjustingFocus

        do {
            let calibration = try calibrate(with: frame)
            let crop = calibration.focalMatchedCrop()
            snapshot.focal = calibration.achievedFocal(crop)
            snapshot.focalMatched = calibration.focalIsMatched(crop)
            snapshot.visibleFraction = detector?.visibleVoxelFraction ?? 0
            snapshot.guides = calibration.groundGuides()
            snapshot.cropDescription = "\(crop.width)x\(crop.height) at (\(crop.x), \(crop.y))"

            guard let preprocessor, let detector else { return }
            var mark = DispatchTime.now()
            let input = try preprocessor.fill(from: frame.pixelBuffer, crop: crop)
            snapshot.preprocessMS = Self.ms(since: mark)

            mark = DispatchTime.now()
            snapshot.detections = try detector.detect(image: input,
                                                      scoreThreshold: scoreThreshold,
                                                      rejectImplausible: rejectImplausible)
            snapshot.inferenceMS = Self.ms(since: mark)
            recorder.append(detections: snapshot.detections, at: frame.presentationTime)
        } catch {
            snapshot.note = String(describing: error)
        }

        onSnapshot?(snapshot)
    }

    /// Build (or refresh) the calibration for this frame. The LUT is rebuilt
    /// only when the pose actually moves -- it depends on calibration, not on
    /// the image, and rebuilding it per frame would be pure waste.
    private func calibrate(with frame: CameraSession.Frame) throws -> Calibration {
        let intrinsics = frame.intrinsics ?? Self.fallbackIntrinsics(width: frame.width,
                                                                    height: frame.height)
        let updated = Calibration(pose: pose,
                                  K: intrinsics,
                                  frameWidth: frame.width,
                                  frameHeight: frame.height)
        if let existing = calibration {
            // ~0.03 deg on any axis. The LUT depends on the pose, not the
            // image, so rebuilding it per frame would be pure waste.
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
        calibration = updated
        return updated
    }

    /// If intrinsic delivery is unavailable, assume a 60-degree horizontal
    /// field of view. Stated rather than silent: every range would then carry
    /// an unmeasured scale error.
    private static func fallbackIntrinsics(width: Int, height: Int) -> simd_double3x3 {
        let focal = Double(width) / (2.0 * tan(60.0 * .pi / 180.0 / 2.0))
        return simd_double3x3(rows: [
            SIMD3<Double>(focal, 0, Double(width) / 2),
            SIMD3<Double>(0, focal, Double(height) / 2),
            SIMD3<Double>(0, 0, 1),
        ])
    }

    private static func ms(since mark: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - mark.uptimeNanoseconds) / 1e6
    }
}
