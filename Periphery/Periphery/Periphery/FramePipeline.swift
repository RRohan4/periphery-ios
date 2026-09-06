// Live camera and motion adapter for the source-neutral PerceptionEngine. The
// capture queue owns sensor history; only value-type snapshots reach the UI.

import AVFoundation
import Foundation
import simd

/// Confined to the capture queue after `start()`. The published snapshot is a
/// plain value type handed to the main actor once per frame.
final class FramePipeline: @unchecked Sendable {

    struct Snapshot {
        var detections: [Detection] = []
        var trackedObjects: [TrackedVehicle] = []
        /// Full source-frame geometry for the presentation adapter. Rendering
        /// consumes it but never feeds changes back into perception.
        var calibration: PerceptionCalibrationSnapshot?
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

        var lensPosition: Float = 0
        var focusLocked = false
        var focusHunting = false

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
    private var engine: PerceptionEngine?
    private let egoMotion = LiveEgoMotion()
    private let perceptionLock = NSLock()
    private var perceptionEnabled = true
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

    var currentPose: MountPose { pose }

    /// Raw gravity pitch in radians, whether or not pitch is locked to it.
    var currentGravityPitch: Double { gravityPitch }
    var currentMeasuredYaw: Double? { measuredYaw }

    // MARK: - Pose edits

    /// An explicit choice by a person; always accepted.
    func setPitch(degrees: Double, from provenance: MountPose.Provenance = .manual) {
        pose.pitchDegrees = degrees
        pose.pitchFrom = provenance
        pose.save()
    }

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
            attitudeRateHz: motion.attitudeRate,
            focus: {
                switch camera.focusPolicy {
                case .autoFar: return "auto (far)"
                case .locked(let p): return String(format: "locked at %.3f", p)
                }
            }())
    }

    func start() async throws {
        guard await CameraSession.requestAccess() else { throw CameraSession.CameraError.denied }
        try camera.configure()
        engine = try PerceptionEngine()
        startMotion()
        camera.onFrame = { [weak self] frame in self?.handle(frame) }
        camera.start()
    }

    func stop() {
        camera.stop()
        motion.stop()
        engine?.resetTemporalState()
        egoMotion.reset()
    }

    /// Replay owns the camera decoder and accelerator. Stop every live source,
    /// including optical flow, so a five-minute replay is the only heavy job.
    func suspendForReplay() {
        perceptionLock.withLock { perceptionEnabled = false }
        camera.stop()
        motion.stop()
        foe.reset()
        engine?.resetTemporalState()
        egoMotion.reset()
    }

    func resumeAfterReplay() {
        engine?.resetTemporalState()
        egoMotion.reset()
        startMotion()
        camera.start()
        perceptionLock.withLock { perceptionEnabled = true }
    }

    // MARK: Cold-start pose

    private func startMotion() {
        motion.onAttitude = { [weak self] attitude in
            guard let self else { return }
            self.egoMotion.append(attitude: attitude)
            self.gravityPitch = attitude.gravityPitch

            if MountPose.Provenance.gravity.mayOverwrite(self.pose.pitchFrom) {
                self.pose.pitch = self.pose.pitch * 0.98 + attitude.gravityPitch * 0.02
                self.pose.pitchFrom = .gravity
            }

            self.smoothedRoll = self.smoothedRoll == nil
                ? attitude.roll
                : self.smoothedRoll! * 0.98 + attitude.roll * 0.02
            let roll = self.smoothedRoll ?? 0
            self.measuredRoll = roll
            if abs(roll) <= Self.rollLimit {
                self.pose.roll = roll
                self.pose.rollFrom = .gravity
            } else {

                self.pose.roll = 0
                self.pose.rollFrom = .fallback
            }

            self.recorder.append(attitude: attitude)

            self.foe.append(rotationRate: attitude.rotationRate, at: attitude.timestamp)

            if let heading = attitude.cameraHeading, let fix = self.motion.latestLocation,
               fix.course >= 0, fix.courseAccuracy >= 0, fix.courseAccuracy < 5.0,
               fix.speed > 5.0 {
                self.measuredYaw = MotionSource.mountYaw(cameraHeading: heading,
                                                         course: fix.course)
            }
        }
        motion.onLocation = { [weak self] location in
            self?.egoMotion.append(location: location)
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

    private func acceptFOE(_ estimate: FocusOfExpansion.Estimate) {

        guard estimate.gates.writesToPose else { return }
        guard autoApplyFOE, estimate.converged else { return }
        guard MountPose.Provenance.estimated.mayOverwrite(pose.pitchFrom) else { return }
        pose.pitchDegrees = estimate.pitchDegrees
        pose.pitchFrom = .estimated
        pose.pitchSigmaDegrees = estimate.sigmaDegrees
    }

    private func handle(_ frame: CameraSession.Frame) {

        recorder.append(frame: frame)

        if let calibration = engine?.currentCalibration {
            foe.feed(frame: frame, calibration: calibration,
                     speed: motion.latestLocation?.speed ?? -1)
        }
        let estimate = foe.estimate
        acceptFOE(estimate)

        guard perceptionLock.withLock({ perceptionEnabled }) else { return }

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
            guard let engine else { return }
            let intrinsics = frame.intrinsics ?? Self.fallbackIntrinsics(
                width: frame.width, height: frame.height)
            let perceptionFrame = PerceptionFrame(
                pixelBuffer: frame.pixelBuffer,
                timestamp: CMTimeGetSeconds(frame.presentationTime),
                intrinsics: intrinsics,
                width: frame.width,
                height: frame.height,
                mountPose: pose,
                egoMotion: egoMotion.delta(at: CMTimeGetSeconds(frame.presentationTime)))
            let result = try engine.process(
                frame: perceptionFrame,
                configuration: PerceptionConfiguration(
                    scoreThreshold: scoreThreshold,
                    rejectImplausible: rejectImplausible))
            let crop = result.calibration.crop
            snapshot.focal = result.calibration.focal
            snapshot.calibration = result.calibration
            snapshot.focalMatched = result.calibration.focalMatched
            snapshot.visibleFraction = result.calibration.visibleVoxelFraction
            snapshot.guides = result.calibration.guides
            snapshot.cropDescription = "\(crop.width)x\(crop.height) at (\(crop.x), \(crop.y))"
            snapshot.preprocessMS = result.timings.preprocessMS
            snapshot.inferenceMS = result.timings.inferenceMS
            snapshot.detections = result.rawDetections
            snapshot.trackedObjects = result.trackedObjects
            recorder.append(result: result, at: frame.presentationTime)
        } catch {
            snapshot.note = String(describing: error)
        }

        onSnapshot?(snapshot)
    }

    private static func fallbackIntrinsics(width: Int, height: Int) -> simd_double3x3 {
        let focal = Double(width) / (2.0 * tan(60.0 * .pi / 180.0 / 2.0))
        return simd_double3x3(rows: [
            SIMD3<Double>(focal, 0, Double(width) / 2),
            SIMD3<Double>(0, focal, Double(height) / 2),
            SIMD3<Double>(0, 0, 1),
        ])
    }

}
