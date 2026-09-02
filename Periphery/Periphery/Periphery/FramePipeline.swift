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
import CoreMotion
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
    }

    private let camera = CameraSession()
    private let motion = CMMotionManager()
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

    var onSnapshot: ((Snapshot) -> Void)?
    var session: AVCaptureSession { camera.session }

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
        motion.stopDeviceMotionUpdates()
    }

    // MARK: Cold-start pose

    private func startMotion() {
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 1.0 / 30.0
        motion.startDeviceMotionUpdates(to: .main) { [weak self] motion, _ in
            guard let self, let gravity = motion?.gravity else { return }
            // The rear camera looks along -z in device axes, so the camera's
            // elevation above horizontal is asin(g.z). Independent of roll,
            // which is why a landscape or portrait mount both work.
            let measured = asin(max(-1.0, min(1.0, gravity.z)))
            // Heavy smoothing: this is the seconds-to-minutes timescale, and
            // per-frame rattle belongs to the gyro, not here.
            self.pose.pitch = self.pose.pitch * 0.98 + measured * 0.02
            self.pose.pitchFrom = .gravity
        }
    }

    // MARK: Per frame

    private func handle(_ frame: CameraSession.Frame) {
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
        snapshot.thermal = Benchmark.describe(ProcessInfo.processInfo.thermalState)

        do {
            let calibration = try calibrate(with: frame)
            let crop = calibration.focalMatchedCrop()
            snapshot.focal = calibration.achievedFocal(crop)
            snapshot.cropDescription = "\(crop.width)x\(crop.height) at (\(crop.x), \(crop.y))"

            guard let preprocessor, let detector else { return }
            var mark = DispatchTime.now()
            let input = try preprocessor.fill(from: frame.pixelBuffer, crop: crop)
            snapshot.preprocessMS = Self.ms(since: mark)

            mark = DispatchTime.now()
            snapshot.detections = try detector.detect(image: input)
            snapshot.inferenceMS = Self.ms(since: mark)
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
