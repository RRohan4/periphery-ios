//  CameraSession.swift
//  Live frames, with the two settings this pipeline cannot work without.
//
//  1. Video stabilisation OFF. EIS and OIS change per-frame geometry without
//     reporting it, which breaks both the fixed intrinsics the focal-matched
//     crop is computed from and the known extrinsics the LUT is built from.
//     Apple confirms the incompatibility implicitly: intrinsic matrix delivery
//     is unavailable while stabilisation is on.
//  2. Intrinsic matrix delivery ON. The crop that lands on the trained focal is
//     computed from the live K, not guessed.
//
//  FOCUS. Autofocus moves the lens, which moves the focal length, which
//  silently rescales every range estimate. So the lens must not wander during a
//  drive. That much was right.
//
//  What was wrong was the number. This used to hard-code
//  `setFocusModeLocked(lensPosition: 1.0)` with the comment "1.0 is the far end
//  of the lens range". It is -- but the far MECHANICAL end is not the same as
//  infinity FOCUS. Voice-coil actuators are built with overtravel past infinity
//  so that infinity stays reachable across temperature and unit-to-unit spread,
//  so parking at 1.0 lands past focus and everything distant goes soft. There is
//  no portable constant for infinity: it is a different number on every unit.
//
//  A blurry frame is not a cosmetic problem here. Optical flow is differences of
//  local intensity, and blur is a low-pass filter -- it removes exactly the
//  high-frequency content the flow field is computed from. Soft focus degrades
//  the pitch estimator far more than it degrades the detector.
//
//  So focus is now FOUND rather than guessed: autofocus, restricted to the far
//  range, until someone locks it. `Calibration` shows the lens position and
//  whether it is locked, because an unlocked lens is a real caveat on every
//  range and must not be silent.

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import simd

final class CameraSession: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    struct Frame {
        let pixelBuffer: CVPixelBuffer
        /// Delivered per frame by AVFoundation, for the buffer as it arrives.
        let intrinsics: simd_double3x3?
        let width: Int
        let height: Int
        /// The buffer itself, kept so the recorder can hand it to an
        /// AVAssetWriter without a second capture path.
        let sampleBuffer: CMSampleBuffer
        /// Host time clock -- the same mach_absolute_time domain CoreMotion and
        /// CMAltimeter stamp their samples in, so video, IMU and barometer share
        /// a timeline with no conversion. CoreLocation does not; see
        /// MotionSource's clock anchor.
        let presentationTime: CMTime
    }

    enum CameraError: Error, CustomStringConvertible {
        case noCamera
        case denied
        case cannotAdd(String)

        var description: String {
            switch self {
            case .noCamera: return "no back wide-angle camera"
            case .denied: return "camera permission denied"
            case .cannotAdd(let what): return "cannot add \(what) to the session"
            }
        }
    }

    /// Where the lens sits, and whether it is allowed to move.
    ///
    /// `autoFar` is the honest default: the correct lens position for infinity
    /// is device-specific and cannot be hard-coded, so let the camera find it.
    /// `locked` is what a measured drive wants, at a position someone chose by
    /// looking at the picture.
    enum FocusPolicy: Equatable {
        case autoFar
        case locked(Float)
    }

    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.periphery.camera", qos: .userInitiated)
    private var device: AVCaptureDevice?

    /// Called on the capture queue, not the main thread.
    var onFrame: ((Frame) -> Void)?
    private(set) var stabilizationDisabled = false
    private(set) var intrinsicsAvailable = false

    static func requestAccess() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .video)
        default: return false
        }
    }

    func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.sessionPreset = .hd1920x1080

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                   for: .video, position: .back) else {
            throw CameraError.noCamera
        }
        self.device = device
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.cannotAdd("camera input") }
        session.addInput(input)

        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        // Drop frames rather than queue them: a stale frame is worse than no
        // frame when the output is a collision warning.
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw CameraError.cannotAdd("video output") }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            if connection.isVideoStabilizationSupported {
                connection.preferredVideoStabilizationMode = .off
                stabilizationDisabled = true
            }
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
                intrinsicsAvailable = true
            }
        }

        try? device.lockForConfiguration()
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        device.unlockForConfiguration()

        apply(FocusPolicy.load())
    }

    // MARK: - Focus

    /// Where the lens is now, 0 (closest) to 1 (mechanically furthest).
    var lensPosition: Float { device?.lensPosition ?? 0 }
    var focusIsLocked: Bool { device?.focusMode == .locked }
    /// True while the camera is still hunting; a frame captured now may be soft.
    var isAdjustingFocus: Bool { device?.isAdjustingFocus ?? false }

    private(set) var focusPolicy: FocusPolicy = .autoFar

    func apply(_ policy: FocusPolicy) {
        guard let device, (try? device.lockForConfiguration()) != nil else { return }
        defer { device.unlockForConfiguration() }
        switch policy {
        case .autoFar:
            // Restricting the range keeps the camera from racking to macro on
            // the dashboard or a raindrop on the glass, which is the failure
            // this restriction exists for.
            if device.isAutoFocusRangeRestrictionSupported {
                device.autoFocusRangeRestriction = .far
            }
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
        case .locked(let position):
            guard device.isFocusModeSupported(.locked) else { return }
            device.setFocusModeLocked(lensPosition: min(max(position, 0), 1))
        }
        focusPolicy = policy
        policy.save()
    }

    /// Freeze the lens exactly where autofocus has just put it.
    ///
    /// This is the whole point: the right lens position for infinity is a
    /// property of the individual camera, so the only reliable way to get it is
    /// to let autofocus find it on a distant scene and then stop the lens
    /// moving. `currentLensPosition` is Apple's sentinel for "lock here",
    /// which avoids a read-then-write race against a lens still in motion.
    @discardableResult
    func lockFocusHere() -> Float {
        guard let device, (try? device.lockForConfiguration()) != nil else { return 0 }
        defer { device.unlockForConfiguration() }
        guard device.isFocusModeSupported(.locked) else { return device.lensPosition }
        device.setFocusModeLocked(lensPosition: AVCaptureDevice.currentLensPosition)
        let position = device.lensPosition
        focusPolicy = .locked(position)
        focusPolicy.save()
        return position
    }
}

// MARK: - Persistence

extension CameraSession.FocusPolicy {
    private static let key = "CameraSession.FocusPolicy.v1"

    /// A locked position belongs to one physical phone, so it survives a
    /// restart. It does NOT survive a reset, and it should not: a value carried
    /// over from another unit would be worse than none.
    static func load(from defaults: UserDefaults = .standard) -> Self {
        guard let value = defaults.object(forKey: key) as? Double else { return .autoFar }
        return .locked(Float(value))
    }

    func save(to defaults: UserDefaults = .standard) {
        switch self {
        case .autoFar: defaults.removeObject(forKey: Self.key)
        case .locked(let position): defaults.set(Double(position), forKey: Self.key)
        }
    }

    func start() {
        queue.async { [session] in
            if !session.isRunning { session.startRunning() }
        }
    }

    func stop() {
        queue.async { [session] in
            if session.isRunning { session.stopRunning() }
        }
    }

    // MARK: - Delegate

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(Frame(pixelBuffer: pixelBuffer,
                       intrinsics: Self.intrinsics(from: sampleBuffer),
                       width: CVPixelBufferGetWidth(pixelBuffer),
                       height: CVPixelBufferGetHeight(pixelBuffer),
                       sampleBuffer: sampleBuffer,
                       presentationTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer)))
    }

    /// The per-frame intrinsic matrix AVFoundation attaches when delivery is
    /// enabled. Column-major float3x3, for the buffer's own pixel dimensions.
    private static func intrinsics(from sampleBuffer: CMSampleBuffer) -> simd_double3x3? {
        guard let attachment = CMGetAttachment(
            sampleBuffer,
            key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
            attachmentModeOut: nil) as? Data,
              attachment.count >= MemoryLayout<matrix_float3x3>.size else { return nil }
        let matrix: matrix_float3x3 = attachment.withUnsafeBytes {
            $0.loadUnaligned(as: matrix_float3x3.self)
        }
        return simd_double3x3(columns: (
            SIMD3<Double>(Double(matrix.columns.0.x), Double(matrix.columns.0.y), Double(matrix.columns.0.z)),
            SIMD3<Double>(Double(matrix.columns.1.x), Double(matrix.columns.1.y), Double(matrix.columns.1.z)),
            SIMD3<Double>(Double(matrix.columns.2.x), Double(matrix.columns.2.y), Double(matrix.columns.2.z))
        ))
    }
}
