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
//  Focus is locked at infinity as well. Autofocus moves the lens, which moves
//  the focal length, which silently rescales every range estimate -- the same
//  failure mode as feeding the wrong focal in the first place.

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
        /// Exposure time in seconds.
        ///
        /// Recorded because it is the direct measure of MOTION BLUR, and blur is
        /// what starves optical flow: at 30 m/s a 1/60 s exposure smears a point
        /// 20 m away across several pixels, which is the same order as the flow
        /// being measured. Without this, a drive where the estimator quietly
        /// degraded at dusk is indistinguishable from one where the maths is
        /// wrong.
        let exposureSeconds: Double
        /// Sensor gain. High ISO means a noisy image, which is the other way
        /// flow quality dies.
        let iso: Double
        /// Where the lens is, 0-1. Constant if the focus lock is holding; if it
        /// moves, the focal length moved with it and every range in that stretch
        /// carries a scale error.
        let lensPosition: Double
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

    let session = AVCaptureSession()
    /// Held so per-frame exposure, gain and lens position can be stamped onto
    /// each Frame. Read-only after configure().
    private var device: AVCaptureDevice?
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.periphery.camera", qos: .userInitiated)

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
        if device.isFocusModeSupported(.locked) {
            // 1.0 is the far end of the lens range.
            device.setFocusModeLocked(lensPosition: 1.0)
        }
        if device.isExposureModeSupported(.continuousAutoExposure) {
            device.exposureMode = .continuousAutoExposure
        }
        device.unlockForConfiguration()
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
        let exposure = device.map { CMTimeGetSeconds($0.exposureDuration) } ?? 0
        onFrame?(Frame(pixelBuffer: pixelBuffer,
                       intrinsics: Self.intrinsics(from: sampleBuffer),
                       width: CVPixelBufferGetWidth(pixelBuffer),
                       height: CVPixelBufferGetHeight(pixelBuffer),
                       sampleBuffer: sampleBuffer,
                       presentationTime: CMSampleBufferGetPresentationTimeStamp(sampleBuffer),
                       exposureSeconds: exposure.isFinite ? exposure : 0,
                       iso: Double(device?.iso ?? 0),
                       lensPosition: Double(device?.lensPosition ?? 0)))
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
