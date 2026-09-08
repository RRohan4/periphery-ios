
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

    enum FocusPolicy: Equatable {
        case autoFar
        case locked(Float)
    }

    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.periphery.camera", qos: .userInitiated)
    /// Held so the focus policy can be applied and read back.
    private var device: AVCaptureDevice?

    /// Called on the capture queue, not the main thread.
    var onFrame: ((Frame) -> Void)?

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
            }
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
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

    func apply(_ policy: FocusPolicy) {
        guard let device, (try? device.lockForConfiguration()) != nil else { return }
        defer { device.unlockForConfiguration() }
        switch policy {
        case .autoFar:

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
        policy.save()
    }

    @discardableResult
    func lockFocusHere() -> Float {
        guard let device, (try? device.lockForConfiguration()) != nil else { return 0 }
        defer { device.unlockForConfiguration() }
        guard device.isFocusModeSupported(.locked) else { return device.lensPosition }
        device.setFocusModeLocked(lensPosition: AVCaptureDevice.currentLensPosition)
        let position = device.lensPosition
        FocusPolicy.locked(position).save()
        return position
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

// MARK: - Persistence

extension CameraSession.FocusPolicy {
    private static let key = "CameraSession.FocusPolicy.v1"

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
}
