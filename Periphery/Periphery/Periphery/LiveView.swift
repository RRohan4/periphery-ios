//  LiveView.swift
//  Camera -> preprocess -> detect -> bird's-eye overlay, at whatever rate the
//  phone holds.
//
//  Structure: `FramePipeline` owns the camera, the preprocessor and the
//  detector, and is confined to the capture queue. Only value types cross to
//  the main actor, once per frame, for drawing. Nothing here keeps temporal
//  state -- each frame is independent, exactly as the contract says.
//
//  Calibration is cold-start only: intrinsics come live from AVFoundation, and
//  pitch comes from gravity. Gravity-referenced pitch absorbs road grade
//  (measured: a 2.18 deg mean grade walked 1.91 deg into the estimate over 40
//  seconds), so this is a starting pose, not the drive-time estimator. The
//  travel-referenced version -- rotate frame-to-frame displacement into device
//  axes, take the median above 5 m/s -- is the next piece of work.

import AVFoundation
import Combine
import CoreMotion
import Foundation
import SwiftUI
import simd

// MARK: - Pipeline

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
    /// Gravity-referenced, radians, positive nose-up. Written by CoreMotion,
    /// read on the capture queue.
    private var pitch = -3.659 * Double.pi / 180.0

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
            self.pitch = self.pitch * 0.98 + measured * 0.02
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
        snapshot.pitchDegrees = pitch * 180.0 / .pi
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
        var updated = Calibration(pitch: pitch,
                                  height: 1.20,
                                  forwardOfOrigin: 1.65,
                                  K: intrinsics,
                                  frameWidth: frame.width,
                                  frameHeight: frame.height)
        if let existing = calibration {
            let pitchMoved = abs(existing.pitch - updated.pitch) > 0.0005   // ~0.03 deg
            let opticsMoved = existing.K[0][0] != updated.K[0][0]
                || existing.frameWidth != updated.frameWidth
            if !pitchMoved && !opticsMoved {
                return existing
            }
        }
        if detector == nil {
            detector = try Detector(calibration: updated)
        } else {
            detector?.updateCalibration(updated)
        }
        updated.pitch = pitch
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

// MARK: - Camera preview

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.layer.session = session
        view.layer.videoGravity = .resizeAspect
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {}

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        override var layer: AVCaptureVideoPreviewLayer {
            super.layer as! AVCaptureVideoPreviewLayer
        }
    }
}

// MARK: - Bird's-eye overlay

/// Forward 0-40 m up the view, lateral +-10 m across it. Ego at the bottom
/// centre. Anything outside that region is not drawn, because outside it a box
/// is a decode artefact rather than a detection.
struct BEVOverlay: View {
    let detections: [Detection]

    var body: some View {
        Canvas { context, size in
            let forwardMax = Contract.forwardRange.max
            let lateralMax = Contract.lateralRange.max
            let scaleX = size.width / (2 * lateralMax)
            let scaleY = size.height / forwardMax

            func point(x forward: Double, y lateral: Double) -> CGPoint {
                CGPoint(x: size.width / 2 - lateral * scaleX,
                        y: size.height - forward * scaleY)
            }

            // Range rings every 10 m.
            for range in stride(from: 10.0, through: forwardMax, by: 10.0) {
                let y = size.height - range * scaleY
                context.stroke(Path { path in
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: size.width, y: y))
                }, with: .color(.gray.opacity(0.25)), lineWidth: 1)
                context.draw(Text("\(Int(range)) m").font(.caption2).foregroundStyle(.secondary),
                             at: CGPoint(x: 22, y: y - 8))
            }
            // Lane-width reference at +-1.8 m.
            for lateral in [-1.8, 1.8] {
                let x = size.width / 2 - lateral * scaleX
                context.stroke(Path { path in
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                }, with: .color(.gray.opacity(0.2)),
                   style: StrokeStyle(lineWidth: 1, dash: [4, 6]))
            }

            for detection in detections {
                let corners = [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]
                    .map { (along: Double, across: Double) -> CGPoint in
                        let dx = along * detection.length
                        let dy = across * detection.width
                        let c = cos(detection.yaw), s = sin(detection.yaw)
                        return point(x: detection.x + dx * c - dy * s,
                                     y: detection.y + dx * s + dy * c)
                    }
                var path = Path()
                path.addLines(corners)
                path.closeSubpath()
                let hue = Double(detection.score)
                context.fill(path, with: .color(.green.opacity(0.15 + 0.35 * hue)))
                context.stroke(path, with: .color(.green), lineWidth: 1.5)
                context.draw(
                    Text(String(format: "%.0f m", detection.x))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.primary),
                    at: point(x: detection.x, y: detection.y))
            }

            // Ego.
            let ego = point(x: 0, y: 0)
            context.fill(Path(ellipseIn: CGRect(x: ego.x - 4, y: ego.y - 4,
                                                width: 8, height: 8)),
                         with: .color(.blue))
        }
        .background(Color(white: 0.08))
    }
}

// MARK: - Screen

struct LiveView: View {
    @StateObject private var model = LiveModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    if let session = model.session {
                        CameraPreview(session: session)
                            .aspectRatio(16.0 / 9.0, contentMode: .fit)
                    } else {
                        Color.black.aspectRatio(16.0 / 9.0, contentMode: .fit)
                            .overlay(Text(model.status).font(.caption)
                                .foregroundStyle(.white))
                    }
                }
                BEVOverlay(detections: model.snapshot.detections)
                stats
            }
            .navigationTitle("Live")
            .navigationBarTitleDisplayMode(.inline)
        }
        .task { await model.start() }
        .onDisappear { model.stop() }
    }

    private var stats: some View {
        let snapshot = model.snapshot
        return VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "%.1f fps · preprocess %.1f ms · inference %.1f ms · %d boxes",
                        snapshot.fps, snapshot.preprocessMS, snapshot.inferenceMS,
                        snapshot.detections.count))
            Text(String(format: "pitch %.2f deg (gravity) · focal %.1f px · crop %@",
                        snapshot.pitchDegrees, snapshot.focal, snapshot.cropDescription))
            Text("thermal \(snapshot.thermal) · dropped \(snapshot.dropped)"
                 + (snapshot.note.isEmpty ? "" : " · \(snapshot.note)"))
                .foregroundStyle(snapshot.note.isEmpty ? Color.secondary : Color.red)
        }
        .font(.system(size: 11, design: .monospaced))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
    }
}

@MainActor
final class LiveModel: ObservableObject {
    @Published var snapshot = FramePipeline.Snapshot()
    @Published var status = "starting camera…"
    @Published var session: AVCaptureSession?

    private let pipeline = FramePipeline()
    private var started = false

    func start() async {
        guard !started else { return }
        started = true
        pipeline.onSnapshot = { [weak self] snapshot in
            Task { @MainActor in self?.snapshot = snapshot }
        }
        do {
            try await pipeline.start()
            session = pipeline.session
            status = "running"
        } catch {
            status = String(describing: error)
        }
    }

    func stop() {
        pipeline.stop()
        started = false
    }
}
