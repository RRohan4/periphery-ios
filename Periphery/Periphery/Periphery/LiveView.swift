//  LiveView.swift
//  The Live screen: camera preview, bird's-eye view, and the stats strip.
//
//  The pipeline lives in FramePipeline.swift and the renderer in WorldView.swift.
//  What is left here is SwiftUI and the main-actor model that feeds it.
//
//  Calibration is cold-start only: intrinsics come live from AVFoundation, and
//  pitch comes from gravity. Gravity-referenced pitch absorbs road grade
//  (measured: a 2.18 deg mean grade walked 1.91 deg into the estimate over 40
//  seconds), so this is a starting pose, not the drive-time estimator.

import AVFoundation
import Combine
import Foundation
import SwiftUI

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

// MARK: - Ground guides over the preview

/// The horizon and a ladder of ground-distance lines, drawn over the live
/// image.
///
/// This is the cheapest honest check in the app. Pitch error is invisible as a
/// number -- half a degree reads 40 m as 56 m and still looks like a plausible
/// angle -- but the same error puts the horizon visibly off the road. Roll tips
/// the line; yaw slides it sideways. Two of the three signs in
/// `vehicleToSensor` have no golden vector behind them, so this is how they get
/// checked: by looking.
struct GroundGuideOverlay: View {
    let guides: GroundGuides

    var body: some View {
        Canvas { context, size in
            guard guides.frameWidth > 0, guides.frameHeight > 0 else { return }
            // The preview is 16:9 and so is the buffer, shown with
            // .resizeAspect, so source pixels scale uniformly onto the view.
            let scale = size.width / Double(guides.frameWidth)
            func map(_ p: SIMD2<Double>) -> CGPoint {
                CGPoint(x: p.x * scale, y: p.y * scale)
            }
            if let horizon = guides.horizon {
                var path = Path()
                path.move(to: map(horizon.a))
                path.addLine(to: map(horizon.b))
                context.stroke(path, with: .color(.cyan.opacity(0.8)),
                               style: StrokeStyle(lineWidth: 1.5, dash: [8, 5]))
            }
            for segment in guides.ranges {
                var path = Path()
                path.move(to: map(segment.a))
                path.addLine(to: map(segment.b))
                context.stroke(path, with: .color(.yellow.opacity(0.55)), lineWidth: 1)
                context.draw(Text("\(Int(segment.range))")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.yellow.opacity(0.9)),
                             at: CGPoint(x: map(segment.b).x + 4, y: map(segment.b).y),
                             anchor: .leading)
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Screen

struct LiveView: View {
    @ObservedObject private var model = LiveSession.shared

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    if let session = model.session {
                        CameraPreview(session: session)
                            .aspectRatio(16.0 / 9.0, contentMode: .fit)
                            .overlay {
                                if model.showGuides {
                                    GroundGuideOverlay(guides: model.snapshot.guides)
                                }
                            }
                    } else {
                        Color.black.aspectRatio(16.0 / 9.0, contentMode: .fit)
                            .overlay(Text(model.status).font(.caption)
                                .foregroundStyle(.white))
                    }
                }
                WorldView(detections: model.snapshot.detections,
                          focal: model.snapshot.focal > 0
                              ? model.snapshot.focal : Contract.trainedFocal,
                          egoSpeed: model.snapshot.speed)
                stats
            }
            .navigationTitle("Live")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                Button {
                    model.showGuides.toggle()
                } label: {
                    Image(systemName: model.showGuides
                          ? "grid.circle.fill" : "grid.circle")
                }
            }
        }
        .task { await model.ensureStarted() }
    }

    private var stats: some View {
        let snapshot = model.snapshot
        return VStack(alignment: .leading, spacing: 2) {
            Text(String(format: "%.1f fps · preprocess %.1f ms · inference %.1f ms · %d boxes",
                        snapshot.fps, snapshot.preprocessMS, snapshot.inferenceMS,
                        snapshot.detections.count))
            Text(String(format: "pitch %+.2f (%@) · roll %+.2f (%@) · yaw %@",
                        snapshot.pose.pitchDegrees, snapshot.pose.pitchFrom.label,
                        snapshot.measuredRollDegrees, snapshot.pose.rollFrom.label,
                        snapshot.measuredYawDegrees.map { String(format: "%+.2f", $0) } ?? "--"))
            Text(String(format: "focal %.1f px · crop %@ · %@ · %@",
                        snapshot.focal, snapshot.cropDescription,
                        snapshot.speed >= 0 ? String(format: "%.1f m/s", snapshot.speed) : "no fix",
                        snapshot.relativeAltitude.map { String(format: "%+.2f m baro", $0) } ?? "no baro"))
            Text("thermal \(snapshot.thermal) · dropped \(snapshot.dropped)"
                 + (snapshot.note.isEmpty ? "" : " · \(snapshot.note)"))
                .foregroundStyle(snapshot.note.isEmpty ? Color.secondary : Color.red)
            if !snapshot.mountWarning.isEmpty {
                Text(snapshot.mountWarning).foregroundStyle(Color.orange)
            }
        }
        .font(.system(size: 11, design: .monospaced))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
    }
}

/// One camera, one pipeline, one recorder.
///
/// Live, Record and Calibrate are three views onto the same running session,
/// so it cannot live inside any one of them -- and the camera cannot be opened
/// twice. Recording in particular has to survive leaving the Live tab, which is
/// why nothing here stops on `onDisappear`.
@MainActor
final class LiveSession: ObservableObject {
    static let shared = LiveSession()

    @Published var snapshot = FramePipeline.Snapshot()
    @Published var status = "idle"
    @Published var session: AVCaptureSession?
    /// Horizon and ground lines over the preview. On by default: they are how
    /// a bad pose becomes visible instead of silently absorbed.
    @Published var showGuides = true

    let pipeline = FramePipeline()
    private var starting = false
    private var started = false

    private init() {}

    /// Idempotent and safe to call from any tab.
    func ensureStarted() async {
        guard !started, !starting else { return }
        starting = true
        defer { starting = false }
        status = "starting camera…"
        pipeline.onSnapshot = { [weak self] snapshot in
            Task { @MainActor in self?.snapshot = snapshot }
        }
        do {
            try await pipeline.start()
            session = pipeline.session
            status = "running"
            started = true
        } catch {
            status = String(describing: error)
        }
    }

    /// Only on the way out of the app. A drive in progress outlives any tab.
    func stop() {
        guard started else { return }
        pipeline.stop()
        started = false
        session = nil
    }
}
