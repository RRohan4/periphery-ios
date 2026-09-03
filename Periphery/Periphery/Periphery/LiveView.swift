//  LiveView.swift
//  The Live screen: camera preview, bird's-eye view, and the stats strip.
//
//  The pipeline lives in FramePipeline.swift and the renderer in WorldView.swift.
//  What is left here is SwiftUI and the main-actor model that feeds it.
//
//  LANDSCAPE. The app is landscape-locked and this screen is split down the
//  middle: camera on the left, bird's-eye on the right. That is not a taste
//  decision. The capture buffer is landscape however the phone is held, so a
//  portrait mount does not rotate the image, it lays the road sideways across a
//  crop computed for the other axis -- see MotionSource.cameraRoll. A screen
//  that stacks a 16:9 preview above a world view in portrait wastes most of the
//  glass on letterboxing and shows both panes too small to read while driving.
//
//  Intrinsics come live from AVFoundation. Pitch starts from gravity, which
//  absorbs road grade one-for-one (2.45 deg p95 over 237 segments), and is
//  superseded by the camera estimator in FocusOfExpansion once it converges.

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
        // Zero rotation relative to the sensor's native readout, i.e. show the
        // buffer exactly as it arrives. GroundGuideOverlay draws in SOURCE
        // PIXELS and maps them with a single uniform scale, so any rotation
        // here would silently slide the horizon off the road while leaving the
        // number in the stats strip looking fine.
        pinRotation(view)
        return view
    }

    // The layer's connection is not always established by the time makeUIView
    // returns, so pin it again once the view is in the hierarchy.
    func updateUIView(_ uiView: PreviewView, context: Context) { pinRotation(uiView) }

    private func pinRotation(_ view: PreviewView) {
        guard let connection = view.layer.connection,
              connection.isVideoRotationAngleSupported(0),
              connection.videoRotationAngle != 0 else { return }
        connection.videoRotationAngle = 0
    }

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
        // No NavigationStack: a title bar costs a fifth of the height in
        // landscape and this screen has nothing to navigate to.
        HStack(spacing: 1) {
            cameraPane
            worldPane
        }
        .background(Color.black)
        .overlay(alignment: .bottomLeading) { stats }
        .overlay(alignment: .topTrailing) { guidesToggle }
        .statusBarHidden()
        .task { await model.ensureStarted() }
    }

    /// Left half. `.fill` rather than `.fit`: at a 50/50 split the pane is
    /// roughly square and fitting a 16:9 preview into it would letterbox away
    /// most of the image. The guides are drawn in the same geometry, so they
    /// stay registered to the road either way.
    private var cameraPane: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black
                if let session = model.session {
                    CameraPreview(session: session)
                        .aspectRatio(16.0 / 9.0, contentMode: .fill)
                        // Attached BEFORE the frame, so the overlay inherits
                        // the preview's aspect-filled bounds and the two are
                        // clipped together. Attach it after and the guides get
                        // the pane's geometry instead, which is a different
                        // scale and slides the horizon off the road.
                        .overlay {
                            if model.showGuides {
                                GroundGuideOverlay(guides: model.snapshot.guides)
                            }
                        }
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                } else {
                    Text(model.status)
                        .font(.caption)
                        .foregroundStyle(.white)
                }
            }
        }
    }

    /// Right half.
    private var worldPane: some View {
        WorldView(detections: model.snapshot.detections,
                  focal: model.snapshot.focal > 0
                      ? model.snapshot.focal : Contract.trainedFocal,
                  egoSpeed: model.snapshot.speed)
    }

    private var guidesToggle: some View {
        Button {
            model.showGuides.toggle()
        } label: {
            Image(systemName: model.showGuides ? "grid.circle.fill" : "grid.circle")
                .font(.title3)
                .foregroundStyle(.white)
                .padding(8)
                .background(.black.opacity(0.45), in: Circle())
        }
        .padding(10)
    }

    private var stats: some View {
        let snapshot = model.snapshot
        return VStack(alignment: .leading, spacing: 1) {
            Text(String(format: "%.1f fps · pre %.1f · inf %.1f ms · %d boxes @ %.2f",
                        snapshot.fps, snapshot.preprocessMS, snapshot.inferenceMS,
                        snapshot.detections.count, snapshot.scoreThreshold))
            Text(String(format: "pitch %+.2f (%@) · roll %+.2f · yaw %+.2f (%@)",
                        snapshot.pose.pitchDegrees, snapshot.pose.pitchFrom.label,
                        snapshot.measuredRollDegrees,
                        snapshot.pose.yawDegrees, snapshot.pose.yawFrom.label))
            foeLine(snapshot.foe)
            Text(String(format: "focal %.0f px · lens %.2f %@ · %@ · %@ · drop %d",
                        snapshot.focal, snapshot.lensPosition,
                        snapshot.focusHunting ? "hunting"
                            : (snapshot.focusLocked ? "locked" : "auto"),
                        snapshot.speed >= 0 ? String(format: "%.1f m/s", snapshot.speed) : "no fix",
                        snapshot.thermal, snapshot.dropped))
                .foregroundStyle(snapshot.focusHunting ? Color.orange
                                 : Color.white.opacity(0.92))
            if !snapshot.note.isEmpty {
                Text(snapshot.note).foregroundStyle(Color.red)
            }
            if !snapshot.mountWarning.isEmpty {
                Text(snapshot.mountWarning).foregroundStyle(Color.orange)
            }
        }
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(.white.opacity(0.92))
        .padding(6)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 6))
        .padding(8)
    }

    /// The camera estimator, in one line. When it is not accumulating, the
    /// reason is the point -- a night drive and a slow street look different,
    /// and both look different from a bug.
    private func foeLine(_ foe: FocusOfExpansion.Estimate) -> some View {
        let text: String
        let colour: Color
        if foe.reportable {
            text = String(format: "camera pitch %+.2f ±%.2f · %d/%d · %d inliers",
                          foe.pitchDegrees, foe.sigmaDegrees,
                          foe.samples, foe.windowSize, foe.inliers)
            colour = foe.converged ? .green : .yellow
        } else {
            text = "camera pitch — \(foe.gate.isEmpty ? "warming up" : foe.gate)"
                 + (foe.samples > 0 ? " · \(foe.samples) samples" : "")
            colour = .white.opacity(0.6)
        }
        return Text(text).foregroundStyle(colour)
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
