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
