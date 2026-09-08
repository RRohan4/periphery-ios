// Live camera screen. FramePipeline supplies snapshots; this file owns the
// SwiftUI layout and the main-actor model.

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
        view.layer.videoGravity = .resizeAspectFill

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

// MARK: - Screen

struct LiveView: View {
    @ObservedObject private var model = LiveSession.shared

    var body: some View {
        PerceptionSplitView(objects: model.snapshot.trackedObjects,
                            calibration: model.snapshot.calibration,
                            egoSpeed: model.snapshot.speed) {
            ZStack {
                Color.black
                if let session = model.session {
                    CameraPreview(session: session)
                } else {
                    Text(model.status).font(.caption).foregroundStyle(.white)
                }
            }
        }
        .statusBarHidden()
        .task { await model.ensureStarted() }
    }
}

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

    /// Only on the way out of the app; the camera outlives any tab switch.
    func stop() {
        guard started else { return }
        pipeline.stop()
        started = false
        session = nil
    }
}
