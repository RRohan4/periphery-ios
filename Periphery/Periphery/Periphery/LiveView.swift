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
