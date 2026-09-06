import Combine
import AVFoundation
import os
import SwiftUI
import UIKit

struct RecordedVideoView: UIViewRepresentable {
    let player: AVPlayer

    func makeUIView(context: Context) -> PlayerView {
        let view = PlayerView()
        view.layer.player = player
        view.layer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PlayerView, context: Context) { view.layer.player = player }

    final class PlayerView: UIView {
        override class var layerClass: AnyClass { AVPlayerLayer.self }
        override var layer: AVPlayerLayer { super.layer as! AVPlayerLayer }
    }
}

struct ReplayView: View {
    @StateObject private var model = ReplayModel()
    /// Deliberately local, brittle demo controls. They affect only the replay
    /// drawing and never rewrite the recorded sidecar or touch inference.
    @State private var demoHeight = 1.50
    @State private var cameraRight = 0.50
    @State private var controlsVisible = true

    @ViewBuilder
    var body: some View {
        if let displayFrame = demoDisplayFrame, !model.processing {
            ZStack {
                PerceptionSplitView(objects: displayFrame.objects,
                                    calibration: displayFrame.calibration,
                                    egoSpeed: displayFrame.egoSpeed,
                                    cameraRight: cameraRight) {
                    RecordedVideoView(player: model.player)
                }
                .contentShape(Rectangle())
                .onTapGesture {
                    withAnimation(.easeOut(duration: 0.2)) {
                        controlsVisible.toggle()
                    }
                }
                if controlsVisible {
                    replayControls(displayFrame)
                        .transition(.opacity)
                }
            }
            .statusBarHidden()
        } else {
            replayBrowser
        }
    }

    private var demoDisplayFrame: ReplayDisplayFrame? {
        model.currentDisplayFrame?.demoAdjusted(height: demoHeight,
                                                cameraRight: cameraRight)
    }

    private var replayBrowser: some View {
        NavigationStack {
            List {
                Section("Local drives") {
                    ForEach(model.drives, id: \.self) { drive in
                        Button(drive.lastPathComponent) { model.select(drive) }
                    }
                }
                if let drive = model.selected {
                    Section("Replay") {
                        Text(drive.lastPathComponent).font(.system(.caption, design: .monospaced))
                        if model.processing {
                            ProgressView(value: model.progress)
                            Text("\(model.processedFrames) / \(model.totalFrames) frames · "
                                 + String(format: "%.1f%%", model.progress * 100))
                                .font(.system(.caption, design: .monospaced))
                            Text(model.rateLine).font(.system(.caption2, design: .monospaced))
                            Text(model.pipelineLine).font(.system(.caption2, design: .monospaced))
                            Text(model.stageLine).font(.system(.caption2, design: .monospaced))
                            Text("thermal · \(model.thermal)")
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundStyle(model.thermal == "nominal"
                                    ? Color.secondary : Color.orange)
                            Text(model.memoryLine)
                                .font(.system(.caption2, design: .monospaced))
                            Text("Keep Replay open. Auto-lock is disabled during this run.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Button("Cancel", role: .destructive) { model.cancel() }
                        } else {
                            Button("Reprocess entire drive") { model.reprocess() }
                        }
                        if !model.frames.isEmpty {
                            HStack {
                                Button("◀︎") { model.step(-1) }
                                Button(model.playing ? "Pause" : "Play") { model.togglePlay() }
                                Button("▶︎") { model.step(1) }
                            }
                            Slider(value: $model.position, in: 0...Double(model.frames.count - 1), step: 1)
                            let frame = model.frames[Int(model.position)]
                            LabeledContent("frame", value: "\(frame.index + 1)/\(model.frames.count)")
                            LabeledContent("raw / tracked", value: "\(frame.rawCount) / \(frame.trackCount)")
                        }
                    }
                }
                if let message = model.message { Text(message).font(.caption).foregroundStyle(.secondary) }
            }
            .navigationTitle("Replay")
            .task { model.refresh() }
        }
    }

    private func replayControls(_ frame: ReplayDisplayFrame) -> some View {
        VStack {
            HStack {
                Button("Drives") { model.closePlayback() }
                Spacer()
                Text("frame \(frame.summary.index + 1)/\(model.displayFrames.count)")
                    .font(.system(size: 10, design: .monospaced))
                Button("Reprocess") { model.reprocess() }
            }
            Spacer()
            demoCalibrationControls
            HStack {
                Button("◀︎") { model.step(-1) }
                Button(model.playing ? "Pause" : "Play") { model.togglePlay() }
                Button("▶︎") { model.step(1) }
                Slider(value: Binding(
                    get: { model.position },
                    set: { value in model.position = value; model.seekVideo() }),
                       in: 0...Double(max(model.displayFrames.count - 1, 0)), step: 1)
            }
        }
        .buttonStyle(.borderedProminent)
        .font(.caption)
        .padding(10)
    }

    private var demoCalibrationControls: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("DEMO CALIBRATION — REPLAY ONLY")
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.65))
            HStack(spacing: 8) {
                Text(String(format: "height %.2f m", demoHeight))
                    .frame(width: 112, alignment: .leading)
                Slider(value: $demoHeight, in: 1.0...2.0, step: 0.01)
            }
            HStack(spacing: 8) {
                Text(String(format: "camera right %.2f m", cameraRight))
                    .frame(width: 112, alignment: .leading)
                Slider(value: $cameraRight, in: 0.0...1.0, step: 0.05)
            }
        }
        .font(.system(size: 10, design: .monospaced))
        .foregroundStyle(.white.opacity(0.9))
        .padding(8)
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 6))
    }
}

private extension ReplayDisplayFrame {
    /// Apply the user's temporary demo geometry to an already-produced replay.
    /// This mirrors the offline height rescale, then translates camera-relative
    /// lateral coordinates into the vehicle-centreline frame. It is intentionally
    /// not part of ReplayProcessor or the perception pipeline.
    func demoAdjusted(height: Double, cameraRight: Double) -> ReplayDisplayFrame {
        guard let source = calibration else { return self }
        let sourceHeight = source.calibration.height
        let scale = sourceHeight > 1e-6 ? height / sourceHeight : 1.0

        let adjustedObjects = objects.map { original -> TrackedVehicle in
            var object = original
            object.x *= scale
            object.y = object.y * scale - cameraRight
            object.z *= scale
            object.length *= scale
            object.width *= scale
            object.height *= scale
            if let velocity = object.velocity {
                object.velocity = velocity * scale
            }
            object.trail = object.trail.map {
                VehicleTrailPoint(timestamp: $0.timestamp,
                                  x: $0.x * scale,
                                  y: $0.y * scale - cameraRight)
            }
            return object
        }

        var pose = source.calibration.pose
        pose.height = height
        let correctedCalibration = Calibration(
            pose: pose,
            K: source.calibration.K,
            frameWidth: source.calibration.frameWidth,
            frameHeight: source.calibration.frameHeight)
        var adjustedSnapshot = source
        adjustedSnapshot.calibration = correctedCalibration
        adjustedSnapshot.guides = correctedCalibration.groundGuides()

        return ReplayDisplayFrame(summary: summary,
                                  objects: adjustedObjects,
                                  calibration: adjustedSnapshot,
                                  egoSpeed: egoSpeed)
    }
}

@MainActor
final class ReplayModel: ObservableObject {
    @Published var drives: [URL] = []
    @Published var selected: URL?
    @Published var frames: [ReplayFrameSummary] = []
    @Published var displayFrames: [ReplayDisplayFrame] = []
    @Published var position = 0.0
    @Published var progress = 0.0
    @Published var processing = false
    @Published var processedFrames = 0
    @Published var totalFrames = 0
    @Published var playing = false
    @Published var message: String?
    @Published var rateLine = "starting…"
    @Published var pipelineLine = "waiting for first frame…"
    /// The per-stage split of `inferenceMS`. Separate line because it is the one
    /// that says WHICH stage is slow, which is the whole point of showing it.
    @Published var stageLine = "—"
    @Published var thermal = "nominal"
    @Published var memoryLine = "available memory —"
    private var processor: ReplayProcessor?
    private var playTask: Task<Void, Never>?
    private var replayStarted = Date()
    private var priorIdleTimerDisabled = false
    let player = AVPlayer()

    var currentDisplayFrame: ReplayDisplayFrame? {
        guard !displayFrames.isEmpty else { return nil }
        return displayFrames[min(max(Int(position), 0), displayFrames.count - 1)]
    }

    func refresh() { drives = DriveRecorder.sessions() }
    func select(_ drive: URL) {
        selected = drive; frames = ReplayProcessor.loadSummaries(drive: drive)
        displayFrames = ReplayProcessor.loadDisplayFrames(drive: drive)
        player.replaceCurrentItem(with: AVPlayerItem(
            url: drive.appendingPathComponent("video.mov")))
        position = 0; message = frames.isEmpty ? "No replay sidecar yet." : "Loaded replay-safety40-v1.jsonl"
        seekVideo()
    }

    func reprocess() {
        guard let drive = selected, !processing else { return }
        guard !LiveSession.shared.pipeline.recorder.isRecording else {
            message = "Stop recording before Replay."; return
        }
        let worker = ReplayProcessor(); processor = worker; processing = true; progress = 0
        processedFrames = 0; totalFrames = 0
        replayStarted = Date(); rateLine = "starting…"
        pipelineLine = "waiting for first frame…"; stageLine = "—"; thermal = "nominal"
        memoryLine = "available memory —"
        LiveSession.shared.pipeline.suspendForReplay()
        priorIdleTimerDisabled = UIApplication.shared.isIdleTimerDisabled
        UIApplication.shared.isIdleTimerDisabled = true
        message = "Running the shared perception engine…"
        Task.detached {
            do {
                let result = try await worker.process(drive: drive) { update in
                    Task { @MainActor in
                        self.processedFrames = update.completed; self.totalFrames = update.total
                        self.progress = Double(update.completed) / Double(max(update.total, 1))
                        let elapsed = max(Date().timeIntervalSince(self.replayStarted), 0.001)
                        let fps = Double(update.completed) / elapsed
                        let remaining = fps > 0
                            ? Double(max(update.total - update.completed, 0)) / fps : 0
                        self.rateLine = String(format: "avg %.1f fps · elapsed %.0fs · ETA %.0fs",
                                               fps, elapsed, remaining)
                        self.pipelineLine = String(
                            format: "latest raw %d · tracked %d · pre %.1f ms · inference %.1f ms",
                            update.latest.rawCount, update.latest.trackCount,
                            update.preprocessMS, update.inferenceMS)
                        // The split inference is actually made of, averaged over the
                        // run so far rather than sampled from one frame. `other` is
                        // the part of inference no stage claims -- the image
                        // precision conversion ahead of the first stage mark.
                        let t = update.mean
                        self.stageLine = String(
                            format: "mean · backbone %.1f · gather %.1f · head %.1f "
                                  + "· decode %.1f · other %.1f ms",
                            t.backboneMS, t.gatherMS, t.headMS, t.decodeMS, t.unaccountedMS)
                        self.thermal = Benchmark.describe(ProcessInfo.processInfo.thermalState)
                        self.memoryLine = String(
                            format: "available memory %.0f MB",
                            Double(os_proc_available_memory()) / 1_048_576.0)
                    }
                }
                await MainActor.run {
                    self.frames = result; self.position = 0; self.processing = false
                    self.displayFrames = ReplayProcessor.loadDisplayFrames(drive: drive)
                    self.finishProcessing()
                    self.message = "Replay complete; versioned sidecar saved."
                }
            } catch {
                await MainActor.run {
                    self.processing = false; self.message = String(describing: error)
                    self.finishProcessing()
                }
            }
        }
    }

    private func finishProcessing() {
        LiveSession.shared.pipeline.resumeAfterReplay()
        UIApplication.shared.isIdleTimerDisabled = priorIdleTimerDisabled
    }

    func cancel() { processor?.cancel() }
    func step(_ amount: Int) {
        playing = false; playTask?.cancel()
        player.pause()
        position = min(max(position + Double(amount), 0),
                       Double(max(displayFrames.count - 1, 0)))
        seekVideo()
    }
    func togglePlay() {
        playing.toggle(); playTask?.cancel()
        guard playing else { player.pause(); return }
        player.play()
        playTask = Task {
            while !Task.isCancelled, playing, Int(position) < displayFrames.count - 1 {
                try? await Task.sleep(for: .milliseconds(33))
                guard let first = displayFrames.first else { break }
                let target = first.summary.timestamp + player.currentTime().seconds
                position = Double(nearestFrameIndex(to: target))
            }
            player.pause(); playing = false
        }
    }

    func seekVideo() {
        guard let first = displayFrames.first, let frame = currentDisplayFrame else { return }
        let seconds = max(frame.summary.timestamp - first.summary.timestamp, 0)
        player.seek(to: CMTime(seconds: seconds, preferredTimescale: 600),
                    toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func closePlayback() {
        playing = false; playTask?.cancel(); player.pause()
        displayFrames = []
    }

    private func nearestFrameIndex(to timestamp: Double) -> Int {
        var low = 0, high = displayFrames.count
        while low < high {
            let middle = (low + high) / 2
            if displayFrames[middle].summary.timestamp < timestamp {
                low = middle + 1
            } else {
                high = middle
            }
        }
        if low == 0 { return 0 }
        if low == displayFrames.count { return displayFrames.count - 1 }
        let before = displayFrames[low - 1].summary.timestamp
        let after = displayFrames[low].summary.timestamp
        return timestamp - before <= after - timestamp ? low - 1 : low
    }
}
