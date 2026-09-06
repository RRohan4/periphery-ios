import Combine
import SwiftUI

struct ReplayView: View {
    @StateObject private var model = ReplayModel()

    var body: some View {
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
}

@MainActor
final class ReplayModel: ObservableObject {
    @Published var drives: [URL] = []
    @Published var selected: URL?
    @Published var frames: [ReplayFrameSummary] = []
    @Published var position = 0.0
    @Published var progress = 0.0
    @Published var processing = false
    @Published var processedFrames = 0
    @Published var totalFrames = 0
    @Published var playing = false
    @Published var message: String?
    @Published var rateLine = "starting…"
    private var processor: ReplayProcessor?
    private var playTask: Task<Void, Never>?
    private var replayStarted = Date()

    func refresh() { drives = DriveRecorder.sessions() }
    func select(_ drive: URL) {
        selected = drive; frames = ReplayProcessor.loadSummaries(drive: drive)
        position = 0; message = frames.isEmpty ? "No replay sidecar yet." : "Loaded replay-safety40-v1.jsonl"
    }

    func reprocess() {
        guard let drive = selected, !processing else { return }
        guard !LiveSession.shared.pipeline.recorder.isRecording else {
            message = "Stop recording before Replay."; return
        }
        let worker = ReplayProcessor(); processor = worker; processing = true; progress = 0
        processedFrames = 0; totalFrames = 0
        replayStarted = Date(); rateLine = "starting…"
        LiveSession.shared.pipeline.suspendForReplay()
        message = "Running the shared perception engine…"
        Task.detached {
            do {
                let result = try worker.process(drive: drive) { done, total in
                    Task { @MainActor in
                        self.processedFrames = done; self.totalFrames = total
                        self.progress = Double(done) / Double(max(total, 1))
                        let elapsed = max(Date().timeIntervalSince(self.replayStarted), 0.001)
                        let fps = Double(done) / elapsed
                        let remaining = fps > 0 ? Double(max(total - done, 0)) / fps : 0
                        self.rateLine = String(format: "%.1f fps · elapsed %.0fs · ETA %.0fs",
                                               fps, elapsed, remaining)
                    }
                }
                await MainActor.run {
                    self.frames = result; self.position = 0; self.processing = false
                    LiveSession.shared.pipeline.resumeAfterReplay()
                    self.message = "Replay complete; versioned sidecar saved."
                }
            } catch {
                await MainActor.run {
                    self.processing = false; self.message = String(describing: error)
                    LiveSession.shared.pipeline.resumeAfterReplay()
                }
            }
        }
    }

    func cancel() { processor?.cancel() }
    func step(_ amount: Int) {
        playing = false; playTask?.cancel()
        position = min(max(position + Double(amount), 0), Double(max(frames.count - 1, 0)))
    }
    func togglePlay() {
        playing.toggle(); playTask?.cancel()
        guard playing else { return }
        playTask = Task {
            while !Task.isCancelled, playing, Int(position) < frames.count - 1 {
                try? await Task.sleep(for: .milliseconds(100)); position += 1
            }
            playing = false
        }
    }
}
