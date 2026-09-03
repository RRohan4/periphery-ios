//  RecordView.swift
//  Start a drive, watch it fill the disk, get it off the phone.
//
//  The export story is deliberately not a share sheet. A session is a DIRECTORY
//  of eight files, and a share sheet per file is unusable at that shape --
//  UIFileSharingEnabled and LSSupportsOpeningDocumentsInPlace put the whole
//  Documents folder in Files and in Finder, so a drive drags out intact.

import Combine
import SwiftUI

/// Footer copy, hoisted out of the ViewBuilders. A chain of `+`-concatenated
/// literals inside a ViewBuilder is a well-known way to blow the type checker's
/// budget; multi-line literals in a plain enum cost it nothing.
private enum Help {
    static let whileRecording = """
        Encoding is real thermal load. The Latency tab's numbers are not valid while \
        this is running.
        """
    static let beforeRecording = """
        1080p is about 4.5 GB per hour. Video, every sensor at its own rate, the \
        detections, and the clock anchors that tie them together.
        """
    static let export = """
        Files → On My iPhone → Periphery, or plug into a Mac and open the device in \
        Finder. Each drive is one folder; copy the whole thing.

        Offline: scripts/ingest_iphone_drive.py in the periphery repo turns it into a \
        segment the existing chain already reads.
        """
}

struct RecordView: View {
    @StateObject private var model = RecordModel()

    var body: some View {
        NavigationStack {
            List {
                Section {
                    if model.status.recording {
                        Button(role: .destructive) { model.stop() } label: {
                            Label("Stop", systemImage: "stop.circle.fill")
                        }
                    } else {
                        Picker("Quality", selection: $model.quality) {
                            ForEach(DriveRecorder.Quality.allCases, id: \.self) {
                                Text($0.rawValue).tag($0)
                            }
                        }
                        Button { model.start() } label: {
                            Label("Record a drive", systemImage: "record.circle")
                        }
                    }
                } header: {
                    Text("Recorder")
                } footer: {
                    Text(model.status.recording ? Help.whileRecording : Help.beforeRecording)
                        .font(.caption2)
                }

                if model.status.recording {
                    Section("Live") {
                        LabeledContent("session", value: model.status.name)
                        LabeledContent("elapsed", value: Self.clock(model.status.elapsed))
                        LabeledContent("frames", value: "\(model.status.videoFrames)")
                        LabeledContent("dropped", value: "\(model.status.droppedFrames)")
                        LabeledContent("written",
                                       value: String(format: "%.0f MB", model.status.megabytes))
                        LabeledContent("free",
                                       value: String(format: "%.1f GB", model.status.freeGigabytes))
                    }
                }

                Section {
                    if model.sessions.isEmpty {
                        Text("no drives yet").foregroundStyle(.secondary).font(.caption)
                    }
                    ForEach(model.sessions, id: \.name) { session in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(session.name)
                                .font(.system(.caption, design: .monospaced))
                            Text(String(format: "%.0f MB", session.megabytes))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    .onDelete { model.delete(at: $0) }
                } header: {
                    Text("Drives")
                } footer: {
                    Text(Help.export).font(.caption2)
                }

                if let error = model.error {
                    Text(error).font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.red)
                }
            }
            .navigationTitle("Record")
            .toolbar { EditButton() }
        }
        .task { await model.watch() }
    }

    private static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d:%02d", total / 3600, (total / 60) % 60, total % 60)
    }
}

@MainActor
final class RecordModel: ObservableObject {

    struct Session: Sendable {
        var name: String
        var url: URL
        var bytes: Int64
        var megabytes: Double { Double(bytes) / 1_048_576.0 }
    }

    @Published var status = DriveRecorder.Status()
    @Published var sessions: [Session] = []
    @Published var quality: DriveRecorder.Quality = .full
    @Published var error: String?

    /// The recorder is owned by the running pipeline -- there is only one
    /// camera, so there is only one recorder, and the Live tab holds it.
    private var recorder: DriveRecorder? { LiveSession.shared.pipeline.recorder }

    func start() {
        error = nil
        do {
            try recorder?.start(pose: LiveSession.shared.pipeline.currentPose,
                                quality: quality,
                                capture: LiveSession.shared.pipeline.captureFlags)
        } catch {
            self.error = String(describing: error)
        }
        refresh()
    }

    func stop() {
        recorder?.stop { _ in
            Task { @MainActor in self.refresh() }
        }
    }

    func delete(at offsets: IndexSet) {
        for index in offsets {
            try? FileManager.default.removeItem(at: sessions[index].url)
        }
        refresh()
    }

    func refresh() {
        status = recorder?.status ?? DriveRecorder.Status()
        sessions = DriveRecorder.sessions().map {
            Session(name: $0.lastPathComponent, url: $0, bytes: DriveRecorder.size(of: $0))
        }
    }

    /// One second is plenty: the interesting numbers move slowly and this view
    /// competes with a 30 Hz detector for the main actor.
    func watch() async {
        // Recording needs frames, and the user may never have opened Live.
        await LiveSession.shared.ensureStarted()
        refresh()
        while !Task.isCancelled {
            try? await Task.sleep(for: .seconds(1))
            status = recorder?.status ?? DriveRecorder.Status()
        }
    }
}
