//
//  ContentView.swift
//  Periphery
//
//  Two screens, both of which answer a question before any camera exists:
//
//  Self-check  the golden vectors from Resources/ against the hand-written half
//              of the port -- the half the CoreML export gate says nothing
//              about. Six rows, all expected green.
//  Compute     where Core ML plans to run each operation. Meaningless in the
//              simulator (no Neural Engine); run it on the phone.
//  Latency     the number the port exists for: per-frame cost of backbone,
//              gather, head and decode, and whether it survives ten minutes.
//  Live        the whole pipeline on camera frames, the 2.5D world view, and
//              the horizon laid over the image so a bad pose is visible.
//  Record      video, every sensor and the detections, into one directory per
//              drive -- the corpus this project has never had of its own.
//  Calibrate   the four numbers the projection depends on, ordered by how much
//              damage each does when wrong.
//  Flow        the camera pitch estimator made visible, on foot: walk forward
//              and watch the focus of expansion, the inliers and the angle.
//              Exercises the whole path -- Vision, gyro axes, three sign
//              conventions -- in ten seconds, indoors, without a car.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            LiveView()
                .tabItem { Label("Live", systemImage: "car.side") }
            RecordView()
                .tabItem { Label("Record", systemImage: "record.circle") }
            CalibrationView()
                .tabItem { Label("Calibrate", systemImage: "level") }
            FlowView()
                .tabItem { Label("Flow", systemImage: "arrow.up.right.circle") }
            SelfCheckView()
                .tabItem { Label("Self-check", systemImage: "checkmark.seal") }
            ComputePlanView()
                .tabItem { Label("Compute", systemImage: "cpu") }
            BenchmarkView()
                .tabItem { Label("Latency", systemImage: "speedometer") }
        }
    }
}

// MARK: - Self-check

struct SelfCheckView: View {
    @State private var results: [SelfCheck.Result] = []
    @State private var elapsed: Double = 0

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(Array(results.enumerated()), id: \.offset) { _, result in
                        HStack(alignment: .top, spacing: 10) {
                            Image(systemName: result.passed
                                  ? "checkmark.circle.fill" : "xmark.octagon.fill")
                                .foregroundStyle(result.passed ? .green : .red)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(result.name)
                                    .font(.body.weight(.medium))
                                Text(result.detail)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                        .padding(.vertical, 2)
                    }
                } header: {
                    Text(header)
                } footer: {
                    if !results.isEmpty {
                        Text("safety40-locked-v1 · threshold 0.50 · circular NMS 2.0 m\n"
                             + String(format: "checked in %.1f ms", elapsed * 1000))
                            .font(.caption2)
                    }
                }
            }
            .navigationTitle("Periphery")
            .toolbar { Button("Re-run") { run() } }
        }
        .task { run() }
    }

    private var header: String {
        if results.isEmpty { return "running…" }
        let failed = results.filter { !$0.passed }.count
        return failed == 0
            ? "PASS — \(results.count) checks"
            : "FAIL — \(failed) of \(results.count)"
    }

    private func run() {
        let start = DispatchTime.now()
        let outcome = SelfCheck.run()
        elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1e9
        results = outcome
        print(SelfCheck.summary(outcome))
    }
}

// MARK: - Compute plan

struct ComputePlanView: View {
    @State private var summaries: [ComputePlanSummary] = []
    @State private var failures: [String] = []
    @State private var loading = false

    var body: some View {
        NavigationStack {
            List {
                if isSimulator {
                    Section {
                        Label("Simulator has no Neural Engine — these numbers "
                              + "only mean something on the phone.",
                              systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
                ForEach(summaries) { summary in
                    Section {
                        LabeledContent("Neural Engine", value: "\(summary.neuralEngine)")
                        LabeledContent("GPU", value: "\(summary.gpu)")
                        LabeledContent("CPU", value: "\(summary.cpu)")
                        LabeledContent("estimated cost on ANE",
                                       value: String(format: "%.1f%%",
                                                     summary.neuralEngineCost * 100))
                        if summary.strays.isEmpty {
                            Text("every operation planned for the Neural Engine")
                                .font(.caption)
                                .foregroundStyle(.green)
                        } else {
                            DisclosureGroup("\(summary.strays.count) off the ANE") {
                                ForEach(summary.strays) { stray in
                                    HStack {
                                        Text(stray.op)
                                            .font(.system(.caption, design: .monospaced))
                                        Spacer()
                                        Text(stray.device)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(String(format: "%.1f%%", stray.cost * 100))
                                            .font(.system(.caption, design: .monospaced))
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                        }
                    } header: {
                        Text(summary.model)
                    } footer: {
                        Text(summary.headline).font(.caption2)
                    }
                }
                ForEach(failures, id: \.self) { failure in
                    Text(failure)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.red)
                }
            }
            .navigationTitle("Compute plan")
            .overlay {
                if loading { ProgressView("planning…") }
            }
            .toolbar { Button("Reload") { Task { await load() } } }
        }
        .task { await load() }
    }

    private var isSimulator: Bool {
        #if targetEnvironment(simulator)
        return true
        #else
        return false
        #endif
    }

    private func load() async {
        guard #available(iOS 17.0, *) else {
            failures = ["MLComputePlan needs iOS 17"]
            return
        }
        loading = true
        defer { loading = false }
        var loaded = [ComputePlanSummary]()
        var errors = [String]()
        for result in await ComputePlanProbe.summarizeAll() {
            switch result {
            case .success(let summary):
                loaded.append(summary)
                print("\(summary.model): \(summary.headline)")
                for stray in summary.strays {
                    print(String(format: "  %@ on %@ (%.2f%%)",
                                 stray.op, stray.device, stray.cost * 100))
                }
            case .failure(let error):
                errors.append(String(describing: error))
            }
        }
        summaries = loaded
        failures = errors
    }
}

// MARK: - Latency

struct BenchmarkView: View {
    @State private var report: BenchmarkReport?
    @State private var error: String?
    @State private var running = false
    @State private var progress = ""

    /// Short burst answers "how fast"; the long run answers "for how long",
    /// which is a different question and the one phones usually fail.
    private let burst = 200
    private let sustained = 18_000

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button("Burst — \(burst) frames") { start(frames: burst) }
                    Button("Sustained — 10 min at 30 fps") {
                        start(frames: sustained, targetFPS: 30)
                    }
                    Button("Saturated — 2000 frames flat out") { start(frames: 2000) }
                }
                .disabled(running)

                if running {
                    Section { HStack { ProgressView(); Text(progress).font(.caption) } }
                }

                if let report {
                    Section("Result") {
                        LabeledContent("fps", value: String(format: "%.1f", report.fps))
                        LabeledContent("median", value: ms(report.totalMedian))
                        LabeledContent("p95", value: ms(report.totalP95))
                        LabeledContent("worst", value: ms(report.totalWorst))
                    }
                    Section("Per stage (median)") {
                        LabeledContent("backbone", value: ms(report.backboneMedian))
                        LabeledContent("gather", value: ms(report.gatherMedian))
                        LabeledContent("head", value: ms(report.headMedian))
                        LabeledContent("decode", value: ms(report.decodeMedian))
                    }
                    Section("Thermal") {
                        LabeledContent("start", value: report.startThermal)
                        LabeledContent("end", value: report.endThermal)
                        if report.thermalTransitions.isEmpty {
                            Text("no transitions").font(.caption).foregroundStyle(.secondary)
                        } else {
                            ForEach(report.thermalTransitions, id: \.self) { transition in
                                Text(transition).font(.system(.caption, design: .monospaced))
                            }
                        }
                        LabeledContent("low power",
                                       value: report.lowPowerMode ? "ON" : "off")
                        LabeledContent("memory available",
                                       value: String(format: "%.0f MB", report.availableMemoryMB))
                    }
                    Section {
                        Text(report.summary)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                    } header: {
                        Text("copyable")
                    } footer: {
                        Text("Desktop reference: 7.6 ms CUDA, 58.8 ms CPU. "
                             + "Synthetic input, preprocessing excluded.")
                    }
                }

                if let error {
                    Text(error)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.red)
                }
            }
            .navigationTitle("Latency")
        }
    }

    private func ms(_ seconds: Double) -> String {
        String(format: "%.1f ms", seconds * 1000)
    }

    private func start(frames: Int, targetFPS: Double? = nil) {
        running = true
        error = nil
        report = nil
        progress = "warming up…"
        // A ten-minute run outlives the screen timeout, and a locked screen
        // suspends the app mid-measurement.
        UIApplication.shared.isIdleTimerDisabled = true
        // Off the main actor: a ten-minute run would otherwise wedge the UI and
        // the watchdog would kill the app.
        //
        // WARNING, AND IT IS A REAL ONE. The target sets
        // SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor, so `enum Benchmark` is
        // implicitly main-actor isolated and this `Task.detached` hops straight
        // back to the main actor -- exactly what the line above says it is
        // avoiding. Swift 5 language mode reports it as a warning rather than an
        // error, which is why it built.
        //
        // The fix is `nonisolated` on the compute chain (Benchmark, Detector,
        // Preprocessor), not a cast here. Until that is done and verified,
        // BURST is fine and SUSTAINED will freeze the UI for ten minutes.
        Task.detached(priority: .userInitiated) {
            do {
                let outcome = try Benchmark.run(frames: frames,
                                                targetFPS: targetFPS) { done, total in
                    if done % 20 == 0 || done == total {
                        Task { @MainActor in
                            progress = "\(done) / \(total)"
                        }
                    }
                }
                await MainActor.run {
                    report = outcome
                    running = false
                    UIApplication.shared.isIdleTimerDisabled = false
                    print(outcome.summary)
                }
            } catch {
                await MainActor.run {
                    self.error = String(describing: error)
                    running = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
