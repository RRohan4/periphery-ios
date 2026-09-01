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
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            SelfCheckView()
                .tabItem { Label("Self-check", systemImage: "checkmark.seal") }
            ComputePlanView()
                .tabItem { Label("Compute", systemImage: "cpu") }
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

#Preview {
    ContentView()
}
