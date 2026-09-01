//
//  ContentView.swift
//  Periphery
//
//  The self-check screen. No camera, no model load: this runs the golden
//  vectors from Resources/ against the hand-written half of the port, which is
//  the half CoreML's export gate says nothing about.
//
//  Six checks are expected to pass. If "projection LUT" or "decode" fails,
//  every later measurement would be measuring a broken pipeline, so nothing
//  else should be wired up until they are green.
//

import SwiftUI

struct ContentView: View {
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
            .toolbar {
                Button("Re-run") { run() }
            }
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
        // Also to the console, so a failure is copyable out of Xcode.
        print(SelfCheck.summary(outcome))
    }
}

#Preview {
    ContentView()
}
