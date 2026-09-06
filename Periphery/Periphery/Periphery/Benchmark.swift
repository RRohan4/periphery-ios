// Synthetic latency and sustained thermal measurements for the inference path.
// Warm-up frames are excluded; preprocessing is measured separately.

import CoreML
import Foundation
import os
import simd

struct LatencySample {
    var backbone: Double
    var gather: Double
    var head: Double
    var decode: Double
    var total: Double { backbone + gather + head + decode }
}

struct BenchmarkReport {
    var frames: Int = 0
    var detections: Int = 0
    var backboneMedian: Double = 0
    var gatherMedian: Double = 0
    var headMedian: Double = 0
    var decodeMedian: Double = 0
    var totalMedian: Double = 0
    var totalP95: Double = 0
    var totalWorst: Double = 0
    var fps: Double = 0
    var wallClock: Double = 0
    var thermalTransitions: [String] = []
    var startThermal: String = ""
    var endThermal: String = ""
    var lowPowerMode: Bool = false
    var availableMemoryMB: Double = 0
    /// What precision Core ML actually used for the tensors crossing its edge.
    var precision: String = ""
    /// nil when the run was flat out; otherwise the frame rate it was held to.
    var targetFPS: Double?

    var summary: String {
        String(format: """
            %d frames in %.1f s -> %.1f fps
            backbone %.1f ms | gather %.1f ms | head %.1f ms | decode %.1f ms
            total median %.1f ms, p95 %.1f ms, worst %.1f ms
            thermal %@ -> %@%@
            %.0f MB available, low power %@
            tensors: %@ | %@
            """,
            frames, wallClock, fps,
            backboneMedian * 1000, gatherMedian * 1000,
            headMedian * 1000, decodeMedian * 1000,
            totalMedian * 1000, totalP95 * 1000, totalWorst * 1000,
            startThermal, endThermal,
            thermalTransitions.isEmpty ? "" : " (" + thermalTransitions.joined(separator: ", ") + ")",
            availableMemoryMB,
            lowPowerMode ? "ON" : "off",
            precision,
            targetFPS.map { String(format: "paced to %.0f fps, duty cycle %.0f%%",
                                   $0, totalMedian * $0 * 100) } ?? "flat out")
    }
}

enum Benchmark {

    /// Calibration used for the LUT. Any plausible mount does; the gather cost
    /// depends on the voxel count, not on where the voxels land.
    static func referenceCalibration() -> Calibration {
        let focal = 910.0                     // comma EON, the measured rig
        let intrinsics = simd_double3x3(rows: [
            SIMD3<Double>(focal, 0, 582),
            SIMD3<Double>(0, focal, 437),
            SIMD3<Double>(0, 0, 1),
        ])
        return Calibration(pose: .fallback,
                           K: intrinsics,
                           frameWidth: 1164,
                           frameHeight: 874)
    }

    static func syntheticInput() throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, 3,
                                             NSNumber(value: Contract.inputHeight),
                                             NSNumber(value: Contract.inputWidth)],
                                     dataType: .float32)
        var generator = SystemRandomNumberGenerator()
        let values = (0..<array.count).map { _ in
            Float.random(in: -2.0...2.0, using: &generator)
        }
        try MLMultiArrayTransfer.writeFloats(values, to: array, named: "benchmark image")
        return array
    }

    static func run(frames: Int,
                    warmup: Int = 10,
                    targetFPS: Double? = nil,
                    progress: ((Int, Int) -> Void)? = nil) throws -> BenchmarkReport {
        let detector = try Detector(calibration: referenceCalibration())
        let input = try syntheticInput()

        for _ in 0..<warmup {

            try autoreleasepool {
                _ = try detector.detect(image: input)
            }
        }

        var samples = [LatencySample]()
        samples.reserveCapacity(frames)
        var detections = 0
        var report = BenchmarkReport()
        report.startThermal = describe(ProcessInfo.processInfo.thermalState)
        var lastThermal = ProcessInfo.processInfo.thermalState

        let interval = targetFPS.map { 1.0 / $0 }
        let start = DispatchTime.now()
        for frame in 0..<frames {
            let frameStart = DispatchTime.now()
            try autoreleasepool {
                let found = try detector.detect(image: input)
                detections += found.count
            }
            let timing = detector.lastTiming
            samples.append(LatencySample(backbone: timing.backbone,
                                         gather: timing.gather,
                                         head: timing.head,
                                         decode: timing.decode))

            if let interval {
                let spent = seconds(since: frameStart)
                if spent < interval { Thread.sleep(forTimeInterval: interval - spent) }
            }
            let thermal = ProcessInfo.processInfo.thermalState
            if thermal != lastThermal {
                let elapsed = seconds(since: start)
                report.thermalTransitions.append(
                    String(format: "%@ at %.0f s", describe(thermal), elapsed))
                lastThermal = thermal
            }
            progress?(frame + 1, frames)
        }
        report.wallClock = seconds(since: start)

        report.frames = samples.count
        report.detections = detections
        report.backboneMedian = median(samples.map(\.backbone))
        report.gatherMedian = median(samples.map(\.gather))
        report.headMedian = median(samples.map(\.head))
        report.decodeMedian = median(samples.map(\.decode))
        let totals = samples.map(\.total).sorted()
        report.totalMedian = median(totals)
        report.totalP95 = percentile(totals, 0.95)
        report.totalWorst = totals.last ?? 0
        report.fps = report.wallClock > 0 ? Double(report.frames) / report.wallClock : 0
        report.endThermal = describe(ProcessInfo.processInfo.thermalState)
        report.lowPowerMode = ProcessInfo.processInfo.isLowPowerModeEnabled
        report.availableMemoryMB = Double(os_proc_available_memory()) / 1_048_576.0
        report.precision = detector.precisionNote
        report.targetFPS = targetFPS
        return report
    }

    // MARK: - Helpers

    static func describe(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func median(_ values: [Double]) -> Double {
        percentile(values.sorted(), 0.5)
    }

    private static func percentile(_ sorted: [Double], _ fraction: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let index = Int((Double(sorted.count - 1) * fraction).rounded())
        return sorted[index]
    }

    private static func seconds(since mark: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - mark.uptimeNanoseconds) / 1e9
    }
}
