//  ReplayProcessor.swift
//  Recorded-drive adapter for the same PerceptionEngine used by Live.

import AVFoundation
import CoreVideo
import Foundation
import simd

struct ReplayFrameSummary: Codable, Sendable {
    var index: Int
    var timestamp: Double
    var rawCount: Int
    var trackCount: Int
}

struct ReplayProgress: Sendable {
    var completed: Int
    var total: Int
    var latest: ReplayFrameSummary
    var preprocessMS: Double
    var inferenceMS: Double
}

final class ReplayProcessor: @unchecked Sendable {
    enum ReplayError: Error, CustomStringConvertible {
        case missing(String), invalid(String), cancelled
        var description: String {
            switch self {
            case .missing(let value): return "missing \(value)"
            case .invalid(let value): return "invalid replay data: \(value)"
            case .cancelled: return "cancelled"
            }
        }
    }

    private struct FrameRow {
        var pts: Double; var K: simd_double3x3?; var width: Int; var height: Int
    }
    private struct Rate { var t: Double; var yaw: Double }
    private struct Speed { var t: Double; var value: Double }
    private struct Manifest: Decodable {
        struct Pose: Decodable {
            var pitchDeg, rollDeg, yawDeg, height, forward: Double
            var pitchFrom, rollFrom, yawFrom, heightFrom: String
            enum CodingKeys: String, CodingKey {
                case pitchDeg = "pitch_deg", rollDeg = "roll_deg", yawDeg = "yaw_deg"
                case height = "height_m", forward = "forward_of_origin_m"
                case pitchFrom = "pitch_from", rollFrom = "roll_from"
                case yawFrom = "yaw_from", heightFrom = "height_from"
            }
        }
        var checkpoint: String
        var pose: Pose
    }

    private let lock = NSLock()
    private var cancelled = false
    func cancel() { lock.withLock { cancelled = true } }

    static func sidecarURL(for drive: URL) -> URL {
        drive.appendingPathComponent("replay-safety40-v1.jsonl")
    }

    func process(drive: URL, progress: @escaping @Sendable (ReplayProgress) -> Void) throws
        -> [ReplayFrameSummary] {
        lock.withLock { cancelled = false }
        let manifest = try loadManifest(drive)
        let pose = mountPose(manifest.pose)
        let rows = try loadFrames(drive.appendingPathComponent("frames.csv"))
        let rates = try loadRates(drive.appendingPathComponent("motion.csv"))
        let speeds = try loadSpeeds(drive.appendingPathComponent("location.csv"))
        guard !rows.isEmpty else { throw ReplayError.invalid("frames.csv is empty") }

        let asset = AVAsset(url: drive.appendingPathComponent("video.mov"))
        guard let track = asset.tracks(withMediaType: .video).first else {
            throw ReplayError.missing("video track")
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw ReplayError.invalid("video reader output") }
        reader.add(output)
        guard reader.startReading() else {
            throw ReplayError.invalid(reader.error?.localizedDescription ?? "reader start")
        }

        let destination = Self.sidecarURL(for: drive)
        let temporary = drive.appendingPathComponent(".replay-safety40-v1.tmp")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        let file = try FileHandle(forWritingTo: temporary)
        var completed = false
        defer {
            try? file.close()
            if !completed { try? FileManager.default.removeItem(at: temporary) }
        }
        try line(Self.header(checkpoint: manifest.checkpoint), to: file)

        let engine = try PerceptionEngine()
        var summaries: [ReplayFrameSummary] = []
        var firstVideoPTS: Double?
        var videoToRecorded = 0.0
        var previousRecordedPTS: Double?
        var rateIndex = 0
        var speedIndex = 0
        var frameIndex = 0
        var latestPreprocessMS = 0.0
        var latestInferenceMS = 0.0

        while let sample = output.copyNextSampleBuffer() {
            if lock.withLock({ cancelled }) { reader.cancelReading(); throw ReplayError.cancelled }
            guard let pixel = CMSampleBufferGetImageBuffer(sample) else { continue }
            let videoPTS = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
            if firstVideoPTS == nil {
                firstVideoPTS = videoPTS
                videoToRecorded = rows[0].pts - videoPTS
            }
            let recordedPTS = videoPTS + videoToRecorded
            let metadata = nearestFrame(to: recordedPTS, rows: rows)
            let ego = egoDelta(from: previousRecordedPTS, to: metadata.pts,
                               rates: rates, speeds: speeds,
                               rateIndex: &rateIndex, speedIndex: &speedIndex)
            previousRecordedPTS = metadata.pts
            let intrinsics = metadata.K ?? Self.fallbackIntrinsics(
                width: metadata.width, height: metadata.height)
            let result = try engine.process(frame: PerceptionFrame(
                pixelBuffer: pixel, timestamp: metadata.pts, intrinsics: intrinsics,
                width: metadata.width, height: metadata.height,
                mountPose: pose, egoMotion: ego))
            let summary = ReplayFrameSummary(index: frameIndex, timestamp: metadata.pts,
                                             rawCount: result.rawDetections.count,
                                             trackCount: result.trackedObjects.count)
            summaries.append(summary)
            try line(Self.encode(result: result, summary: summary), to: file)
            frameIndex += 1
            latestPreprocessMS = result.timings.preprocessMS
            latestInferenceMS = result.timings.inferenceMS
            if frameIndex.isMultiple(of: 10) {
                progress(ReplayProgress(completed: frameIndex, total: rows.count,
                                        latest: summary,
                                        preprocessMS: result.timings.preprocessMS,
                                        inferenceMS: result.timings.inferenceMS))
            }
        }
        guard reader.status == .completed else {
            throw ReplayError.invalid(reader.error?.localizedDescription ?? "video decode")
        }
        try file.synchronize()
        try file.close()
        if FileManager.default.fileExists(atPath: destination.path) {
            _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary)
        } else {
            try FileManager.default.moveItem(at: temporary, to: destination)
        }
        completed = true
        if let latest = summaries.last {
            progress(ReplayProgress(completed: frameIndex, total: frameIndex,
                                    latest: latest,
                                    preprocessMS: latestPreprocessMS,
                                    inferenceMS: latestInferenceMS))
        }
        return summaries
    }

    static func loadSummaries(drive: URL) -> [ReplayFrameSummary] {
        guard let text = try? String(contentsOf: sidecarURL(for: drive), encoding: .utf8) else {
            return []
        }
        return text.split(separator: "\n").dropFirst().compactMap { row in
            guard let data = row.data(using: .utf8),
                  let decoded = try? JSONSerialization.jsonObject(with: data),
                  let object = decoded as? [String: Any],
                  let index = object["index"] as? Int,
                  let pts = object["pts"] as? Double,
                  let raw = object["raw_count"] as? Int,
                  let tracks = object["track_count"] as? Int else { return nil }
            return ReplayFrameSummary(index: index, timestamp: pts,
                                      rawCount: raw, trackCount: tracks)
        }
    }

    private func loadManifest(_ drive: URL) throws -> Manifest {
        let url = drive.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url) else { throw ReplayError.missing("manifest.json") }
        return try JSONDecoder().decode(Manifest.self, from: data)
    }

    private func mountPose(_ value: Manifest.Pose) -> MountPose {
        func source(_ raw: String) -> MountPose.Provenance {
            MountPose.Provenance(rawValue: raw) ?? .fallback
        }
        return MountPose(pitch: value.pitchDeg * .pi / 180,
                         roll: value.rollDeg * .pi / 180,
                         yaw: value.yawDeg * .pi / 180,
                         height: value.height, forwardOfOrigin: value.forward,
                         pitchFrom: source(value.pitchFrom), rollFrom: source(value.rollFrom),
                         yawFrom: source(value.yawFrom), heightFrom: source(value.heightFrom),
                         pitchSigmaDegrees: nil)
    }

    private func loadFrames(_ url: URL) throws -> [FrameRow] {
        try csv(url).dropFirst().compactMap { fields in
            guard fields.count >= 12, let pts = Double(fields[0]),
                  let width = Int(fields[10]), let height = Int(fields[11]) else { return nil }
            let values = fields[1...9].compactMap { Double($0) }
            let K = values.count == 9 ? simd_double3x3(columns: (
                SIMD3(values[0], values[1], values[2]),
                SIMD3(values[3], values[4], values[5]),
                SIMD3(values[6], values[7], values[8]))) : nil
            return FrameRow(pts: pts, K: K, width: width, height: height)
        }
    }

    private func loadRates(_ url: URL) throws -> [Rate] {
        try csv(url).dropFirst().compactMap { f in
            guard f.count > 9, let t = Double(f[0]), let gx = Double(f[1]),
                  let gy = Double(f[2]), let gz = Double(f[3]),
                  let rx = Double(f[7]), let ry = Double(f[8]), let rz = Double(f[9]) else { return nil }
            let length = sqrt(gx * gx + gy * gy + gz * gz)
            guard length > 1e-6 else { return nil }
            return Rate(t: t, yaw: -(rx * gx + ry * gy + rz * gz) / length)
        }
    }

    private func loadSpeeds(_ url: URL) throws -> [Speed] {
        try csv(url).dropFirst().compactMap { f in
            guard f.count > 7, let t = Double(f[0]), let speed = Double(f[7]), speed >= 0 else {
                return nil
            }
            return Speed(t: t, value: speed)
        }
    }

    private func csv(_ url: URL) throws -> [[Substring]] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else {
            throw ReplayError.missing(url.lastPathComponent)
        }
        return text.split(separator: "\n").map {
            $0.split(separator: ",", omittingEmptySubsequences: false)
        }
    }

    private func nearestFrame(to pts: Double, rows: [FrameRow]) -> FrameRow {
        var low = 0, high = rows.count
        while low < high {
            let middle = (low + high) / 2
            if rows[middle].pts < pts { low = middle + 1 } else { high = middle }
        }
        if low == 0 { return rows[0] }
        if low == rows.count { return rows[rows.count - 1] }
        return abs(rows[low].pts - pts) < abs(rows[low - 1].pts - pts) ? rows[low] : rows[low - 1]
    }

    private func egoDelta(from previous: Double?, to current: Double,
                          rates: [Rate], speeds: [Speed],
                          rateIndex: inout Int, speedIndex: inout Int) -> EgoDelta {
        guard let previous else { return EgoDelta(valid: false) }
        let dt = current - previous
        guard dt > 0, dt <= 1 else { return EgoDelta(dt: dt, valid: false) }
        while rateIndex < rates.count, rates[rateIndex].t <= previous { rateIndex += 1 }
        var scan = rateIndex, yawSum = 0.0, yawCount = 0
        while scan < rates.count, rates[scan].t <= current {
            yawSum += rates[scan].yaw; yawCount += 1; scan += 1
        }
        rateIndex = scan
        while speedIndex + 1 < speeds.count, speeds[speedIndex + 1].t <= current {
            speedIndex += 1
        }
        guard yawCount > 0, speedIndex < speeds.count,
              speeds[speedIndex].t <= current, current - speeds[speedIndex].t <= 2 else {
            return EgoDelta(dt: dt, valid: false)
        }
        return LiveEgoMotion.arcDelta(speed: speeds[speedIndex].value,
                                      yawRate: yawSum / Double(yawCount), dt: dt)
    }

    private func line(_ text: String, to file: FileHandle) throws {
        try file.write(contentsOf: Data((text + "\n").utf8))
    }

    private static func header(checkpoint: String) -> String {
        "{\"schema\":1,\"kind\":\"periphery-replay\",\"checkpoint\":\"\(checkpoint)\","
            + "\"tracker\":\"safety40-tracker-v1\",\"score_threshold\":0.5,"
            + "\"reject_implausible\":true,\"units\":\"SI\"}"
    }

    private static func encode(result: PerceptionResult, summary: ReplayFrameSummary) -> String {
        let raw = result.rawDetections.map { d in
            "[\(d.score),\(d.label),\(d.x),\(d.y),\(d.z),\(d.length),\(d.width),\(d.height),\(d.yaw)]"
        }.joined(separator: ",")
        let tracks = result.trackedObjects.map { t in
            let velocity = t.velocity.map { "[\($0.x),\($0.y)]" } ?? "null"
            return "[\(t.id),\(t.x),\(t.y),\(t.z),\(t.length),\(t.width),\(t.height),"
                + "\(t.yaw),\(t.modelYaw),\(t.label),\(t.score),\(t.hits),\(t.observed),"
                + "\"\(t.state.rawValue)\",\(velocity)]"
        }.joined(separator: ",")
        let ego = result.egoMotion
        let calibration = result.calibration
        let pose = calibration.calibration.pose
        return "{\"index\":\(summary.index),\"pts\":\(summary.timestamp),"
            + "\"raw_count\":\(summary.rawCount),\"track_count\":\(summary.trackCount),"
            + "\"ego\":[\(ego.dx),\(ego.dy),\(ego.dyaw),\(ego.dt)],"
            + "\"timing_ms\":[\(result.timings.preprocessMS),\(result.timings.inferenceMS)],"
            + "\"calibration\":{\"focal\":\(calibration.focal),"
            + "\"crop\":[\(calibration.crop.x),\(calibration.crop.y),"
            + "\(calibration.crop.width),\(calibration.crop.height)],"
            + "\"pose\":[\(pose.pitch),\(pose.roll),\(pose.yaw),\(pose.height),"
            + "\(pose.forwardOfOrigin)]},\"raw\":[\(raw)],\"tracks\":[\(tracks)]}"
    }

    private static func fallbackIntrinsics(width: Int, height: Int) -> simd_double3x3 {
        let focal = Double(width) / (2 * tan(60 * .pi / 180 / 2))
        return simd_double3x3(rows: [SIMD3(focal, 0, Double(width) / 2),
                                     SIMD3(0, focal, Double(height) / 2), SIMD3(0, 0, 1)])
    }
}
