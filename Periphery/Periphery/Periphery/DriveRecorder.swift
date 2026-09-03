//  DriveRecorder.swift
//  One drive = one directory: video, every sensor at its native rate, the
//  detections the phone produced, and the clock anchors that tie them together.
//
//  This is note 6's step 0 -- "ship a logger, not an estimator" -- widened to
//  carry frames. The reason for the widening is that it stops being a
//  validation artefact and becomes a corpus generator. Every number in the
//  periphery repo comes from comma2k19: a highway corpus, someone else's
//  windshield, someone else's optics, and only two distinct devices. A drive
//  recorded here replays through the same offline chain
//  (fastbev_detect_comma -> track_comma -> build_comma_viewer) on your camera
//  and your mount.
//
//  NOTHING IS RESAMPLED OR JOINED HERE. Each stream is written at its own rate
//  with its own timestamps and alignment is resolved offline, exactly as note 6
//  specifies. Pairing CoreMotion with CoreLocation on device would silently
//  decimate a 100 Hz attitude stream to 1 Hz.
//
//  Layout:
//      drive_2026-09-02_14-31-07/
//        manifest.json      device, preset, checkpoint, pose, first video PTS
//        video.mov          HEVC at the capture rate
//        frames.csv         pts, the full per-frame K, exposure, ISO, lens
//        motion.csv         gravity, user accel, rotation rate, quaternion
//        location.csv       both clock domains, speed, course, accuracies
//        altimeter.csv      relative altitude and pressure
//        heading.csv        CLHeading -- the magnetometer's independent answer
//        health.csv         thermal state, fps and drops, once a second
//        anchors.csv        wall clock against boot clock, every 60 s
//        detections.jsonl   what the on-device model saw, per frame
//
//  Two things recording must not do. It must not block the capture queue: the
//  writer runs on its own serial queue and a dropped RECORDING frame is
//  preferable to a dropped DETECTION frame. And it must not be mistaken for a
//  benchmark -- encoding is real thermal load, so the Latency tab's numbers are
//  not valid while this is running.

import AVFoundation
import CoreMedia
import Foundation
import UIKit
import simd

final class DriveRecorder: @unchecked Sendable {

    // MARK: - Status

    struct Status: Sendable {
        var recording = false
        var name = ""
        var elapsed: TimeInterval = 0
        var videoFrames = 0
        var droppedFrames = 0
        var bytes: Int64 = 0
        var freeBytes: Int64 = 0
        var note = ""
        /// Boot-clock time the session began, for the elapsed readout.
        var startBoot: TimeInterval = 0

        var megabytes: Double { Double(bytes) / 1_048_576.0 }
        var freeGigabytes: Double { Double(freeBytes) / 1_073_741_824.0 }
    }

    enum Quality: String, CaseIterable, Sendable {
        /// The capture resolution, ~10 Mbit/s. About 4.5 GB per hour.
        case full = "1080p"
        /// Downscaled by the encoder, ~4 Mbit/s. For long shakedown drives
        /// where the video is wanted for context rather than for replay.
        case compact = "720p"

        var size: CGSize {
            switch self {
            case .full: return CGSize(width: 1920, height: 1080)
            case .compact: return CGSize(width: 1280, height: 720)
            }
        }
        var bitrate: Int {
            switch self {
            case .full: return 10_000_000
            case .compact: return 4_000_000
            }
        }
    }

    enum RecorderError: Error, CustomStringConvertible {
        case alreadyRecording
        case cannotCreate(String)
        case noSpace(Double)

        var description: String {
            switch self {
            case .alreadyRecording: return "already recording"
            case .cannotCreate(let what): return "cannot create \(what)"
            case .noSpace(let gb): return String(format: "only %.1f GB free", gb)
            }
        }
    }

    /// Stop rather than fill the device mid-drive.
    static let freeSpaceFloorBytes: Int64 = 2 * 1_073_741_824

    // MARK: - State

    private let queue = DispatchQueue(label: "com.periphery.recorder", qos: .utility)
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var directory: URL?
    private var started = false
    private var firstPTS: CMTime?
    private var frames = 0
    private var dropped = 0
    private var inFlight = 0
    private var lastAnchor: TimeInterval = 0
    private var lastSpaceCheck: TimeInterval = 0
    private var cachedFree: Int64 = 0

    private var framesCSV: CSVWriter?
    private var motionCSV: CSVWriter?
    private var locationCSV: CSVWriter?
    private var altimeterCSV: CSVWriter?
    private var anchorCSV: CSVWriter?
    private var headingCSV: CSVWriter?
    private var healthCSV: CSVWriter?
    private var lastHealth: Double = 0
    private var detectionLog: CSVWriter?

    private let lock = NSLock()
    private var _status = Status()
    var status: Status {
        lock.withLock {
            var s = _status
            if s.recording { s.elapsed = ProcessInfo.processInfo.systemUptime - s.startBoot }
            return s
        }
    }

    /// More than this many frames waiting on the writer queue and the next one
    /// is dropped instead of queued. Bounded, so a slow encoder cannot grow the
    /// backlog until the capture pool starves the detector.
    private static let maxInFlight = 4

    var isRecording: Bool { lock.withLock { _status.recording } }

    // MARK: - Where drives live

    static var root: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    static func sessions() -> [URL] {
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.creationDateKey])) ?? []
        return contents
            .filter { $0.lastPathComponent.hasPrefix("drive_") }
            .sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    static func size(of session: URL) -> Int64 {
        let keys: [URLResourceKey] = [.fileSizeKey]
        guard let e = FileManager.default.enumerator(at: session,
                                                     includingPropertiesForKeys: keys) else {
            return 0
        }
        var total: Int64 = 0
        for case let url as URL in e {
            total += Int64((try? url.resourceValues(forKeys: Set(keys)))?.fileSize ?? 0)
        }
        return total
    }

    static func freeBytes() -> Int64 {
        let values = try? root.resourceValues(
            forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? 0
    }

    // MARK: - Lifecycle

    /// Whether the recorded streams mean anything.
    ///
    /// These are not statistics, they are validity flags. Heading is
    /// meaningless outside the true-north reference frame; if intrinsics never
    /// arrived, every K column in frames.csv is empty; if stabilisation stayed
    /// on, the geometry moved per frame without saying so. Reading a drive back
    /// without knowing these is guesswork, so they go in the manifest.
    struct Capture {
        var referenceFrame = ""
        var headingIsTrueNorth = false
        var stabilizationDisabled = false
        var intrinsicsAvailable = false
        var altimeterAvailable = false
        var attitudeRateHz = 0.0
    }

    func start(pose: MountPose, quality: Quality = .full, note: String = "",
               capture: Capture = Capture()) throws {
        guard !isRecording else { throw RecorderError.alreadyRecording }
        let free = Self.freeBytes()
        guard free > Self.freeSpaceFloorBytes else {
            throw RecorderError.noSpace(Double(free) / 1_073_741_824.0)
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let name = "drive_" + formatter.string(from: Date())
        let dir = Self.root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let movie = dir.appendingPathComponent("video.mov")
        let assetWriter = try AVAssetWriter(outputURL: movie, fileType: .mov)
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: Int(quality.size.width),
            AVVideoHeightKey: Int(quality.size.height),
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: quality.bitrate,
                AVVideoExpectedSourceFrameRateKey: 30,
            ],
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        guard assetWriter.canAdd(input) else { throw RecorderError.cannotCreate("video input") }
        assetWriter.add(input)
        guard assetWriter.startWriting() else {
            let reason = assetWriter.error?.localizedDescription ?? "unknown"
            throw RecorderError.cannotCreate("writer: \(reason)")
        }

        queue.sync {
            self.writer = assetWriter
            self.videoInput = input
            self.directory = dir
            self.started = true
            self.firstPTS = nil
            self.frames = 0
            self.dropped = 0
            self.inFlight = 0
            self.lastAnchor = 0
            self.lastSpaceCheck = 0
            self.cachedFree = free

            self.framesCSV = CSVWriter(dir.appendingPathComponent("frames.csv"),
                header: "pts,k00,k01,k02,k10,k11,k12,k20,k21,k22,width,height,"
                      + "exposure_s,iso,lens")
            self.motionCSV = CSVWriter(dir.appendingPathComponent("motion.csv"),
                header: "t,gx,gy,gz,ax,ay,az,rx,ry,rz,qx,qy,qz,qw,gravity_pitch,roll,heading")
            self.locationCSV = CSVWriter(dir.appendingPathComponent("location.csv"),
                header: "t,t_wall,lat,lon,alt,h_acc,v_acc,speed,speed_acc,course,course_acc")
            self.altimeterCSV = CSVWriter(dir.appendingPathComponent("altimeter.csv"),
                header: "t,relative_altitude,pressure_kpa")
            self.anchorCSV = CSVWriter(dir.appendingPathComponent("anchors.csv"),
                header: "wall,boot")
            self.headingCSV = CSVWriter(dir.appendingPathComponent("heading.csv"),
                header: "t,t_wall,true_heading,magnetic_heading,accuracy")
            self.healthCSV = CSVWriter(dir.appendingPathComponent("health.csv"),
                header: "t,thermal,frames,dropped,low_power")
            self.detectionLog = CSVWriter(dir.appendingPathComponent("detections.jsonl"),
                header: nil)
        }

        writeManifest(dir: dir, pose: pose, quality: quality, note: note, capture: capture)

        lock.withLock {
            _status = Status(recording: true, name: name, elapsed: 0,
                             videoFrames: 0, droppedFrames: 0,
                             bytes: 0, freeBytes: free, note: note,
                             startBoot: ProcessInfo.processInfo.systemUptime)
        }
    }

    /// Finishes the movie and flushes every CSV. Asynchronous because
    /// `finishWriting` is; the completion fires once the file is playable.
    func stop(completion: (@Sendable (URL?) -> Void)? = nil) {
        guard isRecording else { completion?(nil); return }
        lock.withLock { _status.recording = false }
        queue.async {
            self.started = false
            let dir = self.directory
            [self.framesCSV, self.motionCSV, self.locationCSV,
             self.altimeterCSV, self.anchorCSV, self.detectionLog].forEach { $0?.close() }
            self.framesCSV = nil; self.motionCSV = nil; self.locationCSV = nil
            self.altimeterCSV = nil; self.anchorCSV = nil; self.detectionLog = nil

            self.videoInput?.markAsFinished()
            self.writer?.finishWriting {
                self.queue.async {
                    self.writer = nil
                    self.videoInput = nil
                    self.directory = nil
                    if let dir { self.appendFinalManifest(dir: dir) }
                    completion?(dir)
                }
            }
        }
    }

    // MARK: - Sinks

    /// Called on the CAPTURE queue. Retains the buffer, hands it to the writer
    /// queue and returns immediately. If the writer is already `maxInFlight`
    /// behind, the frame is dropped rather than queued -- detection latency is
    /// the thing being protected.
    func append(frame: CameraSession.Frame) {
        guard isRecording else { return }
        let backlog = lock.withLock { () -> Bool in inFlight >= Self.maxInFlight }
        if backlog {
            queue.async { self.dropped += 1 }
            return
        }
        lock.withLock { inFlight += 1 }
        let buffer = frame.sampleBuffer
        let pts = frame.presentationTime
        let k = frame.intrinsics
        let width = frame.width, height = frame.height
        let exposure = frame.exposureSeconds
        let iso = frame.iso
        let lens = frame.lensPosition
        queue.async {
            defer { self.lock.withLock { self.inFlight -= 1 } }
            guard self.started, let writer = self.writer, let input = self.videoInput else { return }
            if self.firstPTS == nil {
                self.firstPTS = pts
                writer.startSession(atSourceTime: pts)
            }
            if input.isReadyForMoreMediaData, writer.status == .writing,
               input.append(buffer) {
                self.frames += 1
            } else {
                self.dropped += 1
            }
            let seconds = CMTimeGetSeconds(pts)
            let optics = [Self.f(exposure, 6), Self.f(iso, 1), Self.f(lens, 4)]
            if let k {
                self.framesCSV?.row([
                    Self.f(seconds, 6),
                    Self.f(k[0][0]), Self.f(k[1][0]), Self.f(k[2][0]),
                    Self.f(k[0][1]), Self.f(k[1][1]), Self.f(k[2][1]),
                    Self.f(k[0][2]), Self.f(k[1][2]), Self.f(k[2][2]),
                    String(width), String(height),
                ] + optics)
            } else {
                self.framesCSV?.row([Self.f(seconds, 6), "", "", "", "", "", "", "", "", "",
                                     String(width), String(height)] + optics)
            }
            // Once a second: did the phone throttle, and did the frame rate
            // survive it? A drive that thermally degraded halfway through looks
            // exactly like a drive where the maths got worse, unless this is on
            // disk.
            if seconds - self.lastHealth >= 1.0 {
                self.lastHealth = seconds
                let info = ProcessInfo.processInfo
                self.healthCSV?.row([
                    Self.f(seconds, 3),
                    Benchmark.describe(info.thermalState),
                    String(self.frames), String(self.dropped),
                    info.isLowPowerModeEnabled ? "1" : "0",
                ])
            }
            self.tick()
        }
    }

    func append(attitude a: MotionSource.Attitude) {
        guard isRecording else { return }
        queue.async {
            self.motionCSV?.row([
                Self.f(a.timestamp, 6),
                Self.f(a.gravity.x), Self.f(a.gravity.y), Self.f(a.gravity.z),
                Self.f(a.userAcceleration.x), Self.f(a.userAcceleration.y), Self.f(a.userAcceleration.z),
                Self.f(a.rotationRate.x), Self.f(a.rotationRate.y), Self.f(a.rotationRate.z),
                Self.f(a.quaternion.imag.x), Self.f(a.quaternion.imag.y),
                Self.f(a.quaternion.imag.z), Self.f(a.quaternion.real),
                Self.f(a.gravityPitch), Self.f(a.roll),
                a.cameraHeading.map { Self.f($0) } ?? "",
            ])
        }
    }

    func append(location l: MotionSource.Location) {
        guard isRecording else { return }
        queue.async {
            self.locationCSV?.row([
                Self.f(l.timestamp, 6), Self.f(l.wallTimestamp, 6),
                Self.f(l.latitude, 8), Self.f(l.longitude, 8), Self.f(l.altitude, 3),
                Self.f(l.horizontalAccuracy, 2), Self.f(l.verticalAccuracy, 2),
                Self.f(l.speed, 3), Self.f(l.speedAccuracy, 3),
                Self.f(l.course, 3), Self.f(l.courseAccuracy, 3),
            ])
        }
    }

    func append(altitude a: MotionSource.Altitude) {
        guard isRecording else { return }
        queue.async {
            self.altimeterCSV?.row([Self.f(a.timestamp, 6),
                                    Self.f(a.relativeAltitude, 4),
                                    Self.f(a.pressureKPa, 5)])
        }
    }

    func append(heading h: MotionSource.Heading) {
        guard isRecording else { return }
        queue.async {
            self.headingCSV?.row([Self.f(h.timestamp, 6), Self.f(h.wallTimestamp, 6),
                                  Self.f(h.trueHeading, 3),
                                  Self.f(h.magneticHeading, 3),
                                  Self.f(h.accuracy, 3)])
        }
    }

    /// One JSON line per frame, keyed to the video's own presentation time.
    /// Diffing this against the Python detector's output on the same frames is
    /// the end-to-end float16 number the README currently lists as unquantified.
    func append(detections: [Detection], at pts: CMTime) {
        guard isRecording else { return }
        queue.async {
            let rows = detections.map { d in
                "[\(Self.f(Double(d.score), 4)),\(d.label),\(Self.f(d.x, 3)),\(Self.f(d.y, 3)),"
                + "\(Self.f(d.z, 3)),\(Self.f(d.length, 3)),\(Self.f(d.width, 3)),"
                + "\(Self.f(d.height, 3)),\(Self.f(d.yaw, 4))]"
            }.joined(separator: ",")
            self.detectionLog?.line(
                "{\"pts\":\(Self.f(CMTimeGetSeconds(pts), 6)),\"d\":[\(rows)]}")
        }
    }

    // MARK: - Housekeeping, on the writer queue

    /// Clock anchors every 60 s and a free-space check every 10 s.
    ///
    /// The anchors are the only thing that lets CoreLocation's wall-clock
    /// timestamps sit on the same timeline as the video offline. They are
    /// RECORDED rather than applied, so wall-clock drift over a long drive
    /// stays measurable instead of being silently absorbed.
    private func tick() {
        let now = ProcessInfo.processInfo.systemUptime
        if now - lastAnchor > 60 || lastAnchor == 0 {
            lastAnchor = now
            let anchor = MotionSource.sampleAnchor()
            anchorCSV?.row([Self.f(anchor.wallSeconds, 6), Self.f(anchor.bootSeconds, 6)])
        }
        if now - lastSpaceCheck > 10 {
            lastSpaceCheck = now
            cachedFree = Self.freeBytes()
            let bytes = directory.map { Self.size(of: $0) } ?? 0
            let frames = self.frames, dropped = self.dropped, free = cachedFree
            lock.withLock {
                _status.videoFrames = frames
                _status.droppedFrames = dropped
                _status.bytes = bytes
                _status.freeBytes = free
            }
            if free < Self.freeSpaceFloorBytes {
                lock.withLock { _status.note = "stopped: disk nearly full" }
                stop()
            }
        } else {
            let frames = self.frames, dropped = self.dropped
            lock.withLock {
                _status.videoFrames = frames
                _status.droppedFrames = dropped
            }
        }
    }

    // MARK: - Manifest

    private func writeManifest(dir: URL, pose: MountPose, quality: Quality, note: String,
                               capture: Capture) {
        let anchor = MotionSource.sampleAnchor()
        let device = UIDevice.current
        var manifest: [String: Any] = [
            "schema": 1,
            "checkpoint": "safety40-locked-v1",
            "created_wall": anchor.wallSeconds,
            "created_boot": anchor.bootSeconds,
            "device_model": Self.hardwareModel(),
            "device_name": device.model,
            "system": device.systemName + " " + device.systemVersion,
            "quality": quality.rawValue,
            "video_width": Int(quality.size.width),
            "video_height": Int(quality.size.height),
            "video_bitrate": quality.bitrate,
            "note": note,
            "pose": [
                "pitch_deg": pose.pitchDegrees,
                "roll_deg": pose.rollDegrees,
                "yaw_deg": pose.yawDegrees,
                "height_m": pose.height,
                "forward_of_origin_m": pose.forwardOfOrigin,
                "pitch_from": pose.pitchFrom.rawValue,
                "roll_from": pose.rollFrom.rawValue,
                "yaw_from": pose.yawFrom.rawValue,
                "height_from": pose.heightFrom.rawValue,
            ],
            // Whether the streams above can be trusted at all. Heading is
            // meaningless outside the true-north frame; intrinsics that never
            // arrived mean every K column is empty; stabilisation left on means
            // the geometry moved per frame without saying so.
            "capture": [
                "reference_frame": capture.referenceFrame,
                "heading_is_true_north": capture.headingIsTrueNorth,
                "stabilization_disabled": capture.stabilizationDisabled,
                "intrinsics_available": capture.intrinsicsAvailable,
                "altimeter_available": capture.altimeterAvailable,
                "attitude_rate_hz": capture.attitudeRateHz,
            ],
            "clocks": [
                "video_pts": "host time clock (mach_absolute_time), seconds",
                "motion_t": "seconds since boot, same domain as video_pts",
                "altimeter_t": "seconds since boot, same domain as video_pts",
                "location_t": "republished into the boot domain via anchors.csv",
                "location_t_wall": "CLLocation.timestamp, seconds since 1970",
            ],
        ]
        manifest["trained_focal_px"] = Contract.trainedFocal
        manifest["score_threshold"] = Contract.scoreThreshold
        manifest["nms_radius_m"] = Contract.nmsRadius
        manifest["class_names"] = Contract.classNames
        write(manifest, to: dir.appendingPathComponent("manifest.json"))
    }

    /// Rewritten at stop so the first video PTS and the frame count are on
    /// record. Offline, `first_pts` is what maps video time onto the boot clock.
    private func appendFinalManifest(dir: URL) {
        let url = dir.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url),
              var manifest = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return }
        let anchor = MotionSource.sampleAnchor()
        manifest["first_pts"] = firstPTS.map { CMTimeGetSeconds($0) } ?? 0
        manifest["video_frames"] = frames
        manifest["dropped_frames"] = dropped
        manifest["closed_wall"] = anchor.wallSeconds
        manifest["closed_boot"] = anchor.bootSeconds
        write(manifest, to: url)
    }

    private func write(_ object: [String: Any], to url: URL) {
        guard let data = try? JSONSerialization.data(withJSONObject: object,
                                                     options: [.prettyPrinted, .sortedKeys])
        else { return }
        try? data.write(to: url, options: .atomic)
    }

    private static func hardwareModel() -> String {
        var info = utsname()
        uname(&info)
        return withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
    }

    private static func f(_ value: Double, _ places: Int = 6) -> String {
        value.isFinite ? String(format: "%.\(places)f", value) : ""
    }
}

// MARK: - CSV

/// Buffered append-only text. Flushes on a size threshold rather than per row:
/// a 100 Hz stream is 100 write(2) calls a second otherwise, which is real cost
/// on the writer queue for no benefit.
private final class CSVWriter {
    private var handle: FileHandle?
    private var buffer = ""

    init(_ url: URL, header: String?) {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
        if let header { buffer = header + "\n" }
    }

    func row(_ fields: [String]) { line(fields.joined(separator: ",")) }

    func line(_ text: String) {
        buffer += text
        buffer += "\n"
        if buffer.utf8.count > 32_768 { flush() }
    }

    func flush() {
        guard !buffer.isEmpty, let data = buffer.data(using: .utf8) else { return }
        try? handle?.write(contentsOf: data)
        buffer.removeAll(keepingCapacity: true)
    }

    func close() {
        flush()
        try? handle?.close()
        handle = nil
    }
}
