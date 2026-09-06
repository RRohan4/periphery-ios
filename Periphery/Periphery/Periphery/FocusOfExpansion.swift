// Estimates mount pitch from optical flow and gyro de-rotation. RANSAC selects
// the static-scene consensus; accepted estimates may update the driving pose.

import Accelerate
import CoreMedia
import CoreVideo
import Foundation
import Vision
import simd

final class FocusOfExpansion: @unchecked Sendable {

    // MARK: - Constants that are not negotiable

    /// Flow shorter than this in the WORKING image is discarded: its direction
    /// is dominated by the flow field's own quantisation.
    static let minFlowPixels = 1.5

    static let pairSeconds = 0.100
    /// Fallback until the real frame interval is known.
    static let defaultFrameStride = 3
    /// Working resolution for the flow field. Vision returns a buffer the size
    /// of its input, so 1920x1080 would be 16 MB per frame.
    static let workingWidth = 448
    /// Samples are taken on this grid in the working image.
    static let sampleStride = 4
    /// Consensus below this is not an estimate, it is a coincidence.
    static let minInliers = 30

    // MARK: - Gates

    struct Gates: Sendable, Equatable {
        /// Ground speed floor, m/s. Negative disables the gate entirely.
        var minSpeed: Double
        /// Above this the pure-expansion model stops holding well enough to be
        /// worth the frames. De-rotation handles what is left below it.
        var maxYawRateDegrees: Double
        /// Pairs the reported median is taken over.
        var windowSize: Int
        /// Enough of the window to report at all.
        var minSamplesToReport: Int
        /// ...and to hand to the pose without a person asking.
        var minSamplesToApply: Int
        /// May an estimate from this profile reach MountPose?
        var writesToPose: Bool
        var name: String

        /// The validated profile. Changing any of these invalidates the numbers
        /// in this file's header.
        static let driving = Gates(minSpeed: 8.0,
                                   maxYawRateDegrees: 1.5,
                                   windowSize: 400,
                                   minSamplesToReport: 60,
                                   minSamplesToApply: 200,
                                   writesToPose: true,
                                   name: "driving")

        static let handheld = Gates(minSpeed: -1.0,
                                    maxYawRateDegrees: 8.0,
                                    windowSize: 60,
                                    minSamplesToReport: 8,
                                    minSamplesToApply: .max,
                                    writesToPose: false,
                                    name: "handheld")
    }

    /// Read on the capture queue and on `queue`; written from the main actor
    /// when a tab switches profile. A torn read would cost one frame.
    private var _gates = Gates.driving
    var gates: Gates {
        get { lock.withLock { _gates } }
        set {
            lock.withLock { _gates = newValue }
            reset()
        }
    }

    // MARK: - What the UI reads

    struct Estimate: Sendable {
        /// Mount pitch against travel, degrees, positive nose-up. The number.
        var pitchDegrees: Double = 0
        /// Mount yaw, degrees, positive camera-points-left. Falls out of the
        /// same fit at no extra cost, and needs no compass.
        var yawDegrees: Double = 0
        /// Robust spread of the window: half the 16th-84th percentile gap,
        /// which is a one-sigma equivalent that outliers cannot inflate.
        var sigmaDegrees: Double = 0

        var pitchWithoutDerotationDegrees: Double = 0
        /// Samples in the window, out of `windowSize`.
        var samples: Int = 0
        /// Median inlier count of the accepted fits. Low means the scene is
        /// thin; zero-consensus frames never get here at all.
        var inliers: Int = 0
        /// The profile this estimate came from, carried along so a consumer
        /// cannot apply a handheld reading to the mount by accident.
        var gates = Gates.driving
        var windowSize: Int { gates.windowSize }
        /// True once the window is long enough to hand to the pose.
        var converged: Bool {
            gates.writesToPose && samples >= gates.minSamplesToApply
        }
        var reportable: Bool { samples >= gates.minSamplesToReport }
        /// Why nothing is being accumulated right now, or "" when it is.
        var gate: String = "waiting for a frame"
        /// Rolling accept rate over recent pairs, so a scene that is being
        /// silently rejected (night) looks different from one that is gated.
        var acceptRate: Double = 0
    }

    struct Debug: Sendable {
        /// Working-image size the coordinates below are in.
        var width = 0
        var height = 0
        /// The fit, in working-image pixels. Nil when there was no consensus --
        /// which is itself the interesting case.
        var foe: SIMD2<Double>?
        /// Sampled flow, thinned for drawing. `vectors` is AFTER de-rotation,
        /// so what is drawn is what the fit actually consumed.
        var points: [SIMD2<Float>] = []
        var vectors: [SIMD2<Float>] = []

        var inlier: [Bool] = []

        var medianFoe: SIMD2<Double>?
        /// This pair alone, not the window median.
        var pitchDegrees: Double = 0
        var yawDegrees: Double = 0
        var inliers = 0
        var total = 0
        /// Rotation subtracted from this pair, camera axes, deg/s.
        var rotationDegreesPerSecond = SIMD3<Double>()
        var note = ""
    }

    /// Set from the Flow tab. Costs a copy per pair, so it stays off otherwise.
    var publishDebug = false

    /// Called on the estimator's own queue after every processed pair.
    var onEstimate: ((Estimate) -> Void)?
    /// Called on the estimator's own queue when `publishDebug` is set.
    var onDebug: ((Debug) -> Void)?

    private let lock = NSLock()
    private var _estimate = Estimate()
    var estimate: Estimate { lock.withLock { _estimate } }

    // MARK: - Machinery

    private let queue = DispatchQueue(label: "com.periphery.foe", qos: .utility)
    /// Under `lock`: written from the capture queue, cleared from `queue`.
    private var _busy = false
    private var frameCounter = 0
    /// Derived from the measured capture interval so the pair spacing is the
    /// same duration whatever rate the camera is running at.
    private var frameStride = FocusOfExpansion.defaultFrameStride
    private var lastFeedTime: Double = 0
    private var smoothedInterval: Double = 0

    private var previous: CVPixelBuffer?
    private var previousTime: Double = 0
    private var pool: CVPixelBufferPool?
    private var poolWidth = 0
    private var poolHeight = 0

    private struct Optics {
        var fx = 1.0, fy = 1.0, cx = 0.0, cy = 0.0
        init() {}
        init(K: simd_double3x3, sourceWidth: Int) {
            let s = Double(FocusOfExpansion.workingWidth) / Double(sourceWidth)
            fx = K[0][0] * s
            fy = K[1][1] * s
            cx = K[2][0] * s
            cy = K[2][1] * s
        }
    }
    /// Written and read on `queue` only.
    private var optics = Optics()
    private var fx: Double { optics.fx }
    private var fy: Double { optics.fy }
    private var cx: Double { optics.cx }
    private var cy: Double { optics.cy }

    /// (timestamp, rotationRate in DEVICE axes), for averaging over a pair's
    /// interval. 100 Hz for a couple of seconds.
    private var gyro = [(t: Double, w: SIMD3<Double>)]()

    private var foeWindow = [SIMD2<Double>]()
    private var pitchWindow = [Double]()
    private var yawWindow = [Double]()
    private var rawWindow = [Double]()          // no de-rotation, diagnostic only
    private var inlierWindow = [Int]()
    private var attempts = [Bool]()

    // MARK: - Input

    /// Every attitude sample, from MotionSource's queue.
    func append(rotationRate: SIMD3<Double>, at timestamp: Double) {
        lock.withLock {
            gyro.append((timestamp, rotationRate))
            // Two seconds is far more than any pair interval needs.
            if gyro.count > 256 { gyro.removeFirst(gyro.count - 256) }
        }
    }

    func reset() {
        queue.async {
            self.previous = nil
            self.foeWindow.removeAll()
            self.pitchWindow.removeAll()
            self.yawWindow.removeAll()
            self.rawWindow.removeAll()
            self.inlierWindow.removeAll()
            self.attempts.removeAll()
            self.publish(gate: "reset")
        }
    }

    func feed(frame: CameraSession.Frame,
              calibration: Calibration,
              speed: Double) {
        let now = CMTimeGetSeconds(frame.presentationTime)
        updateStride(at: now)
        frameCounter += 1
        guard frameCounter % frameStride == 0 else { return }

        let floor = gates.minSpeed
        if floor >= 0, !(speed >= floor) {
            publishGate(String(format: "speed %.1f m/s < %.0f", max(speed, 0), floor))
            return
        }
        // Claim the slot before doing any work, and drop the frame if the
        // previous pair is still in Vision.
        let claimed = lock.withLock { () -> Bool in
            if _busy { return false }
            _busy = true
            return true
        }
        guard claimed else { return }

        let time = now
        guard let small = downscale(frame.pixelBuffer,
                                    sourceWidth: frame.width,
                                    sourceHeight: frame.height) else {
            lock.withLock { _busy = false }
            return
        }
        let optics = Optics(K: calibration.K, sourceWidth: frame.width)
        let roll = calibration.roll
        queue.async { [weak self] in
            guard let self else { return }
            defer { self.lock.withLock { self._busy = false } }
            self.optics = optics
            self.process(current: small, at: time, roll: roll)
        }
    }

    private func updateStride(at time: Double) {
        defer { lastFeedTime = time }
        guard lastFeedTime > 0 else { return }
        let delta = time - lastFeedTime
        // Ignore a gap from backgrounding or a dropped run of frames.
        guard delta > 0.001, delta < 0.2 else { return }
        smoothedInterval = smoothedInterval == 0
            ? delta
            : smoothedInterval * 0.95 + delta * 0.05
        guard smoothedInterval > 0 else { return }
        frameStride = max(1, Int((Self.pairSeconds / smoothedInterval).rounded()))
    }

    // MARK: - The pair

    private func process(current: CVPixelBuffer, at time: Double, roll: Double) {
        defer { previous = current; previousTime = time }
        guard let first = previous else { return }
        let dt = time - previousTime
        // A stale pair -- the app was backgrounded, or frames were dropped --
        // is not a measurement.
        guard dt > 0.01, dt < 0.5 else { return }

        // Average the gyro over exactly this interval. Rotation is the one
        // thing that can move the focus of expansion without the car turning.
        let w = averageRotationRate(from: previousTime, to: time)

        let wCam = SIMD3<Double>(-w.y, -w.x, -w.z)

        let yawRate = abs(wCam.y) * 180.0 / .pi
        guard yawRate <= gates.maxYawRateDegrees else {
            publishGate(String(format: "turning %.1f deg/s", yawRate))
            return
        }

        guard let flow = opticalFlow(from: first, to: current) else {
            publishGate("flow unavailable")
            emitDebug { $0.note = "flow unavailable" }
            return
        }
        let flowWidth = CVPixelBufferGetWidth(flow)
        let flowHeight = CVPixelBufferGetHeight(flow)
        var points = [SIMD2<Double>]()
        var vectors = [SIMD2<Double>]()
        sample(flow: flow, points: &points, vectors: &vectors)
        guard points.count > 3 * Self.minInliers else {
            record(accepted: false)
            publishGate("only \(points.count) usable flow vectors")
            emitDebug {
                $0.width = flowWidth; $0.height = flowHeight
                $0.total = points.count
                $0.note = "too little flow — hold the phone still and walk forward"
            }
            return
        }

        let derotated = derotate(points: points, vectors: vectors, omega: wCam, dt: dt)
        guard let fit = ransac(points: points, vectors: derotated) else {
            record(accepted: false)
            // The night failure mode lands exactly here: plenty of flow, no
            // agreement about where it comes from.
            publishGate("no consensus (\(points.count) vectors)")
            emitDebug {
                $0.width = flowWidth; $0.height = flowHeight
                $0.total = points.count
                $0.rotationDegreesPerSecond = wCam * 180.0 / .pi
                $0.note = "no consensus — nothing in view agrees on a direction"
                self.thin(points: points, vectors: derotated, inlier: nil, into: &$0)
            }
            return
        }
        // The same fit without the rotation correction, as a running check on
        // the axis mapping above. Diagnostic only -- never fed to the pose.
        let rawFit = ransac(points: points, vectors: vectors)

        let angles = pitchAndYaw(foe: fit.foe, roll: roll)
        record(accepted: true)
        append(foe: fit.foe, pitch: angles.pitch, yaw: angles.yaw,
               raw: rawFit.map { pitchAndYaw(foe: $0.foe, roll: roll).pitch },
               inliers: fit.inliers)
        let median = medianFoe()
        emitDebug {
            $0.width = flowWidth; $0.height = flowHeight
            $0.foe = fit.foe
            $0.medianFoe = median
            $0.pitchDegrees = angles.pitch * 180.0 / .pi
            $0.yawDegrees = angles.yaw * 180.0 / .pi
            $0.inliers = fit.inliers
            $0.total = points.count
            $0.rotationDegreesPerSecond = wCam * 180.0 / .pi
            self.thin(points: points, vectors: derotated,
                      inlier: self.inlierMask(points: points, vectors: derotated,
                                              foe: fit.foe),
                      into: &$0)
        }
    }

    // MARK: - Debug payload

    private func emitDebug(_ fill: (inout Debug) -> Void) {
        guard publishDebug, let onDebug else { return }
        var debug = Debug()
        fill(&debug)
        onDebug(debug)
    }

    /// At most `limit` evenly spaced samples. Drawing two thousand arrows is
    /// slower than computing them and reads as a smear.
    private func thin(points: [SIMD2<Double>],
                      vectors: [SIMD2<Double>],
                      inlier: [Bool]?,
                      into debug: inout Debug,
                      limit: Int = 260) {
        let step = max(1, points.count / limit)
        for i in stride(from: 0, to: points.count, by: step) {
            debug.points.append(SIMD2<Float>(Float(points[i].x), Float(points[i].y)))
            debug.vectors.append(SIMD2<Float>(Float(vectors[i].x), Float(vectors[i].y)))
            debug.inlier.append(inlier?[i] ?? false)
        }
    }

    /// Which samples agreed with the winning focus, by the same residual the
    /// fit used.
    private func inlierMask(points: [SIMD2<Double>],
                            vectors: [SIMD2<Double>],
                            foe: SIMD2<Double>,
                            threshold: Double = 2.0) -> [Bool] {
        var mask = [Bool](repeating: false, count: points.count)
        for i in points.indices {
            let length = (vectors[i].x * vectors[i].x
                          + vectors[i].y * vectors[i].y).squareRoot()
            guard length > 1e-6 else { continue }
            let nx = -vectors[i].y / length, ny = vectors[i].x / length
            let offset = nx * points[i].x + ny * points[i].y
            mask[i] = abs(nx * foe.x + ny * foe.y - offset) < threshold
        }
        return mask
    }

    // MARK: - Sampling the flow field

    private func sample(flow: CVPixelBuffer,
                        points: inout [SIMD2<Double>],
                        vectors: inout [SIMD2<Double>]) {
        CVPixelBufferLockBaseAddress(flow, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(flow, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(flow) else { return }
        let width = CVPixelBufferGetWidth(flow)
        let height = CVPixelBufferGetHeight(flow)
        let rowBytes = CVPixelBufferGetBytesPerRow(flow)
        let minimum = Self.minFlowPixels * Self.minFlowPixels

        points.reserveCapacity(4096)
        vectors.reserveCapacity(4096)
        for y in stride(from: 0, to: height, by: Self.sampleStride) {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float.self)
            for x in stride(from: 0, to: width, by: Self.sampleStride) {
                let dx = Double(row[x * 2])
                let dy = Double(row[x * 2 + 1])
                guard dx.isFinite, dy.isFinite else { continue }
                guard dx * dx + dy * dy >= minimum else { continue }
                points.append(SIMD2<Double>(Double(x), Double(y)))
                vectors.append(SIMD2<Double>(dx, dy))
            }
        }
    }

    // MARK: - De-rotation

    private func derotate(points: [SIMD2<Double>],
                          vectors: [SIMD2<Double>],
                          omega: SIMD3<Double>,
                          dt: Double) -> [SIMD2<Double>] {
        let wx = omega.x * dt, wy = omega.y * dt, wz = omega.z * dt
        var out = [SIMD2<Double>]()
        out.reserveCapacity(vectors.count)
        for i in points.indices {
            let x = points[i].x - cx
            let y = points[i].y - cy
            let u = wx * x * y / fx - wy * (fx + x * x / fx) + wz * y
            let v = wx * (fy + y * y / fy) - wy * x * y / fy - wz * x
            out.append(SIMD2<Double>(vectors[i].x - u, vectors[i].y - v))
        }
        return out
    }

    // MARK: - RANSAC

    private func ransac(points: [SIMD2<Double>],
                        vectors: [SIMD2<Double>],
                        iterations: Int = 200,
                        threshold: Double = 2.0) -> (foe: SIMD2<Double>, inliers: Int)? {
        let n = points.count
        guard n >= Self.minInliers else { return nil }

        var normals = [SIMD2<Double>]()
        var offsets = [Double]()
        normals.reserveCapacity(n)
        offsets.reserveCapacity(n)
        for i in 0..<n {
            let length = (vectors[i].x * vectors[i].x + vectors[i].y * vectors[i].y).squareRoot()
            guard length > 1e-6 else { continue }
            let normal = SIMD2<Double>(-vectors[i].y / length, vectors[i].x / length)
            normals.append(normal)
            offsets.append(normal.x * points[i].x + normal.y * points[i].y)
        }
        let count = normals.count
        guard count >= Self.minInliers else { return nil }

        var seed: UInt64 = 0x9E3779B97F4A7C15
        func next(_ bound: Int) -> Int {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17
            return Int(seed % UInt64(bound))
        }

        var best: SIMD2<Double>?
        var bestCount = 0
        for _ in 0..<iterations {
            let a = next(count), b = next(count)
            guard a != b else { continue }
            guard let candidate = intersect(normals[a], offsets[a], normals[b], offsets[b]) else {
                continue
            }
            // Reject candidates far outside the frame before paying for a vote.
            guard abs(candidate.x - cx) < 4 * fx, abs(candidate.y - cy) < 4 * fy else { continue }
            var votes = 0
            for i in 0..<count {
                let residual = abs(normals[i].x * candidate.x
                                   + normals[i].y * candidate.y - offsets[i])
                if residual < threshold { votes += 1 }
            }
            if votes > bestCount { bestCount = votes; best = candidate }
        }
        guard let seedPoint = best, bestCount >= Self.minInliers else { return nil }

        var m00 = 0.0, m01 = 0.0, m11 = 0.0, r0 = 0.0, r1 = 0.0
        var inliers = 0
        for i in 0..<count {
            let residual = abs(normals[i].x * seedPoint.x + normals[i].y * seedPoint.y - offsets[i])
            guard residual < threshold else { continue }
            let nx = normals[i].x, ny = normals[i].y
            m00 += nx * nx; m01 += nx * ny; m11 += ny * ny
            r0 += nx * offsets[i]; r1 += ny * offsets[i]
            inliers += 1
        }
        let determinant = m00 * m11 - m01 * m01
        guard inliers >= Self.minInliers, abs(determinant) > 1e-9 else {
            return (seedPoint, bestCount)
        }
        let foe = SIMD2<Double>((m11 * r0 - m01 * r1) / determinant,
                                (m00 * r1 - m01 * r0) / determinant)
        guard foe.x.isFinite, foe.y.isFinite else { return (seedPoint, bestCount) }
        return (foe, inliers)
    }

    private func intersect(_ n0: SIMD2<Double>, _ d0: Double,
                           _ n1: SIMD2<Double>, _ d1: Double) -> SIMD2<Double>? {
        let determinant = n0.x * n1.y - n0.y * n1.x
        // Near-parallel lines intersect somewhere useless.
        guard abs(determinant) > 0.05 else { return nil }
        return SIMD2<Double>((d0 * n1.y - d1 * n0.y) / determinant,
                             (n0.x * d1 - n1.x * d0) / determinant)
    }

    // MARK: - Focus of expansion -> mount angles

    private func pitchAndYaw(foe: SIMD2<Double>, roll: Double) -> (pitch: Double, yaw: Double) {
        Self.mountAngles(foe: foe, fx: fx, fy: fy, cx: cx, cy: cy, roll: roll)
    }

    static func mountAngles(foe: SIMD2<Double>,
                            fx: Double, fy: Double, cx: Double, cy: Double,
                            roll: Double) -> (pitch: Double, yaw: Double) {
        // Pixel to a unit direction in image axes (x right, y down, z forward).
        let direction = simd_normalize(SIMD3<Double>((foe.x - cx) / fx,
                                                     (foe.y - cy) / fy,
                                                     1.0))

        let sensor = Contract.sensorToImageAxes.transpose * direction
        // Undo Rx(-roll), leaving Ry(pitch) * Rz(-yaw) applied to (1, 0, 0),
        // which is (cos p cos y, -sin y, -sin p cos y).
        let cr = cos(roll), sr = sin(roll)
        let level = SIMD3<Double>(sensor.x,
                                  cr * sensor.y - sr * sensor.z,
                                  sr * sensor.y + cr * sensor.z)
        let pitch = atan2(-level.z, level.x)
        let yaw = asin(max(-1.0, min(1.0, -level.y)))
        return (pitch, yaw)
    }

    // MARK: - The window

    private func append(foe: SIMD2<Double>, pitch: Double, yaw: Double,
                        raw: Double?, inliers: Int) {
        foeWindow.append(foe)
        pitchWindow.append(pitch * 180.0 / .pi)
        yawWindow.append(yaw * 180.0 / .pi)
        rawWindow.append((raw ?? pitch) * 180.0 / .pi)
        inlierWindow.append(inliers)
        let windowSize = gates.windowSize
        if pitchWindow.count > windowSize {
            let excess = pitchWindow.count - windowSize
            foeWindow.removeFirst(excess)
            pitchWindow.removeFirst(excess)
            yawWindow.removeFirst(excess)
            rawWindow.removeFirst(excess)
            inlierWindow.removeFirst(excess)
        }
        publish(gate: "")
    }

    /// Component-wise median, which is enough for a crosshair and cannot be
    /// dragged off by one bad fit the way a mean can.
    private func medianFoe() -> SIMD2<Double>? {
        guard !foeWindow.isEmpty else { return nil }
        return SIMD2<Double>(Self.median(foeWindow.map(\.x)),
                             Self.median(foeWindow.map(\.y)))
    }

    private func record(accepted: Bool) {
        attempts.append(accepted)
        if attempts.count > 60 { attempts.removeFirst(attempts.count - 60) }
    }

    private func publishGate(_ reason: String) {
        queue.async { [weak self] in self?.publish(gate: reason) }
    }

    private func publish(gate: String) {
        var estimate = Estimate()
        estimate.gates = gates
        estimate.samples = pitchWindow.count
        estimate.gate = gate
        if !attempts.isEmpty {
            estimate.acceptRate = Double(attempts.filter { $0 }.count) / Double(attempts.count)
        }
        if !pitchWindow.isEmpty {
            // Median, not mean: one bad fit should not move the answer, and a
            // scene full of a single moving truck can produce a bad fit.
            estimate.pitchDegrees = Self.median(pitchWindow)
            estimate.yawDegrees = Self.median(yawWindow)
            estimate.pitchWithoutDerotationDegrees = Self.median(rawWindow)
            estimate.sigmaDegrees = Self.robustSigma(pitchWindow)
            estimate.inliers = Int(Self.median(inlierWindow.map { Double($0) }))
        }
        lock.withLock { _estimate = estimate }
        onEstimate?(estimate)
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        return sorted.count % 2 == 1
            ? sorted[middle]
            : (sorted[middle - 1] + sorted[middle]) / 2
    }

    /// Half the 16th-to-84th percentile gap: the one-sigma equivalent for a
    /// distribution with outliers in it, which this one has by construction.
    private static func robustSigma(_ values: [Double]) -> Double {
        guard values.count >= 8 else { return 0 }
        let sorted = values.sorted()
        let low = sorted[Int(Double(sorted.count) * 0.16)]
        let high = sorted[min(sorted.count - 1, Int(Double(sorted.count) * 0.84))]
        return (high - low) / 2
    }

    // MARK: - Gyro averaging

    private func averageRotationRate(from start: Double, to end: Double) -> SIMD3<Double> {
        let samples = lock.withLock { gyro }
        var sum = SIMD3<Double>.zero
        var count = 0
        for sample in samples where sample.t >= start && sample.t <= end {
            sum += sample.w
            count += 1
        }
        if count > 0 { return sum / Double(count) }
        // No sample landed inside; fall back to the nearest one rather than
        // silently de-rotating by zero.
        return samples.min { abs($0.t - end) < abs($1.t - end) }?.w ?? .zero
    }

    // MARK: - Vision

    private func opticalFlow(from first: CVPixelBuffer,
                             to second: CVPixelBuffer) -> CVPixelBuffer? {
        let request = VNGenerateOpticalFlowRequest(targetedCVPixelBuffer: second)

        request.computationAccuracy = .low
        request.outputPixelFormat = kCVPixelFormatType_TwoComponent32Float
        let handler = VNImageRequestHandler(cvPixelBuffer: first, options: [:])
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        return (request.results?.first as? VNPixelBufferObservation)?.pixelBuffer
    }

    // MARK: - Downscale

    private func downscale(_ source: CVPixelBuffer,
                           sourceWidth: Int,
                           sourceHeight: Int) -> CVPixelBuffer? {
        let width = Self.workingWidth
        let height = Int((Double(sourceHeight) * Double(width) / Double(sourceWidth)).rounded())
        guard preparePool(width: width, height: height), let pool else { return nil }

        var destinationBuffer: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destinationBuffer) == kCVReturnSuccess,
              let destination = destinationBuffer else { return nil }

        CVPixelBufferLockBaseAddress(source, .readOnly)
        CVPixelBufferLockBaseAddress(destination, [])
        defer {
            CVPixelBufferUnlockBaseAddress(destination, [])
            CVPixelBufferUnlockBaseAddress(source, .readOnly)
        }
        guard let sourceBase = CVPixelBufferGetBaseAddress(source),
              let destinationBase = CVPixelBufferGetBaseAddress(destination) else { return nil }

        var input = vImage_Buffer(data: sourceBase,
                                  height: vImagePixelCount(sourceHeight),
                                  width: vImagePixelCount(sourceWidth),
                                  rowBytes: CVPixelBufferGetBytesPerRow(source))
        var output = vImage_Buffer(data: destinationBase,
                                   height: vImagePixelCount(height),
                                   width: vImagePixelCount(width),
                                   rowBytes: CVPixelBufferGetBytesPerRow(destination))
        guard vImageScale_ARGB8888(&input, &output, nil,
                                   vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError else {
            return nil
        }
        return destination
    }

    private func preparePool(width: Int, height: Int) -> Bool {
        if pool != nil, poolWidth == width, poolHeight == height { return true }
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as CFDictionary,
        ]
        var created: CVPixelBufferPool?
        guard CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary,
                                      &created) == kCVReturnSuccess else { return false }
        pool = created
        poolWidth = width
        poolHeight = height
        return true
    }
}
