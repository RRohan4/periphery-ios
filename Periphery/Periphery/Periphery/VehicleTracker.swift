//  VehicleTracker.swift
//  Exact Swift port of periphery/perception/associate.py and heading_state.py.

import Foundation

struct EgoDelta {
    var dx: Double = 0
    var dy: Double = 0
    var dyaw: Double = 0
    var dt: Double = 0
    var valid: Bool = true
}

enum VehicleMotionState: String, Codable {
    case unknown
    case parked
    case moving
}

enum VehicleHeadingSource: String, Codable {
    case model
    case motion
}

struct VehicleTrailPoint {
    var timestamp: TimeInterval
    var x: Double
    var y: Double
}

struct TrackedVehicle {
    var id: Int
    var x: Double
    var y: Double
    var z: Double
    var length: Double
    var width: Double
    var height: Double
    var yaw: Double
    var modelYaw: Double
    var headingSource: VehicleHeadingSource
    var label: Int
    var score: Float
    var hits: Int
    var observed: Bool
    var ageSeconds: Double
    var state: VehicleMotionState
    var velocity: SIMD2<Double>?
    var trail: [VehicleTrailPoint]
}

struct VehicleTrackerConfiguration {
    var sigmaRadialA = 0.18
    var sigmaRadialB = 0.0123
    var sigmaLateralA = 0.08
    var sigmaLateralB = 0.005
    var rangeAwareGate = true
    var gateFloor = 2.0
    var gateSpeed = 12.0
    var gateSigmas = 4.0
    var confirmHits = 3
    var holdSeconds = 0.5
    var coastMinimumHits: Int? = 8
    var baselineSeconds = 1.0
    var stateObservations: Int? = 5
    var predictionObservations: Int? = 4
    var parkedBelow = 1.0
    var parkedSigmas = 1.5
    var parkedConfirmationFrames = 2
    var outputVelocityAlpha = 0.35
    var stateMaximumRange = 40.0
    var coastFreezeSeconds = 0.2
    var suppressOverlap = true
    var overlapShrink = 1.0
    var trailSeconds = 1.5
    var headingBiasDegrees = 3.24
    var motionHeadingMinimumSpeed = 5.0
}

final class VehicleTracker {
    private struct WorldSample {
        var t: Double
        var x: Double
        var y: Double
    }

    private struct Shape {
        var length: Double
        var width: Double
        var height: Double
    }

    private final class Track {
        var id: Int
        var x: Double
        var y: Double
        var z: Double
        var yaw: Double
        var length: Double
        var width: Double
        var height: Double
        var label: Int
        var score: Float
        var hits = 1
        var age = 0
        var misses = 0
        var firstT: Double
        var lastSeenT: Double
        var state = VehicleMotionState.unknown
        var parkedVotes = 0
        var outputWorldVelocity: SIMD2<Double>?
        var world: [WorldSample] = []
        var shapes: [Shape]

        init(id: Int, detection: Detection, t: Double) {
            self.id = id
            x = detection.x; y = detection.y; z = detection.z; yaw = detection.yaw
            length = detection.length; width = detection.width; height = detection.height
            label = detection.label; score = detection.score
            firstT = t; lastSeenT = t
            shapes = [Shape(length: detection.length, width: detection.width,
                            height: detection.height)]
        }

        var observed: Bool { misses == 0 }
    }

    private struct Pose2D {
        var x = 0.0
        var y = 0.0
        var yaw = 0.0

        mutating func advance(_ ego: EgoDelta) {
            let c = cos(yaw), s = sin(yaw)
            x += c * ego.dx - s * ego.dy
            y += s * ego.dx + c * ego.dy
            yaw += ego.dyaw
        }

        func toWorld(x ex: Double, y ey: Double) -> SIMD2<Double> {
            let c = cos(yaw), s = sin(yaw)
            return SIMD2(x + c * ex - s * ey, y + s * ex + c * ey)
        }

        func toEgo(x wx: Double, y wy: Double) -> SIMD2<Double> {
            let c = cos(-yaw), s = sin(-yaw)
            let tx = wx - x, ty = wy - y
            return SIMD2(c * tx - s * ty, s * tx + c * ty)
        }
    }

    let configuration: VehicleTrackerConfiguration
    private var tracks: [Track] = []
    private var pose = Pose2D()
    private var nextID = 0
    private(set) var births = 0
    private(set) var matches = 0
    private(set) var detectionSuppressed = 0
    private(set) var trackSuppressed = 0
    var suppressed: Int { detectionSuppressed + trackSuppressed }

    init(configuration: VehicleTrackerConfiguration = VehicleTrackerConfiguration()) {
        self.configuration = configuration
    }

    func reset() {
        tracks.removeAll(keepingCapacity: true)
        pose = Pose2D()
        nextID = 0
        births = 0; matches = 0
        detectionSuppressed = 0; trackSuppressed = 0
    }

    func step(detections input: [Detection], ego: EgoDelta,
              timestamp t: TimeInterval) -> [TrackedVehicle] {
        var detections = input
        if configuration.suppressOverlap {
            let filtered = filterOverlapping(detections)
            detections = filtered.detections
            detectionSuppressed += filtered.removed
        }

        predict(ego)
        pose.advance(ego)

        let pairs = match(detections, dt: max(ego.dt, 1e-3),
                          egoStep: hypot(ego.dx, ego.dy))
        let matchedTracks = Set(pairs.map { $0.0 })
        let matchedDetections = Set(pairs.map { $0.1 })
        matches += pairs.count

        for (trackIndex, detectionIndex) in pairs {
            let track = tracks[trackIndex]
            let detection = detections[detectionIndex]
            track.x = detection.x; track.y = detection.y; track.z = detection.z
            track.yaw = detection.yaw
            track.shapes.append(Shape(length: detection.length, width: detection.width,
                                      height: detection.height))
            let shape = robustShape(track.shapes)
            track.length = shape.length; track.width = shape.width; track.height = shape.height
            track.label = detection.label; track.score = detection.score
            track.hits += 1; track.misses = 0; track.lastSeenT = t
        }

        for index in tracks.indices where !matchedTracks.contains(index) {
            tracks[index].misses += 1
        }

        for index in detections.indices where !matchedDetections.contains(index) {
            nextID += 1; births += 1
            tracks.append(Track(id: nextID, detection: detections[index], t: t))
        }

        let keepSamples = configuration.baselineSeconds + configuration.trailSeconds
        for track in tracks {
            if track.observed {
                let world = pose.toWorld(x: track.x, y: track.y)
                track.world.append(WorldSample(t: t, x: world.x, y: world.y))
            }
            let cutoff = t - keepSamples - 0.5
            if let first = track.world.first, first.t < cutoff {
                track.world.removeAll { $0.t < cutoff }
            }
        }

        tracks.removeAll { track in
            t - track.lastSeenT > configuration.holdSeconds
                || (track.misses > 0
                    && configuration.coastMinimumHits != nil
                    && track.hits < configuration.coastMinimumHits!)
        }
        if configuration.suppressOverlap {
            trackSuppressed += suppressTrackOverlaps()
        }
        labelStates(at: t)
        return confirmed(at: t)
    }

    // MARK: Prediction and association

    private func predict(_ ego: EgoDelta) {
        let dt = max(ego.dt, 1e-3)
        let c = cos(-ego.dyaw), s = sin(-ego.dyaw)
        for track in tracks {
            let velocity = egoVelocity(track) ?? .zero
            let tx = track.x + velocity.x * dt - ego.dx
            let ty = track.y + velocity.y * dt - ego.dy
            track.x = c * tx - s * ty
            track.y = s * tx + c * ty
            track.yaw -= ego.dyaw
            track.age += 1
        }
    }

    private func match(_ detections: [Detection], dt: Double,
                       egoStep: Double) -> [(Int, Int)] {
        guard !tracks.isEmpty, !detections.isEmpty else { return [] }
        var costs = Array(repeating: Array(repeating: 1e6, count: detections.count),
                          count: tracks.count)
        for (i, track) in tracks.enumerated() {
            let gate = gateMetres(dt: dt, misses: track.misses) + motionSigma(track, egoStep)
            for (j, detection) in detections.enumerated()
                where hypot(detection.x - track.x, detection.y - track.y) <= gate {
                costs[i][j] = normalizedDistance(track, detection, egoStep)
            }
        }
        let limit = configuration.rangeAwareGate
            ? configuration.gateSigmas : gateMetres(dt: dt)
        return HungarianAssignment.solve(costs).filter { costs[$0.0][$0.1] < limit }
    }

    private func gateMetres(dt: Double, misses: Int = 0) -> Double {
        (configuration.gateFloor + configuration.gateSpeed * max(dt, 0))
            * Double(1 + misses)
    }

    private func sigmas(x: Double, y: Double) -> (radial: Double, lateral: Double, range: Double) {
        let range = hypot(x, y)
        return (configuration.sigmaRadialA + configuration.sigmaRadialB * range,
                configuration.sigmaLateralA + configuration.sigmaLateralB * range,
                range)
    }

    private func motionSigma(_ track: Track, _ egoStep: Double) -> Double {
        if configuration.predictionObservations != nil {
            return worldVelocity(track, over: configuration.baselineSeconds) == nil ? egoStep : 0
        }
        return predictionVelocity(track) == nil ? egoStep : 0
    }

    private func normalizedDistance(_ track: Track, _ detection: Detection,
                                    _ egoStep: Double) -> Double {
        let dx = detection.x - track.x, dy = detection.y - track.y
        guard configuration.rangeAwareGate else { return hypot(dx, dy) }
        let sigma = sigmas(x: track.x, y: track.y)
        let motion = motionSigma(track, egoStep)
        let radialSigma = hypot(sigma.radial, motion)
        let lateralSigma = hypot(sigma.lateral, 0.25 * motion)
        guard sigma.range >= 1e-6 else {
            return hypot(dx / radialSigma, dy / lateralSigma)
        }
        let cosBearing = track.x / sigma.range, sinBearing = track.y / sigma.range
        let radial = dx * cosBearing + dy * sinBearing
        let lateral = -dx * sinBearing + dy * cosBearing
        return hypot(radial / radialSigma, lateral / lateralSigma)
    }

    // MARK: Velocity and state

    private func predictionVelocity(_ track: Track) -> SIMD2<Double>? {
        if let count = configuration.predictionObservations {
            return worldVelocity(track, observations: count)
        }
        return worldVelocity(track, over: configuration.baselineSeconds)
    }

    private func stateVelocity(_ track: Track) -> SIMD2<Double>? {
        if let count = configuration.stateObservations {
            return fittedWorldVelocity(track, observations: count)
        }
        return worldVelocity(track, over: configuration.baselineSeconds)
    }

    private func egoVelocity(_ track: Track) -> SIMD2<Double>? {
        if track.state == .parked { return .zero }
        guard let velocity = predictionVelocity(track) else { return nil }
        return rotateToEgo(velocity)
    }

    private func outputVelocity(_ track: Track) -> SIMD2<Double>? {
        if track.state == .parked { return .zero }
        guard let velocity = track.outputWorldVelocity else { return nil }
        return rotateToEgo(velocity)
    }

    private func rotateToEgo(_ velocity: SIMD2<Double>) -> SIMD2<Double> {
        let c = cos(-pose.yaw), s = sin(-pose.yaw)
        return SIMD2(c * velocity.x - s * velocity.y,
                     s * velocity.x + c * velocity.y)
    }

    private func worldVelocity(_ track: Track, over baseline: Double) -> SIMD2<Double>? {
        guard track.world.count >= 2, let end = track.world.last else { return nil }
        let cutoff = end.t - baseline
        var oldest: WorldSample?
        for sample in track.world {
            if sample.t <= cutoff { oldest = sample } else { break }
        }
        if oldest == nil { oldest = track.world.first }
        guard let start = oldest else { return nil }
        let span = end.t - start.t
        guard span >= 0.8 * baseline else { return nil }
        return SIMD2((end.x - start.x) / span, (end.y - start.y) / span)
    }

    private func worldVelocity(_ track: Track, observations: Int) -> SIMD2<Double>? {
        guard observations >= 2, track.world.count >= observations,
              let end = track.world.last else { return nil }
        let start = track.world[track.world.count - observations]
        let span = end.t - start.t
        guard span > 0 else { return nil }
        return SIMD2((end.x - start.x) / span, (end.y - start.y) / span)
    }

    private func fittedWorldVelocity(_ track: Track, observations: Int) -> SIMD2<Double>? {
        guard observations >= 2, track.world.count >= observations else { return nil }
        let samples = track.world.suffix(observations)
        let count = Double(observations)
        let meanT = samples.reduce(0) { $0 + $1.t } / count
        let denominator = samples.reduce(0) { $0 + ($1.t - meanT) * ($1.t - meanT) }
        guard denominator > 0 else { return nil }
        let meanX = samples.reduce(0) { $0 + $1.x } / count
        let meanY = samples.reduce(0) { $0 + $1.y } / count
        let vx = samples.reduce(0) { $0 + ($1.t - meanT) * ($1.x - meanX) } / denominator
        let vy = samples.reduce(0) { $0 + ($1.t - meanT) * ($1.y - meanY) } / denominator
        return SIMD2(vx, vy)
    }

    private func labelStates(at t: Double) {
        for track in tracks {
            if t - track.lastSeenT > configuration.coastFreezeSeconds { continue }
            let velocity = stateVelocity(track)
            let speed = velocity.map { hypot($0.x, $0.y) }
            if track.observed, let velocity {
                if let old = track.outputWorldVelocity {
                    let alpha = configuration.outputVelocityAlpha
                    track.outputWorldVelocity = old + (velocity - old) * alpha
                } else {
                    track.outputWorldVelocity = velocity
                }
            }
            if speed == nil || hypot(track.x, track.y) > configuration.stateMaximumRange {
                track.state = .unknown
                track.parkedVotes = 0
            } else if speed! < parkedThreshold(track, span: velocitySpan(track)) {
                track.parkedVotes += 1
                if track.parkedVotes >= configuration.parkedConfirmationFrames {
                    track.state = .parked
                }
            } else {
                track.state = .moving
                track.parkedVotes = 0
            }
        }
    }

    private func velocitySpan(_ track: Track) -> Double? {
        guard let observations = configuration.stateObservations,
              track.world.count >= observations, let last = track.world.last else { return nil }
        return max(last.t - track.world[track.world.count - observations].t, 1e-3)
    }

    private func parkedThreshold(_ track: Track, span: Double?) -> Double {
        let sigma = sigmas(x: track.x, y: track.y).radial
        let baseline = span ?? configuration.baselineSeconds
        return max(configuration.parkedBelow,
                   configuration.parkedSigmas * sqrt(2) * sigma / baseline)
    }

    // MARK: Shapes and overlap suppression

    private func robustShape(_ shapes: [Shape]) -> Shape {
        func median(_ values: [Double]) -> Double {
            let sorted = values.sorted()
            let middle = sorted.count / 2
            return sorted.count.isMultiple(of: 2)
                ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        }
        let target = Shape(length: median(shapes.map(\.length)),
                           width: median(shapes.map(\.width)),
                           height: median(shapes.map(\.height)))
        return shapes.enumerated().min { lhs, rhs in
            shapeDistance(lhs.element, target) < shapeDistance(rhs.element, target)
        }!.element
    }

    private func shapeDistance(_ shape: Shape, _ target: Shape) -> Double {
        abs(log(max(shape.length, 1e-3) / max(target.length, 1e-3)))
            + abs(log(max(shape.width, 1e-3) / max(target.width, 1e-3)))
            + abs(log(max(shape.height, 1e-3) / max(target.height, 1e-3)))
    }

    private func filterOverlapping(_ detections: [Detection])
        -> (detections: [Detection], removed: Int) {
        guard detections.count >= 2 else { return (detections, 0) }
        let order = detections.indices.sorted { lhs, rhs in
            let lr = detections[lhs].range, rr = detections[rhs].range
            if lr != rr { return lr < rr }
            if detections[lhs].score != detections[rhs].score {
                return detections[lhs].score > detections[rhs].score
            }
            return lhs < rhs
        }
        var kept: [(index: Int, corners: [SIMD2<Double>], family: Int)] = []
        for index in order {
            let detection = detections[index]
            let corners = Self.corners(x: detection.x, y: detection.y, yaw: detection.yaw,
                                       length: detection.length * configuration.overlapShrink,
                                       width: detection.width * configuration.overlapShrink)
            let family = Self.family(detection.label)
            if kept.contains(where: { $0.family == family
                && Self.overlap(corners, $0.corners) }) { continue }
            kept.append((index, corners, family))
        }
        let indices = kept.map { $0.index }.sorted()
        return (indices.map { detections[$0] }, detections.count - indices.count)
    }

    private func suppressTrackOverlaps() -> Int {
        guard tracks.count >= 2 else { return 0 }
        let ordered = tracks.sorted { lhs, rhs in
            let lr = hypot(lhs.x, lhs.y), rr = hypot(rhs.x, rhs.y)
            if lr != rr { return lr < rr }
            if lhs.hits != rhs.hits { return lhs.hits > rhs.hits }
            return lhs.id < rhs.id
        }
        var kept: [(track: Track, corners: [SIMD2<Double>], family: Int)] = []
        var dropped = Set<ObjectIdentifier>()
        for track in ordered {
            let corners = Self.corners(x: track.x, y: track.y, yaw: track.yaw,
                                       length: track.length * configuration.overlapShrink,
                                       width: track.width * configuration.overlapShrink)
            let family = Self.family(track.label)
            if kept.contains(where: { $0.family == family
                && Self.overlap(corners, $0.corners) }) {
                dropped.insert(ObjectIdentifier(track)); continue
            }
            kept.append((track, corners, family))
        }
        tracks.removeAll { dropped.contains(ObjectIdentifier($0)) }
        return dropped.count
    }

    private static func family(_ label: Int) -> Int {
        if label == 3 { return 1 }
        if label == 2 { return 2 }
        return 0
    }

    private static func corners(x: Double, y: Double, yaw: Double,
                                length: Double, width: Double) -> [SIMD2<Double>] {
        let c = cos(yaw), s = sin(yaw)
        let half = [SIMD2(0.5, 0.5), SIMD2(0.5, -0.5),
                    SIMD2(-0.5, -0.5), SIMD2(-0.5, 0.5)]
        return half.map { point in
            let lx = point.x * length, ly = point.y * width
            return SIMD2(x + lx * c - ly * s, y + lx * s + ly * c)
        }
    }

    private static func overlap(_ a: [SIMD2<Double>], _ b: [SIMD2<Double>]) -> Bool {
        for quad in [a, b] {
            for i in quad.indices {
                let edge = quad[(i + 1) % quad.count] - quad[i]
                let axis = SIMD2(-edge.y, edge.x)
                let pa = a.map { $0.x * axis.x + $0.y * axis.y }
                let pb = b.map { $0.x * axis.x + $0.y * axis.y }
                if pa.max()! < pb.min()! || pb.max()! < pa.min()! { return false }
            }
        }
        return true
    }

    // MARK: Public output

    private func confirmed(at t: Double) -> [TrackedVehicle] {
        tracks.filter { $0.hits >= configuration.confirmHits }.map { track in
            let velocity = outputVelocity(track)
            let heading: Double
            let source: VehicleHeadingSource
            if let velocity, hypot(velocity.x, velocity.y) >= configuration.motionHeadingMinimumSpeed {
                heading = Self.wrap(atan2(velocity.y, velocity.x)); source = .motion
            } else {
                heading = Self.wrap(track.yaw
                    - configuration.headingBiasDegrees * Double.pi / 180)
                source = .model
            }
            var trail: [VehicleTrailPoint] = []
            for sample in track.world where t - sample.t <= configuration.trailSeconds {
                if let last = trail.last, sample.t - last.timestamp < 0.1 { continue }
                let point = pose.toEgo(x: sample.x, y: sample.y)
                trail.append(VehicleTrailPoint(timestamp: sample.t, x: point.x, y: point.y))
            }
            return TrackedVehicle(
                id: track.id, x: track.x, y: track.y, z: track.z,
                length: track.length, width: track.width, height: track.height,
                yaw: heading, modelYaw: track.yaw, headingSource: source,
                label: track.label, score: track.score, hits: track.hits,
                observed: track.observed, ageSeconds: t - track.firstT,
                state: track.state, velocity: velocity, trail: trail)
        }
    }

    private static func wrap(_ angle: Double) -> Double {
        var value = (angle + Double.pi).truncatingRemainder(dividingBy: 2 * Double.pi)
        if value < 0 { value += 2 * Double.pi }
        return value - Double.pi
    }
}

/// Rectangular minimum-cost assignment. This is the same global optimization
/// used by scipy's linear_sum_assignment, not a greedy nearest-neighbour pass.
private enum HungarianAssignment {
    static func solve(_ costs: [[Double]]) -> [(Int, Int)] {
        guard let columns = costs.first?.count, !costs.isEmpty, columns > 0 else { return [] }
        if costs.count <= columns { return solveRows(costs) }
        let transposed = (0..<columns).map { column in costs.map { $0[column] } }
        return solveRows(transposed).map { ($0.1, $0.0) }.sorted { $0.0 < $1.0 }
    }

    private static func solveRows(_ a: [[Double]]) -> [(Int, Int)] {
        let n = a.count, m = a[0].count
        var u = [Double](repeating: 0, count: n + 1)
        var v = [Double](repeating: 0, count: m + 1)
        var p = [Int](repeating: 0, count: m + 1)
        var way = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            p[0] = i
            var j0 = 0
            var minv = [Double](repeating: .infinity, count: m + 1)
            var used = [Bool](repeating: false, count: m + 1)
            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta = Double.infinity, j1 = 0
                for j in 1...m where !used[j] {
                    let current = a[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j] { minv[j] = current; way[j] = j0 }
                    if minv[j] < delta { delta = minv[j]; j1 = j }
                }
                for j in 0...m {
                    if used[j] { u[p[j]] += delta; v[j] -= delta }
                    else { minv[j] -= delta }
                }
                j0 = j1
            } while p[j0] != 0
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }
        return (1...m).compactMap { j in p[j] == 0 ? nil : (p[j] - 1, j - 1) }
            .sorted { $0.0 < $1.0 }
    }
}
