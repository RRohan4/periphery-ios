// 2.5D bird's-eye renderer for confirmed tracked vehicles. It only draws the
// supplied state; association and filtering happen upstream.

import SwiftUI
import simd

// MARK: - Palette

/// The viewer's dark palette. A phone in a windscreen mount is the dark case.
private enum Palette {
    static let ground = Color(red: 0.059, green: 0.075, blue: 0.090)   // #0f1317
    static let panel2 = Color(red: 0.122, green: 0.149, blue: 0.180)   // #1f262e
    static let line = Color(red: 0.173, green: 0.208, blue: 0.243)     // #2c353e
    static let ink2 = Color(red: 0.580, green: 0.635, blue: 0.690)     // #94a2b0
    static let ink3 = Color(red: 0.420, green: 0.471, blue: 0.522)     // #6b7885
    static let vision = Color(red: 0.910, green: 0.639, blue: 0.239)   // #e8a33d
    static let visionSoft = Color(red: 0.910, green: 0.639, blue: 0.239).opacity(0.16)
    static let radar = Color(red: 0.271, green: 0.702, blue: 0.769)    // #45b3c4
    static let radarSoft = Color(red: 0.271, green: 0.702, blue: 0.769).opacity(0.16)
    static let parkedSoft = Color(red: 0.580, green: 0.635, blue: 0.690).opacity(0.12)
    static let cabin = Color(red: 0.188, green: 0.243, blue: 0.282).opacity(0.68)
    static let cabinTop = Color(red: 0.251, green: 0.322, blue: 0.369).opacity(0.72)
    static let glass = Color(red: 0.698, green: 0.827, blue: 0.863).opacity(0.32)
}

// MARK: - Vehicle profiles

struct VehicleProfile {
    let body: [SIMD2<Double>]
    let cabin: [SIMD2<Double>]
    let glass: [SIMD2<Double>]
    let bodyZ: Double
    let roofZ: Double
    let defaultHeight: Double

    static let car = VehicleProfile(
        body: p([(0.50, 0.27), (0.41, 0.49), (-0.30, 0.49), (-0.50, 0.34),
                 (-0.50, -0.34), (-0.30, -0.49), (0.41, -0.49), (0.50, -0.27)]),
        cabin: p([(0.27, 0.24), (0.14, 0.35), (-0.27, 0.34), (-0.35, 0.23),
                  (-0.35, -0.23), (-0.27, -0.34), (0.14, -0.35), (0.27, -0.24)]),
        glass: p([(0.20, 0.21), (0.10, 0.29), (-0.23, 0.28), (-0.29, 0.19),
                  (-0.29, -0.19), (-0.23, -0.28), (0.10, -0.29), (0.20, -0.21)]),
        bodyZ: 0.42, roofZ: 0.91, defaultHeight: 1.5)

    static let suv = VehicleProfile(
        body: p([(0.50, 0.34), (0.43, 0.50), (-0.38, 0.50), (-0.50, 0.38),
                 (-0.50, -0.38), (-0.38, -0.50), (0.43, -0.50), (0.50, -0.34)]),
        cabin: p([(0.31, 0.29), (0.20, 0.40), (-0.31, 0.39), (-0.40, 0.29),
                  (-0.40, -0.29), (-0.31, -0.39), (0.20, -0.40), (0.31, -0.29)]),
        glass: p([(0.23, 0.25), (0.14, 0.33), (-0.26, 0.32), (-0.34, 0.24),
                  (-0.34, -0.24), (-0.26, -0.32), (0.14, -0.33), (0.23, -0.25)]),
        bodyZ: 0.47, roofZ: 0.96, defaultHeight: 1.5)

    static let truck = VehicleProfile(
        body: p([(0.50, 0.34), (0.38, 0.50), (-0.46, 0.50), (-0.50, 0.38),
                 (-0.50, -0.38), (-0.46, -0.50), (0.38, -0.50), (0.50, -0.34)]),
        cabin: p([(0.45, 0.25), (0.30, 0.33), (-0.08, 0.33),
                  (-0.08, -0.33), (0.30, -0.33), (0.45, -0.25)]),
        glass: p([(0.39, 0.22), (0.28, 0.27), (0.01, 0.27),
                  (0.01, -0.27), (0.28, -0.27), (0.39, -0.22)]),
        bodyZ: 0.48, roofZ: 0.94, defaultHeight: 3.0)

    static let bike = VehicleProfile(
        body: p([(0.50, 0.12), (0.30, 0.24), (-0.33, 0.20), (-0.50, 0.09),
                 (-0.50, -0.09), (-0.33, -0.20), (0.30, -0.24), (0.50, -0.12)]),
        cabin: p([(0.24, 0.10), (0.08, 0.15), (-0.18, 0.13), (-0.28, 0.07),
                  (-0.28, -0.07), (-0.18, -0.13), (0.08, -0.15), (0.24, -0.10)]),
        glass: p([(0.16, 0.07), (0.05, 0.10), (-0.12, 0.09), (-0.18, 0.04),
                  (-0.18, -0.04), (-0.12, -0.09), (0.05, -0.10), (0.16, -0.07)]),
        bodyZ: 0.40, roofZ: 0.82, defaultHeight: 0.95)

    static let pedestrian = VehicleProfile(
        body: p([(0.25, 0.25), (0.25, -0.25), (-0.25, -0.25), (-0.25, 0.25)]),
        cabin: p([(0.16, 0.16), (0.16, -0.16), (-0.16, -0.16), (-0.16, 0.16)]),
        glass: [], bodyZ: 0.35, roofZ: 0.92, defaultHeight: 1.7)

    private static func p(_ pairs: [(Double, Double)]) -> [SIMD2<Double>] {
        pairs.map { SIMD2<Double>($0.0, $0.1) }
    }

    static func profile(label: Int, length: Double, width: Double) -> VehicleProfile {
        switch Contract.classNames[label] {
        case "large_vehicle": return .truck
        case "two_wheeler": return .bike
        case "pedestrian": return .pedestrian
        default: return (length > 5.2 || width > 2.1) ? .suv : .car
        }
    }

    /// The centre seam that separates windshield from rear glass at range is
    /// meaningless on these two.
    var hasSeam: Bool { !glass.isEmpty && bodyZ != 0.40 && bodyZ != 0.35 }
}

// MARK: - Framing

enum WorldFraming {

    static let tiltRange: ClosedRange<Double> = 16...45
    static let defaultTilt: Double = 20
    static let key = "world.tiltDegrees"

    static func clamp(_ degrees: Double) -> Double {
        guard degrees.isFinite else { return defaultTilt }
        return min(max(degrees, tiltRange.lowerBound), tiltRange.upperBound)
    }

    /// Past the 39.45 m where the BEV grid ends, so no framing throws away a
    /// measurement that exists.
    static let reach = 42.0
    /// The ego's own rear end; the drawn body is 4.6 m long. It is the only
    /// thing behind the car worth a pixel, and it binds the bottom edge.
    static let back = 2.6

    static let headroom = 2.6
    /// Room above that roof for the label itself, in points.
    static let labelPad = 22.0

    static let aimRange: ClosedRange<Double> = 6...24
}

// MARK: - The virtual camera

private struct WorldCamera {
    let width: Double
    let height: Double
    let camX: Double
    let camZ: Double
    let fx: Double, fz: Double
    let ux: Double, uz: Double
    let focal: Double

    init(size: CGSize, tiltDegrees: Double, sensorFocal: Double) {
        let viewWidth = Double(size.width)
        let viewHeight = Double(size.height)
        let tilt = WorldFraming.clamp(tiltDegrees) * .pi / 180.0
        // Hoisted: the solver below runs a few hundred times per draw and these
        // do not vary inside it.
        let ct = cos(tilt), st = sin(tilt)
        let targetZ = 0.9
        let f = viewHeight / (2 * tan(26.0 * .pi / 360.0))
        let margin = min(10.0, min(viewWidth, viewHeight) * 0.04)
        let top = margin + WorldFraming.labelPad

        // The wedge the detector can actually see, on the road and again at
        // roof height. See WorldFraming.
        var patch = [SIMD3<Double>(-WorldFraming.back, -1.1, 0),
                     SIMD3<Double>(-WorldFraming.back, 1.1, 0)]
        let hfov = 2 * atan(Double(Contract.inputWidth) / (2 * max(sensorFocal, 1)))
        for i in 0...8 {
            let a = -hfov / 2 + hfov * Double(i) / 8
            let x = WorldFraming.reach * cos(a), y = WorldFraming.reach * sin(a)
            patch.append(SIMD3<Double>(x, y, 0))
            patch.append(SIMD3<Double>(x, y, WorldFraming.headroom))
        }

        func fits(aim: Double, distance: Double) -> Bool {
            let cameraX = aim - distance * ct
            let cameraZ = targetZ + distance * st
            for p in patch {
                let dx = p.x - cameraX, dz = p.z - cameraZ
                let depth = dx * ct - dz * st
                guard depth > 0.5 else { return false }
                let sx = viewWidth / 2 - p.y * f / depth
                let sy = viewHeight / 2 - (dx * st + dz * ct) * f / depth
                if sx < margin || sx > viewWidth - margin
                    || sy < top || sy > viewHeight - margin { return false }
            }
            return true
        }
        // For a fixed aim the patch shrinks monotonically as the camera pulls
        // back, so the closest distance that still holds it is one bisection.
        func nearest(aim: Double) -> Double? {
            guard fits(aim: aim, distance: 520) else { return nil }
            if fits(aim: aim, distance: 12) { return 12 }
            var low = 12.0, high = 520.0
            for _ in 0..<26 {
                if high - low <= 0.02 { break }
                let middle = (low + high) / 2
                if fits(aim: aim, distance: middle) { high = middle } else { low = middle }
            }
            return high
        }
        // Aim has no such monotonicity, so it is scanned rather than bisected.
        // Half a metre is far finer than the eye can tell.
        var tried = [(aim: Double, distance: Double)]()
        var best = Double.infinity
        var aim = WorldFraming.aimRange.lowerBound
        while aim <= WorldFraming.aimRange.upperBound + 1e-9 {
            if let d = nearest(aim: aim) {
                tried.append((aim, d))
                best = min(best, d)
            }
            aim += 0.5
        }
        // Of everything within 1% of the closest, the LONGEST aim: the same
        // zoom for the least perspective. The scan runs short to long.
        var chosenAim = WorldFraming.reach / 2, distance = 120.0
        if let pick = tried.last(where: { $0.distance <= best * 1.01 }) {
            chosenAim = pick.aim
            distance = pick.distance
        }
        width = viewWidth
        height = viewHeight
        camX = chosenAim - distance * ct
        camZ = targetZ + distance * st
        fx = ct; fz = -st
        ux = st; uz = ct
        focal = f
    }

    /// Vehicle metres (x forward, y left, z up) to view points, plus depth for
    /// the painter sort. nil when the point is behind the virtual lens.
    func project(_ x: Double, _ y: Double, _ z: Double = 0) -> (point: CGPoint, depth: Double)? {
        let dx = x - camX, dz = z - camZ
        let depth = dx * fx + dz * fz
        guard depth > 0.5 else { return nil }
        return (CGPoint(x: width / 2 - y * focal / depth,
                        y: height / 2 - (dx * ux + dz * uz) * focal / depth), depth)
    }

    func path(_ points: [SIMD3<Double>]) -> Path? {
        var projected = [CGPoint]()
        projected.reserveCapacity(points.count)
        for p in points {
            guard let q = project(p.x, p.y, p.z) else { return nil }
            projected.append(q.point)
        }
        guard projected.count > 1 else { return nil }
        var path = Path()
        path.move(to: projected[0])
        for point in projected.dropFirst() { path.addLine(to: point) }
        return path
    }
}

// MARK: - The view

struct WorldView: View {
    let objects: [TrackedVehicle]
    /// The live focal in network pixels, so the field-of-view wedge shows the
    /// camera actually in use rather than the trained one.
    var focal: Double = Contract.trainedFocal
    /// Ego ground speed, m/s. Negative when there is no fix.
    var egoSpeed: Double = -1

    @AppStorage(WorldFraming.key) private var tiltDegrees: Double = WorldFraming.defaultTilt
    /// Presentation-only lateral camera correction. Zero preserves the live
    /// view's existing origin; replay supplies its temporary demo offset.
    var cameraRight: Double = 0

    var body: some View {
        Canvas { context, size in
            let camera = WorldCamera(size: size, tiltDegrees: tiltDegrees,
                                     sensorFocal: focal)
            drawGround(context: &context, camera: camera)
            drawRangeRings(context: &context, camera: camera)
            drawEgoArrow(context: &context, camera: camera)
            drawObjects(context: &context, camera: camera)
            // Ego last: it is nearest to the virtual lens.
            drawVehicle(context: &context, camera: camera,
                        x: 0, y: 0, yaw: 0, length: 4.6, width: 1.9, height: 1.5,
                        profile: .car, fill: Palette.parkedSoft, stroke: Palette.ink2)
        }
        .background(Palette.ground)
    }

    // MARK: Scene

    private func drawGround(context: inout GraphicsContext, camera: WorldCamera) {
        let quad = [SIMD3<Double>(-60, -2000, 0), SIMD3<Double>(4000, -2000, 0),
                    SIMD3<Double>(4000, 2000, 0), SIMD3<Double>(-60, 2000, 0)]
        if var path = camera.path(quad) {
            path.closeSubpath()
            context.fill(path, with: .color(Palette.panel2))
            context.stroke(path, with: .color(Palette.line), lineWidth: 1)
        }
        // What the network can actually see, from the LIVE focal.
        let hfov = 2 * atan(Double(Contract.inputWidth) / (2 * max(focal, 1)))
        var wedge = [SIMD3<Double>(0, -cameraRight, 0.03)]
        for i in 0...24 {
            let a = -hfov / 2 + hfov * Double(i) / 24
            wedge.append(SIMD3<Double>(40 * cos(a), -cameraRight + 40 * sin(a), 0.03))
        }
        if var path = camera.path(wedge) {
            path.closeSubpath()
            context.fill(path, with: .color(Palette.radarSoft))
            context.stroke(path, with: .color(Palette.radar), lineWidth: 1)
        }
    }

    private func drawRangeRings(context: inout GraphicsContext, camera: WorldCamera) {
        for r in stride(from: 10.0, through: 40.0, by: 10.0) {
            var ring = [SIMD3<Double>]()
            for i in 0...40 {
                let a = -1.25 + 2.5 * Double(i) / 40
                ring.append(SIMD3<Double>(r * cos(a), r * sin(a), 0.05))
            }
            if let path = camera.path(ring) {
                context.stroke(path, with: .color(Palette.line), lineWidth: 1)
            }

        }
        for y in [-1.85, 1.85] {
            if let path = camera.path([SIMD3<Double>(0, y, 0.06), SIMD3<Double>(40, y, 0.06)]) {
                context.stroke(path, with: .color(Palette.line),
                               style: StrokeStyle(lineWidth: 1, dash: [4, 5]))
            }
        }
    }

    private func drawEgoArrow(context: inout GraphicsContext, camera: WorldCamera) {
        guard egoSpeed > 0.3 else { return }
        drawArrow(context: &context, camera: camera,
                  from: SIMD3<Double>(0, 0, 0.10), to: SIMD3<Double>(egoSpeed, 0, 0.10),
                  color: Palette.ink2, lineWidth: 2, dash: [5, 4])
        if let p = camera.project(egoSpeed, 0, 0.12) {
            context.draw(Text(String(format: "ego %.0f km/h", egoSpeed * 3.6))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Palette.ink2),
                         at: CGPoint(x: p.point.x + 6, y: p.point.y), anchor: .leading)
        }
    }

    // MARK: Detections

    private func drawObjects(context: inout GraphicsContext, camera: WorldCamera) {
        // Far to near, so nearer bodies paint over further ones.
        let sorted = objects
            .compactMap { object -> (TrackedVehicle, Double)? in
                guard let p = camera.project(object.x, object.y, 0) else { return nil }
                return (object, p.depth)
            }
            .sorted { $0.1 > $1.1 }

        // Trails are ground evidence and therefore paint below every body.
        for (object, _) in sorted where object.trail.count > 1 {
            let points = object.trail.map { SIMD3<Double>($0.x, $0.y, 0.08) }
            if let path = camera.path(points) {
                context.stroke(path, with: .color(PerceptionColour.state(object.state).opacity(0.65)),
                               lineWidth: 1.5)
            }
        }

        for (object, _) in sorted {
            let profile = VehicleProfile.profile(label: object.label,
                                                 length: object.length, width: object.width)
            let h = object.height.isFinite && object.height > 0.2
                ? object.height : profile.defaultHeight
            let colour = PerceptionColour.state(object.state)
            drawVehicle(context: &context, camera: camera,
                        x: object.x, y: object.y, yaw: object.yaw,
                        length: object.length, width: object.width,
                        height: h, profile: profile,
                        fill: colour.opacity(object.observed ? 0.16 : 0.07), stroke: colour,
                        dashed: !object.observed)
        }

        // Labels are a second pass, so no subsequently drawn body can cover them.
        for (object, _) in sorted.reversed() {
            let h = max(object.height, 1.0)
            if let label = camera.project(object.x, object.y, h + 0.5) {

                let speed = object.velocity.map { hypot($0.x, $0.y) * 3.6 }
                let text = speed.map { String(format: "%.0f m %.0f km/h", object.x, $0) }
                    ?? String(format: "%.0f m", object.x)
                context.draw(Text(text)
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .foregroundStyle(PerceptionColour.state(object.state)),
                             at: CGPoint(x: label.point.x + 3, y: label.point.y), anchor: .leading)
            }
        }
    }

    /// Body walls, body top, cabin walls, cabin top, glass, seam -- the order
    /// matters, because each layer paints over the one below it.
    private func drawVehicle(context: inout GraphicsContext, camera: WorldCamera,
                             x: Double, y: Double, yaw: Double,
                             length: Double, width: Double, height: Double,
                             profile: VehicleProfile, fill: Color, stroke: Color,
                             dashed: Bool = false) {
        let c = cos(yaw), s = sin(yaw)
        func point(_ p: SIMD2<Double>, _ z: Double) -> SIMD3<Double> {
            SIMD3<Double>(x + p.x * length * c - p.y * width * s,
                          y + p.x * length * s + p.y * width * c, z)
        }
        func shape(_ points: [SIMD2<Double>], _ z: Double) -> [SIMD3<Double>] {
            points.map { point($0, z) }
        }
        func walls(_ lower: [SIMD3<Double>], _ upper: [SIMD3<Double>],
                   _ sideFill: Color, _ sideStroke: Color) {
            context.opacity = 0.72
            for i in lower.indices {
                let j = (i + 1) % lower.count
                if var path = camera.path([lower[i], lower[j], upper[j], upper[i]]) {
                    path.closeSubpath()
                    context.fill(path, with: .color(sideFill))
                    context.stroke(path, with: .color(sideStroke),
                                   style: StrokeStyle(lineWidth: 1.05,
                                                      dash: dashed ? [5, 4] : []))
                }
            }
            context.opacity = 1.0
        }
        func cap(_ ring: [SIMD3<Double>], _ capFill: Color, _ lineWidth: Double) {
            if var path = camera.path(ring) {
                path.closeSubpath()
                context.fill(path, with: .color(capFill))
                context.stroke(path, with: .color(stroke),
                               style: StrokeStyle(lineWidth: lineWidth,
                                                  dash: dashed ? [5, 4] : []))
            }
        }

        let bodyLow = shape(profile.body, 0.04)
        let bodyHigh = shape(profile.body, profile.bodyZ * height)
        walls(bodyLow, bodyHigh, fill, stroke)
        context.opacity = 0.96
        cap(bodyHigh, fill, 1.55)
        context.opacity = 1.0

        let cabinLow = shape(profile.cabin, profile.bodyZ * height)
        let cabinHigh = shape(profile.cabin, profile.roofZ * height)
        walls(cabinLow, cabinHigh, Palette.cabin, stroke)
        cap(cabinHigh, Palette.cabinTop, 1.3)

        if !profile.glass.isEmpty {
            cap(shape(profile.glass, profile.roofZ * height + 0.012), Palette.glass, 1.0)
        }
        if profile.hasSeam {
            let seam = [point(SIMD2<Double>(0.12, -0.30), profile.roofZ * height + 0.018),
                        point(SIMD2<Double>(0.12, 0.30), profile.roofZ * height + 0.018)]
            if let path = camera.path(seam) {
                context.stroke(path, with: .color(stroke), lineWidth: 0.9)
            }
        }
    }

    private func drawArrow(context: inout GraphicsContext, camera: WorldCamera,
                           from a: SIMD3<Double>, to b: SIMD3<Double>,
                           color: Color, lineWidth: Double, dash: [CGFloat] = []) {
        guard let start = camera.project(a.x, a.y, a.z),
              let end = camera.project(b.x, b.y, b.z) else { return }
        let angle = atan2(end.point.y - start.point.y, end.point.x - start.point.x)
        let head = 7.0
        var path = Path()
        path.move(to: start.point)
        path.addLine(to: end.point)
        path.addLine(to: CGPoint(x: end.point.x - head * cos(angle - 0.4),
                                 y: end.point.y - head * sin(angle - 0.4)))
        path.move(to: end.point)
        path.addLine(to: CGPoint(x: end.point.x - head * cos(angle + 0.4),
                                 y: end.point.y - head * sin(angle + 0.4)))
        context.stroke(path, with: .color(color),
                       style: StrokeStyle(lineWidth: lineWidth, dash: dash))
    }
}
