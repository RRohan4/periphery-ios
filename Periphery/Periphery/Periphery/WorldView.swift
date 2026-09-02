//  WorldView.swift
//  The 2.5D world view: detections extruded on the ground plane, seen from a
//  fixed virtual camera behind and above the car.
//
//  A transliteration of drawBev() in periphery/scripts/comma_viewer_template.html,
//  which was written as scalar math for exactly this reason -- its own comment
//  says "the Swift port can copy project(), box(), and the painter order without
//  a 3D dependency". Names, constants and painter order are kept identical so
//  the two stay diffable; where this file and that one disagree, that one is
//  the reference.
//
//  There is no tracker on device yet, so there are no ids, no moving/parked
//  colours, no velocity arrows and no trails. The row struct is shaped the way
//  build_comma_viewer.py shapes its BEV rows -- the geometric fields first,
//  everything else appended and optional -- so porting associate.py later
//  touches the tracker and not this file.

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

/// Small normalised profiles, enough to read a car as an object rather than a
/// polygon without a mesh format. Coordinates are [forward, left] in [-.5, .5]
/// and heights are fractions of the measured object height. Copied verbatim
/// from MODEL_PROFILES.
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

    /// Label first, then dimensions -- the same fallback order as
    /// build_comma_viewer.model_kind(), so both renderers pick the same body
    /// for the same detection.
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

// MARK: - The virtual camera

/// A fixed pose behind and above the car. Copied from drawBev(): 27 degrees of
/// tilt, 132 m back, aimed 22 m ahead, 26 degree vertical field of view.
private struct WorldCamera {
    let width: Double
    let height: Double
    let camX: Double
    let camZ: Double
    let fx: Double, fz: Double
    let ux: Double, uz: Double
    let focal: Double

    init(size: CGSize) {
        width = size.width
        height = size.height
        let tilt = 27.0 * .pi / 180.0
        let distance = 132.0, targetX = 22.0, targetZ = 0.9
        camX = targetX - distance * cos(tilt)
        camZ = targetZ + distance * sin(tilt)
        fx = cos(tilt); fz = -sin(tilt)
        ux = sin(tilt); uz = cos(tilt)
        focal = height / (2 * tan(26.0 * .pi / 360.0))
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
    let detections: [Detection]
    /// The live focal in network pixels, so the field-of-view wedge shows the
    /// camera actually in use rather than the trained one.
    var focal: Double = Contract.trainedFocal
    /// Ego ground speed, m/s. Negative when there is no fix.
    var egoSpeed: Double = -1

    var body: some View {
        Canvas { context, size in
            let camera = WorldCamera(size: size)
            drawGround(context: &context, camera: camera)
            drawRangeRings(context: &context, camera: camera)
            drawEgoArrow(context: &context, camera: camera)
            drawDetections(context: &context, camera: camera)
            // Ego last: it is nearest to the virtual lens.
            drawVehicle(context: &context, camera: camera,
                        x: 0, y: 0, yaw: 0, length: 4.6, width: 1.9, height: 1.5,
                        profile: .car, fill: Palette.parkedSoft, stroke: Palette.ink2)
        }
        .background(Palette.ground)
    }

    // MARK: Scene

    private func drawGround(context: inout GraphicsContext, camera: WorldCamera) {
        let quad = [SIMD3<Double>(-12, -23, 0), SIMD3<Double>(56, -23, 0),
                    SIMD3<Double>(56, 23, 0), SIMD3<Double>(-12, 23, 0)]
        if var path = camera.path(quad) {
            path.closeSubpath()
            context.fill(path, with: .color(Palette.panel2))
            context.stroke(path, with: .color(Palette.line), lineWidth: 1)
        }
        // What the network can actually see, from the LIVE focal.
        let hfov = 2 * atan(Double(Contract.inputWidth) / (2 * max(focal, 1)))
        var wedge = [SIMD3<Double>(0, 0, 0.03)]
        for i in 0...24 {
            let a = -hfov / 2 + hfov * Double(i) / 24
            wedge.append(SIMD3<Double>(40 * cos(a), 40 * sin(a), 0.03))
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
            if let label = camera.project(r, -0.5, 0.06) {
                context.draw(Text("\(Int(r)) m")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(Palette.ink3),
                             at: label.point, anchor: .leading)
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
            context.draw(Text(String(format: "ego %.1f m/s", egoSpeed))
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(Palette.ink2),
                         at: CGPoint(x: p.point.x + 6, y: p.point.y), anchor: .leading)
        }
    }

    // MARK: Detections

    private func drawDetections(context: inout GraphicsContext, camera: WorldCamera) {
        // Far to near, so nearer bodies paint over further ones.
        let sorted = detections
            .compactMap { d -> (Detection, Double)? in
                guard let p = camera.project(d.x, d.y, 0) else { return nil }
                return (d, p.depth)
            }
            .sorted { $0.1 > $1.1 }

        for (d, _) in sorted {
            let profile = VehicleProfile.profile(label: d.label, length: d.length, width: d.width)
            let h = d.height.isFinite && d.height > 0.2 ? d.height : profile.defaultHeight
            drawVehicle(context: &context, camera: camera,
                        x: d.x, y: d.y, yaw: d.yaw, length: d.length, width: d.width,
                        height: h, profile: profile,
                        fill: Palette.visionSoft, stroke: Palette.vision)
            if let label = camera.project(d.x, d.y, h + 0.5) {
                context.draw(Text(String(format: "%.0f m", d.x))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(Palette.vision),
                             at: CGPoint(x: label.point.x + 3, y: label.point.y),
                             anchor: .leading)
            }
        }
    }

    /// Body walls, body top, cabin walls, cabin top, glass, seam -- the order
    /// matters, because each layer paints over the one below it.
    private func drawVehicle(context: inout GraphicsContext, camera: WorldCamera,
                             x: Double, y: Double, yaw: Double,
                             length: Double, width: Double, height: Double,
                             profile: VehicleProfile, fill: Color, stroke: Color) {
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
                    context.stroke(path, with: .color(sideStroke), lineWidth: 1.05)
                }
            }
            context.opacity = 1.0
        }
        func cap(_ ring: [SIMD3<Double>], _ capFill: Color, _ lineWidth: Double) {
            if var path = camera.path(ring) {
                path.closeSubpath()
                context.fill(path, with: .color(capFill))
                context.stroke(path, with: .color(stroke), lineWidth: lineWidth)
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
