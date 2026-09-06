// Shared presentation adapter for Live and Replay. It projects tracked results
// without changing inference or tracking state.

import SwiftUI
import simd

enum PerceptionColour {
    static let moving = Color(red: 0.910, green: 0.639, blue: 0.239)
    static let parked = Color(red: 0.580, green: 0.635, blue: 0.690)
    static let unknown = Color(red: 0.420, green: 0.471, blue: 0.522)

    static func state(_ state: VehicleMotionState) -> Color {
        switch state {
        case .moving: return moving
        case .parked: return parked
        case .unknown: return unknown
        }
    }
}

struct CameraBoxOverlay: View {
    let objects: [TrackedVehicle]
    let calibration: PerceptionCalibrationSnapshot?
    /// Replay-only presentation correction. Positive means the camera is to
    /// the vehicle's right; live remains on the pipeline's zero-offset path.
    var cameraRight: Double = 0

    var body: some View {
        Canvas { context, size in
            guard let calibration else { return }
            let sourceWidth = Double(calibration.calibration.frameWidth)
            let sourceHeight = Double(calibration.calibration.frameHeight)
            guard sourceWidth > 0, sourceHeight > 0 else { return }
            func display(_ point: SIMD2<Double>) -> CGPoint {
                Self.aspectFill(point: point,
                                source: CGSize(width: sourceWidth, height: sourceHeight),
                                destination: size) ?? .zero
            }

            for object in objects {
                let corners = Self.projectedCorners(object,
                                                    calibration: calibration.calibration,
                                                    cameraRight: cameraRight)
                guard corners.count == 8 else { continue }
                let points = corners.map(display)
                var path = Path()
                for face in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4],
                             [0, 4], [1, 5], [2, 6], [3, 7]] {
                    path.move(to: points[face[0]])
                    for index in face.dropFirst() { path.addLine(to: points[index]) }
                }
                let colour = PerceptionColour.state(object.state)
                context.stroke(path, with: .color(colour),
                               style: StrokeStyle(lineWidth: object.observed ? 2 : 1.5,
                                                  dash: object.observed ? [] : [5, 4]))
                let anchor = points[4...7].min(by: { $0.y < $1.y }) ?? points[4]
                context.draw(Text("#\(object.id)")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(colour),
                             at: CGPoint(x: anchor.x, y: anchor.y - 4), anchor: .bottom)
            }
        }
        .allowsHitTesting(false)
    }

    static func aspectFill(point: SIMD2<Double>, source: CGSize,
                           destination: CGSize) -> CGPoint? {
        guard source.width > 0, source.height > 0,
              destination.width > 0, destination.height > 0 else { return nil }
        let scale = max(destination.width / source.width,
                        destination.height / source.height)
        let offsetX = (destination.width - source.width * scale) / 2
        let offsetY = (destination.height - source.height * scale) / 2
        return CGPoint(x: offsetX + point.x * scale,
                       y: offsetY + point.y * scale)
    }

    static func projectedCorners(_ object: TrackedVehicle,
                                 calibration: Calibration,
                                 cameraRight: Double = 0) -> [SIMD2<Double>] {
        let c = cos(object.yaw), s = sin(object.yaw)
        let bottom = object.z - object.height / 2
        let top = object.z + object.height / 2
        let footprint = [SIMD2(0.5, 0.5), SIMD2(0.5, -0.5),
                         SIMD2(-0.5, -0.5), SIMD2(-0.5, 0.5)]
        var result: [SIMD2<Double>] = []
        for z in [bottom, top] {
            for p in footprint {
                let localX = p.x * object.length, localY = p.y * object.width
                let point = SIMD3(object.x + localX * c - localY * s,
                                  object.y + localX * s + localY * c, z)
                guard let projected = sourcePoint(point, calibration: calibration,
                                                  cameraRight: cameraRight) else { return [] }
                result.append(projected)
            }
        }
        return result
    }

    private static func sourcePoint(_ vehicle: SIMD3<Double>,
                                    calibration: Calibration,
                                    cameraRight: Double) -> SIMD2<Double>? {
        guard abs(cameraRight) > 1e-9 else { return calibration.sourcePoint(vehicle) }
        let rotation = Calibration.vehicleToSensor(pitch: calibration.pose.pitch,
                                                   roll: calibration.pose.roll,
                                                   yaw: calibration.pose.yaw)
        let camera = SIMD3<Double>(calibration.forwardOfOrigin,
                                   -cameraRight,
                                   calibration.height)
        let image = Contract.sensorToImageAxes * (rotation * (vehicle - camera))
        guard image.z > 1e-6 else { return nil }
        let projected = calibration.K * image
        return SIMD2<Double>(projected.x / projected.z,
                             projected.y / projected.z)
    }

    static func remainsInFieldOfViewArc(_ object: TrackedVehicle,
                                        focal: Double,
                                        cameraRight: Double = 0) -> Bool {
        let halfFOV = atan(Double(Contract.inputWidth) / (2 * max(focal, 1)))
        let cameraY = -cameraRight
        let c = cos(object.yaw), s = sin(object.yaw)
        let footprint = [SIMD2(0.5, 0.5), SIMD2(0.5, -0.5),
                         SIMD2(-0.5, -0.5), SIMD2(-0.5, 0.5)]

        return footprint.contains { p in
            let localX = p.x * object.length, localY = p.y * object.width
            let x = object.x + localX * c - localY * s
            let y = object.y + localX * s + localY * c
            let forward = x
            let left = y - cameraY
            let range = hypot(forward, left)
            return forward > 0 && range <= 40.0
                && abs(atan2(left, forward)) <= halfFOV
        }
    }
}

/// Product display shared by live camera and recorded replay. The caller owns
/// the image source; this view owns only layout and perception rendering.
struct PerceptionSplitView<CameraContent: View>: View {
    let objects: [TrackedVehicle]
    let calibration: PerceptionCalibrationSnapshot?
    let egoSpeed: Double
    let cameraRight: Double
    let cameraContent: CameraContent

    private var displayedObjects: [TrackedVehicle] {
        objects.filter { object in
            guard let calibration else { return object.observed }
            return CameraBoxOverlay.remainsInFieldOfViewArc(object,
                                                            focal: calibration.focal,
                                                            cameraRight: cameraRight)
        }
    }

    init(objects: [TrackedVehicle], calibration: PerceptionCalibrationSnapshot?,
         egoSpeed: Double, cameraRight: Double = 0,
         @ViewBuilder cameraContent: () -> CameraContent) {
        self.objects = objects
        self.calibration = calibration
        self.egoSpeed = egoSpeed
        self.cameraRight = cameraRight
        self.cameraContent = cameraContent()
    }

    var body: some View {
        GeometryReader { geometry in
            let gap = 8.0
            let cameraWidth = max(geometry.size.width * 0.38, 260)
            HStack(spacing: gap) {
                WorldView(objects: displayedObjects,
                          focal: calibration?.focal ?? Contract.trainedFocal,
                          egoSpeed: egoSpeed,
                          cameraRight: cameraRight)
                VStack(spacing: gap) {
                    cameraContent
                        .overlay {
                            CameraBoxOverlay(objects: displayedObjects,
                                             calibration: calibration,
                                             cameraRight: cameraRight)
                        }
                        .aspectRatio(2.0, contentMode: .fill)
                        .frame(width: cameraWidth)
                        .clipped()
                    hud.frame(maxWidth: .infinity, maxHeight: .infinity)
                }
                .frame(width: cameraWidth)
            }
            .padding(8)
        }
        .background(Color(red: 0.035, green: 0.043, blue: 0.051))
    }

    private var hud: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack {
                Spacer()
                Text("PERIPHERY")
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.7))
            }
            Spacer()
            Text(egoSpeed >= 0 ? String(format: "%.0f", egoSpeed * 3.6) : "—")
                .font(.system(size: 46, weight: .medium, design: .rounded))
                .foregroundStyle(.white)
            Text("KM/H")
                .font(.system(size: 11, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.55))
            Text("\(displayedObjects.count) OBSERVED")
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.white.opacity(0.55))
        }
        .padding(12)
        .background(Color(red: 0.059, green: 0.075, blue: 0.090))
    }
}
