//  PerceptionVisualization.swift
//  Shared, read-only presentation adapter for Live and Replay.
//
//  This file projects already-decided tracked results. It does not associate,
//  classify, smooth, suppress, or otherwise change perception state.

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
                let corners = Self.corners(object, calibration: calibration.calibration)
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

    private static func corners(_ object: TrackedVehicle,
                                calibration: Calibration) -> [SIMD2<Double>] {
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
                guard let projected = calibration.sourcePoint(point) else { return [] }
                result.append(projected)
            }
        }
        return result
    }
}

/// Product display shared by live camera and recorded replay. The caller owns
/// the image source; this view owns only layout and perception rendering.
struct PerceptionSplitView<CameraContent: View>: View {
    let objects: [TrackedVehicle]
    let calibration: PerceptionCalibrationSnapshot?
    let egoSpeed: Double
    let sourceLabel: String
    let cameraContent: CameraContent

    /// Coasting remains tracker-internal so a brief miss can recover the same
    /// ID. The rider display only presents boxes supported by the current
    /// image; predicted motion after an object leaves view is not evidence.
    private var displayedObjects: [TrackedVehicle] { objects.filter(\.observed) }

    init(objects: [TrackedVehicle], calibration: PerceptionCalibrationSnapshot?,
         egoSpeed: Double, sourceLabel: String,
         @ViewBuilder cameraContent: () -> CameraContent) {
        self.objects = objects
        self.calibration = calibration
        self.egoSpeed = egoSpeed
        self.sourceLabel = sourceLabel
        self.cameraContent = cameraContent()
    }

    var body: some View {
        GeometryReader { geometry in
            let gap = 8.0
            let cameraWidth = max(geometry.size.width * 0.38, 260)
            HStack(spacing: gap) {
                WorldView(objects: displayedObjects,
                          focal: calibration?.focal ?? Contract.trainedFocal,
                          egoSpeed: egoSpeed)
                VStack(spacing: gap) {
                    cameraContent
                        .overlay {
                            CameraBoxOverlay(objects: displayedObjects, calibration: calibration)
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
                Circle().fill(PerceptionColour.moving).frame(width: 7, height: 7)
                Text(sourceLabel.uppercased())
                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                    .foregroundStyle(PerceptionColour.moving)
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
