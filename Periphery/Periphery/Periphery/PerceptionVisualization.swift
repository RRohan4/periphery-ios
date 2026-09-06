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
                let corners = Self.projectedCorners(object,
                                                    calibration: calibration.calibration)
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

    /// A coast is presentable only while the predicted box still overlaps the
    /// pixels the detector actually saw. This distinguishes a temporary visual
    /// occlusion (wiper, glare, another car) from an object that left the lens.
    static func remainsInInferenceView(_ object: TrackedVehicle,
                                       snapshot: PerceptionCalibrationSnapshot) -> Bool {
        let points = projectedCorners(object, calibration: snapshot.calibration)
        guard points.count == 8,
              let minX = points.map(\.x).min(), let maxX = points.map(\.x).max(),
              let minY = points.map(\.y).min(), let maxY = points.map(\.y).max()
        else { return false }
        let box = CGRect(x: minX, y: minY,
                         width: max(maxX - minX, 1), height: max(maxY - minY, 1))
        let crop = snapshot.crop
        let inferenceRegion = CGRect(x: crop.x, y: crop.y,
                                     width: crop.width, height: crop.height)
        return box.intersects(inferenceRegion)
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

    /// Show a brief coast for an object that should still be visible (for
    /// example behind a windshield wiper), but not after its predicted box has
    /// actually left the detector's image region. Tracker state is retained in
    /// both cases; this is only the presentation boundary.
    private var displayedObjects: [TrackedVehicle] {
        objects.filter { object in
            if object.observed { return true }
            guard let calibration else { return false }
            return CameraBoxOverlay.remainsInInferenceView(object, snapshot: calibration)
        }
    }

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
