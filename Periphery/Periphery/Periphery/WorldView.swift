//  WorldView.swift
//  The bird's-eye rendering of what the detector found.
//
//  Split out of LiveView.swift. Currently a flat top-down plan; the 2.5D world
//  view from periphery/scripts/comma_viewer_template.html replaces it next.

import SwiftUI

/// Forward 0-40 m up the view, lateral +-10 m across it. Ego at the bottom
/// centre. Anything outside that region is not drawn, because outside it a box
/// is a decode artefact rather than a detection.
struct BEVOverlay: View {
    let detections: [Detection]

    var body: some View {
        Canvas { context, size in
            let forwardMax = Contract.forwardRange.max
            let lateralMax = Contract.lateralRange.max
            let scaleX = size.width / (2 * lateralMax)
            let scaleY = size.height / forwardMax

            func point(x forward: Double, y lateral: Double) -> CGPoint {
                CGPoint(x: size.width / 2 - lateral * scaleX,
                        y: size.height - forward * scaleY)
            }

            // Range rings every 10 m.
            for range in stride(from: 10.0, through: forwardMax, by: 10.0) {
                let y = size.height - range * scaleY
                context.stroke(Path { path in
                    path.move(to: CGPoint(x: 0, y: y))
                    path.addLine(to: CGPoint(x: size.width, y: y))
                }, with: .color(.gray.opacity(0.25)), lineWidth: 1)
                context.draw(Text("\(Int(range)) m").font(.caption2).foregroundStyle(.secondary),
                             at: CGPoint(x: 22, y: y - 8))
            }
            // Lane-width reference at +-1.8 m.
            for lateral in [-1.8, 1.8] {
                let x = size.width / 2 - lateral * scaleX
                context.stroke(Path { path in
                    path.move(to: CGPoint(x: x, y: 0))
                    path.addLine(to: CGPoint(x: x, y: size.height))
                }, with: .color(.gray.opacity(0.2)),
                   style: StrokeStyle(lineWidth: 1, dash: [4, 6]))
            }

            for detection in detections {
                let corners = [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (-0.5, 0.5)]
                    .map { (along: Double, across: Double) -> CGPoint in
                        let dx = along * detection.length
                        let dy = across * detection.width
                        let c = cos(detection.yaw), s = sin(detection.yaw)
                        return point(x: detection.x + dx * c - dy * s,
                                     y: detection.y + dx * s + dy * c)
                    }
                var path = Path()
                path.addLines(corners)
                path.closeSubpath()
                let hue = Double(detection.score)
                context.fill(path, with: .color(.green.opacity(0.15 + 0.35 * hue)))
                context.stroke(path, with: .color(.green), lineWidth: 1.5)
                context.draw(
                    Text(String(format: "%.0f m", detection.x))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundStyle(.primary),
                    at: point(x: detection.x, y: detection.y))
            }

            // Ego.
            let ego = point(x: 0, y: 0)
            context.fill(Path(ellipseIn: CGRect(x: ego.x - 4, y: ego.y - 4,
                                                width: 8, height: 8)),
                         with: .color(.blue))
        }
        .background(Color(white: 0.08))
    }
}
