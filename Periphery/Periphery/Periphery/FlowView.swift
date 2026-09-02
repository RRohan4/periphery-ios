//  FlowView.swift
//  The camera pitch estimator, made visible, on foot.
//
//  WHY A TAB
//
//  FocusOfExpansion works on comma2k19 -- 0.103 deg mean absolute error -- but
//  that was measured offline, in Python, against a dataset. On the phone it is
//  a different optical flow implementation, a different gyro, a different axis
//  convention, and three sign choices that would each produce a confident wrong
//  answer rather than a crash. Waiting for a drive to find that out is slow and
//  the failure would be invisible: a wrong pitch just reads 40 m as 56 m.
//
//  Walking with the phone exercises the entire path in ten seconds, indoors,
//  and makes every part of it visible:
//
//    * Walk forward. The cross should sit on the spot you are walking TOWARD.
//    * Tilt the phone down, still walking the same way. The cross stays GLUED
//      TO THAT SPOT in the world -- and because tilting the camera down slides
//      the whole scene up the frame, the cross rides up with it. What must not
//      happen is the cross drifting off the spot. Meanwhile the reported pitch
//      goes NEGATIVE by however far you tilted, since pitch is nose-up
//      positive. That one test proves the sign, the intrinsics and the roll
//      unwind at once.
//    * Turn while walking. If de-rotation works, the number barely moves. If
//      the gyro axis mapping is wrong, it swings wildly. The handheld profile
//      deliberately allows 8 deg/s so this is testable at all.
//    * Point at a passing person or car. They light up red -- outliers -- with
//      nothing in the app having recognised them as anything.
//
//  WHAT IT IS NOT
//
//  Not a calibration. The angle measured here is the angle of your HAND
//  relative to the direction you are walking, and it is refused by MountPose
//  on the way in (Gates.handheld.writesToPose is false). The mount pitch comes
//  from the driving profile, in a car, or from the Calibrate tab.

import SwiftUI
import simd

struct FlowView: View {
    @StateObject private var model = FlowModel()
    @ObservedObject private var live = LiveSession.shared

    var body: some View {
        HStack(spacing: 1) {
            preview
            readout
                .frame(width: 260)
        }
        .background(Color.black)
        .statusBarHidden()
        .task { await model.attach() }
        .onDisappear { model.detach() }
    }

    // MARK: - Left: the flow field

    private var preview: some View {
        ZStack {
            Color.black
            if let session = live.session {
                // `.fit`, unlike the Live tab: here the whole frame has to be
                // visible, because an outlier patch cropped off the edge is the
                // one you most wanted to see.
                CameraPreview(session: session)
                    .aspectRatio(16.0 / 9.0, contentMode: .fit)
                    .overlay { FlowOverlay(debug: model.debug) }
            } else {
                Text(live.status).font(.caption).foregroundStyle(.white)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Right: the numbers

    private var readout: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text("Flow")
                    .font(.headline)
                    .foregroundStyle(.white)

                Picker("mode", selection: $model.handheld) {
                    Text("Handheld").tag(true)
                    Text("Driving").tag(false)
                }
                .pickerStyle(.segmented)

                if let debug = model.debug, debug.foe != nil {
                    angle("pitch", model.estimate.pitchDegrees, debug.pitchDegrees)
                    angle("yaw", model.estimate.yawDegrees, debug.yawDegrees)
                    row("consensus", "\(debug.inliers) / \(debug.total)")
                    row("window", "\(model.estimate.samples) / \(model.estimate.windowSize)")
                    row("rotation", String(format: "%+.1f, %+.1f, %+.1f °/s",
                                           debug.rotationDegreesPerSecond.x,
                                           debug.rotationDegreesPerSecond.y,
                                           debug.rotationDegreesPerSecond.z))
                } else {
                    Text(model.debug?.note ?? "waiting for motion")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(.orange)
                }

                Divider().overlay(Color.white.opacity(0.2))

                legend
                Text(Help.body)
                    .font(.system(size: 10))
                    .foregroundStyle(.white.opacity(0.55))

                if !model.handheld {
                    Text("Driving profile: gated at 8 m/s, so nothing will "
                         + "accumulate on foot.")
                        .font(.system(size: 10))
                        .foregroundStyle(.yellow)
                }
                Button("Reset the window") { model.reset() }
                    .font(.caption)
                    .buttonStyle(.bordered)
            }
            .padding(12)
        }
        .background(Color(white: 0.07))
    }

    /// Window median large, this-frame value small beside it. The gap between
    /// them is how noisy a single pair is, which is the thing the window exists
    /// to average away.
    private func angle(_ name: String, _ median: Double, _ instant: Double) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name.uppercased())
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(String(format: "%+.2f°", median))
                    .font(.system(size: 26, weight: .medium, design: .monospaced))
                    .foregroundStyle(.white)
                Text(String(format: "now %+.2f", instant))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.45))
            }
        }
    }

    private func row(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.5))
            Spacer()
            Text(value)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(.white.opacity(0.85))
        }
    }

    private var legend: some View {
        VStack(alignment: .leading, spacing: 3) {
            legendRow(.green, "agrees with the focus — static world")
            legendRow(.red, "disagrees — moving, or mistracked")
            legendRow(.cyan, "the focus of expansion")
        }
    }

    private func legendRow(_ colour: Color, _ text: String) -> some View {
        HStack(spacing: 6) {
            Circle().fill(colour).frame(width: 6, height: 6)
            Text(text)
                .font(.system(size: 10))
                .foregroundStyle(.white.opacity(0.6))
        }
    }

    private enum Help {
        static let body = """
            Walk forward: the cross should sit on the spot you are walking \
            toward. Tilt the phone down and it should stay stuck to that spot \
            — riding up the frame as the scene does — while pitch goes \
            negative. Turn as you walk: if de-rotation works, the number \
            barely moves.

            Handheld readings measure your hand, not the mount, and cannot be \
            applied to the pose.
            """
    }
}

// MARK: - The drawing

/// Flow vectors and the fitted focus, over the live image.
private struct FlowOverlay: View {
    let debug: FocusOfExpansion.Debug?

    var body: some View {
        Canvas { context, size in
            guard let debug, debug.width > 0, debug.height > 0 else { return }
            // The working image and the preview are both the full frame, so a
            // single uniform scale maps one onto the other.
            let scale = size.width / Double(debug.width)
            func map(_ x: Double, _ y: Double) -> CGPoint {
                CGPoint(x: x * scale, y: y * scale)
            }

            for i in debug.points.indices {
                let p = debug.points[i]
                let v = debug.vectors[i]
                // Vectors are a few pixels long at walking pace; drawn at true
                // length they would be invisible.
                let gain = 3.0
                var path = Path()
                path.move(to: map(Double(p.x), Double(p.y)))
                path.addLine(to: map(Double(p.x) + Double(v.x) * gain,
                                     Double(p.y) + Double(v.y) * gain))
                context.stroke(path,
                               with: .color(debug.inlier[i]
                                            ? .green.opacity(0.75)
                                            : .red.opacity(0.8)),
                               lineWidth: 1.2)
            }

            guard let foe = debug.foe else { return }
            let centre = map(foe.x, foe.y)
            let arm = 14.0
            var cross = Path()
            cross.move(to: CGPoint(x: centre.x - arm, y: centre.y))
            cross.addLine(to: CGPoint(x: centre.x + arm, y: centre.y))
            cross.move(to: CGPoint(x: centre.x, y: centre.y - arm))
            cross.addLine(to: CGPoint(x: centre.x, y: centre.y + arm))
            context.stroke(cross, with: .color(.cyan), lineWidth: 2)
            context.stroke(Path(ellipseIn: CGRect(x: centre.x - 9, y: centre.y - 9,
                                                  width: 18, height: 18)),
                           with: .color(.cyan), lineWidth: 1.5)
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Model

@MainActor
final class FlowModel: ObservableObject {
    @Published var debug: FocusOfExpansion.Debug?
    @Published var estimate = FocusOfExpansion.Estimate()
    @Published var handheld = true {
        didSet { applyProfile() }
    }

    private var estimator: FocusOfExpansion { LiveSession.shared.pipeline.foe }
    private var attached = false

    /// Taking over the shared estimator, rather than running a second one: two
    /// optical-flow passes per frame would double the cost of the most
    /// expensive thing on the phone, to show the same field twice.
    func attach() async {
        await LiveSession.shared.ensureStarted()
        guard !attached else { return }
        attached = true
        estimator.publishDebug = true
        estimator.onDebug = { [weak self] debug in
            Task { @MainActor in self?.debug = debug }
        }
        estimator.onEstimate = { [weak self] estimate in
            Task { @MainActor in self?.estimate = estimate }
        }
        applyProfile()
    }

    /// Hand the estimator back on the way out, or the Live tab would spend the
    /// drive in handheld mode -- accumulating, but refusing to write the pose.
    func detach() {
        guard attached else { return }
        attached = false
        estimator.publishDebug = false
        estimator.onDebug = nil
        estimator.onEstimate = nil
        estimator.gates = .driving
        debug = nil
    }

    func reset() { estimator.reset() }

    private func applyProfile() {
        // Setting gates resets the window: a median mixing two profiles would
        // be a median of two different measurements.
        estimator.gates = handheld ? .handheld : .driving
    }
}
