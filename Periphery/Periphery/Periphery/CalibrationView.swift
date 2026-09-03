//  CalibrationView.swift
//  The four numbers the projection depends on, and how much to trust each.
//
//  Ordered by how much damage each does. Pitch first, because range error goes
//  as r^2/h: at 1.20 m a point 40 m ahead sits 1.72 degrees below horizontal,
//  so half a degree of pitch error reads 40 m as 56 m. Then height, which is a
//  pure scale. Then yaw, 0.70 m of lateral error per degree at 40 m. Roll is
//  last because gravity already measures it honestly and it needs no input.

import Combine
import SwiftUI

/// Footer copy, hoisted out of the ViewBuilders.
///
/// A chain of `+`-concatenated string literals inside a ViewBuilder is a
/// well-known way to blow the type checker's budget -- it did, on the
/// diagnostics section. Multi-line literals in a plain enum cost it nothing.
private enum Help {
    static let pitchDegrees = """
        The dominant error term. Range sensitivity is r²/h, so 0.25° is the design \
        budget, 0.50° is tolerable and 1.00° is a declared failure.

        This is a STARTING GUESS, not a lock. It outranks gravity, which measures \
        mount + road grade and cannot separate them — over 237 drive segments it \
        tracks grade one-for-one, costing 2.45° p95 — so gravity will not walk your \
        number back. The drive-time estimator outranks both and will supersede it \
        once it has enough road, which is the whole point of seeding it.
        """
    static let height = """
        Forgiving: d(range)/range = d(height)/height, a pure scale factor, so 5 cm \
        is about 4% of range.

        The barometer works — 1.2 m of lift is ~0.14 hPa against ~0.02 hPa of noise \
        — but the limit is cabin pressure, not the sensor. A door or the HVAC moves \
        the reading further than the height being measured. A tape measure is ±2 cm \
        and free, which is why the slider is the source of truth.
        """
    static let camera = """
        The drive-time estimator, and the only one of these that measures the \
        mount rather than guessing at it.

        Optical flow radiates from the direction the car is travelling. That \
        point is the vanishing point of travel, so its offset from the optical \
        axis IS the mount angle — and it is immune to road grade, because on a \
        hill the travel direction and the camera tilt together. Measured on \
        comma2k19 against carrier-phase GNSS/INS: 0.103° mean absolute error, \
        inside the 0.25° budget, converging in about 20 s of straight driving.

        Moving cars are excluded by consensus, not by recognising them. Flow \
        from everything nailed down agrees on one focus; a car moving \
        independently does not, and is outvoted.

        NIGHT DOES NOT WORK. The only trackable features after dark are \
        headlights and reflections, which move on their own — on the one night \
        segment tested, zero frames reached consensus. It fails by producing \
        nothing, which is why it is safe to leave running.
        """
    static let detections = """
        0.50 is the measured operating point: precision 0.647, recall 0.678. \
        Roughly a third of the boxes at that threshold are false, and that is \
        the model, not a bug in the decode — raise the threshold to trade \
        recall for a cleaner view.

        The shape filter is separate and cheap. The box coder exponentiates its \
        size codes, so a confident candidate can decode to a fourteen-metre car \
        or a box floating two metres off the road. Those are impossible rather \
        than merely unlikely, so they are dropped regardless of score.
        """
    static let focus = """
        Not cosmetic. Optical flow is built from local intensity differences and         blur is a low-pass filter, so a soft image starves the pitch estimator         long before it bothers the detector.

        There is no portable number for infinity. Lens position 1.0 is the far         MECHANICAL end, and actuators are built to travel past infinity so that         infinity stays reachable across temperature and unit spread — parking         there lands past focus. So: point at something far away, let autofocus         settle, then lock it. That finds the right value for this particular         camera instead of guessing one.

        Locking matters because a moving lens moves the focal length, which         rescales every range. An unlocked lens is workable but it is a caveat         on the whole drive, which is why it is shown rather than hidden.
        """
    static let yaw = """
        Camera heading minus course, so it needs the car moving above 5 m/s with a \
        trustworthy course. 1° is 0.70 m of lateral error at 40 m, applied to every \
        box — cheap, but not free.
        """
    static let roll = """
        No input needed. Unlike pitch, roll is honestly measurable from gravity: its \
        contaminant is road camber, ~0.6° and mean-zero, not the 2.45° of grade.

        Zero means the phone's +x edge — the right edge when held upright — points \
        up. A portrait mount reads −90°, and that is unrecoverable rather than merely \
        wrong: the capture buffer is landscape however the phone is held, so portrait \
        lays the road sideways across the crop.
        """
    static let diagnostics = """
        A sane windshield mount lands 50–70% of voxels on the feature map. Near zero \
        means the pitch sign is flipped.

        The heading cross-check compares the attitude-derived camera bearing against \
        CLHeading, which comes from a different stack. A persistent 90° or 180° gap \
        means the attitude convention is wrong, not the mount.
        """
}

struct CalibrationView: View {
    @ObservedObject private var live = LiveSession.shared
    @StateObject private var model = CalibrationModel()

    private var snapshot: FramePipeline.Snapshot { live.snapshot }

    var body: some View {
        NavigationStack {
            List {
                focusSection
                cameraSection
                pitchSection
                heightSection
                yawSection
                rollSection
                detectionSection
                diagnosticsSection
                Section {
                    Button("Reset to defaults", role: .destructive) { model.reset() }
                }
            }
            .navigationTitle("Calibrate")
        }
        .task { await model.attach() }
    }

    // MARK: - Focus

    private var focusSection: some View {
        Section {
            LabeledContent("state") {
                Text(focusState)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(focusColour)
            }
            LabeledContent("lens position") {
                Text(String(format: "%.3f", snapshot.lensPosition))
                    .font(.system(.body, design: .monospaced))
            }
            Button("Point at something far away, then lock focus here") {
                model.lockFocus()
            }
            Button("Hand it back to autofocus") { model.autoFocus() }
                .font(.caption)
            HStack {
                Text(String(format: "%.2f", model.lensPosition))
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 70, alignment: .leading)
                Slider(value: $model.lensPosition, in: 0...1, step: 0.01)
            }
            Button("Lock at that position") { model.applyLensPosition() }
                .font(.caption)
        } header: {
            Text("Focus")
        } footer: {
            Text(Help.focus).font(.caption2)
        }
    }

    private var focusState: String {
        if snapshot.focusHunting { return "hunting" }
        return snapshot.focusLocked ? "locked" : "autofocus (far)"
    }

    private var focusColour: Color {
        if snapshot.focusHunting { return .orange }
        return snapshot.focusLocked ? .green : .yellow
    }

    // MARK: - Camera estimator

    private var cameraSection: some View {
        Section {
            LabeledContent("state") {
                Text(cameraState)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(cameraStateColour)
            }
            if snapshot.foe.reportable {
                LabeledContent("pitch") {
                    Text(String(format: "%+.2f° ± %.2f°",
                                snapshot.foe.pitchDegrees, snapshot.foe.sigmaDegrees))
                        .font(.system(.body, design: .monospaced))
                }
                LabeledContent("yaw") {
                    Text(String(format: "%+.2f°", snapshot.foe.yawDegrees))
                        .font(.system(.body, design: .monospaced))
                }
                LabeledContent("window") {
                    Text("\(snapshot.foe.samples) / \(snapshot.foe.windowSize) "
                         + "· \(snapshot.foe.inliers) inliers")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                LabeledContent("accepted") {
                    Text(String(format: "%.0f%% of recent pairs",
                                snapshot.foe.acceptRate * 100))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                // The ablation, live. On comma2k19 de-rotation moved the answer
                // by 0.00-0.12 deg and helped every time. A LARGE gap here means
                // the gyro-to-camera axis mapping is wrong, not that the
                // correction is working hard.
                LabeledContent("without de-rotation") {
                    Text(String(format: "%+.2f°  (Δ%+.2f°)",
                                snapshot.foe.pitchWithoutDerotationDegrees,
                                snapshot.foe.pitchDegrees
                                    - snapshot.foe.pitchWithoutDerotationDegrees))
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(derotationColour)
                }
            }
            Toggle("Apply it automatically once converged", isOn: $model.autoApplyFOE)
            Button("Use this pitch now") { model.applyEstimatedPitch() }
                .disabled(!canApplyFOE)
            Button("Use this yaw now") { model.applyEstimatedYaw() }
                .disabled(!canApplyFOE)
            if !snapshot.foe.gates.writesToPose {
                Text("The Flow tab has the estimator in handheld mode. That "
                     + "measures the angle of your hand, so it cannot be "
                     + "applied to the mount.")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            Button("Start the window again") { model.resetFOE() }
                .font(.caption)
        } header: {
            Text("Drive-time pitch — camera")
        } footer: {
            Text(Help.camera).font(.caption2)
        }
    }

    private var canApplyFOE: Bool {
        snapshot.foe.reportable && snapshot.foe.gates.writesToPose
    }

    private var cameraState: String {
        let foe = snapshot.foe
        if foe.converged { return "converged" }
        if foe.reportable { return "settling" }
        return foe.gate.isEmpty ? "warming up" : foe.gate
    }

    private var cameraStateColour: Color {
        let foe = snapshot.foe
        if foe.converged { return .green }
        if foe.reportable { return .orange }
        return .secondary
    }

    /// De-rotation should be a small correction. Anything past a degree is an
    /// axis-convention bug wearing the costume of a working feature.
    private var derotationColour: Color {
        let delta = abs(snapshot.foe.pitchDegrees
                        - snapshot.foe.pitchWithoutDerotationDegrees)
        return delta > 1.0 ? .red : .secondary
    }

    // MARK: - Detections

    private var detectionSection: some View {
        Section {
            HStack {
                Text(String(format: "%.2f", model.scoreThreshold))
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 70, alignment: .leading)
                Slider(value: $model.scoreThreshold, in: 0.20...0.95, step: 0.01)
            }
            .onChange(of: model.scoreThreshold) { _, value in
                model.applyScoreThreshold(value)
            }
            Text(model.scoreThreshold == Contract.scoreThreshold
                 ? "the measured operating point"
                 : String(format: "%+.2f from the measured operating point",
                          model.scoreThreshold - Contract.scoreThreshold))
                .font(.caption2)
                .foregroundStyle(.secondary)
            Toggle("Drop impossible box shapes", isOn: $model.rejectImplausible)
            LabeledContent("drawn now", value: "\(snapshot.detections.count) boxes")
        } header: {
            Text("Detections")
        } footer: {
            Text(Help.detections).font(.caption2)
        }
    }

    // MARK: - Pitch

    private var pitchSection: some View {
        Section {
            LabeledContent("in use") {
                Text(String(format: "%+.2f°  (%@)",
                            snapshot.pose.pitchDegrees, snapshot.pose.pitchFrom.label))
                    .font(.system(.body, design: .monospaced))
            }
            LabeledContent("gravity says") {
                Text(String(format: "%+.2f°", snapshot.gravityPitchDegrees))
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            HStack {
                Text(String(format: "%+.2f°", model.pitchDegrees))
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 70, alignment: .leading)
                Slider(value: $model.pitchDegrees, in: -15...15, step: 0.05)
            }
            HStack {
                Button("−0.25°") { model.nudgePitch(-0.25) }
                Spacer()
                Button("−0.05°") { model.nudgePitch(-0.05) }
                Spacer()
                Button("+0.05°") { model.nudgePitch(0.05) }
                Spacer()
                Button("+0.25°") { model.nudgePitch(0.25) }
            }
            .buttonStyle(.bordered)
            .font(.system(.caption, design: .monospaced))
            Button("Use this as the starting guess") { model.applyPitch() }
            Button("Hand it back to gravity") { model.releasePitch() }
                .font(.caption)
        } header: {
            Text("Mount pitch")
        } footer: {
            Text(Help.pitchDegrees)
                .font(.caption2)
        }
    }

    // MARK: - Height

    private var heightSection: some View {
        Section {
            LabeledContent("in use") {
                Text(String(format: "%.2f m  (%@)",
                            snapshot.pose.height, snapshot.pose.heightFrom.label))
                    .font(.system(.body, design: .monospaced))
            }
            HStack {
                Text(String(format: "%.2f m", model.height))
                    .font(.system(.body, design: .monospaced))
                    .frame(width: 70, alignment: .leading)
                Slider(value: $model.height, in: 0.6...2.2, step: 0.01)
            }
            Button("Use this height") { model.applyHeight() }

            switch model.calibrator.phase {
            case .idle, .done:
                Button("Measure it with the barometer") { model.calibrator.startFloor() }
            case .floor:
                holdRow("Hold still on the floor")
            case .moved:
                Text("Now put the phone in the mount.")
                    .font(.caption).foregroundStyle(.secondary)
                Button("I've mounted it — measure") { model.calibrator.startMount() }
                Button("Cancel", role: .cancel) { model.calibrator.cancel() }
            case .mount:
                holdRow("Hold still in the mount")
            }

            if let outcome = model.calibrator.outcome {
                LabeledContent("measured") {
                    Text(String(format: "%.2f ± %.2f m", outcome.height, outcome.sigma))
                        .font(.system(.body, design: .monospaced))
                }
                ForEach(outcome.warnings, id: \.self) { warning in
                    Text(warning).font(.caption2).foregroundStyle(.orange)
                }
                if outcome.warnings.isEmpty {
                    Button("Use the measured height") { model.applyMeasuredHeight() }
                }
            }
        } header: {
            Text("Camera height")
        } footer: {
            Text(Help.height)
                .font(.caption2)
        }
    }

    private func holdRow(_ title: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption)
            ProgressView(value: model.calibrator.progress)
            Text("\(model.calibrator.samplesInHold) samples")
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - Yaw

    private var yawSection: some View {
        Section {
            LabeledContent("in use") {
                Text(String(format: "%+.2f°  (%@)",
                            snapshot.pose.yawDegrees, snapshot.pose.yawFrom.label))
                    .font(.system(.body, design: .monospaced))
            }
            LabeledContent("measured") {
                Text(snapshot.measuredYawDegrees.map { String(format: "%+.2f°", $0) }
                     ?? "needs a fix above 5 m/s")
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Button("Use the measured yaw") { model.applyYaw() }
                .disabled(snapshot.measuredYawDegrees == nil)
            Button("Clear to zero") { model.clearYaw() }.font(.caption)
        } header: {
            Text("Mount yaw")
        } footer: {
            Text(Help.yaw)
                .font(.caption2)
        }
    }

    // MARK: - Roll

    private var rollSection: some View {
        Section {
            LabeledContent("measured") {
                Text(String(format: "%+.2f°", snapshot.measuredRollDegrees))
                    .font(.system(.body, design: .monospaced))
            }
            LabeledContent("applied") {
                Text(String(format: "%+.2f°  (%@)",
                            snapshot.pose.rollDegrees, snapshot.pose.rollFrom.label))
                    .font(.system(.body, design: .monospaced))
            }
            if !snapshot.mountWarning.isEmpty {
                Text(snapshot.mountWarning).font(.caption2).foregroundStyle(.orange)
            }
        } header: {
            Text("Mount roll")
        } footer: {
            Text(Help.roll)
                .font(.caption2)
        }
    }

    // MARK: - Diagnostics

    private var diagnosticsSection: some View {
        Section {
            LabeledContent("visible voxels") {
                Text(visibleText)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(visibleColor)
            }
            LabeledContent("focal") {
                Text(focalText)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(focalColor)
            }
            LabeledContent("crop", value: snapshot.cropDescription)
            LabeledContent("reference frame", value: referenceFrameText)
            LabeledContent("heading vs compass", value: model.headingCrossCheck)
        } header: {
            Text("Diagnostics")
        } footer: {
            Text(Help.diagnostics).font(.caption2)
        }
    }

    // Precomputed so the ViewBuilder above stays trivial to type-check.
    private var visibleText: String {
        String(format: "%.1f%%", snapshot.visibleFraction * 100)
    }
    private var visibleColor: Color {
        snapshot.visibleFraction > 0.35 ? .primary : .red
    }
    private var focalText: String {
        let suffix = snapshot.focalMatched ? "" : " — UNMATCHED"
        return String(format: "%.1f px", snapshot.focal) + suffix
    }
    private var focalColor: Color {
        snapshot.focalMatched ? .primary : .orange
    }
    private var referenceFrameText: String {
        model.trueNorth ? "true north" : "arbitrary — yaw unusable"
    }
}

@MainActor
final class CalibrationModel: ObservableObject {
    @Published var pitchDegrees: Double = 0
    @Published var height: Double = 1.20
    @Published var calibrator = HeightCalibrator()
    @Published var scoreThreshold: Double = Contract.scoreThreshold
    @Published var lensPosition: Double = 1.0
    @Published var rejectImplausible = true {
        didSet { pipeline.setRejectImplausible(rejectImplausible) }
    }
    @Published var autoApplyFOE = true {
        didSet { pipeline.setAutoApplyFOE(autoApplyFOE) }
    }

    private var pipeline: FramePipeline { LiveSession.shared.pipeline }
    private var attached = false

    var trueNorth: Bool { pipeline.motion.headingIsTrueNorth }

    /// Two independent estimates of the same bearing. They come from different
    /// stacks, so agreement is evidence and disagreement is a bug -- see the
    /// footer.
    var headingCrossCheck: String {
        guard let attitude = pipeline.motion.latestAttitude?.cameraHeading,
              let compass = pipeline.motion.latestTrueHeading else { return "—" }
        let mine = attitude * 180.0 / .pi
        let delta = MotionSource.wrapToPi((mine - compass) * .pi / 180.0) * 180.0 / .pi
        return String(format: "%.0f° vs %.0f°  (Δ%+.0f°)", mine < 0 ? mine + 360 : mine,
                      compass, delta)
    }

    func attach() async {
        await LiveSession.shared.ensureStarted()
        guard !attached else { return }
        attached = true
        pitchDegrees = pipeline.currentPose.pitchDegrees
        height = pipeline.currentPose.height
        scoreThreshold = pipeline.currentScoreThreshold
        lensPosition = Double(pipeline.camera.lensPosition)
        rejectImplausible = pipeline.currentRejectImplausible
        autoApplyFOE = pipeline.currentAutoApplyFOE
        // The altimeter stream already feeds the recorder; tee it here too
        // rather than opening a second CMAltimeter.
        let existing = pipeline.motion.onAltitude
        pipeline.motion.onAltitude = { [weak self] sample in
            existing?(sample)
            Task { @MainActor in
                self?.calibrator.feed(relativeAltitude: sample.relativeAltitude,
                                      at: sample.timestamp)
            }
        }
    }

    func nudgePitch(_ delta: Double) {
        pitchDegrees = ((pitchDegrees + delta) * 100).rounded() / 100
    }

    func applyPitch() { pipeline.setPitch(degrees: pitchDegrees) }
    func releasePitch() {
        pipeline.releasePitchToGravity()
        pitchDegrees = pipeline.currentPose.pitchDegrees
    }
    func applyHeight() { pipeline.setHeight(height) }
    func applyMeasuredHeight() {
        guard let outcome = calibrator.outcome else { return }
        height = outcome.height
        pipeline.setHeight(outcome.height, from: .barometer)
    }
    func applyYaw() { pipeline.applyMeasuredYaw() }
    func clearYaw() { pipeline.clearYaw() }
    func applyEstimatedPitch() {
        pipeline.applyEstimatedPitch()
        pitchDegrees = pipeline.currentPose.pitchDegrees
    }
    func applyEstimatedYaw() { pipeline.applyEstimatedYaw() }
    func resetFOE() { pipeline.foe.reset() }
    func applyScoreThreshold(_ value: Double) { pipeline.setScoreThreshold(value) }

    /// Freeze the lens where autofocus has just put it. The only reliable way
    /// to get infinity: it is a different number on every camera.
    func lockFocus() {
        lensPosition = Double(pipeline.camera.lockFocusHere())
    }
    func autoFocus() { pipeline.camera.apply(.autoFar) }
    func applyLensPosition() {
        pipeline.camera.apply(.locked(Float(lensPosition)))
    }

    func reset() {
        pipeline.resetPose()
        pitchDegrees = pipeline.currentPose.pitchDegrees
        height = pipeline.currentPose.height
        calibrator.cancel()
    }
}
