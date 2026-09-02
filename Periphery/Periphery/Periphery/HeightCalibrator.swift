//  HeightCalibrator.swift
//  Camera height from the barometric lift between the floor and the mount.
//
//  The physics works. Air pressure falls about 12 Pa per metre, so a 1.2 m lift
//  is ~0.14 hPa against roughly 0.02 hPa of per-sample sensor noise. Averaged
//  over a hold at each end that lands near +-5 cm, and height is forgiving
//  anyway: d(range)/range = d(height)/height, a pure scale factor, so 5 cm is
//  ~4% of range where the same effort spent on pitch buys far more.
//
//  THE LIMIT IS NOT SENSOR NOISE, IT IS CABIN PRESSURE. A door, the HVAC or a
//  window moves the reading further than the 1.2 m being measured. That is why
//  this reports the spread of its samples and the drift between the two holds
//  rather than just a number, and why the slider stays the source of truth: a
//  tape measure is +-2 cm and free.
//
//  CMAltimeter.relativeAltitude is a running total from when updates began, so
//  as long as MotionSource keeps the altimeter alive the two holds are directly
//  differenceable with no re-zeroing.

import Combine
import Foundation

@MainActor
final class HeightCalibrator: ObservableObject {

    enum Phase: Equatable {
        case idle
        /// Collecting with the phone on the floor of the car.
        case floor
        /// Floor captured; waiting for the phone to be moved to the mount.
        case moved
        /// Collecting with the phone in the mount.
        case mount
        case done
    }

    /// One hold: the mean relative altitude and how much it wandered.
    struct Reading: Equatable {
        var mean: Double
        var spread: Double          // sample standard deviation, metres
        var count: Int
        var startBoot: TimeInterval
        var endBoot: TimeInterval

        /// Standard error of the mean.
        var sigma: Double { count > 1 ? spread / Double(count).squareRoot() : spread }
    }

    struct Outcome: Equatable {
        var height: Double
        var sigma: Double
        var floor: Reading
        var mount: Reading
        /// Seconds between the end of the floor hold and the start of the mount
        /// hold. Barometer drift is ~2 mm/s, so a long gap is a real error term.
        var gap: TimeInterval
        var warnings: [String]
    }

    /// Long enough to average down the noise, short enough that nobody gives
    /// up. At ~1 Hz this is about 10 samples, so the standard error is roughly a
    /// third of the per-sample noise.
    static let holdSeconds: TimeInterval = 10

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var progress: Double = 0
    @Published private(set) var outcome: Outcome?
    @Published private(set) var samplesInHold = 0

    private var samples: [Double] = []
    private var holdStart: TimeInterval = 0
    private var floorReading: Reading?

    // MARK: - Driving it

    func startFloor() {
        outcome = nil
        floorReading = nil
        beginHold(.floor)
    }

    func startMount() {
        guard floorReading != nil else { return }
        beginHold(.mount)
    }

    func cancel() {
        phase = .idle
        progress = 0
        samples.removeAll()
        samplesInHold = 0
    }

    private func beginHold(_ next: Phase) {
        phase = next
        samples.removeAll()
        samplesInHold = 0
        progress = 0
        holdStart = ProcessInfo.processInfo.systemUptime
    }

    /// Fed from MotionSource's altimeter stream, ~1 Hz. Hopped to the main
    /// actor by the caller; at that rate the cost is irrelevant and the
    /// alternative is a lock around a UI-driven state machine.
    func feed(relativeAltitude: Double, at boot: TimeInterval) {
        guard phase == .floor || phase == .mount else { return }
        samples.append(relativeAltitude)
        samplesInHold = samples.count
        let elapsed = boot - holdStart
        progress = min(1.0, elapsed / Self.holdSeconds)
        guard elapsed >= Self.holdSeconds, samples.count >= 3 else { return }

        let reading = Self.summarise(samples, from: holdStart, to: boot)
        if phase == .floor {
            floorReading = reading
            phase = .moved
        } else if let floor = floorReading {
            outcome = Self.combine(floor: floor, mount: reading)
            phase = .done
        }
        samples.removeAll()
    }

    // MARK: - Arithmetic

    private static func summarise(_ values: [Double],
                                  from start: TimeInterval,
                                  to end: TimeInterval) -> Reading {
        let n = Double(values.count)
        let mean = values.reduce(0, +) / n
        let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / max(1, n - 1)
        return Reading(mean: mean, spread: variance.squareRoot(),
                       count: values.count, startBoot: start, endBoot: end)
    }

    /// The mount is ABOVE the floor, so the relative altitude rises and the
    /// height is `mount - floor`.
    private static func combine(floor: Reading, mount: Reading) -> Outcome {
        let height = mount.mean - floor.mean
        let sigma = (floor.sigma * floor.sigma + mount.sigma * mount.sigma).squareRoot()
        let gap = mount.startBoot - floor.endBoot

        var warnings = [String]()
        if height <= 0 {
            warnings.append("the mount read LOWER than the floor — the two holds "
                            + "were probably swapped, or a door changed the cabin pressure")
        }
        if height > 2.0 {
            warnings.append(String(format: "%.2f m is too tall for a windshield mount", height))
        }
        if sigma > 0.10 {
            warnings.append(String(format: "±%.0f cm — the pressure was moving. Close the "
                                   + "doors, leave the HVAC alone, and try again", sigma * 100))
        }
        // 2.3 mm/s of barometer random walk, so a slow calibration accumulates
        // drift that is indistinguishable from height.
        if gap > 60 {
            warnings.append(String(format: "%.0f s between holds — about %.0f cm of "
                                   + "barometer drift", gap, gap * 0.0023 * 100))
        }
        if max(floor.spread, mount.spread) > 0.15 {
            warnings.append("one hold was much noisier than the other")
        }
        return Outcome(height: height, sigma: sigma, floor: floor, mount: mount,
                       gap: gap, warnings: warnings)
    }
}
