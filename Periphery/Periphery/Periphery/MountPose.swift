//  MountPose.swift
//  Where the camera sits and which way it points -- and, for each number,
//  where that number came from.
//
//  These six values were previously six literals scattered across
//  FramePipeline, Benchmark and SelfCheck. They are the numbers every range in
//  the pipeline is scaled by, so they get one type, one place to persist, and a
//  provenance tag each.
//
//  Provenance is not decoration. A gravity-derived pitch is `mount + road
//  grade` and cannot separate the two -- measured at 2.45 deg p95 against a
//  1.00 deg failure line -- so the UI has to be able to show it as the PRIOR it
//  is rather than as an answer. The pipeline treats every pose identically;
//  only the display cares.
//
//  Angles are radians throughout, degrees only at the UI edge.

import Foundation

struct MountPose: Codable, Equatable, Sendable {

    /// Where a number came from, worst to best.
    enum Provenance: String, Codable, Sendable, CaseIterable {
        /// The built-in starting guess. Measured on nobody's car.
        case fallback
        /// Typed or dialled in by a person.
        case manual
        /// asin(gravity.z) for pitch, or the gravity vector's tilt for roll.
        /// Exact for roll; grade-blind for pitch.
        case gravity
        /// The floor-to-mount barometric lift.
        case barometer
        /// The drive-time run/rise/level estimator.
        case estimated

        var label: String {
            switch self {
            case .fallback: return "default"
            case .manual: return "manual"
            case .gravity: return "gravity"
            case .barometer: return "baro"
            case .estimated: return "drive"
            }
        }

        /// Precedence. An automatic source may only overwrite a value whose
        /// rank it meets or beats; an explicit action by a person always wins,
        /// because it is a person choosing.
        ///
        /// The point of the ordering is that a typed-in pitch is an INITIAL
        /// GUESS, not a lock. It has to outrank gravity, which measures
        /// mount + road grade and would walk a good number back over a few
        /// seconds. It must NOT outrank the drive-time estimator, which is the
        /// thing the guess exists to seed.
        var rank: Int {
            switch self {
            case .fallback: return 0
            case .gravity: return 1
            case .manual, .barometer: return 2
            case .estimated: return 3
            }
        }

        /// May a sample from `self` overwrite a value that currently came from
        /// `current`, with no person in the loop?
        func mayOverwrite(_ current: Provenance) -> Bool { rank >= current.rank }

        /// True when the value is a starting point rather than a measurement of
        /// the mount itself.
        var isPrior: Bool { self == .fallback || self == .gravity }
    }

    // MARK: - The pose

    /// Positive NOSE-UP, radians. Against the direction of travel, not gravity.
    var pitch: Double
    /// Positive when the camera is rolled CLOCKWISE seen from behind, i.e. its
    /// left side is higher. Radians.
    var roll: Double
    /// Positive when the camera points LEFT of the direction of travel, ISO
    /// 8855 z-up. Radians.
    var yaw: Double
    /// Camera height above the road, metres. Forgiving: d(range)/range =
    /// d(height)/height, so this is a pure scale factor.
    var height: Double
    /// Camera position forward of the vehicle origin, metres.
    var forwardOfOrigin: Double

    var pitchFrom: Provenance
    var rollFrom: Provenance
    var yawFrom: Provenance
    var heightFrom: Provenance

    /// One-sigma uncertainty on pitch, degrees, when whoever set it knew.
    /// The budget from note 12: 0.25 design target, 0.50 tolerable, 1.00 a
    /// declared failure that must be visible rather than silently absorbed.
    var pitchSigmaDegrees: Double?

    // MARK: - Degrees, for the UI only

    var pitchDegrees: Double {
        get { pitch * 180.0 / .pi }
        set { pitch = newValue * .pi / 180.0 }
    }
    var rollDegrees: Double {
        get { roll * 180.0 / .pi }
        set { roll = newValue * .pi / 180.0 }
    }
    var yawDegrees: Double {
        get { yaw * 180.0 / .pi }
        set { yaw = newValue * .pi / 180.0 }
    }

    // MARK: - Defaults

    /// The pose the app starts from. -3.659 deg and 1.20 m are the comma EON
    /// rig the corpus was measured on; they are a plausible windshield mount and
    /// nothing more, which is what `.fallback` says.
    static let fallback = MountPose(pitch: -3.659 * .pi / 180.0,
                                    roll: 0.0,
                                    yaw: 0.0,
                                    height: 1.20,
                                    forwardOfOrigin: 1.65,
                                    pitchFrom: .fallback,
                                    rollFrom: .fallback,
                                    yawFrom: .fallback,
                                    heightFrom: .fallback,
                                    pitchSigmaDegrees: nil)

    // MARK: - Sanity

    /// Plausible windshield mounts only. Outside these the projection is not
    /// wrong so much as meaningless, and it is better to say so than to draw
    /// boxes on the sky.
    var isPlausible: Bool {
        abs(pitchDegrees) <= 25.0
            && abs(rollDegrees) <= 25.0
            && abs(yawDegrees) <= 25.0
            && height > 0.3 && height < 3.0
            && forwardOfOrigin > -2.0 && forwardOfOrigin < 6.0
    }

    /// A short, honest one-liner for the stats strip.
    var summary: String {
        String(format: "pitch %+.2f (%@) · roll %+.2f (%@) · yaw %+.2f (%@) · h %.2f m (%@)",
               pitchDegrees, pitchFrom.label,
               rollDegrees, rollFrom.label,
               yawDegrees, yawFrom.label,
               height, heightFrom.label)
    }

    // MARK: - Persistence

    private static let storageKey = "MountPose.v1"

    /// Round-trips through UserDefaults. A mount survives an app restart; it
    /// does not survive being moved, which is what the drive-time estimator is
    /// for.
    static func load(from defaults: UserDefaults = .standard) -> MountPose {
        guard let data = defaults.data(forKey: storageKey),
              let pose = try? JSONDecoder().decode(MountPose.self, from: data),
              pose.isPlausible else {
            return .fallback
        }
        return pose
    }

    func save(to defaults: UserDefaults = .standard) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.storageKey)
    }

    static func clear(from defaults: UserDefaults = .standard) {
        defaults.removeObject(forKey: storageKey)
    }
}
