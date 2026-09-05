//  LiveEgoMotion.swift
//  Timestamp-aligned Live adapter from phone sensors to neutral EgoDelta.

import Foundation

final class LiveEgoMotion: @unchecked Sendable {
    private struct Rate { var t: Double; var yaw: Double }
    private let lock = NSLock()
    private var rates: [Rate] = []
    private var locations: [MotionSource.Location] = []
    private var previousFrame: Double?

    func append(attitude: MotionSource.Attitude) {
        lock.withLock {
            // Project device-axis angular velocity onto gravity-up. At the
            // nominal landscape mount this is +device-x, while this form also
            // remains correct through the measured phone roll and pitch.
            let gravityLength = sqrt(attitude.gravity.x * attitude.gravity.x
                                   + attitude.gravity.y * attitude.gravity.y
                                   + attitude.gravity.z * attitude.gravity.z)
            let up = gravityLength > 1e-6 ? -attitude.gravity / gravityLength : .zero
            rates.append(Rate(t: attitude.timestamp,
                              yaw: attitude.rotationRate.x * up.x
                                 + attitude.rotationRate.y * up.y
                                 + attitude.rotationRate.z * up.z))
            rates.removeAll { $0.t < attitude.timestamp - 3 }
        }
    }

    func append(location: MotionSource.Location) {
        lock.withLock {
            locations.append(location)
            locations.removeAll { $0.timestamp < location.timestamp - 10 }
        }
    }

    func delta(at timestamp: Double) -> EgoDelta {
        lock.withLock {
            guard timestamp.isFinite, let previous = previousFrame else {
                previousFrame = timestamp
                return EgoDelta(dt: 0, valid: false)
            }
            let dt = timestamp - previous
            previousFrame = timestamp
            guard dt > 0, dt <= 1 else { return EgoDelta(dt: dt, valid: false) }

            let interval = rates.filter { $0.t > previous && $0.t <= timestamp }
            let latestRate = rates.last(where: { $0.t <= timestamp })
            guard !interval.isEmpty || latestRate.map({ timestamp - $0.t <= 0.2 }) == true else {
                return EgoDelta(dt: dt, valid: false)
            }
            let yawRate = interval.isEmpty ? latestRate!.yaw
                : interval.reduce(0) { $0 + $1.yaw } / Double(interval.count)
            let location = locations.last { $0.timestamp <= timestamp && $0.speed >= 0 }
            guard let location, timestamp - location.timestamp <= 2 else {
                return EgoDelta(dt: dt, valid: false)
            }
            let speed = location.speed
            return Self.arcDelta(speed: speed, yawRate: yawRate, dt: dt)
        }
    }

    /// Constant-twist arc expressed in the old ego frame.
    static func arcDelta(speed: Double, yawRate: Double, dt: Double) -> EgoDelta {
        let dyaw = yawRate * dt
        if abs(yawRate) < 1e-6 {
            return EgoDelta(dx: speed * dt, dy: 0, dyaw: dyaw, dt: dt)
        }
        return EgoDelta(dx: speed * sin(dyaw) / yawRate,
                        dy: speed * (1 - cos(dyaw)) / yawRate,
                        dyaw: dyaw, dt: dt)
    }

    func reset() {
        lock.withLock { previousFrame = nil; rates.removeAll(); locations.removeAll() }
    }
}
