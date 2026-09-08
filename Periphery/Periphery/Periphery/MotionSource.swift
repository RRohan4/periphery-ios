
import CoreLocation
import CoreMotion
import Foundation
import simd

/// Everything the pipeline and the estimator read.
final class MotionSource: NSObject, @unchecked Sendable {

    // MARK: - Samples

    /// CoreMotion, at its native rate. Raw values plus the two derived angles,
    /// because deriving them twice in two consumers is how they drift apart.
    struct Attitude: Sendable {
        /// Seconds since boot (mach_absolute_time domain).
        var timestamp: TimeInterval
        var gravity: SIMD3<Double>
        var rotationRate: SIMD3<Double>
        /// Camera elevation above horizontal, radians, positive nose-up.
        /// This is `mount + road grade` and cannot separate the two.
        var gravityPitch: Double
        /// Camera roll, radians, positive clockwise seen from behind.
        var roll: Double
        /// Camera bearing, radians clockwise from true north, or nil when the
        /// reference frame is not true-north referenced.
        var cameraHeading: Double?
    }

    /// CoreLocation, ~1 Hz.
    struct Location: Sendable {
        /// Republished into the boot domain via the recorded anchor.
        var timestamp: TimeInterval
        var speed: Double
        var course: Double
        var courseAccuracy: Double
    }

    /// CMAltimeter, ~1 Hz. Relative only -- the absolute value is useless.
    struct Altitude: Sendable {
        var timestamp: TimeInterval
        var relativeAltitude: Double
    }

    struct ClockAnchor: Sendable {
        var wallSeconds: TimeInterval
        var bootSeconds: TimeInterval
    }

    // MARK: - Configuration

    var attitudeRate = 100.0

    // MARK: - Outputs

    /// Called on the motion queue, not the main thread.
    var onAttitude: ((Attitude) -> Void)?
    /// Called on the main queue, where CLLocationManager delivers.
    var onLocation: ((Location) -> Void)?
    /// Called on the motion queue.
    var onAltitude: ((Altitude) -> Void)?

    // MARK: - Latched state, for readers that only want "now"

    private let lock = NSLock()
    private var _latestAttitude: Attitude?
    private var _latestLocation: Location?
    private var _latestAltitude: Altitude?
    private var _latestTrueHeading: Double?
    private var _anchor = ClockAnchor(wallSeconds: 0, bootSeconds: 0)

    var latestAttitude: Attitude? { lock.withLock { _latestAttitude } }
    var latestLocation: Location? { lock.withLock { _latestLocation } }
    var latestAltitude: Altitude? { lock.withLock { _latestAltitude } }

    var latestTrueHeading: Double? { lock.withLock { _latestTrueHeading } }

    // MARK: - Machinery

    private let motion = CMMotionManager()
    private let altimeter = CMAltimeter()
    private let location = CLLocationManager()
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "com.periphery.motion"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInitiated
        return queue
    }()

    private(set) var referenceFrame: CMAttitudeReferenceFrame = .xArbitraryZVertical
    private(set) var running = false

    var headingIsTrueNorth: Bool { referenceFrame == .xTrueNorthZVertical }

    override init() {
        super.init()
        location.delegate = self
        location.desiredAccuracy = kCLLocationAccuracyBestForNavigation
        location.activityType = .automotiveNavigation
        // Course and speed are the whole point; do not let iOS coalesce them.
        location.distanceFilter = kCLDistanceFilterNone
    }

    // MARK: - Lifecycle

    func start() {
        guard !running else { return }
        running = true
        setAnchor()

        location.requestWhenInUseAuthorization()
        location.startUpdatingLocation()
        if CLLocationManager.headingAvailable() {
            location.startUpdatingHeading()
        }

        startAltimeter()
        startMotion()
    }

    func stop() {
        guard running else { return }
        running = false
        motion.stopDeviceMotionUpdates()
        altimeter.stopRelativeAltitudeUpdates()
        location.stopUpdatingLocation()
        location.stopUpdatingHeading()
    }

    private func startMotion() {
        guard motion.isDeviceMotionAvailable else { return }
        motion.deviceMotionUpdateInterval = 1.0 / attitudeRate

        let available = CMMotionManager.availableAttitudeReferenceFrames()
        referenceFrame = available.contains(.xTrueNorthZVertical)
            ? .xTrueNorthZVertical
            : .xArbitraryZVertical

        motion.startDeviceMotionUpdates(using: referenceFrame, to: queue) { [weak self] m, _ in
            guard let self, let m else { return }
            let sample = self.attitude(from: m)
            self.lock.withLock { self._latestAttitude = sample }
            self.onAttitude?(sample)
        }
    }

    private func startAltimeter() {
        guard CMAltimeter.isRelativeAltitudeAvailable() else { return }
        altimeter.startRelativeAltitudeUpdates(to: queue) { [weak self] data, _ in
            guard let self, let data else { return }
            let sample = Altitude(timestamp: data.timestamp,
                                  relativeAltitude: data.relativeAltitude.doubleValue)
            self.lock.withLock { self._latestAltitude = sample }
            self.onAltitude?(sample)
        }
    }

    static func sampleAnchor() -> ClockAnchor {
        ClockAnchor(wallSeconds: Date().timeIntervalSince1970,
                    bootSeconds: ProcessInfo.processInfo.systemUptime)
    }

    private func setAnchor() {
        let anchor = Self.sampleAnchor()
        lock.withLock { _anchor = anchor }
    }

    // MARK: - Derived angles

    private func attitude(from m: CMDeviceMotion) -> Attitude {
        let g = SIMD3<Double>(m.gravity.x, m.gravity.y, m.gravity.z)
        return Attitude(
            timestamp: m.timestamp,
            gravity: g,
            rotationRate: SIMD3<Double>(m.rotationRate.x, m.rotationRate.y, m.rotationRate.z),
            gravityPitch: Self.gravityPitch(g),
            roll: Self.cameraRoll(g),
            cameraHeading: headingIsTrueNorth ? Self.cameraHeading(m.attitude) : nil)
    }

    static func gravityPitch(_ gravity: SIMD3<Double>) -> Double {
        asin(max(-1.0, min(1.0, gravity.z)))
    }

    static func cameraRoll(_ gravity: SIMD3<Double>) -> Double {
        atan2(gravity.y, -gravity.x)
    }

    static func cameraHeading(_ attitude: CMAttitude) -> Double {
        let m = attitude.rotationMatrix

        let north = -m.m31
        let west = -m.m32
        return atan2(-west, north)
    }

    static func mountYaw(cameraHeading: Double, course: Double) -> Double {

        wrapToPi(course * .pi / 180.0 - cameraHeading)
    }

    static func wrapToPi(_ angle: Double) -> Double {
        var a = angle
        while a > .pi { a -= 2 * .pi }
        while a < -.pi { a += 2 * .pi }
        return a
    }
}

// MARK: - CoreLocation

extension MotionSource: CLLocationManagerDelegate {

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {

        let anchor = Self.sampleAnchor()
        lock.withLock { _anchor = anchor }
        for location in locations {
            let wall = location.timestamp.timeIntervalSince1970
            let sample = Location(
                // Boot domain, so this sits on the same timeline as the video
                // and the IMU without a second conversion at every consumer.
                timestamp: anchor.bootSeconds + (wall - anchor.wallSeconds),
                speed: location.speed,
                course: location.course,
                courseAccuracy: location.courseAccuracy)
            lock.withLock { _latestLocation = sample }
            onLocation?(sample)
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        guard newHeading.headingAccuracy >= 0 else { return }
        lock.withLock { _latestTrueHeading = newHeading.trueHeading }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        // Deliberately silent. A location fix that never arrives is a normal
        // condition in a garage, and the pose falls back to gravity.
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        guard running, !headingIsTrueNorth else { return }
        switch manager.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            motion.stopDeviceMotionUpdates()
            startMotion()
        default:
            break
        }
    }
}
