//  MotionSource.swift
//  The one owner of CoreMotion, CoreLocation and the barometer.
//
//  Previously FramePipeline started its own CMMotionManager and delivered
//  updates to `.main`, which put a 30 Hz attitude stream behind the UI's run
//  loop and left nothing for anyone else to read. Every consumer -- the pose,
//  the drive recorder, the pitch estimator -- wants the same samples, so there
//  is one source and it publishes.
//
//  Three things here are easy to get wrong and expensive to debug:
//
//  1. THE REFERENCE FRAME. `attitude.yaw` is measured against whatever frame
//     the manager started in. `CLLocation.course` is true north. They are only
//     comparable under `.xTrueNorthZVertical`, which needs location
//     authorisation to be live BEFORE motion starts. Get this wrong and
//     `deviceYaw - course` is a random number that looks plausible.
//
//  2. THE CLOCK DOMAINS. CMDeviceMotion.timestamp and CMAltitudeData.timestamp
//     are seconds since boot -- the mach_absolute_time domain, the same one
//     CMSampleBufferGetPresentationTimeStamp uses, so motion, altimeter and
//     video align for free. CLLocation.timestamp is an NSDate: WALL CLOCK, a
//     different domain, and it can jump when network time updates. Everything
//     here is republished in the boot domain, with the anchor recorded so the
//     conversion is auditable rather than assumed.
//
//  3. THE CAMERA AXES. Roll is only meaningful against the camera's own axes,
//     not the device's. See `cameraRoll`.

import CoreLocation
import CoreMotion
import Foundation
import simd

/// Everything the pipeline, the recorder and the estimator read.
final class MotionSource: NSObject, @unchecked Sendable {

    // MARK: - Samples

    /// CoreMotion, at its native rate. Raw values plus the two derived angles,
    /// because deriving them twice in two consumers is how they drift apart.
    struct Attitude: Sendable {
        /// Seconds since boot (mach_absolute_time domain).
        var timestamp: TimeInterval
        var gravity: SIMD3<Double>
        var userAcceleration: SIMD3<Double>
        var rotationRate: SIMD3<Double>
        /// Attitude as a quaternion in the active reference frame.
        var quaternion: simd_quatd
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
        /// The original NSDate value, seconds since 1970, kept verbatim.
        var wallTimestamp: TimeInterval
        var latitude: Double
        var longitude: Double
        var altitude: Double
        var horizontalAccuracy: Double
        var verticalAccuracy: Double
        /// HORIZONTAL ground speed, m/s, Doppler. Negative means invalid.
        /// There is no vertical-speed property, which is why the barometer is
        /// not optional.
        var speed: Double
        var speedAccuracy: Double
        var course: Double
        var courseAccuracy: Double
    }

    /// CMAltimeter, ~1 Hz. Relative only -- the absolute value is useless.
    struct Altitude: Sendable {
        var timestamp: TimeInterval
        var relativeAltitude: Double
        var pressureKPa: Double
    }

    /// Wall clock against boot clock, so CoreLocation can be placed on the same
    /// timeline as the video and the IMU offline. Re-sampled periodically
    /// because the wall clock drifts and can step.
    struct ClockAnchor: Sendable {
        var wallSeconds: TimeInterval
        var bootSeconds: TimeInterval
    }

    // MARK: - Configuration

    /// 100 Hz. Note 13 wants attitude averaged over each accepted interval at
    /// CoreMotion's own rate; pairing it with CoreLocation by index would
    /// silently decimate it to 1 Hz.
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
    /// CLHeading.trueHeading, degrees. An INDEPENDENT witness to
    /// `Attitude.cameraHeading`: the two are derived from different stacks, so
    /// a persistent 90 or 180 degree gap between them means the attitude-matrix
    /// convention below is wrong. Surfaced in the Calibrate tab for exactly
    /// that reason.
    var latestTrueHeading: Double? { lock.withLock { _latestTrueHeading } }
    var clockAnchor: ClockAnchor { lock.withLock { _anchor } }

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
    private(set) var altimeterAvailable = false
    private(set) var running = false

    /// True when `attitude.cameraHeading` means anything, i.e. when the
    /// reference frame is true-north locked. Mount yaw is not computable
    /// otherwise.
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

    /// Location first, then motion. The ordering is the point: the true-north
    /// reference frame is only available once location services are authorised
    /// and running, so starting motion first silently downgrades the frame and
    /// mount yaw quietly becomes meaningless.
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
        altimeterAvailable = true
        altimeter.startRelativeAltitudeUpdates(to: queue) { [weak self] data, _ in
            guard let self, let data else { return }
            let sample = Altitude(timestamp: data.timestamp,
                                  relativeAltitude: data.relativeAltitude.doubleValue,
                                  pressureKPa: data.pressure.doubleValue)
            self.lock.withLock { self._latestAltitude = sample }
            self.onAltitude?(sample)
        }
    }

    /// A fresh reading of the wall clock against the boot clock, taken
    /// adjacently. This is the pair that makes `CLLocation.timestamp` placeable
    /// on the video's timeline; 10 ms of sync error at 30 deg/s is 0.3 deg, the
    /// entire pitch budget, so it is not bookkeeping.
    static func sampleAnchor() -> ClockAnchor {
        ClockAnchor(wallSeconds: Date().timeIntervalSince1970,
                    bootSeconds: ProcessInfo.processInfo.systemUptime)
    }

    /// The session's anchor is fixed at `start()` and never moved: re-anchoring
    /// mid-session would step every republished CoreLocation timestamp
    /// discontinuously, which is worse than the drift it would correct. Drift
    /// is instead handled by RECORDING fresh anchors periodically as data, so
    /// it is measurable offline rather than silently absorbed here.
    private func setAnchor() {
        let anchor = Self.sampleAnchor()
        lock.withLock { _anchor = anchor }
    }

    // MARK: - Derived angles

    private func attitude(from m: CMDeviceMotion) -> Attitude {
        let g = SIMD3<Double>(m.gravity.x, m.gravity.y, m.gravity.z)
        let q = m.attitude.quaternion
        return Attitude(
            timestamp: m.timestamp,
            gravity: g,
            userAcceleration: SIMD3<Double>(m.userAcceleration.x,
                                            m.userAcceleration.y,
                                            m.userAcceleration.z),
            rotationRate: SIMD3<Double>(m.rotationRate.x, m.rotationRate.y, m.rotationRate.z),
            quaternion: simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w),
            gravityPitch: Self.gravityPitch(g),
            roll: Self.cameraRoll(g),
            cameraHeading: headingIsTrueNorth ? Self.cameraHeading(m.attitude) : nil)
    }

    /// Camera elevation above horizontal, radians, positive nose-up.
    ///
    /// The rear camera looks along -z in device axes, so the elevation of the
    /// optical axis is asin(g.z). Independent of roll, which is why it survives
    /// any mounting rotation.
    ///
    /// This is `mount + road grade`. Gravity measures tilt against the EARTH
    /// and cannot see past the road: measured over 237 drive segments it
    /// regresses on grade with slope +1.006, costing 2.45 deg p95 against a
    /// 1.00 deg failure line. A prior, never the answer.
    static func gravityPitch(_ gravity: SIMD3<Double>) -> Double {
        asin(max(-1.0, min(1.0, gravity.z)))
    }

    /// Camera roll, radians, positive clockwise seen from behind the camera
    /// (the camera's left side higher).
    ///
    /// Device axes are +x right, +y top edge, +z out of the screen. The rear
    /// camera's native buffer needs a 90 degree CLOCKWISE rotation to sit
    /// upright in portrait, which puts the buffer's left edge at the display's
    /// top. That fixes the camera axes in device terms:
    ///
    ///     image right  u = -y      image down  v = -x      optical  w = -z
    ///
    /// and u x v = w confirms the handedness. Gravity's components in the image
    /// plane are then g_u = -g.y and g_v = -g.x, and a camera rolled clockwise
    /// tips gravity toward its left, i.e. toward -u. Hence:
    ///
    ///     roll = atan2(-g_u, g_v) = atan2(g.y, -g.x)
    ///
    /// Zero roll therefore means the device's +x edge points UP -- landscape,
    /// rotated counterclockwise from portrait. A PORTRAIT MOUNT READS -90 deg,
    /// which is the point: the capture buffer is always landscape regardless of
    /// how the phone is held, so a portrait mount does not rotate the image, it
    /// lays the road sideways across the crop. That is unrecoverable here, and
    /// reading -90 is how it becomes visible instead of silent.
    ///
    /// Unlike pitch, roll is honestly measurable from gravity: its contaminant
    /// is road camber, ~0.6 deg and mean-zero, not the 2.45 deg of grade.
    static func cameraRoll(_ gravity: SIMD3<Double>) -> Double {
        atan2(gravity.y, -gravity.x)
    }

    /// Camera bearing, radians clockwise from true north.
    ///
    /// Only meaningful under `.xTrueNorthZVertical`, whose reference frame is
    /// +x north, +z up, and therefore +y west. The rotation matrix maps the
    /// reference frame into the device frame, so its transpose carries the
    /// optical axis (-z in device axes) back out to world coordinates.
    static func cameraHeading(_ attitude: CMAttitude) -> Double {
        let m = attitude.rotationMatrix
        // m maps the reference frame INTO the device frame, so carrying the
        // optical axis back out uses the transpose. Transpose applied to
        // (0, 0, -1) is the negated third ROW of m.
        let north = -m.m31
        let west = -m.m32
        return atan2(-west, north)
    }

    /// Mount yaw: how far the camera points LEFT of the direction of travel,
    /// radians. Nearly free once heading and course are both in hand, and not
    /// optional -- 1 deg is 0.70 m of lateral error at 40 m, applied to every
    /// box.
    static func mountYaw(cameraHeading: Double, course: Double) -> Double {
        // Course is degrees clockwise from north; heading is radians the same
        // way. Left of travel is counterclockwise, so the offset is
        // course - heading.
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
        let anchor = clockAnchor
        for location in locations {
            let wall = location.timestamp.timeIntervalSince1970
            let sample = Location(
                // Boot domain, so this sits on the same timeline as the video
                // and the IMU without a second conversion at every consumer.
                timestamp: anchor.bootSeconds + (wall - anchor.wallSeconds),
                wallTimestamp: wall,
                latitude: location.coordinate.latitude,
                longitude: location.coordinate.longitude,
                altitude: location.altitude,
                horizontalAccuracy: location.horizontalAccuracy,
                verticalAccuracy: location.verticalAccuracy,
                speed: location.speed,
                speedAccuracy: location.speedAccuracy,
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

    /// Authorisation can arrive after `start()`. The true-north reference frame
    /// is unavailable until it does, so motion is restarted once it lands --
    /// otherwise the app spends the whole drive on an arbitrary frame and mount
    /// yaw is quietly meaningless.
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
