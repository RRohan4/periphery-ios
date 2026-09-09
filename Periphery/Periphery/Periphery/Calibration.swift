// Mount pose, focal matching, and grid-to-feature projection. The transforms
// mirror the corresponding Python geometry.

import Foundation
import simd

/// A centre crop of the source frame plus the letterbox that maps it onto the
/// 512x256 network canvas.
struct ImageCrop {
    var x: Int
    var y: Int
    var width: Int
    var height: Int

    var resize: Double {
        min(Double(Contract.inputWidth) / Double(width),
            Double(Contract.inputHeight) / Double(height))
    }
    var scaledWidth: Int { Int((Double(width) * resize).rounded()) }
    var scaledHeight: Int { Int((Double(height) * resize).rounded()) }
    var offsetX: Int { (Contract.inputWidth - scaledWidth) / 2 }
    var offsetY: Int { (Contract.inputHeight - scaledHeight) / 2 }
}

struct Calibration {

    /// Where the camera is and which way it points.
    var pose: MountPose
    /// Intrinsics of the SOURCE frame, before any crop or resize.
    var K: simd_double3x3
    /// Size of the source frame, pixels.
    var frameWidth: Int
    var frameHeight: Int

    init(pose: MountPose, K: simd_double3x3, frameWidth: Int, frameHeight: Int) {
        self.pose = pose
        self.K = K
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
    }

    // MARK: Convenience

    var pitch: Double {
        get { pose.pitch }
        set { pose.pitch = newValue }
    }
    var roll: Double { pose.roll }
    var yaw: Double { pose.yaw }
    /// Camera height above the road, metres. Forgiving: d(range)/d(height) is
    /// range/height, so 1 cm is a pure 0.8% scale factor.
    var height: Double { pose.height }
    /// Camera position forward of the vehicle origin, metres.
    var forwardOfOrigin: Double { pose.forwardOfOrigin }

    // MARK: - Mount pose
    //this is the transform from the vehicle frame to the sensor frame
    var sensorTVehicle: simd_double4x4 {
        let rotation = Self.vehicleToSensor(pitch: pose.pitch, roll: pose.roll, yaw: pose.yaw)
        let camera = SIMD3<Double>(pose.forwardOfOrigin, 0.0, pose.height)
        let translation = -(rotation * camera)
        return simd_double4x4(rows: [
            SIMD4<Double>(rotation[0][0], rotation[1][0], rotation[2][0], translation.x),
            SIMD4<Double>(rotation[0][1], rotation[1][1], rotation[2][1], translation.y),
            SIMD4<Double>(rotation[0][2], rotation[1][2], rotation[2][2], translation.z),
            SIMD4<Double>(0.0, 0.0, 0.0, 1.0),
        ])
    }

    static func vehicleToSensor(pitch: Double, roll: Double, yaw: Double) -> simd_double3x3 {
        let cr = cos(-roll), sr = sin(-roll)
        let cp = cos(pitch), sp = sin(pitch)
        let cy = cos(-yaw), sy = sin(-yaw)
        let rx = simd_double3x3(rows: [
            SIMD3<Double>(1.0, 0.0, 0.0),
            SIMD3<Double>(0.0, cr, -sr),
            SIMD3<Double>(0.0, sr, cr),
        ])
        let ry = simd_double3x3(rows: [
            SIMD3<Double>(cp, 0.0, sp),
            SIMD3<Double>(0.0, 1.0, 0.0),
            SIMD3<Double>(-sp, 0.0, cp),
        ])
        let rz = simd_double3x3(rows: [
            SIMD3<Double>(cy, -sy, 0.0),
            SIMD3<Double>(sy, cy, 0.0),
            SIMD3<Double>(0.0, 0.0, 1.0),
        ])
        return rx * ry * rz
    }

    //take a real 3d point in car coords and turn it into a 2d point in the image
    func sourcePoint(_ vehicle: SIMD3<Double>) -> SIMD2<Double>? {
        let rotation = Self.vehicleToSensor(pitch: pose.pitch, roll: pose.roll, yaw: pose.yaw)
        let camera = SIMD3<Double>(pose.forwardOfOrigin, 0.0, pose.height)
        let image = Contract.sensorToImageAxes * (rotation * (vehicle - camera))
        guard image.z > 1e-6 else { return nil }
        let projected = K * image
        return SIMD2<Double>(projected.x / projected.z, projected.y / projected.z)
    }

    // MARK: - Focal matching
    //match the focal we trained on tot he focal we have
    func focalMatchedCrop(targetFocal: Double = Contract.trainedFocal) -> ImageCrop {
        let scale = targetFocal / K[0][0]
        var cropW = Int((Double(Contract.inputWidth) / scale).rounded())
        var cropH = Int((Double(Contract.inputHeight) / scale).rounded())
        cropW = min(cropW, frameWidth)
        cropH = min(cropH, frameHeight)
        var x0 = Int((K[2][0] - Double(cropW) / 2.0).rounded())
        var y0 = Int((K[2][1] - Double(cropH) / 2.0).rounded())
        x0 = max(0, min(x0, frameWidth - cropW))
        y0 = max(0, min(y0, frameHeight - cropH))
        return ImageCrop(x: x0, y: y0, width: cropW, height: cropH)
    }

    func focalIsMatched(_ crop: ImageCrop) -> Bool {
        abs(achievedFocal(crop) - Contract.trainedFocal) < 1.0
    }

    func achievedFocal(_ crop: ImageCrop) -> Double {
        K[0][0] * crop.resize
    }

    // MARK: - Projection

    /// Source intrinsics carried through the crop and the letterbox resize onto
    /// the network canvas.
    //basically updates matrix K with the crop and the resize
    func adjustedIntrinsics(for crop: ImageCrop) -> simd_double3x3 {
        var adjusted = K
        // Crop is a principal-point shift.
        adjusted[2][0] -= Double(crop.x)
        adjusted[2][1] -= Double(crop.y)
        // Resize scales the first two rows, the letterbox offsets them.
        let r = crop.resize
        adjusted[0][0] *= r; adjusted[1][0] *= r; adjusted[2][0] *= r
        adjusted[0][1] *= r; adjusted[1][1] *= r; adjusted[2][1] *= r
        adjusted[2][0] += Double(crop.offsetX)
        adjusted[2][1] += Double(crop.offsetY)
        return adjusted
    }

    func projection(crop: ImageCrop) -> simd_double4x3 {
        var adjusted = adjustedIntrinsics(for: crop)
        // The LUT indexes the stride-4 feature map, not the input image.
        let s = 1.0 / Double(Contract.stride)
        adjusted[0][0] *= s; adjusted[1][0] *= s; adjusted[2][0] *= s
        adjusted[0][1] *= s; adjusted[1][1] *= s; adjusted[2][1] *= s

        let gridToSensor = sensorTVehicle * Contract.gridToVehicle
        let upper3x4 = simd_double4x3(columns: (
            SIMD3<Double>(gridToSensor[0][0], gridToSensor[0][1], gridToSensor[0][2]),
            SIMD3<Double>(gridToSensor[1][0], gridToSensor[1][1], gridToSensor[1][2]),
            SIMD3<Double>(gridToSensor[2][0], gridToSensor[2][1], gridToSensor[2][2]),
            SIMD3<Double>(gridToSensor[3][0], gridToSensor[3][1], gridToSensor[3][2])
        ))
        return adjusted * (Contract.sensorToImageAxes * upper3x4)
    }
}
