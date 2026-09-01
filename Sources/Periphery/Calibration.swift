//  Calibration.swift
//  Mount pose, focal matching, and the 3x4 grid-to-feature-pixel projection.
//
//  Ports periphery/sources/comma2k19.py (sensor_T_vehicle, focal_matched_crop)
//  and periphery/perception/fastbev.py (image_transform, cityscapes_projection).

import Foundation
import simd

/// A centre crop of the source frame plus the letterbox that maps it onto the
/// 512x256 network canvas.
struct ImageCrop {
    var x: Int
    var y: Int
    var width: Int
    var height: Int

    /// Uniform resize onto the network canvas, and where the resized image
    /// lands on it. Mirrors fastbev.image_transform: the whole crop is fitted,
    /// never re-cropped, and any remainder is padding.
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

    /// Mount pitch against the DIRECTION OF TRAVEL, positive nose-up, radians.
    /// Not against gravity: road grade cancels identically in the travel
    /// reference and does not in the gravity one, where a 2 deg mean grade
    /// walks straight into the pitch estimate.
    var pitch: Double
    /// Camera height above the road, metres. Forgiving: d(range)/d(height) is
    /// range/height, so 1 cm is a pure 0.8% scale factor.
    var height: Double
    /// Camera position forward of the vehicle origin, metres.
    var forwardOfOrigin: Double
    /// Intrinsics of the SOURCE frame, before any crop or resize.
    var K: simd_double3x3
    /// Size of the source frame, pixels.
    var frameWidth: Int
    var frameHeight: Int

    // MARK: - Mount pose

    /// 4x4 ISO 8855 vehicle -> sensor transform (x forward, y left, z up).
    ///
    /// A camera looking DOWN by angle d is rotated by +d about +y under the
    /// right-hand rule, so vehicle -> sensor is Ry(-d), and `pitch` is positive
    /// nose-up, i.e. d = -pitch.
    ///
    /// This sign is the single most dangerous line in the pipeline. Backwards,
    /// it applies twice the mount angle instead of cancelling it, the ground
    /// plane lands in the wrong image rows, and the failure looks like a bad
    /// detector rather than a bad transform. Ported literally, variable names
    /// and all, from comma2k19.sensor_T_vehicle.
    var sensorTVehicle: simd_double4x4 {
        let down = -pitch
        let c = cos(-down), s = sin(-down)
        let rotation = simd_double3x3(rows: [
            SIMD3<Double>(c, 0.0, s),
            SIMD3<Double>(0.0, 1.0, 0.0),
            SIMD3<Double>(-s, 0.0, c),
        ])
        let camera = SIMD3<Double>(forwardOfOrigin, 0.0, height)
        let translation = -(rotation * camera)
        return simd_double4x4(rows: [
            SIMD4<Double>(rotation[0][0], rotation[1][0], rotation[2][0], translation.x),
            SIMD4<Double>(rotation[0][1], rotation[1][1], rotation[2][1], translation.y),
            SIMD4<Double>(rotation[0][2], rotation[1][2], rotation[2][2], translation.z),
            SIMD4<Double>(0.0, 0.0, 0.0, 1.0),
        ])
    }

    /// Closed-form horizon row in the network image, for the self-check the
    /// Python detect script runs: the vanishing row of the ground plane.
    var horizonRow: Double {
        let crop = focalMatchedCrop()
        let adjusted = adjustedIntrinsics(for: crop)
        return adjusted[2][1] + adjusted[1][1] * tan(pitch)
    }

    // MARK: - Focal matching

    /// Centre crop chosen so that crop-then-resize lands on the trained focal.
    ///
    /// The lift is pure geometry and correct for any K, but the BACKBONE never
    /// sees K. It sees pixels, and apparent scale is f*W/d, so shrinking the
    /// focal by s is indistinguishable from moving every object 1/s farther.
    /// The crop folds into K as a principal-point shift; the discarded
    /// periphery is field of view we choose not to use.
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

    /// True when the source frame is too narrow to reach the trained focal, in
    /// which case every range is scaled by `achievedFocal / trainedFocal` and
    /// the mismatch has to be stated, not absorbed.
    func focalIsMatched(_ crop: ImageCrop) -> Bool {
        abs(achievedFocal(crop) - Contract.trainedFocal) < 1.0
    }

    func achievedFocal(_ crop: ImageCrop) -> Double {
        K[0][0] * crop.resize
    }

    // MARK: - Projection

    /// Source intrinsics carried through the crop and the letterbox resize onto
    /// the network canvas.
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

    /// 3x4 matrix mapping a homogeneous grid point to feature-map pixels,
    /// i.e. cityscapes_projection: K_feature @ SENSOR_TO_IMAGE_AXES @
    /// (sensor_T_vehicle @ GRID_TO_VEHICLE)[:3].
    ///
    /// Depends only on calibration. Rebuild it when the pose estimate moves,
    /// not per frame.
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
