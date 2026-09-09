// Calibration-dependent gather between the backbone and detection head. The LUT
// is rebuilt when pose or intrinsics change, not for every frame.

import Foundation
import simd

/// One entry per voxel: which feature column to read, and whether the voxel is
/// in front of the camera and inside the feature map at all.
struct ProjectionLUT {
    /// v * featureWidth + u, clamped into range. [voxelCount]
    let indices: [Int32]
    /// 1 for visible, 0 otherwise. Invisible voxels are zeroed, not skipped --
    /// the volume keeps its fixed shape.
    let visibility: [Float]

    /// Precompute the voxel → feature-map lookup. Runs when calibration changes,
    /// not each frame.
   
    init(projection: simd_double4x3,
         points: [SIMD3<Double>] = Contract.voxelPoints(),
         featureWidth: Int = Contract.featureWidth,
         featureHeight: Int = Contract.featureHeight) {
        var indices = [Int32](repeating: 0, count: points.count)
        var visibility = [Float](repeating: 0, count: points.count)
        for i in points.indices {
            let p = points[i]

            // Grid-frame voxel center → homogeneous feature-map coords (before ÷z).
            let projected = projection * SIMD4<Double>(p.x, p.y, p.z, 1.0)
            let depth = projected.z
            let safeDepth = depth > 0 ? depth : 1.0

            // Pinhole divide: (u, v) on the stride-4 map, not the 512×256 input.
            let u = (projected.x / safeDepth).rounded()
            let v = (projected.y / safeDepth).rounded()

            // Behind the camera or off the map → gather as zero, not skipped.
            let visible = depth > 0
                && u >= 0 && u < Double(featureWidth)
                && v >= 0 && v < Double(featureHeight)

            // Clamp so `indices[i]` is always in range even when invisible.
            let cu = Int(min(max(u, 0), Double(featureWidth - 1)))
            let cv = Int(min(max(v, 0), Double(featureHeight - 1)))

            // Flatten (u, v) to one index into each channel's 128×64 plane.
            indices[i] = Int32(cv * featureWidth + cu)
            visibility[i] = visible ? 1.0 : 0.0
        }
        self.indices = indices
        self.visibility = visibility
    }

    /// Fraction of voxels that land on the feature map. A sane windshield mount
    /// sits around 0.5-0.7; a number near zero means the pitch sign is flipped.
    var visibleFraction: Double {
        guard !visibility.isEmpty else { return 0 }
        return Double(visibility.reduce(0, +)) / Double(visibility.count)
    }

    /// Lift 2D backbone features into the 3D volume the head expects. Runs every
    /// frame;  only copies using the table built in `init`.
    func backproject(features: UnsafePointer<Float>,
                     volume: UnsafeMutablePointer<Float>,
                     channels: Int = Contract.featureChannels,
                     plane: Int = Contract.featureWidth * Contract.featureHeight) {
        let count = indices.count
        indices.withUnsafeBufferPointer { index in
            visibility.withUnsafeBufferPointer { mask in

                for c in 0..<channels {
                    let source = features + c * plane
                    let destination = volume + c * count
                    for i in 0..<count {
                        destination[i] = source[Int(index[i])] * mask[i]
                    }
                }
            }
        }
    }
}
