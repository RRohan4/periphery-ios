//  ProjectionLUT.swift
//  The geometry step between the two networks.
//
//  Ports periphery/training/model.py: _projection_lut and backproject. This is
//  the op that was deliberately kept out of the CoreML graph -- it is a gather,
//  not arithmetic, and it costs 0.3-0.5 ms on a desktop CPU. Both halves of the
//  model stay plain convolution stacks because of it.

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

    /// Build from a 3x4 grid-to-feature-pixel projection.
    /// Depends only on calibration; rebuild on pose update, not per frame.
    init(projection: simd_double4x3,
         points: [SIMD3<Double>] = Contract.voxelPoints(),
         featureWidth: Int = Contract.featureWidth,
         featureHeight: Int = Contract.featureHeight) {
        var indices = [Int32](repeating: 0, count: points.count)
        var visibility = [Float](repeating: 0, count: points.count)
        for i in points.indices {
            let p = points[i]
            let projected = projection * SIMD4<Double>(p.x, p.y, p.z, 1.0)
            let depth = projected.z
            let safeDepth = depth > 0 ? depth : 1.0
            let u = (projected.x / safeDepth).rounded()
            let v = (projected.y / safeDepth).rounded()
            let visible = depth > 0
                && u >= 0 && u < Double(featureWidth)
                && v >= 0 && v < Double(featureHeight)
            let cu = Int(min(max(u, 0), Double(featureWidth - 1)))
            let cv = Int(min(max(v, 0), Double(featureHeight - 1)))
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

    /// Gather image features into the BEV volume.
    ///
    /// `features` is [1, 64, 64, 128] contiguous, `volume` is [1, 64, 41, 80, 2]
    /// contiguous. Both flatten channel-major, and the voxel flatten order of
    /// Contract.voxelPoints() is exactly the volume's trailing (gx, gy, gz), so
    /// each channel is one contiguous run of gathers.
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
