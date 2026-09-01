//  Contract.swift
//  Geometry and label constants for safety40-locked-v1.
//
//  Every number here is derived from, and checked against, the Python that
//  trained the checkpoint:
//    periphery/training/contract.py   GridSpec, make_anchors, decode_boxes
//    periphery/perception/fastbev.py  N_VOXELS, VOXEL_SIZE, ORIGIN, GRID_TO_VEHICLE
//    configs/fastbev_cityscapes/D0_safety40_2h_512x256.json
//  Where Swift and Python disagree, Python wins and this file is the bug.

import Foundation
import simd

enum Contract {

    // MARK: - Network shapes

    static let inputWidth = 512
    static let inputHeight = 256
    static let stride = 4
    static let featureWidth = 128          // inputWidth / stride
    static let featureHeight = 64          // inputHeight / stride
    static let featureChannels = 64

    /// Focal length, in pixels, that the backbone was trained at for a
    /// 512-wide input. Feeding a different focal silently rescales range.
    static let trainedFocal: Double = 565.6

    // MARK: - Voxel grid (safety40)

    /// GridSpec.n_voxels for D0_safety40_2h_512x256: (gx, gy, gz).
    static let nx = 41
    static let ny = 80
    static let nz = 2
    static var voxelCount: Int { nx * ny * nz }

    /// Released virtual grid: 200x200x4 cells of 0.5 x 0.5 x 1.5 m about ORIGIN.
    /// The safety40 crop is (gx 80..<121, gy 99..<179, gz slices 1 and 2).
    private static let voxelSize = SIMD3<Double>(0.5, 0.5, 1.5)
    private static let fullOrigin = SIMD3<Double>(0.0, 0.0, -1.0)
    private static let fullCount = SIMD3<Double>(200, 200, 4)
    private static let cropX0 = 80
    private static let cropY0 = 99
    static let heightIndices = [1, 2]      // metric z -0.66 m and 0.84 m

    /// Voxel centres in grid metres, in the flatten order the volume expects:
    /// gx outermost, then gy, then the two height slices.
    /// Matches GridSpec.points_np() exactly, [6560, 3].
    static func voxelPoints() -> [SIMD3<Double>] {
        let base = fullOrigin - fullCount * voxelSize / 2.0
        var points = [SIMD3<Double>]()
        points.reserveCapacity(voxelCount)
        for ix in 0..<nx {
            let gx = Double(cropX0 + ix) * voxelSize.x + base.x
            for iy in 0..<ny {
                let gy = Double(cropY0 + iy) * voxelSize.y + base.y
                for k in heightIndices {
                    let gz = Double(k) * voxelSize.z + base.z
                    points.append(SIMD3<Double>(gx, gy, gz))
                }
            }
        }
        return points
    }

    // MARK: - Frames

    /// vehicle = GRID_TO_VEHICLE @ [gx, gy, gz, 1].
    /// forward = gy + 0.944, lateral = -gx (positive left), z = gz + 1.840.
    ///
    /// The 1.840 is the released virtual frame's z offset, not a claim about
    /// this camera's height: cityscapes_target_rows() subtracted the same
    /// constant when the labels were encoded, so decoding with it returns z in
    /// the label frame. Change it and the boxes move, the model does not.
    static let gridToVehicle = simd_double4x4(rows: [
        SIMD4<Double>(0.0, 1.0, 0.0, 0.944),
        SIMD4<Double>(-1.0, 0.0, 0.0, 0.0),
        SIMD4<Double>(0.0, 0.0, 1.0, 1.840),
        SIMD4<Double>(0.0, 0.0, 0.0, 1.0),
    ])

    /// ISO 8855 sensor axes (x forward, y left, z up) to image axes
    /// (x right, y down, z forward).
    static let sensorToImageAxes = simd_double3x3(rows: [
        SIMD3<Double>(0.0, -1.0, 0.0),
        SIMD3<Double>(0.0, 0.0, -1.0),
        SIMD3<Double>(1.0, 0.0, 0.0),
    ])

    // MARK: - Anchors

    /// BEV feature size the head predicts on: GridSpec.feature_size.
    static let fx = 21
    static let fy = 40
    static let anchorSizes: [SIMD3<Double>] = [
        SIMD3<Double>(0.866, 2.5981, 1.0),
        SIMD3<Double>(0.5774, 1.7321, 1.0),
        SIMD3<Double>(1.0, 1.0, 1.0),
        SIMD3<Double>(0.4, 0.4, 1.0),
    ]
    static let anchorRotations: [Double] = [0.0, 1.57]
    static var anchorsPerCell: Int { anchorSizes.count * anchorRotations.count }
    static var candidateCount: Int { fx * fy * anchorsPerCell }   // 6720

    /// GridSpec.anchor_range for safety40: (x0, y0, z, x1, y1, z).
    private static let anchorRange = (x0: -10.0, y0: -0.5, z: -1.8, x1: 10.0, y1: 39.0)

    /// One anchor row: [x, y, z_bottom, w, l, h, yaw] in grid metres.
    struct Anchor {
        var x, y, z, w, l, h, yaw: Double
        /// sqrt(w^2 + l^2), the DeltaXYZWLHRBBoxCoder normaliser.
        var diagonal: Double { (w * w + l * l).squareRoot() }
    }

    /// Cell-centre ordering of AlignedAnchor3DRangeGenerator, flattened the way
    /// FastBEVHead flattens its output maps: gx outer, gy, size, rotation.
    /// Index of a candidate is ((ix * fy + iy) * 4 + size) * 2 + rotation.
    static func anchors() -> [Anchor] {
        var rows = [Anchor]()
        rows.reserveCapacity(candidateCount)
        let spanX = anchorRange.x1 - anchorRange.x0
        let spanY = anchorRange.y1 - anchorRange.y0
        for ix in 0..<fx {
            let x = anchorRange.x0 + spanX * (Double(ix) + 0.5) / Double(fx)
            for iy in 0..<fy {
                let y = anchorRange.y0 + spanY * (Double(iy) + 0.5) / Double(fy)
                for size in anchorSizes {
                    for rotation in anchorRotations {
                        rows.append(Anchor(x: x, y: y, z: anchorRange.z,
                                           w: size.x, l: size.y, h: size.z,
                                           yaw: rotation))
                    }
                }
            }
        }
        return rows
    }

    // MARK: - Decode policy

    /// Half-period fold offset used by the direction classifier.
    static let dirOffset = 0.7854
    /// Operating point. Locked with the checkpoint and the NMS radius.
    static let scoreThreshold = 0.50
    /// Use 0.05 only to sweep the precision/recall curve.
    static let decodeScoreFloor = 0.05
    static let nmsRadius = 2.0

    /// INTERNAL_CLASSES. The pedestrian head is frozen and untrained; the
    /// measured operating point counts labels 0, 1, 2 only.
    static let classNames = ["light_vehicle", "large_vehicle", "two_wheeler", "pedestrian"]
    static let vehicleLabels: Set<Int> = [0, 1, 2]

    /// Evaluation region in vehicle metres. Anything outside is a decode
    /// artefact, not a detection.
    static let forwardRange = (min: 0.0, max: 40.0)
    static let lateralRange = (min: -10.0, max: 10.0)

    // MARK: - Preprocessing

    /// Per-channel RGB mean and standard deviation in 0-255 units, NOT 0-1.
    static let mean = SIMD3<Float>(123.675, 116.28, 103.53)
    static let std = SIMD3<Float>(58.395, 57.12, 57.375)
}
