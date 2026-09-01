//  Preprocess.swift
//  Camera frame -> [1, 3, 256, 512] float32 network input.
//
//  Contract section 1. Two things that are easy to get wrong and expensive to
//  debug:
//
//    * MEAN and STD are in 0-255 units. Divide by 255 first and every
//      activation is off by two orders of magnitude.
//    * The letterbox is filled with MEAN, so padding normalises to exactly
//      zero. Padding with black injects a strong negative signal at the edges.
//      Here the canvas is zeroed once and only the resized region is written,
//      which is the same thing without the subtraction.
//
//  The crop comes from Calibration.focalMatchedCrop(): apparent scale is what
//  the backbone learned, so the crop is not an optimisation, it is correctness.

import Accelerate
import CoreML
import CoreVideo
import simd

final class Preprocessor {

    /// Reusable [1, 3, 256, 512] float32 input.
    let input: MLMultiArray
    private var scaled: vImage_Buffer
    private var planes: [vImage_Buffer]
    private var scaledWidth = 0
    private var scaledHeight = 0

    init() throws {
        input = try MLMultiArray(shape: [1, 3,
                                         NSNumber(value: Contract.inputHeight),
                                         NSNumber(value: Contract.inputWidth)],
                                 dataType: .float32)
        scaled = vImage_Buffer()
        planes = []
    }

    deinit {
        free(scaled.data)
        planes.forEach { free($0.data) }
    }

    /// Fill `input` from a 32BGRA pixel buffer.
    ///
    /// Video stabilisation must be off on the capture session: EIS and OIS
    /// change per-frame geometry unreported, which breaks both the fixed
    /// intrinsics this crop is computed from and the known extrinsics the LUT
    /// is built from.
    func fill(from pixelBuffer: CVPixelBuffer, crop: ImageCrop) throws -> MLMultiArray {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
            throw DetectorError.unsupportedDataType("pixel buffer is not 32BGRA")
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw DetectorError.unsupportedDataType("pixel buffer has no base address")
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)

        try prepareScratch(width: crop.scaledWidth, height: crop.scaledHeight)

        var source = vImage_Buffer(
            data: base.advanced(by: crop.y * rowBytes + crop.x * 4),
            height: vImagePixelCount(crop.height),
            width: vImagePixelCount(crop.width),
            rowBytes: rowBytes)
        var destination = scaled
        guard vImageScale_ARGB8888(&source, &destination, nil,
                                   vImage_Flags(kvImageHighQualityResampling)) == kvImageNoError else {
            throw DetectorError.unexpectedShape("vImageScale failed")
        }

        // Memory order of 32BGRA is B, G, R, A.
        var blue = planes[0], green = planes[1], red = planes[2], alpha = planes[3]
        guard vImageConvert_ARGB8888toPlanar8(&destination, &blue, &green, &red, &alpha,
                                              vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
            throw DetectorError.unexpectedShape("vImageConvert_ARGB8888toPlanar8 failed")
        }

        // Zero the canvas so the letterbox is exactly the normalised mean, then
        // write the three channels into their sub-rects in RGB order.
        let plane = Contract.inputWidth * Contract.inputHeight
        try input.withUnsafeMutableBufferPointer(ofType: Float.self) { buffer, _ in
            guard let canvas = buffer.baseAddress else { return }
            canvas.update(repeating: 0, count: 3 * plane)
            let sources = [red, green, blue]
            for channel in 0..<3 {
                let mean = Contract.mean[channel], std = Contract.std[channel]
                var sourcePlane = sources[channel]
                let origin = canvas
                    .advanced(by: channel * plane)
                    .advanced(by: crop.offsetY * Contract.inputWidth + crop.offsetX)
                var target = vImage_Buffer(data: origin,
                                           height: vImagePixelCount(crop.scaledHeight),
                                           width: vImagePixelCount(crop.scaledWidth),
                                           rowBytes: Contract.inputWidth * MemoryLayout<Float>.size)
                // maxFloat/minFloat implement (pixel - MEAN) / STD directly:
                // p = 0 maps to -MEAN/STD, p = 255 maps to (255 - MEAN)/STD.
                let status = vImageConvert_Planar8toPlanarF(&sourcePlane, &target,
                                                            (255.0 - mean) / std,
                                                            -mean / std,
                                                            vImage_Flags(kvImageNoFlags))
                guard status == kvImageNoError else {
                    throw DetectorError.unexpectedShape(
                        "vImageConvert_Planar8toPlanarF failed: \(status)")
                }
            }
        }
        return input
    }

    private func prepareScratch(width: Int, height: Int) throws {
        guard width != scaledWidth || height != scaledHeight else { return }
        free(scaled.data)
        planes.forEach { free($0.data) }
        planes = []

        var buffer = vImage_Buffer()
        guard vImageBuffer_Init(&buffer, vImagePixelCount(height), vImagePixelCount(width),
                                32, vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
            throw DetectorError.unexpectedShape("could not allocate \(width)x\(height) scratch")
        }
        scaled = buffer
        for _ in 0..<4 {
            var planarBuffer = vImage_Buffer()
            guard vImageBuffer_Init(&planarBuffer, vImagePixelCount(height),
                                    vImagePixelCount(width), 8,
                                    vImage_Flags(kvImageNoFlags)) == kvImageNoError else {
                throw DetectorError.unexpectedShape("could not allocate planar scratch")
            }
            planes.append(planarBuffer)
        }
        scaledWidth = width
        scaledHeight = height
    }
}
