//  MLMultiArrayTransfer.swift
//  Logical-order tensor copies that respect Core ML's reported strides.

import Accelerate
import CoreML
import Foundation

enum MLMultiArrayTransfer {

    static func readFloats(_ array: MLMultiArray,
                           into destination: inout [Float],
                           named what: String) throws {
        guard array.count == destination.count else {
            throw DetectorError.unexpectedShape(
                "\(what) has \(array.count) elements, expected \(destination.count)")
        }

        if isDenselyPacked(array) {
            try readDense(array, into: &destination, named: what)
            return
        }

        switch array.dataType {
        case .float32:
            let source = array.dataPointer.assumingMemoryBound(to: Float.self)
            forEachLogicalOffset(array) { logical, physical in
                destination[logical] = source[physical]
            }
        case .float16:
            let source = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            forEachLogicalOffset(array) { logical, physical in
                destination[logical] = Float(Float16(bitPattern: source[physical]))
            }
        default:
            throw DetectorError.unsupportedDataType("\(what) is \(typeName(array.dataType))")
        }
    }

    static func writeFloats(_ source: [Float],
                            to array: MLMultiArray,
                            named what: String) throws {
        guard array.count == source.count else {
            throw DetectorError.unexpectedShape(
                "\(what) wants \(array.count) elements, have \(source.count)")
        }

        if isDenselyPacked(array) {
            try writeDense(source, to: array, named: what)
            return
        }

        switch array.dataType {
        case .float32:
            let destination = array.dataPointer.assumingMemoryBound(to: Float.self)
            forEachLogicalOffset(array) { logical, physical in
                destination[physical] = source[logical]
            }
        case .float16:
            let destination = array.dataPointer.assumingMemoryBound(to: UInt16.self)
            forEachLogicalOffset(array) { logical, physical in
                destination[physical] = Float16(source[logical]).bitPattern
            }
        default:
            throw DetectorError.unsupportedDataType("\(what) is \(typeName(array.dataType))")
        }
    }

    static func denseStrides(for shape: [Int]) -> [Int] {
        guard !shape.isEmpty else { return [] }
        var result = [Int](repeating: 1, count: shape.count)
        if shape.count > 1 {
            for axis in stride(from: shape.count - 2, through: 0, by: -1) {
                result[axis] = result[axis + 1] * shape[axis + 1]
            }
        }
        return result
    }

    static func isDenselyPacked(_ array: MLMultiArray) -> Bool {
        array.strides.map(\.intValue) == denseStrides(for: array.shape.map(\.intValue))
    }

    private static func forEachLogicalOffset(
        _ array: MLMultiArray,
        _ body: (_ logical: Int, _ physical: Int) -> Void
    ) {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var indices = [Int](repeating: 0, count: shape.count)
        var physical = 0

        for logical in 0..<array.count {
            body(logical, physical)
            for axis in shape.indices.reversed() {
                indices[axis] += 1
                physical += strides[axis]
                if indices[axis] < shape[axis] { break }
                indices[axis] = 0
                physical -= shape[axis] * strides[axis]
            }
        }
    }

    private static func readDense(_ array: MLMultiArray,
                                  into destination: inout [Float],
                                  named what: String) throws {
        let count = array.count
        switch array.dataType {
        case .float32:
            destination.withUnsafeMutableBufferPointer { buffer in
                buffer.baseAddress!.update(
                    from: array.dataPointer.assumingMemoryBound(to: Float.self), count: count)
            }
        case .float16:
            var input = vImage_Buffer(data: array.dataPointer, height: 1,
                                      width: vImagePixelCount(count), rowBytes: count * 2)
            try destination.withUnsafeMutableBufferPointer { buffer in
                var output = vImage_Buffer(data: buffer.baseAddress!, height: 1,
                                           width: vImagePixelCount(count), rowBytes: count * 4)
                guard vImageConvert_Planar16FtoPlanarF(&input, &output, 0) == kvImageNoError else {
                    throw DetectorError.unsupportedDataType("\(what): float16 conversion failed")
                }
            }
        default:
            throw DetectorError.unsupportedDataType("\(what) is \(typeName(array.dataType))")
        }
    }

    private static func writeDense(_ source: [Float],
                                   to array: MLMultiArray,
                                   named what: String) throws {
        let count = array.count
        switch array.dataType {
        case .float32:
            source.withUnsafeBufferPointer { buffer in
                array.dataPointer.assumingMemoryBound(to: Float.self)
                    .update(from: buffer.baseAddress!, count: count)
            }
        case .float16:
            try source.withUnsafeBufferPointer { buffer in
                var input = vImage_Buffer(
                    data: UnsafeMutableRawPointer(mutating: buffer.baseAddress!), height: 1,
                    width: vImagePixelCount(count), rowBytes: count * 4)
                var output = vImage_Buffer(data: array.dataPointer, height: 1,
                                           width: vImagePixelCount(count), rowBytes: count * 2)
                guard vImageConvert_PlanarFtoPlanar16F(&input, &output, 0) == kvImageNoError else {
                    throw DetectorError.unsupportedDataType("\(what): float16 conversion failed")
                }
            }
        default:
            throw DetectorError.unsupportedDataType("\(what) is \(typeName(array.dataType))")
        }
    }

    private static func typeName(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .float16: return "float16"
        case .float32: return "float32"
        case .double: return "float64"
        case .int32: return "int32"
        default: return "raw \(type.rawValue)"
        }
    }
}
