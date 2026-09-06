
import Accelerate
import CoreML
import Foundation

enum DetectorError: Error, CustomStringConvertible {
    case modelMissing(String)
    case unexpectedShape(String)
    case unsupportedDataType(String)

    var description: String {
        switch self {
        case .modelMissing(let name): return "model \(name).mlmodelc not in bundle"
        case .unexpectedShape(let detail): return "unexpected tensor shape: \(detail)"
        case .unsupportedDataType(let detail): return "unsupported MLMultiArray dtype: \(detail)"
        }
    }
}

/// Per-frame timings, for the measurement this port exists to produce.
struct InferenceTiming {
    var backbone: Double = 0
    var gather: Double = 0
    var head: Double = 0
    var decode: Double = 0
    var total: Double { backbone + gather + head + decode }
}

final class Detector {

    private let backbone: MLModel
    private let head: MLModel
    private let backboneInputName: String
    private let backboneOutputName: String
    private let headInputName: String
    private let anchors = Contract.anchors()

    /// Reused across frames. Allocation is not part of the measurement, and
    /// per-frame churn of the volume alone would be 1.6 MB at 30 Hz.
    private let volume: MLMultiArray
    private let imageBuffer: MLMultiArray?      // non-nil when the model wants float16
    private var imageScratch: [Float]
    private var featureScratch: [Float]
    private var volumeScratch: [Float]
    private var classScratch: [Float]
    private var boxScratch: [Float]
    private var directionScratch: [Float]
    private var lut: ProjectionLUT

    private(set) var lastTiming = InferenceTiming()
    /// What Core ML actually handed back, for the record.
    private(set) var precisionNote = ""

    init(calibration: Calibration,
         configuration: MLModelConfiguration = Detector.defaultConfiguration()) throws {
        backbone = try Detector.load("backbone_static", configuration)
        head = try Detector.load("head_static", configuration)
        backboneInputName = try Detector.soleInputName(backbone)
        backboneOutputName = try Detector.soleOutputName(backbone)
        headInputName = try Detector.soleInputName(head)

        let volumeType = Detector.inputDataType(head, headInputName)
        volume = try MLMultiArray(shape: [1,
                                          NSNumber(value: Contract.featureChannels),
                                          NSNumber(value: Contract.nx),
                                          NSNumber(value: Contract.ny),
                                          NSNumber(value: Contract.nz)],
                                  dataType: volumeType)
        let imageType = Detector.inputDataType(backbone, backboneInputName)
        imageBuffer = imageType == .float32 ? nil
            : try MLMultiArray(shape: [1, 3,
                                       NSNumber(value: Contract.inputHeight),
                                       NSNumber(value: Contract.inputWidth)],
                               dataType: imageType)
        precisionNote = "image \(Detector.name(imageType)), volume \(Detector.name(volumeType))"

        imageScratch = [Float](repeating: 0,
                               count: imageBuffer == nil ? 0
                                    : 3 * Contract.inputWidth * Contract.inputHeight)
        featureScratch = [Float](repeating: 0,
                                 count: Contract.featureChannels
                                      * Contract.featureWidth * Contract.featureHeight)
        volumeScratch = [Float](repeating: 0,
                                count: Contract.featureChannels * Contract.voxelCount)
        classScratch = [Float](repeating: 0,
                               count: Contract.candidateCount * Contract.classNames.count)
        boxScratch = [Float](repeating: 0, count: Contract.candidateCount * 9)
        directionScratch = [Float](repeating: 0, count: Contract.candidateCount * 2)

        let crop = calibration.focalMatchedCrop()
        lut = ProjectionLUT(projection: calibration.projection(crop: crop))
    }

    /// Default to every compute unit. Pin to `.cpuAndNeuralEngine` only to
    /// answer a question about ANE residency, never as the shipping default.
    static func defaultConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        return configuration
    }

    /// Rebuild the projection LUT after a pose update. Cheap (6560 voxels), and
    /// deliberately not a per-frame cost: the LUT depends only on calibration.
    func updateCalibration(_ calibration: Calibration) {
        let crop = calibration.focalMatchedCrop()
        lut = ProjectionLUT(projection: calibration.projection(crop: crop))
    }

    var visibleVoxelFraction: Double { lut.visibleFraction }

    /// One frame. `image` is [1, 3, 256, 512] float32, normalised per the
    /// contract: (pixel - MEAN) / STD on 0-255 values, RGB.
    func detect(image: MLMultiArray,
                scoreThreshold: Double = Contract.scoreThreshold,
                rejectImplausible: Bool = false) throws -> [Detection] {
        var timing = InferenceTiming()

        // The image arrives float32 from Preprocessor; convert only if the
        // model asked for something else.
        let modelImage: MLMultiArray
        if let imageBuffer {
            try Detector.readFloats(image, into: &imageScratch, "image")
            try Detector.writeFloats(imageScratch, to: imageBuffer, "image")
            modelImage = imageBuffer
        } else {
            modelImage = image
        }

        var mark = DispatchTime.now()
        let outputs = try run(backbone, input: modelImage, name: backboneInputName)
        guard let features = outputs.featureValue(for: backboneOutputName)?.multiArrayValue else {
            throw DetectorError.unexpectedShape("backbone produced no array")
        }
        timing.backbone = Detector.seconds(since: mark)

        mark = DispatchTime.now()
        try Detector.readFloats(features, into: &featureScratch, "features")
        try gather()
        timing.gather = Detector.seconds(since: mark)

        mark = DispatchTime.now()
        let headOutputs = try run(head, input: volume, name: headInputName)
        let (classes, boxes, directions) = try Detector.headTensors(headOutputs)
        try Detector.readFloats(classes, into: &classScratch, "classes")
        try Detector.readFloats(boxes, into: &boxScratch, "boxes")
        try Detector.readFloats(directions, into: &directionScratch, "directions")
        Detector.dumpTensors([Detector.describe(features, "backbone.features"),
                              Detector.describe(volume, "head.input.volume"),
                              Detector.describe(classes, "head.classes"),
                              Detector.describe(boxes, "head.boxes"),
                              Detector.describe(directions, "head.directions")])
        timing.head = Detector.seconds(since: mark)

        mark = DispatchTime.now()
        var detections = [Detection]()
        classScratch.withUnsafeBufferPointer { classPointer in
            boxScratch.withUnsafeBufferPointer { boxPointer in
                directionScratch.withUnsafeBufferPointer { directionPointer in
                    detections = Decode.detections(classes: classPointer.baseAddress!,
                                                   boxes: boxPointer.baseAddress!,
                                                   directions: directionPointer.baseAddress!,
                                                   anchors: anchors,
                                                   scoreThreshold: scoreThreshold,
                                                   rejectImplausible: rejectImplausible)
                }
            }
        }
        timing.decode = Detector.seconds(since: mark)

        lastTiming = timing
        return detections
    }

    // MARK: - The gather

    /// Feature columns into the BEV volume, then into whatever dtype the head
    /// wants. The gather itself always runs in float32.
    private func gather() throws {
        featureScratch.withUnsafeBufferPointer { source in
            volumeScratch.withUnsafeMutableBufferPointer { destination in
                lut.backproject(features: source.baseAddress!,
                                volume: destination.baseAddress!)
            }
        }
        try Detector.writeFloats(volumeScratch, to: volume, "volume")
    }

    // MARK: - Tensor conversion

    /// Copy an MLMultiArray into a float32 buffer, converting from float16 when
    /// that is what the Neural Engine returned.
    private static func readFloats(_ array: MLMultiArray,
                                   into destination: inout [Float],
                                   _ what: String) throws {
        guard array.count == destination.count else {
            throw DetectorError.unexpectedShape(
                "\(what) has \(array.count) elements, expected \(destination.count)")
        }
        try MLMultiArrayTransfer.readFloats(array, into: &destination, named: what)
    }

    // MARK: - Raw tensor dump

    static func describe(_ array: MLMultiArray, _ name: String,
                         sample: Int = 24) -> [String: Any] {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        let dense = MLMultiArrayTransfer.denseStrides(for: shape)
        let isDense = strides == dense

        func value(atOffset offset: Int) -> Float {
            switch array.dataType {
            case .float32:
                return array.dataPointer.assumingMemoryBound(to: Float.self)[offset]
            case .float16:
                let bits = array.dataPointer.assumingMemoryBound(to: UInt16.self)[offset]
                return Float(Float16(bitPattern: bits))
            default:
                return .nan
            }
        }
        // Logical index -> offset through the reported strides.
        func strided(_ logical: Int) -> Float {
            var remainder = logical, offset = 0
            for axis in 0..<shape.count {
                let index = (remainder / dense[axis]) % shape[axis]
                remainder -= index * dense[axis]
                offset += index * strides[axis]
            }
            return value(atOffset: offset)
        }

        let n = min(sample, array.count)
        let linear = (0..<n).map { value(atOffset: $0) }
        let viaStrides = (0..<n).map { strided($0) }
        var linearZeros = 0
        var logicalZeros = 0
        for i in 0..<array.count {
            if value(atOffset: i) == 0 { linearZeros += 1 }
            if strided(i) == 0 { logicalZeros += 1 }
        }

        return [
            "name": name,
            "shape": shape,
            "strides": strides,
            "denseStrides": dense,
            "isDenselyPacked": isDense,
            "count": array.count,
            "dataType": array.dataType == .float16 ? "float16"
                      : array.dataType == .float32 ? "float32" : "other",
            "zerosReadLinearly": linearZeros,
            "fractionZeroLinearly": Double(linearZeros) / Double(max(array.count, 1)),
            "zerosReadLogically": logicalZeros,
            "fractionZeroLogically": Double(logicalZeros) / Double(max(array.count, 1)),
            "firstLinear": linear.map { Double($0) },
            "firstViaStrides": viaStrides.map { Double($0) },
            "readingsAgree": linear == viaStrides,
        ]
    }

    /// Written once per launch to Documents, next to the drives.
    private static var dumpWritten = false

    static func dumpTensors(_ entries: @autoclosure () -> [[String: Any]]) {
        guard !dumpWritten else { return }
        dumpWritten = true
        let payload: [String: Any] = [
            "note": "Tensor transfers use reported strides. firstLinear is diagnostic only; "
                  + "firstViaStrides is the logical order passed to decode. sigmoid(0)=0.5.",
            "tensors": entries(),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload,
                                                     options: [.prettyPrinted, .sortedKeys]),
              let dir = FileManager.default.urls(for: .documentDirectory,
                                                 in: .userDomainMask).first
        else { return }
        try? data.write(to: dir.appendingPathComponent("head_dump.json"))
    }

    /// The reverse: a float32 buffer into an MLMultiArray of either precision.
    private static func writeFloats(_ source: [Float],
                                    to array: MLMultiArray,
                                    _ what: String) throws {
        guard array.count == source.count else {
            throw DetectorError.unexpectedShape(
                "\(what) wants \(array.count) elements, have \(source.count)")
        }
        try MLMultiArrayTransfer.writeFloats(source, to: array, named: what)
    }

    // MARK: - CoreML plumbing

    private func run(_ model: MLModel, input: MLMultiArray, name: String) throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(
            dictionary: [name: MLFeatureValue(multiArray: input)])
        return try model.prediction(from: provider)
    }

    private static func load(_ name: String, _ configuration: MLModelConfiguration) throws -> MLModel {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw DetectorError.modelMissing(name)
        }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    private static func soleInputName(_ model: MLModel) throws -> String {
        let names = Array(model.modelDescription.inputDescriptionsByName.keys)
        guard names.count == 1, let name = names.first else {
            throw DetectorError.unexpectedShape("expected one input, found \(names)")
        }
        return name
    }

    private static func soleOutputName(_ model: MLModel) throws -> String {
        let names = Array(model.modelDescription.outputDescriptionsByName.keys)
        guard names.count == 1, let name = names.first else {
            throw DetectorError.unexpectedShape("expected one output, found \(names)")
        }
        return name
    }

    private static func inputDataType(_ model: MLModel, _ name: String) -> MLMultiArrayDataType {
        model.modelDescription.inputDescriptionsByName[name]?
            .multiArrayConstraint?.dataType ?? .float32
    }

    private static func name(_ type: MLMultiArrayDataType) -> String {
        switch type {
        case .float16: return "float16"
        case .float32: return "float32"
        case .double: return "float64"
        case .int32: return "int32"
        default: return "raw \(type.rawValue)"
        }
    }

    /// Sort the head's three outputs by trailing dimension.
    private static func headTensors(_ outputs: MLFeatureProvider) throws
        -> (classes: MLMultiArray, boxes: MLMultiArray, directions: MLMultiArray) {
        var byTrailingDimension = [Int: MLMultiArray]()
        for name in outputs.featureNames {
            guard let array = outputs.featureValue(for: name)?.multiArrayValue,
                  let trailing = array.shape.last?.intValue else { continue }
            byTrailingDimension[trailing] = array
        }
        guard let classes = byTrailingDimension[Contract.classNames.count],
              let boxes = byTrailingDimension[9],
              let directions = byTrailingDimension[2] else {
            throw DetectorError.unexpectedShape(
                "head outputs \(byTrailingDimension.keys.sorted()), expected 4, 9 and 2")
        }
        return (classes, boxes, directions)
    }

    private static func seconds(since mark: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - mark.uptimeNanoseconds) / 1e9
    }
}
