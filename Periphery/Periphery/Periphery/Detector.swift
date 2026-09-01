//  Detector.swift
//  backbone_static -> LUT gather -> head_static -> decode.
//
//  The two .mlpackages are plain convolution stacks; everything between and
//  after them lives here. Models are addressed by resource name rather than by
//  the generated Swift classes so that a re-export cannot silently rename the
//  call site, and head outputs are identified by their trailing dimension
//  (4 = classes, 9 = boxes, 2 = directions) rather than by the auto-generated
//  coremltools output names.

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

/// Per-frame timings, nanosecond resolution, for the measurement this port
/// exists to produce.
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
    private let headInputName: String
    private let anchors = Contract.anchors()

    /// Reused across frames: allocation is not part of the measurement, and a
    /// per-frame allocation of the volume would be 1.6 MB of churn at 30 Hz.
    private let volume: MLMultiArray
    private var lut: ProjectionLUT

    private(set) var lastTiming = InferenceTiming()

    init(calibration: Calibration,
         configuration: MLModelConfiguration = Detector.defaultConfiguration()) throws {
        backbone = try Detector.load("backbone_static", configuration)
        head = try Detector.load("head_static", configuration)
        backboneInputName = try Detector.soleInputName(backbone)
        headInputName = try Detector.soleInputName(head)
        volume = try MLMultiArray(shape: [1,
                                          NSNumber(value: Contract.featureChannels),
                                          NSNumber(value: Contract.nx),
                                          NSNumber(value: Contract.ny),
                                          NSNumber(value: Contract.nz)],
                                  dataType: .float32)
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

    /// Rebuild the projection LUT after a pose update. Cheap (6560 voxels), but
    /// it is not a per-frame cost -- the LUT depends only on calibration.
    func updateCalibration(_ calibration: Calibration) {
        let crop = calibration.focalMatchedCrop()
        lut = ProjectionLUT(projection: calibration.projection(crop: crop))
    }

    var visibleVoxelFraction: Double { lut.visibleFraction }

    /// One frame. `image` is [1, 3, 256, 512] float32, normalised per the
    /// contract: (pixel - MEAN) / STD on 0-255 values, RGB.
    func detect(image: MLMultiArray,
                scoreThreshold: Double = Contract.scoreThreshold) throws -> [Detection] {
        var timing = InferenceTiming()

        var mark = DispatchTime.now()
        let features = try run(backbone, input: image, name: backboneInputName).featureValue(
            for: try Detector.soleOutputName(backbone))!.multiArrayValue!
        timing.backbone = Detector.seconds(since: mark)

        mark = DispatchTime.now()
        try Detector.requireFloat32(features, "backbone output")
        try Detector.requireCount(features,
                                  Contract.featureChannels * Contract.featureWidth * Contract.featureHeight,
                                  "features")
        features.withUnsafeBufferPointer(ofType: Float.self) { source in
            volume.withUnsafeMutableBufferPointer(ofType: Float.self) { destination, _ in
                lut.backproject(features: source.baseAddress!,
                                volume: destination.baseAddress!)
            }
        }
        timing.gather = Detector.seconds(since: mark)

        mark = DispatchTime.now()
        let outputs = try run(head, input: volume, name: headInputName)
        let (classes, boxes, directions) = try Detector.headTensors(outputs)
        timing.head = Detector.seconds(since: mark)

        mark = DispatchTime.now()
        var detections = [Detection]()
        classes.withUnsafeBufferPointer(ofType: Float.self) { classPointer in
            boxes.withUnsafeBufferPointer(ofType: Float.self) { boxPointer in
                directions.withUnsafeBufferPointer(ofType: Float.self) { directionPointer in
                    detections = Decode.detections(classes: classPointer.baseAddress!,
                                                   boxes: boxPointer.baseAddress!,
                                                   directions: directionPointer.baseAddress!,
                                                   anchors: anchors,
                                                   scoreThreshold: scoreThreshold)
                }
            }
        }
        timing.decode = Detector.seconds(since: mark)

        lastTiming = timing
        return detections
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

    /// Sort the head's three outputs by trailing dimension.
    private static func headTensors(_ outputs: MLFeatureProvider) throws
        -> (classes: MLMultiArray, boxes: MLMultiArray, directions: MLMultiArray) {
        var byTrailingDimension = [Int: MLMultiArray]()
        for name in outputs.featureNames {
            guard let array = outputs.featureValue(for: name)?.multiArrayValue,
                  let trailing = array.shape.last?.intValue else { continue }
            try requireFloat32(array, "head output \(name)")
            byTrailingDimension[trailing] = array
        }
        guard let classes = byTrailingDimension[Contract.classNames.count],
              let boxes = byTrailingDimension[9],
              let directions = byTrailingDimension[2] else {
            throw DetectorError.unexpectedShape(
                "head outputs \(byTrailingDimension.keys.sorted()), expected 4, 9 and 2")
        }
        try requireCount(classes, Contract.candidateCount * Contract.classNames.count, "classes")
        try requireCount(boxes, Contract.candidateCount * 9, "boxes")
        try requireCount(directions, Contract.candidateCount * 2, "directions")
        return (classes, boxes, directions)
    }

    private static func requireFloat32(_ array: MLMultiArray, _ what: String) throws {
        guard array.dataType == .float32 else {
            throw DetectorError.unsupportedDataType("\(what) is \(array.dataType)")
        }
    }

    private static func requireCount(_ array: MLMultiArray, _ expected: Int, _ what: String) throws {
        guard array.count == expected else {
            throw DetectorError.unexpectedShape("\(what) has \(array.count) elements, expected \(expected)")
        }
    }

    private static func seconds(since mark: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - mark.uptimeNanoseconds) / 1e9
    }
}
