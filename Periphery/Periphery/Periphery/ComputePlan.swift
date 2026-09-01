//  ComputePlan.swift
//  Which compute unit does each operation actually land on?
//
//  The op inventory (conv, relu, add, upsample, concat, max_pool for the
//  backbone; conv, relu, add, reshape, transpose, stack for the head) says only
//  that nothing is inherently unsupported. It does NOT say where Core ML puts
//  them. Placement is chosen at load time, per operation, and a graph can be
//  split into segments that hand off between the ANE and the CPU -- where the
//  handoffs cost more than the ops.
//
//  MLComputePlan (iOS 17+) reports that decision without running anything. It
//  is the substitute for Instruments' Core ML template, which needs a
//  USB-tethered device and is therefore unavailable when the Mac is rented.
//
//  Two caveats worth stating whenever a number from here is quoted:
//    * this is the PLANNED placement, not a measurement of executed work;
//    * in the simulator there is no Neural Engine, so everything reports CPU or
//      GPU and the answer is meaningless. Run it on the phone.

import CoreML
import Foundation

enum ComputePlanError: Error, CustomStringConvertible {
    case modelMissing(String)
    case notAProgram(String)

    var description: String {
        switch self {
        case .modelMissing(let name): return "\(name).mlmodelc not in bundle"
        case .notAProgram(let name): return "\(name) is not an ML Program"
        }
    }
}

/// One operation Core ML does not intend to put on the Neural Engine.
struct StrayOperation: Identifiable {
    let id = UUID()
    let op: String
    let device: String
    /// Relative estimated cost, 0...1 across the whole model.
    let cost: Double
}

struct ComputePlanSummary: Identifiable {
    var id: String { model }
    let model: String
    let total: Int
    let neuralEngine: Int
    let gpu: Int
    let cpu: Int
    /// Estimated cost that lands on the Neural Engine, 0...1. More honest than
    /// counting operations: one convolution outweighs twenty reshapes.
    let neuralEngineCost: Double
    let strays: [StrayOperation]

    var residency: Double { total == 0 ? 0 : Double(neuralEngine) / Double(total) }

    var headline: String {
        String(format: "%d/%d ops on ANE (%.0f%% of estimated cost)",
               neuralEngine, total, neuralEngineCost * 100)
    }
}

@available(iOS 17.0, *)
enum ComputePlanProbe {

    /// Both halves of the model, in pipeline order.
    static func summarizeAll(computeUnits: MLComputeUnits = .all) async -> [Result<ComputePlanSummary, Error>] {
        var results = [Result<ComputePlanSummary, Error>]()
        for name in ["backbone_static", "head_static"] {
            do {
                results.append(.success(try await summarize(name, computeUnits: computeUnits)))
            } catch {
                results.append(.failure(error))
            }
        }
        return results
    }

    static func summarize(_ name: String,
                          computeUnits: MLComputeUnits = .all) async throws -> ComputePlanSummary {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw ComputePlanError.modelMissing(name)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits

        let plan = try await MLComputePlan.load(contentsOf: url, configuration: configuration)
        guard case let .program(program) = plan.modelStructure,
              let main = program.functions["main"] else {
            throw ComputePlanError.notAProgram(name)
        }

        var total = 0, ane = 0, gpu = 0, cpu = 0
        var aneCost = 0.0
        var strays = [StrayOperation]()

        for operation in main.block.operations {
            let opName = operation.operatorName
            // Weights and literals are not work; counting them would inflate
            // the CPU column with things that never execute.
            if opName == "const" { continue }
            total += 1
            let cost = plan.estimatedCost(of: operation)?.weight ?? 0
            let device = plan.deviceUsage(for: operation)?.preferred
            switch device {
            case .some(.neuralEngine):
                ane += 1
                aneCost += cost
            case .some(.gpu):
                gpu += 1
                strays.append(StrayOperation(op: opName, device: "GPU", cost: cost))
            case .some(.cpu):
                cpu += 1
                strays.append(StrayOperation(op: opName, device: "CPU", cost: cost))
            default:
                cpu += 1
                strays.append(StrayOperation(op: opName, device: "unknown", cost: cost))
            }
        }

        // Costliest first: a single conv off the ANE matters, six reshapes
        // usually do not.
        strays.sort { $0.cost > $1.cost }
        return ComputePlanSummary(model: name, total: total, neuralEngine: ane,
                                  gpu: gpu, cpu: cpu, neuralEngineCost: aneCost,
                                  strays: strays)
    }
}
