import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class DeformGraphSystem {
    let device: MTLDevice
    let mpsDevice: MPSGraphDevice
    let graph: MPSGraph
    var executable: MPSGraphExecutable?
    public let useFP16: Bool
    
    // Tensors
    var inputXYZTensor: MPSGraphTensor?
    var inputTTensor: MPSGraphTensor?
    var outXYZTensor: MPSGraphTensor?
    var outRotTensor: MPSGraphTensor?
    var outScaleTensor: MPSGraphTensor?
    
    var weightDataMap: [String: (Data, [NSNumber], MPSDataType)] = [:]
    
    // Deformation Network Params
    let W = 256
    let D = 8
    let SKIP_LAYER = 4
    let MULTIRES = 10
    let T_MULTIRES = 10
    
    // Larger batch size = fewer kernel launches = less overhead
    // 131072 should work on most devices; reduce if GPU crashes
    let SAFE_BATCH_SIZE = 16384
    var flatten = true
    public private(set) var lastMaskedCount: Int = 0
    public init(device: MTLDevice, useFP16: Bool) {
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.graph = MPSGraph()
        #if arch(x86_64)
        self.useFP16 = false
        #else
        self.useFP16 = useFP16
        #endif
    }
    
    private static func float32DataToFloat16Data(_ data: Data) -> Data {
#if arch(x86_64)
        // Float16 isn't available on x86_64 (we alias it to Float elsewhere for buildability),
        // and FP16 deformation is disabled on that architecture anyway.
        return Data()
#else
        let floatCount = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { raw in
            let floats = raw.bindMemory(to: Float.self)
            var halfBits = [UInt16](repeating: 0, count: floatCount)
            for i in 0..<floatCount {
                halfBits[i] = Float16(floats[i]).bitPattern
            }
            return Data(bytes: halfBits, count: halfBits.count * MemoryLayout<UInt16>.size)
        }
#endif
    }
    
    public func loadWeights(flatData: Data) {
        var offset = 0
        let floatSize = MemoryLayout<Float>.size
        let totalCh = 63 + 21
        
        func extract(outDim: Int, inDim: Int, name: String) {
            let byteCount = outDim * inDim * floatSize
            guard (offset + byteCount) <= flatData.count else { return }
            
            let subDataF32 = flatData.subdata(in: offset..<(offset + byteCount))
            if useFP16 {
                let subDataF16 = Self.float32DataToFloat16Data(subDataF32)
                weightDataMap["\(name)_w"] = (subDataF16, [NSNumber(value: outDim), NSNumber(value: inDim)], .float16)
            } else {
                weightDataMap["\(name)_w"] = (subDataF32, [NSNumber(value: outDim), NSNumber(value: inDim)], .float32)
            }
            offset += byteCount
            
            let biasBytes = outDim * floatSize
            let biasDataF32 = flatData.subdata(in: offset..<(offset + biasBytes))
            if useFP16 {
                let biasDataF16 = Self.float32DataToFloat16Data(biasDataF32)
                weightDataMap["\(name)_b"] = (biasDataF16, [NSNumber(value: 1), NSNumber(value: outDim)], .float16)
            } else {
                weightDataMap["\(name)_b"] = (biasDataF32, [NSNumber(value: 1), NSNumber(value: outDim)], .float32)
            }
            offset += biasBytes
        }
        
        extract(outDim: W, inDim: totalCh, name: "L0")
        for i in 0..<(D-1) {
            let inDim = (i == SKIP_LAYER) ? (W + totalCh) : W
            extract(outDim: W, inDim: inDim, name: "L\(i+1)")
        }
        extract(outDim: 3, inDim: W, name: "Head_XYZ")
        extract(outDim: 4, inDim: W, name: "Head_Rot")
        extract(outDim: 3, inDim: W, name: "Head_Scale")
    }
    
    public func buildAndCompile() {
        let batchDim = NSNumber(value: -1) // Dynamic size
        let xyz: MPSGraphTensor
        let t: MPSGraphTensor
        
        // For some reason, I have to flatten the tensors otherwise the output doesn't match with the output generated from PyTorch :(
        if (self.flatten) {
            self.inputXYZTensor = graph.placeholder(shape: [batchDim], dataType: .float32, name: "in_xyz_flat")
            self.inputTTensor   = graph.placeholder(shape: [batchDim], dataType: .float32, name: "in_t_flat")
            guard let xyzFlat = self.inputXYZTensor, let tFlat = self.inputTTensor else { return }
            xyz = graph.reshape(xyzFlat, shape: [NSNumber(value: -1), NSNumber(value: 3)], name: "reshape_xyz")
            t   = graph.reshape(tFlat,   shape: [NSNumber(value: -1), NSNumber(value: 1)], name: "reshape_t")
        }
        else {
            self.inputXYZTensor = graph.placeholder(shape: [batchDim, 3], dataType: .float32, name: "in_xyz")
            self.inputTTensor   = graph.placeholder(shape: [batchDim, 1], dataType: .float32, name: "in_t")
            guard let inputXYZ = self.inputXYZTensor, let inputT = self.inputTTensor else { return }
            xyz = inputXYZ
            t   = inputT
        }
        
        let xyzTyped: MPSGraphTensor
        let tTyped: MPSGraphTensor
        if useFP16 {
            xyzTyped = graph.cast(xyz, to: .float16, name: "cast_xyz_f16")
            tTyped = graph.cast(t, to: .float16, name: "cast_t_f16")
        } else {
            xyzTyped = xyz
            tTyped = t
        }

        let embXYZ = positionalEncoding(input: xyzTyped, numFreqs: MULTIRES, dataType: useFP16 ? .float16 : .float32)
        let embT   = positionalEncoding(input: tTyped, numFreqs: T_MULTIRES, dataType: useFP16 ? .float16 : .float32)
        let inputs = graph.concatTensors([embXYZ, embT], dimension: 1, name: "input_concat")
        
        var h = inputs
        h = denseLayer(input: h, name: "L0", activation: true)
        
        for i in 0..<(D-1) {
            if i == SKIP_LAYER { h = graph.concatTensors([inputs, h], dimension: 1, name: "skip") }
            h = denseLayer(input: h, name: "L\(i+1)", activation: true)
        }
        
        // Output Layers
        let outXYZRaw = denseLayer(input: h, name: "Head_XYZ", activation: false)
        let outRotRaw = denseLayer(input: h, name: "Head_Rot", activation: false)
        let outScaleRaw = denseLayer(input: h, name: "Head_Scale", activation: false)
        
        // We always write float32 into the output MTLBuffers.
        let outXYZ = useFP16 ? graph.cast(outXYZRaw, to: .float32, name: "out_xyz_f32") : outXYZRaw
        let outRot = useFP16 ? graph.cast(outRotRaw, to: .float32, name: "out_rot_f32") : outRotRaw
        let outScale = useFP16 ? graph.cast(outScaleRaw, to: .float32, name: "out_scale_f32") : outScaleRaw
        
        let feedsDict: [MPSGraphTensor : MPSGraphShapedType]
        if (self.flatten) {
            self.outXYZTensor = graph.reshape(outXYZ, shape: [batchDim], name: "out_xyz_flat")
            self.outRotTensor = graph.reshape(outRot, shape: [batchDim], name: "out_rot_flat")
            self.outScaleTensor = graph.reshape(outScale, shape: [batchDim], name: "out_scale_flat")
            feedsDict = [
                inputXYZTensor!: MPSGraphShapedType(shape: [batchDim], dataType: .float32),
                inputTTensor!:   MPSGraphShapedType(shape: [batchDim], dataType: .float32)
            ]
        }
        else {
            self.outXYZTensor   = outXYZ
            self.outRotTensor   = outRot
            self.outScaleTensor = outScale
            feedsDict = [
                inputXYZTensor!: MPSGraphShapedType(shape: [batchDim, 3], dataType: .float32),
                inputTTensor!:   MPSGraphShapedType(shape: [batchDim, 1], dataType: .float32)
            ]
        }
        
        self.executable = graph.compile(with: mpsDevice,
                                        feeds: feedsDict,
                                        targetTensors: [outXYZTensor!, outRotTensor!, outScaleTensor!],
                                        targetOperations: nil,
                                        compilationDescriptor: nil)
    }
    
    // Helper function to slice the buffer with byte offset
    private func createTensorView(buffer: MTLBuffer,
                                  offset: Int,
                                  shape: [NSNumber]) -> MPSGraphTensorData {
        
        let desc = MPSNDArrayDescriptor(dataType: .float32, shape: shape)
        let ndArray = MPSNDArray(buffer: buffer, offset: offset, descriptor: desc)
        return MPSGraphTensorData(ndArray)
    }
    
    /// Returns elapsed time in milliseconds
    @discardableResult
    public func run(commandQueue: MTLCommandQueue,
                    xyzBuffer: MTLBuffer,
                    tBuffer: MTLBuffer,
                    outXYZ: MTLBuffer,
                    outRot: MTLBuffer,
                    outScale: MTLBuffer,
                    count: Int) -> Double {
        
        guard let exec = executable else { return 0 }
        let totalStart = CFAbsoluteTimeGetCurrent()
        let floatSize = MemoryLayout<Float>.size
        let numBatches = (count + SAFE_BATCH_SIZE - 1) / SAFE_BATCH_SIZE

        let cb = MPSCommandBuffer(from: commandQueue)
        cb.label = "DeformGraphSystem.run"

        for i in stride(from: 0, to: count, by: SAFE_BATCH_SIZE) {
            autoreleasepool {
                let batchStart = CFAbsoluteTimeGetCurrent()
                let currentCount = min(SAFE_BATCH_SIZE, count - i)
                
                // Define the offset
                let offsetXYZ = i * 3 * floatSize
                let offsetT   = i * 1 * floatSize
                let offsetOutXYZ = i * 3 * floatSize
                let offsetOutRot = i * 4 * floatSize
                let offsetOutScale = i * 3 * floatSize
                
                let xyzShape: [NSNumber]
                let tShape: [NSNumber]
                let outRotShape: [NSNumber]
                let outScaleShape: [NSNumber]
                
                // Get the size of the batched buffer
                if (self.flatten) {
                    xyzShape = [NSNumber(value: currentCount * 3)]
                    tShape = [NSNumber(value: currentCount * 1)]
                    outRotShape = [NSNumber(value: currentCount * 4)]
                    outScaleShape = [NSNumber(value: currentCount * 3)]
                }
                else {
                    xyzShape   = [NSNumber(value: currentCount), 3]
                    tShape     = [NSNumber(value: currentCount), 1]
                    outRotShape   = [NSNumber(value: currentCount), 4]
                    outScaleShape = [NSNumber(value: currentCount), 3]
                }
                
                // Get the batched buffer
                let xyzData = createTensorView(buffer: xyzBuffer, offset: offsetXYZ, shape: xyzShape)
                let tData = createTensorView(buffer: tBuffer, offset: offsetT, shape: tShape)
                
                let outXYZData = createTensorView(buffer: outXYZ, offset: offsetOutXYZ, shape: xyzShape)
                let outRotData = createTensorView(buffer: outRot, offset: offsetOutRot, shape: outRotShape)
                let outScaleData = createTensorView(buffer: outScale, offset: offsetOutScale, shape: outScaleShape)
                
                var inputsArray: [MPSGraphTensorData] = []
                var resultsArray: [MPSGraphTensorData] = []
                
                if let feedTensors = exec.feedTensors {
                    for tensor in feedTensors {
                        let opName = tensor.operation.name
                        if opName == "in_xyz" { inputsArray.append(xyzData) }
                        else if opName == "in_t" { inputsArray.append(tData) }
                        // Fallback: Check standard names just in case
                        else if opName == "in_xyz_flat" { inputsArray.append(xyzData) }
                        else if opName == "in_t_flat" { inputsArray.append(tData) }
                        // Fallback: Check shape size
                        else if (tensor.shape?[1].intValue ?? 0) == 3 { inputsArray.append(xyzData) }
                        else { inputsArray.append(tData) }
                    }
                }

                resultsArray.append(outXYZData)
                resultsArray.append(outRotData)
                resultsArray.append(outScaleData)
                
                if inputsArray.count == (exec.feedTensors?.count ?? 0) {
                    exec.encode(to: cb,
                                inputs: inputsArray,
                                results: resultsArray,
                                executionDescriptor: nil)
                }
                
                let batchElapsedMs = (CFAbsoluteTimeGetCurrent() - batchStart) * 1000.0
                if numBatches > 1 {
                    print("DeformGraph batch \(i / SAFE_BATCH_SIZE) encoded: \(batchElapsedMs) ms")
                }
            }
        }

        cb.commit()
        cb.waitUntilCompleted()

        let totalElapsedMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0
        print("DeformGraph total (\(numBatches) batches): \(totalElapsedMs) ms")
        return totalElapsedMs
    }

    /// Run deformation network only on a subset of indices (for masked deformation)
    /// - Parameters:
    ///   - commandQueue: Metal command queue
    ///   - xyzBuffer: Input XYZ positions (full buffer)
    ///   - tBuffer: Input time buffer (full buffer)
    ///   - outXYZ: Output position deltas (full buffer)
    ///   - outRot: Output rotation deltas (full buffer)
    ///   - outScale: Output scale deltas (full buffer)
    ///   - maskBuffer: Mask buffer (1.0 = deform, 0.0 = skip)
    ///   - count: Total number of splats
    /// Returns elapsed time in milliseconds
    @discardableResult
    public func runMasked(commandQueue: MTLCommandQueue,
                          xyzBuffer: MTLBuffer,
                          tBuffer: MTLBuffer,
                          outXYZ: MTLBuffer,
                          outRot: MTLBuffer,
                          outScale: MTLBuffer,
                          maskBuffer: MTLBuffer,
                          threshold: Float,
                          count: Int) -> Double {

        guard let exec = executable else { return 0 }

        // First, count how many masked splats we have and gather their indices
        let maskData = maskBuffer.contents().assumingMemoryBound(to: Float.self)
        var maskedIndices: [Int] = []
        maskedIndices.reserveCapacity(count / 10)  // Assume ~10% are dynamic

        for i in 0..<count {
            if maskData[i] > threshold {
                maskedIndices.append(i)
            }
        }

        let maskedCount = maskedIndices.count
        lastMaskedCount = maskedCount
        guard maskedCount > 0 else {
            print("DeformGraph: No masked splats to deform")
            return 0
        }

        print("DeformGraph: Running masked deformation on \(maskedCount)/\(count) splats")

        let floatSize = MemoryLayout<Float>.size

        // Create temporary buffers for the masked subset
        // Pad to next SAFE_BATCH_SIZE boundary to avoid MPS alignment issues
        let paddedCount = ((maskedCount + SAFE_BATCH_SIZE - 1) / SAFE_BATCH_SIZE) * SAFE_BATCH_SIZE
        let tempXYZ = device.makeBuffer(length: paddedCount * 3 * floatSize, options: .storageModeShared)!
        let tempT = device.makeBuffer(length: paddedCount * 1 * floatSize, options: .storageModeShared)!
        let tempOutXYZ = device.makeBuffer(length: paddedCount * 3 * floatSize, options: .storageModeShared)!
        let tempOutRot = device.makeBuffer(length: paddedCount * 4 * floatSize, options: .storageModeShared)!
        let tempOutScale = device.makeBuffer(length: paddedCount * 3 * floatSize, options: .storageModeShared)!

        // Copy masked data to temp buffers using memcpy for efficiency
        let xyzSrc = xyzBuffer.contents()
        let tSrc = tBuffer.contents()
        let xyzDst = tempXYZ.contents()
        let tDst = tempT.contents()

        // Use memcpy for each masked index - more efficient than element-by-element
        for (i, idx) in maskedIndices.enumerated() {
            let srcOffsetXYZ = idx * 3 * floatSize
            let dstOffsetXYZ = i * 3 * floatSize
            memcpy(xyzDst.advanced(by: dstOffsetXYZ), xyzSrc.advanced(by: srcOffsetXYZ), 3 * floatSize)

            let srcOffsetT = idx * 1 * floatSize
            let dstOffsetT = i * 1 * floatSize
            memcpy(tDst.advanced(by: dstOffsetT), tSrc.advanced(by: srcOffsetT), 1 * floatSize)
        }

        // Run deformation on the masked subset
        let totalStart = CFAbsoluteTimeGetCurrent()
        let numBatches = (maskedCount + SAFE_BATCH_SIZE - 1) / SAFE_BATCH_SIZE

        let cb = MPSCommandBuffer(from: commandQueue)
        cb.label = "DeformGraphSystem.runMasked"

        for batchStartIdx in stride(from: 0, to: maskedCount, by: SAFE_BATCH_SIZE) {
            autoreleasepool {
                let currentCount = min(SAFE_BATCH_SIZE, maskedCount - batchStartIdx)

                let xyzShape = [NSNumber(value: currentCount * 3)]
                let tShape = [NSNumber(value: currentCount * 1)]
                let outRotShape = [NSNumber(value: currentCount * 4)]
                let outScaleShape = [NSNumber(value: currentCount * 3)]

                let xyzData = createTensorView(buffer: tempXYZ, offset: batchStartIdx * 3 * floatSize, shape: xyzShape)
                let tData = createTensorView(buffer: tempT, offset: batchStartIdx * 1 * floatSize, shape: tShape)

                let outXYZData = createTensorView(buffer: tempOutXYZ, offset: batchStartIdx * 3 * floatSize, shape: xyzShape)
                let outRotData = createTensorView(buffer: tempOutRot, offset: batchStartIdx * 4 * floatSize, shape: outRotShape)
                let outScaleData = createTensorView(buffer: tempOutScale, offset: batchStartIdx * 3 * floatSize, shape: outScaleShape)

                var inputsArray: [MPSGraphTensorData] = []
                var resultsArray: [MPSGraphTensorData] = []

                if let feedTensors = exec.feedTensors {
                    for tensor in feedTensors {
                        let opName = tensor.operation.name
                        if opName == "in_xyz" || opName == "in_xyz_flat" { inputsArray.append(xyzData) }
                        else if opName == "in_t" || opName == "in_t_flat" { inputsArray.append(tData) }
                        else if (tensor.shape?[1].intValue ?? 0) == 3 { inputsArray.append(xyzData) }
                        else { inputsArray.append(tData) }
                    }
                }

                resultsArray.append(outXYZData)
                resultsArray.append(outRotData)
                resultsArray.append(outScaleData)

                if inputsArray.count == (exec.feedTensors?.count ?? 0) {
                    exec.encode(to: cb,
                                inputs: inputsArray,
                                results: resultsArray,
                                executionDescriptor: nil)
                }
            }
        }

        cb.commit()
        cb.waitUntilCompleted()

        // Copy results back to full output buffers at correct positions using memcpy
        let outXYZBase = outXYZ.contents()
        let outRotBase = outRot.contents()
        let outScaleBase = outScale.contents()
        let tempOutXYZData = tempOutXYZ.contents()
        let tempOutRotData = tempOutRot.contents()
        let tempOutScaleData = tempOutScale.contents()

        for (i, idx) in maskedIndices.enumerated() {
            // Copy XYZ
            let dstOffsetXYZ = idx * 3 * floatSize
            let srcOffsetXYZ = i * 3 * floatSize
            memcpy(outXYZBase.advanced(by: dstOffsetXYZ), tempOutXYZData.advanced(by: srcOffsetXYZ), 3 * floatSize)

            // Copy Rot
            let dstOffsetRot = idx * 4 * floatSize
            let srcOffsetRot = i * 4 * floatSize
            memcpy(outRotBase.advanced(by: dstOffsetRot), tempOutRotData.advanced(by: srcOffsetRot), 4 * floatSize)

            // Copy Scale
            let dstOffsetScale = idx * 3 * floatSize
            let srcOffsetScale = i * 3 * floatSize
            memcpy(outScaleBase.advanced(by: dstOffsetScale), tempOutScaleData.advanced(by: srcOffsetScale), 3 * floatSize)
        }

        let totalElapsedMs = (CFAbsoluteTimeGetCurrent() - totalStart) * 1000.0
        print("DeformGraph masked total (\(numBatches) batches): \(totalElapsedMs) ms")
        return totalElapsedMs
    }

    func positionalEncoding(input: MPSGraphTensor, numFreqs: Int, dataType: MPSDataType) -> MPSGraphTensor {
        var tensors = [input]
        for i in 0..<numFreqs {
            let freq = Float(pow(2.0, Double(i)))
            let freqConst = graph.constant(Double(freq), dataType: dataType)
            let scaled = graph.multiplication(input, freqConst, name: nil)
            tensors.append(graph.sin(with: scaled, name: nil))
            tensors.append(graph.cos(with: scaled, name: nil))
        }
        return graph.concatTensors(tensors, dimension: 1, name: nil)
    }
    
    func denseLayer(input: MPSGraphTensor, name: String, activation: Bool) -> MPSGraphTensor {
        guard let (wD, wS, wType) = weightDataMap["\(name)_w"],
              let (bD, bS, bType) = weightDataMap["\(name)_b"] else {
            print("CRITICAL ERROR: Missing weights for layer '\(name)'. Loaded keys: \(weightDataMap.keys)")
            fatalError("Missing weights for \(name)")
        }
        let w = graph.constant(wD, shape: wS, dataType: wType)
        let b = graph.constant(bD, shape: bS, dataType: bType)
        let wT = graph.transposeTensor(w, dimension: 0, withDimension: 1, name: nil)
        var out = graph.matrixMultiplication(primary: input, secondary: wT, name: nil)
        out = graph.addition(out, b, name: nil)
        if activation { out = graph.reLU(with: out, name: nil) }
        return out
    }
}
