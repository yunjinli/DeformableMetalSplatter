import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

public class DeformGraphSystem {
    let device: MTLDevice
    let mpsDevice: MPSGraphDevice
    let graph: MPSGraph
    var executable: MPSGraphExecutable?
    
    // Tensors
    var inputXYZTensor: MPSGraphTensor?
    var inputTTensor: MPSGraphTensor?
    var outXYZTensor: MPSGraphTensor?
    var outRotTensor: MPSGraphTensor?
    var outScaleTensor: MPSGraphTensor?
    
    var weightDataMap: [String: (Data, [NSNumber])] = [:]
    
    // Deformation Network Params
    let W = 256
    let D = 8
    let SKIP_LAYER = 4
    let MULTIRES = 10
    let T_MULTIRES = 10
    
    // When passing all points, the GPU crashed..., regularizing it with a batch size
//    let SAFE_BATCH_SIZE = 16384 * 8
    let SAFE_BATCH_SIZE = 4096
    var flatten = true
    public init(device: MTLDevice) {
        self.device = device
        self.mpsDevice = MPSGraphDevice(mtlDevice: device)
        self.graph = MPSGraph()
    }
    
    public func loadWeights(flatData: Data) {
        var offset = 0
        let floatSize = MemoryLayout<Float>.size
        let totalCh = 63 + 21
        
        func extract(outDim: Int, inDim: Int, name: String) {
            let byteCount = outDim * inDim * floatSize
            guard (offset + byteCount) <= flatData.count else { return }
            
            let subData = flatData.subdata(in: offset..<(offset + byteCount))
            weightDataMap["\(name)_w"] = (subData, [NSNumber(value: outDim), NSNumber(value: inDim)])
            offset += byteCount
            
            let biasBytes = outDim * floatSize
            let biasData = flatData.subdata(in: offset..<(offset + biasBytes))
            weightDataMap["\(name)_b"] = (biasData, [NSNumber(value: 1), NSNumber(value: outDim)])
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

        let embXYZ = positionalEncoding(input: xyz, numFreqs: MULTIRES)
        let embT   = positionalEncoding(input: t, numFreqs: T_MULTIRES)
        let inputs = graph.concatTensors([embXYZ, embT], dimension: 1, name: "input_concat")
        
        var h = inputs
        h = denseLayer(input: h, name: "L0", activation: true)
        
        for i in 0..<(D-1) {
            if i == SKIP_LAYER { h = graph.concatTensors([inputs, h], dimension: 1, name: "skip") }
            h = denseLayer(input: h, name: "L\(i+1)", activation: true)
        }
        
        // Output Layers
        let outXYZ = denseLayer(input: h, name: "Head_XYZ", activation: false)
        let outRot = denseLayer(input: h, name: "Head_Rot", activation: false)
        let outScale = denseLayer(input: h, name: "Head_Scale", activation: false)
        
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
    
    public func run(commandQueue: MTLCommandQueue,
                    xyzBuffer: MTLBuffer,
                    tBuffer: MTLBuffer,
                    outXYZ: MTLBuffer,
                    outRot: MTLBuffer,
                    outScale: MTLBuffer,
                    count: Int) {
        
        guard let exec = executable else { return }
        let floatSize = MemoryLayout<Float>.size
        for i in stride(from: 0, to: count, by: SAFE_BATCH_SIZE) {
            autoreleasepool {
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
                    let _ = exec.run(with: commandQueue,
                                     inputs: inputsArray,
                                     results: resultsArray,
                                     executionDescriptor: nil)
                }
            }
        }
        if let fenceBuffer = commandQueue.makeCommandBuffer() {
            fenceBuffer.label = "Deformation Fence"
            fenceBuffer.commit()
            fenceBuffer.waitUntilCompleted()
        }
    }
    
    func positionalEncoding(input: MPSGraphTensor, numFreqs: Int) -> MPSGraphTensor {
        var tensors = [input]
        for i in 0..<numFreqs {
            let freq = Float(pow(2.0, Double(i)))
            let scaled = graph.multiplication(input, graph.constant(Double(freq), dataType: .float32), name: nil)
            tensors.append(graph.sin(with: scaled, name: nil))
            tensors.append(graph.cos(with: scaled, name: nil))
        }
        return graph.concatTensors(tensors, dimension: 1, name: nil)
    }
    
    func denseLayer(input: MPSGraphTensor, name: String, activation: Bool) -> MPSGraphTensor {
        guard let (wD, wS) = weightDataMap["\(name)_w"],
              let (bD, bS) = weightDataMap["\(name)_b"] else {
            print("CRITICAL ERROR: Missing weights for layer '\(name)'. Loaded keys: \(weightDataMap.keys)")
            fatalError("Missing weights for \(name)")
        }
        let w = graph.constant(wD, shape: wS, dataType: .float32)
        let b = graph.constant(bD, shape: bS, dataType: .float32)
        let wT = graph.transposeTensor(w, dimension: 0, withDimension: 1, name: nil)
        var out = graph.matrixMultiplication(primary: input, secondary: wT, name: nil)
        out = graph.addition(out, b, name: nil)
        if activation { out = graph.reLU(with: out, name: nil) }
        return out
    }
}
