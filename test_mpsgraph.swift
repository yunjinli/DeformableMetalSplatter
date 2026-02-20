import Foundation
import Metal
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!

let graph = MPSGraph()
let t1 = graph.placeholder(shape: [1], dataType: .float32, name: "in")
let out = graph.addition(t1, t1, name: "out")

let exe = graph.compile(with: MPSGraphDevice(mtlDevice: device),
                        feeds: [t1 : MPSGraphShapedType(shape: [1], dataType: .float32)],
                        targetTensors: [out], targetOperations: nil, compilationDescriptor: nil)

let inBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
inBuf.contents().storeBytes(of: Float(5.0), as: Float.self)

let outBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
let inData = MPSGraphTensorData(MPSNDArray(buffer: inBuf, offset: 0, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: [1])))
let outData = MPSGraphTensorData(MPSNDArray(buffer: outBuf, offset: 0, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: [1])))

let start = CFAbsoluteTimeGetCurrent()
_ = exe.run(with: queue, inputs: [inData], results: [outData], executionDescriptor: nil)
let end = CFAbsoluteTimeGetCurrent()

print("Elapsed: \((end - start)*1000) ms")
let result = outBuf.contents().load(as: Float.self)
print("Result: \(result)")
