import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let cb = MPSCommandBuffer(from: queue)
cb.commit()
cb.waitUntilCompleted()
print("Works")
